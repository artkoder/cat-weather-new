import asyncio
import hashlib
import json
import logging
import shutil
import sqlite3
import sys
from copy import deepcopy
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import piexif
import pytest
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

import observability
from api.uploads import (
    UploadMetricsRecorder,
    UploadsConfig,
    _serialize_for_log,
    register_upload_jobs,
)
from data_access import DataAccess, create_device, insert_upload
from ingestion import (
    ImageMetadataResult,
    IngestionCallbacks,
    IngestionResult,
    UploadIngestionContext,
    ingest_photo,
)
from ingestion import (
    extract_image_metadata as _extract_image_metadata,
)
from jobs import Job, JobQueue
from main import Bot, apply_migrations
from openai_client import OpenAIResponse
from tests.fixtures.ingestion_utils import (
    OpenAIStub,
    StorageStub,
    TelegramStub,
    create_sample_image,
    create_sample_image_with_invalid_gps,
    create_sample_image_without_gps,
)


class DummyStorage(StorageStub):
    pass


class DummyTelegram(TelegramStub):
    pass


class FailingTelegram(TelegramStub):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    async def send_photo(
        self, *, chat_id: int, photo: Path, caption: str | None = None
    ) -> dict[str, object]:
        raise self.error

    async def send_document(
        self,
        *,
        chat_id: int,
        document,
        file_name: str,
        caption: str | None = None,
        content_type: str | None = None,
    ) -> dict[str, object]:
        raise self.error


class FlakyTelegram(TelegramStub):
    def __init__(self, *, failures: int = 1, error: Exception | None = None) -> None:
        super().__init__()
        self.failures = failures
        self.error = error or RuntimeError("transient error")

    async def send_photo(
        self, *, chat_id: int, photo: Path, caption: str | None = None
    ) -> dict[str, object]:
        if self.failures > 0:
            self.failures -= 1
            raise self.error
        return await super().send_photo(chat_id=chat_id, photo=photo, caption=caption)

    async def send_document(
        self,
        *,
        chat_id: int,
        document,
        file_name: str,
        caption: str | None = None,
        content_type: str | None = None,
    ) -> dict[str, object]:
        if self.failures > 0:
            self.failures -= 1
            raise self.error
        return await super().send_document(
            chat_id=chat_id,
            document=document,
            file_name=file_name,
            caption=caption,
            content_type=content_type,
        )



class CaptionGeoOpenAIStub:
    def __init__(self, response: OpenAIResponse | None = None) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []
        self.api_key = "test-key"

    async def generate_json(self, **kwargs: Any):  # type: ignore[override]
        self.calls.append(kwargs)
        return self.response

def _setup_connection(
    *, asset_channel_id: int | None = None, recognition_channel_id: int | None = None
) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_migrations(conn)
    conn.execute("DELETE FROM asset_channel")
    if asset_channel_id is not None:
        conn.execute(
            "INSERT INTO asset_channel (channel_id) VALUES (?)",
            (asset_channel_id,),
        )
    conn.execute("DELETE FROM recognition_channel")
    if recognition_channel_id is not None:
        conn.execute(
            "INSERT INTO recognition_channel (channel_id) VALUES (?)",
            (recognition_channel_id,),
        )
    conn.commit()
    return conn


def _prepare_upload(conn: sqlite3.Connection, *, file_key: str, gps_redacted: bool = False) -> str:
    create_device(
        conn,
        device_id="device-1",
        user_id=1,
        name="Device 1",
        secret="secret",
    )
    upload_id = insert_upload(
        conn,
        id="upload-1",
        device_id="device-1",
        idempotency_key="key-1",
        file_ref=file_key,
        gps_redacted_by_client=gps_redacted,
    )
    conn.commit()
    return upload_id


@pytest.mark.asyncio
async def test_extract_image_metadata_reads_jpeg_exif(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    image_path = create_sample_image(tmp_path / "sample.jpg")

    with caplog.at_level(logging.INFO):
        metadata_result = _extract_image_metadata(image_path)
        (
            mime_type,
            width,
            height,
            exif_payload,
            gps_payload,
            exif_ifds,
        ) = metadata_result

    assert mime_type == "image/jpeg"
    assert width == 640
    assert height == 480
    assert exif_payload["DateTimeOriginal"] == "2023:12:24 15:30:45"
    assert pytest.approx(gps_payload["latitude"], rel=1e-6) == 55.5
    assert pytest.approx(gps_payload["longitude"], rel=1e-6) == 37.6
    assert gps_payload["captured_at"].startswith("2023-12-24T15:30:45")
    assert metadata_result.photo is not None
    assert metadata_result.photo.latitude == pytest.approx(55.5, rel=1e-6)
    assert metadata_result.photo.longitude == pytest.approx(37.6, rel=1e-6)

    metadata_log = next(
        record for record in caplog.records if record.message == "MOBILE_IMAGE_METADATA"
    )
    assert metadata_log.path.endswith("sample.jpg")
    assert metadata_log.gps_present is True
    assert metadata_log.latitude == pytest.approx(55.5, rel=1e-6)
    assert metadata_log.longitude == pytest.approx(37.6, rel=1e-6)

    gps_log = next(record for record in caplog.records if record.message == "MOBILE_GPS_EXIF")
    assert gps_log.path.endswith("sample.jpg")
    lat_ref = gps_log.gps_tags["GPSLatitudeRef"]
    lon_ref = gps_log.gps_tags["GPSLongitudeRef"]

    def _decode_ref(value: Any) -> str:
        if isinstance(value, str) and len(value) == 2:
            try:
                decoded = bytes.fromhex(value).decode("utf-8")
            except Exception:
                return value
            return decoded
        return value

    assert _decode_ref(lat_ref) == "N"
    assert _decode_ref(lon_ref) == "E"
    gps_block = exif_payload["GPS"]
    assert gps_block["GPSLatitudeRef"] == "N"
    assert gps_block["GPSLongitudeRef"] == "E"
    gps_ifd = exif_ifds.get("GPS") or {}
    assert gps_ifd.get("GPSLatitude") == [[55, 1], [30, 1], [0, 1]]
    assert gps_ifd.get("GPSLongitude") == [[37, 1], [36, 1], [0, 1]]


def test_extract_image_metadata_logs_missing_coordinates(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    image_path = create_sample_image_with_invalid_gps(tmp_path / "no-coords.jpg")

    with caplog.at_level(logging.INFO):
        metadata_result = _extract_image_metadata(image_path)

    assert metadata_result.gps is not None
    assert metadata_result.gps.get("captured_at") == "2023-12-24T15:30:45+00:00"

    metadata_log = next(
        record for record in caplog.records if record.message == "MOBILE_IMAGE_METADATA"
    )
    assert metadata_log.path.endswith("no-coords.jpg")
    assert metadata_log.gps_present is False
    assert metadata_log.latitude is None
    assert metadata_log.longitude is None


def test_extract_image_metadata_handles_nested_rationals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "nested.jpg"
    Image.new("RGB", (32, 32), color=(128, 64, 32)).save(image_path, format="JPEG")

    nested_lat = (
        ((55, 1), (1, 1)),
        ((30, 1), (2, 1)),
    )
    nested_lon = (
        ((37, 1), (1, 1)),
        ((36, 1), (1, 1)),
        ((12, 1), (2, 1)),
    )

    gps_payload = {
        piexif.GPSIFD.GPSLatitudeRef: "S",
        piexif.GPSIFD.GPSLatitude: nested_lat,
        piexif.GPSIFD.GPSLongitudeRef: "W",
        piexif.GPSIFD.GPSLongitude: nested_lon,
    }

    def fake_getexif(self: Image.Image, *args: Any, **kwargs: Any):  # type: ignore[override]
        return {
            piexif.ExifIFD.DateTimeOriginal: "2024:01:02 03:04:05",
            34853: gps_payload,
        }

    monkeypatch.setattr(Image.Image, "getexif", fake_getexif)

    metadata_result = _extract_image_metadata(image_path)
    _, _, _, exif_payload, coords, exif_ifds = metadata_result

    assert coords["latitude"] == pytest.approx(-55.25, rel=1e-6)
    expected_lon = -(37 + 36 / 60.0 + 6 / 3600.0)
    assert coords["longitude"] == pytest.approx(expected_lon, rel=1e-6)
    assert metadata_result.photo is not None
    assert metadata_result.photo.latitude == pytest.approx(-55.25, rel=1e-6)
    assert metadata_result.photo.longitude == pytest.approx(expected_lon, rel=1e-6)

    gps_info = exif_payload["GPS"]
    assert gps_info == exif_payload["GPSInfo"]
    assert gps_info["GPSLatitudeRef"] == "S"
    assert gps_info["GPSLongitudeRef"] == "W"
    raw_gps_ifd = exif_ifds.get("GPS") or {}
    assert raw_gps_ifd == gps_info
    assert gps_info["GPSLatitude"] == [
        [[55, 1], [1, 1]],
        [[30, 1], [2, 1]],
    ]
    assert gps_info["GPSLongitude"] == [
        [[37, 1], [1, 1]],
        [[36, 1], [1, 1]],
        [[12, 1], [2, 1]],
    ]


def test_serialize_for_log_coerces_bytes_payload() -> None:
    photo_meta_log_payload = {
        "raw_bytes": b"\x00\xff",
        "nested": {
            "exif": [b"\x01\x02", Fraction(1, 3)],
            "captured_at": datetime(2024, 1, 2, 3, 4, 5),
        },
    }

    serialized = _serialize_for_log(photo_meta_log_payload)

    assert isinstance(serialized, str)
    decoded = json.loads(serialized)
    assert decoded["raw_bytes"] == "00ff"
    assert decoded["nested"]["exif"][0] == "0102"
    assert decoded["nested"]["exif"][1] == pytest.approx(1 / 3)
    assert decoded["nested"]["captured_at"] == "2024-01-02T03:04:05"


@pytest.mark.asyncio
async def test_ingest_photo_populates_result_gps_from_offset_ifd(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "offset.jpg"
    image = Image.new("RGB", (128, 96), color=(90, 120, 150))
    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef: b"N",
        piexif.GPSIFD.GPSLatitude: ((55, 1), (30, 1), (1234, 100)),
        piexif.GPSIFD.GPSLongitudeRef: b"E",
        piexif.GPSIFD.GPSLongitude: ((37, 1), (36, 1), (678, 100)),
    }
    exif_dict = {
        "0th": {piexif.ImageIFD.Make: b"UnitTest"},
        "Exif": {},
        "GPS": gps_ifd,
        "1st": {},
        "thumbnail": None,
    }
    image.save(image_path, format="JPEG", exif=piexif.dump(exif_dict))

    with Image.open(image_path) as created:
        raw_gps_value = created.getexif().get(34853)
    assert isinstance(raw_gps_value, int), "GPS IFD should be referenced via offset"

    conn = _setup_connection(asset_channel_id=-50001)
    data = DataAccess(conn)
    upload_id = _prepare_upload(conn, file_key="offset-key")

    telegram = TelegramStub()
    metrics = UploadMetricsRecorder()
    context = UploadIngestionContext(
        upload_id=upload_id,
        storage_key="offset-key",
        metrics=metrics,
        source="mobile",
    )
    config = UploadsConfig(
        assets_channel_id=-50001,
        vision_enabled=False,
    )

    result = await ingest_photo(
        data=data,
        telegram=telegram,
        openai=None,
        supabase=None,
        config=config,
        context=context,
        file_path=image_path,
        cleanup_file=False,
    )

    expected_lat = 55 + 30 / 60 + 12.34 / 3600
    expected_lon = 37 + 36 / 60 + 6.78 / 3600

    assert result.asset_id
    assert result.gps
    assert result.gps["latitude"] == pytest.approx(expected_lat, rel=1e-6)
    assert result.gps["longitude"] == pytest.approx(expected_lon, rel=1e-6)
    assert result.exif.get("GPS")
    assert result.exif["GPS"]["GPSLatitudeRef"] == "N"
    assert result.exif["GPS"]["GPSLongitudeRef"] == "E"

    asset = data.get_asset(result.asset_id)
    assert asset is not None
    assert asset.exif
    gps_block = asset.exif.get("GPS") if asset.exif else None
    assert gps_block
    assert gps_block["GPSLatitudeRef"] == "N"
    assert gps_block["GPSLongitudeRef"] == "E"

    metadata_payload = {"exif": result.exif, "gps": result.gps}
    data.update_asset(
        result.asset_id,
        metadata=metadata_payload,
        latitude=result.gps["latitude"],
        longitude=result.gps["longitude"],
        exif_present=True,
    )

    updated_asset = data.get_asset(result.asset_id)
    assert updated_asset is not None
    assert updated_asset.latitude == pytest.approx(expected_lat, rel=1e-6)
    assert updated_asset.longitude == pytest.approx(expected_lon, rel=1e-6)
    stored_metadata = updated_asset.metadata or {}
    stored_gps = stored_metadata.get("gps") or {}
    assert stored_gps.get("latitude") == pytest.approx(expected_lat, rel=1e-6)
    assert stored_gps.get("longitude") == pytest.approx(expected_lon, rel=1e-6)
    assert telegram.calls[0]["method"] == "sendPhoto"


@pytest.mark.asyncio
async def test_process_upload_job_success_records_asset(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    legacy_channel_id = -10001
    recognition_channel_id = -20001
    conn = _setup_connection(
        asset_channel_id=legacy_channel_id,
        recognition_channel_id=recognition_channel_id,
    )
    data = DataAccess(conn)
    image_path = create_sample_image(tmp_path / "asset.jpg")
    file_key = "sample-key"
    storage = StorageStub({file_key: image_path})
    telegram = TelegramStub()
    metrics = UploadMetricsRecorder()
    queue = JobQueue(conn)
    config = UploadsConfig(
        assets_channel_id=recognition_channel_id,
        vision_enabled=False,
    )
    register_upload_jobs(
        queue,
        conn,
        storage=storage,
        data=data,
        telegram=telegram,
        metrics=metrics,
        config=config,
    )

    upload_id = _prepare_upload(conn, file_key=file_key)
    job = Job(
        id=1,
        name="process_upload",
        payload={"upload_id": upload_id},
        status="queued",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    await queue.handlers["process_upload"](job)

    row = conn.execute(
        "SELECT status, error, asset_id FROM uploads WHERE id=?", (upload_id,)
    ).fetchone()
    assert row["status"] == "done"
    assert row["error"] is None
    assert row["asset_id"] is not None

    asset_row = conn.execute(
        "SELECT sha256, width, height, exif_json, tg_message_id, payload_json FROM assets WHERE id=?",
        (row["asset_id"],),
    ).fetchone()
    assert asset_row is not None
    expected_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    assert asset_row["sha256"] == expected_sha
    assert asset_row["width"] == 640
    assert asset_row["height"] == 480
    assert "GPSInfo" in (asset_row["exif_json"] or "")
    expected_identifier = f"{config.assets_channel_id}:{telegram.calls[0]['message_id']}"
    assert asset_row["tg_message_id"] == expected_identifier
    payload_blob = asset_row["payload_json"]
    assert payload_blob
    payload_map = json.loads(payload_blob)
    assert payload_map.get("file_id") == telegram.calls[0]["file_id"]

    assert len(telegram.calls) == 1
    assert telegram.calls[0]["method"] == "sendPhoto"
    assert telegram.calls[0]["chat_id"] == config.assets_channel_id
    assert storage.get_calls == [file_key]

    asset = data.get_asset(row["asset_id"])
    assert asset is not None
    assert asset.tg_chat_id == config.assets_channel_id
    assert asset.message_id == telegram.calls[0]["message_id"]
    assert asset.payload.get("tg_chat_id") == config.assets_channel_id
    assert asset.payload.get("message_id") == telegram.calls[0]["message_id"]
    assert asset.source == "mobile"
    metadata = asset.metadata or {}
    assert metadata.get("exif") == asset.exif
    gps_metadata = metadata.get("gps")
    assert gps_metadata
    assert pytest.approx(gps_metadata.get("latitude"), rel=1e-6) == 55.5
    assert pytest.approx(gps_metadata.get("longitude"), rel=1e-6) == 37.6
    assert metadata.get("exif_datetime_original") == "2023:12:24 15:30:45"
    assert metadata.get("exif_datetime_digitized") == "2023:12:24 15:30:45"
    assert metadata.get("exif_datetime") == "2023:12:24 15:30:45"
    assert metadata.get("exif_datetime_best") == "2023:12:24 15:30:45"
    assert asset.latitude == pytest.approx(55.5, rel=1e-6)
    assert asset.longitude == pytest.approx(37.6, rel=1e-6)
    assert asset.exif_present is True

    fetched = data.get_asset_by_message(config.assets_channel_id, telegram.calls[0]["message_id"])
    assert fetched is not None
    assert fetched.id == asset.id

    queued = conn.execute(
        "SELECT name, payload FROM jobs_queue WHERE name='ingest' ORDER BY id",
    ).fetchall()
    assert queued
    queued_payload = json.loads(queued[0]["payload"])
    assert queued_payload == {"asset_id": row["asset_id"]}

    assert metrics.counters.get("assets_created_total") == 1
    assert metrics.counters.get("upload_process_fail_total", 0) == 0
    assert "process_upload_ms" in metrics.timings

    exif_log = next(
        record for record in caplog.records if record.message == "MOBILE_EXIF_EXTRACTED"
    )
    assert exif_log.asset_id == row["asset_id"]
    assert exif_log.upload_id == upload_id
    assert exif_log.exif_payload is True
    assert exif_log.gps_payload is True
    assert exif_log.latitude == pytest.approx(55.5, rel=1e-6)
    assert exif_log.longitude == pytest.approx(37.6, rel=1e-6)

    raw_log = next(record for record in caplog.records if record.message == "MOBILE_EXIF_RAW")
    assert raw_log.asset_id == row["asset_id"]
    assert raw_log.upload_id == upload_id
    assert raw_log.has_exif is True
    assert raw_log.has_gps is True
    assert raw_log.gps_ifd_present is True
    assert len(raw_log.photo_meta_raw) <= 64 * 1024
    photo_meta_raw = json.loads(raw_log.photo_meta_raw)
    assert photo_meta_raw["has_exif"] is True
    assert photo_meta_raw["has_gps"] is True
    assert photo_meta_raw["gps_ifd_present"] is True
    assert pytest.approx(photo_meta_raw["latitude"], rel=1e-6) == 55.5
    assert pytest.approx(photo_meta_raw["longitude"], rel=1e-6) == 37.6
    raw_exif_sections = photo_meta_raw.get("raw_exif") or {}
    raw_zeroth = raw_exif_sections.get("0th") or {}
    raw_exif = raw_exif_sections.get("Exif") or {}
    raw_gps = photo_meta_raw.get("raw_gps") or {}
    assert raw_zeroth
    assert "Make" in raw_zeroth
    assert raw_exif
    original_raw = raw_exif.get("DateTimeOriginal")
    digitized_raw = raw_exif.get("DateTimeDigitized")

    def _decode(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return bytes.fromhex(value).decode("utf-8")
            except ValueError:
                return value
        return value

    assert _decode(original_raw) == "2023:12:24 15:30:45"
    assert _decode(digitized_raw) == "2023:12:24 15:30:45"
    assert raw_gps
    assert _decode(raw_gps.get("GPSLatitudeRef")) == "N"
    assert _decode(raw_gps.get("GPSLongitudeRef")) == "E"
    assert raw_gps.get("GPSLatitude") == [[55, 1], [30, 1], [0, 1]]
    assert raw_gps.get("GPSLongitude") == [[37, 1], [36, 1], [0, 1]]

    mobile_done = next(
        record for record in caplog.records if record.message == "MOBILE_UPLOAD_DONE"
    )
    assert mobile_done.upload_id == upload_id
    assert mobile_done.device_id == "device-1"
    assert mobile_done.tg_chat_id == recognition_channel_id
    assert mobile_done.source == "mobile"
    assert mobile_done.size_bytes == image_path.stat().st_size
    assert isinstance(mobile_done.timestamp, str) and mobile_done.timestamp


@pytest.mark.asyncio
async def test_ingest_photo_skips_gps_when_redacted(tmp_path: Path) -> None:
    conn = _setup_connection(asset_channel_id=-50001)
    data = DataAccess(conn)
    image_path = create_sample_image(tmp_path / "redacted.jpg")
    upload_id = _prepare_upload(conn, file_key="redacted-key", gps_redacted=True)
    telegram = TelegramStub()
    metrics = UploadMetricsRecorder()
    context = UploadIngestionContext(
        upload_id=upload_id,
        storage_key="redacted-key",
        metrics=metrics,
        source="mobile",
        gps_redacted_by_client=True,
    )
    config = UploadsConfig(
        assets_channel_id=-50001,
        vision_enabled=False,
    )

    result = await ingest_photo(
        data=data,
        telegram=telegram,
        openai=None,
        supabase=None,
        config=config,
        context=context,
        file_path=image_path,
        cleanup_file=False,
    )

    assert result.asset_id
    assert "latitude" not in result.gps
    assert "longitude" not in result.gps
    assert result.photo is not None
    assert result.photo.latitude is None
    assert result.photo.longitude is None

    asset = data.get_asset(result.asset_id)
    assert asset is not None
    assert asset.latitude is None
    assert asset.longitude is None
    gps_metadata = (asset.metadata or {}).get("gps") or {}
    assert "latitude" not in gps_metadata
    assert "longitude" not in gps_metadata


@pytest.mark.asyncio
async def test_process_upload_skips_gps_for_redacted_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recognition_channel_id = -22001
    conn = _setup_connection(
        asset_channel_id=-21001,
        recognition_channel_id=recognition_channel_id,
    )
    data = DataAccess(conn)
    image_path = create_sample_image(tmp_path / "redacted-job.jpg")
    file_key = "redacted-job"
    storage = StorageStub({file_key: image_path})
    telegram = TelegramStub()
    metrics = UploadMetricsRecorder()
    queue = JobQueue(conn)
    config = UploadsConfig(
        assets_channel_id=recognition_channel_id,
        vision_enabled=False,
    )

    base_extract = _extract_image_metadata
    skip_flags: list[bool] = []

    def capture_extract_image_metadata(
        path: Path, *, skip_gps: bool = False
    ) -> ImageMetadataResult:
        skip_flags.append(skip_gps)
        return base_extract(path, skip_gps=skip_gps)

    monkeypatch.setattr("ingestion.extract_image_metadata", capture_extract_image_metadata)

    register_upload_jobs(
        queue,
        conn,
        storage=storage,
        data=data,
        telegram=telegram,
        metrics=metrics,
        config=config,
    )

    upload_id = _prepare_upload(conn, file_key=file_key, gps_redacted=True)
    job = Job(
        id=1,
        name="process_upload",
        payload={"upload_id": upload_id},
        status="queued",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    await queue.handlers["process_upload"](job)

    assert skip_flags == [True]

    row = conn.execute("SELECT asset_id FROM uploads WHERE id=?", (upload_id,)).fetchone()
    assert row is not None and row["asset_id"] is not None
    asset = data.get_asset(row["asset_id"])
    assert asset is not None
    assert asset.latitude is None
    assert asset.longitude is None
    gps_metadata = (asset.metadata or {}).get("gps") or {}
    assert "latitude" not in gps_metadata
    assert "longitude" not in gps_metadata


@pytest.mark.asyncio
async def test_process_upload_serializes_complex_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recognition_channel_id = -30001
    conn = _setup_connection(
        asset_channel_id=-20001,
        recognition_channel_id=recognition_channel_id,
    )
    data = DataAccess(conn)
    image_path = create_sample_image(tmp_path / "complex.jpg")
    file_key = "complex-key"
    storage = StorageStub({file_key: image_path})
    telegram = TelegramStub()
    metrics = UploadMetricsRecorder()
    queue = JobQueue(conn)

    base_extract = _extract_image_metadata

    def fake_extract_image_metadata(path: Path, *, skip_gps: bool = False) -> ImageMetadataResult:
        base = base_extract(path, skip_gps=skip_gps)
        gps_block = dict(base.exif.get("GPS") or {})
        gps_block.update(
            {
                "AltitudeRatio": Fraction(5, 2),
                "Bytes": [b"\x0e", b"\x0f"],
            }
        )
        mutated_exif = dict(base.exif)
        mutated_exif["GPS"] = gps_block
        mutated_exif["GPSInfo"] = dict(gps_block)
        mutated_exif["BinaryTop"] = b"\x0c\x0d"
        return ImageMetadataResult(
            mime_type=base.mime_type,
            width=base.width,
            height=base.height,
            exif=mutated_exif,
            gps=dict(base.gps),
            exif_ifds=base.exif_ifds,
            photo=base.photo,
        )

    monkeypatch.setattr("ingestion.extract_image_metadata", fake_extract_image_metadata)

    vision_payload: dict[str, Any] = {
        "caption": "complex",
        "categories": ["test"],
        "raw_bytes": b"\x99",
        "stats": {"ratio": Fraction(2, 5), "chunks": [b"\x01\x02"]},
    }

    class VisionStub:
        def __init__(self, payload: dict[str, Any]) -> None:
            self.payload = payload

        async def classify_image(self, **_: Any) -> OpenAIResponse:
            return OpenAIResponse(
                self.payload,
                {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                    "request_id": "req-complex",
                },
            )

    openai_stub = VisionStub(vision_payload)

    config = UploadsConfig(
        assets_channel_id=recognition_channel_id,
        vision_enabled=True,
        openai_vision_model="vision-test",
    )

    register_upload_jobs(
        queue,
        conn,
        storage=storage,
        data=data,
        telegram=telegram,
        openai=openai_stub,
        metrics=metrics,
        config=config,
    )

    upload_id = _prepare_upload(conn, file_key=file_key)
    job = Job(
        id=1,
        name="process_upload",
        payload={"upload_id": upload_id},
        status="queued",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    await queue.handlers["process_upload"](job)

    upload_row = conn.execute(
        "SELECT status, error, asset_id FROM uploads WHERE id=?",
        (upload_id,),
    ).fetchone()
    assert upload_row["status"] == "done"
    assert upload_row["error"] is None
    assert upload_row["asset_id"] is not None

    asset_row = conn.execute(
        "SELECT exif_json, labels_json FROM assets WHERE id=?",
        (upload_row["asset_id"],),
    ).fetchone()
    assert asset_row is not None
    assert asset_row["exif_json"]
    assert asset_row["labels_json"]

    exif_payload = json.loads(asset_row["exif_json"])
    gps_block = exif_payload.get("GPS") or {}
    assert exif_payload["BinaryTop"] == "0c0d"
    assert gps_block.get("Bytes") == ["0e", "0f"]
    assert gps_block.get("AltitudeRatio") == pytest.approx(2.5, rel=1e-6)

    labels_payload = json.loads(asset_row["labels_json"])
    assert labels_payload.get("raw_bytes") == "99"
    stats_payload = labels_payload.get("stats") or {}
    assert stats_payload.get("ratio") == pytest.approx(0.4, rel=1e-6)
    assert stats_payload.get("chunks") == ["0102"]


@pytest.mark.asyncio
async def test_process_upload_injects_exif_into_file_meta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recognition_channel_id = -42001
    conn = _setup_connection(
        asset_channel_id=recognition_channel_id,
        recognition_channel_id=recognition_channel_id,
    )
    data = DataAccess(conn)
    image_path = create_sample_image(tmp_path / "file-meta-exif.jpg")
    file_key = "file-meta-exif"
    storage = StorageStub({file_key: image_path})
    telegram = TelegramStub()
    metrics = UploadMetricsRecorder()
    queue = JobQueue(conn)
    config = UploadsConfig(
        assets_channel_id=recognition_channel_id,
        vision_enabled=False,
    )

    register_upload_jobs(
        queue,
        conn,
        storage=storage,
        data=data,
        telegram=telegram,
        metrics=metrics,
        config=config,
    )

    upload_id = _prepare_upload(conn, file_key=file_key)

    injected_exif: dict[str, Any] = {
        "BinaryTag": b"\x00\x01",
        "ExposureTime": (1, 200),
        "FractionTag": Fraction(3, 2),
        "GPS": {
            "GPSLatitude": [(55, 1), (30, 1), (0, 1)],
            "MakerNote": b"\x02\x03",
        },
    }

    original_save_asset = DataAccess.save_asset

    def save_asset_with_exif(self: DataAccess, *args: Any, **kwargs: Any) -> str:
        file_meta = dict(kwargs.get("file_meta") or {})
        file_meta["exif"] = deepcopy(injected_exif)
        kwargs["file_meta"] = file_meta
        return original_save_asset(self, *args, **kwargs)

    monkeypatch.setattr(DataAccess, "save_asset", save_asset_with_exif)

    job = Job(
        id=99,
        name="process_upload",
        payload={"upload_id": upload_id},
        status="queued",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    await queue.handlers["process_upload"](job)

    upload_row = conn.execute(
        "SELECT status, error, asset_id FROM uploads WHERE id=?",
        (upload_id,),
    ).fetchone()
    assert upload_row["status"] == "done"
    assert upload_row["error"] is None
    assert upload_row["asset_id"]

    asset_row = conn.execute(
        "SELECT exif_json FROM assets WHERE id=?",
        (upload_row["asset_id"],),
    ).fetchone()
    assert asset_row is not None
    assert asset_row["exif_json"]

    exif_payload = json.loads(asset_row["exif_json"])
    assert exif_payload["BinaryTag"] == "0001"
    assert exif_payload["ExposureTime"] == pytest.approx(1 / 200, rel=1e-6)
    assert exif_payload["FractionTag"] == pytest.approx(1.5, rel=1e-6)
    gps_payload = exif_payload.get("GPS") or {}
    assert gps_payload.get("GPSLatitude") == [55.0, 30.0, 0.0]
    assert gps_payload.get("MakerNote") == "0203"


@pytest.mark.asyncio
async def test_process_upload_job_parses_comma_separated_gps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recognition_channel_id = -40001
    conn = _setup_connection(
        asset_channel_id=recognition_channel_id,
        recognition_channel_id=recognition_channel_id,
    )
    data = DataAccess(conn)
    image_path = create_sample_image(tmp_path / "asset-commas.jpg")
    file_key = "comma-key"
    storage = StorageStub({file_key: image_path})
    telegram = TelegramStub()
    metrics = UploadMetricsRecorder()
    queue = JobQueue(conn)
    config = UploadsConfig(
        assets_channel_id=recognition_channel_id,
        vision_enabled=False,
    )
    register_upload_jobs(
        queue,
        conn,
        storage=storage,
        data=data,
        telegram=telegram,
        metrics=metrics,
        config=config,
    )

    upload_id = _prepare_upload(conn, file_key=file_key)
    job = Job(
        id=3,
        name="process_upload",
        payload={"upload_id": upload_id},
        status="queued",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    original_load = piexif.load

    def _stringified_gps_load(*args: Any, **kwargs: Any):
        data = original_load(*args, **kwargs)
        gps = data.get("GPS")
        if isinstance(gps, dict):
            lat_tag = piexif.GPSIFD.GPSLatitude
            lon_tag = piexif.GPSIFD.GPSLongitude
            if lat_tag in gps:
                gps[lat_tag] = "55/1,30/1,1234/100"
            if lon_tag in gps:
                gps[lon_tag] = "37/1 36/1 600/100"
        return data

    monkeypatch.setattr(piexif, "load", _stringified_gps_load)

    await queue.handlers["process_upload"](job)

    row = conn.execute(
        "SELECT asset_id FROM uploads WHERE id=?",
        (upload_id,),
    ).fetchone()
    assert row is not None and row["asset_id"]

    asset = data.get_asset(row["asset_id"])
    assert asset is not None

    expected_lat = 55 + 30 / 60 + 12.34 / 3600
    expected_lon = 37 + 36 / 60 + 6 / 3600
    assert asset.latitude == pytest.approx(expected_lat, rel=1e-6)
    assert asset.longitude == pytest.approx(expected_lon, rel=1e-6)

    metadata = asset.metadata or {}
    gps_metadata = metadata.get("gps") or {}
    assert gps_metadata.get("latitude") == pytest.approx(expected_lat, rel=1e-6)
    assert gps_metadata.get("longitude") == pytest.approx(expected_lon, rel=1e-6)

    caption_text = telegram.calls[0]["caption"]
    assert isinstance(caption_text, str)
    expected_coordinates = f"Координаты: {expected_lat:.5f}, {expected_lon:.5f}"
    assert expected_coordinates in caption_text
    assert asset.caption is not None
    assert expected_coordinates in asset.caption


@pytest.mark.asyncio
async def test_process_upload_marks_exif_present_without_gps(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    recognition_channel_id = -30001
    conn = _setup_connection(
        asset_channel_id=recognition_channel_id,
        recognition_channel_id=recognition_channel_id,
    )
    data = DataAccess(conn)
    image_path = create_sample_image_with_invalid_gps(tmp_path / "asset-invalid-gps.jpg")
    file_key = "invalid-gps-key"
    storage = StorageStub({file_key: image_path})
    telegram = TelegramStub()
    metrics = UploadMetricsRecorder()
    queue = JobQueue(conn)
    config = UploadsConfig(
        assets_channel_id=recognition_channel_id,
        vision_enabled=False,
    )
    register_upload_jobs(
        queue,
        conn,
        storage=storage,
        data=data,
        telegram=telegram,
        metrics=metrics,
        config=config,
    )

    upload_id = _prepare_upload(conn, file_key=file_key)
    job = Job(
        id=2,
        name="process_upload",
        payload={"upload_id": upload_id},
        status="queued",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    await queue.handlers["process_upload"](job)

    row = conn.execute(
        "SELECT asset_id FROM uploads WHERE id=?",
        (upload_id,),
    ).fetchone()
    assert row is not None and row["asset_id"] is not None

    asset = data.get_asset(row["asset_id"])
    assert asset is not None

    metadata = asset.metadata or {}
    assert metadata.get("exif") == asset.exif
    gps_metadata = metadata.get("gps") or {}
    assert "latitude" not in gps_metadata
    assert "longitude" not in gps_metadata
    assert asset.latitude is None
    assert asset.longitude is None
    assert asset.exif_present is True

    exif_log = next(
        record for record in caplog.records if record.message == "MOBILE_EXIF_EXTRACTED"
    )
    assert exif_log.asset_id == row["asset_id"]
    assert exif_log.upload_id == upload_id
    assert exif_log.exif_payload is True
    assert exif_log.gps_payload is False
    assert exif_log.latitude is None
    assert exif_log.longitude is None

    raw_log = next(record for record in caplog.records if record.message == "MOBILE_EXIF_RAW")
    assert raw_log.asset_id == row["asset_id"]
    assert raw_log.upload_id == upload_id
    assert raw_log.has_exif is True
    assert raw_log.has_gps is False
    assert raw_log.gps_ifd_present is True
    assert len(raw_log.photo_meta_raw) <= 64 * 1024
    photo_meta_raw = json.loads(raw_log.photo_meta_raw)
    assert photo_meta_raw["has_exif"] is True
    assert photo_meta_raw["has_gps"] is False
    assert photo_meta_raw["gps_ifd_present"] is True
    raw_gps = photo_meta_raw.get("raw_gps") or {}
    assert raw_gps

    def _decode(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return bytes.fromhex(value).decode("utf-8")
            except ValueError:
                return value
        return value

    assert _decode(raw_gps.get("GPSLatitudeRef")) == "Q"
    assert _decode(raw_gps.get("GPSLongitudeRef")) == "R"
    assert raw_gps.get("GPSLatitude") == [[55, 1], [30, 1], [0, 1]]
    assert raw_gps.get("GPSLongitude") == [[37, 1], [36, 1], [0, 1]]

    metadata_log = next(
        record for record in caplog.records if record.message == "MOBILE_IMAGE_METADATA"
    )
    assert metadata_log.path.endswith("asset-invalid-gps.jpg")
    assert metadata_log.gps_present is False
    assert metadata_log.latitude is None
    assert metadata_log.longitude is None


@pytest.mark.asyncio
async def test_ingest_job_uses_stored_coordinates_when_result_missing_gps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "bot.sqlite"
    bot = Bot("test-token", str(db_path))
    bot.asset_storage = tmp_path

    stored_lat = 55.5
    stored_lon = 37.6
    metadata_payload = {
        "gps": {"latitude": stored_lat, "longitude": stored_lon},
        "exif": {"DateTimeOriginal": "2023:12:24 15:30:45"},
    }

    asset_channel = -40001
    asset_id = bot.data.save_asset(
        asset_channel,
        101,
        None,
        None,
        tg_chat_id=asset_channel,
        caption="Test asset",
        kind="photo",
        file_meta={
            "file_id": "file-101",
            "file_unique_id": "file-101-uniq",
            "mime_type": "image/jpeg",
            "width": 640,
            "height": 480,
        },
        metadata=metadata_payload,
        origin="mobile",
        source="mobile",
        author_user_id=777,
    )

    bot.data.update_asset(
        asset_id,
        metadata=metadata_payload,
        latitude=stored_lat,
        longitude=stored_lon,
        exif_present=True,
    )

    telegram_copy = create_sample_image_without_gps(tmp_path / "telegram.jpg")

    async def fake_download_file(self: Bot, file_id: str, dest_path):
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(telegram_copy, dest)
        return dest

    async def fake_api_request(self: Bot, method: str, payload: dict[str, Any]):
        return {"ok": True, "result": {"file_path": "dummy"}}

    reverse_calls: list[tuple[float, float]] = []

    async def fake_reverse_geocode(self: Bot, lat: float, lon: float):
        reverse_calls.append((lat, lon))
        return {"city": "Москва", "country": "Россия"}


    async def fake_ingest_photo(**kwargs):
        callbacks: IngestionCallbacks | None = kwargs.get("callbacks")
        overrides: dict[str, Any] = kwargs.get("input_overrides") or {}
        if callbacks and callbacks.save_asset:
            callbacks.save_asset(
                {
                    "channel_id": overrides.get("channel_id", asset_channel),
                    "message_id": 202,
                    "tg_chat_id": overrides.get("tg_chat_id", asset_channel),
                }
            )
        return SimpleNamespace(gps={})

    monkeypatch.setattr(Bot, "_download_file", fake_download_file)
    monkeypatch.setattr(Bot, "api_request", fake_api_request)
    monkeypatch.setattr(Bot, "_reverse_geocode", fake_reverse_geocode)
    monkeypatch.setattr(sys.modules["main"], "ingest_photo", fake_ingest_photo)

    job = Job(
        id=301,
        name="ingest",
        payload={"asset_id": asset_id},
        status="queued",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    await bot._job_ingest(job)

    updated_asset = bot.data.get_asset(asset_id)
    assert updated_asset is not None
    assert updated_asset.latitude == pytest.approx(stored_lat, rel=1e-6)
    assert updated_asset.longitude == pytest.approx(stored_lon, rel=1e-6)
    assert updated_asset.city == "Москва"
    assert updated_asset.country == "Россия"
    assert reverse_calls == [(stored_lat, stored_lon)]
    assert ingest_calls, "ingest_photo should be invoked"


    bot.db.close()


@pytest.mark.asyncio
async def test_ingest_job_extracts_geo_from_caption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "bot.sqlite"
    bot = Bot("test-token", str(db_path))
    bot.asset_storage = tmp_path

    channel_id = -43001
    caption_text = "Координаты: N54°43.200' E20°30.400'. Снято 21.05.2024 18:32"
    asset_id = bot.data.save_asset(
        channel_id,
        777,
        None,
        "#geo",
        tg_chat_id=channel_id,
        caption=caption_text,
        kind="photo",
        file_meta={
            "file_id": "file-geo",
            "file_unique_id": "file-geo-uniq",
            "mime_type": "image/jpeg",
            "width": 64,
            "height": 64,
        },
        origin="recognition",
    )

    sample_path = create_sample_image_without_gps(tmp_path / "caption-geo.jpg")

    async def fake_download_file(self: Bot, file_id: str, dest_path):
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sample_path, dest)
        return dest

    async def fake_api_request(self: Bot, method: str, payload: dict[str, Any]):
        return {"ok": True, "result": {"file_path": "dummy"}}

    reverse_calls: list[tuple[float, float]] = []

    async def fake_reverse_geocode(self: Bot, lat: float, lon: float):
        reverse_calls.append((lat, lon))
        return {"city": "Калининград", "country": "Россия"}

    response = OpenAIResponse(
        {"latitude": 54.72, "longitude": 20.51, "captured_at": "2024-05-21T18:32:00+02:00"},
        {
            "prompt_tokens": 12,
            "completion_tokens": 6,
            "total_tokens": 18,
            "request_id": "req-caption-geo",
        },
    )
    bot.openai = CaptionGeoOpenAIStub(response)

    ingest_calls: list[dict[str, Any]] = []
    kaliningrad_tz = ZoneInfo("Europe/Kaliningrad")

    async def fake_ingest_photo(**kwargs):
        ingest_calls.append(kwargs)
        overrides = kwargs.get("input_overrides") or {}
        gps_override = dict(overrides.get("gps") or {})
        capture_iso = gps_override.get("captured_at")
        shot_at_utc = None
        shot_doy = None
        if capture_iso:
            capture_dt = datetime.fromisoformat(capture_iso.replace("Z", "+00:00"))
            if capture_dt.tzinfo is None:
                capture_dt = capture_dt.replace(tzinfo=kaliningrad_tz)
            shot_at_utc = int(capture_dt.astimezone(UTC).timestamp())
            shot_doy = capture_dt.astimezone(kaliningrad_tz).timetuple().tm_yday
        callbacks: IngestionCallbacks | None = kwargs.get("callbacks")
        if callbacks and callbacks.save_asset:
            callbacks.save_asset(
                {
                    "channel_id": overrides.get("channel_id", channel_id),
                    "message_id": 321,
                    "tg_chat_id": overrides.get("tg_chat_id", channel_id),
                    "caption": overrides.get("caption"),
                    "kind": overrides.get("kind", "photo"),
                    "file_meta": {
                        "file_id": "file-geo",
                        "file_unique_id": "file-geo-uniq",
                        "mime_type": "image/jpeg",
                        "width": 64,
                        "height": 64,
                        "sha256": "sha",
                        "exif": {},
                    },
                    "metadata": {"gps": gps_override, "exif": {}},
                    "latitude": gps_override.get("latitude"),
                    "longitude": gps_override.get("longitude"),
                    "exif_present": bool(gps_override),
                    "shot_at_utc": shot_at_utc,
                    "shot_doy": shot_doy,
                    "photo_doy": shot_doy,
                }
            )
        return SimpleNamespace(gps=gps_override)

    monkeypatch.setattr(Bot, "_download_file", fake_download_file)
    monkeypatch.setattr(Bot, "api_request", fake_api_request)
    monkeypatch.setattr(Bot, "_reverse_geocode", fake_reverse_geocode)
    monkeypatch.setattr(sys.modules["main"], "ingest_photo", fake_ingest_photo)

    job = Job(
        id=401,
        name="ingest",
        payload={"asset_id": asset_id},
        status="queued",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    await bot._job_ingest(job)

    updated_asset = bot.data.get_asset(asset_id)
    assert updated_asset is not None
    assert updated_asset.latitude == pytest.approx(54.72, rel=1e-6)
    assert updated_asset.longitude == pytest.approx(20.51, rel=1e-6)
    capture_dt = datetime(2024, 5, 21, 18, 32, tzinfo=kaliningrad_tz)
    expected_iso = capture_dt.astimezone(UTC).isoformat()
    assert updated_asset.captured_at == expected_iso
    assert updated_asset.shot_at_utc == int(capture_dt.astimezone(UTC).timestamp())
    assert reverse_calls == [
        (pytest.approx(54.72, rel=1e-6), pytest.approx(20.51, rel=1e-6))
    ]
    assert bot.openai.calls, "caption geo extraction should call OpenAI"

    await bot.close()


@pytest.mark.asyncio
async def test_ingest_job_skips_caption_geo_when_coordinates_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "bot.sqlite"
    bot = Bot("test-token", str(db_path))
    bot.asset_storage = tmp_path

    channel_id = -43002
    caption_text = "54.6500, 20.3500 21.05.2024 18:32"
    asset_id = bot.data.save_asset(
        channel_id,
        778,
        None,
        "#geo",
        tg_chat_id=channel_id,
        caption=caption_text,
        kind="photo",
        file_meta={
            "file_id": "file-existing",
            "file_unique_id": "file-existing-uniq",
            "mime_type": "image/jpeg",
            "width": 64,
            "height": 64,
        },
        origin="recognition",
    )

    kaliningrad_tz = ZoneInfo("Europe/Kaliningrad")
    capture_dt = datetime(2024, 5, 20, 17, 15, tzinfo=kaliningrad_tz)
    capture_iso = capture_dt.isoformat()
    shot_at_utc = int(capture_dt.astimezone(UTC).timestamp())

    bot.data.update_asset(
        asset_id,
        latitude=54.65,
        longitude=20.35,
        shot_at_utc=shot_at_utc,
        metadata={"gps": {"captured_at": capture_iso}},
    )

    response = OpenAIResponse(
        {"latitude": 1.0, "longitude": 2.0, "captured_at": None},
        {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "request_id": "req-caption-skip",
        },
    )
    stub = CaptionGeoOpenAIStub(response)
    bot.openai = stub

    sample_path = create_sample_image_without_gps(tmp_path / "caption-geo-skip.jpg")

    async def fake_download_file(self: Bot, file_id: str, dest_path):
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sample_path, dest)
        return dest

    async def fake_api_request(self: Bot, method: str, payload: dict[str, Any]):
        return {"ok": True, "result": {"file_path": "dummy"}}

    reverse_calls: list[tuple[float, float]] = []

    async def fake_reverse_geocode(self: Bot, lat: float, lon: float):
        reverse_calls.append((lat, lon))
        return {"city": "Калининград", "country": "Россия"}

    async def fake_ingest_photo(**kwargs):
        overrides = kwargs.get("input_overrides") or {}
        gps_override = dict(overrides.get("gps") or {})
        capture_iso_override = gps_override.get("captured_at")
        shot_at_override = None
        shot_doy_override = None
        if capture_iso_override:
            capture_dt_override = datetime.fromisoformat(capture_iso_override)
            if capture_dt_override.tzinfo is None:
                capture_dt_override = capture_dt_override.replace(tzinfo=kaliningrad_tz)
            shot_at_override = int(capture_dt_override.astimezone(UTC).timestamp())
            shot_doy_override = capture_dt_override.astimezone(kaliningrad_tz).timetuple().tm_yday
        callbacks: IngestionCallbacks | None = kwargs.get("callbacks")
        if callbacks and callbacks.save_asset:
            callbacks.save_asset(
                {
                    "channel_id": overrides.get("channel_id", channel_id),
                    "message_id": 654,
                    "tg_chat_id": overrides.get("tg_chat_id", channel_id),
                    "caption": overrides.get("caption"),
                    "kind": overrides.get("kind", "photo"),
                    "file_meta": {
                        "file_id": "file-existing",
                        "file_unique_id": "file-existing-uniq",
                        "mime_type": "image/jpeg",
                        "width": 64,
                        "height": 64,
                        "sha256": "sha",
                        "exif": {},
                    },
                    "metadata": {"gps": gps_override, "exif": {}},
                    "latitude": gps_override.get("latitude"),
                    "longitude": gps_override.get("longitude"),
                    "exif_present": True,
                    "shot_at_utc": shot_at_override,
                    "shot_doy": shot_doy_override,
                    "photo_doy": shot_doy_override,
                }
            )
        return SimpleNamespace(gps=gps_override)

    monkeypatch.setattr(Bot, "_download_file", fake_download_file)
    monkeypatch.setattr(Bot, "api_request", fake_api_request)
    monkeypatch.setattr(Bot, "_reverse_geocode", fake_reverse_geocode)
    monkeypatch.setattr(sys.modules["main"], "ingest_photo", fake_ingest_photo)

    job = Job(
        id=402,
        name="ingest",
        payload={"asset_id": asset_id},
        status="queued",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    await bot._job_ingest(job)

    updated_asset = bot.data.get_asset(asset_id)
    assert updated_asset is not None
    assert updated_asset.latitude == pytest.approx(54.65, rel=1e-6)
    assert updated_asset.longitude == pytest.approx(20.35, rel=1e-6)
    assert updated_asset.captured_at == capture_iso
    assert reverse_calls == [
        (pytest.approx(54.65, rel=1e-6), pytest.approx(20.35, rel=1e-6))
    ]
    assert not stub.calls, "caption geo extraction should be skipped when data exists"

    await bot.close()


@pytest.mark.asyncio
async def test_ingest_job_mobile_adapter_handles_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "bot.sqlite"
    bot = Bot("test-token", str(db_path))
    bot.asset_storage = tmp_path

    channel_id = -42001
    message_id = 777
    asset_id = bot.data.save_asset(
        channel_id,
        message_id,
        None,
        None,
        tg_chat_id=channel_id,
        caption="Existing mobile asset",
        kind="photo",
        file_meta={
            "file_id": "file-mobile",
            "file_unique_id": "file-mobile-unique",
            "file_name": "mobile.jpg",
            "mime_type": "image/jpeg",
            "file_size": 64,
            "width": 32,
            "height": 32,
        },
        origin="mobile",
        source="mobile",
    )

    source_path = tmp_path / "downloaded.jpg"
    source_bytes = b"test-image-data"
    source_path.write_bytes(source_bytes)

    async def fake_download(self, file_id, dest_path=None):
        assert dest_path is not None
        destination = Path(dest_path)
        destination.write_bytes(source_path.read_bytes())
        return destination

    async def fail_publish_as_photo(self, chat_id, local_path, caption, *, caption_entities=None):
        raise AssertionError("send_photo should reuse existing message")

    publish_calls: list[dict[str, Any]] = []

    async def fake_publish_mobile_document(
        self,
        chat_id,
        document,
        filename,
        caption,
        *,
        caption_entities=None,
        content_type=None,
    ):
        data_preview = None
        if hasattr(document, "read"):
            try:
                position = document.tell()
            except Exception:
                position = None
            data_preview = document.read()
            try:
                if position is not None:
                    document.seek(position)
                else:
                    document.seek(0)
            except Exception:
                pass
        publish_calls.append(
            {
                "chat_id": chat_id,
                "filename": filename,
                "caption": caption,
                "content_type": content_type,
                "stream_type": type(document).__name__,
                "data": data_preview,
            }
        )
        return {"message_id": 900 + len(publish_calls), "chat": {"id": chat_id}}

    monkeypatch.setattr(Bot, "_download_file", fake_download)
    monkeypatch.setattr(Bot, "_publish_as_photo", fail_publish_as_photo)
    monkeypatch.setattr(Bot, "_publish_mobile_document", fake_publish_mobile_document)

    document_responses: list[tuple[str, dict[str, Any] | None]] = []
    photo_responses: list[dict[str, Any]] = []

    async def fake_ingest_photo(
        *,
        data,
        telegram,
        openai,
        supabase,
        config,
        context: UploadIngestionContext,
        file_path: Path,
        cleanup_file: bool,
        callbacks: IngestionCallbacks,
        input_overrides: dict[str, Any],
    ) -> IngestionResult:
        channel = input_overrides["channel_id"]
        photo_response = await telegram.send_photo(
            chat_id=channel,
            photo=Path(file_path),
            caption="photo",
        )
        photo_responses.append(photo_response)
        reuse_response = await telegram.send_document(
            chat_id=channel,
            document=b"reuse-bytes",
            file_name="reuse.bin",
            caption="reuse",
            content_type="application/octet-stream",
        )
        document_responses.append(("reuse", reuse_response))
        with open(file_path, "rb") as stream:
            stream_response = await telegram.send_document(
                chat_id=channel + 1,
                document=stream,
                file_name="stream.bin",
                caption="stream",
                content_type="application/octet-stream",
            )
        document_responses.append(("stream", stream_response))
        bytes_response = await telegram.send_document(
            chat_id=channel + 2,
            document=b"bytes-data",
            file_name="bytes.bin",
            caption="bytes",
            content_type="application/octet-stream",
        )
        document_responses.append(("bytes", bytes_response))
        if callbacks.save_asset:
            callbacks.save_asset(
                {
                    "channel_id": channel,
                    "message_id": photo_response["message_id"],
                    "tg_chat_id": input_overrides.get("tg_chat_id", channel),
                    "caption": "stub caption",
                    "kind": "photo",
                    "file_meta": {"file_id": "file-mobile", "mime_type": "image/jpeg"},
                    "metadata": {},
                    "source": "mobile",
                }
            )
        return IngestionResult(
            asset_id=input_overrides.get("asset_id"),
            message_id=photo_response["message_id"],
            chat_id=photo_response["chat"]["id"],
            caption="stub caption",
            sha256="sha",
            mime_type="image/jpeg",
            width=32,
            height=32,
            exif={},
            gps={},
            exif_ifds={},
            vision=None,
            telegram_file=None,
            metrics=context.metrics,
        )

    monkeypatch.setattr(sys.modules["main"], "ingest_photo", fake_ingest_photo)

    job = Job(
        id=501,
        name="ingest",
        payload={"asset_id": asset_id},
        status="queued",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    await bot._job_ingest(job)

    assert photo_responses and photo_responses[0]["message_id"] == message_id
    assert document_responses[0][0] == "reuse"
    reuse_payload = document_responses[0][1]
    assert reuse_payload is not None and reuse_payload["message_id"] == message_id
    assert len(publish_calls) == 2
    assert publish_calls[0]["chat_id"] == channel_id + 1
    assert publish_calls[0]["data"] == source_bytes
    assert publish_calls[1]["chat_id"] == channel_id + 2
    assert publish_calls[1]["stream_type"] == "BytesIO"
    assert publish_calls[1]["data"] == b"bytes-data"

    await bot.close()


@pytest.mark.asyncio
async def test_ingest_job_preserves_existing_exif_flag_on_parse_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "bot.sqlite"
    bot = Bot("test-token", str(db_path))
    bot.asset_storage = tmp_path

    stored_lat = 55.5
    stored_lon = 37.6
    metadata_payload = {
        "gps": {"latitude": stored_lat, "longitude": stored_lon},
        "exif": {"DateTimeOriginal": "2023:12:24 15:30:45"},
    }

    asset_channel = -50001
    asset_id = bot.data.save_asset(
        asset_channel,
        101,
        None,
        None,
        tg_chat_id=asset_channel,
        caption="Test asset",
        kind="photo",
        file_meta={
            "file_id": "file-101",
            "file_unique_id": "file-101-uniq",
            "mime_type": "image/jpeg",
            "width": 640,
            "height": 480,
        },
        metadata=metadata_payload,
        origin="mobile",
        source="mobile",
        author_user_id=777,
    )

    bot.data.update_asset(
        asset_id,
        metadata=metadata_payload,
        latitude=stored_lat,
        longitude=stored_lon,
        exif_present=True,
    )

    telegram_copy = create_sample_image_without_gps(tmp_path / "telegram.jpg")

    async def fake_download_file(self: Bot, file_id: str, dest_path):
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(telegram_copy, dest)
        return dest

    async def fake_api_request(self: Bot, method: str, payload: dict[str, Any]):
        return {"ok": True, "result": {"file_path": "dummy"}}

    reverse_calls: list[tuple[float, float]] = []

    async def fake_reverse_geocode(self: Bot, lat: float, lon: float):
        reverse_calls.append((lat, lon))
        return {"city": "Москва", "country": "Россия"}

    extract_calls: list[str] = []

    def fake_extract_gps(self: Bot, image_source):
        extract_calls.append(str(image_source))
        return None

    ingest_calls: list[dict[str, Any]] = []

    async def fake_ingest_photo(**kwargs):
        ingest_calls.append(kwargs)
        callbacks: IngestionCallbacks | None = kwargs.get("callbacks")
        overrides: dict[str, Any] = kwargs.get("input_overrides") or {}
        bot._extract_gps(kwargs.get("file_path"))
        if callbacks and callbacks.save_asset:
            callbacks.save_asset(
                {
                    "channel_id": overrides.get("channel_id", asset_channel),
                    "message_id": 404,
                    "tg_chat_id": overrides.get("tg_chat_id", asset_channel),
                    "exif_present": False,
                    "latitude": None,
                    "longitude": None,
                }
            )
        return SimpleNamespace(gps={})

    monkeypatch.setattr(Bot, "_download_file", fake_download_file)
    monkeypatch.setattr(Bot, "api_request", fake_api_request)
    monkeypatch.setattr(Bot, "_reverse_geocode", fake_reverse_geocode)
    monkeypatch.setattr(Bot, "_extract_gps", fake_extract_gps)
    monkeypatch.setattr(sys.modules["main"], "ingest_photo", fake_ingest_photo)

    job = Job(
        id=302,
        name="ingest",
        payload={"asset_id": asset_id},
        status="queued",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    await bot._job_ingest(job)

    updated_asset = bot.data.get_asset(asset_id)
    assert updated_asset is not None
    assert updated_asset.latitude == pytest.approx(stored_lat, rel=1e-6)
    assert updated_asset.longitude == pytest.approx(stored_lon, rel=1e-6)
    assert updated_asset.exif_present is True
    assert reverse_calls == [(stored_lat, stored_lon)]
    assert extract_calls, "_extract_gps should be invoked by ingestion"
    assert ingest_calls, "ingest_photo should be invoked"

    bot.db.close()


@pytest.mark.asyncio
async def test_process_upload_mobile_upload_increments_metric(tmp_path: Path) -> None:
    legacy_channel_id = -10001
    recognition_channel_id = -20001
    conn = _setup_connection(
        asset_channel_id=legacy_channel_id,
        recognition_channel_id=recognition_channel_id,
    )
    data = DataAccess(conn)
    image_path = create_sample_image(tmp_path / "metric.jpg")
    file_key = "metric-key"
    storage = StorageStub({file_key: image_path})
    telegram = TelegramStub()
    metrics = UploadMetricsRecorder()
    queue = JobQueue(conn)
    config = UploadsConfig(
        assets_channel_id=recognition_channel_id,
        vision_enabled=False,
    )
    register_upload_jobs(
        queue,
        conn,
        storage=storage,
        data=data,
        telegram=telegram,
        metrics=metrics,
        config=config,
    )

    upload_id = _prepare_upload(conn, file_key=file_key)
    job = Job(
        id=101,
        name="process_upload",
        payload={"upload_id": upload_id},
        status="queued",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    def _metric_value() -> float:
        samples_attr = observability._MOBILE_PHOTOS_TOTAL._samples
        if callable(samples_attr):
            samples = samples_attr()
            for sample in samples:
                labels = getattr(sample, "labels", None)
                if labels:
                    continue
                value = getattr(sample, "value", None)
                if value is not None:
                    return float(value)
            return 0.0
        if isinstance(samples_attr, dict):
            sample = samples_attr.get(())
            if sample is None:
                return 0.0
            value = getattr(sample, "value", None)
            if value is not None:
                return float(value)
            if isinstance(sample, tuple) and len(sample) >= 3:
                return float(sample[2])
        return 0.0

    before = _metric_value()
    await queue.handlers["process_upload"](job)
    after = _metric_value()

    assert after == before + 1


@pytest.mark.asyncio
async def test_process_upload_job_vision_failure_sets_failed_status(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    legacy_channel_id = -10002
    recognition_channel_id = -20002
    conn = _setup_connection(
        asset_channel_id=legacy_channel_id,
        recognition_channel_id=recognition_channel_id,
    )
    data = DataAccess(conn)
    image_path = create_sample_image(tmp_path / "vision.jpg")
    file_key = "vision-key"
    storage = DummyStorage({file_key: image_path})
    telegram = DummyTelegram()
    metrics = UploadMetricsRecorder()
    queue = JobQueue(conn)
    config = UploadsConfig(
        assets_channel_id=recognition_channel_id,
        vision_enabled=True,
        openai_vision_model="vision-test",
    )
    openai_client = OpenAIStub(error=RuntimeError("vision boom"))
    register_upload_jobs(
        queue,
        conn,
        storage=storage,
        data=data,
        telegram=telegram,
        openai=openai_client,
        metrics=metrics,
        config=config,
    )

    upload_id = _prepare_upload(conn, file_key=file_key)
    job = Job(
        id=2,
        name="process_upload",
        payload={"upload_id": upload_id},
        status="queued",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    with pytest.raises(RuntimeError, match="vision boom"):
        await queue.handlers["process_upload"](job)

    row = conn.execute(
        "SELECT status, error, asset_id FROM uploads WHERE id=?", (upload_id,)
    ).fetchone()
    assert row["status"] == "failed"
    assert "vision boom" in (row["error"] or "")
    assert row["asset_id"] is None
    assert metrics.counters.get("upload_process_fail_total") == 1
    assert metrics.counters.get("assets_created_total", 0) == 0

    failure_event = next(
        record for record in caplog.records if record.message == "MOBILE_UPLOAD_FAILED"
    )
    assert failure_event.upload_id == upload_id
    assert failure_event.device_id == "device-1"
    assert failure_event.source == "mobile"
    assert failure_event.size_bytes == image_path.stat().st_size
    assert failure_event.error == "vision boom"
    assert isinstance(failure_event.timestamp, str) and failure_event.timestamp


@pytest.mark.asyncio
async def test_mobile_upload_file_lifecycle_until_vision_cleanup(tmp_path: Path) -> None:
    file_key = "mobile-key"
    image_path = create_sample_image(tmp_path / "mobile.jpg")

    db_path = tmp_path / "mobile.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    apply_migrations(conn)
    conn.execute("DELETE FROM asset_channel")
    conn.execute("INSERT INTO asset_channel (channel_id) VALUES (?)", (-40001,))
    conn.execute("DELETE FROM recognition_channel")
    conn.execute("INSERT INTO recognition_channel (channel_id) VALUES (?)", (-50001,))
    conn.commit()

    data = DataAccess(conn)
    storage = StorageStub({file_key: image_path})
    telegram = TelegramStub()
    metrics = UploadMetricsRecorder()
    queue = JobQueue(conn)
    config = UploadsConfig(
        assets_channel_id=-50001,
        vision_enabled=False,
    )
    register_upload_jobs(
        queue,
        conn,
        storage=storage,
        data=data,
        telegram=telegram,
        metrics=metrics,
        config=config,
    )

    upload_id = _prepare_upload(conn, file_key=file_key)
    process_job = Job(
        id=200,
        name="process_upload",
        payload={"upload_id": upload_id},
        status="queued",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    await queue.handlers["process_upload"](process_job)

    upload_row = conn.execute(
        "SELECT status, asset_id FROM uploads WHERE id=?",
        (upload_id,),
    ).fetchone()
    assert upload_row is not None
    assert upload_row["status"] == "done"
    asset_id = upload_row["asset_id"]
    assert asset_id

    asset = data.get_asset(asset_id)
    assert asset is not None
    local_path = asset.local_path
    assert local_path
    local_path_path = Path(local_path)
    assert local_path_path.exists()
    assert storage.delete_calls == []
    assert storage.get_calls == [file_key]

    bot = Bot("vision-test", str(db_path))
    bot.dry_run = False
    bot.storage = storage
    bot.upload_metrics = UploadMetricsRecorder()
    openai_stub = OpenAIStub()
    openai_stub.api_key = "stub"
    bot.openai = openai_stub

    async def _fake_api_request(method: str, payload: dict[str, Any], files: Any | None = None):
        if method in {"copyMessage", "sendDocument"}:
            return {"ok": True, "result": {"message_id": 9001}}
        if method == "deleteMessage":
            return {"ok": True}
        return {"ok": True}

    bot.api_request = _fake_api_request  # type: ignore[assignment]
    bot.api_request_multipart = _fake_api_request  # type: ignore[assignment]

    vision_job = Job(
        id=201,
        name="vision",
        payload={"asset_id": asset_id, "tz_offset": "+00:00"},
        status="queued",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    await bot._job_vision_locked(vision_job)

    updated_asset = bot.data.get_asset(asset_id)
    assert updated_asset is not None
    vision_results = updated_asset.vision_results
    assert isinstance(vision_results, dict)
    assert vision_results.get("status") == "error"
    assert vision_results.get("stage") == "parse_result"
    error_message = vision_results.get("error") or ""
    assert "missing framing" in error_message
    assert updated_asset.local_path is None
    assert storage.delete_calls == [file_key]
    assert not local_path_path.exists()
    assert storage.get_calls == [file_key]
    assert len(openai_stub.calls) == 1

    bot.db.close()
    conn.close()


@pytest.mark.asyncio
async def test_process_upload_job_publish_failure_sets_failed_status(
    tmp_path: Path,
) -> None:
    legacy_channel_id = -10003
    recognition_channel_id = -20003
    conn = _setup_connection(
        asset_channel_id=legacy_channel_id,
        recognition_channel_id=recognition_channel_id,
    )
    data = DataAccess(conn)
    image_path = create_sample_image(tmp_path / "publish.jpg")
    file_key = "publish-key"
    storage = StorageStub({file_key: image_path})
    telegram = FailingTelegram(RuntimeError("telegram down"))
    metrics = UploadMetricsRecorder()
    queue = JobQueue(conn)
    config = UploadsConfig(
        assets_channel_id=recognition_channel_id,
        vision_enabled=False,
    )
    register_upload_jobs(
        queue,
        conn,
        storage=storage,
        data=data,
        telegram=telegram,
        metrics=metrics,
        config=config,
    )

    upload_id = _prepare_upload(conn, file_key=file_key)
    job = Job(
        id=3,
        name="process_upload",
        payload={"upload_id": upload_id},
        status="queued",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    with pytest.raises(RuntimeError, match="telegram down"):
        await queue.handlers["process_upload"](job)

    row = conn.execute(
        "SELECT status, error, asset_id FROM uploads WHERE id=?", (upload_id,)
    ).fetchone()
    assert row["status"] == "failed"
    assert "telegram down" in (row["error"] or "")
    assert row["asset_id"] is None
    assert metrics.counters.get("upload_process_fail_total") == 1


@pytest.mark.asyncio
async def test_process_upload_job_retry_after_failure(tmp_path: Path) -> None:
    legacy_channel_id = -10005
    recognition_channel_id = -20005
    conn = _setup_connection(
        asset_channel_id=legacy_channel_id,
        recognition_channel_id=recognition_channel_id,
    )
    data = DataAccess(conn)
    image_path = create_sample_image(tmp_path / "retry.jpg")
    file_key = "retry-key"
    storage = StorageStub({file_key: image_path})
    telegram = FlakyTelegram(failures=1, error=RuntimeError("telegram glitch"))
    metrics = UploadMetricsRecorder()
    queue = JobQueue(conn)
    config = UploadsConfig(
        assets_channel_id=recognition_channel_id,
        vision_enabled=False,
    )
    register_upload_jobs(
        queue,
        conn,
        storage=storage,
        data=data,
        telegram=telegram,
        metrics=metrics,
        config=config,
    )

    upload_id = _prepare_upload(conn, file_key=file_key)
    first_job = Job(
        id=4,
        name="process_upload",
        payload={"upload_id": upload_id},
        status="queued",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    with pytest.raises(RuntimeError, match="telegram glitch"):
        await queue.handlers["process_upload"](first_job)

    row = conn.execute(
        "SELECT status, error, asset_id FROM uploads WHERE id=?",
        (upload_id,),
    ).fetchone()
    assert row["status"] == "failed"
    assert "telegram glitch" in (row["error"] or "")
    assert row["asset_id"] is None

    retry_job = Job(
        id=5,
        name="process_upload",
        payload={"upload_id": upload_id},
        status="queued",
        attempts=1,
        available_at=None,
        last_error="telegram glitch",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    await queue.handlers["process_upload"](retry_job)

    row = conn.execute(
        "SELECT status, error, asset_id FROM uploads WHERE id=?",
        (upload_id,),
    ).fetchone()
    assert row["status"] == "done"
    assert row["error"] is None
    assert row["asset_id"] is not None

    assert metrics.counters.get("upload_process_fail_total") == 1
    assert metrics.counters.get("assets_created_total") == 1
    assert len(metrics.timings.get("process_upload_ms", [])) == 2


@pytest.mark.asyncio
async def test_register_upload_jobs_integration_flow(tmp_path: Path) -> None:
    legacy_channel_id = -10004
    recognition_channel_id = -20004
    conn = _setup_connection(
        asset_channel_id=legacy_channel_id,
        recognition_channel_id=recognition_channel_id,
    )
    data = DataAccess(conn)
    image_path = create_sample_image(tmp_path / "integration.jpg")
    file_key = "integration-key"
    storage = StorageStub({file_key: image_path})
    telegram = DummyTelegram()
    metrics = UploadMetricsRecorder()
    queue = JobQueue(conn, concurrency=1, poll_interval=0.01)
    config = UploadsConfig(
        assets_channel_id=recognition_channel_id,
        vision_enabled=False,
    )
    register_upload_jobs(
        queue,
        conn,
        storage=storage,
        data=data,
        telegram=telegram,
        metrics=metrics,
        config=config,
    )

    upload_id = _prepare_upload(conn, file_key=file_key)
    queue.enqueue("process_upload", {"upload_id": upload_id})

    await queue.start()
    try:
        for _ in range(100):
            row = conn.execute(
                "SELECT status, asset_id FROM uploads WHERE id=?", (upload_id,)
            ).fetchone()
            if row and row["status"] == "done" and row["asset_id"]:
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("upload job did not complete in time")
    finally:
        await queue.stop()

    assert len(telegram.calls) == 1
    assert metrics.counters.get("assets_created_total") == 1
    assert metrics.counters.get("upload_process_fail_total", 0) == 0
    job_row = conn.execute(
        "SELECT status, last_error FROM jobs_queue WHERE name='process_upload'"
    ).fetchone()
    assert job_row["status"] == "done"
    assert job_row["last_error"] is None
