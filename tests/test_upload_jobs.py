import asyncio
import hashlib
import json
import logging
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import piexif
import pytest
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

import observability
from api.uploads import (
    UploadMetricsRecorder,
    UploadsConfig,
    _extract_image_metadata,
    register_upload_jobs,
)
from data_access import DataAccess, create_device, insert_upload
from jobs import Job, JobQueue
from main import Bot, apply_migrations
from ingestion import IngestionCallbacks
from tests.fixtures.ingestion_utils import (
    OpenAIStub,
    StorageStub,
    TelegramStub,
    create_sample_image,
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

    async def send_photo(self, *, chat_id: int, photo: Path, caption: str | None = None) -> dict[str, object]:
        raise self.error


class FlakyTelegram(TelegramStub):
    def __init__(self, *, failures: int = 1, error: Exception | None = None) -> None:
        super().__init__()
        self.failures = failures
        self.error = error or RuntimeError("transient error")

    async def send_photo(self, *, chat_id: int, photo: Path, caption: str | None = None) -> dict[str, object]:
        if self.failures > 0:
            self.failures -= 1
            raise self.error
        return await super().send_photo(chat_id=chat_id, photo=photo, caption=caption)


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


def _prepare_upload(conn: sqlite3.Connection, *, file_key: str) -> str:
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
    )
    conn.commit()
    return upload_id


@pytest.mark.asyncio
async def test_extract_image_metadata_reads_jpeg_exif(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    image_path = create_sample_image(tmp_path / "sample.jpg")

    with caplog.at_level(logging.INFO):
        mime_type, width, height, exif_payload, gps_payload = _extract_image_metadata(
            image_path
        )

    assert mime_type == "image/jpeg"
    assert width == 640
    assert height == 480
    assert exif_payload["DateTimeOriginal"] == "2023:12:24 15:30:45"
    assert pytest.approx(gps_payload["latitude"], rel=1e-6) == 55.5
    assert pytest.approx(gps_payload["longitude"], rel=1e-6) == 37.6
    assert gps_payload["captured_at"].startswith("2023-12-24T15:30:45")

    metadata_log = next(
        record for record in caplog.records if record.message == "MOBILE_IMAGE_METADATA"
    )
    assert metadata_log.path.endswith("sample.jpg")
    assert metadata_log.gps_present is True
    assert metadata_log.latitude == pytest.approx(55.5, rel=1e-6)
    assert metadata_log.longitude == pytest.approx(37.6, rel=1e-6)

    gps_log = next(record for record in caplog.records if record.message == "MOBILE_GPS_EXIF")
    assert gps_log.path.endswith("sample.jpg")
    assert gps_log.gps_tags["GPSLatitudeRef"] == "N"
    assert gps_log.gps_tags["GPSLongitudeRef"] == "E"


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

    _, _, _, exif_payload, coords = _extract_image_metadata(image_path)

    assert coords["latitude"] == pytest.approx(-55.25, rel=1e-6)
    expected_lon = -(37 + 36 / 60.0 + 6 / 3600.0)
    assert coords["longitude"] == pytest.approx(expected_lon, rel=1e-6)

    gps_info = exif_payload["GPSInfo"]
    assert gps_info["GPSLatitudeRef"] == "S"
    assert gps_info["GPSLongitudeRef"] == "W"
    assert gps_info["GPSLatitude"] == [
        [[55, 1], [1, 1]],
        [[30, 1], [2, 1]],
    ]
    assert gps_info["GPSLongitude"] == [
        [[37, 1], [1, 1]],
        [[36, 1], [1, 1]],
        [[12, 1], [2, 1]],
    ]


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

    fetched = data.get_asset_by_message(
        config.assets_channel_id, telegram.calls[0]["message_id"]
    )
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

    raw_log = next(
        record for record in caplog.records if record.message == "MOBILE_EXIF_RAW"
    )
    assert raw_log.asset_id == row["asset_id"]
    assert raw_log.upload_id == upload_id
    raw_exif = json.loads(raw_log.exif_raw)
    raw_gps = json.loads(raw_log.gps_raw)
    assert raw_exif
    assert raw_gps
    assert raw_gps.get("latitude") == pytest.approx(55.5, rel=1e-6)
    assert raw_gps.get("longitude") == pytest.approx(37.6, rel=1e-6)

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
    image_path = create_sample_image_without_gps(tmp_path / "asset-nogps.jpg")
    file_key = "nogps-key"
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
    gps_metadata = metadata.get("gps")
    assert not gps_metadata
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

    raw_log = next(
        record for record in caplog.records if record.message == "MOBILE_EXIF_RAW"
    )
    assert raw_log.asset_id == row["asset_id"]
    assert raw_log.upload_id == upload_id
    assert json.loads(raw_log.gps_raw) == {}

    metadata_log = next(
        record
        for record in caplog.records
        if record.message == "MOBILE_IMAGE_METADATA" and record.gps_present is False
    )
    assert metadata_log.path.endswith("asset-nogps.jpg")


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

    ingest_calls: list[dict[str, Any]] = []

    async def fake_ingest_photo(**kwargs):
        ingest_calls.append(kwargs)
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
    monkeypatch.setattr(
        sys.modules["main"], "ingest_photo", fake_ingest_photo
    )

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
