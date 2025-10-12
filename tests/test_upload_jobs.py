import asyncio
import hashlib
import sqlite3
import sys
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import piexif
import pytest
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api.uploads import (
    UploadMetricsRecorder,
    UploadsConfig,
    _extract_image_metadata,
    register_upload_jobs,
)
from data_access import DataAccess, create_device, insert_upload
from jobs import Job, JobQueue
from main import apply_migrations


@pytest.fixture(autouse=True)
def _patch_pillow_exif(monkeypatch: pytest.MonkeyPatch) -> None:
    original_getexif = Image.Image.getexif

    gps_tag = 34853
    lat_tag = 2
    lon_tag = 4

    def _patched_getexif(self: Image.Image, *args: Any, **kwargs: Any):  # type: ignore[override]
        exif_bytes = getattr(self, "info", {}).get("exif")
        if exif_bytes:
            data = piexif.load(exif_bytes)
            combined: dict[int, Any] = {}
            for ifd_key in ("0th", "Exif"):
                for tag_id, value in data.get(ifd_key, {}).items():
                    combined[int(tag_id)] = value
            gps = data.get("GPS")
            if gps:
                processed_gps: dict[int, Any] = {}
                for sub_id, raw in gps.items():
                    if sub_id in {lat_tag, lon_tag} and isinstance(raw, (list, tuple)):
                        rationals: tuple[Any, ...] = tuple(
                            Fraction(part[0], part[1])
                            if isinstance(part, (list, tuple)) and len(part) == 2 and part[1]
                            else part
                            for part in raw
                        )
                        processed_gps[int(sub_id)] = rationals
                    else:
                        processed_gps[int(sub_id)] = raw
                combined[gps_tag] = processed_gps
            return combined
        return original_getexif(self, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "getexif", _patched_getexif)


class DummyStorage:
    def __init__(self, mapping: dict[str, Path]) -> None:
        self.mapping = mapping
        self.get_calls: list[str] = []

    async def put_stream(self, *, key: str, stream, content_type: str) -> str:  # pragma: no cover - not used
        raise NotImplementedError

    async def get_url(self, *, key: str) -> str:
        self.get_calls.append(key)
        try:
            return self.mapping[key].as_uri()
        except KeyError as exc:  # pragma: no cover - defensive
            raise FileNotFoundError(key) from exc


class DummyTelegram:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.next_message_id = 100

    async def send_photo(self, *, chat_id: int, photo: Path, caption: str | None = None) -> dict[str, object]:
        message_id = self.next_message_id
        call = {
            "chat_id": chat_id,
            "photo": Path(photo),
            "caption": caption,
            "message_id": message_id,
        }
        self.calls.append(call)
        self.next_message_id += 1
        return {"message_id": message_id, "chat": {"id": chat_id}}


class FailingTelegram(DummyTelegram):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    async def send_photo(self, *, chat_id: int, photo: Path, caption: str | None = None) -> dict[str, object]:
        raise self.error


class FlakyTelegram(DummyTelegram):
    def __init__(self, *, failures: int = 1, error: Exception | None = None) -> None:
        super().__init__()
        self.failures = failures
        self.error = error or RuntimeError("transient error")

    async def send_photo(self, *, chat_id: int, photo: Path, caption: str | None = None) -> dict[str, object]:
        if self.failures > 0:
            self.failures -= 1
            raise self.error
        return await super().send_photo(chat_id=chat_id, photo=photo, caption=caption)


class DummyOpenAI:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def classify_image(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return None


def _create_sample_image(path: Path) -> Path:
    image = Image.new("RGB", (640, 480), color=(10, 20, 30))
    exif_dict = {
        "0th": {
            piexif.ImageIFD.Make: "UnitTest".encode("utf-8"),
            piexif.ImageIFD.DateTime: "2023:12:24 15:30:45",
        },
        "Exif": {
            piexif.ExifIFD.DateTimeOriginal: "2023:12:24 15:30:45",
            piexif.ExifIFD.DateTimeDigitized: "2023:12:24 15:30:45",
        },
        "GPS": {
            piexif.GPSIFD.GPSLatitudeRef: "N",
            piexif.GPSIFD.GPSLatitude: [(55, 1), (30, 1), (0, 1)],
            piexif.GPSIFD.GPSLongitudeRef: "E",
            piexif.GPSIFD.GPSLongitude: [(37, 1), (36, 1), (0, 1)],
        },
        "1st": {},
        "thumbnail": None,
    }
    exif_bytes = piexif.dump(exif_dict)
    image.save(path, format="JPEG", exif=exif_bytes)
    return path


def _setup_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_migrations(conn)
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
async def test_extract_image_metadata_reads_jpeg_exif(tmp_path: Path) -> None:
    image_path = _create_sample_image(tmp_path / "sample.jpg")

    mime_type, width, height, exif_payload, gps_payload = _extract_image_metadata(image_path)

    assert mime_type == "image/jpeg"
    assert width == 640
    assert height == 480
    assert exif_payload["DateTimeOriginal"] == "2023:12:24 15:30:45"
    assert pytest.approx(gps_payload["latitude"], rel=1e-6) == 55.5
    assert pytest.approx(gps_payload["longitude"], rel=1e-6) == 37.6
    assert gps_payload["captured_at"].startswith("2023-12-24T15:30:45")


@pytest.mark.asyncio
async def test_process_upload_job_success_records_asset(tmp_path: Path) -> None:
    conn = _setup_connection()
    data = DataAccess(conn)
    image_path = _create_sample_image(tmp_path / "asset.jpg")
    file_key = "sample-key"
    storage = DummyStorage({file_key: image_path})
    telegram = DummyTelegram()
    metrics = UploadMetricsRecorder()
    queue = JobQueue(conn)
    config = UploadsConfig(assets_channel_id=-10001, vision_enabled=False)
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

    assert len(telegram.calls) == 1
    assert telegram.calls[0]["chat_id"] == config.assets_channel_id
    assert storage.get_calls == [file_key]

    asset = data.get_asset(row["asset_id"])
    assert asset is not None
    assert asset.tg_chat_id == config.assets_channel_id
    assert asset.message_id == telegram.calls[0]["message_id"]
    assert asset.payload.get("tg_chat_id") == config.assets_channel_id
    assert asset.payload.get("message_id") == telegram.calls[0]["message_id"]

    fetched = data.get_asset_by_message(
        config.assets_channel_id, telegram.calls[0]["message_id"]
    )
    assert fetched is not None
    assert fetched.id == asset.id

    assert metrics.counters.get("assets_created_total") == 1
    assert metrics.counters.get("upload_process_fail_total", 0) == 0
    assert "process_upload_ms" in metrics.timings


@pytest.mark.asyncio
async def test_process_upload_job_vision_failure_sets_failed_status(tmp_path: Path) -> None:
    conn = _setup_connection()
    data = DataAccess(conn)
    image_path = _create_sample_image(tmp_path / "vision.jpg")
    file_key = "vision-key"
    storage = DummyStorage({file_key: image_path})
    telegram = DummyTelegram()
    metrics = UploadMetricsRecorder()
    queue = JobQueue(conn)
    config = UploadsConfig(
        assets_channel_id=-10002,
        vision_enabled=True,
        openai_vision_model="vision-test",
    )
    openai_client = DummyOpenAI(error=RuntimeError("vision boom"))
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


@pytest.mark.asyncio
async def test_process_upload_job_publish_failure_sets_failed_status(tmp_path: Path) -> None:
    conn = _setup_connection()
    data = DataAccess(conn)
    image_path = _create_sample_image(tmp_path / "publish.jpg")
    file_key = "publish-key"
    storage = DummyStorage({file_key: image_path})
    telegram = FailingTelegram(RuntimeError("telegram down"))
    metrics = UploadMetricsRecorder()
    queue = JobQueue(conn)
    config = UploadsConfig(assets_channel_id=-10003, vision_enabled=False)
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
    conn = _setup_connection()
    data = DataAccess(conn)
    image_path = _create_sample_image(tmp_path / "retry.jpg")
    file_key = "retry-key"
    storage = DummyStorage({file_key: image_path})
    telegram = FlakyTelegram(failures=1, error=RuntimeError("telegram glitch"))
    metrics = UploadMetricsRecorder()
    queue = JobQueue(conn)
    config = UploadsConfig(assets_channel_id=-10005, vision_enabled=False)
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
    conn = _setup_connection()
    data = DataAccess(conn)
    image_path = _create_sample_image(tmp_path / "integration.jpg")
    file_key = "integration-key"
    storage = DummyStorage({file_key: image_path})
    telegram = DummyTelegram()
    metrics = UploadMetricsRecorder()
    queue = JobQueue(conn, concurrency=1, poll_interval=0.01)
    config = UploadsConfig(assets_channel_id=-10004, vision_enabled=False)
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
