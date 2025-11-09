import json
import shutil
import sqlite3
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api.uploads import UploadMetricsRecorder, UploadsConfig, register_upload_jobs
from data_access import DataAccess, create_device, insert_upload
from ingestion import extract_categories
from jobs import Job, JobQueue
from main import Bot, apply_migrations
from tests.fixtures.ingestion_utils import (
    DEFAULT_VISION_PAYLOAD,
    OpenAIStub,
    StorageStub,
    TelegramStub,
    compute_sha256,
    create_sample_image,
)


class SupabaseStub:
    async def insert_token_usage(self, *args, **kwargs):  # pragma: no cover - test stub
        return False, {}, "disabled"

    async def aclose(self) -> None:  # pragma: no cover - test stub
        return None


def _setup_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_migrations(conn)
    return conn


async def _run_mobile_upload(
    *,
    image_path: Path,
    telegram: TelegramStub,
    openai_payload: dict[str, object],
    vision_enabled: bool,
) -> tuple[
    DataAccess,
    sqlite3.Connection,
    str,
    UploadsConfig,
    list[dict[str, Any]],
    int,
    int,
]:
    conn = _setup_connection()
    data = DataAccess(conn)
    asset_channel_id = -100777
    recognition_channel_id = -200777
    conn.execute("DELETE FROM asset_channel")
    conn.execute(
        "INSERT INTO asset_channel (channel_id) VALUES (?)",
        (asset_channel_id,),
    )
    conn.execute("DELETE FROM recognition_channel")
    conn.execute(
        "INSERT INTO recognition_channel (channel_id) VALUES (?)",
        (recognition_channel_id,),
    )
    conn.commit()
    file_key = "shared-file"
    storage = StorageStub({file_key: image_path})
    metrics = UploadMetricsRecorder()
    queue = JobQueue(conn)
    config = UploadsConfig(
        assets_channel_id=recognition_channel_id * 2,
        vision_enabled=vision_enabled,
        openai_vision_model="vision-test" if vision_enabled else None,
    )
    openai = OpenAIStub(payload=openai_payload)
    register_upload_jobs(
        queue,
        conn,
        storage=storage,
        data=data,
        telegram=telegram,
        openai=openai,
        metrics=metrics,
        config=config,
    )

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
        "SELECT asset_id FROM uploads WHERE id=?",
        (upload_id,),
    ).fetchone()
    assert row is not None and row["asset_id"]
    asset_id = row["asset_id"]
    call_kwargs = list(openai.calls)
    assert telegram.calls and telegram.calls[0]["chat_id"] == recognition_channel_id
    config = replace(config, assets_channel_id=recognition_channel_id)
    return (
        data,
        conn,
        asset_id,
        config,
        call_kwargs,
        recognition_channel_id,
        asset_channel_id,
    )


@pytest.mark.asyncio
async def test_ingestion_helper_mobile_and_telegram_payloads_align(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = create_sample_image(tmp_path / "sample.jpg")
    expected_sha = compute_sha256(image_path)
    telegram_mobile = TelegramStub(start_message_id=500)
    openai_payload = json.loads(json.dumps(DEFAULT_VISION_PAYLOAD))

    (
        data_mobile,
        conn_mobile,
        mobile_asset_id,
        config,
        mobile_openai_calls,
        recognition_channel_id,
        legacy_channel_id,
    ) = await _run_mobile_upload(
        image_path=image_path,
        telegram=telegram_mobile,
        openai_payload=openai_payload,
        vision_enabled=False,
    )

    asset_mobile = data_mobile.get_asset(mobile_asset_id)
    assert asset_mobile is not None

    assert len(telegram_mobile.calls) == 1
    mobile_call = telegram_mobile.calls[0]
    mobile_message_id = mobile_call["message_id"]
    mobile_chat_id = mobile_call["chat_id"]

    assert asset_mobile.sha256 == expected_sha
    assert asset_mobile.source == "mobile"
    assert mobile_chat_id == config.assets_channel_id

    # Telegram ingest job setup
    monkeypatch.setenv("VISION_ENABLED", "1")
    monkeypatch.setenv("OPENAI_VISION_MODEL", "vision-test")

    bot_db_path = tmp_path / "bot.sqlite"
    bot = Bot("dummy-token", str(bot_db_path))
    try:
        bot.set_asset_channel(legacy_channel_id)
        bot.set_recognition_channel(recognition_channel_id)
        bot.uploads_config.vision_enabled = True
        bot.uploads_config.openai_vision_model = "vision-test"
        bot.upload_metrics = UploadMetricsRecorder()
        bot.openai = OpenAIStub(payload=openai_payload)
        bot.supabase = SupabaseStub()

        telegram_ingest = TelegramStub(start_message_id=mobile_message_id)

        async def fake_publish(
            chat_id: int, local_path: str, caption: str | None, *, caption_entities=None
        ):
            response = await telegram_ingest.send_photo(
                chat_id=chat_id,
                photo=Path(local_path),
                caption=caption,
            )
            return response, "photo"

        async def fake_download(file_id: str, dest_path: str | Path | None = None):
            destination = Path(dest_path or (tmp_path / f"{file_id}.jpg"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image_path, destination)
            return destination

        async def fake_api_request(*args, **kwargs):
            return {"ok": True}

        bot._publish_as_photo = fake_publish  # type: ignore[assignment]
        bot._download_file = fake_download  # type: ignore[assignment]
        bot.api_request = fake_api_request  # type: ignore[assignment]

        async def fake_reverse_geocode(*args, **kwargs):
            return None

        bot._reverse_geocode = fake_reverse_geocode  # type: ignore[assignment]
        bot.jobs.enqueue = lambda *args, **kwargs: 99  # type: ignore[assignment]

        file_meta = {
            "file_id": "tg-file",
            "file_unique_id": "tg-unique",
            "file_name": "sample.jpg",
            "mime_type": "image/jpeg",
            "file_size": image_path.stat().st_size,
            "width": asset_mobile.width,
            "height": asset_mobile.height,
        }

        recognition_asset_id = bot.data.save_asset(
            channel_id=config.assets_channel_id,
            message_id=321,
            template=None,
            hashtags=None,
            tg_chat_id=config.assets_channel_id,
            caption="Исходный пост",
            kind="photo",
            file_meta=file_meta,
            metadata=None,
            origin="recognition",
            source="telegram",
        )

        recognition_asset = bot.data.get_asset(recognition_asset_id)
        assert recognition_asset is not None
        payload = dict(recognition_asset.payload)
        payload.pop("tg_chat_id", None)
        payload.pop("message_id", None)
        payload.pop("channel_id", None)
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True) if payload else None
        bot.db.execute(
            "UPDATE assets SET tg_message_id=NULL, payload_json=? WHERE id=?",
            (payload_json, recognition_asset_id),
        )
        bot.db.commit()

        ingest_job = Job(
            id=42,
            name="ingest",
            payload={"asset_id": recognition_asset_id},
            status="queued",
            attempts=0,
            available_at=None,
            last_error=None,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        await bot._job_ingest(ingest_job)

        assert len(telegram_ingest.calls) == 1

        identifier = f"{legacy_channel_id}:{mobile_message_id}"
        row = bot.db.execute(
            "SELECT id FROM assets WHERE tg_message_id=?",
            (identifier,),
        ).fetchone()
        assert row is not None
        asset_ingest = bot.data.get_asset(row["id"])
        assert asset_ingest is not None

        assert asset_ingest.sha256 == expected_sha
        assert asset_ingest.width == asset_mobile.width
        assert asset_ingest.height == asset_mobile.height
        assert asset_ingest.tg_message_id == identifier
        assert asset_ingest.source == "telegram"

        mobile_exif = json.loads(asset_mobile.exif_json or "{}")
        ingest_exif = json.loads(asset_ingest.exif_json or "{}")
        assert mobile_exif == ingest_exif

        mobile_metadata = asset_mobile.metadata or {}
        ingest_metadata = asset_ingest.metadata or {}
        assert mobile_metadata.get("exif") == ingest_metadata.get("exif")
        assert mobile_metadata.get("gps") == ingest_metadata.get("gps")
        assert asset_ingest.latitude == asset_mobile.latitude
        assert asset_ingest.longitude == asset_mobile.longitude
        assert asset_ingest.exif_present == asset_mobile.exif_present

        mobile_labels = json.loads(asset_mobile.labels_json or "{}")
        ingest_labels = json.loads(asset_ingest.labels_json or "[]")
        expected_categories = extract_categories(mobile_labels)
        assert ingest_labels == expected_categories

        assert asset_ingest.payload.get("tg_chat_id") == legacy_channel_id
        assert asset_mobile.payload.get("tg_chat_id") == recognition_channel_id
        assert asset_ingest.payload.get("message_id") == asset_mobile.payload.get("message_id")

        assert not mobile_openai_calls
        assert not bot.openai.calls
    finally:
        await bot.close()
        conn_mobile.close()
