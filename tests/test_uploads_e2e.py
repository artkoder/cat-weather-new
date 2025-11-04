from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import piexif
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

import main as main_module
from api.rate_limit import create_rate_limit_middleware
from api.security import (
    _body_sha256,
    _canonical_query,
    _canonical_string,
    _decode_secret,
    _normalize_path,
    create_hmac_middleware,
)
from api.uploads import (
    UploadMetricsRecorder,
    UploadsConfig,
    register_upload_jobs,
    setup_upload_routes,
)
from data_access import DataAccess, create_device, insert_upload, set_upload_status
from jobs import JobQueue
from main import apply_migrations
from openai_client import OpenAIResponse
from storage import LocalStorage

DEVICE_SECRET = "ab" * 32


class FakeTelegramClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.next_message_id = 1000

    async def send_photo(
        self,
        *,
        chat_id: int,
        photo: Path,
        caption: str | None = None,
    ) -> dict[str, Any]:
        with Image.open(photo) as img:
            width, height = img.size
        call = {
            "chat_id": chat_id,
            "photo": Path(photo),
            "caption": caption,
            "message_id": self.next_message_id,
        }
        self.calls.append(call)
        message_id = self.next_message_id
        self.next_message_id += 1
        file_id = f"photo-{message_id}"
        photo_sizes = [
            {
                "file_id": f"{file_id}-s",
                "file_unique_id": f"{file_id}-s-uniq",
                "width": max(1, width // 2),
                "height": max(1, height // 2),
                "file_size": max(1, (width // 2) * (height // 2) * 2),
                "mime_type": "image/jpeg",
            },
            {
                "file_id": file_id,
                "file_unique_id": f"{file_id}-uniq",
                "width": width,
                "height": height,
                "file_size": max(1, width * height * 3),
                "mime_type": "image/jpeg",
            },
        ]
        call["photo_sizes"] = photo_sizes
        call["file_id"] = file_id
        return {"message_id": message_id, "chat": {"id": chat_id}, "photo": photo_sizes}


class FakeOpenAIClient:
    def __init__(self, response: OpenAIResponse | None = None) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def classify_image(self, **kwargs: Any) -> OpenAIResponse | None:
        self.calls.append(kwargs)
        return self.response


class FakeSupabaseClient:
    def __init__(self, succeed: bool = True) -> None:
        self.succeed = succeed
        self.calls: list[dict[str, Any]] = []

    async def insert_token_usage(self, **kwargs: Any) -> tuple[bool, dict[str, Any], str | None]:
        self.calls.append(kwargs)
        if self.succeed:
            return True, kwargs, None
        return False, kwargs, "error"


def _make_test_image_bytes(*, size: tuple[int, int] = (32, 24), color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


DEFAULT_IMAGE_BYTES = _make_test_image_bytes()


def _make_test_image_with_exif_bytes(
    *,
    size: tuple[int, int] = (64, 48),
    color: tuple[int, int, int] = (20, 30, 40),
    captured: str = "2023:12:24 15:30:45",
) -> bytes:
    buffer = BytesIO()
    image = Image.new("RGB", size, color=color)
    exif_dict = {
        "0th": {piexif.ImageIFD.DateTime: captured},
        "Exif": {
            piexif.ExifIFD.DateTimeOriginal: captured,
            piexif.ExifIFD.DateTimeDigitized: captured,
        },
        "GPS": {},
        "1st": {},
        "thumbnail": None,
    }
    exif_bytes = piexif.dump(exif_dict)
    image.save(buffer, format="JPEG", exif=exif_bytes)
    return buffer.getvalue()


EXIF_IMAGE_BYTES = _make_test_image_with_exif_bytes()


def _sign(
    *,
    secret: str,
    method: str,
    path: str,
    query: dict[str, str] | None,
    body: bytes,
    timestamp: int,
    nonce: str,
    device_id: str,
    idempotency_key: str | None,
) -> str:
    canonical = _canonical_string(
        method,
        _normalize_path(path),
        _canonical_query(query or {}),
        timestamp,
        nonce,
        device_id,
        _body_sha256(body),
        idempotency_key,
    )
    secret_bytes = _decode_secret(secret)
    return hmac.new(secret_bytes, canonical.encode("utf-8"), hashlib.sha256).hexdigest()


@dataclass
class UploadTestEnv:
    root: Path
    conn: sqlite3.Connection | None = None
    jobs: JobQueue | None = None
    client: TestClient | None = None
    server: TestServer | None = None
    storage: LocalStorage | None = None
    config: UploadsConfig | None = None
    data_access: DataAccess | None = None
    telegram: FakeTelegramClient | None = None
    metrics: UploadMetricsRecorder | None = None
    openai_client: FakeOpenAIClient | None = None
    supabase_client: FakeSupabaseClient | None = None
    assets_channel_id: int | None = None
    recognition_channel_id: int | None = None
    legacy_channel_id: int | None = None
    _env_backup: dict[str, str | None] = field(default_factory=dict)

    async def start(
        self,
        *,
        max_upload_mb: float = 10.0,
        recognition_channel_id: int | None = -200456,
        legacy_channel_id: int | None = -100123,
        telegram_client: FakeTelegramClient | None = None,
        openai_client: FakeOpenAIClient | None = None,
        supabase_client: FakeSupabaseClient | None = None,
        metrics: UploadMetricsRecorder | None = None,
        vision_enabled: bool | None = None,
        vision_model: str = "test-vision",
        cleanup_local_after_publish: bool | None = None,
    ) -> UploadTestEnv:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        apply_migrations(conn)
        self.conn = conn
        jobs = JobQueue(conn, concurrency=1)
        self.data_access = DataAccess(conn)
        self.telegram = telegram_client or FakeTelegramClient()
        self.metrics = metrics or UploadMetricsRecorder()
        self.openai_client = openai_client
        self.supabase_client = supabase_client
        vision_flag = vision_enabled if vision_enabled is not None else openai_client is not None
        self._env_backup = {
            "VISION_ENABLED": os.getenv("VISION_ENABLED"),
            "OPENAI_VISION_MODEL": os.getenv("OPENAI_VISION_MODEL"),
            "MAX_IMAGE_SIDE": os.getenv("MAX_IMAGE_SIDE"),
            "CLEANUP_LOCAL_AFTER_PUBLISH": os.getenv("CLEANUP_LOCAL_AFTER_PUBLISH"),
        }
        if vision_flag:
            os.environ["VISION_ENABLED"] = "1"
            os.environ["OPENAI_VISION_MODEL"] = vision_model
        else:
            os.environ["VISION_ENABLED"] = "0"
            os.environ.pop("OPENAI_VISION_MODEL", None)
        if cleanup_local_after_publish is None:
            os.environ.pop("CLEANUP_LOCAL_AFTER_PUBLISH", None)
        else:
            os.environ["CLEANUP_LOCAL_AFTER_PUBLISH"] = "1" if cleanup_local_after_publish else "0"
        self.recognition_channel_id = recognition_channel_id
        self.legacy_channel_id = legacy_channel_id
        target_channel_id = (
            recognition_channel_id if recognition_channel_id is not None else legacy_channel_id
        )
        self.assets_channel_id = target_channel_id

        conn.execute("DELETE FROM asset_channel")
        if legacy_channel_id is not None:
            conn.execute(
                "INSERT INTO asset_channel (channel_id) VALUES (?)",
                (legacy_channel_id,),
            )
        conn.execute("DELETE FROM recognition_channel")
        if recognition_channel_id is not None:
            conn.execute(
                "INSERT INTO recognition_channel (channel_id) VALUES (?)",
                (recognition_channel_id,),
            )
        conn.commit()
        storage = LocalStorage(base_path=self.root / "uploads")
        self.storage = storage
        register_upload_jobs(
            jobs,
            conn,
            storage=storage,
            data=self.data_access,
            telegram=self.telegram,
            openai=self.openai_client,
            supabase=self.supabase_client,
            metrics=self.metrics,
        )
        await jobs.start()
        self.jobs = jobs

        config = UploadsConfig(
            max_upload_mb=max_upload_mb,
            assets_channel_id=target_channel_id,
            vision_enabled=vision_flag,
            openai_vision_model=vision_model if vision_flag else None,
            max_image_side=None,
        )
        app = web.Application(
            middlewares=[create_hmac_middleware(conn), create_rate_limit_middleware()],
            client_max_size=config.max_upload_bytes + 1024,
        )
        setup_upload_routes(app, storage=storage, conn=conn, jobs=jobs, config=config)
        self.config = config

        server = TestServer(app)
        client = TestClient(server)
        await server.start_server()
        await client.start_server()
        self.server = server
        self.client = client
        return self

    async def close(self) -> None:
        if self.client:
            await self.client.close()
        if self.server:
            await self.server.close()
        if self.jobs:
            await self.jobs.stop()
        if self.conn:
            self.conn.close()
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._env_backup.clear()

    def create_device(self, *, device_id: str, secret: str = DEVICE_SECRET) -> None:
        assert self.conn is not None
        create_device(
            self.conn,
            device_id=device_id,
            user_id=101,
            name="Android",
            secret=secret,
        )
        self.conn.commit()


def _multipart_body(content: bytes, *, filename: str = "photo.jpg", content_type: str = "image/jpeg") -> tuple[bytes, str]:
    boundary = "catweatherboundary"
    parts = [
        f"--{boundary}\r\n",
        f"Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n",
        f"Content-Type: {content_type}\r\n\r\n",
    ]
    closing = ["\r\n", f"--{boundary}--\r\n"]
    body = "".join(parts).encode("utf-8") + content + "".join(closing).encode("utf-8")
    return body, boundary


async def _signed_post(
    env: UploadTestEnv,
    *,
    path: str,
    body: bytes,
    boundary: str,
    device_id: str,
    secret: str,
    idempotency_key: str,
    extra_headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    assert env.client is not None
    timestamp = int(time.time())
    nonce = f"nonce-{time.time_ns()}"
    content_sha = _body_sha256(body)
    signature = _sign(
        secret=secret,
        method="POST",
        path=path,
        query=None,
        body=body,
        timestamp=timestamp,
        nonce=nonce,
        device_id=device_id,
        idempotency_key=idempotency_key,
    )
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "X-Device-Id": device_id,
        "X-Timestamp": str(timestamp),
        "X-Nonce": nonce,
        "Idempotency-Key": idempotency_key,
        "X-Signature": signature,
        "X-Content-SHA256": content_sha,
    }
    if extra_headers:
        headers.update(extra_headers)
    response = await env.client.post(path, data=body, headers=headers)
    payload = await response.json()
    return response.status, payload


async def _signed_get(
    env: UploadTestEnv,
    *,
    path: str,
    device_id: str,
    secret: str,
) -> tuple[int, dict[str, Any]]:
    assert env.client is not None
    timestamp = int(time.time())
    nonce = f"nonce-{time.time_ns()}"
    body = b""
    content_sha = _body_sha256(body)
    signature = _sign(
        secret=secret,
        method="GET",
        path=path,
        query=None,
        body=body,
        timestamp=timestamp,
        nonce=nonce,
        device_id=device_id,
        idempotency_key=None,
    )
    headers = {
        "X-Device-Id": device_id,
        "X-Timestamp": str(timestamp),
        "X-Nonce": nonce,
        "X-Signature": signature,
        "X-Content-SHA256": content_sha,
    }
    response = await env.client.get(path, headers=headers)
    payload = await response.json()
    return response.status, payload


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_app_registers_process_upload_and_completes_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploads_dir = tmp_path / "storage"

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    monkeypatch.setenv("WEBHOOK_URL", "https://example.com/bot")
    monkeypatch.setattr(main_module, "DB_PATH", str(tmp_path / "bot.db"))

    def _fake_storage_from_env(*, base_path=None, supabase=None):
        return LocalStorage(base_path=uploads_dir)

    monkeypatch.setattr(main_module, "create_storage_from_env", _fake_storage_from_env)

    app = main_module.create_app()
    bot = app["bot"]

    async def _fake_publish(self, chat_id, local_path, caption, *, caption_entities=None):
        return {"message_id": 123, "chat": {"id": chat_id}}, "original"

    monkeypatch.setattr(bot, "_publish_as_photo", MethodType(_fake_publish, bot))

    assert app["uploads_config"] is bot.uploads_config
    assert "process_upload" in bot.jobs.handlers

    bot.db.execute("DELETE FROM asset_channel")
    bot.db.execute("DELETE FROM recognition_channel")
    bot.db.execute("INSERT INTO asset_channel (channel_id) VALUES (?)", (-100123,))
    bot.db.execute("INSERT INTO recognition_channel (channel_id) VALUES (?)", (-200123,))
    bot.db.commit()

    create_device(
        bot.db,
        device_id="device-1",
        user_id=101,
        name="Android",
        secret=DEVICE_SECRET,
    )
    bot.db.commit()

    server = TestServer(app)
    client = TestClient(server)
    await server.start_server()
    await client.start_server()
    helper_env = SimpleNamespace(client=client)

    try:
        body, boundary = _multipart_body(EXIF_IMAGE_BYTES)
        status, payload = await _signed_post(
            helper_env,
            path="/v1/uploads",
            body=body,
            boundary=boundary,
            device_id="device-1",
            secret=DEVICE_SECRET,
            idempotency_key="idem-create-app",
        )

        assert status == 202
        assert payload["status"] == "queued"
        upload_id = payload["id"]

        for _ in range(50):
            status_check, status_payload = await _signed_get(
                helper_env,
                path=f"/v1/uploads/{upload_id}/status",
                device_id="device-1",
                secret=DEVICE_SECRET,
            )
            assert status_check == 200
            if status_payload["status"] == "done":
                break
            await asyncio.sleep(0.1)
        else:
            pytest.fail("process_upload did not complete in time")
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_uploads_rejects_when_channel_missing(tmp_path: Path) -> None:
    env = UploadTestEnv(tmp_path)
    await env.start(recognition_channel_id=None, legacy_channel_id=None)
    try:
        env.create_device(device_id="device-1")

        body, boundary = _multipart_body(EXIF_IMAGE_BYTES)
        status, payload = await _signed_post(
            env,
            path="/v1/uploads",
            body=body,
            boundary=boundary,
            device_id="device-1",
            secret=DEVICE_SECRET,
            idempotency_key="idem-missing",
        )

        assert status == 500
        assert payload == {
            "error": "asset_channel_not_configured",
            "message": "Recognition upload channel is not configured.",
        }
        assert env.conn is not None
        total_uploads = env.conn.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
        assert total_uploads == 0
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_uploads_records_gps_headers(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    env = UploadTestEnv(tmp_path)
    await env.start()
    try:
        env.create_device(device_id="device-1")

        body, boundary = _multipart_body(EXIF_IMAGE_BYTES)
        caplog.set_level(logging.INFO)
        status, payload = await _signed_post(
            env,
            path="/v1/uploads",
            body=body,
            boundary=boundary,
            device_id="device-1",
            secret=DEVICE_SECRET,
            idempotency_key="idem-gps",
            extra_headers={"X-Has-GPS": "0", "X-EXIF-Source": "ios"},
        )

        assert status == 202
        upload_id = payload["id"]
        assert env.conn is not None
        row = env.conn.execute(
            "SELECT gps_redacted_by_client FROM uploads WHERE id=?",
            (upload_id,),
        ).fetchone()
        assert row is not None
        assert row[0] == 1

        messages = "\n".join(record.getMessage() for record in caplog.records)
        assert "UPLOAD gps-metadata" in messages
        assert "MOBILE_UPLOAD_GPS_REDACTED_BY_CLIENT" in messages
    finally:
        await env.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_uploads_e2e_happy_path(tmp_path: Path):
    env = UploadTestEnv(tmp_path)
    await env.start()
    try:
        env.create_device(device_id="device-1")

        image_bytes = DEFAULT_IMAGE_BYTES
        body, boundary = _multipart_body(image_bytes)
        status, payload = await _signed_post(
            env,
            path="/v1/uploads",
            body=body,
            boundary=boundary,
            device_id="device-1",
            secret=DEVICE_SECRET,
            idempotency_key="idem-1",
        )
        assert status == 202
        assert payload["status"] == "queued"
        upload_id = payload["id"]

        deadline = time.time() + 5
        final_payload: dict[str, Any] | None = None
        while time.time() < deadline:
            status_resp, status_payload = await _signed_get(
                env,
                path=f"/v1/uploads/{upload_id}/status",
                device_id="device-1",
                secret=DEVICE_SECRET,
            )
            assert status_resp == 200
            if status_payload.get("status") in {"done", "failed"}:
                final_payload = status_payload
                break
            await asyncio.sleep(0.2)
        assert final_payload is not None
        assert final_payload["status"] == "done"
        assert "error" in final_payload
        assert final_payload["error"] is None

        stored_files = list((env.root / "uploads").rglob("*"))
        assert any(path.is_file() for path in stored_files)
        assert env.conn is not None
        row = env.conn.execute(
            "SELECT * FROM assets WHERE upload_id=?",
            (upload_id,),
        ).fetchone()
        assert row is not None
        assert final_payload["asset_id"] == row["id"]
        upload_row = env.conn.execute(
            "SELECT asset_id FROM uploads WHERE id=?",
            (upload_id,),
        ).fetchone()
        assert upload_row is not None
        assert upload_row["asset_id"] == row["id"]
        expected_sha = hashlib.sha256(image_bytes).hexdigest()
        assert row["sha256"] == expected_sha
        assert row["content_type"] == "image/jpeg"
        assert row["width"] == 32
        assert row["height"] == 24
        assert row["labels_json"] is None
        assert row["exif_json"] is None
        assert env.telegram is not None
        assert env.telegram.calls
        telegram_call = env.telegram.calls[0]
        expected_identifier = f"{env.assets_channel_id}:{telegram_call['message_id']}"
        assert row["tg_message_id"] == expected_identifier
        payload_blob = row["payload_json"]
        assert payload_blob
        payload_map = json.loads(payload_blob)
        assert payload_map["tg_chat_id"] == env.assets_channel_id
        assert payload_map["message_id"] == telegram_call["message_id"]
        assert payload_map["file_id"] == telegram_call.get("file_id")
        assert telegram_call["chat_id"] == env.assets_channel_id
        assert telegram_call["photo"].exists()
        assert "Новая загрузка" in (telegram_call["caption"] or "")
        assert env.metrics is not None
        assert env.metrics.counters.get("assets_created_total") == 1
        assert env.metrics.counters.get("upload_process_fail_total", 0) == 0
        assert "process_upload_ms" in env.metrics.timings
        assert env.metrics.counters.get("vision_tokens_total", 0) == 0

        queued_jobs = env.conn.execute(
            "SELECT name, payload FROM jobs_queue WHERE name='ingest' ORDER BY id"
        ).fetchall()
        assert queued_jobs
        ingest_payload = json.loads(queued_jobs[0]["payload"])
        assert ingest_payload == {"asset_id": row["id"]}
    finally:
        await env.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_flag", [False, True], ids=["retain", "cleanup"])
async def test_process_upload_local_cleanup(tmp_path: Path, cleanup_flag: bool):
    env = UploadTestEnv(tmp_path)
    await env.start(cleanup_local_after_publish=cleanup_flag)
    try:
        env.create_device(device_id="device-1")

        body, boundary = _multipart_body(DEFAULT_IMAGE_BYTES)
        status, payload = await _signed_post(
            env,
            path="/v1/uploads",
            body=body,
            boundary=boundary,
            device_id="device-1",
            secret=DEVICE_SECRET,
            idempotency_key=f"idem-clean-{cleanup_flag}",
        )
        assert status == 202
        assert payload["status"] == "queued"
        upload_id = payload["id"]

        for _ in range(30):
            status_check, status_payload = await _signed_get(
                env,
                path=f"/v1/uploads/{upload_id}/status",
                device_id="device-1",
                secret=DEVICE_SECRET,
            )
            assert status_check == 200
            if status_payload["status"] == "done":
                break
            await asyncio.sleep(0.05)
        else:  # pragma: no cover - defensive
            pytest.fail("process_upload did not complete in time")

        assert env.conn is not None
        row = env.conn.execute(
            "SELECT file_ref FROM uploads WHERE id=?",
            (upload_id,),
        ).fetchone()
        assert row is not None
        stored_path = env.root / "uploads" / row["file_ref"]
        if cleanup_flag:
            assert not stored_path.exists()
        else:
            assert stored_path.exists()
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_uploads_idempotency_returns_same_id(tmp_path: Path):
    env = UploadTestEnv(tmp_path)
    await env.start()
    try:
        env.create_device(device_id="device-1")

        body, boundary = _multipart_body(DEFAULT_IMAGE_BYTES)
        status1, payload1 = await _signed_post(
            env,
            path="/v1/uploads",
            body=body,
            boundary=boundary,
            device_id="device-1",
            secret=DEVICE_SECRET,
            idempotency_key="idem-42",
        )
        assert status1 == 202
        assert payload1["status"] == "queued"
        status2, payload2 = await _signed_post(
            env,
            path="/v1/uploads",
            body=body,
            boundary=boundary,
            device_id="device-1",
            secret=DEVICE_SECRET,
            idempotency_key="idem-42",
        )
        assert status2 == 409
        assert payload2["error"] == "conflict"
        assert payload2["message"] == "An upload with this idempotency key already exists."
        assert payload2["id"] == payload1["id"]
        assert payload2["status"] in {"queued", "processing", "done", "failed"}
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_uploads_reject_large_file(tmp_path: Path):
    env = UploadTestEnv(tmp_path)
    await env.start()
    try:
        env.create_device(device_id="device-1")
        env.config = UploadsConfig(
            max_upload_mb=0.001,
            assets_channel_id=env.assets_channel_id,
            vision_enabled=False,
            openai_vision_model=None,
            max_image_side=None,
        )
        assert env.server is not None
        env.server.app['uploads_config'] = env.config  # type: ignore[index]

        body, boundary = _multipart_body(b"x" * 4096)
        status, payload = await _signed_post(
            env,
            path="/v1/uploads",
            body=body,
            boundary=boundary,
            device_id="device-1",
            secret=DEVICE_SECRET,
            idempotency_key="idem-big",
        )
        assert status == 413
        assert payload["error"] == "file_too_large"
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_upload_status_for_other_device_not_found(tmp_path: Path):
    env = UploadTestEnv(tmp_path)
    await env.start()
    try:
        env.create_device(device_id="device-1")
        env.create_device(device_id="device-2", secret="cd" * 32)

        body, boundary = _multipart_body(DEFAULT_IMAGE_BYTES)
        status, payload = await _signed_post(
            env,
            path="/v1/uploads",
            body=body,
            boundary=boundary,
            device_id="device-1",
            secret=DEVICE_SECRET,
            idempotency_key="idem-main",
        )
        assert status == 202
        assert payload["status"] == "queued"
        upload_id = payload["id"]

        status_resp, payload_resp = await _signed_get(
            env,
            path=f"/v1/uploads/{upload_id}/status",
            device_id="device-2",
            secret="cd" * 32,
        )
        assert status_resp == 404
        assert payload_resp["error"] == "not_found"
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_upload_status_returns_error_for_failed_upload(tmp_path: Path):
    env = UploadTestEnv(tmp_path)
    await env.start()
    try:
        env.create_device(device_id="device-1")
        assert env.conn is not None
        upload_id = insert_upload(
            env.conn,
            id="upload-error",
            device_id="device-1",
            idempotency_key="idem-error",
        )
        set_upload_status(env.conn, id=upload_id, status="processing")
        set_upload_status(env.conn, id=upload_id, status="failed", error="boom")
        env.conn.commit()

        status_resp, payload = await _signed_get(
            env,
            path=f"/v1/uploads/{upload_id}/status",
            device_id="device-1",
            secret=DEVICE_SECRET,
        )
        assert status_resp == 200
        assert payload["status"] == "failed"
        assert payload["error"] == "boom"
        assert payload["asset_id"] is None
    finally:
        await env.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_upload_processing_with_vision(tmp_path: Path):
    env = UploadTestEnv(tmp_path)
    vision_response = OpenAIResponse(
        content={"caption": "Красный цветок", "categories": ["flower", "outdoor"]},
        usage={
            "prompt_tokens": 12,
            "completion_tokens": 5,
            "total_tokens": 17,
            "request_id": "req-vision",
            "endpoint": "/v1/responses",
        },
        meta=None,
    )
    openai_client = FakeOpenAIClient(vision_response)
    supabase_client = FakeSupabaseClient()
    await env.start(
        openai_client=openai_client,
        supabase_client=supabase_client,
        vision_enabled=True,
        vision_model="vision-test",
    )
    try:
        env.create_device(device_id="device-vision")

        body, boundary = _multipart_body(EXIF_IMAGE_BYTES)
        status, payload = await _signed_post(
            env,
            path="/v1/uploads",
            body=body,
            boundary=boundary,
            device_id="device-vision",
            secret=DEVICE_SECRET,
            idempotency_key="idem-vision",
        )
        assert status == 202
        assert payload["status"] == "queued"
        upload_id = payload["id"]

        deadline = time.time() + 5
        final_payload: dict[str, Any] | None = None
        while time.time() < deadline:
            status_resp, status_payload = await _signed_get(
                env,
                path=f"/v1/uploads/{upload_id}/status",
                device_id="device-vision",
                secret=DEVICE_SECRET,
            )
            assert status_resp == 200
            if status_payload.get("status") in {"done", "failed"}:
                final_payload = status_payload
                break
            await asyncio.sleep(0.2)
        assert final_payload is not None
        assert final_payload["status"] == "done"

        assert env.conn is not None
        asset_row = env.conn.execute(
            "SELECT id, labels_json, tg_message_id, payload_json FROM assets WHERE upload_id=?",
            (upload_id,),
        ).fetchone()
        assert asset_row is not None
        assert final_payload["asset_id"] is not None
        assert final_payload["asset_id"] == asset_row["id"]
        labels = json.loads(asset_row["labels_json"])
        assert labels.get("caption") == "Красный цветок"
        assert "flower" in labels.get("categories", [])
        expected_identifier = (
            f"{env.assets_channel_id}:{env.telegram.calls[0]['message_id']}"
        )
        assert asset_row["tg_message_id"] == expected_identifier
        payload_blob = asset_row["payload_json"]
        assert payload_blob
        payload_map = json.loads(payload_blob)
        assert payload_map["tg_chat_id"] == env.assets_channel_id
        assert payload_map["message_id"] == env.telegram.calls[0]["message_id"]
        assert payload_map["file_id"] == env.telegram.calls[0].get("file_id")
        metadata_map = payload_map.get("metadata") or {}
        assert metadata_map.get("exif_datetime_original") == "2023:12:24 15:30:45"
        assert metadata_map.get("exif_datetime_digitized") == "2023:12:24 15:30:45"
        assert metadata_map.get("exif_datetime") == "2023:12:24 15:30:45"
        assert metadata_map.get("exif_datetime_best") == "2023:12:24 15:30:45"
        usage_row = env.conn.execute("SELECT model, total_tokens FROM token_usage").fetchone()
        assert usage_row is not None
        assert usage_row["model"] == "vision-test"
        assert usage_row["total_tokens"] == 17
        assert env.metrics is not None
        assert env.metrics.counters.get("assets_created_total") == 1
        assert env.metrics.counters.get("vision_tokens_total") == 17
        assert env.metrics.counters.get("upload_process_fail_total", 0) == 0
        assert "vision_ms" in env.metrics.timings
        assert "tg_ms" in env.metrics.timings
        assert env.telegram is not None
        assert env.telegram.calls
        caption = env.telegram.calls[0]["caption"] or ""
        assert "Категории" in caption
        assert "Дата съёмки: 2023-12-24T15:30:45+00:00" in caption
        assert openai_client.calls
        assert openai_client.calls[0]["model"] == "vision-test"
        assert supabase_client.calls
        supabase_meta = supabase_client.calls[0]["meta"]
        assert supabase_meta["upload_id"] == upload_id
        assert supabase_client.calls[0]["model"] == "vision-test"

        queued_jobs = env.conn.execute(
            "SELECT name, payload FROM jobs_queue WHERE name='ingest' ORDER BY id"
        ).fetchall()
        assert queued_jobs
        ingest_payload = json.loads(queued_jobs[0]["payload"])
        assert ingest_payload == {"asset_id": asset_row["id"]}
    finally:
        await env.close()
