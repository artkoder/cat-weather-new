from __future__ import annotations

import asyncio
import hmac
import hashlib
import asyncio
import hashlib
import hmac
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

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
from main import SlidingWindowRateLimiter, apply_migrations
from jobs import JobQueue
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
        call = {
            "chat_id": chat_id,
            "photo": Path(photo),
            "caption": caption,
            "message_id": self.next_message_id,
        }
        self.calls.append(call)
        message_id = self.next_message_id
        self.next_message_id += 1
        return {"message_id": message_id, "chat": {"id": chat_id}}


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
    assets_channel_id: int = -100123
    _env_backup: dict[str, str | None] = field(default_factory=dict)

    async def start(
        self,
        *,
        max_upload_mb: float = 10.0,
        assets_channel_id: int = -100123,
        telegram_client: FakeTelegramClient | None = None,
        openai_client: FakeOpenAIClient | None = None,
        supabase_client: FakeSupabaseClient | None = None,
        metrics: UploadMetricsRecorder | None = None,
        vision_enabled: bool | None = None,
        vision_model: str = "test-vision",
    ) -> "UploadTestEnv":
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
            "ASSETS_CHANNEL_ID": os.getenv("ASSETS_CHANNEL_ID"),
            "VISION_ENABLED": os.getenv("VISION_ENABLED"),
            "OPENAI_VISION_MODEL": os.getenv("OPENAI_VISION_MODEL"),
            "MAX_IMAGE_SIDE": os.getenv("MAX_IMAGE_SIDE"),
        }
        os.environ["ASSETS_CHANNEL_ID"] = str(assets_channel_id)
        if vision_flag:
            os.environ["VISION_ENABLED"] = "1"
            os.environ["OPENAI_VISION_MODEL"] = vision_model
        else:
            os.environ["VISION_ENABLED"] = "0"
            os.environ.pop("OPENAI_VISION_MODEL", None)
        self.assets_channel_id = assets_channel_id
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

        app = web.Application(middlewares=[create_hmac_middleware(conn)])
        app['upload_rate_limiter'] = SlidingWindowRateLimiter(20, 60)
        app['upload_status_rate_limiter'] = SlidingWindowRateLimiter(5, 1)
        config = UploadsConfig(
            max_upload_mb=max_upload_mb,
            assets_channel_id=assets_channel_id,
            vision_enabled=vision_flag,
            openai_vision_model=vision_model if vision_flag else None,
            max_image_side=None,
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
) -> tuple[int, dict[str, Any]]:
    assert env.client is not None
    timestamp = int(time.time())
    nonce = f"nonce-{time.time_ns()}"
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
    }
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
    }
    response = await env.client.get(path, headers=headers)
    payload = await response.json()
    return response.status, payload


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
        assert status == 201
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
        assert row["tg_message_id"] == str(telegram_call["message_id"])
        assert telegram_call["chat_id"] == env.assets_channel_id
        assert telegram_call["photo"].exists()
        assert "Новая загрузка" in (telegram_call["caption"] or "")
        assert env.metrics is not None
        assert env.metrics.counters.get("upload.process.success") == 1
        assert "upload.process.duration" in env.metrics.timings
        assert env.metrics.counters.get("upload.vision.attempts", 0) == 0
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
        assert status1 == 201
        status2, payload2 = await _signed_post(
            env,
            path="/v1/uploads",
            body=body,
            boundary=boundary,
            device_id="device-1",
            secret=DEVICE_SECRET,
            idempotency_key="idem-42",
        )
        assert status2 == 201
        assert payload1["id"] == payload2["id"]
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
        assert status == 201
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
    finally:
        await env.close()


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

        body, boundary = _multipart_body(DEFAULT_IMAGE_BYTES)
        status, payload = await _signed_post(
            env,
            path="/v1/uploads",
            body=body,
            boundary=boundary,
            device_id="device-vision",
            secret=DEVICE_SECRET,
            idempotency_key="idem-vision",
        )
        assert status == 201
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
            "SELECT labels_json, tg_message_id FROM assets WHERE upload_id=?",
            (upload_id,),
        ).fetchone()
        assert asset_row is not None
        labels = json.loads(asset_row["labels_json"])
        assert labels.get("caption") == "Красный цветок"
        assert "flower" in labels.get("categories", [])
        assert asset_row["tg_message_id"] == str(env.telegram.calls[0]["message_id"])
        usage_row = env.conn.execute("SELECT model, total_tokens FROM token_usage").fetchone()
        assert usage_row is not None
        assert usage_row["model"] == "vision-test"
        assert usage_row["total_tokens"] == 17
        assert env.metrics is not None
        assert env.metrics.counters.get("upload.process.success") == 1
        assert env.metrics.counters.get("upload.vision.success") == 1
        assert "upload.vision.duration" in env.metrics.timings
        assert env.metrics.counters.get("upload.telegram.success") == 1
        assert env.telegram is not None
        assert env.telegram.calls
        caption = env.telegram.calls[0]["caption"] or ""
        assert "Категории" in caption
        assert openai_client.calls
        assert openai_client.calls[0]["model"] == "vision-test"
        assert supabase_client.calls
        supabase_meta = supabase_client.calls[0]["meta"]
        assert supabase_meta["upload_id"] == upload_id
        assert supabase_client.calls[0]["model"] == "vision-test"
    finally:
        await env.close()
