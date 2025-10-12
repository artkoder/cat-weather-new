from __future__ import annotations

import asyncio
import hmac
import hashlib
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api.security import (
    _body_sha256,
    _canonical_query,
    _canonical_string,
    _decode_secret,
    _normalize_path,
    create_hmac_middleware,
)
from api.uploads import UploadsConfig, register_upload_jobs, setup_upload_routes
from data_access import create_device
from main import SlidingWindowRateLimiter, apply_migrations
from jobs import JobQueue
from storage import LocalStorage


DEVICE_SECRET = "ab" * 32


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

    async def start(self, *, max_upload_mb: float = 10.0) -> "UploadTestEnv":
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        apply_migrations(conn)
        self.conn = conn
        jobs = JobQueue(conn, concurrency=1)
        register_upload_jobs(jobs, conn)
        await jobs.start()
        self.jobs = jobs

        app = web.Application(middlewares=[create_hmac_middleware(conn)])
        app['upload_rate_limiter'] = SlidingWindowRateLimiter(20, 60)
        storage = LocalStorage(base_path=self.root / "uploads")
        config = UploadsConfig(max_upload_mb=max_upload_mb)
        setup_upload_routes(app, storage=storage, conn=conn, jobs=jobs, config=config)
        self.storage = storage
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

        body, boundary = _multipart_body(b"hello cat")
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

        stored_files = list((env.root / "uploads").rglob("*"))
        assert any(path.is_file() for path in stored_files)
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_uploads_idempotency_returns_same_id(tmp_path: Path):
    env = UploadTestEnv(tmp_path)
    await env.start()
    try:
        env.create_device(device_id="device-1")

        body, boundary = _multipart_body(b"meow")
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
        env.config = UploadsConfig(max_upload_mb=0.001)
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

        body, boundary = _multipart_body(b"cat")
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
