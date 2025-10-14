from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from multidict import MultiDict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api.security import (
    _body_sha256,
    _canonical_query,
    _canonical_string,
    _decode_secret,
    _normalize_path,
    create_hmac_middleware,
)
from data_access import create_device
from main import apply_migrations


DEVICE_ID = "dev-test"
DEVICE_SECRET = "a0" * 32


def _sign(
    secret: str,
    *,
    method: str,
    path: str,
    query: MultiDict[str] | None,
    body: bytes,
    timestamp: int,
    nonce: str,
    device_id: str,
    idempotency_key: str | None,
    body_sha_override: str | None = None,
) -> str:
    body_sha = body_sha_override if body_sha_override is not None else _body_sha256(body)
    canonical = _canonical_string(
        method,
        _normalize_path(path),
        _canonical_query(query or MultiDict()),
        timestamp,
        nonce,
        device_id,
        body_sha,
        idempotency_key,
    )
    secret_bytes = _decode_secret(secret)
    return hmac.new(secret_bytes, canonical.encode("utf-8"), hashlib.sha256).hexdigest()


async def _create_client(conn: sqlite3.Connection):
    app = web.Application(middlewares=[create_hmac_middleware(conn)])

    async def uploads(request: web.Request) -> web.Response:
        payload = await request.json()
        return web.json_response({"ok": True, "payload": payload, "device": request["device_id"]})

    async def health(_: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    app.router.add_post("/v1/uploads", uploads)
    app.router.add_get("/v1/health", health)

    server = TestServer(app)
    client = TestClient(server)
    await server.start_server()
    await client.start_server()
    return app, server, client


@pytest.fixture()
def connection():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    apply_migrations(conn)
    create_device(
        conn,
        device_id=DEVICE_ID,
        user_id=1,
        name="Test",
        secret=DEVICE_SECRET,
    )
    yield conn
    conn.close()


@pytest.mark.asyncio
async def test_valid_signature(connection):
    app, server, client = await _create_client(connection)
    try:
        timestamp = int(time.time())
        nonce = "nonce-valid-1234567890"
        body = json.dumps({"hello": "world"}).encode()
        body_sha = _body_sha256(body)
        headers = {
            "X-Device-Id": DEVICE_ID,
            "X-Timestamp": str(timestamp),
            "X-Nonce": nonce,
            "Idempotency-Key": "idem-1",
            "Content-Type": "application/json",
            "X-Content-SHA256": body_sha,
        }
        headers["X-Signature"] = _sign(
            DEVICE_SECRET,
            method="POST",
            path="/v1/uploads",
            query=None,
            body=body,
            timestamp=timestamp,
            nonce=nonce,
            device_id=DEVICE_ID,
            idempotency_key=headers["Idempotency-Key"],
            body_sha_override=body_sha,
        )
        response = await client.post("/v1/uploads", data=body, headers=headers)
        assert response.status == 200
        payload = await response.json()
        assert payload["ok"] is True
        assert payload["device"] == DEVICE_ID
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_content_sha_mismatch_rejected(connection):
    app, server, client = await _create_client(connection)
    try:
        timestamp = int(time.time())
        nonce = "nonce-bad-body-123456"
        signed_body = json.dumps({"hello": "world"}).encode()
        actual_body = json.dumps({"hello": "tampered"}).encode()
        signed_body_sha = _body_sha256(signed_body)
        headers = {
            "X-Device-Id": DEVICE_ID,
            "X-Timestamp": str(timestamp),
            "X-Nonce": nonce,
            "Content-Type": "application/json",
            "X-Content-SHA256": signed_body_sha,
            "X-Signature": _sign(
                DEVICE_SECRET,
                method="POST",
                path="/v1/uploads",
                query=None,
                body=signed_body,
                timestamp=timestamp,
                nonce=nonce,
                device_id=DEVICE_ID,
                idempotency_key=None,
                body_sha_override=signed_body_sha,
            ),
        }
        response = await client.post("/v1/uploads", data=actual_body, headers=headers)
        assert response.status == 400
        data = await response.json()
        assert data["error"] == "invalid_body_digest"
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_missing_content_sha_header(connection):
    app, server, client = await _create_client(connection)
    try:
        timestamp = int(time.time())
        nonce = "nonce-missing-digest-1234"
        body = json.dumps({"hello": "world"}).encode()
        headers = {
            "X-Device-Id": DEVICE_ID,
            "X-Timestamp": str(timestamp),
            "X-Nonce": nonce,
            "Content-Type": "application/json",
            "X-Signature": _sign(
                DEVICE_SECRET,
                method="POST",
                path="/v1/uploads",
                query=None,
                body=body,
                timestamp=timestamp,
                nonce=nonce,
                device_id=DEVICE_ID,
                idempotency_key=None,
            ),
        }
        response = await client.post("/v1/uploads", data=body, headers=headers)
        assert response.status == 400
        data = await response.json()
        assert data["error"] == "invalid_headers"
        assert data["message"] == "Missing required HMAC headers."
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_timestamp_outside_window(connection):
    app, server, client = await _create_client(connection)
    try:
        timestamp = int(time.time()) - 400
        nonce = "nonce-old-1234567890"
        body = b"{}"
        body_sha = _body_sha256(body)
        headers = {
            "X-Device-Id": DEVICE_ID,
            "X-Timestamp": str(timestamp),
            "X-Nonce": nonce,
            "Content-Type": "application/json",
            "X-Content-SHA256": body_sha,
            "X-Signature": _sign(
                DEVICE_SECRET,
                method="POST",
                path="/v1/uploads",
                query=None,
                body=body,
                timestamp=timestamp,
                nonce=nonce,
                device_id=DEVICE_ID,
                idempotency_key=None,
                body_sha_override=body_sha,
            ),
        }
        response = await client.post("/v1/uploads", data=body, headers=headers)
        assert response.status == 401
        payload = await response.json()
        assert payload["error"] == "stale_timestamp"
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_reused_nonce_rejected(connection):
    app, server, client = await _create_client(connection)
    try:
        timestamp = int(time.time())
        nonce = "nonce-repeat-1234567890"
        body = b"{}"
        body_sha = _body_sha256(body)
        headers = {
            "X-Device-Id": DEVICE_ID,
            "X-Timestamp": str(timestamp),
            "X-Nonce": nonce,
            "Content-Type": "application/json",
            "X-Content-SHA256": body_sha,
            "X-Signature": _sign(
                DEVICE_SECRET,
                method="POST",
                path="/v1/uploads",
                query=None,
                body=body,
                timestamp=timestamp,
                nonce=nonce,
                device_id=DEVICE_ID,
                idempotency_key=None,
                body_sha_override=body_sha,
            ),
        }
        first = await client.post("/v1/uploads", data=body, headers=headers)
        assert first.status == 200
        second = await client.post("/v1/uploads", data=body, headers=headers)
        assert second.status == 403
        payload = await second.json()
        assert payload["error"] == "nonce_reused"
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_upload_happy_path_with_query(connection):
    app, server, client = await _create_client(connection)
    try:
        timestamp = int(time.time())
        nonce = "nonce-query-abcdef123456"
        body = json.dumps({"foo": "bar"}).encode()
        body_sha = _body_sha256(body)
        query = MultiDict([("foo", "1")])
        headers = {
            "X-Device-Id": DEVICE_ID,
            "X-Timestamp": str(timestamp),
            "X-Nonce": nonce,
            "Idempotency-Key": "idem-xyz",
            "Content-Type": "application/json",
            "X-Content-SHA256": body_sha,
        }
        headers["X-Signature"] = _sign(
            DEVICE_SECRET,
            method="POST",
            path="/v1/uploads",
            query=query,
            body=body,
            timestamp=timestamp,
            nonce=nonce,
            device_id=DEVICE_ID,
            idempotency_key=headers["Idempotency-Key"],
            body_sha_override=body_sha,
        )
        response = await client.post("/v1/uploads?foo=1", data=body, headers=headers)
        assert response.status == 200
        payload = await response.json()
        assert payload["payload"]["foo"] == "bar"
    finally:
        await client.close()
        await server.close()
