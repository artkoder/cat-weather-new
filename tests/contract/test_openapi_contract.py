from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import secrets
import sqlite3
import time
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qsl, urljoin, urlsplit

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from PIL import Image

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from api.rate_limit import TokenBucketLimiter, create_rate_limit_middleware
from api.security import (
    _body_sha256,
    _canonical_query,
    _canonical_string,
    _decode_secret,
    _normalize_path,
    create_hmac_middleware,
)
from api.uploads import UploadsConfig, setup_upload_routes
from data_access import create_pairing_token
from main import apply_migrations, attach_device
from storage import LocalStorage

schemathesis = pytest.importorskip("schemathesis")
requests = pytest.importorskip("requests")

ASSET_CHANNEL_ID = -100123
_PAIRING_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


@dataclass
class DummyJobQueue:
    handlers: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.handlers is None:
            self.handlers = {}
        self.enqueued: list[dict[str, object]] = []

    def register_handler(self, name: str, handler: object) -> None:
        self.handlers[name] = handler

    def enqueue(self, name: str, payload: dict[str, object] | None = None, **_: object) -> int:
        self.enqueued.append({"name": name, "payload": payload or {}})
        return len(self.enqueued)


@dataclass
class ContractServerContext:
    base_url: str
    conn: sqlite3.Connection
    server: TestServer

    def issue_pairing_token(
        self,
        *,
        user_id: int = 101,
        device_name: str = "Contract Pixel",
        code: str | None = None,
    ) -> str:
        token = code or "".join(secrets.choice(_PAIRING_ALPHABET) for _ in range(6))
        create_pairing_token(
            self.conn,
            code=token,
            user_id=user_id,
            device_name=device_name,
        )
        self.conn.commit()
        return token

    def url_for(self, path: str) -> str:
        return urljoin(self.base_url.rstrip("/") + "/", path.lstrip("/"))


def _make_test_image_bytes(size: tuple[int, int] = (48, 32)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color=(255, 160, 122)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _prepare_signature(
    prepared: requests.PreparedRequest,
    *,
    device_id: str,
    secret: str,
    nonce: str,
    timestamp: int,
    idempotency_key: str | None,
) -> None:
    body = prepared.body
    if body is None:
        body_bytes = b""
    elif isinstance(body, bytes):
        body_bytes = body
    else:
        body_bytes = body.encode("utf-8")

    content_sha = _body_sha256(body_bytes)

    parsed = urlsplit(prepared.url)
    query_items: dict[str, list[str]] = {}
    if parsed.query:
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            query_items.setdefault(key, []).append(value)

    canonical = _canonical_string(
        prepared.method or "",
        _normalize_path(parsed.path),
        _canonical_query(query_items),
        timestamp,
        nonce,
        device_id,
        content_sha,
        idempotency_key,
    )
    secret_bytes = _decode_secret(secret)
    signature = hmac.new(secret_bytes, canonical.encode("utf-8"), hashlib.sha256).hexdigest()

    prepared.headers.setdefault("X-Device-Id", device_id)
    prepared.headers.setdefault("X-Timestamp", str(timestamp))
    prepared.headers.setdefault("X-Nonce", nonce)
    prepared.headers["X-Content-SHA256"] = content_sha
    prepared.headers["X-Signature"] = signature


def _send_prepared(prepared: requests.PreparedRequest) -> requests.Response:
    with requests.Session() as session:
        return session.send(prepared)


@pytest.fixture(scope="session")
def openapi_schema() -> schemathesis.schemas.BaseSchema:
    contract_path = Path(__file__).resolve().parents[2] / "api/contract/openapi/openapi.yaml"
    if not contract_path.is_file():
        pytest.skip(
            "OpenAPI contract is unavailable; fetch api/contract submodule to run contract tests.",
            allow_module_level=True,
        )
    return schemathesis.from_path(str(contract_path))


@pytest.fixture
def contract_server(
    event_loop: asyncio.AbstractEventLoop,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterable[ContractServerContext]:
    for name in (
        "RL_ATTACH_IP_PER_MIN",
        "RL_ATTACH_USER_PER_MIN",
        "RL_UPLOADS_PER_MIN",
        "RL_UPLOAD_STATUS_PER_MIN",
    ):
        monkeypatch.setenv(name, "500")
    for name in (
        "RL_ATTACH_IP_WINDOW_SEC",
        "RL_ATTACH_USER_WINDOW_SEC",
        "RL_UPLOADS_WINDOW_SEC",
        "RL_UPLOAD_STATUS_WINDOW_SEC",
    ):
        monkeypatch.setenv(name, "60")

    db_path = tmp_path / "contract-tests.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    apply_migrations(conn)
    conn.execute("DELETE FROM asset_channel")
    conn.execute("INSERT OR REPLACE INTO asset_channel (channel_id) VALUES (?)", (ASSET_CHANNEL_ID,))
    conn.commit()

    class _BotStub:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self.db = connection

    app = web.Application(
        middlewares=[create_rate_limit_middleware(), create_hmac_middleware(conn)]
    )
    app["bot"] = _BotStub(conn)
    app["attach_user_rate_limiter"] = TokenBucketLimiter(500, 60)
    storage = LocalStorage(tmp_path / "uploads")
    app["storage"] = storage
    app["db"] = conn
    jobs = DummyJobQueue()
    app["jobs"] = jobs
    app["uploads_config"] = UploadsConfig(
        max_upload_mb=5.0,
        allowed_prefixes=("image/",),
        allowed_exact=("application/pdf",),
        assets_channel_id=ASSET_CHANNEL_ID,
    )

    setup_upload_routes(
        app,
        storage=storage,
        conn=conn,
        jobs=jobs,
        config=app["uploads_config"],
    )
    app.router.add_post("/v1/devices/attach", attach_device)

    server = TestServer(app, loop=event_loop)
    event_loop.run_until_complete(server.start_server())
    context = ContractServerContext(
        base_url=str(server.make_url("/")),
        conn=conn,
        server=server,
    )
    try:
        yield context
    finally:
        event_loop.run_until_complete(server.close())
        conn.close()


@pytest.mark.asyncio
async def test_contract_attach_device(contract_server: ContractServerContext, openapi_schema):
    token = contract_server.issue_pairing_token(user_id=1201, device_name="Spec Attach")
    operation = openapi_schema["/v1/devices/attach"]["post"]
    case = operation.make_case()
    case.media_type = "application/json"
    case.body = {"token": token, "name": "Spec Attach"}
    response = await asyncio.to_thread(case.call, base_url=contract_server.base_url)
    case.validate_response(response)
    payload = response.json()
    assert payload["device_id"]
    assert payload["device_secret"]


@pytest.mark.asyncio
async def test_contract_upload_and_status(contract_server: ContractServerContext, openapi_schema):
    token = contract_server.issue_pairing_token(user_id=1301, device_name="Spec Upload")
    attach_operation = openapi_schema["/v1/devices/attach"]["post"]
    attach_case = attach_operation.make_case()
    attach_case.media_type = "application/json"
    attach_case.body = {"token": token, "name": "Spec Upload"}
    attach_response = await asyncio.to_thread(attach_case.call, base_url=contract_server.base_url)
    attach_case.validate_response(attach_response)
    device_payload = attach_response.json()
    device_id = device_payload["device_id"]
    device_secret = device_payload["device_secret"]

    upload_operation = openapi_schema["/v1/uploads"]["post"]
    upload_url = contract_server.url_for("/v1/uploads")
    idempotency_key = str(uuid.uuid4())
    upload_request = requests.Request(
        method=upload_operation.method.upper(),
        url=upload_url,
        headers={
            "Accept": "application/json",
            "Idempotency-Key": idempotency_key,
        },
        files={
            "file": ("contract-test.jpg", _make_test_image_bytes(), "image/jpeg"),
        },
    )
    prepared = upload_request.prepare()
    nonce = secrets.token_hex(16)
    timestamp = int(time.time())
    _prepare_signature(
        prepared,
        device_id=device_id,
        secret=device_secret,
        nonce=nonce,
        timestamp=timestamp,
        idempotency_key=idempotency_key,
    )
    upload_response = await asyncio.to_thread(_send_prepared, prepared)
    upload_case = upload_operation.make_case()
    upload_case.validate_response(upload_response)
    assert upload_response.status_code == 202
    upload_payload = upload_response.json()
    upload_id = upload_payload["id"]
    assert upload_payload["status"]

    status_operation = openapi_schema["/v1/uploads/{id}/status"]["get"]
    status_url = contract_server.url_for(f"/v1/uploads/{upload_id}/status")
    status_request = requests.Request(
        method=status_operation.method.upper(),
        url=status_url,
        headers={"Accept": "application/json"},
    )
    prepared_status = status_request.prepare()
    status_nonce = secrets.token_hex(16)
    status_timestamp = int(time.time())
    _prepare_signature(
        prepared_status,
        device_id=device_id,
        secret=device_secret,
        nonce=status_nonce,
        timestamp=status_timestamp,
        idempotency_key=None,
    )
    status_response = await asyncio.to_thread(_send_prepared, prepared_status)
    status_case = status_operation.make_case(path_parameters={"id": upload_id})
    status_case.validate_response(status_response)
    status_payload = status_response.json()
    assert status_payload["id"] == upload_id
    assert status_payload["status"]


@pytest.mark.asyncio
async def test_contract_rejects_bad_signature(contract_server: ContractServerContext, openapi_schema):
    token = contract_server.issue_pairing_token(user_id=1401, device_name="Spec Invalid")
    attach_operation = openapi_schema["/v1/devices/attach"]["post"]
    attach_case = attach_operation.make_case()
    attach_case.media_type = "application/json"
    attach_case.body = {"token": token, "name": "Spec Invalid"}
    attach_response = await asyncio.to_thread(attach_case.call, base_url=contract_server.base_url)
    attach_case.validate_response(attach_response)
    device_payload = attach_response.json()

    status_operation = openapi_schema["/v1/uploads/{id}/status"]["get"]
    status_url = contract_server.url_for(f"/v1/uploads/{uuid.uuid4()}/status")
    status_request = requests.Request(
        method=status_operation.method.upper(),
        url=status_url,
        headers={"Accept": "application/json"},
    )
    prepared = status_request.prepare()
    nonce = secrets.token_hex(16)
    timestamp = int(time.time())
    _prepare_signature(
        prepared,
        device_id=device_payload["device_id"],
        secret=device_payload["device_secret"],
        nonce=nonce,
        timestamp=timestamp,
        idempotency_key=None,
    )
    prepared.headers["X-Signature"] = "0" * 64
    response = await asyncio.to_thread(_send_prepared, prepared)
    status_case = status_operation.make_case(path_parameters={"id": "123"})
    status_case.validate_response(response)
    assert response.status_code in {401, 403}
    error = response.json().get("error")
    assert error in {"invalid_signature", "nonce_reused", "device_revoked", "invalid_headers"}
