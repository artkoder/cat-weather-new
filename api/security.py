from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import logging
from datetime import datetime, timezone
from time import perf_counter
from typing import Mapping, Sequence
from urllib.parse import quote

from aiohttp import web
from aiohttp.streams import StreamReader
from multidict import MultiMapping

from data_access import get_device_secret, register_nonce
from observability import record_hmac_failure


_CANONICAL_EMPTY_BODY_SHA = hashlib.sha256(b"").hexdigest()
_ALLOWED_PATH_PREFIX = "/v1/"
_TIME_WINDOW_SECONDS = 300
_NONCE_TTL_SECONDS = 600
_SKIP_ROUTES = {
    ("GET", "/v1/health"),
    ("POST", "/v1/devices/attach"),
}


def _normalize_path(path: str) -> str:
    if not path:
        return "/"
    if not path.startswith("/"):
        path = "/" + path
    if path != "/":
        path = path.rstrip("/")
        if not path:
            return "/"
    return path


def _canonical_query(params: MultiMapping[str] | Mapping[str, Sequence[str]]) -> str:
    items: list[tuple[str, str]] = []
    if isinstance(params, MultiMapping):
        keys = sorted({k for k in params})
        for key in keys:
            values = sorted(params.getall(key))  # type: ignore[attr-defined]
            for value in values:
                items.append((key, value))
    else:
        for key, values in sorted(params.items()):
            if isinstance(values, (list, tuple)):
                for value in sorted(values):
                    items.append((key, str(value)))
            else:
                items.append((key, str(values)))
    if not items:
        return "-"
    encoded = []
    for key, value in items:
        encoded.append(
            f"{quote(str(key), safe='~-._')}={quote(str(value), safe='~-._')}"
        )
    return "&".join(encoded)


def _body_sha256(body: bytes) -> str:
    if not body:
        return _CANONICAL_EMPTY_BODY_SHA
    return hashlib.sha256(body).hexdigest()


def _decode_secret(secret: str) -> bytes:
    secret = secret.strip()
    if not secret:
        raise ValueError("Device secret cannot be empty")
    lowered = secret.lower()
    if all(c in "0123456789abcdef" for c in lowered) and len(secret) % 2 == 0:
        try:
            return bytes.fromhex(secret)
        except ValueError:
            pass
    try:
        return base64.b64decode(secret, validate=True)
    except (binascii.Error, ValueError):
        return secret.encode("utf-8")


def _canonical_string(
    method: str,
    path: str,
    query: str,
    timestamp: int,
    nonce: str,
    device_id: str,
    body_sha: str,
    idempotency_key: str | None,
) -> str:
    key = idempotency_key if idempotency_key else "-"
    parts = [
        method.upper(),
        path,
        query,
        str(timestamp),
        nonce,
        device_id,
        body_sha,
        key,
    ]
    return "\n".join(parts)


def _json_error(status: int, error: str, message: str) -> web.Response:
    return web.json_response({"error": error, "message": message}, status=status)


async def _read_body(request: web.Request) -> bytes:
    body = await request.read()
    request._read_bytes = body  # type: ignore[attr-defined]
    payload = getattr(request, "_payload", None)
    if payload is not None:
        protocol = getattr(payload, "_protocol", None)
        if protocol is not None:
            loop = getattr(payload, "_loop", None)
            limit = getattr(payload, "_limit", None) or 2**16
            reader = StreamReader(protocol, limit, loop=loop)
            reader.feed_data(body)
            reader.feed_eof()
            request._payload = reader  # type: ignore[attr-defined]
    return body


def create_hmac_middleware(conn) -> web.middleware:
    """Return middleware that enforces HMAC signatures for protected routes."""

    @web.middleware
    async def middleware(request: web.Request, handler):
        start = perf_counter()
        method = request.method.upper()
        path = _normalize_path(request.rel_url.path)
        if (method, path) in _SKIP_ROUTES or not path.startswith(_ALLOWED_PATH_PREFIX):
            return await handler(request)

        headers = request.headers
        device_id = headers.get("X-Device-Id")
        timestamp_raw = headers.get("X-Timestamp")
        nonce = headers.get("X-Nonce")
        signature = headers.get("X-Signature")
        content_sha = headers.get("X-Content-SHA256")
        idempotency_key = headers.get("Idempotency-Key")

        if (
            not device_id
            or not timestamp_raw
            or not nonce
            or not signature
            or not content_sha
        ):
            record_hmac_failure("missing_headers")
            return _json_error(400, "invalid_headers", "Missing required HMAC headers.")

        content_sha = content_sha.strip().lower()
        if not (
            len(content_sha) == 64
            and all(c in "0123456789abcdef" for c in content_sha)
        ):
            record_hmac_failure("content_sha_format")
            return _json_error(
                400,
                "invalid_headers",
                "X-Content-SHA256 must be a 64-character lowercase hex digest.",
            )

        try:
            timestamp = int(timestamp_raw)
        except (TypeError, ValueError):
            record_hmac_failure("invalid_timestamp")
            return _json_error(400, "invalid_headers", "X-Timestamp must be an integer.")

        now = datetime.now(timezone.utc)
        if abs(int(now.timestamp()) - timestamp) > _TIME_WINDOW_SECONDS:
            logging.warning("SEC reject stale m=%s p=%s ts=%s", method, path, timestamp)
            record_hmac_failure("timestamp_window")
            return _json_error(401, "stale_timestamp", "Request timestamp outside allowed window.")

        if len(nonce) < 16:
            record_hmac_failure("nonce_short")
            return _json_error(400, "invalid_headers", "X-Nonce must be at least 16 characters.")

        if not (len(signature) == 64 and all(c in "0123456789abcdef" for c in signature)):
            record_hmac_failure("signature_format")
            return _json_error(400, "invalid_headers", "X-Signature must be 64 lowercase hex characters.")

        record = get_device_secret(conn, device_id=device_id)
        if not record:
            logging.warning("SEC reject missing-device id=%s", device_id)
            record_hmac_failure("unknown_device")
            return _json_error(401, "invalid_signature", "Signature verification failed.")

        secret, revoked_at = record
        if revoked_at:
            logging.warning("SEC reject revoked id=%s", device_id)
            record_hmac_failure("revoked")
            return _json_error(403, "device_revoked", "Device secret has been revoked.")

        body = await _read_body(request)
        body_sha = _body_sha256(body)
        if not hmac.compare_digest(body_sha, content_sha):
            record_hmac_failure("content_sha_mismatch")
            return _json_error(
                400,
                "invalid_body_digest",
                "X-Content-SHA256 does not match request body.",
            )
        query = _canonical_query(request.rel_url.query)
        canonical = _canonical_string(
            method,
            path,
            query,
            timestamp,
            nonce,
            device_id,
            content_sha,
            idempotency_key,
        )

        try:
            secret_bytes = _decode_secret(secret)
        except ValueError:
            logging.warning("SEC reject secret-format id=%s", device_id)
            record_hmac_failure("secret_format")
            return _json_error(403, "device_revoked", "Device secret has been revoked.")

        computed = hmac.new(secret_bytes, canonical.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(computed, signature):
            logging.warning("SEC reject signature id=%s", device_id)
            record_hmac_failure("bad_signature")
            return _json_error(401, "invalid_signature", "Signature verification failed.")

        if not register_nonce(conn, device_id=device_id, nonce=nonce, ttl_seconds=_NONCE_TTL_SECONDS):
            logging.warning("SEC reject nonce-reuse id=%s nonce=%s", device_id, nonce)
            record_hmac_failure("nonce_replay")
            return _json_error(403, "nonce_reused", "Nonce has already been used.")

        request["device_id"] = device_id
        try:
            response = await handler(request)
            latency = (perf_counter() - start) * 1000
            logging.info("SEC ok m=%s p=%s ts=%s lat=%.2fms", method, path, timestamp, latency)
            return response
        except Exception:
            latency = (perf_counter() - start) * 1000
            logging.exception("SEC handler error m=%s p=%s lat=%.2fms", method, path, latency)
            raise

    return middleware


__all__ = [
    "create_hmac_middleware",
    "_canonical_string",
    "_normalize_path",
    "_canonical_query",
    "_body_sha256",
    "_decode_secret",
]
