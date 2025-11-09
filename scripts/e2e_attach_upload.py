"""End-to-end smoke test for device attach and upload processing."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import os
import random
import secrets
import sqlite3
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import httpx

_CANONICAL_EMPTY_BODY_SHA = hashlib.sha256(b"").hexdigest()
_POLL_DELAY_RANGE = (0.5, 0.8)

_PNG_1X1_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO30jO8AAAAASUVORK5CYII="
)


@dataclass(slots=True)
class DeviceCredentials:
    device_id: str
    secret: str
    name: str


@dataclass(slots=True)
class UploadStatus:
    upload_id: str
    status: str
    error: str | None
    asset_id: str | None


def _log(message: str) -> None:
    timestamp = datetime.utcnow().isoformat(timespec="milliseconds")
    print(f"[{timestamp}] {message}")


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    text = value.strip()
    return text if text else default


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


def _canonical_query(items: Sequence[tuple[str, str]]) -> str:
    if not items:
        return "-"
    grouped: dict[str, list[str]] = {}
    for key, value in items:
        grouped.setdefault(str(key), []).append(str(value))
    encoded: list[str] = []
    for key in sorted(grouped):
        values = sorted(grouped[key])
        for value in values:
            encoded.append(f"{quote(str(key), safe='~-._')}={quote(str(value), safe='~-._')}")
    return "&".join(encoded) if encoded else "-"


def _body_sha256(body: bytes) -> str:
    if not body:
        return _CANONICAL_EMPTY_BODY_SHA
    return hashlib.sha256(body).hexdigest()


def _decode_secret(secret: str) -> bytes:
    lowered = secret.strip()
    if not lowered:
        raise ValueError("Device secret is empty")
    candidate = lowered.lower()
    if len(candidate) % 2 == 0 and all(c in "0123456789abcdef" for c in candidate):
        try:
            return bytes.fromhex(candidate)
        except ValueError:
            pass
    try:
        return base64.b64decode(lowered, validate=True)
    except Exception:
        return lowered.encode("utf-8")


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


def _build_signed_request(
    client: httpx.Client,
    *,
    method: str,
    url: str,
    device_id: str,
    secret: str,
    idempotency_key: str | None = None,
    **kwargs: Any,
) -> httpx.Response:
    headers = kwargs.pop("headers", {})
    if idempotency_key:
        headers = {**headers, "Idempotency-Key": idempotency_key}
    request = client.build_request(method, url, headers=headers, **kwargs)
    body = request.content or b""
    timestamp = int(time.time())
    nonce = secrets.token_hex(16)
    parsed = request.url
    path = _normalize_path(parsed.path)
    query_items = list(parsed.params.multi_items())  # type: ignore[attr-defined]
    query = _canonical_query([(str(k), str(v)) for k, v in query_items])
    body_sha = _body_sha256(body)
    canonical = _canonical_string(
        method,
        path,
        query,
        timestamp,
        nonce,
        device_id,
        body_sha,
        idempotency_key,
    )
    secret_bytes = _decode_secret(secret)
    signature = hmac.new(secret_bytes, canonical.encode("utf-8"), hashlib.sha256).hexdigest()
    request.headers["X-Device-Id"] = device_id
    request.headers["X-Timestamp"] = str(timestamp)
    request.headers["X-Nonce"] = nonce
    request.headers["X-Signature"] = signature
    return client.send(request)


def _run_helper_create_pairing(
    *,
    user_id: int,
    device_name: str,
    ttl: int,
    db_path: str | None,
) -> str:
    cmd = [
        sys.executable,
        "-m",
        "tools.e2e",
        "create-pairing",
        "--user-id",
        str(user_id),
        "--device-name",
        device_name,
        "--ttl",
        str(ttl),
    ]
    if db_path:
        cmd.extend(["--db-path", db_path])
    _log(f"Issuing helper command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or "unknown error"
        raise RuntimeError(f"Helper failed with exit code {result.returncode}: {detail}")
    output = result.stdout.strip().splitlines()
    if not output:
        raise RuntimeError("Helper produced no output")
    code = output[-1].strip()
    if not code:
        raise RuntimeError("Helper returned an empty pairing code")
    return code


def _attach_device(
    client: httpx.Client, base_url: str, code: str, device_name: str
) -> DeviceCredentials:
    url = f"{base_url}/devices/attach"
    _log("Calling /devices/attach")
    response = client.post(url, json={"token": code, "name": device_name})
    if response.status_code != 200:
        raise RuntimeError(
            f"Attach failed with HTTP {response.status_code}: {response.text.strip()}"
        )
    payload = response.json()
    device_id = payload.get("device_id") or payload.get("id")
    secret = payload.get("device_secret") or payload.get("secret")
    if not device_id or not secret:
        raise RuntimeError("Attach response missing device credentials")
    name = payload.get("name") or device_name
    return DeviceCredentials(
        device_id=str(device_id),
        secret=str(secret),
        name=str(name),
    )


def make_test_image() -> tuple[str, str, bytes]:
    data = base64.b64decode(_PNG_1X1_B64)
    return "e2e.png", "image/png", data


def build_multipart(file_name: str, content_type: str, data: bytes) -> tuple[bytes, str]:
    boundary = "e2e-" + secrets.token_hex(12)
    parts = []
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{file_name}"\r\n'.encode()
    )
    parts.append(f"Content-Type: {content_type}\r\n\r\n".encode())
    parts.append(data)
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    ct_header = f"multipart/form-data; boundary={boundary}"
    return body, ct_header


def _poll_status(
    client: httpx.Client,
    *,
    base_url: str,
    creds: DeviceCredentials,
    upload_id: str,
    timeout: float,
) -> UploadStatus:
    deadline = time.perf_counter() + timeout
    status_url = f"{base_url}/uploads/{upload_id}/status"
    while True:
        if time.perf_counter() > deadline:
            raise TimeoutError(f"Upload {upload_id} did not complete within {timeout:.1f}s")
        response = _build_signed_request(
            client,
            method="GET",
            url=status_url,
            device_id=creds.device_id,
            secret=creds.secret,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Status check failed ({response.status_code}): {response.text.strip()}"
            )
        payload = response.json()
        status = str(payload.get("status"))
        error = payload.get("error")
        asset_id = payload.get("asset_id")
        _log(f"Status {status} (asset={asset_id})")
        if status == "done":
            return UploadStatus(
                upload_id=upload_id,
                status=status,
                error=str(error) if error else None,
                asset_id=str(asset_id) if asset_id else None,
            )
        if status == "failed":
            raise RuntimeError(f"Upload {upload_id} failed with error: {error or 'unknown error'}")
        time.sleep(random.uniform(*_POLL_DELAY_RANGE))


def _fetch_asset_record(db_path: str, upload_id: str) -> dict[str, Any] | None:
    uri = db_path.startswith("file:")
    conn = sqlite3.connect(db_path, uri=uri)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            """
            SELECT id, file_ref, content_type, sha256, created_at
            FROM assets
            WHERE upload_id=?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (upload_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {key: row[key] for key in row.keys()}
    finally:
        conn.close()


def _verify_supabase_object(
    client: httpx.Client,
    *,
    file_ref: str,
    supabase_url: str,
    bucket: str,
) -> None:
    normalized = file_ref.lstrip("/")
    public_url = f"{supabase_url.rstrip('/')}/storage/v1/object/public/{bucket}/{normalized}"
    _log(f"Checking Supabase object availability at {public_url}")
    response = client.get(public_url)
    if response.status_code >= 400:
        raise RuntimeError(
            f"Supabase object check failed ({response.status_code}): {response.text[:200]}"
        )


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the attach->upload->process E2E flow")
    parser.add_argument(
        "--base-url", default=_env("E2E_BASE_URL"), help="API base URL, e.g. https://host/v1"
    )
    parser.add_argument(
        "--user-id", type=int, default=int(_env("E2E_USER_ID", "0")), help="Telegram user id"
    )
    parser.add_argument("--device-name", default=_env("E2E_DEVICE_NAME", "E2E Android"))
    parser.add_argument("--timeout", type=float, default=float(_env("E2E_TIMEOUT_S", "60")))
    parser.add_argument(
        "--db-path",
        default=_env("E2E_DB_PATH") or _env("DB_PATH"),
        help="Path to SQLite database for verification",
    )
    parser.add_argument(
        "--pairing-ttl",
        type=int,
        default=int(_env("E2E_PAIRING_TTL", "600")),
        help="Lifetime of generated pairing code in seconds",
    )
    parser.add_argument(
        "--storage-backend",
        default=_env("STORAGE_BACKEND", "local").lower(),
        help="Expected storage backend (local or supabase)",
    )
    parser.add_argument(
        "--supabase-url",
        default=_env("SUPABASE_URL"),
        help="Supabase project URL (required when storage backend is supabase)",
    )
    parser.add_argument(
        "--supabase-bucket",
        default=_env("SUPABASE_BUCKET", "uploads"),
        help="Supabase bucket name",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.base_url:
        parser.error("--base-url or E2E_BASE_URL must be provided")
    if not args.user_id:
        parser.error("--user-id or E2E_USER_ID must be provided")
    if not args.db_path:
        parser.error("--db-path, E2E_DB_PATH, or DB_PATH must be provided for verification")

    base_url = args.base_url.rstrip("/")
    timeout = max(float(args.timeout), 1.0)

    overall_start = time.perf_counter()

    code = _run_helper_create_pairing(
        user_id=int(args.user_id),
        device_name=args.device_name,
        ttl=int(args.pairing_ttl),
        db_path=args.db_path,
    )
    _log(f"Received pairing code {code}")

    timeout_config = httpx.Timeout(30.0, read=30.0)
    with httpx.Client(timeout=timeout_config) as client:
        attach_start = time.perf_counter()
        creds = _attach_device(client, base_url, code, args.device_name)
        attach_duration = time.perf_counter() - attach_start
        _log(f"Attached device {creds.name} (id={creds.device_id}) in {attach_duration:.2f}s")

        filename, content_type, file_bytes = make_test_image()
        body, multipart_content_type = build_multipart(
            filename,
            content_type,
            file_bytes,
        )
        idempotency_key = str(uuid4())
        body_sha = hashlib.sha256(body).hexdigest()
        _log(
            f"Uploading generated {filename} ({len(file_bytes)} bytes, sha={body_sha}) with idempotency {idempotency_key}"
        )
        upload_start = time.perf_counter()
        response = _build_signed_request(
            client,
            method="POST",
            url=f"{base_url}/uploads",
            device_id=creds.device_id,
            secret=creds.secret,
            idempotency_key=idempotency_key,
            headers={"Content-Type": multipart_content_type},
            content=body,
        )
        if response.status_code != 201:
            raise RuntimeError(
                f"Upload failed with HTTP {response.status_code}: {response.text.strip()}"
            )
        payload = response.json()
        upload_id_raw = payload.get("id")
        if not upload_id_raw:
            raise RuntimeError("Upload response missing identifier")
        upload_id = str(upload_id_raw)
        upload_duration = time.perf_counter() - upload_start
        _log(f"Upload created with id {upload_id} in {upload_duration:.2f}s")

        status_start = time.perf_counter()
        status = _poll_status(
            client,
            base_url=base_url,
            creds=creds,
            upload_id=upload_id,
            timeout=timeout,
        )
        status_duration = time.perf_counter() - status_start
        _log(
            f"Upload {status.upload_id} completed with asset {status.asset_id} in {status_duration:.2f}s"
        )

        record = _fetch_asset_record(args.db_path, upload_id)
        if not record:
            raise RuntimeError(f"Asset record for upload {upload_id} was not found in the database")
        if not status.asset_id:
            raise RuntimeError("Upload finished without reporting an asset identifier")
        if status.asset_id and record["id"] != status.asset_id:
            raise RuntimeError(
                f"Asset mismatch: API reported {status.asset_id} but DB has {record['id']}"
            )

        if args.storage_backend == "supabase":
            if not args.supabase_url:
                raise RuntimeError("SUPABASE_URL must be provided when storage backend is supabase")
            if not record.get("file_ref"):
                raise RuntimeError(
                    "Asset record does not contain file_ref for storage verification"
                )
            _verify_supabase_object(
                client,
                file_ref=str(record["file_ref"]),
                supabase_url=args.supabase_url,
                bucket=args.supabase_bucket,
            )

    total_duration = time.perf_counter() - overall_start
    _log(
        f"E2E flow succeeded in {total_duration:.2f}s (attach {attach_duration:.2f}s, upload {upload_duration:.2f}s, status {status_duration:.2f}s)"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(argv)
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - surfaced to caller
        _log(f"E2E flow failed: {exc}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
