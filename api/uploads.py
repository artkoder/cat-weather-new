from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator
from uuid import uuid4

from aiohttp import web

from data_access import get_upload, insert_upload, set_upload_status
from jobs import Job, JobQueue
from storage import Storage


class UploadTooLargeError(Exception):
    pass


@dataclass(slots=True)
class UploadsConfig:
    max_upload_mb: float = 10.0
    allowed_prefixes: tuple[str, ...] = ("image/",)
    allowed_exact: tuple[str, ...] = ("application/pdf",)

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


def load_uploads_config() -> UploadsConfig:
    raw = os.getenv("MAX_UPLOAD_MB", "10")
    try:
        value = float(raw)
        if value <= 0:
            raise ValueError
    except ValueError:
        logging.warning("Invalid MAX_UPLOAD_MB=%s, defaulting to 10 MB", raw)
        value = 10.0
    return UploadsConfig(max_upload_mb=value)


def _json_error(status: int, error: str, message: str) -> web.Response:
    return web.json_response({"error": error, "message": message}, status=status)


def _ensure_limiter(app: web.Application):
    limiter = app.get("upload_rate_limiter")
    if not limiter:
        raise RuntimeError("Upload rate limiter is not configured")
    return limiter


def _ensure_storage(app: web.Application) -> Storage:
    storage = app.get("storage")
    if not storage:
        raise RuntimeError("Storage is not configured")
    return storage


def _ensure_jobs(app: web.Application) -> JobQueue:
    jobs = app.get("jobs")
    if not jobs:
        raise RuntimeError("Job queue is not configured")
    return jobs


def _ensure_db(app: web.Application):
    conn = app.get("db")
    if not conn:
        raise RuntimeError("Database connection is not configured")
    return conn


def _ensure_config(app: web.Application) -> UploadsConfig:
    config = app.get("uploads_config")
    if not config:
        config = load_uploads_config()
        app["uploads_config"] = config
    return config


@dataclass(slots=True)
class StreamStats:
    size: int = 0


async def _iter_file(
    part,
    *,
    max_bytes: int,
    hasher,
    stats: StreamStats,
) -> AsyncIterator[bytes]:
    while True:
        chunk = await part.read_chunk(65536)
        if not chunk:
            break
        stats.size += len(chunk)
        if stats.size > max_bytes:
            raise UploadTooLargeError
        hasher.update(chunk)
        yield chunk


def _is_allowed_type(content_type: str, config: UploadsConfig) -> bool:
    normalized = content_type.lower()
    if normalized in (value.lower() for value in config.allowed_exact):
        return True
    return any(normalized.startswith(prefix.lower()) for prefix in config.allowed_prefixes)


async def handle_create_upload(request: web.Request) -> web.Response:
    device_id = request.get("device_id")
    if not device_id:
        return _json_error(401, "unauthorized", "Missing device context")
    limiter = _ensure_limiter(request.app)
    if not await limiter.allow(f"upload:{device_id}"):
        logging.warning("UPLOAD rate-limit device=%s", device_id)
        return _json_error(429, "rate_limited", "Too many uploads. Try later.")

    idempotency_key = request.headers.get("Idempotency-Key")
    if not idempotency_key or not (1 <= len(idempotency_key) <= 128):
        return _json_error(400, "invalid_idempotency_key", "Idempotency-Key header is required.")

    if request.content_type is None or not request.content_type.startswith("multipart/"):
        return _json_error(400, "invalid_content_type", "Expected multipart/form-data payload.")

    conn = _ensure_db(request.app)
    config = _ensure_config(request.app)

    reader = await request.multipart()
    file_part = None
    async for part in reader:
        if part.name == "file":
            file_part = part
            break
    if not file_part:
        return _json_error(400, "missing_file", "Multipart field 'file' is required.")

    content_type = file_part.headers.get("Content-Type", "application/octet-stream")
    if not _is_allowed_type(content_type, config):
        await file_part.release()
        return _json_error(400, "unsupported_media_type", "Unsupported file type.")

    upload_id = str(uuid4())
    now = datetime.now(timezone.utc)
    storage_key = f"{now:%Y/%m}/{upload_id}"

    created_id = insert_upload(
        conn,
        id=upload_id,
        device_id=device_id,
        idempotency_key=idempotency_key,
        file_ref=storage_key,
    )
    conn.commit()

    if created_id != upload_id:
        logging.info("UPLOAD idempotent-return device=%s upload=%s", device_id, created_id)
        await file_part.release()
        return web.json_response({"id": created_id}, status=201)

    storage = _ensure_storage(request.app)
    jobs = _ensure_jobs(request.app)
    hasher = hashlib.sha256()
    stats = StreamStats()
    stream = _iter_file(
        file_part,
        max_bytes=config.max_upload_bytes,
        hasher=hasher,
        stats=stats,
    )

    try:
        await storage.put_stream(
            key=storage_key,
            stream=stream,
            content_type=content_type,
        )
    except UploadTooLargeError:
        logging.warning("UPLOAD too-large device=%s upload=%s", device_id, upload_id)
        set_upload_status(conn, id=upload_id, status="failed", error="file_too_large")
        conn.commit()
        await file_part.release()
        return _json_error(
            413,
            "file_too_large",
            f"Uploaded file exceeds the allowed size of {config.max_upload_mb:.1f} MB.",
        )
    except Exception:
        logging.exception("UPLOAD storage-error device=%s upload=%s", device_id, upload_id)
        set_upload_status(conn, id=upload_id, status="failed", error="storage_error")
        conn.commit()
        await file_part.release()
        return _json_error(500, "storage_error", "Failed to persist uploaded file.")

    digest = hasher.hexdigest()
    await file_part.release()
    logging.info(
        "UPLOAD stored device=%s upload=%s bytes=%s sha=%s",
        device_id,
        upload_id,
        stats.size,
        digest,
    )
    jobs.enqueue("process_upload", {"upload_id": upload_id})
    return web.json_response({"id": upload_id}, status=201)


async def handle_get_upload_status(request: web.Request) -> web.Response:
    device_id = request.get("device_id")
    if not device_id:
        return _json_error(401, "unauthorized", "Missing device context")
    limiter = request.app.get("upload_status_rate_limiter")
    if limiter and not await limiter.allow(f"upload-status:{device_id}"):
        logging.warning("UPLOAD_STATUS rate-limit device=%s", device_id)
        return _json_error(429, "rate_limited", "Too many status checks. Try later.")
    upload_id = request.match_info.get("id")
    conn = _ensure_db(request.app)
    record = get_upload(conn, device_id=device_id, upload_id=str(upload_id))
    if not record:
        return _json_error(404, "not_found", "Upload not found.")
    payload = {
        "id": record["id"],
        "status": record["status"],
        "error": record["error"],
    }
    logging.info(
        "UPLOAD_STATUS id=%s status=%s 200",
        payload["id"],
        payload["status"],
    )
    return web.json_response(payload)


def register_upload_jobs(queue: JobQueue, conn) -> None:
    if "process_upload" in queue.handlers:
        return

    async def process_upload(job: Job) -> None:
        upload_id = str(job.payload.get("upload_id"))
        if not upload_id:
            logging.warning("UPLOAD job missing upload_id")
            return
        logging.info("UPLOAD job start upload=%s", upload_id)
        set_upload_status(conn, id=upload_id, status="processing")
        conn.commit()
        await asyncio.sleep(0.5)
        set_upload_status(conn, id=upload_id, status="done")
        conn.commit()
        logging.info("UPLOAD job done upload=%s", upload_id)

    queue.register_handler("process_upload", process_upload)


def setup_upload_routes(
    app: web.Application,
    *,
    storage: Storage,
    conn,
    jobs: JobQueue,
    config: UploadsConfig | None = None,
) -> None:
    app["storage"] = storage
    app["db"] = conn
    app["jobs"] = jobs
    if config:
        app["uploads_config"] = config
    if "upload_rate_limiter" not in app:
        raise RuntimeError("Upload rate limiter must be configured on the app")
    app.router.add_post("/v1/uploads", handle_create_upload)
    app.router.add_get("/v1/uploads/{id}/status", handle_get_upload_status)
