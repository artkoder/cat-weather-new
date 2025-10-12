from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import tempfile
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ContextManager
from urllib.parse import unquote, urlparse
from uuid import uuid4

import httpx
from aiohttp import web

from data_access import (
    DataAccess,
    fetch_upload_record,
    get_device,
    get_upload,
    insert_upload,
    link_upload_asset,
    set_upload_status,
)
from observability import (
    context,
    record_job_processed,
    record_storage_put_bytes,
    record_upload_created,
    record_upload_status_change,
)
from jobs import Job, JobQueue
from openai_client import OpenAIClient
from storage import LocalStorage, Storage
from supabase_client import SupabaseClient


from ingestion import (
    TelegramClient,
    UploadIngestionContext,
    UploadMetricsRecorder,
    extract_image_metadata as _extract_image_metadata,
    ingest_photo,
)


@dataclass(slots=True)
class UploadJobDependencies:
    storage: Storage
    data: DataAccess
    telegram: TelegramClient
    openai: OpenAIClient | None = None
    supabase: SupabaseClient | None = None
    metrics: UploadMetricsRecorder | None = None


@dataclass(slots=True)
class DownloadedFile:
    path: Path
    cleanup: bool = False


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    logging.warning("Invalid boolean env %s=%s, using %s", name, raw, default)
    return default


async def _download_from_storage(storage: Storage, *, key: str) -> DownloadedFile:
    url = await storage.get_url(key=key)
    parsed = urlparse(url)
    if parsed.scheme in {"", "file"}:
        path = Path(unquote(parsed.path))
        if not path.exists():
            raise FileNotFoundError(f"Stored file missing at {path}")
        return DownloadedFile(path=path, cleanup=False)

    tmp_fd, tmp_name = tempfile.mkstemp(prefix="upload-", suffix=Path(key).suffix or "")
    os.close(tmp_fd)
    destination = Path(tmp_name)
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        with destination.open("wb") as handle:
            async for chunk in response.aiter_bytes():
                handle.write(chunk)
    return DownloadedFile(path=destination, cleanup=True)


class UploadTooLargeError(Exception):
    pass


@dataclass(slots=True)
class UploadsConfig:
    max_upload_mb: float = 10.0
    allowed_prefixes: tuple[str, ...] = ("image/",)
    allowed_exact: tuple[str, ...] = ("application/pdf",)
    assets_channel_id: int = 0
    vision_enabled: bool = False
    openai_vision_model: str | None = None
    max_image_side: int | None = None

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


def load_uploads_config() -> UploadsConfig:
    raw_max_upload = os.getenv("MAX_UPLOAD_MB", "10")
    try:
        max_upload_mb = float(raw_max_upload)
        if max_upload_mb <= 0:
            raise ValueError
    except ValueError:
        logging.warning("Invalid MAX_UPLOAD_MB=%s, defaulting to 10 MB", raw_max_upload)
        max_upload_mb = 10.0

    channel_raw = os.getenv("ASSETS_CHANNEL_ID")
    if not channel_raw:
        raise RuntimeError("ASSETS_CHANNEL_ID is not configured")
    try:
        assets_channel_id = int(channel_raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Invalid ASSETS_CHANNEL_ID={channel_raw}") from exc

    vision_enabled = _env_bool("VISION_ENABLED", False)
    model_raw = os.getenv("OPENAI_VISION_MODEL")
    openai_vision_model = model_raw.strip() if model_raw else None
    if openai_vision_model == "":
        openai_vision_model = None
    if vision_enabled and not openai_vision_model:
        logging.warning(
            "VISION_ENABLED is set but OPENAI_VISION_MODEL is missing; recognition will be skipped"
        )

    max_image_side_raw = os.getenv("MAX_IMAGE_SIDE")
    max_image_side: int | None = None
    if max_image_side_raw:
        try:
            parsed = int(max_image_side_raw)
            if parsed <= 0:
                raise ValueError
        except ValueError:
            logging.warning("Invalid MAX_IMAGE_SIDE=%s; ignoring downscale override", max_image_side_raw)
        else:
            max_image_side = parsed

    return UploadsConfig(
        max_upload_mb=max_upload_mb,
        assets_channel_id=assets_channel_id,
        vision_enabled=vision_enabled,
        openai_vision_model=openai_vision_model,
        max_image_side=max_image_side,
    )


def _json_error(status: int, error: str, message: str) -> web.Response:
    return web.json_response({"error": error, "message": message}, status=status)


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
    request["upload_id"] = upload_id
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

    record_upload_created()

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
        record_upload_status_change()
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
        record_upload_status_change()
        conn.commit()
        await file_part.release()
        return _json_error(500, "storage_error", "Failed to persist uploaded file.")

    digest = hasher.hexdigest()
    await file_part.release()
    record_storage_put_bytes(stats.size)
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
    upload_id = request.match_info.get("id")
    if upload_id:
        request["upload_id"] = str(upload_id)
    conn = _ensure_db(request.app)
    record = get_upload(conn, device_id=device_id, upload_id=str(upload_id))
    if not record:
        return _json_error(404, "not_found", "Upload not found.")
    payload = {
        "id": record["id"],
        "status": record["status"],
        "error": record["error"],
        "asset_id": record.get("asset_id"),
    }
    logging.info(
        "UPLOAD_STATUS id=%s status=%s 200",
        payload["id"],
        payload["status"],
    )
    return web.json_response(payload)


def register_upload_jobs(
    queue: JobQueue,
    conn,
    *,
    storage: Storage,
    data: DataAccess,
    telegram: TelegramClient,
    openai: OpenAIClient | None = None,
    supabase: SupabaseClient | None = None,
    metrics: UploadMetricsRecorder | None = None,
    config: UploadsConfig | None = None,
) -> None:
    if "process_upload" in queue.handlers:
        return

    metrics_recorder = metrics or UploadMetricsRecorder()
    uploads_config = config or load_uploads_config()
    cleanup_local_after_publish = _env_bool("CLEANUP_LOCAL_AFTER_PUBLISH", False)
    storage_is_local = isinstance(storage, LocalStorage)

    async def process_upload(job: Job) -> None:
        upload_id = str(job.payload.get("upload_id") or "")
        if not upload_id:
            logging.warning("UPLOAD job missing upload_id")
            return

        with context(upload_id=upload_id, job=job.name):
            with metrics_recorder.measure_process():
                logging.info("UPLOAD job start upload=%s", upload_id)
                try:
                    processed_path: Path | None = None
                    processed_cleanup = False
                    download: DownloadedFile | None = None
                    should_cleanup_local_download = False
                    set_upload_status(conn, id=upload_id, status="processing")
                    record_upload_status_change()
                    conn.commit()
                    record = fetch_upload_record(conn, upload_id=upload_id)
                    if not record:
                        logging.warning("UPLOAD missing record upload=%s", upload_id)
                        return
                    file_ref = record.get("file_ref")
                    if not file_ref:
                        raise RuntimeError("upload missing file_ref")

                    download = await _download_from_storage(storage, key=str(file_ref))
                    try:
                        device_id_value = record.get("device_id")
                        device_user_id: int | None = None
                        if device_id_value:
                            device_row = get_device(conn, device_id=str(device_id_value))
                            if device_row:
                                raw_user = None
                                try:
                                    raw_user = device_row["user_id"]  # type: ignore[index]
                                except (KeyError, TypeError):
                                    try:
                                        raw_user = device_row[1]  # type: ignore[index]
                                    except Exception:
                                        raw_user = None
                                if raw_user is not None:
                                    try:
                                        device_user_id = int(raw_user)
                                    except (TypeError, ValueError):
                                        device_user_id = None

                        ingestion_context = UploadIngestionContext(
                            upload_id=upload_id,
                            storage_key=str(file_ref),
                            metrics=metrics_recorder,
                            source="mobile",
                            device_id=str(device_id_value) if device_id_value else None,
                            user_id=device_user_id,
                            job_id=job.id,
                            job_name=job.name,
                        )

                        result = await ingest_photo(
                            data=data,
                            telegram=telegram,
                            openai=openai,
                            supabase=supabase,
                            config=uploads_config,
                            context=ingestion_context,
                            file_path=download.path,
                            cleanup_file=download.cleanup,
                        )

                        asset_id = result.asset_id
                        if not asset_id:
                            raise RuntimeError("asset creation failed for upload")
                        link_upload_asset(conn, upload_id=upload_id, asset_id=asset_id)
                        logging.info(
                            "UPLOAD asset stored upload=%s asset=%s sha=%s mime=%s",
                            upload_id,
                            asset_id,
                            result.sha256,
                            result.mime_type,
                        )
                        if (
                            cleanup_local_after_publish
                            and storage_is_local
                            and download
                            and not download.cleanup
                        ):
                            should_cleanup_local_download = True
                    finally:
                        if download and download.cleanup and download.path.exists():
                            with contextlib.suppress(Exception):
                                download.path.unlink()

                    set_upload_status(conn, id=upload_id, status="done")
                    record_upload_status_change()
                    conn.commit()
                    if (
                        should_cleanup_local_download
                        and download
                        and download.path.exists()
                    ):
                        with contextlib.suppress(Exception):
                            download.path.unlink()
                    logging.info("UPLOAD job done upload=%s", upload_id)
                    record_job_processed("process_upload", "ok")
                except Exception as exc:
                    logging.exception("UPLOAD job failed upload=%s", upload_id)
                    try:
                        set_upload_status(
                            conn,
                            id=upload_id,
                            status="failed",
                            error=str(exc)[:200],
                        )
                        record_upload_status_change()
                        conn.commit()
                    except Exception:
                        logging.exception(
                            "UPLOAD failed to persist error status upload=%s", upload_id
                        )
                    metrics_recorder.record_process_failure()
                    record_job_processed("process_upload", "failed")
                    raise

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
    app.router.add_post("/v1/uploads", handle_create_upload)
    app.router.add_get("/v1/uploads/{id}/status", handle_get_upload_status)
