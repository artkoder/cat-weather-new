from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import tempfile
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

import httpx
from aiohttp import web

from data_access import (
    DataAccess,
    fetch_upload_record,
    get_asset_channel_id,
    get_device,
    get_recognition_channel_id,
    get_upload,
    insert_upload,
    link_upload_asset,
    set_upload_status,
)
from ingestion import (
    TelegramClient,
    UploadIngestionContext,
    UploadMetricsRecorder,
    ingest_photo,
)
from ingestion import (
    extract_exif_datetimes as _extract_exif_datetimes,
)
from jobs import Job, JobQueue
from observability import (
    context,
    record_job_processed,
    record_mobile_photo_ingested,
    record_storage_put_bytes,
    record_upload_created,
    record_upload_status_change,
)
from openai_client import OpenAIClient
from storage import LocalStorage, Storage
from supabase_client import SupabaseClient


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
    assets_channel_id: int | None = None
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
            logging.warning(
                "Invalid MAX_IMAGE_SIDE=%s; ignoring downscale override", max_image_side_raw
            )
        else:
            max_image_side = parsed

    return UploadsConfig(
        max_upload_mb=max_upload_mb,
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
        return _json_error(415, "invalid_content_type", "Expected multipart/form-data payload.")

    conn = _ensure_db(request.app)
    config = _ensure_config(request.app)

    has_gps_header = request.headers.get("X-Has-GPS")
    exif_source_header = request.headers.get("X-EXIF-Source")
    has_gps_value = has_gps_header.strip() if has_gps_header is not None else None
    gps_redacted_by_client = has_gps_value == "0"

    recognition_channel_id = get_recognition_channel_id(conn)
    asset_channel_id = (
        recognition_channel_id if recognition_channel_id is not None else get_asset_channel_id(conn)
    )
    if asset_channel_id is None:
        logging.error(
            "MOBILE_UPLOAD_RECOGNITION_CHANNEL_MISSING - установите recognition_channel в БД",
            extra={
                "device_id": device_id,
                "source": "mobile",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        return _json_error(
            500,
            "asset_channel_not_configured",
            "Recognition upload channel is not configured.",
        )

    config = replace(config, assets_channel_id=asset_channel_id)

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
        return _json_error(415, "unsupported_media_type", "Unsupported file type.")

    upload_id = str(uuid4())
    request["upload_id"] = upload_id
    now = datetime.now(UTC)
    storage_key = f"{now:%Y/%m}/{upload_id}"

    with context(device_id=device_id, upload_id=upload_id, source="mobile"):
        created_id = insert_upload(
            conn,
            id=upload_id,
            device_id=device_id,
            idempotency_key=idempotency_key,
            file_ref=storage_key,
            gps_redacted_by_client=gps_redacted_by_client,
        )
        conn.commit()

        logging.info(
            "UPLOAD gps-metadata device=%s upload=%s has_gps=%s exif_source=%s",
            device_id,
            created_id,
            has_gps_value,
            exif_source_header,
        )
        if gps_redacted_by_client:
            logging.info(
                "MOBILE_UPLOAD_GPS_REDACTED_BY_CLIENT device=%s upload=%s has_gps=%s exif_source=%s",
                device_id,
                created_id,
                has_gps_value,
                exif_source_header,
            )
        request["upload_id"] = created_id

        if created_id != upload_id:
            logging.info("UPLOAD idempotent-return device=%s upload=%s", device_id, created_id)
            await file_part.release()
            request["upload_id"] = created_id
            existing = get_upload(conn, device_id=device_id, upload_id=created_id)
            payload: dict[str, Any] = {
                "error": "conflict",
                "message": "An upload with this idempotency key already exists.",
                "id": created_id,
            }
            if existing:
                payload["status"] = existing.get("status")
                if existing.get("error") is not None:
                    payload["upload_error"] = existing.get("error")
                if existing.get("asset_id") is not None:
                    payload["asset_id"] = existing.get("asset_id")
            return web.json_response(payload, status=409)

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
        logging.info(
            "MOBILE_UPLOAD_ACCEPTED",
            extra={
                "upload_id": upload_id,
                "device_id": device_id,
                "size_bytes": stats.size,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        return web.json_response({"id": upload_id, "status": "queued"}, status=202)


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
    }
    error_value = record.get("error")
    if error_value is not None:
        payload["error"] = error_value
    payload["asset_id"] = record.get("asset_id")
    logging.info(
        "UPLOAD_STATUS id=%s status=%s 200",
        payload["id"],
        payload["status"],
    )
    return web.json_response(payload)


def register_upload_jobs(
    jobs: JobQueue,
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
    if "process_upload" in jobs.handlers:
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

        with context(upload_id=upload_id, job=job.name, source="mobile"):
            with metrics_recorder.measure_process():
                logging.info("UPLOAD job start upload=%s", upload_id)
                device_id_str: str | None = None
                size_bytes: int | None = None
                asset_channel_id: int | None = None
                try:
                    processed_path: Path | None = None
                    processed_cleanup = False
                    download: DownloadedFile | None = None
                    should_cleanup_local_download = False
                    upload_source: str | None = None
                    set_upload_status(conn, id=upload_id, status="processing")
                    record_upload_status_change()
                    conn.commit()
                    record = fetch_upload_record(conn, upload_id=upload_id)
                    if not record:
                        logging.warning("UPLOAD missing record upload=%s", upload_id)
                        return
                    source_value = record.get("source")
                    if source_value:
                        upload_source = str(source_value)
                    gps_redacted_by_client = bool(record.get("gps_redacted_by_client"))
                    file_ref = record.get("file_ref")
                    if not file_ref:
                        raise RuntimeError("upload missing file_ref")

                    device_id_value = record.get("device_id")
                    if device_id_value:
                        device_id_str = str(device_id_value)

                    recognition_channel_id = get_recognition_channel_id(conn)
                    asset_channel_id = (
                        recognition_channel_id
                        if recognition_channel_id is not None
                        else get_asset_channel_id(conn)
                    )
                    if asset_channel_id is None:
                        logging.error(
                            "MOBILE_UPLOAD_RECOGNITION_CHANNEL_MISSING - установите recognition_channel в БД",
                            extra={
                                "upload_id": upload_id,
                                "device_id": device_id_str,
                                "source": "mobile",
                                "timestamp": datetime.now(UTC).isoformat(),
                            },
                        )
                        set_upload_status(
                            conn,
                            id=upload_id,
                            status="failed",
                            error="asset_channel_not_configured",
                        )
                        record_upload_status_change()
                        conn.commit()
                        metrics_recorder.record_process_failure()
                        record_job_processed("process_upload", "failed")
                        return

                    job_config = replace(uploads_config, assets_channel_id=asset_channel_id)

                    download = await _download_from_storage(storage, key=str(file_ref))
                    try:
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

                        with context(device_id=device_id_str):
                            if download and download.path.exists():
                                with contextlib.suppress(OSError):
                                    size_bytes = download.path.stat().st_size

                            ingestion_context = UploadIngestionContext(
                                upload_id=upload_id,
                                storage_key=str(file_ref),
                                metrics=metrics_recorder,
                                source="mobile",
                                device_id=device_id_str,
                                user_id=device_user_id,
                                job_id=job.id,
                                job_name=job.name,
                                gps_redacted_by_client=gps_redacted_by_client,
                            )

                            result = await ingest_photo(
                                data=data,
                                telegram=telegram,
                                openai=openai,
                                supabase=supabase,
                                config=job_config,
                                context=ingestion_context,
                                file_path=download.path,
                                cleanup_file=False,
                            )

                            asset_id = result.asset_id
                            if not asset_id:
                                raise RuntimeError("asset creation failed for upload")

                            photo_meta = result.photo
                            exif_payload = dict(result.exif or {})
                            gps_payload = dict(result.gps or {})
                            if gps_redacted_by_client:
                                for key in ("latitude", "longitude", "altitude"):
                                    gps_payload.pop(key, None)
                            exif_ifds_payload = dict(result.exif_ifds or {})
                            exif_sections_payload: dict[str, dict[str, Any]] = {}
                            for ifd_name, raw_ifd in exif_ifds_payload.items():
                                if isinstance(raw_ifd, Mapping):
                                    exif_sections_payload[ifd_name] = dict(raw_ifd)
                            for required_ifd in ("0th", "Exif", "GPS"):
                                exif_sections_payload.setdefault(required_ifd, {})
                            exif_datetime_payload: dict[str, Any] = {}
                            if download and download.path.exists():
                                exif_datetime_payload = _extract_exif_datetimes(download.path)
                            metadata_payload = {
                                "exif": exif_payload,
                                "gps": gps_payload,
                            }
                            if photo_meta:
                                photo_meta_payload: dict[str, Any] = {}
                                if photo_meta.captured_at:
                                    photo_meta_payload["captured_at"] = (
                                        photo_meta.captured_at.isoformat()
                                    )
                                if photo_meta.latitude is not None:
                                    photo_meta_payload["latitude"] = photo_meta.latitude
                                if photo_meta.longitude is not None:
                                    photo_meta_payload["longitude"] = photo_meta.longitude
                                if photo_meta.altitude is not None:
                                    photo_meta_payload["altitude"] = photo_meta.altitude
                                if photo_meta.make:
                                    photo_meta_payload["make"] = photo_meta.make
                                if photo_meta.model:
                                    photo_meta_payload["model"] = photo_meta.model
                                if photo_meta.orientation is not None:
                                    photo_meta_payload["orientation"] = photo_meta.orientation
                                photo_meta_payload["source"] = photo_meta.source
                                if photo_meta_payload:
                                    metadata_payload["photo_meta"] = photo_meta_payload
                            if exif_datetime_payload:
                                metadata_payload.update(exif_datetime_payload)
                            update_kwargs: dict[str, Any] = {
                                "metadata": metadata_payload,
                                "exif_present": bool(exif_payload) or bool(gps_payload),
                            }
                            if not gps_redacted_by_client:
                                if photo_meta and photo_meta.latitude is not None:
                                    update_kwargs["latitude"] = photo_meta.latitude
                                elif gps_payload.get("latitude") is not None:
                                    update_kwargs["latitude"] = gps_payload.get("latitude")
                                if photo_meta and photo_meta.longitude is not None:
                                    update_kwargs["longitude"] = photo_meta.longitude
                                elif gps_payload.get("longitude") is not None:
                                    update_kwargs["longitude"] = gps_payload.get("longitude")
                            latitude = update_kwargs.get("latitude")
                            longitude = update_kwargs.get("longitude")
                            raw_exif_payload: dict[str, Any] = {}
                            raw_gps_payload: dict[str, Any] = {}
                            if photo_meta:
                                if photo_meta.raw_exif:
                                    raw_exif_payload = {
                                        key: dict(value)
                                        for key, value in photo_meta.raw_exif.items()
                                    }
                                if photo_meta.raw_gps:
                                    raw_gps_payload = dict(photo_meta.raw_gps)
                            if not raw_exif_payload:
                                raw_exif_payload = dict(exif_sections_payload.get("Exif") or {})
                            if not raw_gps_payload:
                                raw_gps_payload = dict(exif_sections_payload.get("GPS") or {})

                            gps_ifd_present = bool(raw_gps_payload)
                            gps_coordinates_present = latitude is not None and longitude is not None
                            photo_meta_log_payload = {
                                "has_exif": bool(raw_exif_payload),
                                "has_gps": gps_coordinates_present,
                                "gps_ifd_present": gps_ifd_present,
                                "latitude": latitude,
                                "longitude": longitude,
                                "altitude": photo_meta.altitude if photo_meta else None,
                                "source": photo_meta.source if photo_meta else None,
                                "raw_exif": raw_exif_payload,
                                "raw_gps": raw_gps_payload,
                            }

                            logging.info(
                                "MOBILE_EXIF_RAW",
                                extra={
                                    "asset_id": asset_id,
                                    "upload_id": upload_id,
                                    "has_exif": photo_meta_log_payload["has_exif"],
                                    "has_gps": photo_meta_log_payload["has_gps"],
                                    "gps_ifd_present": photo_meta_log_payload["gps_ifd_present"],
                                    "photo_meta_raw": _serialize_for_log(photo_meta_log_payload),
                                },
                            )

                            logging.info(
                                "MOBILE_EXIF_EXTRACTED",
                                extra={
                                    "asset_id": asset_id,
                                    "upload_id": upload_id,
                                    "exif_payload": bool(exif_payload),
                                    "gps_payload": gps_coordinates_present,
                                    "latitude": latitude,
                                    "longitude": longitude,
                                },
                            )
                            data.update_asset(asset_id, **update_kwargs)

                            link_upload_asset(conn, upload_id=upload_id, asset_id=asset_id)
                            logging.info(
                                "UPLOAD asset stored upload=%s asset=%s sha=%s mime=%s",
                                upload_id,
                                asset_id,
                                result.sha256,
                                result.mime_type,
                            )
                            telegram_file_meta = result.telegram_file or {}
                            file_meta_payload: dict[str, Any] = {
                                "file_ref": str(file_ref),
                                "sha256": result.sha256,
                            }
                            for key in ("file_id", "file_unique_id", "file_ref"):
                                value = telegram_file_meta.get(key)
                                if value:
                                    file_meta_payload[key] = value
                            mime_value = telegram_file_meta.get("mime_type") or result.mime_type
                            if mime_value:
                                file_meta_payload["mime_type"] = mime_value

                            def _as_int(value: Any) -> int | None:
                                try:
                                    return int(value)
                                except (TypeError, ValueError):
                                    return None

                            width_value = telegram_file_meta.get("width")
                            if width_value is None:
                                width_value = result.width
                            height_value = telegram_file_meta.get("height")
                            if height_value is None:
                                height_value = result.height
                            for key, raw in (
                                ("width", width_value),
                                ("height", height_value),
                                ("file_size", telegram_file_meta.get("file_size")),
                            ):
                                coerced = _as_int(raw)
                                if coerced is not None:
                                    file_meta_payload[key] = coerced

                            saved_asset_id = data.save_asset(
                                result.chat_id,
                                result.message_id,
                                None,
                                None,
                                tg_chat_id=result.chat_id,
                                caption=result.caption,
                                kind="photo",
                                file_meta=file_meta_payload,
                                metadata=metadata_payload,
                                origin="mobile",
                                source="mobile",
                            )
                            if saved_asset_id != asset_id:
                                logging.warning(
                                    "Asset id mismatch after save (expected %s, got %s)",
                                    asset_id,
                                    saved_asset_id,
                                )
                            jobs.enqueue("ingest", {"asset_id": asset_id}, dedupe=True)
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

                    with context(device_id=device_id_str):
                        set_upload_status(conn, id=upload_id, status="done")
                        record_upload_status_change()
                        conn.commit()
                        if should_cleanup_local_download and download and download.path.exists():
                            with contextlib.suppress(Exception):
                                download.path.unlink()
                        if upload_source and upload_source.lower() == "mobile":
                            record_mobile_photo_ingested()
                        logging.info("UPLOAD job done upload=%s", upload_id)
                        logging.info(
                            "MOBILE_UPLOAD_DONE",
                            extra={
                                "upload_id": upload_id,
                                "device_id": device_id_str,
                                "tg_chat_id": asset_channel_id,
                                "source": "mobile",
                                "size_bytes": size_bytes,
                                "timestamp": datetime.now(UTC).isoformat(),
                            },
                        )
                        record_job_processed("process_upload", "ok")
                except Exception as exc:
                    with context(device_id=device_id_str):
                        logging.exception("UPLOAD job failed upload=%s", upload_id)
                        logging.error(
                            "MOBILE_UPLOAD_FAILED",
                            extra={
                                "upload_id": upload_id,
                                "device_id": device_id_str,
                                "size_bytes": size_bytes,
                                "timestamp": datetime.now(UTC).isoformat(),
                                "error": str(exc),
                            },
                        )
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

    jobs.register_handler("process_upload", process_upload)


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


RAW_EXIF_LOG_MAX_LENGTH = 64 * 1024


def _serialize_for_log(payload: Mapping[str, Any]) -> str:
    """Serialize payload to JSON and trim overly large entries."""

    safe_payload = DataAccess._make_json_safe(payload)
    raw = json.dumps(safe_payload, ensure_ascii=False, sort_keys=True)
    if len(raw) > RAW_EXIF_LOG_MAX_LENGTH:
        return raw[:RAW_EXIF_LOG_MAX_LENGTH]
    return raw
