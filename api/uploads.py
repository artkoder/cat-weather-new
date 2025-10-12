from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import tempfile
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, ContextManager, Protocol
from urllib.parse import unquote, urlparse
from uuid import uuid4

import httpx
from aiohttp import web
from PIL import ExifTags, Image, ImageOps

from data_access import (
    DataAccess,
    fetch_upload_record,
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
from openai_client import OpenAIClient, OpenAIResponse
from storage import Storage
from supabase_client import SupabaseClient


class TelegramClient(Protocol):
    async def send_photo(
        self,
        *,
        chat_id: int,
        photo: Path,
        caption: str | None = None,
    ) -> Any:
        ...


class MetricsEmitter(Protocol):
    def increment(self, name: str, value: float = 1.0) -> None:
        ...

    def observe(self, name: str, value: float) -> None:
        ...


@dataclass(slots=True)
class UploadMetricsRecorder:
    counters: dict[str, float] = field(default_factory=dict)
    timings: dict[str, list[float]] = field(default_factory=dict)
    emitter: MetricsEmitter | None = None

    def increment(self, name: str, value: float = 1.0) -> None:
        self.counters[name] = self.counters.get(name, 0.0) + value
        if self.emitter:
            with contextlib.suppress(Exception):
                self.emitter.increment(name, value)

    @contextlib.contextmanager
    def timer(self, name: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            duration_ms = (perf_counter() - start) * 1000.0
            self.timings.setdefault(name, []).append(duration_ms)
            if self.emitter:
                with contextlib.suppress(Exception):
                    self.emitter.observe(name, duration_ms)

    def measure_process(self) -> ContextManager[None]:
        return self.timer("process_upload_ms")

    def measure_exif(self) -> ContextManager[None]:
        return self.timer("exif_ms")

    def measure_vision(self) -> ContextManager[None]:
        return self.timer("vision_ms")

    def measure_telegram(self) -> ContextManager[None]:
        return self.timer("tg_ms")

    def record_asset_created(self, count: int = 1) -> None:
        if count > 0:
            self.increment("assets_created_total", count)

    def record_process_failure(self) -> None:
        self.increment("upload_process_fail_total")

    def record_vision_tokens(self, tokens: int | None) -> None:
        if tokens and tokens > 0:
            self.increment("vision_tokens_total", tokens)


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


VISION_SYSTEM_PROMPT = (
    "Ты ассистент проекта Котопогода. Проанализируй изображение и верни JSON, "
    "строго соответствующий схеме asset_vision_v1."
)

VISION_USER_PROMPT = (
    "Опиши главный сюжет, перечисли заметные категории и объекты."
    " Используй краткие формулировки."
)

VISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "asset_vision_v1",
    "additionalProperties": False,
    "properties": {
        "caption": {"type": ["string", "null"]},
        "categories": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
        },
        "objects": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
        },
    },
}


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


def _compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _downscale_image_if_needed(source: Path, *, max_side: int) -> tuple[Path, bool]:
    """Return path to a file respecting ``max_side`` and a cleanup flag."""

    with Image.open(source) as original:
        width, height = original.size
        if max(width, height) <= max_side:
            return source, False

        if hasattr(Image, "Resampling"):
            resample = Image.Resampling.LANCZOS
        else:  # pragma: no cover - Pillow < 9 compatibility
            resample = Image.LANCZOS  # type: ignore[attr-defined]

        scale = max_side / float(max(width, height))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        resized = original.resize((new_width, new_height), resample)
        suffix = source.suffix or ".jpg"
        fd, name = tempfile.mkstemp(prefix="upload-resized-", suffix=suffix)
        os.close(fd)
        temp_path = Path(name)
        format_name = original.format or suffix.lstrip(".").upper()
        try:
            resized.save(temp_path, format=format_name)
        finally:
            resized.close()
    return temp_path, True


def _normalize_exif_value(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="ignore")
        except Exception:
            return value.hex()
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_normalize_exif_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _normalize_exif_value(v) for k, v in value.items()}
    try:
        return float(value)
    except Exception:
        return str(value)


def _extract_gps_decimal(gps_info: dict[str, Any]) -> tuple[float | None, float | None]:
    def _to_decimal(values: list[Any], ref: str | None) -> float | None:
        if not values:
            return None
        try:
            degrees = [_normalize_exif_value(part) for part in values]
            parts = [float(part) for part in degrees]
            while len(parts) < 3:
                parts.append(0.0)
        except Exception:
            return None
        decimal = parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
        if ref and ref.upper() in {"S", "W"}:
            decimal *= -1.0
        return decimal

    lat = _to_decimal(
        gps_info.get("GPSLatitude") or [],
        str(gps_info.get("GPSLatitudeRef") or "") or None,
    )
    lon = _to_decimal(
        gps_info.get("GPSLongitude") or [],
        str(gps_info.get("GPSLongitudeRef") or "") or None,
    )
    return lat, lon


def _extract_capture_datetime(exif: dict[str, Any]) -> str | None:
    candidates = [
        "DateTimeOriginal",
        "DateTimeDigitized",
        "DateTime",
    ]
    for key in candidates:
        raw = exif.get(key)
        if not raw:
            continue
        text = str(raw).strip()
        if not text:
            continue
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(text, fmt)
            except ValueError:
                continue
            return dt.replace(tzinfo=timezone.utc).isoformat()
        return text
    return None


def _extract_image_metadata(path: Path) -> tuple[str | None, int | None, int | None, dict[str, Any], dict[str, Any]]:
    exif_payload: dict[str, Any] = {}
    gps_payload: dict[str, Any] = {}
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None

    with Image.open(path) as raw_image:
        original_format = raw_image.format
        image = ImageOps.exif_transpose(raw_image)
        try:
            width, height = image.size
            format_name = image.format or original_format
            if format_name and format_name in Image.MIME:
                mime_type = Image.MIME[format_name]
            exif_data = image.getexif() if hasattr(image, "getexif") else None
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    if tag_name == "GPSInfo" and isinstance(value, dict):
                        gps_info = {
                            ExifTags.GPSTAGS.get(sub_id, str(sub_id)): _normalize_exif_value(sub_val)
                            for sub_id, sub_val in value.items()
                        }
                        exif_payload["GPSInfo"] = gps_info
                        lat, lon = _extract_gps_decimal(gps_info)
                        if lat is not None:
                            gps_payload["latitude"] = lat
                        if lon is not None:
                            gps_payload["longitude"] = lon
                    else:
                        exif_payload[tag_name] = _normalize_exif_value(value)
        finally:
            if image is not raw_image:
                with contextlib.suppress(Exception):
                    image.close()

    capture = _extract_capture_datetime(exif_payload)
    if capture:
        gps_payload.setdefault("captured_at", capture)
    return mime_type, width, height, exif_payload, gps_payload


def _extract_categories(vision: dict[str, Any] | None) -> list[str]:
    if not vision:
        return []
    categories: list[str] = []
    for key in ("categories", "objects", "labels"):
        value = vision.get(key)
        if isinstance(value, (list, tuple)):
            for item in value:
                text = str(item).strip()
                if text and text not in categories:
                    categories.append(text)
    return categories


def _build_caption(
    *,
    gps: dict[str, Any],
    categories: list[str],
    capture_iso: str | None,
) -> str:
    parts: list[str] = []
    if capture_iso:
        parts.append(f"Дата съёмки: {capture_iso}")
    lat = gps.get("latitude")
    lon = gps.get("longitude")
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        parts.append(f"Координаты: {lat:.5f}, {lon:.5f}")
    if categories:
        parts.append("Категории: " + ", ".join(categories[:6]))
    if not parts:
        parts.append("Новая загрузка готова к обработке.")
    return "\n".join(parts)


def _extract_message_id(response: Any) -> int | None:
    if isinstance(response, dict):
        if "result" in response:
            return _extract_message_id(response.get("result"))
        message_id = response.get("message_id")
        if isinstance(message_id, int):
            return message_id
        if isinstance(message_id, str) and message_id.isdigit():
            return int(message_id)
    return None


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
                        sha256 = _compute_sha256(download.path)
                        with metrics_recorder.measure_exif():
                            (
                                mime_type,
                                width,
                                height,
                                exif_payload,
                                gps_payload,
                            ) = _extract_image_metadata(download.path)

                        vision_payload: dict[str, Any] | None = None
                        if "captured_at" not in gps_payload:
                            capture_candidate = _extract_capture_datetime(exif_payload)
                            if capture_candidate:
                                gps_payload["captured_at"] = capture_candidate

                        processed_path = download.path
                        if uploads_config.max_image_side:
                            processed_path, processed_cleanup = _downscale_image_if_needed(
                                download.path,
                                max_side=uploads_config.max_image_side,
                            )

                        if (
                            uploads_config.vision_enabled
                            and openai
                            and uploads_config.openai_vision_model
                        ):
                            model = uploads_config.openai_vision_model
                            with metrics_recorder.measure_vision():
                                response = await openai.classify_image(
                                    model=model,
                                    system_prompt=VISION_SYSTEM_PROMPT,
                                    user_prompt=VISION_USER_PROMPT,
                                    image_path=processed_path,
                                    schema=VISION_SCHEMA,
                                )
                            if isinstance(response, OpenAIResponse):
                                vision_payload = response.content
                                metrics_recorder.record_vision_tokens(response.total_tokens)
                                data.log_token_usage(
                                    model=model,
                                    prompt_tokens=response.prompt_tokens,
                                    completion_tokens=response.completion_tokens,
                                    total_tokens=response.total_tokens,
                                    job_id=job.id,
                                    request_id=response.request_id,
                                )
                                if supabase:
                                    try:
                                        await supabase.insert_token_usage(
                                            bot="kotopogoda",
                                            model=model,
                                            prompt_tokens=response.prompt_tokens,
                                            completion_tokens=response.completion_tokens,
                                            total_tokens=response.total_tokens,
                                            request_id=response.request_id,
                                            endpoint=response.endpoint or "/v1/responses",
                                            meta={
                                                "upload_id": upload_id,
                                                "job_id": job.id,
                                            },
                                        )
                                    except Exception:
                                        logging.exception(
                                            "UPLOAD vision token usage logging failed upload=%s",
                                            upload_id,
                                        )
                        elif uploads_config.vision_enabled and not openai:
                            logging.warning(
                                "VISION_ENABLED is set but OpenAI client is unavailable; skipping vision"
                            )
                        elif (
                            uploads_config.vision_enabled
                            and not uploads_config.openai_vision_model
                        ):
                            logging.warning(
                                "VISION_ENABLED but OPENAI_VISION_MODEL missing; skipping vision"
                            )

                        categories = _extract_categories(vision_payload)
                        caption = _build_caption(
                            gps=gps_payload,
                            categories=categories,
                            capture_iso=gps_payload.get("captured_at"),
                        )

                        chat_id = uploads_config.assets_channel_id
                        with metrics_recorder.measure_telegram():
                            telegram_response = await telegram.send_photo(
                                chat_id=chat_id,
                                photo=processed_path,
                                caption=caption,
                            )
                        message_id = _extract_message_id(telegram_response)
                        if message_id is None:
                            raise RuntimeError("telegram response missing message_id")

                        message_identifier = f"{chat_id}:{message_id}"
                        asset_id = data.create_asset(
                            upload_id=upload_id,
                            file_ref=str(file_ref),
                            content_type=mime_type,
                            sha256=sha256,
                            width=width,
                            height=height,
                            exif=exif_payload or None,
                            labels=vision_payload or None,
                            tg_message_id=message_identifier,
                            tg_chat_id=chat_id,
                        )
                        link_upload_asset(conn, upload_id=upload_id, asset_id=asset_id)
                        logging.info(
                            "UPLOAD asset stored upload=%s asset=%s sha=%s mime=%s",
                            upload_id,
                            asset_id,
                            sha256,
                            mime_type,
                        )
                    finally:
                        if (
                            processed_cleanup
                            and processed_path is not None
                            and processed_path != download.path
                            and processed_path.exists()
                        ):
                            with contextlib.suppress(Exception):
                                processed_path.unlink()
                        if download.cleanup and download.path.exists():
                            with contextlib.suppress(Exception):
                                download.path.unlink()

                    set_upload_status(conn, id=upload_id, status="done")
                    record_upload_status_change()
                    conn.commit()
                    metrics_recorder.record_asset_created()
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
