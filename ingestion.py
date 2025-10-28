from __future__ import annotations

import contextlib
import re
import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from typing import TYPE_CHECKING, Any, BinaryIO, Callable, Mapping, Protocol, Sequence, TypedDict

from PIL import ExifTags, Image, ImageOps
import piexif

from observability import context as telemetry_context

if TYPE_CHECKING:  # pragma: no cover - typing only
    from api.uploads import UploadsConfig
    from data_access import DataAccess

try:  # pragma: no cover - optional typing imports
    from openai_client import OpenAIClient, OpenAIResponse
except Exception:  # pragma: no cover - typing fallback
    OpenAIClient = None  # type: ignore[assignment]
    OpenAIResponse = None  # type: ignore[assignment]

try:  # pragma: no cover - optional typing imports
    from supabase_client import SupabaseClient
except Exception:  # pragma: no cover - typing fallback
    SupabaseClient = None  # type: ignore[assignment]


class MetricsEmitter(Protocol):
    def increment(self, name: str, value: float = 1.0) -> None:
        ...

    def observe(self, name: str, value: float) -> None:
        ...


class TelegramClient(Protocol):
    async def send_photo(
        self,
        *,
        chat_id: int,
        photo: Path,
        caption: str | None = None,
    ) -> Any:
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
    def timer(self, name: str):
        start = perf_counter()
        try:
            yield
        finally:
            duration_ms = (perf_counter() - start) * 1000.0
            self.timings.setdefault(name, []).append(duration_ms)
            if self.emitter:
                with contextlib.suppress(Exception):
                    self.emitter.observe(name, duration_ms)

    def measure_process(self):
        return self.timer("process_upload_ms")

    def measure_exif(self):
        return self.timer("exif_ms")

    def measure_vision(self):
        return self.timer("vision_ms")

    def measure_telegram(self):
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
class UploadIngestionContext:
    upload_id: str
    storage_key: str
    metrics: UploadMetricsRecorder
    source: str = "mobile"
    device_id: str | None = None
    user_id: int | None = None
    job_id: int | None = None
    job_name: str | None = None

    def telemetry_payload(self) -> Mapping[str, Any] | None:
        payload: dict[str, Any] = {"upload_id": self.upload_id}
        if self.job_name:
            payload["job"] = self.job_name
        if self.job_id is not None:
            payload["job_id"] = self.job_id
        if self.device_id:
            payload["device_id"] = self.device_id
        if self.user_id is not None:
            payload["user_id"] = self.user_id
        return payload or None


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


@dataclass(slots=True)
class IngestionFile:
    path: Path
    cleanup: bool = False


@dataclass(slots=True)
class IngestionVisionConfig:
    enabled: bool = False
    model: str | None = None


class TokenUsagePayload(TypedDict, total=False):
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    job_id: int | None
    request_id: str | None


@dataclass(slots=True)
class IngestionInputs:
    source: str
    channel_id: int
    file: IngestionFile
    upload_id: str | None = None
    asset_id: str | None = None
    file_ref: str | None = None
    file_metadata: dict[str, Any] | None = None
    job_id: int | None = None
    job_name: str | None = None
    telemetry: Mapping[str, Any] | None = None
    max_image_side: int | None = None
    exif: dict[str, Any] | None = None
    gps: dict[str, Any] | None = None
    vision: IngestionVisionConfig = field(default_factory=IngestionVisionConfig)
    tg_chat_id: int | None = None
    template: str | None = None
    hashtags: str | None = None
    caption: str | None = None
    kind: str | None = None
    metadata: dict[str, Any] | None = None
    categories: list[str] | None = None
    rubric_id: int | None = None
    origin: str | None = None
    author_user_id: int | None = None
    author_username: str | None = None
    sender_chat_id: int | None = None
    via_bot_id: int | None = None
    forward_from_user: int | None = None
    forward_from_chat: int | None = None


class CreateAssetPayload(TypedDict, total=False):
    upload_id: str
    file_ref: str
    content_type: str | None
    sha256: str
    width: int | None
    height: int | None
    exif: dict[str, Any] | None
    labels: dict[str, Any] | None
    tg_message_id: str | int | None
    tg_chat_id: int | None
    source: str


class SaveAssetPayload(TypedDict, total=False):
    channel_id: int
    message_id: int
    template: str | None
    hashtags: str | None
    tg_chat_id: int
    caption: str | None
    kind: str | None
    file_meta: dict[str, Any] | None
    metadata: dict[str, Any] | None
    categories: list[str] | None
    rubric_id: int | None
    origin: str
    source: str
    author_user_id: int | None
    author_username: str | None
    sender_chat_id: int | None
    via_bot_id: int | None
    forward_from_user: int | None
    forward_from_chat: int | None
    latitude: float | None
    longitude: float | None
    exif_present: bool | None


@dataclass(slots=True)
class IngestionCallbacks:
    create_asset: Callable[[CreateAssetPayload], str] | None = None
    save_asset: Callable[[SaveAssetPayload], str] | None = None
    link_upload_asset: Callable[[str], None] | None = None


@dataclass(slots=True)
class IngestionContext:
    telegram: TelegramClient
    metrics: UploadMetricsRecorder
    openai: OpenAIClient | None = None
    supabase: SupabaseClient | None = None
    token_logger: Callable[[TokenUsagePayload], None] | None = None


@dataclass(slots=True)
class IngestionResult:
    asset_id: str | None
    message_id: int
    chat_id: int
    caption: str
    sha256: str
    mime_type: str | None
    width: int | None
    height: int | None
    exif: dict[str, Any]
    gps: dict[str, Any]
    vision: dict[str, Any] | None
    telegram_file: dict[str, Any] | None
    metrics: UploadMetricsRecorder


def _compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


_RATIO_SPLIT_PATTERN = re.compile(r"[\s,]+")


def _parse_ratio_token(token: str) -> float | None:
    """Parse a single ``numerator/denominator`` token into a float."""

    stripped = token.strip()
    if not stripped:
        return None
    if "/" in stripped:
        numerator_text, denominator_text = stripped.split("/", 1)
        try:
            numerator_value = float(numerator_text)
            denominator_value = float(denominator_text)
        except Exception:
            return None
        if denominator_value == 0:
            return None
        return numerator_value / denominator_value
    try:
        return float(stripped)
    except Exception:
        return None


def _extract_ratio_numbers(text: str) -> list[float] | None:
    """Extract numeric values from a ratio string like ``"10/1,20/1"``."""

    stripped = text.strip()
    if not stripped:
        return None
    if "/" not in stripped and "," not in stripped:
        return None
    tokens = [token for token in _RATIO_SPLIT_PATTERN.split(stripped) if token]
    if not tokens:
        return None
    numbers: list[float] = []
    for token in tokens:
        parsed = _parse_ratio_token(token)
        if parsed is None:
            return None
        numbers.append(parsed)
    return numbers


def _to_float_ratio(value: Any) -> float | None:
    """Convert EXIF rational representations to float when possible."""

    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        ratio_numbers = _extract_ratio_numbers(stripped)
        if ratio_numbers and len(ratio_numbers) == 1:
            return ratio_numbers[0]
        if "/" in stripped:
            parts = stripped.split("/", 1)
            if len(parts) == 2:
                try:
                    numerator_value = float(parts[0])
                    denominator_value = float(parts[1])
                except Exception:
                    return None
                if denominator_value == 0:
                    return None
                return numerator_value / denominator_value
        try:
            return float(stripped)
        except Exception:
            return None

    numerator = getattr(value, "numerator", None)
    denominator = getattr(value, "denominator", None)
    if numerator is not None and denominator is not None:
        try:
            numerator_value = float(numerator)
            denominator_value = float(denominator)
        except Exception:
            return None
        if denominator_value == 0:
            return None
        return numerator_value / denominator_value

    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
        sequence = list(value)
        if not sequence:
            return None
        if len(sequence) == 1:
            return _to_float_ratio(sequence[0])
        if len(sequence) == 2:
            numerator_value = _to_float_ratio(sequence[0])
            denominator_value = _to_float_ratio(sequence[1])
            if numerator_value is None or denominator_value in (None, 0.0):
                return None
            return numerator_value / denominator_value
        return None

    try:
        return float(value)
    except Exception:
        return None


def _normalize_exif_value(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="ignore")
        except Exception:
            return value.hex()
    if isinstance(value, str):
        ratio_numbers = _extract_ratio_numbers(value)
        if ratio_numbers:
            if len(ratio_numbers) == 1:
                return ratio_numbers[0]
            return ratio_numbers
        return value
    if isinstance(value, (int, float)):
        return value
    ratio_value = _to_float_ratio(value)
    if ratio_value is not None and not isinstance(value, (list, tuple)):
        return ratio_value
    if isinstance(value, (list, tuple)):
        return [_normalize_exif_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _normalize_exif_value(v) for k, v in value.items()}
    try:
        return float(value)
    except Exception:
        return str(value)


def _extract_gps_decimal(gps_info: dict[str, Any]) -> tuple[float | None, float | None]:
    def _coerce_sequence(value: Any) -> list[Any]:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            ratio_numbers = _extract_ratio_numbers(stripped)
            if ratio_numbers:
                return ratio_numbers
            if re.search(r"[\s,]", stripped):
                tokens = [token for token in re.split(r"[\s,]+", stripped) if token]
                if tokens:
                    return tokens
            return [value]

        if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
            return list(value)
        return [value]

    def _to_decimal(values: Any, ref: str | None) -> float | None:
        if not values:
            return None

        raw_parts = _coerce_sequence(values)
        numeric_parts: list[float] = []

        for raw_part in raw_parts:
            if isinstance(raw_part, str):
                ratio_values = _extract_ratio_numbers(raw_part)
                if ratio_values:
                    numeric_parts.extend(ratio_values)
                    continue
            normalized = _normalize_exif_value(raw_part)
            number: float | None = None

            if isinstance(normalized, (int, float)):
                number = float(normalized)
            else:
                ratio_value = _to_float_ratio(normalized)
                if ratio_value is None:
                    ratio_value = _to_float_ratio(raw_part)
                if ratio_value is not None:
                    number = ratio_value

            if number is None:
                try:
                    number = float(normalized)
                except Exception:
                    return None

            numeric_parts.append(number)

        while len(numeric_parts) < 3:
            numeric_parts.append(0.0)

        decimal = (
            numeric_parts[0]
            + numeric_parts[1] / 60.0
            + numeric_parts[2] / 3600.0
        )
        if ref and ref.upper() in {"S", "W"}:
            decimal *= -1.0
        return decimal

    lat_ref = gps_info.get("GPSLatitudeRef")
    lon_ref = gps_info.get("GPSLongitudeRef")
    lat = _to_decimal(gps_info.get("GPSLatitude") or [], str(lat_ref) if lat_ref else None)
    lon = _to_decimal(
        gps_info.get("GPSLongitude") or [], str(lon_ref) if lon_ref else None
    )
    return lat, lon


def _decode_exif_datetime_value(value: Any) -> str | None:
    if isinstance(value, bytes):
        for encoding in ("utf-8", "ascii", "latin-1"):
            try:
                decoded = value.decode(encoding, errors="ignore")
            except Exception:
                continue
            decoded = decoded.strip().strip("\x00")
            if decoded:
                return decoded
        return None
    if isinstance(value, str):
        decoded = value.strip().strip("\x00")
        return decoded or None
    return None


def extract_exif_datetimes(image_source: str | Path | BinaryIO) -> dict[str, str]:
    try:
        with Image.open(image_source, mode="r") as img:
            exif_bytes = img.info.get("exif")
        if exif_bytes:
            exif_dict = piexif.load(exif_bytes)
        else:
            if isinstance(image_source, (str, Path)):
                exif_dict = piexif.load(str(image_source))
            else:
                return {}
    except Exception:
        logging.exception("Failed to parse EXIF metadata")
        return {}

    result: dict[str, str] = {}
    exif_ifd = exif_dict.get("Exif") or {}
    original_value = _decode_exif_datetime_value(
        exif_ifd.get(piexif.ExifIFD.DateTimeOriginal)
    )
    if original_value:
        result["exif_datetime_original"] = original_value
    digitized_value = _decode_exif_datetime_value(
        exif_ifd.get(piexif.ExifIFD.DateTimeDigitized)
    )
    if digitized_value:
        result["exif_datetime_digitized"] = digitized_value
    zeroth_ifd = exif_dict.get("0th") or {}
    image_datetime_value = _decode_exif_datetime_value(
        zeroth_ifd.get(piexif.ImageIFD.DateTime)
    )
    if image_datetime_value:
        result["exif_datetime"] = image_datetime_value
    best_value = (
        result.get("exif_datetime_original")
        or result.get("exif_datetime_digitized")
        or result.get("exif_datetime")
    )
    if best_value:
        result["exif_datetime_best"] = best_value
    return result


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


def extract_image_metadata(path: Path) -> tuple[str | None, int | None, int | None, dict[str, Any], dict[str, Any]]:
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
    logging.info(
        "MOBILE_IMAGE_METADATA",
        extra={
            "path": str(path),
            "mime_type": mime_type,
            "exif_present": bool(exif_payload),
            "gps_present": bool(gps_payload),
            "latitude": gps_payload.get("latitude"),
            "longitude": gps_payload.get("longitude"),
        },
    )
    return mime_type, width, height, exif_payload, gps_payload


def extract_categories(vision: dict[str, Any] | None) -> list[str]:
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


def build_caption(*, gps: dict[str, Any], categories: list[str], capture_iso: str | None) -> str:
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


def extract_message_id(response: Any) -> int | None:
    if isinstance(response, dict):
        if "result" in response:
            return extract_message_id(response.get("result"))
        message_id = response.get("message_id")
        if isinstance(message_id, int):
            return message_id
        if isinstance(message_id, str) and message_id.isdigit():
            return int(message_id)
    return None


def _ensure_telemetry(telemetry: Mapping[str, Any] | None):
    if telemetry:
        return telemetry_context(**telemetry)
    return contextlib.nullcontext()


def _select_best_photo_size(photo_sizes: Sequence[Any] | None) -> dict[str, Any] | None:
    if not photo_sizes:
        return None
    best: dict[str, Any] | None = None
    best_score = -1
    for entry in photo_sizes:
        if not isinstance(entry, MappingABC):
            continue
        try:
            width = int(entry.get("width") or 0)
        except (TypeError, ValueError):
            width = 0
        try:
            height = int(entry.get("height") or 0)
        except (TypeError, ValueError):
            height = 0
        try:
            file_size = int(entry.get("file_size") or 0)
        except (TypeError, ValueError):
            file_size = 0
        score = width * height
        if score <= 0 and file_size <= 0:
            continue
        if best is None:
            best = dict(entry)
            best_score = score
            continue
        best_file_size = 0
        try:
            best_file_size = int(best.get("file_size") or 0)
        except (TypeError, ValueError):
            best_file_size = 0
        if score > best_score or (score == best_score and file_size > best_file_size):
            best = dict(entry)
            best_score = score
    return best


def _extract_telegram_photo_meta(response: Any) -> dict[str, Any] | None:
    if isinstance(response, MappingABC):
        photo_sizes = response.get("photo")
        if isinstance(photo_sizes, SequenceABC):
            best = _select_best_photo_size(photo_sizes)
            if best:
                meta: dict[str, Any] = {}
                for key in ("file_id", "file_unique_id", "file_ref", "mime_type"):
                    value = best.get(key)
                    if value is not None:
                        meta[key] = value
                for key in ("width", "height", "file_size"):
                    try:
                        value = best.get(key)
                        if value is not None:
                            meta[key] = int(value)
                    except (TypeError, ValueError):
                        continue
                if "mime_type" not in meta:
                    meta["mime_type"] = "image/jpeg"
                return meta
        for nested_key in ("result", "message", "channel_post"):
            nested = response.get(nested_key)
            meta = _extract_telegram_photo_meta(nested)
            if meta:
                return meta
    if isinstance(response, SequenceABC) and not isinstance(response, (str, bytes, bytearray)):
        for item in response:
            meta = _extract_telegram_photo_meta(item)
            if meta:
                return meta
    return None


async def _ingest_photo_internal(
    inputs: IngestionInputs,
    context: IngestionContext,
    callbacks: IngestionCallbacks,
) -> IngestionResult:
    metrics = context.metrics
    processed_path = inputs.file.path
    processed_cleanup = False
    vision_payload: dict[str, Any] | None = None

    with _ensure_telemetry(inputs.telemetry):
        sha256 = _compute_sha256(inputs.file.path)
        with metrics.measure_exif():
            mime_type, width, height, extracted_exif, extracted_gps = extract_image_metadata(
                inputs.file.path
            )
        exif_payload = inputs.exif or extracted_exif
        gps_payload = dict(extracted_gps)
        if inputs.gps:
            gps_payload.update(inputs.gps)
        if "captured_at" not in gps_payload:
            capture_candidate = _extract_capture_datetime(exif_payload)
            if capture_candidate:
                gps_payload["captured_at"] = capture_candidate

        if inputs.max_image_side:
            processed_path, processed_cleanup = _downscale_image_if_needed(
                inputs.file.path, max_side=inputs.max_image_side
            )

        if inputs.vision.enabled and context.openai and inputs.vision.model:
            model = inputs.vision.model
            with metrics.measure_vision():
                response = await context.openai.classify_image(
                    model=model,
                    system_prompt=VISION_SYSTEM_PROMPT,
                    user_prompt=VISION_USER_PROMPT,
                    image_path=processed_path,
                    schema=VISION_SCHEMA,
                )
            response_types: tuple[type, ...]
            if OpenAIResponse is None:
                response_types = ()
            else:
                response_types = (OpenAIResponse,)
            if isinstance(response, response_types):
                vision_payload = response.content
                metrics.record_vision_tokens(response.total_tokens)
                if context.token_logger:
                    payload: TokenUsagePayload = {
                        "model": model,
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                        "total_tokens": response.total_tokens,
                        "job_id": inputs.job_id,
                        "request_id": response.request_id,
                    }
                    try:
                        context.token_logger(payload)
                    except Exception:
                        logging.exception("Token usage logger failed for model %s", model)
                if context.supabase:
                    try:
                        await context.supabase.insert_token_usage(
                            bot="kotopogoda",
                            model=model,
                            prompt_tokens=response.prompt_tokens,
                            completion_tokens=response.completion_tokens,
                            total_tokens=response.total_tokens,
                            request_id=response.request_id,
                            endpoint=response.endpoint or "/v1/responses",
                            meta={
                                "upload_id": inputs.upload_id,
                                "asset_id": inputs.asset_id,
                                "job_id": inputs.job_id,
                                "job_name": inputs.job_name,
                                "source": inputs.source,
                            },
                        )
                    except Exception:
                        logging.exception("Vision token usage logging failed for source %s", inputs.source)
        elif inputs.vision.enabled and not context.openai:
            logging.warning(
                "Vision requested for %s but OpenAI client is unavailable; skipping", inputs.source
            )
        elif inputs.vision.enabled and not inputs.vision.model:
            logging.warning(
                "Vision requested for %s but model is missing; skipping", inputs.source
            )

        categories = extract_categories(vision_payload)
        caption = build_caption(
            gps=gps_payload, categories=categories, capture_iso=gps_payload.get("captured_at")
        )

        with metrics.measure_telegram():
            response = await context.telegram.send_photo(
                chat_id=inputs.channel_id,
                photo=processed_path,
                caption=caption,
            )
        message_id = extract_message_id(response)
        if message_id is None:
            raise RuntimeError("telegram response missing message_id")

        telegram_file_meta = _extract_telegram_photo_meta(response)
        if telegram_file_meta:
            if "mime_type" not in telegram_file_meta and mime_type:
                telegram_file_meta["mime_type"] = mime_type
            if "width" not in telegram_file_meta and width is not None:
                telegram_file_meta["width"] = width
            if "height" not in telegram_file_meta and height is not None:
                telegram_file_meta["height"] = height
            if inputs.file_ref and "file_ref" not in telegram_file_meta:
                telegram_file_meta["file_ref"] = inputs.file_ref

        tg_chat_id = inputs.tg_chat_id or inputs.channel_id
        message_identifier = f"{tg_chat_id}:{message_id}"
        asset_id: str | None = inputs.asset_id

        if inputs.source in {"upload", "mobile"} and callbacks.create_asset:
            if not inputs.upload_id or not inputs.file_ref:
                raise RuntimeError("upload ingestion requires upload_id and file_ref")
            source_value = inputs.source if inputs.source in {"mobile", "telegram"} else "mobile"
            asset_id = callbacks.create_asset(
                {
                    "upload_id": inputs.upload_id,
                    "file_ref": inputs.file_ref,
                    "content_type": mime_type,
                    "sha256": sha256,
                    "width": width,
                    "height": height,
                    "exif": exif_payload or None,
                    "labels": vision_payload or None,
                    "tg_message_id": message_identifier,
                    "tg_chat_id": inputs.channel_id,
                    "source": source_value,
                }
            )
        elif inputs.source != "upload" and callbacks.save_asset:
            combined_file_meta: dict[str, Any] = dict(inputs.file_metadata or {})
            if inputs.file_ref:
                combined_file_meta.setdefault("file_ref", inputs.file_ref)
            if mime_type is not None:
                combined_file_meta["mime_type"] = mime_type
            if telegram_file_meta:
                for key, value in telegram_file_meta.items():
                    if value is not None:
                        combined_file_meta[key] = value
            combined_file_meta["sha256"] = sha256
            if width is not None:
                combined_file_meta["width"] = width
            if height is not None:
                combined_file_meta["height"] = height
            if exif_payload:
                combined_file_meta["exif"] = exif_payload
            if vision_payload:
                combined_file_meta["labels"] = vision_payload

            metadata_payload: dict[str, Any] = dict(inputs.metadata or {})
            metadata_payload["exif"] = exif_payload
            metadata_payload["gps"] = gps_payload

            combined_categories: list[str] = []
            seen_categories: set[str] = set()
            for value in (inputs.categories or []):
                text = str(value).strip()
                if text and text not in seen_categories:
                    combined_categories.append(text)
                    seen_categories.add(text)
            for value in categories:
                text = str(value).strip()
                if text and text not in seen_categories:
                    combined_categories.append(text)
                    seen_categories.add(text)

            save_payload: SaveAssetPayload = {
                "channel_id": inputs.channel_id,
                "message_id": message_id,
                "template": inputs.template,
                "hashtags": inputs.hashtags,
                "tg_chat_id": tg_chat_id,
                "caption": inputs.caption if inputs.caption is not None else caption,
                "kind": inputs.kind,
                "file_meta": combined_file_meta or None,
                "metadata": metadata_payload or None,
                "categories": combined_categories or None,
                "rubric_id": inputs.rubric_id,
                "origin": inputs.origin or inputs.source,
                "source": inputs.source,
                "author_user_id": inputs.author_user_id,
                "author_username": inputs.author_username,
                "sender_chat_id": inputs.sender_chat_id,
                "via_bot_id": inputs.via_bot_id,
                "forward_from_user": inputs.forward_from_user,
                "forward_from_chat": inputs.forward_from_chat,
                "latitude": gps_payload.get("latitude"),
                "longitude": gps_payload.get("longitude"),
                "exif_present": bool(exif_payload) or bool(gps_payload),
            }
            asset_id = callbacks.save_asset(save_payload)

        if asset_id and callbacks.link_upload_asset and inputs.upload_id:
            callbacks.link_upload_asset(asset_id)
        metrics.record_asset_created(1 if asset_id else 0)

    if processed_cleanup and processed_path.exists():
        with contextlib.suppress(Exception):
            processed_path.unlink()
    if inputs.file.cleanup and inputs.file.path.exists():
        with contextlib.suppress(Exception):
            inputs.file.path.unlink()

    return IngestionResult(
        asset_id=asset_id,
        message_id=message_id,
        chat_id=inputs.channel_id,
        caption=caption,
        sha256=sha256,
        mime_type=mime_type,
        width=width,
        height=height,
        exif=exif_payload,
        gps=gps_payload,
        vision=vision_payload,
        telegram_file=telegram_file_meta,
        metrics=metrics,
    )


async def ingest_photo(
    *,
    data: "DataAccess",
    telegram: TelegramClient,
    openai: OpenAIClient | None,
    supabase: SupabaseClient | None,
    config: "UploadsConfig",
    context: UploadIngestionContext,
    file_path: Path,
    cleanup_file: bool = False,
    callbacks: IngestionCallbacks | None = None,
    input_overrides: Mapping[str, Any] | None = None,
) -> IngestionResult:
    metrics = context.metrics
    telemetry = context.telemetry_payload()

    vision_config = IngestionVisionConfig(
        enabled=config.vision_enabled,
        model=config.openai_vision_model,
    )

    ingestion_inputs = IngestionInputs(
        source=context.source,
        channel_id=config.assets_channel_id,
        file=IngestionFile(path=file_path, cleanup=cleanup_file),
        upload_id=context.upload_id,
        file_ref=context.storage_key,
        job_id=context.job_id,
        job_name=context.job_name,
        telemetry=telemetry,
        max_image_side=config.max_image_side,
        vision=vision_config,
    )

    if input_overrides:
        for key, value in input_overrides.items():
            if hasattr(ingestion_inputs, key):
                setattr(ingestion_inputs, key, value)
            else:
                logging.debug("Ignoring unknown ingestion override %s", key)

    def _log_tokens(payload: TokenUsagePayload) -> None:
        model = payload.get("model")
        if not model:
            return
        data.log_token_usage(
            model,
            payload.get("prompt_tokens"),
            payload.get("completion_tokens"),
            payload.get("total_tokens"),
            job_id=payload.get("job_id") or context.job_id,
            request_id=payload.get("request_id"),
        )

    ingestion_context = IngestionContext(
        telegram=telegram,
        metrics=metrics,
        openai=openai,
        supabase=supabase,
        token_logger=_log_tokens,
    )

    effective_callbacks = callbacks or IngestionCallbacks()
    if effective_callbacks.create_asset is None and effective_callbacks.save_asset is None:
        effective_callbacks.create_asset = lambda payload: data.create_asset(**payload)

    return await _ingest_photo_internal(
        ingestion_inputs,
        ingestion_context,
        effective_callbacks,
    )
