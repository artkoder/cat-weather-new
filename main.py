from __future__ import annotations

import asyncio
import contextlib
import gc
import html
import io
import json
import logging
import math
import mimetypes
import os
import random
import re
import secrets
import sqlite3
import tempfile
import time
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta, timezone
from datetime import time as dtime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, BinaryIO, ClassVar
from uuid import uuid4
from zoneinfo import ZoneInfo

import piexif
import psutil
from aiohttp import ClientSession, FormData, web
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:  # pragma: no cover - optional dependency fallback
    import qrcode  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback to text image
    qrcode = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from PIL import ImageCms  # type: ignore
except Exception:  # pragma: no cover - fallback when LittleCMS is unavailable
    ImageCms = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import exifread  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback when exifread is unavailable
    exifread = None  # type: ignore[assignment]

from api.pairing import PairingTokenError, normalize_pairing_token
from api.rate_limit import TokenBucketLimiter, create_rate_limit_middleware
from api.security import create_hmac_middleware
from api.uploads import (
    UploadMetricsRecorder,
    UploadsConfig,
    load_uploads_config,
    register_upload_jobs,
    setup_upload_routes,
)
from data_access import (
    Asset,
    DataAccess,
    Rubric,
    consume_pairing_token,
    create_device,
    create_pairing_token,
    get_asset_channel_id,
    get_recognition_channel_id,
    list_user_devices,
    revoke_device,
    rotate_device_secret,
)
from facts.loader import Fact, load_baltic_facts
from flowers_patterns import (
    FlowerKnowledgeBase,
    FlowerPattern,
    load_flowers_knowledge,
)
from ingestion import (
    IngestionCallbacks,
    UploadIngestionContext,
    extract_exif_datetimes,
    ingest_photo,
)
from jobs import Job, JobDelayed, JobQueue, cleanup_expired_records
from observability import (
    context,
    metrics_handler,
    observability_middleware,
    observe_health_latency,
    record_mobile_photo_ingested,
    record_rate_limit_drop,
    setup_logging,
)
from openai_client import OpenAIClient
from sea_selection import (
    NormalizedSky,
    STAGE_CONFIGS,
    StageConfig,
    calc_wave_penalty,
    sky_similarity,
)
from storage import create_storage_from_env
from supabase_client import SupabaseClient
from utils_wave import wave_m_to_score
from weather_migration import migrate_weather_publish_channels

if TYPE_CHECKING:
    from openai_client import OpenAIResponse

setup_logging()


# Prompt leak sanitization pattern (defense-in-depth)
PROMPT_LEAK_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:#{2,}|\*{2,}|[-=]{3,}|>+)\s*"
    r"(?:Верните JSON|JSON-ответ|построить результат|post text v1|"
    r"json_schema|schema|формат|шаблон|инструкция|промпт)\b",
    re.IGNORECASE,
)


def sanitize_prompt_leaks(text: str) -> str:
    """Remove prompt/schema leaks if detected (defense-in-depth)."""
    m = PROMPT_LEAK_PATTERN.search(text)
    if m:
        trimmed = text[: m.start()].rstrip()
        logging.warning(
            "SEA_RUBRIC caption_prompt_leak_detected trimmed_at_pos=%d original_len=%d trimmed_len=%d",
            m.start(),
            len(text),
            len(trimmed),
        )
        return trimmed
    return text


class LoggingMetricsEmitter:
    """Lightweight metrics sink that logs counter and timer updates."""

    def increment(self, name: str, value: float = 1.0) -> None:
        logging.debug("METRIC counter %s += %s", name, value)

    def observe(self, name: str, value: float) -> None:
        logging.debug("METRIC timer %s=%.3fms", name, value)


def _read_version_from_changelog() -> str:
    changelog_path = Path(__file__).resolve().parent / "CHANGELOG.md"
    try:
        with changelog_path.open("r", encoding="utf-8") as changelog:
            for line in changelog:
                line = line.strip()
                if line.startswith("## ["):
                    closing = line.find("]", 4)
                    if closing == -1:
                        continue
                    version = line[4:closing].strip()
                    if not version:
                        continue
                    if version.lower() == "unreleased":
                        return "unreleased"
                    return version
    except FileNotFoundError:
        logging.warning("CHANGELOG.md not found while resolving version")
    return "dev"


APP_VERSION = os.getenv("APP_VERSION") or _read_version_from_changelog()


# Default database path points to /data which is mounted as a Fly.io volume.
# This ensures information like registered channels and scheduled posts
# persists across deployments unless DB_PATH is explicitly overridden.
def _env_flag(value: str) -> bool:
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logging.warning("Invalid %s=%s, defaulting to %s", name, raw, default)
        return default


DB_PATH = os.getenv("DB_PATH", "/data/bot.db")
TZ_OFFSET = os.getenv("TZ_OFFSET", "+00:00")
SCHED_INTERVAL_SEC = int(os.getenv("SCHED_INTERVAL_SEC", "30"))
ASSETS_DEBUG_EXIF = _env_flag(os.getenv("ASSETS_DEBUG_EXIF", "0"))
DEBUG_SEA_PICK = _env_flag(os.getenv("DEBUG_SEA_PICK", "0"))
VISION_CONCURRENCY_RAW = os.getenv("VISION_CONCURRENCY", "1")
try:
    VISION_CONCURRENCY = max(1, int(VISION_CONCURRENCY_RAW))
except ValueError:
    logging.warning("Invalid VISION_CONCURRENCY=%s, defaulting to 1", VISION_CONCURRENCY_RAW)
    VISION_CONCURRENCY = 1
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
WMO_EMOJI = {
    0: "\u2600\ufe0f",
    1: "\U0001f324",
    2: "\u26c5",
    3: "\u2601\ufe0f",
    45: "\U0001f32b",
    48: "\U0001f32b",
    51: "\U0001f327",
    53: "\U0001f327",
    55: "\U0001f327",
    61: "\U0001f327",
    63: "\U0001f327",
    65: "\U0001f327",
    71: "\u2744\ufe0f",
    73: "\u2744\ufe0f",
    75: "\u2744\ufe0f",
    80: "\U0001f327",
    81: "\U0001f327",
    82: "\U0001f327",
    95: "\u26c8\ufe0f",
    96: "\u26c8\ufe0f",
    99: "\u26c8\ufe0f",
}

LOVE_COLLECTION_LINK = '<a href="https://t.me/addlist/sW-rkrslxqo1NTVi">📂 Полюбить 39</a>'


def build_rubric_link_block(rubric_code: str) -> str:
    if rubric_code == "sea":
        return LOVE_COLLECTION_LINK
    return ""


CLOUD_WORDS = {"облачн", "пасмурн", "ясн", "солнечн", "дожд", "гроза"}
NUM_PAT = re.compile(r"(\d+(?:[.,]\d+)?\s?(?:км/ч|м/с|%))", re.IGNORECASE)


def sanitize_sea_intro(text: str) -> tuple[str, list[str]]:
    removed_tokens: list[str] = []

    def _remove_numeric(match: re.Match[str]) -> str:
        token = match.group(0)
        if token:
            removed_tokens.append(token)
        return ""

    cleaned = NUM_PAT.sub(_remove_numeric, text)
    paragraphs = cleaned.split("\n\n")
    if paragraphs:
        intro = paragraphs[0]

        def _drop_stem(stem: str, value: str) -> str:
            pattern = re.compile(rf"\b\w*{stem}\w*\b", re.IGNORECASE)

            def _collect(match: re.Match[str]) -> str:
                token = match.group(0)
                if token:
                    removed_tokens.append(token)
                return ""

            return pattern.sub(_collect, value)

        for stem in CLOUD_WORDS:
            intro = _drop_stem(stem, intro)
        intro = re.sub(r"[ \t]{2,}", " ", intro)
        intro = re.sub(r"\s+([,.!?…])", r"\1", intro)
        intro_clean = intro.strip()
        if intro_clean:
            paragraphs[0] = intro_clean
        else:
            paragraphs = paragraphs[1:]

    sanitized = "\n\n".join(paragraphs)
    sanitized = re.sub(r"[ \t]{2,}", " ", sanitized)
    sanitized = re.sub(r"\s+([,.!?…])", r"\1", sanitized)
    sanitized = sanitized.strip()
    return sanitized, removed_tokens


def _isoformat_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt.isoformat().replace("+00:00", "Z")


def weather_emoji(code: int, is_day: int | None) -> str:
    emoji = WMO_EMOJI.get(code, "")
    if code == 0 and is_day == 0:
        return "\U0001f319"  # crescent moon
    return emoji


WEATHER_SEPARATOR = "\u2219"  # "∙" used to split header from original text
WEATHER_HEADER_PATTERN = re.compile(
    r"(°\s*[cf]?|шторм|м/с|ветер|давлен|влажн|осадки|old)",
    re.IGNORECASE,
)


PHOTO_MIME_TYPES: set[str] = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}


def is_photo_mime(mime: str | None) -> bool:
    if not mime:
        return False
    return mime.lower() in PHOTO_MIME_TYPES


def bytes_10mb() -> int:
    return 10 * 1024 * 1024


def detect_mime_and_size(
    path: str | os.PathLike[str] | Path,
) -> tuple[str | None, int]:
    file_path = Path(path)
    try:
        size = file_path.stat().st_size
    except FileNotFoundError:
        logging.warning("Unable to stat file for MIME detection: %s", path)
        return None, 0
    mime: str | None = None
    try:
        with Image.open(file_path) as image:
            format_name = image.format
            if format_name:
                mime = Image.MIME.get(format_name.upper()) or Image.MIME.get(format_name)
    except Exception:
        logging.debug("Failed to detect MIME via Pillow for %s", path, exc_info=True)
    if not mime:
        guessed, _ = mimetypes.guess_type(str(file_path))
        mime = guessed
    return mime.lower() if mime else None, size


WEATHER_ALLOWED_VALUES: set[str] = {
    "sunny",
    "partly_cloudy",
    "overcast",
    "rain",
    "snow",
    "fog",
    "night",
}

_PAIRING_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_PAIRING_DEFAULT_NAME = "Android"

MARINE_TAG_SYNONYMS: set[str] = {
    "sea",
    "seaside",
    "seashore",
    "seafront",
    "coast",
    "coastal",
    "coastline",
    "shore",
    "shoreline",
    "ocean",
    "oceanic",
    "marine",
    "maritime",
    "beach",
}

FRAMING_ALLOWED_VALUES: set[str] = {"close_up", "medium", "wide"}

FRAMING_ALIAS_MAP: dict[str, str] = {
    "closeup": "close_up",
    "close": "close_up",
    "medium_shot": "medium",
    "mediumshot": "medium",
    "wide_shot": "wide",
    "wideshot": "wide",
    "panorama": "wide",
    "aerial_shot": "wide",
    "detail": "close_up",
}

WEATHER_ALIAS_MAP: dict[str, str | None] = {
    "sunny": "sunny",
    "clear": "sunny",
    "ясно": "sunny",
    "солнечно": "sunny",
    "bright": "sunny",
    "indoor": "sunny",
    "inside": "sunny",
    "room": "sunny",
    "partly_cloudy": "partly_cloudy",
    "partlycloudy": "partly_cloudy",
    "переменная_облачность": "partly_cloudy",
    "облачно": "partly_cloudy",
    "cloudy": "overcast",
    "overcast": "overcast",
    "пасмурно": "overcast",
    "mostly_cloudy": "overcast",
    "rain": "rain",
    "rainy": "rain",
    "дождь": "rain",
    "дождливо": "rain",
    "drizzle": "rain",
    "shower": "rain",
    "sleet": "rain",
    "hail": "rain",
    "storm": "rain",
    "stormy": "rain",
    "thunderstorm": "rain",
    "гроза": "rain",
    "snow": "snow",
    "snowy": "snow",
    "snowfall": "snow",
    "снег": "snow",
    "снежно": "snow",
    "blizzard": "snow",
    "fog": "fog",
    "foggy": "fog",
    "mist": "fog",
    "дымка": "fog",
    "туман": "fog",
    "haze": "fog",
    "smog": "fog",
    "night": "night",
    "clear_night": "night",
    "ночь": "night",
    "twilight": "night",
    "dusk": "night",
    "evening": "night",
}

FLOWERS_PREVIEW_MAX_LENGTH = 4000

WEATHER_TAG_TRANSLATIONS: dict[str, str] = {
    "sunny": "солнечно",
    "partly_cloudy": "переменная облачность",
    "overcast": "пасмурно",
    "rain": "дождь",
    "snow": "снег",
    "fog": "туман",
    "night": "ночь",
}

SEASON_BY_MONTH: dict[int, str] = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}

SEASON_TRANSLATIONS: dict[str, str] = {
    "spring": "весна",
    "summer": "лето",
    "autumn": "осень",
    "fall": "осень",
    "winter": "зима",
}


SEASON_ADJACENCY: dict[str, set[str]] = {
    "spring": {"winter", "summer"},
    "summer": {"spring", "autumn"},
    "autumn": {"summer", "winter"},
    "winter": {"autumn", "spring"},
}


def classify_wind_kph(speed_kmh: float | None) -> str | None:
    if speed_kmh is None:
        return None
    try:
        value = float(speed_kmh)
    except (TypeError, ValueError):
        return None
    if value >= 35.0:
        return "very_strong"
    if value >= 25.0:
        return "strong"
    return None


def resolve_wind_speed(value: Any, units: str | None) -> tuple[float | None, float | None]:
    if value is None:
        return None, None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None, None
    unit = (units or "").strip().lower()
    if unit in {"km/h", "kmh", "kilometres_per_hour", "kilometers_per_hour"}:
        kmh = numeric
        ms = kmh / 3.6
        return kmh, ms
    if unit in {"m/s", "ms", "meter_per_second", "metres_per_second", "meters_per_second"}:
        ms = numeric
        kmh = ms * 3.6
        return kmh, ms
    # Default to treating the value as km/h when units are unknown
    kmh = numeric
    ms = kmh / 3.6
    return kmh, ms


def bucket_clouds(cloud_pct: float | None) -> str | None:
    if cloud_pct is None:
        return None
    try:
        value = float(cloud_pct)
    except (TypeError, ValueError):
        return None
    if value < 0:
        value = 0.0
    if value <= 20:
        return "clear"
    if value <= 40:
        return "mostly_clear"
    if value <= 60:
        return "partly_cloudy"
    if value <= 80:
        return "mostly_cloudy"
    return "overcast"


def compatible_skies(bucket: str | None, daypart: str | None) -> set[NormalizedSky]:
    mapping = {
        "clear": {"sunny", "mostly_clear", "partly_cloudy"},
        "mostly_clear": {"sunny", "mostly_clear", "partly_cloudy"},
        "partly_cloudy": {"sunny", "mostly_clear", "partly_cloudy", "mostly_cloudy"},
        "mostly_cloudy": {"mostly_cloudy", "overcast"},
        "overcast": {"mostly_cloudy", "overcast"},
    }
    normalized_daypart = daypart or "day"
    weather_tags = mapping.get(bucket)
    if weather_tags is None:
        weather_tags = {"sunny", "mostly_clear", "partly_cloudy", "mostly_cloudy", "overcast"}

    allowed = {NormalizedSky(daypart=normalized_daypart, weather_tag=tag) for tag in weather_tags}

    if normalized_daypart == "day" and bucket in {"clear", "mostly_clear"}:
        for evening_tag in {"sunny", "mostly_clear"}:
            allowed.add(NormalizedSky(daypart="evening", weather_tag=evening_tag))

    return allowed


def compute_season_window(reference: date, window_days: int = 45) -> set[str]:
    seasons: set[str] = set()
    for offset in range(-window_days, window_days + 1):
        candidate = reference + timedelta(days=offset)
        seasons.add(SEASON_BY_MONTH.get(candidate.month, "winter"))
    return seasons


def map_hour_to_day_part(hour: int) -> str:
    """Map hour (0-23) to day part.

    morning: 05:00–10:59
    day:     11:00–16:59
    evening: 17:00–21:59
    night:   22:00–04:59
    """
    if 5 <= hour < 11:
        return "morning"
    elif 11 <= hour < 17:
        return "day"
    elif 17 <= hour < 22:
        return "evening"
    else:
        return "night"


def is_in_season_window(shot_doy: int | None, *, today_doy: int, window: int = 45) -> bool:
    """
    Check if a shot day-of-year falls within a seasonal window around today.

    Args:
        shot_doy: Day of year when photo was taken (1-366), or None
        today_doy: Current day of year (1-366)
        window: Number of days before and after today to include (default 45)

    Returns:
        True if shot_doy is within the window, False otherwise (including when shot_doy is None)
    """
    if shot_doy is None:
        return False

    try:
        shot_value = int(shot_doy)
    except (TypeError, ValueError):
        return False

    try:
        today_value = int(today_doy)
    except (TypeError, ValueError):
        return False

    try:
        window_value = int(window)
    except (TypeError, ValueError):
        window_value = 0

    if window_value < 0:
        window_value = 0

    if not 1 <= shot_value <= 366:
        return False
    if not 1 <= today_value <= 366:
        return False

    period = 365
    normalized_shot = shot_value if shot_value <= 365 else 365
    normalized_today = today_value if today_value <= 365 else 365

    forward_dist = (normalized_shot - normalized_today) % period
    backward_dist = (normalized_today - normalized_shot) % period
    min_dist = min(forward_dist, backward_dist)

    return min_dist <= window_value


def season_match(season_guess: str | None, allowed: set[str]) -> bool:
    if not season_guess:
        return False
    normalized = str(season_guess).strip().lower()
    return normalized in allowed


ASSET_VISION_V1_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "asset_vision_v1",
    "description": (
        "Структурированное описание фото для классификации рубрик, "
        "угадывания города и оценки безопасности."
    ),
    "additionalProperties": False,
    "properties": {
        "arch_view": {
            "type": "boolean",
            "description": "Присутствует ли в кадре архитектурный ракурс (здания, фасады, панорамы).",
        },
        "caption": {
            "type": "string",
            "description": "Короткое описание основного сюжета (на русском языке).",
            "minLength": 1,
        },
        "objects": {
            "type": "array",
            "description": (
                "Список заметных объектов или деталей. Если присутствуют цветы, укажи их вид."
            ),
            "items": {
                "type": "string",
                "minLength": 1,
            },
            "default": [],
        },
        "is_outdoor": {
            "type": "boolean",
            "description": "True, если сцена снята на улице (иначе — в помещении).",
        },
        "guess_country": {
            "type": ["string", "null"],
            "description": "Предполагаемая страна, если есть контекст.",
        },
        "guess_city": {
            "type": ["string", "null"],
            "description": "Предполагаемый город, если распознаётся.",
        },
        "location_confidence": {
            "type": "number",
            "description": "Числовая уверенность в локации (0 — нет уверенности, 1 — полностью уверен).",
            "minimum": 0,
            "maximum": 1,
        },
        "landmarks": {
            "type": "array",
            "description": "Имена распознанных достопримечательностей или ориентиров.",
            "items": {
                "type": "string",
                "minLength": 1,
            },
            "default": [],
        },
        "tags": {
            "type": "array",
            "description": (
                "Набор тегов (на английском в нижнем регистре) для downstream-логики: architecture, flowers, people, animals и т.п."
            ),
            "items": {
                "type": "string",
                "minLength": 1,
            },
            "minItems": 3,
            "maxItems": 12,
            "default": [],
        },
        "framing": {
            "type": "string",
            "description": (
                "Кадровка/ракурс снимка. Используй один из вариантов: close_up, medium, " "wide."
            ),
            "enum": [
                "close_up",
                "medium",
                "wide",
            ],
        },
        "architecture_close_up": {
            "type": "boolean",
            "description": "Есть ли крупный план архитектурных деталей.",
        },
        "architecture_wide": {
            "type": "boolean",
            "description": "Есть ли широкий архитектурный план или панорама.",
        },
        "weather_image": {
            "type": "string",
            "description": (
                "Краткое описание погодных условий на фото (на английском). Выбирай из категорий: "
                "sunny, partly_cloudy, overcast, rain, snow, fog, night."
            ),
            "enum": [
                "sunny",
                "partly_cloudy",
                "overcast",
                "rain",
                "snow",
                "fog",
                "night",
            ],
        },
        "is_sea": {
            "type": "boolean",
            "description": "True, если на фото море, океан, пляж или береговая линия.",
        },
        "sea_wave_score": {
            "type": ["number", "null"],
            "description": "Оценка волнения моря по шкале 0..10 (0 — гладь, 10 — сильный шторм).",
            "minimum": 0,
            "maximum": 10,
        },
        "photo_sky": {
            "type": "string",
            "description": "Класс неба на снимке.",
            "enum": [
                "sunny",
                "partly_cloudy",
                "mostly_cloudy",
                "overcast",
                "night",
                "unknown",
            ],
        },
        "sky_visible": {
            "type": "boolean",
            "description": "True, если на фото видимо небо (даже частично), иначе False.",
        },
        "is_sunset": {
            "type": "boolean",
            "description": "True, если на фото закат или выраженные закатные оттенки.",
        },
        "season_guess": {
            "type": ["string", "null"],
            "description": "Предполагаемый сезон (spring, summer, autumn, winter) или null, если неясно.",
            "enum": [
                "spring",
                "summer",
                "autumn",
                "winter",
                None,
            ],
        },
        "arch_style": {
            "type": ["object", "null"],
            "description": (
                "Предполагаемый архитектурный стиль. Либо null, либо объект с label (строка) "
                "и confidence (число 0..1)."
            ),
            "additionalProperties": False,
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Название архитектурного стиля (на английском).",
                    "minLength": 1,
                },
                "confidence": {
                    "type": ["number", "null"],
                    "description": "Уверенность в определении стиля (0 — неизвестно, 1 — уверен).",
                    "minimum": 0,
                    "maximum": 1,
                },
            },
            "required": ["label", "confidence"],
        },
        "safety": {
            "type": "object",
            "description": ("Информация о чувствительном контенте: nsfw и краткая причина."),
            "additionalProperties": False,
            "properties": {
                "nsfw": {"type": "boolean"},
                "reason": {
                    "type": "string",
                    "description": "Краткое пояснение статуса безопасности (на русском).",
                    "minLength": 1,
                },
            },
            "required": ["nsfw", "reason"],
        },
    },
    "required": [
        "arch_view",
        "caption",
        "objects",
        "is_outdoor",
        "guess_country",
        "guess_city",
        "location_confidence",
        "landmarks",
        "tags",
        "framing",
        "architecture_close_up",
        "architecture_wide",
        "weather_image",
        "is_sea",
        "sea_wave_score",
        "photo_sky",
        "sky_visible",
        "is_sunset",
        "season_guess",
        "safety",
    ],
}


CHANNEL_PICKER_PAGE_SIZE = 6
CITY_PICKER_PAGE_SIZE = 8
CHANNEL_SEARCH_CHARSETS = {
    "rus": list("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"),
    "lat": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    "num": list("0123456789"),
}
CHANNEL_SEARCH_LABELS = {
    "rus": "АБВ",
    "lat": "ABC",
    "num": "123",
}
CHANNEL_SEARCH_CONTROLS = [
    ("⬅️", "rubric_channel_search_del"),
    ("Пробел", "rubric_channel_search_add:20"),
    ("Сбросить", "rubric_channel_search_clear"),
    ("Готово", "rubric_channel_search_done"),
]
CITY_SEARCH_CONTROLS = [
    ("⬅️", "rubric_city_search_del"),
    ("Пробел", "rubric_city_search_add:20"),
    ("Сбросить", "rubric_city_search_clear"),
    ("Готово", "rubric_city_search_done"),
]


DEFAULT_RUBRIC_PRESETS: dict[str, dict[str, Any]] = {
    "flowers": {
        "title": "Цветы",
        "config": {
            "enabled": False,
            "schedules": [],
            "assets": {"min": 1, "max": 6, "categories": ["flowers"]},
            "weather_city": "Kaliningrad",
        },
    },
    "guess_arch": {
        "title": "Угадай архитектуру",
        "config": {
            "enabled": False,
            "schedules": [],
            "assets": {"min": 4, "max": 4, "categories": ["architecture"]},
            "weather_city": "Kaliningrad",
            "overlays_dir": "overlays",
        },
    },
    "sea": {
        "title": "Море / Закат на море",
        "config": {
            "enabled": False,
            "schedules": [],
            "assets": {"min": 1, "max": 1, "categories": ["sea"]},
            "sea_id": 1,
            "enable_facts": True,
        },
    },
}


def apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply SQL migrations stored in the migrations directory."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    applied = {row["id"] for row in conn.execute("SELECT id FROM schema_migrations")}
    migration_files = sorted(p for p in MIGRATIONS_DIR.iterdir() if p.suffix in {".sql", ".py"})
    for path in migration_files:
        migration_id = path.stem
        if migration_id in applied:
            continue
        logging.info("Applying migration %s", migration_id)
        with conn:
            if path.suffix == ".sql":
                sql = path.read_text(encoding="utf-8")
                conn.executescript(sql)
            else:
                namespace: dict[str, Any] = {}
                exec(path.read_text(encoding="utf-8"), namespace)
                runner = namespace.get("run")
                if not callable(runner):
                    raise ValueError(f"Migration {migration_id} missing run()")
                runner(conn)
            conn.execute(
                "INSERT INTO schema_migrations (id, applied_at) VALUES (?, ?)",
                (migration_id, datetime.utcnow().isoformat()),
            )


CREATE_TABLES = [
    """CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            is_superadmin INTEGER DEFAULT 0,
            tz_offset TEXT
        )""",
    """CREATE TABLE IF NOT EXISTS pending_users (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            requested_at TEXT
        )""",
    """CREATE TABLE IF NOT EXISTS rejected_users (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            rejected_at TEXT
        )""",
    """CREATE TABLE IF NOT EXISTS channels (
            chat_id INTEGER PRIMARY KEY,
            title TEXT
        )""",
    """CREATE TABLE IF NOT EXISTS schedule (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_chat_id INTEGER,
            message_id INTEGER,
            target_chat_id INTEGER,
            publish_time TEXT,
            sent INTEGER DEFAULT 0,
            sent_at TEXT
        )""",
    """CREATE TABLE IF NOT EXISTS cities (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            UNIQUE(name)
        )""",
    """CREATE TABLE IF NOT EXISTS weather_cache_day (
            city_id INTEGER NOT NULL,
            day DATE NOT NULL,
            temperature REAL,
            weather_code INTEGER,
            wind_speed REAL,
            PRIMARY KEY (city_id, day)
        )""",
    """CREATE TABLE IF NOT EXISTS weather_cache_hour (
            city_id INTEGER NOT NULL,
            timestamp DATETIME NOT NULL,
            temperature REAL,
            weather_code INTEGER,
            wind_speed REAL,
            is_day INTEGER,
            PRIMARY KEY (city_id, timestamp)
        )""",
    """CREATE TABLE IF NOT EXISTS seas (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            UNIQUE(name)
        )""",
    """CREATE TABLE IF NOT EXISTS sea_cache (
            sea_id INTEGER PRIMARY KEY,
            updated TEXT,
            current REAL,
            morning REAL,
            day REAL,
            evening REAL,
            night REAL,
            wave REAL,
            morning_wave REAL,
            day_wave REAL,
            evening_wave REAL,
            night_wave REAL
        )""",
    """CREATE TABLE IF NOT EXISTS sea_conditions (
            sea_id INTEGER PRIMARY KEY,
            updated TEXT,
            wave_height_m REAL,
            wind_speed_10m_ms REAL,
            wind_speed_10m_kmh REAL,
            wind_gusts_10m_ms REAL,
            wind_gusts_10m_kmh REAL,
            wind_units TEXT,
            wind_gusts_units TEXT,
            wind_time_ref TEXT,
            cloud_cover_pct REAL
        )""",
    """CREATE TABLE IF NOT EXISTS weather_posts (

            chat_id BIGINT NOT NULL,
            message_id BIGINT NOT NULL,
            template TEXT NOT NULL,
            base_text TEXT,

            base_caption TEXT,
            reply_markup TEXT,

            UNIQUE(chat_id, message_id)
        )""",
    """CREATE TABLE IF NOT EXISTS asset_channel (
            channel_id INTEGER PRIMARY KEY
        )""",
    """CREATE TABLE IF NOT EXISTS recognition_channel (
            channel_id INTEGER PRIMARY KEY
        )""",
    """CREATE TABLE IF NOT EXISTS weather_cache_period (
            city_id INTEGER PRIMARY KEY,
            updated TEXT,
            morning_temp REAL,
            morning_code INTEGER,
            morning_wind REAL,
            day_temp REAL,
            day_code INTEGER,
            day_wind REAL,
            evening_temp REAL,
            evening_code INTEGER,
            evening_wind REAL,
            night_temp REAL,
            night_code INTEGER,
            night_wind REAL
        )""",
    """CREATE TABLE IF NOT EXISTS latest_weather_post (
            chat_id BIGINT,
            message_id BIGINT,
            published_at TEXT
        )""",
    """CREATE TABLE IF NOT EXISTS weather_link_posts (
            chat_id BIGINT NOT NULL,
            message_id BIGINT NOT NULL,
            base_markup TEXT,
            button_texts TEXT,
            UNIQUE(chat_id, message_id)
        )""",
    """CREATE TABLE IF NOT EXISTS amber_channels (
            channel_id INTEGER PRIMARY KEY
        )""",
    """CREATE TABLE IF NOT EXISTS amber_state (
            sea_id INTEGER PRIMARY KEY,
            storm_start TEXT,
            active INTEGER DEFAULT 0
        )""",
]


class Bot:
    def __init__(self, token: str, db_path: str):
        self.token = token
        self.api_url = f"https://api.telegram.org/bot{token}"
        self.dry_run = token == "dummy"
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        apply_migrations(self.db)
        global TZ_OFFSET
        TZ_OFFSET = os.getenv("TZ_OFFSET", "+00:00")
        for stmt in CREATE_TABLES:
            self.db.execute(stmt)
        self.db.commit()
        self._pairing_rng = random.SystemRandom()
        self.data = DataAccess(self.db)
        self._rubric_category_cache: dict[str, int] = {}
        self._ensure_default_rubrics()
        if migrate_weather_publish_channels(self.db, tz_offset=TZ_OFFSET):
            self.db.commit()
        self.jobs = JobQueue(self.db, concurrency=1)
        self.jobs.register_handler("ingest", self._job_ingest)
        self.jobs.register_handler("vision", self._job_vision)
        self.jobs.register_handler("publish_rubric", self._job_publish_rubric)
        self.jobs.add_periodic_task(
            300,
            lambda: cleanup_expired_records(self.db),
            name="cleanup_expired_records",
        )
        self.openai = OpenAIClient(os.getenv("4O_API_KEY"))
        self.supabase = SupabaseClient()
        self.uploads_config: UploadsConfig = load_uploads_config()
        self.upload_metrics: UploadMetricsRecorder | None = None
        self._model_limits = self._load_model_limits()
        asset_dir = os.getenv("ASSET_STORAGE_DIR")
        self.asset_storage = Path(asset_dir).expanduser() if asset_dir else Path("/tmp/bot_assets")
        self.asset_storage.mkdir(parents=True, exist_ok=True)
        self._last_geocode_at: datetime | None = None
        ttl_hours_raw = os.getenv("REVGEO_CACHE_TTL_HOURS", "24")
        try:
            ttl_hours = float(ttl_hours_raw)
        except ValueError:
            logging.warning("Invalid REVGEO_CACHE_TTL_HOURS=%s, defaulting to 24h", ttl_hours_raw)
            ttl_hours = 24.0
        fail_ttl_minutes_raw = os.getenv("REVGEO_FAIL_TTL_MINUTES", "15")
        try:
            fail_ttl_minutes = float(fail_ttl_minutes_raw)
        except ValueError:
            logging.warning(
                "Invalid REVGEO_FAIL_TTL_MINUTES=%s, defaulting to 15m",
                fail_ttl_minutes_raw,
            )
            fail_ttl_minutes = 15.0
        self._revgeo_ttl = timedelta(hours=max(ttl_hours, 0.0) or 24.0)
        self._revgeo_fail_ttl = timedelta(minutes=max(fail_ttl_minutes, 0.0) or 15.0)
        self._revgeo_cache: dict[tuple[float, float], tuple[dict[str, Any], datetime]] = {}
        self._revgeo_fallback_cache: dict[tuple[float, float], tuple[str, datetime]] = {}
        self._revgeo_fail_cache: dict[tuple[float, float], datetime] = {}
        self._revgeo_semaphore = asyncio.Semaphore(1)
        self._vision_semaphore = asyncio.Semaphore(VISION_CONCURRENCY)
        self._twogis_api_key = os.getenv("TWOGIS_API_KEY")
        self._twogis_backoff_seconds = 1.0
        self._shoreline_cache: dict[tuple[float, float], tuple[bool, datetime]] = {}
        self._shoreline_fail_cache: dict[tuple[float, float], datetime] = {}
        self._sea_publish_guard: dict[str, float] = {}
        # ensure new columns exist when upgrading
        for table, column in (
            ("users", "username"),
            ("users", "tz_offset"),
            ("pending_users", "username"),
            ("rejected_users", "username"),
            ("weather_posts", "template"),
            ("weather_posts", "base_text"),
            ("weather_posts", "base_caption"),
            ("weather_posts", "reply_markup"),
            ("sea_cache", "updated"),
            ("sea_cache", "current"),
            ("sea_cache", "morning"),
            ("sea_cache", "day"),
            ("sea_cache", "evening"),
            ("sea_cache", "night"),
            ("sea_cache", "wave"),
            ("sea_cache", "morning_wave"),
            ("sea_cache", "day_wave"),
            ("sea_cache", "evening_wave"),
            ("sea_cache", "night_wave"),
        ):
            cur = self.db.execute(f"PRAGMA table_info({table})")
            names = [r[1] for r in cur.fetchall()]
            if column not in names:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")
        self.db.commit()
        self.pending = {}
        self.rubric_dashboards: dict[int, dict[str, int]] = {}
        self.rubric_overview_messages: dict[int, dict[str, dict[str, int]]] = {}
        self.failed_fetches: dict[int, tuple[int, datetime]] = {}
        self.weather_assets_channel_id = self.get_weather_assets_channel()
        self.recognition_channel_id = self.get_recognition_channel()
        self.uploads_config = replace(
            self.uploads_config,
            assets_channel_id=self.weather_assets_channel_id,
        )

        self.session: ClientSession | None = None
        self.running = False
        self.manual_buttons: dict[tuple[int, int], dict[str, list[list[dict]]]] = {}
        self.rubric_pending_runs: dict[tuple[int, str], str] = {}
        self.pending_flowers_previews: dict[int, dict[str, Any]] = {}
        self.flowers_kb: FlowerKnowledgeBase | None = None
        try:
            self.flowers_kb = load_flowers_knowledge()
        except Exception:
            logging.exception("Failed to load flowers knowledge base")
        self._backfill_waves_lock = asyncio.Lock()
        self._backfill_waves_running = False

    def _generate_pairing_code(self) -> str:
        length = self._pairing_rng.randint(6, 8)
        return "".join(self._pairing_rng.choice(_PAIRING_ALPHABET) for _ in range(length))

    def _get_active_pairing_token(self, user_id: int) -> tuple[str, datetime, str] | None:
        now = datetime.utcnow()
        cur = self.db.execute(
            """
            SELECT code, device_name, expires_at
            FROM pairing_tokens
            WHERE user_id=? AND used_at IS NULL
            ORDER BY expires_at DESC
            LIMIT 1
            """,
            (user_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        if isinstance(row, sqlite3.Row):
            code = str(row["code"])
            device_name = str(row["device_name"])
            expires_raw = row["expires_at"]
        else:
            code = str(row[0])
            device_name = str(row[1])
            expires_raw = row[2]
        try:
            expires_at = datetime.fromisoformat(str(expires_raw))
        except (TypeError, ValueError):
            return None
        if expires_at <= now:
            return None
        return code, expires_at, device_name

    def _issue_pairing_token(self, user_id: int, device_name: str) -> tuple[str, datetime]:
        normalized_name = (device_name or "").strip() or _PAIRING_DEFAULT_NAME
        for _ in range(20):
            code = self._generate_pairing_code()
            cur = self.db.execute(
                "SELECT user_id FROM pairing_tokens WHERE code=?",
                (code,),
            )
            row = cur.fetchone()
            if row:
                existing_user = int(row["user_id"]) if isinstance(row, sqlite3.Row) else int(row[0])
                if existing_user != user_id:
                    continue
            with self.db:
                self.db.execute(
                    "DELETE FROM pairing_tokens WHERE user_id=? AND used_at IS NULL",
                    (user_id,),
                )
                create_pairing_token(
                    self.db,
                    code=code,
                    user_id=user_id,
                    device_name=normalized_name,
                )
            cur = self.db.execute(
                "SELECT expires_at FROM pairing_tokens WHERE code=?",
                (code,),
            )
            saved = cur.fetchone()
            if saved:
                expires_raw = saved["expires_at"] if isinstance(saved, sqlite3.Row) else saved[0]
                try:
                    expires_at = datetime.fromisoformat(str(expires_raw))
                except (TypeError, ValueError):
                    expires_at = datetime.utcnow() + timedelta(minutes=10)
                logging.info("PAIR code issued user=%s name=%s", user_id, normalized_name)
                return code, expires_at
        raise RuntimeError("Failed to generate pairing token")

    @staticmethod
    def _format_pairing_message(code: str, expires_at: datetime, existing: bool) -> str:
        if existing:
            remaining = max(0, int((expires_at - datetime.utcnow()).total_seconds()))
            if remaining <= 60:
                return (
                    f"Активный код для привязки: {code}. Истекает менее чем через минуту. "
                    "Введите код в приложении."
                )
            minutes = max(1, math.ceil(remaining / 60))
            return (
                f"Активный код для привязки: {code}. Истекает через {minutes} мин. "
                "Введите код в приложении."
            )
        return f"Код для привязки: {code}. Действует 10 минут. Введите код в приложении."

    async def _send_pairing_code_message(
        self,
        user_id: int,
        code: str,
        expires_at: datetime,
        *,
        existing: bool,
        message: Mapping[str, Any] | None = None,
    ) -> None:
        keyboard = {
            "inline_keyboard": [
                [
                    {
                        "text": "Сгенерировать новый",
                        "callback_data": "pairing_regen",
                    }
                ]
            ]
        }
        payload = {
            "chat_id": user_id,
            "text": self._format_pairing_message(code, expires_at, existing),
            "reply_markup": keyboard,
        }
        if message and message.get("message_id"):
            payload["message_id"] = message["message_id"]
            await self.api_request("editMessageText", payload)
        else:
            await self.api_request("sendMessage", payload)

    def _render_pairing_qr(self, code: str) -> io.BytesIO:
        payload_text = f"catweather://pair?token={code}"
        if qrcode is not None:
            qr_image = qrcode.make(payload_text)
        else:
            size = 320
            qr_image = Image.new("RGB", (size, size), color="white")
            draw = ImageDraw.Draw(qr_image)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 36)
            except Exception:  # pragma: no cover - fallback when font missing
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), payload_text, font=font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            position = ((size - width) // 2, (size - height) // 2)
            draw.rectangle((0, 0, size - 1, size - 1), outline="black", width=4)
            draw.text(position, payload_text, fill="black", font=font)
        buffer = io.BytesIO()
        qr_image.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer

    def _format_mobile_caption(
        self,
        code: str,
        expires_at: datetime,
        devices: Sequence[Mapping[str, Any]],
    ) -> str:
        now = datetime.utcnow()
        remaining = int((expires_at - now).total_seconds())
        if remaining <= 0:
            ttl_line = "Срок действия истёк — сгенерируйте новый код."
        else:
            if remaining <= 60:
                ttl_text = "менее минуты"
            else:
                minutes, seconds = divmod(remaining, 60)
                if minutes < 60:
                    ttl_text = f"{minutes} мин"
                else:
                    hours, minutes = divmod(minutes, 60)
                    if hours < 24:
                        ttl_text = f"{hours} ч {minutes} мин"
                    else:
                        days, hours = divmod(hours, 24)
                        ttl_text = f"{days} дн {hours} ч"
            expiry_dt = expires_at
            if expiry_dt.tzinfo is None:
                expiry_dt = expiry_dt.replace(tzinfo=UTC)
            else:
                expiry_dt = expiry_dt.astimezone(UTC)
            expires_text = expiry_dt.strftime("%Y-%m-%d %H:%M UTC")
            ttl_line = f"Действует ещё {ttl_text} (до {expires_text})."

        caption_lines = [
            "<b>Привязка мобильного приложения</b>",
            f"Код: <code>{html.escape(code)}</code>",
            ttl_line,
        ]
        max_caption_length = 1024

        def _caption_length(lines: Sequence[str]) -> int:
            if not lines:
                return 0
            return sum(len(line) for line in lines) + len(lines) - 1

        current_length = _caption_length(caption_lines)

        def _can_append(line: str) -> bool:
            extra = len(line)
            if caption_lines:
                extra += 1
            return current_length + extra <= max_caption_length

        def _append_line(line: str) -> None:
            nonlocal current_length
            extra = len(line)
            if caption_lines:
                extra += 1
            caption_lines.append(line)
            current_length += extra

        def _format_timestamp(raw: str | None) -> str | None:
            if not raw:
                return None
            value = str(raw).strip()
            if not value:
                return None
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError:
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")

        if devices:
            if _can_append(""):
                _append_line("")
            if _can_append("<b>Устройства</b>:"):
                _append_line("<b>Устройства</b>:")
            for index, device in enumerate(devices):
                name = html.escape(str(device.get("name") or _PAIRING_DEFAULT_NAME))
                revoked_raw = device.get("revoked_at") or ""
                status = "отозван" if str(revoked_raw).strip() else "активен"
                last_seen = _format_timestamp(device.get("last_seen_at"))
                entry = f"• {name} — {status}"
                if last_seen:
                    entry += f" (последнее событие {last_seen})"
                if _can_append(entry):
                    _append_line(entry)
                    continue
                omitted_count = len(devices) - index
                if omitted_count > 0:
                    summary_line = f"… и ещё {omitted_count} устройств"
                    if not _can_append(summary_line):
                        summary_line = "…"
                    if _can_append(summary_line):
                        _append_line(summary_line)
                break
        else:
            if _can_append(""):
                _append_line("")
            if _can_append("Нет активных устройств."):
                _append_line("Нет активных устройств.")

        return "\n".join(caption_lines)

    def _build_mobile_keyboard(
        self, devices: Sequence[Mapping[str, Any]]
    ) -> dict[str, list[list[dict[str, str]]]]:
        keyboard: list[list[dict[str, str]]] = []
        for device in devices:
            device_id = str(device.get("id"))
            name = str(device.get("name") or _PAIRING_DEFAULT_NAME)
            label = name if len(name) <= 24 else f"{name[:21]}…"
            revoked_raw = str(device.get("revoked_at") or "").strip()
            buttons: list[dict[str, str]] = []
            if not revoked_raw:
                buttons.append(
                    {
                        "text": f"🔁 {label}",
                        "callback_data": f"dev:rotate:{device_id}",
                    }
                )
            buttons.append(
                {
                    "text": f"🛑 {label}",
                    "callback_data": f"dev:revoke:{device_id}",
                }
            )
            keyboard.append(buttons)
        keyboard.append([{"text": "🔄 Новый код", "callback_data": "pair:new"}])
        return {"inline_keyboard": keyboard}

    async def _send_mobile_pairing_card(
        self,
        user_id: int,
        code: str,
        expires_at: datetime,
        devices: Sequence[Mapping[str, Any]],
        *,
        message: Mapping[str, Any] | None = None,
        replace_photo: bool = False,
    ) -> None:
        active_devices = [
            device for device in devices if not str(device.get("revoked_at") or "").strip()
        ]
        caption = self._format_mobile_caption(code, expires_at, active_devices)
        keyboard = self._build_mobile_keyboard(active_devices)
        if message and message.get("message_id"):
            message_id = message["message_id"]
            if replace_photo:
                buffer = self._render_pairing_qr(code)
                media = {
                    "type": "photo",
                    "media": "attach://photo",
                    "caption": caption,
                    "parse_mode": "HTML",
                }
                payload = {
                    "chat_id": user_id,
                    "message_id": message_id,
                    "media": media,
                    "reply_markup": keyboard,
                }
                files = {"photo": ("pairing.png", buffer, "image/png")}
                await self.api_request_multipart(
                    "editMessageMedia",
                    payload,
                    files=files,
                )
            else:
                payload = {
                    "chat_id": user_id,
                    "message_id": message_id,
                    "caption": caption,
                    "parse_mode": "HTML",
                    "reply_markup": keyboard,
                }
                await self.api_request("editMessageCaption", payload)
            return
        buffer = self._render_pairing_qr(code)
        payload = {
            "chat_id": user_id,
            "caption": caption,
            "parse_mode": "HTML",
            "reply_markup": keyboard,
        }
        files = {"photo": ("pairing.png", buffer, "image/png")}
        await self.api_request_multipart("sendPhoto", payload, files=files)

    def _ensure_default_rubrics(self) -> None:
        created: list[str] = []
        for code, preset in DEFAULT_RUBRIC_PRESETS.items():
            if self.data.get_rubric_by_code(code):
                continue
            title = preset.get("title") or code.title()
            raw_config = preset.get("config") or {}
            config = deepcopy(raw_config)
            self.data.upsert_rubric(code, title, config=config)
            created.append(code)
        if created:
            logging.info("Initialized default rubrics: %s", ", ".join(created))
            self._rubric_category_cache.clear()

    @staticmethod
    def _is_convertible_image_document(asset: Asset) -> bool:
        """Return True when a Telegram document can be safely converted to a photo."""

        if asset.kind != "document":
            return False

        mime = (asset.mime_type or "").lower()
        if mime.startswith("image/") and not mime.endswith("svg+xml"):
            return True

        if asset.file_name:
            ext = Path(asset.file_name).suffix.lower()
            if ext in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".heic", ".heif", ".bmp"}:
                return True

        return False

    def _reencode_to_jpeg_under_10mb(self, local_path: str | os.PathLike[str] | Path) -> Path:
        source_path = Path(local_path)
        fd, temp_name = tempfile.mkstemp(prefix="tg-photo-", suffix=".jpg")
        os.close(fd)
        temp_path = Path(temp_name)
        image: Image.Image | None = None
        try:
            with Image.open(source_path) as original:
                transposed = ImageOps.exif_transpose(original)
                if transposed is not original:
                    working = transposed
                else:
                    working = original.copy()
            if working.mode not in {"RGB", "L"}:
                image = working.convert("RGB")
                working.close()
            elif working.mode == "L":
                image = working.convert("RGB")
                working.close()
            else:
                image = working
            qualities = [95, 92, 88, 84, 80, 76, 72, 68, 64, 60, 56, 52, 48, 44, 40, 36, 32]
            for quality in qualities:
                image.save(
                    temp_path,
                    format="JPEG",
                    quality=quality,
                    progressive=True,
                    optimize=True,
                    exif=b"",
                )
                if temp_path.stat().st_size <= bytes_10mb():
                    break
            else:
                logging.warning("Unable to reduce %s below 10MB at lowest quality", source_path)
            return temp_path
        except Exception:
            self._remove_file(str(temp_path))
            raise
        finally:
            if image is not None:
                image.close()
            gc.collect()

    def _prepare_photo_for_upload(
        self, local_path: str
    ) -> tuple[Path, Path | None, str, str, str, str | None, int | None]:
        mime, size = detect_mime_and_size(local_path)
        send_path = Path(local_path)
        cleanup_path: Path | None = None
        content_type = mime or "image/jpeg"
        mode = "original"
        if not (mime and is_photo_mime(mime) and size <= bytes_10mb()):
            cleanup_path = self._reencode_to_jpeg_under_10mb(local_path)
            send_path = cleanup_path
            content_type = "image/jpeg"
            mode = "reencoded"
        path_obj = Path(local_path)
        filename = path_obj.name or "photo.jpg"
        if mode != "original":
            filename = f"{path_obj.stem or 'photo'}.jpg"
        else:
            normalized = {
                "image/jpeg": ".jpg",
                "image/jpg": ".jpg",
                "image/png": ".png",
                "image/webp": ".webp",
            }
            desired_suffix = normalized.get(content_type.lower()) if content_type else None
            if desired_suffix and not filename.lower().endswith(desired_suffix):
                filename = f"{path_obj.stem or 'photo'}{desired_suffix}"
        return send_path, cleanup_path, filename, content_type, mode, mime, size

    async def _publish_as_photo(
        self,
        chat_id: int,
        local_path: str,
        caption: str | None,
        *,
        caption_entities: Sequence[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        send_path, cleanup_path, filename, content_type, mode, mime, size = (
            self._prepare_photo_for_upload(local_path)
        )
        logging.info(
            "Publishing photo for chat %s via %s file (mime=%s size=%s)",
            chat_id,
            mode,
            mime or "?",
            size,
        )
        try:
            with open(send_path, "rb") as fh:
                payload: dict[str, Any] = {
                    "chat_id": chat_id,
                    "caption": caption or None,
                }
                if caption_entities:
                    payload["caption_entities"] = caption_entities
                response = await self.api_request_multipart(
                    "sendPhoto",
                    payload,
                    files={"photo": (filename, fh, content_type)},
                )
        finally:
            if cleanup_path is not None:
                self._remove_file(str(cleanup_path))
            gc.collect()
        return response, mode

    async def _publish_mobile_document(
        self,
        chat_id: int,
        document: BinaryIO,
        filename: str,
        caption: str | None,
        *,
        caption_entities: Sequence[dict[str, Any]] | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any] | None:
        resolved_type = (
            content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        )
        logging.info(
            "Publishing mobile document for chat %s via sendDocument (filename=%s, content_type=%s)",
            chat_id,
            filename,
            resolved_type,
        )
        payload: dict[str, Any] = {"chat_id": chat_id, "caption": caption or None}
        if caption_entities:
            payload["caption_entities"] = caption_entities
        return await self.api_request_multipart(
            "sendDocument",
            payload,
            files={"document": (filename, document, resolved_type)},
        )

    @staticmethod
    def _extract_photo_file_meta(
        photo_sizes: Sequence[dict[str, Any]] | None,
    ) -> dict[str, Any] | None:
        if not photo_sizes:
            return None
        best: dict[str, Any] | None = None
        best_score = -1
        for entry in photo_sizes:
            if not isinstance(entry, dict):
                continue
            width = int(entry.get("width") or 0)
            height = int(entry.get("height") or 0)
            file_size = int(entry.get("file_size") or 0)
            score = width * height
            if score <= 0 and file_size <= 0:
                continue
            if score > best_score or (
                score == best_score and file_size > int(best.get("file_size") or 0)
                if best
                else True
            ):
                best = entry
                best_score = score
        if not best:
            return None
        return {
            "file_id": best.get("file_id"),
            "file_unique_id": best.get("file_unique_id"),
            "width": best.get("width"),
            "height": best.get("height"),
            "file_size": best.get("file_size"),
            "mime_type": "image/jpeg",
        }

    def _get_rubric_overview_target(self, user_id: int, code: str) -> dict[str, Any] | None:
        stored = self.rubric_overview_messages.get(user_id, {}).get(code)
        if not stored:
            return None
        chat_id = stored.get("chat_id")
        message_id = stored.get("message_id")
        if chat_id is None or message_id is None:
            return None
        return {"chat": {"id": chat_id}, "message_id": message_id}

    def _remember_rubric_overview(
        self, user_id: int, code: str, *, chat_id: int, message_id: int
    ) -> None:
        self.rubric_overview_messages.setdefault(user_id, {})[code] = {
            "chat_id": chat_id,
            "message_id": message_id,
        }

    def _cleanup_rubric_overviews(self, user_id: int, valid_codes: Iterable[str]) -> None:
        stored = self.rubric_overview_messages.get(user_id)
        if not stored:
            return
        valid = set(valid_codes)
        for code in list(stored.keys()):
            if code not in valid:
                stored.pop(code, None)
                self._clear_rubric_pending_run(user_id, code)
        if not stored:
            self.rubric_overview_messages.pop(user_id, None)

    def _get_rubric_pending_run(self, user_id: int, code: str) -> str | None:
        return self.rubric_pending_runs.get((user_id, code))

    def _set_rubric_pending_run(self, user_id: int, code: str, mode: str) -> None:
        self.rubric_pending_runs[(user_id, code)] = mode

    def _clear_rubric_pending_run(self, user_id: int, code: str) -> None:
        self.rubric_pending_runs.pop((user_id, code), None)

    async def _render_rubric_cards(self, user_id: int, rubrics: Sequence[Rubric]) -> None:
        for rubric in rubrics:
            target = self._get_rubric_overview_target(user_id, rubric.code)
            await self._send_rubric_overview(user_id, rubric.code, message=target)
        self._cleanup_rubric_overviews(user_id, [rubric.code for rubric in rubrics])

    def _tz_offset_delta(self, tz_offset: str | None) -> timedelta:
        tzinfo = self._parse_tz_offset(tz_offset or TZ_OFFSET)
        offset = tzinfo.utcoffset(datetime.utcnow())
        return offset if offset is not None else timedelta()

    def _next_usage_reset(
        self, *, now: datetime | None = None, tz_offset: str | None = None
    ) -> datetime:
        reference = now or datetime.utcnow()
        offset = self._tz_offset_delta(tz_offset)
        local_reference = reference + offset
        next_day = local_reference.date() + timedelta(days=1)
        local_reset = datetime.combine(next_day, dtime(hour=0, minute=5))
        return local_reset - offset

    def _load_model_limits(self) -> dict[str, int]:
        def parse_limit(raw: str | None, env_name: str) -> int | None:
            if not raw:
                return None
            try:
                value = int(raw)
            except ValueError:
                logging.warning("Invalid %s: %s", env_name, raw)
                return None
            if value <= 0:
                logging.warning("Ignoring non-positive %s: %s", env_name, raw)
                return None
            return value

        limits: dict[str, int] = {}
        default_limit = parse_limit(
            os.getenv("OPENAI_DAILY_TOKEN_LIMIT"), "OPENAI_DAILY_TOKEN_LIMIT"
        )
        per_model_env_vars: Sequence[tuple[str, Sequence[str]]] = (
            (
                "gpt-4o",
                (
                    "OPENAI_DAILY_TOKEN_LIMIT_GPT_4O",
                    "OPENAI_DAILY_TOKEN_LIMIT_4O",
                ),
            ),
            (
                "gpt-4o-mini",
                (
                    "OPENAI_DAILY_TOKEN_LIMIT_GPT_4O_MINI",
                    "OPENAI_DAILY_TOKEN_LIMIT_4O_MINI",
                ),
            ),
        )
        for model, env_names in per_model_env_vars:
            for env_name in env_names:
                limit = parse_limit(os.getenv(env_name), env_name)
                if limit is not None:
                    limits[model] = limit
                    break
        if not limits and default_limit is not None:
            limits = {model: default_limit for model, _ in per_model_env_vars}
        return limits

    async def run_openai_health_check(self) -> None:
        if not self.openai or not self.openai.api_key:
            logging.warning("OpenAI FAIL: API key missing; skipping health check")
            return

        schema = {
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"},
            },
            "required": ["ok"],
            "additionalProperties": False,
        }

        try:
            response = await self.openai.generate_json(
                model="gpt-4o",
                system_prompt="You are a readiness probe.",
                user_prompt="Return a JSON object with ok=true.",
                schema=schema,
                schema_name="health_check_v1",
                temperature=0.0,
            )
        except Exception:
            logging.exception("OpenAI health check request failed")
            logging.warning("OpenAI FAIL: request error")
            return

        await self._record_openai_usage("gpt-4o", response)

        if response and isinstance(response.content, dict) and response.content.get("ok") is True:
            logging.info("OpenAI OK")
        else:
            logging.warning(
                "OpenAI FAIL: unexpected response %s", response.content if response else None
            )

    def _enforce_openai_limit(self, job: Job | None, model: str) -> None:
        if (
            job is None
            or model not in self._model_limits
            or not self.openai
            or not self.openai.api_key
        ):
            return
        tz_offset = None
        if job and isinstance(job.payload, dict):
            tz_offset = job.payload.get("tz_offset")
        tz_offset = tz_offset or TZ_OFFSET
        limit = self._model_limits[model]
        total_today = self.data.get_daily_token_usage_total(models={model}, tz_offset=tz_offset)
        if total_today >= limit:
            resume_at = self._next_usage_reset(tz_offset=tz_offset)
            tzinfo = self._parse_tz_offset(tz_offset)
            resume_local = (
                resume_at.replace(tzinfo=UTC).astimezone(tzinfo)
                if resume_at.tzinfo is None
                else resume_at.astimezone(tzinfo)
            )
            reason = (
                f"Превышен дневной лимит токенов {model}: "
                f"{total_today}/{limit}. Задача перенесена до {resume_at.isoformat()} UTC"
                f" (локально {resume_local.isoformat()})"
            )
            logging.warning(reason)
            raise JobDelayed(resume_at, reason)

    async def _record_openai_usage(
        self,
        model: str,
        response: OpenAIResponse | None,
        *,
        job: Job | None = None,
        record_supabase: bool = True,
        supabase_meta: dict[str, Any] | None = None,
        supabase_endpoint: str | None = None,
        supabase_bot: str | None = None,
    ) -> None:
        if response is None:
            return
        self.data.log_token_usage(
            model,
            response.prompt_tokens,
            response.completion_tokens,
            response.total_tokens,
            job_id=job.id if job else None,
            request_id=response.request_id,
        )
        tz_offset = None
        if job and isinstance(job.payload, dict):
            tz_offset = job.payload.get("tz_offset")
        tz_offset = tz_offset or TZ_OFFSET
        if model in self._model_limits:
            total_today = self.data.get_daily_token_usage_total(models={model}, tz_offset=tz_offset)
            limit = self._model_limits[model]
            logging.info(
                "Суммарное потребление токенов %s за сегодня: %s из %s",
                model,
                total_today,
                limit,
            )
        meta: dict[str, Any] = {}
        if response.meta:
            meta.update(response.meta)
        if job is not None:
            job_meta: dict[str, Any] = {
                "id": job.id,
                "name": job.name,
            }
            if isinstance(job.payload, dict):
                job_meta["payload_keys"] = sorted(job.payload.keys())
            meta["job"] = job_meta
        if supabase_meta:
            meta.update(supabase_meta)
        if not record_supabase:
            return
        usage = response.usage if isinstance(response.usage, dict) else {}
        success, payload, error = await self.supabase.insert_token_usage(
            bot=supabase_bot or "kotopogoda",
            model=model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            request_id=response.request_id,
            endpoint=supabase_endpoint or usage.get("endpoint") or "/v1/responses",
            meta=meta or None,
        )
        log_context = {"log_token_usage": payload}
        if success:
            logging.info("Supabase token usage insert succeeded", extra=log_context)
        else:
            if error == "disabled":
                logging.debug("Supabase client disabled; token usage skipped", extra=log_context)
            elif error:
                logging.error("Supabase token usage insert failed: %s", error, extra=log_context)
            else:
                logging.error("Supabase token usage insert failed", extra=log_context)

    async def start(self) -> None:
        self.session = ClientSession()
        self.running = True
        await self.jobs.start()

    async def close(self) -> None:
        self.running = False
        await self.jobs.stop()
        if self.session:
            await self.session.close()
        await self.supabase.aclose()

        self.db.close()

    async def backfill_waves(self, *, dry_run: bool = False) -> dict[str, int]:
        """Backfill wave_score_0_10, wave_conf, and sky_code from vision_results.

        Runs in batches of 200 with async yields. Guarded by lock to prevent concurrent runs.
        Returns stats dict with counts of updated, skipped, and error assets.
        """
        async with self._backfill_waves_lock:
            if self._backfill_waves_running:
                logging.info("Backfill waves already running, skipping duplicate trigger")
                return {"updated": 0, "skipped": 0, "errors": 0, "already_running": 1}

            self._backfill_waves_running = True
            try:
                stats = {"updated": 0, "skipped": 0, "errors": 0}
                batch_size = 200
                batch_count = 0
                error_details_count = 0
                max_error_details = 10
                skip_reasons: dict[str, int] = {}
                skipped_samples: list[dict[str, Any]] = []
                max_samples = 10

                assets_to_process = []
                for asset in self.data.iter_assets():
                    has_wave = asset.vision_wave_score is not None
                    has_conf = asset.vision_wave_conf is not None
                    has_sky = asset.vision_sky_bucket is not None

                    if has_wave and has_conf and has_sky:
                        stats["skipped"] += 1
                        skip_reason = "already_complete"
                        skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
                        if len(skipped_samples) < max_samples:
                            skipped_samples.append({"asset_id": asset.id, "reason": skip_reason})
                        continue

                    assets_to_process.append(asset)

                logging.info(
                    "Backfill waves: found %d assets to process, %d already complete",
                    len(assets_to_process),
                    stats["skipped"],
                )

                for i in range(0, len(assets_to_process), batch_size):
                    batch = assets_to_process[i : i + batch_size]
                    batch_count += 1

                    for asset in batch:
                        try:
                            vision = asset.vision_results

                            # Log asset discovery with current state
                            logging.debug(
                                "Backfill discovered asset_id=%s vision_results_exists=%s "
                                "current_wave_score=%s current_sky_code=%s",
                                asset.id,
                                vision is not None,
                                asset.vision_wave_score,
                                asset.vision_sky_bucket,
                            )

                            if vision is None:
                                skip_reason = "no_vision_results"
                                logging.debug(
                                    "Backfill skip asset_id=%s skip_reason=%s",
                                    asset.id,
                                    skip_reason,
                                )
                                stats["skipped"] += 1
                                skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
                                if len(skipped_samples) < max_samples:
                                    skipped_samples.append(
                                        {"asset_id": asset.id, "reason": skip_reason}
                                    )
                                continue

                            if not isinstance(vision, dict):
                                skip_reason = "invalid_vision_type"
                                logging.error(
                                    "Backfill error asset_id=%s error_type=InvalidVisionType "
                                    "error_msg='vision_results is not a dict: %s'",
                                    asset.id,
                                    type(vision).__name__,
                                )
                                stats["errors"] += 1
                                skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
                                if len(skipped_samples) < max_samples:
                                    skipped_samples.append(
                                        {"asset_id": asset.id, "reason": skip_reason}
                                    )
                                continue

                            wave_score, wave_conf = self.data._parse_wave_score_from_vision(vision)
                            sky_bucket = self.data._parse_sky_bucket_from_vision(vision)

                            if wave_score is None and wave_conf is None and sky_bucket is None:
                                skip_reason = "no_extractable_data"
                                logging.debug(
                                    "Backfill skip asset_id=%s skip_reason=%s",
                                    asset.id,
                                    skip_reason,
                                )
                                stats["skipped"] += 1
                                skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
                                if len(skipped_samples) < max_samples:
                                    skipped_samples.append(
                                        {"asset_id": asset.id, "reason": skip_reason}
                                    )
                                continue

                            updates: list[str] = []
                            params: list[Any] = []

                            if asset.vision_wave_score is None and wave_score is not None:
                                updates.append("vision_wave_score=?")
                                params.append(float(wave_score))

                            if asset.vision_wave_conf is None and wave_conf is not None:
                                updates.append("vision_wave_conf=?")
                                params.append(float(wave_conf))

                            if asset.vision_sky_bucket is None and sky_bucket is not None:
                                updates.append("vision_sky_bucket=?")
                                params.append(str(sky_bucket))

                            if not updates:
                                skip_reason = "fields_already_populated"
                                logging.debug(
                                    "Backfill skip asset_id=%s skip_reason=%s",
                                    asset.id,
                                    skip_reason,
                                )
                                stats["skipped"] += 1
                                skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
                                if len(skipped_samples) < max_samples:
                                    skipped_samples.append(
                                        {"asset_id": asset.id, "reason": skip_reason}
                                    )
                                continue

                            if not dry_run:
                                sql = f"UPDATE assets SET {', '.join(updates)} WHERE id=?"
                                params.append(str(asset.id))
                                self.db.execute(sql, params)

                            # Log successful update with details
                            final_wave = (
                                wave_score
                                if asset.vision_wave_score is None
                                else asset.vision_wave_score
                            )
                            final_conf = (
                                wave_conf
                                if asset.vision_wave_conf is None
                                else asset.vision_wave_conf
                            )
                            final_sky = (
                                sky_bucket
                                if asset.vision_sky_bucket is None
                                else asset.vision_sky_bucket
                            )
                            logging.info(
                                "Backfill updated asset_id=%s wave_score_0_10=%s sky_code=%s confidence=%s",
                                asset.id,
                                final_wave,
                                final_sky,
                                final_conf,
                            )

                            stats["updated"] += 1

                        except Exception as e:
                            error_type = type(e).__name__
                            error_msg = str(e)
                            logging.error(
                                "Backfill error asset_id=%s error_type=%s error_msg=%s",
                                asset.id,
                                error_type,
                                error_msg,
                            )
                            if error_details_count < max_error_details:
                                logging.debug(
                                    "Backfill error detail asset_id=%s",
                                    asset.id,
                                    exc_info=True,
                                )
                                error_details_count += 1
                            stats["errors"] += 1
                            skip_reason = "exception"
                            skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1

                    if not dry_run:
                        self.db.commit()

                    logging.info(
                        "Backfill waves: processed batch %d (%d assets), stats: %s",
                        batch_count,
                        len(batch),
                        stats,
                    )
                    await asyncio.sleep(0)

                # Log sample of skipped assets for quick diagnostics
                if skipped_samples:
                    logging.info(
                        "Backfill sample (first %d skipped): %s",
                        len(skipped_samples),
                        skipped_samples,
                    )

                # Build reason breakdown string
                reason_parts = []
                for reason, count in sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True):
                    reason_parts.append(f"{reason}={count}")
                reason_breakdown = ", ".join(reason_parts) if reason_parts else "none"

                logging.info(
                    "Backfill waves completed: processed=%d updated=%d skipped=%d errors=%d "
                    "(skip reasons: %s)",
                    len(assets_to_process),
                    stats["updated"],
                    stats["skipped"],
                    stats["errors"],
                    reason_breakdown,
                )

                return stats

            finally:
                self._backfill_waves_running = False

    async def handle_edited_message(self, message: Any) -> None:
        chat_id = message.get("chat", {}).get("id")
        if chat_id is None:
            return
        is_weather_channel = chat_id == self.weather_assets_channel_id
        is_recognition_channel = (
            chat_id is not None
            and self.recognition_channel_id is not None
            and chat_id == self.recognition_channel_id
            and self.recognition_channel_id != self.weather_assets_channel_id
        )
        if not is_weather_channel and not is_recognition_channel:
            return
        info = self._collect_asset_metadata(message)
        message_id = info.get("message_id")
        tg_chat_id = info.get("tg_chat_id")
        if not message_id or not tg_chat_id:
            return
        if self.data.is_recognized_message(tg_chat_id, message_id):
            logging.info(
                "Skipping recognized edit %s in channel %s",
                message_id,
                tg_chat_id,
            )
            return
        origin = "recognition" if is_recognition_channel else "weather"
        existing = self.data.get_asset_by_message(tg_chat_id, message_id)
        if existing:
            self.data.update_asset(
                existing.id,
                template=info.get("caption"),
                caption=info.get("caption"),
                hashtags=info.get("hashtags"),
                kind=info.get("kind"),
                file_meta=info.get("file_meta"),
                author_user_id=info.get("author_user_id"),
                author_username=info.get("author_username"),
                sender_chat_id=info.get("sender_chat_id"),
                via_bot_id=info.get("via_bot_id"),
                forward_from_user=info.get("forward_from_user"),
                forward_from_chat=info.get("forward_from_chat"),
                metadata=info.get("metadata"),
                origin=origin,
            )
            asset_id = existing.id
        else:
            asset_id = self.add_asset(
                message_id,
                info.get("hashtags", ""),
                info.get("caption"),
                channel_id=tg_chat_id,
                metadata=info.get("metadata"),
                tg_chat_id=tg_chat_id,
                kind=info.get("kind"),
                file_meta=info.get("file_meta"),
                author_user_id=info.get("author_user_id"),
                author_username=info.get("author_username"),
                sender_chat_id=info.get("sender_chat_id"),
                via_bot_id=info.get("via_bot_id"),
                forward_from_user=info.get("forward_from_user"),
                forward_from_chat=info.get("forward_from_chat"),
                origin=origin,
            )
        if asset_id and is_recognition_channel:
            self._schedule_ingest_job(asset_id, reason="edit")
        return

    async def api_request(
        self,
        method: str,
        data: dict | None = None,
        *,
        files: dict[str, tuple[str, bytes]] | None = None,
    ) -> Any:
        if self.dry_run:
            logging.debug("Simulated API call %s with %s", method, data)
            return {"ok": True, "result": {}}
        url = f"{self.api_url}/{method}"
        if files:
            form = FormData()
            for key, value in (data or {}).items():
                if isinstance(value, (dict, list)):
                    form.add_field(key, json.dumps(value))
                else:
                    form.add_field(key, str(value))
            for name, (filename, blob) in files.items():
                form.add_field(name, blob, filename=filename)
            async with self.session.post(url, data=form) as resp:
                text = await resp.text()
        else:
            async with self.session.post(url, json=data) as resp:
                text = await resp.text()
        if resp.status != 200:
            logging.error("API HTTP %s for %s: %s", resp.status, method, text)
        try:
            result = json.loads(text)
        except Exception:
            logging.exception("Invalid response for %s: %s", method, text)
            return {}
        if not result.get("ok"):
            logging.error("API call %s failed: %s", method, result)
        else:
            logging.info("API call %s succeeded", method)
        return result

    async def api_request_multipart(
        self,
        method: str,
        data: dict | None = None,
        *,
        files: dict[str, tuple[str, BinaryIO, str]] | None = None,
    ) -> dict:
        if self.dry_run:
            logging.debug("Simulated multipart API call %s with %s", method, data)
            return {"ok": True, "result": {}}
        url = f"{self.api_url}/{method}"
        form = FormData()
        for key, value in (data or {}).items():
            if isinstance(value, (dict, list)):
                form.add_field(key, json.dumps(value))
            else:
                form.add_field(key, str(value))
        for name, (filename, fh, content_type) in (files or {}).items():
            if hasattr(fh, "seek"):
                try:
                    fh.seek(0)
                except Exception:
                    pass
            form.add_field(
                name,
                fh,
                filename=filename,
                content_type=content_type,
            )
        async with self.session.post(url, data=form) as resp:
            text = await resp.text()
        if resp.status != 200:
            logging.error("API HTTP %s for %s: %s", resp.status, method, text)
        try:
            result = json.loads(text)
        except Exception:
            logging.exception("Invalid response for %s: %s", method, text)
            return {}
        if not result.get("ok"):
            logging.error("API call %s failed: %s", method, result)
        else:
            logging.info("API call %s succeeded", method)
        return result

    async def fetch_open_meteo(self, lat: float, lon: float) -> dict | None:
        url = (
            "https://api.open-meteo.com/v1/forecast?latitude="
            f"{lat}&longitude={lon}&current_weather=true"
            "&hourly=temperature_2m,weather_code,wind_speed_10m,is_day"
            "&forecast_days=2&timezone=auto&wind_speed_unit=kmh"
        )
        try:
            async with self.session.get(url) as resp:
                text = await resp.text()

        except Exception:
            logging.exception("Failed to fetch weather")
            return None

        logging.info("Weather API raw response: %s", text)
        if resp.status != 200:
            logging.error("Open-Meteo HTTP %s", resp.status)
            return None
        try:
            data = json.loads(text)
        except Exception:
            logging.exception("Invalid weather JSON")
            return None

        if "current_weather" in data and "current" not in data:
            cw = data["current_weather"]
            data["current"] = {
                "temperature_2m": cw.get("temperature") or cw.get("temperature_2m"),
                "weather_code": cw.get("weather_code") or cw.get("weathercode"),
                "wind_speed_10m": cw.get("wind_speed_10m") or cw.get("windspeed"),
                "is_day": cw.get("is_day"),
            }

        logging.info("Weather response: %s", data.get("current"))
        return data

    async def fetch_open_meteo_sea(self, lat: float, lon: float) -> dict | None:
        url = (
            "https://marine-api.open-meteo.com/v1/marine?latitude="
            f"{lat}&longitude={lon}"
            "&current=wave_height,wind_wave_height,swell_wave_height,"
            "sea_surface_temperature,sea_level_height_msl"
            "&hourly=wave_height,wind_wave_height,swell_wave_height,"
            "sea_surface_temperature"
            "&daily=wave_height_max,wind_wave_height_max,swell_wave_height_max"
            "&forecast_days=2&timezone=auto&wind_speed_unit=kmh"
        )
        logging.info("SEA_RUBRIC api_request url=%s", url)
        try:
            async with self.session.get(url) as resp:
                text = await resp.text()
        except Exception:
            logging.exception("SEA_RUBRIC api_request failed url=%s", url)
            return None

        logging.info("SEA_RUBRIC api_response %s", text)
        if resp.status != 200:
            logging.error("SEA_RUBRIC api_http status=%s", resp.status)
            return None
        try:
            data = json.loads(text)
        except Exception:
            logging.exception("SEA_RUBRIC api_response invalid_json")
            return None
        return data

    async def fetch_open_meteo_sea_conditions(self, lat: float, lon: float) -> dict | None:
        url = (
            "https://api.open-meteo.com/v1/forecast?latitude="
            f"{lat}&longitude={lon}"
            "&current=wind_speed_10m,wind_gusts_10m,cloud_cover"
            "&hourly=wind_speed_10m,wind_gusts_10m,cloudcover"
            "&past_hours=3&forecast_hours=0"
            "&timezone=auto&wind_speed_unit=kmh"
        )
        logging.info("SEA_RUBRIC weather_api_request url=%s", url)
        try:
            async with self.session.get(url) as resp:
                text = await resp.text()
        except Exception:
            logging.exception("SEA_RUBRIC weather_api_request failed url=%s", url)
            return None
        logging.info("SEA_RUBRIC weather_api_response %s", text)
        if resp.status != 200:
            logging.error("SEA_RUBRIC weather_api_http status=%s", resp.status)
            return None
        try:
            data = json.loads(text)
        except Exception:
            logging.exception("SEA_RUBRIC weather_api_response invalid_json")
            return None
        return data

    async def collect_weather(self, force: bool = False) -> None:

        cur = self.db.execute("SELECT id, lat, lon, name FROM cities")
        updated: set[int] = set()
        for c in cur.fetchall():
            try:
                row = self.db.execute(
                    "SELECT timestamp FROM weather_cache_hour WHERE city_id=? ORDER BY timestamp DESC LIMIT 1",
                    (c["id"],),
                ).fetchone()
                now = datetime.utcnow()
                last_success = datetime.fromisoformat(row["timestamp"]) if row else datetime.min

                attempts, last_attempt = self.failed_fetches.get(c["id"], (0, datetime.min))

                if not force:
                    if not self.dry_run and last_success > now - timedelta(minutes=30):
                        continue
                    if attempts >= 3 and (now - last_attempt) < timedelta(minutes=30):
                        continue
                    if attempts > 0 and (now - last_attempt) < timedelta(minutes=1):
                        continue
                    if attempts >= 3 and (now - last_attempt) >= timedelta(minutes=30):
                        attempts = 0

                data = await self.fetch_open_meteo(c["lat"], c["lon"])
                if not data or "current" not in data:
                    self.failed_fetches[c["id"]] = (attempts + 1, now)
                    continue

                self.failed_fetches.pop(c["id"], None)

                w = data["current"]
                ts = datetime.utcnow().replace(microsecond=0).isoformat()
                day = ts.split("T")[0]
                self.db.execute(
                    "INSERT OR REPLACE INTO weather_cache_hour (city_id, timestamp, temperature, weather_code, wind_speed, is_day) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        c["id"],
                        ts,
                        w.get("temperature_2m"),
                        w.get("weather_code"),
                        w.get("wind_speed_10m"),
                        w.get("is_day"),
                    ),
                )
                self.db.execute(
                    "INSERT OR REPLACE INTO weather_cache_day (city_id, day, temperature, weather_code, wind_speed) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        c["id"],
                        day,
                        w.get("temperature_2m"),
                        w.get("weather_code"),
                        w.get("wind_speed_10m"),
                    ),
                )

                # parse hourly forecast for tomorrow using averages

                h = data.get("hourly", {})
                times = h.get("time") or []
                temps = h.get("temperature_2m") or []
                codes = h.get("weather_code") or []
                winds = h.get("wind_speed_10m") or []
                tomorrow = (datetime.utcnow() + timedelta(days=1)).date()

                buckets = {
                    "morning": ([], [], []),  # temp, code, wind
                    "day": ([], [], []),
                    "evening": ([], [], []),
                    "night": ([], [], []),
                }

                for tstr, temp, code, wind in zip(times, temps, codes, winds):
                    dt_obj = datetime.fromisoformat(tstr)
                    if dt_obj.date() != tomorrow:
                        continue

                    h = dt_obj.hour
                    if 6 <= h < 12:
                        b = buckets["morning"]
                    elif 12 <= h < 18:
                        b = buckets["day"]
                    elif 18 <= h < 24:
                        b = buckets["evening"]
                    else:
                        b = buckets["night"]
                    b[0].append(temp)
                    b[1].append(code)
                    b[2].append(wind)

                def avg(lst: list[float]) -> float | None:
                    return sum(lst) / len(lst) if lst else None

                def mid_code(lst: list[int]) -> int | None:
                    return lst[len(lst) // 2] if lst else None

                mt = avg(buckets["morning"][0])
                mc = mid_code(buckets["morning"][1])
                mw = avg(buckets["morning"][2])
                dt_val = avg(buckets["day"][0])
                dc = mid_code(buckets["day"][1])
                dw = avg(buckets["day"][2])
                et = avg(buckets["evening"][0])
                ec = mid_code(buckets["evening"][1])
                ew = avg(buckets["evening"][2])
                nt = avg(buckets["night"][0])
                nc = mid_code(buckets["night"][1])
                nw = avg(buckets["night"][2])

                self.db.execute(
                    "INSERT OR REPLACE INTO weather_cache_period (city_id, updated, morning_temp, morning_code, morning_wind, day_temp, day_code, day_wind, evening_temp, evening_code, evening_wind, night_temp, night_code, night_wind) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        c["id"],
                        now.isoformat(),
                        mt,
                        mc,
                        mw,
                        dt_val,
                        dc,
                        dw,
                        et,
                        ec,
                        ew,
                        nt,
                        nc,
                        nw,
                    ),
                )
                self.db.commit()
                logging.info(
                    "Cached weather for city %s: %s°C code %s",
                    c["id"],
                    w.get("temperature_2m"),
                    w.get("weather_code"),
                )
                updated.add(c["id"])
            except Exception:
                logging.exception("Error processing weather for city %s", c["id"])
        if updated:
            await self.update_weather_posts(updated)

    async def collect_sea(self, force: bool = False) -> None:
        cur = self.db.execute("SELECT id, lat, lon FROM seas")
        updated: set[int] = set()
        for s in cur.fetchall():
            row = self.db.execute(
                "SELECT updated FROM sea_cache WHERE sea_id=?",
                (s["id"],),
            ).fetchone()
            now = datetime.utcnow()
            last = datetime.fromisoformat(row["updated"]) if row else datetime.min
            if not force and last > now - timedelta(minutes=30):
                continue

            data = await self.fetch_open_meteo_sea(s["lat"], s["lon"])
            if not data or "hourly" not in data:
                continue
            temps = data["hourly"].get("water_temperature") or data["hourly"].get(
                "sea_surface_temperature"
            )
            waves = data["hourly"].get("wave_height")
            times = data["hourly"].get("time")
            if not temps or not times:
                continue
            if waves is None:
                waves = [0.0 for _ in temps]
            current = temps[0]
            current_wave = data.get("current", {}).get("wave_height")
            conditions_payload = await self.fetch_open_meteo_sea_conditions(s["lat"], s["lon"])
            wind_speed_ms: float | None = None
            wind_speed_kmh: float | None = None
            wind_gust_ms: float | None = None
            wind_gust_kmh: float | None = None
            wind_units: str | None = None
            wind_gust_units: str | None = None
            wind_time_ref: str | None = None
            cloud_cover_pct = None
            if conditions_payload:
                current_conditions = conditions_payload.get("current") or {}
                current_units = conditions_payload.get("current_units") or {}
                hourly_units = conditions_payload.get("hourly_units") or {}

                raw_wind_units = current_units.get("wind_speed_10m") or hourly_units.get(
                    "wind_speed_10m"
                )
                if raw_wind_units is not None:
                    wind_units = str(raw_wind_units).strip() or None
                raw_gust_units = current_units.get("wind_gusts_10m") or hourly_units.get(
                    "wind_gusts_10m"
                )
                if raw_gust_units is not None:
                    wind_gust_units = str(raw_gust_units).strip() or None

                wind_value = current_conditions.get("wind_speed_10m")
                gust_value = current_conditions.get("wind_gusts_10m")
                wind_speed_kmh, wind_speed_ms = resolve_wind_speed(wind_value, wind_units)
                wind_gust_kmh, wind_gust_ms = resolve_wind_speed(gust_value, wind_gust_units)
                if wind_speed_ms is None and wind_speed_kmh is not None:
                    wind_speed_ms = wind_speed_kmh / 3.6
                if wind_speed_kmh is None and wind_speed_ms is not None:
                    wind_speed_kmh = wind_speed_ms * 3.6
                if wind_gust_ms is None and wind_gust_kmh is not None:
                    wind_gust_ms = wind_gust_kmh / 3.6
                if wind_gust_kmh is None and wind_gust_ms is not None:
                    wind_gust_kmh = wind_gust_ms * 3.6

                cloud_cover_pct = self._safe_float(current_conditions.get("cloud_cover"))
                wind_time_ref = current_conditions.get("time") or (
                    conditions_payload.get("current_weather") or {}
                ).get("time")

                logging.info(
                    "SEA_RUBRIC weather: wind_source=wind_speed_10m units=%s gusts_units=%s time_ref=%s",
                    wind_units or "unknown",
                    wind_gust_units or "unknown",
                    wind_time_ref or "unknown",
                )
            else:
                logging.info(
                    "SEA_RUBRIC weather: wind_source=wind_speed_10m units=%s gusts_units=%s time_ref=%s",
                    "missing",
                    "missing",
                    "missing",
                )
            tomorrow = date.today() + timedelta(days=1)
            morn = day_temp = eve = night = None
            mwave = dwave = ewave = nwave = None
            for t, temp, wave in zip(times, temps, waves):
                dt = datetime.fromisoformat(t)
                if dt.date() != tomorrow:
                    continue
                if dt.hour == 6 and morn is None:
                    morn = temp
                    mwave = wave
                elif dt.hour == 12 and day_temp is None:
                    day_temp = temp
                    dwave = wave
                elif dt.hour == 18 and eve is None:
                    eve = temp
                    ewave = wave
                elif dt.hour == 0 and night is None:
                    night = temp
                    nwave = wave
                if None not in (morn, day_temp, eve, night, mwave, dwave, ewave, nwave):
                    break

            wave_height_for_cache = self._safe_float(current_wave)
            if wave_height_for_cache is None and waves:
                wave_height_for_cache = self._safe_float(waves[0])

            self.db.execute(
                "INSERT OR REPLACE INTO sea_cache (sea_id, updated, current, morning, day, evening, night, wave, morning_wave, day_wave, evening_wave, night_wave) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    s["id"],
                    now.isoformat(),
                    current,
                    morn,
                    day_temp,
                    eve,
                    night,
                    wave_height_for_cache,
                    mwave,
                    dwave,
                    ewave,
                    nwave,
                ),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO sea_conditions (sea_id, updated, wave_height_m, wind_speed_10m_ms, wind_speed_10m_kmh, wind_gusts_10m_ms, wind_gusts_10m_kmh, wind_units, wind_gusts_units, wind_time_ref, cloud_cover_pct) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    s["id"],
                    now.isoformat(),
                    wave_height_for_cache,
                    wind_speed_ms,
                    wind_speed_kmh,
                    wind_gust_ms,
                    wind_gust_kmh,
                    wind_units,
                    wind_gust_units,
                    wind_time_ref,
                    cloud_cover_pct,
                ),
            )
            self.db.commit()
            wave_log = (
                f"{wave_height_for_cache:.3f}"
                if isinstance(wave_height_for_cache, (int, float))
                else "None"
            )
            wind_kmh_log = (
                f"{wind_speed_kmh:.2f}" if isinstance(wind_speed_kmh, (int, float)) else "None"
            )
            wind_ms_log = (
                f"{wind_speed_ms:.2f}" if isinstance(wind_speed_ms, (int, float)) else "None"
            )
            gust_kmh_log = (
                f"{wind_gust_kmh:.2f}" if isinstance(wind_gust_kmh, (int, float)) else "None"
            )
            gust_ms_log = (
                f"{wind_gust_ms:.2f}" if isinstance(wind_gust_ms, (int, float)) else "None"
            )
            cloud_log = (
                f"{cloud_cover_pct:.1f}" if isinstance(cloud_cover_pct, (int, float)) else "None"
            )

            # Enhanced structured logging for SEA_RUBRIC weather
            logging.info(
                "SEA_RUBRIC weather sea_id=%s lat=%s lon=%s time_ref=%s wave_height_m=%s wind_speed_kmh=%s wind_speed_ms=%s wind_units=%s wind_gusts_kmh=%s wind_gusts_ms=%s wind_gusts_units=%s cloud_cover_pct=%s",
                s["id"],
                s["lat"],
                s["lon"],
                wind_time_ref or "unknown",
                wave_log,
                wind_kmh_log,
                wind_ms_log,
                wind_units or "unknown",
                gust_kmh_log,
                gust_ms_log,
                wind_gust_units or "unknown",
                cloud_log,
            )
            updated.add(s["id"])
        if updated:
            await self.update_weather_posts()
            await self.check_amber()

    async def check_amber(self) -> None:
        state = self.db.execute(
            "SELECT sea_id, storm_start, active FROM amber_state LIMIT 1"
        ).fetchone()
        if not state:
            return
        sea_id = state["sea_id"]
        row = self._get_sea_cache(sea_id)
        if not row or row["wave"] is None:
            return
        wave_raw = row["wave"]
        try:
            wave = float(wave_raw)
        except (TypeError, ValueError):
            logging.warning("Unable to parse wave height %r for sea %s", wave_raw, sea_id)
            return
        now = datetime.utcnow()
        if wave >= 1.5:
            if not state["active"]:
                self.db.execute(
                    "UPDATE amber_state SET storm_start=?, active=1 WHERE sea_id=?",
                    (now.isoformat(), sea_id),
                )
                self.db.commit()
            return
        if state["active"] and state["storm_start"]:
            start = datetime.fromisoformat(state["storm_start"])
            if now - start >= timedelta(hours=1):
                tz_offset = self.get_default_tz_offset()
                start_str = self.format_time(start.isoformat(), tz_offset)
                end_str = self.format_time(now.isoformat(), tz_offset)
                text = (
                    "Время собирать янтарь. Закончился шторм, длившийся с "
                    f"{start_str} по {end_str}, теперь в окрестностях Светлогорска, Отрадного, Донского и Балтийска можно идти собирать янтарь на пляже (вывоз за пределы региона по закону запрещён).\n\n"
                    "Сообщение сформировано автоматически сервисом #котопогода от Полюбить Калининград"
                )
                for r in self.db.execute("SELECT channel_id FROM amber_channels"):
                    try:
                        await self.api_request(
                            "sendMessage", {"chat_id": r["channel_id"], "text": text}
                        )
                        logging.info("Amber message sent to %s", r["channel_id"])
                    except Exception:
                        logging.exception("Amber message failed for %s", r["channel_id"])
            self.db.execute(
                "UPDATE amber_state SET storm_start=NULL, active=0 WHERE sea_id=?", (sea_id,)
            )
            self.db.commit()

    async def handle_update(self, update: Any) -> None:
        message = update.get("message") or update.get("channel_post")
        if message:
            await self.handle_message(message)

        elif "edited_channel_post" in update:
            await self.handle_edited_message(update["edited_channel_post"])

        elif "callback_query" in update:
            await self.handle_callback(update["callback_query"])
        elif "my_chat_member" in update:
            await self.handle_my_chat_member(update["my_chat_member"])

    async def handle_my_chat_member(self, chat_update: Any) -> None:
        chat = chat_update["chat"]
        status = chat_update["new_chat_member"]["status"]
        if status in {"administrator", "creator"}:
            self.db.execute(
                "INSERT OR REPLACE INTO channels (chat_id, title) VALUES (?, ?)",
                (chat["id"], chat.get("title", chat.get("username", ""))),
            )
            self.db.commit()
            logging.info("Added channel %s", chat["id"])
        else:
            self.db.execute("DELETE FROM channels WHERE chat_id=?", (chat["id"],))
            self.db.commit()
            logging.info("Removed channel %s", chat["id"])

    def get_user(self, user_id: Any) -> Any:
        cur = self.db.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
        return cur.fetchone()

    def is_pending(self, user_id: int) -> bool:
        cur = self.db.execute("SELECT 1 FROM pending_users WHERE user_id=?", (user_id,))
        return cur.fetchone() is not None

    def pending_count(self) -> int:
        cur = self.db.execute("SELECT COUNT(*) FROM pending_users")
        return cur.fetchone()[0]

    def approve_user(self, uid: int) -> bool:
        if not self.is_pending(uid):
            return False
        cur = self.db.execute("SELECT username FROM pending_users WHERE user_id=?", (uid,))
        row = cur.fetchone()
        username = row["username"] if row else None
        self.db.execute("DELETE FROM pending_users WHERE user_id=?", (uid,))
        self.db.execute(
            "INSERT OR IGNORE INTO users (user_id, username, tz_offset) VALUES (?, ?, ?)",
            (uid, username, TZ_OFFSET),
        )
        if username:
            self.db.execute("UPDATE users SET username=? WHERE user_id=?", (username, uid))
        self.db.execute("DELETE FROM rejected_users WHERE user_id=?", (uid,))
        self.db.commit()
        logging.info("Approved user %s", uid)
        return True

    def reject_user(self, uid: int) -> bool:
        if not self.is_pending(uid):
            return False
        cur = self.db.execute("SELECT username FROM pending_users WHERE user_id=?", (uid,))
        row = cur.fetchone()
        username = row["username"] if row else None
        self.db.execute("DELETE FROM pending_users WHERE user_id=?", (uid,))
        self.db.execute(
            "INSERT OR REPLACE INTO rejected_users (user_id, username, rejected_at) VALUES (?, ?, ?)",
            (uid, username, datetime.utcnow().isoformat()),
        )
        self.db.commit()
        logging.info("Rejected user %s", uid)
        return True

    def is_rejected(self, user_id: int) -> bool:
        cur = self.db.execute("SELECT 1 FROM rejected_users WHERE user_id=?", (user_id,))
        return cur.fetchone() is not None

    def list_scheduled(self) -> Any:
        cur = self.db.execute(
            "SELECT s.id, s.target_chat_id, c.title as target_title, "
            "s.publish_time, s.from_chat_id, s.message_id "
            "FROM schedule s LEFT JOIN channels c ON s.target_chat_id=c.chat_id "
            "WHERE s.sent=0 ORDER BY s.publish_time"
        )
        return cur.fetchall()

    def add_schedule(self, from_chat: int, msg_id: int, targets: set[int], pub_time: str) -> None:
        for chat_id in targets:
            self.db.execute(
                "INSERT INTO schedule (from_chat_id, message_id, target_chat_id, publish_time) VALUES (?, ?, ?, ?)",
                (from_chat, msg_id, chat_id, pub_time),
            )
        self.db.commit()
        logging.info("Scheduled %s -> %s at %s", msg_id, list(targets), pub_time)

    def remove_schedule(self, sid: int) -> None:
        self.db.execute("DELETE FROM schedule WHERE id=?", (sid,))
        self.db.commit()
        logging.info("Cancelled schedule %s", sid)

    def update_schedule_time(self, sid: int, pub_time: str) -> None:
        self.db.execute("UPDATE schedule SET publish_time=? WHERE id=?", (pub_time, sid))
        self.db.commit()
        logging.info("Rescheduled %s to %s", sid, pub_time)

    @staticmethod
    def format_user(user_id: int, username: str | None) -> str:
        label = f"@{username}" if username else str(user_id)
        return f"[{label}](tg://user?id={user_id})"

    @staticmethod
    def parse_offset(offset: str) -> timedelta:
        sign = -1 if offset.startswith("-") else 1
        h, m = offset.lstrip("+-").split(":")
        return timedelta(minutes=sign * (int(h) * 60 + int(m)))

    def next_weather_run(
        self,
        post_time: str,
        offset: str,
        reference: datetime | None = None,
        allow_past: bool = False,
    ) -> datetime:
        """Compute the next UTC datetime for the given local post time."""

        if reference is None:
            reference = datetime.utcnow()
        tz_delta = self.parse_offset(offset)
        local_ref = reference + tz_delta
        hour, minute = map(int, post_time.split(":"))
        candidate = local_ref.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= local_ref:
            if allow_past:
                candidate = local_ref
            else:
                candidate += timedelta(days=1)
        return candidate - tz_delta

    def format_time(self, ts: str, offset: str) -> str:
        dt = datetime.fromisoformat(ts)
        dt += self.parse_offset(offset)
        return dt.strftime("%H:%M %d.%m.%Y")

    def get_default_tz_offset(self) -> str:
        """Return the timezone offset configured in settings or fallback to global default."""

        def _extract(config: Mapping[str, Any] | None) -> str | None:
            if not isinstance(config, Mapping):
                return None
            raw_value = config.get("tz")
            if isinstance(raw_value, str):
                candidate = raw_value.strip()
                if candidate:
                    try:
                        self.parse_offset(candidate)
                    except Exception:
                        return None
                    return candidate
            return None

        preferred_codes = (
            "weather",
            "weather_new",
            "weather_daily",
            "weather_hourly",
        )
        for code in preferred_codes:
            try:
                config = self.data.get_rubric_config(code)
            except Exception:
                continue
            tz_value = _extract(config)
            if tz_value:
                return tz_value

        try:
            rubrics = self.data.list_rubrics()
        except Exception:
            rubrics = []
        for rubric in rubrics:
            config = self._normalize_rubric_config(rubric.config)
            tz_value = _extract(config)
            if tz_value:
                return tz_value

        return TZ_OFFSET

    def get_tz_offset(self, user_id: int) -> str:
        cur = self.db.execute("SELECT tz_offset FROM users WHERE user_id=?", (user_id,))
        row = cur.fetchone()
        return row["tz_offset"] if row and row["tz_offset"] else TZ_OFFSET

    def is_authorized(self, user_id: Any) -> bool:
        return self.get_user(user_id) is not None

    def is_superadmin(self, user_id: Any) -> bool:
        row = self.get_user(user_id)
        return row and row["is_superadmin"]

    def get_superadmin_ids(self) -> list[int]:
        """Return list of all superadmin user IDs."""
        cur = self.db.execute("SELECT user_id FROM users WHERE is_superadmin=1")
        return [row["user_id"] for row in cur.fetchall()]

    def get_amber_sea(self) -> int | None:
        row = self.db.execute("SELECT sea_id FROM amber_state LIMIT 1").fetchone()
        return row["sea_id"] if row else None

    def set_amber_sea(self, sea_id: int) -> None:
        self.db.execute("DELETE FROM amber_state")
        self.db.execute(
            "INSERT INTO amber_state (sea_id, storm_start, active) VALUES (?, NULL, 0)",
            (sea_id,),
        )
        self.db.commit()

    def get_amber_channels(self) -> set[int]:
        cur = self.db.execute("SELECT channel_id FROM amber_channels")
        return {r["channel_id"] for r in cur.fetchall()}

    def is_amber_channel(self, channel_id: int) -> bool:
        cur = self.db.execute("SELECT 1 FROM amber_channels WHERE channel_id=?", (channel_id,))
        return cur.fetchone() is not None

    async def show_amber_channels(self, user_id: int) -> None:
        enabled = self.get_amber_channels()
        cur = self.db.execute("SELECT chat_id, title FROM channels")
        rows = cur.fetchall()
        if not rows:
            await self.api_request(
                "sendMessage", {"chat_id": user_id, "text": "No channels available"}
            )
            return
        for r in rows:
            cid = r["chat_id"]
            icon = "✅" if cid in enabled else "❌"
            btn = "Выключить отправку" if cid in enabled else "Включить отправку"
            keyboard = {
                "inline_keyboard": [[{"text": btn, "callback_data": f"amber_toggle:{cid}"}]]
            }
            await self.api_request(
                "sendMessage",
                {
                    "chat_id": user_id,
                    "text": f"{r['title'] or cid} {icon}",
                    "reply_markup": keyboard,
                },
            )

    async def parse_post_url(self, url: str) -> tuple[int, int] | None:
        """Return chat_id and message_id from a Telegram post URL."""
        m = re.search(r"/c/(\d+)/(\d+)", url)
        if m:
            return int(f"-100{m.group(1)}"), int(m.group(2))
        m = re.search(r"t.me/([^/]+)/(\d+)", url)
        if m:
            resp = await self.api_request("getChat", {"chat_id": f"@{m.group(1)}"})
            if resp.get("ok"):
                return resp["result"]["id"], int(m.group(2))
        return None

    def _get_cached_weather(self, city_id: int) -> Any:
        return self.db.execute(
            "SELECT temperature, weather_code, wind_speed, is_day FROM weather_cache_hour "
            "WHERE city_id=? ORDER BY timestamp DESC LIMIT 1",
            (city_id,),
        ).fetchone()

    def _get_period_weather(self, city_id: int) -> Any:
        return self.db.execute(
            "SELECT * FROM weather_cache_period WHERE city_id=?",
            (city_id,),
        ).fetchone()

    def _get_sea_cache(self, sea_id: int) -> Any:
        return self.db.execute(
            "SELECT current, morning, day, evening, night, wave, "
            "morning_wave, day_wave, evening_wave, night_wave FROM sea_cache WHERE sea_id=?",
            (sea_id,),
        ).fetchone()

    def _get_sea_conditions(self, sea_id: int) -> dict[str, Any] | None:
        row = self.db.execute(
            """
            SELECT
                wave_height_m,
                wind_speed_10m_ms,
                wind_speed_10m_kmh,
                wind_gusts_10m_ms,
                wind_gusts_10m_kmh,
                wind_units,
                wind_gusts_units,
                wind_time_ref,
                cloud_cover_pct,
                updated
            FROM sea_conditions
            WHERE sea_id=?
            """,
            (sea_id,),
        ).fetchone()
        if not row:
            return None

        keys = set(row.keys()) if hasattr(row, "keys") else set()

        def _get(name: str) -> Any:
            return row[name] if name in keys else None

        return {
            "wave_height_m": self._safe_float(row["wave_height_m"]),
            "wind_speed_10m_ms": self._safe_float(row["wind_speed_10m_ms"]),
            "wind_speed_10m_kmh": self._safe_float(_get("wind_speed_10m_kmh")),
            "wind_gusts_10m_ms": self._safe_float(_get("wind_gusts_10m_ms")),
            "wind_gusts_10m_kmh": self._safe_float(_get("wind_gusts_10m_kmh")),
            "wind_units": (_get("wind_units") or None),
            "wind_gusts_units": (_get("wind_gusts_units") or None),
            "wind_time_ref": _get("wind_time_ref"),
            "cloud_cover_pct": self._safe_float(row["cloud_cover_pct"]),
            "updated": row["updated"],
        }

    def _select_baltic_fact(
        self,
        facts: Sequence[Fact],
        *,
        now: datetime,
        rng: random.Random | None = None,
    ) -> tuple[Fact | None, dict[str, Any]]:
        if not facts:
            return None, {"reason": "empty"}
        total = len(facts)
        window_days = max(7, min(21, math.ceil(total * 0.6)))
        now_dt = now.astimezone(UTC) if now.tzinfo else now.replace(tzinfo=UTC)
        now_ts = int(now_dt.timestamp())
        day_utc = now_ts // 86400
        start_day = day_utc - window_days + 1
        recent_rows = self.data.get_fact_rollout_range(start_day, end_day=day_utc - 1)
        recent_ids = {fact_id for _day, fact_id in recent_rows}
        candidates = [fact for fact in facts if fact.id not in recent_ids]
        fallback_reason: str | None = None
        if not candidates:
            fallback_reason = "window_relaxed"
            yesterday_id = self.data.get_fact_rollout_for_day(day_utc - 1)
            if yesterday_id:
                candidates = [fact for fact in facts if fact.id != yesterday_id]
            else:
                candidates = list(facts)
        if not candidates:
            info = {
                "window_days": window_days,
                "recent_ids": sorted(recent_ids),
                "candidates": [],
            }
            if fallback_reason:
                info["fallback"] = fallback_reason
            return None, info
        rng_obj: Any = rng or random
        usage_map = self.data.get_fact_usage_map()
        weights: list[float] = []
        fact_ids: list[str] = []
        weight_map: dict[str, float] = {}
        for fact in candidates:
            uses_count, _last_used = usage_map.get(fact.id, (0, None))
            weight = 1.0 / (1.0 + float(uses_count))
            if weight <= 0:
                weight = 1e-6
            weights.append(weight)
            fact_ids.append(fact.id)
            weight_map[fact.id] = weight
        try:
            selected = rng_obj.choices(candidates, weights=weights, k=1)[0]
        except Exception:
            logging.exception("SEA_RUBRIC facts weighted_choice_failed")
            selected = rng_obj.choice(candidates)
        self.data.record_fact_selection(selected.id, now_ts, day_utc)
        info = {
            "window_days": window_days,
            "recent_ids": sorted(recent_ids),
            "candidates": fact_ids,
            "weights": weight_map,
            "chosen_id": selected.id,
            "chosen_preview": selected.text[:80],
        }
        if fallback_reason:
            info["fallback"] = fallback_reason
        return selected, info

    def _prepare_sea_fact(
        self,
        *,
        sea_id: int,
        storm_state: str,
        enable_facts: bool,
        now: datetime,
        rng: random.Random | None = None,
    ) -> tuple[str | None, str | None, dict[str, Any]]:
        if not enable_facts:
            logging.info("SEA_RUBRIC facts skip reason=disabled")
            return None, None, {"reason": "disabled"}
        if storm_state == "strong_storm":
            logging.info("SEA_RUBRIC facts skip reason=strong_storm")
            return None, None, {"reason": "strong_storm"}
        facts = load_baltic_facts()
        if not facts:
            logging.warning("SEA_RUBRIC facts skip reason=no_facts")
            return None, None, {"reason": "empty"}
        selected, info = self._select_baltic_fact(facts, now=now, rng=rng)
        if selected is None:
            logging.info(
                "SEA_RUBRIC facts choose_failed window_days=%s candidates=%s",
                info.get("window_days"),
                len(info.get("candidates") or []),
            )
            return None, None, info
        preview = selected.text.replace("\n", " ")[:80]
        logging.info(
            'SEA_RUBRIC facts choose window_days=%s candidates=%s chosen={id:%s, preview:"%s"} reason="lowest uses_count, rnd"',
            info.get("window_days"),
            len(info.get("candidates") or []),
            selected.id,
            preview,
        )
        return selected.text, selected.id, info

    def _is_storm_persisting(
        self,
        sea_id: int,
        *,
        now: datetime,
        current_state: str,
    ) -> tuple[bool, str | None]:
        normalized_state = str(current_state or "").strip().lower()
        storm_states = {"storm", "strong_storm", "heavy_storm"}
        if normalized_state not in storm_states:
            return False, None

        now_dt = now.astimezone(UTC) if now.tzinfo else now.replace(tzinfo=UTC)
        threshold = now_dt - timedelta(hours=36)

        try:
            lookup_time = now_dt - timedelta(minutes=1)
            record = self.data.get_last_sea_post_entry(sea_id, before=lookup_time)
        except Exception:
            logging.exception("SEA_RUBRIC storm_persist lookup_failed sea_id=%s", sea_id)
            record = None

        if record:
            prev_published_at, prev_metadata = record
            prev_state = str(prev_metadata.get("storm_state") or "").strip().lower()
            if prev_state in storm_states:
                prev_dt = (
                    prev_published_at.astimezone(UTC)
                    if prev_published_at.tzinfo
                    else prev_published_at.replace(tzinfo=UTC)
                )
                if prev_dt >= threshold:
                    return True, f"found in publish_history at {prev_dt.isoformat()}"

        weather_hit = self._find_recent_sea_storm_event(sea_id, since=threshold)
        if weather_hit:
            ts, _state = weather_hit
            return True, f"found in weather_history at {ts.isoformat()}"

        return False, None

    def _find_recent_sea_storm_event(
        self,
        sea_id: int,
        *,
        since: datetime,
    ) -> tuple[datetime, str] | None:
        def _to_utc(dt: datetime | None) -> datetime | None:
            if dt is None:
                return None
            if dt.tzinfo:
                return dt.astimezone(UTC)
            return dt.replace(tzinfo=UTC)

        def _classify_wave(value: Any) -> str | None:
            try:
                wave_height = float(value)
            except (TypeError, ValueError):
                return None
            if wave_height > 1.5:
                return "strong_storm"
            if wave_height > 0.5:
                return "storm"
            return None

        conn = self.data.conn
        since_utc = since.astimezone(UTC) if since.tzinfo else since.replace(tzinfo=UTC)
        since_iso = since_utc.isoformat()
        history_tables: list[str] = []
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'sea_%history%'"
            ).fetchall()
        except Exception:
            rows = []
        for row in rows:
            table_name = row["name"] if "name" in row.keys() else row[0]
            if not table_name:
                continue
            safe_table_name = table_name.replace("'", "''")
            safe_table_quoted = table_name.replace('"', '""')
            try:
                pragma = conn.execute(f"PRAGMA table_info('{safe_table_name}')").fetchall()
            except Exception:
                continue
            column_names = {str(col["name"] if "name" in col.keys() else col[1]) for col in pragma}
            if "sea_id" not in column_names:
                continue
            ts_column = next(
                (
                    candidate
                    for candidate in (
                        "observed_at",
                        "timestamp",
                        "recorded_at",
                        "created_at",
                        "updated",
                    )
                    if candidate in column_names
                ),
                None,
            )
            if not ts_column:
                continue
            state_column = next(
                (
                    candidate
                    for candidate in (
                        "storm_state",
                        "state",
                        "storm",
                        "status",
                    )
                    if candidate in column_names
                ),
                None,
            )
            wave_column = next(
                (
                    candidate
                    for candidate in (
                        "wave_height_m",
                        "wave",
                        "wave_height",
                        "sea_wave_height",
                    )
                    if candidate in column_names
                ),
                None,
            )
            if not state_column and not wave_column:
                continue
            ts_quoted = ts_column.replace('"', '""')
            select_parts = [f'"{ts_quoted}" AS ts']
            if state_column:
                state_quoted = state_column.replace('"', '""')
                select_parts.append(f'"{state_quoted}" AS state')
            if wave_column:
                wave_quoted = wave_column.replace('"', '""')
                select_parts.append(f'"{wave_quoted}" AS wave')
            query = (
                f"SELECT {', '.join(select_parts)} FROM \"{safe_table_quoted}\" "
                f'WHERE "sea_id"=? AND "{ts_quoted}" >= ? '
                f'ORDER BY "{ts_quoted}" DESC LIMIT 20'
            )
            try:
                history_rows = conn.execute(query, (sea_id, since_iso)).fetchall()
            except Exception:
                continue
            for history_row in history_rows:
                ts_raw = history_row["ts"] if "ts" in history_row.keys() else history_row[0]
                try:
                    parsed = datetime.fromisoformat(str(ts_raw))
                except Exception:
                    continue
                ts_utc = _to_utc(parsed)
                if ts_utc is None or ts_utc < since_utc:
                    continue
                state_value: str | None = None
                if "state" in history_row.keys():
                    state_value = str(history_row["state"] or "").strip().lower() or None
                if state_value in {"storm", "heavy_storm", "strong_storm"}:
                    normalized = "strong_storm" if state_value != "storm" else "storm"
                    return ts_utc, normalized
                if "wave" in history_row.keys():
                    guess = _classify_wave(history_row["wave"])
                    if guess:
                        return ts_utc, guess

        # Fallback to cached conditions
        for table_name, ts_column, wave_column in (
            ("sea_conditions", "updated", "wave_height_m"),
            ("sea_cache", "updated", "wave"),
        ):
            table_quoted = table_name.replace('"', '""')
            ts_quoted = ts_column.replace('"', '""')
            wave_quoted = wave_column.replace('"', '""')
            try:
                row = conn.execute(
                    f'SELECT "{ts_quoted}" AS ts, "{wave_quoted}" AS wave FROM "{table_quoted}" WHERE "sea_id"=?',
                    (sea_id,),
                ).fetchone()
            except Exception:
                continue
            if not row:
                continue
            ts_raw = row["ts"] if "ts" in row.keys() else row[0]
            try:
                parsed = datetime.fromisoformat(str(ts_raw))
            except Exception:
                continue
            ts_utc = _to_utc(parsed)
            if ts_utc is None or ts_utc < since_utc:
                continue
            guess = _classify_wave(row["wave"] if "wave" in row.keys() else row[1])
            if guess:
                return ts_utc, guess

        return None

    @staticmethod
    def strip_header(text: str | None) -> str | None:

        if not text:
            return text
        if WEATHER_SEPARATOR not in text:
            return text
        prefix, rest = text.split(WEATHER_SEPARATOR, 1)
        if WEATHER_HEADER_PATTERN.search(prefix.strip()):
            return rest.lstrip()
        return text

    @staticmethod
    def _parse_coords(text: str) -> tuple[float, float] | None:
        """Parse latitude and longitude from string allowing comma separator."""
        parts = [p for p in re.split(r"[ ,]+", text.strip()) if p]
        if len(parts) != 2:
            return None
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            return None

    def _render_template(self, template: str) -> str | None:
        """Replace placeholders in template with cached weather values."""

        months = [
            "января",
            "февраля",
            "марта",
            "апреля",
            "мая",
            "июня",
            "июля",
            "августа",
            "сентября",
            "октября",
            "ноября",
            "декабря",
        ]

        def repl(match: re.Match[str]) -> str:
            cid = int(match.group(1))
            period = match.group(2)
            field = match.group(3)

            if field in {"seatemperature", "seatemp"}:
                row = self._get_sea_cache(cid)
                if not row:
                    raise ValueError(f"no sea data for {cid}")
                key = {
                    "nm": "morning",
                    "nd": "day",
                    "ny": "evening",
                    "nn": "night",
                }.get(period, "current")
                if row[key] is None:
                    raise ValueError(f"no sea {key} for {cid}")
                emoji = "\U0001f30a"
                return f"{emoji} {row[key]:.1f}\u00b0C"

            if field == "seastorm":
                row = self._get_sea_cache(cid)
                if not row:
                    raise ValueError(f"no sea data for {cid}")
                t_key = {
                    "nm": "morning",
                    "nd": "day",
                    "ny": "evening",
                    "nn": "night",
                }.get(period, "current")
                wave_key = {
                    "nm": "morning_wave",
                    "nd": "day_wave",
                    "ny": "evening_wave",
                    "nn": "night_wave",
                }.get(period, "wave")
                temp = row[t_key]
                wave = row[wave_key]
                if wave is None or temp is None:
                    raise ValueError(f"no sea storm data for {cid}")

                try:
                    wave_val = float(wave)
                    temp_val = float(temp)
                except (TypeError, ValueError):
                    raise ValueError(f"invalid sea storm data for {cid}")

                emoji = "\U0001f30a"
                if wave_val < 0.5:
                    return f"{emoji} {temp_val:.1f}\u00b0C"
                if wave_val >= 1.5:
                    return f"{emoji} сильный шторм"
                return f"{emoji} шторм"

            row = self._get_cached_weather(cid)
            period_row = self._get_period_weather(cid) if period else None
            if not row and not period_row:
                raise ValueError(f"no data for city {cid}")

            if field in {"temperature", "temp", "wind"}:
                if period_row and period:
                    key_map = {
                        "nm": ("morning_temp", "morning_code", "morning_wind", 1),
                        "nd": ("day_temp", "day_code", "day_wind", 1),
                        "ny": ("evening_temp", "evening_code", "evening_wind", 1),
                        "nn": ("night_temp", "night_code", "night_wind", 0),
                    }
                    t_key, c_key, w_key, is_day_val = key_map[period]
                    if period_row[t_key] is not None:
                        if field in {"temperature", "temp"}:
                            emoji = weather_emoji(period_row[c_key], is_day_val)

                            return f"{emoji} {period_row[t_key]:.0f}\u00b0C"

                        if field == "wind":
                            return f"{period_row[w_key]:.1f}"
                if not row:
                    raise ValueError(f"no current data for city {cid}")
                is_day = row["is_day"] if "is_day" in row.keys() else None
                if field in {"temperature", "temp"}:
                    emoji = weather_emoji(row["weather_code"], is_day)
                    return f"{emoji} {row['temperature']:.1f}\u00b0C"

                return f"{row['wind_speed']:.1f}"
            return ""

        try:

            rendered = re.sub(r"{(\d+)\|(?:(nm|nd|ny|nn)-)?(\w+)}", repl, template)
            tomorrow = date.today() + timedelta(days=1)
            rendered = rendered.replace("{next-day-date}", tomorrow.strftime("%d"))
            rendered = rendered.replace("{next-day-month}", months[tomorrow.month - 1])

            return rendered
        except ValueError as e:
            logging.info("%s", e)
            return None

    @staticmethod
    def post_url(chat_id: int, message_id: int) -> str:
        if str(chat_id).startswith("-100"):
            return f"https://t.me/c/{str(chat_id)[4:]}/{message_id}"
        return f"https://t.me/{chat_id}/{message_id}"

    async def update_weather_posts(self, cities: set[int] | None = None) -> None:
        """Update all registered posts using cached weather."""
        cur = self.db.execute(
            "SELECT id, chat_id, message_id, template, base_text, base_caption, reply_markup FROM weather_posts"
        )
        rows = cur.fetchall()
        for r in rows:
            tpl_cities = {int(m.group(1)) for m in re.finditer(r"{(\d+)\|", r["template"])}
            if cities is not None and not (tpl_cities & cities):
                continue
            header = self._render_template(r["template"])
            if header is None:
                continue

            markup = json.loads(r["reply_markup"]) if r["reply_markup"] else None
            if r["base_caption"] is not None:
                caption = f"{header}{WEATHER_SEPARATOR}{r['base_caption']}"
                payload = {
                    "chat_id": r["chat_id"],
                    "message_id": r["message_id"],
                    "caption": caption,
                }
                if markup:
                    payload["reply_markup"] = markup
                resp = await self.api_request(
                    "editMessageCaption",
                    payload,
                )
            else:
                text = (
                    f"{header}{WEATHER_SEPARATOR}{r['base_text']}"
                    if r["base_text"] is not None
                    else header
                )

                payload = {
                    "chat_id": r["chat_id"],
                    "message_id": r["message_id"],
                    "text": text,
                }
                if markup:
                    payload["reply_markup"] = markup
                resp = await self.api_request(
                    "editMessageText",
                    payload,
                )
            if resp.get("ok"):
                logging.info("Updated weather post %s", r["id"])
            elif resp.get("error_code") == 400 and "message is not modified" in resp.get(
                "description", ""
            ):
                logging.info("Weather post %s already up to date", r["id"])
            else:
                logging.error("Failed to update weather post %s: %s", r["id"], resp)

    def latest_weather_url(self) -> str | None:
        cur = self.db.execute("SELECT chat_id, message_id FROM latest_weather_post LIMIT 1")
        row = cur.fetchone()
        if row:
            return self.post_url(row["chat_id"], row["message_id"])
        return None

    def set_latest_weather_post(self, chat_id: int, message_id: int) -> None:
        self.db.execute("DELETE FROM latest_weather_post")
        self.db.execute(
            "INSERT INTO latest_weather_post (chat_id, message_id, published_at) VALUES (?, ?, ?)",
            (chat_id, message_id, datetime.utcnow().isoformat()),
        )
        self.db.commit()

    async def update_weather_buttons(self) -> None:
        url = self.latest_weather_url()
        if not url:
            return
        cur = self.db.execute(
            "SELECT chat_id, message_id, base_markup, button_texts FROM weather_link_posts"
        )
        for r in cur.fetchall():
            base = json.loads(r["base_markup"]) if r["base_markup"] else {"inline_keyboard": []}
            buttons = base.get("inline_keyboard", [])

            weather_buttons = []
            for t in json.loads(r["button_texts"]):
                rendered = self._render_template(t) or t
                weather_buttons.append({"text": rendered, "url": url})
            if weather_buttons:
                buttons.append(weather_buttons)

            resp = await self.api_request(
                "editMessageReplyMarkup",
                {
                    "chat_id": r["chat_id"],
                    "message_id": r["message_id"],
                    "reply_markup": {"inline_keyboard": buttons},
                },
            )

            if not resp.get("ok") and not (
                resp.get("error_code") == 400
                and "message is not modified" in resp.get("description", "")
            ):
                logging.error("Failed to update buttons for %s: %s", r["message_id"], resp)

            self.db.execute(
                "UPDATE weather_posts SET reply_markup=? WHERE chat_id=? AND message_id=?",
                (json.dumps({"inline_keyboard": buttons}), r["chat_id"], r["message_id"]),
            )
        self.db.commit()

    def add_weather_channel(self, channel_id: int, post_time: str) -> None:
        next_run = self.next_weather_run(post_time, TZ_OFFSET, allow_past=True)
        self.data.upsert_weather_job(channel_id, post_time, next_run)

    def remove_weather_channel(self, channel_id: int) -> None:
        self.data.remove_weather_job(channel_id)

    def list_weather_channels(self) -> Any:
        jobs = self.data.list_weather_jobs()
        rows = []
        for job in jobs:
            title_row = self.db.execute(
                "SELECT title FROM channels WHERE chat_id=?",
                (job.channel_id,),
            ).fetchone()
            rows.append(
                {
                    "channel_id": job.channel_id,
                    "post_time": job.post_time,
                    "last_published_at": job.last_run_at.isoformat() if job.last_run_at else None,
                    "next_run_at": job.run_at.isoformat(),
                    "title": title_row["title"] if title_row else None,
                }
            )
        return rows

    def set_asset_channel(self, channel_id: int) -> None:
        self.set_weather_assets_channel(channel_id)
        self.set_recognition_channel(channel_id)

    def set_weather_assets_channel(self, channel_id: int | None) -> None:
        self._store_single_channel("asset_channel", channel_id)
        self.weather_assets_channel_id = channel_id
        self.uploads_config = replace(
            self.uploads_config,
            assets_channel_id=channel_id,
        )

    def get_weather_assets_channel(self) -> int | None:
        cur = self.db.execute("SELECT channel_id FROM asset_channel LIMIT 1")
        row = cur.fetchone()
        return row["channel_id"] if row else None

    def set_recognition_channel(self, channel_id: int | None) -> None:
        self._store_single_channel("recognition_channel", channel_id)
        self.recognition_channel_id = channel_id

    def get_recognition_channel(self) -> int | None:
        cur = self.db.execute("SELECT channel_id FROM recognition_channel LIMIT 1")
        row = cur.fetchone()
        return row["channel_id"] if row else None

    def _store_single_channel(self, table: str, channel_id: int | None) -> None:
        self.db.execute(f"DELETE FROM {table}")
        if channel_id is not None:
            self.db.execute(
                f"INSERT INTO {table} (channel_id) VALUES (?)",
                (channel_id,),
            )
        self.db.commit()

    def add_asset(
        self,
        message_id: int,
        hashtags: str,
        template: str | None = None,
        *,
        channel_id: int | None = None,
        metadata: dict[str, Any] | None = None,
        tg_chat_id: int | None = None,
        kind: str | None = None,
        file_meta: dict[str, Any] | None = None,
        author_user_id: int | None = None,
        author_username: str | None = None,
        sender_chat_id: int | None = None,
        via_bot_id: int | None = None,
        forward_from_user: int | None = None,
        forward_from_chat: int | None = None,
        origin: str = "weather",
    ) -> str:
        source_channel = channel_id or self.weather_assets_channel_id or 0
        asset_id = self.data.save_asset(
            source_channel,
            message_id,
            template,
            hashtags,
            tg_chat_id=tg_chat_id or source_channel,
            caption=template,
            kind=kind,
            file_meta=file_meta,
            author_user_id=author_user_id,
            author_username=author_username,
            sender_chat_id=sender_chat_id,
            via_bot_id=via_bot_id,
            forward_from_user=forward_from_user,
            forward_from_chat=forward_from_chat,
            metadata=metadata,
            origin=origin,
        )
        logging.info("Stored asset %s tags=%s", message_id, hashtags)
        return asset_id

    async def _prompt_channel_selection(
        self,
        user_id: int,
        *,
        pending_key: str,
        callback_prefix: str,
        prompt: str,
    ) -> None:
        cur = self.db.execute("SELECT chat_id, title FROM channels")
        rows = cur.fetchall()
        if not rows:
            await self.api_request(
                "sendMessage",
                {"chat_id": user_id, "text": "No channels available"},
            )
            return
        keyboard = {
            "inline_keyboard": [
                [{"text": r["title"], "callback_data": f'{callback_prefix}:{r["chat_id"]}'}]
                for r in rows
            ]
        }
        self.pending[user_id] = {pending_key: True}
        await self.api_request(
            "sendMessage",
            {
                "chat_id": user_id,
                "text": prompt,
                "reply_markup": keyboard,
            },
        )

    def _schedule_ingest_job(self, asset_id: str | None, *, reason: str) -> None:
        if not asset_id:
            logging.warning("Skipping ingest scheduling with missing asset id (%s)", reason)
            return
        asset = self.data.get_asset(asset_id)
        if not asset:
            logging.warning("Asset %s not found when scheduling ingest (%s)", asset_id, reason)
            return
        job_id = self.jobs.enqueue("ingest", {"asset_id": asset_id}, dedupe=True)
        logging.info(
            "Scheduled ingest job %s for asset %s (kind=%s, file_id=%s, author=%s, reason=%s)",
            job_id,
            asset_id,
            asset.kind or "unknown",
            asset.file_id,
            asset.author_user_id,
            reason,
        )

    def _collect_asset_metadata(self, message: dict[str, Any]) -> dict[str, Any]:
        caption = message.get("caption") or message.get("text") or ""
        tags = " ".join(re.findall(r"#\S+", caption))
        chat_id = message.get("chat", {}).get("id", 0)
        message_id = message.get("message_id", 0)
        from_user = message.get("from") or {}
        sender_chat_id = (
            message.get("sender_chat", {}).get("id") if message.get("sender_chat") else None
        )
        via_bot_id = message.get("via_bot", {}).get("id") if message.get("via_bot") else None
        forward_from_user = (
            message.get("forward_from", {}).get("id") if message.get("forward_from") else None
        )
        forward_from_chat = (
            message.get("forward_from_chat", {}).get("id")
            if message.get("forward_from_chat")
            else None
        )
        metadata: dict[str, Any] = {
            "date": message.get("date"),
        }
        file_meta: dict[str, Any] = {}
        kind = None
        if message.get("photo"):
            kind = "photo"
            photo = sorted(message["photo"], key=lambda p: p.get("file_size", 0))[-1]
            file_meta = {
                "file_id": photo.get("file_id"),
                "file_unique_id": photo.get("file_unique_id"),
                "file_size": photo.get("file_size"),
                "width": photo.get("width"),
                "height": photo.get("height"),
            }
        elif message.get("document"):
            kind = "document"
            doc = message["document"]
            file_meta = {
                "file_id": doc.get("file_id"),
                "file_unique_id": doc.get("file_unique_id"),
                "file_name": doc.get("file_name"),
                "mime_type": doc.get("mime_type"),
                "file_size": doc.get("file_size"),
            }
        elif message.get("video"):
            kind = "video"
            vid = message["video"]
            file_meta = {
                "file_id": vid.get("file_id"),
                "file_unique_id": vid.get("file_unique_id"),
                "duration": vid.get("duration"),
                "width": vid.get("width"),
                "height": vid.get("height"),
                "file_size": vid.get("file_size"),
            }
        elif message.get("animation"):
            kind = "animation"
            anim = message["animation"]
            file_meta = {
                "file_id": anim.get("file_id"),
                "file_unique_id": anim.get("file_unique_id"),
                "file_name": anim.get("file_name"),
                "mime_type": anim.get("mime_type"),
                "duration": anim.get("duration"),
                "file_size": anim.get("file_size"),
            }
        return {
            "metadata": metadata,
            "hashtags": tags,
            "caption": caption,
            "tg_chat_id": chat_id,
            "message_id": message_id,
            "kind": kind,
            "file_meta": file_meta or None,
            "author_user_id": from_user.get("id"),
            "author_username": from_user.get("username"),
            "sender_chat_id": sender_chat_id,
            "via_bot_id": via_bot_id,
            "forward_from_user": forward_from_user,
            "forward_from_chat": forward_from_chat,
        }

    async def _download_file(
        self, file_id: str, dest_path: str | Path | None = None
    ) -> Path | bytes | None:
        if self.dry_run:
            return b""
        if not self.session:
            logging.warning("HTTP session is not initialised for download")
            return None
        resp = await self.api_request("getFile", {"file_id": file_id})
        file_path = resp.get("result", {}).get("file_path") if resp else None
        if not file_path:
            logging.error("No file_path returned for file %s", file_id)
            return None
        if dest_path is None:
            logging.error("Destination path is required for file %s", file_id)
            return None
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_dest = (
            dest.with_suffix(dest.suffix + ".part")
            if dest.suffix
            else dest.with_name(dest.name + ".part")
        )
        url = f"https://api.telegram.org/file/bot{self.token}/{file_path}"
        try:
            async with self.session.get(url) as file_resp:
                if file_resp.status != 200:
                    logging.error("Failed to download file %s: HTTP %s", file_id, file_resp.status)
                    return None
                with tmp_dest.open("wb") as fh:
                    async for chunk in file_resp.content.iter_chunked(64 * 1024):
                        if not chunk:
                            continue
                        fh.write(chunk)
            tmp_dest.replace(dest)
            return dest
        except Exception:
            logging.exception("Failed to stream download for file %s", file_id)
            with contextlib.suppress(FileNotFoundError):
                tmp_dest.unlink()
            return None

    @staticmethod
    def _normalize_gps_ref(ref: Any) -> str | None:
        if ref is None:
            return None
        original_ref = ref
        if isinstance(ref, (bytes, bytearray)):
            for encoding in ("ascii", "latin-1"):
                try:
                    ref = ref.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                ref = original_ref.decode("latin-1", errors="ignore")
        if not isinstance(ref, str):
            ref = str(ref)
        normalized = ref.replace("\x00", "").strip().upper()
        if not normalized:
            return None
        candidate = normalized[0]
        if candidate in {"N", "S", "E", "W"}:
            return candidate
        return None

    @staticmethod
    def _convert_gps(value: Any, ref: Any, axis: str, source: str) -> float | None:
        if not value:
            return None
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            components = list(value)
        else:
            components = [value]
        coerced: list[float] = []
        for index, component in enumerate(components):
            numerator: float | None
            denominator: float | None
            if hasattr(component, "num") and hasattr(component, "den"):
                numerator = getattr(component, "num", None)
                denominator = getattr(component, "den", None)
            elif isinstance(component, tuple) and len(component) == 2:
                numerator, denominator = component
            else:
                numerator = component
                denominator = 1
            try:
                num_value = float(numerator)
                den_value = float(denominator)
            except (TypeError, ValueError):
                logging.debug(
                    "GPS %s conversion from %s failed to coerce component[%s]; value=%r (type=%s), ref=%r (type=%s)",
                    axis,
                    source,
                    index,
                    component,
                    type(component).__name__,
                    ref,
                    type(ref).__name__ if ref is not None else "NoneType",
                )
                return None
            if den_value == 0:
                logging.debug(
                    "GPS %s conversion from %s encountered zero denominator in component[%s]; value=%r (type=%s), ref=%r (type=%s)",
                    axis,
                    source,
                    index,
                    component,
                    type(component).__name__,
                    ref,
                    type(ref).__name__ if ref is not None else "NoneType",
                )
                return None
            coerced.append(num_value / den_value)
        if len(coerced) < 3:
            logging.debug(
                "GPS %s conversion from %s has insufficient components; value=%r (type=%s), ref=%r (type=%s)",
                axis,
                source,
                value,
                type(value).__name__,
                ref,
                type(ref).__name__ if ref is not None else "NoneType",
            )
            return None
        decimal = coerced[0] + coerced[1] / 60 + coerced[2] / 3600
        normalized_ref = Bot._normalize_gps_ref(ref)
        if normalized_ref is None:
            logging.debug(
                "GPS %s ref normalization from %s failed; raw=%r (type=%s)",
                axis,
                source,
                ref,
                type(ref).__name__ if ref is not None else "NoneType",
            )
        elif normalized_ref in {"S", "W"}:
            decimal = -decimal
        return decimal

    def _extract_gps(self, image_source: str | Path | BinaryIO) -> tuple[float, float] | None:
        exif_dict: dict[str, Any] | None = None
        loaded_from_path = False
        try:
            with Image.open(image_source, mode="r") as img:
                exif_bytes = img.info.get("exif")
        except Exception:
            logging.exception("Failed to parse EXIF metadata")
            exif_bytes = None
        if exif_bytes:
            try:
                exif_dict = piexif.load(exif_bytes)
            except Exception:
                logging.exception("Failed to load EXIF metadata from embedded bytes")
                exif_dict = None
        if exif_dict is None and isinstance(image_source, (str, Path)):
            try:
                exif_dict = piexif.load(str(image_source))
                loaded_from_path = True
            except Exception:
                logging.exception("Failed to load EXIF metadata from file path")
                exif_dict = None
        if exif_dict is not None and not loaded_from_path and isinstance(image_source, (str, Path)):
            gps_ifd = exif_dict.get("GPS") or {}
            zeroth_source = exif_dict.get("0th")
            zeroth_ifd = zeroth_source if isinstance(zeroth_source, dict) else {}
            has_gps_pointer = any(key in zeroth_ifd for key in (piexif.ImageIFD.GPSTag, "GPSInfo"))
            if has_gps_pointer and not gps_ifd:
                try:
                    exif_dict = piexif.load(str(image_source))
                    loaded_from_path = True
                except Exception:
                    logging.debug(
                        "Failed to reload EXIF metadata for GPS pointer fallback", exc_info=True
                    )
        if exif_dict:
            gps = exif_dict.get("GPS") or {}
            lat = self._convert_gps(
                gps.get(piexif.GPSIFD.GPSLatitude),
                gps.get(piexif.GPSIFD.GPSLatitudeRef),
                "latitude",
                "piexif",
            )
            lon = self._convert_gps(
                gps.get(piexif.GPSIFD.GPSLongitude),
                gps.get(piexif.GPSIFD.GPSLongitudeRef),
                "longitude",
                "piexif",
            )
            if lat is not None and lon is not None:
                return lat, lon
        if exifread is not None and isinstance(image_source, (str, Path)):
            try:
                with Path(image_source).open("rb") as fh:
                    tags = exifread.process_file(fh, details=False)
            except Exception:
                logging.debug("exifread failed to process file for GPS extraction", exc_info=True)
            else:
                lat_field = tags.get("GPS GPSLatitude")
                lon_field = tags.get("GPS GPSLongitude")
                lat_ref_field = tags.get("GPS GPSLatitudeRef")
                lon_ref_field = tags.get("GPS GPSLongitudeRef")

                def _field_values(field: Any) -> Any:
                    if field is None:
                        return None
                    return getattr(field, "values", field)

                def _single_value(field: Any) -> Any:
                    if isinstance(field, Sequence) and not isinstance(
                        field, (str, bytes, bytearray)
                    ):
                        return field[0] if field else None
                    return field

                lat_value = _field_values(lat_field)
                lon_value = _field_values(lon_field)
                lat_ref_value = _single_value(_field_values(lat_ref_field))
                lon_ref_value = _single_value(_field_values(lon_ref_field))

                lat = self._convert_gps(lat_value, lat_ref_value, "latitude", "exifread")
                lon = self._convert_gps(lon_value, lon_ref_value, "longitude", "exifread")
                if lat is not None and lon is not None:
                    return lat, lon
        return None

    def _extract_exif_full(self, image_source: str | Path | BinaryIO) -> dict[str, dict[str, Any]]:
        try:
            with Image.open(image_source, mode="r") as img:
                exif_bytes = img.info.get("exif")
            if exif_bytes:
                exif_dict = piexif.load(exif_bytes)
            else:
                exif_dict = piexif.load(str(image_source))
        except Exception:
            logging.exception("Failed to parse EXIF metadata")
            return {}

        def _format_value(value: Any) -> Any:
            if isinstance(value, bytes):
                try:
                    return value.decode("utf-8")
                except UnicodeDecodeError:
                    return value.hex()
            if isinstance(value, dict):
                return {k: _format_value(v) for k, v in value.items()}
            if isinstance(value, tuple):
                return [_format_value(v) for v in value]
            if isinstance(value, list):
                return [_format_value(v) for v in value]
            return value

        readable: dict[str, dict[str, Any]] = {}
        for ifd in ("0th", "Exif", "GPS", "1st"):
            source = exif_dict.get(ifd)
            if not isinstance(source, dict):
                continue
            tag_map = piexif.TAGS.get(ifd, {})
            formatted: dict[str, Any] = {}
            for tag_id, raw_value in source.items():
                tag_info = tag_map.get(tag_id)
                tag_name = tag_info.get("name") if tag_info else None
                name = tag_name or str(tag_id)
                formatted[name] = _format_value(raw_value)
            readable[ifd] = formatted
        return readable

    @staticmethod
    def _parse_exif_datetime_month(value: str | bytes | None) -> int | None:
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    value = value.decode("latin-1")
                except UnicodeDecodeError:
                    return None
        if not value:
            return None
        text = str(value).strip()
        if not text:
            return None
        cleaned = text.replace("-", ":").replace(".", ":")
        date_part = cleaned.split()[0]
        bits = date_part.split(":")
        if len(bits) >= 2 and bits[1].isdigit():
            month = int(bits[1])
            if 1 <= month <= 12:
                return month
        # fallback: try to parse integer month from entire string
        match = re.search(r"(?:^|\D)([0-1]?\d)(?:\D|$)", text)
        if match:
            month = int(match.group(1))
            if 1 <= month <= 12:
                return month
        return None

    @classmethod
    def _extract_exif_datetimes(cls, image_source: str | Path | BinaryIO) -> dict[str, str]:
        return extract_exif_datetimes(image_source)

    @staticmethod
    def _normalize_month_value(value: Any) -> int | None:
        if isinstance(value, int):
            if 1 <= value <= 12:
                return value
            return None
        if isinstance(value, float):
            month = int(value)
            if 1 <= month <= 12:
                return month
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            month = Bot._parse_exif_datetime_month(text)
            if month:
                return month
            try:
                month = int(text)
            except ValueError:
                return None
            return month if 1 <= month <= 12 else None
        return None

    @classmethod
    def _extract_month_from_metadata(cls, metadata: dict[str, Any] | None) -> int | None:
        if not isinstance(metadata, dict):
            return None
        stack: list[Any] = [metadata]
        seen: set[int] = set()
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                obj_id = id(item)
                if obj_id in seen:
                    continue
                seen.add(obj_id)
                for key, value in item.items():
                    if isinstance(key, str):
                        lowered = key.lower()
                        if "month" in lowered or lowered in {"month"}:
                            month = cls._normalize_month_value(value)
                            if month:
                                return month
                        if "date" in lowered:
                            month = cls._normalize_month_value(value)
                            if month:
                                return month
                    if isinstance(value, (dict, list, tuple)):
                        stack.append(value)
            elif isinstance(item, (list, tuple)):
                for value in item:
                    if isinstance(value, (dict, list, tuple)):
                        stack.append(value)
        return None

    def _extract_exif_month(self, image_source: str | Path | BinaryIO) -> int | None:
        try:
            with Image.open(image_source, mode="r") as img:
                exif_bytes = img.info.get("exif")
            if exif_bytes:
                exif_dict = piexif.load(exif_bytes)
            else:
                if isinstance(image_source, (str, Path)):
                    exif_dict = piexif.load(str(image_source))
                else:
                    return None
        except Exception:
            logging.exception("Failed to parse EXIF metadata")
            return None
        exif_ifd = exif_dict.get("Exif") or {}
        for tag in (
            piexif.ExifIFD.DateTimeOriginal,
            piexif.ExifIFD.DateTimeDigitized,
        ):
            if tag in exif_ifd:
                month = self._parse_exif_datetime_month(exif_ifd.get(tag))
                if month:
                    return month
        zeroth_ifd = exif_dict.get("0th") or {}
        if piexif.ImageIFD.DateTime in zeroth_ifd:
            month = self._parse_exif_datetime_month(zeroth_ifd.get(piexif.ImageIFD.DateTime))
            if month:
                return month
        return None

    @staticmethod
    def _normalize_weather_enum(value: Any) -> str | None:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if not normalized:
                return None
            normalized = re.sub(r"[\s\-]+", "_", normalized)
            if normalized in WEATHER_ALLOWED_VALUES:
                return normalized
            alias = WEATHER_ALIAS_MAP.get(normalized)
            if isinstance(alias, str) and alias in WEATHER_ALLOWED_VALUES:
                return alias
            return None
        if isinstance(value, (list, tuple)):
            for item in value:
                normalized = Bot._normalize_weather_enum(item)
                if normalized:
                    return normalized
            return None
        if isinstance(value, dict):
            for key in ("enum", "code", "value", "tag", "name", "weather"):
                if key in value:
                    normalized = Bot._normalize_weather_enum(value.get(key))
                    if normalized:
                        return normalized
            return None
        return None

    @classmethod
    def _extract_weather_enum_from_metadata(cls, metadata: dict[str, Any] | None) -> str | None:
        if not isinstance(metadata, dict):
            return None
        stack: list[Any] = [metadata]
        seen: set[int] = set()
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                obj_id = id(item)
                if obj_id in seen:
                    continue
                seen.add(obj_id)
                for key, value in item.items():
                    if isinstance(key, str) and "weather" in key.lower():
                        normalized = cls._normalize_weather_enum(value)
                        if normalized:
                            return normalized
                    if isinstance(value, (dict, list, tuple)):
                        stack.append(value)
            elif isinstance(item, (list, tuple)):
                for value in item:
                    if isinstance(value, (dict, list, tuple)):
                        stack.append(value)
        return None

    @staticmethod
    def _weather_display(value: str | None) -> str | None:
        if not value:
            return None
        normalized = Bot._normalize_weather_enum(value)
        if not normalized:
            return None
        translation = WEATHER_TAG_TRANSLATIONS.get(normalized)
        if translation:
            return translation
        return normalized

    @staticmethod
    def _normalize_season(value: str | None) -> str | None:
        if not value:
            return None
        season = value.strip().lower()
        if not season:
            return None
        if season == "fall":
            return "autumn"
        if season in {"spring", "summer", "autumn", "winter"}:
            return season
        return None

    @staticmethod
    def _season_from_month(value: int | None) -> str | None:
        if value is None:
            return None
        return SEASON_BY_MONTH.get(value)

    def _resolve_asset_season(self, asset: Asset) -> str | None:
        season: str | None = None
        vision_payload = asset.vision_results if isinstance(asset.vision_results, Mapping) else None
        if vision_payload:
            raw_season = vision_payload.get("season_final")
            if isinstance(raw_season, str):
                season = self._normalize_season(raw_season)
        if not season:
            month = self._extract_month_from_metadata(asset.metadata)
            season_from_month = self._season_from_month(month)
            season = self._normalize_season(season_from_month)
        return season

    def _collect_asset_seasons(self, assets: Sequence[Asset]) -> dict[int, str]:
        seasons: dict[int, str] = {}
        for asset in assets:
            season = self._resolve_asset_season(asset)
            if season:
                seasons[asset.id] = season
        return seasons

    def _filter_flower_assets_by_season(
        self,
        assets: Sequence[Asset],
        *,
        desired_count: int | None = None,
        min_count: int | None = None,
        max_count: int,
    ) -> tuple[list[Asset], dict[str, str]] | None:
        target = desired_count if desired_count is not None else min_count
        if target is None:
            return None
        required_count = int(target)
        if not assets or required_count <= 0:
            return None
        asset_seasons: dict[str, str] = {}
        filtered_assets: list[Asset] = []
        for asset in assets:
            season = self._resolve_asset_season(asset)
            if not season:
                continue
            asset_seasons[asset.id] = season
            filtered_assets.append(asset)
        if len(filtered_assets) < required_count:
            return None

        unique_seasons = {asset_seasons[asset.id] for asset in filtered_assets}
        candidate_sets: list[set[str]] = []
        for season in unique_seasons:
            candidate_sets.append({season})
            for neighbor in SEASON_ADJACENCY.get(season, set()):
                if neighbor in unique_seasons:
                    candidate_sets.append({season, neighbor})

        best_selection: list[Asset] | None = None

        for allowed_seasons in candidate_sets:
            selection = [
                asset for asset in filtered_assets if asset_seasons.get(asset.id) in allowed_seasons
            ]
            if len(selection) < required_count:
                continue
            selection = selection[:max_count]
            if not best_selection or len(selection) > len(best_selection):
                best_selection = selection
            elif best_selection and len(selection) == len(best_selection):
                current_diversity = len({asset_seasons[a.id] for a in best_selection})
                new_diversity = len({asset_seasons[a.id] for a in selection})
                if new_diversity < current_diversity:
                    best_selection = selection

        if not best_selection:
            return None

        final_seasons = {asset.id: asset_seasons[asset.id] for asset in best_selection}
        return best_selection, final_seasons

    async def reverse_geocode_osm(self, lat: float, lon: float) -> dict[str, Any]:
        if self.dry_run or not self.session:
            return {}

        async with self._revgeo_semaphore:
            now = datetime.utcnow()
            if self._last_geocode_at:
                elapsed = (now - self._last_geocode_at).total_seconds()
                if elapsed < 1:
                    await asyncio.sleep(1 - elapsed)

            url = "https://nominatim.openstreetmap.org/reverse"
            params = {
                "lat": f"{lat:.6f}",
                "lon": f"{lon:.6f}",
                "format": "jsonv2",
                "zoom": "18",
                "addressdetails": "1",
                "accept-language": "ru",
            }
            headers = {"User-Agent": "kotopogoda-bot/1.0 (+контакт)"}

            try:
                async with self.session.get(url, params=params, headers=headers) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"HTTP {resp.status}")
                    data = await resp.json()
            finally:
                self._last_geocode_at = datetime.utcnow()

        if not isinstance(data, dict):
            return {}

        address = data.get("address")
        if not isinstance(address, dict):
            return {}

        def _pick(keys: Sequence[str]) -> str | None:
            for key in keys:
                value = address.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return None

        return {
            "street": _pick(["road", "pedestrian", "footway", "residential"]),
            "house_number": _pick(["house_number"]),
            "district": _pick(["suburb", "district"]),
            "city": _pick(["city", "town", "village", "hamlet"]),
            "state": _pick(["state"]),
            "country": _pick(["country"]),
            "postcode": _pick(["postcode"]),
        }

    async def _reverse_geocode(self, lat: float, lon: float) -> dict[str, Any]:
        empty = {
            "street": None,
            "house_number": None,
            "district": None,
            "city": None,
            "state": None,
            "country": None,
            "postcode": None,
            "fallback": None,
        }

        if self.dry_run or not self.session:
            return empty

        key = (round(lat, 5), round(lon, 5))
        now = datetime.utcnow()

        cached = self._revgeo_cache.get(key)
        if cached and now - cached[1] < self._revgeo_ttl:
            result = {**cached[0], "fallback": None}
            fallback_cached = self._revgeo_fallback_cache.get(key)
            if fallback_cached and now - fallback_cached[1] < self._revgeo_ttl:
                result["fallback"] = fallback_cached[0]
            return result

        fallback_cached = self._revgeo_fallback_cache.get(key)
        if fallback_cached and now - fallback_cached[1] < self._revgeo_ttl:
            return {**empty, "fallback": fallback_cached[0]}

        fail_cached = self._revgeo_fail_cache.get(key)
        if fail_cached and now - fail_cached < self._revgeo_fail_ttl:
            return empty

        try:
            components = await self.reverse_geocode_osm(lat, lon)
        except Exception as exc:
            logging.warning(
                "REVGEO fail provider=osm lat=%.5f lon=%.5f error=%s",
                lat,
                lon,
                exc,
            )
            components = {}

        if components:
            logging.info(
                "REVGEO ok provider=osm lat=%.5f lon=%.5f street=%s city=%s country=%s",
                lat,
                lon,
                components.get("street"),
                components.get("city"),
                components.get("country"),
            )
            self._revgeo_cache[key] = (components, now)
            self._revgeo_fail_cache.pop(key, None)
            return {**components, "fallback": None}

        twogis_fallback = await self._reverse_geocode_twogis(lat, lon)
        if twogis_fallback:
            logging.info(
                "REVGEO ok provider=2gis lat=%.5f lon=%.5f text=%s",
                lat,
                lon,
                twogis_fallback,
            )
            formatted = f"Адрес (2ГИС): {twogis_fallback}"
            self._revgeo_fallback_cache[key] = (formatted, now)
            self._revgeo_fail_cache.pop(key, None)
            return {**empty, "fallback": formatted}

        self._revgeo_fail_cache[key] = now
        return empty

    async def _reverse_geocode_twogis(self, lat: float, lon: float) -> str | None:
        if not self._twogis_api_key or not self.session:
            return None

        params = {
            "lat": f"{lat:.6f}",
            "lon": f"{lon:.6f}",
            "type": "building",
            "fields": "items.full_name,items.address_name",
            "key": self._twogis_api_key,
        }
        url = "https://catalog.api.2gis.com/3.0/items/geocode"

        try:
            async with self.session.get(url, params=params) as resp:
                if resp.status == 429:
                    delay = min(self._twogis_backoff_seconds, 60.0)
                    logging.warning(
                        "REVGEO warn provider=2gis lat=%.5f lon=%.5f status=429 retry_in=%.1fs",
                        lat,
                        lon,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    self._twogis_backoff_seconds = min(self._twogis_backoff_seconds * 2, 60.0)
                    return None
                if resp.status != 200:
                    text = await resp.text()
                    delay = min(self._twogis_backoff_seconds, 60.0)
                    logging.warning(
                        "REVGEO warn provider=2gis lat=%.5f lon=%.5f status=%s body=%s retry_in=%.1fs",
                        lat,
                        lon,
                        resp.status,
                        text,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    self._twogis_backoff_seconds = min(self._twogis_backoff_seconds * 2, 60.0)
                    return None
                data = await resp.json()
        except Exception as exc:
            delay = min(self._twogis_backoff_seconds, 60.0)
            logging.warning(
                "REVGEO warn provider=2gis lat=%.5f lon=%.5f error=%s retry_in=%.1fs",
                lat,
                lon,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
            self._twogis_backoff_seconds = min(self._twogis_backoff_seconds * 2, 60.0)
            return None

        self._twogis_backoff_seconds = 1.0

        items = data.get("result", {}).get("items") if isinstance(data, dict) else None
        if not items:
            return None
        best = items[0]
        if not isinstance(best, dict):
            return None
        return best.get("full_name") or best.get("address_name") or best.get("name")

    async def _is_near_sea(self, lat: float, lon: float) -> bool:
        if self.dry_run or not self.session:
            return False

        key = (round(lat, 5), round(lon, 5))
        now = datetime.utcnow()

        cached = self._shoreline_cache.get(key)
        if cached:
            value, ts = cached
            if now - ts < self._revgeo_ttl:
                return value

        fail_cached = self._shoreline_fail_cache.get(key)
        if fail_cached and now - fail_cached < self._revgeo_fail_ttl:
            return False

        query = (
            "[out:json][timeout:25];("  # noqa: E501
            f'way(around:250,{lat:.6f},{lon:.6f})["natural"="coastline"];'
            f'relation(around:250,{lat:.6f},{lon:.6f})["natural"="coastline"];'
            f'way(around:250,{lat:.6f},{lon:.6f})["natural"="water"]["water"~"sea|ocean"];'
            f'relation(around:250,{lat:.6f},{lon:.6f})["natural"="water"]["water"~"sea|ocean"];'
            f'way(around:250,{lat:.6f},{lon:.6f})["place"="sea"];'
            f'relation(around:250,{lat:.6f},{lon:.6f})["place"="sea"];'
            ");out ids;"
        )
        url = "https://overpass-api.de/api/interpreter"
        headers = {"User-Agent": "kotopogoda-bot/1.0 (+контакт)"}

        try:
            async with self.session.post(url, data={"data": query}, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")
                payload = await resp.json()
        except Exception as exc:
            logging.warning(
                "Marine proximity lookup failed lat=%.5f lon=%.5f error=%s",
                lat,
                lon,
                exc,
            )
            self._shoreline_fail_cache[key] = now
            return False

        elements = []
        if isinstance(payload, dict):
            raw_elements = payload.get("elements")
            if isinstance(raw_elements, list):
                elements = raw_elements

        is_marine = bool(elements)
        self._shoreline_cache[key] = (is_marine, now)
        if is_marine:
            self._shoreline_fail_cache.pop(key, None)
        else:
            self._shoreline_fail_cache[key] = now
        return is_marine

    async def _maybe_append_marine_tag(self, asset: Asset, tags: list[str]) -> None:
        normalized = {tag.strip().lower() for tag in tags if isinstance(tag, str) and tag.strip()}
        if "sea" in normalized:
            return

        if normalized.intersection(MARINE_TAG_SYNONYMS):
            tags.append("sea")
            return

        lat_raw = getattr(asset, "latitude", None)
        lon_raw = getattr(asset, "longitude", None)
        if lat_raw is None or lon_raw is None:
            return

        try:
            lat = float(lat_raw)
            lon = float(lon_raw)
        except (TypeError, ValueError):
            return

        try:
            if await self._is_near_sea(lat, lon):
                tags.append("sea")
        except Exception:
            logging.exception(
                "Marine proximity enrichment failed for asset %s", getattr(asset, "id", "unknown")
            )

    def _format_exif_address_caption(
        self, address: dict[str, Any] | None, lat: float, lon: float
    ) -> tuple[str, set[str], bool]:
        address = address or {}

        street = address.get("street")
        house_number = address.get("house_number")
        city = address.get("city")
        state = address.get("state")
        country = address.get("country")

        street_parts: list[str] = []
        if isinstance(street, str) and street.strip():
            street_parts.append(street.strip())
        if isinstance(house_number, str) and house_number.strip():
            street_parts.append(house_number.strip())
        street_line = " ".join(street_parts)

        location_parts: list[str] = []
        district = address.get("district")
        if isinstance(city, str) and city.strip():
            location_parts.append(city.strip())
        elif isinstance(district, str) and district.strip():
            location_parts.append(district.strip())
        if isinstance(state, str) and state.strip():
            location_parts.append(state.strip())
        if isinstance(country, str) and country.strip():
            location_parts.append(country.strip())

        outside_region = (
            isinstance(state, str) and state.strip() and state.strip() != "Калининградская область"
        )

        location_line = ", ".join(location_parts)
        if location_line and outside_region:
            location_line += " [вне региона]"

        has_osm_components = bool(street_line or location_line)

        address_parts: list[str] = []
        if street_line:
            address_parts.append(street_line)
        if location_line:
            address_parts.append(location_line)

        coords_fragment = f"(lat {lat:.5f}, lon {lon:.5f})"
        if address_parts:
            caption_line = f"Адрес (EXIF): {', '.join(address_parts)} {coords_fragment}"
        else:
            caption_line = f"Адрес (EXIF): lat {lat:.5f}, lon {lon:.5f}"

        dedupe_values: set[str] = set()
        for value in (street, district, city, state, country):
            if isinstance(value, str) and value.strip():
                dedupe_values.add(value.strip().lower())

        return caption_line, dedupe_values, has_osm_components

    def _build_local_file_path(self, asset_id: str, file_meta: dict[str, Any]) -> Path:
        suffix = ""
        file_name = file_meta.get("file_name")
        if file_name:
            suffix = Path(file_name).suffix
        if not suffix and file_meta.get("mime_type"):
            if "/" in file_meta["mime_type"]:
                suffix = "." + file_meta["mime_type"].split("/")[-1]
        unique = file_meta.get("file_unique_id") or str(asset_id)
        filename = f"{asset_id}_{unique}{suffix}" if suffix else f"{asset_id}_{unique}"
        return self.asset_storage / filename

    def _store_local_file(self, asset_id: str, file_meta: dict[str, Any], data: bytes) -> str:
        path = self._build_local_file_path(asset_id, file_meta)
        try:
            path.write_bytes(data)
        except Exception:
            logging.exception("Failed to write asset file %s", path)
        return str(path)

    async def _job_ingest(self, job: Job) -> None:
        asset_id = job.payload.get("asset_id") if job.payload else None
        if not asset_id:
            logging.warning("Ingest job %s missing asset_id", job.id)
            return
        asset = self.data.get_asset(asset_id)
        if not asset:
            logging.warning("Asset %s not found for ingest", asset_id)
            return
        file_id = asset.file_id
        if not file_id:
            logging.warning("Asset %s has no file information", asset_id)
            return
        if self.dry_run:
            logging.info("Dry run ingest for asset %s", asset_id)
            self.data.update_asset(
                asset_id,
                metadata={"ingest_skipped": True},
            )
            tz_offset = (
                self.get_tz_offset(asset.author_user_id) if asset.author_user_id else TZ_OFFSET
            )
            vision_job = self.jobs.enqueue("vision", {"asset_id": asset_id, "tz_offset": tz_offset})
            logging.info(
                "Asset %s queued for vision job %s after ingest (dry run)", asset_id, vision_job
            )
            return
        file_meta = {
            "file_id": asset.file_id,
            "file_unique_id": asset.file_unique_id,
            "file_name": asset.file_name,
            "mime_type": asset.mime_type,
            "duration": asset.duration,
            "file_size": asset.file_size,
            "width": asset.width,
            "height": asset.height,
        }
        target_path = self._build_local_file_path(asset_id, file_meta)
        downloaded_path = await self._download_file(file_id, target_path)
        if not downloaded_path:
            raise RuntimeError(f"Failed to download file for asset {asset_id}")

        metrics = self.upload_metrics or UploadMetricsRecorder(emitter=LoggingMetricsEmitter())

        upload_id_value = str(asset.upload_id or asset_id)
        storage_key_value = str(asset.file_ref or asset.file_id or asset.file_unique_id or asset_id)
        ingestion_context = UploadIngestionContext(
            upload_id=upload_id_value,
            storage_key=storage_key_value,
            metrics=metrics,
            source=asset.source or "telegram",
            device_id=None,
            user_id=asset.author_user_id,
            job_id=job.id,
            job_name=job.name,
        )

        target_channel_id = (
            asset.tg_chat_id or asset.channel_id or self.uploads_config.assets_channel_id
        )
        if target_channel_id is None:
            raise RuntimeError(f"Asset {asset_id} missing channel for ingest")

        base_metadata = dict(asset.metadata or {})
        exif_datetimes = self._extract_exif_datetimes(downloaded_path)
        if exif_datetimes:
            base_metadata.update(exif_datetimes)

        file_metadata = {k: v for k, v in file_meta.items() if v is not None}
        existing_categories = list(asset.categories)

        overrides: dict[str, Any] = {
            "asset_id": asset_id,
            "channel_id": target_channel_id,
            "tg_chat_id": asset.tg_chat_id or target_channel_id,
            "template": asset.caption_template,
            "hashtags": asset.hashtags,
            "caption": asset.caption,
            "kind": asset.kind,
            "metadata": base_metadata,
            "categories": existing_categories,
            "rubric_id": asset.rubric_id,
            "origin": asset.origin,
            "author_user_id": asset.author_user_id,
            "author_username": asset.author_username,
            "sender_chat_id": asset.sender_chat_id,
            "via_bot_id": asset.via_bot_id,
            "forward_from_user": asset.forward_from_user,
            "forward_from_chat": asset.forward_from_chat,
            "file_metadata": file_metadata,
        }

        stored_exif = asset.exif or base_metadata.get("exif")
        if stored_exif:
            overrides["exif"] = stored_exif

        gps_override: dict[str, Any] = {}
        metadata_gps = base_metadata.get("gps") if base_metadata else None
        if isinstance(metadata_gps, dict):
            gps_override.update(metadata_gps)
        stored_latitude = asset.latitude
        stored_longitude = asset.longitude
        latitude_value = stored_latitude
        if latitude_value is not None:
            gps_override.setdefault("latitude", latitude_value)
        longitude_value = stored_longitude
        if longitude_value is not None:
            gps_override.setdefault("longitude", longitude_value)
        if gps_override:
            overrides["gps"] = gps_override

        existing_metadata_for_callback = base_metadata.copy()

        class _TelegramReuseAdapter:
            def __init__(
                self,
                bot: Bot,
                *,
                message_id: int | None,
                chat_id: int | None,
            ) -> None:
                self._bot = bot
                self._message_id = message_id
                self._chat_id = chat_id

            async def send_photo(
                self,
                *,
                chat_id: int,
                photo: Path,
                caption: str | None = None,
            ) -> dict[str, Any]:
                if self._message_id and (self._chat_id is None or self._chat_id == chat_id):
                    logging.debug(
                        "Reusing existing Telegram message %s for asset %s",
                        self._message_id,
                        asset_id,
                    )
                    return {
                        "message_id": self._message_id,
                        "chat": {"id": self._chat_id or chat_id},
                    }
                response, _ = await self._bot._publish_as_photo(chat_id, str(photo), caption)
                return response or {}

            async def send_document(
                self,
                *,
                chat_id: int,
                document: BinaryIO | bytes,
                file_name: str,
                caption: str | None = None,
                content_type: str | None = None,
            ) -> dict[str, Any]:
                if self._message_id and (self._chat_id is None or self._chat_id == chat_id):
                    logging.debug(
                        "Reusing existing Telegram message %s for asset %s",
                        self._message_id,
                        asset_id,
                    )
                    return {
                        "message_id": self._message_id,
                        "chat": {"id": self._chat_id or chat_id},
                    }
                if isinstance(document, (bytes, bytearray)):
                    document_stream: BinaryIO = io.BytesIO(document)
                else:
                    document_stream = document
                response = await self._bot._publish_mobile_document(
                    chat_id,
                    document_stream,
                    file_name,
                    caption,
                    content_type=content_type,
                )
                return response or {}

        def _save_asset(payload: dict[str, Any]) -> str:
            merged_metadata = existing_metadata_for_callback.copy()
            incoming_metadata = payload.get("metadata") or {}
            if incoming_metadata:
                merged_metadata.update(incoming_metadata)
            categories_payload = payload.get("categories") or existing_categories
            result_id = self.data.save_asset(
                payload["channel_id"],
                payload["message_id"],
                (
                    payload.get("template")
                    if payload.get("template") is not None
                    else asset.caption_template
                ),
                payload.get("hashtags") if payload.get("hashtags") is not None else asset.hashtags,
                tg_chat_id=payload.get("tg_chat_id") or target_channel_id,
                caption=(
                    payload.get("caption") if payload.get("caption") is not None else asset.caption
                ),
                kind=payload.get("kind") if payload.get("kind") is not None else asset.kind,
                file_meta=payload.get("file_meta"),
                metadata=merged_metadata or None,
                categories=categories_payload or None,
                rubric_id=(
                    payload.get("rubric_id")
                    if payload.get("rubric_id") is not None
                    else asset.rubric_id
                ),
                origin=payload.get("origin") or asset.origin or "telegram",
                source=payload.get("source") or asset.source or "telegram",
                author_user_id=(
                    payload.get("author_user_id")
                    if payload.get("author_user_id") is not None
                    else asset.author_user_id
                ),
                author_username=(
                    payload.get("author_username")
                    if payload.get("author_username") is not None
                    else asset.author_username
                ),
                sender_chat_id=(
                    payload.get("sender_chat_id")
                    if payload.get("sender_chat_id") is not None
                    else asset.sender_chat_id
                ),
                via_bot_id=(
                    payload.get("via_bot_id")
                    if payload.get("via_bot_id") is not None
                    else asset.via_bot_id
                ),
                forward_from_user=(
                    payload.get("forward_from_user")
                    if payload.get("forward_from_user") is not None
                    else asset.forward_from_user
                ),
                forward_from_chat=(
                    payload.get("forward_from_chat")
                    if payload.get("forward_from_chat") is not None
                    else asset.forward_from_chat
                ),
            )
            update_kwargs: dict[str, Any] = {"local_path": None}
            latitude = payload.get("latitude")
            longitude = payload.get("longitude")
            if latitude is not None:
                update_kwargs["latitude"] = latitude
            if longitude is not None:
                update_kwargs["longitude"] = longitude

            exif_flag = payload.get("exif_present")
            gps_present = latitude is not None and longitude is not None
            if exif_flag is True:
                update_kwargs["exif_present"] = True
            elif gps_present:
                update_kwargs["exif_present"] = True
            self.data.update_asset(result_id, **update_kwargs)
            return result_id

        telegram_adapter = _TelegramReuseAdapter(
            self,
            message_id=asset.message_id,
            chat_id=asset.tg_chat_id,
        )
        telegram_config = replace(
            self.uploads_config,
            assets_channel_id=target_channel_id,
            vision_enabled=False,
        )

        callbacks = IngestionCallbacks(save_asset=_save_asset)

        try:
            result = await ingest_photo(
                data=self.data,
                telegram=telegram_adapter,
                openai=self.openai,
                supabase=self.supabase,
                config=telegram_config,
                context=ingestion_context,
                file_path=downloaded_path,
                cleanup_file=False,
                callbacks=callbacks,
                input_overrides=overrides,
            )
        finally:
            self._remove_file(str(downloaded_path))

        if asset.local_path and asset.local_path != str(downloaded_path):
            self._remove_file(asset.local_path)

        should_extract_gps = asset.kind == "photo" or self._is_convertible_image_document(asset)
        gps_payload = dict(result.gps or {})
        lat = gps_payload.get("latitude")
        lon = gps_payload.get("longitude")
        override_latitude = gps_override.get("latitude") if gps_override else None
        override_longitude = gps_override.get("longitude") if gps_override else None
        if lat is None:
            fallback_latitude = (
                override_latitude if override_latitude is not None else stored_latitude
            )
            if fallback_latitude is not None:
                gps_payload["latitude"] = fallback_latitude
                lat = fallback_latitude
        if lon is None:
            fallback_longitude = (
                override_longitude if override_longitude is not None else stored_longitude
            )
            if fallback_longitude is not None:
                gps_payload["longitude"] = fallback_longitude
                lon = fallback_longitude
        if result.gps is not gps_payload:
            result.gps = gps_payload
        if should_extract_gps and (lat is None or lon is None):
            author_id = asset.author_user_id
            if author_id:
                await self.api_request(
                    "sendMessage",
                    {
                        "chat_id": author_id,
                        "text": "В изображении отсутствуют EXIF-данные с координатами.",
                    },
                )
        update_kwargs: dict[str, Any] = {}
        if lat is not None and lon is not None:
            update_kwargs["latitude"] = lat
            update_kwargs["longitude"] = lon
            logging.info(
                "MOBILE_REVERSE_GEOCODE_INPUT",
                extra={
                    "asset_id": asset_id,
                    "stored_latitude": stored_latitude,
                    "stored_longitude": stored_longitude,
                    "override_latitude": override_latitude,
                    "override_longitude": override_longitude,
                    "ingested_latitude": lat,
                    "ingested_longitude": lon,
                },
            )
            address = await self._reverse_geocode(lat, lon)
            if address:
                city = address.get("city") or address.get("town") or address.get("village")
                country = address.get("country")
                if city:
                    update_kwargs["city"] = city
                if country:
                    update_kwargs["country"] = country
        else:
            stored_has_coords = stored_latitude is not None and stored_longitude is not None
            override_has_coords = override_latitude is not None and override_longitude is not None
            ingest_has_coords = lat is not None and lon is not None
            if not stored_has_coords and not override_has_coords and not ingest_has_coords:
                logging.warning(
                    "MOBILE_REVERSE_GEOCODE_MISSING_COORDS",
                    extra={
                        "asset_id": asset_id,
                        "stored_latitude": stored_latitude,
                        "stored_longitude": stored_longitude,
                        "override_latitude": override_latitude,
                        "override_longitude": override_longitude,
                        "ingested_latitude": lat,
                        "ingested_longitude": lon,
                    },
                )
        if update_kwargs:
            self.data.update_asset(asset_id, **update_kwargs)

        if asset.source == "mobile":
            record_mobile_photo_ingested()
        tz_offset = self.get_tz_offset(asset.author_user_id) if asset.author_user_id else TZ_OFFSET
        vision_job = self.jobs.enqueue("vision", {"asset_id": asset_id, "tz_offset": tz_offset})
        logging.info("Asset %s queued for vision job %s after ingest", asset_id, vision_job)

    async def _job_vision(self, job: Job) -> None:
        async with self._vision_semaphore:
            await self._job_vision_locked(job)

    async def _job_vision_locked(self, job: Job) -> None:
        def _utf16_length(text: str) -> int:
            return len(text.encode("utf-16-le")) // 2

        start_time = datetime.utcnow()
        asset_id = job.payload.get("asset_id") if job.payload else None
        cleanup_paths: list[str] = []
        logging.info("Starting vision job %s for asset %s", job.id, asset_id)
        try:
            if not asset_id:
                logging.warning("Vision job %s missing asset_id", job.id)
                return
            asset = self.data.get_asset(asset_id)
            if not asset:
                logging.warning("Asset %s missing for vision", asset_id)
                return
            file_id = asset.file_id
            if not file_id:
                logging.warning("Asset %s has no file for vision", asset_id)
                return

            file_meta = {
                "file_id": asset.file_id,
                "file_unique_id": asset.file_unique_id,
                "file_name": asset.file_name,
                "mime_type": asset.mime_type,
                "duration": asset.duration,
                "file_size": asset.file_size,
                "width": asset.width,
                "height": asset.height,
            }
            local_path = asset.local_path if asset.local_path else None
            if local_path and os.path.exists(local_path):
                logging.info(
                    "Vision job %s using cached file %s for asset %s",
                    job.id,
                    local_path,
                    asset_id,
                )
                cleanup_paths.append(local_path)
            else:
                local_path = None
            if not local_path and not self.dry_run:
                target_path = self._build_local_file_path(asset_id, file_meta)
                downloaded_path = await self._download_file(file_id, target_path)
                if downloaded_path:
                    local_path = str(downloaded_path)
                    cleanup_paths.append(local_path)
                    logging.info(
                        "Vision job %s downloaded asset %s to %s",
                        job.id,
                        asset_id,
                        local_path,
                    )
            if self.openai and not self.openai.api_key:
                self.openai.refresh_api_key()
            if self.dry_run or not self.openai or not self.openai.api_key:
                if self.dry_run:
                    logging.info(
                        "Vision job %s skipped for asset %s: dry run enabled",
                        job.id,
                        asset_id,
                    )
                else:
                    logging.warning(
                        "Vision job %s skipped for asset %s: OpenAI key missing",
                        job.id,
                        asset_id,
                    )
                self.data.update_asset(
                    asset_id, vision_results={"status": "skipped"}, local_path=None
                )
                return
            process = psutil.Process(os.getpid())

            def log_rss(stage: str) -> None:
                try:
                    rss = process.memory_info().rss // (1024 * 1024)
                    logging.info("MEM rss=%sMB stage=%s", rss, stage)
                except Exception:
                    logging.debug("Failed to capture RSS at stage=%s", stage)

            if not local_path or not os.path.exists(local_path):
                raise RuntimeError(f"Local file for asset {asset_id} not found")

            system_prompt = (
                "Ты ассистент проекта Котопогода. Проанализируй изображение и верни JSON, строго соответствующий схеме asset_vision_v1. "
                "Структура включает arch_view (boolean), caption (строка на русском), objects (массив строк), is_outdoor (boolean), guess_country/guess_city (строка или null), "
                "location_confidence (число 0..1), landmarks (массив строк), tags (3-12 элементов в нижнем регистре), framing, архитектурные признаки, погодное описание, сезон и безопасность. "
                "Поле framing обязательно и принимает только close_up, medium, wide. "
                "weather_image описывает нюансы погоды и выбирается из sunny, partly_cloudy, overcast, rain, snow, fog, night. "
                "season_guess — spring, summer, autumn, winter или null. arch_style либо null, либо объект с label (название стиля на английском) и confidence (0..1). "
                "В objects перечисляй заметные элементы, цветы называй видами. В tags используй английские слова в нижнем регистре и обязательно включай погодный тег. "
                "Поле safety содержит nsfw:boolean и reason:string, где reason всегда непустая строка на русском. "
                "Дополнительно определи, есть ли море, океан, пляж или береговая линия — поле is_sea. "
                "Если is_sea=true, оцени sea_wave_score по шкале 0..10 (0 — гладь, 10 — шквал), укажи photo_sky одной из категорий sunny/partly_cloudy/mostly_cloudy/overcast/night/unknown и выставь is_sunset=true, когда заметен закат. "
                "Если моря нет, sea_wave_score ставь null, но всё равно классифицируй photo_sky по видимому небу. "
                "Обязательно укажи sky_visible=true, если на фото видно небо (даже частично), иначе sky_visible=false. Если небо не видно или неясно, ставь photo_sky=unknown."
            )
            user_prompt = (
                "Опиши сцену, перечисли объекты, теги, достопримечательности, архитектуру и безопасность фото. Укажи кадровку (framing), "
                "наличие архитектуры крупным планом и панорам, погодный тег (weather_image), сезон и стиль архитектуры (если можно). "
                "Если локация неочевидна, ставь guess_country/guess_city = null и указывай низкую числовую уверенность. "
                "Отдельно отметь, присутствует ли море/океан/пляж (is_sea), оцени sea_wave_score 0..10, классифицируй небо photo_sky и укажи is_sunset для закатных кадров. "
                "Обязательно определи, видно ли небо на фото (sky_visible), и если небо не видно или неясно, используй photo_sky=unknown."
            )
            self._enforce_openai_limit(job, "gpt-4o-mini")
            logging.info(
                "Vision job %s classifying asset %s using gpt-4o-mini from %s",
                job.id,
                asset_id,
                local_path,
            )
            response = await self.openai.classify_image(
                model="gpt-4o-mini",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_path=Path(local_path),
                schema=ASSET_VISION_V1_SCHEMA,
            )
            log_rss("after_openai")
            gc.collect()
            if response is None:
                logging.warning(
                    "Vision job %s for asset %s returned no response",
                    job.id,
                    asset_id,
                )
                self.data.update_asset(asset_id, vision_results={"status": "skipped"})
                gc.collect()
                return
            result = response.content
            if not isinstance(result, dict):
                raise RuntimeError("Invalid response from vision model")
            framing_raw = result.get("framing")
            framing: str | None = None
            if isinstance(framing_raw, str):
                framing = re.sub(r"[\s\-]+", "_", framing_raw.strip().lower()) or None
            elif framing_raw is not None:
                framing = (
                    re.sub(
                        r"[\s\-]+",
                        "_",
                        str(framing_raw).strip().lower(),
                    )
                    or None
                )
            if not framing:
                raise RuntimeError("Invalid response from vision model: missing framing")
            if framing not in FRAMING_ALLOWED_VALUES:
                alias = FRAMING_ALIAS_MAP.get(framing)
                if alias in FRAMING_ALLOWED_VALUES:
                    framing = alias
                else:
                    raise RuntimeError("Invalid response from vision model: unknown framing")
            architecture_close_up_raw = result.get("architecture_close_up")
            architecture_close_up = (
                bool(architecture_close_up_raw)
                if isinstance(architecture_close_up_raw, bool)
                else str(architecture_close_up_raw).strip().lower() in {"1", "true", "yes", "да"}
            )
            architecture_wide_raw = result.get("architecture_wide")
            architecture_wide = (
                bool(architecture_wide_raw)
                if isinstance(architecture_wide_raw, bool)
                else str(architecture_wide_raw).strip().lower() in {"1", "true", "yes", "да"}
            )
            weather_image_raw = result.get("weather_image")
            weather_image: str | None = None
            if isinstance(weather_image_raw, str):
                weather_image = (
                    re.sub(
                        r"[\s\-]+",
                        "_",
                        weather_image_raw.strip().lower(),
                    )
                    or None
                )
            elif weather_image_raw is not None:
                weather_image = (
                    re.sub(
                        r"[\s\-]+",
                        "_",
                        str(weather_image_raw).strip().lower(),
                    )
                    or None
                )
            if not weather_image:
                raise RuntimeError("Invalid response from vision model: missing weather_image")
            normalized_weather = self._normalize_weather_enum(weather_image)
            if not normalized_weather:
                raise RuntimeError("Invalid response from vision model: unknown weather_image")
            weather_image = normalized_weather
            season_guess_raw = result.get("season_guess")
            if isinstance(season_guess_raw, str):
                season_guess = self._normalize_season(season_guess_raw)
            elif season_guess_raw is None:
                season_guess = None
            else:
                season_guess = self._normalize_season(str(season_guess_raw))
            arch_style_raw = result.get("arch_style")
            arch_style: dict[str, Any] | None
            if isinstance(arch_style_raw, dict):
                label_raw = arch_style_raw.get("label")
                if isinstance(label_raw, str):
                    label = label_raw.strip()
                elif label_raw is None:
                    label = ""
                else:
                    label = str(label_raw).strip()
                if label:
                    confidence_value: float | None = None
                    confidence_raw = arch_style_raw.get("confidence")
                    if isinstance(confidence_raw, (int, float)):
                        confidence_value = float(confidence_raw)
                    elif isinstance(confidence_raw, str):
                        try:
                            confidence_value = float(confidence_raw.strip())
                        except ValueError:
                            confidence_value = None
                    if confidence_value is not None:
                        confidence_value = min(max(confidence_value, 0.0), 1.0)
                    arch_style = {"label": label, "confidence": confidence_value}
                else:
                    arch_style = None
            elif isinstance(arch_style_raw, str):
                label = arch_style_raw.strip()
                arch_style = {"label": label, "confidence": None} if label else None
            else:
                arch_style = None
            usage = response.usage if isinstance(response.usage, dict) else {}
            caption = str(result.get("caption", "")).strip()
            guess_country_raw = result.get("guess_country")
            guess_city_raw = result.get("guess_city")
            if isinstance(guess_country_raw, str):
                guess_country = guess_country_raw.strip() or None
            elif guess_country_raw is None:
                guess_country = None
            else:
                guess_country = str(guess_country_raw).strip() or None
            if isinstance(guess_city_raw, str):
                guess_city = guess_city_raw.strip() or None
            elif guess_city_raw is None:
                guess_city = None
            else:
                guess_city = str(guess_city_raw).strip() or None
            arch_view_raw = result.get("arch_view")
            arch_view = (
                bool(arch_view_raw)
                if isinstance(arch_view_raw, bool)
                else str(arch_view_raw).strip().lower() in {"1", "true", "yes", "да"}
            )
            is_outdoor_raw = result.get("is_outdoor")
            is_outdoor = (
                bool(is_outdoor_raw)
                if isinstance(is_outdoor_raw, bool)
                else str(is_outdoor_raw).strip().lower() in {"1", "true", "yes", "да"}
            )
            is_sea_raw = result.get("is_sea")
            is_sea = (
                bool(is_sea_raw)
                if isinstance(is_sea_raw, bool)
                else str(is_sea_raw).strip().lower() in {"1", "true", "yes", "да"}
            )
            photo_sky_raw = result.get("photo_sky")
            photo_sky_result: str | None = None
            if isinstance(photo_sky_raw, str):
                photo_sky_result = photo_sky_raw.strip() or None
            elif photo_sky_raw is not None:
                photo_sky_result = str(photo_sky_raw).strip() or None
            is_sunset_raw = result.get("is_sunset")
            is_sunset = (
                bool(is_sunset_raw)
                if isinstance(is_sunset_raw, bool)
                else str(is_sunset_raw).strip().lower() in {"1", "true", "yes", "да"}
            )
            sky_visible_raw = result.get("sky_visible")
            sky_visible = (
                bool(sky_visible_raw)
                if isinstance(sky_visible_raw, bool)
                else str(sky_visible_raw).strip().lower() in {"1", "true", "yes", "да"}
            )
            raw_objects = result.get("objects")
            objects: list[str] = []
            if isinstance(raw_objects, list):
                seen_objects: set[str] = set()
                for item in raw_objects:
                    text = str(item).strip()
                    if not text or text in seen_objects:
                        continue
                    seen_objects.add(text)
                    objects.append(text)
            raw_landmarks = result.get("landmarks")
            landmarks: list[str] = []
            if isinstance(raw_landmarks, list):
                seen_landmarks: set[str] = set()
                for item in raw_landmarks:
                    text = str(item).strip()
                    normalized = text.lower()
                    if not text or normalized in seen_landmarks:
                        continue
                    seen_landmarks.add(normalized)
                    landmarks.append(text)
            raw_tags = result.get("tags")
            tags: list[str] = []
            if isinstance(raw_tags, list):
                seen_tags: set[str] = set()
                for tag in raw_tags:
                    text = str(tag).strip().lower()
                    if not text or text in seen_tags:
                        continue
                    seen_tags.add(text)
                    tags.append(text)
            if weather_image and weather_image not in tags:
                tags.append(weather_image)
            if architecture_close_up and "architecture_close_up" not in tags:
                tags.append("architecture_close_up")
            if architecture_wide and "architecture_wide" not in tags:
                tags.append("architecture_wide")
            await self._maybe_append_marine_tag(asset, tags)
            metadata_dict = asset.metadata if isinstance(asset.metadata, dict) else {}
            capture_datetime: datetime | None = None
            capture_time_display: str | None = None
            timestamp_keys = [
                "exif_datetime_best",
                "exif_datetime_original",
                "exif_datetime",
                "exif_datetime_digitized",
            ]
            for ts_key in timestamp_keys:
                raw_value = metadata_dict.get(ts_key)
                if raw_value is None:
                    continue
                if isinstance(raw_value, (list, tuple)):
                    candidate = next(
                        (str(item).strip() for item in raw_value if str(item).strip()),
                        "",
                    )
                else:
                    candidate = str(raw_value).strip()
                if not candidate or candidate.lower() == "none":
                    continue
                try:
                    parsed_dt = datetime.strptime(candidate, "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    parsed_dt = None
                if parsed_dt:
                    capture_datetime = parsed_dt
                    capture_time_display = parsed_dt.strftime("%Y-%m-%d %H:%M")
                else:
                    capture_time_display = candidate
                break
            exif_month = self._extract_month_from_metadata(metadata_dict)
            if exif_month is None and local_path and os.path.exists(local_path):
                exif_month = self._extract_exif_month(local_path)
            season_from_exif = self._season_from_month(exif_month)
            season_final = self._normalize_season(season_from_exif or season_guess)
            season_final_display = SEASON_TRANSLATIONS.get(season_final) if season_final else None
            fallback_weather = self._normalize_weather_enum(weather_image)
            model_weather: str | None = None
            model_weather_display: str | None = None
            for tag_value in tags:
                normalized_tag = self._normalize_weather_enum(tag_value)
                if not normalized_tag:
                    continue
                translated = WEATHER_TAG_TRANSLATIONS.get(normalized_tag)
                if translated:
                    model_weather = normalized_tag
                    model_weather_display = translated
                    break
            if not model_weather and fallback_weather:
                model_weather = fallback_weather
                model_weather_display = WEATHER_TAG_TRANSLATIONS.get(fallback_weather)
            metadata_weather = self._extract_weather_enum_from_metadata(metadata_dict)
            weather_final = metadata_weather or model_weather or fallback_weather
            weather_final = self._normalize_weather_enum(weather_final)
            weather_final_display = self._weather_display(weather_final)
            if not weather_final_display and weather_final:
                weather_final_display = weather_final
            if weather_final and weather_final not in tags:
                tags.append(weather_final)
            photo_weather = weather_final or model_weather
            photo_weather_display: str | None = weather_final_display
            if not photo_weather_display and model_weather_display:
                photo_weather_display = model_weather_display
            if not photo_weather_display and photo_weather:
                photo_weather_display = photo_weather
            supabase_meta = {
                "asset_id": asset_id,
                "channel_id": asset.channel_id,
                "architecture_close_up": architecture_close_up,
                "architecture_wide": architecture_wide,
                "weather_final": photo_weather,
                "weather_final_display": photo_weather_display,
                "season_final": season_final,
                "season_final_display": season_final_display,
            }
            if arch_style:
                supabase_meta["arch_style"] = arch_style
            success, payload, error = await self.supabase.insert_token_usage(
                bot="kotopogoda",
                model="gpt-4o-mini",
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
                request_id=response.request_id,
                endpoint=usage.get("endpoint") or "/v1/responses",
                meta=supabase_meta,
            )
            log_context = {
                "log_token_usage": payload,
                "weather_final": photo_weather,
                "season_final": season_final,
            }
            if arch_style:
                log_context["arch_style"] = arch_style
            if success:
                logging.info("Supabase token usage insert succeeded", extra=log_context)
            else:
                if error == "disabled":
                    logging.debug(
                        "Supabase client disabled; token usage skipped", extra=log_context
                    )
                elif error:
                    logging.error(
                        "Supabase token usage insert failed: %s", error, extra=log_context
                    )
                else:
                    logging.error("Supabase token usage insert failed", extra=log_context)
            safety_raw = result.get("safety")
            nsfw_flag = False
            safety_reason: str | None = None
            if isinstance(safety_raw, dict):
                nsfw_value = safety_raw.get("nsfw")
                if isinstance(nsfw_value, bool):
                    nsfw_flag = nsfw_value
                elif nsfw_value is not None:
                    nsfw_flag = str(nsfw_value).strip().lower() in {"1", "true", "yes", "да"}
                reason_raw = safety_raw.get("reason")
                if isinstance(reason_raw, str):
                    safety_reason = reason_raw.strip() or None
                elif reason_raw is not None:
                    safety_reason = str(reason_raw).strip() or None
            if not safety_reason:
                safety_reason = "обнаружен чувствительный контент" if nsfw_flag else "безопасно"
            location_confidence_raw = result.get("location_confidence")
            location_confidence: float | None = None
            if isinstance(location_confidence_raw, (int, float)):
                location_confidence = float(location_confidence_raw)
            elif isinstance(location_confidence_raw, str):
                try:
                    location_confidence = float(location_confidence_raw.strip())
                except ValueError:
                    location_confidence = None
            if location_confidence is not None:
                location_confidence = min(max(location_confidence, 0.0), 1.0)
            if not caption:
                raise RuntimeError("Invalid response from vision model")
            category = self._derive_primary_scene(caption, tags)
            # Force sea category when is_sea=true, regardless of heuristics
            if is_sea:
                category = "sea"
            rubric_id = self._resolve_rubric_id_for_category(category)
            flower_varieties: list[str] = []
            normalized_tag_set = {tag.lower() for tag in tags if tag}
            if normalized_tag_set.intersection({"flowers", "flower"}):
                flower_varieties = [obj for obj in objects if obj]

            sea_wave_score_data: dict[str, Any] | None = None
            is_sea_asset = is_sea or (
                category == "sea" or normalized_tag_set.intersection({"sea", "ocean"})
            )
            if (
                is_sea_asset
                and self.openai
                and self.openai.api_key
                and local_path
                and os.path.exists(local_path)
            ):
                try:
                    sea_wave_prompt = (
                        "Analyze the sea/ocean in this image and return a JSON with sea wave intensity score. "
                        "Score criteria: 0 = calm/flat, 1-3 = small waves, 4-6 = moderate waves/storm, "
                        "7-8 = strong storm (many whitecaps), 9-10 = very strong storm (massive whitecaps, foam, spray everywhere). "
                        'Evaluate only the sea/ocean visible. Return: {"sea_wave_score": 0-10 (integer), "confidence": 0.0-1.0 (float)}'
                    )
                    sea_wave_schema = {
                        "type": "object",
                        "properties": {
                            "sea_wave_score": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 10,
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                        },
                        "required": ["sea_wave_score", "confidence"],
                    }
                    logging.info(
                        "Vision job %s calling 4o-mini for sea_wave_score on asset %s",
                        job.id,
                        asset_id,
                    )
                    self._enforce_openai_limit(job, "gpt-4o-mini")
                    sea_wave_response = await self.openai.classify_image(
                        model="gpt-4o-mini",
                        system_prompt="You are analyzing sea/ocean conditions. Return only the requested JSON.",
                        user_prompt=sea_wave_prompt,
                        image_path=Path(local_path),
                        schema=sea_wave_schema,
                        schema_name="sea_wave_score_v1",
                    )
                    if sea_wave_response and isinstance(sea_wave_response.content, dict):
                        wave_score_raw = sea_wave_response.content.get("sea_wave_score")
                        confidence_raw = sea_wave_response.content.get("confidence")
                        if isinstance(wave_score_raw, int) and isinstance(
                            confidence_raw, (int, float)
                        ):
                            wave_score = max(0, min(10, wave_score_raw))
                            confidence = max(0.0, min(1.0, float(confidence_raw)))
                            sea_wave_score_data = {
                                "value": wave_score,
                                "confidence": confidence,
                                "model": "gpt-4o-mini",
                            }
                            logging.info(
                                "Sea wave score for asset %s: score=%s conf=%.2f",
                                asset_id,
                                wave_score,
                                confidence,
                            )
                            await self._record_openai_usage(
                                "gpt-4o-mini",
                                sea_wave_response,
                                job=job,
                                record_supabase=True,
                            )
                except Exception:
                    logging.exception(
                        "Failed to get sea_wave_score for asset %s, continuing without it",
                        asset_id,
                    )

            location_parts: list[str] = []
            existing_lower: set[str] = set()
            if asset.city:
                location_parts.append(asset.city)
                existing_lower.add(asset.city.lower())
            if asset.country and asset.country.lower() not in existing_lower:
                location_parts.append(asset.country)
                existing_lower.add(asset.country.lower())
            if guess_city and guess_city.lower() not in existing_lower:
                location_parts.insert(0, guess_city)
                existing_lower.add(guess_city.lower())
            if guess_country and guess_country.lower() not in existing_lower:
                location_parts.append(guess_country)
                existing_lower.add(guess_country.lower())
            exif_coords: tuple[float, float] | None = None
            if asset.latitude is not None and asset.longitude is not None:
                try:
                    exif_coords = (float(asset.latitude), float(asset.longitude))
                except (TypeError, ValueError):
                    exif_coords = None
            if not exif_coords and local_path and os.path.exists(local_path):
                exif_retry = self._extract_gps(local_path)
                if exif_retry:
                    exif_coords = exif_retry
            confidence_display: str | None = None
            if location_confidence is not None and math.isfinite(location_confidence):
                confidence_percent = int(round(location_confidence * 100))
                confidence_percent = max(0, min(100, confidence_percent))
                confidence_display = f"{confidence_percent}%"
            caption_lines = [f"Распознано: {caption}"]
            if exif_coords:
                exif_lat, exif_lon = exif_coords
                exif_address: dict[str, Any] | None = None
                try:
                    exif_address = await self._reverse_geocode(exif_lat, exif_lon)
                except Exception:
                    logging.exception(
                        "Reverse geocode failed for EXIF coordinates of asset %s",
                        asset_id,
                    )
                    exif_address = {}

                fallback_text: str | None = None
                if isinstance(exif_address, dict):
                    fallback_value = exif_address.get("fallback")
                    if isinstance(fallback_value, str):
                        fallback_text = fallback_value

                formatted_exif, dedupe_values, has_osm_components = (
                    self._format_exif_address_caption(
                        exif_address if isinstance(exif_address, dict) else None,
                        exif_lat,
                        exif_lon,
                    )
                )

                for value in dedupe_values:
                    existing_lower.add(value)

                if formatted_exif and formatted_exif not in caption_lines:
                    caption_lines.append(formatted_exif)

                if fallback_text and not has_osm_components and fallback_text not in caption_lines:
                    caption_lines.append(fallback_text)
            if location_parts:
                location_line = ", ".join(location_parts)
                if confidence_display:
                    location_line += f" (уверенность: {confidence_display})"
                caption_lines.append("Локация: " + location_line)
            elif confidence_display:
                caption_lines.append(f"Уверенность в локации: {confidence_display}")
            if photo_weather_display:
                caption_lines.append(f"Обстановка: {photo_weather_display}")
            caption_lines.append(f"На улице: {'да' if is_outdoor else 'нет'}")
            caption_lines.append(f"Архитектура: {'да' if arch_view else 'нет'}")
            season_caption_display = season_final_display or "неизвестно"
            weather_caption_display = photo_weather_display or "неизвестно"
            capture_display_value = capture_time_display or "неизвестно"
            caption_lines.append(f"Время съёмки: {capture_display_value}")
            caption_lines.append(f"Погода: {weather_caption_display}")
            caption_lines.append(f"Сезон: {season_caption_display}")
            if arch_style and arch_style.get("label"):
                confidence_value = arch_style.get("confidence")
                style_line = f"Стиль: {arch_style['label']}"
                confidence_note: str
                if isinstance(confidence_value, (int, float)) and math.isfinite(confidence_value):
                    confidence_float = float(confidence_value)
                    confidence_pct = int(round(confidence_float * 100))
                    confidence_pct = max(0, min(100, confidence_pct))
                    if confidence_float >= 0.4:
                        confidence_note = f"(≈{confidence_pct}%)"
                    else:
                        confidence_note = f"(низкая уверенность ≈{confidence_pct}%)"
                else:
                    confidence_note = "(уверенность неизвестна)"
                caption_lines.append(f"{style_line} {confidence_note}".strip())
            if landmarks:
                caption_lines.append("Ориентиры: " + ", ".join(landmarks))
            if flower_varieties:
                caption_lines.append("Цветы: " + ", ".join(flower_varieties))
            flower_set = set(flower_varieties)
            remaining_objects = [obj for obj in objects if obj not in flower_set]
            if remaining_objects:
                caption_lines.append("Объекты: " + ", ".join(remaining_objects))
            if tags:
                caption_lines.append("Теги: " + ", ".join(tags))
            if sea_wave_score_data:
                wave_val = sea_wave_score_data["value"]
                wave_conf = sea_wave_score_data["confidence"]
                caption_lines.append(f"Волнение моря: {wave_val}/10 (conf={wave_conf:.2f})")
            if nsfw_flag:
                caption_lines.append(
                    "⚠️ Чувствительный контент: " + (safety_reason or "потенциально NSFW")
                )

            attribution_line = "Адрес: OSM/Nominatim"
            if attribution_line not in caption_lines:
                caption_lines.append(attribution_line)

            caption_text = "\n".join(line for line in caption_lines if line)
            caption_entities: list[dict[str, Any]] | None = None
            if caption_text:
                caption_entities = [
                    {
                        "type": "expandable_blockquote",
                        "offset": 0,
                        "length": _utf16_length(caption_text),
                    }
                ]
            location_log_parts: list[str] = []
            if guess_city:
                location_log_parts.append(guess_city)
            if guess_country and (not guess_city or guess_country.lower() != guess_city.lower()):
                location_log_parts.append(guess_country)
            location_log = ", ".join(location_log_parts) or "-"
            confidence_log = (
                f"{location_confidence:.3f}"
                if location_confidence is not None and math.isfinite(location_confidence)
                else "-"
            )
            request_id = response.request_id if response else None
            logging.info(
                "VISION_RESULT asset=%s model=%s request_id=%s description=%s location=%s confidence=%s caption_len=%s weather=%s season=%s style=%s",
                asset_id,
                "gpt-4o-mini",
                request_id or "-",
                caption,
                location_log,
                confidence_log,
                len(caption_text),
                photo_weather or "-",
                season_final or "-",
                arch_style["label"] if arch_style and arch_style.get("label") else "-",
            )
            result_payload = {
                "status": "ok",
                "provider": "gpt-4o-mini",
                "arch_view": arch_view,
                "caption": caption,
                "objects": objects,
                "is_outdoor": is_outdoor,
                "guess_country": guess_country,
                "guess_city": guess_city,
                "location_confidence": location_confidence,
                "landmarks": landmarks,
                "tags": tags,
                "framing": framing,
                "architecture_close_up": architecture_close_up,
                "architecture_wide": architecture_wide,
                "weather_image": weather_image,
                "season_guess": season_guess,
                "season_final": season_final,
                "season_final_display": season_final_display,
                "arch_style": arch_style,
                "safety": {"nsfw": nsfw_flag, "reason": safety_reason},
                "category": category,
                "photo_weather": photo_weather,
                "photo_weather_display": photo_weather_display,
                "weather_final": photo_weather,
                "weather_final_display": photo_weather_display,
                "flower_varieties": flower_varieties,
                "is_sea": is_sea,
                "photo_sky": photo_sky_result,
                "sky_visible": sky_visible,
                "is_sunset": is_sunset,
            }
            if rubric_id is not None:
                result_payload["rubric_id"] = rubric_id
            if sea_wave_score_data:
                result_payload["sea_wave_score"] = sea_wave_score_data
            logging.info(
                "Vision job %s classified asset %s: scene=%s, arch=%s, tags=%s, weather_tag=%s",
                job.id,
                asset_id,
                caption,
                arch_view,
                ", ".join(tags) if tags else "-",
                photo_weather or "-",
            )
            await self._record_openai_usage(
                "gpt-4o-mini",
                response,
                job=job,
                record_supabase=False,
            )
            logging.info(
                "OpenAI request_id=%s usage in/out/total=%s/%s/%s",
                request_id or "-",
                response.prompt_tokens if response and response.prompt_tokens is not None else "-",
                (
                    response.completion_tokens
                    if response and response.completion_tokens is not None
                    else "-"
                ),
                response.total_tokens if response and response.total_tokens is not None else "-",
            )
            delete_original_after_post = False
            if asset.kind == "photo":
                method_used = "copyMessage"
                copy_payload: dict[str, Any] = {
                    "chat_id": asset.channel_id,
                    "from_chat_id": asset.channel_id,
                    "message_id": asset.message_id,
                    "caption": caption_text or None,
                }
                if caption_entities:
                    copy_payload["caption_entities"] = caption_entities
                resp = await self.api_request("copyMessage", copy_payload)
                if resp.get("ok"):
                    delete_original_after_post = True
                else:
                    logging.error(
                        "Vision job %s failed to copy message for asset %s: %s",
                        job.id,
                        asset_id,
                        resp,
                    )
                    fallback_method = "sendPhoto" if asset.kind == "photo" else "sendDocument"
                    file_field = "photo" if fallback_method == "sendPhoto" else "document"
                    fallback_payload: dict[str, Any] = {
                        "chat_id": asset.channel_id,
                        file_field: file_id,
                        "caption": caption_text or None,
                    }
                    if caption_entities:
                        fallback_payload["caption_entities"] = caption_entities
                    resp = await self.api_request(fallback_method, fallback_payload)
                    method_used = fallback_method
                    if resp.get("ok"):
                        delete_original_after_post = fallback_method == "sendPhoto"
                    else:
                        logging.error(
                            "Vision job %s failed to publish result for asset %s via %s: %s",
                            job.id,
                            asset_id,
                            fallback_method,
                            resp,
                        )
                        raise RuntimeError(f"Failed to publish vision result: {resp}")
            elif self._is_convertible_image_document(asset):
                if not local_path or not os.path.exists(local_path):
                    if not self.dry_run:
                        target_path = self._build_local_file_path(asset_id, file_meta)
                        downloaded_path = await self._download_file(file_id, target_path)
                        if downloaded_path:
                            local_path = str(downloaded_path)
                            cleanup_paths.append(local_path)
                if not local_path or not os.path.exists(local_path):
                    raise RuntimeError("Unable to load asset for conversion")

                resp: dict[str, Any] | None = None
                publish_mode = "original"
                log_rss("before_sendPhoto")
                try:
                    resp, publish_mode = await self._publish_as_photo(
                        asset.channel_id,
                        local_path,
                        caption_text or None,
                        caption_entities=caption_entities,
                    )
                finally:
                    log_rss("after_sendPhoto")
                    gc.collect()
                method_used = "sendPhoto"
                delete_original_after_post = True
                logging.info(
                    "Vision job %s published document asset %s via sendPhoto (%s)",
                    job.id,
                    asset_id,
                    publish_mode,
                )
                if resp and resp.get("ok"):
                    result_payload = resp.get("result")
                    photo_sizes = None
                    if isinstance(result_payload, dict):
                        photo_sizes = result_payload.get("photo")
                    photo_meta = self._extract_photo_file_meta(photo_sizes)
                    if photo_meta and photo_meta.get("file_id"):
                        self.data.update_asset(
                            asset_id,
                            kind="photo",
                            file_meta=photo_meta,
                            metadata={"original_document_file_id": file_id},
                        )
                        new_file_id = photo_meta.get("file_id")
                        asset.payload["kind"] = "photo"
                        if new_file_id is not None:
                            asset.payload["file_id"] = new_file_id
                        file_unique = photo_meta.get("file_unique_id")
                        if file_unique is not None:
                            asset.payload["file_unique_id"] = file_unique
                        mime_type = photo_meta.get("mime_type")
                        if mime_type is not None:
                            asset.payload["mime_type"] = mime_type
                        file_size = photo_meta.get("file_size")
                        if file_size is not None:
                            asset.payload["file_size"] = file_size
                        width_value = photo_meta.get("width")
                        height_value = photo_meta.get("height")
                        asset.width = Asset._to_int(width_value)
                        asset.height = Asset._to_int(height_value)
                    else:
                        logging.warning(
                            "Vision job %s missing photo metadata in response for asset %s: %s",
                            job.id,
                            asset_id,
                            resp,
                        )
                if not resp or not resp.get("ok"):
                    logging.error(
                        "Vision job %s failed to publish converted photo for asset %s: %s",
                        job.id,
                        asset_id,
                        resp,
                    )
                    fallback_doc_payload: dict[str, Any] = {
                        "chat_id": asset.channel_id,
                        "document": file_id,
                        "caption": caption_text or None,
                    }
                    if caption_entities:
                        fallback_doc_payload["caption_entities"] = caption_entities
                    resp = await self.api_request("sendDocument", fallback_doc_payload)
                    method_used = "sendDocument"
                    delete_original_after_post = False
                    if not resp.get("ok"):
                        logging.error(
                            "Vision job %s failed to publish result for asset %s via sendDocument: %s",
                            job.id,
                            asset_id,
                            resp,
                        )
                        raise RuntimeError(f"Failed to publish vision result: {resp}")
            else:
                document_payload: dict[str, Any] = {
                    "chat_id": asset.channel_id,
                    "document": file_id,
                    "caption": caption_text or None,
                }
                if caption_entities:
                    document_payload["caption_entities"] = caption_entities
                resp = await self.api_request("sendDocument", document_payload)
                method_used = "sendDocument"
                if not resp.get("ok"):
                    logging.error(
                        "Vision job %s failed to publish result for asset %s via sendDocument: %s",
                        job.id,
                        asset_id,
                        resp,
                    )
                    raise RuntimeError(f"Failed to publish vision result: {resp}")
            new_mid = resp.get("result", {}).get("message_id") if resp.get("result") else None
            logging.info(
                "Vision job %s posted classification for asset %s via %s: message_id=%s",
                job.id,
                asset_id,
                method_used,
                new_mid,
            )
            weather_display_log = (
                weather_final_display or photo_weather_display or photo_weather or "-"
            )
            weather_source_log: str | None
            if metadata_weather:
                weather_source_log = "metadata"
            elif model_weather:
                weather_source_log = "model"
            elif fallback_weather:
                weather_source_log = "fallback"
            else:
                weather_source_log = None
            if weather_source_log:
                weather_display_log = f"{weather_display_log} ({weather_source_log})"
            season_display_log = season_final_display or season_final or "-"
            arch_style_label = arch_style.get("label") if isinstance(arch_style, dict) else None
            arch_style_confidence = (
                arch_style.get("confidence") if isinstance(arch_style, dict) else None
            )
            arch_confidence_log = (
                f"{float(arch_style_confidence):.3f}"
                if isinstance(arch_style_confidence, (int, float))
                else "-"
            )
            logging.info(
                "VISION: framing=%s weather=%s season=%s arch_style=%s arch_confidence=%s",
                framing,
                weather_display_log,
                season_display_log,
                arch_style_label or "-",
                arch_confidence_log,
            )
            asset_update_kwargs = {
                "recognized_message_id": new_mid,
                "vision_results": result_payload,
                "vision_category": category,
                "vision_arch_view": "yes" if arch_view else "",
                "vision_photo_weather": photo_weather,
                "vision_confidence": location_confidence,
                "vision_flower_varieties": flower_varieties,
                "vision_caption": caption_text,
                "local_path": None,
            }
            if rubric_id is not None:
                asset_update_kwargs["rubric_id"] = rubric_id
            self.data.update_asset(asset_id, **asset_update_kwargs)
            if tags and any(t in {"sunset", "закат", "golden hour"} for t in tags):
                try:
                    self.data.update_asset_categories_merge(asset_id, ["закат"])
                except Exception:
                    logging.exception("Failed to add закат category to asset %s", asset_id)
            if ASSETS_DEBUG_EXIF and not self.dry_run and new_mid:
                try:
                    debug_path: str | None = (
                        local_path if local_path and os.path.exists(local_path) else None
                    )
                    if not debug_path:
                        target_path = self._build_local_file_path(asset_id, file_meta)
                        downloaded_path = await self._download_file(file_id, target_path)
                        if downloaded_path:
                            debug_path = str(downloaded_path)
                            cleanup_paths.append(debug_path)
                    if debug_path and os.path.exists(debug_path):
                        exif_payload = self._extract_exif_full(debug_path)
                        exif_json = json.dumps(exif_payload, ensure_ascii=False, indent=2)
                        message_text = (
                            f"EXIF (raw)\n```json\n{exif_json}\n```" if exif_json else "EXIF (raw)"
                        )
                        exif_bytes = exif_json.encode("utf-8") if exif_json else b""
                        if len(message_text) <= 3500:
                            await self.api_request(
                                "sendMessage",
                                {
                                    "chat_id": asset.channel_id,
                                    "text": message_text,
                                    "reply_to_message_id": new_mid,
                                },
                            )
                        else:
                            buffer = io.BytesIO(exif_bytes)
                            await self.api_request(
                                "sendDocument",
                                {
                                    "chat_id": asset.channel_id,
                                    "caption": "EXIF (raw)",
                                    "reply_to_message_id": new_mid,
                                },
                                files={"document": ("exif.json", buffer.getvalue())},
                            )
                except Exception:
                    logging.exception("Failed to publish EXIF debug for asset %s", asset_id)
            if delete_original_after_post and not self.dry_run and new_mid and asset.message_id:
                logging.info(
                    "Vision job %s deleting original document message %s for asset %s",
                    job.id,
                    asset.message_id,
                    asset_id,
                )
                delete_resp = await self.api_request(
                    "deleteMessage",
                    {"chat_id": asset.channel_id, "message_id": asset.message_id},
                )
                if not delete_resp.get("ok"):
                    logging.error(
                        "Vision job %s failed to delete original message %s for asset %s: %s",
                        job.id,
                        asset.message_id,
                        asset_id,
                        delete_resp,
                    )
        finally:
            for path in cleanup_paths:
                self._remove_file(path)
            duration = (datetime.utcnow() - start_time).total_seconds()
            logging.info(
                "Vision job %s for asset %s completed in %.2fs",
                job.id,
                asset_id,
                duration,
            )

    def next_asset(self, tags: set[str] | None) -> Any:
        logging.info("Selecting asset for tags=%s", tags)
        asset = self.data.get_next_asset(tags)
        if asset:
            logging.info("Picked asset %s", asset.message_id)
        else:
            logging.info("No asset available")
        if not asset:
            return None
        return {
            "id": asset.id,
            "channel_id": asset.channel_id,
            "message_id": asset.message_id,
            "template": asset.caption_template,
            "hashtags": asset.hashtags,
            "categories": asset.categories,
            "recognized_message_id": asset.recognized_message_id,
        }

    @staticmethod
    def _is_missing_source_message_error(resp: dict[str, Any]) -> bool:
        if resp.get("ok", False):
            return False
        if resp.get("error_code") != 400:
            return False
        description = (resp.get("description") or "").lower()
        if not description:
            return False
        indicators = (
            "message to copy",
            "message to forward",
        )
        return any(
            indicator in description and "not found" in description for indicator in indicators
        )

    def _cleanup_missing_source_asset(self, asset: dict[str, Any]) -> None:
        asset_id = asset.get("id")
        channel_id = asset.get("channel_id")
        message_id = asset.get("message_id")
        logging.info(
            "Cleaning up missing source asset id=%s channel=%s message=%s",
            asset_id,
            channel_id,
            message_id,
        )
        if asset_id is not None:
            self.data.delete_assets([asset_id])

    async def publish_weather(
        self,
        channel_id: int,
        tags: set[str] | None = None,
        record: bool = True,
    ) -> bool:

        full_asset: Asset | None = None

        while True:
            asset = self.next_asset(tags)
            caption = asset["template"] if asset and asset.get("template") else ""
            if caption:
                caption = self._render_template(caption) or caption
            from_chat = None
            if asset:
                from_chat = asset.get("channel_id") or self.weather_assets_channel_id

            if asset and from_chat:

                logging.info("Copying asset %s to %s", asset["message_id"], channel_id)
                resp = await self.api_request(
                    "copyMessage",
                    {
                        "chat_id": channel_id,
                        "from_chat_id": from_chat,
                        "message_id": asset["message_id"],
                        "caption": caption or None,
                    },
                )

                ok = resp.get("ok", False)
                if ok and asset and asset.get("id") is not None:
                    asset_id = asset["id"]
                    try:
                        full_asset = self.data.get_asset(asset_id)
                    except Exception:
                        logging.exception("Failed to load asset %s for cleanup", asset_id)
                        full_asset = None
                if not ok and self._is_missing_source_message_error(resp) and asset:
                    self._cleanup_missing_source_asset(asset)
                    continue
            elif caption:
                logging.info("Sending text weather to %s", channel_id)
                resp = await self.api_request(
                    "sendMessage",
                    {"chat_id": channel_id, "text": caption},
                )
                ok = resp.get("ok", False)
            else:
                logging.info("No asset and no caption; nothing to publish")
                return False
            break

        if ok and record:
            if resp.get("result"):
                mid = resp["result"].get("message_id")
                if mid:
                    self.set_latest_weather_post(channel_id, mid)
                    await self.update_weather_buttons()
                    self.data.record_post_history(
                        channel_id,
                        mid,
                        asset["id"] if asset else None,
                        full_asset.rubric_id if full_asset else None,
                        {
                            "caption": caption,
                            "categories": asset["categories"] if asset else [],
                        },
                    )
        elif not ok:
            logging.error("Failed to publish weather: %s", resp)

        if ok and asset:
            await self._finalize_published_asset(asset, full_asset)

        return ok

    async def handle_message(self, message: Any) -> None:
        global TZ_OFFSET

        chat_id = message.get("chat", {}).get("id")
        is_weather_channel = chat_id == self.weather_assets_channel_id
        is_recognition_channel = (
            chat_id is not None
            and self.recognition_channel_id is not None
            and chat_id == self.recognition_channel_id
            and self.recognition_channel_id != self.weather_assets_channel_id
        )
        if chat_id and (is_weather_channel or is_recognition_channel):
            info = self._collect_asset_metadata(message)
            message_id = info.get("message_id", 0)
            tg_chat_id = info.get("tg_chat_id", 0)
            if (
                message_id
                and tg_chat_id
                and self.data.is_recognized_message(tg_chat_id, message_id)
            ):
                logging.info(
                    "Skipping recognized message %s in channel %s",
                    message_id,
                    tg_chat_id,
                )
                return
            origin = "recognition" if is_recognition_channel else "weather"
            asset_id = self.add_asset(
                message_id,
                info.get("hashtags", ""),
                info.get("caption"),
                channel_id=tg_chat_id,
                metadata=info.get("metadata"),
                tg_chat_id=tg_chat_id,
                kind=info.get("kind"),
                file_meta=info.get("file_meta"),
                author_user_id=info.get("author_user_id"),
                author_username=info.get("author_username"),
                sender_chat_id=info.get("sender_chat_id"),
                via_bot_id=info.get("via_bot_id"),
                forward_from_user=info.get("forward_from_user"),
                forward_from_chat=info.get("forward_from_chat"),
                origin=origin,
            )
            if asset_id and is_recognition_channel:
                self._schedule_ingest_job(asset_id, reason="new_message")
            return

        if "from" not in message:
            # ignore channel posts when asset channel is not configured
            return

        text = message.get("text", "")
        user_id = message["from"]["id"]
        username = message["from"].get("username")

        preview_state = self.pending_flowers_previews.get(user_id)
        if preview_state and preview_state.get("awaiting_instruction"):
            reply_to = message.get("reply_to_message") or {}
            prompt_id = preview_state.get("instruction_prompt_id")
            if prompt_id and reply_to.get("message_id") == prompt_id:
                instructions_text = text.strip()
                preview_state["instructions"] = instructions_text
                preview_state["awaiting_instruction"] = False
                preview_state["instruction_prompt_id"] = None
                rubric_code = preview_state.get("rubric_code")
                rubric = self.data.get_rubric_by_code(rubric_code) if rubric_code else None
                if rubric:
                    assets = list(preview_state.get("assets") or [])
                    weather_block = preview_state.get("weather_block")
                    plan_override = preview_state.get("plan")
                    if isinstance(plan_override, dict):
                        plan_override = deepcopy(plan_override)
                        plan_override["instructions"] = instructions_text
                    else:
                        plan_override = None
                    channel_hint = preview_state.get("default_channel_id")
                    if not isinstance(channel_hint, int):
                        for key in ("channel_id", "test_channel_id"):
                            value = preview_state.get(key)
                            if isinstance(value, int):
                                channel_hint = value
                                break
                    plan_meta_override = preview_state.get("plan_meta")
                    greeting, hashtags, plan, plan_meta = await self._generate_flowers_copy(
                        rubric,
                        assets,
                        channel_id=channel_hint if isinstance(channel_hint, int) else None,
                        weather_block=weather_block,
                        instructions=instructions_text,
                        plan=plan_override,
                        plan_meta=(
                            plan_meta_override if isinstance(plan_meta_override, dict) else None
                        ),
                    )
                    if isinstance(plan, dict):
                        plan["instructions"] = instructions_text
                    (
                        preview_caption,
                        publish_caption,
                        publish_parse_mode,
                        prepared_hashtags,
                    ) = self._build_flowers_caption(
                        greeting,
                        list(preview_state.get("cities") or []),
                        hashtags,
                        weather_block,
                    )
                    preview_state["plan"] = plan
                    preview_state["plan_meta"] = plan_meta or {}
                    preview_state["pattern_ids"] = list((plan_meta or {}).get("pattern_ids", []))
                    prompt_payload = self._build_flowers_prompt_payload(plan, plan_meta)
                    preview_state["serialized_plan"] = str(
                        prompt_payload.get("serialized_plan") or "{}"
                    )
                    plan_system_prompt = str(prompt_payload.get("system_prompt") or "")
                    plan_user_prompt = str(prompt_payload.get("user_prompt") or "")
                    plan_request_text = str(prompt_payload.get("request_text") or "")
                    preview_state["plan_system_prompt"] = plan_system_prompt
                    preview_state["plan_user_prompt"] = plan_user_prompt
                    preview_state["plan_request_text"] = plan_request_text
                    preview_state["plan_prompt"] = plan_user_prompt
                    preview_state["plan_prompt_length"] = prompt_payload.get("prompt_length")
                    preview_state["plan_prompt_fallback"] = bool(
                        prompt_payload.get("used_fallback")
                    )
                    await self._update_flowers_preview_caption_state(
                        preview_state,
                        preview_caption=preview_caption,
                        publish_caption=publish_caption,
                        publish_parse_mode=publish_parse_mode,
                        greeting=greeting,
                        hashtags=hashtags,
                        prepared_hashtags=prepared_hashtags,
                    )
                    await self.api_request(
                        "sendMessage",
                        {
                            "chat_id": user_id,
                            "text": "Инструкция сохранена, подпись обновлена.",
                        },
                    )
                else:
                    await self.api_request(
                        "sendMessage",
                        {
                            "chat_id": user_id,
                            "text": "Не удалось обновить подпись: рубрика недоступна.",
                        },
                    )
                return

        if user_id in self.pending and self.pending[user_id].get("rubric_input"):
            if not self.is_superadmin(user_id):
                del self.pending[user_id]
            else:
                await self.api_request(
                    "sendMessage",
                    {
                        "chat_id": user_id,
                        "text": "Используйте кнопки для настройки рубрики.",
                    },
                )
            return

        command = text.split(maxsplit=1)[0] if text else ""
        if command == "/privet" or command.startswith("/privet@"):
            await self.api_request(
                "sendMessage",
                {
                    "chat_id": user_id,
                    "text": "Привет",
                },
            )
            return

        if text.startswith("/help"):
            help_messages = [
                (
                    "*Быстрый старт*\n\n"
                    "- `/start` — запросить доступ или подтвердить, что бот уже активирован.\n"
                    "- `/help` — показать эту памятку.\n"
                    "- `/pending` — очередь заявок с кнопками `Approve`/`Reject`.\n"
                    "- `/list_users` — список администраторов и операторов.\n"
                    "- `/tz` — смена личного часового пояса (бот попросит выбрать значение).\n"
                ),
                (
                    "*Рубрики*\n"
                    "- `/rubrics` — карточки `flowers` и `guess_arch` со всеми кнопками управления.\n"
                    "  • `Включить/Выключить` меняет статус.\n"
                    "  • `Канал` и `Тест-канал` открывают кнопочный список каналов.\n"
                    "  • `Добавить расписание` запускает пошаговый мастер (время → дни → канал → сохранение).\n"
                    "  • `▶️ Запустить` и `🧪 Тест` просят подтверждение перед постановкой задачи в очередь.\n"
                ),
                (
                    "*Мобильные устройства*\n"
                    "- `/mobile` — карточка с QR-кодом для подключения тестовых устройств без ручного ввода.\n"
                    "- `/mobile_stats` — сводка загрузок через мобильный канал за сегодня, 7 и 30 дней с топами устройств.\n"
                    "- `/pair <название>` — создать или повторно показать активный код привязки с выбранной меткой устройства.\n"
                    "- `/attach <название>` — сокращённый алиас `/pair` для ручной выдачи нового кода подключения.\n"
                ),
                (
                    "*Управление пользователями*\n"
                    "- `/add_user <id>` — добавить пользователя в операторы (только для супер-админа).\n"
                    "- `/remove_user <id>` — удалить оператора и отозвать доступ.\n"
                    "- `/approve <id>` — утвердить заявку из `/pending` и выдать доступ.\n"
                    "- `/reject <id>` — отклонить заявку и уведомить пользователя.\n"
                ),
                (
                    "*Каналы и погода*\n"
                    "- `/channels` — список подключённых каналов.\n"
                    "- `/set_weather_assets_channel` и `/set_recognition_channel` открывают кнопочный выбор канала.\n"
                    "- `/set_assets_channel` — единый выбор канала для ассетов и распознавания (требует подтверждения).\n"
                    "- `/setup_weather` и `/list_weather_channels` показывают расписания с кнопками `Run now`/`Stop`.\n"
                    "- `/weather`, `/history`, `/scheduled` — статусные отчёты, где управлять публикациями можно прямо из inline-кнопок.\n"
                    "- `/amber` — кнопочное управление каналом Янтарный."
                ),
                (
                    "*Кнопки и посты*\n"
                    "- `/addbutton <post_url> <text> <url>` — добавить кастомную inline-кнопку к любому посту.\n"
                    "- `/delbutton <post_url>` — убрать все кнопки и очистить сохранённые настройки.\n"
                    "- `/addweatherbutton <post_url> <text> [url]` — добавить погодную кнопку с шаблонами и ссылкой на прогноз.\n"
                ),
                (
                    "*Города и море*\n"
                    "- `/addcity <name> <lat> <lon>` — добавить город в погодную базу (координаты с точностью до десятых и выше).\n"
                    "- `/cities` — показать список городов с кнопками удаления.\n"
                    "- `/addsea <name> <lat> <lon>` — добавить морскую точку для температуры воды.\n"
                    "- `/seas` — показать все морские точки и удалить ненужные.\n"
                    "- `/dump_sea` — экспортировать CSV со всеми морскими ассетами, метаданными волн/неба и vision_json (только для супер-админов).\n"
                    "- `/backfill_waves [dry-run]` — заполнить волны/небо из vision_results (используйте dry-run для проверки без изменений).\n"
                    "- `/inv_sea` — остатки фото «Море» по небу и волне.\n"
                    "- `/sea_audit` — проверка и очистка «мёртвых душ» в базе.\n"
                    "- `/audit_assets` — запустить аудит всех таблиц ассетов, проверить наличие файлов в Telegram и удалить несуществующие записи; вывести отчёт с количеством удалённых записей по рубрикам.\n"
                ),
                (
                    "*Погодные рассылки*\n"
                    "- `/weatherposts` — список зарегистрированных погодных постов; `update` обновит заголовки и кнопки сразу.\n"
                    "- `/regweather <post_url> <template>` — привязать пост к шаблону погоды для автоматических обновлений.\n"
                ),
            ]
            if not self.is_authorized(user_id):
                help_messages.insert(
                    0,
                    (
                        "*Доступ по приглашению*\n"
                        "- Первый администратор отправляет `/start` и получает статус супер-админа.\n"
                        "- Остальные пользователи вызывают `/start`, попадают в очередь и ждут утверждения через `/pending`."
                    ),
                )
            help_messages.append(
                "Подробная документация: файл `README.md` → раздел *Operator Interface* и журнал изменений `CHANGELOG.md`."
            )
            for chunk in help_messages:
                await self.api_request(
                    "sendMessage",
                    {
                        "chat_id": user_id,
                        "text": chunk,
                        "parse_mode": "Markdown",
                    },
                )
            return

        # first /start registers superadmin or puts user in queue
        if text.startswith("/start"):
            if self.get_user(user_id):
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Bot is working"}
                )
                return

            if self.is_rejected(user_id):
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Access denied by administrator"}
                )
                return

            if self.is_pending(user_id):
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Awaiting approval"}
                )
                return

            cur = self.db.execute("SELECT COUNT(*) FROM users")
            user_count = cur.fetchone()[0]
            if user_count == 0:
                self.db.execute(
                    "INSERT INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, ?)",
                    (user_id, username, TZ_OFFSET),
                )
                self.db.commit()
                logging.info("Registered %s as superadmin", user_id)
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "You are superadmin"}
                )
                return

            if self.pending_count() >= 10:
                await self.api_request(
                    "sendMessage",
                    {"chat_id": user_id, "text": "Registration queue full, try later"},
                )
                logging.info("Registration rejected for %s due to full queue", user_id)
                return

            self.db.execute(
                "INSERT OR IGNORE INTO pending_users (user_id, username, requested_at) VALUES (?, ?, ?)",
                (user_id, username, datetime.utcnow().isoformat()),
            )
            self.db.commit()
            logging.info("User %s added to pending queue", user_id)
            await self.api_request(
                "sendMessage", {"chat_id": user_id, "text": "Registration pending approval"}
            )
            return

        if text.startswith("/pair") or text.startswith("/attach"):
            if not self.is_authorized(user_id):
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Not authorized"}
                )
                return
            parts = text.split(maxsplit=1)
            requested_name = parts[1].strip() if len(parts) > 1 else ""
            existing_token = self._get_active_pairing_token(user_id)
            if existing_token and not requested_name:
                code, expires_at, _ = existing_token
                await self._send_pairing_code_message(
                    user_id,
                    code,
                    expires_at,
                    existing=True,
                )
                return
            device_label = requested_name or (
                existing_token[2] if existing_token else _PAIRING_DEFAULT_NAME
            )
            try:
                code, expires_at = self._issue_pairing_token(user_id, device_label)
            except RuntimeError:
                logging.error("Failed to generate pairing code for user %s", user_id)
                await self.api_request(
                    "sendMessage",
                    {
                        "chat_id": user_id,
                        "text": "Не удалось сгенерировать код, попробуйте снова.",
                    },
                )
                return
            await self._send_pairing_code_message(
                user_id,
                code,
                expires_at,
                existing=False,
            )
            return

        if text.startswith("/mobile_stats"):
            if not self.is_authorized(user_id):
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Not authorized"}
                )
                return
            stats = self.data.get_mobile_upload_stats()
            lines = [
                "📱 Mobile uploads",
                f"Total: {stats.total}",
                f"Today: {stats.today}",
                f"7d: {stats.seven_days}",
                f"30d: {stats.thirty_days}",
            ]
            if stats.top_devices:
                lines.append("")
                lines.append("Top devices:")
                for device in stats.top_devices:
                    label = device.name or device.device_id
                    lines.append(f"• {label} — {device.count}")
            await self.api_request(
                "sendMessage",
                {
                    "chat_id": user_id,
                    "text": "\n".join(lines),
                },
            )
            return

        if text.startswith("/mobile"):
            if not self.is_authorized(user_id):
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Not authorized"}
                )
                return
            existing_token = self._get_active_pairing_token(user_id)
            if existing_token:
                code, expires_at, _ = existing_token
            else:
                try:
                    code, expires_at = self._issue_pairing_token(user_id, _PAIRING_DEFAULT_NAME)
                except RuntimeError:
                    logging.error("Failed to generate pairing code for user %s", user_id)
                    await self.api_request(
                        "sendMessage",
                        {
                            "chat_id": user_id,
                            "text": "Не удалось сгенерировать код, попробуйте снова.",
                        },
                    )
                    return
            devices = list_user_devices(self.db, user_id=user_id)
            logging.info(
                "MOBILE_PAIR_UI",
                extra={
                    "user_id": user_id,
                    "has_devices": bool(devices),
                    "code_len": len(code),
                },
            )
            await self._send_mobile_pairing_card(user_id, code, expires_at, devices)
            return

        if text.startswith("/dump_sea") and self.is_superadmin(user_id):
            try:
                from datetime import datetime as dt

                csv_content = self.data.dump_sea_assets_csv()

                timestamp = dt.utcnow().strftime("%Y%m%d_%H%M%S")
                filename = f"sea_assets_{timestamp}.csv"

                csv_bytes = csv_content.encode("utf-8")

                logging.info(
                    "Dumping sea assets CSV for user %s: %d bytes, %d rows",
                    user_id,
                    len(csv_bytes),
                    csv_content.count("\n"),
                )

                await self.api_request(
                    "sendDocument",
                    {
                        "chat_id": user_id,
                        "caption": f"Sea assets dump generated at {timestamp} UTC",
                    },
                    files={"document": (filename, csv_bytes)},
                )
            except Exception as e:
                logging.exception("Failed to generate sea assets CSV dump")
                await self.api_request(
                    "sendMessage",
                    {"chat_id": user_id, "text": f"❌ Error generating CSV: {e}"},
                )
            return

        if text.startswith("/backfill_waves"):
            if not self.is_authorized(user_id):
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Not authorized"}
                )
                return

            parts = text.split()
            dry_run = len(parts) > 1 and parts[1].lower() in ("dry-run", "dry_run", "dryrun")

            await self.api_request(
                "sendMessage",
                {
                    "chat_id": user_id,
                    "text": f"🔄 Starting wave backfill {'(DRY RUN)' if dry_run else ''}...",
                },
            )

            try:
                stats = await self.backfill_waves(dry_run=dry_run)

                if "already_running" in stats and stats["already_running"] > 0:
                    message = "⚠️ Backfill already running, skipped"
                else:
                    mode_label = "DRY RUN" if dry_run else "COMPLETED"
                    message = (
                        f"✅ Backfill waves {mode_label}:\n"
                        f"Updated: {stats['updated']}\n"
                        f"Skipped: {stats['skipped']}\n"
                        f"Errors: {stats['errors']}"
                    )

                await self.api_request(
                    "sendMessage",
                    {"chat_id": user_id, "text": message},
                )
            except Exception as e:
                logging.exception("Failed to run wave backfill")
                await self.api_request(
                    "sendMessage",
                    {"chat_id": user_id, "text": f"❌ Backfill error: {e}"},
                )
            return

        if text.startswith("/inv_sea"):
            if not self.is_authorized(user_id):
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Not authorized"}
                )
                return

            await self._send_sea_inventory_report(is_prod=False, initiator_id=user_id)
            await self.api_request("sendMessage", {"chat_id": user_id, "text": "✓ Отчёт отправлен"})
            return

        if text.startswith("/sea_audit"):
            if not self.is_authorized(user_id):
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Not authorized"}
                )
                return

            await self.api_request(
                "sendMessage",
                {"chat_id": user_id, "text": "🔍 Начинаю аудит..."},
            )

            BATCH_SIZE = 50
            checked = 0
            deleted = 0
            kept = 0

            logging.info("SEA_AUDIT_STARTED")

            # Fetch all sea asset records in batches
            sea_assets = self.db.execute(
                """
                SELECT a.id, a.payload_json, a.tg_message_id
                FROM assets a
                WHERE LOWER(json_extract(a.payload_json, '$.vision_category')) = 'sea'
                """
            ).fetchall()

            for i in range(0, len(sea_assets), BATCH_SIZE):
                batch = sea_assets[i : i + BATCH_SIZE]

                for row in batch:
                    asset_id = row["id"]
                    payload_json = row["payload_json"]
                    tg_message_id = row["tg_message_id"]

                    # Parse tg_chat_id and message_id
                    chat_id = None
                    msg_id = None

                    if payload_json:
                        try:
                            payload = json.loads(payload_json)
                            chat_id = payload.get("tg_chat_id")
                            msg_id = payload.get("message_id")
                        except json.JSONDecodeError:
                            pass

                    if (chat_id is None or msg_id is None) and tg_message_id:
                        if ":" in str(tg_message_id):
                            parts = str(tg_message_id).split(":", 1)
                            try:
                                chat_id = int(parts[0])
                                msg_id = int(parts[1])
                            except ValueError:
                                pass
                        else:
                            try:
                                msg_id = int(tg_message_id)
                            except ValueError:
                                pass

                    if not chat_id or not msg_id:
                        # Skip assets without valid TG message info
                        kept += 1
                        checked += 1
                        continue

                    checked += 1

                    # Check if message exists via copyMessage (safe check)
                    try:
                        # Copy message to operator chat (non-destructive)
                        copy_result = await self.api_request(
                            "copyMessage",
                            {
                                "from_chat_id": chat_id,
                                "message_id": msg_id,
                                "chat_id": user_id,
                                "disable_notification": True,
                            },
                        )
                        # If successful, delete the copy immediately (was only for verification)
                        if copy_result.get("ok") and copy_result.get("result"):
                            copy_msg_id = copy_result["result"].get("message_id")
                            if copy_msg_id:
                                await self.api_request(
                                    "deleteMessage",
                                    {
                                        "chat_id": user_id,
                                        "message_id": copy_msg_id,
                                    },
                                )
                        kept += 1

                    except Exception as e:
                        error_str = str(e).lower()
                        # If message not found (400), mark as dead soul and remove from DB
                        if (
                            "message to copy not found" in error_str
                            or "message not found" in error_str
                            or "message can't be copied" in error_str
                        ):
                            logging.warning(
                                "SEA_AUDIT_DEAD_SOUL asset_id=%s chat_id=%s msg_id=%s",
                                asset_id,
                                chat_id,
                                msg_id,
                            )
                            self.db.execute("DELETE FROM assets WHERE id=?", (asset_id,))
                            self.db.commit()
                            deleted += 1
                        else:
                            # If other error, keep record (err on side of caution)
                            logging.warning(
                                "SEA_AUDIT_CHECK_ERROR asset_id=%s err=%s", asset_id, str(e)[:200]
                            )
                            kept += 1

            logging.info(
                "SEA_AUDIT_FINISHED checked=%d deleted=%d kept=%d",
                checked,
                deleted,
                kept,
            )

            report = (
                f"✓ Аудит завершён\n\n"
                f"Проверено: {checked}\n"
                f"Удалено (мёртвые): {deleted}\n"
                f"Оставлено: {kept}"
            )
            await self.api_request(
                "sendMessage",
                {"chat_id": user_id, "text": report},
            )
            return

        if text.startswith("/audit_assets"):
            if not self.is_authorized(user_id):
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Not authorized"}
                )
                return

            await self.api_request(
                "sendMessage",
                {"chat_id": user_id, "text": "🔍 Начинаю аудит всех ассетов..."},
            )

            BATCH_SIZE = 50
            BATCH_DELAY_MS = 100
            total_checked = 0
            total_removed = 0

            logging.info("ASSETS_AUDIT_STARTED")

            # Get rubric mapping for better reporting
            rubric_map: dict[int, str] = {}
            rubric_checked: dict[str, int] = {}
            rubric_removed: dict[str, int] = {}
            try:
                rubrics_rows = self.db.execute("SELECT id, code FROM rubrics").fetchall()
                for row in rubrics_rows:
                    rubric_map[row["id"]] = row["code"]
                    rubric_checked[row["code"]] = 0
                    rubric_removed[row["code"]] = 0
                rubric_checked["unassigned"] = 0
                rubric_removed["unassigned"] = 0
            except Exception as e:
                logging.warning("ASSETS_AUDIT_RUBRIC_MAP_ERROR err=%s", str(e)[:200])

            # Fetch all assets
            try:
                all_assets = self.db.execute(
                    """
                    SELECT id, tg_message_id, payload_json
                    FROM assets
                    """
                ).fetchall()
            except Exception as e:
                logging.error("ASSETS_AUDIT_FETCH_ERROR err=%s", str(e)[:200])
                await self.api_request(
                    "sendMessage",
                    {
                        "chat_id": user_id,
                        "text": f"❌ Ошибка при получении ассетов: {str(e)[:100]}",
                    },
                )
                return

            # Process in batches
            for batch_idx in range(0, len(all_assets), BATCH_SIZE):
                batch = all_assets[batch_idx : batch_idx + BATCH_SIZE]

                for row in batch:
                    asset_id = row["id"]
                    tg_message_id = row["tg_message_id"] if "tg_message_id" in row.keys() else None
                    payload_json = row["payload_json"] if "payload_json" in row.keys() else None

                    # Parse tg_message_id to extract chat_id and message_id
                    chat_id = None
                    msg_id = None
                    if tg_message_id:
                        parts = str(tg_message_id).split(":", 1)
                        if len(parts) == 2:
                            try:
                                chat_id = int(parts[0])
                                msg_id = int(parts[1])
                            except (ValueError, TypeError):
                                pass

                    # Parse rubric_id from payload_json
                    rubric_id = None
                    if payload_json:
                        try:
                            payload = json.loads(payload_json)
                            rubric_id = payload.get("rubric_id")
                        except (json.JSONDecodeError, TypeError):
                            pass

                    # Determine rubric name for reporting
                    rubric_name = (
                        rubric_map.get(rubric_id, "unassigned") if rubric_id else "unassigned"
                    )

                    if not chat_id or not msg_id:
                        # Skip assets without valid TG message info
                        total_checked += 1
                        if rubric_name in rubric_checked:
                            rubric_checked[rubric_name] += 1
                        continue

                    total_checked += 1
                    if rubric_name in rubric_checked:
                        rubric_checked[rubric_name] += 1

                    # Check if message exists via copyMessage (safe, non-destructive)
                    logging.info(
                        "ASSETS_AUDIT_CHECKING asset_id=%s chat_id=%s msg_id=%s rubric=%s",
                        asset_id,
                        chat_id,
                        msg_id,
                        rubric_name,
                    )
                    try:
                        copy_result = await self.api_request(
                            "copyMessage",
                            {
                                "from_chat_id": chat_id,
                                "message_id": msg_id,
                                "chat_id": user_id,
                                "disable_notification": True,
                            },
                        )
                        # If successful, delete the copy (was only for verification)
                        logging.info(
                            "ASSETS_AUDIT_EXISTS asset_id=%s chat_id=%s msg_id=%s rubric=%s",
                            asset_id,
                            chat_id,
                            msg_id,
                            rubric_name,
                        )
                        if copy_result.get("ok") and copy_result.get("result"):
                            copy_msg_id = copy_result["result"].get("message_id")
                            if copy_msg_id:
                                try:
                                    await self.api_request(
                                        "deleteMessage",
                                        {
                                            "chat_id": user_id,
                                            "message_id": copy_msg_id,
                                        },
                                    )
                                except Exception as e:
                                    logging.warning(
                                        "ASSETS_AUDIT_COPY_DELETE_FAILED asset_id=%s err=%s",
                                        asset_id,
                                        str(e)[:100],
                                    )

                    except Exception as e:
                        error_str = str(e).lower()
                        logging.info(
                            "ASSETS_AUDIT_COPY_FAILED asset_id=%s chat_id=%s msg_id=%s rubric=%s error=%s",
                            asset_id,
                            chat_id,
                            msg_id,
                            rubric_name,
                            str(e)[:200],
                        )
                        # If message not found (400), treat as dead soul
                        # Check for various message deletion/not found errors
                        is_dead_soul = (
                            "message to copy not found" in error_str
                            or "message not found" in error_str
                            or "message can't be copied" in error_str
                            or "message_id_invalid" in error_str
                            or "message to get not found" in error_str
                            or "message to forward not found" in error_str
                            or "message identifier is not specified" in error_str
                            or "chat not found" in error_str
                        )
                        if is_dead_soul:
                            logging.warning(
                                "ASSETS_AUDIT_DEAD_SOUL asset_id=%s chat_id=%s msg_id=%s rubric=%s",
                                asset_id,
                                chat_id,
                                msg_id,
                                rubric_name,
                            )

                            # Delete record from DB
                            try:
                                self.db.execute("DELETE FROM assets WHERE id=?", (asset_id,))
                                self.db.commit()
                                total_removed += 1
                                if rubric_name in rubric_removed:
                                    rubric_removed[rubric_name] += 1
                                logging.info(
                                    "ASSETS_AUDIT_DELETED asset_id=%s rubric=%s total_removed=%d",
                                    asset_id,
                                    rubric_name,
                                    total_removed,
                                )
                            except Exception as delete_error:
                                logging.error(
                                    "ASSETS_AUDIT_DB_DELETE_FAILED asset_id=%s err=%s",
                                    asset_id,
                                    str(delete_error)[:200],
                                )
                        else:
                            # Other TG errors: log and continue (don't fail audit)
                            logging.warning(
                                "ASSETS_AUDIT_TG_ERROR asset_id=%s err=%s",
                                asset_id,
                                str(e)[:200],
                            )

                # Delay between batches to avoid rate limits
                if batch_idx + BATCH_SIZE < len(all_assets):
                    await asyncio.sleep(BATCH_DELAY_MS / 1000.0)

            logging.info(
                "ASSETS_AUDIT_FINISHED checked=%d removed=%d",
                total_checked,
                total_removed,
            )

            # Build detailed report
            report_lines = [
                "🔎 Аудит ассетов завершён\n",
                f"Проверено: {total_checked}",
                f"Удалено «мёртвых душ»: {total_removed}\n",
            ]

            # Add per-rubric breakdown if we have data
            if rubric_checked:
                report_lines.append("По рубрикам:")
                for rubric_name in sorted(rubric_checked.keys()):
                    checked_count = rubric_checked.get(rubric_name, 0)
                    removed_count = rubric_removed.get(rubric_name, 0)
                    if checked_count > 0:
                        report_lines.append(
                            f"  • {rubric_name}: проверено {checked_count}, удалено {removed_count}"
                        )

            report = "\n".join(report_lines)
            await self.api_request(
                "sendMessage",
                {"chat_id": user_id, "text": report},
            )
            return

        if text.startswith("/add_user") and self.is_superadmin(user_id):
            parts = text.split()
            if len(parts) == 2:
                uid = int(parts[1])
                if not self.get_user(uid):
                    self.db.execute("INSERT INTO users (user_id) VALUES (?)", (uid,))
                    self.db.commit()
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": f"User {uid} added"}
                )
            return

        if text.startswith("/remove_user") and self.is_superadmin(user_id):
            parts = text.split()
            if len(parts) == 2:
                uid = int(parts[1])
                self.db.execute("DELETE FROM users WHERE user_id=?", (uid,))
                self.db.commit()
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": f"User {uid} removed"}
                )
            return

        if text.startswith("/tz"):
            parts = text.split()
            if not self.is_authorized(user_id):
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Not authorized"}
                )
                return
            if len(parts) != 2:
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Usage: /tz +02:00"}
                )
                return
            try:
                self.parse_offset(parts[1])
            except Exception:
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Invalid offset"}
                )
                return
            self.db.execute("UPDATE users SET tz_offset=? WHERE user_id=?", (parts[1], user_id))
            self.db.commit()
            TZ_OFFSET = parts[1]
            await self.api_request(
                "sendMessage", {"chat_id": user_id, "text": f"Timezone set to {parts[1]}"}
            )
            return

        if text.startswith("/list_users") and self.is_superadmin(user_id):
            cur = self.db.execute("SELECT user_id, username, is_superadmin FROM users")
            rows = cur.fetchall()
            msg = "\n".join(
                f"{self.format_user(r['user_id'], r['username'])} {'(admin)' if r['is_superadmin'] else ''}"
                for r in rows
            )
            await self.api_request(
                "sendMessage",
                {"chat_id": user_id, "text": msg or "No users", "parse_mode": "Markdown"},
            )
            return

        if text.startswith("/pending") and self.is_superadmin(user_id):
            cur = self.db.execute("SELECT user_id, username, requested_at FROM pending_users")
            rows = cur.fetchall()
            if not rows:
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "No pending users"}
                )
                return

            msg = "\n".join(
                f"{self.format_user(r['user_id'], r['username'])} requested {r['requested_at']}"
                for r in rows
            )
            keyboard = {
                "inline_keyboard": [
                    [
                        {"text": "Approve", "callback_data": f'approve:{r["user_id"]}'},
                        {"text": "Reject", "callback_data": f'reject:{r["user_id"]}'},
                    ]
                    for r in rows
                ]
            }
            await self.api_request(
                "sendMessage",
                {
                    "chat_id": user_id,
                    "text": msg,
                    "parse_mode": "Markdown",
                    "reply_markup": keyboard,
                },
            )
            return

        if text.startswith("/rubrics") and self.is_superadmin(user_id):
            self.rubric_dashboards.pop(user_id, None)
            self.rubric_overview_messages.pop(user_id, None)
            await self._send_rubric_dashboard(user_id)
            return

        if text.startswith("/approve") and self.is_superadmin(user_id):
            parts = text.split()
            if len(parts) == 2:
                uid = int(parts[1])
                if self.approve_user(uid):
                    cur = self.db.execute("SELECT username FROM users WHERE user_id=?", (uid,))
                    row = cur.fetchone()
                    uname = row["username"] if row else None
                    await self.api_request(
                        "sendMessage",
                        {
                            "chat_id": user_id,
                            "text": f"{self.format_user(uid, uname)} approved",
                            "parse_mode": "Markdown",
                        },
                    )
                    await self.api_request(
                        "sendMessage", {"chat_id": uid, "text": "You are approved"}
                    )
                else:
                    await self.api_request(
                        "sendMessage", {"chat_id": user_id, "text": "User not in pending list"}
                    )
            return

        if text.startswith("/reject") and self.is_superadmin(user_id):
            parts = text.split()
            if len(parts) == 2:
                uid = int(parts[1])
                if self.reject_user(uid):
                    cur = self.db.execute(
                        "SELECT username FROM rejected_users WHERE user_id=?", (uid,)
                    )
                    row = cur.fetchone()
                    uname = row["username"] if row else None
                    await self.api_request(
                        "sendMessage",
                        {
                            "chat_id": user_id,
                            "text": f"{self.format_user(uid, uname)} rejected",
                            "parse_mode": "Markdown",
                        },
                    )
                    await self.api_request(
                        "sendMessage", {"chat_id": uid, "text": "Your registration was rejected"}
                    )
                else:
                    await self.api_request(
                        "sendMessage", {"chat_id": user_id, "text": "User not in pending list"}
                    )
            return

        if text.startswith("/channels") and self.is_superadmin(user_id):
            cur = self.db.execute("SELECT chat_id, title FROM channels")
            rows = cur.fetchall()
            msg = "\n".join(f"{r['title']} ({r['chat_id']})" for r in rows)
            await self.api_request(
                "sendMessage", {"chat_id": user_id, "text": msg or "No channels"}
            )
            return

        if text.startswith("/history"):
            cur = self.db.execute(
                "SELECT target_chat_id, sent_at FROM schedule WHERE sent=1 ORDER BY sent_at DESC LIMIT 10"
            )
            rows = cur.fetchall()
            offset = self.get_tz_offset(user_id)
            msg = "\n".join(
                f"{r['target_chat_id']} at {self.format_time(r['sent_at'], offset)}" for r in rows
            )
            await self.api_request("sendMessage", {"chat_id": user_id, "text": msg or "No history"})
            return

        if text.startswith("/scheduled") and self.is_authorized(user_id):
            rows = self.list_scheduled()
            if not rows:
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "No scheduled posts"}
                )
                return
            offset = self.get_tz_offset(user_id)
            for r in rows:
                ok = False
                try:
                    resp = await self.api_request(
                        "forwardMessage",
                        {
                            "chat_id": user_id,
                            "from_chat_id": r["from_chat_id"],
                            "message_id": r["message_id"],
                        },
                    )
                    ok = resp.get("ok", False)
                    if (
                        not ok
                        and resp.get("error_code") == 400
                        and "not" in resp.get("description", "").lower()
                    ):
                        resp = await self.api_request(
                            "copyMessage",
                            {
                                "chat_id": user_id,
                                "from_chat_id": r["from_chat_id"],
                                "message_id": r["message_id"],
                            },
                        )
                        ok = resp.get("ok", False)
                except Exception:
                    logging.exception("Failed to forward message %s", r["id"])
                if not ok:
                    link = None
                    if str(r["from_chat_id"]).startswith("-100"):
                        cid = str(r["from_chat_id"])[4:]
                        link = f'https://t.me/c/{cid}/{r["message_id"]}'
                    await self.api_request(
                        "sendMessage",
                        {
                            "chat_id": user_id,
                            "text": link or f'Message {r["message_id"]} from {r["from_chat_id"]}',
                        },
                    )
                keyboard = {
                    "inline_keyboard": [
                        [
                            {"text": "Cancel", "callback_data": f'cancel:{r["id"]}'},
                            {"text": "Reschedule", "callback_data": f'resch:{r["id"]}'},
                        ]
                    ]
                }
                target = (
                    f"{r['target_title']} ({r['target_chat_id']})"
                    if r["target_title"]
                    else str(r["target_chat_id"])
                )
                await self.api_request(
                    "sendMessage",
                    {
                        "chat_id": user_id,
                        "text": f"{r['id']}: {target} at {self.format_time(r['publish_time'], offset)}",
                        "reply_markup": keyboard,
                    },
                )
            return

        if text.startswith("/addbutton"):
            if not self.is_authorized(user_id):
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Not authorized"}
                )
                return

            parts = text.split()
            if len(parts) < 4:
                await self.api_request(
                    "sendMessage",
                    {"chat_id": user_id, "text": "Usage: /addbutton <post_url> <text> <url>"},
                )
                return
            parsed = await self.parse_post_url(parts[1])
            if not parsed:
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Invalid post URL"}
                )
                return
            chat_id, msg_id = parsed
            keyboard_text = " ".join(parts[2:-1])
            fwd = await self.api_request(
                "forwardMessage",
                {
                    "chat_id": user_id,
                    "from_chat_id": chat_id,
                    "message_id": msg_id,
                },
            )

            markup = None
            caption = None
            caption_entities = None
            if fwd.get("ok") and fwd.get("result"):
                message = fwd["result"]
                markup = message.get("reply_markup")
                caption = message.get("caption")
                caption_entities = message.get("caption_entities")
                await self.api_request(
                    "deleteMessage",
                    {"chat_id": user_id, "message_id": message.get("message_id")},
                )
            key = (chat_id, msg_id)
            info = self.manual_buttons.get(key)
            if info is None:
                base_buttons = markup.get("inline_keyboard", []) if markup else []
                info = {
                    "base": [[dict(btn) for btn in row] for row in base_buttons],
                    "custom": [],
                }
            new_row = [{"text": keyboard_text, "url": parts[-1]}]
            info["custom"].append([dict(btn) for btn in new_row])
            self.manual_buttons[key] = info

            keyboard = {
                "inline_keyboard": [
                    [dict(btn) for btn in row] for row in info["base"] + info["custom"]
                ]
            }

            payload = {
                "chat_id": chat_id,
                "message_id": msg_id,
                "reply_markup": keyboard,
            }
            method = "editMessageReplyMarkup"
            if caption is not None:
                method = "editMessageCaption"
                payload["caption"] = caption
                if caption_entities:
                    payload["caption_entities"] = caption_entities

            resp = await self.api_request(method, payload)

            if (
                not resp.get("ok")
                and resp.get("error_code") == 400
                and "message is not modified" in resp.get("description", "")
            ):
                resp["ok"] = True
            if resp.get("ok"):
                logging.info("Updated message %s with button", msg_id)
                cur = self.db.execute(
                    "SELECT 1 FROM weather_posts WHERE chat_id=? AND message_id=?",
                    (chat_id, msg_id),
                )
                if cur.fetchone():
                    self.db.execute(
                        "UPDATE weather_posts SET reply_markup=? WHERE chat_id=? AND message_id=?",
                        (json.dumps(keyboard), chat_id, msg_id),
                    )
                    self.db.commit()

                await self.api_request("sendMessage", {"chat_id": user_id, "text": "Button added"})
            else:
                logging.error("Failed to add button to %s: %s", msg_id, resp)
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Failed to add button"}
                )
            return

        if text.startswith("/delbutton"):
            if not self.is_authorized(user_id):
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Not authorized"}
                )
                return

            parts = text.split()
            if len(parts) != 2:
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Usage: /delbutton <post_url>"}
                )
                return
            parsed = await self.parse_post_url(parts[1])
            if not parsed:
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Invalid post URL"}
                )
                return
            chat_id, msg_id = parsed

            resp = await self.api_request(
                "editMessageReplyMarkup",
                {
                    "chat_id": chat_id,
                    "message_id": msg_id,
                    "reply_markup": {},
                },
            )

            if (
                not resp.get("ok")
                and resp.get("error_code") == 400
                and "message is not modified" in resp.get("description", "")
            ):
                resp["ok"] = True
            if resp.get("ok"):

                logging.info("Removed buttons from message %s", msg_id)
                self.db.execute(
                    "DELETE FROM weather_link_posts WHERE chat_id=? AND message_id=?",
                    (chat_id, msg_id),
                )
                self.db.execute(
                    "UPDATE weather_posts SET reply_markup=NULL WHERE chat_id=? AND message_id=?",
                    (chat_id, msg_id),
                )
                self.db.commit()
                self.manual_buttons.pop((chat_id, msg_id), None)
            else:
                logging.error("Failed to remove button from %s: %s", msg_id, resp)
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Failed to remove button"}
                )
            return

        if text.startswith("/addweatherbutton") and self.is_superadmin(user_id):
            parts = text.split()
            if len(parts) < 3:
                await self.api_request(
                    "sendMessage",
                    {
                        "chat_id": user_id,
                        "text": "Usage: /addweatherbutton <post_url> <text> [url]",
                    },
                )
                return

            url = None
            if len(parts) > 3 and parts[-1].startswith(("http://", "https://")):
                url = parts[-1]
                btn_text = " ".join(parts[2:-1])
            else:
                btn_text = " ".join(parts[2:])
                url = self.latest_weather_url()
                if not url:
                    await self.api_request(
                        "sendMessage",
                        {"chat_id": user_id, "text": "Specify forecast URL after text"},
                    )
                    return

            parsed = await self.parse_post_url(parts[1])
            if not parsed:
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Invalid post URL"}
                )
                return
            chat_id, msg_id = parsed
            fwd = await self.api_request(
                "copyMessage",
                {"chat_id": user_id, "from_chat_id": chat_id, "message_id": msg_id},
            )
            markup = None
            if not fwd.get("ok"):
                fwd = await self.api_request(
                    "forwardMessage",
                    {"chat_id": user_id, "from_chat_id": chat_id, "message_id": msg_id},
                )
            if fwd.get("ok") and fwd.get("result"):
                markup = fwd["result"].get("reply_markup")
                await self.api_request(
                    "deleteMessage",
                    {
                        "chat_id": user_id,
                        "message_id": fwd["result"].get("message_id"),
                    },
                )

            row = self.db.execute(
                "SELECT base_markup, button_texts FROM weather_link_posts WHERE chat_id=? AND message_id=?",
                (chat_id, msg_id),
            ).fetchone()
            base_markup = row["base_markup"] if row else json.dumps(markup) if markup else None
            texts = json.loads(row["button_texts"]) if row else []
            if row is None:
                base_buttons = markup.get("inline_keyboard", []) if markup else []
            else:
                base_buttons = json.loads(base_markup)["inline_keyboard"] if base_markup else []
            texts.append(btn_text)

            rendered_texts = [self._render_template(t) or t for t in texts]
            weather_buttons = [{"text": t, "url": url} for t in rendered_texts]
            keyboard_buttons = base_buttons + [weather_buttons]

            resp = await self.api_request(
                "editMessageReplyMarkup",
                {
                    "chat_id": chat_id,
                    "message_id": msg_id,
                    "reply_markup": {"inline_keyboard": keyboard_buttons},
                },
            )
            if resp.get("ok"):
                self.db.execute(
                    "INSERT OR REPLACE INTO weather_link_posts (chat_id, message_id, base_markup, button_texts) VALUES (?, ?, ?, ?)",
                    (chat_id, msg_id, base_markup, json.dumps(texts)),
                )
                self.db.execute(
                    "UPDATE weather_posts SET reply_markup=? WHERE chat_id=? AND message_id=?",
                    (json.dumps({"inline_keyboard": keyboard_buttons}), chat_id, msg_id),
                )
                self.db.commit()
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Weather button added"}
                )
            else:
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Failed to add weather button"}
                )
            return

        if text.startswith("/addcity") and self.is_superadmin(user_id):
            parts = text.split(maxsplit=2)
            if len(parts) == 3:
                name = parts[1]
                coords = self._parse_coords(parts[2])
                if not coords:
                    await self.api_request(
                        "sendMessage", {"chat_id": user_id, "text": "Invalid coordinates"}
                    )
                    return
                lat, lon = coords
                try:
                    self.db.execute(
                        "INSERT INTO cities (name, lat, lon) VALUES (?, ?, ?)", (name, lat, lon)
                    )
                    self.db.commit()
                    await self.api_request(
                        "sendMessage", {"chat_id": user_id, "text": f"City {name} added"}
                    )
                except sqlite3.IntegrityError:
                    await self.api_request(
                        "sendMessage", {"chat_id": user_id, "text": "City already exists"}
                    )
            else:
                await self.api_request(
                    "sendMessage",
                    {"chat_id": user_id, "text": "Usage: /addcity <name> <lat> <lon>"},
                )
            return

        if text.startswith("/addsea") and self.is_superadmin(user_id):

            parts = text.split(maxsplit=2)
            if len(parts) == 3:
                name = parts[1]
                coords = self._parse_coords(parts[2])
                if not coords:
                    await self.api_request(
                        "sendMessage", {"chat_id": user_id, "text": "Invalid coordinates"}
                    )
                    return
                lat, lon = coords

                try:
                    self.db.execute(
                        "INSERT INTO seas (name, lat, lon) VALUES (?, ?, ?)", (name, lat, lon)
                    )
                    self.db.commit()
                    await self.api_request(
                        "sendMessage", {"chat_id": user_id, "text": f"Sea {name} added"}
                    )
                except sqlite3.IntegrityError:
                    await self.api_request(
                        "sendMessage", {"chat_id": user_id, "text": "Sea already exists"}
                    )
            else:
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Usage: /addsea <name> <lat> <lon>"}
                )
            return

        if text.startswith("/cities") and self.is_superadmin(user_id):
            cur = self.db.execute("SELECT id, name, lat, lon FROM cities ORDER BY id")
            rows = cur.fetchall()
            if not rows:
                await self.api_request("sendMessage", {"chat_id": user_id, "text": "No cities"})
                return
            for r in rows:
                keyboard = {
                    "inline_keyboard": [
                        [{"text": "Delete", "callback_data": f'city_del:{r["id"]}'}]
                    ]
                }
                await self.api_request(
                    "sendMessage",
                    {
                        "chat_id": user_id,
                        "text": f"{r['id']}: {r['name']} ({r['lat']:.6f}, {r['lon']:.6f})",
                        "reply_markup": keyboard,
                    },
                )
            return

        if text.startswith("/seas") and self.is_superadmin(user_id):
            cur = self.db.execute("SELECT id, name, lat, lon FROM seas ORDER BY id")
            rows = cur.fetchall()
            if not rows:
                await self.api_request("sendMessage", {"chat_id": user_id, "text": "No seas"})
                return
            for r in rows:
                keyboard = {
                    "inline_keyboard": [[{"text": "Delete", "callback_data": f'sea_del:{r["id"]}'}]]
                }
                await self.api_request(
                    "sendMessage",
                    {
                        "chat_id": user_id,
                        "text": f"{r['id']}: {r['name']} ({r['lat']:.6f}, {r['lon']:.6f})",
                        "reply_markup": keyboard,
                    },
                )
            return

        if text.startswith("/amber") and self.is_superadmin(user_id):
            sea_id = self.get_amber_sea()
            if sea_id is None:
                cur = self.db.execute("SELECT id, name FROM seas ORDER BY id")
                rows = cur.fetchall()
                if not rows:
                    await self.api_request("sendMessage", {"chat_id": user_id, "text": "No seas"})
                    return
                keyboard = {
                    "inline_keyboard": [
                        [{"text": r["name"], "callback_data": f'amber_sea:{r["id"]}'}] for r in rows
                    ]
                }
                self.pending[user_id] = {"amber_sea": True}
                await self.api_request(
                    "sendMessage",
                    {"chat_id": user_id, "text": "Select sea", "reply_markup": keyboard},
                )
            else:
                await self.show_amber_channels(user_id)
            return

        if text.startswith("/weatherposts") and self.is_superadmin(user_id):
            parts = text.split(maxsplit=1)
            force = len(parts) > 1 and parts[1] == "update"
            if force:
                await self.update_weather_posts()
                await self.update_weather_buttons()
            cur = self.db.execute(
                "SELECT chat_id, message_id, template FROM weather_posts ORDER BY id"
            )
            post_rows = cur.fetchall()
            for r in post_rows:
                header = self._render_template(r["template"])
                url = self.post_url(r["chat_id"], r["message_id"])
                text = f"{url} {header}" if header else f"{url} no data"
                keyboard = {
                    "inline_keyboard": [
                        [
                            {
                                "text": "Stop weather",
                                "callback_data": f'wpost_del:{r["chat_id"]}:{r["message_id"]}',
                            }
                        ]
                    ]
                }
                await self.api_request(
                    "sendMessage",
                    {"chat_id": user_id, "text": text, "reply_markup": keyboard},
                )
            cur = self.db.execute(
                "SELECT chat_id, message_id, button_texts FROM weather_link_posts ORDER BY rowid"
            )
            rows = cur.fetchall()
            if not rows and not post_rows:
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "No weather posts"}
                )
                return
            for r in rows:

                rendered = [self._render_template(t) or t for t in json.loads(r["button_texts"])]
                texts = ", ".join(rendered)

                keyboard = {
                    "inline_keyboard": [
                        [
                            {
                                "text": "Remove buttons",
                                "callback_data": f'wbtn_del:{r["chat_id"]}:{r["message_id"]}',
                            }
                        ]
                    ]
                }
                await self.api_request(
                    "sendMessage",
                    {
                        "chat_id": user_id,
                        "text": f"{self.post_url(r['chat_id'], r['message_id'])} buttons: {texts}",
                        "reply_markup": keyboard,
                    },
                )
            return

        if text.startswith("/setup_weather") and self.is_superadmin(user_id):
            cur = self.db.execute("SELECT chat_id, title FROM channels")
            rows = cur.fetchall()
            existing = {r["channel_id"] for r in self.list_weather_channels()}
            options = [r for r in rows if r["chat_id"] not in existing]
            if not options:
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "No channels available"}
                )
                return
            keyboard = {
                "inline_keyboard": [
                    [{"text": r["title"], "callback_data": f'ws_ch:{r["chat_id"]}'}]
                    for r in options
                ]
            }
            self.pending[user_id] = {"setup_weather": True}
            await self.api_request(
                "sendMessage",
                {"chat_id": user_id, "text": "Select channel", "reply_markup": keyboard},
            )
            return

        if text.startswith("/list_weather_channels") and self.is_superadmin(user_id):
            rows = self.list_weather_channels()
            if not rows:
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "No weather channels"}
                )
                return
            for r in rows:

                last = r["last_published_at"]
                if last:
                    last = self.format_time(last, self.get_tz_offset(user_id))
                else:
                    last = "never"
                keyboard = {
                    "inline_keyboard": [
                        [
                            {"text": "Run now", "callback_data": f'wrnow:{r["channel_id"]}'},
                            {"text": "Stop", "callback_data": f'wstop:{r["channel_id"]}'},
                        ]
                    ]
                }
                await self.api_request(
                    "sendMessage",
                    {
                        "chat_id": user_id,
                        "text": f"{r['title'] or r['channel_id']} at {r['post_time']} last {last}",
                        "reply_markup": keyboard,
                    },
                )

            return

        if text.startswith("/set_assets_channel") and self.is_superadmin(user_id):
            parts = text.split(maxsplit=1)
            confirmed = len(parts) > 1 and parts[1].strip().lower() == "confirm"
            if not confirmed:
                await self.api_request(
                    "sendMessage",
                    {
                        "chat_id": user_id,
                        "text": (
                            "Команда `/set_assets_channel` устанавливает один и тот же канал для "
                            "хранилища погоды и распознавания. Используйте её только если это "
                            "действительно необходимо и подтвердите действие командой "
                            "`/set_assets_channel confirm`. Для раздельных складов вызовите "
                            "по очереди `/set_weather_assets_channel` и `/set_recognition_channel`."
                        ),
                        "parse_mode": "Markdown",
                    },
                )
                return
            await self._prompt_channel_selection(
                user_id,
                pending_key="set_assets",
                callback_prefix="asset_ch",
                prompt="Select asset channel",
            )
            return

        if text.startswith("/set_weather_assets_channel") and self.is_superadmin(user_id):
            await self._prompt_channel_selection(
                user_id,
                pending_key="set_weather_assets",
                callback_prefix="weather_ch",
                prompt="Select weather assets channel",
            )
            return

        if text.startswith("/set_recognition_channel") and self.is_superadmin(user_id):
            await self._prompt_channel_selection(
                user_id,
                pending_key="set_recognition",
                callback_prefix="recognition_ch",
                prompt="Select recognition channel",
            )
            return

        if text.startswith("/weather") and self.is_superadmin(user_id):

            parts = text.split(maxsplit=1)
            if len(parts) > 1 and parts[1].lower() == "now":
                await self.collect_weather(force=True)
                await self.collect_sea(force=True)

            cur = self.db.execute("SELECT id, name FROM cities ORDER BY id")
            rows = cur.fetchall()
            if not rows:
                await self.api_request("sendMessage", {"chat_id": user_id, "text": "No cities"})
                return
            lines = []
            for r in rows:
                w = self.db.execute(
                    "SELECT temperature, weather_code, wind_speed, is_day, timestamp FROM weather_cache_hour WHERE city_id=? ORDER BY timestamp DESC LIMIT 1",
                    (r["id"],),
                ).fetchone()
                if w:
                    emoji = weather_emoji(w["weather_code"], w["is_day"])
                    lines.append(
                        f"{r['name']}: {w['temperature']:.1f}°C {emoji} wind {w['wind_speed']:.1f} m/s at {w['timestamp']}"
                    )
                else:
                    lines.append(f"{r['name']}: no data")

            cur = self.db.execute("SELECT id, name FROM seas ORDER BY id")
            sea_rows = cur.fetchall()
            for r in sea_rows:
                row = self._get_sea_cache(r["id"])
                if row and row["current"] is not None:
                    emoji = "\U0001f30a"
                    lines.append(
                        f"{r['name']}: {emoji} {row['current']:.1f}°C {row['morning']:.1f}/{row['day']:.1f}/{row['evening']:.1f}/{row['night']:.1f}"
                    )
                else:
                    lines.append(f"{r['name']}: no data")
            await self.api_request("sendMessage", {"chat_id": user_id, "text": "\n".join(lines)})
            return

        if text.startswith("/regweather") and self.is_superadmin(user_id):
            parts = text.split(maxsplit=2)
            if len(parts) < 3:
                await self.api_request(
                    "sendMessage",
                    {"chat_id": user_id, "text": "Usage: /regweather <post_url> <template>"},
                )
                return
            parsed = await self.parse_post_url(parts[1])
            if not parsed:
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Invalid post URL"}
                )
                return
            template = parts[2]
            chat_id, msg_id = parsed
            resp = await self.api_request(
                "copyMessage",
                {
                    "chat_id": user_id,
                    "from_chat_id": chat_id,
                    "message_id": msg_id,
                },
            )
            if not resp.get("ok") or not resp.get("result"):
                resp = await self.api_request(
                    "forwardMessage",
                    {
                        "chat_id": user_id,
                        "from_chat_id": chat_id,
                        "message_id": msg_id,
                    },
                )
            elif not resp["result"].get("text") and not resp["result"].get("caption"):
                await self.api_request(
                    "deleteMessage",
                    {"chat_id": user_id, "message_id": resp["result"]["message_id"]},
                )
                resp = await self.api_request(
                    "forwardMessage",
                    {
                        "chat_id": user_id,
                        "from_chat_id": chat_id,
                        "message_id": msg_id,
                    },
                )
            if not resp.get("ok"):
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Cannot read post"}
                )
                return

            base_text = resp["result"].get("text")
            base_caption = resp["result"].get("caption")
            base_text = self.strip_header(base_text)
            base_caption = self.strip_header(base_caption)
            markup = resp["result"].get("reply_markup")

            if base_text is None and base_caption is None:
                base_text = ""
            await self.api_request(
                "deleteMessage", {"chat_id": user_id, "message_id": resp["result"]["message_id"]}
            )
            self.db.execute(
                "INSERT OR REPLACE INTO weather_posts (chat_id, message_id, template, base_text, base_caption, reply_markup) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    chat_id,
                    msg_id,
                    template,
                    base_text,
                    base_caption,
                    json.dumps(markup) if markup else None,
                ),
            )
            self.db.commit()
            # Ensure data is available for the placeholders right away
            # so the post gets updated immediately after registration.
            await self.collect_weather(force=True)
            await self.collect_sea(force=True)
            await self.update_weather_posts(
                {int(m.group(1)) for m in re.finditer(r"{(\d+)\|", template)}
            )
            await self.api_request(
                "sendMessage", {"chat_id": user_id, "text": "Weather post registered"}
            )
            return

        # handle time input for scheduling
        if user_id in self.pending and "await_time" in self.pending[user_id]:
            time_str = text.strip()
            try:
                if len(time_str.split()) == 1:
                    dt = datetime.strptime(time_str, "%H:%M")
                    pub_time = datetime.combine(date.today(), dt.time())
                else:
                    pub_time = datetime.strptime(time_str, "%d.%m.%Y %H:%M")
            except ValueError:
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Invalid time format"}
                )
                return
            offset = self.get_tz_offset(user_id)
            pub_time_utc = pub_time - self.parse_offset(offset)
            if pub_time_utc <= datetime.utcnow():
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Time must be in future"}
                )
                return
            data = self.pending.pop(user_id)
            if "reschedule_id" in data:
                self.update_schedule_time(data["reschedule_id"], pub_time_utc.isoformat())
                await self.api_request(
                    "sendMessage",
                    {
                        "chat_id": user_id,
                        "text": f"Rescheduled for {self.format_time(pub_time_utc.isoformat(), offset)}",
                    },
                )
            else:
                test = await self.api_request(
                    "forwardMessage",
                    {
                        "chat_id": user_id,
                        "from_chat_id": data["from_chat_id"],
                        "message_id": data["message_id"],
                    },
                )
                if not test.get("ok"):
                    await self.api_request(
                        "sendMessage",
                        {
                            "chat_id": user_id,
                            "text": f"Add the bot to channel {data['from_chat_id']} (reader role) first",
                        },
                    )
                    return
                self.add_schedule(
                    data["from_chat_id"],
                    data["message_id"],
                    data["selected"],
                    pub_time_utc.isoformat(),
                )
                await self.api_request(
                    "sendMessage",
                    {
                        "chat_id": user_id,
                        "text": f"Scheduled to {len(data['selected'])} channels for {self.format_time(pub_time_utc.isoformat(), offset)}",
                    },
                )
            return

        if user_id in self.pending and self.pending[user_id].get("weather_time"):
            time_str = text.strip()
            try:
                dt = datetime.strptime(time_str, "%H:%M")
            except ValueError:
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Invalid time format"}
                )
                return
            self.add_weather_channel(self.pending[user_id]["channel"], time_str)
            del self.pending[user_id]
            await self.api_request(
                "sendMessage", {"chat_id": user_id, "text": "Weather channel registered"}
            )
            return

        # start scheduling on forwarded message
        if "forward_from_chat" in message and self.is_authorized(user_id):
            from_chat = message["forward_from_chat"]["id"]
            msg_id = message["forward_from_message_id"]
            cur = self.db.execute("SELECT chat_id, title FROM channels")
            rows = cur.fetchall()
            if not rows:
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "No channels available"}
                )
                return
            keyboard = {
                "inline_keyboard": [
                    [{"text": r["title"], "callback_data": f'addch:{r["chat_id"]}'}] for r in rows
                ]
                + [[{"text": "Done", "callback_data": "chdone"}]]
            }
            self.pending[user_id] = {
                "from_chat_id": from_chat,
                "message_id": msg_id,
                "selected": set(),
            }
            await self.api_request(
                "sendMessage",
                {"chat_id": user_id, "text": "Select channels", "reply_markup": keyboard},
            )
            return
        else:
            if not self.is_authorized(user_id):
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Not authorized"}
                )
            else:
                await self.api_request(
                    "sendMessage",
                    {"chat_id": user_id, "text": "Please forward a post from a channel"},
                )

    async def handle_callback(self, query: Any) -> None:
        user_id = query["from"]["id"]
        data = query["data"]
        if data == "pair:new":
            if not self.is_authorized(user_id):
                await self.api_request(
                    "answerCallbackQuery",
                    {
                        "callback_query_id": query["id"],
                        "text": "Недостаточно прав",
                        "show_alert": True,
                    },
                )
                return
            message = query.get("message") or {}
            existing = self._get_active_pairing_token(user_id)
            device_label = existing[2] if existing else _PAIRING_DEFAULT_NAME
            try:
                code, expires_at = self._issue_pairing_token(user_id, device_label)
            except RuntimeError:
                logging.error("Failed to regenerate pairing code for user %s", user_id)
                await self.api_request(
                    "answerCallbackQuery",
                    {
                        "callback_query_id": query["id"],
                        "text": "Не удалось сгенерировать код",
                        "show_alert": True,
                    },
                )
                return
            devices = list_user_devices(self.db, user_id=user_id)
            await self._send_mobile_pairing_card(
                user_id,
                code,
                expires_at,
                devices,
                message=message,
                replace_photo=True,
            )
            await self.api_request(
                "answerCallbackQuery",
                {
                    "callback_query_id": query["id"],
                    "text": "Новый код создан",
                },
            )
            return
        if data.startswith("dev:revoke:"):
            if not self.is_authorized(user_id):
                await self.api_request(
                    "answerCallbackQuery",
                    {
                        "callback_query_id": query["id"],
                        "text": "Недостаточно прав",
                        "show_alert": True,
                    },
                )
                return
            device_id = data.split(":", 2)[2]
            message = query.get("message") or {}
            revoked = False
            with self.db:
                revoked = revoke_device(self.db, device_id=device_id, expected_user_id=user_id)
            if not revoked:
                await self.api_request(
                    "answerCallbackQuery",
                    {
                        "callback_query_id": query["id"],
                        "text": "Устройство не найдено",
                        "show_alert": True,
                    },
                )
                return
            existing = self._get_active_pairing_token(user_id)
            replace_photo = False
            if existing:
                code, expires_at, _ = existing
            else:
                try:
                    code, expires_at = self._issue_pairing_token(user_id, _PAIRING_DEFAULT_NAME)
                    replace_photo = True
                except RuntimeError:
                    logging.error("Failed to regenerate pairing code for user %s", user_id)
                    await self.api_request(
                        "answerCallbackQuery",
                        {
                            "callback_query_id": query["id"],
                            "text": "Не удалось обновить код",
                            "show_alert": True,
                        },
                    )
                    return
            devices = list_user_devices(self.db, user_id=user_id)
            await self._send_mobile_pairing_card(
                user_id,
                code,
                expires_at,
                devices,
                message=message,
                replace_photo=replace_photo,
            )
            await self.api_request(
                "answerCallbackQuery",
                {
                    "callback_query_id": query["id"],
                    "text": "Устройство отозвано",
                },
            )
            return
        if data.startswith("dev:rotate:"):
            if not self.is_authorized(user_id):
                await self.api_request(
                    "answerCallbackQuery",
                    {
                        "callback_query_id": query["id"],
                        "text": "Недостаточно прав",
                        "show_alert": True,
                    },
                )
                return
            device_id = data.split(":", 2)[2]
            message = query.get("message") or {}
            with self.db:
                rotated = rotate_device_secret(
                    self.db,
                    device_id=device_id,
                    expected_user_id=user_id,
                )
            if not rotated:
                await self.api_request(
                    "answerCallbackQuery",
                    {
                        "callback_query_id": query["id"],
                        "text": "Устройство не найдено",
                        "show_alert": True,
                    },
                )
                return
            secret, _, device_name = rotated
            existing = self._get_active_pairing_token(user_id)
            replace_photo = False
            if existing:
                code, expires_at, _ = existing
            else:
                try:
                    code, expires_at = self._issue_pairing_token(
                        user_id, device_name or _PAIRING_DEFAULT_NAME
                    )
                    replace_photo = True
                except RuntimeError:
                    logging.error("Failed to regenerate pairing code for user %s", user_id)
                    await self.api_request(
                        "answerCallbackQuery",
                        {
                            "callback_query_id": query["id"],
                            "text": "Не удалось обновить код",
                            "show_alert": True,
                        },
                    )
                    return
            devices = list_user_devices(self.db, user_id=user_id)
            await self._send_mobile_pairing_card(
                user_id,
                code,
                expires_at,
                devices,
                message=message,
                replace_photo=replace_photo,
            )
            await self.api_request(
                "answerCallbackQuery",
                {
                    "callback_query_id": query["id"],
                    "text": f"Новый секрет: {secret}",
                    "show_alert": True,
                },
            )
            return
        if data == "pairing_regen":
            if not self.is_authorized(user_id):
                await self.api_request(
                    "answerCallbackQuery",
                    {
                        "callback_query_id": query["id"],
                        "text": "Недостаточно прав",
                        "show_alert": True,
                    },
                )
                return
            message = query.get("message") or {}
            existing = self._get_active_pairing_token(user_id)
            device_label = existing[2] if existing else _PAIRING_DEFAULT_NAME
            try:
                code, expires_at = self._issue_pairing_token(user_id, device_label)
            except RuntimeError:
                logging.error("Failed to regenerate pairing code for user %s", user_id)
                await self.api_request(
                    "answerCallbackQuery",
                    {
                        "callback_query_id": query["id"],
                        "text": "Не удалось сгенерировать код",
                        "show_alert": True,
                    },
                )
                return
            await self._send_pairing_code_message(
                user_id,
                code,
                expires_at,
                existing=False,
                message=message,
            )
            await self.api_request(
                "answerCallbackQuery",
                {
                    "callback_query_id": query["id"],
                    "text": "Новый код создан",
                },
            )
            return
        if data.startswith("addch:") and user_id in self.pending:
            chat_id = int(data.split(":")[1])
            if "selected" in self.pending[user_id]:
                s = self.pending[user_id]["selected"]
                if chat_id in s:
                    s.remove(chat_id)
                else:
                    s.add(chat_id)
        elif data == "chdone" and user_id in self.pending:
            info = self.pending[user_id]
            if not info.get("selected"):
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "Select at least one channel"}
                )
            else:
                self.pending[user_id]["await_time"] = True
                await self.api_request(
                    "sendMessage",
                    {"chat_id": user_id, "text": "Enter time (HH:MM or DD.MM.YYYY HH:MM)"},
                )
        elif (
            data.startswith("ws_ch:")
            and user_id in self.pending
            and self.pending[user_id].get("setup_weather")
        ):
            cid = int(data.split(":")[1])
            self.pending[user_id] = {"channel": cid, "weather_time": False, "setup_weather": True}
            keyboard = {
                "inline_keyboard": [
                    [
                        {"text": "17:55", "callback_data": "ws_time:17:55"},
                        {"text": "Custom", "callback_data": "ws_custom"},
                    ]
                ]
            }
            await self.api_request(
                "sendMessage", {"chat_id": user_id, "text": "Select time", "reply_markup": keyboard}
            )
        elif (
            data == "ws_custom"
            and user_id in self.pending
            and self.pending[user_id].get("setup_weather")
        ):
            self.pending[user_id]["weather_time"] = True
            await self.api_request("sendMessage", {"chat_id": user_id, "text": "Enter time HH:MM"})
        elif (
            data.startswith("ws_time:")
            and user_id in self.pending
            and self.pending[user_id].get("setup_weather")
        ):
            time_str = data.split(":", 1)[1]
            self.add_weather_channel(self.pending[user_id]["channel"], time_str)
            del self.pending[user_id]
            await self.api_request(
                "sendMessage", {"chat_id": user_id, "text": "Weather channel registered"}
            )
        elif data.startswith("flowers_preview:"):
            if not self.is_superadmin(user_id):
                await self.api_request(
                    "answerCallbackQuery",
                    {
                        "callback_query_id": query["id"],
                        "text": "Недостаточно прав",
                        "show_alert": True,
                    },
                )
                return
            action = data.split(":", 1)[1] if ":" in data else ""
            await self._handle_flowers_preview_callback(user_id, action, query)
            return
        elif (
            data.startswith("asset_ch:")
            and user_id in self.pending
            and self.pending[user_id].get("set_assets")
        ):
            cid = int(data.split(":")[1])
            self.set_asset_channel(cid)
            del self.pending[user_id]
            await self.api_request("sendMessage", {"chat_id": user_id, "text": "Asset channel set"})
        elif (
            data.startswith("weather_ch:")
            and user_id in self.pending
            and self.pending[user_id].get("set_weather_assets")
        ):
            cid = int(data.split(":")[1])
            self.set_weather_assets_channel(cid)
            del self.pending[user_id]
            await self.api_request(
                "sendMessage", {"chat_id": user_id, "text": "Weather assets channel set"}
            )
        elif (
            data.startswith("recognition_ch:")
            and user_id in self.pending
            and self.pending[user_id].get("set_recognition")
        ):
            cid = int(data.split(":")[1])
            self.set_recognition_channel(cid)
            del self.pending[user_id]
            await self.api_request(
                "sendMessage", {"chat_id": user_id, "text": "Recognition channel set"}
            )
        elif data.startswith("wrnow:") and self.is_superadmin(user_id):
            cid = int(data.split(":")[1])

            ok = await self.publish_weather(cid, None, record=False)
            msg = "Posted" if ok else "No asset to publish"
            await self.api_request("sendMessage", {"chat_id": user_id, "text": msg})

        elif data.startswith("wstop:") and self.is_superadmin(user_id):
            cid = int(data.split(":")[1])
            self.remove_weather_channel(cid)
            await self.api_request("sendMessage", {"chat_id": user_id, "text": "Channel removed"})
        elif data.startswith("wbtn_del:") and self.is_superadmin(user_id):
            _, cid, mid = data.split(":")
            chat_id = int(cid)
            msg_id = int(mid)
            row = self.db.execute(
                "SELECT base_markup FROM weather_link_posts WHERE chat_id=? AND message_id=?",
                (chat_id, msg_id),
            ).fetchone()
            markup = json.loads(row["base_markup"]) if row and row["base_markup"] else {}
            await self.api_request(
                "editMessageReplyMarkup",
                {
                    "chat_id": chat_id,
                    "message_id": msg_id,
                    "reply_markup": markup,
                },
            )
            self.db.execute(
                "UPDATE weather_posts SET reply_markup=? WHERE chat_id=? AND message_id=?",
                (json.dumps(markup) if markup else None, chat_id, msg_id),
            )
            self.db.execute(
                "DELETE FROM weather_link_posts WHERE chat_id=? AND message_id=?",
                (chat_id, msg_id),
            )
            self.db.commit()
            await self.api_request(
                "sendMessage", {"chat_id": user_id, "text": "Weather buttons removed"}
            )
        elif data.startswith("wpost_del:") and self.is_superadmin(user_id):
            _, cid, mid = data.split(":")
            chat_id = int(cid)
            msg_id = int(mid)
            row = self.db.execute(
                "SELECT base_text, base_caption, reply_markup FROM weather_posts WHERE chat_id=? AND message_id=?",
                (chat_id, msg_id),
            ).fetchone()
            if row:
                markup = json.loads(row["reply_markup"]) if row["reply_markup"] else None
                if row["base_caption"] is not None:
                    payload = {
                        "chat_id": chat_id,
                        "message_id": msg_id,
                        "caption": row["base_caption"],
                    }
                    method = "editMessageCaption"
                else:
                    payload = {
                        "chat_id": chat_id,
                        "message_id": msg_id,
                        "text": row["base_text"] or "",
                    }
                    method = "editMessageText"
                if markup:
                    payload["reply_markup"] = markup
                await self.api_request(method, payload)
                self.db.execute(
                    "DELETE FROM weather_posts WHERE chat_id=? AND message_id=?",
                    (chat_id, msg_id),
                )
                self.db.commit()
            await self.api_request("sendMessage", {"chat_id": user_id, "text": "Weather removed"})
        elif data.startswith("approve:") and self.is_superadmin(user_id):
            uid = int(data.split(":")[1])
            if self.approve_user(uid):
                cur = self.db.execute("SELECT username FROM users WHERE user_id=?", (uid,))
                row = cur.fetchone()
                uname = row["username"] if row else None
                await self.api_request(
                    "sendMessage",
                    {
                        "chat_id": user_id,
                        "text": f"{self.format_user(uid, uname)} approved",
                        "parse_mode": "Markdown",
                    },
                )
                await self.api_request("sendMessage", {"chat_id": uid, "text": "You are approved"})
            else:
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "User not in pending list"}
                )
        elif data.startswith("reject:") and self.is_superadmin(user_id):
            uid = int(data.split(":")[1])
            if self.reject_user(uid):
                cur = self.db.execute("SELECT username FROM rejected_users WHERE user_id=?", (uid,))
                row = cur.fetchone()
                uname = row["username"] if row else None
                await self.api_request(
                    "sendMessage",
                    {
                        "chat_id": user_id,
                        "text": f"{self.format_user(uid, uname)} rejected",
                        "parse_mode": "Markdown",
                    },
                )
                await self.api_request(
                    "sendMessage", {"chat_id": uid, "text": "Your registration was rejected"}
                )
            else:
                await self.api_request(
                    "sendMessage", {"chat_id": user_id, "text": "User not in pending list"}
                )
        elif data.startswith("cancel:") and self.is_authorized(user_id):
            sid = int(data.split(":")[1])
            self.remove_schedule(sid)
            await self.api_request(
                "sendMessage", {"chat_id": user_id, "text": f"Schedule {sid} cancelled"}
            )
        elif data.startswith("resch:") and self.is_authorized(user_id):
            sid = int(data.split(":")[1])
            self.pending[user_id] = {"reschedule_id": sid, "await_time": True}
            await self.api_request("sendMessage", {"chat_id": user_id, "text": "Enter new time"})
        elif data.startswith("city_del:") and self.is_superadmin(user_id):
            cid = int(data.split(":")[1])
            self.db.execute("DELETE FROM cities WHERE id=?", (cid,))
            self.db.commit()
            await self.api_request(
                "editMessageReplyMarkup",
                {
                    "chat_id": query["message"]["chat"]["id"],
                    "message_id": query["message"]["message_id"],
                    "reply_markup": {},
                },
            )
            await self.api_request(
                "sendMessage", {"chat_id": user_id, "text": f"City {cid} deleted"}
            )
        elif data.startswith("sea_del:") and self.is_superadmin(user_id):
            sid = int(data.split(":")[1])
            self.db.execute("DELETE FROM seas WHERE id=?", (sid,))
            self.db.commit()
            await self.api_request(
                "editMessageReplyMarkup",
                {
                    "chat_id": query["message"]["chat"]["id"],
                    "message_id": query["message"]["message_id"],
                    "reply_markup": {},
                },
            )
            await self.api_request(
                "sendMessage", {"chat_id": user_id, "text": f"Sea {sid} deleted"}
            )

        elif (
            data.startswith("amber_sea:")
            and user_id in self.pending
            and self.pending[user_id].get("amber_sea")
        ):
            sid = int(data.split(":")[1])
            self.set_amber_sea(sid)
            del self.pending[user_id]
            await self.api_request("sendMessage", {"chat_id": user_id, "text": "Sea selected"})
            await self.show_amber_channels(user_id)
        elif data.startswith("amber_toggle:") and self.is_superadmin(user_id):
            cid = int(data.split(":")[1])
            if self.is_amber_channel(cid):
                self.db.execute("DELETE FROM amber_channels WHERE channel_id=?", (cid,))
                self.db.commit()
                enabled = False
            else:
                self.db.execute(
                    "INSERT OR IGNORE INTO amber_channels (channel_id) VALUES (?)", (cid,)
                )
                self.db.commit()
                enabled = True
            row = self.db.execute("SELECT title FROM channels WHERE chat_id=?", (cid,)).fetchone()
            title = row["title"] if row else str(cid)
            icon = "✅" if enabled else "❌"
            btn = "Выключить отправку" if enabled else "Включить отправку"
            keyboard = {
                "inline_keyboard": [[{"text": btn, "callback_data": f"amber_toggle:{cid}"}]]
            }
            await self.api_request(
                "editMessageText",
                {
                    "chat_id": query["message"]["chat"]["id"],
                    "message_id": query["message"]["message_id"],
                    "text": f"{title} {icon}",
                    "reply_markup": keyboard,
                },
            )
        elif data == "rubric_dashboard" and self.is_superadmin(user_id):
            self.rubric_overview_messages.pop(user_id, None)
            await self._send_rubric_dashboard(user_id, message=query.get("message"))
        elif data.startswith("rubric_overview:") and self.is_superadmin(user_id):
            code = data.split(":", 1)[1]
            await self._send_rubric_overview(user_id, code, message=query.get("message"))
        elif data.startswith("rubric_publish_confirm:") and self.is_superadmin(user_id):
            parts = data.split(":", 2)
            if len(parts) < 3:
                await self.api_request(
                    "answerCallbackQuery",
                    {
                        "callback_query_id": query["id"],
                        "text": "Некорректный запрос рубрики",
                        "show_alert": True,
                    },
                )
                return
            _, code, mode = parts
            if mode not in {"prod", "test"}:
                await self.api_request(
                    "answerCallbackQuery",
                    {
                        "callback_query_id": query["id"],
                        "text": "Неизвестный режим запуска",
                        "show_alert": True,
                    },
                )
                return
            self._set_rubric_pending_run(user_id, code, mode)
            await self._send_rubric_overview(user_id, code, message=query.get("message"))
            await self.api_request(
                "answerCallbackQuery",
                {
                    "callback_query_id": query["id"],
                    "text": "Подтвердите запуск кнопкой ниже",
                },
            )
        elif data.startswith("rubric_publish_cancel:") and self.is_superadmin(user_id):
            code = data.split(":", 1)[1]
            self._clear_rubric_pending_run(user_id, code)
            await self._send_rubric_overview(user_id, code, message=query.get("message"))
        elif data.startswith("rubric_publish_execute:") and self.is_superadmin(user_id):
            parts = data.split(":", 2)
            if len(parts) < 3:
                await self.api_request(
                    "answerCallbackQuery",
                    {
                        "callback_query_id": query["id"],
                        "text": "Некорректный запрос рубрики",
                        "show_alert": True,
                    },
                )
                return
            _, code, mode = parts
            is_test = mode == "test"
            run_label = "Тестовая" if is_test else "Рабочая"
            try:
                job_id = self.enqueue_rubric(
                    code,
                    test=is_test,
                    initiator_id=user_id,
                )
            except Exception as exc:  # noqa: PERF203 - feedback path
                logging.exception("Failed to enqueue rubric %s (test=%s)", code, is_test)
                self._clear_rubric_pending_run(user_id, code)
                await self._send_rubric_overview(user_id, code, message=query.get("message"))
                await self.api_request(
                    "answerCallbackQuery",
                    {
                        "callback_query_id": query["id"],
                        "text": "Ошибка запуска рубрики",
                        "show_alert": True,
                    },
                )
                reason = str(exc).strip() or "неизвестная ошибка"
                await self.api_request(
                    "sendMessage",
                    {
                        "chat_id": user_id,
                        "text": (
                            f"⚠️ {run_label.lower()} публикация рубрики {code} не запущена.\n"
                            f"Причина: {reason}"
                        ),
                    },
                )
            else:
                logging.info(
                    "Enqueued %s publication for rubric %s (job_id=%s, user_id=%s)",
                    "test" if is_test else "prod",
                    code,
                    job_id,
                    user_id,
                )
                self._clear_rubric_pending_run(user_id, code)
                await self._send_rubric_overview(user_id, code, message=query.get("message"))
                await self.api_request(
                    "answerCallbackQuery",
                    {
                        "callback_query_id": query["id"],
                        "text": "Задача поставлена в очередь",
                    },
                )
                await self.api_request(
                    "sendMessage",
                    {
                        "chat_id": user_id,
                        "text": (
                            f"✅ {run_label} публикация рубрики {code} поставлена в очередь"
                            f" (задача #{job_id})."
                        ),
                    },
                )
        elif data.startswith("rubric_toggle:") and self.is_superadmin(user_id):
            code = data.split(":", 1)[1]
            self._clear_rubric_pending_run(user_id, code)
            rubric = self.data.get_rubric_by_code(code)
            if rubric:
                config = self._normalize_rubric_config(rubric.config)
                was_enabled = config.get("enabled", True)
                config["enabled"] = not was_enabled
                self.data.save_rubric_config(code, config)
                if was_enabled:
                    self._delete_future_rubric_jobs(code, "rubric_disabled")
                await self._send_rubric_overview(user_id, code, message=query.get("message"))
        elif data.startswith("rubric_channel:") and self.is_superadmin(user_id):
            parts = data.split(":")
            if len(parts) >= 3:
                code = parts[1]
                self._clear_rubric_pending_run(user_id, code)
                target = parts[2]
                field = "channel_id" if target == "main" else "test_channel_id"
                self.pending[user_id] = {
                    "rubric_input": {
                        "mode": "channel_picker",
                        "code": code,
                        "field": field,
                        "message": query.get("message"),
                        "page": 0,
                        "search": "",
                        "search_mode": False,
                        "search_charset": "rus",
                        "return_mode": None,
                    }
                }
                await self._edit_rubric_input_message(user_id)
        elif data.startswith("rubric_city:") and self.is_superadmin(user_id):
            code = data.split(":", 1)[1] if ":" in data else ""
            if code:
                self._clear_rubric_pending_run(user_id, code)
                self.pending[user_id] = {
                    "rubric_input": {
                        "mode": "city_picker",
                        "code": code,
                        "message": query.get("message"),
                        "page": 0,
                        "search": "",
                        "search_mode": False,
                        "search_charset": "rus",
                    }
                }
                await self._edit_rubric_input_message(user_id)
        elif data.startswith("rubric_channel_page:") and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "channel_picker":
                try:
                    page = int(data.split(":", 1)[1])
                except ValueError:
                    page = 0
                state["page"] = max(page, 0)
                await self._edit_rubric_input_message(user_id)
        elif data.startswith("rubric_city_page:") and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "city_picker":
                try:
                    page = int(data.split(":", 1)[1])
                except ValueError:
                    page = 0
                state["page"] = max(page, 0)
                await self._edit_rubric_input_message(user_id)
        elif data == "rubric_channel_search_toggle" and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "channel_picker":
                state["search_mode"] = not state.get("search_mode", False)
                if state["search_mode"]:
                    state.setdefault("search_charset", "rus")
                await self._edit_rubric_input_message(user_id)
        elif data == "rubric_city_search_toggle" and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "city_picker":
                state["search_mode"] = not state.get("search_mode", False)
                if state["search_mode"]:
                    state.setdefault("search_charset", "rus")
                await self._edit_rubric_input_message(user_id)
        elif data.startswith("rubric_channel_search_charset:") and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "channel_picker":
                key = data.split(":", 1)[1]
                if key in CHANNEL_SEARCH_CHARSETS:
                    state["search_charset"] = key
                await self._edit_rubric_input_message(user_id)
        elif data.startswith("rubric_city_search_charset:") and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "city_picker":
                key = data.split(":", 1)[1]
                if key in CHANNEL_SEARCH_CHARSETS:
                    state["search_charset"] = key
                await self._edit_rubric_input_message(user_id)
        elif data.startswith("rubric_channel_search_add:") and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "channel_picker":
                hex_value = data.split(":", 1)[1]
                try:
                    char = bytes.fromhex(hex_value).decode("utf-8")
                except ValueError:
                    char = ""
                if char:
                    state["search"] = (state.get("search") or "") + char
                    state["page"] = 0
                await self._edit_rubric_input_message(user_id)
        elif data.startswith("rubric_city_search_add:") and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "city_picker":
                hex_value = data.split(":", 1)[1]
                try:
                    char = bytes.fromhex(hex_value).decode("utf-8")
                except ValueError:
                    char = ""
                if char:
                    state["search"] = (state.get("search") or "") + char
                    state["page"] = 0
                await self._edit_rubric_input_message(user_id)
        elif data == "rubric_channel_search_del" and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "channel_picker":
                current = state.get("search") or ""
                if current:
                    state["search"] = current[:-1]
                    state["page"] = 0
                await self._edit_rubric_input_message(user_id)
        elif data == "rubric_city_search_del" and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "city_picker":
                current = state.get("search") or ""
                if current:
                    state["search"] = current[:-1]
                    state["page"] = 0
                await self._edit_rubric_input_message(user_id)
        elif data == "rubric_channel_search_clear" and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "channel_picker":
                state["search"] = ""
                state["page"] = 0
                await self._edit_rubric_input_message(user_id)
        elif data == "rubric_city_search_clear" and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "city_picker":
                state["search"] = ""
                state["page"] = 0
                await self._edit_rubric_input_message(user_id)
        elif data == "rubric_channel_search_done" and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "channel_picker":
                state["search_mode"] = False
                state["page"] = 0
                await self._edit_rubric_input_message(user_id)
        elif data == "rubric_city_search_done" and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "city_picker":
                state["search_mode"] = False
                state["page"] = 0
                await self._edit_rubric_input_message(user_id)
        elif data.startswith("rubric_channel_set:") and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "channel_picker":
                value = data.split(":", 1)[1]
                channel_id: int | None
                try:
                    channel_id = int(value)
                except ValueError:
                    channel_id = None
                if state.get("return_mode") == "schedule_wizard":
                    schedule = state.setdefault("schedule", {})
                    schedule["channel_id"] = channel_id
                    state["mode"] = "schedule_wizard"
                    state["step"] = "main"
                    state["search_mode"] = False
                    await self._edit_rubric_input_message(user_id)
                else:
                    code = state.get("code")
                    field = state.get("field")
                    if code and field:
                        config = self._normalize_rubric_config(
                            self.data.get_rubric_config(code) or {}
                        )
                        old_value = config.get(field)
                        if channel_id is None:
                            config.pop(field, None)
                        else:
                            config[field] = channel_id
                        self.data.save_rubric_config(code, config)
                        if old_value != channel_id:
                            if field == "channel_id":
                                reason = "prod_channel_changed"
                            else:
                                reason = "test_channel_changed"
                            self._delete_future_rubric_jobs(code, reason)
                    message_obj = state.get("message")
                    del self.pending[user_id]
                    if code:
                        await self._send_rubric_overview(
                            user_id,
                            code,
                            message=message_obj,
                        )
        elif data.startswith("rubric_city_set:") and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "city_picker":
                raw_value = data.split(":", 1)[1]
                city_id: int | None
                try:
                    city_id = int(raw_value)
                except ValueError:
                    city_id = None
                code = state.get("code")
                if code:
                    config = self._normalize_rubric_config(self.data.get_rubric_config(code) or {})
                    if city_id is None:
                        config.pop("weather_city", None)
                        config.pop("weather_city_id", None)
                    else:
                        row = self.db.execute(
                            "SELECT id, name FROM cities WHERE id=?",
                            (city_id,),
                        ).fetchone()
                        if row:
                            config["weather_city_id"] = int(row["id"])
                            config["weather_city"] = row["name"]
                        else:
                            config.pop("weather_city", None)
                            config.pop("weather_city_id", None)
                    self.data.save_rubric_config(code, config)
                message_obj = state.get("message")
                del self.pending[user_id]
                if code:
                    await self._send_rubric_overview(
                        user_id,
                        code,
                        message=message_obj,
                    )
        elif data == "rubric_channel_clear" and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "channel_picker":
                if state.get("return_mode") == "schedule_wizard":
                    schedule = state.setdefault("schedule", {})
                    schedule["channel_id"] = None
                    state["mode"] = "schedule_wizard"
                    state["step"] = "main"
                    state["search_mode"] = False
                    await self._edit_rubric_input_message(user_id)
                else:
                    code = state.get("code")
                    field = state.get("field")
                    if code and field:
                        config = self._normalize_rubric_config(
                            self.data.get_rubric_config(code) or {}
                        )
                        old_value = config.get(field)
                        config.pop(field, None)
                        self.data.save_rubric_config(code, config)
                        if old_value is not None:
                            if field == "channel_id":
                                reason = "prod_channel_changed"
                            else:
                                reason = "test_channel_changed"
                            self._delete_future_rubric_jobs(code, reason)
                    message_obj = state.get("message")
                    del self.pending[user_id]
                    if code:
                        await self._send_rubric_overview(
                            user_id,
                            code,
                            message=message_obj,
                        )
        elif data == "rubric_city_clear" and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "city_picker":
                code = state.get("code")
                if code:
                    config = self._normalize_rubric_config(self.data.get_rubric_config(code) or {})
                    config.pop("weather_city", None)
                    config.pop("weather_city_id", None)
                    self.data.save_rubric_config(code, config)
                message_obj = state.get("message")
                del self.pending[user_id]
                if code:
                    await self._send_rubric_overview(
                        user_id,
                        code,
                        message=message_obj,
                    )
        elif data == "rubric_channel_cancel" and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state:
                if state.get("return_mode") == "schedule_wizard":
                    state["mode"] = "schedule_wizard"
                    state["step"] = "main"
                    state["search_mode"] = False
                    await self._edit_rubric_input_message(user_id)
                else:
                    code = state.get("code")
                    message_obj = state.get("message")
                    del self.pending[user_id]
                    if code:
                        await self._send_rubric_overview(
                            user_id,
                            code,
                            message=message_obj,
                        )
        elif data == "rubric_city_cancel" and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "city_picker":
                code = state.get("code")
                message_obj = state.get("message")
                del self.pending[user_id]
                if code:
                    await self._send_rubric_overview(
                        user_id,
                        code,
                        message=message_obj,
                    )
        elif data.startswith("rubric_sched_add:") and self.is_superadmin(user_id):
            code = data.split(":", 1)[1]
            self._clear_rubric_pending_run(user_id, code)
            rubric = self.data.get_rubric_by_code(code)
            if not rubric:
                return
            config = self._normalize_rubric_config(rubric.config)
            default_days = config.get("days")
            if isinstance(default_days, (list, tuple)):
                days_value: Any = list(default_days)
            elif default_days:
                days_value = default_days
            else:
                days_value = []
            schedule = {
                "time": None,
                "tz": config.get("tz") or TZ_OFFSET,
                "days": days_value,
                "channel_id": config.get("channel_id"),
                "enabled": True,
            }
            self.pending[user_id] = {
                "rubric_input": {
                    "mode": "schedule_wizard",
                    "code": code,
                    "action": "schedule_add",
                    "message": query.get("message"),
                    "schedule": schedule,
                    "step": "main",
                    "search": "",
                    "search_mode": False,
                    "search_charset": "rus",
                    "page": 0,
                }
            }
            await self._edit_rubric_input_message(user_id)
        elif data.startswith("rubric_sched_edit:") and self.is_superadmin(user_id):
            parts = data.split(":")
            if len(parts) == 3:
                code, idx_str = parts[1], parts[2]
                self._clear_rubric_pending_run(user_id, code)
                try:
                    idx = int(idx_str)
                except ValueError:
                    idx = -1
                rubric = self.data.get_rubric_by_code(code)
                if not rubric:
                    return
                config = self._normalize_rubric_config(rubric.config)
                schedules = config.get("schedules", [])
                if 0 <= idx < len(schedules):
                    schedule = dict(schedules[idx])
                    self.pending[user_id] = {
                        "rubric_input": {
                            "mode": "schedule_wizard",
                            "code": code,
                            "action": "schedule_edit",
                            "index": idx,
                            "message": query.get("message"),
                            "schedule": schedule,
                            "step": "main",
                            "search": "",
                            "search_mode": False,
                            "search_charset": "rus",
                            "page": 0,
                        }
                    }
                    await self._edit_rubric_input_message(user_id)
        elif data.startswith("rubric_sched_toggle:") and self.is_superadmin(user_id):
            parts = data.split(":")
            if len(parts) == 3:
                code, idx_str = parts[1], parts[2]
                self._clear_rubric_pending_run(user_id, code)
                rubric = self.data.get_rubric_by_code(code)
                if rubric:
                    config = self._normalize_rubric_config(rubric.config)
                    schedules = config.get("schedules", [])
                    try:
                        idx = int(idx_str)
                    except ValueError:
                        idx = -1
                    if 0 <= idx < len(schedules):
                        schedule = schedules[idx]
                        schedule["enabled"] = not schedule.get("enabled", True)
                        self.data.save_rubric_config(code, config)
                        await self._send_rubric_overview(
                            user_id, code, message=query.get("message")
                        )
        elif data == "rubric_sched_time" and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "schedule_wizard":
                state["step"] = "time_hours"
                await self._edit_rubric_input_message(user_id)
        elif data.startswith("rubric_sched_hour:") and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "schedule_wizard":
                try:
                    hour = int(data.split(":", 1)[1])
                except ValueError:
                    hour = 0
                state["temp_hour"] = max(0, min(hour, 23))
                state["step"] = "time_minutes"
                await self._edit_rubric_input_message(user_id)
        elif data.startswith("rubric_sched_minute:") and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "schedule_wizard":
                try:
                    minute = int(data.split(":", 1)[1])
                except ValueError:
                    minute = 0
                hour = int(state.get("temp_hour", 0))
                minute = max(0, min(minute, 59))
                state.pop("temp_hour", None)
                schedule = state.setdefault("schedule", {})
                schedule["time"] = f"{hour:02d}:{minute:02d}"
                state["step"] = "main"
                await self._edit_rubric_input_message(user_id)
        elif data == "rubric_sched_time_back" and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "schedule_wizard":
                state["step"] = "time_hours"
                await self._edit_rubric_input_message(user_id)
        elif data == "rubric_sched_time_cancel" and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "schedule_wizard":
                state.pop("temp_hour", None)
                state["step"] = "main"
                await self._edit_rubric_input_message(user_id)
        elif data == "rubric_sched_days" and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "schedule_wizard":
                state["step"] = "days"
                await self._edit_rubric_input_message(user_id)
        elif data.startswith("rubric_sched_day:") and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "schedule_wizard":
                day = data.split(":", 1)[1]
                schedule = state.setdefault("schedule", {})
                days = schedule.get("days")
                if not isinstance(days, list):
                    days = list(days) if isinstance(days, tuple) else []
                if day in days:
                    days.remove(day)
                else:
                    days.append(day)
                schedule["days"] = days
                await self._edit_rubric_input_message(user_id)
        elif data == "rubric_sched_days_all" and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "schedule_wizard":
                schedule = state.setdefault("schedule", {})
                schedule["days"] = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
                await self._edit_rubric_input_message(user_id)
        elif data == "rubric_sched_days_clear" and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "schedule_wizard":
                schedule = state.setdefault("schedule", {})
                schedule["days"] = []
                await self._edit_rubric_input_message(user_id)
        elif data == "rubric_sched_days_done" and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "schedule_wizard":
                state["step"] = "main"
                await self._edit_rubric_input_message(user_id)
        elif data == "rubric_sched_toggle_enabled" and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "schedule_wizard":
                schedule = state.setdefault("schedule", {})
                schedule["enabled"] = not schedule.get("enabled", True)
                await self._edit_rubric_input_message(user_id)
        elif data == "rubric_sched_channel" and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "schedule_wizard":
                state["mode"] = "channel_picker"
                state["return_mode"] = "schedule_wizard"
                state["field"] = "channel_id"
                state["page"] = 0
                state["search_mode"] = False
                await self._edit_rubric_input_message(user_id)
        elif data == "rubric_sched_save" and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "schedule_wizard":
                code = state.get("code")
                if code:
                    self._clear_rubric_pending_run(user_id, code)
                schedule_data = dict(state.get("schedule") or {})
                if isinstance(schedule_data.get("days"), tuple):
                    schedule_data["days"] = list(schedule_data["days"])
                action = state.get("action")
                message_obj = state.get("message")
                try:
                    if action == "schedule_edit":
                        index = int(state.get("index", 0))
                        self.data.update_rubric_schedule(code, index, schedule_data)
                    else:
                        self.data.add_rubric_schedule(code, schedule_data)
                finally:
                    del self.pending[user_id]
                if code:
                    await self._send_rubric_overview(
                        user_id,
                        code,
                        message=message_obj,
                    )
        elif data == "rubric_sched_cancel" and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get("rubric_input")
            if state and state.get("mode") == "schedule_wizard":
                code = state.get("code")
                if code:
                    self._clear_rubric_pending_run(user_id, code)
                message_obj = state.get("message")
                del self.pending[user_id]
                if code:
                    await self._send_rubric_overview(
                        user_id,
                        code,
                        message=message_obj,
                    )
        elif data.startswith("rubric_sched_del:") and self.is_superadmin(user_id):
            parts = data.split(":")
            if len(parts) == 3:
                code, idx_str = parts[1], parts[2]
                self._clear_rubric_pending_run(user_id, code)
                try:
                    idx = int(idx_str)
                except ValueError:
                    idx = -1
                if idx >= 0 and self.data.remove_rubric_schedule(code, idx):
                    await self._send_rubric_overview(user_id, code, message=query.get("message"))
                else:
                    await self.api_request(
                        "sendMessage",
                        {
                            "chat_id": user_id,
                            "text": "Не удалось удалить расписание",
                        },
                    )
        await self.api_request("answerCallbackQuery", {"callback_query_id": query["id"]})

    async def process_due(self) -> None:
        """Publish due scheduled messages."""
        now = datetime.utcnow().isoformat()
        logging.info("Scheduler check at %s", now)
        cur = self.db.execute(
            "SELECT * FROM schedule WHERE sent=0 AND publish_time<=? ORDER BY publish_time",
            (now,),
        )
        rows = cur.fetchall()
        logging.info("Due ids: %s", [r["id"] for r in rows])
        for row in rows:
            try:
                resp = await self.api_request(
                    "forwardMessage",
                    {
                        "chat_id": row["target_chat_id"],
                        "from_chat_id": row["from_chat_id"],
                        "message_id": row["message_id"],
                    },
                )
                ok = resp.get("ok", False)
                if (
                    not ok
                    and resp.get("error_code") == 400
                    and "not" in resp.get("description", "").lower()
                ):
                    resp = await self.api_request(
                        "copyMessage",
                        {
                            "chat_id": row["target_chat_id"],
                            "from_chat_id": row["from_chat_id"],
                            "message_id": row["message_id"],
                        },
                    )
                    ok = resp.get("ok", False)
                if ok:
                    self.db.execute(
                        "UPDATE schedule SET sent=1, sent_at=? WHERE id=?",
                        (datetime.utcnow().isoformat(), row["id"]),
                    )
                    self.db.commit()
                    logging.info("Published schedule %s", row["id"])
                else:
                    logging.error("Failed to publish %s: %s", row["id"], resp)
            except Exception:
                logging.exception("Error publishing schedule %s", row["id"])

    async def process_weather_channels(self) -> None:
        now_utc = datetime.utcnow()
        jobs = self.data.due_weather_jobs(now_utc)
        for job in jobs:
            try:
                ok = await self.publish_weather(job.channel_id, None)
                if ok:
                    next_run = self.next_weather_run(job.post_time, TZ_OFFSET, reference=now_utc)
                    self.data.mark_weather_job_run(job.id, next_run)
                else:
                    self.data.record_weather_job_failure(job.id, "publish failed")
            except Exception:
                logging.exception("Failed to publish weather for %s", job.channel_id)

    async def process_rubric_schedule(self, reference: datetime | None = None) -> None:
        now = reference or datetime.utcnow()
        rubrics = self.data.list_rubrics()
        for rubric in rubrics:
            config = rubric.config or {}
            if not config.get("enabled", True):
                continue
            schedules = config.get("schedules") or config.get("schedule") or []
            if isinstance(schedules, dict):
                schedules = [schedules]
            for idx, schedule in enumerate(schedules):
                if not isinstance(schedule, dict):
                    continue
                if not schedule.get("enabled", True):
                    continue
                slot_channel_id = schedule.get("channel_id")
                default_channel_id = config.get("channel_id")
                if not slot_channel_id and not default_channel_id:
                    continue
                time_str = schedule.get("time")
                if not time_str:
                    continue
                tz_value = schedule.get("tz") or config.get("tz") or TZ_OFFSET
                days = schedule.get("days") or config.get("days")
                if slot_channel_id:
                    key = schedule.get("key") or f"{slot_channel_id}:{idx}:{time_str}"
                else:
                    key = schedule.get("key") or f"{rubric.code}:{idx}:{time_str}"
                state = self.data.get_rubric_schedule_state(rubric.code, key)
                next_run = state.next_run_at if state else None
                if next_run is None or next_run <= now:
                    next_run = self._compute_next_rubric_run(
                        time_str=time_str,
                        tz_offset=tz_value,
                        days=days,
                        reference=now,
                    )
                    self.data.set_rubric_schedule_state(
                        rubric.code,
                        key,
                        next_run_at=next_run,
                        last_run_at=state.last_run_at if state else None,
                    )
                if not next_run:
                    continue
                if self._rubric_job_exists(rubric.code, key):
                    continue
                payload: dict[str, Any] = {
                    "rubric_code": rubric.code,
                    "schedule_key": key,
                    "scheduled_at": next_run.isoformat(),
                    "tz_offset": tz_value,
                    "slot_index": idx,
                }
                if slot_channel_id:
                    payload["slot_channel_id"] = slot_channel_id
                self.jobs.enqueue("publish_rubric", payload, run_at=next_run)

    def _rubric_job_exists(self, rubric_code: str, schedule_key: str) -> bool:
        row = self.db.execute(
            """
            SELECT id FROM jobs_queue
            WHERE name='publish_rubric'
              AND status IN ('queued', 'delayed', 'running')
              AND json_extract(payload, '$.rubric_code') = ?
              AND json_extract(payload, '$.schedule_key') = ?
            LIMIT 1
            """,
            (rubric_code, schedule_key),
        ).fetchone()
        return bool(row)

    def _delete_future_rubric_jobs(self, rubric_code: str, reason: str) -> int:
        """Delete all future scheduled jobs for a rubric (excluding manual runs)."""
        now = datetime.utcnow().isoformat()
        rows = self.db.execute(
            """
            SELECT id, payload FROM jobs_queue
            WHERE name='publish_rubric'
              AND status IN ('queued', 'delayed')
              AND json_extract(payload, '$.rubric_code') = ?
              AND available_at >= ?
            """,
            (rubric_code, now),
        ).fetchall()
        deleted = 0
        for row in rows:
            payload = json.loads(row["payload"]) if row["payload"] else {}
            schedule_key = payload.get("schedule_key", "")
            if schedule_key and not schedule_key.startswith("manual"):
                self.db.execute("DELETE FROM jobs_queue WHERE id=?", (row["id"],))
                deleted += 1
        if deleted > 0:
            self.db.commit()
            logging.info(
                "Deleted future rubric jobs: rubric=%s, jobs_cleared=%d, reason=%s",
                rubric_code,
                deleted,
                reason,
            )
        return deleted

    @staticmethod
    def _normalize_rubric_config(config: dict[str, Any] | None) -> dict[str, Any]:
        data = dict(config or {})
        schedules = data.get("schedules") or data.get("schedule") or []
        if isinstance(schedules, dict):
            schedules = [schedules]
        elif not isinstance(schedules, list):
            schedules = []
        normalized: list[dict[str, Any]] = []
        for item in schedules:
            if isinstance(item, dict):
                normalized.append(dict(item))
        data["schedules"] = normalized
        data.pop("schedule", None)
        if "enabled" not in data:
            data["enabled"] = False
        return data

    @staticmethod
    def _normalize_rubric_category(category: str | None) -> str | None:
        if not category:
            return None
        normalized = re.sub(r"[\s\-]+", "_", str(category).strip().lower())
        return normalized or None

    def _build_rubric_category_cache(self) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for rubric in self.data.list_rubrics():
            config = rubric.config or {}
            assets_cfg = config.get("assets")
            if not isinstance(assets_cfg, dict):
                continue
            raw_categories = assets_cfg.get("categories")
            if isinstance(raw_categories, (list, tuple, set)):
                categories_iter = raw_categories
            elif isinstance(raw_categories, str):
                categories_iter = [raw_categories]
            else:
                continue
            for raw_category in categories_iter:
                if not isinstance(raw_category, str):
                    continue
                normalized = self._normalize_rubric_category(raw_category)
                if not normalized or normalized in mapping:
                    continue
                mapping[normalized] = rubric.id
        return mapping

    def _resolve_rubric_id_for_category(self, category: str | None) -> int | None:
        normalized = self._normalize_rubric_category(category)
        if not normalized:
            return None
        cached = self._rubric_category_cache.get(normalized)
        if cached is not None:
            return cached
        self._rubric_category_cache = self._build_rubric_category_cache()
        return self._rubric_category_cache.get(normalized)

    def _format_rubric_schedule_line(
        self,
        index: int,
        schedule: dict[str, Any],
        *,
        fallback_channel: int | None,
        fallback_tz: str | None,
        fallback_days: Any,
    ) -> str:
        time_str = schedule.get("time") or "—"
        tz_value = schedule.get("tz") or fallback_tz or TZ_OFFSET
        schedule_channel = schedule.get("channel_id") or fallback_channel
        enabled = schedule.get("enabled", True)
        days = schedule.get("days") if schedule.get("days") is not None else fallback_days
        if isinstance(days, (list, tuple)):
            days_repr = ",".join(str(d) for d in days)
        else:
            days_repr = str(days) if days else "—"
        channel_repr = str(schedule_channel) if schedule_channel is not None else "—"
        flag = "✅" if enabled else "❌"
        key = schedule.get("key")
        suffix = f" key={key}" if key else ""
        return f"#{index + 1}: {time_str} (tz {tz_value}) → {channel_repr}, дни: {days_repr} {flag}{suffix}"

    def _get_channel_title(self, chat_id: int | None) -> str:
        if chat_id is None:
            return "—"
        row = self.db.execute(
            "SELECT title FROM channels WHERE chat_id=?",
            (chat_id,),
        ).fetchone()
        title = row["title"] if row and row["title"] else None
        return title or str(chat_id)

    def _get_city_name(self, city_id: int | None) -> str | None:
        if city_id is None:
            return None
        row = self.db.execute(
            "SELECT name FROM cities WHERE id=?",
            (city_id,),
        ).fetchone()
        name = row["name"] if row and row["name"] else None
        return name

    @staticmethod
    def _weekday_label(day: str) -> str:
        mapping = {
            "mon": "Пн",
            "tue": "Вт",
            "wed": "Ср",
            "thu": "Чт",
            "fri": "Пт",
            "sat": "Сб",
            "sun": "Вс",
        }
        return mapping.get(day, day)

    def _format_weekdays(self, days: Iterable[str] | str | None) -> str:
        if not days:
            return "—"
        if isinstance(days, str):
            return days
        labels = [self._weekday_label(day) for day in days]
        return ", ".join(labels) if labels else "—"

    def _get_rubric_input_message_target(self, state: dict[str, Any]) -> tuple[int, int] | None:
        message = state.get("message")
        if not message:
            return None
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        message_id = message.get("message_id")
        if chat_id is None or message_id is None:
            return None
        return chat_id, message_id

    def _render_channel_search_keyboard(self, state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        search = state.get("search") or ""
        charset_key = state.get("search_charset") or "rus"
        charset = CHANNEL_SEARCH_CHARSETS.get(charset_key) or CHANNEL_SEARCH_CHARSETS["rus"]
        header = state.get("code") or ""
        lines = [
            f"Поиск каналов для {header}",
            f"Запрос: {search or '—'}",
            "Нажмите символы для фильтрации.",
        ]
        keyboard_rows: list[list[dict[str, Any]]] = []
        for idx in range(0, len(charset), 6):
            row_buttons: list[dict[str, Any]] = []
            for ch in charset[idx : idx + 6]:
                encoded = ch.encode("utf-8").hex()
                row_buttons.append(
                    {
                        "text": ch,
                        "callback_data": f"rubric_channel_search_add:{encoded}",
                    }
                )
            if row_buttons:
                keyboard_rows.append(row_buttons)
        switch_row: list[dict[str, Any]] = []
        for key, label in CHANNEL_SEARCH_LABELS.items():
            prefix = "• " if key == charset_key else ""
            switch_row.append(
                {
                    "text": f"{prefix}{label}",
                    "callback_data": f"rubric_channel_search_charset:{key}",
                }
            )
        keyboard_rows.append(switch_row)
        control_row: list[dict[str, Any]] = []
        for label, callback in CHANNEL_SEARCH_CONTROLS:
            control_row.append({"text": label, "callback_data": callback})
        keyboard_rows.append(control_row)
        return "\n".join(lines), {"inline_keyboard": keyboard_rows}

    def _render_city_search_keyboard(self, state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        search = state.get("search") or ""
        charset_key = state.get("search_charset") or "rus"
        charset = CHANNEL_SEARCH_CHARSETS.get(charset_key) or CHANNEL_SEARCH_CHARSETS["rus"]
        header = state.get("code") or ""
        lines = [
            f"Поиск города для {header}",
            f"Запрос: {search or '—'}",
            "Нажмите символы для фильтрации.",
        ]
        keyboard_rows: list[list[dict[str, Any]]] = []
        for idx in range(0, len(charset), 6):
            row_buttons: list[dict[str, Any]] = []
            for ch in charset[idx : idx + 6]:
                encoded = ch.encode("utf-8").hex()
                row_buttons.append(
                    {
                        "text": ch,
                        "callback_data": f"rubric_city_search_add:{encoded}",
                    }
                )
            if row_buttons:
                keyboard_rows.append(row_buttons)
        switch_row: list[dict[str, Any]] = []
        for key, label in CHANNEL_SEARCH_LABELS.items():
            prefix = "• " if key == charset_key else ""
            switch_row.append(
                {
                    "text": f"{prefix}{label}",
                    "callback_data": f"rubric_city_search_charset:{key}",
                }
            )
        keyboard_rows.append(switch_row)
        control_row: list[dict[str, Any]] = []
        for label, callback in CITY_SEARCH_CONTROLS:
            control_row.append({"text": label, "callback_data": callback})
        keyboard_rows.append(control_row)
        return "\n".join(lines), {"inline_keyboard": keyboard_rows}

    def _render_channel_picker(self, state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if state.get("search_mode"):
            return self._render_channel_search_keyboard(state)
        code = state.get("code") or ""
        field = state.get("field", "channel_id")
        search = state.get("search") or ""
        page = max(int(state.get("page", 0)), 0)
        params: list[Any] = []
        where_clause = ""
        if search:
            where_clause = " WHERE title LIKE ?"
            params.append(f"%{search}%")
        count_row = self.db.execute(
            f"SELECT COUNT(*) FROM channels{where_clause}",
            params,
        ).fetchone()
        total = count_row[0] if count_row else 0
        per_page = CHANNEL_PICKER_PAGE_SIZE
        max_page = max((total - 1) // per_page, 0)
        page = min(page, max_page)
        state["page"] = page
        offset = page * per_page
        rows = self.db.execute(
            f"SELECT chat_id, title FROM channels{where_clause} ORDER BY rowid DESC LIMIT ? OFFSET ?",
            params + [per_page, offset],
        ).fetchall()
        if state.get("return_mode") == "schedule_wizard":
            schedule = state.get("schedule") or {}
            current_id = schedule.get("channel_id")
            if current_id is None:
                config = self._normalize_rubric_config(self.data.get_rubric_config(code) or {})
                current_id = config.get("channel_id")
        else:
            config = self._normalize_rubric_config(self.data.get_rubric_config(code) or {})
            current_id = config.get(field)
        lines = [
            f"Выбор канала для {code}",
            f"Поиск: {search or '—'}",
        ]
        if not rows:
            lines.append("Каналы не найдены")
        total_pages = max_page + 1 if total else 1
        lines.append(f"Страница {page + 1}/{total_pages}")
        keyboard_rows: list[list[dict[str, Any]]] = []
        for row in rows:
            chat_id = row["chat_id"]
            title = row["title"] or str(chat_id)
            if len(title) > 50:
                title = title[:47] + "…"
            prefix = "✅ " if current_id == chat_id else ""
            keyboard_rows.append(
                [
                    {
                        "text": f"{prefix}{title}",
                        "callback_data": f"rubric_channel_set:{chat_id}",
                    }
                ]
            )
        nav_row: list[dict[str, Any]] = []
        if page > 0:
            nav_row.append({"text": "◀️", "callback_data": f"rubric_channel_page:{page - 1}"})
        if page < max_page:
            nav_row.append({"text": "▶️", "callback_data": f"rubric_channel_page:{page + 1}"})
        if nav_row:
            keyboard_rows.append(nav_row)
        keyboard_rows.append(
            [
                {
                    "text": "🔍 Поиск",
                    "callback_data": "rubric_channel_search_toggle",
                },
                {
                    "text": "Очистить",
                    "callback_data": "rubric_channel_clear",
                },
            ]
        )
        cancel_text = "Назад" if state.get("return_mode") == "schedule_wizard" else "Отмена"
        keyboard_rows.append([{"text": cancel_text, "callback_data": "rubric_channel_cancel"}])
        return "\n".join(lines), {"inline_keyboard": keyboard_rows}

    def _render_city_picker(self, state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if state.get("search_mode"):
            return self._render_city_search_keyboard(state)
        code = state.get("code") or ""
        search = state.get("search") or ""
        page = max(int(state.get("page", 0)), 0)
        params: list[Any] = []
        where_clause = ""
        if search:
            where_clause = " WHERE name LIKE ?"
            params.append(f"%{search}%")
        count_row = self.db.execute(
            f"SELECT COUNT(*) FROM cities{where_clause}",
            params,
        ).fetchone()
        total = count_row[0] if count_row else 0
        per_page = CITY_PICKER_PAGE_SIZE
        max_page = max((total - 1) // per_page, 0)
        page = min(page, max_page)
        state["page"] = page
        offset = page * per_page
        rows = self.db.execute(
            f"SELECT id, name FROM cities{where_clause} ORDER BY name ASC LIMIT ? OFFSET ?",
            params + [per_page, offset],
        ).fetchall()
        config = self._normalize_rubric_config(self.data.get_rubric_config(code) or {})
        current_id = config.get("weather_city_id")
        current_name = config.get("weather_city")
        lines = [
            f"Выбор города погоды для {code}",
            f"Поиск: {search or '—'}",
        ]
        if current_name:
            lines.append(f"Текущий: {current_name}")
        if not rows:
            lines.append("Города не найдены")
        total_pages = max_page + 1 if total else 1
        lines.append(f"Страница {page + 1}/{total_pages}")
        keyboard_rows: list[list[dict[str, Any]]] = []
        for row in rows:
            city_id = row["id"]
            name = row["name"] or str(city_id)
            display = name if len(name) <= 50 else name[:47] + "…"
            prefix = (
                "✅ "
                if current_id == city_id
                or (current_id is None and current_name and name == current_name)
                else ""
            )
            keyboard_rows.append(
                [
                    {
                        "text": f"{prefix}{display}",
                        "callback_data": f"rubric_city_set:{city_id}",
                    }
                ]
            )
        nav_row: list[dict[str, Any]] = []
        if page > 0:
            nav_row.append({"text": "◀️", "callback_data": f"rubric_city_page:{page - 1}"})
        if page < max_page:
            nav_row.append({"text": "▶️", "callback_data": f"rubric_city_page:{page + 1}"})
        if nav_row:
            keyboard_rows.append(nav_row)
        keyboard_rows.append(
            [
                {
                    "text": "🔍 Поиск",
                    "callback_data": "rubric_city_search_toggle",
                },
                {
                    "text": "Очистить",
                    "callback_data": "rubric_city_clear",
                },
            ]
        )
        keyboard_rows.append([{"text": "Отмена", "callback_data": "rubric_city_cancel"}])
        return "\n".join(lines), {"inline_keyboard": keyboard_rows}

    def _build_time_hours_keyboard(self) -> dict[str, Any]:
        rows: list[list[dict[str, Any]]] = []
        for start in range(0, 24, 6):
            row: list[dict[str, Any]] = []
            for hour in range(start, min(start + 6, 24)):
                label = f"{hour:02d}"
                row.append(
                    {
                        "text": label,
                        "callback_data": f"rubric_sched_hour:{hour}",
                    }
                )
            rows.append(row)
        rows.append(
            [
                {"text": "Отмена", "callback_data": "rubric_sched_time_cancel"},
            ]
        )
        return {"inline_keyboard": rows}

    def _build_time_minutes_keyboard(self) -> dict[str, Any]:
        rows: list[list[dict[str, Any]]] = []
        for start in range(0, 60, 15):
            row: list[dict[str, Any]] = []
            for minute in range(start, min(start + 15, 60), 5):
                label = f"{minute:02d}"
                row.append(
                    {
                        "text": label,
                        "callback_data": f"rubric_sched_minute:{minute}",
                    }
                )
            rows.append(row)
        rows.append(
            [
                {"text": "⬅️", "callback_data": "rubric_sched_time_back"},
                {"text": "Отмена", "callback_data": "rubric_sched_time_cancel"},
            ]
        )
        return {"inline_keyboard": rows}

    def _build_days_keyboard(self, schedule: dict[str, Any]) -> dict[str, Any]:
        current = schedule.get("days")
        if not isinstance(current, list):
            current = list(current) if isinstance(current, tuple) else []
        rows: list[list[dict[str, Any]]] = []
        order = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        for start in range(0, len(order), 3):
            row: list[dict[str, Any]] = []
            for day in order[start : start + 3]:
                label = self._weekday_label(day)
                if day in current:
                    label = f"✅ {label}"
                row.append(
                    {
                        "text": label,
                        "callback_data": f"rubric_sched_day:{day}",
                    }
                )
            rows.append(row)
        rows.append(
            [
                {
                    "text": "Все",
                    "callback_data": "rubric_sched_days_all",
                },
                {
                    "text": "Очистить",
                    "callback_data": "rubric_sched_days_clear",
                },
                {
                    "text": "Готово",
                    "callback_data": "rubric_sched_days_done",
                },
            ]
        )
        return {"inline_keyboard": rows}

    def _render_schedule_wizard(self, state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        code = state.get("code") or ""
        schedule = state.setdefault("schedule", {})
        config = self._normalize_rubric_config(self.data.get_rubric_config(code) or {})
        schedule.setdefault("tz", schedule.get("tz") or config.get("tz") or TZ_OFFSET)
        if schedule.get("days") is None and config.get("days") is not None:
            fallback_days = config.get("days")
            schedule["days"] = (
                list(fallback_days) if isinstance(fallback_days, (list, tuple)) else fallback_days
            )
        lines = [f"Настройка расписания для {code}"]
        time_value = schedule.get("time") or "--:--"
        lines.append(f"Время: {time_value} (TZ {schedule.get('tz')})")
        lines.append(f"Дни: {self._format_weekdays(schedule.get('days'))}")
        channel_id = schedule.get("channel_id")
        if channel_id is None and config.get("channel_id") is not None:
            channel_text = f"{self._get_channel_title(config.get('channel_id'))} (по умолчанию)"
        else:
            channel_text = self._get_channel_title(channel_id)
        lines.append(f"Канал: {channel_text}")
        enabled = schedule.get("enabled", True)
        lines.append(f"Статус: {'✅' if enabled else '❌'}")
        step = state.get("step", "main")
        if step == "time_hours":
            lines.append("Выберите часы отправки")
            keyboard = self._build_time_hours_keyboard()
        elif step == "time_minutes":
            lines.append("Выберите минуты отправки")
            keyboard = self._build_time_minutes_keyboard()
        elif step == "days":
            lines.append("Выберите дни недели")
            keyboard = self._build_days_keyboard(schedule)
        else:
            keyboard_rows: list[list[dict[str, Any]]] = [
                [
                    {
                        "text": f"🕒 Время: {time_value}",
                        "callback_data": "rubric_sched_time",
                    }
                ],
                [
                    {
                        "text": f"📅 Дни: {self._format_weekdays(schedule.get('days'))}",
                        "callback_data": "rubric_sched_days",
                    }
                ],
                [
                    {
                        "text": f"📡 Канал: {channel_text}",
                        "callback_data": "rubric_sched_channel",
                    }
                ],
                [
                    {
                        "text": "✅ Включено" if enabled else "❌ Выключено",
                        "callback_data": "rubric_sched_toggle_enabled",
                    }
                ],
                [
                    {
                        "text": "Сохранить",
                        "callback_data": "rubric_sched_save",
                    },
                    {
                        "text": "Отмена",
                        "callback_data": "rubric_sched_cancel",
                    },
                ],
            ]
            keyboard = {"inline_keyboard": keyboard_rows}
        return "\n".join(lines), keyboard

    async def _edit_rubric_input_message(self, user_id: int) -> None:
        state = self.pending.get(user_id, {}).get("rubric_input")
        if not state:
            return
        target = self._get_rubric_input_message_target(state)
        if not target:
            return
        chat_id, message_id = target
        if state.get("mode") == "channel_picker":
            text, keyboard = self._render_channel_picker(state)
        elif state.get("mode") == "city_picker":
            text, keyboard = self._render_city_picker(state)
        elif state.get("mode") == "schedule_wizard":
            text, keyboard = self._render_schedule_wizard(state)
        else:
            return
        payload = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "reply_markup": keyboard,
        }
        await self.api_request("editMessageText", payload)

    async def _send_rubric_dashboard(
        self,
        user_id: int,
        *,
        message: dict[str, Any] | None = None,
    ) -> None:
        target = message
        if target is None:
            stored = self.rubric_dashboards.get(user_id)
            if stored:
                target = {
                    "chat": {"id": stored.get("chat_id")},
                    "message_id": stored.get("message_id"),
                }
        rubrics = self.data.list_rubrics()
        lines: list[str] = [
            "Карточки рубрик",
            "",
            (
                "Бот автоматически создаёт рубрики `flowers` и `guess_arch`."
                " Управляйте ими прямо в карточках:"
                " включение, выбор каналов, расписания и ручные запуски."
            ),
        ]
        if rubrics:
            lines.append("")
            lines.append("Состояние:")
            for rubric in rubrics:
                config = self._normalize_rubric_config(rubric.config)
                enabled = config.get("enabled", False)
                status = "✅" if enabled else "❌"
                lines.append(f"{status} {rubric.title} ({rubric.code})")
        else:
            lines.append("")
            lines.append("Рубрики ещё не созданы.")
        keyboard_rows: list[list[dict[str, Any]]] = [
            [
                {
                    "text": "Обновить карточки",
                    "callback_data": "rubric_dashboard",
                }
            ]
        ]
        keyboard = {"inline_keyboard": keyboard_rows}
        text = "\n".join(lines).strip()
        chat_id: int | None = None
        message_id: int | None = None
        if target:
            chat = target.get("chat") or {}
            chat_id = chat.get("id")
            message_id = target.get("message_id")
            if chat_id is not None and message_id is not None:
                payload = {
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "text": text,
                    "reply_markup": keyboard,
                }
                response = await self.api_request("editMessageText", payload)
                if response and response.get("ok"):
                    logging.info(
                        "Rubric dashboard: reply_mode=edit, user_id=%s, msg_id=%s",
                        user_id,
                        message_id,
                    )
                    chat_id = chat.get("id")
                    message_id = message_id
                else:
                    error_desc = response.get("description", "") if response else ""
                    error_code = response.get("error_code") if response else None
                    if error_code in {400, 404}:
                        logging.info(
                            "Rubric dashboard: edit failed (code=%s, desc=%s), falling back to send. user_id=%s, old_msg_id=%s",
                            error_code,
                            error_desc,
                            user_id,
                            message_id,
                        )
                        chat_id = None
                        message_id = None
                    else:
                        logging.warning(
                            "Rubric dashboard: edit failed unexpectedly (code=%s, desc=%s). user_id=%s, msg_id=%s",
                            error_code,
                            error_desc,
                            user_id,
                            message_id,
                        )
                        chat_id = None
                        message_id = None
        if chat_id is None or message_id is None:
            response = await self.api_request(
                "sendMessage",
                {"chat_id": user_id, "text": text, "reply_markup": keyboard},
            )
            if response and response.get("ok"):
                result = response.get("result")
                if isinstance(result, dict):
                    chat = result.get("chat") or {}
                    chat_id = chat.get("id")
                    message_id = result.get("message_id")
                    reply_mode = "fallback_send" if target else "send"
                    logging.info(
                        "Rubric dashboard: reply_mode=%s, user_id=%s, new_msg_id=%s",
                        reply_mode,
                        user_id,
                        message_id,
                    )
        if chat_id is not None and message_id is not None:
            self.rubric_dashboards[user_id] = {
                "chat_id": chat_id,
                "message_id": message_id,
            }
        await self._render_rubric_cards(user_id, rubrics)

    def _build_rubric_overview(
        self, rubric: Rubric, *, pending_mode: str | None = None
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        config = self._normalize_rubric_config(rubric.config)
        title_line = f"{rubric.title} ({rubric.code})"
        enabled = config.get("enabled", True)
        flag = "✅" if enabled else "❌"
        channel = config.get("channel_id")
        channel_line = f"Канал: {channel if channel is not None else '—'}"
        test_channel = config.get("test_channel_id")
        test_line = f"Тестовый канал: {test_channel if test_channel is not None else '—'}"
        tz_value = config.get("tz")
        tz_line = f"Часовой пояс по умолчанию: {tz_value or TZ_OFFSET}"
        days_default = config.get("days")
        days_line = (
            f"Дни по умолчанию: {','.join(days_default) if isinstance(days_default, (list, tuple)) else days_default}"
            if days_default
            else "Дни по умолчанию: —"
        )
        weather_city_line: str | None = None
        if rubric.code == "flowers":
            city_name = config.get("weather_city")
            city_id = config.get("weather_city_id")
            if (not city_name or not str(city_name).strip()) and isinstance(city_id, int):
                resolved = self._get_city_name(city_id)
                if resolved:
                    city_name = resolved
            city_display = city_name if city_name else "—"
            weather_city_line = f"Город погоды: {city_display}"
        lines = [
            title_line,
            f"Статус: {flag}",
            channel_line,
            test_line,
        ]
        if weather_city_line:
            lines.append(weather_city_line)
        lines.extend([tz_line, days_line, "Расписания:"])
        schedules = config.get("schedules", [])
        if schedules:
            for idx, schedule in enumerate(schedules):
                lines.append(
                    self._format_rubric_schedule_line(
                        idx,
                        schedule,
                        fallback_channel=channel,
                        fallback_tz=tz_value,
                        fallback_days=days_default,
                    )
                )
        else:
            lines.append("—")
        if pending_mode:
            mode_label = "рабочую" if pending_mode == "prod" else "тестовую"
            lines.append(f"Запуск: подтвердите {mode_label} публикацию.")

        keyboard_rows: list[list[dict[str, Any]]] = []
        toggle_text = "Выключить" if enabled else "Включить"
        keyboard_rows.append(
            [
                {"text": toggle_text, "callback_data": f"rubric_toggle:{rubric.code}"},
            ]
        )
        if pending_mode:
            keyboard_rows.append(
                [
                    {
                        "text": "✅ Подтвердить",
                        "callback_data": f"rubric_publish_execute:{rubric.code}:{pending_mode}",
                    },
                    {
                        "text": "✖️ Отмена",
                        "callback_data": f"rubric_publish_cancel:{rubric.code}",
                    },
                ]
            )
        else:
            keyboard_rows.append(
                [
                    {
                        "text": "▶️ Запустить",
                        "callback_data": f"rubric_publish_confirm:{rubric.code}:prod",
                    },
                    {
                        "text": "🧪 Тест",
                        "callback_data": f"rubric_publish_confirm:{rubric.code}:test",
                    },
                ]
            )
        keyboard_rows.append(
            [
                {
                    "text": "Канал",
                    "callback_data": f"rubric_channel:{rubric.code}:main",
                },
                {
                    "text": "Тест-канал",
                    "callback_data": f"rubric_channel:{rubric.code}:test",
                },
            ]
        )
        if rubric.code == "flowers":
            keyboard_rows.append(
                [
                    {
                        "text": "Город погоды",
                        "callback_data": f"rubric_city:{rubric.code}",
                    }
                ]
            )
        keyboard_rows.append(
            [
                {
                    "text": "Добавить расписание",
                    "callback_data": f"rubric_sched_add:{rubric.code}",
                }
            ]
        )
        for idx, schedule in enumerate(schedules):
            keyboard_rows.append(
                [
                    {
                        "text": f"#{idx + 1} Изменить",
                        "callback_data": f"rubric_sched_edit:{rubric.code}:{idx}",
                    },
                    {
                        "text": "Вкл/Выкл",
                        "callback_data": f"rubric_sched_toggle:{rubric.code}:{idx}",
                    },
                    {
                        "text": "Удалить",
                        "callback_data": f"rubric_sched_del:{rubric.code}:{idx}",
                    },
                ]
            )
        keyboard_rows.append(
            [
                {
                    "text": "↩️ Управление рубриками",
                    "callback_data": "rubric_dashboard",
                }
            ]
        )
        keyboard = {"inline_keyboard": keyboard_rows}
        return "\n".join(lines), config, keyboard

    async def _send_rubric_overview(
        self,
        user_id: int,
        code: str,
        *,
        message: dict[str, Any] | None = None,
    ) -> None:
        rubric = self.data.get_rubric_by_code(code)
        if not rubric:
            payload = {
                "chat_id": message.get("chat", {}).get("id") if message else user_id,
                "text": f"Рубрика {code} не найдена",
            }
            if message:
                payload["message_id"] = message.get("message_id")
                await self.api_request("editMessageText", payload)
            else:
                await self.api_request("sendMessage", payload)
            return
        pending_mode = self._get_rubric_pending_run(user_id, code)
        text, _, keyboard = self._build_rubric_overview(rubric, pending_mode=pending_mode)
        payload = {"text": text, "reply_markup": keyboard}
        sent_new_message = False
        if message:
            chat_id = message.get("chat", {}).get("id", user_id)
            message_id = message.get("message_id")
            payload.update({"chat_id": chat_id, "message_id": message_id})
            response = await self.api_request("editMessageText", payload)
            if response and response.get("ok"):
                logging.info(
                    "Rubric overview %s: reply_mode=edit, user_id=%s, msg_id=%s",
                    code,
                    user_id,
                    message_id,
                )
                if chat_id is not None and message_id is not None:
                    self._remember_rubric_overview(
                        user_id, code, chat_id=chat_id, message_id=message_id
                    )
            else:
                error_desc = response.get("description", "") if response else ""
                error_code = response.get("error_code") if response else None
                if error_code in {400, 404}:
                    logging.info(
                        "Rubric overview %s: edit failed (code=%s, desc=%s), falling back to send. user_id=%s, old_msg_id=%s",
                        code,
                        error_code,
                        error_desc,
                        user_id,
                        message_id,
                    )
                    sent_new_message = True
                else:
                    logging.warning(
                        "Rubric overview %s: edit failed unexpectedly (code=%s, desc=%s). user_id=%s, msg_id=%s",
                        code,
                        error_code,
                        error_desc,
                        user_id,
                        message_id,
                    )
                    sent_new_message = True
        else:
            sent_new_message = True
        if sent_new_message:
            payload = {"text": text, "reply_markup": keyboard, "chat_id": user_id}
            response = await self.api_request("sendMessage", payload)
            if response and response.get("ok"):
                result = response.get("result")
                if isinstance(result, dict):
                    chat = result.get("chat") or {}
                    chat_id = chat.get("id", user_id)
                    message_id = result.get("message_id")
                    reply_mode = "fallback_send" if message else "send"
                    logging.info(
                        "Rubric overview %s: reply_mode=%s, user_id=%s, new_msg_id=%s",
                        code,
                        reply_mode,
                        user_id,
                        message_id,
                    )
                    if chat_id is not None and message_id is not None:
                        self._remember_rubric_overview(
                            user_id,
                            code,
                            chat_id=chat_id,
                            message_id=message_id,
                        )

    def enqueue_rubric(
        self,
        code: str,
        *,
        channel_id: int | None = None,
        test: bool = False,
        initiator_id: int | None = None,
        instructions: str | None = None,
    ) -> int:
        rubric = self.data.get_rubric_by_code(code)
        if not rubric:
            raise ValueError(f"Unknown rubric {code}")
        config = rubric.config or {}
        target = channel_id
        if target is None:
            prod_channel = config.get("channel_id")
            test_channel = config.get("test_channel_id")
            target = test_channel if test else prod_channel
            logging.info(
                "Channel resolved: rubric=%s, test=%s, prod_channel=%s, "
                "test_channel=%s, resolved=%s",
                code,
                test,
                prod_channel,
                test_channel,
                target,
            )
        if target is None:
            raise ValueError("Channel id is required for rubric publication")
        payload = {
            "rubric_code": code,
            "channel_id": target,
            "schedule_key": "manual-test" if test else "manual",
            "scheduled_at": datetime.utcnow().isoformat(),
            "test": test,
            "tz_offset": config.get("tz") or TZ_OFFSET,
        }
        if initiator_id is not None:
            payload["initiator_id"] = initiator_id
        if instructions:
            payload["instructions"] = instructions
        return self.jobs.enqueue("publish_rubric", payload)

    async def _job_publish_rubric(self, job: Job) -> None:
        payload = job.payload or {}
        code = payload.get("rubric_code")
        if not code:
            logging.warning("Rubric job %s missing code", job.id)
            return
        test_mode = bool(payload.get("test"))
        schedule_key = payload.get("schedule_key")
        scheduled_at = payload.get("scheduled_at")
        old_payload_channel = payload.get("channel_id")
        resolved_channel: int | None = None
        if schedule_key and not schedule_key.startswith("manual"):
            rubric = self.data.get_rubric_by_code(code)
            if rubric:
                config = rubric.config or {}
                slot_channel_id = payload.get("slot_channel_id")
                if slot_channel_id:
                    resolved_channel = slot_channel_id
                elif test_mode:
                    resolved_channel = config.get("test_channel_id")
                else:
                    resolved_channel = config.get("channel_id")
                if old_payload_channel and old_payload_channel != resolved_channel:
                    logging.info(
                        "Channel resolved at execution: rubric=%s, old_payload_channel=%s, resolved=%s",
                        code,
                        old_payload_channel,
                        resolved_channel,
                    )
                elif not old_payload_channel and resolved_channel:
                    logging.debug(
                        "Channel resolved at execution: rubric=%s, resolved=%s",
                        code,
                        resolved_channel,
                    )
        else:
            resolved_channel = old_payload_channel
            logging.info(
                "_job_publish_rubric (manual): rubric=%s, test_mode=%s, "
                "schedule_key=%s, payload_channel=%s, resolved=%s",
                code,
                test_mode,
                schedule_key,
                old_payload_channel,
                resolved_channel,
            )
        success = await self.publish_rubric(
            code,
            channel_id=resolved_channel,
            test=test_mode,
            job=job,
            initiator_id=payload.get("initiator_id"),
            instructions=payload.get("instructions"),
        )
        if success and schedule_key and scheduled_at:
            try:
                run_at = datetime.fromisoformat(scheduled_at)
            except ValueError:
                run_at = datetime.utcnow()
            self.data.mark_rubric_run(code, schedule_key, run_at)
        if not success:
            raise RuntimeError(f"Failed to publish rubric {code}")

    async def publish_rubric(
        self,
        code: str,
        channel_id: int | None = None,
        *,
        test: bool = False,
        job: Job | None = None,
        initiator_id: int | None = None,
        instructions: str | None = None,
    ) -> bool:
        rubric = self.data.get_rubric_by_code(code)
        if not rubric:
            logging.warning("Rubric %s not found", code)
            return False
        config = rubric.config or {}
        target = channel_id
        channel_source = "provided" if channel_id is not None else "config"
        if target is None:
            prod_channel = config.get("channel_id")
            test_channel = config.get("test_channel_id")
            target = test_channel if test else prod_channel
        logging.info(
            "publish_rubric: rubric=%s, test=%s, prod_channel=%s, "
            "test_channel=%s, channel_source=%s, resolved=%s",
            code,
            test,
            config.get("channel_id"),
            config.get("test_channel_id"),
            channel_source,
            target,
        )
        if target is None:
            logging.warning("Rubric %s missing channel configuration", code)
            return False
        handler = getattr(self, f"_publish_{code}", None)
        if not handler:
            logging.warning("No handler defined for rubric %s", code)
            return False
        return await handler(
            rubric,
            int(target),
            test=test,
            job=job,
            initiator_id=initiator_id,
            instructions=instructions,
        )

    def _parse_tz_offset(self, value: str | None) -> timezone:
        offset = (value or "+00:00").strip()
        sign = 1
        if offset.startswith("-"):
            sign = -1
            offset = offset[1:]
        elif offset.startswith("+"):
            offset = offset[1:]
        hours_str, _, minutes_str = offset.partition(":")
        try:
            hours = int(hours_str or "0")
            minutes = int(minutes_str or "0")
        except ValueError:
            return UTC
        delta = timedelta(hours=hours, minutes=minutes)
        return timezone(sign * delta)

    def _normalize_days(self, raw_days: Any) -> set[int]:
        if not raw_days:
            return set()
        if isinstance(raw_days, str):
            items = [raw_days]
        elif isinstance(raw_days, Iterable):
            items = list(raw_days)
        else:
            return set()
        mapping = {
            "mon": 0,
            "tue": 1,
            "wed": 2,
            "thu": 3,
            "fri": 4,
            "sat": 5,
            "sun": 6,
        }
        result: set[int] = set()
        for item in items:
            if isinstance(item, int):
                result.add(item % 7)
            elif isinstance(item, str):
                key = item.strip().lower()[:3]
                if key in mapping:
                    result.add(mapping[key])
        return result

    async def _ensure_asset_source(self, asset: Asset) -> tuple[str | None, bool]:
        """Provide a local path for an asset and flag whether it should be removed."""

        local_path = asset.local_path if asset.local_path else None
        if local_path and os.path.exists(local_path):
            return local_path, True
        if not asset.file_id or self.dry_run:
            return None, False
        file_meta = {
            "file_id": asset.file_id,
            "file_unique_id": asset.file_unique_id,
            "file_name": asset.file_name,
            "mime_type": asset.mime_type,
            "duration": asset.duration,
            "file_size": asset.file_size,
            "width": asset.width,
            "height": asset.height,
        }
        target_path = self._build_local_file_path(asset.id, file_meta)
        downloaded_path = await self._download_file(asset.file_id, target_path)
        if not isinstance(downloaded_path, Path):
            return None, False
        return str(downloaded_path), True

    def _derive_primary_scene(self, primary_scene: str, tags: Sequence[str]) -> str:
        normalized_tags = [tag.lower() for tag in tags if tag]
        normalized_set = set(normalized_tags)
        if "architecture" in normalized_set:
            return "architecture"
        if normalized_set.intersection({"flowers", "flower"}):
            return "flowers"
        if normalized_tags:
            return normalized_tags[0]
        return primary_scene or "unknown"

    def _compute_next_rubric_run(
        self,
        *,
        time_str: str,
        tz_offset: str,
        days: Any,
        reference: datetime,
    ) -> datetime:
        tzinfo = self._parse_tz_offset(tz_offset)
        if reference.tzinfo is None:
            local_ref = reference.replace(tzinfo=UTC).astimezone(tzinfo)
        else:
            local_ref = reference.astimezone(tzinfo)
        try:
            hour, minute = [int(part) for part in time_str.split(":", 1)]
        except ValueError:
            hour, minute = 0, 0
        candidate = local_ref.replace(hour=hour, minute=minute, second=0, microsecond=0)
        allowed_days = self._normalize_days(days)
        if not allowed_days:
            allowed_days = set(range(7))
        while candidate <= local_ref or candidate.weekday() not in allowed_days:
            candidate = candidate + timedelta(days=1)
            candidate = candidate.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return candidate.astimezone(UTC).replace(tzinfo=None)

    @staticmethod
    def _parse_positive_int(value: Any) -> int | None:
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        if number < 1:
            return None
        return number

    def _resolve_flowers_asset_limits(self, rubric: Rubric) -> tuple[int, int]:
        config = rubric.config or {}
        asset_cfg = config.get("assets") or {}
        min_raw = asset_cfg.get("min")
        min_value = self._parse_positive_int(min_raw)
        if min_value is None:
            if min_raw not in (None, "", 0):
                logging.warning(
                    "Invalid flowers min asset config for rubric %s: %r",
                    rubric.code,
                    min_raw,
                )
            min_count = 1
        else:
            min_count = min_value
        max_raw = asset_cfg.get("max")
        max_value = self._parse_positive_int(max_raw)
        if max_value is None:
            max_count = max(min_count, 6)
        else:
            max_count = max(min_count, min(max_value, 6))
        return min_count, max_count

    def _asset_media_kind(self, asset: Asset) -> tuple[str, bool]:
        """Return media kind and whether a fresh upload is required."""

        kind = (asset.kind or "").strip().lower()
        missing_file = not asset.file_id
        if kind == "photo":
            return "photo", missing_file
        if kind == "document":
            if self._is_convertible_image_document(asset):
                return "photo", True
            if is_photo_mime(asset.mime_type):
                return "photo", True
            return "document", missing_file
        if is_photo_mime(asset.mime_type):
            return "photo", missing_file
        return "document", missing_file

    def _flowers_seed(self, channel_id: int | None) -> str:
        today = datetime.utcnow().strftime("%Y%m%d")
        return f"{today}:{channel_id or 0}"

    def _flowers_recent_pattern_ids(
        self, rubric: Rubric, limit: int = 14
    ) -> tuple[set[str], set[str], list[list[str]]]:
        pattern_history = self.data.get_recent_rubric_pattern_ids(rubric.code, limit=limit)
        result: set[str] = set()
        normalized_history: list[list[str]] = []
        for entries in pattern_history:
            normalized = [str(item).strip() for item in entries if str(item).strip()]
            normalized_history.append(normalized)
            result.update(normalized)
        consecutive_repeats: set[str] = set()
        if len(normalized_history) >= 2:
            latest = set(normalized_history[0])
            previous = set(normalized_history[1])
            consecutive_repeats = latest & previous
        return result, consecutive_repeats, normalized_history

    def _flowers_rotation_order(
        self,
        candidates: Sequence[FlowerPattern],
        history: Sequence[Sequence[str]],
        weather_pattern: FlowerPattern | None,
    ) -> list[FlowerPattern]:
        if not candidates:
            return []
        sorted_candidates = sorted(candidates, key=lambda pattern: pattern.id)
        weather_id = weather_pattern.id if weather_pattern else None

        recent_rotation: list[str] = []
        for day in history:
            for pattern_id in day:
                if weather_id and pattern_id == weather_id:
                    continue
                recent_rotation.append(pattern_id)
                break
        order_ids = [pattern.id for pattern in sorted_candidates]
        filtered_recent = [pid for pid in recent_rotation if pid in order_ids]
        last_rotation = filtered_recent[0] if filtered_recent else None
        exclusions = {pid for pid in filtered_recent[:2] if pid}
        if exclusions and len(exclusions) >= len(order_ids):
            exclusions = set()
        start_index = 0
        if last_rotation and last_rotation in order_ids:
            start_index = (order_ids.index(last_rotation) + 1) % len(order_ids)
        ordered_ids: list[str] = []
        used: set[str] = set()
        for offset in range(len(order_ids)):
            index = (start_index + offset) % len(order_ids)
            pattern_id = order_ids[index]
            if pattern_id in used:
                continue
            if exclusions and pattern_id in exclusions:
                continue
            ordered_ids.append(pattern_id)
            used.add(pattern_id)
        for pattern_id in order_ids:
            if pattern_id not in used:
                ordered_ids.append(pattern_id)
                used.add(pattern_id)
        lookup = {pattern.id: pattern for pattern in sorted_candidates}
        return [lookup[pattern_id] for pattern_id in ordered_ids if pattern_id in lookup]

    def _extract_flower_features(
        self,
        assets: Sequence[Asset],
        weather_block: dict[str, Any] | None,
        *,
        seed_rng: random.Random,
        asset_seasons: Mapping[int, str] | None = None,
    ) -> dict[str, Any]:
        kb = self.flowers_kb
        flowers: list[dict[str, Any]] = []
        flower_ids: list[str] = []
        has_photo = all(asset.kind == "photo" for asset in assets)
        seasons_lookup = dict(asset_seasons or {})

        def _normalize_descriptors(value: Any) -> list[str]:
            if value is None:
                return []
            if isinstance(value, str):
                cleaned = value.replace("•", ",").replace("\n", ",")
                cleaned = cleaned.replace("—", ",").replace("–", ",")
                cleaned = re.sub(r"\s+и\s+", ", ", cleaned, flags=re.IGNORECASE)
                parts = [
                    part.strip(" \t\r\f\v-–—•·")
                    for part in re.split(r"[,;/]", cleaned)
                    if part.strip()
                ]
                return [re.sub(r"\s+", " ", part) for part in parts if part]
            if isinstance(value, (list, tuple, set)):
                descriptors: list[str] = []
                for item in value:
                    descriptors.extend(_normalize_descriptors(item))
                return descriptors
            if isinstance(value, dict):
                descriptors: list[str] = []
                for key in ("name", "label", "title", "description", "value", "text"):
                    if key in value:
                        descriptors.extend(_normalize_descriptors(value[key]))
                if not descriptors:
                    for item in value.values():
                        descriptors.extend(_normalize_descriptors(item))
                return descriptors
            text = str(value).strip()
            return [text] if text else []

        def _build_palette_entry(
            title: str | None,
            descriptors: Iterable[Any] | str,
            *,
            mood: str | None = None,
            default_title: str | None = None,
        ) -> dict[str, Any] | None:
            normalized_title = str(title or "").strip()
            if not normalized_title and default_title:
                normalized_title = default_title
            normalized_descriptors: list[str] = []
            seen: set[str] = set()
            if isinstance(descriptors, str):
                descriptor_iterable: Iterable[Any] = [descriptors]
            else:
                descriptor_iterable = descriptors
            for descriptor in descriptor_iterable:
                text = str(descriptor or "").strip()
                if not text:
                    continue
                cleaned = re.sub(r"\s+", " ", text)
                lowered = cleaned.casefold()
                if lowered in seen:
                    continue
                seen.add(lowered)
                normalized_descriptors.append(cleaned)
            if not normalized_descriptors and not normalized_title:
                return None
            entry: dict[str, Any] = {"descriptors": normalized_descriptors}
            if normalized_title:
                entry["title"] = normalized_title
            if mood:
                mood_text = str(mood).strip()
                if mood_text:
                    entry["mood"] = mood_text
            return entry

        def _collect_color_palettes_from_results(
            result_payload: dict[str, Any],
        ) -> list[dict[str, Any]]:
            raw_colors = result_payload.get("colors") if isinstance(result_payload, dict) else None
            if not raw_colors:
                return []

            def _flatten_palette(payload: Any) -> list[tuple[str | None, list[str]]]:
                flattened: list[tuple[str | None, list[str]]] = []
                if isinstance(payload, dict):
                    if isinstance(payload.get("palettes"), (list, tuple, set)):
                        for item in payload["palettes"]:
                            flattened.extend(_flatten_palette(item))
                    else:
                        title = (
                            payload.get("title")
                            or payload.get("name")
                            or payload.get("label")
                            or payload.get("palette")
                            or payload.get("theme")
                            or payload.get("caption")
                        )
                        descriptors: list[str] = []
                        for key in (
                            "descriptors",
                            "colors",
                            "swatches",
                            "keywords",
                            "tones",
                            "dominant",
                            "primary",
                            "values",
                            "names",
                            "items",
                            "description",
                            "summary",
                        ):
                            if key in payload:
                                descriptors.extend(_normalize_descriptors(payload[key]))
                        if not descriptors and len(payload) == 1:
                            only_value = next(iter(payload.values()))
                            descriptors.extend(_normalize_descriptors(only_value))
                        flattened.append((title, descriptors))
                elif isinstance(payload, (list, tuple, set)):
                    for item in payload:
                        flattened.extend(_flatten_palette(item))
                else:
                    flattened.append((None, _normalize_descriptors(payload)))
                return flattened

            entries: list[dict[str, Any]] = []
            for title, descriptors in _flatten_palette(raw_colors):
                entry = _build_palette_entry(title, descriptors, default_title="Палитра снимка")
                if entry:
                    entries.append(entry)
            return entries

        def _collect_color_palettes_from_caption(caption_text: str) -> list[dict[str, Any]]:
            palettes: list[dict[str, Any]] = []
            if not caption_text:
                return palettes
            for raw_line in caption_text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                match = re.match(r"(?P<title>.+?)([:\-—]\s*)(?P<body>.+)", line)
                if not match:
                    continue
                title = match.group("title").strip()
                if not re.search(r"палитр|цвет", title, re.IGNORECASE):
                    continue
                body = match.group("body").strip()
                descriptors = _normalize_descriptors(body)
                entry = _build_palette_entry(title, descriptors, default_title="Палитра снимка")
                if entry:
                    palettes.append(entry)
            return palettes

        vision_palettes: list[dict[str, Any]] = []
        photo_context: list[dict[str, Any]] = []

        for asset in assets:
            varieties = asset.vision_flower_varieties or []
            for variety in varieties:
                if not kb:
                    continue
                resolved = kb.resolve_flower(variety)
                if not resolved:
                    continue
                if resolved in flower_ids:
                    continue
                flower_ids.append(resolved)
                flower_payload = kb.flowers.get(resolved) or {}
                flowers.append(
                    {
                        "id": resolved,
                        "name": flower_payload.get("name") or resolved,
                        "symbolism": flower_payload.get("symbolism"),
                    }
                )
            result_palettes: list[dict[str, Any]] = []
            if isinstance(asset.vision_results, dict):
                result_palettes = _collect_color_palettes_from_results(asset.vision_results)
                vision_palettes.extend(result_palettes)
            if asset.vision_caption and not result_palettes:
                vision_palettes.extend(_collect_color_palettes_from_caption(asset.vision_caption))

            recognized_flowers = [
                str(variety).strip()
                for variety in (asset.vision_flower_varieties or [])
                if str(variety or "").strip()
            ]

            photo_hints: list[str] = []

            def _add_hint(value: Any) -> None:
                if not value:
                    return
                text = str(value).strip()
                if not text:
                    return
                if text not in photo_hints:
                    photo_hints.append(text)

            if asset.vision_caption:
                _add_hint(asset.vision_caption)

            if isinstance(asset.vision_results, Mapping):
                results_mapping: Mapping[str, Any] = asset.vision_results
                for key in ("caption", "description", "summary"):
                    _add_hint(results_mapping.get(key))
                tags_value = results_mapping.get("tags")
                if isinstance(tags_value, (list, tuple, set)):
                    tags = [str(tag or "").strip() for tag in tags_value if str(tag or "").strip()]
                    if tags:
                        _add_hint("Теги: " + ", ".join(tags))
                display_map = {
                    "weather_final_display": "Погода: {}",
                    "season_final_display": "Сезон: {}",
                    "time_of_day_display": "Время суток: {}",
                }
                for display_key, template in display_map.items():
                    value = results_mapping.get(display_key)
                    if isinstance(value, str) and value.strip():
                        _add_hint(template.format(value.strip()))
                arch_style = results_mapping.get("arch_style")
                if isinstance(arch_style, Mapping):
                    label = str(arch_style.get("label") or "").strip()
                    confidence = arch_style.get("confidence")
                    if label:
                        if isinstance(confidence, (int, float)):
                            percent = int(round(confidence * 100))
                            _add_hint(f"Архитектура: {label} (~{percent}% уверенность)")
                        else:
                            _add_hint(f"Архитектура: {label}")
                colors_info = results_mapping.get("colors")
                if isinstance(colors_info, Mapping):
                    palettes_value = colors_info.get("palettes")
                    if isinstance(palettes_value, (list, tuple)):
                        for palette in palettes_value:
                            if not isinstance(palette, Mapping):
                                continue
                            title = str(palette.get("title") or "").strip()
                            descriptors_raw = palette.get("descriptors")
                            descriptors = [
                                str(desc or "").strip()
                                for desc in (descriptors_raw or [])
                                if str(desc or "").strip()
                            ]
                            if title and descriptors:
                                _add_hint(f"{title}: {', '.join(descriptors)}")
                            elif title:
                                _add_hint(title)
                            elif descriptors:
                                _add_hint(", ".join(descriptors))
                _add_hint(results_mapping.get("notes"))

            if len(photo_hints) > 8:
                photo_hints = photo_hints[:8]

            location = str(asset.city or "").strip()
            photo_entry: dict[str, Any] = {}
            if recognized_flowers:
                photo_entry["flowers"] = recognized_flowers
            if photo_hints:
                photo_entry["hints"] = photo_hints
            if location:
                photo_entry["location"] = location
            season_value = seasons_lookup.get(asset.id)
            if season_value:
                photo_entry["season"] = season_value
                display_value = SEASON_TRANSLATIONS.get(season_value, season_value)
                photo_entry["season_display"] = display_value
            photo_context.append(photo_entry)

        palette_cycle: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, tuple[str, ...]]] = set()

        def _register_palette(entry: dict[str, Any]) -> None:
            descriptors = entry.get("descriptors") or []
            if not isinstance(descriptors, list) or not descriptors:
                return
            title_value = str(entry.get("title") or "").strip()
            key = (
                title_value.casefold(),
                tuple(descriptor.casefold() for descriptor in descriptors),
            )
            if key in seen_keys:
                return
            seen_keys.add(key)
            palette_cycle.append(entry)

        for entry in vision_palettes:
            _register_palette(entry)
        weather_class: str | None = None
        if weather_block and isinstance(weather_block, dict):
            city_snapshot = weather_block.get("city") or {}
            weather_class = city_snapshot.get("weather_class")
            if not weather_class:
                weather_code = city_snapshot.get("weather_code")
                weather_class = self._classify_weather_code(weather_code)
            if not weather_class:
                condition_value = city_snapshot.get("weather_condition")
                weather_class = Bot._normalize_weather_enum(condition_value)
            if not weather_class:
                today_metrics = weather_block.get("today")
                if isinstance(today_metrics, Mapping):
                    condition_metric = today_metrics.get("condition")
                    weather_class = Bot._normalize_weather_enum(condition_metric)
            if weather_class is None and (
                weather_block.get("today") or weather_block.get("yesterday")
            ):
                weather_class = "overcast"
        season_key: str | None = None
        season_description: str | None = None
        if kb and kb.seasons:
            now = datetime.utcnow()
            month = now.month
            for key, payload in kb.seasons.items():
                months = payload.get("months") or []
                if month in months:
                    season_key = str(key)
                    season_description = payload.get("description")
                    break
        holiday_note: dict[str, str] | None = None
        if kb and kb.holidays:
            today_key = datetime.utcnow().strftime("%m-%d")
            holiday_payload = kb.holidays.get(today_key)
            if isinstance(holiday_payload, dict):
                title = holiday_payload.get("title")
                note = holiday_payload.get("note")
                if title or note:
                    holiday_note = {"title": title or "", "note": note or ""}
        wisdom: dict[str, str] | None = None
        if kb and kb.wisdom:
            wisdom = seed_rng.choice(kb.wisdom)
        engagement: dict[str, str] | None = None
        if kb and kb.micro_engagement:
            engagement = seed_rng.choice(kb.micro_engagement)
        return {
            "has_photo": has_photo,
            "flowers": flowers,
            "flower_ids": flower_ids,
            "palettes": palette_cycle,
            "photo_context": photo_context,
            "weather": weather_class,
            "season": season_key,
            "season_description": season_description,
            "holiday": holiday_note,
            "wisdom": wisdom,
            "engagement": engagement,
        }

    def _render_flower_pattern(
        self,
        pattern: FlowerPattern,
        *,
        features: dict[str, Any],
        kb: FlowerKnowledgeBase | None,
        rng: random.Random,
    ) -> dict[str, Any] | None:
        if not kb:
            return None
        template = pattern.template
        if not template:
            return None
        payload: dict[str, Any] = {
            "id": pattern.id,
            "kind": pattern.kind,
            "photo_dependent": pattern.photo_dependent,
        }
        if pattern.kind == "color":
            palettes = features.get("palettes") or []
            if not palettes:
                return None
            palette = rng.choice(palettes)
            descriptors = palette.get("descriptors") or []
            if descriptors:
                sample = rng.sample(descriptors, k=min(2, len(descriptors)))
            else:
                sample = []
            payload["instruction"] = template.format(
                palette_title=palette.get("title") or "палитра утра",
                palette_descriptors=(
                    ", ".join(sample) if sample else palette.get("title") or "ласковый свет"
                ),
            )
        elif pattern.kind == "tradition":
            flowers = features.get("flowers") or []
            if not flowers:
                return None
            flower_entry = rng.choice(flowers)
            notes = (
                kb.traditions.get(flower_entry["id"], {}).get("notes")
                if flower_entry.get("id")
                else None
            )
            if not notes:
                return None
            payload["instruction"] = template.format(
                flower_name=flower_entry.get("name") or "цветы",
                tradition_note=rng.choice(notes),
            )
        elif pattern.kind == "micro_engagement":
            engagement = features.get("engagement") or {}
            prompt_text = engagement.get("text") if isinstance(engagement, dict) else None
            if not prompt_text and kb.micro_engagement:
                prompt_text = rng.choice(kb.micro_engagement).get("text")
            if not prompt_text:
                return None
            payload["instruction"] = template.format(prompt=prompt_text)
        elif pattern.kind == "wisdom":
            wisdom = features.get("wisdom") or {}
            text = wisdom.get("text") if isinstance(wisdom, dict) else None
            if not text and kb.wisdom:
                text = rng.choice(kb.wisdom).get("text")
            if not text:
                return None
            payload["instruction"] = template.format(wisdom_text=text)
        elif pattern.kind == "texture":
            palettes = features.get("palettes") or []
            mood = None
            if palettes:
                palette = rng.choice(palettes)
                mood = palette.get("mood")
            if not mood:
                mood = "уют"
            payload["instruction"] = template.format(texture_mood=mood)
        elif pattern.kind == "weather":
            payload["instruction"] = template
        else:
            payload["instruction"] = template
        if not payload.get("instruction"):
            return None
        return payload

    def _build_flowers_plan(
        self,
        rubric: Rubric,
        assets: Sequence[Asset],
        weather_block: dict[str, Any] | None,
        *,
        channel_id: int | None,
        instructions: str | None = None,
        asset_seasons: Mapping[int, str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        kb = self.flowers_kb
        seed = self._flowers_seed(channel_id)
        rng = random.Random(seed)
        features = self._extract_flower_features(
            assets,
            weather_block,
            seed_rng=rng,
            asset_seasons=asset_seasons,
        )
        photo_context = features.get("photo_context") or []
        _banned_recent, consecutive_repeats, pattern_history = self._flowers_recent_pattern_ids(
            rubric
        )
        available: list[FlowerPattern] = []
        weather_pattern: FlowerPattern | None = None
        if kb:
            for pattern in kb.patterns:
                if not pattern.matches_context(features):
                    continue
                if pattern.id in consecutive_repeats and not pattern.always_include:
                    continue
                if pattern.always_include and pattern.kind == "weather":
                    weather_pattern = pattern
                available.append(pattern)
        selected: list[dict[str, Any]] = []
        pattern_ids: list[str] = []
        if weather_pattern:
            rendered = self._render_flower_pattern(
                weather_pattern,
                features=features,
                kb=kb,
                rng=rng,
            )
            if rendered:
                selected.append(rendered)
                pattern_ids.append(weather_pattern.id)
        pool = [pattern for pattern in available if pattern is not weather_pattern]
        candidate_order = self._flowers_rotation_order(pool, pattern_history, weather_pattern)
        for choice_pattern in candidate_order:
            rendered = self._render_flower_pattern(
                choice_pattern,
                features=features,
                kb=kb,
                rng=rng,
            )
            if not rendered:
                continue
            selected.append(rendered)
            pattern_ids.append(choice_pattern.id)
            break
        cities = sorted({asset.city for asset in assets if asset.city})

        def _filter_metrics(payload: Mapping[str, Any] | None) -> dict[str, Any]:
            if not isinstance(payload, Mapping):
                return {}
            allowed_scalar_keys = ("temperature", "wind_speed", "condition")
            filtered: dict[str, Any] = {}
            for key in allowed_scalar_keys:
                value = payload.get(key)
                if value is not None:
                    filtered[key] = value
            parts_value = payload.get("parts")
            if isinstance(parts_value, Mapping):
                normalized_parts: dict[str, str] = {}
                for part_name, part_condition in parts_value.items():
                    if not isinstance(part_name, str):
                        continue
                    if isinstance(part_condition, str):
                        trimmed = part_condition.strip()
                        if trimmed:
                            normalized_parts[part_name] = trimmed
                if normalized_parts:
                    filtered["parts"] = normalized_parts
            return filtered

        if weather_block:
            today_metrics = (
                weather_block.get("today") if isinstance(weather_block.get("today"), dict) else {}
            )
            yesterday_metrics = (
                weather_block.get("yesterday")
                if isinstance(weather_block.get("yesterday"), dict)
                else {}
            )
            weather_summary = {
                "cities": weather_block.get("cities"),
                "today": _filter_metrics(today_metrics),
                "yesterday": _filter_metrics(yesterday_metrics),
            }
            if isinstance(weather_block.get("sea"), dict):
                weather_summary["sea"] = weather_block.get("sea")
            city_snapshot = weather_block.get("city")
            if isinstance(city_snapshot, dict) and city_snapshot.get("name"):
                weather_summary["city_name"] = city_snapshot.get("name")
        else:
            weather_summary = {
                "cities": None,
                "today": {},
                "yesterday": {},
            }
        flower_entries: list[dict[str, Any]] = []
        for flower in features.get("flowers") or []:
            if not isinstance(flower, dict):
                continue
            name = str(flower.get("name") or "").strip()
            if not name:
                continue
            entry: dict[str, Any] = {"name": name}
            if flower.get("id"):
                entry["id"] = flower.get("id")
            flower_entries.append(entry)
        previous_text: str | None = None
        rubric_id = getattr(rubric, "id", None)
        if isinstance(rubric_id, int):
            previous_text, _ = self._lookup_previous_flowers_post(rubric_id)
        plan = {
            "patterns": selected,
            "weather": weather_summary,
            "flowers": flower_entries,
            "previous_text": previous_text,
            "instructions": instructions or "",
            "photo_context": photo_context,
        }
        banned_words = sorted((kb.banned_words if kb else set()) or [])
        plan_meta = {
            "pattern_ids": pattern_ids,
            "banned_words": banned_words,
            "length": {"min": 420, "max": 520},
            "cities": cities,
        }
        return plan, plan_meta

    def _flowers_contains_banned_word(self, text: str, banned_words: Iterable[str]) -> bool:
        lowered = text.casefold()
        tokens = {token.strip() for token in re.split(r"\W+", lowered) if token.strip()}
        for word in banned_words:
            normalized = str(word).strip().casefold()
            if not normalized:
                continue
            if normalized in tokens:
                return True
            pattern = re.escape(normalized)
            if re.search(rf"(?<!\w){pattern}(?!\w)", lowered):
                return True
        return False

    @staticmethod
    def _jaccard_similarity(a: str, b: str) -> float:
        words_a = [token for token in re.split(r"\W+", a.lower()) if token]
        words_b = [token for token in re.split(r"\W+", b.lower()) if token]
        bigrams_a = {f"{words_a[i]} {words_a[i + 1]}" for i in range(len(words_a) - 1)}
        bigrams_b = {f"{words_b[i]} {words_b[i + 1]}" for i in range(len(words_b) - 1)}
        if not bigrams_a or not bigrams_b:
            return 0.0
        intersection = len(bigrams_a & bigrams_b)
        union = len(bigrams_a | bigrams_b)
        if union == 0:
            return 0.0
        return intersection / union

    def _latest_flowers_text(self, rubric: Rubric) -> str | None:
        recent = self.data.get_recent_rubric_metadata(rubric.code, limit=1)
        if not recent:
            return None
        entry = recent[0] or {}
        text = entry.get("greeting") or entry.get("text")
        if not text:
            return None
        return str(text)

    async def _prepare_flowers_drop(
        self,
        rubric: Rubric,
        *,
        job: Job | None = None,
        instructions: str | None = None,
        channel_id: int | None = None,
    ) -> (
        tuple[
            list[Asset],
            list[int],
            list[str],
            list[str],
            list[str],
            str,
            list[str],
            dict[int, dict[str, Any]],
            dict[str, Any] | None,
            dict[str, Any],
            dict[str, Any],
        ]
        | None
    ):
        min_count, max_count = self._resolve_flowers_asset_limits(rubric)
        assets = self.data.fetch_assets_by_vision_category(
            "flowers",
            rubric_id=rubric.id,
            limit=max_count,
            random_order=True,
        )
        candidate_assets = list(assets)
        if len(candidate_assets) < min_count:
            logging.warning(
                "Not enough assets for flowers rubric: have %s, need %s",
                len(candidate_assets),
                min_count,
            )
        if not candidate_assets:
            logging.warning("No assets available for flowers rubric")
            return None
        max_candidate_count = min(len(candidate_assets), max_count)
        if max_candidate_count <= 0:
            logging.warning("Unable to select flowers assets: empty candidate list")
            return None
        attempt_limit = 10
        selected_assets: list[Asset] | None = None
        asset_seasons: dict[int, str] | None = None
        selected_target: int | None = None
        total_attempts = 0
        recorded_attempts: int | None = None
        for target_count in range(max_candidate_count, 0, -1):
            attempts = 0
            while attempts < attempt_limit:
                attempts += 1
                total_attempts += 1
                if attempts == 1:
                    combination = candidate_assets[:]
                else:
                    subset_size = min(len(candidate_assets), max_count)
                    if subset_size <= 0:
                        break
                    if subset_size == len(candidate_assets):
                        combination = candidate_assets[:]
                        random.shuffle(combination)
                    else:
                        combination = random.sample(candidate_assets, k=subset_size)
                filtered = self._filter_flower_assets_by_season(
                    combination,
                    desired_count=target_count,
                    max_count=target_count,
                )
                if filtered:
                    selected_assets, asset_seasons = filtered
                    selected_target = target_count
                    recorded_attempts = total_attempts
                    break
            if selected_assets:
                break
        if not selected_assets or not asset_seasons:
            logging.warning(
                "Unable to assemble seasonal-consistent flowers assets: have seasons %s",
                sorted(
                    {self._resolve_asset_season(asset) or "unknown" for asset in candidate_assets}
                ),
            )
            return None
        if selected_target is not None and selected_target < min_count:
            logging.info(
                "Falling back to %s flowers assets below configured minimum %s",
                selected_target,
                min_count,
            )
        assets = selected_assets
        file_ids: list[str] = []
        asset_kinds: list[str] = []
        conversion_map: dict[int, dict[str, Any]] = {}
        for asset in assets:
            file_id = asset.file_id
            if not file_id:
                logging.warning("Asset %s missing file_id", asset.id)
                return None
            media_kind, needs_upload = self._asset_media_kind(asset)
            if needs_upload:
                conversion_map[asset.id] = {
                    "file_id": file_id,
                    "file_unique_id": asset.file_unique_id,
                    "file_name": asset.file_name,
                    "mime_type": asset.mime_type,
                    "file_size": asset.file_size,
                    "width": asset.width,
                    "height": asset.height,
                }
            file_ids.append(file_id)
            asset_kinds.append(media_kind)
        cities = sorted({asset.city for asset in assets if asset.city})
        weather_block = self._compose_flowers_weather_block(cities, rubric)
        plan, plan_meta = self._build_flowers_plan(
            rubric,
            assets,
            weather_block,
            channel_id=channel_id,
            instructions=instructions,
            asset_seasons=asset_seasons,
        )
        if isinstance(plan_meta, dict):
            plan_meta["photo_selection_attempts"] = recorded_attempts or total_attempts
        greeting, hashtags, plan, plan_meta = await self._generate_flowers_copy(
            rubric,
            assets,
            channel_id=channel_id,
            weather_block=weather_block,
            job=job,
            instructions=instructions,
            plan=plan,
            plan_meta=plan_meta,
        )
        asset_ids = [asset.id for asset in assets]
        return (
            assets,
            asset_ids,
            file_ids,
            asset_kinds,
            cities,
            greeting,
            hashtags,
            conversion_map,
            weather_block,
            plan,
            plan_meta,
        )

    async def _send_flowers_media_bundle(
        self,
        *,
        chat_id: int,
        assets: list[Asset],
        file_ids: list[str],
        asset_kinds: list[str],
        caption: str | None,
        parse_mode: str | None = None,
        conversion_map: dict[int, dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
        conversion_map = dict(conversion_map or {})
        cleanup_paths: list[str] = []
        file_handles: dict[str, Any] = {}
        response: dict[str, Any] | None = None

        def _update_asset_after_conversion(
            asset: Asset,
            original_file_id: str,
            result_payload: dict[str, Any] | None,
        ) -> str | None:
            if not isinstance(result_payload, dict):
                return None
            photo_meta = self._extract_photo_file_meta(result_payload.get("photo"))
            if not photo_meta or not photo_meta.get("file_id"):
                return None
            new_file_id_raw = photo_meta.get("file_id")
            if not new_file_id_raw:
                return None
            new_file_id = str(new_file_id_raw)
            self.data.update_asset(
                asset.id,
                kind="photo",
                file_meta=photo_meta,
                metadata={"original_document_file_id": original_file_id},
            )
            asset.payload["kind"] = "photo"
            asset.payload["file_id"] = new_file_id
            file_unique = photo_meta.get("file_unique_id")
            if file_unique is not None:
                asset.payload["file_unique_id"] = file_unique
            mime_type = photo_meta.get("mime_type")
            if mime_type is not None:
                asset.payload["mime_type"] = mime_type
            file_size = photo_meta.get("file_size")
            if file_size is not None:
                asset.payload["file_size"] = file_size
            asset.width = Asset._to_int(photo_meta.get("width"))
            asset.height = Asset._to_int(photo_meta.get("height"))
            return new_file_id

        try:
            if not assets:
                return {"ok": True, "result": []}, conversion_map
            asset_count = len(assets)
            if asset_count == 1:
                asset = assets[0]
                file_id = file_ids[0] if file_ids else asset.file_id or ""
                media_kind = asset_kinds[0] if asset_kinds else "photo"
                needs_conversion = asset.id in conversion_map
                payload: dict[str, Any] = {"chat_id": chat_id}
                if caption:
                    payload["caption"] = caption
                if parse_mode and caption:
                    payload["parse_mode"] = parse_mode
                if media_kind == "photo" and needs_conversion:
                    source_meta = conversion_map.get(asset.id) or {}
                    source_file_id = source_meta.get("file_id") or file_id
                    file_meta = {
                        "file_id": source_file_id,
                        "file_unique_id": asset.file_unique_id,
                        "file_name": asset.file_name,
                        "mime_type": asset.mime_type,
                    }
                    target_path = self._build_local_file_path(asset.id, file_meta)
                    downloaded_path = await self._download_file(source_file_id, target_path)
                    if not downloaded_path:
                        raise RuntimeError("Failed to download asset for conversion")
                    send_path, cleanup_path, filename, content_type, _, _, _ = (
                        self._prepare_photo_for_upload(str(downloaded_path))
                    )
                    cleanup_paths.append(str(downloaded_path))
                    if cleanup_path is not None and str(cleanup_path) != str(downloaded_path):
                        cleanup_paths.append(str(cleanup_path))
                    with open(send_path, "rb") as fh:
                        files = {"photo": (filename, fh, content_type)}
                        response = await self.api_request_multipart(
                            "sendPhoto",
                            payload,
                            files=files,
                        )
                    if response.get("ok"):
                        new_file_id = _update_asset_after_conversion(
                            asset,
                            source_file_id,
                            response.get("result"),
                        )
                        if new_file_id:
                            file_ids[0] = new_file_id
                        asset_kinds[0] = "photo"
                        conversion_map.pop(asset.id, None)
                    return response, conversion_map
                if media_kind == "photo":
                    payload["photo"] = file_id
                    response = await self.api_request("sendPhoto", payload)
                    return response, conversion_map
                payload["document"] = file_id
                response = await self.api_request("sendDocument", payload)
                return response, conversion_map

            media_payload: list[dict[str, Any]] = []
            attachments: dict[str, tuple[str, Any, str]] = {}
            for idx, asset in enumerate(assets):
                file_id = file_ids[idx] if idx < len(file_ids) else asset.file_id or ""
                media_kind = asset_kinds[idx] if idx < len(asset_kinds) else "photo"
                needs_conversion = asset.id in conversion_map and media_kind == "photo"
                item: dict[str, Any] = {"type": "photo" if media_kind == "photo" else "document"}
                if idx == 0 and caption:
                    item["caption"] = caption
                    if parse_mode:
                        item["parse_mode"] = parse_mode
                if needs_conversion:
                    source_meta = conversion_map.get(asset.id) or {}
                    source_file_id = source_meta.get("file_id") or file_id
                    file_meta = {
                        "file_id": source_file_id,
                        "file_unique_id": asset.file_unique_id,
                        "file_name": asset.file_name,
                        "mime_type": asset.mime_type,
                    }
                    target_path = self._build_local_file_path(asset.id, file_meta)
                    downloaded_path = await self._download_file(source_file_id, target_path)
                    if not downloaded_path:
                        raise RuntimeError("Failed to download asset for conversion")
                    send_path, cleanup_path, filename, content_type, _, _, _ = (
                        self._prepare_photo_for_upload(str(downloaded_path))
                    )
                    cleanup_paths.append(str(downloaded_path))
                    if cleanup_path is not None and str(cleanup_path) != str(downloaded_path):
                        cleanup_paths.append(str(cleanup_path))
                    attach_name = f"photo{idx}"
                    fh = open(send_path, "rb")
                    attachments[attach_name] = (filename, fh, content_type)
                    file_handles[attach_name] = fh
                    item["media"] = f"attach://{attach_name}"
                else:
                    item["media"] = file_id
                media_payload.append(item)

            if attachments:
                response = await self.api_request_multipart(
                    "sendMediaGroup",
                    {"chat_id": chat_id, "media": media_payload},
                    files=attachments,
                )
            else:
                response = await self.api_request(
                    "sendMediaGroup",
                    {"chat_id": chat_id, "media": media_payload},
                )
            if response and response.get("ok") and isinstance(response.get("result"), list):
                results_list = response.get("result")
                for idx, asset in enumerate(assets):
                    if asset.id not in conversion_map:
                        continue
                    if idx >= len(results_list):
                        continue
                    new_file_id = _update_asset_after_conversion(
                        asset,
                        conversion_map[asset.id].get("file_id") or file_ids[idx],
                        results_list[idx] if isinstance(results_list[idx], dict) else None,
                    )
                    if new_file_id:
                        file_ids[idx] = new_file_id
                for asset in assets:
                    conversion_map.pop(asset.id, None)
            return response or {"ok": False}, conversion_map
        finally:
            for fh in file_handles.values():
                with contextlib.suppress(Exception):
                    fh.close()
            for path in cleanup_paths:
                self._remove_file(path)

    def _build_flowers_caption(
        self,
        greeting: str,
        cities: Sequence[str],
        hashtags: Sequence[str],
        weather_block: dict[str, Any] | None = None,
    ) -> tuple[str, str, str | None, list[str]]:
        greeting_text = str(greeting or "").strip()
        city_hashtags, trailing_hashtags = self._prepare_flowers_hashtag_sections(
            cities,
            hashtags,
        )
        caption_parts: list[str] = []
        if greeting_text:
            caption_parts.append(greeting_text)
        if city_hashtags:
            caption_parts.append(" ".join(city_hashtags))
        if trailing_hashtags:
            caption_parts.append(" ".join(trailing_hashtags))
        preview_caption = "\n\n".join(caption_parts)
        publish_caption, parse_mode = self._build_flowers_publish_caption(preview_caption)
        combined_hashtags = city_hashtags + trailing_hashtags
        return preview_caption, publish_caption, parse_mode, combined_hashtags

    def _prepare_flowers_hashtag_sections(
        self,
        cities: Sequence[str],
        hashtags: Sequence[str],
    ) -> tuple[list[str], list[str]]:
        city_hashtags: list[str] = []
        seen_city: set[str] = set()
        for city in cities:
            normalized = self._normalize_city_hashtag(city)
            if not normalized:
                continue
            key = normalized.casefold()
            if key in seen_city:
                continue
            seen_city.add(key)
            city_hashtags.append(normalized)
        prepared_hashtags = self._prepare_hashtags(hashtags)
        trailing_hashtags: list[str] = []
        seen_other: set[str] = set()
        for tag in prepared_hashtags:
            key = tag.casefold()
            if key in seen_city or key in seen_other:
                continue
            seen_other.add(key)
            trailing_hashtags.append(tag)
        return city_hashtags, trailing_hashtags

    def _normalize_city_hashtag(self, city: str | None) -> str | None:
        text = unicodedata.normalize("NFKC", str(city or "")).strip()
        if not text:
            return None
        text = text.replace("ё", "е").replace("Ё", "Е")
        lowered = text.casefold()
        cleaned = re.sub(r"[^\w]+", "", lowered, flags=re.UNICODE)
        if not cleaned:
            return None
        return f"#{cleaned}"

    def _build_flowers_publish_caption(self, preview_caption: str) -> tuple[str, str | None]:
        preview_text = str(preview_caption or "").strip()
        link = '<a href="https://t.me/addlist/sW-rkrslxqo1NTVi">📂 Полюбить 39</a>'
        parse_mode = "HTML"
        if not preview_text:
            return link, parse_mode
        escaped = html.escape(preview_text)
        return f"{escaped}\n\n{link}", parse_mode

    async def _publish_flowers(
        self,
        rubric: Rubric,
        channel_id: int,
        *,
        test: bool = False,
        job: Job | None = None,
        initiator_id: int | None = None,
        instructions: str | None = None,
    ) -> bool:
        min_count, _ = self._resolve_flowers_asset_limits(rubric)
        prepared = await self._prepare_flowers_drop(
            rubric,
            job=job,
            instructions=instructions,
            channel_id=channel_id,
        )
        if not prepared:
            if initiator_id is not None:
                title = rubric.title or rubric.code
                await self.api_request(
                    "sendMessage",
                    {
                        "chat_id": initiator_id,
                        "text": (
                            f"Для рубрики «{title}» не набралось минимальное количество "
                            f"фото (нужно {min_count}). Добавьте новые и повторите попытку."
                        ),
                    },
                )
                return True
            return False
        (
            assets,
            asset_ids,
            file_ids,
            asset_kinds,
            cities,
            greeting,
            hashtags,
            conversion_map,
            weather_block,
            plan,
            plan_meta,
        ) = prepared
        (
            preview_caption,
            publish_caption,
            publish_parse_mode,
            hashtag_list,
        ) = self._build_flowers_caption(
            greeting,
            cities,
            hashtags,
            weather_block,
        )
        if initiator_id is not None:
            await self._send_flowers_preview(
                rubric,
                initiator_id,
                default_channel=channel_id,
                test_requested=test,
                assets=assets,
                asset_ids=asset_ids,
                file_ids=file_ids,
                asset_kinds=asset_kinds,
                conversion_map=conversion_map,
                cities=cities,
                greeting=greeting,
                hashtags=hashtags,
                preview_caption=preview_caption,
                publish_caption=publish_caption,
                publish_parse_mode=publish_parse_mode,
                prepared_hashtags=hashtag_list,
                instructions=instructions,
                weather_block=weather_block,
                plan=plan,
                plan_meta=plan_meta,
            )
            return True
        response, _ = await self._send_flowers_media_bundle(
            chat_id=channel_id,
            assets=assets,
            file_ids=file_ids,
            asset_kinds=asset_kinds,
            caption=publish_caption,
            parse_mode=publish_parse_mode,
            conversion_map=conversion_map,
        )
        if not response.get("ok"):
            logging.error("Failed to publish flowers rubric: %s", response)
            return False
        result_payload = response.get("result")
        message_id: int
        if isinstance(result_payload, list) and result_payload:
            message_id = int(result_payload[0].get("message_id") or 0)
        elif isinstance(result_payload, dict):
            message_id = int(result_payload.get("message_id") or 0)
        else:
            message_id = 0
        self.data.mark_assets_used(asset_ids)
        previous_weather_line: str | None = None
        rubric_id = getattr(rubric, "id", None)
        if isinstance(rubric_id, int):
            _, previous_weather_line = self._lookup_previous_flowers_post(rubric_id)
        today_line_source: str | None = None
        yesterday_line_source: str | None = None
        if isinstance(weather_block, dict):
            location_label = self._extract_weather_location_label(weather_block)
            today_metrics_line = self._compose_weather_metrics_preview_line(
                weather_block.get("today"),
                label="Сегодня",
            )
            yesterday_metrics_line = self._compose_weather_metrics_preview_line(
                weather_block.get("yesterday"),
                label="Вчера",
            )
            if location_label:
                today_line_source = f"{location_label}: {today_metrics_line}"
                yesterday_line_source = f"{location_label}: {yesterday_metrics_line}"
            else:
                today_line_source = today_metrics_line
                yesterday_line_source = yesterday_metrics_line
        if not yesterday_line_source and previous_weather_line:
            yesterday_line_source = previous_weather_line
        weather_today_line = self._normalize_weather_preview_line(today_line_source)
        weather_yesterday_line = self._normalize_weather_preview_line(yesterday_line_source)
        metadata = {
            "rubric_code": rubric.code,
            "asset_ids": asset_ids,
            "test": test,
            "cities": cities,
            "greeting": greeting,
            "hashtags": hashtag_list,
            "weather": weather_block,
            "weather_today_line": weather_today_line,
            "weather_yesterday_line": weather_yesterday_line,
            "weather_line": weather_today_line,
            "pattern_ids": (
                list(plan_meta.get("pattern_ids", [])) if isinstance(plan_meta, dict) else None
            ),
            "plan": plan,
        }
        self.data.record_post_history(
            channel_id,
            message_id,
            assets[0].id if assets else None,
            rubric.id,
            metadata,
        )
        await self._cleanup_assets(assets)
        return True

    def _resolve_flowers_target(self, state: dict[str, Any], *, to_test: bool) -> int | None:
        key = "test_channel_id" if to_test else "channel_id"
        value = state.get(key)
        if isinstance(value, int):
            return value
        default_type = state.get("default_channel_type")
        default_value = state.get("default_channel_id")
        if isinstance(default_value, int) and (
            (to_test and default_type == "test") or (not to_test and default_type == "main")
        ):
            return default_value
        return None

    def _flowers_preview_keyboard(self, state: dict[str, Any]) -> dict[str, Any]:
        rows: list[list[dict[str, Any]]] = [
            [
                {
                    "text": "♻️ Перегенерировать фото",
                    "callback_data": "flowers_preview:regen_photos",
                }
            ],
            [
                {
                    "text": "✍️ Перегенерировать подпись",
                    "callback_data": "flowers_preview:regen_caption",
                },
                {
                    "text": "✍️➕ Инструкция",
                    "callback_data": "flowers_preview:instruction",
                },
            ],
            [
                {
                    "text": "⬇️ Скачать инструкцию",
                    "callback_data": "flowers_preview:download_prompt",
                }
            ],
        ]
        send_row: list[dict[str, Any]] = []
        if self._resolve_flowers_target(state, to_test=True) is not None:
            send_row.append({"text": "🧪 В тест", "callback_data": "flowers_preview:send_test"})
        if self._resolve_flowers_target(state, to_test=False) is not None:
            send_row.append({"text": "📣 В канал", "callback_data": "flowers_preview:send_main"})
        if send_row:
            rows.append(send_row)
        rows.append([{"text": "✖️ Отмена", "callback_data": "flowers_preview:cancel"}])
        return {"inline_keyboard": rows}

    @staticmethod
    def _normalize_weather_preview_line(value: Any) -> str:
        line = str(value or "").strip()
        return line if line else "не публиковалось"

    def _lookup_previous_flowers_post(self, rubric_id: int) -> tuple[str | None, str | None]:
        previous_text: str | None = None
        previous_weather: str | None = None
        try:
            rows = self.db.execute(
                """
                SELECT metadata
                FROM posts_history
                WHERE rubric_id=?
                ORDER BY published_at DESC, id DESC
                LIMIT 15
                """,
                (rubric_id,),
            ).fetchall()
        except Exception:
            rows = []
        for row in rows:
            raw = row["metadata"] if row is not None else None
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if payload.get("test"):
                continue
            if previous_text is None:
                text = payload.get("greeting") or payload.get("text") or payload.get("caption")
                if text:
                    previous_text = str(text)
            if previous_weather is None:
                candidate = payload.get("weather_today_line") or payload.get("weather_line")
                candidate_str = str(candidate or "").strip()
                if candidate_str:
                    previous_weather = candidate_str
            if previous_text and previous_weather:
                break
        return previous_text, previous_weather

    @staticmethod
    def _safe_preview_truncate(text: str, limit: int) -> str:
        if limit <= 0:
            return ""
        if len(text) <= limit:
            return text
        ellipsis = "\u2026"
        available = max(limit - len(ellipsis), 0)
        truncated = text[:available]
        last_amp = truncated.rfind("&")
        last_semicolon = truncated.rfind(";")
        if last_amp > last_semicolon:
            truncated = truncated[:last_amp]
        last_lt = truncated.rfind("<")
        last_gt = truncated.rfind(">")
        if last_lt > last_gt:
            truncated = truncated[:last_lt]
        return truncated.rstrip() + ellipsis

    def _render_flowers_preview_text(self, state: dict[str, Any]) -> str:
        @dataclass
        class _PreviewSection:
            text: str
            priority: int
            fallback: str | None = None
            used_fallback: bool = False

        sections: list[_PreviewSection] = []

        def _add_section(text: str, priority: int, *, fallback: str | None = None) -> None:
            normalized = text.strip("\n")
            if not normalized:
                return
            sections.append(_PreviewSection(normalized, priority, fallback))

        def _escape_block_line(text: str) -> str:
            return html.escape(text)

        caption = str(state.get("preview_caption") or "").strip()
        if caption:
            _add_section("Подпись на медиа показана выше.", 0)
        else:
            _add_section("Подпись пока не сгенерирована.", 0)
        weather_today_line = self._normalize_weather_preview_line(state.get("weather_today_line"))
        weather_yesterday_line = self._normalize_weather_preview_line(
            state.get("weather_yesterday_line")
        )

        def _ensure_weather_html(key: str, value: str) -> str:
            html_key = f"{key}_html"
            cached_html = state.get(html_key)
            cached_value = state.get(key)
            if (
                isinstance(cached_html, str)
                and isinstance(cached_value, str)
                and cached_value == value
            ):
                return cached_html
            state[key] = value
            escaped = html.escape(value)
            state[html_key] = escaped
            return escaped

        weather_today_html = _ensure_weather_html("weather_today_line", weather_today_line)
        weather_yesterday_html = _ensure_weather_html(
            "weather_yesterday_line", weather_yesterday_line
        )

        _add_section(f"Погода сегодня: {weather_today_html}", 0)
        _add_section(f"Погода вчера: {weather_yesterday_html}", 0)
        instructions = str(state.get("instructions") or "").strip()
        if instructions:
            escaped_instructions = _escape_block_line(instructions)
            instructions_block = (
                "Инструкции оператора:\n"
                f'<blockquote expandable="true">{escaped_instructions}</blockquote>'
            )
            fallback_block: str | None = None
            if len(instructions_block) > 600:
                truncated = self._safe_preview_truncate(instructions, 600)
                escaped_truncated = _escape_block_line(truncated)
                fallback_block = (
                    "Инструкции оператора:\n"
                    f'<blockquote expandable="true">{escaped_truncated}</blockquote>'
                )
                if fallback_block == instructions_block:
                    fallback_block = None
            _add_section(instructions_block, 2, fallback=fallback_block)
        attempts_raw = state.get("photo_selection_attempts")
        attempts_count: int | None = None
        if isinstance(attempts_raw, int):
            attempts_count = attempts_raw
        elif isinstance(attempts_raw, float):
            if math.isfinite(attempts_raw):
                attempts_count = int(round(attempts_raw))
        elif isinstance(attempts_raw, str):
            try:
                attempts_count = int(attempts_raw.strip())
            except ValueError:
                attempts_count = None
        if attempts_count and attempts_count > 0:
            _add_section(f"Попытки подбора фото: {attempts_count}", 2)
        channels: list[str] = []
        main_target = self._resolve_flowers_target(state, to_test=False)
        if main_target is not None:
            channels.append(f"📣 {main_target}")
        test_target = self._resolve_flowers_target(state, to_test=True)
        if test_target is not None:
            channels.append(f"🧪 {test_target}")
        if channels:
            _add_section("Доступные каналы: " + ", ".join(channels), 1)
        plan_raw = state.get("plan")
        plan_dict = plan_raw if isinstance(plan_raw, dict) else {}

        pattern_lines: list[str] = []
        for idx, raw_pattern in enumerate(plan_dict.get("patterns") or [], 1):
            if not isinstance(raw_pattern, dict):
                continue
            instruction = str(raw_pattern.get("instruction") or "").strip()
            if not instruction:
                continue
            tags: list[str] = []
            kind = str(raw_pattern.get("kind") or "").strip()
            if kind:
                tags.append(kind)
            if raw_pattern.get("photo_dependent"):
                tags.append("про фото")
            tag_prefix = f"[{', '.join(tags)}] " if tags else ""
            pattern_lines.append(f"{idx}. {tag_prefix}{instruction}")
        if pattern_lines:
            escaped_patterns = "\n".join(_escape_block_line(line) for line in pattern_lines)
            pattern_block = (
                f"Паттерны:\n" f'<blockquote expandable="true">{escaped_patterns}</blockquote>'
            )
            _add_section(pattern_block, 3)

        weather_lines: list[str] = []
        seen_weather: set[str] = set()

        def _add_weather_line(value: str) -> None:
            normalized = value.strip()
            if not normalized or normalized in seen_weather:
                return
            seen_weather.add(normalized)
            weather_lines.append(normalized)

        plan_weather = plan_dict.get("weather")
        if isinstance(plan_weather, dict):
            location_label = self._extract_weather_location_label(plan_weather)
            today_metrics = plan_weather.get("today")
            if isinstance(today_metrics, Mapping) and today_metrics:
                today_line = self._compose_weather_metrics_preview_line(
                    today_metrics,
                    label="Сегодня",
                )
                _add_weather_line(
                    today_line if not location_label else f"{location_label}: {today_line}"
                )
            yesterday_metrics = plan_weather.get("yesterday")
            if isinstance(yesterday_metrics, Mapping) and yesterday_metrics:
                yesterday_line = self._compose_weather_metrics_preview_line(
                    yesterday_metrics,
                    label="Вчера",
                )
                _add_weather_line(
                    yesterday_line if not location_label else f"{location_label}: {yesterday_line}"
                )
            sea_info = plan_weather.get("sea")
            if isinstance(sea_info, Mapping) and sea_info:
                sea_parts: list[str] = []
                temp = sea_info.get("temperature")
                if isinstance(temp, (int, float)):
                    sea_parts.append(f"вода {int(round(temp))}°C")
                wave = sea_info.get("wave")
                if isinstance(wave, (int, float)):
                    sea_parts.append(f"волна {round(float(wave), 1)} м")
                if sea_parts:
                    _add_weather_line("Море: " + ", ".join(sea_parts))

        weather_details_raw = state.get("weather_details")
        weather_details = weather_details_raw if isinstance(weather_details_raw, dict) else {}
        if weather_details:
            details_location = self._extract_weather_location_label(weather_details)
            if (
                (not isinstance(plan_weather, Mapping))
                or not isinstance(plan_weather.get("today"), Mapping)
                or not plan_weather.get("today")
            ):
                today_metrics = weather_details.get("today")
                if isinstance(today_metrics, Mapping) and today_metrics:
                    today_line = self._compose_weather_metrics_preview_line(
                        today_metrics,
                        label="Сегодня",
                    )
                    _add_weather_line(
                        today_line if not details_location else f"{details_location}: {today_line}"
                    )
            if (
                (not isinstance(plan_weather, Mapping))
                or not isinstance(plan_weather.get("yesterday"), Mapping)
                or not plan_weather.get("yesterday")
            ):
                yesterday_metrics = weather_details.get("yesterday")
                if isinstance(yesterday_metrics, Mapping) and yesterday_metrics:
                    yesterday_line = self._compose_weather_metrics_preview_line(
                        yesterday_metrics,
                        label="Вчера",
                    )
                    _add_weather_line(
                        yesterday_line
                        if not details_location
                        else f"{details_location}: {yesterday_line}"
                    )
            if (
                (not isinstance(plan_weather, Mapping))
                or not isinstance(plan_weather.get("sea"), Mapping)
                or not plan_weather.get("sea")
            ):
                sea_snapshot = weather_details.get("sea")
                if isinstance(sea_snapshot, Mapping):
                    sea_parts: list[str] = []
                    temp = sea_snapshot.get("temperature")
                    if isinstance(temp, (int, float)):
                        sea_parts.append(f"вода {int(round(temp))}°C")
                    wave = sea_snapshot.get("wave")
                    if isinstance(wave, (int, float)):
                        sea_parts.append(f"волна {round(float(wave), 1)} м")
                    description = str(sea_snapshot.get("description") or "").strip()
                    if description:
                        sea_parts.append(description)
                    detail = str(sea_snapshot.get("detail") or "").strip()
                    if detail:
                        sea_parts.append(detail)
                    if sea_parts:
                        _add_weather_line("Море: " + ", ".join(sea_parts))

        if weather_lines:
            escaped_weather = "\n".join(_escape_block_line(line) for line in weather_lines)
            weather_block = (
                f"Погода:\n" f'<blockquote expandable="true">{escaped_weather}</blockquote>'
            )
            _add_section(weather_block, 3)

        system_prompt = str(state.get("plan_system_prompt") or "").strip()
        user_prompt = str(state.get("plan_user_prompt") or "").strip()
        if system_prompt or user_prompt:
            length_value = state.get("plan_prompt_length")
            fallback_used = bool(state.get("plan_prompt_fallback"))
            meta_parts: list[str] = []
            if isinstance(length_value, int):
                meta_parts.append(f"длина {length_value}")
            if fallback_used:
                meta_parts.append("fallback")
            suffix = f" ({', '.join(meta_parts)})" if meta_parts else ""
            block_sections: list[str] = []
            if system_prompt:
                escaped_system = _escape_block_line(system_prompt)
                block_sections.append(f"<b>System prompt</b>:\n{escaped_system}")
            if user_prompt:
                escaped_user = _escape_block_line(user_prompt)
                block_sections.append(f"<b>User prompt</b>:\n{escaped_user}")
            if block_sections:
                block_html = "\n\n".join(block_sections)
                service_block = (
                    f"Служебно{suffix}:\n"
                    f'<blockquote expandable="true">{block_html}</blockquote>'
                )
                _add_section(service_block, 3)
        if "previous_main_post_text" in state:
            previous_text = str(state.get("previous_main_post_text") or "").strip()
            if previous_text:
                escaped_previous_text = html.escape(previous_text)
                full_block = f"Предыдущая публикация: {escaped_previous_text}"
                fallback_text: str | None = None
                if len(full_block) > 600:
                    truncated_prev = self._safe_preview_truncate(previous_text, 600)
                    escaped_truncated_prev = html.escape(truncated_prev)
                    fallback_text = f"Предыдущая публикация: {escaped_truncated_prev}"
                    if fallback_text == full_block:
                        fallback_text = None
                _add_section(full_block, 2, fallback=fallback_text)
            else:
                _add_section("Предыдущая публикация: не публиковалось", 1)
        _add_section("Выберите действие на клавиатуре ниже.", 0)

        def _total_length(items: Sequence[_PreviewSection]) -> int:
            if not items:
                return 0
            return sum(len(section.text) for section in items) + 2 * (len(items) - 1)

        limit = FLOWERS_PREVIEW_MAX_LENGTH
        total_length = _total_length(sections)
        if total_length > limit:
            priorities = sorted({section.priority for section in sections}, reverse=True)
            for priority in priorities:
                idx = len(sections) - 1
                while idx >= 0 and total_length > limit:
                    section = sections[idx]
                    if section.priority != priority:
                        idx -= 1
                        continue
                    if section.fallback and not section.used_fallback:
                        if section.fallback != section.text:
                            section.text = section.fallback
                            section.used_fallback = True
                            total_length = _total_length(sections)
                            if total_length <= limit:
                                break
                            continue
                    sections.pop(idx)
                    total_length = _total_length(sections)
                    idx -= 1
                if total_length <= limit:
                    break

        final_text = "\n\n".join(section.text for section in sections)
        if len(final_text) > limit:
            final_text = self._safe_preview_truncate(final_text, limit)
        return final_text

    async def _delete_flowers_preview_messages(
        self, state: dict[str, Any], *, keep_prompt: bool = False
    ) -> None:
        chat_id = state.get("preview_chat_id")
        if chat_id is None:
            return
        for message_id in list(state.get("media_message_ids") or []):
            try:
                await self.api_request(
                    "deleteMessage",
                    {"chat_id": chat_id, "message_id": message_id},
                )
            except Exception:
                logging.exception(
                    "Failed to delete flowers preview media message %s for chat %s",
                    message_id,
                    chat_id,
                )
        caption_id = state.get("caption_message_id")
        if caption_id:
            try:
                await self.api_request(
                    "deleteMessage",
                    {"chat_id": chat_id, "message_id": caption_id},
                )
            except Exception:
                logging.exception(
                    "Failed to delete flowers preview caption message %s for chat %s",
                    caption_id,
                    chat_id,
                )
        state["media_message_ids"] = []
        state["caption_message_id"] = None
        if not keep_prompt:
            prompt_id = state.get("instruction_prompt_id")
            if prompt_id:
                try:
                    await self.api_request(
                        "deleteMessage",
                        {"chat_id": chat_id, "message_id": prompt_id},
                    )
                except Exception:
                    logging.exception(
                        "Failed to delete flowers instruction prompt %s for chat %s",
                        prompt_id,
                        chat_id,
                    )
            state["instruction_prompt_id"] = None
            state["awaiting_instruction"] = False

    async def _send_flowers_preview(
        self,
        rubric: Rubric,
        initiator_id: int,
        *,
        default_channel: int,
        test_requested: bool,
        assets: list[Asset],
        asset_ids: list[str],
        file_ids: list[str],
        asset_kinds: list[str],
        conversion_map: dict[int, dict[str, Any]],
        cities: list[str],
        greeting: str,
        hashtags: list[str],
        preview_caption: str,
        publish_caption: str,
        publish_parse_mode: str | None,
        prepared_hashtags: list[str],
        instructions: str | None,
        weather_block: dict[str, Any] | None,
        plan: dict[str, Any],
        plan_meta: dict[str, Any] | None,
    ) -> None:
        previous_state = self.pending_flowers_previews.get(initiator_id)
        if previous_state:
            await self._delete_flowers_preview_messages(previous_state, keep_prompt=True)
        config = rubric.config or {}
        main_channel_raw = config.get("channel_id")
        test_channel_raw = config.get("test_channel_id")

        def _to_int(value: Any) -> int | None:
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                try:
                    return int(value)
                except ValueError:
                    return None
            return None

        prompt_payload = self._build_flowers_prompt_payload(plan, plan_meta)
        serialized_plan = str(prompt_payload.get("serialized_plan") or "{}")
        plan_system_prompt = str(prompt_payload.get("system_prompt") or "")
        plan_user_prompt = str(prompt_payload.get("user_prompt") or "")
        plan_request_text = str(prompt_payload.get("request_text") or "")
        plan_prompt_length = prompt_payload.get("prompt_length")
        plan_prompt_fallback = bool(prompt_payload.get("used_fallback"))
        selection_attempts: int | None = None
        if isinstance(plan_meta, Mapping):
            attempts_raw = plan_meta.get("photo_selection_attempts")
            if isinstance(attempts_raw, int):
                if attempts_raw > 0:
                    selection_attempts = attempts_raw
            elif isinstance(attempts_raw, str):
                try:
                    parsed_attempts = int(attempts_raw.strip())
                except ValueError:
                    parsed_attempts = None
                if parsed_attempts and parsed_attempts > 0:
                    selection_attempts = parsed_attempts
        weather_details: dict[str, Any] | None = None
        if isinstance(weather_block, dict):
            weather_details = {
                "city": weather_block.get("city"),
                "sea": weather_block.get("sea"),
                "positive_intro": weather_block.get("positive_intro"),
                "trend_summary": weather_block.get("trend_summary"),
                "today": weather_block.get("today"),
                "yesterday": weather_block.get("yesterday"),
            }
        previous_main_post_text: str | None = None
        previous_weather_line: str | None = None
        rubric_id = getattr(rubric, "id", None)
        if isinstance(rubric_id, int):
            prev_text, prev_weather = self._lookup_previous_flowers_post(rubric_id)
            previous_main_post_text = prev_text
            previous_weather_line = prev_weather
        if previous_state:
            if previous_main_post_text is None:
                stored_text = previous_state.get("previous_main_post_text")
                if stored_text:
                    previous_main_post_text = str(stored_text)
            if previous_weather_line is None:
                stored_weather = previous_state.get("weather_yesterday_line") or previous_state.get(
                    "weather_line"
                )
                stored_weather_str = str(stored_weather or "").strip()
                if stored_weather_str:
                    previous_weather_line = stored_weather_str
        today_line_source: str | None = None
        yesterday_line_source: str | None = None
        if isinstance(weather_block, dict):
            location_label = self._extract_weather_location_label(weather_block)
            today_metrics_line = self._compose_weather_metrics_preview_line(
                weather_block.get("today"),
                label="Сегодня",
            )
            yesterday_metrics_line = self._compose_weather_metrics_preview_line(
                weather_block.get("yesterday"),
                label="Вчера",
            )
            if location_label:
                today_line_source = f"{location_label}: {today_metrics_line}"
                yesterday_line_source = f"{location_label}: {yesterday_metrics_line}"
            else:
                today_line_source = today_metrics_line
                yesterday_line_source = yesterday_metrics_line
        if not yesterday_line_source and previous_weather_line:
            yesterday_line_source = previous_weather_line
        weather_today_line = self._normalize_weather_preview_line(today_line_source)
        weather_yesterday_line = self._normalize_weather_preview_line(yesterday_line_source)
        weather_today_line_html = html.escape(weather_today_line)
        weather_yesterday_line_html = html.escape(weather_yesterday_line)
        state: dict[str, Any] = {
            "rubric_code": rubric.code,
            "rubric_id": rubric.id,
            "assets": assets,
            "asset_ids": asset_ids,
            "file_ids": file_ids,
            "asset_kinds": asset_kinds,
            "conversion_map": conversion_map,
            "cities": cities,
            "greeting": greeting,
            "hashtags": hashtags,
            "prepared_hashtags": prepared_hashtags,
            "preview_caption": preview_caption,
            "publish_caption": publish_caption,
            "publish_parse_mode": publish_parse_mode,
            "instructions": (instructions or "").strip(),
            "weather_block": weather_block,
            "weather_today_line": weather_today_line,
            "weather_yesterday_line": weather_yesterday_line,
            "weather_today_line_html": weather_today_line_html,
            "weather_yesterday_line_html": weather_yesterday_line_html,
            "weather_line": weather_today_line,
            "plan": plan,
            "pattern_ids": list((plan_meta or {}).get("pattern_ids", [])),
            "serialized_plan": serialized_plan,
            "plan_prompt": plan_user_prompt,
            "plan_system_prompt": plan_system_prompt,
            "plan_user_prompt": plan_user_prompt,
            "plan_request_text": plan_request_text,
            "plan_prompt_length": plan_prompt_length,
            "plan_prompt_fallback": plan_prompt_fallback,
            "weather_details": weather_details,
            "previous_main_post_text": previous_main_post_text,
            "preview_chat_id": initiator_id,
            "media_message_ids": [],
            "caption_message_id": None,
            "instruction_prompt_id": (
                previous_state.get("instruction_prompt_id") if previous_state else None
            ),
            "awaiting_instruction": (
                previous_state.get("awaiting_instruction") if previous_state else False
            ),
            "channel_id": _to_int(main_channel_raw),
            "test_channel_id": _to_int(test_channel_raw),
            "default_channel_id": int(default_channel),
            "default_channel_type": "test" if test_requested else "main",
            "plan_meta": plan_meta or {},
            "photo_selection_attempts": selection_attempts,
        }
        normalized_caption = str(preview_caption or "")
        response, remaining_conversion = await self._send_flowers_media_bundle(
            chat_id=initiator_id,
            assets=assets,
            file_ids=file_ids,
            asset_kinds=asset_kinds,
            caption=normalized_caption,
            parse_mode=None,
            conversion_map=conversion_map,
        )
        if not response.get("ok"):
            logging.error("Failed to send flowers preview media: %s", response)
            await self.api_request(
                "sendMessage",
                {
                    "chat_id": initiator_id,
                    "text": "Не удалось отправить предпросмотр медиагруппы.",
                },
            )
            if previous_state:
                self.pending_flowers_previews[initiator_id] = previous_state
            return
        result_payload = response.get("result")
        media_ids: list[int] = []
        if isinstance(result_payload, list):
            for item in result_payload:
                message_id = item.get("message_id")
                if isinstance(message_id, int):
                    media_ids.append(message_id)
        elif isinstance(result_payload, dict):
            message_id = result_payload.get("message_id")
            if isinstance(message_id, int):
                media_ids.append(message_id)
        state["media_message_ids"] = media_ids
        state["file_ids"] = list(file_ids)
        state["asset_kinds"] = list(asset_kinds)
        state["conversion_map"] = remaining_conversion
        text = self._render_flowers_preview_text(state)
        caption_response = await self.api_request(
            "sendMessage",
            {
                "chat_id": initiator_id,
                "text": text,
                "parse_mode": "HTML",
                "reply_markup": self._flowers_preview_keyboard(state),
            },
        )
        if not caption_response.get("ok"):
            logging.error("Failed to send flowers preview caption: %s", caption_response)
            await self._delete_flowers_preview_messages(state)
            await self.api_request(
                "sendMessage",
                {
                    "chat_id": initiator_id,
                    "text": "Не удалось отправить подпись предпросмотра.",
                },
            )
            if previous_state:
                self.pending_flowers_previews[initiator_id] = previous_state
            return
        caption_result = caption_response.get("result")
        if isinstance(caption_result, dict):
            message_id = caption_result.get("message_id")
            if isinstance(message_id, int):
                state["caption_message_id"] = message_id
        state["instruction_prompt_id"] = None
        state["awaiting_instruction"] = False
        self.pending_flowers_previews[initiator_id] = state

    async def _update_flowers_preview_caption_state(
        self,
        state: dict[str, Any],
        *,
        preview_caption: str,
        publish_caption: str,
        publish_parse_mode: str | None,
        greeting: str,
        hashtags: list[str],
        prepared_hashtags: list[str],
    ) -> None:
        state["preview_caption"] = preview_caption
        state["publish_caption"] = publish_caption
        state["publish_parse_mode"] = publish_parse_mode
        state["greeting"] = greeting
        state["hashtags"] = hashtags
        state["prepared_hashtags"] = prepared_hashtags
        weather_block = state.get("weather_block")
        today_line_source: str | None = None
        yesterday_line_source: str | None = None
        if isinstance(weather_block, Mapping):
            location_label = self._extract_weather_location_label(weather_block)
            today_metrics_line = self._compose_weather_metrics_preview_line(
                weather_block.get("today"),
                label="Сегодня",
            )
            yesterday_metrics_line = self._compose_weather_metrics_preview_line(
                weather_block.get("yesterday"),
                label="Вчера",
            )
            if location_label:
                today_line_source = f"{location_label}: {today_metrics_line}"
                yesterday_line_source = f"{location_label}: {yesterday_metrics_line}"
            else:
                today_line_source = today_metrics_line
                yesterday_line_source = yesterday_metrics_line
        weather_today_line = self._normalize_weather_preview_line(today_line_source)
        if not yesterday_line_source:
            yesterday_line_source = state.get("weather_yesterday_line")
        weather_yesterday_line = self._normalize_weather_preview_line(yesterday_line_source)
        state["weather_today_line"] = weather_today_line
        state["weather_yesterday_line"] = weather_yesterday_line
        state["weather_today_line_html"] = html.escape(weather_today_line)
        state["weather_yesterday_line_html"] = html.escape(weather_yesterday_line)
        state["weather_line"] = weather_today_line
        chat_id = state.get("preview_chat_id")
        message_id = state.get("caption_message_id")
        if chat_id is None or message_id is None:
            return
        text = self._render_flowers_preview_text(state)
        await self.api_request(
            "editMessageText",
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": text,
                "parse_mode": "HTML",
                "reply_markup": self._flowers_preview_keyboard(state),
            },
        )

    async def _finalize_flowers_preview(
        self,
        user_id: int,
        *,
        to_test: bool,
    ) -> bool:
        state = self.pending_flowers_previews.get(user_id)
        if not state:
            await self.api_request(
                "sendMessage",
                {
                    "chat_id": user_id,
                    "text": "Предпросмотр не найден.",
                },
            )
            return False
        channel_id = self._resolve_flowers_target(state, to_test=to_test)
        if channel_id is None:
            await self.api_request(
                "sendMessage",
                {
                    "chat_id": user_id,
                    "text": "Не настроен канал для отправки.",
                },
            )
            return False
        file_ids = list(state.get("file_ids") or [])
        asset_kinds = list(state.get("asset_kinds") or [])
        caption = str(state.get("publish_caption") or "")
        parse_mode = state.get("publish_parse_mode")
        conversion_map = dict(state.get("conversion_map") or {})
        assets_list = list(state.get("assets") or [])
        response, _ = await self._send_flowers_media_bundle(
            chat_id=channel_id,
            assets=assets_list,
            file_ids=file_ids,
            asset_kinds=asset_kinds,
            caption=caption,
            parse_mode=str(parse_mode) if parse_mode else None,
            conversion_map=conversion_map,
        )
        if not response.get("ok"):
            logging.error("Failed to finalize flowers preview: %s", response)
            await self.api_request(
                "sendMessage",
                {
                    "chat_id": user_id,
                    "text": "Не удалось отправить публикацию.",
                },
            )
            return False
        result_payload = response.get("result")
        message_id: int
        if isinstance(result_payload, list) and result_payload:
            message_id = int(result_payload[0].get("message_id") or 0)
        elif isinstance(result_payload, dict):
            message_id = int(result_payload.get("message_id") or 0)
        else:
            message_id = 0
        rubric_code = state.get("rubric_code")
        rubric_id = state.get("rubric_id")
        asset_ids = list(state.get("asset_ids") or [])
        if isinstance(rubric_id, int) and rubric_code:
            metadata = {
                "rubric_code": rubric_code,
                "asset_ids": asset_ids,
                "test": to_test,
                "cities": list(state.get("cities") or []),
                "greeting": state.get("greeting"),
                "hashtags": list(state.get("prepared_hashtags") or []),
                "weather": state.get("weather_block"),
                "weather_today_line": state.get("weather_today_line"),
                "weather_yesterday_line": state.get("weather_yesterday_line"),
                "weather_line": state.get("weather_today_line"),
                "pattern_ids": list(state.get("pattern_ids") or []),
                "plan": state.get("plan"),
            }
            self.data.mark_assets_used(asset_ids)
            first_asset = asset_ids[0] if asset_ids else None
            self.data.record_post_history(
                channel_id,
                message_id,
                first_asset,
                rubric_id,
                metadata,
            )
        assets = list(state.get("assets") or [])
        if assets:
            await self._cleanup_assets(assets)
        await self._delete_flowers_preview_messages(state)
        self.pending_flowers_previews.pop(user_id, None)
        confirmation_text = "Публикация отправлена."
        if message_id:
            confirmation_text = f"{confirmation_text}\n{self.post_url(channel_id, message_id)}"
        await self.api_request(
            "sendMessage",
            {
                "chat_id": user_id,
                "text": confirmation_text,
            },
        )
        return True

    async def _handle_flowers_preview_callback(
        self,
        user_id: int,
        action: str,
        query: dict[str, Any],
    ) -> None:
        state = self.pending_flowers_previews.get(user_id)
        if (
            action
            in {
                "regen_photos",
                "regen_caption",
                "instruction",
                "download_prompt",
                "send_test",
                "send_main",
                "cancel",
            }
            and not state
        ):
            await self.api_request(
                "answerCallbackQuery",
                {
                    "callback_query_id": query["id"],
                    "text": "Предпросмотр не найден.",
                    "show_alert": True,
                },
            )
            return
        if action == "regen_photos":
            rubric_code = state.get("rubric_code") if state else None
            rubric = self.data.get_rubric_by_code(rubric_code) if rubric_code else None
            if not rubric:
                await self.api_request(
                    "answerCallbackQuery",
                    {
                        "callback_query_id": query["id"],
                        "text": "Рубрика недоступна.",
                        "show_alert": True,
                    },
                )
                return
            channel_hint = state.get("default_channel_id")
            if not isinstance(channel_hint, int):
                for key in ("channel_id", "test_channel_id"):
                    value = state.get(key)
                    if isinstance(value, int):
                        channel_hint = value
                        break
            prepared = await self._prepare_flowers_drop(
                rubric,
                instructions=state.get("instructions"),
                channel_id=channel_hint if isinstance(channel_hint, int) else None,
            )
            if not prepared:
                await self.api_request(
                    "answerCallbackQuery",
                    {
                        "callback_query_id": query["id"],
                        "text": "Не удалось подобрать фото.",
                        "show_alert": True,
                    },
                )
                return
            (
                assets,
                asset_ids,
                file_ids,
                asset_kinds,
                cities,
                greeting,
                hashtags,
                conversion_map,
                weather_block,
                plan,
                plan_meta,
            ) = prepared
            (
                preview_caption,
                publish_caption,
                publish_parse_mode,
                prepared_hashtags,
            ) = self._build_flowers_caption(
                greeting,
                cities,
                hashtags,
                weather_block,
            )
            default_channel = state.get("default_channel_id")
            if not isinstance(default_channel, int):
                fallback_channel = state.get("channel_id") or state.get("test_channel_id")
                if isinstance(fallback_channel, int):
                    default_channel = fallback_channel
            if not isinstance(default_channel, int):
                await self.api_request(
                    "answerCallbackQuery",
                    {
                        "callback_query_id": query["id"],
                        "text": "Нет доступного канала для предпросмотра.",
                        "show_alert": True,
                    },
                )
                return
            test_requested = state.get("default_channel_type") == "test"
            await self._send_flowers_preview(
                rubric,
                user_id,
                default_channel=default_channel,
                test_requested=bool(test_requested),
                assets=assets,
                asset_ids=asset_ids,
                file_ids=file_ids,
                asset_kinds=asset_kinds,
                conversion_map=conversion_map,
                cities=cities,
                greeting=greeting,
                hashtags=hashtags,
                preview_caption=preview_caption,
                publish_caption=publish_caption,
                publish_parse_mode=publish_parse_mode,
                prepared_hashtags=prepared_hashtags,
                instructions=state.get("instructions"),
                weather_block=weather_block,
                plan=plan,
                plan_meta=plan_meta,
            )
            await self.api_request(
                "answerCallbackQuery",
                {
                    "callback_query_id": query["id"],
                    "text": "Фото обновлены.",
                },
            )
            return
        if action == "regen_caption":
            rubric_code = state.get("rubric_code") if state else None
            rubric = self.data.get_rubric_by_code(rubric_code) if rubric_code else None
            if not rubric:
                await self.api_request(
                    "answerCallbackQuery",
                    {
                        "callback_query_id": query["id"],
                        "text": "Рубрика недоступна.",
                        "show_alert": True,
                    },
                )
                return
            assets = list(state.get("assets") or [])
            weather_block = state.get("weather_block")
            plan_override = state.get("plan")
            instructions_text = str(state.get("instructions") or "")
            if isinstance(plan_override, dict):
                plan_override = deepcopy(plan_override)
                plan_override["instructions"] = instructions_text
            else:
                plan_override = None
            channel_hint = state.get("default_channel_id")
            if not isinstance(channel_hint, int):
                for key in ("channel_id", "test_channel_id"):
                    value = state.get(key)
                    if isinstance(value, int):
                        channel_hint = value
                        break
            plan_meta_override = state.get("plan_meta")
            greeting, hashtags, plan, plan_meta = await self._generate_flowers_copy(
                rubric,
                assets,
                channel_id=channel_hint if isinstance(channel_hint, int) else None,
                weather_block=weather_block,
                instructions=instructions_text,
                plan=plan_override,
                plan_meta=plan_meta_override if isinstance(plan_meta_override, dict) else None,
            )
            if isinstance(plan, dict):
                plan["instructions"] = instructions_text
            (
                preview_caption,
                publish_caption,
                publish_parse_mode,
                prepared_hashtags,
            ) = self._build_flowers_caption(
                greeting,
                list(state.get("cities") or []),
                hashtags,
                weather_block,
            )
            state["plan"] = plan
            state["plan_meta"] = plan_meta or {}
            state["pattern_ids"] = list((plan_meta or {}).get("pattern_ids", []))
            prompt_payload = self._build_flowers_prompt_payload(plan, plan_meta)
            state["serialized_plan"] = str(prompt_payload.get("serialized_plan") or "{}")
            plan_system_prompt = str(prompt_payload.get("system_prompt") or "")
            plan_user_prompt = str(prompt_payload.get("user_prompt") or "")
            plan_request_text = str(prompt_payload.get("request_text") or "")
            state["plan_system_prompt"] = plan_system_prompt
            state["plan_user_prompt"] = plan_user_prompt
            state["plan_request_text"] = plan_request_text
            state["plan_prompt"] = plan_user_prompt
            state["plan_prompt_length"] = prompt_payload.get("prompt_length")
            state["plan_prompt_fallback"] = bool(prompt_payload.get("used_fallback"))
            await self._update_flowers_preview_caption_state(
                state,
                preview_caption=preview_caption,
                publish_caption=publish_caption,
                publish_parse_mode=publish_parse_mode,
                greeting=greeting,
                hashtags=hashtags,
                prepared_hashtags=prepared_hashtags,
            )
            await self.api_request(
                "answerCallbackQuery",
                {
                    "callback_query_id": query["id"],
                    "text": "Подпись обновлена.",
                },
            )
            return
        if action == "instruction":
            if state.get("awaiting_instruction"):
                await self.api_request(
                    "answerCallbackQuery",
                    {
                        "callback_query_id": query["id"],
                        "text": "Уже ожидаю инструкцию.",
                    },
                )
                return
            prompt_response = await self.api_request(
                "sendMessage",
                {
                    "chat_id": user_id,
                    "text": "Напишите инструкцию для подписи в ответ на это сообщение.",
                    "reply_markup": {
                        "force_reply": True,
                        "input_field_placeholder": "Например: добавь упоминание тюльпанов",
                    },
                },
            )
            if prompt_response.get("ok"):
                result = prompt_response.get("result")
                if isinstance(result, dict):
                    prompt_id = result.get("message_id")
                    if isinstance(prompt_id, int):
                        state["instruction_prompt_id"] = prompt_id
                        state["awaiting_instruction"] = True
            await self.api_request(
                "answerCallbackQuery",
                {
                    "callback_query_id": query["id"],
                    "text": "Жду инструкцию в ответном сообщении.",
                },
            )
            return
        if action == "download_prompt":
            plan_request_text = str(state.get("plan_request_text") or "")
            if not plan_request_text.strip():
                await self.api_request(
                    "answerCallbackQuery",
                    {
                        "callback_query_id": query["id"],
                        "text": "Промпт недоступен.",
                        "show_alert": True,
                    },
                )
                return
            buffer = io.BytesIO(plan_request_text.encode("utf-8"))
            files = {"document": ("flowers_prompt.txt", buffer.getvalue())}
            await self.api_request(
                "sendDocument",
                {
                    "chat_id": user_id,
                },
                files=files,
            )
            await self.api_request(
                "answerCallbackQuery",
                {
                    "callback_query_id": query["id"],
                    "text": "Промпт отправлен.",
                },
            )
            return
        if action == "send_test":
            success = await self._finalize_flowers_preview(user_id, to_test=True)
            await self.api_request(
                "answerCallbackQuery",
                {
                    "callback_query_id": query["id"],
                    "text": "Отправлено." if success else "Ошибка отправки.",
                },
            )
            return
        if action == "send_main":
            success = await self._finalize_flowers_preview(user_id, to_test=False)
            await self.api_request(
                "answerCallbackQuery",
                {
                    "callback_query_id": query["id"],
                    "text": "Отправлено." if success else "Ошибка отправки.",
                },
            )
            return
        if action == "cancel":
            await self._delete_flowers_preview_messages(state)
            self.pending_flowers_previews.pop(user_id, None)
            await self.api_request(
                "sendMessage",
                {
                    "chat_id": user_id,
                    "text": "Предпросмотр отменён.",
                },
            )
            await self.api_request(
                "answerCallbackQuery",
                {
                    "callback_query_id": query["id"],
                    "text": "Отменено.",
                },
            )
            return
        await self.api_request(
            "answerCallbackQuery",
            {
                "callback_query_id": query["id"],
                "text": "Неизвестное действие.",
                "show_alert": True,
            },
        )

    def _build_flowers_prompt_payload(
        self,
        plan: dict[str, Any] | None,
        plan_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        plan_dict = plan if isinstance(plan, dict) else {}
        meta = plan_meta if isinstance(plan_meta, dict) else {}
        length_cfg_raw = meta.get("length") if isinstance(meta, dict) else {}
        length_cfg = length_cfg_raw if isinstance(length_cfg_raw, dict) else {}
        min_len = int(length_cfg.get("min") or 420)
        max_len = int(length_cfg.get("max") or 520)
        banned_words_raw = meta.get("banned_words") if isinstance(meta, dict) else []
        banned_words = {str(word).strip() for word in (banned_words_raw or []) if str(word).strip()}
        patterns = [
            item
            for item in plan_dict.get("patterns") or []
            if isinstance(item, dict) and str(item.get("instruction") or "").strip()
        ]
        pattern_lines: list[str] = []
        for idx, pattern in enumerate(patterns, 1):
            instruction = str(pattern.get("instruction") or "").strip()
            kind = str(pattern.get("kind") or "").strip()
            tags: list[str] = []
            if kind:
                tags.append(kind)
            if pattern.get("photo_dependent"):
                tags.append("про фото")
            tag_prefix = f"[{', '.join(tags)}] " if tags else ""
            pattern_lines.append(f"{idx}. {tag_prefix}{instruction}")

        def _smart_trim(text: str, limit: int) -> str:
            if len(text) <= limit:
                return text
            trimmed = text[:limit].rsplit(" ", 1)[0].strip()
            if len(trimmed) < int(limit * 0.6):
                trimmed = text[:limit].strip()
            return trimmed.rstrip(",.;:-") + "…"

        photo_context_raw = plan_dict.get("photo_context")
        photo_entries: list[dict[str, Any]] = []
        if isinstance(photo_context_raw, list):
            for idx, entry in enumerate(photo_context_raw, 1):
                flowers_list: list[str] = []
                hints_list: list[str] = []
                location_text = ""
                season_code = ""
                season_display = ""
                if isinstance(entry, Mapping):
                    raw_flowers = entry.get("flowers")
                    if isinstance(raw_flowers, (list, tuple, set)):
                        flowers_list = [
                            re.sub(r"\s+", " ", str(value or "").strip())
                            for value in raw_flowers
                            if str(value or "").strip()
                        ]
                    elif isinstance(raw_flowers, str) and raw_flowers.strip():
                        flowers_list = [re.sub(r"\s+", " ", raw_flowers.strip())]
                    raw_hints = entry.get("hints")
                    if isinstance(raw_hints, (list, tuple, set)):
                        hints_list = [
                            re.sub(r"\s+", " ", str(value or "").strip())
                            for value in raw_hints
                            if str(value or "").strip()
                        ]
                    elif isinstance(raw_hints, str) and raw_hints.strip():
                        hints_list = [re.sub(r"\s+", " ", raw_hints.strip())]
                    location_value = entry.get("location")
                    if isinstance(location_value, str) and location_value.strip():
                        location_text = re.sub(r"\s+", " ", location_value.strip())
                    raw_season = entry.get("season")
                    if isinstance(raw_season, str) and raw_season.strip():
                        season_code = re.sub(r"\s+", " ", raw_season.strip())
                    raw_season_display = entry.get("season_display")
                    if isinstance(raw_season_display, str) and raw_season_display.strip():
                        season_display = re.sub(r"\s+", " ", raw_season_display.strip())
                elif isinstance(entry, str) and entry.strip():
                    hints_list = [re.sub(r"\s+", " ", entry.strip())]
                if hints_list and len(hints_list) > 5:
                    hints_list = hints_list[:5]
                photo_entries.append(
                    {
                        "idx": idx,
                        "flowers": flowers_list,
                        "hints": hints_list,
                        "location": location_text,
                        "season": season_code,
                        "season_display": season_display,
                    }
                )

        weather_info = (
            plan_dict.get("weather") if isinstance(plan_dict.get("weather"), dict) else {}
        )
        raw_weather_payload: dict[str, Any] = {}
        if isinstance(weather_info, dict) and weather_info:
            cities_value = weather_info.get("cities")
            if isinstance(cities_value, (list, tuple, set)):
                normalized_cities = [
                    str(city or "").strip() for city in cities_value if str(city or "").strip()
                ]
                if normalized_cities:
                    raw_weather_payload["cities"] = normalized_cities
            else:
                city_str = str(cities_value or "").strip()
                if city_str:
                    raw_weather_payload["cities"] = city_str
            city_name = str(weather_info.get("city_name") or "").strip()
            if city_name:
                raw_weather_payload["city_name"] = city_name

            def _extract_metrics(payload: Mapping[str, Any] | None) -> dict[str, Any]:
                if not isinstance(payload, Mapping):
                    return {}
                metrics: dict[str, Any] = {}
                for key in ("temperature", "wind_speed", "condition"):
                    value = payload.get(key)
                    if value is None:
                        continue
                    if isinstance(value, str):
                        trimmed = value.strip()
                        if not trimmed:
                            continue
                        metrics[key] = trimmed
                    else:
                        metrics[key] = value
                parts_value = payload.get("parts")
                if isinstance(parts_value, Mapping):
                    normalized_parts: dict[str, str] = {}
                    for part_name, part_condition in parts_value.items():
                        if not isinstance(part_name, str):
                            continue
                        if isinstance(part_condition, str):
                            trimmed_part = part_condition.strip()
                            if trimmed_part:
                                normalized_parts[part_name] = trimmed_part
                    if normalized_parts:
                        metrics["parts"] = normalized_parts
                return metrics

            today_metrics = _extract_metrics(weather_info.get("today"))
            if today_metrics:
                raw_weather_payload["today"] = today_metrics
            yesterday_metrics = _extract_metrics(weather_info.get("yesterday"))
            if yesterday_metrics:
                raw_weather_payload["yesterday"] = yesterday_metrics
            sea_info = weather_info.get("sea")
            if isinstance(sea_info, Mapping):
                sea_payload: dict[str, Any] = {}
                for key in ("temperature", "wave"):
                    value = sea_info.get(key)
                    if value is not None:
                        sea_payload[key] = value
                if sea_payload:
                    raw_weather_payload["sea"] = sea_payload

        flowers = plan_dict.get("flowers") if isinstance(plan_dict.get("flowers"), list) else []
        flower_names = [
            str(flower.get("name") or "").strip()
            for flower in flowers
            if isinstance(flower, dict) and str(flower.get("name") or "").strip()
        ]
        previous_text = str(plan_dict.get("previous_text") or "").strip()
        extra_instructions = str(plan_dict.get("instructions") or "").strip()

        rule_items: list[str] = [
            "Используй все идеи из раздела паттернов, вплетая их в один абзац без списков.",
            "Фото-зависимые шаблоны свяжи с визуальными деталями снимков.",
            "Сравнивай сегодня и вчера, опираясь на сырые погодные данные; если изменения ощутимые, коротко озвучь их (например, «заметно потеплело»), без выдуманных показателей.",
            "Игнорируй микроколебания погоды — делись только заметными изменениями.",
            "Сопоставляй температуру, ветер и другие показатели, чтобы сделать единый вывод о тренде.",
            "Погоду интегрируй мягко, без сухих перечислений.",
            "Пиши естественно, как живой человек; уместные эмодзи допустимы без перебора.",
            f"Итоговый текст должен быть от {min_len} до {max_len} символов.",
            "Текст один абзац без двойных переводов строк.",
            "Будь заботливым, избегай рекламных интонаций.",
            "Приветствие опирай на реальные фото: используй подсказки и распознанные цветы, не выдумывай несоответствующие детали.",
            "Помни: фотографии сделаны заранее, поэтому не описывай их как отражение сегодняшней погоды.",
        ]
        system_prompt = (
            "Ты редактор телеграм-канала о погоде, домашнем уюте и цветах. "
            "Твоя задача — создавать тёплые утренние приветственные подписи к реальным фотографиям цветов, звуча заботливо и по-доброму, полагаясь на описания снимков и погодный контекст, а не на фантазию. "
            "Пиши как живой человек, допускай уместные эмодзи и подбирай образный язык, но избегай клише. "
            "Фотографии сделаны заранее, поэтому не описывай их как отражение сегодняшней погоды. "
            "Сохраняй утреннюю интонацию: скажи «Доброе утро» и фразу вроде «Порадую вас цветами…», мягко перечисляя несколько распознанных цветов и подчеркивая заботу о читателях."
        )
        if banned_words:
            rule_items.append("Не используй слова: " + ", ".join(sorted(banned_words)))
        rules = [f"{idx}. {text}" for idx, text in enumerate(rule_items, 1)]
        header = (
            "Доброе утро! Ты — редактор телеграм-канала про погоду, уют и цветы. "
            "Подготовь заботливое утреннее приветствие: начни с тёплого обращения, произнеси «Порадую вас цветами…» и назови несколько распознанных цветов, опираясь на контекст снимков и погоды без выдуманных деталей. "
            "Подбери подходящие хэштеги, уместные эмодзи допустимы."
        )

        def _render_photo_descriptions(
            entries: Sequence[Mapping[str, Any]], *, max_hints: int | None
        ) -> list[str]:
            descriptions: list[str] = []
            for entry in entries:
                hints: list[str] = list(entry.get("hints") or [])
                if max_hints is not None and len(hints) > max_hints:
                    hints = hints[:max_hints]
                parts: list[str] = []
                flowers = [str(value) for value in entry.get("flowers") or [] if str(value)]
                if flowers:
                    parts.append("Цветы: " + ", ".join(flowers))
                season_display = str(entry.get("season_display") or "").strip()
                season_code = str(entry.get("season") or "").strip()
                if season_display or season_code:
                    season_label = season_display or season_code
                    parts.append("Сезон: " + season_label)
                if hints:
                    parts.append("Подсказки: " + "; ".join(hints))
                location = str(entry.get("location") or "").strip()
                if location:
                    parts.append("Локация: " + location)
                if not parts:
                    parts.append("Подсказок нет — ориентируйся на настроение кадра.")
                descriptions.append(f"Фото {entry.get('idx')}: " + "; ".join(parts))
            return descriptions

        raw_weather_text_pretty = ""
        raw_weather_text_compact = ""
        if raw_weather_payload:
            raw_weather_text_pretty = json.dumps(
                raw_weather_payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            raw_weather_text_compact = json.dumps(
                raw_weather_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ": "),
            )
        flower_names_text = ", ".join(flower_names)

        def _assemble_prompt(config: dict[str, Any]) -> dict[str, Any]:
            context_sections: list[str] = []
            if pattern_lines:
                context_sections.append("Паттерны:\n" + "\n".join(pattern_lines))

            photos_lines: list[str] = []
            if photo_entries:
                photos_lines = _render_photo_descriptions(
                    photo_entries,
                    max_hints=config.get("photo_hint_limit"),
                )
                context_sections.append("Фотографии:\n" + "\n".join(photos_lines))

            weather_payload_text = ""
            if config.get("include_weather") and raw_weather_payload:
                if config.get("use_compact_weather") and raw_weather_text_compact:
                    weather_payload_text = raw_weather_text_compact
                else:
                    weather_payload_text = (
                        raw_weather_text_pretty
                        or raw_weather_text_compact
                        or json.dumps(raw_weather_payload, ensure_ascii=False, sort_keys=True)
                    )
                context_sections.append("Сырые данные погоды (JSON):\n" + weather_payload_text)

            flower_text = ""
            if config.get("include_flowers") and flower_names_text:
                flower_text = flower_names_text
                context_sections.append("Цветы на фото (распознаны): " + flower_text)

            trimmed_previous = previous_text
            previous_limit = config.get("previous_limit")
            if trimmed_previous and previous_limit:
                trimmed_previous = _smart_trim(trimmed_previous, int(previous_limit))
            if trimmed_previous:
                context_sections.append("Вчерашний текст: " + trimmed_previous)

            trimmed_instructions = extra_instructions
            instructions_limit = config.get("instructions_limit")
            if trimmed_instructions and instructions_limit:
                trimmed_instructions = _smart_trim(trimmed_instructions, int(instructions_limit))
            if trimmed_instructions:
                context_sections.append("Доп. инструкция: " + trimmed_instructions)

            context_block = "\n\n".join(context_sections).strip()

            sections: list[str] = [header]
            if context_block:
                sections.append("Контекст:\n" + context_block)
            sections.append("Правила:\n" + "\n".join(rules))
            sections.append(
                "Верни JSON с ключами greeting (готовый текст) и hashtags (не менее двух тегов)."
            )
            user_prompt_text = "\n\n".join(
                section.strip() for section in sections if section.strip()
            )
            return {
                "user_prompt": user_prompt_text,
                "context_block": context_block,
                "photos_lines": photos_lines,
                "weather_payload_text": weather_payload_text,
                "flower_text": flower_text,
                "previous_text": trimmed_previous,
                "instructions_text": trimmed_instructions,
            }

        prompt_limit = 2200
        config: dict[str, Any] = {
            "photo_hint_limit": 4 if photo_entries else None,
            "use_compact_weather": False,
            "include_weather": bool(raw_weather_payload),
            "include_flowers": bool(flower_names_text),
            "previous_limit": None,
            "instructions_limit": None,
        }

        render_state = _assemble_prompt(config)
        user_prompt = render_state["user_prompt"]

        limit_keys = {"previous_limit", "instructions_limit", "photo_hint_limit"}

        def _apply_config_change(key: str, value: Any) -> bool:
            current_value = config.get(key)
            if key in limit_keys and current_value is not None:
                if current_value <= value:
                    return False
            if current_value == value:
                return False
            config[key] = value
            return True

        def _maybe_adjust(
            key: str,
            value: Any,
            condition: Callable[[], bool],
        ) -> None:
            nonlocal render_state, user_prompt
            if len(user_prompt) <= prompt_limit:
                return
            try:
                should_apply = bool(condition())
            except Exception:
                should_apply = False
            if not should_apply:
                return
            if _apply_config_change(key, value):
                render_state = _assemble_prompt(config)
                user_prompt = render_state["user_prompt"]

        # Prefer compact weather representation if available.
        _maybe_adjust("use_compact_weather", True, lambda: bool(raw_weather_text_pretty))

        # Gradually trim the lengthier optional sections before falling back.
        for limit_value in (480, 360, 240, 180, 140, 100, 80, 60):
            _maybe_adjust("previous_limit", limit_value, lambda: bool(previous_text))
            _maybe_adjust("instructions_limit", limit_value, lambda: bool(extra_instructions))
            if len(user_prompt) <= prompt_limit:
                break

        # Reduce per-photo hint verbosity while keeping all photo descriptions present.
        for hint_cap in (3, 2, 1):
            _maybe_adjust("photo_hint_limit", hint_cap, lambda: bool(photo_entries))
            if len(user_prompt) <= prompt_limit:
                break

        # Remove optional sections entirely only as a last resort.
        _maybe_adjust("include_flowers", False, lambda: bool(flower_names_text))
        _maybe_adjust("include_weather", False, lambda: bool(raw_weather_text_pretty))

        def _build_fallback_prompt(state: Mapping[str, Any]) -> str:
            ideas = [re.sub(r"^\d+\.\s*", "", line) for line in pattern_lines[:3]]
            ideas_text = "; ".join(idea for idea in ideas if idea)
            if len(ideas_text) > 360:
                ideas_text = ideas_text[:357].rstrip() + "…"
            fallback_lines = [
                header,
                f"Собери один абзац длиной {min_len}-{max_len} символов.",
            ]
            if ideas_text:
                fallback_lines.append("Основные идеи: " + ideas_text)
            context_lines: list[str] = []
            weather_payload_text = str(state.get("weather_payload_text") or "").strip()
            if weather_payload_text:
                context_lines.append("Сырые данные погоды:\n" + weather_payload_text)
                context_lines.append(
                    "Модель 4o должна сама сравнить сегодняшний и вчерашний день, опираясь на эти показатели."
                )
            photos_lines = list(state.get("photos_lines") or [])
            if photos_lines:
                context_lines.append("Фотографии:")
                context_lines.extend(photos_lines)
            flower_text = str(state.get("flower_text") or "").strip()
            if flower_text:
                context_lines.append("Цветы на фото (распознаны): " + flower_text)
            instructions_text = str(state.get("instructions_text") or "").strip()
            if instructions_text:
                context_lines.append("Доп. инструкция: " + instructions_text)
            if context_lines:
                fallback_lines.append("Контекст:")
                fallback_lines.extend(context_lines)
            if banned_words:
                fallback_lines.append("Избегай слов: " + ", ".join(sorted(list(banned_words))[:8]))
            fallback_lines.append(
                "Пиши естественно, как живой человек; уместные эмодзи допустимы без перебора."
            )
            fallback_lines.append("Верни JSON с ключами greeting и hashtags (не менее двух тегов).")
            return "\n".join(fallback_lines)

        used_fallback = False
        if len(user_prompt) > prompt_limit:
            user_prompt = _build_fallback_prompt(render_state)
            used_fallback = True

        prompt_length = len(user_prompt)
        request_sections = []
        if system_prompt:
            request_sections.append("System prompt:\n" + system_prompt)
        if user_prompt:
            request_sections.append("User prompt:\n" + user_prompt)
        request_text = "\n\n".join(request_sections)
        serialized_plan = json.dumps(plan_dict, ensure_ascii=False, sort_keys=True)
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "request_text": request_text,
            "serialized_plan": serialized_plan,
            "rules": rules,
            "min_length": min_len,
            "max_length": max_len,
            "banned_words": banned_words,
            "prompt_length": prompt_length,
            "used_fallback": used_fallback,
        }

    async def _generate_flowers_copy(
        self,
        rubric: Rubric,
        assets: Sequence[Asset],
        *,
        channel_id: int | None,
        weather_block: dict[str, Any] | None = None,
        job: Job | None = None,
        instructions: str | None = None,
        plan: dict[str, Any] | None = None,
        plan_meta: dict[str, Any] | None = None,
    ) -> tuple[str, list[str], dict[str, Any], dict[str, Any]]:
        if plan is None or not isinstance(plan, dict) or not isinstance(plan_meta, dict):
            asset_seasons = self._collect_asset_seasons(assets)
            plan, plan_meta = self._build_flowers_plan(
                rubric,
                assets,
                weather_block,
                channel_id=channel_id,
                instructions=instructions,
                asset_seasons=asset_seasons if asset_seasons else None,
            )
        resolved_plan = plan
        resolved_meta = plan_meta or {}
        cities = list(resolved_meta.get("cities") or [])
        prompt_payload = self._build_flowers_prompt_payload(resolved_plan, resolved_meta)
        banned_words = set(prompt_payload.get("banned_words") or [])
        if not self.openai or not self.openai.api_key:
            return (
                self._default_flowers_greeting(cities),
                self._default_hashtags("flowers"),
                resolved_plan,
                resolved_meta,
            )
        system_prompt = str(prompt_payload.get("system_prompt") or "")
        user_prompt = str(prompt_payload.get("user_prompt") or "")
        schema = {
            "type": "object",
            "properties": {
                "greeting": {"type": "string"},
                "hashtags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                },
            },
            "required": ["greeting", "hashtags"],
        }
        previous_text = self._latest_flowers_text(rubric)
        attempts = 3
        for attempt in range(1, attempts + 1):
            temperature = self._creative_temperature()
            try:
                logging.info(
                    "Запрос генерации текста для цветов: модель=%s temperature=%.2f top_p=0.9 попытка %s/%s",
                    "gpt-4o",
                    temperature,
                    attempt,
                    attempts,
                )
                self._enforce_openai_limit(job, "gpt-4o")
                response = await self.openai.generate_json(
                    model="gpt-4o",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema=schema,
                    temperature=temperature,
                    top_p=0.9,
                )
            except Exception:
                logging.exception("Failed to generate flowers copy (attempt %s)", attempt)
                response = None
            if response:
                await self._record_openai_usage("gpt-4o", response, job=job)
            if not response or not isinstance(response.content, dict):
                continue
            greeting = str(response.content.get("greeting") or "").strip()
            raw_hashtags = response.content.get("hashtags") or []
            hashtags = self._deduplicate_hashtags(raw_hashtags)
            if not greeting or not hashtags:
                continue
            if banned_words and self._flowers_contains_banned_word(greeting, banned_words):
                logging.info("flowers copy содержит запрещённые слова, пробуем снова")
                continue
            if previous_text:
                similarity = self._jaccard_similarity(previous_text, greeting)
                if similarity >= 0.4:
                    logging.info(
                        "flowers copy слишком похож на предыдущий (Jaccard=%.2f), повторяем попытку",
                        similarity,
                    )
                    continue
            if self._is_duplicate_rubric_copy("flowers", "greeting", greeting, hashtags):
                logging.info(
                    "Получен повторяющийся текст для рубрики flowers, пробуем снова (%s/%s)",
                    attempt,
                    attempts,
                )
                continue
            return greeting, hashtags, resolved_plan, resolved_meta
        logging.warning(
            "Не удалось получить новый текст для рубрики flowers, используем запасной вариант"
        )
        return (
            self._default_flowers_greeting(cities),
            self._default_hashtags("flowers"),
            resolved_plan,
            resolved_meta,
        )

    def _default_flowers_greeting(self, cities: list[str]) -> str:
        if cities:
            if len(cities) == 1:
                location = cities[0]
            else:
                location = ", ".join(cities[:-1]) + f" и {cities[-1]}"
            return f"Доброе утро, {location}! Делимся цветами вашего города."
        return "Доброе утро! Пусть этот букет сделает день ярче."

    def _creative_temperature(self) -> float:
        return round(random.uniform(0.9, 1.1), 2)

    def _deduplicate_hashtags(self, tags: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for tag in tags:
            text = str(tag or "").strip()
            if not text:
                continue
            normalized = text.lstrip("#").lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            result.append(text)
        return result

    def _is_duplicate_rubric_copy(
        self,
        rubric_code: str,
        key: str,
        text: str,
        hashtags: Iterable[str],
    ) -> bool:
        normalized_text = text.strip().lower()
        normalized_tags = sorted(
            {str(tag).lstrip("#").lower() for tag in hashtags if str(tag).strip()}
        )
        if not normalized_text:
            return False
        recent_metadata = self.data.get_recent_rubric_metadata(rubric_code, limit=5)
        for item in recent_metadata:
            if not isinstance(item, dict):
                continue
            previous_text = str(item.get(key) or "").strip().lower()
            if not previous_text:
                continue
            raw_tags = item.get("hashtags")
            previous_tags: list[str]
            if isinstance(raw_tags, (list, tuple, set)):
                previous_tags = [str(tag) for tag in raw_tags]
            elif isinstance(raw_tags, str):
                previous_tags = [piece.strip() for piece in raw_tags.split() if piece.strip()]
            else:
                previous_tags = []
            normalized_previous_tags = sorted(
                {tag.lstrip("#").lower() for tag in previous_tags if tag.strip()}
            )
            if previous_text == normalized_text and normalized_previous_tags == normalized_tags:
                return True
        return False

    async def _publish_guess_arch(
        self,
        rubric: Rubric,
        channel_id: int,
        *,
        test: bool = False,
        job: Job | None = None,
        initiator_id: int | None = None,
        instructions: str | None = None,
    ) -> bool:
        config = rubric.config or {}
        asset_cfg = config.get("assets") or {}
        min_config = int(asset_cfg.get("min") or 4)
        max_config_raw = asset_cfg.get("max")
        min_count = max(4, min_config)
        if max_config_raw is None:
            max_count = max(min_count, 6)
        else:
            max_count = max(min_count, min(int(max_config_raw), 6))
        weather_city = config.get("weather_city") or "Kaliningrad"
        weather_text, weather_class = self._get_city_weather_info(weather_city)
        allowed_photo_weather = self._compatible_photo_weather_classes(weather_class)
        candidate_limit = max_count * 3 if allowed_photo_weather else max_count
        candidates = self.data.fetch_assets_by_vision_category(
            "architecture",
            rubric_id=rubric.id,
            limit=candidate_limit,
            require_arch_view=True,
            random_order=True,
        )
        assets: list[Asset] = []
        for asset in candidates:
            photo_label = self._normalize_weather_label(asset.vision_photo_weather)
            if not photo_label:
                continue
            if photo_label == "indoor":
                assets.append(asset)
            elif not allowed_photo_weather:
                assets.append(asset)
            elif photo_label in allowed_photo_weather:
                assets.append(asset)
            if len(assets) >= max_count:
                break
        if len(assets) < min_count:
            logging.warning(
                "Not enough assets for guess_arch rubric: have %s, need %s",
                len(assets),
                min_count,
            )
            return False
        overlay_paths: list[str] = []
        source_files: list[tuple[int, str, bool]] = []
        try:
            for idx, asset in enumerate(assets, start=1):
                source_path, should_cleanup = await self._ensure_asset_source(asset)
                if not source_path:
                    logging.warning("Asset %s missing source file for guess_arch overlay", asset.id)
                    for created in overlay_paths:
                        self._remove_file(created)
                    return False
                source_files.append((asset.id, source_path, should_cleanup))
                path = self._overlay_number(asset, idx, config, source_path=source_path)
                if not path:
                    for created in overlay_paths:
                        self._remove_file(created)
                    return False
                overlay_paths.append(path)
        except Exception:
            logging.exception("Failed to prepare overlays for guess_arch")
            for created in overlay_paths:
                self._remove_file(created)
            return False
        finally:
            for asset_id, source_path, should_cleanup in source_files:
                if should_cleanup:
                    self._remove_file(source_path)
                    try:
                        self.data.update_asset(asset_id, local_path=None)
                    except Exception:
                        logging.exception(
                            "Failed to clear local_path for asset %s after overlay", asset_id
                        )
        caption_text, hashtags = await self._generate_guess_arch_copy(
            rubric, len(assets), weather_text, job=job
        )
        hashtag_list = self._prepare_hashtags(hashtags)
        caption_parts = [caption_text.strip()] if caption_text else []
        if weather_text:
            caption_parts.append(weather_text)
        if hashtag_list:
            caption_parts.append(" ".join(hashtag_list))
        caption = "\n\n".join(part for part in caption_parts if part)
        media: list[dict[str, Any]] = []
        files: dict[str, tuple[str, bytes]] = {}
        for idx, path in enumerate(overlay_paths):
            attach_name = f"file{idx}"
            try:
                data = Path(path).read_bytes()
            except Exception:
                logging.exception("Failed to read overlay %s", path)
                for created in overlay_paths:
                    self._remove_file(created)
                return False
            files[attach_name] = (Path(path).name, data)
            item = {"type": "photo", "media": f"attach://{attach_name}"}
            if idx == 0 and caption:
                item["caption"] = caption
            media.append(item)
        response = await self.api_request(
            "sendMediaGroup",
            {"chat_id": channel_id, "media": media},
            files=files,
        )
        if not response.get("ok"):
            logging.error("Failed to publish guess_arch rubric: %s", response)
            for created in overlay_paths:
                self._remove_file(created)
            return False
        result_payload = response.get("result")
        if isinstance(result_payload, list) and result_payload:
            message_id = int(result_payload[0].get("message_id") or 0)
        elif isinstance(result_payload, dict):
            message_id = int(result_payload.get("message_id") or 0)
        else:
            message_id = 0
        self.data.mark_assets_used(asset.id for asset in assets)
        metadata = {
            "rubric_code": rubric.code,
            "asset_ids": [asset.id for asset in assets],
            "test": test,
            "weather": weather_text,
            "caption": caption_text,
            "hashtags": hashtag_list,
        }
        self.data.record_post_history(
            channel_id,
            message_id,
            assets[0].id if assets else None,
            rubric.id,
            metadata,
        )
        await self._cleanup_assets(assets, extra_paths=overlay_paths)
        return True

    async def _publish_sea(
        self,
        rubric: Rubric,
        channel_id: int,
        *,
        test: bool = False,
        job: Job | None = None,
        initiator_id: int | None = None,
        instructions: str | None = None,
    ) -> bool:
        publish_start_time = time.perf_counter()
        timeline: dict[str, float] = {}

        config = rubric.config or {}
        enable_facts = bool(config.get("enable_facts", True))
        sea_id = int(config.get("sea_id") or 1)
        is_prod = not test
        original_channel_id = channel_id

        def _coerce_channel(value: Any) -> int | None:
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return None
                try:
                    return int(text)
                except ValueError:
                    logging.warning("SEA_RUBRIC invalid_channel value=%s", value)
                    return None
            return None

        prod_channel_cfg = config.get("channel_id")
        test_channel_cfg = config.get("test_channel_id")
        prod_channel = _coerce_channel(prod_channel_cfg)
        test_channel = _coerce_channel(test_channel_cfg)
        selected_channel = prod_channel if is_prod else test_channel
        if selected_channel is None:
            fallback_channel = _coerce_channel(original_channel_id)
            if fallback_channel is not None:
                selected_channel = fallback_channel
        if selected_channel is None:
            logging.error(
                "SEA_RUBRIC channel_unresolved prod=%d prod_channel=%s test_channel=%s fallback=%s",
                int(is_prod),
                prod_channel_cfg,
                test_channel_cfg,
                original_channel_id,
            )
            return False
        channel_id = int(selected_channel)
        logging.info(
            "SEA_RUBRIC channel_resolved prod=%d channel_id=%s prod_channel=%s test_channel=%s fallback=%s",
            int(is_prod),
            channel_id,
            prod_channel_cfg,
            test_channel_cfg,
            original_channel_id,
        )

        timeline["start"] = 0.0

        step_start = time.perf_counter()
        conditions = self._get_sea_conditions(sea_id) or {}
        cache_row = self._get_sea_cache(sea_id)
        timeline["read_sea_cache"] = round((time.perf_counter() - step_start) * 1000, 2)
        water_temp = cache_row["current"] if cache_row and "current" in cache_row.keys() else None
        wave_height_value = self._safe_float(conditions.get("wave_height_m"))
        if wave_height_value is None and cache_row:
            wave_height_value = self._safe_float(cache_row["wave"])
        wave_score = wave_m_to_score(wave_height_value)
        if wave_score <= 2:
            storm_state = "calm"
        elif wave_score < 6:
            storm_state = "storm"
        else:
            storm_state = "strong_storm"
        target_wave_score = wave_score

        publish_key = f"sea_{sea_id}_{storm_state}"
        if hasattr(self, "_sea_publish_guard"):
            guard = self._sea_publish_guard
            last_publish_time = guard.get(publish_key)
            if last_publish_time is not None:
                elapsed = time.time() - last_publish_time
                if elapsed < 60:
                    logging.info(
                        "SEA_RUBRIC idempotency_skip sea_id=%s storm_state=%s elapsed_sec=%.1f",
                        sea_id,
                        storm_state,
                        elapsed,
                    )
                    return True
        else:
            self._sea_publish_guard = {}
        self._sea_publish_guard[publish_key] = time.time()
        wind_ms = self._safe_float(conditions.get("wind_speed_10m_ms"))
        wind_kmh = self._safe_float(conditions.get("wind_speed_10m_kmh"))
        if wind_ms is not None:
            wind_kmh = wind_ms * 3.6
        elif wind_kmh is not None:
            wind_ms = wind_kmh / 3.6
        wind_gust_ms = self._safe_float(conditions.get("wind_gusts_10m_ms"))
        wind_gust_kmh = self._safe_float(conditions.get("wind_gusts_10m_kmh"))
        if wind_gust_ms is not None:
            wind_gust_kmh = wind_gust_ms * 3.6
        elif wind_gust_kmh is not None:
            wind_gust_ms = wind_gust_kmh / 3.6
        wind_units_raw = conditions.get("wind_units")
        wind_units = wind_units_raw.strip() if isinstance(wind_units_raw, str) else None
        wind_units = wind_units or None
        gust_units_raw = conditions.get("wind_gusts_units")
        wind_gust_units = gust_units_raw.strip() if isinstance(gust_units_raw, str) else None
        wind_gust_units = wind_gust_units or None
        wind_time_ref = conditions.get("wind_time_ref")

        wind_class = classify_wind_kph(wind_kmh)
        cloud_cover_pct = self._safe_float(conditions.get("cloud_cover_pct"))
        sky_bucket = bucket_clouds(cloud_cover_pct) or "partly_cloudy"
        clear_guard_hard = cloud_cover_pct is not None and cloud_cover_pct <= 10.0
        clear_guard_soft = cloud_cover_pct is not None and cloud_cover_pct <= 20.0

        clouds_label_map = {
            "clear": "ясное небо",
            "mostly_clear": "почти ясное небо",
            "partly_cloudy": "переменная облачность",
            "mostly_cloudy": "пасмурно",
            "overcast": "сплошная облачность",
        }
        clouds_label = clouds_label_map.get(sky_bucket, "облачность неопределена")

        now = datetime.utcnow()
        storm_persisting, storm_persisting_reason = self._is_storm_persisting(
            sea_id,
            now=now,
            current_state=storm_state,
        )
        kaliningrad_tz = ZoneInfo("Europe/Kaliningrad")
        now_local = now.replace(tzinfo=UTC).astimezone(kaliningrad_tz)
        today_doy = now_local.timetuple().tm_yday
        season_window_days = 45

        now_local_iso = now_local.isoformat()
        day_part = map_hour_to_day_part(now_local.hour)
        allowed_photo_skies = compatible_skies(sky_bucket, day_part)
        tz_name = "Europe/Kaliningrad"

        def _normalize_for_log(value: Any) -> Any:
            if isinstance(value, float):
                return round(value, 3)
            if isinstance(value, dict):
                return {key: _normalize_for_log(val) for key, val in value.items()}
            if isinstance(value, (list, tuple)):
                return [_normalize_for_log(item) for item in value]
            if isinstance(value, set):
                return [_normalize_for_log(item) for item in sorted(value)]
            return value

        def _stringify_for_log(value: Any) -> str:
            normalized = _normalize_for_log(value)
            if isinstance(normalized, dict):
                items = [
                    f"{key}={_stringify_for_log(val)}" for key, val in sorted(normalized.items())
                ]
                return ";".join(items)
            if isinstance(normalized, (list, tuple)):
                return ",".join(_stringify_for_log(item) for item in normalized)
            return str(normalized)

        def _sky_token(value: NormalizedSky | None) -> str | None:
            if isinstance(value, NormalizedSky):
                return value.token()
            return None

        def _sky_tokens(values: Iterable[NormalizedSky]) -> list[str]:
            tokens = {token for token in (_sky_token(item) for item in values) if token}
            return sorted(tokens)

        def _emit_log(label: str, **fields: Any) -> None:
            payload = {key: value for key, value in fields.items() if value is not None}
            if payload:
                message = " ".join(
                    f"{key}={_stringify_for_log(value)}" for key, value in payload.items()
                )
                logging.info("SEA_RUBRIC %s %s", label, message)
            else:
                logging.info("SEA_RUBRIC %s", label)

        def sea_log(event: str, **details: Any) -> None:
            if event == "weather":
                _emit_log(
                    "weather",
                    wave_height_m=details.get("wave_height_m"),
                    wave_target_score=details.get("wave_target_score"),
                    wind_ms=details.get("wind_ms"),
                    wind_kmh=details.get("wind_kmh"),
                    wind_class=details.get("wind_class"),
                    cloud_cover_pct=details.get("cloud_cover_pct"),
                    sky_bucket=details.get("sky_bucket"),
                    allowed_skies=details.get("allowed_skies"),
                    sky_daypart=details.get("sky_daypart"),
                    today_doy=details.get("today_doy"),
                    season_window_days=details.get("season_window_days"),
                )
            elif event == "season":
                _emit_log(
                    "season window",
                    doy_now=details.get("doy_now"),
                    doy_range=details.get("doy_range"),
                    kept=details.get("kept"),
                    removed=details.get("removed"),
                    null_doy=details.get("null_doy"),
                    season_removed=details.get("season_removed"),
                    pool_after_season=details.get("pool_after_season"),
                )
            elif event == "stage":
                label = f"stage {details.get('name')}"
                _emit_log(
                    label,
                    sky_policy=details.get("sky"),
                    corridor=details.get("corridor"),
                    pool_size=details.get("pool_size"),
                )
            elif event == "pool_counts":
                payload = {
                    "pool_after_season": details.get("pool_after_season"),
                    "pool_after_B1": details.get("pool_after_B1"),
                    "pool_after_B2": details.get("pool_after_B2"),
                    "pool_after_AN": details.get("pool_after_AN"),
                    "pool_after_B0": details.get("pool_after_B0"),
                }
                _emit_log("pool after", **payload)
                _emit_log("pool_counts", **payload)
            elif event == "top5":
                index = details.get("index")
                label = f"top5 #{index}" if index is not None else "top5"
                payload = {key: val for key, val in details.items() if key != "index"}
                _emit_log(label, **payload)
            elif event == "selected":
                _emit_log("selected", **details)
            else:
                _emit_log(event, **details)

        sea_log(
            "weather",
            wave_height_m=wave_height_value,
            wave_target_score=target_wave_score,
            wind_ms=wind_ms,
            wind_kmh=wind_kmh,
            wind_gust_ms=wind_gust_ms,
            wind_gust_kmh=wind_gust_kmh,
            wind_class=wind_class,
            wind_units=wind_units,
            wind_gust_units=wind_gust_units,
            wind_time_ref=wind_time_ref,
            cloud_cover_pct=cloud_cover_pct,
            sky_bucket=sky_bucket,
            allowed_skies=_sky_tokens(allowed_photo_skies),
            sky_daypart=day_part,
            today_doy=today_doy,
            season_window_days=season_window_days,
        )

        doy_low = (today_doy - season_window_days) % 365 + 1
        doy_high = (today_doy + season_window_days - 1) % 365 + 1

        step_start = time.perf_counter()
        candidates = self.data.fetch_sea_candidates(
            rubric.id,
            limit=48,
            season_range=(doy_low, doy_high),
        )
        timeline["select_candidates"] = round((time.perf_counter() - step_start) * 1000, 2)
        if not candidates:
            sea_log(
                "selection_empty",
                reason="no_candidates",
                wave_height_m=wave_height_value,
                wave_target_score=target_wave_score,
                wind_ms=wind_ms,
                wind_kmh=wind_kmh,
                wind_gust_ms=wind_gust_ms,
                wind_gust_kmh=wind_gust_kmh,
                wind_class=wind_class,
                sky_bucket=sky_bucket,
                clouds_label=clouds_label,
            )
            logging.warning("SEA_RUBRIC no_candidates sea_id=%s", sea_id)
            temp_want_sunset = storm_state != "strong_storm" and sky_bucket in {
                "clear",
                "mostly_clear",
                "partly_cloudy",
            }
            await self._handle_sea_no_candidates(
                rubric,
                channel_id,
                sea_id=sea_id,
                storm_state=storm_state,
                storm_persisting=storm_persisting,
                storm_reason=storm_persisting_reason,
                wave_height=wave_height_value,
                wave_score=wave_score,
                wind_ms=wind_ms,
                wind_kmh=wind_kmh,
                wind_class=wind_class,
                clouds_label=clouds_label,
                sky_bucket=sky_bucket,
                want_sunset=temp_want_sunset,
                sky_daypart=day_part,
                test=test,
                initiator_id=initiator_id,
            )
            return True

        # Apply seasonal filter based on day-of-year
        kept_asset_ids: list[int] = []
        removed_asset_ids: list[int] = []
        null_doy_asset_ids: list[int] = []

        for candidate in candidates:
            shot_doy = candidate.get("shot_doy")
            if shot_doy is None:
                candidate["season_match"] = False
                null_doy_asset_ids.append(candidate["asset"].id)
            else:
                match = is_in_season_window(
                    shot_doy, today_doy=today_doy, window=season_window_days
                )
                candidate["season_match"] = match
                if match:
                    kept_asset_ids.append(candidate["asset"].id)
                else:
                    removed_asset_ids.append(candidate["asset"].id)

        season_matches = sum(1 for candidate in candidates if candidate["season_match"])
        filtered_count = len(candidates) - season_matches

        if DEBUG_SEA_PICK and filtered_count > 0:
            rejected = [c for c in candidates if not c["season_match"]][:5]
            for c in rejected:
                logging.debug(
                    "SEA_RUBRIC season_rejected asset_id=%s shot_doy=%s today_doy=%s",
                    c["asset"].id,
                    c.get("shot_doy"),
                    today_doy,
                )

        working_candidates = [candidate for candidate in candidates if candidate["season_match"]]
        season_filter_removed = False
        season_removal_reason: str | None = None
        if not working_candidates:
            season_filter_removed = True
            season_removal_reason = "no_match"
            working_candidates = candidates

        step_start = time.perf_counter()

        sea_log(
            "season",
            doy_now=today_doy,
            doy_range=[doy_low, doy_high],
            kept=kept_asset_ids,
            removed=removed_asset_ids,
            null_doy=null_doy_asset_ids,
            season_removed=season_filter_removed,
            total_candidates=len(candidates),
            matched=season_matches,
            filtered=filtered_count,
            pool_after_season=len(working_candidates),
        )

        target_wave_value = float(target_wave_score)

        # Apply calm seas guard rules
        calm_guard_active = False
        calm_guard_filtered: list[str] = []
        calm_guard_threshold_wave = 2.0
        calm_guard_threshold_conf = 0.85
        known_wave_count = 0
        unknown_wave_count = 0

        # Count wave score distribution
        for candidate in working_candidates:
            asset_obj = candidate["asset"]
            wave_val_candidate = getattr(asset_obj, "wave_score_0_10", None)
            if wave_val_candidate is None:
                wave_val_candidate = candidate.get("photo_wave") or candidate.get("wave_score")
            if wave_val_candidate is not None:
                known_wave_count += 1
            else:
                unknown_wave_count += 1

        if target_wave_score == 0:
            calm_guard_active = True
            # Filter out assets where wave_score_0_10 >= 2 AND wave_conf >= 0.85
            filtered_candidates = []
            for candidate in working_candidates:
                asset_obj = candidate["asset"]
                wave_score_val = getattr(asset_obj, "wave_score_0_10", None)
                wave_conf_val = getattr(asset_obj, "wave_conf", None)

                # Fallback to candidate dict if asset fields are None
                if wave_score_val is None:
                    wave_score_val = candidate.get("photo_wave") or candidate.get("wave_score")

                should_filter = False
                if wave_score_val is not None and wave_conf_val is not None:
                    try:
                        wave_float = float(wave_score_val)
                        conf_float = float(wave_conf_val)
                        if (
                            wave_float >= calm_guard_threshold_wave
                            and conf_float >= calm_guard_threshold_conf
                        ):
                            should_filter = True
                            calm_guard_filtered.append(str(asset_obj.id))
                    except (TypeError, ValueError):
                        pass

                if not should_filter:
                    filtered_candidates.append(candidate)

            working_candidates = filtered_candidates

            sea_log(
                "calm_guard",
                active=True,
                target_wave=target_wave_score,
                threshold_wave=calm_guard_threshold_wave,
                threshold_conf=calm_guard_threshold_conf,
                known_wave=known_wave_count,
                unknown_wave=unknown_wave_count,
                filtered_ids_count=len(calm_guard_filtered),
                filtered_ids=calm_guard_filtered,
                pool_before=known_wave_count + unknown_wave_count,
                pool_after_calm_guard=len(working_candidates),
            )
        else:
            sea_log(
                "wave_source",
                known_wave=known_wave_count,
                unknown_wave=unknown_wave_count,
                using_col="wave_score_0_10",
            )

        allowed_hard = set(allowed_photo_skies)

        stage_sequence: list[StageConfig] = [
            STAGE_CONFIGS["B0"],
            STAGE_CONFIGS["B1"],
            STAGE_CONFIGS["B2"],
            STAGE_CONFIGS["AN"],
        ]

        def build_corridor(stage_cfg: StageConfig) -> tuple[float, float]:
            tolerance = stage_cfg.wave_tolerance
            low = max(0.0, target_wave_value - tolerance)
            high = min(10.0, target_wave_value + tolerance)
            return low, high

        def evaluate_stage_candidate(
            candidate: dict[str, Any],
            stage_cfg: StageConfig,
            corridor: tuple[float, float],
        ) -> tuple[float, dict[str, Any]] | None:
            asset_obj = candidate["asset"]
            # Use wave_score_0_10 as primary source (not obsolete vision_wave_score)
            wave_score_0_10 = getattr(asset_obj, "wave_score_0_10", None)
            wave_conf = getattr(asset_obj, "wave_conf", None)
            vision_sky_bucket_val = getattr(asset_obj, "vision_sky_bucket", None)

            # Prefer wave_score_0_10 from asset object, then candidate dict values
            photo_wave = candidate.get("photo_wave")
            if photo_wave is None and wave_score_0_10 is not None:
                photo_wave = wave_score_0_10
            if photo_wave is None:
                photo_wave = candidate.get("wave_score")
            photo_wave_val = None
            photo_wave_conf = wave_conf
            if photo_wave is not None:
                try:
                    photo_wave_val = float(photo_wave)
                except (TypeError, ValueError):
                    photo_wave_val = None

            wave_delta = None
            if photo_wave_val is not None:
                wave_delta = abs(photo_wave_val - target_wave_value)
            elif target_wave_value is not None:
                wave_delta = None

            sky_visible = candidate.get("sky_visible")

            photo_sky = candidate.get("photo_sky_struct")
            similarity = sky_similarity(photo_sky, allowed_hard)
            season_match = candidate.get("season_match")

            components: dict[str, float] = {
                "SkyMatchBonus": 0.0,
                "SkyUnknownPenalty": 0.0,
                "SkyMismatchPenalty": 0.0,
                "FalseSkyPenalty": 0.0,
                "StrictSkyPenalty": 0.0,
                "RequiredSkyPenalty": 0.0,
                "WaveDeltaPenalty": 0.0,
                "WaveCorridorPenalty": 0.0,
                "CalmWavePenalty": 0.0,
                "CalmWaveBonus": 0.0,
                "CalmGuardNullWavePenalty": 0.0,
                "SeasonMismatchPenalty": 0.0,
                "NoDoyPenalty": 0.0,
                "AgeBonus": 0.0,
                "VisibleSkyBonus": 0.0,
            }

            score = 0.0
            age_bonus_val = float(candidate.get("age_bonus") or 0.0)
            components["AgeBonus"] = age_bonus_val
            score += age_bonus_val

            # CalmWaveBonus: only grant when wave_score_0_10=0 AND wave_conf >= 0.85
            if target_wave_score <= 2 and photo_wave_val is not None:
                if photo_wave_val == 0.0 and (photo_wave_conf is None or photo_wave_conf >= 0.85):
                    calm_bonus = 5.0
                    components["CalmWaveBonus"] = calm_bonus
                    score += calm_bonus

            wave_penalty = calc_wave_penalty(photo_wave_val, target_wave_value, stage_cfg)
            components["WaveDeltaPenalty"] = wave_penalty
            score -= wave_penalty

            # Apply calm guard null wave penalty for B0/B1 stages
            if calm_guard_active and photo_wave_val is None and stage_cfg.name in {"B0", "B1"}:
                calm_guard_null_penalty = 0.8
                components["CalmGuardNullWavePenalty"] = calm_guard_null_penalty
                score -= calm_guard_null_penalty

            if wave_delta is not None and wave_delta > stage_cfg.wave_tolerance:
                overshoot = wave_delta - stage_cfg.wave_tolerance
                corridor_penalty = overshoot * stage_cfg.outside_corridor_multiplier
                if corridor_penalty:
                    corridor_penalty = round(corridor_penalty, 6)
                    components["WaveCorridorPenalty"] = corridor_penalty
                    score -= corridor_penalty

            if (
                stage_cfg.calm_wave_cap is not None
                and target_wave_score <= 2
                and photo_wave_val is not None
                and photo_wave_val > stage_cfg.calm_wave_cap
            ):
                calm_penalty = stage_cfg.calm_wave_penalty
                if calm_penalty:
                    components["CalmWavePenalty"] = calm_penalty
                    score -= calm_penalty

            if sky_visible is None:
                components["SkyUnknownPenalty"] += stage_cfg.unknown_sky_penalty
                score -= stage_cfg.unknown_sky_penalty
                if not stage_cfg.allow_unknown_sky and stage_cfg.strict_unknown_sky_penalty:
                    components["StrictSkyPenalty"] += stage_cfg.strict_unknown_sky_penalty
                    score -= stage_cfg.strict_unknown_sky_penalty
            elif sky_visible is False:
                components["SkyMismatchPenalty"] += stage_cfg.mismatch_penalty
                score -= stage_cfg.mismatch_penalty
                if not stage_cfg.allow_false_sky and stage_cfg.false_sky_penalty:
                    components["FalseSkyPenalty"] += stage_cfg.false_sky_penalty
                    score -= stage_cfg.false_sky_penalty
            elif (
                sky_visible is True
                and stage_cfg.require_visible_sky
                and stage_cfg.visible_sky_bonus
            ):
                components["VisibleSkyBonus"] = stage_cfg.visible_sky_bonus
                score += stage_cfg.visible_sky_bonus

            if similarity == "match":
                components["SkyMatchBonus"] += stage_cfg.match_bonus
                score += stage_cfg.match_bonus
            elif similarity == "close":
                close_bonus = round(stage_cfg.match_bonus * 0.6, 3)
                components["SkyMatchBonus"] += close_bonus
                score += close_bonus
            elif similarity == "mismatch" and components["SkyMismatchPenalty"] == 0.0:
                components["SkyMismatchPenalty"] += stage_cfg.mismatch_penalty
                score -= stage_cfg.mismatch_penalty

            if stage_cfg.require_allowed_sky:
                if (
                    similarity != "match"
                    and photo_sky is not None
                    and photo_sky.weather_tag != "unknown"
                ):
                    penalty = stage_cfg.required_sky_penalty
                    if penalty:
                        components["RequiredSkyPenalty"] += penalty
                        score -= penalty
                elif photo_sky is None or (photo_sky and photo_sky.weather_tag == "unknown"):
                    penalty = stage_cfg.required_sky_unknown_penalty
                    if penalty:
                        components["RequiredSkyPenalty"] += penalty
                        score -= penalty

            if stage_cfg.season_required and not season_match:
                season_penalty = stage_cfg.season_penalty + stage_cfg.season_mismatch_extra
                components["SeasonMismatchPenalty"] = season_penalty
                score -= season_penalty
            elif not season_match:
                season_penalty = stage_cfg.season_penalty + stage_cfg.season_mismatch_extra
                components["SeasonMismatchPenalty"] = season_penalty
                score -= season_penalty

            if (
                stage_cfg.name == "AN"
                and candidate.get("photo_doy") is None
                and candidate.get("shot_doy") is None
            ):
                components["NoDoyPenalty"] = 0.3
                score -= 0.3

            reasons = {
                "stage": stage_cfg.name,
                "wave_corridor": [round(val, 2) for val in corridor],
                "wave_delta": round(wave_delta, 2) if wave_delta is not None else None,
                "score_components": {k: round(v, 3) for k, v in components.items()},
                "season_match": season_match,
                "sky_visible": sky_visible,
                "sky_similarity": similarity,
            }
            return score, reasons

        selected_candidate: dict[str, Any] | None = None
        selected_details: dict[str, Any] = {}
        sky_critical_mismatch = False
        pool_counts: dict[str, int] = {
            "pool_after_season": len(working_candidates),
            "pool_after_B0": 0,
            "pool_after_B1": 0,
            "pool_after_B2": 0,
            "pool_after_AN": 0,
        }

        for stage_cfg in stage_sequence:
            pool_before = len(working_candidates)
            corridor = build_corridor(stage_cfg)
            stage_results: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
            for candidate in working_candidates:
                evaluation = evaluate_stage_candidate(candidate, stage_cfg, corridor)
                if evaluation is None:
                    continue
                score, reasons = evaluation
                stage_results.append((score, reasons, candidate))

            # B0 corridor enforcement: only allow wave_score_0_10 in [0, 1]
            if stage_cfg.name == "B0" and target_wave_score <= 1:
                b0_passed_known = 0
                b0_passed_unknown = 0
                b0_excluded_known = 0
                filtered_b0_results = []
                for score_val, reason_val, candidate_val in stage_results:
                    asset_obj = candidate_val["asset"]
                    wave_score_val = getattr(asset_obj, "wave_score_0_10", None)
                    if wave_score_val is None:
                        wave_score_val = candidate_val.get("photo_wave") or candidate_val.get(
                            "wave_score"
                        )

                    if wave_score_val is not None:
                        try:
                            wave_float = float(wave_score_val)
                            if wave_float <= 1.0:
                                filtered_b0_results.append((score_val, reason_val, candidate_val))
                                b0_passed_known += 1
                            else:
                                b0_excluded_known += 1
                        except (TypeError, ValueError):
                            filtered_b0_results.append((score_val, reason_val, candidate_val))
                            b0_passed_unknown += 1
                    else:
                        # Unknown wave: pass corridor but will receive penalty later
                        filtered_b0_results.append((score_val, reason_val, candidate_val))
                        b0_passed_unknown += 1

                stage_results = filtered_b0_results
                sea_log(
                    "attempt:B0 corridor_check",
                    corridor_range="0-1",
                    in_range=b0_passed_known,
                    unknown=b0_passed_unknown,
                    excluded=b0_excluded_known,
                    pool_before=pool_before,
                    pool_after=len(stage_results),
                    wave_col="wave_score_0_10",
                )

            pool_counts[f"pool_after_{stage_cfg.name}"] = len(stage_results)

            if stage_results:

                def _ordering(
                    item: tuple[float, dict[str, Any], dict[str, Any]],
                ) -> tuple[float, float, int]:
                    score_value, _reason_payload, candidate_payload = item
                    age_value = float(candidate_payload.get("age_bonus") or 0.0)
                    asset_obj = candidate_payload["asset"]
                    try:
                        asset_id_int = int(str(asset_obj.id))
                    except Exception:
                        asset_id_int = 0
                    return (round(score_value, 4), round(age_value, 4), -asset_id_int)

                sorted_results = sorted(stage_results, key=_ordering, reverse=True)
            else:
                sorted_results = []

            sky_policy_desc = "strict_visible" if stage_cfg.require_visible_sky else "relaxed"
            if stage_cfg.allow_false_sky:
                sky_policy_desc = "permissive"
            sea_log(
                f"attempt:{stage_cfg.name}",
                sky_policy=sky_policy_desc,
                corridor=corridor,
                pool_before=pool_before,
                pool_after=len(sorted_results),
            )

            sea_log(
                "pool_counts",
                pool_after_season=pool_counts.get("pool_after_season", 0),
                pool_after_B0=pool_counts.get("pool_after_B0", 0),
                pool_after_B1=pool_counts.get("pool_after_B1", 0),
                pool_after_B2=pool_counts.get("pool_after_B2", 0),
                pool_after_AN=pool_counts.get("pool_after_AN", 0),
            )

            for idx, (score_value, reason_payload, candidate_payload) in enumerate(
                sorted_results[:5], start=1
            ):
                asset_obj = candidate_payload["asset"]
                component_payload = reason_payload.get("score_components") or {}
                photo_wave_log = candidate_payload.get("photo_wave")
                wave_delta_log = reason_payload.get("wave_delta")
                sky_bucket_log = candidate_payload.get("photo_sky_struct")
                sky_bucket_str = (
                    sky_bucket_log.weather_tag if hasattr(sky_bucket_log, "weather_tag") else None
                )
                age_bonus_log = candidate_payload.get("age_bonus")
                sea_log(
                    f"top5:{stage_cfg.name}",
                    rank=idx,
                    asset_id=getattr(asset_obj, "id", None),
                    wave_target=target_wave_score,
                    wave_photo=photo_wave_log,
                    delta=wave_delta_log,
                    sky_photo=sky_bucket_str,
                    penalties=component_payload,
                    total_score=score_value,
                    freshness_bonus=age_bonus_log,
                    sky_visible=candidate_payload.get("sky_visible"),
                    photo_sky=_sky_token(candidate_payload.get("photo_sky_struct")),
                    photo_sky_daypart=candidate_payload.get("photo_sky_daypart"),
                    season_match=candidate_payload.get("season_match"),
                    photo_doy=candidate_payload.get("photo_doy")
                    or candidate_payload.get("shot_doy"),
                )

            if sorted_results:
                best_score, best_reasons, best_candidate = sorted_results[0]
                best_reasons["stage"] = stage_cfg.name

                # Determine selection reason
                selection_reason = "highest_score"
                if len(sorted_results) > 1:
                    second_score = sorted_results[1][0]
                    if abs(best_score - second_score) < 0.01:
                        # Tie-breaker scenarios
                        best_age_bonus = float(best_candidate.get("age_bonus") or 0.0)
                        second_age_bonus = float(sorted_results[1][2].get("age_bonus") or 0.0)
                        if abs(best_age_bonus - second_age_bonus) > 0.01:
                            selection_reason = "freshest"
                        else:
                            selection_reason = "random_tiebreak"
                    else:
                        # Check what made the difference
                        best_wave = best_candidate.get("photo_wave")
                        if best_wave is not None:
                            wave_delta = abs(float(best_wave) - target_wave_value)
                            if wave_delta < 0.5:
                                selection_reason = "lowest_wave_delta"
                        components = best_reasons.get("score_components", {})
                        sky_match_bonus = components.get("SkyMatchBonus", 0.0)
                        if sky_match_bonus > 1.0:
                            selection_reason = "sky_match"
                        age_bonus = components.get("AgeBonus", 0.0)
                        if age_bonus > 1.0:
                            selection_reason = "freshest"

                selected_candidate = best_candidate
                selected_details = {
                    "score": best_score,
                    "reasons": best_reasons,
                    "stage": stage_cfg.name,
                    "corridor": corridor,
                    "selection_reason": selection_reason,
                }

                if stage_cfg.name == "AN" and clear_guard_hard:
                    chosen_sky = best_candidate.get("photo_sky_struct")
                    if chosen_sky and chosen_sky.weather_tag in {"mostly_cloudy", "overcast"}:
                        sky_critical_mismatch = True

                break

        if selected_candidate is None:
            sea_log(
                "selection_fallback",
                reason="age_priority",
                candidate_total=len(working_candidates),
            )
            if not working_candidates:
                sea_log("selection_fallback", reason="empty_pool", candidate_total=0)
                logging.warning("SEA_RUBRIC fallback_no_candidates sea_id=%s", sea_id)
                return False

            def _fallback_key(candidate: dict[str, Any]) -> datetime:
                last_used_at = candidate.get("last_used_at")
                if isinstance(last_used_at, datetime):
                    return last_used_at
                return datetime.min

            selected_candidate = min(working_candidates, key=_fallback_key)
            selected_details = {
                "score": 0.0,
                "reasons": {"fallback": "age_priority"},
                "stage": "fallback_age",
                "corridor": build_corridor(STAGE_CONFIGS["AN"]),
                "selection_reason": "age_priority_fallback",
            }

        asset = selected_candidate["asset"]
        sunset_selected = bool(selected_candidate.get("is_sunset"))
        sky_visible = selected_candidate.get("sky_visible", True)
        chosen_photo_sky = selected_candidate.get("photo_sky_struct")
        chosen_photo_sky_tag = (
            chosen_photo_sky.weather_tag if isinstance(chosen_photo_sky, NormalizedSky) else None
        )

        want_sunset = (
            storm_state != "strong_storm"
            and sky_visible
            and not (clear_guard_hard and chosen_photo_sky_tag in {"mostly_cloudy", "overcast"})
        )
        if enable_facts:
            fact_sentence_value, fact_id_value, fact_info = self._prepare_sea_fact(
                sea_id=sea_id,
                storm_state=storm_state,
                enable_facts=enable_facts,
                now=now,
            )
            fact_sentence = fact_sentence_value.strip() if fact_sentence_value else None
            fact_id = fact_id_value
        else:
            logging.info("SEA_RUBRIC facts skip reason=disabled")
            fact_sentence = None
            fact_id = None
            fact_info = {"reason": "disabled"}
        fact_log_info: dict[str, Any] | None
        if isinstance(fact_info, dict) and fact_info:
            fact_log_info = {
                "reason": fact_info.get("reason"),
                "fallback": fact_info.get("fallback"),
                "window_days": fact_info.get("window_days"),
                "candidates_count": len(fact_info.get("candidates") or []),
                "recent_ids_count": len(fact_info.get("recent_ids") or []),
                "weights_count": len(fact_info.get("weights") or []),
                "chosen_id": fact_info.get("chosen_id"),
            }
        else:
            fact_log_info = None
        photo_wave_selected = selected_candidate.get("photo_wave") or selected_candidate.get(
            "wave_score"
        )
        wave_delta_selected = None
        if photo_wave_selected is not None:
            wave_delta_selected = abs(float(photo_wave_selected) - target_wave_value)

        chosen_sky_struct = selected_candidate.get("photo_sky_struct")
        sky_photo_selected = (
            chosen_sky_struct.weather_tag if hasattr(chosen_sky_struct, "weather_tag") else None
        )

        sea_log(
            "selected",
            stage=selected_details.get("stage"),
            asset_id=asset.id,
            wave_target=target_wave_score,
            wave_photo=photo_wave_selected,
            delta=wave_delta_selected,
            sky_photo=sky_photo_selected,
            penalties=selected_details.get("reasons", {}).get("score_components"),
            total_score=selected_details.get("score"),
            reason=selected_details.get("selection_reason"),
            shot_doy=selected_candidate.get("shot_doy"),
            photo_doy=selected_candidate.get("photo_doy"),
            photo_sky=_sky_token(selected_candidate.get("photo_sky_struct")),
            photo_sky_daypart=selected_candidate.get("photo_sky_daypart"),
            season_match=selected_candidate.get("season_match"),
            sunset_selected=sunset_selected,
            want_sunset=want_sunset,
            storm_persisting=storm_persisting,
            wave_corridor=selected_details.get("corridor"),
            sky_visible=sky_visible,
            sky_critical_mismatch=sky_critical_mismatch,
            fact_id=fact_id,
            fact_info=fact_log_info,
        )

        timeline["build_context"] = round((time.perf_counter() - step_start) * 1000, 2)

        place_hashtag: str | None = None
        if asset.latitude is not None and asset.longitude is not None:
            geo_data = await self._reverse_geocode(asset.latitude, asset.longitude)
            if geo_data and geo_data.get("city"):
                city_name = str(geo_data["city"]).strip()
                if city_name:
                    sanitized_city = re.sub(r"\s+", "", city_name)
                    if sanitized_city:
                        place_hashtag = f"#{sanitized_city}"

        step_start = time.perf_counter()
        caption_text, model_hashtags, openai_metadata = (
            await self._generate_sea_caption_with_timeout(
                storm_state=storm_state,
                storm_persisting=storm_persisting,
                wave_height_m=wave_height_value,
                wave_score=wave_score,
                wind_class=wind_class,
                wind_ms=wind_ms,
                wind_kmh=wind_kmh,
                clouds_label=clouds_label,
                sunset_selected=sunset_selected,
                want_sunset=want_sunset,
                place_hashtag=place_hashtag,
                fact_sentence=fact_sentence,
                now_local_iso=now_local_iso,
                day_part=day_part,
                tz_name=tz_name,
                job=job,
            )
        )
        stripped_caption = self.strip_header(caption_text)
        raw_caption_text = (stripped_caption or caption_text or "").strip()

        fallback_seed = ""
        if storm_state == "strong_storm":
            fallback_seed = "Сегодня сильный шторм на море — волны гремят у самого берега."
        elif storm_state == "storm":
            if storm_persisting:
                fallback_seed = "Продолжает штормить на море — волны всё ещё бьют о берег."
            else:
                fallback_seed = "Сегодня шторм на море — волны упрямо разбиваются о берег."
        elif sunset_selected:
            fallback_seed = "Порадую закатом над морем — побережье дышит теплом."
        else:
            fallback_seed = "Порадую вас морем — побережье зовёт вдохнуть глубже."
        if wind_class == "very_strong":
            fallback_seed += " Ветер срывает шапки на набережной."
        elif wind_class == "strong":
            fallback_seed += " Ветер ощутимо тянет к морю."
        if fact_sentence:
            fallback_seed += f" {fact_sentence}"
        fallback_sentences = [
            segment.strip()
            for segment in re.split(r"(?<=[.!?…])\s+", fallback_seed.strip())
            if segment.strip()
        ]
        fallback_caption_plain = (
            " ".join(fallback_sentences[:3]) if fallback_sentences else fallback_seed.strip()
        )
        main_plain = raw_caption_text or fallback_caption_plain
        main_plain = main_plain.strip()
        sanitized_main_plain, removed_tokens = sanitize_sea_intro(main_plain)
        removed_clean: list[str] = []
        for token in removed_tokens:
            trimmed = token.strip()
            if trimmed and trimmed not in removed_clean:
                removed_clean.append(trimmed)
        if removed_clean:
            logging.warning(
                "SEA_RUBRIC sanitize removed_tokens=%s",
                ", ".join(removed_clean),
            )
        main_plain = sanitized_main_plain

        def _ensure_paragraph_break(value: str) -> str:
            if not value or "\n\n" in value:
                return value
            match = re.search(r"([.!?…])\s+", value)
            if match and match.end() < len(value):
                head_end = match.end(1)
                tail_start = match.end()
                head = value[:head_end]
                tail = value[tail_start:]
                return f"{head}\n\n{tail.lstrip()}"
            return value

        main_plain = _ensure_paragraph_break(main_plain)
        fallback_caption_plain = _ensure_paragraph_break(fallback_caption_plain)

        exclusions = self._hashtag_exclusions(rubric.code)
        deduped_model_tags = self._deduplicate_hashtags(model_hashtags or [])
        seen_tags: set[str] = set()
        final_hashtags: list[str] = []

        def append_tag(tag: str) -> None:
            text = str(tag or "").strip()
            if not text:
                return
            normalized = text if text.startswith("#") else f"#{text.lstrip('#')}"
            stripped = normalized.lstrip("#").strip()
            if not stripped:
                return
            key = stripped.lower()
            if key in exclusions or key in seen_tags:
                return
            seen_tags.add(key)
            final_hashtags.append(normalized)

        append_tag("#море")
        append_tag("#БалтийскоеМоре")
        if place_hashtag:
            append_tag(place_hashtag)
        for tag in deduped_model_tags:
            append_tag(tag)

        active_hashtags = list(final_hashtags)

        link_block = build_rubric_link_block("sea")

        caption_text = main_plain or fallback_caption_plain
        if not caption_text:
            caption_text = fallback_caption_plain
        caption_text = _ensure_paragraph_break(caption_text.strip())

        def _trim_plain_to_html_limit(value: str, html_limit: int) -> str:
            if html_limit <= 0:
                return ""
            if len(html.escape(value)) <= html_limit:
                return value
            low, high = 0, len(value)
            best = ""
            while low <= high:
                mid = (low + high) // 2
                candidate = value[:mid]
                if len(html.escape(candidate)) <= html_limit:
                    best = candidate
                    low = mid + 1
                else:
                    high = mid - 1
            return best

        def compose_caption(
            caption_plain: str,
            include_link: bool,
            hashtags: list[str],
        ) -> tuple[str, list[str]]:
            caption_html = html.escape(caption_plain) if caption_plain else ""
            hashtags_line = " ".join(hashtags)
            hashtags_html = html.escape(hashtags_line) if hashtags_line else ""
            composed: list[str] = []
            if caption_html:
                composed.append(caption_html)
            if include_link and link_block:
                composed.append(link_block)
            if hashtags_html:
                composed.append(hashtags_html)
            full = "\n\n".join(composed)
            return full, composed

        include_link_block = bool(link_block)
        caption_plain_final = caption_text
        full_caption, caption_segments = compose_caption(
            caption_text,
            include_link_block,
            active_hashtags,
        )
        caption_segments = [segment for segment in caption_segments if segment]
        full_caption = "\n\n".join(caption_segments)
        logging.info("SEA_RUBRIC caption_length=%s", len(full_caption))

        CAP_LIMIT = 990
        # Allow moderate overflow so natural captions remain intact while
        # still trimming aggressively when captions are drastically over budget.
        TRIM_OVERFLOW_THRESHOLD = 200
        original_len = len(full_caption)
        overflow = original_len - CAP_LIMIT
        if overflow > 0:
            should_trim = overflow >= TRIM_OVERFLOW_THRESHOLD
            if should_trim:
                reserved = 0
                for segment in caption_segments[1:]:
                    reserved += len("\n\n") + len(segment)
                available_html = CAP_LIMIT - reserved
                if available_html < 0:
                    available_html = 0
                if caption_segments and caption_text:
                    trimmed_plain = _trim_plain_to_html_limit(caption_text, available_html)
                    if trimmed_plain:
                        trimmed_plain = _ensure_paragraph_break(trimmed_plain)
                    caption_plain_final = trimmed_plain if trimmed_plain else ""
                    caption_segments[0] = html.escape(trimmed_plain) if trimmed_plain else ""
                caption_segments = [segment for segment in caption_segments if segment]
                full_caption = "\n\n".join(caption_segments)
                if len(full_caption) > CAP_LIMIT:
                    full_caption = full_caption[:CAP_LIMIT].rstrip()
                    caption_segments = [
                        segment for segment in full_caption.split("\n\n") if segment
                    ]
                logging.warning(
                    "SEA_RUBRIC caption_trim applied original=%d final=%d",
                    original_len,
                    len(full_caption),
                )
        final_hashtags = list(active_hashtags)

        rendered_segments = [segment for segment in full_caption.split("\n\n") if segment]
        caption_plain_segments: list[str] = []
        for segment in rendered_segments:
            if segment == link_block or segment.startswith("<a "):
                break
            if segment.startswith("#"):
                break
            caption_plain_segments.append(html.unescape(segment))
        if caption_plain_segments:
            caption_plain_final = "\n\n".join(caption_plain_segments)
        caption_text = caption_plain_final

        caption_paragraphs = len([p for p in caption_text.split("\n\n") if p.strip()])
        has_link_final = bool(include_link_block and link_block and link_block in full_caption)
        has_hashtags_final = bool(final_hashtags)
        block_count = 1 + int(has_link_final) + int(has_hashtags_final)
        logging.info(
            "SEA_RUBRIC compose_blocks caption_paragraphs=%d has_link=%s has_hashtags=%s blocks=%d",
            caption_paragraphs,
            has_link_final,
            has_hashtags_final,
            block_count,
        )

        timeline["openai_generate_caption"] = round((time.perf_counter() - step_start) * 1000, 2)

        step_start = time.perf_counter()
        logging.info(
            "SEA_RUBRIC SEND channel_id=%s prod=%d message_type=photo",
            channel_id,
            int(is_prod),
        )
        file_id = asset.file_id
        if file_id:
            response = await self.api_request(
                "sendPhoto",
                {
                    "chat_id": channel_id,
                    "photo": file_id,
                    "caption": full_caption,
                    "parse_mode": "HTML",
                },
            )
        else:
            source_path, should_cleanup = await self._ensure_asset_source(asset)
            if not source_path:
                logging.warning("Asset %s missing source file for sea publication", asset.id)
                return False
            try:
                file_data = Path(source_path).read_bytes()
            except Exception:
                logging.exception("Failed to read source file for asset %s", asset.id)
                if should_cleanup:
                    self._remove_file(source_path)
                return False
            finally:
                if should_cleanup:
                    self._remove_file(source_path)
                    try:
                        self.data.update_asset(asset.id, local_path=None)
                    except Exception:
                        logging.exception(
                            "Failed to clear local_path for asset %s after sea publication",
                            asset.id,
                        )

            response = await self.api_request(
                "sendPhoto",
                {
                    "chat_id": channel_id,
                    "caption": full_caption,
                    "parse_mode": "HTML",
                },
                files={"photo": ("photo.jpg", file_data)},
            )

        tg_elapsed = round((time.perf_counter() - step_start) * 1000, 2)
        timeline["sendPhoto"] = tg_elapsed

        tg_rate_limited = 0
        if not response.get("ok"):
            error_code = response.get("error_code")
            if error_code in {429, 420}:
                tg_rate_limited = 1
            logging.error(
                "SEA_RUBRIC tg_api_error status=%s time_ms=%.1f tg_rate_limited=%d response=%s",
                error_code or "unknown",
                tg_elapsed,
                tg_rate_limited,
                response,
            )
            return False

        logging.info(
            "SEA_RUBRIC tg_api_success status=200 time_ms=%.1f tg_rate_limited=0",
            tg_elapsed,
        )
        result_payload = response.get("result")
        if isinstance(result_payload, dict):
            message_id = int(result_payload.get("message_id") or 0)
        else:
            message_id = 0

        self.data.mark_assets_used([asset.id])
        scoring_payload = dict(selected_details.get("reasons") or {})
        corridor_values = scoring_payload.get("wave_corridor")
        if isinstance(corridor_values, tuple):
            scoring_payload["wave_corridor"] = [round(value, 2) for value in corridor_values]

        total_elapsed = round((time.perf_counter() - publish_start_time) * 1000, 2)
        timeline["done"] = total_elapsed

        logging.info(
            "SEA_RUBRIC PUBLISH_TIMELINE sea_id=%s total_ms=%.1f start=%.1f read_sea_cache=%.1f "
            "select_candidates=%.1f build_context=%.1f openai_generate_caption=%.1f sendPhoto=%.1f",
            sea_id,
            total_elapsed,
            timeline.get("start", 0.0),
            timeline.get("read_sea_cache", 0.0),
            timeline.get("select_candidates", 0.0),
            timeline.get("build_context", 0.0),
            timeline.get("openai_generate_caption", 0.0),
            timeline.get("sendPhoto", 0.0),
        )

        metadata = {
            "rubric_code": rubric.code,
            "asset_ids": [asset.id],
            "test": test,
            "sea_id": sea_id,
            "enable_facts": enable_facts,
            "storm_state": storm_state,
            "storm_persisting": storm_persisting,
            "storm_persisting_reason": storm_persisting_reason,
            "wave_height_m": wave_height_value,
            "wave_score": wave_score,
            "water_temp": water_temp,
            "wind_speed": wind_ms,
            "wind_speed_ms": wind_ms,
            "wind_speed_kmh": wind_kmh,
            "wind_gust_ms": wind_gust_ms,
            "wind_gust_kmh": wind_gust_kmh,
            "wind_units": wind_units,
            "wind_gust_units": wind_gust_units,
            "wind_time_ref": wind_time_ref,
            "wind_class": wind_class,
            "cloud_cover_pct": cloud_cover_pct,
            "clear_guard_hard": clear_guard_hard,
            "clear_guard_soft": clear_guard_soft,
            "sky_bucket": sky_bucket,
            "sky_daypart": day_part,
            "allowed_photo_sky": _sky_tokens(allowed_photo_skies),
            "selected_photo_sky": selected_candidate.get("photo_sky"),
            "selected_photo_sky_token": _sky_token(selected_candidate.get("photo_sky_struct")),
            "selected_photo_sky_daypart": selected_candidate.get("photo_sky_daypart"),
            "selected_photo_sky_weather": selected_candidate.get("photo_sky_weather"),
            "sky_visible": sky_visible,
            "sky_critical_mismatch": sky_critical_mismatch,
            "want_sunset": want_sunset,
            "sunset_selected": sunset_selected,
            "season_match": selected_candidate.get("season_match"),
            "score": selected_details.get("score"),
            "scoring": scoring_payload,
            "stage": selected_details.get("stage"),
            "wave_corridor": [round(val, 2) for val in selected_details.get("corridor", [])],
            "pool_counts": pool_counts,
            "caption": caption_text,
            "hashtags": final_hashtags,
            "place_hashtag": place_hashtag,
            "fact_id": fact_id,
            "fact_text": fact_sentence,
            "facts_info": fact_info,
            "openai_metadata": openai_metadata,
            "timeline_ms": timeline,
        }
        self.data.record_post_history(
            channel_id,
            message_id,
            asset.id,
            rubric.id,
            metadata,
        )
        await self._cleanup_assets([asset])

        if test and initiator_id:
            wave_text = f"{wave_height_value:.2f}" if wave_height_value is not None else "н/д"
            resolved_wind_ms = wind_ms
            resolved_wind_kmh = wind_kmh
            resolved_wind_class = wind_class
            if resolved_wind_ms is None or resolved_wind_kmh is None:
                fallback_speed, fallback_class = self._get_sea_wind(sea_id)
                if fallback_speed is not None:
                    logging.warning(
                        "SEA_RUBRIC test_wind_fallback sea_id=%s source=nearest_cache",
                        sea_id,
                    )
                    resolved_wind_ms = fallback_speed
                    resolved_wind_kmh = fallback_speed * 3.6
                    resolved_wind_class = fallback_class or classify_wind_kph(resolved_wind_kmh)
                else:
                    logging.warning(
                        "SEA_RUBRIC test_wind_cache_missing sea_id=%s",
                        sea_id,
                    )
            if resolved_wind_ms is None or resolved_wind_kmh is None:
                logging.warning("SEA_RUBRIC test_wind_zero_fallback sea_id=%s", sea_id)
                resolved_wind_ms = 0.0
                resolved_wind_kmh = 0.0
                resolved_wind_class = "n/a"
            wind_class_display = resolved_wind_class or "n/a"
            resolved_wind_gust_kmh = wind_gust_kmh
            if resolved_wind_gust_kmh is None and wind_gust_ms is not None:
                resolved_wind_gust_kmh = wind_gust_ms * 3.6
            wind_line = (
                f"• Ветер: {resolved_wind_kmh:.0f} км/ч и {resolved_wind_ms:.1f} м/с "
                f"({wind_class_display})"
            )
            if resolved_wind_gust_kmh is not None:
                wind_line += f", порывы до {resolved_wind_gust_kmh:.0f} км/ч"
            wind_line += "."
            summary = (
                "🧪 Тестовая публикация «Море / Закат на море» успешно.\n"
                f"• Волна: {wave_text} м ({storm_state}).\n"
                f"{wind_line}\n"
                f"• Облачность: {clouds_label}."
            )
            notify_response = await self.api_request(
                "sendMessage",
                {"chat_id": initiator_id, "text": summary},
            )
            if notify_response.get("ok"):
                logging.info("SEA_RUBRIC test_notify sent to %s", initiator_id)
            else:
                logging.warning(
                    "SEA_RUBRIC test_notify failed for %s: %s",
                    initiator_id,
                    notify_response,
                )

        # After successful publish, handle prod deletion and inventory report
        await self._on_sea_publish_success([asset.id], is_prod=is_prod)
        await self._send_sea_inventory_report(is_prod=is_prod, initiator_id=initiator_id)

        return True

    async def _on_sea_publish_success(
        self,
        asset_ids: list[str],
        is_prod: bool,
    ) -> None:
        """Delete sea assets from TG + DB after successful prod publish."""
        if not is_prod:
            return

        # Fetch asset metadata
        placeholders = ",".join("?" * len(asset_ids))
        query = f"SELECT id, payload_json, tg_message_id FROM assets WHERE id IN ({placeholders})"
        rows = self.db.execute(query, asset_ids).fetchall()

        deleted_count = 0
        for row in rows:
            asset_id = row["id"]
            payload_json = row["payload_json"]
            tg_message_id = row["tg_message_id"]

            # Parse tg_chat_id and message_id from payload or tg_message_id
            chat_id = None
            msg_id = None

            if payload_json:
                try:
                    payload = json.loads(payload_json)
                    chat_id = payload.get("tg_chat_id")
                    msg_id = payload.get("message_id")
                except json.JSONDecodeError:
                    pass

            if (chat_id is None or msg_id is None) and tg_message_id:
                if ":" in str(tg_message_id):
                    parts = str(tg_message_id).split(":", 1)
                    try:
                        chat_id = int(parts[0])
                        msg_id = int(parts[1])
                    except ValueError:
                        pass
                else:
                    try:
                        msg_id = int(tg_message_id)
                    except ValueError:
                        pass

            # 1. Delete message from TG assets channel (hard delete, no trash)
            if chat_id and msg_id:
                try:
                    await self.api_request(
                        "deleteMessage",
                        {
                            "chat_id": chat_id,
                            "message_id": msg_id,
                        },
                    )
                except Exception as e:
                    error_str = str(e).lower()
                    # If message not found, log but continue to DB delete
                    if (
                        "message to delete not found" in error_str
                        or "message not found" in error_str
                    ):
                        logging.warning(
                            "SEA_TG_DELETE_SKIP_NOT_FOUND asset_id=%s",
                            asset_id,
                        )
                    else:
                        logging.warning(
                            "SEA_TG_DELETE_FAILED asset_id=%s err=%s",
                            asset_id,
                            str(e)[:200],
                        )

            # 2. Delete record from DB (cascades handle tags/relations)
            try:
                self.db.execute("DELETE FROM assets WHERE id=?", (asset_id,))
                self.db.commit()
                deleted_count += 1
            except Exception as e:
                logging.error("SEA_DB_DELETE_FAILED asset_id=%s err=%s", asset_id, str(e)[:200])

        logging.info(
            "SEA_ASSET_DELETED prod=1 count=%d ids=%s",
            deleted_count,
            [str(row["id"]) for row in rows],
        )

    async def _send_sea_inventory_report(
        self, is_prod: bool, initiator_id: int | None = None
    ) -> None:
        """Send sea assets inventory breakdown to operator chat."""
        use_sea_assets = self._database_has_table_or_view("sea_assets")
        if use_sea_assets:
            total_row = self.db.execute("SELECT COUNT(*) AS cnt FROM sea_assets").fetchone()
            sky_rows = self.db.execute(
                """
                SELECT sky_bucket, COUNT(*) AS cnt
                FROM sea_assets
                WHERE sky_bucket IS NOT NULL
                GROUP BY sky_bucket
                """
            ).fetchall()
            wave_rows = self.db.execute(
                """
                SELECT wave_score_0_10 AS wave_score, COUNT(*) AS cnt
                FROM sea_assets
                WHERE wave_score_0_10 IS NOT NULL
                GROUP BY wave_score_0_10
                ORDER BY wave_score_0_10
                """
            ).fetchall()
        else:
            total_row = self.db.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM assets a
                WHERE LOWER(json_extract(a.payload_json, '$.vision_category')) = 'sea'
                """
            ).fetchone()
            sky_rows = self.db.execute(
                """
                SELECT a.vision_sky_bucket AS sky_bucket, COUNT(*) AS cnt
                FROM assets a
                WHERE LOWER(json_extract(a.payload_json, '$.vision_category')) = 'sea'
                  AND a.vision_sky_bucket IS NOT NULL
                GROUP BY a.vision_sky_bucket
                """
            ).fetchall()
            wave_rows = self.db.execute(
                """
                SELECT a.wave_score_0_10 AS wave_score, COUNT(*) AS cnt
                FROM assets a
                WHERE LOWER(json_extract(a.payload_json, '$.vision_category')) = 'sea'
                  AND a.wave_score_0_10 IS NOT NULL
                GROUP BY a.wave_score_0_10
                ORDER BY a.wave_score_0_10
                """
            ).fetchall()

        total_count = self._safe_int(total_row["cnt"]) if total_row else 0

        sky_counts: dict[str, int] = {}
        for row in sky_rows:
            sky_value = row["sky_bucket"]
            if sky_value is None:
                continue
            sky_counts[str(sky_value)] = self._safe_int(row["cnt"])

        wave_counts: dict[int, int] = {}
        for row in wave_rows:
            raw_wave = row["wave_score"]
            if raw_wave is None:
                continue
            try:
                wave_key = int(float(raw_wave))
            except (TypeError, ValueError):
                logging.debug("SEA_INVENTORY skip_wave_value value=%s", raw_wave)
                continue
            wave_counts[wave_key] = self._safe_int(row["cnt"])

        logging.info("SEA_INVENTORY_SKY_COUNTS raw=%s", dict(sky_counts))
        logging.info("SEA_INVENTORY_WAVE_COUNTS raw=%s", dict(wave_counts))

        logging.info(
            "SEA_INVENTORY_REPORT prod=%d total=%d sky_buckets=%s wave_scores=%s",
            int(is_prod),
            total_count,
            dict(sky_counts),
            dict(wave_counts),
        )

        # Build report text
        SKY_LABELS = {
            "clear": "Солнечно",
            "mostly_clear": "Преимущественно ясно",
            "partly_cloudy": "Переменная облачность",
            "mostly_cloudy": "Преимущественно облачно",
            "overcast": "Пасмурно",
        }

        report_lines = [
            f"🗂 Остатки фото «Море»: {total_count}\n",
            "Небо",
        ]

        for sky, label in SKY_LABELS.items():
            count = sky_counts.get(sky, 0)
            warning = " ⚠️ мало" if count < 10 else ""
            report_lines.append(f"• {label}: {count}{warning}")

        report_lines.append("\nВолнение (0–10)")

        for wave_score in range(11):
            count = wave_counts.get(wave_score, 0)
            if count == 0:
                continue  # Skip unreported wave scores
            warning = " ⚠️ мало" if count < 10 else ""
            if wave_score == 0:
                report_lines.append(f"• {wave_score}/10 (штиль): {count}{warning}")
            else:
                report_lines.append(f"• {wave_score}/10: {count}{warning}")

        report_text = "\n".join(report_lines)

        # Send to operator chat (send to initiator if provided, otherwise send to all superadmins)
        target_ids = [initiator_id] if initiator_id else self.get_superadmin_ids()

        for target_id in target_ids:
            try:
                await self.api_request(
                    "sendMessage",
                    {
                        "chat_id": target_id,
                        "text": report_text,
                    },
                )
                logging.info(
                    "SEA_INVENTORY_SENT prod=%d total=%d sky_buckets=%d wave_scores=%d target_id=%d",
                    int(is_prod),
                    total_count,
                    len(sky_counts),
                    len(wave_counts),
                    target_id,
                )
            except Exception as e:
                logging.error(
                    "SEA_INVENTORY_SEND_FAILED target_id=%d err=%s", target_id, str(e)[:200]
                )

    async def _handle_sea_no_candidates(
        self,
        rubric: Rubric,
        channel_id: int,
        *,
        sea_id: int,
        storm_state: str,
        storm_persisting: bool,
        storm_reason: str | None,
        wave_height: float | None,
        wave_score: float,
        wind_ms: float | None,
        wind_kmh: float | None,
        wind_class: str | None,
        clouds_label: str,
        sky_bucket: str,
        want_sunset: bool,
        sky_daypart: str | None,
        test: bool,
        initiator_id: int | None,
    ) -> None:
        logging.info(
            "SEA_RUBRIC skip_no_assets sea_id=%s storm_state=%s storm_persisting=%s",
            sea_id,
            storm_state,
            storm_persisting,
        )

        metadata = {
            "rubric_code": rubric.code,
            "asset_ids": [],
            "test": test,
            "sea_id": sea_id,
            "skip_reason": "no_candidates",
            "storm_state": storm_state,
            "storm_persisting": storm_persisting,
            "storm_persisting_reason": storm_reason,
            "wave_height_m": wave_height,
            "wave_score": wave_score,
            "wind_speed_ms": wind_ms,
            "wind_speed_kmh": wind_kmh,
            "wind_class": wind_class,
            "clouds_label": clouds_label,
            "sky_bucket": sky_bucket,
            "want_sunset": want_sunset,
            "sky_daypart": sky_daypart,
        }
        self.data.record_post_history(channel_id, 0, None, rubric.id, metadata)

        recipient_id = initiator_id if initiator_id is not None else None
        if recipient_id is None:
            return

        state_display = {
            "calm": "штиль",
            "storm": "шторм",
            "strong_storm": "сильный шторм",
        }.get(storm_state, storm_state)
        wave_text = f"{wave_height:.2f}" if wave_height is not None else "н/д"
        wind_parts: list[str] = []
        if wind_kmh is not None:
            wind_parts.append(f"{wind_kmh:.0f} км/ч")
        if wind_ms is not None:
            wind_parts.append(f"{wind_ms:.1f} м/с")
        wind_class_display = wind_class or "n/a"
        if wind_parts:
            primary = wind_parts[0]
            extra = wind_parts[1:] + [wind_class_display]
            wind_line = f"• Ветер: {primary} ({', '.join(extra)})"
        else:
            wind_line = "• Ветер: н/д"

        heading = "🧪 Тестовая публикация" if test else "⚠️ Публикация"
        summary_lines = [
            f"{heading} «Море / Закат на море» пропущена.",
            "Причина: не нашлось подходящих фотографий моря.",
            f"• Волна: {wave_text} м ({state_display}).",
            wind_line + "." if not wind_line.endswith(".") else wind_line,
            f"• Облачность: {clouds_label}.",
        ]
        message_text = "\n".join(summary_lines)
        response = await self.api_request(
            "sendMessage",
            {"chat_id": recipient_id, "text": message_text},
        )
        if response.get("ok"):
            logging.info("SEA_RUBRIC skip_notify sent to %s", recipient_id)
        else:
            logging.warning(
                "SEA_RUBRIC skip_notify failed recipient=%s response=%s",
                recipient_id,
                response,
            )

    def _build_final_sea_caption(self, caption: str, hashtags: list[str]) -> tuple[str, list[str]]:
        """Assemble final caption from parsed JSON only (caption + hashtags)."""
        # Deduplicate and normalize hashtags
        cleaned_hashtags = self._deduplicate_hashtags(hashtags)

        # Sanitize caption for any prompt leaks (defense-in-depth)
        sanitized_caption = sanitize_prompt_leaks(caption.strip())

        return sanitized_caption, cleaned_hashtags

    async def _generate_sea_caption(
        self,
        *,
        storm_state: str,
        storm_persisting: bool,
        wave_height_m: float | None,
        wave_score: float,
        wind_class: str | None,
        wind_ms: float | None,
        wind_kmh: float | None,
        clouds_label: str,
        sunset_selected: bool,
        want_sunset: bool,
        place_hashtag: str | None,
        fact_sentence: str | None,
        now_local_iso: str | None = None,
        day_part: str | None = None,
        tz_name: str | None = None,
        job: Job | None = None,
    ) -> tuple[str, list[str]]:
        default_hashtags = self._default_hashtags("sea")

        # LEADS list for soft fact introductions
        LEADS = [
            "А вы знали",
            "Знаете ли вы",
            "Интересный факт:",
            "Это интересно:",
            "Кстати:",
            "К слову о Балтике,",
            "Теперь вы будете знать:",
            "Небольшой факт:",
            "Поделюсь фактом:",
            "К слову",
        ]

        def fallback_caption() -> str:
            if storm_state == "strong_storm":
                opening = "Сегодня сильный шторм на море — волны гремят у самого берега."
            elif storm_state == "storm":
                if storm_persisting:
                    opening = "Продолжает штормить на море — волны всё ещё бьют о берег."
                else:
                    opening = "Сегодня шторм на море — волны упрямо разбиваются о кромку."
            else:
                opening = (
                    "Порадую закатом над морем — побережье дышит теплом."
                    if sunset_selected
                    else "Порадую вас морем — тихий берег и ровный плеск."
                )

            # Build second paragraph with LEADS + fact
            second_para = ""
            if fact_sentence:
                lead = random.choice(LEADS)
                fact_text = fact_sentence.strip()
                # Ensure proper punctuation
                if lead.endswith(":"):
                    second_para = f"{lead} {fact_text}"
                elif lead.endswith(","):
                    second_para = f"{lead} {fact_text}"
                else:
                    # Add comma after lead if it doesn't have punctuation
                    second_para = f"{lead}, {fact_text}"

            if second_para:
                return f"{opening}\n\n{second_para}"
            else:
                # No fact available - add wind/storm info
                if wind_class == "very_strong":
                    return f"{opening} Ветер срывает шапки на набережной."
                elif wind_class == "strong":
                    return f"{opening} Ветер ощутимо тянет к морю."
                elif storm_state == "calm":
                    return f"{opening} На побережье спокойно и хочется задержаться."
                else:
                    return opening

        if not self.openai or not self.openai.api_key:
            raw_fallback = fallback_caption()
            cleaned = self.strip_header(raw_fallback)
            fallback_text = cleaned.strip() if cleaned else raw_fallback.strip()
            return fallback_text, default_hashtags

        day_part_instruction = ""
        if day_part:
            day_part_instruction = (
                "Учитывай локальное время публикации: now_local_iso, day_part (morning|day|evening|night), tz_name. "
                "Пиши уместно текущему времени суток, но не упоминай время явно. "
                "Избегай приветствий и пожеланий, не соответствующих моменту (например, «пусть ваш день будет…», если уже вечер/ночь). "
                "Сохраняй естественный, дружелюбный тон. "
            )

        system_prompt = (
            "Ты редактор телеграм-канала о Балтийском море.\n\n"
            "# Форма и тон\n"
            "• Пиши ровно 2 абзаца: короткое вступление + факт.\n"
            "• Вступление — 1–2 коротких предложения, спокойно и по-дружески, без канцелярита; допустим один уместный эмодзи.\n"
            "• Факт — на тему Балтики. Начни с подводки (одну из: "
            + ", ".join(f'"{lead}"' for lead in LEADS)
            + ") и естественно перефразируй переданный факт без искажения смысла.\n"
            "• При перефразировании сохраняй числа/названия/термины корректными: не меняй значения, единицы, топонимы и термины.\n"
            "• Избегай штампов и клише: «дышит», «шепчет», «манит», «ласкает», «величавый», «бурлит», «одаряет», «нежный бриз» и т. п.\n"
            "• Не используй узкоспециальный жаргон и заумные слова (избегай научных терминов): «термоклин», «галоклин», «денситерм», «бенталь», «синоптическая обстановка», "
            "«фронтальная зона», «инфильтрация», «морфодинамика», и канцеляризмы.\n"
            f"• {day_part_instruction}"
            "• Проверь орфографию и пунктуацию по правилам русского языка. Абзацы разделяй одной пустой строкой.\n"
            "• ЖЁСТКОЕ ОГРАНИЧЕНИЕ: основной текст (без хэштегов и ссылки) — не более 700 символов; НИКОГДА не превышай 900 символов.\n"
        )
        wind_label = wind_class if wind_class in {"strong", "very_strong"} else "none"
        prompt_payload: dict[str, Any] = {
            "storm_state": storm_state,
            "storm_persisting": storm_persisting,
            "wave_height_m": round(wave_height_m, 2) if wave_height_m is not None else None,
            "wave_score": round(wave_score, 2),
            "wind_class": wind_label,
            "wind_ms": round(wind_ms, 1) if wind_ms is not None else None,
            "wind_kmh": round(wind_kmh, 1) if wind_kmh is not None else None,
            "clouds_label": clouds_label,
            "sunset_selected": sunset_selected,
            "want_sunset": want_sunset,
            "blog_tone": True,
        }
        if place_hashtag:
            prompt_payload["place_hashtag"] = place_hashtag
        if storm_state != "strong_storm" and fact_sentence:
            prompt_payload["fact_sentence"] = fact_sentence
        if now_local_iso:
            prompt_payload["now_local_iso"] = now_local_iso
        if day_part:
            prompt_payload["day_part"] = day_part
        if tz_name:
            prompt_payload["tz_name"] = tz_name
        payload_text = json.dumps(prompt_payload, ensure_ascii=False, separators=(",", ": "))
        user_prompt = (
            "Параметры сцены (JSON):\n"
            f"{payload_text}\n\n"
            "Собери подпись как два абзаца.\n\n"
            "Вступление: 1–2 коротких предложения, можно очень кратко («Порадую вас морем»). "
            "Допустим один эмодзи только здесь. Не перечисляй погодные параметры и числа.\n\n"
            "Интересный факт: начни одной подводкой из списка (или естественной эквивалентной, явно помечающей переход к факту) "
            "и передай смысл fact_sentence естественно и без искажений, одним предложением.\n\n"
            "Лимиты: вступление ≤220; общий ≤350 (или ≤400, если fact_sentence длинный). "
            "Без восклицательных знаков и риторических вопросов.\n\n"
            "Абзацы разделены одной пустой строкой.\n\n"
            "Если place_hashtag задан — включи его в массив hashtags.\n"
            "Не вставляй хэштеги в caption; верни их только в массиве hashtags.\n\n"
            'Верни JSON: {{"caption":"<два абзаца>","hashtags":[...]}}\n\n'
            "Перед возвратом проверь логичность и целостность текста: очевидный переход ко второму абзацу, "
            "связность с Балтикой/сценой, корректная пунктуация."
        )
        schema = {
            "type": "object",
            "properties": {
                "caption": {"type": "string"},
                "hashtags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                },
            },
            "required": ["caption", "hashtags"],
        }

        attempts = 3
        for attempt in range(1, attempts + 1):
            temperature = self._creative_temperature()
            attempt_start = time.perf_counter()
            try:
                logging.info(
                    "Запрос генерации текста для sea: модель=%s temperature=%.2f top_p=0.9 попытка %s/%s",
                    "gpt-4o",
                    temperature,
                    attempt,
                    attempts,
                )
                self._enforce_openai_limit(job, "gpt-4o")
                response = await self.openai.generate_json(
                    model="gpt-4o",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema=schema,
                    temperature=temperature,
                    top_p=0.9,
                )
                attempt_latency = (time.perf_counter() - attempt_start) * 1000
            except Exception:
                attempt_latency = (time.perf_counter() - attempt_start) * 1000
                logging.exception(
                    "Failed to generate sea caption (attempt %s) latency_ms=%.1f",
                    attempt,
                    attempt_latency,
                )
                response = None
            if response:
                await self._record_openai_usage("gpt-4o", response, job=job)
                # Extract finish_reason from response metadata if available
                finish_reason = "completed"
                if hasattr(response, "meta") and response.meta and isinstance(response.meta, dict):
                    finish_reason = response.meta.get("finish_reason", "completed")
                logging.info(
                    "SEA_RUBRIC openai_response attempt=%d latency_ms=%.1f finish_reason=%s source=llm",
                    attempt,
                    attempt_latency,
                    finish_reason,
                )

            # Check for valid response with parsed JSON (not raw text fallback)
            if not response or not isinstance(response.content, dict):
                logging.warning(
                    "SEA_RUBRIC json_parse_error attempt=%d (response missing or not dict)",
                    attempt,
                )
                continue

            # Check if OpenAI client returned raw text fallback ({"raw": ...})
            if "raw" in response.content and "caption" not in response.content:
                logging.warning(
                    "SEA_RUBRIC json_parse_error attempt=%d (OpenAI returned raw text, not JSON)",
                    attempt,
                )
                continue

            # Extract caption and hashtags from parsed JSON
            caption_raw = response.content.get("caption")
            raw_hashtags = response.content.get("hashtags")

            if not caption_raw or not isinstance(caption_raw, str):
                logging.warning(
                    "SEA_RUBRIC caption_missing_or_invalid attempt=%d (caption not in response or not string)",
                    attempt,
                )
                continue

            if not raw_hashtags or not isinstance(raw_hashtags, list):
                logging.warning(
                    "SEA_RUBRIC hashtags_missing_or_invalid attempt=%d (using default hashtags)",
                    attempt,
                )
                raw_hashtags = []

            # Clean and process caption
            cleaned_caption = self.strip_header(caption_raw)
            caption = cleaned_caption.strip() if cleaned_caption else caption_raw.strip()

            # Build final caption with sanitization
            caption, hashtags = self._build_final_sea_caption(caption, raw_hashtags)

            # Fatal check: caption must be non-empty after processing
            if not caption:
                logging.warning(
                    "SEA_RUBRIC empty_caption_error attempt=%d (caption empty after processing)",
                    attempt,
                )
                continue

            # Style validation checks (warn-only, do NOT block publish)
            # Check paragraph count
            paragraphs = [p.strip() for p in caption.split("\n\n") if p.strip()]
            paragraph_count = len(paragraphs)
            if paragraph_count != 2:
                logging.warning(
                    "SEA_RUBRIC caption_structure expected 2 paragraphs, got %d", paragraph_count
                )

            # Check for LEADS in second paragraph (if we have 2 paragraphs)
            if paragraph_count >= 2:
                second_para = paragraphs[1]
                has_lead = any(lead in second_para for lead in LEADS)
                if not has_lead:
                    logging.warning(
                        "SEA_RUBRIC caption_leads no standard lead found in paragraph 2"
                    )

            # Check caption length
            caption_length = len(caption)
            if caption_length > 400:
                logging.warning(
                    "SEA_RUBRIC caption_length %d exceeds soft limit 400", caption_length
                )

            # Check emoji placement (emojis should only be in paragraph 1)
            if paragraph_count >= 2:
                # Simple emoji detection pattern
                emoji_pattern = r"[\U0001F300-\U0001F9FF]|[\U0001F600-\U0001F64F]|[\U0001F680-\U0001F6FF]|[\U00002600-\U000027BF]"
                if re.search(emoji_pattern, paragraphs[1]):
                    logging.warning(
                        "SEA_RUBRIC caption_emoji found in paragraph 2 (expected only in para 1)"
                    )
            if self._is_duplicate_rubric_copy("sea", "caption", caption, hashtags):
                logging.info(
                    "Получен повторяющийся текст для рубрики sea, пробуем снова (%s/%s)",
                    attempt,
                    attempts,
                )
                continue
            logging.info("SEA_RUBRIC caption_accepted attempt=%d source=llm", attempt)
            return caption, hashtags

        # All attempts failed, using fallback
        logging.warning(
            "SEA_RUBRIC openai_fallback reason=caption_generation_failed source=fallback attempts=%d",
            attempts,
        )
        raw_fallback = fallback_caption()
        cleaned = self.strip_header(raw_fallback)
        fallback_text = cleaned.strip() if cleaned else raw_fallback.strip()
        # Apply sanitization to fallback as well (defense-in-depth)
        fallback_text = sanitize_prompt_leaks(fallback_text)
        return fallback_text, default_hashtags

    async def _generate_sea_caption_with_timeout(
        self,
        *,
        storm_state: str,
        storm_persisting: bool,
        wave_height_m: float | None,
        wave_score: float,
        wind_class: str | None,
        wind_ms: float | None,
        wind_kmh: float | None,
        clouds_label: str,
        sunset_selected: bool,
        want_sunset: bool,
        place_hashtag: str | None,
        fact_sentence: str | None,
        now_local_iso: str | None = None,
        day_part: str | None = None,
        tz_name: str | None = None,
        job: Job | None = None,
    ) -> tuple[str, list[str], dict[str, Any]]:
        openai_metadata: dict[str, Any] = {
            "openai_calls_per_publish": 0,
            "duration_ms": 0,
            "tokens": 0,
            "retries": 0,
            "timeout_hit": 0,
            "fallback": 0,
        }

        OPENAI_DEADLINE = 90.0
        PER_ATTEMPT_TIMEOUT = 60.0
        MAX_RETRIES = 1
        BACKOFF_DELAYS = [1.5, 2.0]

        global_start = time.perf_counter()

        for retry_idx in range(MAX_RETRIES + 1):
            elapsed_global = time.perf_counter() - global_start
            if elapsed_global >= OPENAI_DEADLINE:
                openai_metadata["timeout_hit"] = 1
                openai_metadata["fallback"] = 1
                logging.warning(
                    "SEA_RUBRIC OPENAI_CALL timeout=global_deadline elapsed_ms=%.1f",
                    elapsed_global * 1000,
                )
                break

            remaining_time = min(PER_ATTEMPT_TIMEOUT, OPENAI_DEADLINE - elapsed_global)
            attempt_start = time.perf_counter()

            try:
                caption_task = asyncio.create_task(
                    self._generate_sea_caption(
                        storm_state=storm_state,
                        storm_persisting=storm_persisting,
                        wave_height_m=wave_height_m,
                        wave_score=wave_score,
                        wind_class=wind_class,
                        wind_ms=wind_ms,
                        wind_kmh=wind_kmh,
                        clouds_label=clouds_label,
                        sunset_selected=sunset_selected,
                        want_sunset=want_sunset,
                        place_hashtag=place_hashtag,
                        fact_sentence=fact_sentence,
                        now_local_iso=now_local_iso,
                        day_part=day_part,
                        tz_name=tz_name,
                        job=job,
                    )
                )
                caption, hashtags = await asyncio.wait_for(caption_task, timeout=remaining_time)

                attempt_duration = (time.perf_counter() - attempt_start) * 1000
                openai_metadata["openai_calls_per_publish"] += 1
                openai_metadata["duration_ms"] = round(attempt_duration, 2)
                openai_metadata["retries"] = retry_idx

                logging.info(
                    "SEA_RUBRIC OPENAI_CALL success attempt=%d duration_ms=%.1f retries=%d",
                    openai_metadata["openai_calls_per_publish"],
                    openai_metadata["duration_ms"],
                    openai_metadata["retries"],
                )

                return caption, hashtags, openai_metadata

            except TimeoutError:
                attempt_duration = (time.perf_counter() - attempt_start) * 1000
                openai_metadata["openai_calls_per_publish"] += 1
                openai_metadata["timeout_hit"] = 1

                logging.warning(
                    "SEA_RUBRIC OPENAI_CALL timeout attempt=%d duration_ms=%.1f timeout_sec=%.1f",
                    openai_metadata["openai_calls_per_publish"],
                    attempt_duration,
                    remaining_time,
                )

                if retry_idx < MAX_RETRIES:
                    openai_metadata["retries"] = retry_idx + 1
                    backoff_delay = BACKOFF_DELAYS[min(retry_idx, len(BACKOFF_DELAYS) - 1)]
                    logging.info("SEA_RUBRIC OPENAI_CALL retry backoff_sec=%.1f", backoff_delay)
                    await asyncio.sleep(backoff_delay)
                else:
                    openai_metadata["fallback"] = 1
                    break

            except Exception as exc:
                attempt_duration = (time.perf_counter() - attempt_start) * 1000
                openai_metadata["openai_calls_per_publish"] += 1

                logging.exception(
                    "SEA_RUBRIC OPENAI_CALL error attempt=%d duration_ms=%.1f",
                    openai_metadata["openai_calls_per_publish"],
                    attempt_duration,
                )

                if retry_idx < MAX_RETRIES:
                    openai_metadata["retries"] = retry_idx + 1
                    backoff_delay = BACKOFF_DELAYS[min(retry_idx, len(BACKOFF_DELAYS) - 1)]
                    logging.info("SEA_RUBRIC OPENAI_CALL retry backoff_sec=%.1f", backoff_delay)
                    await asyncio.sleep(backoff_delay)
                else:
                    openai_metadata["fallback"] = 1
                    break

        openai_metadata["fallback"] = 1
        total_duration = (time.perf_counter() - global_start) * 1000
        openai_metadata["duration_ms"] = round(total_duration, 2)

        logging.info(
            "SEA_RUBRIC OPENAI_FALLBACK total_duration_ms=%.1f calls=%d retries=%d timeout_hit=%d source=fallback",
            openai_metadata["duration_ms"],
            openai_metadata["openai_calls_per_publish"],
            openai_metadata["retries"],
            openai_metadata["timeout_hit"],
        )

        # LEADS list for fallback
        LEADS_FALLBACK = [
            "А вы знали",
            "Знаете ли вы",
            "Интересный факт:",
            "Это интересно:",
            "Кстати:",
            "К слову о Балтике,",
            "Теперь вы будете знать:",
            "Небольшой факт:",
            "Поделюсь фактом:",
            "К слову",
        ]

        fallback_opening = ""
        if storm_state == "strong_storm":
            fallback_opening = "Сегодня сильный шторм на море — волны гремят у самого берега."
        elif storm_state == "storm":
            if storm_persisting:
                fallback_opening = "Продолжает штормить на море — волны всё ещё бьют о берег."
            else:
                fallback_opening = "Сегодня шторм на море — волны упрямо разбиваются о кромку."
        else:
            fallback_opening = (
                "Порадую закатом над морем — побережье дышит теплом."
                if sunset_selected
                else "Порадую вас морем — тихий берег и ровный плеск."
            )

        # Build second paragraph with LEADS + fact
        second_para_fallback = ""
        if fact_sentence:
            lead_fallback = random.choice(LEADS_FALLBACK)
            fact_text_fallback = fact_sentence.strip()
            # Ensure proper punctuation
            if lead_fallback.endswith(":"):
                second_para_fallback = f"{lead_fallback} {fact_text_fallback}"
            elif lead_fallback.endswith(","):
                second_para_fallback = f"{lead_fallback} {fact_text_fallback}"
            else:
                # Add comma after lead if it doesn't have punctuation
                second_para_fallback = f"{lead_fallback}, {fact_text_fallback}"

        if second_para_fallback:
            fallback_caption = f"{fallback_opening}\n\n{second_para_fallback}"
        else:
            # No fact available - add wind/storm info
            if wind_class == "very_strong":
                fallback_caption = f"{fallback_opening} Ветер срывает шапки на набережной."
            elif wind_class == "strong":
                fallback_caption = f"{fallback_opening} Ветер ощутимо тянет к морю."
            elif storm_state == "calm":
                fallback_caption = (
                    f"{fallback_opening} На побережье спокойно и хочется задержаться."
                )
            else:
                fallback_caption = fallback_opening
        default_hashtags = self._default_hashtags("sea")

        # Apply sanitization to fallback (defense-in-depth)
        fallback_caption = sanitize_prompt_leaks(fallback_caption)

        return fallback_caption, default_hashtags, openai_metadata

    async def _generate_guess_arch_copy(
        self,
        rubric: Rubric,
        asset_count: int,
        weather_text: str | None,
        *,
        job: Job | None = None,
    ) -> tuple[str, list[str]]:
        if not self.openai or not self.openai.api_key:
            fallback_caption = (
                "Делимся подборкой калининградской архитектуры — угадайте номера на фото и"
                " делитесь ответами в комментариях!"
            )
            if weather_text:
                fallback_caption += f" {weather_text}"
            return fallback_caption, self._default_hashtags("guess_arch")
        prompt = (
            'Подготовь подпись на русском языке для конкурса "Угадай архитектуру". '
            f"В альбоме {asset_count} пронумерованных фотографий. "
            "Попроси подписчиков написать свои варианты в комментариях."
        )
        if weather_text:
            prompt += f" Добавь аккуратную фразу с погодой: {weather_text}."
        schema = {
            "type": "object",
            "properties": {
                "caption": {"type": "string"},
                "hashtags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "required": ["caption", "hashtags"],
        }
        attempts = 3
        for attempt in range(1, attempts + 1):
            temperature = self._creative_temperature()
            try:
                logging.info(
                    "Запрос генерации текста для guess_arch: модель=%s temperature=%.2f top_p=0.9 попытка %s/%s",
                    "gpt-4o",
                    temperature,
                    attempt,
                    attempts,
                )
                self._enforce_openai_limit(job, "gpt-4o")
                response = await self.openai.generate_json(
                    model="gpt-4o",
                    system_prompt=(
                        "Ты редактор телеграм-канала о погоде и городе. "
                        "Пиши интересно, но коротко."
                    ),
                    user_prompt=prompt,
                    schema=schema,
                    temperature=temperature,
                    top_p=0.9,
                )
            except Exception:
                logging.exception("Failed to generate guess_arch caption (attempt %s)", attempt)
                response = None
            if response:
                await self._record_openai_usage("gpt-4o", response, job=job)
            if not response or not isinstance(response.content, dict):
                continue
            caption = str(response.content.get("caption") or "").strip()
            raw_hashtags = response.content.get("hashtags") or []
            hashtags = self._deduplicate_hashtags(raw_hashtags)
            if not caption or not hashtags:
                continue
            if self._is_duplicate_rubric_copy("guess_arch", "caption", caption, hashtags):
                logging.info(
                    "Получен повторяющийся текст для рубрики guess_arch, пробуем снова (%s/%s)",
                    attempt,
                    attempts,
                )
                continue
            return caption, hashtags
        fallback_caption = (
            "Делимся подборкой калининградской архитектуры — угадайте номера на фото и"
            " делитесь ответами в комментариях!"
        )
        if weather_text:
            fallback_caption += f" {weather_text}"
        return fallback_caption, self._default_hashtags("guess_arch")

    async def _generate_sea_copy(
        self,
        *,
        storm_state: str,
        sunset_selected: bool,
        wind_class: str | None,
        place_hashtag: str | None,
        job: Job | None = None,
    ) -> tuple[str, list[str]]:
        default_hashtags = self._default_hashtags("sea")
        if not self.openai or not self.openai.api_key:
            if storm_state in ("storm", "strong_storm"):
                fallback_caption = "Сегодня шторм на #море. Берегите себя!"
            elif sunset_selected:
                fallback_caption = "Порадую закатом над #море."
            else:
                fallback_caption = "Порадую вас #море."
            cleaned_fallback = self.strip_header(fallback_caption)
            fallback_caption = cleaned_fallback.strip() if cleaned_fallback else fallback_caption
            return fallback_caption, default_hashtags
        system_prompt = (
            "Ты редактор телеграм-канала о погоде и море. "
            "Пиши коротко, тепло и образно. "
            "Береги длину: не больше 1000 символов. "
            "Если шторм — сообщи об этом в первой фразе. "
            "Если штиля — начни с доброжелательной вариации «Порадую вас…» или «Порадую закатом…» (если фото закатное). "
            "Если передан сильный ветер — упомяни кратко и подбери уместные образные выражения (например, «сбивающий с ног»), но формулировки — на твой вкус. "
            "Всегда используй русский язык."
        )
        user_prompt_parts = [
            'Сгенерируй JSON:\n{ "caption": string, "hashtags": string[] }\n',
            f"storm_state: {storm_state}",
            f"sunset_selected: {sunset_selected}",
            f"wind_strength: {wind_class if wind_class else 'null'}",
            f"place_hashtag: {place_hashtag if place_hashtag else 'null'}",
            "\nТребования:",
        ]
        if storm_state != "calm":
            user_prompt_parts.append(
                '- Если storm_state != "calm", первая фраза — про шторм на #море.'
            )
        else:
            if sunset_selected:
                user_prompt_parts.append(
                    '- Если storm_state == "calm" и sunset_selected — вариация «Порадую закатом над #море…»'
                )
            else:
                user_prompt_parts.append(
                    '- Если storm_state == "calm" и не sunset_selected — вариация «Порадую вас #море…»'
                )
        user_prompt_parts.append(
            '- В массиве hashtags обязательно должны быть "#море" и "#БалтийскоеМоре".'
        )
        if place_hashtag:
            user_prompt_parts.append(
                f"- Если place_hashtag задан — добавь его в массив hashtags: {place_hashtag}"
            )
        user_prompt_parts.append("- Итоговая длина < 1000 символов.")
        user_prompt = "\n".join(user_prompt_parts)
        schema = {
            "type": "object",
            "properties": {
                "caption": {"type": "string"},
                "hashtags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "required": ["caption", "hashtags"],
        }
        attempts = 3
        for attempt in range(1, attempts + 1):
            temperature = self._creative_temperature()
            try:
                logging.info(
                    "Запрос генерации текста для sea: модель=%s temperature=%.2f top_p=0.9 попытка %s/%s",
                    "gpt-4o",
                    temperature,
                    attempt,
                    attempts,
                )
                self._enforce_openai_limit(job, "gpt-4o")
                response = await self.openai.generate_json(
                    model="gpt-4o",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema=schema,
                    temperature=temperature,
                    top_p=0.9,
                )
            except Exception:
                logging.exception("Failed to generate sea caption (attempt %s)", attempt)
                response = None
            if response:
                await self._record_openai_usage("gpt-4o", response, job=job)
            if not response or not isinstance(response.content, dict):
                continue
            caption_raw = str(response.content.get("caption") or "")
            cleaned_caption = self.strip_header(caption_raw)
            caption = cleaned_caption.strip() if cleaned_caption else caption_raw.strip()
            raw_hashtags = response.content.get("hashtags") or []
            hashtags = self._deduplicate_hashtags(raw_hashtags)
            if not caption or not hashtags:
                continue
            if self._is_duplicate_rubric_copy("sea", "caption", caption, hashtags):
                logging.info(
                    "Получен повторяющийся текст для рубрики sea, пробуем снова (%s/%s)",
                    attempt,
                    attempts,
                )
                continue
            return caption, hashtags
        if storm_state in ("storm", "strong_storm"):
            fallback_caption = "Сегодня шторм на #море. Берегите себя!"
        elif sunset_selected:
            fallback_caption = "Порадую закатом над #море."
        else:
            fallback_caption = "Порадую вас #море."
        cleaned_fallback = self.strip_header(fallback_caption)
        fallback_caption = cleaned_fallback.strip() if cleaned_fallback else fallback_caption
        return fallback_caption, default_hashtags

    def _prepare_hashtags(self, tags: Iterable[str]) -> list[str]:
        prepared: list[str] = []
        for tag in tags:
            text = str(tag or "").strip()
            if not text:
                continue
            if not text.startswith("#"):
                text = "#" + text.lstrip("#")
            prepared.append(text)
        return prepared

    HASHTAG_EXCLUDE: ClassVar[dict[str, set[str]]] = {
        "sea": {"котопогода"},
    }

    def _hashtag_exclusions(self, code: str) -> set[str]:
        raw = self.HASHTAG_EXCLUDE.get(code)
        if not raw:
            return set()
        normalized: set[str] = set()
        for item in raw:
            text = str(item or "").strip().lstrip("#").lower()
            if text:
                normalized.add(text)
        return normalized

    def _default_hashtags(self, code: str) -> list[str]:
        mapping = {
            "flowers": ["#котопогода", "#цветы"],
            "guess_arch": ["#угадайархитектуру", "#калининград", "#котопогода"],
            "sea": ["#котопогода", "#море", "#БалтийскоеМоре"],
        }
        return mapping.get(code, ["#котопогода"])

    def _safe_int(self, value: Any, default: int = 0) -> int:
        """Safely convert any value to int with fallback."""
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return default
            try:
                return int(text)
            except ValueError:
                try:
                    return int(float(text))
                except ValueError:
                    logging.warning("Cannot convert '%s' to int, using default %s", value, default)
                    return default
        logging.warning(
            "Unexpected type %s for int conversion, using default %s", type(value).__name__, default
        )
        return default

    def _safe_float(self, value: Any, default: float | None = None) -> float | None:
        """Safely convert any value to float with fallback."""
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return default
            try:
                return float(text)
            except ValueError:
                logging.warning("Cannot convert '%s' to float, using default %s", value, default)
                return default
        logging.warning(
            "Unexpected type %s for float conversion, using default %s",
            type(value).__name__,
            default,
        )
        return default

    def _database_has_table_or_view(self, name: str) -> bool:
        query = "SELECT 1 FROM sqlite_master WHERE (type='table' OR type='view') AND name=? LIMIT 1"
        row = self.db.execute(query, (name,)).fetchone()
        return row is not None

    def _parse_datetime_iso(self, value: str | None) -> datetime:
        """Parse ISO8601 datetime string, return datetime.min on failure."""
        if not value:
            return datetime.min
        if not isinstance(value, str):
            logging.warning("Expected string for datetime parsing, got %s", type(value).__name__)
            return datetime.min
        try:
            return datetime.fromisoformat(str(value))
        except (ValueError, AttributeError) as e:
            logging.debug("Failed to parse datetime '%s': %s", value, e)
            return datetime.min

    def _interpolate_wave_to_score(self, wave: float) -> int:
        """Convert wave height (m) to desired sea wave score 0-10."""
        if wave <= 0.0:
            return 0
        if wave <= 0.5:
            return int(round(wave * 4))
        if wave <= 1.0:
            return int(round(2 + (wave - 0.5) * 4))
        if wave <= 1.5:
            return int(round(4 + (wave - 1.0) * 6))
        if wave <= 2.0:
            return int(round(7 + (wave - 1.5) * 4))
        return 10

    def _extract_asset_sky(self, vision_results: dict[str, Any] | None) -> str | None:
        """Extract asset sky (sunny/cloudy) from vision results tags."""
        if not vision_results or not isinstance(vision_results, dict):
            return None

        sunny_syn = {
            "sunny",
            "clear",
            "sun",
            "sunlight",
            "bright",
            "blue_sky",
            "ясно",
            "солнечно",
            "голубое_небо",
            "blue sky",
        }
        cloudy_syn = {"cloudy", "overcast", "rain", "storm clouds", "пасмурно", "облачно", "дождь"}

        tags = vision_results.get("tags")
        if not isinstance(tags, list):
            return None

        tags_lower = {str(t).lower() for t in tags}
        has_cloudy = bool(tags_lower.intersection(cloudy_syn))
        has_sunny = bool(tags_lower.intersection(sunny_syn))

        if has_cloudy:
            return "cloudy"
        if has_sunny:
            return "sunny"

        weather_image = vision_results.get("weather_image")
        if isinstance(weather_image, str):
            weather_lower = weather_image.lower()
            if weather_lower in {"rain", "overcast", "snow"}:
                return "cloudy"
            if weather_lower in {"sunny", "clear"}:
                return "sunny"

        return None

    def _get_asset_wave_score_with_fallback(self, asset: Asset) -> int:
        """Get asset wave score from vision_results or fallback to tag-based heuristic."""
        if asset.vision_results and isinstance(asset.vision_results, dict):
            sea_wave_score = asset.vision_results.get("sea_wave_score")
            if isinstance(sea_wave_score, dict):
                value = sea_wave_score.get("value")
                if value is not None:
                    # Handle both int and string values
                    int_value = self._safe_int(value, default=-1)
                    if int_value >= 0:
                        return max(0, min(10, int_value))

            tags = asset.vision_results.get("tags")
            if isinstance(tags, list):
                tags_lower = {str(t).lower() for t in tags}
                if tags_lower.intersection({"whitecaps", "surf", "шторм", "gale", "spray", "foam"}):
                    return random.randint(8, 9)
                if "storm" in tags_lower:
                    return random.randint(6, 7)
                if "waves" in tags_lower:
                    return random.randint(4, 5)

        return random.randint(1, 2)

    def _pick_sea_asset(
        self,
        assets: list[Asset],
        *,
        desired_wave_score: float | int | str | None,
        desired_sky: str | None,
    ) -> Asset | None:
        if not assets:
            return None

        def norm(value: str | None) -> str:
            return (value or "").strip().lower()

        def _safe_float(value: Any, default: float | None = None) -> float | None:
            if value is None:
                return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _safe_int(value: Any, default: int = 0) -> int:
            if value is None:
                return default
            try:
                return int(value)
            except (TypeError, ValueError):
                try:
                    return int(float(value))
                except (TypeError, ValueError):
                    return default

        def _lru_ts(asset: Asset) -> float:
            payload = getattr(asset, "payload", None)
            if isinstance(payload, dict):
                last_used = payload.get("last_used_at")
                if isinstance(last_used, datetime):
                    return last_used.timestamp()
                if isinstance(last_used, (int, float)):
                    return float(last_used)
                if isinstance(last_used, str):
                    ts_text = last_used.strip()
                    if not ts_text:
                        return 0.0
                    try:
                        return datetime.fromisoformat(ts_text).timestamp()
                    except ValueError:
                        numeric_ts = _safe_float(ts_text, default=None)
                        if numeric_ts is not None:
                            return float(numeric_ts)
            return 0.0

        def extract_tags(asset: Asset) -> set[str]:
            tags_set: set[str] = set()
            vision_results = getattr(asset, "vision_results", None)
            if vision_results and isinstance(vision_results, dict):
                raw_tags = vision_results.get("tags")
                if isinstance(raw_tags, list):
                    tags_set.update(norm(str(tag)) for tag in raw_tags)
            return tags_set

        def extract_categories(asset: Asset) -> set[str]:
            categories_set: set[str] = set()
            categories = getattr(asset, "categories", None)
            if categories:
                categories_set.update(norm(str(category)) for category in categories)
            return categories_set

        def _extract_wave_score(asset: Asset, tags: set[str]) -> float:
            vision_results = getattr(asset, "vision_results", None) or {}
            if not isinstance(vision_results, dict):
                vision_results = {}
            sea_wave_score = vision_results.get("sea_wave_score")
            value: float | None = None
            if isinstance(sea_wave_score, dict):
                value = _safe_float(sea_wave_score.get("value"))
            elif sea_wave_score is not None:
                value = _safe_float(sea_wave_score)
            if value is not None:
                return value
            if {"whitecaps", "surf", "шторм", "gale", "spray", "foam"} & tags:
                return 9.0
            if "storm" in tags:
                return 7.0
            if "waves" in tags:
                return 5.0
            return 1.0

        desired_wave_score_value = _safe_float(desired_wave_score, default=None)
        desired_sky_normalized = norm(desired_sky)
        if not desired_sky_normalized:
            desired_sky_normalized = None
        debug_enabled = DEBUG_SEA_PICK
        debug_records: list[dict[str, Any]] = []
        storm_synonyms = {
            "storm",
            "шторм",
            "waves",
            "буря",
            "surf",
            "whitecaps",
            "gale",
            "spray",
            "foam",
        }

        def build_sort_key(asset: Asset) -> tuple[int, int, float, float, float, int]:
            asset_id_int = _safe_int(getattr(asset, "id", None))
            record: dict[str, Any] = {
                "id": asset_id_int,
                "has_sunset_tag": 0,
                "has_sunset_category": 0,
                "asset_score": None,
                "desired_wave_score": desired_wave_score_value,
                "wave_penalty": 0.0,
                "lru_ts": 0.0,
                "thematic_bias": 0.0,
            }
            sunset_tag_priority = 0
            sunset_category_priority = 0
            try:
                tags = extract_tags(asset)
                categories = extract_categories(asset)
                sunset_tag = "sunset" in tags
                sunset_category = bool({"закат", "sunset"} & categories)
                prefer_sunsets = (
                    desired_wave_score_value is not None and desired_wave_score_value <= 2.0
                )
                if prefer_sunsets and sunset_tag:
                    sunset_tag_priority = 1
                if prefer_sunsets and sunset_category:
                    sunset_category_priority = 1
                record["has_sunset_tag"] = sunset_tag_priority
                record["has_sunset_category"] = sunset_category_priority

                asset_score = _extract_wave_score(asset, tags)
                record["asset_score"] = asset_score
                if desired_wave_score_value is None or asset_score is None:
                    wave_penalty = 0.0
                else:
                    wave_penalty = -abs(asset_score - desired_wave_score_value)
                record["wave_penalty"] = wave_penalty

                thematic_bias = 0.0
                asset_sky = self._extract_asset_sky(getattr(asset, "vision_results", None))
                if desired_sky_normalized and asset_sky:
                    if desired_sky_normalized == asset_sky:
                        thematic_bias += 1.0
                    else:
                        thematic_bias -= 0.5
                if desired_wave_score_value is not None:
                    if desired_wave_score_value <= 2.0:
                        if sunset_tag:
                            thematic_bias += 2.0
                        elif sunset_category:
                            thematic_bias += 1.0
                    if desired_wave_score_value >= 6.0 and storm_synonyms.intersection(tags):
                        storm_bonus = max(0.0, 0.5 + wave_penalty)
                        if storm_bonus:
                            thematic_bias += storm_bonus
                record["thematic_bias"] = thematic_bias

                lru_component = _lru_ts(asset)
                record["lru_ts"] = lru_component
                sort_key = (
                    sunset_tag_priority,
                    sunset_category_priority,
                    float(thematic_bias),
                    float(wave_penalty),
                    -float(lru_component),
                    -int(asset_id_int),
                )
            except Exception:
                logging.exception(
                    "Failed to build sea picker key for asset %s",
                    getattr(asset, "id", None),
                )
                sort_key = (
                    sunset_tag_priority,
                    sunset_category_priority,
                    0.0,
                    0.0,
                    0.0,
                    -int(asset_id_int),
                )
            if debug_enabled and len(debug_records) < 5:
                record["key"] = sort_key
                record["types"] = tuple(type(component).__name__ for component in sort_key)
                debug_records.append(record)
            return sort_key

        scored_assets = [(asset, build_sort_key(asset)) for asset in assets]
        if not scored_assets:
            return None

        winner_asset, winner_key = max(scored_assets, key=lambda item: item[1])
        if debug_enabled and debug_records:
            logging.info(
                (
                    "sea_picker_debug desired_wave_score=%s desired_sky=%s "
                    "winner_id=%s winner_key=%s candidates=%s"
                ),
                desired_wave_score_value,
                desired_sky_normalized,
                _safe_int(getattr(winner_asset, "id", None)),
                winner_key,
                debug_records,
            )
        return winner_asset

    async def _cleanup_assets(
        self, assets: Iterable[Asset], *, extra_paths: Iterable[str] | None = None
    ) -> None:
        asset_list = list(assets)
        ids = [asset.id for asset in asset_list]
        for asset in asset_list:
            await self._delete_asset_message(asset)
            self._remove_file(asset.local_path)
        if extra_paths:
            for path in extra_paths:
                self._remove_file(path)
        self.data.delete_assets(ids)

    async def _finalize_published_asset(
        self, asset: dict[str, Any], full_asset: Asset | None
    ) -> None:
        asset_id = asset.get("id")
        if asset_id is None:
            return
        if full_asset and full_asset.id == asset_id:
            try:
                await self._cleanup_assets([full_asset])
                return
            except Exception:
                logging.exception("Failed to cleanup asset %s after publishing", asset_id)
        await self._fallback_cleanup_asset(asset)

    async def _fallback_cleanup_asset(self, asset: dict[str, Any]) -> None:
        asset_id = asset.get("id")
        channel_id = asset.get("channel_id")
        message_id = asset.get("message_id")
        if channel_id and message_id:
            try:
                response = await self.api_request(
                    "deleteMessage",
                    {"chat_id": channel_id, "message_id": message_id},
                )
                if not response or not response.get("ok"):
                    logging.warning(
                        "Fallback message deletion failed for asset %s: %s",
                        asset_id,
                        response,
                    )
            except Exception:
                logging.exception(
                    "Failed to delete message %s from chat %s during cleanup",
                    message_id,
                    channel_id,
                )
        if asset_id is not None:
            try:
                self.data.delete_assets([asset_id])
            except Exception:
                logging.exception(
                    "Failed to delete asset %s from database during cleanup",
                    asset_id,
                )

    async def _delete_asset_message(self, asset: Asset) -> None:
        chat_id = asset.tg_chat_id or asset.channel_id
        message_id = asset.message_id
        if not chat_id or not message_id:
            return
        try:
            response = await self.api_request(
                "deleteMessage",
                {"chat_id": chat_id, "message_id": message_id},
            )
        except Exception:
            logging.exception(
                "Failed to delete asset %s message %s from chat %s",
                asset.id,
                message_id,
                chat_id,
            )
            return
        if not response or not response.get("ok"):
            logging.warning(
                "Failed to delete message %s from chat %s for asset %s: %s",
                message_id,
                chat_id,
                asset.id,
                response,
            )

    def _remove_file(self, path: str | None) -> None:
        if not path:
            return
        try:
            os.remove(path)
        except FileNotFoundError:
            return
        except Exception:
            logging.exception("Failed to remove file %s", path)

    def _compatible_photo_weather_classes(self, actual_class: str | None) -> set[str] | None:
        if actual_class is None:
            return None
        normalized_actual = Bot._normalize_weather_enum(actual_class)
        if normalized_actual:
            actual_class = normalized_actual
        mapping = {
            "sunny": {"sunny", "partly_cloudy"},
            "partly_cloudy": {"sunny", "partly_cloudy", "overcast"},
            "overcast": {"partly_cloudy", "overcast", "fog"},
            "rain": {"rain"},
            "snow": {"snow"},
            "fog": {"fog", "overcast"},
            "night": {"night", "partly_cloudy"},
        }
        allowed = mapping.get(actual_class)
        if not allowed and normalized_actual and normalized_actual in mapping:
            allowed = mapping[normalized_actual]
        if not allowed:
            allowed = {actual_class}
        return set(allowed)

    def _normalize_weather_label(self, label: str | None) -> str | None:
        normalized = Bot._normalize_weather_enum(label)
        if normalized:
            return normalized
        if not label:
            return None
        text = str(label).strip().lower()
        if not text:
            return None
        keyword_map: list[tuple[tuple[str, ...], str]] = [
            (("ноч", "night"), "night"),
            (("гроза", "storm", "thunder", "молн"), "rain"),
            (("снег", "snow", "снеж", "метел", "blizzard"), "snow"),
            (("дожд", "rain", "ливн", "drizzle", "wet"), "rain"),
            (("туман", "fog", "mist", "дымк", "haze", "смог"), "fog"),
            (("пасмур", "overcast", "сплошн", "тучн", "серое небо"), "overcast"),
            (("облач", "cloud"), "partly_cloudy"),
            (("солне", "ясн", "clear", "sunny", "bright sun"), "sunny"),
        ]
        for needles, token in keyword_map:
            for needle in needles:
                if needle in text:
                    normalized_token = Bot._normalize_weather_enum(token)
                    if normalized_token:
                        return normalized_token
                    return token
        fallback = text.split()[0]
        return Bot._normalize_weather_enum(fallback)

    def _classify_weather_code(self, code: int | None) -> str | None:
        if code is None:
            return None
        mapping = {
            0: "sunny",
            1: "partly_cloudy",
            2: "partly_cloudy",
            3: "overcast",
            45: "fog",
            48: "fog",
            51: "rain",
            53: "rain",
            55: "rain",
            56: "rain",
            57: "rain",
            61: "rain",
            63: "rain",
            65: "rain",
            66: "rain",
            67: "rain",
            71: "snow",
            73: "snow",
            75: "snow",
            77: "snow",
            80: "rain",
            81: "rain",
            82: "rain",
            85: "snow",
            86: "snow",
            95: "rain",
            96: "rain",
            99: "rain",
        }
        return mapping.get(code)

    def _get_city_weather_info(self, city_name: str) -> tuple[str | None, str | None]:
        snapshot = self._fetch_city_weather_snapshot(city_name)
        if not snapshot:
            return None, None
        weather_class = snapshot.get("weather_class")
        if weather_class is None:
            weather_class = self._classify_weather_code(snapshot.get("weather_code"))
        emoji = (
            weather_emoji(snapshot["weather_code"], snapshot.get("is_day"))
            if snapshot.get("weather_code") is not None
            else ""
        )
        parts: list[str] = []
        temp = snapshot.get("temperature")
        wind = snapshot.get("wind_speed")
        if temp is not None:
            parts.append(f"{int(round(temp))}°C")
        if wind is not None:
            parts.append(f"ветер {int(round(wind))} м/с")
        summary = ", ".join(parts)
        city_display = snapshot.get("name") or city_name
        if emoji and summary:
            return f"{emoji} {city_display}: {summary}", weather_class
        if emoji:
            return f"{emoji} {city_display}", weather_class
        if summary:
            return f"{city_display}: {summary}", weather_class
        return city_display, weather_class

    def _get_city_weather_summary(self, city_name: str) -> str | None:
        summary, _ = self._get_city_weather_info(city_name)
        return summary

    def _classify_wind(self, wind_mps: float | None) -> str | None:
        if wind_mps is None or wind_mps < 10:
            return None
        if wind_mps < 15:
            return "strong"
        return "very_strong"

    def _get_sea_weather_sky(self, sea_id: int) -> str | None:
        """Get desired_sky (sunny/cloudy) from nearest city weather to sea_id."""
        sea_row = self.data.conn.execute(
            "SELECT lat, lon FROM seas WHERE id = ?", (sea_id,)
        ).fetchone()
        if not sea_row:
            return None
        sea_lat = sea_row["lat"]
        sea_lon = sea_row["lon"]
        city_rows = self.data.conn.execute("SELECT id, name, lat, lon FROM cities").fetchall()
        if not city_rows:
            return None
        min_distance = None
        closest_city_id = None
        for row in city_rows:
            city_lat = row["lat"]
            city_lon = row["lon"]
            dlat = math.radians(city_lat - sea_lat)
            dlon = math.radians(city_lon - sea_lon)
            a = (
                math.sin(dlat / 2) ** 2
                + math.cos(math.radians(sea_lat))
                * math.cos(math.radians(city_lat))
                * math.sin(dlon / 2) ** 2
            )
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance = 6371 * c
            if min_distance is None or distance < min_distance:
                min_distance = distance
                closest_city_id = row["id"]
        if closest_city_id is None:
            return None
        now_utc = datetime.utcnow()
        weather_row = self.data.conn.execute(
            """
            SELECT weather_code
            FROM weather_cache_hour
            WHERE city_id = ? AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (closest_city_id, now_utc.isoformat()),
        ).fetchone()
        if not weather_row or weather_row["weather_code"] is None:
            return None
        weather_code = int(weather_row["weather_code"])
        weather_class = self._classify_weather_code(weather_code)
        if not weather_class:
            return None
        if weather_class in {"rain", "overcast", "snow", "fog"}:
            return "cloudy"
        if weather_class in {"sunny", "partly_cloudy"}:
            return "sunny"
        return None

    def _get_sea_wind(self, sea_id: int) -> tuple[float | None, str | None]:
        sea_row = self.data.conn.execute(
            "SELECT lat, lon FROM seas WHERE id = ?", (sea_id,)
        ).fetchone()
        if not sea_row:
            return None, None
        sea_lat = sea_row["lat"]
        sea_lon = sea_row["lon"]
        city_rows = self.data.conn.execute("SELECT id, name, lat, lon FROM cities").fetchall()
        if not city_rows:
            return None, None
        min_distance = None
        closest_city_id = None
        for row in city_rows:
            city_lat = row["lat"]
            city_lon = row["lon"]
            dlat = math.radians(city_lat - sea_lat)
            dlon = math.radians(city_lon - sea_lon)
            a = (
                math.sin(dlat / 2) ** 2
                + math.cos(math.radians(sea_lat))
                * math.cos(math.radians(city_lat))
                * math.sin(dlon / 2) ** 2
            )
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance = 6371 * c
            if min_distance is None or distance < min_distance:
                min_distance = distance
                closest_city_id = row["id"]
        if closest_city_id is None:
            return None, None
        now_utc = datetime.utcnow()
        wind_row = self.data.conn.execute(
            """
            SELECT wind_speed
            FROM weather_cache_hour
            WHERE city_id = ? AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (closest_city_id, now_utc.isoformat()),
        ).fetchone()
        if not wind_row or wind_row["wind_speed"] is None:
            return None, None
        wind_speed = float(wind_row["wind_speed"])
        wind_class = self._classify_wind(wind_speed)
        return wind_speed, wind_class

    def _format_temperature_value(self, value: float | None) -> str | None:
        if value is None:
            return None
        return f"{int(round(value))}°C"

    def _format_wind_value(self, value: float | None) -> str | None:
        if value is None:
            return None
        return f"{max(0, int(round(value)))} м/с"

    def _compose_weather_metrics_preview_line(
        self, metrics: Mapping[str, Any] | None, *, label: str
    ) -> str:
        if not isinstance(metrics, Mapping):
            return f"{label}: данные отсутствуют"

        def _coerce_numeric(value: Any) -> float | None:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                stripped = value.strip().replace(",", ".")
                if not stripped:
                    return None
                try:
                    return float(stripped)
                except ValueError:
                    return None
            return None

        parts: list[str] = []
        condition = str(metrics.get("condition") or "").strip()
        if condition:
            parts.append(condition)
        temp_value = _coerce_numeric(metrics.get("temperature"))
        if temp_value is not None:
            formatted = self._format_temperature_value(temp_value)
            if formatted:
                parts.append(f"температура {formatted}")
        wind_value = _coerce_numeric(metrics.get("wind_speed"))
        if wind_value is not None:
            formatted_wind = self._format_wind_value(wind_value)
            if formatted_wind:
                parts.append(f"ветер {formatted_wind}")
        if not parts:
            return f"{label}: данные отсутствуют"
        return f"{label}: " + ", ".join(parts)

    @staticmethod
    def _extract_weather_location_label(
        payload: Mapping[str, Any] | None,
    ) -> str | None:
        if not isinstance(payload, Mapping):
            return None
        cities_value = payload.get("cities")
        if isinstance(cities_value, (list, tuple, set)):
            locations = [
                str(city or "").strip() for city in cities_value if str(city or "").strip()
            ]
            if locations:
                return ", ".join(locations)
        else:
            location = str(cities_value or "").strip()
            if location:
                return location
        city_snapshot = payload.get("city")
        if isinstance(city_snapshot, Mapping):
            name = str(city_snapshot.get("name") or "").strip()
            if name:
                return name
        city_name = payload.get("city_name")
        if isinstance(city_name, str) and city_name.strip():
            return city_name.strip()
        return None

    def _positive_temperature_trend(self, current: float | None, previous: float | None) -> str:
        if current is None:
            return ""
        current_text = self._format_temperature_value(current) or ""
        if previous is None:
            return f"температура около {current_text}".strip()
        diff = current - previous
        diff_value = int(round(abs(diff)))
        if diff >= 1:
            change = f"на {diff_value}°" if diff_value else ""
            return f"стало теплее {change}".strip()
        if diff <= -1:
            return f"свежесть бодрит — около {current_text}".strip()
        return f"температура держится около {current_text}".strip()

    def _positive_wind_trend(self, current: float | None, previous: float | None) -> str:
        if current is None:
            return ""
        current_text = self._format_wind_value(current) or ""
        if previous is None:
            return f"ветер около {current_text}".strip()
        diff = current - previous
        if diff <= -0.5:
            return f"ветер стал спокойнее — около {current_text}".strip()
        if diff >= 0.5:
            return f"ветер бодрит до {current_text}, но остаётся дружелюбным".strip()
        return f"ветер мягкий — около {current_text}".strip()

    def _fetch_city_weather_snapshot(self, city_name: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT id, name FROM cities WHERE lower(name)=lower(?)",
            (city_name,),
        ).fetchone()
        if not row:
            return None
        weather_row = self.db.execute(
            """
            SELECT temperature, weather_code, wind_speed, is_day
            FROM weather_cache_hour
            WHERE city_id=?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (row["id"],),
        ).fetchone()
        if not weather_row:
            return None
        now = datetime.utcnow()
        today = now.date().isoformat()
        yesterday = (now - timedelta(days=1)).date().isoformat()
        hourly_rows = self.db.execute(
            """
            SELECT timestamp, temperature, weather_code, wind_speed, is_day
            FROM weather_cache_hour
            WHERE city_id=?
            ORDER BY timestamp DESC
            LIMIT 60
            """,
            (row["id"],),
        ).fetchall()
        day_row = self.db.execute(
            """
            SELECT temperature, wind_speed, weather_code
            FROM weather_cache_day
            WHERE city_id=? AND day=?
            LIMIT 1
            """,
            (row["id"], today),
        ).fetchone()
        previous_row = self.db.execute(
            """
            SELECT temperature, wind_speed, weather_code
            FROM weather_cache_day
            WHERE city_id=? AND day=?
            LIMIT 1
            """,
            (row["id"], yesterday),
        ).fetchone()

        def _row_to_day_payload(day_row: sqlite3.Row | None) -> dict[str, Any] | None:
            if not day_row:
                return None
            payload: dict[str, Any] = {}
            for key in day_row.keys():
                value = day_row[key]
                if isinstance(value, str):
                    stripped = value.strip()
                    if stripped.startswith("{") or stripped.startswith("["):
                        try:
                            payload[key] = json.loads(stripped)
                            continue
                        except json.JSONDecodeError:
                            pass
                payload[key] = value
            return payload

        def _coerce_numeric(value: Any) -> float | None:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    return None
                try:
                    return float(stripped)
                except ValueError:
                    if stripped.startswith("{") or stripped.startswith("["):
                        try:
                            parsed = json.loads(stripped)
                        except json.JSONDecodeError:
                            return None
                        return _coerce_numeric(parsed)
                    return None
            if isinstance(value, dict):
                for key in ("mean", "avg", "average", "value"):
                    if key in value:
                        coerced = _coerce_numeric(value[key])
                        if coerced is not None:
                            return coerced
                for nested_value in value.values():
                    coerced = _coerce_numeric(nested_value)
                    if coerced is not None:
                        return coerced
            if isinstance(value, (list, tuple)):
                for item in value:
                    coerced = _coerce_numeric(item)
                    if coerced is not None:
                        return coerced
            return None

        def _flatten_day_values(data: dict[str, Any] | None) -> dict[str, float]:
            flattened: dict[str, float] = {}
            if not data:
                return flattened

            def _collect(prefix: str, raw_value: Any) -> None:
                numeric = _coerce_numeric(raw_value)
                if numeric is not None:
                    keys = [prefix]
                    if "." in prefix:
                        keys.append(prefix.rsplit(".", 1)[-1])
                    for key in keys:
                        if key and key not in flattened:
                            flattened[key] = numeric
                    return
                if isinstance(raw_value, dict):
                    for sub_key, sub_value in raw_value.items():
                        next_prefix = f"{prefix}.{sub_key}" if prefix else sub_key
                        _collect(next_prefix, sub_value)
                elif isinstance(raw_value, (list, tuple)):
                    for index, item in enumerate(raw_value):
                        next_prefix = f"{prefix}[{index}]" if prefix else str(index)
                        _collect(next_prefix, item)

            for key, value in data.items():
                if key in {"day", "city_id"}:
                    continue
                _collect(key, value)
            return flattened

        def _select_metric(
            values: dict[str, float],
            preferred: Sequence[str],
            fallbacks: Sequence[str],
        ) -> float | None:
            for key in preferred:
                if key in values:
                    return values[key]
            for key in fallbacks:
                if key in values:
                    return values[key]
            return None

        today_day = _row_to_day_payload(day_row)
        yesterday_day = _row_to_day_payload(previous_row)
        today_values = _flatten_day_values(today_day)
        yesterday_values = _flatten_day_values(yesterday_day)
        temperature_keys = (
            "temperature_2m_mean",
            "temperature_mean",
            "temperature_avg",
            "temperature_average",
            "temp_mean",
            "temp_avg",
            "temperature.mean",
            "temperature.avg",
            "temperature.average",
        )
        temperature_fallbacks = (
            "temperature",
            "temperature_day",
            "temperature.value",
            "temperature.max",
            "temperature_2m_max",
            "temperature_max",
        )
        wind_keys = (
            "wind_speed_10m_max",
            "wind_speed_max",
            "wind_speed.max",
            "wind_speed.gusts_max",
        )
        wind_fallbacks = (
            "wind_speed_10m_mean",
            "wind_speed_mean",
            "wind_speed_avg",
            "wind_speed_average",
            "wind_speed.mean",
            "wind_speed.avg",
            "wind_speed.average",
            "wind_speed",
            "wind",
        )
        day_temperature = _select_metric(today_values, temperature_keys, temperature_fallbacks)
        day_wind = _select_metric(today_values, wind_keys, wind_fallbacks)
        previous_temperature = _select_metric(
            yesterday_values, temperature_keys, temperature_fallbacks
        )
        previous_wind = _select_metric(yesterday_values, wind_keys, wind_fallbacks)
        weather_code = weather_row["weather_code"]
        weather_class = self._classify_weather_code(weather_code)
        weather_condition = (
            WEATHER_TAG_TRANSLATIONS.get(weather_class, weather_class) if weather_class else None
        )
        snapshot: dict[str, Any] = {
            "id": row["id"],
            "name": row["name"],
            "temperature": weather_row["temperature"],
            "wind_speed": weather_row["wind_speed"],
            "weather_code": weather_code,
            "weather_class": weather_class,
            "weather_condition": weather_condition,
            "is_day": weather_row["is_day"],
            "day": today_day,
            "yesterday_day": yesterday_day,
            "day_temperature": day_temperature,
            "day_wind": day_wind,
            "previous_temperature": previous_temperature,
            "previous_wind": previous_wind,
        }
        dayparts: dict[str, dict[str, Any]] = {"today": {}, "yesterday": {}}

        def _parse_timestamp(value: Any) -> datetime | None:
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    return None
                if stripped.endswith("Z"):
                    stripped = stripped[:-1] + "+00:00"
                try:
                    return datetime.fromisoformat(stripped)
                except ValueError:
                    return None
            return None

        def _segment_for_hour(hour: int) -> str | None:
            if 6 <= hour < 12:
                return "morning"
            if 12 <= hour < 18:
                return "day"
            if 18 <= hour < 24:
                return "evening"
            return None

        def _normalize_code(value: Any) -> int | None:
            if isinstance(value, (int, float)):
                return int(round(value))
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    return None
                try:
                    return int(float(stripped.replace(",", ".")))
                except ValueError:
                    return None
            return None

        def _avg(values: list[float]) -> float | None:
            return sum(values) / len(values) if values else None

        hourly_entries: list[tuple[datetime, sqlite3.Row]] = []
        for hourly_row in hourly_rows or []:
            ts = _parse_timestamp(hourly_row["timestamp"])
            if not ts:
                continue
            hourly_entries.append((ts, hourly_row))
        hourly_entries.sort(key=lambda item: item[0])
        today_date = now.date()
        yesterday_date = (now - timedelta(days=1)).date()
        for ts, hourly_row in hourly_entries:
            segment = _segment_for_hour(ts.hour)
            if not segment:
                continue
            day_key: str | None
            if ts.date() == today_date:
                day_key = "today"
            elif ts.date() == yesterday_date:
                day_key = "yesterday"
            else:
                continue
            samples = (
                dayparts.setdefault(day_key, {}).setdefault(segment, {}).setdefault("samples", [])
            )
            samples.append(
                {
                    "timestamp": ts.isoformat(),
                    "temperature": hourly_row["temperature"],
                    "weather_code": hourly_row["weather_code"],
                    "wind_speed": hourly_row["wind_speed"],
                    "is_day": hourly_row["is_day"],
                }
            )

        def _build_fallback_segment(
            day_key: str,
            segment_key: str,
            segment_payload: Any,
        ) -> dict[str, Any] | None:
            samples: list[Any] = []
            if isinstance(segment_payload, Mapping):
                raw_samples = segment_payload.get("samples")
                if isinstance(raw_samples, list):
                    samples = list(raw_samples)
                existing: dict[str, Any] = {}
                for key in ("condition", "code", "class", "temperature", "wind_speed"):
                    if key not in segment_payload:
                        continue
                    value = segment_payload.get(key)
                    if value is None:
                        continue
                    if isinstance(value, str):
                        trimmed = value.strip()
                        if not trimmed:
                            continue
                        existing[key] = trimmed
                    else:
                        existing[key] = value
                if existing:
                    existing["samples"] = samples
                    return existing
            elif isinstance(segment_payload, str):
                trimmed = segment_payload.strip()
                if trimmed:
                    return {"condition": trimmed, "samples": samples}
                return None

            day_payload = today_day if day_key == "today" else yesterday_day
            flattened = today_values if day_key == "today" else yesterday_values
            fallback_temperature = day_temperature if day_key == "today" else previous_temperature
            fallback_wind = day_wind if day_key == "today" else previous_wind
            fallback_code = (
                _normalize_code(weather_row["weather_code"])
                if day_key == "today"
                else (
                    _normalize_code(previous_row["weather_code"])
                    if previous_row and "weather_code" in previous_row.keys()
                    else None
                )
            )

            condition: str | None = None
            code: int | None = fallback_code
            temperature = fallback_temperature
            wind_speed = fallback_wind

            def _ingest(value: Any) -> None:
                nonlocal condition, code, temperature, wind_speed
                if value is None:
                    return
                if isinstance(value, Mapping):
                    if condition is None:
                        for key in ("condition", "summary", "text", "description", "value"):
                            raw = value.get(key)
                            if isinstance(raw, str) and raw.strip():
                                condition = raw.strip()
                                break
                    if code is None:
                        for key in ("code", "weather_code", "wmo_code", "wmo"):
                            candidate = _normalize_code(value.get(key))
                            if candidate is not None:
                                code = candidate
                                break
                    if temperature is None:
                        for key in (
                            "temperature",
                            "temp",
                            "temperature_mean",
                            "temperature_avg",
                            "mean",
                            "avg",
                            "average",
                        ):
                            candidate_temp = _coerce_numeric(value.get(key))
                            if candidate_temp is not None:
                                temperature = candidate_temp
                                break
                    if wind_speed is None:
                        for key in ("wind_speed", "wind", "wind_gust", "gust"):
                            candidate_wind = _coerce_numeric(value.get(key))
                            if candidate_wind is not None:
                                wind_speed = candidate_wind
                                break
                    for nested in value.values():
                        if isinstance(nested, (Mapping, list, tuple, set)):
                            _ingest(nested)
                elif isinstance(value, str):
                    trimmed_value = value.strip()
                    if trimmed_value and condition is None:
                        condition = trimmed_value
                elif isinstance(value, (list, tuple, set)):
                    for item in value:
                        _ingest(item)
                else:
                    if code is None:
                        normalized = _normalize_code(value)
                        if normalized is not None:
                            code = normalized

            if isinstance(day_payload, Mapping):
                _ingest(day_payload.get(segment_key))
                for container_key in (
                    "segments",
                    "dayparts",
                    "parts",
                    "periods",
                    "details",
                    "summary",
                ):
                    container = day_payload.get(container_key)
                    if isinstance(container, Mapping):
                        _ingest(container.get(segment_key))

            if isinstance(flattened, Mapping):
                temperature_keys = [
                    f"{segment_key}_temperature",
                    f"temperature_{segment_key}",
                    f"{segment_key}.temperature",
                    f"temperature.{segment_key}",
                    f"{segment_key}_temp",
                    f"temp_{segment_key}",
                ]
                for key in temperature_keys:
                    if temperature is not None:
                        break
                    if key in flattened:
                        temperature = flattened[key]
                wind_keys = [
                    f"{segment_key}_wind_speed",
                    f"wind_speed_{segment_key}",
                    f"{segment_key}.wind_speed",
                    f"wind_speed.{segment_key}",
                    f"{segment_key}_wind",
                    f"wind_{segment_key}",
                ]
                for key in wind_keys:
                    if wind_speed is not None:
                        break
                    if key in flattened:
                        wind_speed = flattened[key]
                code_keys = [
                    f"{segment_key}_code",
                    f"{segment_key}_weather_code",
                    f"weather_code_{segment_key}",
                    f"{segment_key}.code",
                    f"code.{segment_key}",
                ]
                for key in code_keys:
                    if code is not None:
                        break
                    if key in flattened:
                        candidate_code = _normalize_code(flattened[key])
                        if candidate_code is not None:
                            code = candidate_code

            payload: dict[str, Any] = {}
            if condition is not None and condition:
                payload["condition"] = condition
            if temperature is not None:
                payload["temperature"] = temperature
            if wind_speed is not None:
                payload["wind_speed"] = wind_speed
            if code is not None:
                payload["code"] = code
                weather_class = self._classify_weather_code(code)
                if weather_class:
                    payload["class"] = weather_class
                    if not payload.get("condition"):
                        payload["condition"] = WEATHER_TAG_TRANSLATIONS.get(
                            weather_class, weather_class
                        )
            if not payload:
                return None
            payload["samples"] = samples
            payload["source"] = "daily"
            return payload

        for day_key in ("today", "yesterday"):
            segments = dayparts.setdefault(day_key, {})
            for segment_key in ("morning", "day", "evening"):
                segment_payload = segments.get(segment_key)
                if segment_payload is None:
                    segment_payload = {}
                    segments[segment_key] = segment_payload
                samples = (
                    segment_payload.get("samples") if isinstance(segment_payload, Mapping) else None
                )
                if isinstance(samples, list) and samples:
                    codes = [
                        _normalize_code(sample.get("weather_code"))
                        for sample in samples
                        if isinstance(sample, Mapping)
                    ]
                    codes = [code for code in codes if code is not None]
                    representative_code: int | None = None
                    if codes:
                        representative_code = codes[len(codes) // 2]
                    weather_class_segment = (
                        self._classify_weather_code(representative_code)
                        if representative_code is not None
                        else None
                    )
                    condition_segment = (
                        WEATHER_TAG_TRANSLATIONS.get(weather_class_segment, weather_class_segment)
                        if weather_class_segment
                        else None
                    )
                    temperatures = [
                        float(sample["temperature"])
                        for sample in samples
                        if isinstance(sample, Mapping) and sample.get("temperature") is not None
                    ]
                    winds = [
                        float(sample["wind_speed"])
                        for sample in samples
                        if isinstance(sample, Mapping) and sample.get("wind_speed") is not None
                    ]
                    segment_payload.update(
                        {
                            "code": representative_code,
                            "class": weather_class_segment,
                            "condition": condition_segment,
                            "temperature": _avg(temperatures),
                            "wind_speed": _avg(winds),
                        }
                    )
                    if segment_payload.get("condition") is None and not temperatures and not winds:
                        segments.pop(segment_key, None)
                else:
                    fallback_segment = _build_fallback_segment(
                        day_key, segment_key, segment_payload
                    )
                    if fallback_segment is None:
                        segments.pop(segment_key, None)
                    else:
                        segments[segment_key] = fallback_segment
            if not segments:
                dayparts.pop(day_key, None)

        if dayparts:
            snapshot["dayparts"] = dayparts
        temp_detail = self._format_temperature_value(snapshot["temperature"])
        wind_detail = self._format_wind_value(snapshot["wind_speed"])
        details: list[str] = []
        if temp_detail:
            details.append(temp_detail)
        if wind_detail:
            details.append(f"ветер {wind_detail}")
        snapshot["detail"] = ", ".join(details)
        trend_temperature_value = (
            snapshot.get("day_temperature")
            if snapshot.get("day_temperature") is not None
            else snapshot.get("temperature")
        )
        trend_temperature_previous_value = snapshot.get("previous_temperature")
        trend_wind_value = (
            snapshot.get("day_wind")
            if snapshot.get("day_wind") is not None
            else snapshot.get("wind_speed")
        )
        trend_wind_previous_value = snapshot.get("previous_wind")

        snapshot["trend_temperature_value"] = trend_temperature_value
        snapshot["trend_temperature_previous_value"] = trend_temperature_previous_value
        snapshot["trend_wind_value"] = trend_wind_value
        snapshot["trend_wind_previous_value"] = trend_wind_previous_value

        snapshot["trend_temperature"] = self._positive_temperature_trend(
            trend_temperature_value,
            trend_temperature_previous_value,
        )
        snapshot["trend_wind"] = self._positive_wind_trend(
            trend_wind_value,
            trend_wind_previous_value,
        )
        trends = [
            piece for piece in (snapshot["trend_temperature"], snapshot["trend_wind"]) if piece
        ]
        snapshot["trend_summary"] = " и ".join(trends)
        if snapshot["trend_summary"]:
            snapshot["positive_intro"] = "Утро радует"
        else:
            snapshot["positive_intro"] = "Утро остаётся уютным"
        return snapshot

    def _fetch_coast_weather_snapshot(self) -> dict[str, Any] | None:
        row = self.db.execute(
            """
            SELECT s.name, sc.current, sc.wave
            FROM sea_cache AS sc
            JOIN seas AS s ON sc.sea_id = s.id
            ORDER BY sc.updated DESC
            LIMIT 1
            """
        ).fetchone()
        if not row:
            return None
        temp = row["current"]
        wave = row["wave"]
        parts: list[str] = []
        if temp is not None:
            parts.append(f"вода {int(round(temp))}°C")
        if wave is not None:
            parts.append(f"волна {round(wave, 1)} м")
        joined = ", ".join(parts)
        if joined:
            detail = f"{row['name']}: {joined}"
        else:
            detail = row["name"]
        description: str
        if wave is None or wave < 0.5:
            description = "море спокойное"
        elif wave < 1.2:
            description = "волна мягко бодрит"
        else:
            description = "волна играет и заряжает"
        return {
            "name": row["name"],
            "temperature": temp,
            "wave": wave,
            "detail": detail,
            "description": description,
        }

    def _format_flowers_cities_for_weather(self, cities: Sequence[str], fallback: str) -> str:
        base_city = (fallback or "").strip() or "Калининград"
        base_key = base_city.casefold()
        shooting_locations: list[str] = []
        seen: set[str] = set()
        for raw_city in cities:
            city = str(raw_city or "").strip()
            if not city:
                continue
            key = city.casefold()
            if key == base_key:
                continue
            if key in seen:
                continue
            seen.add(key)
            shooting_locations.append(city)

        if not shooting_locations:
            return base_city

        if len(shooting_locations) == 1:
            locations = shooting_locations[0]
        elif len(shooting_locations) == 2:
            locations = f"{shooting_locations[0]} и {shooting_locations[1]}"
        else:
            locations = ", ".join(shooting_locations[:-1]) + f" и {shooting_locations[-1]}"

        return f"{base_city} (съёмка: {locations})"

    def _compose_flowers_weather_block(
        self,
        cities: Sequence[str],
        rubric: Rubric | None = None,
    ) -> dict[str, Any] | None:
        config = self._normalize_rubric_config(rubric.config if rubric else {})
        configured_city = str(config.get("weather_city") or "").strip()
        if not configured_city:
            city_id = config.get("weather_city_id")
            if isinstance(city_id, int):
                resolved = self._get_city_name(city_id)
                if resolved:
                    configured_city = resolved
        requested_city = configured_city or "Kaliningrad"
        city_snapshot = self._fetch_city_weather_snapshot(requested_city)
        if not city_snapshot and requested_city != "Kaliningrad":
            city_snapshot = self._fetch_city_weather_snapshot("Kaliningrad")
        coast_snapshot = self._fetch_coast_weather_snapshot()
        if not city_snapshot and not coast_snapshot:
            return None
        fallback_city_name = (city_snapshot or {}).get("name") or requested_city or "Калининград"
        city_list = self._format_flowers_cities_for_weather(cities, fallback_city_name)
        positive_intro = (city_snapshot or {}).get("positive_intro") or "Утро радует"
        trend_summary = (city_snapshot or {}).get("trend_summary") or "погода поддерживает уют"
        if trend_summary:
            headline = f"{positive_intro}: {trend_summary}"
        else:
            headline = positive_intro
        detail_parts: list[str] = []
        if city_snapshot and city_snapshot.get("detail"):
            detail_parts.append(f"в городе {city_snapshot['detail']}")
        if coast_snapshot:
            if coast_snapshot.get("detail"):
                detail_parts.append(coast_snapshot["detail"])
            elif coast_snapshot.get("description"):
                detail_parts.append(coast_snapshot["description"])
        details = "; ".join(part for part in detail_parts if part)
        if details:
            line = f"{headline} {city_list}: {details}."
        else:
            line = f"{headline} {city_list}."

        def _condition_from_code(value: Any) -> str | None:
            code: int | None
            if isinstance(value, Mapping):
                for key in ("code", "weather_code", "value"):
                    if key in value:
                        return _condition_from_code(value[key])
                return None
            if isinstance(value, (int, float)):
                code = int(round(value))
            elif isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    return None
                try:
                    code = int(float(stripped.replace(",", ".")))
                except ValueError:
                    return None
            else:
                return None
            weather_class = self._classify_weather_code(code)
            if not weather_class:
                return None
            return WEATHER_TAG_TRANSLATIONS.get(weather_class, weather_class)

        def _extract_daypart_conditions(dayparts_payload: Any, day_key: str) -> dict[str, str]:
            if not isinstance(dayparts_payload, Mapping):
                return {}
            day_bucket = dayparts_payload.get(day_key)
            if not isinstance(day_bucket, Mapping):
                return {}
            parts: dict[str, str] = {}
            for segment_key in ("morning", "day", "evening"):
                segment_value = day_bucket.get(segment_key)
                condition: str | None = None
                if isinstance(segment_value, Mapping):
                    text = segment_value.get("condition")
                    if isinstance(text, str) and text.strip():
                        condition = text.strip()
                    if not condition:
                        weather_class = segment_value.get("class")
                        if isinstance(weather_class, str) and weather_class.strip():
                            normalized = weather_class.strip()
                            condition = WEATHER_TAG_TRANSLATIONS.get(normalized, normalized)
                    if not condition:
                        condition = _condition_from_code(segment_value.get("code"))
                    if not condition:
                        samples = segment_value.get("samples")
                        if isinstance(samples, Sequence):
                            for sample in samples:
                                condition = _condition_from_code(sample)
                                if condition:
                                    break
                elif isinstance(segment_value, str):
                    trimmed = segment_value.strip()
                    if trimmed:
                        condition = trimmed
                else:
                    condition = _condition_from_code(segment_value)
                if condition:
                    parts[segment_key] = condition
            return parts

        today_metrics: dict[str, Any] = {}
        yesterday_metrics: dict[str, Any] = {}
        if city_snapshot:
            today_condition = city_snapshot.get("weather_condition")
            dayparts_payload = city_snapshot.get("dayparts")
            today_parts = _extract_daypart_conditions(dayparts_payload, "today")
            yesterday_parts = _extract_daypart_conditions(dayparts_payload, "yesterday")
            if not today_condition:
                for segment_key in ("day", "morning", "evening"):
                    candidate = today_parts.get(segment_key)
                    if candidate:
                        today_condition = candidate
                        break
            yesterday_condition = None
            for segment_key in ("day", "morning", "evening"):
                candidate = yesterday_parts.get(segment_key)
                if candidate:
                    yesterday_condition = candidate
                    break
            today_metrics = {
                "temperature": city_snapshot.get("trend_temperature_value"),
                "wind_speed": city_snapshot.get("trend_wind_value"),
                "condition": today_condition,
            }
            if today_parts:
                today_metrics["parts"] = today_parts
            yesterday_metrics = {
                "temperature": city_snapshot.get("trend_temperature_previous_value"),
                "wind_speed": city_snapshot.get("trend_wind_previous_value"),
                "condition": yesterday_condition,
            }
            if yesterday_parts:
                yesterday_metrics["parts"] = yesterday_parts

        def _drop_empty(metrics: dict[str, Any]) -> dict[str, Any]:
            return {key: value for key, value in metrics.items() if value is not None}

        return {
            "city": city_snapshot,
            "sea": coast_snapshot,
            "positive_intro": positive_intro,
            "trend_summary": trend_summary,
            "details": details,
            "line": line,
            "cities": city_list,
            "today": _drop_empty(today_metrics),
            "yesterday": _drop_empty(yesterday_metrics),
        }

    def _overlay_number(
        self,
        asset: Asset,
        number: int,
        config: dict[str, Any],
        *,
        source_path: str | None = None,
    ) -> str | None:
        local_path = source_path or asset.local_path
        if not local_path or not os.path.exists(local_path):
            logging.warning("Asset %s missing source file for overlay", asset.id)
            return None
        overlays_dir = (
            config.get("overlays_dir")
            or config.get("overlay_dir")
            or config.get("overlays")
            or "overlays"
        )
        overlay_path = Path(overlays_dir)
        if not overlay_path.is_absolute():
            overlay_path = Path(__file__).resolve().parent / overlay_path
        candidate = overlay_path / f"{number}.png"
        with Image.open(local_path) as src:
            base = src.convert("RGBA")
            overlay_img = self._load_overlay_image(candidate, number, base.size)
            min_side = min(base.width, base.height)
            padding = 12 if min_side < 480 else 24
            offset = (padding, padding)
            base.paste(overlay_img, offset, overlay_img)
            output_path = self.asset_storage / f"{asset.id}_numbered_{number}.png"
            base.convert("RGB").save(output_path)
        return str(output_path)

    def _load_overlay_image(
        self, path: Path, number: int, base_size: tuple[int, int]
    ) -> Image.Image:
        overlay_img: Image.Image
        if path.exists():
            with Image.open(path) as overlay_src:
                overlay_img = overlay_src.convert("RGBA").copy()
        else:
            overlay_img = self._create_number_overlay(number, min(base_size))
        min_side = min(base_size)
        min_target = max(1, math.ceil(min_side * 0.10))
        max_target = max(min_target, math.floor(min_side * 0.16))
        target = round(min_side * 0.12)
        max_side = max(min_target, min(target, max_target))
        overlay_img = overlay_img.resize((max_side, max_side), Image.LANCZOS)
        return overlay_img

    def _create_number_overlay(self, number: int, base_side: int) -> Image.Image:
        size = max(96, base_side // 4)
        image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.ellipse((0, 0, size, size), fill=(0, 0, 0, 180))
        font_size = max(32, size // 2)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
        text = str(number)
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            text_width, text_height = draw.textsize(text, font=font)
        position = ((size - text_width) // 2, (size - text_height) // 2)
        draw.text(position, text, fill=(255, 255, 255, 230), font=font)
        return image

    async def schedule_loop(self) -> None:
        """Background scheduler running at configurable intervals."""

        try:
            logging.info("Scheduler loop started")
            while self.running:
                await self.process_due()
                try:
                    await self.collect_weather()
                    await self.collect_sea()

                    await self.process_weather_channels()
                    await self.process_rubric_schedule()
                except Exception:
                    logging.exception("Weather collection failed")
                await asyncio.sleep(SCHED_INTERVAL_SEC)
        except asyncio.CancelledError:
            pass


async def privet_handler(request: web.Request) -> web.Response:
    return web.Response(text="Привет")


async def health_handler(request: web.Request) -> web.Response:
    bot: Bot = request.app["bot"]
    started_at: datetime = request.app["started_at"]
    version = request.app.get("version", APP_VERSION)

    overall_start = perf_counter()
    checks: dict[str, dict[str, Any]] = {}
    status = 200
    warnings: list[str] = []

    # Database check
    t0 = perf_counter()
    try:
        bot.db.execute("SELECT 1").fetchone()
        latency_ms = (perf_counter() - t0) * 1000.0
        checks["db"] = {"ok": True, "latency_ms": latency_ms}
    except Exception as exc:  # pragma: no cover - defensive
        checks["db"] = {"ok": False, "error": str(exc)}
        status = 503

    # Queue metrics
    t0 = perf_counter()
    try:
        metrics = bot.jobs.metrics()
        latency_ms = (perf_counter() - t0) * 1000.0
        checks["queue"] = {"ok": True, **metrics, "latency_ms": latency_ms}
    except Exception as exc:
        checks["queue"] = {"ok": False, "error": str(exc)}
        status = 503

    try:
        asset_channel_id = get_asset_channel_id(bot.db)
    except Exception:
        logging.exception("Failed to fetch asset channel configuration")
        asset_channel_id = None

    try:
        recognition_channel_id = get_recognition_channel_id(bot.db)
    except Exception:
        logging.exception("Failed to fetch recognition channel configuration")
        recognition_channel_id = None

    config_missing: list[str] = []
    if recognition_channel_id is None:
        config_missing.append("recognition_channel")
    if asset_channel_id is None:
        config_missing.append("asset_channel")
    config_ok = not config_missing
    if config_missing and status == 200:
        status = 207

    # Telegram connectivity
    t0 = perf_counter()
    skipped = getattr(bot, "dry_run", False)
    if skipped:
        checks["telegram"] = {
            "ok": True,
            "method": "getMe",
            "latency_ms": 0.0,
            "skipped": True,
        }
        warnings.append("telegram check skipped (dry_run)")
        if status == 200:
            status = 207
    else:
        try:
            response = await bot.api_request("getMe")
            latency_ms = (perf_counter() - t0) * 1000.0
            ok = bool(response.get("ok"))
            telegram_check: dict[str, Any] = {
                "ok": ok,
                "method": "getMe",
                "latency_ms": latency_ms,
                "skipped": False,
            }
            if not ok:
                telegram_check["error"] = response.get(
                    "description", "telegram api returned ok=false"
                )
                status = 503
            checks["telegram"] = telegram_check
        except Exception as exc:
            latency_ms = (perf_counter() - t0) * 1000.0
            checks["telegram"] = {
                "ok": False,
                "method": "getMe",
                "latency_ms": latency_ms,
                "error": str(exc),
                "skipped": False,
            }
            status = 503

    now_utc = datetime.now(UTC)
    started = started_at
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    uptime_s = max(0.0, (now_utc - started).total_seconds())

    config_payload: dict[str, Any] = {"ok": config_ok}
    if config_missing:
        config_payload["missing"] = config_missing

    payload = {
        "ok": status in (200, 207),
        "version": version,
        "now": _isoformat_utc(now_utc),
        "uptime_s": uptime_s,
        "checks": checks,
        "config": config_payload,
        "warnings": warnings,
    }

    log_parts: list[str] = []
    db_check = checks.get("db", {})
    if db_check.get("ok"):
        log_parts.append(f"db=ok({db_check['latency_ms']:.1f}ms)")
    else:
        log_parts.append(f"db=fail({db_check.get('error', 'unknown')})")

    queue_check = checks.get("queue", {})
    if queue_check.get("ok"):
        log_parts.append(
            "queue=ok(p={pending} a={active} f={failed},{latency:.1f}ms)".format(
                pending=queue_check.get("pending", 0),
                active=queue_check.get("active", 0),
                failed=queue_check.get("failed", 0),
                latency=queue_check.get("latency_ms", 0.0),
            )
        )
    else:
        log_parts.append(f"queue=fail({queue_check.get('error', 'unknown')})")

    telegram_check = checks.get("telegram", {})
    if telegram_check.get("skipped"):
        log_parts.append("tg=skip")
    elif telegram_check.get("ok"):
        log_parts.append(f"tg=ok({telegram_check.get('latency_ms', 0.0):.1f}ms)")
    else:
        log_parts.append(f"tg=fail({telegram_check.get('error', 'unknown')})")

    if config_ok:
        log_parts.append("config=ok")
    else:
        missing_desc = ",".join(config_missing) if config_missing else "unknown"
        log_parts.append(f"config=missing({missing_desc})")

    logging.info("HEALTH %s status=%s", " ".join(log_parts), status)
    observe_health_latency(perf_counter() - overall_start)

    return web.json_response(payload, status=status)


async def ensure_webhook(bot: Bot, base_url: str) -> None:
    expected = base_url.rstrip("/") + "/webhook"
    info = await bot.api_request("getWebhookInfo")
    current = info.get("result", {}).get("url")
    if current != expected:
        logging.info("Registering webhook %s", expected)
        resp = await bot.api_request("setWebhook", {"url": expected})
        if not resp.get("ok"):
            logging.error("Failed to register webhook: %s", resp)
            raise RuntimeError(f"Webhook registration failed: {resp}")
        logging.info("Webhook registered successfully")
    else:
        logging.info("Webhook already registered at %s", current)


async def attach_device(request: web.Request) -> web.Response:
    app = request.app
    bot: Bot = app["bot"]
    user_limiter: TokenBucketLimiter = app["attach_user_rate_limiter"]

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "invalid_payload"}, status=400)

    if not isinstance(payload, dict):
        return web.json_response({"error": "invalid_payload"}, status=400)

    try:
        code = normalize_pairing_token(payload)
    except PairingTokenError as exc:
        logging.warning(
            "DEVICE attach invalid token ip=%s reason=%s",
            request.remote,
            exc.message,
        )
        return web.json_response(
            {"error": "invalid_token", "message": exc.message},
            status=400,
        )

    raw_name = payload.get("name")
    provided_name = str(raw_name).strip() if isinstance(raw_name, str) else ""

    ip = request.remote or "unknown"

    conn = bot.db
    now_iso = datetime.utcnow().isoformat()
    cur = conn.execute(
        """
        SELECT user_id
        FROM pairing_tokens
        WHERE code=? AND used_at IS NULL AND expires_at>?
        """,
        (code, now_iso),
    )
    row = cur.fetchone()
    if row:
        user_for_limit = int(row["user_id"]) if isinstance(row, sqlite3.Row) else int(row[0])
        allowance = await user_limiter.allow(f"user:{user_for_limit}")
        if not allowance.allowed:
            logging.warning("DEVICE attach user rate-limit user=%s ip=%s", user_for_limit, ip)
            request["rate_limit_log"] = {
                "result": "hit",
                "scope": "user",
                "limit": user_limiter.capacity,
                "window": user_limiter.window,
                "key": f"user:{user_for_limit}",
                "retry_after": allowance.retry_after_seconds,
            }
            record_rate_limit_drop("/v1/devices/attach", "user")
            headers: dict[str, str] | None = None
            if (
                allowance.retry_after_seconds is not None
                and allowance.retry_after_seconds >= 0
                and math.isfinite(allowance.retry_after_seconds)
            ):
                headers = {
                    "Retry-After": str(max(0, int(math.ceil(allowance.retry_after_seconds))))
                }
            return web.json_response({"error": "rate_limited"}, status=429, headers=headers)

    with conn:
        info = consume_pairing_token(conn, code=code)
        if not info:
            logging.warning("DEVICE attach invalid token ip=%s", ip)
            return web.json_response(
                {
                    "error": "invalid_token",
                    "message": "Токен недействителен или срок его действия истёк.",
                },
                status=400,
            )

        user_id, default_name = info
        effective_name = provided_name or str(default_name or "").strip() or _PAIRING_DEFAULT_NAME
        device_id = str(uuid4())
        secret = secrets.token_hex(32)
        create_device(
            conn,
            device_id=device_id,
            user_id=user_id,
            name=effective_name,
            secret=secret,
        )

    event_time = datetime.now(UTC).isoformat()
    with context(device_id=device_id, source="mobile"):
        logging.info(
            "MOBILE_ATTACH_OK",
            extra={
                "user_id": user_id,
                "device_id": device_id,
                "device_name": effective_name,
                "timestamp": event_time,
            },
        )
        logging.info(
            "DEVICE attach success user=%s device=%s name=%s ip=%s",
            user_id,
            device_id,
            effective_name,
            ip,
        )
    payload: dict[str, str] = {
        "device_id": device_id,
        "device_secret": secret,
    }
    if effective_name:
        payload["name"] = effective_name

    return web.json_response(payload)


async def handle_webhook(request: Any) -> web.Response:
    bot: Bot = request.app["bot"]
    try:
        data = await request.json()
        logging.info("Received webhook: %s", data)
    except Exception:
        logging.exception("Invalid webhook payload")
        return web.Response(text="bad request", status=400)
    try:
        await bot.handle_update(data)
    except Exception:
        logging.exception("Error handling update")
        return web.Response(text="error", status=500)
    return web.Response(text="ok")


def create_app() -> web.Application:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not found in environment variables")

    bot = Bot(token, DB_PATH)
    uploads_config = bot.uploads_config
    app = web.Application(client_max_size=uploads_config.max_upload_bytes + 1024)
    app["bot"] = bot
    app["uploads_config"] = uploads_config
    app["started_at"] = datetime.now(UTC)
    app["version"] = APP_VERSION
    app["attach_user_rate_limiter"] = TokenBucketLimiter(
        _env_int("RL_ATTACH_USER_PER_MIN", 3),
        _env_int("RL_ATTACH_USER_WINDOW_SEC", 60),
    )

    storage = create_storage_from_env(supabase=bot.supabase)
    upload_metrics = UploadMetricsRecorder(emitter=LoggingMetricsEmitter())
    bot.upload_metrics = upload_metrics
    app["upload_metrics"] = upload_metrics

    class _UploadTelegramAdapter:
        def __init__(self, bot: Bot):
            self._bot = bot

        async def send_photo(
            self,
            *,
            chat_id: int,
            photo: Path,
            caption: str | None = None,
        ) -> dict[str, Any]:
            response, _ = await self._bot._publish_as_photo(chat_id, str(photo), caption)
            return response or {}

        async def send_document(
            self,
            *,
            chat_id: int,
            document: BinaryIO | bytes,
            file_name: str,
            caption: str | None = None,
            content_type: str | None = None,
        ) -> dict[str, Any]:
            if isinstance(document, (bytes, bytearray)):
                document_stream: BinaryIO = io.BytesIO(document)
            else:
                document_stream = document
            response = await self._bot._publish_mobile_document(
                chat_id,
                document_stream,
                file_name,
                caption,
                content_type=content_type,
            )
            return response or {}

    register_upload_jobs(
        bot.jobs,
        bot.db,
        storage=storage,
        data=bot.data,
        telegram=_UploadTelegramAdapter(bot),
        openai=bot.openai,
        supabase=bot.supabase,
        metrics=upload_metrics,
        config=uploads_config,
    )
    setup_upload_routes(
        app,
        storage=storage,
        conn=bot.db,
        jobs=bot.jobs,
        config=uploads_config,
    )

    app.middlewares.append(observability_middleware)
    app.middlewares.append(create_hmac_middleware(bot.db))
    app.middlewares.append(create_rate_limit_middleware())

    app.router.add_post("/webhook", handle_webhook)
    app.router.add_get("/privet", privet_handler)
    app.router.add_get("/v1/health", health_handler)
    app.router.add_post("/v1/devices/attach", attach_device)
    app.router.add_get("/metrics", metrics_handler)

    webhook_base = os.getenv("WEBHOOK_URL")
    if not webhook_base:
        raise RuntimeError("WEBHOOK_URL not found in environment variables")

    async def start_background(app: web.Application) -> None:
        logging.info("Application startup")
        try:
            await bot.start()
            await bot.run_openai_health_check()
            await ensure_webhook(bot, webhook_base)
        except Exception:
            logging.exception("Error during startup")
            raise
        app["schedule_task"] = asyncio.create_task(bot.schedule_loop())

        async def run_startup_backfill() -> None:
            try:
                logging.info("Starting wave backfill in background")
                stats = await bot.backfill_waves(dry_run=False)
                message = (
                    f"🌊 Backfill waves finished:\n"
                    f"Updated: {stats['updated']}\n"
                    f"Skipped: {stats['skipped']}\n"
                    f"Errors: {stats['errors']}"
                )
                logging.info("Wave backfill completed: %s", stats)

                superadmin_ids = bot.get_superadmin_ids()
                for admin_id in superadmin_ids:
                    try:
                        await bot.api_request(
                            "sendMessage",
                            {"chat_id": admin_id, "text": message},
                        )
                    except Exception:
                        logging.exception("Failed to notify superadmin %s", admin_id)
            except Exception:
                logging.exception("Background wave backfill failed")

        asyncio.create_task(run_startup_backfill())

    async def cleanup_background(app: web.Application) -> None:
        await bot.close()
        app["schedule_task"].cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await app["schedule_task"]

    app.on_startup.append(start_background)
    app.on_cleanup.append(cleanup_background)

    return app


if __name__ == "__main__":

    web.run_app(create_app(), port=int(os.getenv("PORT", 8080)))
