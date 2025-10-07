from __future__ import annotations

import asyncio
import gc
import io
import contextlib
import json
import logging
import math
import mimetypes
import os
import random
import re
import sqlite3
import tempfile
from copy import deepcopy
from datetime import datetime, date, timedelta, timezone, time as dtime
from pathlib import Path

from typing import Any, BinaryIO, Iterable, Sequence, TYPE_CHECKING

from aiohttp import web, ClientSession, FormData
from PIL import Image, ImageDraw, ImageFont, ImageOps
import psutil

try:  # pragma: no cover - optional dependency
    from PIL import ImageCms  # type: ignore
except Exception:  # pragma: no cover - fallback when LittleCMS is unavailable
    ImageCms = None  # type: ignore[assignment]
import piexif

from data_access import Asset, DataAccess, Rubric
from jobs import Job, JobDelayed, JobQueue
from openai_client import OpenAIClient
from supabase_client import SupabaseClient
from weather_migration import migrate_weather_publish_channels

if TYPE_CHECKING:
    from openai_client import OpenAIResponse

logging.basicConfig(level=logging.INFO)

# Default database path points to /data which is mounted as a Fly.io volume.
# This ensures information like registered channels and scheduled posts
# persists across deployments unless DB_PATH is explicitly overridden.
def _env_flag(value: str) -> bool:
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


DB_PATH = os.getenv("DB_PATH", "/data/bot.db")
TZ_OFFSET = os.getenv("TZ_OFFSET", "+00:00")
SCHED_INTERVAL_SEC = int(os.getenv("SCHED_INTERVAL_SEC", "30"))
ASSETS_DEBUG_EXIF = _env_flag(os.getenv("ASSETS_DEBUG_EXIF", "0"))
VISION_CONCURRENCY_RAW = os.getenv("VISION_CONCURRENCY", "1")
try:
    VISION_CONCURRENCY = max(1, int(VISION_CONCURRENCY_RAW))
except ValueError:
    logging.warning(
        "Invalid VISION_CONCURRENCY=%s, defaulting to 1", VISION_CONCURRENCY_RAW
    )
    VISION_CONCURRENCY = 1
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
WMO_EMOJI = {
    0: "\u2600\ufe0f",
    1: "\U0001F324",
    2: "\u26c5",
    3: "\u2601\ufe0f",
    45: "\U0001F32B",
    48: "\U0001F32B",
    51: "\U0001F327",
    53: "\U0001F327",
    55: "\U0001F327",
    61: "\U0001F327",
    63: "\U0001F327",
    65: "\U0001F327",
    71: "\u2744\ufe0f",
    73: "\u2744\ufe0f",
    75: "\u2744\ufe0f",
    80: "\U0001F327",
    81: "\U0001F327",
    82: "\U0001F327",
    95: "\u26c8\ufe0f",
    96: "\u26c8\ufe0f",
    99: "\u26c8\ufe0f",
}

def weather_emoji(code: int, is_day: int | None) -> str:
    emoji = WMO_EMOJI.get(code, "")
    if code == 0 and is_day == 0:
        return "\U0001F319"  # crescent moon
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
                mime = Image.MIME.get(format_name.upper()) or Image.MIME.get(
                    format_name
                )
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
    "sunset": "night",
}

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
                "Кадровка/ракурс снимка. Используй один из вариантов: close_up, medium, "
                "wide."
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
            "description": (
                "Информация о чувствительном контенте: nsfw и краткая причина."
            ),
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
        "season_guess",
        "safety",
    ],
}


CHANNEL_PICKER_PAGE_SIZE = 6
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


DEFAULT_RUBRIC_PRESETS: dict[str, dict[str, Any]] = {
    "flowers": {
        "title": "Цветы",
        "config": {
            "enabled": False,
            "schedules": [],
            "assets": {"min": 1, "max": 6, "categories": ["flowers"]},
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
    applied = {
        row["id"]
        for row in conn.execute("SELECT id FROM schema_migrations")
    }
    migration_files = sorted(
        p for p in MIGRATIONS_DIR.iterdir() if p.suffix in {".sql", ".py"}
    )
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

    """CREATE TABLE IF NOT EXISTS weather_posts (
            id INTEGER PRIMARY KEY,
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
        self.data = DataAccess(self.db)
        self._rubric_category_cache: dict[str, int] = {}
        self._ensure_default_rubrics()
        if migrate_weather_publish_channels(self.db, tz_offset=TZ_OFFSET):
            self.db.commit()
        self.jobs = JobQueue(self.db, concurrency=1)
        self.jobs.register_handler("ingest", self._job_ingest)
        self.jobs.register_handler("vision", self._job_vision)
        self.jobs.register_handler("publish_rubric", self._job_publish_rubric)
        self.openai = OpenAIClient(os.getenv("4O_API_KEY"))
        self.supabase = SupabaseClient()
        self._model_limits = self._load_model_limits()
        asset_dir = os.getenv("ASSET_STORAGE_DIR")
        self.asset_storage = Path(asset_dir).expanduser() if asset_dir else Path("/tmp/bot_assets")
        self.asset_storage.mkdir(parents=True, exist_ok=True)
        self._last_geocode_at: datetime | None = None
        ttl_hours_raw = os.getenv("REVGEO_CACHE_TTL_HOURS", "24")
        try:
            ttl_hours = float(ttl_hours_raw)
        except ValueError:
            logging.warning(
                "Invalid REVGEO_CACHE_TTL_HOURS=%s, defaulting to 24h", ttl_hours_raw
            )
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
        self._shoreline_cache: dict[
            tuple[float, float], tuple[bool, datetime]
        ] = {}
        self._shoreline_fail_cache: dict[tuple[float, float], datetime] = {}
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

        self.session: ClientSession | None = None
        self.running = False
        self.manual_buttons: dict[tuple[int, int], dict[str, list[list[dict]]]] = {}
        self.rubric_pending_runs: dict[tuple[int, str], str] = {}
        self.pending_flowers_previews: dict[int, dict[str, Any]] = {}

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

    def _reencode_to_jpeg_under_10mb(
        self, local_path: str | os.PathLike[str] | Path
    ) -> Path:
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
                logging.warning(
                    "Unable to reduce %s below 10MB at lowest quality", source_path
                )
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
            if score > best_score or (score == best_score and file_size > int(best.get("file_size") or 0) if best else True):
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

    def _get_rubric_overview_target(
        self, user_id: int, code: str
    ) -> dict[str, Any] | None:
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

    def _cleanup_rubric_overviews(
        self, user_id: int, valid_codes: Iterable[str]
    ) -> None:
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

    async def _render_rubric_cards(
        self, user_id: int, rubrics: Sequence[Rubric]
    ) -> None:
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
        default_limit = parse_limit(os.getenv("OPENAI_DAILY_TOKEN_LIMIT"), "OPENAI_DAILY_TOKEN_LIMIT")
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
            logging.warning("OpenAI FAIL: unexpected response %s", response.content if response else None)

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
        total_today = self.data.get_daily_token_usage_total(
            models={model}, tz_offset=tz_offset
        )
        if total_today >= limit:
            resume_at = self._next_usage_reset(tz_offset=tz_offset)
            tzinfo = self._parse_tz_offset(tz_offset)
            resume_local = (
                resume_at.replace(tzinfo=timezone.utc).astimezone(tzinfo)
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
        response: "OpenAIResponse" | None,
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
            total_today = self.data.get_daily_token_usage_total(
                models={model}, tz_offset=tz_offset
            )
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

    async def start(self):
        self.session = ClientSession()
        self.running = True
        await self.jobs.start()

    async def close(self):
        self.running = False
        await self.jobs.stop()
        if self.session:
            await self.session.close()
        await self.supabase.aclose()

        self.db.close()

    async def handle_edited_message(self, message):
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
    ):
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
            "&forecast_days=2&timezone=auto"
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
            "&forecast_days=2&timezone=auto"
        )
        logging.info("Sea API request: %s", url)
        try:
            async with self.session.get(url) as resp:
                text = await resp.text()
        except Exception:
            logging.exception("Failed to fetch sea")
            return None

        logging.info("Sea API raw response: %s", text)
        if resp.status != 200:
            logging.error("Open-Meteo sea HTTP %s", resp.status)
            return None
        try:
            data = json.loads(text)
        except Exception:
            logging.exception("Invalid sea JSON")
            return None
        return data

    async def collect_weather(self, force: bool = False):

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

    async def collect_sea(self, force: bool = False):
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
            temps = data["hourly"].get("water_temperature") or data["hourly"].get("sea_surface_temperature")
            waves = data["hourly"].get("wave_height")
            times = data["hourly"].get("time")
            if not temps or not times:
                continue
            if waves is None:
                waves = [0.0 for _ in temps]
            current = temps[0]
            current_wave = data.get("current", {}).get("wave_height")
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
                if (
                    None not in (morn, day_temp, eve, night, mwave, dwave, ewave, nwave)
                ):
                    break

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
                    current_wave,
                    mwave,
                    dwave,
                    ewave,
                    nwave,
                ),
            )
            self.db.commit()
            updated.add(s["id"])
        if updated:
            await self.update_weather_posts()
            await self.check_amber()

    async def check_amber(self):
        state = self.db.execute('SELECT sea_id, storm_start, active FROM amber_state LIMIT 1').fetchone()
        if not state:
            return
        sea_id = state['sea_id']
        row = self._get_sea_cache(sea_id)
        if not row or row['wave'] is None:
            return
        wave_raw = row['wave']
        try:
            wave = float(wave_raw)
        except (TypeError, ValueError):
            logging.warning('Unable to parse wave height %r for sea %s', wave_raw, sea_id)
            return
        now = datetime.utcnow()
        if wave >= 1.5:
            if not state['active']:
                self.db.execute(
                    'UPDATE amber_state SET storm_start=?, active=1 WHERE sea_id=?',
                    (now.isoformat(), sea_id),
                )
                self.db.commit()
            return
        if state['active'] and state['storm_start']:
            start = datetime.fromisoformat(state['storm_start'])
            if now - start >= timedelta(hours=1):
                start_str = self.format_time(start.isoformat(), TZ_OFFSET)
                end_str = self.format_time(now.isoformat(), TZ_OFFSET)
                text = (
                    'Время собирать янтарь. Закончился шторм, длившийся с '
                    f'{start_str} по {end_str}, теперь в окрестностях Светлогорска, Отрадного, Донского и Балтийска можно идти собирать янтарь на пляже (вывоз за пределы региона по закону запрещён).\n\n'
                    'Сообщение сформировано автоматически сервисом #котопогода от Полюбить Калининград'
                )
                for r in self.db.execute('SELECT channel_id FROM amber_channels'):
                    try:
                        await self.api_request('sendMessage', {'chat_id': r['channel_id'], 'text': text})
                        logging.info('Amber message sent to %s', r['channel_id'])
                    except Exception:
                        logging.exception('Amber message failed for %s', r['channel_id'])
            self.db.execute('UPDATE amber_state SET storm_start=NULL, active=0 WHERE sea_id=?', (sea_id,))
            self.db.commit()

    async def handle_update(self, update):
        message = update.get('message') or update.get('channel_post')
        if message:
            await self.handle_message(message)

        elif 'edited_channel_post' in update:
            await self.handle_edited_message(update['edited_channel_post'])

        elif 'callback_query' in update:
            await self.handle_callback(update['callback_query'])
        elif 'my_chat_member' in update:
            await self.handle_my_chat_member(update['my_chat_member'])

    async def handle_my_chat_member(self, chat_update):
        chat = chat_update['chat']
        status = chat_update['new_chat_member']['status']
        if status in {'administrator', 'creator'}:
            self.db.execute(
                'INSERT OR REPLACE INTO channels (chat_id, title) VALUES (?, ?)',
                (chat['id'], chat.get('title', chat.get('username', '')))
            )
            self.db.commit()
            logging.info("Added channel %s", chat['id'])
        else:
            self.db.execute('DELETE FROM channels WHERE chat_id=?', (chat['id'],))
            self.db.commit()
            logging.info("Removed channel %s", chat['id'])

    def get_user(self, user_id):
        cur = self.db.execute('SELECT * FROM users WHERE user_id=?', (user_id,))
        return cur.fetchone()

    def is_pending(self, user_id: int) -> bool:
        cur = self.db.execute('SELECT 1 FROM pending_users WHERE user_id=?', (user_id,))
        return cur.fetchone() is not None

    def pending_count(self) -> int:
        cur = self.db.execute('SELECT COUNT(*) FROM pending_users')
        return cur.fetchone()[0]

    def approve_user(self, uid: int) -> bool:
        if not self.is_pending(uid):
            return False
        cur = self.db.execute('SELECT username FROM pending_users WHERE user_id=?', (uid,))
        row = cur.fetchone()
        username = row['username'] if row else None
        self.db.execute('DELETE FROM pending_users WHERE user_id=?', (uid,))
        self.db.execute(
            'INSERT OR IGNORE INTO users (user_id, username, tz_offset) VALUES (?, ?, ?)',
            (uid, username, TZ_OFFSET)
        )
        if username:
            self.db.execute('UPDATE users SET username=? WHERE user_id=?', (username, uid))
        self.db.execute('DELETE FROM rejected_users WHERE user_id=?', (uid,))
        self.db.commit()
        logging.info('Approved user %s', uid)
        return True

    def reject_user(self, uid: int) -> bool:
        if not self.is_pending(uid):
            return False
        cur = self.db.execute('SELECT username FROM pending_users WHERE user_id=?', (uid,))
        row = cur.fetchone()
        username = row['username'] if row else None
        self.db.execute('DELETE FROM pending_users WHERE user_id=?', (uid,))
        self.db.execute(
            'INSERT OR REPLACE INTO rejected_users (user_id, username, rejected_at) VALUES (?, ?, ?)',
            (uid, username, datetime.utcnow().isoformat()),
        )
        self.db.commit()
        logging.info('Rejected user %s', uid)
        return True

    def is_rejected(self, user_id: int) -> bool:
        cur = self.db.execute('SELECT 1 FROM rejected_users WHERE user_id=?', (user_id,))
        return cur.fetchone() is not None

    def list_scheduled(self):
        cur = self.db.execute(
            'SELECT s.id, s.target_chat_id, c.title as target_title, '
            's.publish_time, s.from_chat_id, s.message_id '
            'FROM schedule s LEFT JOIN channels c ON s.target_chat_id=c.chat_id '
            'WHERE s.sent=0 ORDER BY s.publish_time'
        )
        return cur.fetchall()

    def add_schedule(self, from_chat: int, msg_id: int, targets: set[int], pub_time: str):
        for chat_id in targets:
            self.db.execute(
                'INSERT INTO schedule (from_chat_id, message_id, target_chat_id, publish_time) VALUES (?, ?, ?, ?)',
                (from_chat, msg_id, chat_id, pub_time),
            )
        self.db.commit()
        logging.info('Scheduled %s -> %s at %s', msg_id, list(targets), pub_time)

    def remove_schedule(self, sid: int):
        self.db.execute('DELETE FROM schedule WHERE id=?', (sid,))
        self.db.commit()
        logging.info('Cancelled schedule %s', sid)

    def update_schedule_time(self, sid: int, pub_time: str):
        self.db.execute('UPDATE schedule SET publish_time=? WHERE id=?', (pub_time, sid))
        self.db.commit()
        logging.info('Rescheduled %s to %s', sid, pub_time)

    @staticmethod
    def format_user(user_id: int, username: str | None) -> str:
        label = f"@{username}" if username else str(user_id)
        return f"[{label}](tg://user?id={user_id})"

    @staticmethod
    def parse_offset(offset: str) -> timedelta:
        sign = -1 if offset.startswith('-') else 1
        h, m = offset.lstrip('+-').split(':')
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
        hour, minute = map(int, post_time.split(':'))
        candidate = local_ref.replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )
        if candidate <= local_ref:
            if allow_past:
                candidate = local_ref
            else:
                candidate += timedelta(days=1)
        return candidate - tz_delta

    def format_time(self, ts: str, offset: str) -> str:
        dt = datetime.fromisoformat(ts)
        dt += self.parse_offset(offset)
        return dt.strftime('%H:%M %d.%m.%Y')

    def get_tz_offset(self, user_id: int) -> str:
        cur = self.db.execute('SELECT tz_offset FROM users WHERE user_id=?', (user_id,))
        row = cur.fetchone()
        return row['tz_offset'] if row and row['tz_offset'] else TZ_OFFSET

    def is_authorized(self, user_id):
        return self.get_user(user_id) is not None

    def is_superadmin(self, user_id):
        row = self.get_user(user_id)
        return row and row['is_superadmin']

    def get_amber_sea(self) -> int | None:
        row = self.db.execute('SELECT sea_id FROM amber_state LIMIT 1').fetchone()
        return row['sea_id'] if row else None

    def set_amber_sea(self, sea_id: int):
        self.db.execute('DELETE FROM amber_state')
        self.db.execute(
            'INSERT INTO amber_state (sea_id, storm_start, active) VALUES (?, NULL, 0)',
            (sea_id,),
        )
        self.db.commit()

    def get_amber_channels(self) -> set[int]:
        cur = self.db.execute('SELECT channel_id FROM amber_channels')
        return {r['channel_id'] for r in cur.fetchall()}

    def is_amber_channel(self, channel_id: int) -> bool:
        cur = self.db.execute('SELECT 1 FROM amber_channels WHERE channel_id=?', (channel_id,))
        return cur.fetchone() is not None

    async def show_amber_channels(self, user_id: int):
        enabled = self.get_amber_channels()
        cur = self.db.execute('SELECT chat_id, title FROM channels')
        rows = cur.fetchall()
        if not rows:
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'No channels available'})
            return
        for r in rows:
            cid = r['chat_id']
            icon = '✅' if cid in enabled else '❌'
            btn = 'Выключить отправку' if cid in enabled else 'Включить отправку'
            keyboard = {'inline_keyboard': [[{'text': btn, 'callback_data': f'amber_toggle:{cid}'}]]}
            await self.api_request('sendMessage', {
                'chat_id': user_id,
                'text': f"{r['title'] or cid} {icon}",
                'reply_markup': keyboard,
            })

    async def parse_post_url(self, url: str) -> tuple[int, int] | None:
        """Return chat_id and message_id from a Telegram post URL."""
        m = re.search(r"/c/(\d+)/(\d+)", url)
        if m:
            return int(f"-100{m.group(1)}"), int(m.group(2))
        m = re.search(r"t.me/([^/]+)/(\d+)", url)
        if m:
            resp = await self.api_request('getChat', {'chat_id': f"@{m.group(1)}"})
            if resp.get('ok'):
                return resp['result']['id'], int(m.group(2))
        return None

    def _get_cached_weather(self, city_id: int):
        return self.db.execute(
            "SELECT temperature, weather_code, wind_speed, is_day FROM weather_cache_hour "
            "WHERE city_id=? ORDER BY timestamp DESC LIMIT 1",
            (city_id,),
        ).fetchone()

    def _get_period_weather(self, city_id: int):
        return self.db.execute(
            "SELECT * FROM weather_cache_period WHERE city_id=?",
            (city_id,),
        ).fetchone()

    def _get_sea_cache(self, sea_id: int):
        return self.db.execute(
            "SELECT current, morning, day, evening, night, wave, "
            "morning_wave, day_wave, evening_wave, night_wave FROM sea_cache WHERE sea_id=?",
            (sea_id,),
        ).fetchone()

    @staticmethod
    def strip_header(text: str | None) -> str | None:
        """Remove an existing weather header from text if detected."""
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
                emoji = "\U0001F30A"
                return f"{emoji} {row[key]:.1f}\u00B0C"

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

                emoji = "\U0001F30A"
                if wave_val < 0.5:
                    return f"{emoji} {temp_val:.1f}\u00B0C"
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

                            return f"{emoji} {period_row[t_key]:.0f}\u00B0C"

                        if field == "wind":
                            return f"{period_row[w_key]:.1f}"
                if not row:
                    raise ValueError(f"no current data for city {cid}")
                is_day = row["is_day"] if "is_day" in row.keys() else None
                if field in {"temperature", "temp"}:
                    emoji = weather_emoji(row["weather_code"], is_day)
                    return f"{emoji} {row['temperature']:.1f}\u00B0C"

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


    async def update_weather_posts(self, cities: set[int] | None = None):
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
            elif resp.get("error_code") == 400 and "message is not modified" in resp.get("description", ""):
                logging.info("Weather post %s already up to date", r["id"])
            else:
                logging.error(
                    "Failed to update weather post %s: %s", r["id"], resp
                )

    def latest_weather_url(self) -> str | None:
        cur = self.db.execute(
            "SELECT chat_id, message_id FROM latest_weather_post LIMIT 1"
        )
        row = cur.fetchone()
        if row:
            return self.post_url(row["chat_id"], row["message_id"])
        return None

    def set_latest_weather_post(self, chat_id: int, message_id: int):
        self.db.execute("DELETE FROM latest_weather_post")
        self.db.execute(
            "INSERT INTO latest_weather_post (chat_id, message_id, published_at) VALUES (?, ?, ?)",
            (chat_id, message_id, datetime.utcnow().isoformat()),
        )
        self.db.commit()

    async def update_weather_buttons(self):
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
                resp.get("error_code") == 400 and "message is not modified" in resp.get("description", "")
            ):
                logging.error("Failed to update buttons for %s: %s", r["message_id"], resp)

            self.db.execute(
                "UPDATE weather_posts SET reply_markup=? WHERE chat_id=? AND message_id=?",
                (json.dumps({"inline_keyboard": buttons}), r["chat_id"], r["message_id"]),
            )
        self.db.commit()

    def add_weather_channel(self, channel_id: int, post_time: str):
        next_run = self.next_weather_run(post_time, TZ_OFFSET, allow_past=True)
        self.data.upsert_weather_job(channel_id, post_time, next_run)

    def remove_weather_channel(self, channel_id: int):
        self.data.remove_weather_job(channel_id)

    def list_weather_channels(self):
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
                    "last_published_at": job.last_run_at.isoformat()
                    if job.last_run_at
                    else None,
                    "next_run_at": job.run_at.isoformat(),
                    "title": title_row["title"] if title_row else None,
                }
            )
        return rows

    def set_asset_channel(self, channel_id: int):
        self.set_weather_assets_channel(channel_id)
        self.set_recognition_channel(channel_id)

    def set_weather_assets_channel(self, channel_id: int | None):
        self._store_single_channel("asset_channel", channel_id)
        self.weather_assets_channel_id = channel_id

    def get_weather_assets_channel(self) -> int | None:
        cur = self.db.execute("SELECT channel_id FROM asset_channel LIMIT 1")
        row = cur.fetchone()
        return row["channel_id"] if row else None

    def set_recognition_channel(self, channel_id: int | None):
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
    ) -> int:
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
        cur = self.db.execute('SELECT chat_id, title FROM channels')
        rows = cur.fetchall()
        if not rows:
            await self.api_request(
                'sendMessage',
                {'chat_id': user_id, 'text': 'No channels available'},
            )
            return
        keyboard = {
            'inline_keyboard': [
                [{'text': r['title'], 'callback_data': f'{callback_prefix}:{r["chat_id"]}'}]
                for r in rows
            ]
        }
        self.pending[user_id] = {pending_key: True}
        await self.api_request(
            'sendMessage',
            {
                'chat_id': user_id,
                'text': prompt,
                'reply_markup': keyboard,
            },
        )

    def _schedule_ingest_job(self, asset_id: int, *, reason: str) -> None:
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
        sender_chat_id = message.get("sender_chat", {}).get("id") if message.get("sender_chat") else None
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
            photo = sorted(
                message["photo"], key=lambda p: p.get("file_size", 0)
            )[-1]
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
                    logging.error(
                        "Failed to download file %s: HTTP %s", file_id, file_resp.status
                    )
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
    def _convert_gps(value, ref) -> float | None:
        if not value:
            return None
        try:
            degrees = value[0][0] / value[0][1]
            minutes = value[1][0] / value[1][1]
            seconds = value[2][0] / value[2][1]
            decimal = degrees + minutes / 60 + seconds / 3600
            if ref in {"S", "W"}:
                decimal = -decimal
            return decimal
        except Exception:
            logging.exception("Failed to convert GPS coordinates")
            return None

    def _extract_gps(
        self, image_source: str | Path | BinaryIO
    ) -> tuple[float, float] | None:
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
        gps = exif_dict.get("GPS") or {}
        lat = self._convert_gps(
            gps.get(piexif.GPSIFD.GPSLatitude), gps.get(piexif.GPSIFD.GPSLatitudeRef)
        )
        lon = self._convert_gps(
            gps.get(piexif.GPSIFD.GPSLongitude), gps.get(piexif.GPSIFD.GPSLongitudeRef)
        )
        if lat is None or lon is None:
            return None
        return lat, lon

    def _extract_exif_full(
        self, image_source: str | Path | BinaryIO
    ) -> dict[str, dict[str, Any]]:
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

    def _extract_exif_month(
        self, image_source: str | Path | BinaryIO
    ) -> int | None:
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
    def _extract_weather_enum_from_metadata(
        cls, metadata: dict[str, Any] | None
    ) -> str | None:
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

        items = (
            data.get("result", {}).get("items")
            if isinstance(data, dict)
            else None
        )
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
            f"way(around:250,{lat:.6f},{lon:.6f})[\"natural\"=\"coastline\"];"
            f"relation(around:250,{lat:.6f},{lon:.6f})[\"natural\"=\"coastline\"];"
            f"way(around:250,{lat:.6f},{lon:.6f})[\"natural\"=\"water\"][\"water\"~\"sea|ocean\"];"
            f"relation(around:250,{lat:.6f},{lon:.6f})[\"natural\"=\"water\"][\"water\"~\"sea|ocean\"];"
            f"way(around:250,{lat:.6f},{lon:.6f})[\"place\"=\"sea\"];"
            f"relation(around:250,{lat:.6f},{lon:.6f})[\"place\"=\"sea\"];"
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
            isinstance(state, str)
            and state.strip()
            and state.strip() != "Калининградская область"
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

    def _build_local_file_path(self, asset_id: int, file_meta: dict[str, Any]) -> Path:
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

    def _store_local_file(self, asset_id: int, file_meta: dict[str, Any], data: bytes) -> str:
        path = self._build_local_file_path(asset_id, file_meta)
        try:
            path.write_bytes(data)
        except Exception:
            logging.exception("Failed to write asset file %s", path)
        return str(path)

    async def _job_ingest(self, job: Job):
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
            tz_offset = self.get_tz_offset(asset.author_user_id) if asset.author_user_id else TZ_OFFSET
            vision_job = self.jobs.enqueue(
                "vision", {"asset_id": asset_id, "tz_offset": tz_offset}
            )
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
        cleanup_path = str(downloaded_path)
        try:
            gps = None
            should_extract_gps = asset.kind == "photo" or self._is_convertible_image_document(asset)
            if should_extract_gps:
                gps = self._extract_gps(downloaded_path)
                if not gps:
                    author_id = asset.author_user_id
                    if author_id:
                        await self.api_request(
                            "sendMessage",
                            {
                                "chat_id": author_id,
                                "text": "В изображении отсутствуют EXIF-данные с координатами.",
                            },
                        )
            update_kwargs: dict[str, Any] = {
                "local_path": None,
                "exif_present": bool(gps),
            }
            if gps:
                lat, lon = gps
                update_kwargs["latitude"] = lat
                update_kwargs["longitude"] = lon
                address = await self._reverse_geocode(lat, lon)
                if address:
                    city = address.get("city") or address.get("town") or address.get("village")
                    country = address.get("country")
                    if city:
                        update_kwargs["city"] = city
                    if country:
                        update_kwargs["country"] = country
            self.data.update_asset(asset_id, **update_kwargs)
        finally:
            self._remove_file(cleanup_path)
            if asset.local_path and asset.local_path != cleanup_path:
                self._remove_file(asset.local_path)
        tz_offset = self.get_tz_offset(asset.author_user_id) if asset.author_user_id else TZ_OFFSET
        vision_job = self.jobs.enqueue(
            "vision", {"asset_id": asset_id, "tz_offset": tz_offset}
        )
        logging.info("Asset %s queued for vision job %s after ingest", asset_id, vision_job)


    async def _job_vision(self, job: Job):
        async with self._vision_semaphore:
            await self._job_vision_locked(job)

    async def _job_vision_locked(self, job: Job):
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
                "Поле safety содержит nsfw:boolean и reason:string, где reason всегда непустая строка на русском."
            )
            user_prompt = (
                "Опиши сцену, перечисли объекты, теги, достопримечательности, архитектуру и безопасность фото. Укажи кадровку (framing), "
                "наличие архитектуры крупным планом и панорам, погодный тег (weather_image), сезон и стиль архитектуры (если можно). "
                "Если локация неочевидна, ставь guess_country/guess_city = null и указывай низкую числовую уверенность."
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
                framing = re.sub(
                    r"[\s\-]+",
                    "_",
                    str(framing_raw).strip().lower(),
                ) or None
            if not framing:
                raise RuntimeError("Invalid response from vision model: missing framing")
            if framing not in FRAMING_ALLOWED_VALUES:
                alias = FRAMING_ALIAS_MAP.get(framing)
                if alias in FRAMING_ALLOWED_VALUES:
                    framing = alias
                else:
                    raise RuntimeError(
                        "Invalid response from vision model: unknown framing"
                    )
            architecture_close_up_raw = result.get("architecture_close_up")
            architecture_close_up = (
                bool(architecture_close_up_raw)
                if isinstance(architecture_close_up_raw, bool)
                else str(architecture_close_up_raw)
                .strip()
                .lower()
                in {"1", "true", "yes", "да"}
            )
            architecture_wide_raw = result.get("architecture_wide")
            architecture_wide = (
                bool(architecture_wide_raw)
                if isinstance(architecture_wide_raw, bool)
                else str(architecture_wide_raw)
                .strip()
                .lower()
                in {"1", "true", "yes", "да"}
            )
            weather_image_raw = result.get("weather_image")
            weather_image: str | None = None
            if isinstance(weather_image_raw, str):
                weather_image = re.sub(
                    r"[\s\-]+",
                    "_",
                    weather_image_raw.strip().lower(),
                ) or None
            elif weather_image_raw is not None:
                weather_image = re.sub(
                    r"[\s\-]+",
                    "_",
                    str(weather_image_raw).strip().lower(),
                ) or None
            if not weather_image:
                raise RuntimeError("Invalid response from vision model: missing weather_image")
            normalized_weather = self._normalize_weather_enum(weather_image)
            if not normalized_weather:
                raise RuntimeError(
                    "Invalid response from vision model: unknown weather_image"
                )
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
            exif_month = self._extract_month_from_metadata(metadata_dict)
            if (
                exif_month is None
                and local_path
                and os.path.exists(local_path)
            ):
                exif_month = self._extract_exif_month(local_path)
            season_from_exif = self._season_from_month(exif_month)
            season_final = self._normalize_season(season_from_exif or season_guess)
            season_final_display = (
                SEASON_TRANSLATIONS.get(season_final)
                if season_final
                else None
            )
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
                    logging.error(
                        "Supabase token usage insert failed", extra=log_context
                    )
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
                safety_reason = (
                    "обнаружен чувствительный контент"
                    if nsfw_flag
                    else "безопасно"
                )
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
            rubric_id = self._resolve_rubric_id_for_category(category)
            flower_varieties: list[str] = []
            normalized_tag_set = {tag.lower() for tag in tags if tag}
            if normalized_tag_set.intersection({"flowers", "flower"}):
                flower_varieties = [obj for obj in objects if obj]
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
            if (
                not exif_coords
                and local_path
                and os.path.exists(local_path)
            ):
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

                if (
                    fallback_text
                    and not has_osm_components
                    and fallback_text not in caption_lines
                ):
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
            caption_lines.append(f"Погода: {weather_caption_display}")
            caption_lines.append(f"Сезон: {season_caption_display}")
            if arch_style and arch_style.get("label"):
                confidence_value = arch_style.get("confidence")
                style_line = f"Стиль: {arch_style['label']}"
                confidence_note: str
                if isinstance(confidence_value, (int, float)) and math.isfinite(
                    confidence_value
                ):
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
            if nsfw_flag:
                caption_lines.append(
                    "⚠️ Чувствительный контент: "
                    + (safety_reason or "потенциально NSFW")
                )
            elif safety_reason:
                caption_lines.append("Безопасность: " + safety_reason)

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
            if guess_country and (
                not guess_city or guess_country.lower() != guess_city.lower()
            ):
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
            }
            if rubric_id is not None:
                result_payload["rubric_id"] = rubric_id
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
                response.completion_tokens if response and response.completion_tokens is not None else "-",
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
                        asset.kind = "photo"
                        asset.file_id = new_file_id
                        asset.file_unique_id = photo_meta.get("file_unique_id")
                        asset.mime_type = photo_meta.get("mime_type")
                        asset.file_size = photo_meta.get("file_size")
                        asset.width = photo_meta.get("width")
                        asset.height = photo_meta.get("height")
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
                weather_final_display
                or photo_weather_display
                or photo_weather
                or "-"
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
            arch_style_label = (
                arch_style.get("label") if isinstance(arch_style, dict) else None
            )
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
                            f"EXIF (raw)\n```json\n{exif_json}\n```"
                            if exif_json
                            else "EXIF (raw)"
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
            if (
                delete_original_after_post
                and not self.dry_run
                and new_mid
                and asset.message_id
            ):
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

    def next_asset(self, tags: set[str] | None):
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
        return any(indicator in description and "not found" in description for indicator in indicators)

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



    async def handle_message(self, message):
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


        if 'from' not in message:
            # ignore channel posts when asset channel is not configured
            return


        text = message.get('text', '')
        user_id = message['from']['id']
        username = message['from'].get('username')

        preview_state = self.pending_flowers_previews.get(user_id)
        if preview_state and preview_state.get('awaiting_instruction'):
            reply_to = message.get('reply_to_message') or {}
            prompt_id = preview_state.get('instruction_prompt_id')
            if prompt_id and reply_to.get('message_id') == prompt_id:
                instructions_text = text.strip()
                preview_state['instructions'] = instructions_text
                preview_state['awaiting_instruction'] = False
                preview_state['instruction_prompt_id'] = None
                rubric_code = preview_state.get('rubric_code')
                rubric = (
                    self.data.get_rubric_by_code(rubric_code)
                    if rubric_code
                    else None
                )
                if rubric:
                    cities = list(preview_state.get('cities') or [])
                    asset_count = len(preview_state.get('assets') or [])
                    greeting, hashtags = await self._generate_flowers_copy(
                        rubric,
                        cities,
                        asset_count,
                        instructions=instructions_text,
                        weather=preview_state.get('weather'),
                    )
                    caption, prepared_hashtags = self._build_flowers_caption(
                        greeting,
                        cities,
                        hashtags,
                        preview_state.get('weather'),
                    )
                    await self._update_flowers_preview_caption_state(
                        preview_state,
                        caption=caption,
                        greeting=greeting,
                        hashtags=hashtags,
                        prepared_hashtags=prepared_hashtags,
                    )
                    await self.api_request(
                        'sendMessage',
                        {
                            'chat_id': user_id,
                            'text': 'Инструкция сохранена, подпись обновлена.',
                        },
                    )
                else:
                    await self.api_request(
                        'sendMessage',
                        {
                            'chat_id': user_id,
                            'text': 'Не удалось обновить подпись: рубрика недоступна.',
                        },
                    )
                return

        if user_id in self.pending and self.pending[user_id].get('rubric_input'):
            if not self.is_superadmin(user_id):
                del self.pending[user_id]
            else:
                await self.api_request(
                    'sendMessage',
                    {
                        'chat_id': user_id,
                        'text': 'Используйте кнопки для настройки рубрики.',
                    },
                )
            return

        if text.startswith('/help'):
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
                    "*Каналы и погода*\n"
                    "- `/channels` — список подключённых каналов.\n"
                    "- `/set_weather_assets_channel` и `/set_recognition_channel` открывают кнопочный выбор канала.\n"
                    "- `/setup_weather` и `/list_weather_channels` показывают расписания с кнопками `Run now`/`Stop`.\n"
                    "- `/weather`, `/history`, `/scheduled` — статусные отчёты, где управлять публикациями можно прямо из inline-кнопок.\n"
                    "- `/amber` — кнопочное управление каналом Янтарный."
                )
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
                    'sendMessage',
                    {
                        'chat_id': user_id,
                        'text': chunk,
                        'parse_mode': 'Markdown',
                    },
                )
            return

        # first /start registers superadmin or puts user in queue
        if text.startswith('/start'):
            if self.get_user(user_id):
                await self.api_request('sendMessage', {
                    'chat_id': user_id,
                    'text': 'Bot is working'
                })
                return

            if self.is_rejected(user_id):
                await self.api_request('sendMessage', {
                    'chat_id': user_id,
                    'text': 'Access denied by administrator'
                })
                return

            if self.is_pending(user_id):
                await self.api_request('sendMessage', {
                    'chat_id': user_id,
                    'text': 'Awaiting approval'
                })
                return

            cur = self.db.execute('SELECT COUNT(*) FROM users')
            user_count = cur.fetchone()[0]
            if user_count == 0:
                self.db.execute('INSERT INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, ?)', (user_id, username, TZ_OFFSET))
                self.db.commit()
                logging.info('Registered %s as superadmin', user_id)
                await self.api_request('sendMessage', {
                    'chat_id': user_id,
                    'text': 'You are superadmin'
                })
                return

            if self.pending_count() >= 10:
                await self.api_request('sendMessage', {
                    'chat_id': user_id,
                    'text': 'Registration queue full, try later'
                })
                logging.info('Registration rejected for %s due to full queue', user_id)
                return

            self.db.execute(
                'INSERT OR IGNORE INTO pending_users (user_id, username, requested_at) VALUES (?, ?, ?)',
                (user_id, username, datetime.utcnow().isoformat())
            )
            self.db.commit()
            logging.info('User %s added to pending queue', user_id)
            await self.api_request('sendMessage', {
                'chat_id': user_id,
                'text': 'Registration pending approval'
            })
            return

        if text.startswith('/add_user') and self.is_superadmin(user_id):
            parts = text.split()
            if len(parts) == 2:
                uid = int(parts[1])
                if not self.get_user(uid):
                    self.db.execute('INSERT INTO users (user_id) VALUES (?)', (uid,))
                    self.db.commit()
                await self.api_request('sendMessage', {
                    'chat_id': user_id,
                    'text': f'User {uid} added'
                })
            return

        if text.startswith('/remove_user') and self.is_superadmin(user_id):
            parts = text.split()
            if len(parts) == 2:
                uid = int(parts[1])
                self.db.execute('DELETE FROM users WHERE user_id=?', (uid,))
                self.db.commit()
                await self.api_request('sendMessage', {
                    'chat_id': user_id,
                    'text': f'User {uid} removed'
                })
            return

        if text.startswith('/tz'):
            parts = text.split()
            if not self.is_authorized(user_id):
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Not authorized'})
                return
            if len(parts) != 2:
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Usage: /tz +02:00'})
                return
            try:
                self.parse_offset(parts[1])
            except Exception:
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Invalid offset'})
                return
            self.db.execute('UPDATE users SET tz_offset=? WHERE user_id=?', (parts[1], user_id))
            self.db.commit()
            TZ_OFFSET = parts[1]
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': f'Timezone set to {parts[1]}'})
            return

        if text.startswith('/list_users') and self.is_superadmin(user_id):
            cur = self.db.execute('SELECT user_id, username, is_superadmin FROM users')
            rows = cur.fetchall()
            msg = '\n'.join(
                f"{self.format_user(r['user_id'], r['username'])} {'(admin)' if r['is_superadmin'] else ''}"
                for r in rows
            )
            await self.api_request('sendMessage', {
                'chat_id': user_id,
                'text': msg or 'No users',
                'parse_mode': 'Markdown'
            })
            return

        if text.startswith('/pending') and self.is_superadmin(user_id):
            cur = self.db.execute('SELECT user_id, username, requested_at FROM pending_users')
            rows = cur.fetchall()
            if not rows:
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'No pending users'})
                return

            msg = '\n'.join(
                f"{self.format_user(r['user_id'], r['username'])} requested {r['requested_at']}"
                for r in rows
            )
            keyboard = {
                'inline_keyboard': [
                    [
                        {'text': 'Approve', 'callback_data': f'approve:{r["user_id"]}'},
                        {'text': 'Reject', 'callback_data': f'reject:{r["user_id"]}'}
                    ]
                    for r in rows
                ]
            }
            await self.api_request('sendMessage', {
                'chat_id': user_id,
                'text': msg,
                'parse_mode': 'Markdown',
                'reply_markup': keyboard
            })
            return

        if text.startswith('/rubrics') and self.is_superadmin(user_id):
            await self._send_rubric_dashboard(user_id)
            return




        if text.startswith('/approve') and self.is_superadmin(user_id):
            parts = text.split()
            if len(parts) == 2:
                uid = int(parts[1])
                if self.approve_user(uid):
                    cur = self.db.execute('SELECT username FROM users WHERE user_id=?', (uid,))
                    row = cur.fetchone()
                    uname = row['username'] if row else None
                    await self.api_request('sendMessage', {
                        'chat_id': user_id,
                        'text': f'{self.format_user(uid, uname)} approved',
                        'parse_mode': 'Markdown'
                    })
                    await self.api_request('sendMessage', {'chat_id': uid, 'text': 'You are approved'})
                else:
                    await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'User not in pending list'})
            return

        if text.startswith('/reject') and self.is_superadmin(user_id):
            parts = text.split()
            if len(parts) == 2:
                uid = int(parts[1])
                if self.reject_user(uid):
                    cur = self.db.execute('SELECT username FROM rejected_users WHERE user_id=?', (uid,))
                    row = cur.fetchone()
                    uname = row['username'] if row else None
                    await self.api_request('sendMessage', {
                        'chat_id': user_id,
                        'text': f'{self.format_user(uid, uname)} rejected',
                        'parse_mode': 'Markdown'
                    })
                    await self.api_request('sendMessage', {'chat_id': uid, 'text': 'Your registration was rejected'})
                else:
                    await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'User not in pending list'})
            return

        if text.startswith('/channels') and self.is_superadmin(user_id):
            cur = self.db.execute('SELECT chat_id, title FROM channels')
            rows = cur.fetchall()
            msg = '\n'.join(f"{r['title']} ({r['chat_id']})" for r in rows)
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': msg or 'No channels'})
            return

        if text.startswith('/history'):
            cur = self.db.execute(
                'SELECT target_chat_id, sent_at FROM schedule WHERE sent=1 ORDER BY sent_at DESC LIMIT 10'
            )
            rows = cur.fetchall()
            offset = self.get_tz_offset(user_id)
            msg = '\n'.join(
                f"{r['target_chat_id']} at {self.format_time(r['sent_at'], offset)}"
                for r in rows
            )
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': msg or 'No history'})
            return

        if text.startswith('/scheduled') and self.is_authorized(user_id):
            rows = self.list_scheduled()
            if not rows:
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'No scheduled posts'})
                return
            offset = self.get_tz_offset(user_id)
            for r in rows:
                ok = False
                try:
                    resp = await self.api_request('forwardMessage', {
                        'chat_id': user_id,
                        'from_chat_id': r['from_chat_id'],
                        'message_id': r['message_id']
                    })
                    ok = resp.get('ok', False)
                    if not ok and resp.get('error_code') == 400 and 'not' in resp.get('description', '').lower():
                        resp = await self.api_request('copyMessage', {
                            'chat_id': user_id,
                            'from_chat_id': r['from_chat_id'],
                            'message_id': r['message_id']
                        })
                        ok = resp.get('ok', False)
                except Exception:
                    logging.exception('Failed to forward message %s', r['id'])
                if not ok:
                    link = None
                    if str(r['from_chat_id']).startswith('-100'):
                        cid = str(r['from_chat_id'])[4:]
                        link = f'https://t.me/c/{cid}/{r["message_id"]}'
                    await self.api_request('sendMessage', {
                        'chat_id': user_id,
                        'text': link or f'Message {r["message_id"]} from {r["from_chat_id"]}'
                    })
                keyboard = {
                    'inline_keyboard': [[
                        {'text': 'Cancel', 'callback_data': f'cancel:{r["id"]}'},
                        {'text': 'Reschedule', 'callback_data': f'resch:{r["id"]}'}
                    ]]
                }
                target = (
                    f"{r['target_title']} ({r['target_chat_id']})"
                    if r['target_title'] else str(r['target_chat_id'])
                )
                await self.api_request('sendMessage', {
                    'chat_id': user_id,
                    'text': f"{r['id']}: {target} at {self.format_time(r['publish_time'], offset)}",
                    'reply_markup': keyboard
                })
            return

        if text.startswith('/addbutton'):
            if not self.is_authorized(user_id):
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Not authorized'})
                return

            parts = text.split()
            if len(parts) < 4:
                await self.api_request('sendMessage', {
                    'chat_id': user_id,
                    'text': 'Usage: /addbutton <post_url> <text> <url>'
                })
                return
            parsed = await self.parse_post_url(parts[1])
            if not parsed:
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Invalid post URL'})
                return
            chat_id, msg_id = parsed
            keyboard_text = " ".join(parts[2:-1])
            fwd = await self.api_request(

                'forwardMessage',

                {
                    'chat_id': user_id,
                    'from_chat_id': chat_id,
                    'message_id': msg_id,
                },
            )


            markup = None
            caption = None
            caption_entities = None
            if fwd.get('ok') and fwd.get('result'):
                message = fwd['result']
                markup = message.get('reply_markup')
                caption = message.get('caption')
                caption_entities = message.get('caption_entities')
                await self.api_request(
                    'deleteMessage',
                    {'chat_id': user_id, 'message_id': message.get('message_id')},
                )
            key = (chat_id, msg_id)
            info = self.manual_buttons.get(key)
            if info is None:
                base_buttons = markup.get('inline_keyboard', []) if markup else []
                info = {
                    'base': [
                        [dict(btn) for btn in row]
                        for row in base_buttons
                    ],
                    'custom': [],
                }
            new_row = [{'text': keyboard_text, 'url': parts[-1]}]
            info['custom'].append([dict(btn) for btn in new_row])
            self.manual_buttons[key] = info

            keyboard = {
                'inline_keyboard': [
                    [dict(btn) for btn in row]
                    for row in info['base'] + info['custom']
                ]
            }

            payload = {
                'chat_id': chat_id,
                'message_id': msg_id,
                'reply_markup': keyboard,
            }
            method = 'editMessageReplyMarkup'
            if caption is not None:
                method = 'editMessageCaption'
                payload['caption'] = caption
                if caption_entities:
                    payload['caption_entities'] = caption_entities

            resp = await self.api_request(method, payload)

            if not resp.get('ok') and resp.get('error_code') == 400 and 'message is not modified' in resp.get('description', ''):
                resp['ok'] = True
            if resp.get('ok'):
                logging.info('Updated message %s with button', msg_id)
                cur = self.db.execute(
                    'SELECT 1 FROM weather_posts WHERE chat_id=? AND message_id=?',
                    (chat_id, msg_id),
                )
                if cur.fetchone():
                    self.db.execute(
                        'UPDATE weather_posts SET reply_markup=? WHERE chat_id=? AND message_id=?',
                        (json.dumps(keyboard), chat_id, msg_id),
                    )
                    self.db.commit()

                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Button added'})
            else:
                logging.error('Failed to add button to %s: %s', msg_id, resp)
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Failed to add button'})
            return

        if text.startswith('/delbutton'):
            if not self.is_authorized(user_id):
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Not authorized'})
                return

            parts = text.split()
            if len(parts) != 2:
                await self.api_request('sendMessage', {
                    'chat_id': user_id,
                    'text': 'Usage: /delbutton <post_url>'
                })
                return
            parsed = await self.parse_post_url(parts[1])
            if not parsed:
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Invalid post URL'})
                return
            chat_id, msg_id = parsed

            resp = await self.api_request(
                'editMessageReplyMarkup',
                {
                    'chat_id': chat_id,
                    'message_id': msg_id,
                    'reply_markup': {},
                },
            )

            if not resp.get('ok') and resp.get('error_code') == 400 and 'message is not modified' in resp.get('description', ''):
                resp['ok'] = True
            if resp.get('ok'):

                logging.info('Removed buttons from message %s', msg_id)
                self.db.execute(
                    'DELETE FROM weather_link_posts WHERE chat_id=? AND message_id=?',
                    (chat_id, msg_id),
                )
                self.db.execute(
                    'UPDATE weather_posts SET reply_markup=NULL WHERE chat_id=? AND message_id=?',
                    (chat_id, msg_id),
                )
                self.db.commit()
                self.manual_buttons.pop((chat_id, msg_id), None)
            else:
                logging.error('Failed to remove button from %s: %s', msg_id, resp)
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Failed to remove button'})
            return

        if text.startswith('/addweatherbutton') and self.is_superadmin(user_id):
            parts = text.split()
            if len(parts) < 3:
                await self.api_request(
                    'sendMessage',
                    {
                        'chat_id': user_id,
                        'text': 'Usage: /addweatherbutton <post_url> <text> [url]'
                    },
                )
                return

            url = None
            if len(parts) > 3 and parts[-1].startswith(('http://', 'https://')):
                url = parts[-1]
                btn_text = " ".join(parts[2:-1])
            else:
                btn_text = " ".join(parts[2:])
                url = self.latest_weather_url()
                if not url:
                    await self.api_request(
                        'sendMessage',
                        {
                            'chat_id': user_id,
                            'text': 'Specify forecast URL after text'
                        },
                    )
                    return

            parsed = await self.parse_post_url(parts[1])
            if not parsed:
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Invalid post URL'})
                return
            chat_id, msg_id = parsed
            fwd = await self.api_request(
                'copyMessage',
                {'chat_id': user_id, 'from_chat_id': chat_id, 'message_id': msg_id},
            )
            markup = None
            if not fwd.get('ok'):
                fwd = await self.api_request(
                    'forwardMessage',
                    {'chat_id': user_id, 'from_chat_id': chat_id, 'message_id': msg_id},
                )
            if fwd.get('ok') and fwd.get('result'):
                markup = fwd['result'].get('reply_markup')
                await self.api_request(
                    'deleteMessage',
                    {
                        'chat_id': user_id,
                        'message_id': fwd['result'].get('message_id'),
                    },
                )

            row = self.db.execute(
                'SELECT base_markup, button_texts FROM weather_link_posts WHERE chat_id=? AND message_id=?',
                (chat_id, msg_id),
            ).fetchone()
            base_markup = row['base_markup'] if row else json.dumps(markup) if markup else None
            texts = json.loads(row['button_texts']) if row else []
            if row is None:
                base_buttons = markup.get('inline_keyboard', []) if markup else []
            else:
                base_buttons = json.loads(base_markup)['inline_keyboard'] if base_markup else []
            texts.append(btn_text)

            rendered_texts = [self._render_template(t) or t for t in texts]
            weather_buttons = [{'text': t, 'url': url} for t in rendered_texts]
            keyboard_buttons = base_buttons + [weather_buttons]

            resp = await self.api_request(
                'editMessageReplyMarkup',
                {
                    'chat_id': chat_id,
                    'message_id': msg_id,
                    'reply_markup': {'inline_keyboard': keyboard_buttons},
                },
            )
            if resp.get('ok'):
                self.db.execute(
                    'INSERT OR REPLACE INTO weather_link_posts (chat_id, message_id, base_markup, button_texts) VALUES (?, ?, ?, ?)',
                    (chat_id, msg_id, base_markup, json.dumps(texts)),
                )
                self.db.execute(
                    'UPDATE weather_posts SET reply_markup=? WHERE chat_id=? AND message_id=?',
                    (json.dumps({'inline_keyboard': keyboard_buttons}), chat_id, msg_id),
                )
                self.db.commit()
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Weather button added'})
            else:
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Failed to add weather button'})
            return

        if text.startswith('/addcity') and self.is_superadmin(user_id):
            parts = text.split(maxsplit=2)
            if len(parts) == 3:
                name = parts[1]
                coords = self._parse_coords(parts[2])
                if not coords:
                    await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Invalid coordinates'})
                    return
                lat, lon = coords
                try:
                    self.db.execute('INSERT INTO cities (name, lat, lon) VALUES (?, ?, ?)', (name, lat, lon))
                    self.db.commit()
                    await self.api_request('sendMessage', {'chat_id': user_id, 'text': f'City {name} added'})
                except sqlite3.IntegrityError:
                    await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'City already exists'})
            else:
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Usage: /addcity <name> <lat> <lon>'})
            return

        if text.startswith('/addsea') and self.is_superadmin(user_id):

            parts = text.split(maxsplit=2)
            if len(parts) == 3:
                name = parts[1]
                coords = self._parse_coords(parts[2])
                if not coords:
                    await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Invalid coordinates'})
                    return
                lat, lon = coords

                try:
                    self.db.execute('INSERT INTO seas (name, lat, lon) VALUES (?, ?, ?)', (name, lat, lon))
                    self.db.commit()
                    await self.api_request('sendMessage', {'chat_id': user_id, 'text': f'Sea {name} added'})
                except sqlite3.IntegrityError:
                    await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Sea already exists'})
            else:
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Usage: /addsea <name> <lat> <lon>'})
            return

        if text.startswith('/cities') and self.is_superadmin(user_id):
            cur = self.db.execute('SELECT id, name, lat, lon FROM cities ORDER BY id')
            rows = cur.fetchall()
            if not rows:
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'No cities'})
                return
            for r in rows:
                keyboard = {'inline_keyboard': [[{'text': 'Delete', 'callback_data': f'city_del:{r["id"]}'}]]}
                await self.api_request('sendMessage', {
                    'chat_id': user_id,
                    'text': f"{r['id']}: {r['name']} ({r['lat']:.6f}, {r['lon']:.6f})",
                    'reply_markup': keyboard
                })
            return

        if text.startswith('/seas') and self.is_superadmin(user_id):
            cur = self.db.execute('SELECT id, name, lat, lon FROM seas ORDER BY id')
            rows = cur.fetchall()
            if not rows:
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'No seas'})
                return
            for r in rows:
                keyboard = {'inline_keyboard': [[{'text': 'Delete', 'callback_data': f'sea_del:{r["id"]}'}]]}
                await self.api_request('sendMessage', {
                    'chat_id': user_id,
                    'text': f"{r['id']}: {r['name']} ({r['lat']:.6f}, {r['lon']:.6f})",
                    'reply_markup': keyboard
                })
            return

        if text.startswith('/amber') and self.is_superadmin(user_id):
            sea_id = self.get_amber_sea()
            if sea_id is None:
                cur = self.db.execute('SELECT id, name FROM seas ORDER BY id')
                rows = cur.fetchall()
                if not rows:
                    await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'No seas'})
                    return
                keyboard = {'inline_keyboard': [[{'text': r['name'], 'callback_data': f'amber_sea:{r["id"]}'}] for r in rows]}
                self.pending[user_id] = {'amber_sea': True}
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Select sea', 'reply_markup': keyboard})
            else:
                await self.show_amber_channels(user_id)
            return

        if text.startswith('/weatherposts') and self.is_superadmin(user_id):
            parts = text.split(maxsplit=1)
            force = len(parts) > 1 and parts[1] == 'update'
            if force:
                await self.update_weather_posts()
                await self.update_weather_buttons()
            cur = self.db.execute(
                'SELECT chat_id, message_id, template FROM weather_posts ORDER BY id'
            )
            post_rows = cur.fetchall()
            for r in post_rows:
                header = self._render_template(r['template'])
                url = self.post_url(r['chat_id'], r['message_id'])
                text = f"{url} {header}" if header else f"{url} no data"
                keyboard = {
                    'inline_keyboard': [[
                        {
                            'text': 'Stop weather',
                            'callback_data': f'wpost_del:{r["chat_id"]}:{r["message_id"]}'
                        }
                    ]]
                }
                await self.api_request(
                    'sendMessage',
                    {'chat_id': user_id, 'text': text, 'reply_markup': keyboard},
                )
            cur = self.db.execute('SELECT chat_id, message_id, button_texts FROM weather_link_posts ORDER BY rowid')
            rows = cur.fetchall()
            if not rows and not post_rows:
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'No weather posts'})
                return
            for r in rows:

                rendered = [self._render_template(t) or t for t in json.loads(r['button_texts'])]
                texts = ', '.join(rendered)

                keyboard = {'inline_keyboard': [[{'text': 'Remove buttons', 'callback_data': f'wbtn_del:{r["chat_id"]}:{r["message_id"]}'}]]}
                await self.api_request(
                    'sendMessage',
                    {
                        'chat_id': user_id,
                        'text': f"{self.post_url(r['chat_id'], r['message_id'])} buttons: {texts}",
                        'reply_markup': keyboard,
                    },
                )
            return

        if text.startswith('/setup_weather') and self.is_superadmin(user_id):
            cur = self.db.execute('SELECT chat_id, title FROM channels')
            rows = cur.fetchall()
            existing = {r['channel_id'] for r in self.list_weather_channels()}
            options = [r for r in rows if r['chat_id'] not in existing]
            if not options:
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'No channels available'})
                return
            keyboard = {'inline_keyboard': [[{'text': r['title'], 'callback_data': f'ws_ch:{r["chat_id"]}'}] for r in options]}
            self.pending[user_id] = {'setup_weather': True}
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Select channel', 'reply_markup': keyboard})
            return

        if text.startswith('/list_weather_channels') and self.is_superadmin(user_id):
            rows = self.list_weather_channels()
            if not rows:
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'No weather channels'})
                return
            for r in rows:

                last = r['last_published_at']
                if last:
                    last = self.format_time(last, self.get_tz_offset(user_id))
                else:
                    last = 'never'
                keyboard = {
                    'inline_keyboard': [[
                        {'text': 'Run now', 'callback_data': f'wrnow:{r["channel_id"]}'},
                        {'text': 'Stop', 'callback_data': f'wstop:{r["channel_id"]}'}
                    ]]
                }
                await self.api_request(
                    'sendMessage',
                    {
                        'chat_id': user_id,
                        'text': f"{r['title'] or r['channel_id']} at {r['post_time']} last {last}",
                        'reply_markup': keyboard,
                    },
                )

            return

        if text.startswith('/set_assets_channel') and self.is_superadmin(user_id):
            parts = text.split(maxsplit=1)
            confirmed = len(parts) > 1 and parts[1].strip().lower() == 'confirm'
            if not confirmed:
                await self.api_request(
                    'sendMessage',
                    {
                        'chat_id': user_id,
                        'text': (
                            'Команда `/set_assets_channel` устанавливает один и тот же канал для '
                            'хранилища погоды и распознавания. Используйте её только если это '
                            'действительно необходимо и подтвердите действие командой '
                            '`/set_assets_channel confirm`. Для раздельных складов вызовите '
                            'по очереди `/set_weather_assets_channel` и `/set_recognition_channel`.'
                        ),
                        'parse_mode': 'Markdown',
                    },
                )
                return
            await self._prompt_channel_selection(
                user_id,
                pending_key='set_assets',
                callback_prefix='asset_ch',
                prompt='Select asset channel',
            )
            return

        if text.startswith('/set_weather_assets_channel') and self.is_superadmin(user_id):
            await self._prompt_channel_selection(
                user_id,
                pending_key='set_weather_assets',
                callback_prefix='weather_ch',
                prompt='Select weather assets channel',
            )
            return

        if text.startswith('/set_recognition_channel') and self.is_superadmin(user_id):
            await self._prompt_channel_selection(
                user_id,
                pending_key='set_recognition',
                callback_prefix='recognition_ch',
                prompt='Select recognition channel',
            )
            return



        if text.startswith('/weather') and self.is_superadmin(user_id):

            parts = text.split(maxsplit=1)
            if len(parts) > 1 and parts[1].lower() == 'now':
                await self.collect_weather(force=True)
                await self.collect_sea(force=True)

            cur = self.db.execute('SELECT id, name FROM cities ORDER BY id')
            rows = cur.fetchall()
            if not rows:
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'No cities'})
                return
            lines = []
            for r in rows:
                w = self.db.execute(
                    'SELECT temperature, weather_code, wind_speed, is_day, timestamp FROM weather_cache_hour WHERE city_id=? ORDER BY timestamp DESC LIMIT 1',
                    (r['id'],),
                ).fetchone()
                if w:
                    emoji = weather_emoji(w['weather_code'], w['is_day'])
                    lines.append(
                        f"{r['name']}: {w['temperature']:.1f}°C {emoji} wind {w['wind_speed']:.1f} m/s at {w['timestamp']}"

                    )
                else:
                    lines.append(f"{r['name']}: no data")

            cur = self.db.execute('SELECT id, name FROM seas ORDER BY id')
            sea_rows = cur.fetchall()
            for r in sea_rows:
                row = self._get_sea_cache(r['id'])
                if row and row['current'] is not None:
                    emoji = "\U0001F30A"
                    lines.append(
                        f"{r['name']}: {emoji} {row['current']:.1f}°C {row['morning']:.1f}/{row['day']:.1f}/{row['evening']:.1f}/{row['night']:.1f}"
                    )
                else:
                    lines.append(f"{r['name']}: no data")
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': '\n'.join(lines)})
            return

        if text.startswith('/regweather') and self.is_superadmin(user_id):
            parts = text.split(maxsplit=2)
            if len(parts) < 3:
                await self.api_request('sendMessage', {
                    'chat_id': user_id,
                    'text': 'Usage: /regweather <post_url> <template>'
                })
                return
            parsed = await self.parse_post_url(parts[1])
            if not parsed:
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Invalid post URL'})
                return
            template = parts[2]
            chat_id, msg_id = parsed
            resp = await self.api_request(
                'copyMessage',
                {
                    'chat_id': user_id,
                    'from_chat_id': chat_id,
                    'message_id': msg_id,
                },
            )
            if not resp.get('ok') or not resp.get('result'):
                resp = await self.api_request(
                    'forwardMessage',
                    {
                        'chat_id': user_id,
                        'from_chat_id': chat_id,
                        'message_id': msg_id,
                    },
                )
            elif (
                not resp['result'].get('text')
                and not resp['result'].get('caption')
            ):
                await self.api_request(
                    'deleteMessage',
                    {'chat_id': user_id, 'message_id': resp['result']['message_id']},
                )
                resp = await self.api_request(
                    'forwardMessage',
                    {
                        'chat_id': user_id,
                        'from_chat_id': chat_id,
                        'message_id': msg_id,
                    },
                )
            if not resp.get('ok'):
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Cannot read post'})
                return

            base_text = resp['result'].get('text')
            base_caption = resp['result'].get('caption')
            base_text = self.strip_header(base_text)
            base_caption = self.strip_header(base_caption)
            markup = resp['result'].get('reply_markup')

            if base_text is None and base_caption is None:
                base_text = ''
            await self.api_request('deleteMessage', {'chat_id': user_id, 'message_id': resp['result']['message_id']})
            self.db.execute(

                'INSERT OR REPLACE INTO weather_posts (chat_id, message_id, template, base_text, base_caption, reply_markup) VALUES (?, ?, ?, ?, ?, ?)',
                (chat_id, msg_id, template, base_text, base_caption, json.dumps(markup) if markup else None)

            )
            self.db.commit()
            # Ensure data is available for the placeholders right away
            # so the post gets updated immediately after registration.
            await self.collect_weather(force=True)
            await self.collect_sea(force=True)
            await self.update_weather_posts({int(m.group(1)) for m in re.finditer(r"{(\d+)\|", template)})
            await self.api_request('sendMessage', {
                'chat_id': user_id,
                'text': 'Weather post registered'
            })
            return



        # handle time input for scheduling
        if user_id in self.pending and 'await_time' in self.pending[user_id]:
            time_str = text.strip()
            try:
                if len(time_str.split()) == 1:
                    dt = datetime.strptime(time_str, '%H:%M')
                    pub_time = datetime.combine(date.today(), dt.time())
                else:
                    pub_time = datetime.strptime(time_str, '%d.%m.%Y %H:%M')
            except ValueError:
                await self.api_request('sendMessage', {
                    'chat_id': user_id,
                    'text': 'Invalid time format'
                })
                return
            offset = self.get_tz_offset(user_id)
            pub_time_utc = pub_time - self.parse_offset(offset)
            if pub_time_utc <= datetime.utcnow():
                await self.api_request('sendMessage', {
                    'chat_id': user_id,
                    'text': 'Time must be in future'
                })
                return
            data = self.pending.pop(user_id)
            if 'reschedule_id' in data:
                self.update_schedule_time(data['reschedule_id'], pub_time_utc.isoformat())
                await self.api_request('sendMessage', {
                    'chat_id': user_id,
                    'text': f'Rescheduled for {self.format_time(pub_time_utc.isoformat(), offset)}'
                })
            else:
                test = await self.api_request(
                    'forwardMessage',
                    {
                        'chat_id': user_id,
                        'from_chat_id': data['from_chat_id'],
                        'message_id': data['message_id']
                    }
                )
                if not test.get('ok'):
                    await self.api_request('sendMessage', {
                        'chat_id': user_id,
                        'text': f"Add the bot to channel {data['from_chat_id']} (reader role) first"
                    })
                    return
                self.add_schedule(data['from_chat_id'], data['message_id'], data['selected'], pub_time_utc.isoformat())
                await self.api_request('sendMessage', {
                    'chat_id': user_id,
                    'text': f"Scheduled to {len(data['selected'])} channels for {self.format_time(pub_time_utc.isoformat(), offset)}"
                })
            return

        if user_id in self.pending and self.pending[user_id].get('weather_time'):
            time_str = text.strip()
            try:
                dt = datetime.strptime(time_str, '%H:%M')
            except ValueError:
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Invalid time format'})
                return
            self.add_weather_channel(self.pending[user_id]['channel'], time_str)
            del self.pending[user_id]
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Weather channel registered'})
            return

        # start scheduling on forwarded message
        if 'forward_from_chat' in message and self.is_authorized(user_id):
            from_chat = message['forward_from_chat']['id']
            msg_id = message['forward_from_message_id']
            cur = self.db.execute('SELECT chat_id, title FROM channels')
            rows = cur.fetchall()
            if not rows:
                await self.api_request('sendMessage', {
                    'chat_id': user_id,
                    'text': 'No channels available'
                })
                return
            keyboard = {
                'inline_keyboard': [
                    [{'text': r['title'], 'callback_data': f'addch:{r["chat_id"]}'}] for r in rows
                ] + [[{'text': 'Done', 'callback_data': 'chdone'}]]
            }
            self.pending[user_id] = {
                'from_chat_id': from_chat,
                'message_id': msg_id,
                'selected': set()
            }
            await self.api_request('sendMessage', {
                'chat_id': user_id,
                'text': 'Select channels',
                'reply_markup': keyboard
            })
            return
        else:
            if not self.is_authorized(user_id):
                await self.api_request('sendMessage', {
                    'chat_id': user_id,
                    'text': 'Not authorized'
                })
            else:
                await self.api_request('sendMessage', {
                    'chat_id': user_id,
                    'text': 'Please forward a post from a channel'
                })

    async def handle_callback(self, query):
        user_id = query['from']['id']
        data = query['data']
        if data.startswith('addch:') and user_id in self.pending:
            chat_id = int(data.split(':')[1])
            if 'selected' in self.pending[user_id]:
                s = self.pending[user_id]['selected']
                if chat_id in s:
                    s.remove(chat_id)
                else:
                    s.add(chat_id)
        elif data == 'chdone' and user_id in self.pending:
            info = self.pending[user_id]
            if not info.get('selected'):
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Select at least one channel'})
            else:
                self.pending[user_id]['await_time'] = True
                await self.api_request('sendMessage', {
                    'chat_id': user_id,
                    'text': 'Enter time (HH:MM or DD.MM.YYYY HH:MM)'
                })
        elif data.startswith('ws_ch:') and user_id in self.pending and self.pending[user_id].get('setup_weather'):
            cid = int(data.split(':')[1])
            self.pending[user_id] = {'channel': cid, 'weather_time': False, 'setup_weather': True}
            keyboard = {'inline_keyboard': [[{'text': '17:55', 'callback_data': 'ws_time:17:55'}, {'text': 'Custom', 'callback_data': 'ws_custom'}]]}
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Select time', 'reply_markup': keyboard})
        elif data == 'ws_custom' and user_id in self.pending and self.pending[user_id].get('setup_weather'):
            self.pending[user_id]['weather_time'] = True
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Enter time HH:MM'})
        elif data.startswith('ws_time:') and user_id in self.pending and self.pending[user_id].get('setup_weather'):
            time_str = data.split(':', 1)[1]
            self.add_weather_channel(self.pending[user_id]['channel'], time_str)
            del self.pending[user_id]
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Weather channel registered'})
        elif data.startswith('flowers_preview:'):
            if not self.is_superadmin(user_id):
                await self.api_request(
                    'answerCallbackQuery',
                    {
                        'callback_query_id': query['id'],
                        'text': 'Недостаточно прав',
                        'show_alert': True,
                    },
                )
                return
            action = data.split(':', 1)[1] if ':' in data else ''
            await self._handle_flowers_preview_callback(user_id, action, query)
            return
        elif data.startswith('asset_ch:') and user_id in self.pending and self.pending[user_id].get('set_assets'):
            cid = int(data.split(':')[1])
            self.set_asset_channel(cid)
            del self.pending[user_id]
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Asset channel set'})
        elif data.startswith('weather_ch:') and user_id in self.pending and self.pending[user_id].get('set_weather_assets'):
            cid = int(data.split(':')[1])
            self.set_weather_assets_channel(cid)
            del self.pending[user_id]
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Weather assets channel set'})
        elif data.startswith('recognition_ch:') and user_id in self.pending and self.pending[user_id].get('set_recognition'):
            cid = int(data.split(':')[1])
            self.set_recognition_channel(cid)
            del self.pending[user_id]
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Recognition channel set'})
        elif data.startswith('wrnow:') and self.is_superadmin(user_id):
            cid = int(data.split(':')[1])

            ok = await self.publish_weather(cid, None, record=False)
            msg = 'Posted' if ok else 'No asset to publish'
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': msg})

        elif data.startswith('wstop:') and self.is_superadmin(user_id):
            cid = int(data.split(':')[1])
            self.remove_weather_channel(cid)
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Channel removed'})
        elif data.startswith('wbtn_del:') and self.is_superadmin(user_id):
            _, cid, mid = data.split(':')
            chat_id = int(cid)
            msg_id = int(mid)
            row = self.db.execute(
                'SELECT base_markup FROM weather_link_posts WHERE chat_id=? AND message_id=?',
                (chat_id, msg_id),
            ).fetchone()
            markup = json.loads(row['base_markup']) if row and row['base_markup'] else {}
            await self.api_request(
                'editMessageReplyMarkup',
                {
                    'chat_id': chat_id,
                    'message_id': msg_id,
                    'reply_markup': markup,
                },
            )
            self.db.execute(
                'UPDATE weather_posts SET reply_markup=? WHERE chat_id=? AND message_id=?',
                (json.dumps(markup) if markup else None, chat_id, msg_id),
            )
            self.db.execute(
                'DELETE FROM weather_link_posts WHERE chat_id=? AND message_id=?',
                (chat_id, msg_id),
            )
            self.db.commit()
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Weather buttons removed'})
        elif data.startswith('wpost_del:') and self.is_superadmin(user_id):
            _, cid, mid = data.split(':')
            chat_id = int(cid)
            msg_id = int(mid)
            row = self.db.execute(
                'SELECT base_text, base_caption, reply_markup FROM weather_posts WHERE chat_id=? AND message_id=?',
                (chat_id, msg_id),
            ).fetchone()
            if row:
                markup = json.loads(row['reply_markup']) if row['reply_markup'] else None
                if row['base_caption'] is not None:
                    payload = {
                        'chat_id': chat_id,
                        'message_id': msg_id,
                        'caption': row['base_caption'],
                    }
                    method = 'editMessageCaption'
                else:
                    payload = {
                        'chat_id': chat_id,
                        'message_id': msg_id,
                        'text': row['base_text'] or '',
                    }
                    method = 'editMessageText'
                if markup:
                    payload['reply_markup'] = markup
                await self.api_request(method, payload)
                self.db.execute(
                    'DELETE FROM weather_posts WHERE chat_id=? AND message_id=?',
                    (chat_id, msg_id),
                )
                self.db.commit()
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Weather removed'})
        elif data.startswith('approve:') and self.is_superadmin(user_id):
            uid = int(data.split(':')[1])
            if self.approve_user(uid):
                cur = self.db.execute('SELECT username FROM users WHERE user_id=?', (uid,))
                row = cur.fetchone()
                uname = row['username'] if row else None
                await self.api_request('sendMessage', {
                    'chat_id': user_id,
                    'text': f'{self.format_user(uid, uname)} approved',
                    'parse_mode': 'Markdown'
                })
                await self.api_request('sendMessage', {'chat_id': uid, 'text': 'You are approved'})
            else:
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'User not in pending list'})
        elif data.startswith('reject:') and self.is_superadmin(user_id):
            uid = int(data.split(':')[1])
            if self.reject_user(uid):
                cur = self.db.execute('SELECT username FROM rejected_users WHERE user_id=?', (uid,))
                row = cur.fetchone()
                uname = row['username'] if row else None
                await self.api_request('sendMessage', {
                    'chat_id': user_id,
                    'text': f'{self.format_user(uid, uname)} rejected',
                    'parse_mode': 'Markdown'
                })
                await self.api_request('sendMessage', {'chat_id': uid, 'text': 'Your registration was rejected'})
            else:
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'User not in pending list'})
        elif data.startswith('cancel:') and self.is_authorized(user_id):
            sid = int(data.split(':')[1])
            self.remove_schedule(sid)
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': f'Schedule {sid} cancelled'})
        elif data.startswith('resch:') and self.is_authorized(user_id):
            sid = int(data.split(':')[1])
            self.pending[user_id] = {'reschedule_id': sid, 'await_time': True}
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Enter new time'})
        elif data.startswith('city_del:') and self.is_superadmin(user_id):
            cid = int(data.split(':')[1])
            self.db.execute('DELETE FROM cities WHERE id=?', (cid,))
            self.db.commit()
            await self.api_request('editMessageReplyMarkup', {
                'chat_id': query['message']['chat']['id'],
                'message_id': query['message']['message_id'],
                'reply_markup': {}
            })
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': f'City {cid} deleted'})
        elif data.startswith('sea_del:') and self.is_superadmin(user_id):
            sid = int(data.split(':')[1])
            self.db.execute('DELETE FROM seas WHERE id=?', (sid,))
            self.db.commit()
            await self.api_request('editMessageReplyMarkup', {
                'chat_id': query['message']['chat']['id'],
                'message_id': query['message']['message_id'],
                'reply_markup': {}
            })
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': f'Sea {sid} deleted'})

        elif data.startswith('amber_sea:') and user_id in self.pending and self.pending[user_id].get('amber_sea'):
            sid = int(data.split(':')[1])
            self.set_amber_sea(sid)
            del self.pending[user_id]
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Sea selected'})
            await self.show_amber_channels(user_id)
        elif data.startswith('amber_toggle:') and self.is_superadmin(user_id):
            cid = int(data.split(':')[1])
            if self.is_amber_channel(cid):
                self.db.execute('DELETE FROM amber_channels WHERE channel_id=?', (cid,))
                self.db.commit()
                enabled = False
            else:
                self.db.execute('INSERT OR IGNORE INTO amber_channels (channel_id) VALUES (?)', (cid,))
                self.db.commit()
                enabled = True
            row = self.db.execute('SELECT title FROM channels WHERE chat_id=?', (cid,)).fetchone()
            title = row['title'] if row else str(cid)
            icon = '✅' if enabled else '❌'
            btn = 'Выключить отправку' if enabled else 'Включить отправку'
            keyboard = {'inline_keyboard': [[{'text': btn, 'callback_data': f'amber_toggle:{cid}'}]]}
            await self.api_request('editMessageText', {
                'chat_id': query['message']['chat']['id'],
                'message_id': query['message']['message_id'],
                'text': f"{title} {icon}",
                'reply_markup': keyboard,
            })
        elif data == 'rubric_dashboard' and self.is_superadmin(user_id):
            await self._send_rubric_dashboard(user_id, message=query.get('message'))
        elif data.startswith('rubric_overview:') and self.is_superadmin(user_id):
            code = data.split(':', 1)[1]
            await self._send_rubric_overview(user_id, code, message=query.get('message'))
        elif data.startswith('rubric_publish_confirm:') and self.is_superadmin(user_id):
            parts = data.split(':', 2)
            if len(parts) < 3:
                await self.api_request(
                    'answerCallbackQuery',
                    {
                        'callback_query_id': query['id'],
                        'text': 'Некорректный запрос рубрики',
                        'show_alert': True,
                    },
                )
                return
            _, code, mode = parts
            if mode not in {'prod', 'test'}:
                await self.api_request(
                    'answerCallbackQuery',
                    {
                        'callback_query_id': query['id'],
                        'text': 'Неизвестный режим запуска',
                        'show_alert': True,
                    },
                )
                return
            self._set_rubric_pending_run(user_id, code, mode)
            await self._send_rubric_overview(user_id, code, message=query.get('message'))
            await self.api_request(
                'answerCallbackQuery',
                {
                    'callback_query_id': query['id'],
                    'text': 'Подтвердите запуск кнопкой ниже',
                },
            )
        elif data.startswith('rubric_publish_cancel:') and self.is_superadmin(user_id):
            code = data.split(':', 1)[1]
            self._clear_rubric_pending_run(user_id, code)
            await self._send_rubric_overview(user_id, code, message=query.get('message'))
        elif data.startswith('rubric_publish_execute:') and self.is_superadmin(user_id):
            parts = data.split(':', 2)
            if len(parts) < 3:
                await self.api_request(
                    'answerCallbackQuery',
                    {
                        'callback_query_id': query['id'],
                        'text': 'Некорректный запрос рубрики',
                        'show_alert': True,
                    },
                )
                return
            _, code, mode = parts
            is_test = mode == 'test'
            run_label = 'Тестовая' if is_test else 'Рабочая'
            try:
                job_id = self.enqueue_rubric(
                    code,
                    test=is_test,
                    initiator_id=user_id,
                )
            except Exception as exc:  # noqa: PERF203 - feedback path
                logging.exception('Failed to enqueue rubric %s (test=%s)', code, is_test)
                self._clear_rubric_pending_run(user_id, code)
                await self._send_rubric_overview(user_id, code, message=query.get('message'))
                await self.api_request(
                    'answerCallbackQuery',
                    {
                        'callback_query_id': query['id'],
                        'text': 'Ошибка запуска рубрики',
                        'show_alert': True,
                    },
                )
                reason = str(exc).strip() or 'неизвестная ошибка'
                await self.api_request(
                    'sendMessage',
                    {
                        'chat_id': user_id,
                        'text': (
                            f"⚠️ {run_label.lower()} публикация рубрики {code} не запущена.\n"
                            f"Причина: {reason}"
                        ),
                    },
                )
            else:
                logging.info(
                    'Enqueued %s publication for rubric %s (job_id=%s, user_id=%s)',
                    'test' if is_test else 'prod',
                    code,
                    job_id,
                    user_id,
                )
                self._clear_rubric_pending_run(user_id, code)
                await self._send_rubric_overview(user_id, code, message=query.get('message'))
                await self.api_request(
                    'answerCallbackQuery',
                    {
                        'callback_query_id': query['id'],
                        'text': 'Задача поставлена в очередь',
                    },
                )
                await self.api_request(
                    'sendMessage',
                    {
                        'chat_id': user_id,
                        'text': (
                            f"✅ {run_label} публикация рубрики {code} поставлена в очередь"
                            f" (задача #{job_id})."
                        ),
                    },
                )
        elif data.startswith('rubric_toggle:') and self.is_superadmin(user_id):
            code = data.split(':', 1)[1]
            self._clear_rubric_pending_run(user_id, code)
            rubric = self.data.get_rubric_by_code(code)
            if rubric:
                config = self._normalize_rubric_config(rubric.config)
                config['enabled'] = not config.get('enabled', True)
                self.data.save_rubric_config(code, config)
                await self._send_rubric_overview(user_id, code, message=query.get('message'))
        elif data.startswith('rubric_channel:') and self.is_superadmin(user_id):
            parts = data.split(':')
            if len(parts) >= 3:
                code = parts[1]
                self._clear_rubric_pending_run(user_id, code)
                target = parts[2]
                field = 'channel_id' if target == 'main' else 'test_channel_id'
                self.pending[user_id] = {
                    'rubric_input': {
                        'mode': 'channel_picker',
                        'code': code,
                        'field': field,
                        'message': query.get('message'),
                        'page': 0,
                        'search': '',
                        'search_mode': False,
                        'search_charset': 'rus',
                        'return_mode': None,
                    }
                }
                await self._edit_rubric_input_message(user_id)
        elif data.startswith('rubric_channel_page:') and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get('rubric_input')
            if state and state.get('mode') == 'channel_picker':
                try:
                    page = int(data.split(':', 1)[1])
                except ValueError:
                    page = 0
                state['page'] = max(page, 0)
                await self._edit_rubric_input_message(user_id)
        elif data == 'rubric_channel_search_toggle' and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get('rubric_input')
            if state and state.get('mode') == 'channel_picker':
                state['search_mode'] = not state.get('search_mode', False)
                if state['search_mode']:
                    state.setdefault('search_charset', 'rus')
                await self._edit_rubric_input_message(user_id)
        elif data.startswith('rubric_channel_search_charset:') and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get('rubric_input')
            if state and state.get('mode') == 'channel_picker':
                key = data.split(':', 1)[1]
                if key in CHANNEL_SEARCH_CHARSETS:
                    state['search_charset'] = key
                await self._edit_rubric_input_message(user_id)
        elif data.startswith('rubric_channel_search_add:') and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get('rubric_input')
            if state and state.get('mode') == 'channel_picker':
                hex_value = data.split(':', 1)[1]
                try:
                    char = bytes.fromhex(hex_value).decode('utf-8')
                except ValueError:
                    char = ''
                if char:
                    state['search'] = (state.get('search') or '') + char
                    state['page'] = 0
                await self._edit_rubric_input_message(user_id)
        elif data == 'rubric_channel_search_del' and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get('rubric_input')
            if state and state.get('mode') == 'channel_picker':
                current = state.get('search') or ''
                if current:
                    state['search'] = current[:-1]
                    state['page'] = 0
                await self._edit_rubric_input_message(user_id)
        elif data == 'rubric_channel_search_clear' and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get('rubric_input')
            if state and state.get('mode') == 'channel_picker':
                state['search'] = ''
                state['page'] = 0
                await self._edit_rubric_input_message(user_id)
        elif data == 'rubric_channel_search_done' and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get('rubric_input')
            if state and state.get('mode') == 'channel_picker':
                state['search_mode'] = False
                state['page'] = 0
                await self._edit_rubric_input_message(user_id)
        elif data.startswith('rubric_channel_set:') and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get('rubric_input')
            if state and state.get('mode') == 'channel_picker':
                value = data.split(':', 1)[1]
                channel_id: int | None
                try:
                    channel_id = int(value)
                except ValueError:
                    channel_id = None
                if state.get('return_mode') == 'schedule_wizard':
                    schedule = state.setdefault('schedule', {})
                    schedule['channel_id'] = channel_id
                    state['mode'] = 'schedule_wizard'
                    state['step'] = 'main'
                    state['search_mode'] = False
                    await self._edit_rubric_input_message(user_id)
                else:
                    code = state.get('code')
                    field = state.get('field')
                    if code and field:
                        config = self._normalize_rubric_config(
                            self.data.get_rubric_config(code) or {}
                        )
                        if channel_id is None:
                            config.pop(field, None)
                        else:
                            config[field] = channel_id
                        self.data.save_rubric_config(code, config)
                    message_obj = state.get('message')
                    del self.pending[user_id]
                    if code:
                        await self._send_rubric_overview(
                            user_id,
                            code,
                            message=message_obj,
                        )
        elif data == 'rubric_channel_clear' and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get('rubric_input')
            if state and state.get('mode') == 'channel_picker':
                if state.get('return_mode') == 'schedule_wizard':
                    schedule = state.setdefault('schedule', {})
                    schedule['channel_id'] = None
                    state['mode'] = 'schedule_wizard'
                    state['step'] = 'main'
                    state['search_mode'] = False
                    await self._edit_rubric_input_message(user_id)
                else:
                    code = state.get('code')
                    field = state.get('field')
                    if code and field:
                        config = self._normalize_rubric_config(
                            self.data.get_rubric_config(code) or {}
                        )
                        config.pop(field, None)
                        self.data.save_rubric_config(code, config)
                    message_obj = state.get('message')
                    del self.pending[user_id]
                    if code:
                        await self._send_rubric_overview(
                            user_id,
                            code,
                            message=message_obj,
                        )
        elif data == 'rubric_channel_cancel' and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get('rubric_input')
            if state:
                if state.get('return_mode') == 'schedule_wizard':
                    state['mode'] = 'schedule_wizard'
                    state['step'] = 'main'
                    state['search_mode'] = False
                    await self._edit_rubric_input_message(user_id)
                else:
                    code = state.get('code')
                    message_obj = state.get('message')
                    del self.pending[user_id]
                    if code:
                        await self._send_rubric_overview(
                            user_id,
                            code,
                            message=message_obj,
                        )
        elif data.startswith('rubric_sched_add:') and self.is_superadmin(user_id):
            code = data.split(':', 1)[1]
            self._clear_rubric_pending_run(user_id, code)
            rubric = self.data.get_rubric_by_code(code)
            if not rubric:
                return
            config = self._normalize_rubric_config(rubric.config)
            default_days = config.get('days')
            if isinstance(default_days, (list, tuple)):
                days_value: Any = list(default_days)
            elif default_days:
                days_value = default_days
            else:
                days_value = []
            schedule = {
                'time': None,
                'tz': config.get('tz') or TZ_OFFSET,
                'days': days_value,
                'channel_id': config.get('channel_id'),
                'enabled': True,
            }
            self.pending[user_id] = {
                'rubric_input': {
                    'mode': 'schedule_wizard',
                    'code': code,
                    'action': 'schedule_add',
                    'message': query.get('message'),
                    'schedule': schedule,
                    'step': 'main',
                    'search': '',
                    'search_mode': False,
                    'search_charset': 'rus',
                    'page': 0,
                }
            }
            await self._edit_rubric_input_message(user_id)
        elif data.startswith('rubric_sched_edit:') and self.is_superadmin(user_id):
            parts = data.split(':')
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
                schedules = config.get('schedules', [])
                if 0 <= idx < len(schedules):
                    schedule = dict(schedules[idx])
                    self.pending[user_id] = {
                        'rubric_input': {
                            'mode': 'schedule_wizard',
                            'code': code,
                            'action': 'schedule_edit',
                            'index': idx,
                            'message': query.get('message'),
                            'schedule': schedule,
                            'step': 'main',
                            'search': '',
                            'search_mode': False,
                            'search_charset': 'rus',
                            'page': 0,
                        }
                    }
                    await self._edit_rubric_input_message(user_id)
        elif data.startswith('rubric_sched_toggle:') and self.is_superadmin(user_id):
            parts = data.split(':')
            if len(parts) == 3:
                code, idx_str = parts[1], parts[2]
                self._clear_rubric_pending_run(user_id, code)
                rubric = self.data.get_rubric_by_code(code)
                if rubric:
                    config = self._normalize_rubric_config(rubric.config)
                    schedules = config.get('schedules', [])
                    try:
                        idx = int(idx_str)
                    except ValueError:
                        idx = -1
                    if 0 <= idx < len(schedules):
                        schedule = schedules[idx]
                        schedule['enabled'] = not schedule.get('enabled', True)
                        self.data.save_rubric_config(code, config)
                        await self._send_rubric_overview(user_id, code, message=query.get('message'))
        elif data == 'rubric_sched_time' and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get('rubric_input')
            if state and state.get('mode') == 'schedule_wizard':
                state['step'] = 'time_hours'
                await self._edit_rubric_input_message(user_id)
        elif data.startswith('rubric_sched_hour:') and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get('rubric_input')
            if state and state.get('mode') == 'schedule_wizard':
                try:
                    hour = int(data.split(':', 1)[1])
                except ValueError:
                    hour = 0
                state['temp_hour'] = max(0, min(hour, 23))
                state['step'] = 'time_minutes'
                await self._edit_rubric_input_message(user_id)
        elif data.startswith('rubric_sched_minute:') and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get('rubric_input')
            if state and state.get('mode') == 'schedule_wizard':
                try:
                    minute = int(data.split(':', 1)[1])
                except ValueError:
                    minute = 0
                hour = int(state.get('temp_hour', 0))
                minute = max(0, min(minute, 59))
                state.pop('temp_hour', None)
                schedule = state.setdefault('schedule', {})
                schedule['time'] = f"{hour:02d}:{minute:02d}"
                state['step'] = 'main'
                await self._edit_rubric_input_message(user_id)
        elif data == 'rubric_sched_time_back' and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get('rubric_input')
            if state and state.get('mode') == 'schedule_wizard':
                state['step'] = 'time_hours'
                await self._edit_rubric_input_message(user_id)
        elif data == 'rubric_sched_time_cancel' and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get('rubric_input')
            if state and state.get('mode') == 'schedule_wizard':
                state.pop('temp_hour', None)
                state['step'] = 'main'
                await self._edit_rubric_input_message(user_id)
        elif data == 'rubric_sched_days' and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get('rubric_input')
            if state and state.get('mode') == 'schedule_wizard':
                state['step'] = 'days'
                await self._edit_rubric_input_message(user_id)
        elif data.startswith('rubric_sched_day:') and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get('rubric_input')
            if state and state.get('mode') == 'schedule_wizard':
                day = data.split(':', 1)[1]
                schedule = state.setdefault('schedule', {})
                days = schedule.get('days')
                if not isinstance(days, list):
                    days = list(days) if isinstance(days, tuple) else []
                if day in days:
                    days.remove(day)
                else:
                    days.append(day)
                schedule['days'] = days
                await self._edit_rubric_input_message(user_id)
        elif data == 'rubric_sched_days_all' and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get('rubric_input')
            if state and state.get('mode') == 'schedule_wizard':
                schedule = state.setdefault('schedule', {})
                schedule['days'] = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
                await self._edit_rubric_input_message(user_id)
        elif data == 'rubric_sched_days_clear' and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get('rubric_input')
            if state and state.get('mode') == 'schedule_wizard':
                schedule = state.setdefault('schedule', {})
                schedule['days'] = []
                await self._edit_rubric_input_message(user_id)
        elif data == 'rubric_sched_days_done' and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get('rubric_input')
            if state and state.get('mode') == 'schedule_wizard':
                state['step'] = 'main'
                await self._edit_rubric_input_message(user_id)
        elif data == 'rubric_sched_toggle_enabled' and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get('rubric_input')
            if state and state.get('mode') == 'schedule_wizard':
                schedule = state.setdefault('schedule', {})
                schedule['enabled'] = not schedule.get('enabled', True)
                await self._edit_rubric_input_message(user_id)
        elif data == 'rubric_sched_channel' and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get('rubric_input')
            if state and state.get('mode') == 'schedule_wizard':
                state['mode'] = 'channel_picker'
                state['return_mode'] = 'schedule_wizard'
                state['field'] = 'channel_id'
                state['page'] = 0
                state['search_mode'] = False
                await self._edit_rubric_input_message(user_id)
        elif data == 'rubric_sched_save' and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get('rubric_input')
            if state and state.get('mode') == 'schedule_wizard':
                code = state.get('code')
                if code:
                    self._clear_rubric_pending_run(user_id, code)
                schedule_data = dict(state.get('schedule') or {})
                if isinstance(schedule_data.get('days'), tuple):
                    schedule_data['days'] = list(schedule_data['days'])
                action = state.get('action')
                message_obj = state.get('message')
                try:
                    if action == 'schedule_edit':
                        index = int(state.get('index', 0))
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
        elif data == 'rubric_sched_cancel' and self.is_superadmin(user_id):
            state = self.pending.get(user_id, {}).get('rubric_input')
            if state and state.get('mode') == 'schedule_wizard':
                code = state.get('code')
                if code:
                    self._clear_rubric_pending_run(user_id, code)
                message_obj = state.get('message')
                del self.pending[user_id]
                if code:
                    await self._send_rubric_overview(
                        user_id,
                        code,
                        message=message_obj,
                    )
        elif data.startswith('rubric_sched_del:') and self.is_superadmin(user_id):
            parts = data.split(':')
            if len(parts) == 3:
                code, idx_str = parts[1], parts[2]
                self._clear_rubric_pending_run(user_id, code)
                try:
                    idx = int(idx_str)
                except ValueError:
                    idx = -1
                if idx >= 0 and self.data.remove_rubric_schedule(code, idx):
                    await self._send_rubric_overview(user_id, code, message=query.get('message'))
                else:
                    await self.api_request('sendMessage', {
                        'chat_id': user_id,
                        'text': 'Не удалось удалить расписание',
                    })
        await self.api_request('answerCallbackQuery', {'callback_query_id': query['id']})


    async def process_due(self):
        """Publish due scheduled messages."""
        now = datetime.utcnow().isoformat()
        logging.info("Scheduler check at %s", now)
        cur = self.db.execute(
            'SELECT * FROM schedule WHERE sent=0 AND publish_time<=? ORDER BY publish_time',
            (now,),
        )
        rows = cur.fetchall()
        logging.info("Due ids: %s", [r['id'] for r in rows])
        for row in rows:
            try:
                resp = await self.api_request(
                    'forwardMessage',
                    {
                        'chat_id': row['target_chat_id'],
                        'from_chat_id': row['from_chat_id'],
                        'message_id': row['message_id'],
                    },
                )
                ok = resp.get('ok', False)
                if not ok and resp.get('error_code') == 400 and 'not' in resp.get('description', '').lower():
                    resp = await self.api_request(
                        'copyMessage',
                        {
                            'chat_id': row['target_chat_id'],
                            'from_chat_id': row['from_chat_id'],
                            'message_id': row['message_id'],
                        },
                    )
                    ok = resp.get('ok', False)
                if ok:
                    self.db.execute(
                        'UPDATE schedule SET sent=1, sent_at=? WHERE id=?',
                        (datetime.utcnow().isoformat(), row['id']),
                    )
                    self.db.commit()
                    logging.info('Published schedule %s', row['id'])
                else:
                    logging.error('Failed to publish %s: %s', row['id'], resp)
            except Exception:
                logging.exception('Error publishing schedule %s', row['id'])

    async def process_weather_channels(self):
        now_utc = datetime.utcnow()
        jobs = self.data.due_weather_jobs(now_utc)
        for job in jobs:
            try:
                ok = await self.publish_weather(job.channel_id, None)
                if ok:
                    next_run = self.next_weather_run(
                        job.post_time, TZ_OFFSET, reference=now_utc
                    )
                    self.data.mark_weather_job_run(job.id, next_run)
                else:
                    self.data.record_weather_job_failure(
                        job.id, "publish failed"
                    )
            except Exception:
                logging.exception(
                    "Failed to publish weather for %s", job.channel_id
                )

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
                channel_id = schedule.get("channel_id") or config.get("channel_id")
                if not channel_id:
                    continue
                time_str = schedule.get("time")
                if not time_str:
                    continue
                tz_value = schedule.get("tz") or config.get("tz") or TZ_OFFSET
                days = schedule.get("days") or config.get("days")
                key = schedule.get("key") or f"{channel_id}:{idx}:{time_str}"
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
                payload = {
                    "rubric_code": rubric.code,
                    "channel_id": channel_id,
                    "schedule_key": key,
                    "scheduled_at": next_run.isoformat(),
                    "tz_offset": tz_value,
                }
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
        return (
            f"#{index + 1}: {time_str} (tz {tz_value}) → {channel_repr}, дни: {days_repr} {flag}{suffix}"
        )

    def _get_channel_title(self, chat_id: int | None) -> str:
        if chat_id is None:
            return "—"
        row = self.db.execute(
            "SELECT title FROM channels WHERE chat_id=?",
            (chat_id,),
        ).fetchone()
        title = row["title"] if row and row["title"] else None
        return title or str(chat_id)

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

    def _get_rubric_input_message_target(
        self, state: dict[str, Any]
    ) -> tuple[int, int] | None:
        message = state.get("message")
        if not message:
            return None
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        message_id = message.get("message_id")
        if chat_id is None or message_id is None:
            return None
        return chat_id, message_id

    def _render_channel_search_keyboard(
        self, state: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
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

    def _render_channel_picker(
        self, state: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
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
                config = self._normalize_rubric_config(
                    self.data.get_rubric_config(code) or {}
                )
                current_id = config.get("channel_id")
        else:
            config = self._normalize_rubric_config(
                self.data.get_rubric_config(code) or {}
            )
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
            nav_row.append(
                {"text": "◀️", "callback_data": f"rubric_channel_page:{page - 1}"}
            )
        if page < max_page:
            nav_row.append(
                {"text": "▶️", "callback_data": f"rubric_channel_page:{page + 1}"}
            )
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
        keyboard_rows.append(
            [{"text": cancel_text, "callback_data": "rubric_channel_cancel"}]
        )
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

    def _render_schedule_wizard(
        self, state: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        code = state.get("code") or ""
        schedule = state.setdefault("schedule", {})
        config = self._normalize_rubric_config(
            self.data.get_rubric_config(code) or {}
        )
        schedule.setdefault("tz", schedule.get("tz") or config.get("tz") or TZ_OFFSET)
        if schedule.get("days") is None and config.get("days") is not None:
            fallback_days = config.get("days")
            schedule["days"] = list(fallback_days) if isinstance(fallback_days, (list, tuple)) else fallback_days
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
                await self.api_request("editMessageText", payload)
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
        lines = [title_line, f"Статус: {flag}", channel_line, test_line, tz_line, days_line, "Расписания:"]
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
        keyboard_rows.append([
            {"text": toggle_text, "callback_data": f"rubric_toggle:{rubric.code}"},
        ])
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
        if message:
            chat_id = message.get("chat", {}).get("id", user_id)
            message_id = message.get("message_id")
            payload.update({"chat_id": chat_id, "message_id": message_id})
            await self.api_request("editMessageText", payload)
            if chat_id is not None and message_id is not None:
                self._remember_rubric_overview(
                    user_id, code, chat_id=chat_id, message_id=message_id
                )
        else:
            payload["chat_id"] = user_id
            response = await self.api_request("sendMessage", payload)
            if response and response.get("ok"):
                result = response.get("result")
                if isinstance(result, dict):
                    chat = result.get("chat") or {}
                    chat_id = chat.get("id", user_id)
                    message_id = result.get("message_id")
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
            target = config.get("test_channel_id") if test else config.get("channel_id")
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
        channel_id = payload.get("channel_id")
        test_mode = bool(payload.get("test"))
        schedule_key = payload.get("schedule_key")
        scheduled_at = payload.get("scheduled_at")
        success = await self.publish_rubric(
            code,
            channel_id=channel_id,
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
        if target is None:
            target = config.get("test_channel_id") if test else config.get("channel_id")
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
            return timezone.utc
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
            local_ref = reference.replace(tzinfo=timezone.utc).astimezone(tzinfo)
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
        return candidate.astimezone(timezone.utc).replace(tzinfo=None)

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

    async def _prepare_flowers_drop(
        self,
        rubric: Rubric,
        *,
        job: Job | None = None,
        instructions: str | None = None,
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
        if len(assets) < min_count:
            logging.warning(
                "Not enough assets for flowers rubric: have %s, need %s",
                len(assets),
                min_count,
            )
            return None
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
        weather = self._build_flowers_weather_block()
        greeting, hashtags = await self._generate_flowers_copy(
            rubric,
            cities,
            len(assets),
            job=job,
            instructions=instructions,
            weather=weather,
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
            weather,
            conversion_map,
        )

    async def _send_flowers_media_bundle(
        self,
        *,
        chat_id: int,
        assets: list[Asset],
        file_ids: list[str],
        asset_kinds: list[str],
        caption: str | None,
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
            asset.kind = "photo"
            asset.file_id = new_file_id
            asset.file_unique_id = photo_meta.get("file_unique_id")
            asset.mime_type = photo_meta.get("mime_type")
            asset.file_size = photo_meta.get("file_size")
            asset.width = photo_meta.get("width")
            asset.height = photo_meta.get("height")
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
        weather: dict[str, Any] | None = None,
    ) -> tuple[str, list[str]]:
        hashtag_list = self._prepare_hashtags(hashtags)
        caption_parts: list[str] = []
        weather_line: str | None = None
        if weather:
            raw_line = weather.get("caption_line")
            if isinstance(raw_line, str) and raw_line.strip():
                weather_line = raw_line.strip()
            else:
                positive_summary = str(weather.get("positive_summary") or "").strip()
                locations = weather.get("locations") or []
                location_text = ", ".join(str(loc) for loc in locations if str(loc).strip())
                if not location_text and cities:
                    location_text = ", ".join(cities)
                if not location_text:
                    location_text = "Калининград, побережье"
                if not positive_summary:
                    positive_summary = "Погода дарит приятное настроение."
                positive_summary = positive_summary.rstrip(".")
                weather_line = f"{positive_summary} • {location_text}"
        elif cities:
            weather_line = "Погода дарит приятное настроение • " + ", ".join(cities)
        else:
            weather_line = "Погода дарит приятное настроение • Калининград, побережье"
        if weather_line:
            caption_parts.append(weather_line)
        if greeting:
            caption_parts.append(greeting.strip())
        if cities:
            caption_parts.append("Города съёмки: " + ", ".join(cities))
        if hashtag_list:
            caption_parts.append(" ".join(hashtag_list))
        caption = "\n\n".join(part for part in caption_parts if part)
        return caption, hashtag_list

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
            weather,
            conversion_map,
        ) = prepared
        caption, hashtag_list = self._build_flowers_caption(
            greeting,
            cities,
            hashtags,
            weather,
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
                weather=weather,
                caption=caption,
                prepared_hashtags=hashtag_list,
                instructions=instructions,
            )
            return True
        response, _ = await self._send_flowers_media_bundle(
            chat_id=channel_id,
            assets=assets,
            file_ids=file_ids,
            asset_kinds=asset_kinds,
            caption=caption,
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
        metadata = {
            "rubric_code": rubric.code,
            "asset_ids": asset_ids,
            "test": test,
            "cities": cities,
            "greeting": greeting,
            "hashtags": hashtag_list,
            "weather": weather,
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

    def _resolve_flowers_target(
        self, state: dict[str, Any], *, to_test: bool
    ) -> int | None:
        key = "test_channel_id" if to_test else "channel_id"
        value = state.get(key)
        if isinstance(value, int):
            return value
        default_type = state.get("default_channel_type")
        default_value = state.get("default_channel_id")
        if isinstance(default_value, int) and (
            (to_test and default_type == "test")
            or (not to_test and default_type == "main")
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
        ]
        send_row: list[dict[str, Any]] = []
        if self._resolve_flowers_target(state, to_test=True) is not None:
            send_row.append(
                {"text": "🧪 В тест", "callback_data": "flowers_preview:send_test"}
            )
        if self._resolve_flowers_target(state, to_test=False) is not None:
            send_row.append(
                {"text": "📣 В канал", "callback_data": "flowers_preview:send_main"}
            )
        if send_row:
            rows.append(send_row)
        rows.append([
            {"text": "✖️ Отмена", "callback_data": "flowers_preview:cancel"}
        ])
        return {"inline_keyboard": rows}

    def _render_flowers_preview_text(self, state: dict[str, Any]) -> str:
        parts: list[str] = []
        caption = str(state.get("caption") or "").strip()
        if caption:
            parts.append("Подпись на медиа показана выше.")
        else:
            parts.append("Подпись пока не сгенерирована.")
        instructions = str(state.get("instructions") or "").strip()
        if instructions:
            parts.append(f"Инструкции оператора:\n{instructions}")
        channels: list[str] = []
        main_target = self._resolve_flowers_target(state, to_test=False)
        if main_target is not None:
            channels.append(f"📣 {main_target}")
        test_target = self._resolve_flowers_target(state, to_test=True)
        if test_target is not None:
            channels.append(f"🧪 {test_target}")
        if channels:
            parts.append("Доступные каналы: " + ", ".join(channels))
        parts.append("Выберите действие на клавиатуре ниже.")
        return "\n\n".join(parts)

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
        asset_ids: list[int],
        file_ids: list[str],
        asset_kinds: list[str],
        conversion_map: dict[int, dict[str, Any]],
        cities: list[str],
        greeting: str,
        hashtags: list[str],
        weather: dict[str, Any] | None,
        caption: str,
        prepared_hashtags: list[str],
        instructions: str | None,
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
            "weather": weather,
            "prepared_hashtags": prepared_hashtags,
            "caption": caption,
            "instructions": (instructions or "").strip(),
            "preview_chat_id": initiator_id,
            "media_message_ids": [],
            "caption_message_id": None,
            "instruction_prompt_id": previous_state.get("instruction_prompt_id") if previous_state else None,
            "awaiting_instruction": previous_state.get("awaiting_instruction") if previous_state else False,
            "channel_id": _to_int(main_channel_raw),
            "test_channel_id": _to_int(test_channel_raw),
            "default_channel_id": int(default_channel),
            "default_channel_type": "test" if test_requested else "main",
        }
        normalized_caption = str(caption or "")
        response, remaining_conversion = await self._send_flowers_media_bundle(
            chat_id=initiator_id,
            assets=assets,
            file_ids=file_ids,
            asset_kinds=asset_kinds,
            caption=normalized_caption,
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
        caption: str,
        greeting: str,
        hashtags: list[str],
        prepared_hashtags: list[str],
    ) -> None:
        state["caption"] = caption
        state["greeting"] = greeting
        state["hashtags"] = hashtags
        state["prepared_hashtags"] = prepared_hashtags
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
        caption = str(state.get("caption") or "")
        conversion_map = dict(state.get("conversion_map") or {})
        assets_list = list(state.get("assets") or [])
        response, _ = await self._send_flowers_media_bundle(
            chat_id=channel_id,
            assets=assets_list,
            file_ids=file_ids,
            asset_kinds=asset_kinds,
            caption=caption,
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
                "weather": state.get("weather"),
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
        await self.api_request(
            "sendMessage",
            {
                "chat_id": user_id,
                "text": "Публикация отправлена.",
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
        if action in {
            "regen_photos",
            "regen_caption",
            "instruction",
            "send_test",
            "send_main",
            "cancel",
        } and not state:
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
            rubric = (
                self.data.get_rubric_by_code(rubric_code) if rubric_code else None
            )
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
            prepared = await self._prepare_flowers_drop(
                rubric,
                instructions=state.get("instructions"),
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
                weather,
                conversion_map,
            ) = prepared
            caption, prepared_hashtags = self._build_flowers_caption(
                greeting,
                cities,
                hashtags,
                weather,
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
                weather=weather,
                caption=caption,
                prepared_hashtags=prepared_hashtags,
                instructions=state.get("instructions"),
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
            rubric = (
                self.data.get_rubric_by_code(rubric_code) if rubric_code else None
            )
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
            cities = list(state.get("cities") or [])
            asset_count = len(state.get("assets") or [])
            greeting, hashtags = await self._generate_flowers_copy(
                rubric,
                cities,
                asset_count,
                instructions=state.get("instructions"),
                weather=state.get("weather"),
            )
            caption, prepared_hashtags = self._build_flowers_caption(
                greeting,
                cities,
                hashtags,
                state.get("weather"),
            )
            await self._update_flowers_preview_caption_state(
                state,
                caption=caption,
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

    async def _generate_flowers_copy(
        self,
        rubric: Rubric,
        cities: list[str],
        asset_count: int,
        *,
        job: Job | None = None,
        instructions: str | None = None,
        weather: dict[str, Any] | None = None,
    ) -> tuple[str, list[str]]:
        if not self.openai or not self.openai.api_key:
            return self._default_flowers_greeting(cities), self._default_hashtags("flowers")
        prompt_parts = [
            "Составь короткий приветственный текст для утреннего поста с фото цветов.",
            "Пиши дружелюбно и на русском языке.",
            f"Фотографий: {asset_count}.",
        ]
        if cities:
            prompt_parts.append(
                "Упомяни города: " + ", ".join(cities)
            )
        if instructions:
            prompt_parts.append(f"Дополнительные пожелания: {instructions}")
        weather_positive = "Погода дарит приятное настроение."
        plan_lines: list[str]
        if weather:
            positive_summary_raw = weather.get("positive_summary")
            if isinstance(positive_summary_raw, str) and positive_summary_raw.strip():
                weather_positive = positive_summary_raw.strip()
            plan_lines = [
                str(line).strip()
                for line in weather.get("plan_lines", [])
                if str(line).strip()
            ]
            if not plan_lines:
                plan_lines = [
                    "Текущая погода в Калининграде: утро приятное.",
                    "Море у побережья: атмосфера дружелюбная.",
                    "Вывод о переменах относительно вчера: погода остаётся приятной.",
                    "Позитивные формулировки: Погода дарит приятное настроение.",
                ]
            examples = weather.get("positive_phrases") or []
            positive_examples = " ".join(
                str(example).strip()
                for example in examples
                if str(example).strip()
            )
            if positive_examples:
                prompt_parts.append(
                    "Позитивные формулировки для вдохновения: " + positive_examples
                )
        else:
            plan_lines = [
                "Текущая погода в Калининграде: утро приятное.",
                "Море у побережья: атмосфера дружелюбная.",
                "Вывод о переменах относительно вчера: погода остаётся приятной.",
                "Позитивные формулировки: Погода дарит приятное настроение.",
            ]
        prompt_parts.append(f"Главный вывод о погоде: {weather_positive}")
        plan_text = "\n".join(
            f"{idx}. {line}" for idx, line in enumerate(plan_lines, start=1)
        )
        prompt_parts.append(
            "Структурированный план (не перечисляй номера в ответе, но следуй логике):\n"
            + plan_text
        )
        prompt = " ".join(prompt_parts)
        schema = {
            "type": "object",
            "properties": {
                "greeting": {"type": "string"},
                "hashtags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "required": ["greeting", "hashtags"],
        }
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
                    system_prompt=(
                        "Ты редактор телеграм-канала про погоду и уют. "
                        "Создавай дружелюбные тексты."
                    ),
                    user_prompt=prompt,
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
            if self._is_duplicate_rubric_copy("flowers", "greeting", greeting, hashtags):
                logging.info(
                    "Получен повторяющийся текст для рубрики flowers, пробуем снова (%s/%s)",
                    attempt,
                    attempts,
                )
                continue
            return greeting, hashtags
        logging.warning("Не удалось получить новый текст для рубрики flowers, используем запасной вариант")
        return self._default_flowers_greeting(cities), self._default_hashtags("flowers")

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
        normalized_tags = sorted({str(tag).lstrip("#").lower() for tag in hashtags if str(tag).strip()})
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
                    logging.warning(
                        "Asset %s missing source file for guess_arch overlay", asset.id
                    )
                    for created in overlay_paths:
                        self._remove_file(created)
                    return False
                source_files.append((asset.id, source_path, should_cleanup))
                path = self._overlay_number(
                    asset, idx, config, source_path=source_path
                )
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
            "Подготовь подпись на русском языке для конкурса \"Угадай архитектуру\". "
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

    def _default_hashtags(self, code: str) -> list[str]:
        mapping = {
            "flowers": ["#котопогода", "#цветы"],
            "guess_arch": ["#угадайархитектуру", "#калининград", "#котопогода"],
        }
        return mapping.get(code, ["#котопогода"])

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
                logging.exception(
                    "Failed to cleanup asset %s after publishing", asset_id
                )
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

    def _compatible_photo_weather_classes(
        self, actual_class: str | None
    ) -> set[str] | None:
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
            (("закат", "sunset", "рассвет", "sunrise", "golden hour"), "night"),
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

    def _format_temperature(self, value: Any, *, decimals: int = 0) -> str | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if decimals:
            return f"{number:.{decimals}f}\u00B0C"
        return f"{int(round(number))}\u00B0C"

    def _format_wind(self, value: Any) -> str | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return f"{int(round(number))} м/с"

    def _positive_temperature_phrase(
        self,
        city_name: str,
        current: Any,
        previous: Any,
    ) -> str:
        try:
            current_val = float(current)
        except (TypeError, ValueError):
            current_val = None
        try:
            previous_val = float(previous)
        except (TypeError, ValueError):
            previous_val = None
        if current_val is None:
            return f"Температура в {city_name} остаётся комфортной."
        if previous_val is None:
            return (
                f"Температура в {city_name} около {int(round(current_val))}\u00B0 — хорошее утро."
            )
        diff = current_val - previous_val
        amount = max(1, abs(int(round(diff))))
        if diff > 1:
            return f"В {city_name} теплее на {amount}\u00B0 — отличный повод прогуляться."
        if diff < -1:
            return (
                f"Свежий воздух в {city_name} бодрит (на {amount}\u00B0 прохладнее)."
            )
        return f"Температура в {city_name} держится комфортной."

    def _positive_wind_phrase(self, current: Any, previous: Any) -> str:
        try:
            current_val = float(current)
        except (TypeError, ValueError):
            current_val = None
        try:
            previous_val = float(previous)
        except (TypeError, ValueError):
            previous_val = None
        if current_val is None:
            return "Ветер остаётся спокойным."
        if previous_val is None:
            return "Ветер мягкий и не мешает прогулкам."
        diff = current_val - previous_val
        if diff < -1:
            return "Ветер стал мягче и уютнее."
        if diff > 1:
            return "Лёгкий ветерок добавляет энергии."
        return "Ветер остаётся спокойным."

    def _flowers_change_summary(
        self,
        city_name: str,
        current_temp: Any,
        previous_temp: Any,
        current_wind: Any,
        previous_wind: Any,
    ) -> str:
        try:
            temp_now = float(current_temp)
        except (TypeError, ValueError):
            temp_now = None
        try:
            temp_prev = float(previous_temp)
        except (TypeError, ValueError):
            temp_prev = None
        try:
            wind_now = float(current_wind)
        except (TypeError, ValueError):
            wind_now = None
        try:
            wind_prev = float(previous_wind)
        except (TypeError, ValueError):
            wind_prev = None

        parts: list[str] = []
        if temp_now is not None and temp_prev is not None:
            diff_temp = temp_now - temp_prev
            amount = max(1, abs(int(round(diff_temp))))
            if diff_temp > 1:
                parts.append(f"теплее на {amount}\u00B0")
            elif diff_temp < -1:
                parts.append(f"свежее на {amount}\u00B0 — бодрит")
            else:
                parts.append("температура осталась комфортной")
        else:
            parts.append("температура ощущается комфортной")

        if wind_now is not None and wind_prev is not None:
            diff_wind = wind_now - wind_prev
            if diff_wind < -1:
                parts.append("ветер заметно спокойнее")
            elif diff_wind > 1:
                parts.append("ветер чуть бодрее и освежает")
            else:
                parts.append("ветер почти не изменился")
        else:
            parts.append("ветер мягкий")

        joined = ", ".join(parts)
        return f"В {city_name} {joined}."

    def _describe_wave_height(self, value: Any) -> str | None:
        if value is None:
            return None
        try:
            wave = float(value)
        except (TypeError, ValueError):
            return None
        if wave < 0.5:
            return "волны тихие"
        if wave < 1.5:
            return "волны мягкие"
        return "волны мощные"

    def _sea_positive_phrase(self, name: str, wave: Any) -> str:
        try:
            wave_val = float(wave)
        except (TypeError, ValueError):
            wave_val = None
        if wave_val is None:
            return f"Побережье {name} радует спокойствием."
        if wave_val < 0.5:
            return f"Побережье {name} встречает спокойными волнами."
        if wave_val < 1.5:
            return f"У {name} лёгкие волны — приятно прогуляться вдоль берега."
        return f"Море у {name} мощное и завораживающее."

    def _get_flowers_kaliningrad_weather(self) -> dict[str, Any]:
        default_name = "Калининград"
        row = self.db.execute(
            "SELECT id, name FROM cities WHERE lower(name)=lower(?)",
            ("Kaliningrad",),
        ).fetchone()
        city_id = row["id"] if row else None
        city_name = row["name"] if row and row["name"] else default_name
        info: dict[str, Any] = {
            "name": city_name,
            "location": city_name,
        }
        if not city_id:
            info["summary"] = None
            info["display"] = f"{city_name}: данные уточняются"
            info["positive_phrases"] = [
                f"Погода в {city_name} остаётся уютной."
            ]
            info["change_summary"] = f"В {city_name} погода остаётся приятной."
            info["plan_line"] = (
                f"Текущая погода в {city_name}: данных нет, но утро обещает быть приятным."
            )
            return info
        weather_row = self.db.execute(
            """
            SELECT temperature, weather_code, wind_speed, is_day
            FROM weather_cache_hour
            WHERE city_id=?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (city_id,),
        ).fetchone()
        if not weather_row:
            info["summary"] = None
            info["display"] = f"{city_name}: данные уточняются"
            info["positive_phrases"] = [
                f"Погода в {city_name} остаётся уютной."
            ]
            info["change_summary"] = f"В {city_name} погода остаётся приятной."
            info["plan_line"] = (
                f"Текущая погода в {city_name}: данных нет, но утро обещает быть приятным."
            )
            return info
        temp = weather_row["temperature"]
        wind = weather_row["wind_speed"]
        code = weather_row["weather_code"]
        is_day = weather_row["is_day"]
        temp_str = self._format_temperature(temp)
        wind_str = self._format_wind(wind)
        parts: list[str] = []
        if temp_str:
            parts.append(temp_str)
        if wind_str:
            parts.append(f"ветер {wind_str}")
        summary = ", ".join(parts) if parts else None
        emoji = weather_emoji(int(code), is_day) if code is not None else ""
        display_parts: list[str] = []
        if emoji:
            display_parts.append(emoji)
        if summary:
            display_parts.append(f"{city_name}: {summary}")
        else:
            display_parts.append(city_name)
        info["summary"] = summary
        info["display"] = " ".join(display_parts).strip()
        yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
        previous_row = self.db.execute(
            """
            SELECT temperature, wind_speed
            FROM weather_cache_day
            WHERE city_id=? AND day=?
            """,
            (city_id, yesterday.isoformat()),
        ).fetchone()
        prev_temp = previous_row["temperature"] if previous_row else None
        prev_wind = previous_row["wind_speed"] if previous_row else None
        temp_phrase = self._positive_temperature_phrase(city_name, temp, prev_temp)
        wind_phrase = self._positive_wind_phrase(wind, prev_wind)
        info["positive_phrases"] = [temp_phrase, wind_phrase]
        info["change_summary"] = self._flowers_change_summary(
            city_name, temp, prev_temp, wind, prev_wind
        )
        if summary:
            info["plan_line"] = f"Текущая погода в {city_name}: {summary}."
        else:
            info["plan_line"] = (
                f"Текущая погода в {city_name}: данные появятся позже, но утро всё равно приятное."
            )
        return info

    def _get_flowers_coast_weather(self) -> dict[str, Any]:
        default_name = "Балтийское побережье"
        row = self.db.execute(
            """
            SELECT s.id, s.name, c.current, c.wave
            FROM seas AS s
            LEFT JOIN sea_cache AS c ON c.sea_id = s.id
            ORDER BY COALESCE(c.updated, '') DESC, s.id ASC
            LIMIT 1
            """
        ).fetchone()
        name = row["name"] if row and row["name"] else default_name
        temp = row["current"] if row else None
        wave = row["wave"] if row else None
        temp_str = self._format_temperature(temp, decimals=1)
        wave_desc = self._describe_wave_height(wave)
        parts: list[str] = []
        if temp_str:
            parts.append(f"вода {temp_str}")
        if wave_desc:
            parts.append(wave_desc)
        summary = ", ".join(parts) if parts else None
        info: dict[str, Any] = {
            "name": name,
            "location": name,
            "summary": summary,
            "display": f"{name}: {summary}" if summary else f"{name}: данные уточняются",
            "positive_phrases": [self._sea_positive_phrase(name, wave)],
        }
        if summary:
            info["plan_line"] = f"Море у {name}: {summary}."
        else:
            info["plan_line"] = (
                f"Море у {name}: данные появятся позже, но берег остаётся уютным."
            )
        return info

    def _build_flowers_weather_block(self) -> dict[str, Any]:
        city_info = self._get_flowers_kaliningrad_weather()
        sea_info = self._get_flowers_coast_weather()
        plan_lines: list[str] = []
        city_plan = city_info.get("plan_line")
        if city_plan:
            plan_lines.append(city_plan)
        sea_plan = sea_info.get("plan_line")
        if sea_plan:
            plan_lines.append(sea_plan)
        change_summary = city_info.get("change_summary") or "Погода остаётся приятной."
        plan_lines.append(f"Вывод о переменах относительно вчера: {change_summary}")
        positive_phrases: list[str] = []
        for phrases in (
            city_info.get("positive_phrases"),
            sea_info.get("positive_phrases"),
        ):
            if not phrases:
                continue
            for phrase in phrases:
                text = str(phrase or "").strip()
                if text:
                    positive_phrases.append(text if text.endswith(".") else f"{text}.")
        if not positive_phrases:
            positive_phrases = ["Погода дарит приятное настроение."]
        plan_lines.append(
            "Позитивные формулировки: " + " ".join(positive_phrases)
        )
        positive_summary = positive_phrases[0].rstrip(".") + "."
        locations = [
            city_info.get("location") or city_info.get("name"),
            sea_info.get("location") or sea_info.get("name"),
        ]
        clean_locations = [str(loc) for loc in locations if loc]
        if clean_locations:
            caption_line = f"{positive_summary.rstrip('.')} • {', '.join(clean_locations)}"
        else:
            caption_line = positive_summary
        return {
            "city": city_info,
            "sea": sea_info,
            "change_summary": change_summary,
            "positive_phrases": positive_phrases,
            "positive_summary": positive_summary,
            "plan_lines": plan_lines,
            "caption_line": caption_line,
            "locations": clean_locations,
        }

    def _get_city_weather_info(self, city_name: str) -> tuple[str | None, str | None]:
        row = self.db.execute(
            "SELECT id, name FROM cities WHERE lower(name)=lower(?)",
            (city_name,),
        ).fetchone()
        if not row:
            return None, None
        weather_row = self.db.execute(
            """
            SELECT temperature, weather_code, wind_speed
            FROM weather_cache_hour
            WHERE city_id=?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (row["id"],),
        ).fetchone()
        if not weather_row:
            return None, None
        temp = weather_row["temperature"]
        code = weather_row["weather_code"]
        wind = weather_row["wind_speed"]
        weather_class = self._classify_weather_code(int(code) if code is not None else None)
        emoji = weather_emoji(int(code), None) if code is not None else ""
        parts: list[str] = []
        if temp is not None:
            parts.append(f"{int(round(temp))}°C")
        if wind is not None:
            parts.append(f"ветер {int(round(wind))} м/с")
        summary = ", ".join(parts)
        city_display = row["name"]
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


    async def schedule_loop(self):
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
                    logging.exception('Weather collection failed')
                await asyncio.sleep(SCHED_INTERVAL_SEC)
        except asyncio.CancelledError:
            pass


async def ensure_webhook(bot: Bot, base_url: str):
    expected = base_url.rstrip('/') + '/webhook'
    info = await bot.api_request('getWebhookInfo')
    current = info.get('result', {}).get('url')
    if current != expected:
        logging.info('Registering webhook %s', expected)
        resp = await bot.api_request('setWebhook', {'url': expected})
        if not resp.get('ok'):
            logging.error('Failed to register webhook: %s', resp)
            raise RuntimeError(f"Webhook registration failed: {resp}")
        logging.info('Webhook registered successfully')
    else:
        logging.info('Webhook already registered at %s', current)

async def handle_webhook(request):
    bot: Bot = request.app['bot']
    try:
        data = await request.json()
        logging.info("Received webhook: %s", data)
    except Exception:
        logging.exception("Invalid webhook payload")
        return web.Response(text='bad request', status=400)
    try:
        await bot.handle_update(data)
    except Exception:
        logging.exception("Error handling update")
        return web.Response(text='error', status=500)
    return web.Response(text='ok')

def create_app():
    app = web.Application()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not found in environment variables")

    bot = Bot(token, DB_PATH)
    app['bot'] = bot

    app.router.add_post('/webhook', handle_webhook)

    webhook_base = os.getenv("WEBHOOK_URL")
    if not webhook_base:
        raise RuntimeError("WEBHOOK_URL not found in environment variables")

    async def start_background(app: web.Application):
        logging.info("Application startup")
        try:
            await bot.start()
            await bot.run_openai_health_check()
            await ensure_webhook(bot, webhook_base)
        except Exception:
            logging.exception("Error during startup")
            raise
        app['schedule_task'] = asyncio.create_task(bot.schedule_loop())

    async def cleanup_background(app: web.Application):
        await bot.close()
        app['schedule_task'].cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await app['schedule_task']


    app.on_startup.append(start_background)
    app.on_cleanup.append(cleanup_background)

    return app


if __name__ == '__main__':

    web.run_app(create_app(), port=int(os.getenv("PORT", 8080)))


