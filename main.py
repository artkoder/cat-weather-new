from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import random
import re
import sqlite3
from datetime import datetime, date, timedelta, timezone, time as dtime
from pathlib import Path

from typing import Any, Iterable, TYPE_CHECKING

from aiohttp import web, ClientSession, FormData
from PIL import Image, ImageDraw, ImageFont
import piexif

from data_access import Asset, DataAccess, Rubric
from jobs import Job, JobDelayed, JobQueue
from openai_client import OpenAIClient
from weather_migration import migrate_weather_publish_channels

if TYPE_CHECKING:
    from openai_client import OpenAIResponse

logging.basicConfig(level=logging.INFO)

# Default database path points to /data which is mounted as a Fly.io volume.
# This ensures information like registered channels and scheduled posts
# persists across deployments unless DB_PATH is explicitly overridden.
DB_PATH = os.getenv("DB_PATH", "/data/bot.db")
TZ_OFFSET = os.getenv("TZ_OFFSET", "+00:00")
SCHED_INTERVAL_SEC = int(os.getenv("SCHED_INTERVAL_SEC", "30"))
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
        if migrate_weather_publish_channels(self.db, tz_offset=TZ_OFFSET):
            self.db.commit()
        self.jobs = JobQueue(self.db, concurrency=1)
        self.jobs.register_handler("ingest", self._job_ingest)
        self.jobs.register_handler("vision", self._job_vision)
        self.jobs.register_handler("publish_rubric", self._job_publish_rubric)
        self.openai = OpenAIClient(os.getenv("OPENAI_API_KEY"))
        self._model_limits = self._load_model_limits()
        asset_dir = os.getenv("ASSET_STORAGE_DIR")
        self.asset_storage = Path(asset_dir).expanduser() if asset_dir else Path("/tmp/bot_assets")
        self.asset_storage.mkdir(parents=True, exist_ok=True)
        self._last_geocode_at: datetime | None = None
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
        self.failed_fetches: dict[int, tuple[int, datetime]] = {}
        self.asset_channel_id = self.get_asset_channel()

        self.session: ClientSession | None = None
        self.running = False
        self.manual_buttons: dict[tuple[int, int], dict[str, list[list[dict]]]] = {}

    def _next_usage_reset(self, *, now: datetime | None = None) -> datetime:
        reference = now or datetime.utcnow()
        next_day = reference.date() + timedelta(days=1)
        return datetime.combine(next_day, dtime(hour=0, minute=5))

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
        limit_4o = parse_limit(os.getenv("OPENAI_DAILY_TOKEN_LIMIT_4O"), "OPENAI_DAILY_TOKEN_LIMIT_4O")
        limit_4o_mini = parse_limit(
            os.getenv("OPENAI_DAILY_TOKEN_LIMIT_4O_MINI"), "OPENAI_DAILY_TOKEN_LIMIT_4O_MINI"
        )
        if limit_4o is not None:
            limits["gpt-4o"] = limit_4o
        if limit_4o_mini is not None:
            limits["gpt-4o-mini"] = limit_4o_mini
        if not limits and default_limit is not None:
            limits = {"gpt-4o": default_limit, "gpt-4o-mini": default_limit}
        return limits

    def _enforce_openai_limit(self, job: Job | None, model: str) -> None:
        if (
            job is None
            or model not in self._model_limits
            or not self.openai
            or not self.openai.api_key
        ):
            return
        limit = self._model_limits[model]
        total_today = self.data.get_daily_token_usage_total(models={model})
        if total_today >= limit:
            resume_at = self._next_usage_reset()
            reason = (
                f"Превышен дневной лимит токенов {model}: "
                f"{total_today}/{limit}. Задача перенесена до {resume_at.isoformat()}"
            )
            logging.warning(reason)
            raise JobDelayed(resume_at, reason)

    def _record_openai_usage(
        self,
        model: str,
        response: "OpenAIResponse" | None,
        *,
        job: Job | None = None,
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
        if model in self._model_limits:
            total_today = self.data.get_daily_token_usage_total(models={model})
            limit = self._model_limits[model]
            logging.info(
                "Суммарное потребление токенов %s за сегодня: %s из %s",
                model,
                total_today,
                limit,
            )

    async def start(self):
        self.session = ClientSession()
        self.running = True
        await self.jobs.start()

    async def close(self):
        self.running = False
        await self.jobs.stop()
        if self.session:
            await self.session.close()

        self.db.close()

    async def handle_edited_message(self, message):
        if self.asset_channel_id and message.get('chat', {}).get('id') == self.asset_channel_id:
            info = self._collect_asset_metadata(message)
            message_id = info.get("message_id")
            tg_chat_id = info.get("tg_chat_id") or 0
            if not message_id:
                return
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
                )
            if asset_id:
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
        wave = row['wave']
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
        self.db.execute("DELETE FROM asset_channel")
        self.db.execute(
            "INSERT INTO asset_channel (channel_id) VALUES (?)",
            (channel_id,),
        )
        self.db.commit()
        self.asset_channel_id = channel_id

    def get_asset_channel(self) -> int | None:
        cur = self.db.execute("SELECT channel_id FROM asset_channel LIMIT 1")
        row = cur.fetchone()
        return row["channel_id"] if row else None

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
    ) -> int:
        source_channel = channel_id or self.asset_channel_id or 0
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
        )
        logging.info("Stored asset %s tags=%s", message_id, hashtags)
        return asset_id

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

    async def _download_file(self, file_id: str) -> bytes | None:
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
        url = f"https://api.telegram.org/file/bot{self.token}/{file_path}"
        async with self.session.get(url) as file_resp:
            if file_resp.status != 200:
                logging.error("Failed to download file %s: HTTP %s", file_id, file_resp.status)
                return None
            return await file_resp.read()

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

    def _extract_gps(self, data: bytes) -> tuple[float, float] | None:
        try:
            with Image.open(io.BytesIO(data)) as img:
                exif_bytes = img.info.get("exif")
            if not exif_bytes:
                return None
            exif_dict = piexif.load(exif_bytes)
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

    async def _reverse_geocode(self, lat: float, lon: float) -> dict[str, Any]:
        if self.dry_run or not self.session:
            return {}
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
        }
        headers = {"User-Agent": "KotopogodaBot/1.0"}
        async with self.session.get(url, params=params, headers=headers) as resp:
            if resp.status != 200:
                logging.warning("Reverse geocode failed with HTTP %s", resp.status)
                return {}
            data = await resp.json()
        self._last_geocode_at = datetime.utcnow()
        return data.get("address", {}) if isinstance(data, dict) else {}

    def _store_local_file(self, asset_id: int, file_meta: dict[str, Any], data: bytes) -> str:
        suffix = ""
        file_name = file_meta.get("file_name")
        if file_name:
            suffix = Path(file_name).suffix
        if not suffix and file_meta.get("mime_type"):
            if "/" in file_meta["mime_type"]:
                suffix = "." + file_meta["mime_type"].split("/")[-1]
        unique = file_meta.get("file_unique_id") or str(asset_id)
        filename = f"{asset_id}_{unique}{suffix}" if suffix else f"{asset_id}_{unique}"
        path = self.asset_storage / filename
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
            vision_job = self.jobs.enqueue("vision", {"asset_id": asset_id})
            logging.info(
                "Asset %s queued for vision job %s after ingest (dry run)", asset_id, vision_job
            )
            return
        data = await self._download_file(file_id)
        if not data:
            raise RuntimeError(f"Failed to download file for asset {asset_id}")
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
        local_path = self._store_local_file(asset_id, file_meta, data)
        gps = None
        if asset.kind == "photo":
            gps = self._extract_gps(data)
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
            "local_path": local_path,
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
        vision_job = self.jobs.enqueue("vision", {"asset_id": asset_id})
        logging.info("Asset %s queued for vision job %s after ingest", asset_id, vision_job)

    async def _job_vision(self, job: Job):
        asset_id = job.payload.get("asset_id") if job.payload else None
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
        local_path = asset.local_path
        if not local_path and not self.dry_run:
            data = await self._download_file(file_id)
            if data:
                local_path = self._store_local_file(asset_id, file_meta, data)
        if self.dry_run or not self.openai or not self.openai.api_key:
            self.data.update_asset(asset_id, vision_results={"status": "skipped"})
            return
        if not local_path or not os.path.exists(local_path):
            raise RuntimeError(f"Local file for asset {asset_id} not found")
        with open(local_path, "rb") as fh:
            image_bytes = fh.read()
        schema = {
            "name": "vision_classification",
            "schema": {
                "type": "object",
                "title": "Vision classification payload",
                "description": (
                    "Заполни сведения о категории сюжета, архитектурном виде, погоде на фото и "
                    "цветах согласно §3.1."
                ),
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Основная классификация сюжета фотографии",
                        "minLength": 1,
                    },
                    "arch_view": {
                        "type": "string",
                        "description": "Описание архитектурных элементов или вида (если их нет — пустая строка)",
                        "default": "",
                    },
                    "photo_weather": {
                        "type": "string",
                        "description": "Краткое описание погодных условий, видимых на изображении",
                        "minLength": 1,
                    },
                    "flower_varieties": {
                        "type": "array",
                        "description": "Перечень цветов, различимых на фото",
                        "items": {
                            "type": "string",
                            "minLength": 1,
                        },
                        "minItems": 0,
                        "default": [],
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Уверенность модели (0.0–1.0)",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
                "required": ["category", "photo_weather"],
                "additionalProperties": False,
            },
        }
        system_prompt = (
            "Ты ассистент проекта Котопогода. Проанализируй изображение и верни JSON, строго соответствующий схеме, "
            "с полями category, arch_view, photo_weather, flower_varieties и confidence. "
            "category — краткая классификация сюжета. arch_view — архитектурный ракурс (если нет, оставь пустую строку). "
            "photo_weather — сводка погоды, которую видно на фото. flower_varieties — массив названий цветов или пустой массив. "
            "confidence — число от 0 до 1."
        )
        user_prompt = (
            "Определи по фото категорию, архитектурный вид, погоду и перечисли заметные цветы. Верни только JSON согласно схеме."
        )
        self._enforce_openai_limit(job, "gpt-4o-mini")
        response = await self.openai.classify_image(
            model="gpt-4o-mini",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_bytes=image_bytes,
            schema=schema,
        )
        if response is None:
            self.data.update_asset(asset_id, vision_results={"status": "skipped"})
            return
        result = response.content
        if not isinstance(result, dict):
            raise RuntimeError("Invalid response from vision model")
        category = str(result.get("category", "")).strip()
        photo_weather = str(result.get("photo_weather", "")).strip()
        if not category or not photo_weather:
            raise RuntimeError("Invalid response from vision model")
        arch_view = str(result.get("arch_view", "")).strip()
        raw_flowers = result.get("flower_varieties")
        flower_varieties: list[str] = []
        if isinstance(raw_flowers, list):
            for item in raw_flowers:
                text = str(item).strip()
                if text:
                    flower_varieties.append(text)
        elif raw_flowers:
            text = str(raw_flowers).strip()
            if text:
                flower_varieties.append(text)
        raw_confidence = result.get("confidence")
        confidence: float | None
        if isinstance(raw_confidence, (int, float)):
            confidence = float(raw_confidence)
        elif isinstance(raw_confidence, str):
            try:
                confidence = float(raw_confidence)
            except ValueError:
                confidence = None
        else:
            confidence = None
        location_parts: list[str] = []
        if asset.city:
            location_parts.append(asset.city)
        if asset.country and asset.country not in location_parts:
            location_parts.append(asset.country)
        summary_parts: list[str] = [category]
        if location_parts:
            summary_parts.append(", ".join(location_parts))
        if photo_weather:
            summary_parts.append(f"Погода: {photo_weather}")
        caption_lines = ["Распознано: " + " • ".join(summary_parts)]
        if arch_view:
            caption_lines.append(f"Архитектурный вид: {arch_view}")
        if flower_varieties:
            caption_lines.append("Цветы: " + ", ".join(flower_varieties))
        if confidence is not None:
            display_confidence = confidence * 100 if 0 <= confidence <= 1 else confidence
            caption_lines.append(f"Уверенность модели: {display_confidence:.0f}%")
        caption_text = "\n".join(line for line in caption_lines if line)
        result_payload = {
            "status": "ok",
            "provider": "gpt-4o-mini",
            "category": category,
            "arch_view": arch_view,
            "photo_weather": photo_weather,
            "flower_varieties": flower_varieties,
            "confidence": confidence,
        }
        resp = await self.api_request(
            "sendPhoto",
            {
                "chat_id": asset.channel_id,
                "photo": file_id,
                "caption": caption_text,
            },
        )
        if not resp.get("ok"):
            raise RuntimeError(f"Failed to publish vision result: {resp}")
        new_mid = resp.get("result", {}).get("message_id") if resp.get("result") else None
        self.data.update_asset(
            asset_id,
            recognized_message_id=new_mid,
            vision_results=result_payload,
            vision_category=category,
            vision_arch_view=arch_view,
            vision_photo_weather=photo_weather,
            vision_flower_varieties=flower_varieties,
            vision_confidence=confidence,
            vision_caption=caption_text,
            local_path=local_path,
        )
        self._record_openai_usage("gpt-4o-mini", response, job=job)
        if not self.dry_run and new_mid:
            await self.api_request(
                "deleteMessage",
                {"chat_id": asset.channel_id, "message_id": asset.message_id},
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



    async def publish_weather(
        self,
        channel_id: int,
        tags: set[str] | None = None,
        record: bool = True,
    ) -> bool:

        asset = self.next_asset(tags)
        caption = asset["template"] if asset and asset.get("template") else ""
        if caption:
            caption = self._render_template(caption) or caption
        from_chat = None
        if asset:
            from_chat = asset.get("channel_id") or self.asset_channel_id
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
            await self.api_request(
                "deleteMessage",
                {
                    "chat_id": from_chat,
                    "message_id": asset["message_id"],
                },
            )

            ok = resp.get("ok", False)
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
                        None,
                        {
                            "caption": caption,
                            "categories": asset["categories"] if asset else [],
                        },
                    )
        else:
            logging.error("Failed to publish weather: %s", resp)
        return ok



    async def handle_message(self, message):
        global TZ_OFFSET

        if self.asset_channel_id and message.get('chat', {}).get('id') == self.asset_channel_id:
            info = self._collect_asset_metadata(message)
            asset_id = self.add_asset(
                info.get("message_id", 0),
                info.get("hashtags", ""),
                info.get("caption"),
                channel_id=info.get("tg_chat_id"),
                metadata=info.get("metadata"),
                tg_chat_id=info.get("tg_chat_id"),
                kind=info.get("kind"),
                file_meta=info.get("file_meta"),
                author_user_id=info.get("author_user_id"),
                author_username=info.get("author_username"),
                sender_chat_id=info.get("sender_chat_id"),
                via_bot_id=info.get("via_bot_id"),
                forward_from_user=info.get("forward_from_user"),
                forward_from_chat=info.get("forward_from_chat"),
            )
            if asset_id:
                self._schedule_ingest_job(asset_id, reason="new_message")
            return


        if 'from' not in message:
            # ignore channel posts when asset channel is not configured
            return


        text = message.get('text', '')
        user_id = message['from']['id']
        username = message['from'].get('username')

        if text.startswith('/help'):
            help_messages = [
                (
                    "*Основные команды*\n\n"
                    "*Доступ и настройки*\n"
                    "- `/help` — краткая памятка с ключевыми сценариями.\n"
                    "- `/start` — запросить доступ или подтвердить, что бот уже активирован.\n"
                    "- `/tz <±HH:MM>` — установить личный часовой пояс для расписаний.\n"
                    "- `/list_users` — список администраторов и операторов.\n"
                    "- `/pending` → кнопки `Approve`/`Reject` для очереди заявок.\n"
                    "- `/approve <id>` / `/reject <id>` — ручное утверждение или отказ.\n"
                    "- `/add_user <id>` / `/remove_user <id>` — постоянное добавление или удаление доступа.\n"
                ),
                (
                    "*Каналы и расписания*\n"
                    "- `/channels` — все подключённые каналы (раздел «Каналы» админ-интерфейса).\n"
                    "- `/set_assets_channel` — выбрать канал хранения ассетов перед запуском конвейера.\n"
                    "- `/scheduled` — список очереди публикаций с кнопками `Cancel` и `Reschedule`.\n"
                    "- `/history` — последние отправленные посты с отметкой времени.\n"
                    "- `/setup_weather` — мастер настройки расписаний рубрик для выбранных каналов.\n"
                    "- `/list_weather_channels` — обзор рубрик: показывает время, дату последнего запуска и кнопки `Run now`/`Stop`.\n"
                ),
                (
                    "*Работа с постами, погодой и ручные действия*\n"
                    "- `/addbutton <post_url> <текст> <url>` — добавить кнопку к посту; используйте `t.me/c/...` из истории канала.\n"
                    "- `/delbutton <post_url>` — удалить все кнопки у поста; изменения сохраняются в базе.\n"
                    "- `/addweatherbutton <post_url> <текст> [url]` — быстрый доступ к свежему прогнозу, можно опустить URL после `/weather now`.\n"
                    "- `/weatherposts [update]` — перечень активных погодных шаблонов и кнопка остановки рассылки.\n"
                    "- `/regweather <post_url> <template>` — зарегистрировать новый шаблон для автоподстановки погоды.\n"
                    "- `/weather [now]` — посмотреть кэш погоды и морей или форсировать обновление.\n"
                    "- `/addcity`, `/cities` и `/addsea`, `/seas` — управлять справочниками городов и морей; `/amber` открывает выбор канала с кнопкой «Янтарный».")
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
                "Подробная документация: файл `README.md` → раздел *Commands* и журнал изменений `CHANGELOG.md`."
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
            cur = self.db.execute('SELECT chat_id, title FROM channels')
            rows = cur.fetchall()
            if not rows:
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'No channels available'})
                return
            keyboard = {'inline_keyboard': [[{'text': r['title'], 'callback_data': f'asset_ch:{r["chat_id"]}'}] for r in rows]}
            self.pending[user_id] = {'set_assets': True}
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Select asset channel', 'reply_markup': keyboard})
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
        elif data.startswith('asset_ch:') and user_id in self.pending and self.pending[user_id].get('set_assets'):
            cid = int(data.split(':')[1])
            self.set_asset_channel(cid)
            del self.pending[user_id]
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Asset channel set'})
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

    def enqueue_rubric(
        self,
        code: str,
        *,
        channel_id: int | None = None,
        test: bool = False,
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
        }
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
        return await handler(rubric, int(target), test=test, job=job)

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

    async def _publish_flowers(
        self,
        rubric: Rubric,
        channel_id: int,
        *,
        test: bool = False,
        job: Job | None = None,
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
        assets = self.data.fetch_assets_by_vision_category(
            "flowers",
            rubric_id=rubric.id,
            limit=max_count,
        )
        if len(assets) < min_count:
            logging.warning(
                "Not enough assets for flowers rubric: have %s, need %s",
                len(assets),
                min_count,
            )
            return False
        cities = sorted({asset.city for asset in assets if asset.city})
        greeting, hashtags = await self._generate_flowers_copy(
            rubric, cities, len(assets), job=job
        )
        hashtag_list = self._prepare_hashtags(hashtags)
        caption_parts = [greeting.strip()] if greeting else []
        if cities:
            caption_parts.append("Города съёмки: " + ", ".join(cities))
        if hashtag_list:
            caption_parts.append(" ".join(hashtag_list))
        caption = "\n\n".join(part for part in caption_parts if part)
        media: list[dict[str, Any]] = []
        for idx, asset in enumerate(assets):
            file_id = asset.file_id
            if not file_id:
                logging.warning("Asset %s missing file_id", asset.id)
                return False
            item = {"type": "photo", "media": file_id}
            if idx == 0 and caption:
                item["caption"] = caption
            media.append(item)
        response = await self.api_request(
            "sendMediaGroup",
            {"chat_id": channel_id, "media": media},
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
        metadata = {
            "rubric_code": rubric.code,
            "asset_ids": [asset.id for asset in assets],
            "test": test,
            "cities": cities,
            "greeting": greeting,
            "hashtags": hashtag_list,
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

    async def _generate_flowers_copy(
        self,
        rubric: Rubric,
        cities: list[str],
        asset_count: int,
        *,
        job: Job | None = None,
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
        prompt = " ".join(prompt_parts)
        schema = {
            "name": "flowers_post",
            "schema": {
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
            },
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
                self._record_openai_usage("gpt-4o", response, job=job)
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
        try:
            for idx, asset in enumerate(assets, start=1):
                path = self._overlay_number(asset, idx, config)
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
            "name": "guess_arch_post",
            "schema": {
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
            },
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
                self._record_openai_usage("gpt-4o", response, job=job)
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
        mapping = {
            "clear": {"clear", "partly_cloudy"},
            "partly_cloudy": {"clear", "partly_cloudy", "cloudy"},
            "cloudy": {"partly_cloudy", "cloudy", "overcast"},
            "overcast": {"cloudy", "overcast"},
            "rain": {"rain", "storm"},
            "storm": {"storm", "rain"},
            "snow": {"snow"},
            "fog": {"fog", "overcast", "cloudy"},
            "night": {"night", "clear", "partly_cloudy"},
            "indoor": {"indoor"},
        }
        allowed = mapping.get(actual_class, {actual_class})
        return set(allowed)

    def _normalize_weather_label(self, label: str | None) -> str | None:
        if not label:
            return None
        text = str(label).strip().lower()
        if not text:
            return None
        keyword_map: list[tuple[tuple[str, ...], str]] = [
            (("indoor", "помещ", "inside", "room"), "indoor"),
            (("ноч", "night"), "night"),
            (("гроза", "storm", "thunder", "молн"), "storm"),
            (("снег", "snow", "снеж", "метел", "blizzard"), "snow"),
            (("дожд", "rain", "ливн", "drizzle", "wet"), "rain"),
            (("туман", "fog", "mist", "дымк", "haze", "смог"), "fog"),
            (("пасмур", "overcast", "сплошн", "тучн", "серое небо"), "overcast"),
            (("облач", "cloud"), "cloudy"),
            (("закат", "sunset", "рассвет", "sunrise", "golden hour"), "partly_cloudy"),
            (("солне", "ясн", "clear", "sunny", "bright sun"), "clear"),
        ]
        for needles, token in keyword_map:
            for needle in needles:
                if needle in text:
                    return token
        return text.split()[0]

    def _classify_weather_code(self, code: int | None) -> str | None:
        if code is None:
            return None
        mapping = {
            0: "clear",
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
            95: "storm",
            96: "storm",
            99: "storm",
        }
        return mapping.get(code)

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
        self, asset: Asset, number: int, config: dict[str, Any]
    ) -> str | None:
        local_path = asset.local_path
        if not local_path or not os.path.exists(local_path):
            logging.warning("Asset %s missing local file for overlay", asset.id)
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
            offset = (
                int(base.width * 0.05),
                int(base.height * 0.05),
            )
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
        max_side = max(96, min(base_size) // 4)
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


