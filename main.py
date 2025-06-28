import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime, date, timedelta, timezone
import contextlib
import re

from aiohttp import web, ClientSession

logging.basicConfig(level=logging.INFO)

DB_PATH = os.getenv("DB_PATH", "bot.db")
TZ_OFFSET = os.getenv("TZ_OFFSET", "+00:00")
SCHED_INTERVAL_SEC = int(os.getenv("SCHED_INTERVAL_SEC", "30"))
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
            night REAL
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

    """CREATE TABLE IF NOT EXISTS asset_images (
            message_id INTEGER PRIMARY KEY,
            hashtags TEXT,
            template TEXT,
            used_at TEXT
        )""",

    """CREATE TABLE IF NOT EXISTS asset_channel (
            channel_id INTEGER PRIMARY KEY
        )""",

    """CREATE TABLE IF NOT EXISTS weather_publish_channels (
            channel_id INTEGER PRIMARY KEY,
            post_time TEXT NOT NULL,
            last_published_at TEXT
        )""",
]


class Bot:
    def __init__(self, token: str, db_path: str):
        self.api_url = f"https://api.telegram.org/bot{token}"
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        for stmt in CREATE_TABLES:
            self.db.execute(stmt)
        self.db.commit()
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

    async def start(self):
        self.session = ClientSession()
        self.running = True

    async def close(self):
        self.running = False
        if self.session:
            await self.session.close()

        self.db.close()

    async def api_request(self, method: str, data: dict = None):
        async with self.session.post(f"{self.api_url}/{method}", json=data) as resp:
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
            f"{lat}&longitude={lon}&current=temperature_2m,weather_code,wind_speed_10m,is_day"
            "&timezone=auto"
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
            f"{lat}&longitude={lon}&hourly=sea_surface_temperature&timezone=auto"

        )
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
                    if last_success > now - timedelta(minutes=30):
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
            times = data["hourly"].get("time")
            if not temps or not times:
                continue

            current = temps[0]
            tomorrow = date.today() + timedelta(days=1)
            morn = day_temp = eve = night = None
            for t, temp in zip(times, temps):
                dt = datetime.fromisoformat(t)
                if dt.date() != tomorrow:
                    continue
                if dt.hour == 6 and morn is None:
                    morn = temp
                elif dt.hour == 12 and day_temp is None:
                    day_temp = temp
                elif dt.hour == 18 and eve is None:
                    eve = temp
                elif dt.hour == 0 and night is None:
                    night = temp
                if morn is not None and day_temp is not None and eve is not None and night is not None:
                    break

            self.db.execute(
                "INSERT OR REPLACE INTO sea_cache (sea_id, updated, current, morning, day, evening, night) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    s["id"],
                    now.isoformat(),
                    current,
                    morn,
                    day_temp,
                    eve,
                    night,
                ),
            )
            self.db.commit()
            updated.add(s["id"])
        if updated:
            await self.update_weather_posts()

    async def handle_update(self, update):
        message = update.get('message') or update.get('channel_post')
        if message:
            await self.handle_message(message)
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

    def _get_sea_cache(self, sea_id: int):
        return self.db.execute(
            "SELECT current, morning, day, evening, night FROM sea_cache WHERE sea_id=?",
            (sea_id,),
        ).fetchone()


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

        def repl(match: re.Match[str]) -> str:
            cid = int(match.group(1))
            field = match.group(2)
            if field == "seatemperature":
                row = self._get_sea_cache(cid)
                if not row:
                    raise ValueError(f"no sea data for {cid}")
                emoji = "\U0001F30A"

                return f"{emoji} {row['current']:.1f}\u00B0C"


            row = self._get_cached_weather(cid)
            if not row:
                raise ValueError(f"no data for city {cid}")
            if field == "temperature":

                is_day = row["is_day"] if "is_day" in row.keys() else None
                emoji = weather_emoji(row["weather_code"], is_day)

                return f"{emoji} {row['temperature']:.1f}\u00B0C"
            if field == "wind":
                return f"{row['wind_speed']:.1f}"
            return ""

        try:
            rendered = re.sub(r"{(\d+)\|(\w+)}", repl, template)
            tomorrow = date.today() + timedelta(days=1)
            rendered = rendered.replace("{next-day-date}", tomorrow.strftime("%d"))
            rendered = rendered.replace("{next-day-month}", tomorrow.strftime("%B"))
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
            if r["base_caption"]:
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
                    if r["base_text"]
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
            else:
                logging.error(
                    "Failed to update weather post %s: %s", r["id"], resp
                )

    def add_weather_channel(self, channel_id: int, post_time: str):
        self.db.execute(
            "INSERT OR REPLACE INTO weather_publish_channels (channel_id, post_time) VALUES (?, ?)",
            (channel_id, post_time),
        )
        self.db.commit()

    def remove_weather_channel(self, channel_id: int):
        self.db.execute(
            "DELETE FROM weather_publish_channels WHERE channel_id=?",
            (channel_id,),
        )
        self.db.commit()

    def list_weather_channels(self):
        cur = self.db.execute(
            "SELECT w.channel_id, w.post_time, w.last_published_at, c.title FROM weather_publish_channels w LEFT JOIN channels c ON c.chat_id=w.channel_id ORDER BY w.channel_id"
        )
        return cur.fetchall()

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

    def add_asset(self, message_id: int, hashtags: str, template: str | None = None):
        self.db.execute(
            "INSERT OR REPLACE INTO asset_images (message_id, hashtags, template) VALUES (?, ?, ?)",
            (message_id, hashtags, template),
        )
        self.db.commit()

    def next_asset(self, tags: set[str] | None):
        cur = self.db.execute(
            "SELECT message_id, hashtags, template FROM asset_images WHERE used_at IS NULL ORDER BY message_id"
        )
        rows = cur.fetchall()
        first_no_tag = None
        for r in rows:
            tagset = set(r["hashtags"].split()) if r["hashtags"] else set()
            if tags and tagset & tags:
                self.db.execute(
                    "UPDATE asset_images SET used_at=? WHERE message_id=?",
                    (datetime.utcnow().isoformat(), r["message_id"]),
                )
                self.db.commit()
                return r
            if not tagset and first_no_tag is None:
                first_no_tag = r
        if first_no_tag:
            self.db.execute(
                "UPDATE asset_images SET used_at=? WHERE message_id=?",
                (datetime.utcnow().isoformat(), first_no_tag["message_id"]),
            )
            self.db.commit()
            return first_no_tag
        return None


    async def publish_weather(self, channel_id: int, tags: set[str] | None = None):
        asset = self.next_asset(tags)
        caption = asset["template"] if asset and asset["template"] else ""
        if caption:
            caption = self._render_template(caption) or caption
        if asset and self.asset_channel_id:
            await self.api_request(
                "copyMessage",
                {
                    "chat_id": channel_id,
                    "from_chat_id": self.asset_channel_id,
                    "message_id": asset["message_id"],
                    "caption": caption or None,
                },
            )
            await self.api_request(
                "deleteMessage",
                {"chat_id": self.asset_channel_id, "message_id": asset["message_id"]},
            )
        else:
            if caption:
                await self.api_request(
                    "sendMessage",
                    {"chat_id": channel_id, "text": caption},
                )
        self.db.execute(
            "UPDATE weather_publish_channels SET last_published_at=? WHERE channel_id=?",
            (datetime.utcnow().isoformat(), channel_id),
        )
        self.db.commit()


    async def handle_message(self, message):
        if self.asset_channel_id and message.get('chat', {}).get('id') == self.asset_channel_id:
            caption = message.get('caption') or message.get('text') or ''
            tags = ' '.join(re.findall(r'#\S+', caption))
            self.add_asset(message['message_id'], tags, caption)
            return

        text = message.get('text', '')
        user_id = message['from']['id']
        username = message['from'].get('username')

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
            keyboard = {'inline_keyboard': [[{'text': keyboard_text, 'url': parts[-1]}]]}

            resp = await self.api_request('editMessageReplyMarkup', {
                'chat_id': chat_id,
                'message_id': msg_id,
                'reply_markup': keyboard
            })
            if resp.get('ok'):
                logging.info('Updated message %s with button', msg_id)
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

            resp = await self.api_request('editMessageReplyMarkup', {
                'chat_id': chat_id,
                'message_id': msg_id,
                'reply_markup': {}
            })
            if resp.get('ok'):
                logging.info('Removed buttons from message %s', msg_id)
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Button removed'})
            else:
                logging.error('Failed to remove button from %s: %s', msg_id, resp)
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Failed to remove button'})
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

        if text.startswith('/weatherposts') and self.is_superadmin(user_id):
            parts = text.split(maxsplit=1)
            force = len(parts) > 1 and parts[1] == 'update'
            if force:
                await self.update_weather_posts()
            cur = self.db.execute(
                'SELECT chat_id, message_id, template FROM weather_posts ORDER BY id'
            )
            rows = cur.fetchall()
            if not rows:
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'No weather posts'})
                return
            lines = []
            for r in rows:
                header = self._render_template(r['template'])
                url = self.post_url(r['chat_id'], r['message_id'])
                if header:
                    lines.append(f"{url} {header}")
                else:
                    lines.append(f"{url} no data")
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': '\n'.join(lines)})
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
                last = r['last_published_at'] or 'never'
                keyboard = {'inline_keyboard': [[{'text': 'Run now', 'callback_data': f'wrnow:{r["channel_id"]}'}, {'text': 'Stop', 'callback_data': f'wstop:{r["channel_id"]}'}]]}
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': f"{r['title'] or r['channel_id']} at {r['post_time']} last {last}", 'reply_markup': keyboard})
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
                if row and all(row[k] is not None for k in row.keys()):
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
            resp = await self.api_request('forwardMessage', {
                'chat_id': user_id,
                'from_chat_id': chat_id,
                'message_id': msg_id
            })
            if not resp.get('ok'):
                await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Cannot read post'})
                return

            base_text = resp['result'].get('text')
            base_caption = resp['result'].get('caption')
            if base_text and WEATHER_SEPARATOR in base_text:
                base_text = base_text.split(WEATHER_SEPARATOR, 1)[1]
            if base_caption and WEATHER_SEPARATOR in base_caption:
                base_caption = base_caption.split(WEATHER_SEPARATOR, 1)[1]
            markup = resp['result'].get('reply_markup')

            if base_text is None and base_caption is None:
                base_text = ''
            await self.api_request('deleteMessage', {'chat_id': user_id, 'message_id': resp['result']['message_id']})
            self.db.execute(

                'INSERT OR REPLACE INTO weather_posts (chat_id, message_id, template, base_text, base_caption, reply_markup) VALUES (?, ?, ?, ?, ?, ?)',
                (chat_id, msg_id, template, base_text, base_caption, json.dumps(markup) if markup else None)

            )
            self.db.commit()
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
            await self.publish_weather(cid, None)
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Posted'})
        elif data.startswith('wstop:') and self.is_superadmin(user_id):
            cid = int(data.split(':')[1])
            self.remove_weather_channel(cid)
            await self.api_request('sendMessage', {'chat_id': user_id, 'text': 'Channel removed'})
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
        offset = self.parse_offset(TZ_OFFSET)
        local_now = now_utc + offset
        cur = self.db.execute(
            "SELECT channel_id, post_time, last_published_at FROM weather_publish_channels"
        )
        for r in cur.fetchall():
            try:
                if r["last_published_at"]:
                    last = datetime.fromisoformat(r["last_published_at"])
                    if last.date() == local_now.date():
                        continue
                hh, mm = map(int, r["post_time"].split(":"))
                scheduled = datetime.combine(local_now.date(), datetime.min.time()).replace(hour=hh, minute=mm)
                if local_now >= scheduled:
                    await self.publish_weather(r["channel_id"], None)
            except Exception:
                logging.exception("Failed to publish weather for %s", r["channel_id"])

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


