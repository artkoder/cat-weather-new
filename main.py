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
        ):
            cur = self.db.execute(f"PRAGMA table_info({table})")
            names = [r[1] for r in cur.fetchall()]
            if column not in names:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")
        self.db.commit()
        self.pending = {}
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

    async def handle_update(self, update):
        if 'message' in update:
            await self.handle_message(update['message'])
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

    async def handle_message(self, message):
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

    async def schedule_loop(self):
        """Background scheduler running at configurable intervals."""

        try:
            logging.info("Scheduler loop started")
            while self.running:
                await self.process_due()
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


