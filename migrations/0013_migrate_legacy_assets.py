from __future__ import annotations

import json
import sqlite3
from datetime import datetime


def _extract_tags(hashtags: str | None) -> list[str]:
    if not hashtags:
        return []
    tokens = []
    for raw in hashtags.replace("\n", " ").split(" "):
        tag = raw.strip()
        if not tag:
            continue
        if tag not in tokens:
            tokens.append(tag)
    return tokens


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    )
    return cur.fetchone() is not None


def run(conn: sqlite3.Connection) -> None:
    old_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        if not _table_exists(conn, "asset_images"):
            conn.execute("DROP TABLE IF EXISTS asset_images")
            return

        channel_row = None
        if _table_exists(conn, "asset_channel"):
            channel_row = conn.execute("SELECT channel_id FROM asset_channel LIMIT 1").fetchone()
        if channel_row and channel_row["channel_id"] is not None:
            default_channel = int(channel_row["channel_id"])
        else:
            default_channel = 0

        rows = conn.execute(
            "SELECT message_id, hashtags, template, used_at FROM asset_images"
        ).fetchall()

        if not rows:
            conn.execute("DROP TABLE IF EXISTS asset_images")
            return

        now = datetime.utcnow().isoformat()
        for row in rows:
            message_id = row["message_id"]
            hashtags = row["hashtags"]
            template = row["template"]
            used_raw = row["used_at"]
            used_at = used_raw or None
            created_at = used_at or now
            updated_at = created_at
            categories_json = json.dumps(_extract_tags(hashtags))
            conn.execute(
                """
                INSERT INTO assets (
                    channel_id,
                    tg_chat_id,
                    message_id,
                    caption_template,
                    caption,
                    hashtags,
                    categories,
                    kind,
                    last_used_at,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tg_chat_id, message_id) DO NOTHING
                """,
                (
                    default_channel,
                    default_channel,
                    message_id,
                    template,
                    template,
                    hashtags,
                    categories_json,
                    "photo",
                    used_at,
                    created_at,
                    updated_at,
                ),
            )

        conn.execute("DROP TABLE IF EXISTS asset_images")
    finally:
        conn.row_factory = old_factory
