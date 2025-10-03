from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta


def _parse_offset(offset: str) -> timedelta:
    sign = -1 if offset.startswith("-") else 1
    hours, minutes = offset.lstrip("+-").split(":")
    return timedelta(minutes=sign * (int(hours) * 60 + int(minutes)))


def _next_weather_run(post_time: str, offset: str, reference: datetime | None = None) -> datetime:
    reference_dt = reference or datetime.utcnow()
    tz_delta = _parse_offset(offset)
    local_ref = reference_dt + tz_delta
    hour, minute = map(int, post_time.split(":"))
    candidate = local_ref.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= local_ref:
        candidate += timedelta(days=1)
    return candidate - tz_delta


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    return cur.fetchone() is not None


def migrate_weather_publish_channels(
    conn: sqlite3.Connection, *, tz_offset: str | None = None, reference: datetime | None = None
) -> bool:
    """Backfill legacy weather channels into weather_jobs.

    Returns True if any migration was performed (rows moved or table dropped).
    """

    if not _table_exists(conn, "weather_publish_channels"):
        return False

    rows = conn.execute(
        "SELECT channel_id, post_time, last_published_at FROM weather_publish_channels"
    ).fetchall()

    tz = tz_offset or os.getenv("TZ_OFFSET", "+00:00")
    now = datetime.utcnow().isoformat()
    for row in rows:
        channel_id, post_time, last_published = row
        if not post_time:
            continue
        run_at = _next_weather_run(post_time, tz, reference=reference).isoformat()
        created_at = last_published or now
        conn.execute(
            """
            INSERT INTO weather_jobs (
                channel_id, post_time, run_at, last_run_at, failures, last_error, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 0, NULL, ?, ?)
            ON CONFLICT(channel_id) DO UPDATE SET
                post_time=excluded.post_time,
                run_at=excluded.run_at,
                last_run_at=COALESCE(excluded.last_run_at, weather_jobs.last_run_at),
                failures=0,
                last_error=NULL,
                updated_at=excluded.updated_at
            """,
            (
                channel_id,
                post_time,
                run_at,
                last_published,
                created_at,
                now,
            ),
        )

    conn.execute("DROP TABLE IF EXISTS weather_publish_channels")
    return True
