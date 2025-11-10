from __future__ import annotations

import sqlite3


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    )
    return cur.fetchone() is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def run(conn: sqlite3.Connection) -> None:
    """Add normalized wave and sky metric columns to assets table."""

    if not _table_exists(conn, "assets"):
        return

    columns = _table_columns(conn, "assets")

    # Add wave_score_0_10 column if it doesn't exist
    if "wave_score_0_10" not in columns:
        conn.execute("ALTER TABLE assets ADD COLUMN wave_score_0_10 REAL")

    # Add wave_conf column if it doesn't exist
    if "wave_conf" not in columns:
        conn.execute("ALTER TABLE assets ADD COLUMN wave_conf REAL")

    # Add sky_code column if it doesn't exist
    if "sky_code" not in columns:
        conn.execute("ALTER TABLE assets ADD COLUMN sky_code TEXT")

    conn.commit()
