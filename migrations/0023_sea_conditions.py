"""Migration 0023: ensure sea_conditions table exists."""

from __future__ import annotations

import sqlite3


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def run(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sea_conditions (
            sea_id INTEGER PRIMARY KEY,
            updated TEXT,
            wave_height_m REAL,
            wind_speed_10m_ms REAL,
            cloud_cover_pct REAL
        )
        """
    )

    if _table_exists(conn, "sea_cache"):
        conn.execute(
            """
            INSERT OR IGNORE INTO sea_conditions (sea_id, updated, wave_height_m)
            SELECT sea_id, updated, wave
            FROM sea_cache
            """
        )

    conn.commit()
