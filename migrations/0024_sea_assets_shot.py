"""Migration 0024: add shot metadata columns and wind details."""

from __future__ import annotations

import sqlite3


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    return {row[1] for row in rows}


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _index_exists(conn: sqlite3.Connection, table: str, index: str) -> bool:
    rows = conn.execute(f"PRAGMA index_list('{table}')").fetchall()
    return any(row[1] == index for row in rows)


def run(conn: sqlite3.Connection) -> None:
    assets_columns = _table_columns(conn, "assets")
    if "shot_at_utc" not in assets_columns:
        conn.execute("ALTER TABLE assets ADD COLUMN shot_at_utc INTEGER")
    if "shot_doy" not in assets_columns:
        conn.execute("ALTER TABLE assets ADD COLUMN shot_doy INTEGER")

    if _table_exists(conn, "assets") and not _index_exists(conn, "assets", "idx_assets_shot_doy"):
        conn.execute("CREATE INDEX IF NOT EXISTS idx_assets_shot_doy ON assets(shot_doy)")

    if not _table_exists(conn, "sea_conditions"):
        return

    sea_columns = _table_columns(conn, "sea_conditions")
    if "wind_speed_10m_kmh" not in sea_columns:
        conn.execute("ALTER TABLE sea_conditions ADD COLUMN wind_speed_10m_kmh REAL")
    if "wind_gusts_10m_ms" not in sea_columns:
        conn.execute("ALTER TABLE sea_conditions ADD COLUMN wind_gusts_10m_ms REAL")
    if "wind_gusts_10m_kmh" not in sea_columns:
        conn.execute("ALTER TABLE sea_conditions ADD COLUMN wind_gusts_10m_kmh REAL")
    if "wind_units" not in sea_columns:
        conn.execute("ALTER TABLE sea_conditions ADD COLUMN wind_units TEXT")
    if "wind_gusts_units" not in sea_columns:
        conn.execute("ALTER TABLE sea_conditions ADD COLUMN wind_gusts_units TEXT")
    if "wind_time_ref" not in sea_columns:
        conn.execute("ALTER TABLE sea_conditions ADD COLUMN wind_time_ref TEXT")

    conn.commit()
