from __future__ import annotations

import sqlite3

_ALLOWED_SOURCES = ("mobile", "telegram")


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    cursor = conn.execute(f"PRAGMA table_info('{table}')")
    return {str(row[1]) for row in cursor.fetchall()}


def _ensure_uploads_source(conn: sqlite3.Connection) -> None:
    columns = _column_names(conn, "uploads")
    if "source" in columns:
        return
    allowed = ",".join(f"'{value}'" for value in _ALLOWED_SOURCES)
    conn.execute(
        f"""
        ALTER TABLE uploads
        ADD COLUMN source TEXT NOT NULL DEFAULT 'mobile'
        CHECK (source IN ({allowed}))
        """
    )


def _ensure_assets_source(conn: sqlite3.Connection) -> None:
    columns = _column_names(conn, "assets")
    if "source" in columns:
        return
    allowed = ",".join(f"'{value}'" for value in _ALLOWED_SOURCES)
    conn.execute(
        f"""
        ALTER TABLE assets
        ADD COLUMN source TEXT NOT NULL DEFAULT 'telegram'
        CHECK (source IN ({allowed}))
        """
    )
    conn.execute(
        "UPDATE assets SET source='mobile' WHERE upload_id IS NOT NULL"
    )


def run(conn: sqlite3.Connection) -> None:
    _ensure_uploads_source(conn)
    _ensure_assets_source(conn)
