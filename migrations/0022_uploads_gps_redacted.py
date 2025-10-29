from __future__ import annotations

import sqlite3


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    cursor = conn.execute(f"PRAGMA table_info('{table}')")
    return {str(row[1]) for row in cursor.fetchall()}


def _ensure_gps_redacted_column(conn: sqlite3.Connection) -> None:
    columns = _column_names(conn, "uploads")
    if "gps_redacted_by_client" in columns:
        return
    conn.execute(
        """
        ALTER TABLE uploads
        ADD COLUMN gps_redacted_by_client INTEGER NOT NULL DEFAULT 0
        CHECK (gps_redacted_by_client IN (0,1))
        """
    )


def run(conn: sqlite3.Connection) -> None:
    _ensure_gps_redacted_column(conn)
