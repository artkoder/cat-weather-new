from __future__ import annotations

import sqlite3


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cursor = conn.execute(
        "SELECT 1 FROM pragma_table_info(?) WHERE name=?",
        (table, column),
    )
    if cursor.fetchone() is not None:
        return True
    fallback = conn.execute(f"PRAGMA table_info('{table}')")
    return any(str(row[1]) == column for row in fallback.fetchall())


def run(conn: sqlite3.Connection) -> None:
    """Add postcard_score column to assets table if missing."""

    if _column_exists(conn, "assets", "postcard_score"):
        return

    conn.execute("ALTER TABLE assets ADD COLUMN postcard_score INTEGER")
    conn.commit()
