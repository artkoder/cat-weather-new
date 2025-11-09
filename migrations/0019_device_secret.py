from __future__ import annotations

import secrets
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


def _generate_secret() -> str:
    # 32 bytes hex-encoded -> 64 characters as required by the spec.
    return secrets.token_hex(32)


def run(conn: sqlite3.Connection) -> None:
    """Ensure devices table stores the shared secret used for HMAC signatures."""

    if not _table_exists(conn, "devices"):
        return

    columns = _table_columns(conn, "devices")
    if "secret" in columns:
        return

    conn.execute("ALTER TABLE devices ADD COLUMN secret TEXT")

    rows = conn.execute("SELECT id FROM devices").fetchall()
    for row in rows:
        device_id = row["id"] if isinstance(row, sqlite3.Row) else row[0]
        conn.execute(
            "UPDATE devices SET secret=? WHERE id=?",
            (_generate_secret(), device_id),
        )

    conn.commit()
