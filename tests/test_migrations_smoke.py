from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import apply_migrations


def _table_names(conn: sqlite3.Connection) -> set[str]:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    return {row[0] for row in cur.fetchall()}


def _index_names(conn: sqlite3.Connection, table: str) -> set[str]:
    quoted = table.replace("'", "''")
    cur = conn.execute(f"PRAGMA index_list('{quoted}')")
    return {row[1] for row in cur.fetchall()}


def test_apply_migrations_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "migrations.db"

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        apply_migrations(conn)
        tables = _table_names(conn)
        expected_tables = {"devices", "pairing_tokens", "nonces", "uploads", "assets"}
        assert expected_tables.issubset(tables)
        upload_indexes = _index_names(conn, "uploads")
        assert "uq_uploads_device_idempotency" in upload_indexes
        assert "idx_uploads_device" in upload_indexes
        assert "idx_uploads_status" in upload_indexes
        asset_indexes = _index_names(conn, "assets")
        assert "idx_assets_upload_id" in asset_indexes
        assert "idx_assets_created_at" in asset_indexes

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        apply_migrations(conn)
        tables = _table_names(conn)
        expected_tables = {"devices", "pairing_tokens", "nonces", "uploads", "assets"}
        assert expected_tables.issubset(tables)
