from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from data_access import POSTCARD_RUNTIME_RESTORE_GUARD, DataAccess


def _prepare_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE assets (
            id TEXT PRIMARY KEY,
            payload_json TEXT,
            created_at TEXT NOT NULL,
            source TEXT,
            postcard_score INTEGER,
            captured_at TEXT,
            photo_doy INTEGER
        );

        CREATE TABLE schema_migrations (
            id TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    return conn


def test_data_access_restores_postcard_last_used(tmp_path: Path) -> None:
    conn = _prepare_connection(tmp_path / "restore.db")

    conn.execute(
        """
        INSERT INTO assets (id, payload_json, created_at, source, postcard_score, captured_at, photo_doy)
        VALUES (?, ?, ?, 'telegram', ?, NULL, NULL)
        """,
        (
            "restore-me",
            json.dumps(
                {
                    "city": "Test",
                    "last_used_at": "2024-10-01T12:00:00+00:00",
                },
                ensure_ascii=False,
            ),
            "2024-11-20T00:00:00+00:00",
            7,
        ),
    )
    conn.execute(
        """
        INSERT INTO assets (id, payload_json, created_at, source, postcard_score, captured_at, photo_doy)
        VALUES (?, ?, ?, 'telegram', ?, NULL, NULL)
        """,
        (
            "keep-me",
            json.dumps(
                {
                    "city": "Keep",
                    "last_used_at": "2024-09-15T09:30:00+00:00",
                },
                ensure_ascii=False,
            ),
            "2024-11-20T00:00:00+00:00",
            9,
        ),
    )
    conn.commit()

    DataAccess(conn)

    restore_row = conn.execute("SELECT payload_json FROM assets WHERE id=?", ("restore-me",)).fetchone()
    assert restore_row is not None
    restored_payload = json.loads(restore_row["payload_json"])
    assert "last_used_at" not in restored_payload
    assert "updated_at" in restored_payload
    first_updated_at = restored_payload["updated_at"]

    keep_row = conn.execute("SELECT payload_json FROM assets WHERE id=?", ("keep-me",)).fetchone()
    assert keep_row is not None
    keep_payload = json.loads(keep_row["payload_json"])
    assert keep_payload["last_used_at"] == "2024-09-15T09:30:00+00:00"

    guard_entry = conn.execute(
        "SELECT 1 FROM schema_migrations WHERE id=?",
        (POSTCARD_RUNTIME_RESTORE_GUARD,),
    ).fetchone()
    assert guard_entry is not None

    DataAccess(conn)
    restore_repeat = conn.execute("SELECT payload_json FROM assets WHERE id=?", ("restore-me",)).fetchone()
    assert restore_repeat is not None
    repeat_payload = json.loads(restore_repeat["payload_json"])
    assert repeat_payload["updated_at"] == first_updated_at
