import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_access import DataAccess


def _setup_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE jobs_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            payload TEXT,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            available_at TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    return conn


def _insert_job(
    conn: sqlite3.Connection,
    *,
    status: str,
    rubric_code: str,
    schedule_key: str | None,
    available_at: datetime | None,
    name: str = "publish_rubric",
) -> None:
    now_iso = datetime.utcnow().isoformat()
    payload: dict[str, Any] = {"rubric_code": rubric_code}
    if schedule_key is not None:
        payload["schedule_key"] = schedule_key
    conn.execute(
        """
        INSERT INTO jobs_queue (
            name,
            payload,
            status,
            attempts,
            available_at,
            last_error,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, 0, ?, NULL, ?, ?)
        """,
        (
            name,
            json.dumps(payload),
            status,
            available_at.isoformat() if available_at else None,
            now_iso,
            now_iso,
        ),
    )


def test_delete_future_rubric_jobs_filters_manual_and_statuses() -> None:
    conn = _setup_connection()
    data = DataAccess(conn)

    now = datetime.utcnow()
    _insert_job(
        conn,
        status="queued",
        rubric_code="sea",
        schedule_key="sea:morning",
        available_at=None,
    )
    _insert_job(
        conn,
        status="delayed",
        rubric_code="sea",
        schedule_key="sea:evening",
        available_at=now + timedelta(hours=1),
    )
    _insert_job(
        conn,
        status="queued",
        rubric_code="sea",
        schedule_key="manual",
        available_at=None,
    )
    _insert_job(
        conn,
        status="delayed",
        rubric_code="sea",
        schedule_key="manual-test",
        available_at=now + timedelta(hours=2),
    )
    _insert_job(
        conn,
        status="running",
        rubric_code="sea",
        schedule_key="sea:running",
        available_at=None,
    )
    _insert_job(
        conn,
        status="queued",
        rubric_code="flowers",
        schedule_key="flowers:slot",
        available_at=None,
    )
    _insert_job(
        conn,
        status="queued",
        rubric_code="sea",
        schedule_key="sea:export",
        available_at=None,
        name="export_inventory",
    )
    conn.commit()

    deleted = data.delete_future_rubric_jobs("sea")
    assert deleted == 3

    remaining = conn.execute(
        """
        SELECT json_extract(payload, '$.schedule_key') AS schedule_key,
               json_extract(payload, '$.rubric_code') AS rubric_code,
               status,
               name
        FROM jobs_queue
        ORDER BY id
        """
    ).fetchall()
    sea_keys = {row["schedule_key"] for row in remaining if row["rubric_code"] == "sea"}
    assert sea_keys == {"manual", "manual-test", "sea:running"}
    assert any(row["rubric_code"] == "flowers" for row in remaining)
    assert any(row["name"] == "export_inventory" for row in remaining)

    deleted_manual = data.delete_future_rubric_jobs("sea", include_manual=True)
    assert deleted_manual == 2

    final_rows = conn.execute(
        """
        SELECT json_extract(payload, '$.rubric_code') AS rubric_code
        FROM jobs_queue
        ORDER BY id
        """
    ).fetchall()
    assert all(row["rubric_code"] != "sea" for row in final_rows)

    conn.close()
