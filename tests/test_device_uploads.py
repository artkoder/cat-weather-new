import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_access import (
    create_device,
    create_pairing_token,
    consume_pairing_token,
    insert_upload,
    set_upload_status,
)
from main import apply_migrations


@pytest.fixture()
def conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    apply_migrations(connection)
    yield connection
    connection.close()


def _ensure_device(connection: sqlite3.Connection, device_id: str = "device-1") -> None:
    create_device(connection, device_id=device_id, user_id=42, name="Front door")


def _get_upload(connection: sqlite3.Connection, upload_id: str) -> sqlite3.Row:
    cur = connection.execute("SELECT * FROM uploads WHERE id=?", (upload_id,))
    row = cur.fetchone()
    assert row is not None
    return row


def test_insert_upload_enforces_idempotency_per_device(conn: sqlite3.Connection) -> None:
    _ensure_device(conn)
    first = insert_upload(
        conn,
        id="upload-1",
        device_id="device-1",
        idempotency_key="idem-42",
    )
    assert first == "upload-1"
    second = insert_upload(
        conn,
        id="upload-2",
        device_id="device-1",
        idempotency_key="idem-42",
    )
    assert second == "upload-1"
    count = conn.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
    assert count == 1


def test_pairing_token_expires(conn: sqlite3.Connection) -> None:
    create_pairing_token(
        conn,
        code="pair-1",
        user_id=99,
        device_name="Kitchen",
        ttl_sec=1,
    )
    expired_at = (datetime.utcnow() - timedelta(seconds=5)).isoformat()
    conn.execute(
        "UPDATE pairing_tokens SET expires_at=? WHERE code=?",
        (expired_at, "pair-1"),
    )
    conn.commit()
    assert consume_pairing_token(conn, code="pair-1") is None


def test_pairing_token_once(conn: sqlite3.Connection) -> None:
    create_pairing_token(
        conn,
        code="pair-2",
        user_id=7,
        device_name="Entry",
        ttl_sec=600,
    )
    payload = consume_pairing_token(conn, code="pair-2")
    assert payload == (7, "Entry")
    assert consume_pairing_token(conn, code="pair-2") is None


def test_upload_status_transitions(conn: sqlite3.Connection) -> None:
    _ensure_device(conn)
    upload_id = insert_upload(
        conn,
        id="upload-100",
        device_id="device-1",
        idempotency_key="key-100",
    )
    set_upload_status(conn, id=upload_id, status="processing")
    set_upload_status(conn, id=upload_id, status="done")
    with pytest.raises(ValueError):
        set_upload_status(conn, id=upload_id, status="processing")
    set_upload_status(conn, id=upload_id, status="done")

    failing_id = insert_upload(
        conn,
        id="upload-200",
        device_id="device-1",
        idempotency_key="key-200",
    )
    set_upload_status(conn, id=failing_id, status="processing")
    set_upload_status(conn, id=failing_id, status="failed", error="boom")
    row = _get_upload(conn, failing_id)
    assert row["status"] == "failed"
    assert row["error"] == "boom"
    with pytest.raises(ValueError):
        set_upload_status(conn, id=failing_id, status="queued")
