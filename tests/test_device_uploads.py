import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_access import (
    DataAccess,
    consume_pairing_token,
    create_device,
    create_pairing_token,
    fetch_upload_record,
    insert_upload,
    link_upload_asset,
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
    create_device(
        connection,
        device_id=device_id,
        user_id=42,
        name="Front door",
        secret="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    )


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
    row = _get_upload(conn, first)
    assert row["status"] == "queued"
    count = conn.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
    assert count == 1
    row = conn.execute("SELECT source FROM uploads WHERE id=?", (first,)).fetchone()
    assert row is not None and row["source"] == "mobile"


def test_insert_upload_records_gps_flag(conn: sqlite3.Connection) -> None:
    _ensure_device(conn)
    upload_id = insert_upload(
        conn,
        id="upload-gps",
        device_id="device-1",
        idempotency_key="idem-gps",
        gps_redacted_by_client=True,
    )
    row = conn.execute(
        "SELECT gps_redacted_by_client FROM uploads WHERE id=?",
        (upload_id,),
    ).fetchone()
    assert row is not None
    assert row["gps_redacted_by_client"] == 1

    record = fetch_upload_record(conn, upload_id=upload_id)
    assert record is not None
    assert record["gps_redacted_by_client"] is True


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
    set_upload_status(conn, id=upload_id, status="processing")
    row = _get_upload(conn, upload_id)
    assert row["status"] == "processing"
    assert row["error"] is None
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
    set_upload_status(conn, id=failing_id, status="processing")
    row = _get_upload(conn, failing_id)
    assert row["status"] == "processing"
    assert row["error"] is None
    with pytest.raises(ValueError):
        set_upload_status(conn, id=failing_id, status="queued")


def test_link_upload_asset_updates_row(conn: sqlite3.Connection) -> None:
    _ensure_device(conn)
    upload_id = insert_upload(
        conn,
        id="upload-asset",
        device_id="device-1",
        idempotency_key="key-asset",
    )
    row = _get_upload(conn, upload_id)
    assert row["asset_id"] is None

    data = DataAccess(conn)
    asset_id = data.create_asset(
        upload_id=upload_id,
        file_ref="file:///tmp/test.jpg",
        content_type="image/jpeg",
        sha256="deadbeef",
        width=10,
        height=20,
    )

    asset = data.get_asset(asset_id)
    assert asset is not None and asset.source == "mobile"

    link_upload_asset(conn, upload_id=upload_id, asset_id=asset_id)
    row = _get_upload(conn, upload_id)
    assert row["asset_id"] == asset_id

    with pytest.raises(ValueError):
        link_upload_asset(conn, upload_id=upload_id, asset_id="different")


def test_asset_file_id_falls_back_to_file_ref(conn: sqlite3.Connection) -> None:
    _ensure_device(conn)
    upload_id = insert_upload(
        conn,
        id="upload-file-ref",
        device_id="device-1",
        idempotency_key="key-file-ref",
    )

    data = DataAccess(conn)
    file_ref = "file:///tmp/file-ref.jpg"
    asset_id = data.create_asset(
        upload_id=upload_id,
        file_ref=file_ref,
        content_type="image/jpeg",
        sha256="c0ffee",
        width=100,
        height=200,
    )

    asset = data.get_asset(asset_id)
    assert asset is not None
    assert asset.payload == {}
    assert asset.file_ref == file_ref
    assert asset.file_id == file_ref
    assert asset.source == "mobile"


def test_save_asset_defaults_to_telegram_source(conn: sqlite3.Connection) -> None:
    data = DataAccess(conn)
    asset_id = data.save_asset(
        channel_id=-100,
        message_id=42,
        template=None,
        hashtags="#cats",
        tg_chat_id=-100,
        caption="test",
        kind="photo",
    )
    asset = data.get_asset(asset_id)
    assert asset is not None
    assert asset.source == "telegram"
