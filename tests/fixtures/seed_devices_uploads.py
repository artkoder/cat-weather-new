from __future__ import annotations

import argparse
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

from main import apply_migrations


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Apply migrations before seeding data."""

    apply_migrations(conn)


def _insert_devices(conn: sqlite3.Connection) -> list[str]:
    now = datetime.utcnow()
    devices = [
        {
            "device_id": "device-alpha",
            "user_id": 1,
            "name": "Front door",
            "secret": secrets.token_hex(32),
            "created_at": now - timedelta(days=2),
            "last_seen_at": now - timedelta(hours=1),
            "revoked_at": None,
        },
        {
            "device_id": "device-beta",
            "user_id": 2,
            "name": "Garden",
            "secret": secrets.token_hex(32),
            "created_at": now - timedelta(days=1, hours=3),
            "last_seen_at": None,
            "revoked_at": None,
        },
    ]
    for entry in devices:
        conn.execute(
            """
            INSERT INTO devices (id, user_id, name, secret, created_at, last_seen_at, revoked_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                user_id=excluded.user_id,
                name=excluded.name,
                secret=excluded.secret,
                last_seen_at=excluded.last_seen_at,
                revoked_at=excluded.revoked_at
            """,
            (
                entry["device_id"],
                entry["user_id"],
                entry["name"],
                entry["secret"],
                entry["created_at"].isoformat(),
                entry["last_seen_at"].isoformat() if entry["last_seen_at"] else None,
                entry["revoked_at"].isoformat() if entry["revoked_at"] else None,
            ),
        )
    conn.commit()
    return [entry["device_id"] for entry in devices]


def _insert_pairing_tokens(conn: sqlite3.Connection, device_ids: Iterable[str]) -> None:
    now = datetime.utcnow()
    for idx, device_id in enumerate(device_ids, start=1):
        conn.execute(
            """
            INSERT INTO pairing_tokens (code, user_id, device_name, created_at, expires_at, used_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(code) DO UPDATE SET
                user_id=excluded.user_id,
                device_name=excluded.device_name,
                expires_at=excluded.expires_at,
                used_at=excluded.used_at
            """,
            (
                f"pair-{idx}",
                idx,
                f"Device {idx}",
                now.isoformat(),
                (now + timedelta(hours=idx)).isoformat(),
                now.isoformat() if idx == 1 else None,
            ),
        )


def _insert_nonces(conn: sqlite3.Connection, device_ids: Iterable[str]) -> None:
    now = datetime.utcnow()
    for idx, device_id in enumerate(device_ids, start=1):
        conn.execute(
            """
            INSERT INTO nonces (id, device_id, value, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                device_id=excluded.device_id,
                value=excluded.value,
                created_at=excluded.created_at,
                expires_at=excluded.expires_at
            """,
            (
                f"nonce-{idx}",
                device_id,
                f"seed-nonce-{idx}",
                (now - timedelta(minutes=idx * 2)).isoformat(),
                (now + timedelta(minutes=10)).isoformat(),
            ),
        )


def _insert_uploads(conn: sqlite3.Connection, device_ids: Iterable[str]) -> list[str]:
    now = datetime.utcnow()
    upload_ids: list[str] = []
    for idx, device_id in enumerate(device_ids, start=1):
        conn.execute(
            """
            INSERT INTO uploads (
                id,
                device_id,
                idempotency_key,
                status,
                error,
                file_ref,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                device_id=excluded.device_id,
                idempotency_key=excluded.idempotency_key,
                status=excluded.status,
                error=excluded.error,
                file_ref=excluded.file_ref,
                created_at=excluded.created_at,
                updated_at=excluded.updated_at
            """,
            (
                f"upload-{idx}",
                device_id,
                f"idem-{idx}",
                "done" if idx % 2 == 0 else "queued",
                None,
                f"file-{idx}",
                (now - timedelta(minutes=idx * 5)).isoformat(),
                (now - timedelta(minutes=idx * 3)).isoformat(),
            ),
        )
        upload_ids.append(f"upload-{idx}")
    return upload_ids


def _insert_assets(conn: sqlite3.Connection, upload_ids: Iterable[str]) -> None:
    now = datetime.utcnow()
    for idx, upload_id in enumerate(upload_ids, start=1):
        conn.execute(
            """
            INSERT INTO assets (
                id,
                upload_id,
                file_ref,
                content_type,
                sha256,
                width,
                height,
                exif_json,
                labels_json,
                tg_message_id,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                upload_id=excluded.upload_id,
                file_ref=excluded.file_ref,
                content_type=excluded.content_type,
                sha256=excluded.sha256,
                width=excluded.width,
                height=excluded.height,
                exif_json=excluded.exif_json,
                labels_json=excluded.labels_json,
                tg_message_id=excluded.tg_message_id,
                created_at=excluded.created_at
            """,
            (
                f"asset-{idx}",
                upload_id,
                f"file-{idx}",
                "image/jpeg",
                f"sha256-{idx:02d}",
                1024 + idx,
                768 + idx,
                None,
                None,
                f"{1000 + idx}",
                (now - timedelta(minutes=idx)).isoformat(),
            ),
        )


def seed(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        _ensure_schema(conn)
        device_ids = _insert_devices(conn)
        _insert_pairing_tokens(conn, device_ids)
        _insert_nonces(conn, device_ids)
        upload_ids = _insert_uploads(conn, device_ids)
        _insert_assets(conn, upload_ids)
        conn.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed device pairing/upload demo data.")
    parser.add_argument(
        "db",
        nargs="?",
        default="/tmp/cat-weather-demo.db",
        help="Path to the SQLite database file (default: %(default)s)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed(Path(args.db))
    print(f"Seeded demo data into {args.db}")
