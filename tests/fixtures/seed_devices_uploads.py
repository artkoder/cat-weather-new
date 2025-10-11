from __future__ import annotations

import argparse
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
            "secret_hash": "hash-alpha",
            "created_at": now - timedelta(days=2),
            "last_seen_at": now - timedelta(hours=1),
            "revoked_at": None,
        },
        {
            "device_id": "device-beta",
            "secret_hash": "hash-beta",
            "created_at": now - timedelta(days=1, hours=3),
            "last_seen_at": None,
            "revoked_at": None,
        },
    ]
    for entry in devices:
        conn.execute(
            """
            INSERT INTO devices (device_id, secret_hash, created_at, last_seen_at, revoked_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(device_id) DO UPDATE SET
                secret_hash=excluded.secret_hash,
                last_seen_at=excluded.last_seen_at,
                revoked_at=excluded.revoked_at
            """,
            (
                entry["device_id"],
                entry["secret_hash"],
                entry["created_at"].isoformat(),
                entry["last_seen_at"].isoformat() if entry["last_seen_at"] else None,
                entry["revoked_at"].isoformat() if entry["revoked_at"] else None,
            ),
        )
    return [entry["device_id"] for entry in devices]


def _insert_pairing_tokens(conn: sqlite3.Connection, device_ids: Iterable[str]) -> None:
    now = datetime.utcnow()
    for idx, device_id in enumerate(device_ids, start=1):
        conn.execute(
            """
            INSERT INTO pairing_tokens (token, created_by, expires_at, used_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(token) DO UPDATE SET
                created_by=excluded.created_by,
                expires_at=excluded.expires_at,
                used_at=excluded.used_at
            """,
            (
                f"pair-{idx}",
                device_id,
                (now + timedelta(hours=idx)).isoformat(),
                now.isoformat() if idx == 1 else None,
            ),
        )


def _insert_nonces(conn: sqlite3.Connection) -> None:
    now = datetime.utcnow()
    for idx in range(1, 4):
        conn.execute(
            """
            INSERT INTO nonces (nonce, ts)
            VALUES (?, ?)
            ON CONFLICT(nonce) DO UPDATE SET ts=excluded.ts
            """,
            (f"nonce-{idx}", (now - timedelta(minutes=idx)).isoformat()),
        )


def _insert_uploads(conn: sqlite3.Connection, device_ids: Iterable[str]) -> None:
    now = datetime.utcnow()
    for idx, device_id in enumerate(device_ids, start=1):
        conn.execute(
            """
            INSERT INTO uploads (
                upload_id,
                device_id,
                content_sha256,
                mime,
                size,
                received_at,
                processed_at,
                status,
                error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(upload_id) DO UPDATE SET
                device_id=excluded.device_id,
                content_sha256=excluded.content_sha256,
                mime=excluded.mime,
                size=excluded.size,
                received_at=excluded.received_at,
                processed_at=excluded.processed_at,
                status=excluded.status,
                error=excluded.error
            """,
            (
                f"upload-{idx}",
                device_id,
                f"sha256-{idx}",
                "image/jpeg",
                1024 * idx,
                (now - timedelta(minutes=idx * 5)).isoformat(),
                (now - timedelta(minutes=idx * 2)).isoformat() if idx % 2 == 0 else None,
                "processed" if idx % 2 == 0 else "pending",
                None,
            ),
        )


def seed(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        _ensure_schema(conn)
        device_ids = _insert_devices(conn)
        _insert_pairing_tokens(conn, device_ids)
        _insert_nonces(conn)
        _insert_uploads(conn, device_ids)
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
