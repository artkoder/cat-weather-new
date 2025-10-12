from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Iterable


TARGET_COLUMNS: tuple[str, ...] = (
    "id",
    "upload_id",
    "file_ref",
    "content_type",
    "sha256",
    "width",
    "height",
    "exif_json",
    "labels_json",
    "tg_message_id",
    "created_at",
)

PAYLOAD_COLUMN_CANDIDATES: tuple[str, ...] = ("payload_json", "metadata_json")

LEGACY_REQUIRED_COLUMNS: tuple[str, ...] = (
    "channel_id",
    "tg_chat_id",
    "message_id",
    "file_id",
    "mime_type",
)

MAPPED_LEGACY_COLUMNS: frozenset[str] = frozenset(
    {
        "id",
        "file_id",
        "mime_type",
        "width",
        "height",
        "categories",
        "tg_chat_id",
        "message_id",
        "created_at",
        "sha256",
        "exif_json",
    }
)


def _table_info(conn: sqlite3.Connection, table: str) -> list[sqlite3.Row]:
    cursor = conn.execute(f"PRAGMA table_info('{table}')")
    return cursor.fetchall()


def _column_names(columns: Iterable[sqlite3.Row]) -> list[str]:
    names: list[str] = []
    for column in columns:
        try:
            name = column["name"]  # type: ignore[index]
        except (KeyError, TypeError):
            name = column[1]
        names.append(str(name))
    return names


def _has_target_layout(columns: list[sqlite3.Row]) -> bool:
    if not columns:
        return False
    column_names = _column_names(columns)
    missing = [name for name in TARGET_COLUMNS if name not in column_names]
    if missing:
        return False
    extras = [name for name in column_names if name not in TARGET_COLUMNS]
    for extra in extras:
        if extra not in PAYLOAD_COLUMN_CANDIDATES:
            return False
    return True


def _looks_like_legacy(columns: list[sqlite3.Row]) -> bool:
    if not columns:
        return False
    column_names = set(_column_names(columns))
    return all(name in column_names for name in LEGACY_REQUIRED_COLUMNS)


def _ensure_new_assets_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS assets (
            id TEXT PRIMARY KEY,
            upload_id TEXT REFERENCES uploads(id) ON DELETE CASCADE,
            file_ref TEXT,
            content_type TEXT,
            sha256 TEXT,
            width INTEGER,
            height INTEGER,
            exif_json TEXT,
            labels_json TEXT,
            tg_message_id TEXT,
            payload_json TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_assets_upload_id ON assets(upload_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_assets_created_at ON assets(created_at)"
    )


def _build_tg_message_id(data: dict[str, object]) -> str | None:
    chat_id = data.get("tg_chat_id")
    message_id = data.get("message_id")
    if chat_id is None and message_id is None:
        return None
    if chat_id is not None and message_id is not None:
        return f"{chat_id}:{message_id}"
    value = chat_id if chat_id is not None else message_id
    return str(value) if value is not None else None


def _extract_labels_json(data: dict[str, object]) -> str | None:
    raw = data.get("categories")
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(raw)
    except TypeError:
        return None


def _legacy_created_at(data: dict[str, object]) -> str:
    value = data.get("created_at") or data.get("updated_at")
    if isinstance(value, str) and value.strip():
        return value
    return datetime.utcnow().isoformat()


def _payload_from_legacy(data: dict[str, object]) -> str | None:
    payload: dict[str, object] = {}
    for key, value in data.items():
        if key in MAPPED_LEGACY_COLUMNS:
            continue
        if value is None:
            continue
        payload[key] = value
    if not payload:
        return None
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except TypeError:
        safe_payload: dict[str, object] = {}
        for key, value in payload.items():
            try:
                json.dumps(value)
                safe_payload[key] = value
            except TypeError:
                safe_payload[key] = str(value)
        return json.dumps(safe_payload, ensure_ascii=False, sort_keys=True)


def _copy_legacy_assets(conn: sqlite3.Connection) -> None:
    cursor = conn.execute("SELECT * FROM assets_legacy")
    for row in cursor:
        row_dict = {key: row[key] for key in row.keys()}
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
                payload_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(row_dict.get("id")),
                None,
                row_dict.get("file_id"),
                row_dict.get("mime_type"),
                row_dict.get("sha256"),
                row_dict.get("width"),
                row_dict.get("height"),
                row_dict.get("exif_json"),
                _extract_labels_json(row_dict),
                _build_tg_message_id(row_dict),
                _payload_from_legacy(row_dict),
                _legacy_created_at(row_dict),
            ),
        )


def run(conn: sqlite3.Connection) -> None:
    columns = _table_info(conn, "assets")
    if _has_target_layout(columns):
        with conn:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_assets_upload_id ON assets(upload_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_assets_created_at ON assets(created_at)"
            )
        return

    if not columns:
        with conn:
            conn.execute("PRAGMA foreign_keys=ON")
            _ensure_new_assets_table(conn)
        return

    if not _looks_like_legacy(columns):
        return

    old_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("ALTER TABLE assets RENAME TO assets_legacy")
            _ensure_new_assets_table(conn)
            _copy_legacy_assets(conn)
            conn.execute("DROP TABLE assets_legacy")
    finally:
        conn.row_factory = old_factory
