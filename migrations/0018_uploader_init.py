from __future__ import annotations

import sqlite3

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- 1) Устройства
CREATE TABLE IF NOT EXISTS devices (
  id              TEXT PRIMARY KEY,
  user_id         INTEGER NOT NULL,
  name            TEXT NOT NULL,
  secret          TEXT NOT NULL,
  created_at      TEXT NOT NULL DEFAULT (datetime('now')),
  last_seen_at    TEXT,
  revoked_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_devices_user_id ON devices(user_id);

-- 2) Пейринг-коды (attach)
CREATE TABLE IF NOT EXISTS pairing_tokens (
  code            TEXT PRIMARY KEY,
  user_id         INTEGER NOT NULL,
  device_name     TEXT NOT NULL,
  created_at      TEXT NOT NULL DEFAULT (datetime('now')),
  expires_at      TEXT NOT NULL,
  used_at         TEXT
);
CREATE INDEX IF NOT EXISTS idx_pairing_tokens_expires ON pairing_tokens(expires_at);

-- 3) Нонсы (для анти-replay; связаны с устройством)
CREATE TABLE IF NOT EXISTS nonces (
  id              TEXT PRIMARY KEY,
  device_id       TEXT NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
  value           TEXT NOT NULL,
  created_at      TEXT NOT NULL DEFAULT (datetime('now')),
  expires_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_nonces_device ON nonces(device_id);
CREATE INDEX IF NOT EXISTS idx_nonces_expires ON nonces(expires_at);

-- 4) Загрузки
CREATE TABLE IF NOT EXISTS uploads (
  id              TEXT PRIMARY KEY,
  device_id       TEXT NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
  idempotency_key TEXT NOT NULL,
  status          TEXT NOT NULL CHECK (status IN ('queued','processing','failed','done')),
  error           TEXT,
  file_ref        TEXT,
  asset_id        TEXT,
  created_at      TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_uploads_device_idempotency
  ON uploads(device_id, idempotency_key);
CREATE INDEX IF NOT EXISTS idx_uploads_device ON uploads(device_id);
CREATE INDEX IF NOT EXISTS idx_uploads_status ON uploads(status);
"""

EXPECTED_COLUMNS = {
    "devices": {"id", "user_id", "name", "secret", "created_at", "last_seen_at", "revoked_at"},
    "pairing_tokens": {"code", "user_id", "device_name", "created_at", "expires_at", "used_at"},
    "nonces": {"id", "device_id", "value", "created_at", "expires_at"},
    "uploads": {
        "id",
        "device_id",
        "idempotency_key",
        "status",
        "error",
        "file_ref",
        "asset_id",
        "created_at",
        "updated_at",
    },
}


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    )
    return cur.fetchone() is not None


def run(conn: sqlite3.Connection) -> None:
    for table, expected in EXPECTED_COLUMNS.items():
        if not _table_exists(conn, table):
            continue
        columns = _table_columns(conn, table)
        if columns == expected:
            continue
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.executescript(DDL)
