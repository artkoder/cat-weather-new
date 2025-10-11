BEGIN;

CREATE TABLE IF NOT EXISTS devices (
    device_id TEXT PRIMARY KEY,
    secret TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_seen_at TEXT,
    revoked_at TEXT
);

CREATE TABLE IF NOT EXISTS pairing_tokens (
    token TEXT PRIMARY KEY,
    created_by TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used_at TEXT
);

CREATE TABLE IF NOT EXISTS nonces (
    nonce TEXT PRIMARY KEY,
    ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS uploads (
    upload_id TEXT PRIMARY KEY,
    device_id TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    mime TEXT NOT NULL,
    size INTEGER NOT NULL,
    received_at TEXT NOT NULL,
    processed_at TEXT,
    status TEXT NOT NULL,
    error TEXT,
    FOREIGN KEY (device_id) REFERENCES devices(device_id)
);

CREATE INDEX IF NOT EXISTS idx_uploads_content_sha256 ON uploads(content_sha256);
CREATE INDEX IF NOT EXISTS idx_uploads_device_id ON uploads(device_id);

COMMIT;
