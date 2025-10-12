PRAGMA foreign_keys=ON;

DROP TABLE IF EXISTS assets;

CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY,
    upload_id TEXT NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
    file_ref TEXT,
    content_type TEXT,
    sha256 TEXT,
    width INTEGER,
    height INTEGER,
    exif_json TEXT,
    labels_json TEXT,
    tg_message_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_assets_upload_id ON assets(upload_id);
CREATE INDEX IF NOT EXISTS idx_assets_created_at ON assets(created_at);
