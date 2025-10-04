BEGIN;

CREATE TABLE IF NOT EXISTS rubrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id INTEGER NOT NULL,
    tg_chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    recognized_message_id INTEGER,
    caption_template TEXT,
    caption TEXT,
    hashtags TEXT,
    categories TEXT NOT NULL DEFAULT '[]',
    kind TEXT,
    file_id TEXT,
    file_unique_id TEXT,
    file_name TEXT,
    mime_type TEXT,
    file_size INTEGER,
    width INTEGER,
    height INTEGER,
    duration INTEGER,
    exif_present INTEGER NOT NULL DEFAULT 0,
    latitude REAL,
    longitude REAL,
    city TEXT,
    country TEXT,
    author_user_id INTEGER,
    author_username TEXT,
    sender_chat_id INTEGER,
    via_bot_id INTEGER,
    forward_from_user INTEGER,
    forward_from_chat INTEGER,
    local_path TEXT,
    metadata TEXT,
    rubric_id INTEGER,
    vision_category TEXT,
    vision_arch_view TEXT,
    vision_photo_weather TEXT,
    vision_flower_varieties TEXT,
    vision_confidence REAL,
    vision_caption TEXT,
    last_used_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(rubric_id) REFERENCES rubrics(id) ON DELETE SET NULL,
    UNIQUE(tg_chat_id, message_id)
);

CREATE INDEX IF NOT EXISTS idx_assets_message ON assets(tg_chat_id, message_id);
CREATE INDEX IF NOT EXISTS idx_assets_rubric ON assets(rubric_id);

CREATE TABLE IF NOT EXISTS vision_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id INTEGER NOT NULL,
    provider TEXT,
    status TEXT,
    category TEXT,
    arch_view TEXT,
    photo_weather TEXT,
    flower_varieties TEXT,
    confidence REAL,
    result_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_vision_results_asset ON vision_results(asset_id);

CREATE TABLE IF NOT EXISTS jobs_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    payload TEXT,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    available_at TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_queue_status ON jobs_queue(status, available_at);

CREATE TABLE IF NOT EXISTS posts_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    asset_id INTEGER,
    rubric_id INTEGER,
    metadata TEXT,
    published_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE SET NULL,
    FOREIGN KEY(rubric_id) REFERENCES rubrics(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS token_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    job_id INTEGER,
    request_id TEXT,
    timestamp TEXT NOT NULL
);

COMMIT;
