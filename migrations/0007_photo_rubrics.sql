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
    message_id INTEGER NOT NULL UNIQUE,
    caption_template TEXT,
    hashtags TEXT,
    categories TEXT,
    recognized_message_id INTEGER,
    metadata TEXT,
    latitude REAL,
    longitude REAL,
    city TEXT,
    country TEXT,
    last_used_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

ALTER TABLE assets ADD COLUMN rubric_id INTEGER;

CREATE INDEX IF NOT EXISTS idx_assets_message ON assets(message_id);

CREATE TABLE IF NOT EXISTS vision_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id INTEGER NOT NULL,
    provider TEXT,
    status TEXT,
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

CREATE TABLE IF NOT EXISTS weather_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id INTEGER NOT NULL UNIQUE,
    post_time TEXT NOT NULL,
    run_at TEXT NOT NULL,
    last_run_at TEXT,
    failures INTEGER DEFAULT 0,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

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

DROP TABLE IF EXISTS asset_history;

CREATE TABLE IF NOT EXISTS token_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    job_name TEXT,
    job_id INTEGER,
    asset_id INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    job_name TEXT,
    job_id INTEGER,
    asset_id INTEGER,
    created_at TEXT NOT NULL
);

INSERT INTO token_usage (
    model,
    prompt_tokens,
    completion_tokens,
    total_tokens,
    job_name,
    job_id,
    asset_id,
    created_at
)
SELECT
    model,
    prompt_tokens,
    completion_tokens,
    total_tokens,
    job_name,
    job_id,
    asset_id,
    created_at
FROM ai_usage;

DROP TABLE IF EXISTS ai_usage;

COMMIT;
