PRAGMA foreign_keys=OFF;

BEGIN;

CREATE TABLE assets_new (
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

INSERT INTO assets_new (
    id,
    channel_id,
    tg_chat_id,
    message_id,
    recognized_message_id,
    caption_template,
    caption,
    hashtags,
    categories,
    kind,
    file_id,
    file_unique_id,
    file_name,
    mime_type,
    file_size,
    width,
    height,
    duration,
    exif_present,
    latitude,
    longitude,
    city,
    country,
    author_user_id,
    author_username,
    sender_chat_id,
    via_bot_id,
    forward_from_user,
    forward_from_chat,
    local_path,
    metadata,
    rubric_id,
    vision_category,
    vision_arch_view,
    vision_photo_weather,
    vision_flower_varieties,
    vision_confidence,
    vision_caption,
    last_used_at,
    created_at,
    updated_at
)
SELECT
    id,
    channel_id,
    COALESCE(CAST(json_extract(metadata, '$.chat_id') AS INTEGER), channel_id, 0) AS tg_chat_id,
    message_id,
    recognized_message_id,
    caption_template,
    COALESCE(json_extract(metadata, '$.caption'), caption_template) AS caption,
    hashtags,
    COALESCE(categories, '[]') AS categories,
    json_extract(metadata, '$.kind') AS kind,
    json_extract(metadata, '$.file.file_id') AS file_id,
    json_extract(metadata, '$.file.file_unique_id') AS file_unique_id,
    json_extract(metadata, '$.file.file_name') AS file_name,
    json_extract(metadata, '$.file.mime_type') AS mime_type,
    json_extract(metadata, '$.file.file_size') AS file_size,
    json_extract(metadata, '$.file.width') AS width,
    json_extract(metadata, '$.file.height') AS height,
    json_extract(metadata, '$.file.duration') AS duration,
    CASE
        WHEN json_extract(metadata, '$.exif_present') IN (1, '1', 'true', 'True', 'TRUE') THEN 1
        ELSE 0
    END AS exif_present,
    latitude,
    longitude,
    city,
    country,
    CAST(json_extract(metadata, '$.author_user_id') AS INTEGER) AS author_user_id,
    json_extract(metadata, '$.author_username') AS author_username,
    CAST(json_extract(metadata, '$.sender_chat_id') AS INTEGER) AS sender_chat_id,
    CAST(json_extract(metadata, '$.via_bot_id') AS INTEGER) AS via_bot_id,
    CAST(json_extract(metadata, '$.forward_from_user') AS INTEGER) AS forward_from_user,
    CAST(json_extract(metadata, '$.forward_from_chat') AS INTEGER) AS forward_from_chat,
    json_extract(metadata, '$.local_path') AS local_path,
    CASE
        WHEN metadata IS NULL THEN NULL
        ELSE NULLIF(
            NULLIF(
                json_remove(
                    metadata,
                    '$.chat_id',
                    '$.message_id',
                    '$.caption',
                    '$.kind',
                    '$.author_user_id',
                    '$.author_username',
                    '$.sender_chat_id',
                    '$.via_bot_id',
                    '$.forward_from_user',
                    '$.forward_from_chat',
                    '$.file',
                    '$.local_path',
                    '$.exif_present',
                    '$.vision_caption'
                ),
                '{}'
            ),
            'null'
        )
    END AS metadata,
    rubric_id,
    vision_category,
    vision_arch_view,
    vision_photo_weather,
    vision_flower_varieties,
    vision_confidence,
    json_extract(metadata, '$.vision_caption') AS vision_caption,
    last_used_at,
    created_at,
    updated_at
FROM assets;

CREATE TABLE vision_results_new (
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

INSERT INTO vision_results_new (
    id,
    asset_id,
    provider,
    status,
    category,
    arch_view,
    photo_weather,
    flower_varieties,
    confidence,
    result_json,
    created_at,
    updated_at
)
SELECT
    id,
    asset_id,
    provider,
    status,
    category,
    arch_view,
    photo_weather,
    flower_varieties,
    confidence,
    result_json,
    created_at,
    updated_at
FROM vision_results;

CREATE TABLE jobs_queue_new (
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

INSERT INTO jobs_queue_new (
    id,
    name,
    payload,
    status,
    attempts,
    available_at,
    last_error,
    created_at,
    updated_at
)
SELECT id, name, payload, status, attempts, available_at, last_error, created_at, updated_at
FROM jobs_queue;

CREATE TABLE rubrics_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

INSERT INTO rubrics_new (id, code, title, description, created_at, updated_at)
SELECT id, code, title, description, created_at, updated_at FROM rubrics;

CREATE TABLE posts_history_new (
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

INSERT INTO posts_history_new (
    id,
    channel_id,
    message_id,
    asset_id,
    rubric_id,
    metadata,
    published_at,
    created_at
)
SELECT id, channel_id, message_id, asset_id, rubric_id, metadata, published_at, created_at
FROM posts_history;

CREATE TABLE token_usage_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    job_id INTEGER,
    request_id TEXT,
    timestamp TEXT NOT NULL
);

INSERT INTO token_usage_new (
    id,
    model,
    prompt_tokens,
    completion_tokens,
    total_tokens,
    job_id,
    request_id,
    timestamp
)
SELECT id, model, prompt_tokens, completion_tokens, total_tokens, job_id, request_id, timestamp
FROM token_usage;

DROP TABLE vision_results;
DROP TABLE assets;
DROP TABLE jobs_queue;
DROP TABLE rubrics;
DROP TABLE posts_history;
DROP TABLE token_usage;

ALTER TABLE assets_new RENAME TO assets;
ALTER TABLE vision_results_new RENAME TO vision_results;
ALTER TABLE jobs_queue_new RENAME TO jobs_queue;
ALTER TABLE rubrics_new RENAME TO rubrics;
ALTER TABLE posts_history_new RENAME TO posts_history;
ALTER TABLE token_usage_new RENAME TO token_usage;

CREATE INDEX idx_assets_message ON assets(tg_chat_id, message_id);
CREATE INDEX idx_assets_rubric ON assets(rubric_id);
CREATE INDEX idx_vision_results_asset ON vision_results(asset_id);
CREATE INDEX idx_jobs_queue_status ON jobs_queue(status, available_at);

COMMIT;

PRAGMA foreign_keys=ON;
