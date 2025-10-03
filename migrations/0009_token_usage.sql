BEGIN;

CREATE TABLE IF NOT EXISTS token_usage_new (
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
    model,
    prompt_tokens,
    completion_tokens,
    total_tokens,
    job_id,
    request_id,
    timestamp
)
SELECT
    model,
    prompt_tokens,
    completion_tokens,
    total_tokens,
    job_id,
    NULL,
    COALESCE(created_at, datetime('now'))
FROM token_usage;

DROP TABLE IF EXISTS token_usage;
ALTER TABLE token_usage_new RENAME TO token_usage;

COMMIT;
