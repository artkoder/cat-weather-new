BEGIN;

CREATE TABLE IF NOT EXISTS rubric_schedule_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rubric_code TEXT NOT NULL,
    schedule_key TEXT NOT NULL,
    next_run_at TEXT,
    last_run_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(rubric_code, schedule_key)
);

CREATE INDEX IF NOT EXISTS idx_rubric_schedule_state_code
    ON rubric_schedule_state(rubric_code, schedule_key);

COMMIT;
