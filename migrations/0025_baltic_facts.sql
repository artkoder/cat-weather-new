-- Migration 0025: baltic facts usage tracking

CREATE TABLE IF NOT EXISTS facts_usage (
    fact_id TEXT PRIMARY KEY,
    last_used_at INTEGER,
    uses_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS facts_rollout (
    day_utc INTEGER PRIMARY KEY,
    fact_id TEXT
);
