CREATE TABLE IF NOT EXISTS amber_channels (
    channel_id INTEGER PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS amber_state (
    sea_id INTEGER PRIMARY KEY,
    storm_start TEXT,
    active INTEGER DEFAULT 0
);
