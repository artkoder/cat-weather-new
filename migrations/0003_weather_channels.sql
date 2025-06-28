CREATE TABLE IF NOT EXISTS asset_images (
    message_id INTEGER PRIMARY KEY,
    hashtags TEXT,
    template TEXT,
    used_at TEXT
);

CREATE TABLE IF NOT EXISTS weather_publish_channels (
    channel_id INTEGER PRIMARY KEY,
    post_time TEXT NOT NULL,
    last_published_at TEXT
);
