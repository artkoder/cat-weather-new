BEGIN;

ALTER TABLE assets ADD COLUMN origin TEXT NOT NULL DEFAULT 'weather';

CREATE TABLE IF NOT EXISTS recognition_channel (
    channel_id INTEGER PRIMARY KEY
);

INSERT OR IGNORE INTO recognition_channel (channel_id)
SELECT channel_id FROM asset_channel LIMIT 1;

COMMIT;
