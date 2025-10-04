BEGIN;

CREATE TABLE IF NOT EXISTS recognition_channel_new (
    channel_id INTEGER
);

INSERT INTO recognition_channel_new (channel_id)
SELECT CASE
    WHEN rc.channel_id IS NOT NULL AND rc.channel_id = ac.channel_id THEN NULL
    ELSE rc.channel_id
END
FROM recognition_channel rc
LEFT JOIN asset_channel ac ON 1=1
LIMIT 1;

DROP TABLE IF EXISTS recognition_channel;
ALTER TABLE recognition_channel_new RENAME TO recognition_channel;

COMMIT;
