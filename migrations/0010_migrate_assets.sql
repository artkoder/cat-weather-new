BEGIN;

CREATE TABLE IF NOT EXISTS asset_images (
    message_id INTEGER PRIMARY KEY,
    hashtags TEXT,
    template TEXT,
    used_at TEXT
);

WITH
    channel AS (
        SELECT channel_id
        FROM asset_channel
        ORDER BY rowid
        LIMIT 1
    ),
    src AS (
        SELECT
            ai.message_id,
            ai.template,
            ai.hashtags,
            TRIM(COALESCE(ai.hashtags, '')) AS cleaned
        FROM asset_images ai
    ),
    tokens(message_id, rest, token, idx) AS (
        SELECT
            message_id,
            CASE
                WHEN cleaned = '' THEN ''
                WHEN INSTR(cleaned, ' ') = 0 THEN ''
                ELSE LTRIM(SUBSTR(cleaned, INSTR(cleaned || ' ', ' ')))
            END AS rest,
            CASE
                WHEN cleaned = '' THEN NULL
                WHEN INSTR(cleaned, ' ') = 0 THEN cleaned
                ELSE SUBSTR(cleaned, 1, INSTR(cleaned || ' ', ' ') - 1)
            END AS token,
            0 AS idx
        FROM src
        UNION ALL
        SELECT
            message_id,
            CASE
                WHEN rest = '' THEN ''
                WHEN INSTR(rest, ' ') = 0 THEN ''
                ELSE LTRIM(SUBSTR(rest, INSTR(rest || ' ', ' ')))
            END,
            CASE
                WHEN rest = '' THEN NULL
                WHEN INSTR(rest, ' ') = 0 THEN rest
                ELSE SUBSTR(rest, 1, INSTR(rest || ' ', ' ') - 1)
            END,
            idx + 1
        FROM tokens
        WHERE rest <> ''
    ),
    aggregated AS (
        SELECT
            message_id,
            COALESCE(
                (
                    SELECT json_group_array(token)
                    FROM (
                        SELECT token
                        FROM tokens t2
                        WHERE t2.message_id = t.message_id
                          AND t2.token IS NOT NULL
                          AND t2.token <> ''
                        ORDER BY t2.idx
                    )
                ),
                json('[]')
            ) AS categories
        FROM tokens t
        GROUP BY message_id
    )
INSERT INTO assets (
    channel_id,
    message_id,
    caption_template,
    hashtags,
    categories,
    created_at,
    updated_at
)
SELECT
    (SELECT channel_id FROM channel),
    src.message_id,
    src.template,
    src.hashtags,
    COALESCE(aggregated.categories, json('[]')),
    datetime('now'),
    datetime('now')
FROM src
LEFT JOIN aggregated USING (message_id)
WHERE src.message_id IS NOT NULL
  AND NOT EXISTS (
        SELECT 1 FROM assets WHERE message_id = src.message_id
    );

DROP TABLE IF EXISTS asset_images;
COMMIT;
