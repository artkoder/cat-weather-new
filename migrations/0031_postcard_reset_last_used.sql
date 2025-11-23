-- Reset the last_used_at flag for postcard-quality assets.
-- This restores photos that were marked as used during test runs.
UPDATE assets
SET payload_json = json_remove(payload_json, '$.last_used_at')
WHERE postcard_score >= 7;
