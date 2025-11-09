-- Migration 0026: Enrich sea metadata with vision wave/sky columns

BEGIN;

-- Add new vision-derived columns to assets table
ALTER TABLE assets ADD COLUMN vision_wave_score REAL;
ALTER TABLE assets ADD COLUMN vision_wave_conf REAL;
ALTER TABLE assets ADD COLUMN vision_sky_bucket TEXT;

-- Create index for wave score filtering
CREATE INDEX IF NOT EXISTS idx_assets_wave_score ON assets(vision_wave_score);

-- Create index for sky bucket filtering
CREATE INDEX IF NOT EXISTS idx_assets_sky_bucket ON assets(vision_sky_bucket);

COMMIT;
