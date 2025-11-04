-- Migration 0023: add shot metadata for assets and extended wind columns for sea conditions

ALTER TABLE assets ADD COLUMN shot_at_utc INTEGER;
ALTER TABLE assets ADD COLUMN shot_doy INTEGER;

CREATE INDEX IF NOT EXISTS idx_assets_shot_doy ON assets(shot_doy);

ALTER TABLE sea_conditions ADD COLUMN wind_speed_10m_kmh REAL;
ALTER TABLE sea_conditions ADD COLUMN wind_gusts_10m_ms REAL;
ALTER TABLE sea_conditions ADD COLUMN wind_gusts_10m_kmh REAL;
ALTER TABLE sea_conditions ADD COLUMN wind_units TEXT;
ALTER TABLE sea_conditions ADD COLUMN wind_gusts_units TEXT;
ALTER TABLE sea_conditions ADD COLUMN wind_time_ref TEXT;
