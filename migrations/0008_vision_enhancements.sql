BEGIN;

ALTER TABLE assets ADD COLUMN vision_category TEXT;
ALTER TABLE assets ADD COLUMN vision_arch_view TEXT;
ALTER TABLE assets ADD COLUMN vision_photo_weather TEXT;
ALTER TABLE assets ADD COLUMN vision_flower_varieties TEXT;
ALTER TABLE assets ADD COLUMN vision_confidence REAL;

ALTER TABLE vision_results ADD COLUMN category TEXT;
ALTER TABLE vision_results ADD COLUMN arch_view TEXT;
ALTER TABLE vision_results ADD COLUMN photo_weather TEXT;
ALTER TABLE vision_results ADD COLUMN flower_varieties TEXT;
ALTER TABLE vision_results ADD COLUMN confidence REAL;

COMMIT;
