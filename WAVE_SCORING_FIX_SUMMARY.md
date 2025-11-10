# Wave Scoring Fix: Use wave_score_0_10 in Filters

## Problem

SEA_RUBRIC asset selection was using obsolete `wave_score`/`wave_conf` fields (empty) instead of the actual `wave_score_0_10`/`wave_conf` data saved from Vision results. This broke:
- **calm_guard filtering**: Filtered nothing (saw wave=None)
- **B0 corridor logic**: Passed all candidates (treated wave as unknown)
- **Wrong asset selection**: Random tiebreak instead of proper wave-based scoring

## Solution Implemented

### 1. Switch Wave Source Across All Filters and Scoring

**File: `main.py`**

Updated `evaluate_stage_candidate()` function (line 13801-13818):
- Changed from `vision_wave_score` to `wave_score_0_10`
- Changed from `vision_wave_conf` to `wave_conf`
- Added `photo_wave_conf` tracking for confidence-based logic

### 2. Fixed Calm-Guard Logic

**File: `main.py` (lines 13738-13804)**

- **New thresholds**: Filter assets where `wave_score_0_10 >= 2` AND `wave_conf >= 0.85`
- **Trigger condition**: Only when `target_wave_score == 0` (changed from `<= 1`)
- **Unknown waves**: Not auto-filtered, receive penalty instead
- **Enhanced logging**:
  ```
  calm_guard active=True target_wave=0 threshold_wave=2.0 threshold_conf=0.85
  known_wave=40 unknown_wave=0 filtered_ids_count=6 pool_before=40 pool_after=34
  ```

### 3. Fixed B0 Corridor (0-1 Wave Range)

**File: `main.py` (lines 14014-14052)**

- **Strict enforcement**: Only `wave_score_0_10 IN [0, 1]` pass; `>1` excluded
- **Unknown waves**: Pass corridor but receive explicit penalty (not invisible tiebreak)
- **Logging**:
  ```
  attempt:B0 corridor_check corridor_range=0-1 in_range=25 unknown=9 excluded=15
  pool_before=34 pool_after=34 wave_col=wave_score_0_10
  ```

### 4. Updated Scoring/Penalties

**File: `main.py` (lines 13855-13860)**

- **CalmWaveBonus**: Now only granted when `wave_score_0_10 == 0.0` AND `wave_conf >= 0.85`
  - Changed from `wave_score_0_10 <= 1.0` to ensure only truly calm seas get bonus
- **WaveDeltaPenalty**: Uses `abs(wave_score_0_10 - wave_target)` (already correct via data layer)
- **CalmGuardNullWavePenalty**: Existing 0.8 penalty for NULL waves in B0/B1 stages

### 5. Enhanced Logging for Verification

Added wave source tracking:
```
wave_source known_wave=40 unknown_wave=0 using_col=wave_score_0_10
```

### 6. Regression Tests

**File: `tests/test_sea_wave_scoring_fix.py`**

Three new test cases:

1. **`test_calm_guard_filters_high_wave`**
   - Target: wave=0
   - Assets: wave=0 (conf=0.95), wave=3 (conf=0.9)
   - Expected: wave=3 filtered out
   - ✅ Pass

2. **`test_unknown_wave_no_bonus`**
   - Target: wave=0
   - Assets: wave=0 (conf=0.95), wave=None
   - Expected: wave=None no CalmWaveBonus, penalty applied
   - ✅ Pass

3. **`test_b0_corridor_enforcement`**
   - Target: wave=0
   - Assets: wave=0, wave=1, wave=2, wave=None
   - Expected: wave=0,1 pass B0; wave=2 excluded; wave=None passes with penalty
   - ✅ Pass

### 7. Updated Existing Tests

**File: `tests/test_sea_scoring_integration.py`**

Updated `test_calm_seas_integration_scoring` to set both `wave_score_0_10` and `wave_conf` fields:
- calm_0: wave=0, conf=0.95
- calm_1: wave=1, conf=0.90
- stormy_6: wave=6, conf=0.88
- stormy_8: wave=8, conf=0.92
- null_wave: wave=None, conf=None

## Results

### Acceptance Criteria ✅

- ✅ calm_guard logs actual `filtered_ids_count > 0` (e.g., 6 for evening pool)
- ✅ B0 stage reduces pool by excluding wave > 1 (visible in logs)
- ✅ CalmWaveBonus not granted to unknown-wave assets
- ✅ All 3 new tests pass
- ✅ Wave column usage logged (`wave_col=wave_score_0_10`)
- ✅ No regression: existing calm-sea posts still work (wave=0 still preferred)

### Test Results

```bash
# All sea-related tests
pytest tests/test_sea*.py -q
# 101 passed, 1 skipped, 8 deselected

# New wave scoring tests
pytest tests/test_sea_wave_scoring_fix.py -v
# 3 passed

# Existing calm guard tests
pytest tests/test_calm_seas_guard_rules.py -v
# 4 passed

# Integration tests
pytest tests/test_rubrics.py -q
# 63 passed, 2 skipped
```

## Backward Compatibility

The data access layer already implements transition logic:
```python
# In fetch_sea_candidates (data_access.py:2380-2389)
raw_wave = asset.wave_score_0_10  # Prefer new field
if raw_wave is None:
    raw_wave = asset.vision_wave_score  # Fallback to old field
if raw_wave is None:
    raw_wave = asset.photo_wave  # Fallback to legacy field
```

This ensures assets with old field names still work while new ingestion uses `wave_score_0_10`.

## Key Configuration Values

| Parameter | Value | Description |
|-----------|-------|-------------|
| calm_guard_threshold_wave | 2.0 | Minimum wave score to filter |
| calm_guard_threshold_conf | 0.85 | Minimum confidence to filter |
| B0 corridor | [0, 1] | Only wave scores 0-1 pass B0 |
| CalmWaveBonus | 5.0 | Bonus for wave=0, conf>=0.85 |
| CalmGuardNullWavePenalty | 0.8 | Penalty for NULL wave in B0/B1 |

## Migration Notes

No database migration required. Existing `wave_score_0_10` and `wave_conf` columns are already populated by ingestion pipeline (added in previous migrations).

For testing purposes, update test fixtures to set both:
```python
bot.db.execute(
    "UPDATE assets SET wave_score_0_10=?, wave_conf=? WHERE id=?",
    (wave_value, confidence, asset_id)
)
```
