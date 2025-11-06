# Sea Selection Rework - Implementation Summary

## Overview
This document summarizes the comprehensive rework of the sea selection logic as specified in the "Rework sea selection" ticket. All acceptance criteria have been met, with extensive logging, new test coverage, and backward-compatible changes.

## Changes Implemented

### 1. Seasonal Window with Kaliningrad Timezone
- **Change**: Seasonal window now uses `Europe/Kaliningrad` timezone (`zoneinfo.ZoneInfo`) instead of UTC for day-of-year calculations
- **Implementation**: `main.py` lines 13082-13084
- **Logging**: Enhanced season log includes:
  - `doy_now`: Current day of year in Kaliningrad timezone
  - `doy_range`: [low, high] boundaries of the ±45 day window with wrap-around
  - `kept`: List of asset IDs within season window
  - `removed`: List of asset IDs outside season window  
  - `null_doy`: List of asset IDs with missing shot_doy (now included)
  - `season_removed`: Boolean flag indicating if filter was released when no candidates remained

### 2. Wave Score Storm Classification
- **Change**: Storm states now classified by wave scores instead of raw wave heights
  - calm: score ≤ 3.5
  - storm: 3.5 < score < 6.0
  - strong_storm: score ≥ 6.0
- **Implementation**: `main.py` lines 13035-13040
- **Wave mapping**: Existing interpolation curve preserved (0.0→0.0, 0.5→3.0, 1.0→5.0, 1.5→7.0, 2.0→8.5, 3.0→10.0)

### 3. Corridor Width Computation
- **Change**: Derive corridors based on current `wave_height_m` with staged fallback
  - Calm: center=1.75, halfwidth=1.75+broaden
  - Storm: center=wave_score, halfwidth=1.0+broaden
  - Strong storm: center=9.0, halfwidth=1.0+broaden
- **Implementation**: `main.py` lines 13221-13233
- **Broadening**: +0.8 added for `wave+broaden` stage fallback

### 4. Penalty-Based Candidate Evaluation
- **Change**: New penalty system replaces old bonus-based scoring
  - Wave penalty: `max(0, |score−center|−halfwidth) * 0.75`
  - Calm-day bonus: -1.0 penalty when wave_score is missing on calm days
  - Sky penalty: varies based on match quality (-0.5 exact, 0.0 compatible, 1.0 mismatch)
  - Final score: `-penalty + age_bonus`
- **Implementation**: `main.py` lines 13235-13322
- **Missing scores**: Blocked for storm/strong_storm states until final fallback

### 5. Coastal Cloud Acceptance Matrix
- **Change**: Implemented B0/B1/B2/AN sky policy stages
  - **B0** (season): Strict - requires exact or compatible sky match with allowed_photo_skies
  - **B1**: Moderate - allows only assets in allowed_photo_skies
  - **B2**: Relaxed - blocks only overcast on clear days
  - **AN** (any): Accepts any sky
- **Implementation**: `main.py` lines 13274-13305
- **Clear guards**:
  - `clear_guard_hard`: cloud_cover_pct ≤ 10%
  - `clear_guard_soft`: cloud_cover_pct ≤ 20%
- **Sky mismatch tracking**: `sky_critical_mismatch=true` if hard-clear day falls through to AN with mostly_cloudy/overcast

### 6. Fallback Order
- **Change**: Implemented staged fallback progression
  1. **season**: B0 sky policy, no wave broadening, requires wave scores for storms
  2. **wave**: B0 sky policy, no wave broadening, allows missing wave scores
  3. **B1**: B1 sky policy, no wave broadening, allows missing wave scores
  4. **wave+broaden**: B1 sky policy, +0.8 wave broadening, allows missing wave scores
  5. **B2**: B2 sky policy, +0.8 wave broadening, allows missing wave scores
  6. **AN**: AN sky policy, +0.8 wave broadening, allows missing wave scores
- **Implementation**: `main.py` lines 13324-13331, 13338-13410
- **Pool tracking**: Each stage logs pool size after filtering

### 7. Sky Visibility Integration
- **Change**: New `sky_visible` field determines if sky penalties apply
  - Computed from `vision_results.sky_visible` or inferred from `photo_sky`
  - When `False` or `photo_sky="unknown"`, sky penalties are skipped entirely
  - Disables sunset prioritization for non-visible sky assets
- **Implementation**: 
  - `data_access.py` lines 1728-1732, 1761
  - `main.py` lines 13246, 13276-13305, 13439, 13442-13446

### 8. Reworked Sunset Logic
- **Change**: `want_sunset` now requires:
  - `storm_state != "strong_storm"`
  - `sky_visible = True`
  - NOT (`clear_guard_hard` AND `chosen_sky` in {mostly_cloudy, overcast})
- **Implementation**: `main.py` lines 13442-13446
- **Impact**: Prevents sunset preference on clear days with cloudy assets

### 9. Storm Persistence with New Classes
- **Change**: Existing 36h storm persistence logic updated to use new wave-score based storm states
- **Implementation**: Already compatible - `_is_storm_persisting` and `_find_recent_sea_storm_event` use storm_state strings
- **States recognized**: "calm", "storm", "strong_storm"

### 10. Comprehensive Logging
- **Change**: All sea selection events now emit Grafana-friendly JSON logs
- **Weather summary**: `weather` event with wave_height_m, wave_score, wind metrics, cloud_cover_pct, sky_bucket
- **Season window**: `season` event with doy_now, doy_range, kept/removed/null_doy asset lists
- **Stage progression**: `attempt <policy>` event per fallback stage (B0/B1/B2/AN) with pool_size, top5 scoring dump, wave_corridor
- **Top-5 dump**: Each stage logs top 5 candidates with id, wave, sky, penalties, score
- **Final selection**: `selected` event includes:
  - asset_id, shot_doy, score, wave_score, photo_sky, season_match
  - sunset_selected, want_sunset, storm_persisting
  - wave_corridor, sky_penalty, sky_visible, sky_critical_mismatch
  - pool_counts (per-stage pool sizes)
  - reasons (detailed scoring breakdown)
- **Implementation**: `main.py` lines 13096-13109, 13111-13126, 13208-13219, 13386-13392, 13461-13481

### 11. Metadata Storage
- **Change**: `record_post_history` now stores all new fields
- **New fields**:
  - `clear_guard_hard`, `clear_guard_soft`
  - `sky_visible`, `sky_critical_mismatch`
  - `pool_counts` (dict of stage pool sizes)
  - `want_sunset` (computed per-selection)
- **Implementation**: `main.py` lines 13692-13735

### 12. Test Coverage
- **New test file**: `tests/test_sea_selection_rework.py`
- **Tests**:
  1. `test_season_window_doy_any_year`: Verifies Kaliningrad timezone and ±45 day wrap-around
  2. `test_wave_mapping_and_corridor`: Tests interpolation curve and storm classification thresholds
  3. `test_no_sky_not_filtered`: Verifies sky_visible=False skips sky penalties
  4. `test_coast_cloud_policy`: Tests B0/B1/B2/AN sky policy matrix and clear guard thresholds
  5. `test_want_sunset_requires_visible_sky_and_no_clear_guard_violation`: Validates sunset logic
  6. `test_storm_persisting_true`: Confirms storm persistence uses new wave-score classes
- **All tests passing**: 315 passed, 2 skipped, 0 failed

## Acceptance Criteria Met

✅ Rubric run follows the specified fallback order (season → wave → B1 → wave+broaden → B2 → AN)
✅ Logging shows every stage with required fields (season metadata, pool counts, top-5 scoring, final selection payload)
✅ New tests pass covering all specified behaviors
✅ Existing sea publications remain single-photo with correct storm/sunset decisions (verified by 77 sea-related tests)
✅ Storm persistence logic uses new wave-score classes (calm/storm/strong_storm)
✅ Clear guard enforcement prevents cloudy assets on clear days in sunset mode
✅ Sky visibility integration skips penalties and disables sunset for non-visible sky
✅ Season window uses Kaliningrad timezone and includes NULL shot_doy assets
✅ Penalty-based evaluation with calm-day bonuses and missing score handling

## Files Modified

1. **main.py**: Core sea selection logic rework (lines 13014-13805+)
2. **data_access.py**: Added `sky_visible` field to candidate fetching (lines 1728-1732, 1761)
3. **tests/test_sea_selection_rework.py**: New comprehensive test suite (6 tests)
4. **tests/test_sea_rubric_season_window.py**: Minor lint fixes (lines 131, 325)
5. **CHANGELOG.md**: Added entries documenting all changes

## Backward Compatibility

- All existing tests pass (315 passed)
- Wave mapping interpolation curve unchanged
- Storm persistence logic compatible with new state names
- Metadata schema extended (not breaking - new fields added)
- Logging format enhanced (additive - no breaking changes)

## Performance Considerations

- Staged fallback minimizes candidate re-evaluation
- Pool tracking provides visibility into filtering effectiveness
- Penalty computation is O(1) per candidate
- Season window calculation remains O(n) over candidates

## Future Enhancements

- Consider caching Kaliningrad timezone object (currently created per run)
- Monitor sky_critical_mismatch frequency to tune clear guard thresholds
- Track pool_counts metrics in Prometheus for alerting
- Consider exposing fallback stage distribution in Grafana dashboards
