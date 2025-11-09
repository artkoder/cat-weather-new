# Sea Scoring Tuning - Implementation Summary

## Overview
This update tunes the sea selection logic to better handle calm scenarios and NULL wave scores, while improving observability through enhanced topN logging.

## Changes Made

### 1. Calm Wave Filtering (`data_access.py`)
**Location**: `fetch_sea_candidates()` method

**Changes**:
- Added `target_wave_score` parameter to the function signature
- Implemented calm-scenario filtering: when `target_wave_score ≤ 1` and at least one candidate has `wave_score ≤ 2`, assets with `wave_score ≥ 5` are filtered out
- Added import of `wave_m_to_score` from `utils_wave`
- Fixed wave score handling to properly convert `photo_wave` (stored in meters) to score (0-10 scale) before storing in candidate dict

**Rationale**: In calm conditions, we want to prioritize photos showing calm seas and exclude photos with high waves that don't match the current conditions.

### 2. NULL Wave Score Penalty (`main.py`)
**Location**: `evaluate_stage_candidate()` function within `_publish_sea()`

**Changes**:
- Added `WaveNullPenalty` to the components dictionary
- Applied +0.8 penalty when `photo_wave_val is None` and `stage_cfg.allow_missing_wave` is true
- This ensures NULL scores are penalized but still remain in the pool (not excluded entirely)

**Rationale**: Assets without wave scores should be less preferred than those with scores, but we still want them available as fallback options in later stages (B2, AN).

### 3. Enhanced TopN Logging (`main.py`)
**Location**: topN loop in `_publish_sea()`

**Changes**:
- Restructured log fields to include all required keys:
  - `wave_target`: target wave score from weather conditions
  - `wave_photo`: wave score of the photo
  - `delta`: absolute difference (wave_delta)
  - `sky_photo`: normalized sky token
  - `penalties`: structured breakdown of all penalty components
  - `total_penalties`: sum of all penalties
  - `total_score`: final score after all bonuses/penalties
- Extracted penalties breakdown by filtering components dict for keys containing "Penalty"

**Rationale**: Operators need to see detailed scoring breakdown in logs to understand why certain photos were selected and debug any issues.

### 4. Test Coverage

#### `tests/test_sea_calm_priority.py`
Added two new test cases:

1. **`test_calm_sea_filters_high_waves_when_calm_exists`**
   - Verifies that when target wave ≤1 and calm candidates exist, high wave assets (≥5) are filtered out
   - Creates one calm asset (wave_score=1, photo_wave=0.2m) and one high wave asset (wave_score=6, photo_wave=1.2m)
   - Asserts calm asset is in candidates but high wave asset is not

2. **`test_calm_sea_allows_null_waves_with_penalty`**
   - Verifies that NULL wave scores remain in pool
   - Creates one asset with NULL wave score and one with known wave score
   - Asserts both are present in candidates (penalty is applied but not tested directly here)

#### `tests/test_sea_selection_rework.py`
Added two new test cases:

1. **`test_topn_log_structure`**
   - Documents the required fields in topN logs
   - Validates the structure expectations

2. **`test_null_wave_penalty_applied`**
   - Verifies that B2 and AN stages allow missing wave scores
   - Documents the +0.8 penalty value

#### `tests/test_sea_selection_utils.py`
Fixed import issue by adding sys.path setup.

## API Changes

### `fetch_sea_candidates()`
**Before**:
```python
def fetch_sea_candidates(
    self,
    rubric_id: int,
    *,
    limit: int = 48,
    season_range: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
```

**After**:
```python
def fetch_sea_candidates(
    self,
    rubric_id: int,
    *,
    limit: int = 48,
    season_range: tuple[int, int] | None = None,
    target_wave_score: float | None = None,
) -> list[dict[str, Any]]:
```

**Callers Updated**:
- `main.py`: `_publish_sea()` now passes `target_wave_score=target_wave_score`

## Behavioral Changes

### Calm Scenarios (target_wave_score ≤ 1)
- **Before**: All assets in season window were eligible regardless of wave score
- **After**: If calm candidates exist (wave_score ≤ 2), high wave assets (wave_score ≥ 5) are excluded from the pool

### NULL Wave Scores
- **Before**: NULL scores received stage-dependent unknown_wave_penalty (varies by stage)
- **After**: NULL scores receive both stage-dependent penalty AND additional +0.8 WaveNullPenalty in stages that allow missing waves (B2, AN)

### TopN Logs
- **Before**: Logs included most fields but penalties weren't aggregated
- **After**: Logs include structured `penalties` dict and `total_penalties` sum, plus renamed fields for consistency (`wave_target`, `wave_photo`, `delta`, `sky_photo`, `total_score`)

## Testing
All existing tests pass, plus 4 new tests covering:
- Calm wave filtering logic
- NULL wave score handling
- TopN log structure validation
- Stage configuration for missing waves

Run tests with:
```bash
pytest tests/test_sea_calm_priority.py -v
pytest tests/test_sea_selection_rework.py -v
```

## Guardrails Preserved
- Season filtering logic unchanged
- Stage progression (B0 → B1 → B2 → AN) unchanged
- Sky matching and compatibility unchanged
- Calm wave bonus (target ≤ 2, photo ≤ 1.0 → +5.0) unchanged
- Age-based freshness scoring unchanged
