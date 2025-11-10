# Vision Metrics Parsing Enhancement - Implementation Summary

## Overview
Enhanced the ingestion pipeline to automatically extract and persist `wave_score_0_10`, `wave_conf`, and `sky_code` (mapped to `vision_wave_score`, `vision_wave_conf`, and `vision_sky_bucket` columns) from vision classification results.

## Changes Implemented

### 1. Centralized Parsing Functions (`utils_wave.py`)
- **Added `parse_wave_score_from_vision()`**: Extracts wave score and confidence from multiple JSON layouts:
  - `weather.sea.wave_score` + `weather.sea.confidence`
  - `sea_state.score` + `sea_state.confidence`
  - `sea_wave_score.value` + `sea_wave_score.confidence` (legacy dict)
  - `sea_wave_score` (legacy scalar)
  - Textual fallback: `«Волнение моря: X/10 (conf=Y)»` with confidence
  - Textual fallback: `«Волнение моря: X/10»` without confidence

- **Added `parse_sky_bucket_from_vision()`**: Extracts sky bucket from multiple JSON layouts:
  - `weather.sky.bucket`
  - `sky.bucket`
  - `sky_bucket` (direct)

- **Added `_to_float()` helper**: Safe conversion of values to float

### 2. DataAccess Integration (`data_access.py`)
- **Updated `_parse_wave_score_from_vision()` and `_parse_sky_bucket_from_vision()`**: Now delegate to centralized utils_wave functions for consistency
- **Enhanced `update_asset()` method**: When `vision_results` is provided, automatically:
  1. Parses wave score, confidence, and sky bucket from the vision JSON
  2. Updates `vision_wave_score`, `vision_wave_conf`, and `vision_sky_bucket` columns
  3. Maintains backward compatibility with explicit column updates
- **Enhanced `fetch_sea_candidates()` method**: Now prefers parsed `vision_wave_score` column before falling back to legacy `photo_wave` or vision JSON data

### 3. Backward Compatibility
- Kept existing `vision_wave_*` column names for compatibility
- Maintained support for all legacy JSON formats
- Manual column updates via `update_asset()` still work as before
- When `vision_results` is provided, parsed values overwrite existing columns

### 4. Test Coverage
Added comprehensive test suites:

**`tests/test_vision_parsers.py`** (28 tests):
- Tests for all JSON layout variations
- Tests for textual fallbacks with and without confidence
- Tests for mixed types, string numbers, invalid values
- Tests for sky bucket parsing in all layouts
- Consistency tests between utils_wave and DataAccess methods

**`tests/test_ingestion_helper_consistency.py`** (8 tests):
- Integration tests verifying that vision metrics flow through `update_asset()`
- Tests for wave score, sky bucket, and combined parsing
- Tests for legacy formats (dict and textual)
- Tests for graceful handling of missing metrics
- Tests for overwriting existing metrics
- Tests for interaction with manual column updates

## Usage Example

```python
# When vision classification completes in main.py
result_payload = {
    "status": "ok",
    "provider": "gpt-4o-mini",
    "weather": {
        "sea": {"wave_score": 7.5, "confidence": 0.92},
        "sky": {"bucket": "partly_cloudy"}
    },
    # ... other fields
}

# Update asset with vision results
self.data.update_asset(asset_id, vision_results=result_payload)

# Columns are automatically populated:
# - vision_wave_score = 7.5
# - vision_wave_conf = 0.92
# - vision_sky_bucket = "partly_cloudy"
```

## Benefits
1. **Automatic parsing**: No manual column updates needed when vision results arrive
2. **Reduced NULL cases**: Parsed columns are populated whenever vision data is available
3. **Better data quality**: fetch_sea_candidates prefers parsed columns, reducing reliance on legacy formats
4. **Centralized logic**: All parsing logic in one place (utils_wave.py) for consistency
5. **Comprehensive fallbacks**: Handles multiple JSON layouts and textual formats
6. **Full test coverage**: 36 tests ensure parsing works correctly in all scenarios

## Files Modified
- `utils_wave.py`: Added centralized parsing functions
- `data_access.py`: Enhanced update_asset and fetch_sea_candidates
- `tests/test_vision_parsers.py`: Enhanced with new tests for utils_wave functions
- `tests/test_ingestion_helper_consistency.py`: New integration tests (created)

## Testing
All 36 tests pass successfully:
```bash
pytest tests/test_vision_parsers.py tests/test_ingestion_helper_consistency.py -v
```

## Notes
- Column names use `vision_wave_score` instead of `wave_score_0_10` for consistency with existing schema
- Column names use `vision_sky_bucket` instead of `sky_code` for consistency with existing schema
- The parsing respects priority order: structured JSON layouts are checked before textual fallbacks
- Regular expressions support flexible spacing in textual formats
