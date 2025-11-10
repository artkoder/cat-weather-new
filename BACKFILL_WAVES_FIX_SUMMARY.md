# Backfill Waves Error Handling Fix

## Problem

When backfill_waves ran on production, it resulted in:
- Updated: 2 assets
- Errors: 139 assets (out of 141)

The logging was insufficient - it only showed aggregate results without details about individual asset failures.

## Root Cause Analysis

The original implementation had several issues:

1. **Poor error categorization**: Assets with no parseable data were counted as "errors" instead of "skipped"
2. **Insufficient logging**: No structured logging per asset with error details
3. **Missing edge case handling**: No explicit checks for vision_results being None or non-dict types

## Changes Made

### 1. Enhanced Logging (`main.py:1926-2070`)

**Structured error logging** with asset_id, error_type, and error_msg:
```python
logging.error(
    "Backfill error asset_id=%s error_type=%s error_msg=%s",
    asset.id,
    error_type,
    error_msg,
)
```

**Debug logging for skipped assets** with specific reasons:
- `reason=no_vision_results` - asset has no vision_results at all
- `reason=no_parseable_data` - vision_results exists but contains no extractable wave/sky data
- `reason=fields_already_populated` - all target fields already have values

**Error detail tracebacks**: First 10 errors get full traceback via `exc_info=True` for debugging

**Completion summary**: Final log shows aggregate statistics for the entire backfill run

### 2. Better Error Classification

Changed logic to distinguish between legitimate errors vs. missing data:

**Old behavior** (line 1971-1973):
```python
if wave_score is None and sky_bucket is None:
    stats["errors"] += 1  # Wrong! Missing data isn't an error
    continue
```

**New behavior** (line 1992-1998):
```python
if wave_score is None and wave_conf is None and sky_bucket is None:
    logging.debug("Backfill skip asset_id=%s reason=no_parseable_data", asset.id)
    stats["skipped"] += 1  # Correctly categorized as skipped
    continue
```

### 3. Explicit Type Checking

Added explicit validation for vision_results type:

```python
if vision is None:
    logging.debug("Backfill skip asset_id=%s reason=no_vision_results", asset.id)
    stats["skipped"] += 1
    continue

if not isinstance(vision, dict):
    logging.error(
        "Backfill error asset_id=%s error_type=InvalidVisionType error_msg='vision_results is not a dict: %s'",
        asset.id,
        type(vision).__name__,
    )
    stats["errors"] += 1
    continue
```

### 4. Idempotency Verification

The existing implementation already had good idempotency:
- Line 1946-1953: Skip assets that already have all three fields populated
- Line 2003-2013: Only update fields that are currently None
- This ensures re-running backfill is safe and won't overwrite data

### 5. Parser Robustness

The existing parsers in `utils_wave.py` are already quite robust:
- `parse_wave_score_from_vision()` handles multiple JSON layouts
- `_to_float()` helper gracefully converts strings, ints, and floats
- Returns None for invalid data instead of raising exceptions
- The `vision_results` property in `data_access.py` (line 368-381) already handles JSON parsing errors

## Test Coverage

Created comprehensive test suite in `tests/test_backfill_waves_errors.py`:

1. **test_backfill_waves_handles_invalid_json**: Invalid JSON strings are handled gracefully
2. **test_backfill_waves_handles_missing_vision_results**: Assets without vision_results are skipped
3. **test_backfill_waves_handles_empty_vision_results**: Empty vision_results dicts are skipped
4. **test_backfill_waves_handles_malformed_fields**: Various type mismatches (strings, invalid values) are handled
5. **test_backfill_waves_idempotency**: Re-running backfill doesn't duplicate updates
6. **test_backfill_waves_partial_updates**: Only missing fields are updated, existing ones preserved
7. **test_backfill_waves_text_parsing**: Text-based vision results are parsed correctly

## Expected Outcome

When backfill_waves runs again:

1. **Assets with valid vision data**: Updated (if fields were None) or skipped (if already populated)
2. **Assets without vision_results**: Skipped, logged as `reason=no_vision_results`
3. **Assets with unparseable data**: Skipped, logged as `reason=no_parseable_data`
4. **Assets with actual errors**: Logged with full error details including asset_id, error_type, error_msg

**Previous result**: 2 updated, 139 errors
**Expected result**: 2+ updated (if there are parseable assets), 137+ skipped, 0-2 errors (only for truly exceptional cases)

## Log Output Examples

### Normal processing:
```
INFO: Backfill waves: found 139 assets to process, 2 already complete
INFO: Backfill waves: processed batch 1 (139 assets), stats: {'updated': 0, 'skipped': 139, 'errors': 0}
INFO: Backfill waves completed: processed=139 updated=0 skipped=139 errors=0
```

### With errors (first 10 get full traceback):
```
ERROR: Backfill error asset_id=abc123 error_type=ValueError error_msg=invalid literal for float()
DEBUG: Backfill error detail asset_id=abc123
Traceback (most recent call last):
  ...
```

### Debug-level skip reasons:
```
DEBUG: Backfill skip asset_id=xyz789 reason=no_vision_results
DEBUG: Backfill skip asset_id=def456 reason=no_parseable_data
DEBUG: Backfill skip asset_id=ghi012 reason=fields_already_populated
```

## Deployment Notes

1. The fix is backward compatible - no schema changes required
2. Idempotency is preserved - safe to re-run multiple times
3. Existing tests in `test_backfill_waves_command.py` remain valid
4. New edge case tests added in `test_backfill_waves_errors.py`
