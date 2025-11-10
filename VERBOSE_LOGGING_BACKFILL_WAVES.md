# Verbose Logging for Wave Backfill

## Overview

Enhanced the `backfill_waves` function with comprehensive verbose logging to facilitate debugging and diagnostics when processing assets.

## Problem

The backfill ran twice with confusing results:
1. First run: Updated: 2, Errors: 139
2. Second run: Updated: 0, Skipped: 141, Errors: 0

The logs only showed summary statistics without details about why each asset was skipped, making it impossible to diagnose the root cause.

## Solution

Added detailed logging at multiple levels:

### 1. Per-Asset Discovery Logging (DEBUG)

When discovering each asset to process, log its current state:

```
DEBUG: Backfill discovered asset_id=X vision_results_exists=True current_wave_score=3.5 current_sky_code=clear
```

### 2. Skip Reason Tracking

Track and log specific reasons for skipping assets:

- `already_complete` - All three fields (wave_score, wave_conf, sky_bucket) already populated
- `no_vision_results` - Asset has no vision_results data
- `invalid_vision_type` - vision_results exists but is not a dict (counted as error)
- `no_extractable_data` - vision_results exists but contains no parseable wave/sky data
- `fields_already_populated` - Some fields already exist, no updates needed
- `exception` - An unexpected exception occurred during processing

Each skip is logged as:

```
DEBUG: Backfill skip asset_id=X skip_reason=no_vision_results
```

### 3. Success Logging (INFO)

When an asset is successfully updated, log the final values:

```
INFO: Backfill updated asset_id=X wave_score_0_10=3.5 sky_code=clear confidence=0.85
```

### 4. Sample Collection

Collect first 10 skipped assets with their reasons for quick diagnostics:

```
INFO: Backfill sample (first 10 skipped): [
    {'asset_id': 1, 'reason': 'no_vision_results'},
    {'asset_id': 2, 'reason': 'already_complete'},
    ...
]
```

### 5. Summary with Reason Breakdown

Final summary includes distribution of skip reasons:

```
INFO: Backfill waves completed: processed=141 updated=0 skipped=141 errors=0 
      (skip reasons: no_vision_results=100, already_complete=30, no_extractable_data=11)
```

## Implementation Details

### Data Structures

Added two tracking structures:

```python
skip_reasons: dict[str, int] = {}  # Count of each skip reason
skipped_samples: list[dict[str, Any]] = []  # First 10 skipped assets
max_samples = 10  # Limit sample collection
```

### Skip Reason Recording

Each skip path now:
1. Increments the appropriate skip reason counter
2. Adds to sample collection (if under limit)
3. Logs with specific reason

Example:

```python
skip_reason = "no_vision_results"
logging.debug("Backfill skip asset_id=%s skip_reason=%s", asset.id, skip_reason)
stats["skipped"] += 1
skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
if len(skipped_samples) < max_samples:
    skipped_samples.append({"asset_id": asset.id, "reason": skip_reason})
```

### Final Reporting

Summary includes:
- Total counts (processed, updated, skipped, errors)
- Sorted breakdown of skip reasons (most common first)
- Sample of first 10 skipped assets

## Usage

### Viewing Detailed Logs

Set log level to DEBUG to see all per-asset logs:

```python
logging.basicConfig(level=logging.DEBUG)
```

### Quick Diagnostics

Even at INFO level, you'll see:
- Updated asset details
- Sample of skipped assets
- Summary with reason breakdown

### Example Output

```
INFO: Backfill waves: found 141 assets to process, 0 already complete
DEBUG: Backfill discovered asset_id=1 vision_results_exists=True current_wave_score=None current_sky_code=None
DEBUG: Backfill skip asset_id=1 skip_reason=no_extractable_data
DEBUG: Backfill discovered asset_id=2 vision_results_exists=False current_wave_score=None current_sky_code=None
DEBUG: Backfill skip asset_id=2 skip_reason=no_vision_results
INFO: Backfill waves: processed batch 1 (141 assets), stats: {'updated': 0, 'skipped': 141, 'errors': 0}
INFO: Backfill sample (first 10 skipped): [
    {'asset_id': 1, 'reason': 'no_extractable_data'},
    {'asset_id': 2, 'reason': 'no_vision_results'},
    ...
]
INFO: Backfill waves completed: processed=141 updated=0 skipped=141 errors=0 
     (skip reasons: no_vision_results=100, already_complete=30, no_extractable_data=11)
```

## Benefits

1. **Quick Diagnosis**: Sample shows immediate patterns without scrolling through thousands of lines
2. **Reason Distribution**: Summary breakdown identifies the most common skip reasons
3. **Detailed Investigation**: DEBUG logs provide per-asset information when needed
4. **Performance Friendly**: Only collects first 10 samples, avoiding memory bloat
5. **Backward Compatible**: Existing tests continue to work; logging is non-invasive

## Testing

All existing tests pass:
- `test_backfill_waves_command.py` - Integration tests
- `test_backfill_waves_errors.py` - Error handling tests

The verbose logging is transparent to tests - they verify behavior, not log output.

## Future Improvements

Possible enhancements:
- Add summary statistics per batch (for very large datasets)
- Export skip samples to JSON file for offline analysis
- Add time-based metrics (processing time per asset)
- Configure sample size via parameter
