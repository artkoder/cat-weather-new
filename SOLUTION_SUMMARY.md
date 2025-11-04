# Fix: TypeError '<' between 'str' and 'float' in Sea Picker Sort Key

## Problem
The `_pick_sea_asset` function in the sea rubric publishing workflow was crashing with a `TypeError` when attempting to sort assets. The error occurred because the sort key tuple contained mixed types (str, float, int, datetime) that Python cannot compare with the `<` operator.

## Root Causes
1. **Wave score values**: The `sea_wave_score.value` field from `vision_results` could be either a string (`"7"`) or an int/float, but the code only checked for `isinstance(value, int)`.
2. **Asset IDs**: Asset IDs could be strings or integers, requiring safe conversion.
3. **Datetime parsing**: The `last_used_at` field was a string that needed to be parsed to datetime, but could be invalid, empty, or missing entirely.
4. **Type consistency**: The sort key tuple needed all elements to be consistently typed (numeric types only).

## Solution

### 1. Added Type Normalization Utilities (lines 12741-12795 in main.py)

```python
def _safe_int(self, value: Any, default: int = 0) -> int:
    """Safely convert any value to int with fallback."""
    # Handles: int, float, str (including "3.14"), bool, None
    # Returns default on failure with warning log

def _safe_float(self, value: Any, default: float | None = None) -> float | None:
    """Safely convert any value to float with fallback."""
    # Handles: int, float, str (including "3.14", "7")
    # Returns default on failure with warning log

def _parse_datetime_iso(self, value: str | None) -> datetime:
    """Parse ISO8601 datetime string, return datetime.min on failure."""
    # Handles valid ISO strings, invalid strings, None, empty strings
    # Always returns a datetime (never raises exception)
```

### 2. Updated Wave Score Extraction (lines 12848-12870 in main.py)

Changed `_get_asset_wave_score_with_fallback` to use `_safe_int`:

```python
# Before: only accepted isinstance(value, int)
if isinstance(value, int):
    return max(0, min(10, value))

# After: handles string, int, float
if value is not None:
    int_value = self._safe_int(value, default=-1)
    if int_value >= 0:
        return max(0, min(10, int_value))
```

### 3. Normalized Sort Key Types (lines 12925-12973 in main.py)

Updated `_pick_sea_asset` to ensure all sort key components are numeric:

```python
def compute_lru_score(asset: Asset) -> float:
    """Returns negative timestamp (older = higher score) or float('inf')."""
    last_used_str = asset.payload.get("last_used_at") if asset.payload else None
    if not last_used_str:
        return float('inf')
    
    # Use safe datetime parser (never raises)
    last_used_dt = self._parse_datetime_iso(last_used_str)
    if last_used_dt == datetime.min:
        return float('inf')
    
    return -last_used_dt.timestamp()

def compute_score(asset: Asset) -> tuple[float, int, int, float, int]:
    """Compute sort key tuple with all numeric types."""
    # All components explicitly cast to their expected types
    asset_id = self._safe_int(asset.id, default=0)
    
    return (
        float(wave_and_sky_score),  # float
        int(sunset_tag),            # int (0 or 1)
        int(sunset_cat),            # int (0 or 1)
        float(lru_score),           # float (timestamp or inf)
        -int(asset_id),             # int (for tie-breaking)
    )
```

## Testing

Created comprehensive test suite in `tests/test_sea_picker_type_normalization.py`:

### Unit Tests for Type Conversion Utilities (19 tests)
- `_safe_int`: handles strings, None, invalid strings, various types
- `_safe_float`: handles strings, None, invalid strings, various types  
- `_parse_datetime_iso`: handles valid ISO strings, invalid strings, None, empty

### Integration Tests for Sea Picker
- **Wave score with string values**: `{"value": "7"}` correctly processed
- **Asset IDs as strings**: `asset.id = "101"` correctly handled
- **Invalid datetime strings**: `"not-a-date"` doesn't crash (returns datetime.min)
- **Missing/empty timestamps**: `{}` or `{"last_used_at": ""}` handled gracefully
- **Mixed type consistency**: Assets with mix of string/int/float values sort without TypeError
- **Deterministic sorting**: Same scores produce consistent results via ID tie-breaker

### Regression Tests
- ✅ `test_pick_sea_asset_prioritizes_tags_and_categories` (existing test) still passes
- ✅ All 25 existing sea wave score tests pass
- ✅ All 46 sea-related tests pass

## Validation Results

```
✅ All 19 new type normalization tests pass
✅ All 25 existing sea wave score tests pass
✅ All 46 sea-related tests pass
✅ Linting passes (ruff check)
✅ Existing behavior preserved
```

## Key Benefits

1. **Robust error handling**: Invalid data types are converted safely with logging instead of crashing
2. **Backward compatible**: Existing tests pass; behavior unchanged for valid data
3. **Deterministic**: Sort order is now predictable and consistent
4. **Logging**: Invalid values are logged with warnings for debugging
5. **Type safety**: Sort key tuple has consistent, comparable types throughout

## Modified Files

1. `main.py`: Added 3 utility methods + updated 2 methods (lines 12741-12973)
2. `tests/test_sea_picker_type_normalization.py`: New comprehensive test suite (355 lines)

## No Breaking Changes

- All existing tests pass
- Existing behavior preserved
- No database schema changes
- No config changes
- No API changes
