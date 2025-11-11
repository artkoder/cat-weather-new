# Sea Remainder Report Fix - Zero Count Issue

## Problem
After production publication of the sea rubric, the remainder report was showing all zeros for sky conditions and wave scores, even though unpublished sea photos remained in the database.

Example of the broken output:
```
🗂 Остатки фото «Море»: 0
Небо
• Солнечно: 0 ⚠️ мало
• Преимущественно ясно: 0 ⚠️ мало
... (all conditions show 0)
```

## Root Cause
The `_send_sea_inventory_report()` method was querying the wrong table/column for the vision_category filter.

**Incorrect queries were looking in:**
```sql
json_extract(vr.result_json, '$.vision_category') = 'sea'
```

**But vision_category is actually stored in:**
```sql
json_extract(a.payload_json, '$.vision_category')
```

### Why This Matters
- `vision_results.result_json` contains the raw API response from the vision service
- `assets.payload_json` contains metadata about the asset, including the classified vision_category
- Using the wrong table meant the query found no matching records (returning NULL from the LEFT JOIN), resulting in zero counts

## Solution
Fixed all three SQL queries in `_send_sea_inventory_report()` to use the correct column:

1. **Total count query** - Changed from `vr.result_json` to `a.payload_json`
2. **Sky bucket counts query** - Changed from `vr.result_json` to `a.payload_json`  
3. **Wave score counts query** - Changed from `vr.result_json` to `a.payload_json`

Also removed the unnecessary `LEFT JOIN vision_results` since we don't need vision_results data for these queries.

### Additional Fix
The same bug was found in the `/sea_audit` command query, which was also fixed.

## Changes Made

### main.py Line 6901-6907 (sea_audit command)
**Before:**
```sql
SELECT a.id, a.payload_json, a.tg_message_id
FROM assets a
LEFT JOIN vision_results vr ON vr.asset_id = a.id
WHERE json_extract(vr.result_json, '$.vision_category') = 'sea'
```

**After:**
```sql
SELECT a.id, a.payload_json, a.tg_message_id
FROM assets a
WHERE LOWER(json_extract(a.payload_json, '$.vision_category')) = 'sea'
```

### main.py Line 15192-15227 (_send_sea_inventory_report)
**Before:**
```sql
SELECT COUNT(*) as cnt 
FROM assets a
LEFT JOIN vision_results vr ON vr.asset_id = a.id 
WHERE json_extract(vr.result_json, '$.vision_category') = 'sea'
```

**After:**
```sql
SELECT COUNT(*) as cnt 
FROM assets a
WHERE LOWER(json_extract(a.payload_json, '$.vision_category')) = 'sea'
```

Similar changes for sky bucket and wave score queries.

### Added Debug Logging
Added logging statement to help troubleshoot future issues:
```python
logging.info(
    "SEA_INVENTORY_REPORT prod=%d total=%d sky_buckets=%s wave_scores=%s",
    int(is_prod),
    total_count,
    dict(sky_counts),
    dict(wave_counts),
)
```

## Testing
Created comprehensive test suite in `tests/test_sea_inventory_report.py` with three test cases:
1. **test_sea_inventory_report_counts_correctly** - Verifies correct counts for various sky/wave combinations
2. **test_sea_inventory_report_zero_when_no_assets** - Verifies zeros shown when no assets exist
3. **test_sea_inventory_report_ignores_non_sea_assets** - Verifies only sea assets are counted

## Expected Behavior After Fix
After publishing a sea rubric, the remainder report will correctly show:
- Total count of remaining unpublished sea photos
- Breakdown by sky condition (clear, mostly_clear, partly_cloudy, mostly_cloudy, overcast)
- Breakdown by wave score (0-10)
- Warning indicators (⚠️ мало) when counts are below 10

## Related Code References
- Vision category is set during asset ingestion in `data_access.py`
- The `fetch_assets_by_vision_category()` method (line 2160) shows the correct pattern for querying by vision_category
- All queries for vision_category should use `a.payload_json`, not `vr.result_json`
