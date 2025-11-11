# /audit_assets Command Implementation

## Summary

Implemented `/audit_assets` command to audit and clean stale records across all assets in the database. The command checks if the Telegram messages referenced by assets still exist and removes "dead souls" (DB records for deleted TG messages).

## Key Design Decisions

### Schema Adaptation
- The ticket assumed separate tables (`sea_assets`, `flowers_assets`, `arch_assets`)
- In reality, the codebase uses a single unified `assets` table with a `rubric_id` foreign key
- Implementation audits all assets from the single table and groups results by rubric for reporting

### Implementation Details

1. **Authorization**: Requires `is_authorized()` (operator/superadmin)

2. **Batch Processing**:
   - Batch size: 50 assets per batch
   - Delay: 100ms between batches to avoid rate limits

3. **Safe Verification Pattern**:
   - Uses `copyMessage` to operator chat to verify message existence
   - Immediately deletes the copy with `deleteMessage` (non-destructive verification)
   - If copy fails with "message not found" error → asset is a dead soul → delete from DB
   - Other errors (rate limit, permissions, etc.) → log but keep the asset

4. **Error Handling**:
   - Dead soul detection: `"message to copy not found"`, `"message not found"`, `"message can't be copied"`
   - Individual delete operations (no transactions across batches)
   - Errors logged but don't stop the audit
   - Resilient to partial failures

5. **Reporting**:
   - Total checked/removed counts
   - Per-rubric breakdown (sea, flowers, guess_arch, unassigned)
   - Sent to operator chat after completion

## Files Modified

### `/home/engine/project/main.py`
- Added `/audit_assets` command handler (lines 7020-7194)
- Updated `/help` text to include new command (line 6604)

### `/home/engine/project/tests/test_audit_assets.py`
- New comprehensive test file with 8 test cases:
  - `test_audit_assets_removes_missing`: Verifies dead souls are deleted
  - `test_audit_assets_reports_counts`: Verifies correct report counts
  - `test_audit_assets_continues_on_error`: Verifies non-404 errors don't delete
  - `test_audit_assets_requires_authorization`: Verifies auth requirement
  - `test_help_has_audit_assets`: Verifies help includes new command
  - `test_audit_assets_batching`: Verifies batch processing works
  - `test_audit_assets_per_rubric_reporting`: Verifies per-rubric breakdown

## Logging

New log events:
- `ASSETS_AUDIT_STARTED`: Audit begins
- `ASSETS_AUDIT_RUBRIC_MAP_ERROR`: Error loading rubric mapping
- `ASSETS_AUDIT_FETCH_ERROR`: Error fetching assets from DB
- `ASSETS_AUDIT_DEAD_SOUL`: Dead soul detected (includes asset_id, chat_id, msg_id, rubric)
- `ASSETS_AUDIT_COPY_DELETE_FAILED`: Error deleting verification copy
- `ASSETS_AUDIT_TG_ERROR`: Non-404 Telegram error
- `ASSETS_AUDIT_DB_DELETE_FAILED`: Error deleting dead soul from DB
- `ASSETS_AUDIT_FINISHED`: Audit complete (includes checked/removed counts)

## Usage

```
/audit_assets
```

Operator receives:
1. Initial "🔍 Начинаю аудит всех ассетов..." message
2. Final report with:
   - Total assets checked
   - Total dead souls removed
   - Per-rubric breakdown

## Example Output

```
🔎 Аудит ассетов завершён

Проверено: 245
Удалено «мёртвых душ»: 12

По рубрикам:
  • flowers: проверено 89, удалено 4
  • guess_arch: проверено 67, удалено 3
  • sea: проверено 78, удалено 5
  • unassigned: проверено 11, удалено 0
```

## Acceptance Criteria

✅ `/audit_assets` command works from operator chat
✅ Audits all assets in the unified assets table
✅ copyMessage → deleteMessage pattern for safe check
✅ Records with missing TG messages deleted from DB
✅ Other TG errors logged but don't block audit
✅ Batch processing: 50 records, 100ms delay between batches
✅ Summary sent to operator chat: checked/removed/rubrics
✅ `/help` contains `/audit_assets` entry
✅ All logs present and properly formatted
✅ Unit tests comprehensive
✅ Code compiles without syntax errors
