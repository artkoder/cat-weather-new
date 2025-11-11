# Audit Assets Deletion Debug and Fix

## Problem Description

The `/audit_assets` command was not correctly detecting deleted photos from the assets channel:
- After manually deleting a photo from the assets channel in Telegram, running `/audit_assets` showed "удалено 0" instead of the expected "удалено 1"
- Database counts remained unchanged even after files were deleted from Telegram

## Root Cause Analysis

The issue was that the error pattern matching was too narrow. When the bot tried to verify if a message still exists in Telegram using the `copyMessage` API call, it only checked for a limited set of error messages:
- "message to copy not found"
- "message not found"
- "message can't be copied"

However, Telegram can return various other error messages when a message is deleted or inaccessible, which weren't being caught by the original implementation.

## Changes Made

### 1. Enhanced Error Pattern Detection (main.py, lines 7173-7184)

**Before:**
```python
if (
    "message to copy not found" in error_str
    or "message not found" in error_str
    or "message can't be copied" in error_str
):
```

**After:**
```python
is_dead_soul = (
    "message to copy not found" in error_str
    or "message not found" in error_str
    or "message can't be copied" in error_str
    or "message_id_invalid" in error_str
    or "message to get not found" in error_str
    or "message to forward not found" in error_str
    or "message identifier is not specified" in error_str
    or "chat not found" in error_str
)
if is_dead_soul:
```

This now catches additional Telegram API error patterns:
- `MESSAGE_ID_INVALID` - when the message ID is invalid
- "message to get not found" - alternative error for missing messages
- "message to forward not found" - forwarding-specific error
- "message identifier is not specified" - missing ID error
- "chat not found" - when the entire chat is inaccessible

### 2. Added Comprehensive Logging (main.py)

Added detailed logging at key points in the audit process:

**Line 7120-7126:** Log before checking each asset
```python
logging.info(
    "ASSETS_AUDIT_CHECKING asset_id=%s chat_id=%s msg_id=%s rubric=%s",
    asset_id,
    chat_id,
    msg_id,
    rubric_name,
)
```

**Line 7138-7144:** Log when message exists
```python
logging.info(
    "ASSETS_AUDIT_EXISTS asset_id=%s chat_id=%s msg_id=%s rubric=%s",
    asset_id,
    chat_id,
    msg_id,
    rubric_name,
)
```

**Line 7165-7172:** Log when copy fails with full error message
```python
logging.info(
    "ASSETS_AUDIT_COPY_FAILED asset_id=%s chat_id=%s msg_id=%s rubric=%s error=%s",
    asset_id,
    chat_id,
    msg_id,
    rubric_name,
    str(e)[:200],
)
```

**Line 7187-7192:** Log successful deletion with counter
```python
logging.info(
    "ASSETS_AUDIT_DELETED asset_id=%s rubric=%s total_removed=%d",
    asset_id,
    rubric_name,
    total_removed,
)
```

## Benefits

1. **More Robust Error Detection**: The enhanced error pattern matching catches a wider variety of Telegram API errors, ensuring deleted messages are properly detected
2. **Better Debugging**: Comprehensive logging allows operators to trace exactly what happens during the audit process
3. **Production Visibility**: Log messages include key identifiers (asset_id, chat_id, msg_id, rubric) for easy correlation with production issues

## Testing

### Existing Tests (7 tests)
All existing tests in `tests/test_audit_assets.py` continue to pass:
- `test_audit_assets_removes_missing` - Verifies missing assets are deleted
- `test_audit_assets_reports_counts` - Checks correct counts in report
- `test_audit_assets_continues_on_error` - Ensures non-404 errors don't cause deletion
- `test_audit_assets_requires_authorization` - Verifies authorization check
- `test_help_has_audit_assets` - Confirms command is in help
- `test_audit_assets_batching` - Tests batch processing
- `test_audit_assets_per_rubric_reporting` - Validates per-rubric breakdown

### New Tests (2 tests)
Added `tests/test_audit_assets_enhanced_errors.py` with:
- `test_audit_assets_various_error_patterns` - Tests detection of various error message formats (MESSAGE_ID_INVALID, chat not found, etc.)
- `test_audit_assets_logging` - Verifies comprehensive logging at all key points

All 9 tests pass successfully.

## How to Use

### Running the Audit
Send `/audit_assets` as an authorized operator in Telegram. The bot will:
1. Check all assets in the database
2. Attempt to copy each message to verify it still exists
3. Delete database records for messages that no longer exist
4. Report counts per rubric

### Monitoring Logs
Look for these log patterns in production:
- `ASSETS_AUDIT_CHECKING` - Asset being verified
- `ASSETS_AUDIT_EXISTS` - Message still exists
- `ASSETS_AUDIT_COPY_FAILED` - Copy attempt failed (includes error message)
- `ASSETS_AUDIT_DEAD_SOUL` - Detected as deleted
- `ASSETS_AUDIT_DELETED` - Successfully removed from database

### Debugging Issues
If `/audit_assets` still shows "удалено 0" after deleting a photo:
1. Check logs for `ASSETS_AUDIT_COPY_FAILED` entries
2. Look at the error message in the log
3. If it's a new error pattern not in the list, add it to the `is_dead_soul` check
4. Verify the `tg_message_id` format is correct (`chat_id:message_id`)

## Expected Behavior After Fix

1. Manually delete a photo from the assets channel in Telegram
2. Run `/audit_assets` 
3. The bot should show "удалено 1" for the corresponding rubric
4. Database counts should decrease by 1
5. Logs should show:
   - `ASSETS_AUDIT_CHECKING` for the deleted asset
   - `ASSETS_AUDIT_COPY_FAILED` with the Telegram error
   - `ASSETS_AUDIT_DEAD_SOUL` detection
   - `ASSETS_AUDIT_DELETED` confirmation

## Future Improvements

Consider these enhancements:
1. Add metrics/counters for audit operations (success/failure counts)
2. Implement scheduled automatic audits
3. Add notification when large numbers of dead souls are detected
4. Cache results to avoid re-checking recently verified assets
