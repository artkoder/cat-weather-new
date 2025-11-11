# Assets Publishing via sendPhoto

## Overview

Mobile assets are now published to Telegram channels using `sendPhoto` by default instead of `sendDocument`. This makes asset messages appear as photos (like other channel content) rather than document attachments.

## Configuration

### Environment Variable

Set `ASSETS_UPLOAD_MODE` to control publishing mode:

- `photo` (default): Use `sendPhoto` with automatic fallback to `sendDocument`
- `document`: Always use `sendDocument` (legacy behavior)

Example:
```bash
ASSETS_UPLOAD_MODE=photo  # Default
# or
ASSETS_UPLOAD_MODE=document  # Legacy mode
```

## Features

### Automatic Format Handling

The system automatically handles various image formats:

- **JPEG, PNG, WebP**: Published directly via `sendPhoto`
- **HEIC/HEIF**: Converted to JPEG (quality 90) before publishing
- **EXIF Orientation**: Applied automatically since Telegram strips EXIF data from photos

### Fallback Mechanism

If `sendPhoto` fails (due to format/size issues or Telegram errors), the system automatically falls back to `sendDocument`:

- Catches all exceptions during `sendPhoto`
- Logs fallback reason
- Publishes same file via `sendDocument`
- Increments `assets_publish_fallback` metric

### Caption Truncation

Captions are automatically truncated to 1024 characters (Telegram's limit for `sendPhoto`):

- Soft truncation at word boundaries when possible
- Adds ellipsis (…) to indicate truncation
- Safe for all UTF-8 content

### Structured Logging

All publishing events are logged with structured data:

- `assets_publish_attempt`: Before attempting `sendPhoto`
- `assets_publish_ok`: After successful publish
- `assets_publish_fallback`: When falling back to `sendDocument`
- Log fields: `mode`, `mime`, `size_bytes`, `message_id`, `error`

### Metrics

The following Prometheus metrics are incremented:

- `assets_publish_attempt`: Each `sendPhoto` attempt
- `assets_publish_ok`: Successful publishes (photo or document)
- `assets_publish_fallback`: Fallbacks from photo to document

## Telegram Response Handling

### sendPhoto Response

Photo responses contain an array of sizes. The system extracts the largest size for storage:

```python
{
  "message_id": 123,
  "photo": [
    {"file_id": "...", "width": 90, "height": 67},
    {"file_id": "...", "width": 320, "height": 240},
    {"file_id": "...", "width": 640, "height": 480}  # <- largest, stored
  ]
}
```

### sendDocument Response

Document responses contain direct file metadata:

```python
{
  "message_id": 123,
  "document": {
    "file_id": "...",
    "file_unique_id": "...",
    "mime_type": "image/jpeg",
    "file_size": 12345
  }
}
```

## Database Storage

Both modes store the same fields:

- `message_id`: Telegram message identifier
- `file_id`: Telegram file identifier (largest size for photos)
- `file_unique_id`: Unique file identifier
- `mime_type`: Original MIME type
- `width`, `height`: Image dimensions

## Testing

Run the dedicated test suite:

```bash
pytest tests/test_assets_sendphoto.py -v
```

Test coverage includes:

- Mode configuration (photo/document/invalid)
- Caption truncation (short/long/boundary cases)
- sendPhoto success
- sendDocument mode
- Automatic fallback
- Metrics increments

## Migration Notes

### From sendDocument to sendPhoto

No action required. New uploads will automatically use `sendPhoto` with fallback.

### Reverting to sendDocument

Set environment variable:

```bash
ASSETS_UPLOAD_MODE=document
```

All future uploads will use `sendDocument` without attempting `sendPhoto`.

## Implementation Details

### Code Changes

1. `ingestion.py`:
   - `_get_assets_upload_mode()`: Reads environment variable
   - `_truncate_caption()`: Safe caption truncation
   - Modified `_ingest_photo_internal()`: Try-catch with fallback logic

2. Tests:
   - `tests/test_assets_sendphoto.py`: New test suite
   - Updated existing tests to expect `sendPhoto` by default

### Files Modified

- `ingestion.py`: Core publishing logic
- `tests/test_assets_sendphoto.py`: New tests
- `tests/test_upload_jobs.py`: Updated assertions

## Troubleshooting

### All uploads using sendDocument

Check that `ASSETS_UPLOAD_MODE` is set to `photo` (or unset, defaults to photo).

### High fallback rate

Check logs for fallback reasons:

```bash
grep "assets_publish_fallback" logs/app.log
```

Common reasons:
- Unsupported format
- File size exceeds limits
- Telegram API errors

### Caption truncation issues

Captions longer than 1024 characters are automatically truncated. Check source data if truncation is unexpected.

## Future Enhancements

Possible improvements:

1. Size validation before sendPhoto attempt
2. Format detection with early fallback
3. Configurable truncation length
4. Retry logic with exponential backoff
5. Compression quality tuning
