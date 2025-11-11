# Caption Assembly Fix: Block Raw Text, Sanitize Prompt Leaks

## Changes Made

### 1. Prompt Leak Sanitization (Defense-in-Depth)

**Location:** `main.py` lines 115-136

Added global pattern and sanitization function:
- `PROMPT_LEAK_PATTERN`: Regex to detect prompt/schema leaks
- `sanitize_prompt_leaks()`: Function that detects and removes prompt markers

**Pattern detects:**
- `###`, `***`, `---`, `>` markers followed by:
  - "Верните JSON"
  - "JSON-ответ"
  - "построить результат"
  - "post text v1"
  - "json_schema"
  - "schema"
  - "формат"
  - "шаблон"
  - "инструкция"
  - "промпт"

### 2. Clean Caption Assembly

**Location:** `main.py` lines 14719-14729

Added `_build_final_sea_caption()` method:
- Only combines caption + hashtags
- Deduplicates hashtags
- Applies prompt leak sanitization
- No extra debug markers or build context strings

### 3. JSON-Only Path (Never Publish Raw Text)

**Location:** `main.py` lines 14938-14982

Updated `_generate_sea_caption()` response processing:

**Before:**
```python
if not response or not isinstance(response.content, dict):
    continue
caption_raw = str(response.content.get("caption") or "")
hashtags = self._deduplicate_hashtags(raw_hashtags)
```

**After:**
```python
# Check for valid response with parsed JSON (not raw text fallback)
if not response or not isinstance(response.content, dict):
    logging.warning("SEA_RUBRIC json_parse_error attempt=%d (response missing or not dict)", attempt)
    continue

# Check if OpenAI client returned raw text fallback ({"raw": ...})
if "raw" in response.content and "caption" not in response.content:
    logging.warning("SEA_RUBRIC json_parse_error attempt=%d (OpenAI returned raw text, not JSON)", attempt)
    continue

# Extract caption and hashtags from parsed JSON
caption_raw = response.content.get("caption")
raw_hashtags = response.content.get("hashtags")

# Validate caption exists and is string
if not caption_raw or not isinstance(caption_raw, str):
    logging.warning("SEA_RUBRIC caption_missing_or_invalid attempt=%d", attempt)
    continue

# Build final caption with sanitization
caption, hashtags = self._build_final_sea_caption(caption, raw_hashtags)

# Fatal check: caption must be non-empty after processing
if not caption:
    logging.warning("SEA_RUBRIC empty_caption_error attempt=%d", attempt)
    continue
```

### 4. Enhanced Logging

**Added logging events:**
- `json_parse_error` - When OpenAI response fails to parse or is raw text
- `caption_missing_or_invalid` - When caption field is missing or not string
- `hashtags_missing_or_invalid` - When hashtags field is missing or not list
- `empty_caption_error` - When caption is empty after processing
- `caption_prompt_leak_detected` - When sanitization detects and removes leaks

**All logs include:**
- `attempt=N` - Current attempt number
- `source=llm|fallback` - Whether caption came from LLM or fallback
- `reason=<reason>` - Specific reason when fallback is used

### 5. Fallback Caption Sanitization

**Locations:**
- `main.py` line 15036 (in `_generate_sea_caption`)
- `main.py` line 15238 (in `_generate_sea_caption_with_timeout`)

Applied `sanitize_prompt_leaks()` to fallback captions as defense-in-depth measure.

## Testing

### Manual Test Cases

1. **Clean caption (no leak)**
   - Input: `"Порадую вас морем — тихий берег и ровный плеск.\n\nА вы знали: Балтийское море самое молодое."`
   - Output: Same (no changes)

2. **Caption with ### leak**
   - Input: `"Порадую вас морем — тихий берег.\n\n### Верните JSON\n{...}"`
   - Output: `"Порадую вас морем — тихий берег."`
   - Log: `caption_prompt_leak_detected trimmed_at_pos=X`

3. **Caption with post text v1 leak**
   - Input: `"Море спокойное.\n\n--- post text v1 ---\nformat..."`
   - Output: `"Море спокойное."`
   - Log: `caption_prompt_leak_detected trimmed_at_pos=X`

### Verification Steps

To verify the fix works in production:

1. **Monitor logs** for these patterns:
   - `SEA_RUBRIC json_parse_error` - Should be rare, indicates OpenAI response issues
   - `SEA_RUBRIC caption_prompt_leak_detected` - Should never happen (defense-in-depth)
   - `SEA_RUBRIC caption_accepted attempt=X source=llm` - Normal success path

2. **Check published captions** in Telegram:
   - No `###` markers
   - No "Верните JSON" or similar prompts
   - Clean 2-paragraph format
   - Hashtags at end

3. **Fallback behavior**:
   - If OpenAI fails, fallback caption includes LEADS + fact_sentence
   - Fallback logs show `source=fallback reason=<reason>`

## What Doesn't Change

- ✅ System/user prompts (no changes to prompt text)
- ✅ Response schema: `{"caption": str, "hashtags": [...]}`
- ✅ Hashtag deduplication logic
- ✅ Model parameters (gpt-4o, temperature, top_p)
- ✅ Timeout and retry configuration

## Acceptance Criteria Status

- ✅ No raw_text published; JSON parse → caption or fallback
- ✅ Final assembly only caption + hashtags (verified by code inspection)
- ✅ PROMPT_LEAK_PATTERN regex in place; sanitization active
- ✅ Fallback called only for: timeout, api_error, json_parse_error, empty_caption
- ✅ Style violations (length, structure) logged but don't trigger fallback
- ✅ Logs show: attempt, latency_ms, finish_reason, source=llm|fallback, reason
- ⏳ Manual test: Requires production deployment to verify 3+ captions

## Files Modified

- `main.py`:
  - Added `PROMPT_LEAK_PATTERN` (line 116)
  - Added `sanitize_prompt_leaks()` function (line 124)
  - Added `_build_final_sea_caption()` method (line 14719)
  - Updated `_generate_sea_caption()` response processing (line 14938)
  - Updated fallback logging in `_generate_sea_caption()` (line 15028)
  - Applied sanitization to fallback in both methods (lines 15036, 15238)
