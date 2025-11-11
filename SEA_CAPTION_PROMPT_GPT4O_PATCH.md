# Sea Caption Prompt Patch for GPT-4o

## Summary
Applied targeted prompt improvements to sea rubric caption generation to improve text quality, readability, and emotional resonance.

## Changes Made

### 1. System Prompt Update (`_generate_sea_caption()` in main.py)

Updated system prompt with 6 structured blocks:

#### Block 1: Форма и тон (Form and Tone)
- 2-paragraph structure required
- Warm, calm intro (1-2 short sentences, can be very brief like "Порадую вас морем")
- 1 emoji allowed only in intro (optional)
- Fact as separate paragraph with clear lead-in
- Day_part awareness without explicit time mentions
- No weather parameter listings in intro

#### Block 2: Лексика (Lexicon)
- Anti-cliché list: дышит, шепчет, манит, ласкает, нежится, обнимает, зовёт/зовет, очаровывает/зачаровывает, завораживает, волшебный, по-настоящему, буквально, наполнен, вдохновляет, приглашает окунуться, голос моря, игра волн, умиротворение
- Emotion conveyed through 1-2 concrete sensory details
- No chains of epithets

#### Block 3: LEADS (Fact Transition Phrases)
Predefined list:
- «А вы знали?»
- «Интересный факт:»
- «Это интересно:»
- «Кстати:»
- «К слову о Балтике,»
- «Поделюсь фактом:»
- «Любопытная деталь:»
- «Небольшое пояснение:»
- «Теперь вы будете знать:»
- «К слову о море,»

Natural equivalents allowed if clearly signaling fact content.

#### Block 4: Работа с fact_sentence (Fact Handling)
- Gentle paraphrasing allowed without distorting meaning
- Numbers/names/terms must be correct
- One complete sentence

#### Block 5: Ограничения по длине (Length Limits)
- Intro ≤ 220 characters
- Total caption ≤ 350 characters
- Up to 400 allowed if fact_sentence is long
- Hashtags separate (not in caption text)

#### Block 6: Читабельность и проверка (Readability and Checking)
- Easy, natural reading
- One thought = one sentence
- No bureaucratic style
- Spelling and punctuation check before response
- Paragraphs separated by single blank line

### 2. User Prompt Update

Restructured user prompt with:
- Clear 2-paragraph instruction
- Intro: 1-2 short sentences, 1 emoji allowed, no weather parameter listing
- Fact: starts with LEADS or equivalent, preserves fact_sentence meaning in 1 sentence
- Length limits: intro ≤220, total ≤350 (or ≤400 for long facts)
- No exclamation marks or rhetorical questions
- Paragraphs separated by blank line
- JSON format: `{"caption":"<два абзаца>","hashtags":[...]}`
- Self-check instruction for logical transition, Baltic/scene relevance, correct punctuation

### 3. Test Update

Updated `tests/test_sea_caption_prompt.py`:
- Changed assertion from `"сохрани все цифры"` to `"числа/названия/термины"` to match new prompt wording

## Unchanged Elements

✅ Response schema: `{"caption": str, "hashtags": [...]}`
✅ Model parameters: gpt-4o, temperature, retries, timeouts
✅ Hashtag deduplication logic
✅ Post-processing pipeline
✅ day_part integration logic
✅ Fallback caption mechanism

## Testing Criteria (DoD)

### Format & Structure
- Response is valid JSON with "caption" and "hashtags" keys
- Caption contains exactly 2 paragraphs separated by 1 blank line
- JSON parse succeeds; no unclosed quotes or escaping errors

### Intro (paragraph 1)
- ≤220 characters
- Does NOT list weather parameters or numbers
- Contains ≤1 emoji, placed only in this paragraph (or 0 emojis)
- Free of anti-cliché stop-list words
- Warm, calm tone; natural and conversational
- Uses sensory details rather than chains of epithets

### Fact transition & paragraph 2
- Paragraph 2 starts with LEADS marker OR natural equivalent
- Transition is obvious and readable
- Fact is exactly 1 sentence
- Fact reflects fact_sentence meaning without distortion
- Facts, numbers, toponyms preserved correctly
- Fact connects to Baltic Sea / scene context

### Overall
- Total caption length ≤350 characters; ≤400 if fact_sentence was long
- No exclamation marks or rhetorical questions (except in leads)
- Spelling and punctuation correct
- Text reads naturally; one idea = one sentence; no bureaucratic style

### Emoji
- If emoji present: exactly 1, only in intro
- If emoji absent: acceptable (rule is ≤1, not mandatory)

## Implementation Details

**File Modified:** `/home/engine/project/main.py`
- Method: `_generate_sea_caption()` (line ~14695)
- System prompt: lines 14761-14791
- User prompt: lines 14817-14833

**Test File Modified:** `/home/engine/project/tests/test_sea_caption_prompt.py`
- Updated assertion at line 85

## Notes

- Multiline strings properly escaped
- Day_part instruction integration preserved
- All existing functionality maintained
- No breaking changes to API or response format
