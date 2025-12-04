from __future__ import annotations

import asyncio
import io
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import google.generativeai as genai
from PIL import Image, ImageDraw

from openai_client import OpenAIClient

RAW_ANSWER_HIGHLIGHT_MODEL_ID = "gemini-2.5-flash-lite"
_raw_answer_highlight_model: genai.GenerativeModel | None = None

PADDING = 2


@dataclass
class HighlightExtraction:
    boxes: list[Mapping[str, Any]]
    page_lines: list[Mapping[str, Any]]
    answers: list[str]
    page_bottom_y: float | None


@dataclass
class OCRWord:
    index: int
    text: str
    bbox: tuple[float, float, float, float]
    conf: float | None = None


@dataclass
class PageHighlightResult:
    boxes: list[Mapping[str, float]]
    spans: list[tuple[int, int]]
    page_lines: list[Mapping[str, Any]]
    quote_index_to_boxes: dict[int, list[Mapping[str, float]]] = field(default_factory=dict)


def get_highlight_model() -> genai.GenerativeModel:
    global _raw_answer_highlight_model
    if _raw_answer_highlight_model is None:
        _raw_answer_highlight_model = genai.GenerativeModel(
            RAW_ANSWER_HIGHLIGHT_MODEL_ID,
            generation_config={"response_mime_type": "application/json"},
        )
    return _raw_answer_highlight_model


async def _extract_boxes_with_gemini(image_bytes: bytes, user_query: str) -> HighlightExtraction:
    def _generate() -> HighlightExtraction:
        try:
            model = get_highlight_model()
            try:
                image_part = Image.open(io.BytesIO(image_bytes))
            except Exception as img_exc:
                logging.error(
                    "Gemini highlight: Failed to create PIL Image from bytes: %s", img_exc
                )
                return HighlightExtraction(boxes=[], page_lines=[], answers=[])

            prompt = f"""
You are a document analysis assistant.
User query: "{user_query}"

Task: Identify lines that are relevant to the user's query or contain parts of the answer.

Additionally, transcribe EVERY line of text on the page in reading order so that the operator can review the full content.
For each transcribed line provide normalized vertical coordinates of the top and bottom of the text (y0 and y1 between 0 and 1, relative to the image height).

CRITICAL INSTRUCTIONS:
1. **Relevant Lines:** Identify lines that are relevant to the user's query or contain parts of the answer.
2. **Line Numbers Only:** You must not return bounding boxes. Instead, provide the line numbers of the lines that contain the direct answer.
3. **Per-Line Matches:** Each item should describe a single line or a tight group of adjacent lines that exactly answer the query.
4. **Full Transcription:** Provide the complete text of the page, line by line, in a machine-readable format.

Output format:
Return a JSON object with keys "items" and "page_lines".
"items" is a list of:
- "line_numbers": [list of line numbers where the answer text appears]
- "content": "The exact text string contained in these lines"

Also include "page_bottom_y": the Y coordinate (normalized 0-1, relative to image height) of the lowest text on the page (i.e.,
the bottom of the final line).

"page_lines" is an array of objects ordered from line 1 to the last line of the page.
Each object MUST include:
- "text": the line text string
- "y0": top Y coordinate normalized to the image height (0-1)
- "y1": bottom Y coordinate normalized to the image height (0-1)
"""
            logging.info("DEBUG_GEMINI_PROMPT: %s", prompt)
            response = model.generate_content([image_part, prompt])
            logging.info("DEBUG_GEMINI_RESPONSE: %s", response.text)
            data = json.loads(response.text)
            items = data.get("items", []) if isinstance(data, dict) else []
            if not isinstance(items, list):
                items = []
            normalized: list[Mapping[str, Any]] = []
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                raw_lines = item.get("line_numbers") or item.get("lines") or []
                if not isinstance(raw_lines, Sequence):
                    continue
                try:
                    line_numbers = [int(num) for num in raw_lines if int(num) > 0]
                except (TypeError, ValueError):
                    continue
                if not line_numbers:
                    continue
                content = item.get("content") or item.get("text") or ""
                if not isinstance(content, str):
                    content = ""
                normalized.append(
                    {
                        "line_numbers": line_numbers,
                        "text": content.strip(),
                    }
                )
            page_lines_raw = data.get("page_lines", []) if isinstance(data, dict) else []
            page_lines: list[Mapping[str, Any]] = []
            page_bottom_y: float | None = None
            if isinstance(page_lines_raw, Sequence) and not isinstance(
                page_lines_raw, (str, bytes)
            ):
                for entry in page_lines_raw:
                    if entry is None:
                        continue
                    if isinstance(entry, Mapping):
                        text_value = str(entry.get("text") or entry.get("line") or "").strip()
                        y0_raw = entry.get("y0") if "y0" in entry else entry.get("y_top")
                        y1_raw = entry.get("y1") if "y1" in entry else entry.get("y_bottom")
                    else:
                        text_value = str(entry).strip()
                        y0_raw = None
                        y1_raw = None
                    if not text_value:
                        continue
                    try:
                        y0_val = float(y0_raw) if y0_raw is not None else None
                    except (TypeError, ValueError):
                        y0_val = None
                    try:
                        y1_val = float(y1_raw) if y1_raw is not None else None
                    except (TypeError, ValueError):
                        y1_val = None
                    page_lines.append({"text": text_value, "y0": y0_val, "y1": y1_val})
                    if y1_val is not None:
                        page_bottom_y = max(page_bottom_y or y1_val, y1_val)
            if page_bottom_y is None and isinstance(data, dict):
                try:
                    page_bottom_y = float(data.get("page_bottom_y"))
                except (TypeError, ValueError):
                    page_bottom_y = None
            answers = [item.get("text", "") for item in normalized if item.get("text")]
            return HighlightExtraction(
                boxes=normalized,
                page_lines=page_lines,
                answers=answers,
                page_bottom_y=page_bottom_y,
            )
        except Exception as exc:  # pragma: no cover - external dependency
            logging.exception("Gemini highlight failed: %s", exc)
            return HighlightExtraction(
                boxes=[],
                page_lines=[],
                answers=[],
                page_bottom_y=None,
            )

    return await asyncio.to_thread(_generate)


async def extract_text_coordinates(
    image_bytes: bytes, query_text: str, client: OpenAIClient | None = None
) -> HighlightExtraction:
    logging.debug("RAW_ANSWER using Gemini highlight extraction")
    return await _extract_boxes_with_gemini(image_bytes, query_text)


def build_answer_boxes(
    page_lines: Sequence[Mapping[str, Any]] | None,
    answers: Sequence[str] | None,
) -> list[Mapping[str, float]]:
    if not page_lines or not answers:
        return []

    answers_normalized = {str(ans).strip().lower() for ans in answers if str(ans).strip()}
    if not answers_normalized:
        return []

    def _normalize_coord(raw: Any) -> float | None:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        if value > 1:
            value /= 1000.0
        if value < 0:
            return None
        return min(value, 1.0)

    boxes: list[Mapping[str, float]] = []
    for entry in page_lines:
        if not isinstance(entry, Mapping):
            continue
        text = str(entry.get("text") or entry.get("line") or "").strip()
        if not text or text.lower() not in answers_normalized:
            continue
        y0 = _normalize_coord(entry.get("y0") if "y0" in entry else entry.get("y_top"))
        y1 = _normalize_coord(entry.get("y1") if "y1" in entry else entry.get("y_bottom"))
        if y0 is None or y1 is None:
            continue
        if y0 >= y1:
            continue
        boxes.append({"x0": 0.0, "y0": y0, "x1": 1.0, "y1": y1})

    return boxes


def draw_highlight_overlay(image_bytes: bytes, boxes: Sequence[Mapping[str, Any]]) -> bytes | None:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            img = image.convert("RGBA")
    except Exception:
        logging.exception("RAW_ANSWER failed to open image for highlighting")
        return None

    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = img.size
    for box in boxes:
        try:
            x0 = max(0, min(width, int(round(box.get("x0", 0) * width))))
            y0 = max(0, min(height, int(round(box.get("y0", 0) * height))))
            x1 = max(0, min(width, int(round(box.get("x1", 0) * width))))
            y1 = max(0, min(height, int(round(box.get("y1", 0) * height))))
        except Exception:
            continue
        x0 = max(0, x0 - PADDING)
        y0 = max(0, y0 - PADDING)
        x1 = min(width, x1 + PADDING)
        y1 = min(height, y1 + PADDING)
        if x0 >= x1 or y0 >= y1:
            continue
        draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 0, 100))

    composed = Image.alpha_composite(img, overlay).convert("RGB")
    buffer = io.BytesIO()
    composed.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.read()


_WORD_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _normalize_word_token(text: str) -> str:
    if not text:
        return ""
    token = "".join(_WORD_TOKEN_RE.findall(text)).strip().lower()
    return token or text.strip().lower()


def _tokenize_quote(text: str) -> list[str]:
    if not text:
        return []
    return [
        _normalize_word_token(match.group(0))
        for match in _WORD_TOKEN_RE.finditer(text)
        if _normalize_word_token(match.group(0))
    ]


def _find_quote_span(tokens: Sequence[str], quote_tokens: Sequence[str]) -> tuple[int, int] | None:
    if not tokens or not quote_tokens:
        return None
    max_start = len(tokens) - len(quote_tokens)
    for start in range(max_start + 1):
        window = tokens[start : start + len(quote_tokens)]
        if window == list(quote_tokens):
            return start, start + len(quote_tokens) - 1
    return None


def locate_quotes_in_ocr(
    words: Sequence[OCRWord],
    quotes: Sequence[str],
    *,
    page_size: tuple[int, int] | None = None,
) -> PageHighlightResult | None:
    if not words:
        return None

    tokens = [_normalize_word_token(word.text) for word in words]
    normalized_quotes = [_tokenize_quote(quote) for quote in quotes if isinstance(quote, str)]
    spans: list[tuple[int, int]] = []
    for quote_tokens in normalized_quotes:
        span = _find_quote_span(tokens, quote_tokens)
        if span and span not in spans:
            spans.append(span)

    page_width = page_size[0] if page_size else 0
    page_height = page_size[1] if page_size else 0
    page_lines: list[Mapping[str, Any]] = []
    boxes: list[Mapping[str, float]] = []
    if page_width and page_height:
        page_lines = build_page_lines(words, page_width=page_width, page_height=page_height)
        boxes = _build_span_boxes(
            words,
            spans,
            page_width=page_width,
            page_height=page_height,
        )

    if not spans and not boxes and not page_lines:
        return None

    return PageHighlightResult(
        boxes=boxes,
        spans=spans,
        page_lines=page_lines,
        quote_index_to_boxes={},
    )


async def locate_citations_on_page(
    quotes: Sequence[str],
    ocr_words: Sequence[OCRWord],
    *,
    page_size: tuple[int, int] | None = None,
    llm_client: genai.GenerativeModel | None = None,
) -> PageHighlightResult | None:
    if not quotes or not ocr_words:
        return None

    def _generate() -> PageHighlightResult | None:
        try:
            model = llm_client or get_highlight_model()
            quotes_list = [f"{idx}: {quote}" for idx, quote in enumerate(quotes)]
            quotes_block = "\n".join(quotes_list) if quotes_list else ""
            words_listing = build_words_with_indices(ocr_words)
            page_text = build_page_text(ocr_words)
            prompt = f"""
You are given OCR output for a scanned page.

QUOTES (each item is quote_index: text):
{quotes_block}

PAGE TEXT (reading order):
{page_text}

WORDS (use 0-based inclusive indices to mark spans):
{words_listing}

Return STRICT JSON with a top-level "matches" array. Each item must be an object:
{{"quote_index": <int>, "spans": [[start_idx, end_idx], ...]}}

- Use the WORDS list for indices.
- start_idx and end_idx are inclusive and must reference existing indices.
- If a quote is not present, return an empty spans list for that quote.
"""
            response = model.generate_content(prompt)
            data = json.loads(response.text)
            matches_raw = data.get("matches") if isinstance(data, Mapping) else None

            spans_by_quote: dict[int, list[tuple[int, int]]] = {}
            total_words = len(ocr_words)

            if isinstance(matches_raw, Sequence) and not isinstance(matches_raw, (str, bytes)):
                # Case 1: matches are objects with quote_index
                for entry in matches_raw:
                    quote_index: int | None = None
                    spans_raw: Any = None
                    if isinstance(entry, Mapping):
                        try:
                            quote_index = int(entry.get("quote_index"))
                        except (TypeError, ValueError):
                            quote_index = None
                        spans_raw = entry.get("spans") if "spans" in entry else entry.get("indices")
                    elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
                        if len(entry) == 2 and isinstance(entry[0], int):
                            quote_index = entry[0]
                            spans_raw = entry[1]
                    if quote_index is None or quote_index < 0:
                        continue
                    spans = _normalize_spans(spans_raw, total_words)
                    if spans:
                        spans_by_quote.setdefault(quote_index, []).extend(spans)

                # Case 2: ordered list matching quotes
                if not spans_by_quote and len(matches_raw) == len(quotes):
                    for idx, entry in enumerate(matches_raw):
                        spans = _normalize_spans(entry, total_words)
                        if spans:
                            spans_by_quote[idx] = spans

            if not spans_by_quote:
                logging.info("RAW_ANSWER Gemini returned no citation matches")
                return None

            all_spans: list[tuple[int, int]] = []
            boxes: list[Mapping[str, float]] = []
            quote_index_to_boxes: dict[int, list[Mapping[str, float]]] = {}
            page_lines: list[Mapping[str, Any]] = []
            for spans in spans_by_quote.values():
                for span in spans:
                    if span not in all_spans:
                        all_spans.append(span)
            if page_size and page_size[0] and page_size[1]:
                page_lines = build_page_lines(
                    ocr_words, page_width=page_size[0], page_height=page_size[1]
                )
                for quote_index, spans in spans_by_quote.items():
                    quote_boxes = _build_span_boxes(
                        ocr_words,
                        spans,
                        page_width=page_size[0],
                        page_height=page_size[1],
                    )
                    if quote_boxes:
                        quote_index_to_boxes[quote_index] = quote_boxes
                        boxes.extend(quote_boxes)

            return PageHighlightResult(
                boxes=boxes,
                spans=all_spans,
                page_lines=page_lines,
                quote_index_to_boxes=quote_index_to_boxes,
            )
        except Exception:  # pragma: no cover - external dependency
            logging.exception("RAW_ANSWER citation locating failed")
            return None

    return await asyncio.to_thread(_generate)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_ocr_words(payload: Mapping[str, Any]) -> tuple[list[OCRWord], tuple[int, int] | None]:
    words_raw = payload.get("words")
    if not isinstance(words_raw, Sequence) or isinstance(words_raw, (str, bytes)):
        return [], None

    def _extract_dim(*keys: str) -> int | None:
        for key in keys:
            if key in payload:
                try:
                    return int(payload[key])
                except (TypeError, ValueError):
                    continue
        return None

    width = _extract_dim("width", "page_width", "image_width")
    height = _extract_dim("height", "page_height", "image_height")
    page_size = (width, height) if width and height else None

    words: list[OCRWord] = []
    for idx, raw in enumerate(words_raw):
        if not isinstance(raw, Mapping):
            continue
        text = str(raw.get("text") or "").strip()
        bbox_raw = raw.get("bbox") or {}
        if not isinstance(bbox_raw, Mapping):
            continue
        x = _safe_float(bbox_raw.get("x"))
        y = _safe_float(bbox_raw.get("y"))
        w = _safe_float(bbox_raw.get("w"))
        h = _safe_float(bbox_raw.get("h"))
        if None in (x, y, w, h):
            continue
        conf = _safe_float(raw.get("conf"))
        words.append(OCRWord(index=idx, text=text, bbox=(x, y, w, h), conf=conf))

    return words, page_size


def _group_words_into_lines(
    words: Sequence[OCRWord], *, tolerance_factor: float = 1.25
) -> list[list[OCRWord]]:
    if not words:
        return []

    sorted_words = sorted(words, key=lambda w: (w.bbox[1] + w.bbox[3] / 2.0, w.bbox[0]))
    lines: list[list[OCRWord]] = []
    current_line: list[OCRWord] = []
    current_center: float | None = None
    current_height: float | None = None

    for word in sorted_words:
        center_y = word.bbox[1] + word.bbox[3] / 2.0
        height = word.bbox[3]
        if current_line:
            assert current_center is not None and current_height is not None
            threshold = max(current_height, height) * tolerance_factor
            if abs(center_y - current_center) <= threshold:
                current_line.append(word)
                current_center = (current_center * (len(current_line) - 1) + center_y) / len(
                    current_line
                )
                current_height = max(current_height, height)
            else:
                lines.append(current_line)
                current_line = [word]
                current_center = center_y
                current_height = height
        else:
            current_line = [word]
            current_center = center_y
            current_height = height

    if current_line:
        lines.append(current_line)

    return [sorted(line, key=lambda w: w.bbox[0]) for line in lines]


def build_page_lines(
    words: Sequence[OCRWord], *, page_width: float, page_height: float
) -> list[Mapping[str, Any]]:
    if not words or page_width <= 0 or page_height <= 0:
        return []

    lines: list[Mapping[str, Any]] = []
    for line_words in _group_words_into_lines(words):
        x1 = min(w.bbox[0] for w in line_words)
        x2 = max(w.bbox[0] + w.bbox[2] for w in line_words)
        y1 = min(w.bbox[1] for w in line_words)
        y2 = max(w.bbox[1] + w.bbox[3] for w in line_words)
        text_value = " ".join(w.text for w in line_words if w.text)
        if not text_value:
            continue
        lines.append(
            {
                "text": text_value,
                "y0": max(0.0, y1 / page_height),
                "y1": min(1.0, y2 / page_height),
            }
        )

    return lines


def build_page_text(words: Sequence[OCRWord]) -> str:
    if not words:
        return ""

    lines = _group_words_into_lines(words)
    parts: list[str] = []
    for line_words in lines:
        line_text = " ".join(w.text for w in line_words if w.text)
        if line_text:
            parts.append(line_text)
    return "\n".join(parts)


def build_words_with_indices(words: Sequence[OCRWord]) -> str:
    if not words:
        return "WORDS:\n"
    lines = ["WORDS:"] + [f"{word.index}: {word.text}" for word in words]
    return "\n".join(lines)


def _normalize_spans(spans: Any, total_words: int) -> list[tuple[int, int]]:
    normalized: list[tuple[int, int]] = []
    if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes)):
        return normalized
    for raw in spans:
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            continue
        if len(raw) != 2:
            continue
        try:
            start, end = int(raw[0]), int(raw[1])
        except (TypeError, ValueError):
            continue
        if start < 0 or end < 0 or start >= total_words:
            logging.warning("RAW_ANSWER span outside range: %s", raw)
            continue
        end = min(end, total_words - 1)
        if start > end:
            start, end = end, start
        normalized.append((start, end))
    return normalized


def _build_span_boxes(
    words: Sequence[OCRWord],
    spans: Sequence[tuple[int, int]],
    *,
    page_width: float,
    page_height: float,
) -> list[Mapping[str, float]]:
    if not words or not spans or page_width <= 0 or page_height <= 0:
        return []

    boxes: list[Mapping[str, float]] = []
    word_map = {word.index: word for word in words}

    for start, end in spans:
        span_words = [word_map[idx] for idx in range(start, end + 1) if idx in word_map]
        if not span_words:
            continue
        for line_words in _group_words_into_lines(span_words):
            x1 = min(w.bbox[0] for w in line_words)
            x2 = max(w.bbox[0] + w.bbox[2] for w in line_words)
            y1 = min(w.bbox[1] for w in line_words)
            y2 = max(w.bbox[1] + w.bbox[3] for w in line_words)
            boxes.append(
                {
                    "x0": max(0.0, x1 / page_width),
                    "y0": max(0.0, y1 / page_height),
                    "x1": min(1.0, x2 / page_width),
                    "y1": min(1.0, y2 / page_height),
                }
            )

    return boxes


async def highlight_answer_on_page(
    *,
    question: str,
    chunk_text: str,
    page_text: str,
    words: Sequence[OCRWord],
    scan_image_bytes: bytes,
    source_page_size: tuple[int, int] | None = None,
    model_client: genai.GenerativeModel | None = None,
) -> PageHighlightResult | None:
    if not words:
        return None

    def _generate() -> PageHighlightResult | None:
        try:
            model = model_client or get_highlight_model()
            try:
                image_part = Image.open(io.BytesIO(scan_image_bytes))
            except Exception:
                logging.exception("RAW_ANSWER failed to open image for Gemini highlighting")
                return None

            width, height = image_part.size
            scaled_words = list(words)
            if source_page_size and source_page_size[0] and source_page_size[1]:
                scale_x = width / float(source_page_size[0])
                scale_y = height / float(source_page_size[1])
                scaled_words = [
                    OCRWord(
                        index=w.index,
                        text=w.text,
                        bbox=(
                            w.bbox[0] * scale_x,
                            w.bbox[1] * scale_y,
                            w.bbox[2] * scale_x,
                            w.bbox[3] * scale_y,
                        ),
                        conf=w.conf,
                    )
                    for w in words
                ]

            words_with_indices = build_words_with_indices(scaled_words)
            prompt = f"""Вопрос пользователя:
{question}

Фрагмент текста книги, который мы уже считаем релевантным (chunk):
{chunk_text}

Текст страницы по OCR:
{page_text}

Ниже список слов этой страницы с индексами:
{words_with_indices}

Ты видишь и изображение страницы, и текст.
Найди на странице (по изображению и тексту) слова, которые по смыслу лучше всего отвечают на вопрос.
Используй список WORDS, чтобы указать, какие именно слова входят в ответ.

Верни строго JSON без комментариев и дополнительного текста:
{{"spans": [[start_idx, end_idx], ...]}}
где start_idx и end_idx — целые индексы (0-based) слов из списка WORDS (включительно)."""

            response = model.generate_content([image_part, prompt])
            data = json.loads(response.text)
            spans_raw = data.get("spans") if isinstance(data, Mapping) else None
            spans = _normalize_spans(spans_raw, len(scaled_words))
            if not spans:
                logging.info("RAW_ANSWER Gemini returned no spans for highlighting")
                return None

            boxes = _build_span_boxes(
                scaled_words,
                spans,
                page_width=width,
                page_height=height,
            )
            page_lines = build_page_lines(scaled_words, page_width=width, page_height=height)
            return PageHighlightResult(
                boxes=boxes,
                spans=spans,
                page_lines=page_lines,
                quote_index_to_boxes={},
            )
        except Exception:
            logging.exception("RAW_ANSWER Gemini span extraction failed")
            return None

    return await asyncio.to_thread(_generate)
