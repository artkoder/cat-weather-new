from __future__ import annotations

import asyncio
import io
import json
import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import google.generativeai as genai
from PIL import Image, ImageDraw

from openai_client import OpenAIClient

RAW_ANSWER_HIGHLIGHT_MODEL_ID = "gemini-2.5-flash-lite"
_raw_answer_highlight_model: genai.GenerativeModel | None = None


@dataclass
class HighlightExtraction:
    boxes: list[Mapping[str, Any]]
    page_lines: list[str]


def get_highlight_model() -> genai.GenerativeModel:
    global _raw_answer_highlight_model
    if _raw_answer_highlight_model is None:
        _raw_answer_highlight_model = genai.GenerativeModel(
            RAW_ANSWER_HIGHLIGHT_MODEL_ID,
            generation_config={"response_mime_type": "application/json"},
        )
    return _raw_answer_highlight_model


async def _extract_boxes_with_gemini(
    image_bytes: bytes, user_query: str
) -> HighlightExtraction:
    def _generate() -> HighlightExtraction:
        try:
            model = get_highlight_model()
            try:
                image_part = Image.open(io.BytesIO(image_bytes))
            except Exception as img_exc:
                logging.error(
                    "Gemini highlight: Failed to create PIL Image from bytes: %s", img_exc
                )
                return HighlightExtraction(boxes=[], page_lines=[])

            prompt = f"""
You are a strict document analysis assistant.
User query: "{user_query}"

Task: Identify the EXACT lines of text on this page that directly answer the user's query.

Additionally, transcribe EVERY line of text on the page in reading order so that the operator can review the full content.

CRITICAL INSTRUCTIONS:
1. **Direct Answer Only:** If the page mentions the topic/names but does NOT contain the specific answer to the question, return {{ "items": [] }}. Do not guess.
2. **Line Numbers Only:** You must not return bounding boxes. Instead, provide the line numbers of the lines that contain the direct answer.
3. **Per-Line Matches:** Each item should describe a single line or a tight group of adjacent lines that exactly answer the query.
4. **Full Transcription:** Provide the complete text of the page, line by line, in a machine-readable format.

Output format:
Return a JSON object with keys "items" and "page_lines".
"items" is a list of:
- "line_numbers": [list of line numbers where the answer text appears]
- "content": "The exact text string contained in these lines"

"page_lines" is an array of strings ordered from line 1 to the last line of the page.
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
            page_lines: list[str] = []
            if isinstance(page_lines_raw, Sequence) and not isinstance(page_lines_raw, (str, bytes)):
                for entry in page_lines_raw:
                    if entry is None:
                        continue
                    text_value = str(entry).strip()
                    if text_value:
                        page_lines.append(text_value)
            return HighlightExtraction(boxes=normalized, page_lines=page_lines)
        except Exception as exc:  # pragma: no cover - external dependency
            logging.exception("Gemini highlight failed: %s", exc)
            return HighlightExtraction(boxes=[], page_lines=[])

    return await asyncio.to_thread(_generate)


async def extract_text_coordinates(
    image_bytes: bytes, query_text: str, client: OpenAIClient | None = None
) -> HighlightExtraction:
    logging.debug("RAW_ANSWER using Gemini highlight extraction")
    return await _extract_boxes_with_gemini(image_bytes, query_text)


def draw_highlight_overlay(
    image_bytes: bytes, boxes: Sequence[Mapping[str, Any]]
) -> bytes | None:
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
        if x0 >= x1 or y0 >= y1:
            continue
        draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 0, 100))

    composed = Image.alpha_composite(img, overlay).convert("RGB")
    buffer = io.BytesIO()
    composed.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.read()
