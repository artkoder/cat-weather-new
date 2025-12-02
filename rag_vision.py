from __future__ import annotations

import asyncio
import io
import json
import logging
from typing import Any, Mapping, Sequence

import google.generativeai as genai
from PIL import Image, ImageDraw

from openai_client import OpenAIClient

RAW_ANSWER_HIGHLIGHT_MODEL_ID = "gemini-2.0-flash"
_raw_answer_highlight_model: genai.GenerativeModel | None = None


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
) -> list[Mapping[str, Any]]:
    def _generate() -> list[Mapping[str, Any]]:
        try:
            model = get_highlight_model()
            try:
                image_part = Image.open(io.BytesIO(image_bytes))
            except Exception as img_exc:
                logging.error(
                    "Gemini highlight: Failed to create PIL Image from bytes: %s", img_exc
                )
                return []

            prompt = f"""
You are a strict document analysis assistant.
User query: "{user_query}"

Task: Identify the EXACT lines of text on this page that directly answer the user's query.

CRITICAL INSTRUCTIONS:
1. **Direct Answer Only:** If the page mentions the topic/names but does NOT contain the specific answer to the question, return {{ "items": [] }}. Do not guess.
2. **Line-by-Line Highlighting:** Do NOT highlight entire paragraphs with a single box. You must return separate bounding boxes for EACH visual line of text.
3. **Precision:** Boxes must tightly enclose the text.

Output format:
Return a JSON object with a key "items".
Each item must have:
- "box_2d": [ymin, xmin, ymax, xmax] (normalized 0-1000)
- "content": "The exact text string contained in this box"
"""
            logging.info("DEBUG_GEMINI_PROMPT: %s", prompt)
            response = model.generate_content([image_part, prompt])
            logging.info("DEBUG_GEMINI_RESPONSE: %s", response.text)
            data = json.loads(response.text)
            items = data.get("items", []) if isinstance(data, dict) else []
            if not isinstance(items, list):
                return []
            normalized: list[Mapping[str, Any]] = []
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                box = item.get("box_2d")
                if not (isinstance(box, Sequence) and len(box) == 4):
                    continue
                try:
                    ymin, xmin, ymax, xmax = (
                        float(box[0]) / 1000.0,
                        float(box[1]) / 1000.0,
                        float(box[2]) / 1000.0,
                        float(box[3]) / 1000.0,
                    )
                except (TypeError, ValueError):
                    continue
                content = item.get("content") or item.get("text") or ""
                if not isinstance(content, str):
                    content = ""
                normalized.append(
                    {
                        "x0": xmin,
                        "y0": ymin,
                        "x1": xmax,
                        "y1": ymax,
                        "text": content.strip(),
                    }
                )
            return normalized
        except Exception as exc:  # pragma: no cover - external dependency
            logging.exception("Gemini highlight failed: %s", exc)
            return []

    return await asyncio.to_thread(_generate)


async def extract_text_coordinates(
    image_bytes: bytes, query_text: str, client: OpenAIClient | None = None
) -> list[Mapping[str, Any]]:
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
