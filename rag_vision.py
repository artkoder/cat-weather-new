from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import google.generativeai as genai
from PIL import Image, ImageDraw

from openai_client import OpenAIClient

RAW_ANSWER_HIGHLIGHT_MODEL_ID = "gemini-2.5-flash-lite"
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
) -> list[Mapping[str, float]]:
    def _generate() -> list[Mapping[str, float]]:
        try:
            model = get_highlight_model()
            uploaded = genai.upload_file(path=None, file=image_bytes, display_name="page_scan")
            prompt = f"""
You are a document comprehension assistant.

The user is searching for information about: "{user_query}".

Look at this scanned document page and identify the specific sentences or paragraphs
that directly answer or strongly relate to the user's request.

Return a JSON object with a list of bounding boxes under the key "boxes".
Each bounding box should be in the format [ymin, xmin, ymax, xmax],
where coordinates are normalized from 0 to 1000 (0 is top/left, 1000 is bottom/right).

Be precise: if only one sentence is relevant, highlight just that sentence,
not the entire paragraph.

Example output:
{{
  "boxes": [
    [100, 200, 150, 800],
    [300, 200, 350, 800]
  ]
}}

If the page does not contain relevant content, return:
{{ "boxes": [] }}.
"""
            logging.info("DEBUG_GEMINI_PROMPT: %s", prompt)
            response = model.generate_content([uploaded, prompt])
            logging.info("DEBUG_GEMINI_RESPONSE: %s", response.text)
            data = json.loads(response.text)
            boxes = data.get("boxes", []) if isinstance(data, dict) else []
            if not isinstance(boxes, list):
                return []
            normalized: list[Mapping[str, float]] = []
            for box in boxes:
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
                normalized.append({"x0": xmin, "y0": ymin, "x1": xmax, "y1": ymax})
            return normalized
        except Exception as exc:  # pragma: no cover - external dependency
            logging.warning("Gemini highlight failed: %s", exc)
            return []

    return await asyncio.to_thread(_generate)


async def extract_text_coordinates(
    image_bytes: bytes, query_text: str, client: OpenAIClient | None = None
) -> list[Mapping[str, float]]:
    if client is None or not client.api_key:
        logging.debug("RAW_ANSWER skipping vision: OpenAI client unavailable")
        return await _extract_boxes_with_gemini(image_bytes, query_text)

    fd, temp_name = tempfile.mkstemp(prefix="raw-scan-src-", suffix=".jpg")
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        temp_path.write_bytes(image_bytes)
    except Exception:
        with contextlib.suppress(OSError):
            temp_path.unlink()
        logging.exception("RAW_ANSWER failed to persist temp image for vision")
        return []

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "boxes": {
                "type": "array",
                "minItems": 0,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "x0": {"type": "number"},
                        "y0": {"type": "number"},
                        "x1": {"type": "number"},
                        "y1": {"type": "number"},
                        "confidence": {"type": ["number", "null"]},
                        "reason": {"type": ["string", "null"]},
                    },
                    "required": ["x0", "y0", "x1", "y1"],
                },
            }
        },
        "required": ["boxes"],
        "additionalProperties": False,
    }

    system_prompt = (
        "Ты анализируешь скан страницы книги. Найди 1–4 наиболее релевантных фрагмента текста, "
        "которые отвечают на запрос пользователя. Верни координаты прямоугольников в долях от ширины/высоты изображения."
    )
    query_line = query_text.strip() or ""
    user_prompt = (
        "Запрос пользователя: "
        + (query_line or "(пусто)")
        + "\n"
        "Верни массив boxes. Каждая запись: x0,y0,x1,y1 в диапазоне 0..1, где (x0,y0) — левый верх. "
        "x0<x1 и y0<y1. Добавь confidence 0..1, если уверенность известна."
    )

    try:
        response = await client.classify_image(
            model="gpt-4o-mini",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=temp_path,
            schema=schema,
            schema_name="raw_scan_boxes",
        )
    except Exception:
        logging.exception("RAW_ANSWER vision request failed")
        with contextlib.suppress(OSError):
            temp_path.unlink()
        return []
    finally:
        with contextlib.suppress(OSError):
            temp_path.unlink()

    if not response or not isinstance(response.content, Mapping):
        return []

    boxes_raw = response.content.get("boxes")
    parsed: list[Mapping[str, float]] = []
    if isinstance(boxes_raw, Sequence):
        for item in boxes_raw:
            if not isinstance(item, Mapping):
                continue
            try:
                x0 = float(item.get("x0"))
                y0 = float(item.get("y0"))
                x1 = float(item.get("x1"))
                y1 = float(item.get("y1"))
            except (TypeError, ValueError):
                continue
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                continue
            parsed.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1})
    return parsed


def draw_highlight_overlay(
    image_bytes: bytes, boxes: Sequence[Mapping[str, float]]
) -> bytes | None:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            img = image.convert("RGBA")
    except Exception:
        logging.exception("RAW_ANSWER failed to open image for highlighting")
        return None

    overlay = Image.new("RGBA", img.size, (255, 0, 0, 0))
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
        draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 0, 255), width=max(3, width // 200))
        draw.rectangle((x0, y0, x1, y1), fill=(255, 0, 0, 60))

    composed = Image.alpha_composite(img, overlay).convert("RGB")
    buffer = io.BytesIO()
    composed.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.read()
