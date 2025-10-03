from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict

import httpx


@dataclass
class OpenAIResponse:
    content: Dict[str, Any]
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


class OpenAIClient:
    """Minimal OpenAI wrapper that tracks token usage."""

    def __init__(self, api_key: str | None, *, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
        if not self.api_key:
            logging.warning("OpenAI API key not configured; vision tasks will be skipped")

    async def classify_image(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        schema: dict[str, Any],
    ) -> OpenAIResponse | None:
        if not self.api_key:
            return None
        url = f"{self.base_url}/responses"
        payload = {
            "model": model,
            "input": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "input_image",
                            "image_base64": base64.b64encode(image_bytes).decode("ascii"),
                        },
                    ],
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": schema,
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            logging.error("OpenAI API error %s: %s", response.status_code, response.text)
            raise RuntimeError(f"OpenAI API error {response.status_code}")
        data = response.json()
        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        content = data.get("output") or data.get("response") or {}
        # API returns list under "output"; normalise
        if isinstance(content, list) and content:
            content_item = content[0]
        else:
            content_item = content
        if isinstance(content_item, dict) and "content" in content_item:
            segments = content_item["content"]
            if isinstance(segments, list) and segments:
                message_text = segments[0].get("text")
            else:
                message_text = content_item.get("content")
        elif isinstance(content_item, str):
            message_text = content_item
        else:
            message_text = None
        parsed: Dict[str, Any]
        if message_text:
            try:
                parsed = json.loads(message_text)
            except json.JSONDecodeError:
                parsed = {"raw": message_text}
        else:
            parsed = {}
        return OpenAIResponse(parsed, prompt_tokens, completion_tokens, total_tokens)
