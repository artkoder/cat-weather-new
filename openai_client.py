from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict

import httpx


@dataclass
class OpenAIResponse:
    content: Dict[str, Any]
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    request_id: str | None = None
    meta: Dict[str, Any] | None = None


class OpenAIClient:
    """Minimal OpenAI wrapper that tracks token usage."""

    def __init__(self, api_key: str | None, *, base_url: str | None = None) -> None:
        self.api_key = api_key.strip() if api_key else None
        self.base_url = base_url or "https://api.openai.com/v1"
        if not self.api_key:
            self.api_key = self._load_api_key_from_env()
        if not self.api_key:
            logging.warning("4O API key not configured; vision tasks will be skipped")

    def _load_api_key_from_env(self) -> str | None:
        env_key = os.getenv("4O_API_KEY")
        if env_key and env_key.strip():
            return env_key.strip()
        return None

    def refresh_api_key(self) -> str | None:
        new_key = self._load_api_key_from_env()
        if new_key and new_key != self.api_key:
            logging.info("4O API key refreshed from environment")
        self.api_key = new_key or self.api_key
        if not self.api_key:
            logging.warning("4O API key not configured; vision tasks will be skipped")
        return self.api_key

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
        payload = {
            "model": model,
            "text": {
                "format": self._build_text_format(schema),
            },
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": system_prompt,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                        self._build_image_part(image_bytes),
                    ],
                },
            ],
        }
        return await self._submit_request(payload)

    async def generate_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> OpenAIResponse | None:
        if not self.api_key:
            return None
        payload = {
            "model": model,
            "text": {
                "format": self._build_text_format(schema),
            },
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        return await self._submit_request(payload)

    def _build_text_format(self, schema: dict[str, Any]) -> dict[str, Any]:
        if "name" in schema and "schema" in schema:
            schema_name = schema["name"]
            schema_body = schema["schema"]
        else:
            schema_name = "response"
            schema_body = schema
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema_body,
            },
            "strict": True,
        }

    def _build_image_part(self, image_bytes: bytes) -> dict[str, Any]:
        mime_type = self._detect_mime_type(image_bytes)
        base64_data = base64.b64encode(image_bytes).decode("ascii")
        return {
            "type": "input_image",
            "image_url": f"data:{mime_type};base64,{base64_data}",
        }

    def _detect_mime_type(self, image_bytes: bytes) -> str:
        header = image_bytes[:12]
        if header.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if header[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        if header.startswith(b"BM"):
            return "image/bmp"
        if len(image_bytes) >= 12 and header[0:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
            return "image/webp"
        if header.startswith(b"II*\x00") or header.startswith(b"MM\x00*"):
            return "image/tiff"
        if len(image_bytes) >= 12 and header[4:8] == b"ftyp":
            brand = image_bytes[8:12]
            if brand in {b"heic", b"heix", b"hevc", b"hevx"}:
                return "image/heic"
            if brand in {b"mif1", b"msf1"}:
                return "image/heif"
        return "image/jpeg"

    async def _submit_request(self, payload: Dict[str, Any]) -> OpenAIResponse:
        url = f"{self.base_url}/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(url, json=payload, headers=headers)
        duration = time.perf_counter() - started
        if response.status_code != 200:
            logging.error("OpenAI API error %s: %s", response.status_code, response.text)
            raise RuntimeError(f"OpenAI API error {response.status_code}")
        data = response.json()
        usage = data.get("usage") or {}
        prompt_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
        completion_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        if total_tokens is None:
            tokens_sum = 0
            has_value = False
            if isinstance(prompt_tokens, int):
                tokens_sum += prompt_tokens
                has_value = True
            if isinstance(completion_tokens, int):
                tokens_sum += completion_tokens
                has_value = True
            if has_value:
                total_tokens = tokens_sum
        request_id = data.get("id") or response.headers.get("x-request-id")
        content = data.get("output") or data.get("response") or {}
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
        meta: Dict[str, Any] | None = {
            "model": payload.get("model"),
            "duration_ms": round(duration * 1000, 2),
            "status_code": response.status_code,
        }
        return OpenAIResponse(
            parsed,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            request_id,
            meta,
        )
