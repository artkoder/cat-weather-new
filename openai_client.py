from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
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
            logging.warning("OpenAI API key not configured; vision tasks will be skipped")

    def _load_api_key_from_env(self) -> str | None:
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key and env_key.strip():
            return env_key.strip()
        file_path = os.getenv("OPENAI_API_KEY_FILE")
        if file_path:
            try:
                contents = Path(file_path).expanduser().read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                logging.error("OPENAI_API_KEY_FILE %s not found", file_path)
                return None
            except OSError as exc:
                logging.error("Failed reading OPENAI_API_KEY_FILE %s: %s", file_path, exc)
                return None
            if contents:
                return contents
        return None

    def refresh_api_key(self) -> str | None:
        new_key = self._load_api_key_from_env()
        if new_key and new_key != self.api_key:
            logging.info("OpenAI API key refreshed from environment")
        self.api_key = new_key or self.api_key
        if not self.api_key:
            logging.warning("OpenAI API key not configured; vision tasks will be skipped")
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
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": schema,
            },
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        return await self._submit_request(payload)

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
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
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
