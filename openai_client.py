from __future__ import annotations

import asyncio
import base64
import imghdr
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
        schema_name: str = "asset_vision_v1",
    ) -> OpenAIResponse | None:
        if not self.api_key:
            return None
        payload = {
            "model": model,
            "text": {
                "format": {
                    "type": "json_schema",
                    **self.ensure_json_format(
                        name=schema_name,
                        schema=schema,
                    ),
                }
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
                        self._build_image_part(image_bytes),
                        {"type": "input_text", "text": user_prompt},
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
        schema_name: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> OpenAIResponse | None:
        if not self.api_key:
            return None
        payload = {
            "model": model,
            "text": {
                "format": {
                    "type": "json_schema",
                    **self.ensure_json_format(
                        name=schema_name or "post_text_v1",
                        schema=schema,
                    ),
                }
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

    def ensure_json_format(
        self, *, name: str, schema: dict[str, Any] | None, strict: bool = True
    ) -> dict[str, Any]:
        if not name or not str(name).strip():
            raise ValueError("Structured output schema name must be provided")
        if not isinstance(schema, dict) or not schema:
            raise ValueError("Structured output schema must be a non-empty dict")
        return {
            "name": name,
            "schema": schema,
            "strict": strict,
        }

    def _build_image_part(self, image_bytes: bytes) -> dict[str, Any]:
        image_kind = imghdr.what(None, image_bytes)
        if image_kind:
            mime_type = f"image/{image_kind}"
        else:
            mime_type = "image/jpeg"
        base64_data = base64.b64encode(image_bytes).decode("ascii")
        return {
            "type": "input_image",
            "image_url": f"data:{mime_type};base64,{base64_data}",
        }

    def _truncate_for_log(self, value: str, limit: int = 600) -> str:
        if len(value) <= limit:
            return value
        return value[: limit - 3] + "..."

    async def _submit_request(self, payload: Dict[str, Any]) -> OpenAIResponse:
        url = f"{self.base_url}/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        json_schema_section: dict[str, Any] | None = None
        if isinstance(payload, dict):
            text_section = payload.get("text", {})
            if isinstance(text_section, dict):
                format_section = text_section.get("format", {})
                if isinstance(format_section, dict):
                    format_type = format_section.get("type")
                    if format_type == "json_schema":
                        json_schema_section = format_section
                        legacy_container = format_section.get("json_schema")
                        if isinstance(legacy_container, dict):
                            json_schema_section = legacy_container
                    elif json_schema_section is None:
                        legacy_container = format_section.get("json_schema")
                        if isinstance(legacy_container, dict):
                            json_schema_section = legacy_container
        if json_schema_section is None:
            json_schema_section = {}
        schema_name = json_schema_section.get("name")
        if not schema_name or not str(schema_name).strip():
            raise ValueError(
                "OpenAI payload must include text.format.name (json_schema)"
            )
        schema_body = json_schema_section.get("schema")
        schema_keys: list[str] | None = None
        schema_key_count: int | None = None
        if isinstance(schema_body, dict):
            schema_key_count = len(schema_body)
            schema_keys = sorted(schema_body.keys())[:5]
        strict_flag = json_schema_section.get("strict")
        logging.debug(
            "OpenAI payload schema summary: name=%s strict=%s key_count=%s sample_keys=%s",
            schema_name,
            strict_flag,
            schema_key_count,

            schema_keys,
        )

        max_attempts = 3
        delay = 1.0
        for attempt in range(1, max_attempts + 1):
            started = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    response = await client.post(url, json=payload, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= max_attempts:
                    logging.error("OpenAI API request failed: %s", exc)
                    raise
                logging.warning(
                    "OpenAI API request failed (attempt %s/%s): %s",
                    attempt,
                    max_attempts,
                    exc,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 10)
                continue

            duration = time.perf_counter() - started
            if response.status_code == 200:
                data = response.json()
                break

            error_body: dict[str, Any] | None = None
            try:
                parsed = response.json()
                if isinstance(parsed, dict):
                    error_body = parsed
            except Exception:  # pragma: no cover - response.json may raise
                error_body = None

            should_retry = response.status_code in {408, 429} or response.status_code >= 500
            error_type = None
            if error_body:
                error_info = error_body.get("error")
                if isinstance(error_info, dict):
                    error_type = error_info.get("type")
                elif isinstance(error_body.get("type"), str):
                    error_type = error_body["type"]  # type: ignore[index]
            if error_type == "invalid_json_schema":
                schema_serialized: str
                try:
                    if schema_body is None:
                        schema_serialized = "<missing>"
                    else:
                        schema_serialized = json.dumps(schema_body)
                except TypeError:
                    schema_serialized = str(schema_body)
                max_length = 600
                truncated = schema_serialized
                truncated_flag = False
                if len(truncated) > max_length:
                    truncated = truncated[: max_length - 3] + "..."
                    truncated_flag = True
                logging.error(
                    "OpenAI invalid_json_schema error for schema %s (len=%s truncated=%s): %s",
                    schema_name,
                    len(schema_serialized),
                    truncated_flag,
                    truncated,
                )
            if 400 <= response.status_code < 500 and error_type == "invalid_request_error":
                should_retry = False

            request_id = response.headers.get("x-request-id")
            response_text_for_log = self._truncate_for_log(response.text)
            log_extra = {"status_code": response.status_code, "request_id": request_id}

            if should_retry and attempt < max_attempts:
                logging.warning(
                    "OpenAI API error status=%s request_id=%s (attempt %s/%s): %s",
                    response.status_code,
                    request_id,
                    attempt,
                    max_attempts,
                    response_text_for_log,
                    extra=log_extra,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 10)
                continue

            logging.error(
                "OpenAI API error status=%s request_id=%s: %s",
                response.status_code,
                request_id,
                response_text_for_log,
                extra=log_extra,
            )
            raise RuntimeError(f"OpenAI API error {response.status_code}")
        else:  # pragma: no cover - safeguard
            raise RuntimeError("OpenAI API request failed after retries")

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
