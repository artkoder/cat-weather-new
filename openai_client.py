from __future__ import annotations

import asyncio
import base64
import gc
import json
import logging
import mimetypes
import os
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

from PIL import Image, UnidentifiedImageError



def _ensure_list_with_null(type_value: Any) -> Any:
    if isinstance(type_value, str):
        if type_value == "null":
            return ["null"]
        return [type_value, "null"]
    if isinstance(type_value, list):
        if "null" in type_value:
            return type_value
        return [*type_value, "null"]
    return type_value


def strictify_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Apply strict defaults to schema in-place and return it."""

    if not isinstance(schema, dict):
        raise ValueError("Schema must be a dictionary")

    def process_list(items: list[Any]) -> None:
        for item in items:
            if isinstance(item, dict):
                process_node(item, is_root=False)
            elif isinstance(item, list):
                process_list(item)

    def process_node(node: dict[str, Any], *, is_root: bool) -> None:
        properties = node.get("properties")
        if isinstance(properties, dict):
            for prop_schema in properties.values():
                if isinstance(prop_schema, dict):
                    process_node(prop_schema, is_root=False)
                elif isinstance(prop_schema, list):
                    process_list(prop_schema)

        items_value = node.get("items")
        if isinstance(items_value, dict):
            process_node(items_value, is_root=False)
        elif isinstance(items_value, list):
            process_list(items_value)

        for key, value in list(node.items()):
            if key in {"properties", "items"}:
                continue
            if isinstance(value, dict):
                process_node(value, is_root=False)
            elif isinstance(value, list):
                process_list(value)

        if is_root:
            node["type"] = "object"
        elif "type" in node:
            node["type"] = _ensure_list_with_null(node["type"])
        elif properties:
            node["type"] = _ensure_list_with_null("object")

        if isinstance(properties, dict):
            required_existing = node.get("required")
            required: list[str] = []
            if isinstance(required_existing, list):
                for item in required_existing:
                    if isinstance(item, str) and item not in required:
                        required.append(item)
            for prop_key in properties.keys():
                if prop_key not in required:
                    required.append(prop_key)
            node["required"] = required

        type_value = node.get("type")
        is_object = False
        if isinstance(type_value, str):
            is_object = type_value == "object"
        elif isinstance(type_value, list):
            is_object = "object" in type_value
        elif type_value is None and isinstance(properties, dict):
            is_object = True
        if is_object and "additionalProperties" not in node:
            node["additionalProperties"] = False

    process_node(schema, is_root=True)
    return schema

import httpx


@dataclass
class OpenAIResponse:
    content: Dict[str, Any]
    usage: Dict[str, Any]
    meta: Dict[str, Any] | None = None

    def _usage_int(self, key: str) -> int | None:
        value = self.usage.get(key)
        return value if isinstance(value, int) else None

    @property
    def prompt_tokens(self) -> int | None:
        return self._usage_int("prompt_tokens")

    @property
    def completion_tokens(self) -> int | None:
        return self._usage_int("completion_tokens")

    @property
    def total_tokens(self) -> int | None:
        return self._usage_int("total_tokens")

    @property
    def request_id(self) -> str | None:
        request_id = self.usage.get("request_id")
        if isinstance(request_id, str) and request_id:
            return request_id
        response_id = self.usage.get("response_id")
        if isinstance(response_id, str) and response_id:
            return response_id
        return None

    @property
    def endpoint(self) -> str | None:
        endpoint = self.usage.get("endpoint")
        if isinstance(endpoint, str) and endpoint:
            return endpoint
        return None


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
        image_path: str | os.PathLike[str] | Path,
        schema: dict[str, Any],
        schema_name: str = "asset_vision_v1",
    ) -> OpenAIResponse | None:
        if not self.api_key:
            return None
        strict_schema = strictify_schema(schema)
        payload: dict[str, Any] | None = None
        image_data_url: str | None = None
        image_part: dict[str, Any] | None = None
        response_obj: OpenAIResponse | None = None
        try:
            source_path = Path(image_path)
            image_bytes = source_path.read_bytes()
            mime_type = self._infer_image_mime_type(source_path, image_bytes)
            encoded_image = base64.b64encode(image_bytes).decode("ascii")
            del image_bytes
            image_data_url = f"data:{mime_type};base64,{encoded_image}"
            del encoded_image
            image_part = self._build_image_part(image_data_url)
            payload = {
                "model": model,
                "text": {
                    "format": {
                        "type": "json_schema",
                        **self.ensure_json_format(
                            name=schema_name,
                            schema=strict_schema,
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
                            image_part,
                            {"type": "input_text", "text": user_prompt},
                        ],
                    },
                ],
            }
            response_obj = await self._submit_request(payload)
            return response_obj
        finally:
            payload = None
            image_part = None
            image_data_url = None
            response_obj = None
            gc.collect()

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
        strict_schema = strictify_schema(schema)
        payload = {
            "model": model,
            "text": {
                "format": {
                    "type": "json_schema",
                    **self.ensure_json_format(
                        name=schema_name or "post_text_v1",
                        schema=strict_schema,
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

    def _infer_image_mime_type(self, image_path: Path, data: bytes) -> str:
        mime_type, _ = mimetypes.guess_type(image_path.name)
        if not mime_type or not mime_type.startswith("image/"):
            detected_format: str | None = None
            try:
                with Image.open(BytesIO(data)) as image:
                    detected_format = image.format
            except (UnidentifiedImageError, OSError):
                detected_format = None
            if detected_format:
                format_key = detected_format.lower()
                if format_key:
                    pil_mime_map = {
                        "jpeg": "image/jpeg",
                        "jpg": "image/jpeg",
                        "png": "image/png",
                        "webp": "image/webp",
                        "gif": "image/gif",
                        "tiff": "image/tiff",
                        "bmp": "image/bmp",
                        "heif": "image/heif",
                        "heic": "image/heic",
                    }
                    mime_type = pil_mime_map.get(format_key) or f"image/{format_key}"
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/jpeg"
        if mime_type.endswith("/jpg"):
            mime_type = "image/jpeg"
        return mime_type

    def _build_image_part(self, image_data_url: str) -> dict[str, Any]:
        return {
            "type": "input_image",
            "image_url": image_data_url,
        }

    def _truncate_for_log(self, value: str, limit: int = 600) -> str:
        if len(value) <= limit:
            return value
        return value[: limit - 3] + "..."

    async def _submit_request(self, payload: Dict[str, Any]) -> OpenAIResponse:
        endpoint_path = "/v1/responses"
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

            request_id_header = response.headers.get("x-request-id")
            response_text_for_log = self._truncate_for_log(response.text)
            log_extra = {
                "status_code": response.status_code,
                "request_id": request_id_header,
            }

            if should_retry and attempt < max_attempts:
                logging.warning(
                    "OpenAI API error status=%s request_id=%s (attempt %s/%s): %s",
                    response.status_code,
                    request_id_header,
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
                request_id_header,
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
        request_id_header = response.headers.get("x-request-id")
        response_id = data.get("id") if isinstance(data, dict) else None
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
        usage_payload: Dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "endpoint": endpoint_path,
            "request_id": request_id_header,
        }
        if response_id:
            usage_payload["response_id"] = response_id
        if not usage_payload.get("request_id") and response_id:
            usage_payload["request_id"] = response_id
        meta: Dict[str, Any] | None = {
            "model": payload.get("model"),
            "duration_ms": round(duration * 1000, 2),
            "status_code": response.status_code,
        }
        if response_id:
            meta["response_id"] = response_id
        return OpenAIResponse(
            parsed,
            usage_payload,
            meta,
        )
