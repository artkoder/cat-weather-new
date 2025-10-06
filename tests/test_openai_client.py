import base64
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from openai_client import OpenAIClient


PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)
PNG_BYTES = base64.b64decode(PNG_BASE64)

JPEG_BASE64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAAaACgDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAb/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAH/AP/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAQUCf//EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQMBAT8Bf//EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQIBAT8Bf//EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEABj8Cf//Z"
)
JPEG_BYTES = base64.b64decode(JPEG_BASE64)


class DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.status_code = 200
        self._payload = payload
        self.headers: dict[str, str] = {"x-request-id": "req_123"}

    def json(self) -> dict[str, Any]:
        return self._payload

    @property
    def text(self) -> str:
        return json.dumps(self._payload)


class DummyAsyncClient:
    def __init__(
        self,
        callback: Callable[[str, dict[str, Any], dict[str, str]], None],
        response_payload: dict[str, Any],
    ) -> None:
        self._callback = callback
        self._response_payload = response_payload

    async def __aenter__(self) -> "DummyAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        return None

    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
    ) -> DummyResponse:
        self._callback(url, json, headers)
        return DummyResponse(self._response_payload)


class ErrorResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 400) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers: dict[str, str] = {}

    def json(self) -> dict[str, Any]:
        return self._payload

    @property
    def text(self) -> str:
        return json.dumps(self._payload)


class ErrorAsyncClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    async def __aenter__(self) -> "ErrorAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        return None

    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
    ) -> ErrorResponse:
        return ErrorResponse(self._payload)


@pytest.mark.asyncio
async def test_classify_image_uses_text_response_payload(monkeypatch):
    captured: dict[str, Any] = {}
    schema = {"type": "object"}
    expected_result = {"label": "cat", "confidence": 0.9}
    response_payload = {
        "output": [
            {
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(expected_result),
                    }
                ]
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
        "id": "resp_vision",
    }

    def _capture(url: str, payload: dict[str, Any], headers: dict[str, str]) -> None:
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers

    monkeypatch.setattr(
        "httpx.AsyncClient",
        lambda timeout=120: DummyAsyncClient(_capture, response_payload),
    )

    client = OpenAIClient("test-key")
    result = await client.classify_image(
        model="gpt-vision",
        system_prompt="classify image",
        user_prompt="What do you see?",
        image_bytes=PNG_BYTES,
        schema=schema,
    )

    assert captured["url"].endswith("/responses")
    payload = captured["payload"]
    assert payload["model"] == "gpt-vision"
    text_config = payload["text"]
    format_config = text_config["format"]
    assert format_config["type"] == "json_schema"
    json_schema_config = format_config["json_schema"]
    assert json_schema_config["name"] == "asset_vision_v1"
    assert json_schema_config["schema"] is schema
    assert json_schema_config["strict"] is True
    system_content = payload["input"][0]["content"][0]
    assert system_content == {
        "type": "input_text",
        "text": "classify image",
    }
    image_part = payload["input"][1]["content"][0]
    assert image_part["type"] == "input_image"
    image_url = image_part["image_url"]
    assert image_url.startswith("data:image/png;base64,")
    encoded = image_url.split(",", 1)[1]
    assert base64.b64decode(encoded) == PNG_BYTES
    user_text = payload["input"][1]["content"][1]
    assert user_text == {"type": "input_text", "text": "What do you see?"}

    assert result is not None
    assert result.content == expected_result
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.total_tokens == 15
    assert result.request_id == "resp_vision"


def test_build_image_part_png_data_uri():
    client = OpenAIClient("test-key")
    part = client._build_image_part(PNG_BYTES)
    assert part["type"] == "input_image"
    assert part["image_url"].startswith("data:image/png;base64,")
    encoded = part["image_url"].split(",", 1)[1]
    assert base64.b64decode(encoded) == PNG_BYTES


def test_build_image_part_jpeg_data_uri():
    client = OpenAIClient("test-key")
    part = client._build_image_part(JPEG_BYTES)
    assert part["type"] == "input_image"
    assert part["image_url"].startswith("data:image/jpeg;base64,")
    encoded = part["image_url"].split(",", 1)[1]
    assert base64.b64decode(encoded) == JPEG_BYTES


@pytest.mark.asyncio
async def test_generate_json_uses_text_response_payload(monkeypatch):
    captured: dict[str, Any] = {}
    schema = {"type": "object"}
    expected_result = {"message": "hello"}
    response_payload = {
        "output": [
            {
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(expected_result),
                    }
                ]
            }
        ],
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
        },
        "id": "resp_json",
    }

    def _capture(url: str, payload: dict[str, Any], headers: dict[str, str]) -> None:
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers

    monkeypatch.setattr(
        "httpx.AsyncClient",
        lambda timeout=120: DummyAsyncClient(_capture, response_payload),
    )

    client = OpenAIClient("test-key")
    result = await client.generate_json(
        model="gpt-json",
        system_prompt="You are a helpful assistant",
        user_prompt="Say hello",
        schema=schema,
        temperature=0.1,
        top_p=0.9,
    )

    payload = captured["payload"]
    assert payload["model"] == "gpt-json"
    text_config = payload["text"]
    format_config = text_config["format"]
    assert format_config["type"] == "json_schema"
    json_schema_config = format_config["json_schema"]
    assert json_schema_config["name"] == "post_text_v1"
    assert json_schema_config["schema"] is schema
    assert json_schema_config["strict"] is True
    system_content = payload["input"][0]["content"][0]
    assert system_content == {
        "type": "input_text",
        "text": "You are a helpful assistant",
    }
    user_content = payload["input"][1]["content"][0]
    assert user_content == {"type": "input_text", "text": "Say hello"}
    assert payload["temperature"] == 0.1
    assert payload["top_p"] == 0.9

    assert result is not None
    assert result.content == expected_result
    assert result.total_tokens == 10
    assert result.request_id == "resp_json"


@pytest.mark.asyncio
async def test_logs_invalid_json_schema_details(monkeypatch, caplog):
    error_payload = {
        "error": {
            "type": "invalid_json_schema",
            "message": "Schema validation failed",
        }
    }

    monkeypatch.setattr(
        "httpx.AsyncClient", lambda timeout=120: ErrorAsyncClient(error_payload)
    )

    client = OpenAIClient("test-key")
    oversized_value = "x" * 1200
    schema = {
        "type": "object",
        "properties": {
            "details": {
                "type": "string",
                "description": oversized_value,
            }
        },
        "required": ["details"],
    }

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError):
            await client.generate_json(
                model="gpt-json",
                system_prompt="Validate schema",
                user_prompt="Return data",
                schema=schema,
            )

    messages = [record.getMessage() for record in caplog.records]
    relevant = [message for message in messages if "invalid_json_schema" in message]
    assert relevant, f"Expected invalid_json_schema log, got {messages!r}"
    log_message = relevant[0]
    assert "post_text_v1" in log_message
    assert "truncated=True" in log_message


@pytest.mark.asyncio
async def test_submit_request_requires_json_schema_name():
    client = OpenAIClient("test-key")
    payload = {
        "model": "gpt-json",
        "text": {
            "format": {
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object"}},
            }
        },
    }

    with pytest.raises(ValueError) as exc_info:
        await client._submit_request(payload)

    assert "text.format.json_schema.name" in str(exc_info.value)
