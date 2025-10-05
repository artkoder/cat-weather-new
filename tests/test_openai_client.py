import base64
import json
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from openai_client import OpenAIClient


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


@pytest.mark.asyncio
async def test_classify_image_uses_text_response_payload(monkeypatch):
    captured: dict[str, Any] = {}
    schema = {"name": "vision", "schema": {"type": "object"}}
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
        image_bytes=b"fake-bytes",
        schema=schema,
    )

    assert captured["url"].endswith("/responses")
    payload = captured["payload"]
    assert payload["model"] == "gpt-vision"
    assert payload["modalities"] == ["text"]
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"] is schema
    assert payload["response_format"]["strict"] is True
    system_content = payload["input"][0]["content"][0]
    assert system_content == {
        "type": "input_text",
        "text": "classify image",
    }
    user_text = payload["input"][1]["content"][0]
    assert user_text == {"type": "input_text", "text": "What do you see?"}
    image_part = payload["input"][1]["content"][1]
    assert image_part["type"] == "input_image"
    assert base64.b64decode(image_part["image_base64"]) == b"fake-bytes"

    assert result is not None
    assert result.content == expected_result
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.total_tokens == 15
    assert result.request_id == "resp_vision"


@pytest.mark.asyncio
async def test_generate_json_uses_text_response_payload(monkeypatch):
    captured: dict[str, Any] = {}
    schema = {"name": "json", "schema": {"type": "object"}}
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
    assert payload["modalities"] == ["text"]
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"] is schema
    assert payload["response_format"]["strict"] is True
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
