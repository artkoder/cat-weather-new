from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from supabase_client import SupabaseClient


class _DummyResponse:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> Any:
        return self._payload


@pytest.mark.asyncio
async def test_insert_token_usage_success(monkeypatch):
    client = SupabaseClient(url="https://example.supabase.co", key="test-key")
    response = type("Resp", (), {"status_code": 201, "text": "Created"})()
    post_mock = AsyncMock(return_value=response)
    monkeypatch.setattr(client._client, "post", post_mock)

    meta = {"source": "test", "time": datetime(2024, 1, 1, tzinfo=UTC)}
    success, payload, error = await client.insert_token_usage(
        model="gpt-4o",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        request_id="req-1",
        meta=meta,
    )
    await client.aclose()

    assert success is True
    assert error is None
    assert post_mock.await_count == 1
    assert payload == post_mock.await_args.kwargs["json"]
    assert payload["meta"]["source"] == "test"
    assert payload["meta"]["time"].startswith("2024-01-01")
    assert payload["endpoint"] == "/v1/responses"
    assert payload["bot"] == "kotopogoda"


@pytest.mark.asyncio
async def test_insert_token_usage_http_error(monkeypatch):
    client = SupabaseClient(url="https://example.supabase.co", key="test-key")
    post_mock = AsyncMock(side_effect=httpx.HTTPError("boom"))
    monkeypatch.setattr(client._client, "post", post_mock)

    success, payload, error = await client.insert_token_usage(
        model="gpt-4o",
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        request_id=None,
        meta=None,
    )
    await client.aclose()

    assert success is False
    assert "boom" in (error or "")
    assert payload["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_insert_token_usage_meta_strict():
    client = SupabaseClient(url="https://example.supabase.co", key="test-key")

    with pytest.raises(TypeError):
        await client.insert_token_usage(
            model="gpt-4o",
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
            request_id="req",
            meta={"bad": object()},
        )
    await client.aclose()


@pytest.mark.asyncio
async def test_get_24h_usage_total_returns_sum(monkeypatch):
    client = SupabaseClient(url="https://example.supabase.co", key="test-key")
    payload = [{"sum_total": 12345}]
    response = _DummyResponse(200, payload)
    assert client._client is not None
    get_mock = AsyncMock(return_value=response)
    monkeypatch.setattr(client._client, "get", get_mock)

    used, raw, error = await client.get_24h_usage_total()

    assert used == 12345
    assert raw == payload
    assert error is None
    params = get_mock.await_args.kwargs["params"]
    assert params["select"] == "sum_total:sum(total_tokens)"
    assert params["bot"] == "eq.kotopogoda"
    assert params["model"] == "eq.gpt-4o-mini"
    assert params["at"].startswith("gte.")
    await client.aclose()


@pytest.mark.asyncio
async def test_get_24h_usage_total_disabled_returns_none(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    client = SupabaseClient(url=None, key=None)

    used, raw, error = await client.get_24h_usage_total()

    assert used is None
    assert raw is None
    assert error == "disabled"
