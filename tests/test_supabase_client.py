from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import httpx
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from supabase_client import SupabaseClient


@pytest.mark.asyncio
async def test_insert_token_usage_success(monkeypatch):
    client = SupabaseClient(url="https://example.supabase.co", key="test-key")
    response = type("Resp", (), {"status_code": 201, "text": "Created"})()
    post_mock = AsyncMock(return_value=response)
    monkeypatch.setattr(client._client, "post", post_mock)

    meta = {"source": "test", "time": datetime(2024, 1, 1, tzinfo=timezone.utc)}
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
