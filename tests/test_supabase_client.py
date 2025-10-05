from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import httpx
import pytest

from supabase_client import SupabaseClient


@pytest.mark.asyncio
async def test_insert_token_usage_success(monkeypatch, caplog):
    client = SupabaseClient(url="https://example.supabase.co", key="test-key")
    response = type("Resp", (), {"status_code": 201, "text": "Created"})()
    post_mock = AsyncMock(return_value=response)
    monkeypatch.setattr(client._client, "post", post_mock)
    caplog.set_level(logging.INFO)

    meta = {"source": "test", "time": datetime(2024, 1, 1, tzinfo=timezone.utc)}
    assert await client.insert_token_usage(
        model="gpt-4o",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        request_id="req-1",
        meta=meta,
    )
    await client.aclose()

    assert post_mock.await_count == 1
    payload = post_mock.await_args.kwargs["json"]
    assert payload["model"] == "gpt-4o"
    assert payload["meta"]["source"] == "test"
    assert payload["meta"]["time"].startswith("2024-01-01")
    record = next(record for record in caplog.records if "Supabase token usage insert succeeded" in record.message)
    assert record.log_token_usage["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_insert_token_usage_http_error(monkeypatch, caplog):
    client = SupabaseClient(url="https://example.supabase.co", key="test-key")
    post_mock = AsyncMock(side_effect=httpx.HTTPError("boom"))
    monkeypatch.setattr(client._client, "post", post_mock)
    caplog.set_level(logging.ERROR)

    result = await client.insert_token_usage(
        model="gpt-4o",
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        request_id=None,
        meta=None,
    )
    await client.aclose()

    assert result is False
    record = next(record for record in caplog.records if "Supabase token usage insert error" in record.message)
    assert record.log_token_usage["model"] == "gpt-4o"
