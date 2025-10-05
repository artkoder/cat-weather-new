from __future__ import annotations

import logging
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from main import Bot  # noqa: E402
from openai_client import OpenAIResponse  # noqa: E402


@pytest.mark.asyncio
async def test_record_openai_usage_logs_success(tmp_path, caplog, monkeypatch):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    payload = {
        "bot": "kotopogoda",
        "model": "gpt-4o",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "request_id": "req-1",
        "endpoint": "responses",
        "meta": {"job": {"id": 123, "name": "vision", "payload_keys": ["foo"]}},
        "at": "2024-01-01T00:00:00+00:00",
    }
    mock_insert = AsyncMock(return_value=(True, payload, None))
    monkeypatch.setattr(bot.supabase, "insert_token_usage", mock_insert)
    caplog.set_level(logging.INFO)

    job = SimpleNamespace(id=123, name="vision", payload={"foo": "bar"})
    response = OpenAIResponse({}, 10, 5, 15, request_id="req-1", meta={})
    await bot._record_openai_usage("gpt-4o", response, job=job)

    assert mock_insert.await_count == 1
    insert_kwargs = mock_insert.await_args.kwargs
    assert insert_kwargs["meta"]["job"]["payload_keys"] == ["foo"]

    record = next(rec for rec in caplog.records if rec.message == "Supabase token usage insert succeeded")
    assert record.log_token_usage == payload
    await bot.close()


@pytest.mark.asyncio
async def test_record_openai_usage_logs_failure(tmp_path, caplog, monkeypatch):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    payload = {"bot": "kotopogoda", "model": "gpt-4o"}
    mock_insert = AsyncMock(return_value=(False, payload, "HTTP 500: error"))
    monkeypatch.setattr(bot.supabase, "insert_token_usage", mock_insert)
    caplog.set_level(logging.ERROR)

    response = OpenAIResponse({}, 1, 2, 3, request_id="req-2", meta=None)
    await bot._record_openai_usage("gpt-4o", response)

    record = next(rec for rec in caplog.records if "Supabase token usage insert failed" in rec.message)
    assert record.log_token_usage == payload
    await bot.close()
