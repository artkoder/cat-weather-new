import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import health_handler


class DummyJobs:
    def __init__(self, metrics: dict[str, int]):
        self._metrics = metrics

    def metrics(self) -> dict[str, int]:
        return dict(self._metrics)


class DummyBot:
    def __init__(self, db, jobs, *, dry_run: bool, telegram_response):
        self.db = db
        self.jobs = jobs
        self.dry_run = dry_run
        self._telegram_response = telegram_response
        self.calls = []

    async def api_request(self, method: str, data=None):
        self.calls.append((method, data))
        if isinstance(self._telegram_response, Exception):
            raise self._telegram_response
        return self._telegram_response


async def _call_health(bot: DummyBot):
    app = web.Application()
    app["bot"] = bot
    app["started_at"] = datetime.now(timezone.utc) - timedelta(seconds=5)
    app["version"] = "test-version"
    app.router.add_get("/v1/health", health_handler)

    async with TestServer(app) as server:
        async with TestClient(server) as client:
            response = await client.get("/v1/health")
            payload = await response.json()
            return response.status, payload


@pytest.mark.asyncio
async def test_health_dry_run_skips_telegram():
    conn = sqlite3.connect(":memory:")
    bot = DummyBot(
        conn,
        DummyJobs({"pending": 1, "active": 0, "failed": 0}),
        dry_run=True,
        telegram_response=RuntimeError("should not call"),
    )

    status, payload = await _call_health(bot)

    assert status == 207
    assert payload["ok"] is True
    assert payload["version"] == "test-version"
    assert payload["checks"]["db"]["ok"] is True
    assert payload["checks"]["queue"]["pending"] == 1
    assert payload["checks"]["telegram"]["skipped"] is True
    assert "telegram check skipped" in payload["warnings"][0]
    assert not bot.calls
    conn.close()


@pytest.mark.asyncio
async def test_health_successful_probe():
    conn = sqlite3.connect(":memory:")
    bot = DummyBot(
        conn,
        DummyJobs({"pending": 0, "active": 0, "failed": 0}),
        dry_run=False,
        telegram_response={"ok": True},
    )

    status, payload = await _call_health(bot)

    assert status == 200
    assert payload["ok"] is True
    assert payload["warnings"] == []
    assert bot.calls == [("getMe", None)]
    conn.close()


@pytest.mark.asyncio
async def test_health_telegram_failure():
    conn = sqlite3.connect(":memory:")
    bot = DummyBot(
        conn,
        DummyJobs({"pending": 0, "active": 0, "failed": 0}),
        dry_run=False,
        telegram_response={"ok": False, "description": "bad token"},
    )

    status, payload = await _call_health(bot)

    assert status == 503
    assert payload["ok"] is False
    assert payload["checks"]["telegram"]["ok"] is False
    assert payload["checks"]["telegram"]["error"] == "bad token"
    conn.close()
