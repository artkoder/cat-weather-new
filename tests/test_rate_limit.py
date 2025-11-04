import sys
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api.rate_limit import TokenBucketLimiter, create_rate_limit_middleware


@pytest.mark.asyncio
async def test_token_bucket_limiter_blocks_after_capacity():
    limiter = TokenBucketLimiter(2, 60)
    first = await limiter.allow("key")
    assert first.allowed is True
    assert first.retry_after_seconds is None

    second = await limiter.allow("key")
    assert second.allowed is True
    assert second.retry_after_seconds is None

    third = await limiter.allow("key")
    assert third.allowed is False
    assert third.retry_after_seconds is not None
    assert 29 <= third.retry_after_seconds <= 31


@pytest.mark.asyncio
async def test_rate_limit_middleware_returns_429(monkeypatch):
    monkeypatch.setenv("RL_HEALTH_PER_MIN", "1")
    monkeypatch.setenv("RL_HEALTH_WINDOW_SEC", "60")
    app = web.Application(middlewares=[create_rate_limit_middleware()])

    async def handler(request: web.Request) -> web.Response:
        return web.Response(text="ok")

    app.router.add_get("/v1/health", handler)

    async with TestServer(app) as server:
        async with TestClient(server) as client:
            first = await client.get("/v1/health")
            assert first.status == 200
            second = await client.get("/v1/health")
            assert second.status == 429
            payload = await second.json()
            assert payload == {"error": "rate_limited"}
            assert "Retry-After" in second.headers
            retry_after = int(second.headers["Retry-After"])
            assert retry_after >= 59
