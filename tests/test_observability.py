import io
import json
import logging
import os
import sys

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from api.rate_limit import create_rate_limit_middleware
from observability import metrics_handler, observability_middleware, setup_logging


@pytest.mark.asyncio
async def test_request_id_middleware_propagates():
    app = web.Application(middlewares=[observability_middleware])

    async def handler(request: web.Request) -> web.Response:
        assert request["request_id"] == "req-123"
        return web.json_response({"ok": True})

    app.router.add_get("/", handler)

    async with TestServer(app) as server:
        async with TestClient(server) as client:
            response = await client.get("/", headers={"X-Request-ID": "req-123"})
            assert response.status == 200
            assert response.headers.get("X-Request-ID") == "req-123"


def test_logging_redacts_sensitive_values(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setenv("LOG_FORMAT", "json")
    setup_logging(stream=stream)

    logger = logging.getLogger("test-redaction")
    logger.info("Authorization=secret-token X-Signature=abcdef")

    output = stream.getvalue().strip().splitlines()[-1]
    payload = json.loads(output)
    assert "***" in payload["msg"]
    assert "abcdef" not in payload["msg"]


@pytest.mark.asyncio
async def test_metrics_endpoint_forbidden_without_allowlist(monkeypatch):
    monkeypatch.delenv("ALLOWLIST_CIDR", raising=False)
    app = web.Application(middlewares=[create_rate_limit_middleware()])
    app.router.add_get("/metrics", metrics_handler)

    async with TestServer(app) as server:
        async with TestClient(server) as client:
            response = await client.get("/metrics")
            assert response.status == 403


@pytest.mark.asyncio
async def test_metrics_endpoint_allowed_for_allowlisted_ip(monkeypatch):
    monkeypatch.setenv("ALLOWLIST_CIDR", "127.0.0.1/32")
    monkeypatch.setenv("RL_METRICS_PER_MIN", "10")
    app = web.Application(middlewares=[create_rate_limit_middleware()])
    app.router.add_get("/metrics", metrics_handler)

    async with TestServer(app) as server:
        async with TestClient(server) as client:
            response = await client.get(
                "/metrics", headers={"X-Forwarded-For": "127.0.0.1"}
            )
            text = await response.text()
            assert response.status == 200
            assert response.headers["Content-Type"].startswith("text/plain")
            assert "http_requests_total" in text
