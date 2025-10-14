from __future__ import annotations

import asyncio
import ipaddress
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Iterable

from aiohttp import web

from observability import record_rate_limit_drop


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logging.warning("RATE invalid integer env %s=%s", name, raw)
        return default


@dataclass
class _Bucket:
    tokens: float
    updated_at: float


@dataclass
class _Allowance:
    allowed: bool
    retry_after_seconds: float | None = None


class TokenBucketLimiter:
    """Simple in-memory token bucket limiter."""

    def __init__(self, limit: int, window_seconds: int) -> None:
        self._capacity = max(0, int(limit))
        self._window = max(1, int(window_seconds))
        self._buckets: dict[str, _Bucket] = {}
        self._lock = asyncio.Lock()
        self._refill_rate = (
            self._capacity / self._window if self._window > 0 else float("inf")
        )

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def window(self) -> int:
        return self._window

    async def allow(self, key: str) -> _Allowance:
        if self._capacity <= 0:
            return _Allowance(False, float(self._window))
        now = time.monotonic()
        async with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=float(self._capacity), updated_at=now)
            else:
                elapsed = now - bucket.updated_at
                if elapsed > 0:
                    bucket.tokens = min(
                        float(self._capacity),
                        bucket.tokens + elapsed * self._refill_rate,
                    )
                    bucket.updated_at = now
            if bucket.tokens < 1.0:
                self._buckets[key] = bucket
                missing_tokens = max(0.0, 1.0 - bucket.tokens)
                if self._refill_rate <= 0:
                    retry_after = float(self._window)
                else:
                    retry_after = missing_tokens / self._refill_rate
                return _Allowance(False, retry_after)
            bucket.tokens -= 1.0
            bucket.updated_at = now
            self._buckets[key] = bucket
            return _Allowance(True, None)


class AllowList:
    def __init__(self, cidr_values: Iterable[str] | None) -> None:
        networks: list[ipaddress._BaseNetwork] = []
        if cidr_values:
            for candidate in cidr_values:
                candidate = candidate.strip()
                if not candidate:
                    continue
                try:
                    networks.append(ipaddress.ip_network(candidate, strict=False))
                except ValueError:
                    logging.warning("ALLOW invalid CIDR %s", candidate)
        self._networks = tuple(networks)

    def allows(self, ip: str | None) -> bool:
        if not self._networks:
            return False
        if not ip:
            return False
        try:
            address = ipaddress.ip_address(ip)
        except ValueError:
            logging.warning("ALLOW invalid IP %s", ip)
            return False
        return any(address in network for network in self._networks)


@dataclass
class RateLimitRule:
    method: str
    path: str
    scope: str
    limiter: TokenBucketLimiter


def _client_ip(request: web.Request) -> str | None:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        first = forwarded.split(",", 1)[0].strip()
        if first:
            return first
    return request.remote


def _route_canonical(request: web.Request) -> str | None:
    route = request.match_info.route
    if route is None:
        return None
    resource = getattr(route, "resource", None)
    canonical = getattr(resource, "canonical", None)
    if canonical:
        return canonical
    path = getattr(route, "path", None)
    if path:
        return path
    return request.rel_url.path


def create_rate_limit_middleware() -> web.middleware:
    allow_raw = os.getenv("ALLOWLIST_CIDR", "")
    cidr_values = [value.strip() for value in allow_raw.split(",") if value.strip()]
    allowlist = AllowList(cidr_values)

    rules = [
        RateLimitRule(
            method="POST",
            path="/v1/devices/attach",
            scope="ip",
            limiter=TokenBucketLimiter(
                _env_int("RL_ATTACH_IP_PER_MIN", 10),
                _env_int("RL_ATTACH_IP_WINDOW_SEC", 60),
            ),
        ),
        RateLimitRule(
            method="POST",
            path="/v1/uploads",
            scope="device",
            limiter=TokenBucketLimiter(
                _env_int("RL_UPLOADS_PER_MIN", 20),
                _env_int("RL_UPLOADS_WINDOW_SEC", 60),
            ),
        ),
        RateLimitRule(
            method="GET",
            path="/v1/uploads/{id}/status",
            scope="device",
            limiter=TokenBucketLimiter(
                _env_int("RL_UPLOAD_STATUS_PER_MIN", 60),
                _env_int("RL_UPLOAD_STATUS_WINDOW_SEC", 60),
            ),
        ),
        RateLimitRule(
            method="GET",
            path="/v1/health",
            scope="ip",
            limiter=TokenBucketLimiter(
                _env_int("RL_HEALTH_PER_MIN", 30),
                _env_int("RL_HEALTH_WINDOW_SEC", 60),
            ),
        ),
        RateLimitRule(
            method="GET",
            path="/metrics",
            scope="ip",
            limiter=TokenBucketLimiter(
                _env_int("RL_METRICS_PER_MIN", 5),
                _env_int("RL_METRICS_WINDOW_SEC", 60),
            ),
        ),
    ]

    @web.middleware
    async def middleware(request: web.Request, handler: Callable[[web.Request], web.StreamResponse]):
        canonical = _route_canonical(request)
        route_key = (request.method.upper(), canonical)

        ip = _client_ip(request)
        if canonical == "/metrics" or (canonical and canonical.startswith("/_admin")):
            if not allowlist.allows(ip):
                request["rate_limit_log"] = {
                    "result": "hit",
                    "scope": "allowlist",
                    "key": f"allowlist:{ip or 'unknown'}",
                }
                return web.Response(status=403, text="forbidden")

        for rule in rules:
            if rule.method != route_key[0]:
                continue
            if rule.path != route_key[1]:
                continue
            if rule.scope == "ip":
                key = ip or "unknown"
            elif rule.scope == "device":
                key = request.get("device_id") or request.headers.get("X-Device-Id")
                if not key:
                    key = "unknown"
            else:
                continue
            allowance = await rule.limiter.allow(key)
            request.setdefault(
                "rate_limit_log",
                {
                    "result": "miss",
                    "scope": rule.scope,
                    "limit": rule.limiter.capacity,
                    "window": rule.limiter.window,
                    "key": f"{rule.scope}:{key}",
                },
            )
            if not allowance.allowed:
                retry_after_seconds = allowance.retry_after_seconds
                headers: dict[str, str] | None = None
                if (
                    retry_after_seconds is not None
                    and retry_after_seconds >= 0
                    and math.isfinite(retry_after_seconds)
                ):
                    delay_seconds = max(0, int(math.ceil(retry_after_seconds)))
                    headers = {"Retry-After": str(delay_seconds)}
                request["rate_limit_log"] = {
                    "result": "hit",
                    "scope": rule.scope,
                    "limit": rule.limiter.capacity,
                    "window": rule.limiter.window,
                    "key": f"{rule.scope}:{key}",
                    "retry_after": retry_after_seconds,
                }
                record_rate_limit_drop(rule.path, rule.scope)
                return web.json_response(
                    {"error": "rate_limited"}, status=429, headers=headers
                )
        response = await handler(request)
        return response

    return middleware
