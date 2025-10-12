from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Iterable, Mapping, MutableMapping
from uuid import uuid4

from aiohttp import web
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    ProcessCollector,
    PlatformCollector,
    generate_latest,
)
from prometheus_client import GCCollector


_LOG_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "log_context", default={}
)

_LOG_FORMAT = "json"
_LOG_LEVEL = logging.INFO

_SENSITIVE_KEYS = (
    "secret",
    "token",
    "signature",
    "authorization",
    "password",
    "apikey",
    "api_key",
)

_HEADER_RE = re.compile(
    r"(?i)(x-signature|signature|secret|token|authorization)([:=]\s*)([^\s,;]+)"
)
_JSON_RE = re.compile(
    r"(?i)(\"(?:secret|token|x-signature|authorization)\"\s*:\s*)\"[^\"]*\""
)


def _redact_value(key: str | None, value: Any) -> Any:
    if key and any(token in key.lower() for token in _SENSITIVE_KEYS):
        return "***"
    if isinstance(value, str):
        redacted = value
        redacted = _HEADER_RE.sub(
            lambda match: f"{match.group(1)}{match.group(2)}***",
            redacted,
        )
        redacted = _JSON_RE.sub(
            lambda match: f"{match.group(1)}\"***\"",
            redacted,
        )
        return redacted
    if isinstance(value, Mapping):
        return {k: _redact_value(k, v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        iterable = list(value)
        return type(value)(_redact_value(None, item) for item in iterable)  # type: ignore[call-arg]
    return value


def _current_context() -> dict[str, Any]:
    return dict(_LOG_CONTEXT.get({}))


def bind_context(**updates: Any) -> contextvars.Token[dict[str, Any]]:
    ctx = _current_context()
    for key, value in updates.items():
        if value is None:
            ctx.pop(key, None)
        else:
            ctx[key] = value
    return _LOG_CONTEXT.set(ctx)


@contextlib.contextmanager
def context(**updates: Any):
    token = bind_context(**updates)
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - trivial
        context = _current_context()
        for key, value in context.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        for key in (
            "request_id",
            "route",
            "method",
            "status",
            "duration_ms",
            "device_id",
            "source",
            "ip",
            "job",
            "upload_id",
            "rl",
            "rl_limit",
            "rl_window",
            "rl_key",
            "rl_scope",
        ):
            if not hasattr(record, key):
                setattr(record, key, None)
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
        }
        message = record.getMessage()
        base["msg"] = _redact_value("msg", message)
        for key in (
            "request_id",
            "route",
            "method",
            "status",
            "duration_ms",
            "device_id",
            "source",
            "ip",
            "job",
            "upload_id",
            "rl",
            "rl_limit",
            "rl_window",
            "rl_key",
            "rl_scope",
        ):
            value = getattr(record, key, None)
            if value is not None:
                base[key] = _redact_value(key, value)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key
            not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
            }
        }
        for key, value in extras.items():
            if key.startswith("_"):
                continue
            if key in base:
                continue
            base[key] = _redact_value(key, value)
        if record.exc_info:
            base["error_type"] = getattr(record.exc_info[0], "__name__", "Exception")
            if _LOG_FORMAT == "pretty":
                base["stack"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)


class PrettyFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%fZ")
        message = _redact_value("msg", record.getMessage())
        parts = [f"[{ts}]", record.levelname.ljust(5), str(message)]
        extras: list[str] = []
        for key in (
            "request_id",
            "route",
            "method",
            "status",
            "duration_ms",
            "device_id",
            "source",
            "ip",
            "job",
            "upload_id",
            "rl",
            "rl_limit",
            "rl_window",
            "rl_key",
            "rl_scope",
        ):
            value = getattr(record, key, None)
            if value:
                extras.append(f"{key}={_redact_value(key, value)}")
        if extras:
            parts.append("(" + " ".join(extras) + ")")
        if record.exc_info:
            parts.append("\n" + self.formatException(record.exc_info))
        return " ".join(parts)


def setup_logging(*, stream: Any | None = None) -> None:
    global _LOG_FORMAT, _LOG_LEVEL
    format_name = os.getenv("LOG_FORMAT", "json").strip().lower()
    level_name = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    formatter: logging.Formatter
    if format_name == "pretty":
        formatter = PrettyFormatter()
    else:
        format_name = "json"
        formatter = JsonFormatter()
    _LOG_FORMAT = format_name
    _LOG_LEVEL = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(_LOG_LEVEL)
    root.addFilter(ContextFilter())


def is_pretty_format() -> bool:
    return _LOG_FORMAT == "pretty"


def log_exc(ctx: str, err: BaseException) -> None:
    logger = logging.getLogger("observability")
    extra = {"error_type": type(err).__name__, "error": str(err)}
    if is_pretty_format():
        logger.exception(ctx, extra=extra)
    else:
        logger.error(ctx, extra=extra)


_HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    labelnames=("method", "route", "status"),
)
_HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    labelnames=("route",),
)
_HMAC_FAIL_TOTAL = Counter(
    "hmac_fail_total",
    "Total number of HMAC authentication failures",
    labelnames=("reason",),
)
_UPLOADS_TOTAL = Counter(
    "uploads_total",
    "Upload lifecycle events",
    labelnames=("action",),
)
_MOBILE_PHOTOS_TOTAL = Counter(
    "mobile_photos_total",
    "Total mobile photos ingested",
)
_JOBS_PROCESSED_TOTAL = Counter(
    "jobs_processed_total",
    "Jobs processed outcomes",
    labelnames=("job", "status"),
)
_QUEUE_DEPTH = Gauge(
    "queue_depth",
    "Depth of background job queues",
    labelnames=("queue",),
)
_STORAGE_PUT_BYTES_TOTAL = Counter(
    "storage_put_bytes_total",
    "Total bytes written to storage",
)
_HEALTH_LATENCY_SECONDS = Histogram(
    "health_latency_seconds",
    "Latency of health check internal probes",
)

_RATE_LIMIT_DROPPED_TOTAL = Counter(
    "rate_limit_dropped_total",
    "Requests rejected due to rate limiting",
    labelnames=("route", "key"),
)


_METRICS_INITIALIZED = False


def _ensure_runtime_collectors() -> None:
    global _METRICS_INITIALIZED
    if _METRICS_INITIALIZED:
        return
    for collector in (ProcessCollector, PlatformCollector, GCCollector):
        try:
            collector()
        except ValueError:
            # Already registered in this process.
            continue
    _METRICS_INITIALIZED = True


def _client_ip(request: web.Request) -> str | None:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote


async def metrics_handler(request: web.Request) -> web.Response:
    _ensure_runtime_collectors()
    payload = generate_latest()
    return web.Response(body=payload, headers={"Content-Type": CONTENT_TYPE_LATEST})


def _resolve_route_label(request: web.Request) -> str:
    route = request.match_info.route
    if route is not None:
        resource = getattr(route, "resource", None)
        if resource is not None:
            canonical = getattr(resource, "canonical", None)
            if canonical:
                return canonical
        route_name = getattr(route, "name", None)
        if route_name:
            return str(route_name)
    return request.rel_url.path


@web.middleware
async def observability_middleware(request: web.Request, handler):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    ip = _client_ip(request)
    token = bind_context(
        request_id=request_id,
        method=request.method,
        route=request.rel_url.path,
        ip=ip,
    )
    request["request_id"] = request_id
    start = time.perf_counter()
    response: web.StreamResponse | None = None
    status: int = 500
    try:
        try:
            response = await handler(request)
        except web.HTTPException as http_exc:
            status = http_exc.status
            http_exc.headers.setdefault("X-Request-ID", request_id)
            raise
        else:
            status = response.status
            return response
    finally:
        duration = time.perf_counter() - start
        route_label = _resolve_route_label(request)
        device_id = request.get("device_id")
        upload_id = request.get("upload_id")
        job_name = request.get("job")
        with context(
            route=route_label,
            status=status,
            duration_ms=round(duration * 1000.0, 3),
            device_id=device_id,
            upload_id=upload_id,
            job=job_name,
        ):
            if response is not None and isinstance(response, web.StreamResponse):
                response.headers.setdefault("X-Request-ID", request_id)
            _HTTP_REQUESTS_TOTAL.labels(
                method=request.method,
                route=route_label,
                status=str(status),
            ).inc()
            _HTTP_REQUEST_DURATION.labels(route=route_label).observe(duration)
            rl_info = request.get("rate_limit_log") or {}
            logging.getLogger("aiohttp.access").info(
                "request_completed",
                extra={
                    "request_id": request_id,
                    "route": route_label,
                    "method": request.method,
                    "status": status,
                    "duration_ms": round(duration * 1000.0, 3),
                    "device_id": device_id,
                    "ip": ip,
                    "upload_id": upload_id,
                    "job": job_name,
                    "rl": rl_info.get("result"),
                    "rl_limit": rl_info.get("limit"),
                    "rl_window": rl_info.get("window"),
                    "rl_key": rl_info.get("key"),
                    "rl_scope": rl_info.get("scope"),
                },
            )
        _LOG_CONTEXT.reset(token)


def record_rate_limit_drop(route: str, key: str) -> None:
    _RATE_LIMIT_DROPPED_TOTAL.labels(route=route, key=key).inc()


def record_hmac_failure(reason: str) -> None:
    _HMAC_FAIL_TOTAL.labels(reason=reason).inc()


def record_upload_created() -> None:
    _UPLOADS_TOTAL.labels(action="created").inc()


def record_upload_status_change() -> None:
    _UPLOADS_TOTAL.labels(action="status_change").inc()


def record_job_processed(job: str, status: str) -> None:
    _JOBS_PROCESSED_TOTAL.labels(job=job, status=status).inc()


def record_mobile_photo_ingested() -> None:
    _MOBILE_PHOTOS_TOTAL.inc()


def set_queue_depth(queue: str, depth: int) -> None:
    _QUEUE_DEPTH.labels(queue=queue).set(depth)


def record_storage_put_bytes(size: int) -> None:
    if size < 0:
        return
    _STORAGE_PUT_BYTES_TOTAL.inc(size)


def observe_health_latency(seconds: float) -> None:
    if seconds < 0:
        return
    _HEALTH_LATENCY_SECONDS.observe(seconds)


__all__ = [
    "bind_context",
    "context",
    "log_exc",
    "metrics_handler",
    "observability_middleware",
    "record_rate_limit_drop",
    "record_hmac_failure",
    "record_job_processed",
    "record_mobile_photo_ingested",
    "record_storage_put_bytes",
    "record_upload_created",
    "record_upload_status_change",
    "set_queue_depth",
    "setup_logging",
    "observe_health_latency",
]
