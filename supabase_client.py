from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx


def _json_default(value: Any) -> Any:
    """Fallback serializer that keeps non-JSON objects readable."""

    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalize_meta(meta: Any) -> Any:
    if meta is None:
        return None
    try:
        serialized = json.dumps(meta, default=_json_default)
    except TypeError:
        serialized = json.dumps(str(meta))
    return json.loads(serialized)


@dataclass
class SupabaseConfig:
    url: Optional[str]
    key: Optional[str]


class SupabaseClient:
    def __init__(
        self,
        url: str | None = None,
        key: str | None = None,
        *,
        timeout: float = 10.0,
    ) -> None:
        config = SupabaseConfig(url or os.getenv("SUPABASE_URL"), key or os.getenv("SUPABASE_KEY"))
        self._enabled = bool(config.url and config.key)
        self._client: httpx.AsyncClient | None = None
        if self._enabled:
            base_url = config.url.rstrip("/") + "/rest/v1"
            headers = {
                "apikey": config.key,
                "Authorization": f"Bearer {config.key}",
                "Content-Type": "application/json",
            }
            self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout, headers=headers)

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def aclose(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def insert_token_usage(
        self,
        *,
        model: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        request_id: str | None,
        meta: Dict[str, Any] | None = None,
    ) -> bool:
        if not self._client:
            logging.debug("Supabase client disabled; skipping token usage upload")
            return False

        payload: Dict[str, Any] = {
            "bot": "kotopogoda",
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "request_id": request_id,
            "endpoint": "responses",
            "meta": _normalize_meta(meta),
            "at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            response = await self._client.post(
                "/token_usage",
                json=payload,
                headers={"Prefer": "return=minimal"},
            )
            if response.status_code not in (200, 201, 204):
                logging.error(
                    "Supabase token usage insert failed: %s %s",
                    response.status_code,
                    response.text,
                    extra={"log_token_usage": payload},
                )
                return False
            logging.info(
                "Supabase token usage insert succeeded",
                extra={"log_token_usage": payload},
            )
            return True
        except httpx.HTTPError as exc:
            logging.error(
                "Supabase token usage insert error: %s",
                exc,
                extra={"log_token_usage": payload},
            )
            return False
