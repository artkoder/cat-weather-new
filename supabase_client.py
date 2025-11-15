from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Type {type(value)!r} is not JSON serializable")


def _strict_meta(meta: Any) -> Any:
    if meta is None:
        return None
    try:
        serialized = json.dumps(meta, default=_json_default, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("Supabase meta must be JSON serializable") from exc
    return json.loads(serialized)


@dataclass
class SupabaseConfig:
    url: str | None
    key: str | None


class SupabaseClient:
    def __init__(
        self,
        url: str | None = None,
        key: str | None = None,
        *,
        timeout: float = 10.0,
    ) -> None:
        env_key = key or os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        config = SupabaseConfig(url or os.getenv("SUPABASE_URL"), env_key)
        self._enabled = bool(config.url and config.key)
        self._client: httpx.AsyncClient | None = None
        self._api_base: str | None = None
        self._storage_base: str | None = None
        if self._enabled:
            api_base = config.url.rstrip("/")
            rest_base = api_base + "/rest/v1"
            headers = {
                "apikey": config.key,
                "Authorization": f"Bearer {config.key}",
                "Content-Type": "application/json",
            }
            self._client = httpx.AsyncClient(base_url=rest_base, timeout=timeout, headers=headers)
            self._api_base = api_base
            self._storage_base = api_base + "/storage/v1"

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
        bot: str = "kotopogoda",
        model: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        request_id: str | None,
        endpoint: str = "/v1/responses",
        meta: dict[str, Any] | None = None,
    ) -> tuple[bool, dict[str, Any], str | None]:
        payload: dict[str, Any] = {
            "bot": bot,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "request_id": request_id,
            "endpoint": endpoint,
            "meta": _strict_meta(meta),
            "at": datetime.now(UTC).isoformat(),
        }
        if not self._client:
            logging.debug("Supabase client disabled; skipping token usage upload")
            return False, payload, "disabled"

        try:
            response = await self._client.post(
                "/token_usage",
                json=payload,
                headers={"Prefer": "return=minimal"},
            )
            if response.status_code not in (200, 201, 204):
                message = f"HTTP {response.status_code}: {response.text}".strip()
                return False, payload, message
            return True, payload, None
        except httpx.HTTPError as exc:
            return False, payload, str(exc)

    async def upload_object(
        self,
        *,
        bucket: str,
        key: str,
        stream: AsyncIterator[bytes],
        content_type: str,
    ) -> str:
        if not self._client or not self._storage_base:
            raise RuntimeError("Supabase client is disabled")
        normalized_key = key.lstrip("/")
        url = f"{self._storage_base}/object/{bucket}/{normalized_key}"
        headers = {
            "Content-Type": content_type,
            "x-upsert": "true",
        }
        byte_stream = httpx.AsyncIteratorByteStream(stream)
        response = await self._client.post(url, content=byte_stream, headers=headers)
        if response.status_code not in (200, 201, 204):
            message = response.text.strip() or response.reason_phrase or "upload_failed"
            raise RuntimeError(f"Supabase upload failed: {response.status_code} {message}")
        return normalized_key

    def public_url(self, *, bucket: str, key: str) -> str:
        if not self._api_base:
            raise RuntimeError("Supabase client is disabled")
        normalized_key = key.lstrip("/")
        return f"{self._api_base}/storage/v1/object/public/{bucket}/{normalized_key}"

    async def delete_object(self, *, bucket: str, key: str) -> None:
        if not self._client or not self._storage_base:
            raise RuntimeError("Supabase client is disabled")
        normalized_key = key.lstrip("/")
        url = f"{self._storage_base}/object/{bucket}/{normalized_key}"
        try:
            response = await self._client.delete(url)
        except httpx.HTTPError as exc:
            logging.warning(
                "Supabase delete failed bucket=%s key=%s error=%s",
                bucket,
                normalized_key,
                exc,
            )
            return
        if response.status_code not in (200, 204):
            message = response.text.strip() or response.reason_phrase or "delete_failed"
            logging.warning(
                "Supabase delete failed bucket=%s key=%s status=%s message=%s",
                bucket,
                normalized_key,
                response.status_code,
                message,
            )
