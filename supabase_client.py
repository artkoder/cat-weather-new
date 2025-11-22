from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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

    async def get_24h_usage_total(
        self,
        *,
        bot: str = "kotopogoda",
        model: str = "gpt-4o-mini",
    ) -> tuple[int | None, Any | None, str | None]:
        """
        Возвращает (used_tokens, raw_payload, error_message).
        used_tokens = суммарный total_tokens за последние 24 часа или None при ошибке/disabled.
        """
        if not self._client:
            logging.debug("Supabase client disabled; skipping token usage check")
            return None, None, "disabled"

        since_iso = (datetime.now(UTC) - timedelta(hours=24)).isoformat()
        params = {
            "select": "sum_total:sum(total_tokens)",
            "bot": f"eq.{bot}",
            "model": f"eq.{model}",
            "at": f"gte.{since_iso}",
        }

        try:
            response = await self._client.get("/token_usage", params=params)
        except httpx.HTTPError as exc:
            logging.warning("Failed to fetch 24h OCR usage from Supabase: %s", exc)
            return None, None, str(exc)

        raw_payload: Any | None = None
        parse_error: Exception | None = None
        try:
            raw_payload = response.json()
        except (ValueError, TypeError) as exc:
            parse_error = exc
            raw_payload = None
            if response.status_code != 200:
                logging.warning("Failed to parse Supabase error payload: %s", exc)

        if response.status_code != 200:
            message = f"HTTP {response.status_code}: {response.text}".strip()
            logging.warning("Failed to fetch 24h OCR usage from Supabase: %s", message)
            return None, raw_payload, message

        if parse_error is not None:
            logging.warning("Failed to parse 24h OCR usage response: %s", parse_error)
            return None, None, str(parse_error)

        first_row: Any
        if isinstance(raw_payload, list):
            if not raw_payload:
                return 0, raw_payload, None
            first_row = raw_payload[0]
        elif isinstance(raw_payload, dict):
            first_row = raw_payload
        else:
            logging.warning(
                "Failed to parse 24h OCR usage from Supabase: unexpected format %s",
                raw_payload,
            )
            return None, raw_payload, "invalid_payload"

        sum_present = False
        sum_value: Any | None = None
        if isinstance(first_row, dict):
            if "sum_total" in first_row:
                sum_present = True
                sum_value = first_row.get("sum_total")
            elif "sum" in first_row:
                sum_present = True
                sum_value = first_row.get("sum")
        if not sum_present:
            logging.warning(
                "Failed to parse 24h OCR usage from Supabase: missing sum_total in %s",
                raw_payload,
            )
            return None, raw_payload, "invalid_payload"

        if sum_value is None:
            return 0, raw_payload, None

        try:
            used_tokens = int(sum_value)
        except (TypeError, ValueError) as exc:
            logging.warning("Failed to parse 24h OCR usage total: %s", exc)
            return None, raw_payload, str(exc)

        return max(used_tokens, 0), raw_payload, None
