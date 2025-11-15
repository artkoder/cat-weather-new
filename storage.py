from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from supabase_client import SupabaseClient


class Storage(Protocol):
    async def put_stream(
        self,
        *,
        key: str,
        stream: AsyncIterator[bytes],
        content_type: str,
    ) -> str: ...

    async def get_url(self, *, key: str) -> str: ...

    async def delete(self, *, key: str) -> None: ...


class LocalStorage:
    def __init__(self, base_path: Path | str | None = None) -> None:
        path = Path(base_path) if base_path else Path("data/uploads")
        self._base = path.resolve()
        self._base.mkdir(parents=True, exist_ok=True)

    async def put_stream(
        self,
        *,
        key: str,
        stream: AsyncIterator[bytes],
        content_type: str,
    ) -> str:
        normalized_key = key.lstrip("/")
        destination = self._base / normalized_key
        destination.parent.mkdir(parents=True, exist_ok=True)
        logging.info("UPLOAD store local key=%s path=%s", normalized_key, destination)
        with destination.open("wb") as handle:
            async for chunk in stream:
                handle.write(chunk)
        return normalized_key

    async def get_url(self, *, key: str) -> str:
        normalized_key = key.lstrip("/")
        path = self._base / normalized_key
        return path.as_uri()

    async def delete(self, *, key: str) -> None:
        normalized_key = key.lstrip("/")
        path = self._base / normalized_key
        with contextlib.suppress(FileNotFoundError, OSError):
            path.unlink()


@dataclass
class SupabaseStorage:
    client: SupabaseClient
    bucket: str

    async def put_stream(
        self,
        *,
        key: str,
        stream: AsyncIterator[bytes],
        content_type: str,
    ) -> str:
        normalized_key = key.lstrip("/")
        logging.info("UPLOAD store supabase key=%s bucket=%s", normalized_key, self.bucket)
        return await self.client.upload_object(
            bucket=self.bucket,
            key=normalized_key,
            stream=stream,
            content_type=content_type,
        )

    async def get_url(self, *, key: str) -> str:
        normalized_key = key.lstrip("/")
        return self.client.public_url(bucket=self.bucket, key=normalized_key)

    async def delete(self, *, key: str) -> None:
        normalized_key = key.lstrip("/")
        try:
            await self.client.delete_object(bucket=self.bucket, key=normalized_key)
        except Exception:
            logging.exception(
                "UPLOAD storage-delete failed bucket=%s key=%s",
                self.bucket,
                normalized_key,
            )


def create_storage_from_env(
    *,
    base_path: Path | str | None = None,
    supabase: SupabaseClient | None = None,
) -> Storage:
    backend = (os.getenv("STORAGE_BACKEND") or "local").strip().lower()
    if backend == "supabase":
        if not supabase or not supabase.enabled:
            raise RuntimeError("Supabase storage selected but client is disabled")
        bucket = os.getenv("SUPABASE_BUCKET", "uploads").strip() or "uploads"
        return SupabaseStorage(client=supabase, bucket=bucket)
    if backend and backend != "local":
        logging.warning("Unknown STORAGE_BACKEND=%s, falling back to local", backend)
    return LocalStorage(base_path=base_path)
