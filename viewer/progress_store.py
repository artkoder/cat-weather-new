"""Persistence helpers for review progress."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Dict, Optional

from .models import ReviewPosition

UTC = timezone.utc


def folder_id_from_uri(uri: str) -> str:
    """Return a stable identifier derived from a folder URI."""

    digest = sha256(uri.encode("utf-8")).hexdigest()
    return digest


class ReviewProgressStore:
    """Persists review positions on disk using a JSON document."""

    def __init__(self, storage_path: Path) -> None:
        self._storage_path = storage_path

    def _read(self) -> Dict[str, Dict[str, int]]:
        if not self._storage_path.exists():
            return {}
        data = json.loads(self._storage_path.read_text())
        if not isinstance(data, dict):
            raise ValueError("Corrupted progress store: expected object at root")
        return data

    def _write(self, data: Dict[str, Dict[str, int]]) -> None:
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage_path.write_text(json.dumps(data, separators=(",", ":")))

    def save_position(
        self, folder_id: str, index: int, anchor_date: Optional[datetime]
    ) -> None:
        """Persist the review state for ``folder_id``."""

        if index < 0:
            raise ValueError("index must be non-negative")
        if anchor_date is not None and anchor_date.tzinfo is None:
            raise ValueError("anchor_date must be timezone-aware")

        data = self._read()
        anchor_ms = -1
        if anchor_date is not None:
            anchor_ms = int(anchor_date.astimezone(UTC).timestamp() * 1000)
        data[folder_id] = {"last_index": index, "last_anchor": anchor_ms}
        self._write(data)

    def load_position(self, folder_id: str) -> Optional[ReviewPosition]:
        """Return the persisted position for ``folder_id`` if available."""

        data = self._read()
        stored = data.get(folder_id)
        if not stored:
            return None
        index = stored.get("last_index")
        anchor_ms = stored.get("last_anchor", -1)
        if index is None:
            return None
        anchor_date = None
        if isinstance(anchor_ms, int) and anchor_ms >= 0:
            anchor_date = datetime.fromtimestamp(anchor_ms / 1000, tz=UTC)
        return ReviewPosition(index=index, anchor_date=anchor_date)

    def clear(self, folder_id: str) -> None:
        """Remove stored progress for ``folder_id``."""

        data = self._read()
        if folder_id in data:
            data.pop(folder_id)
            self._write(data)
