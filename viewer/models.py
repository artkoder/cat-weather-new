"""Data structures that describe photo viewer entities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


UTC = timezone.utc


@dataclass(frozen=True, slots=True)
class PhotoItem:
    """Represents a photo entry in the viewer catalogue."""

    id: str
    taken_at: Optional[datetime]

    def normalized_taken_at(self) -> Optional[datetime]:
        """Return the capture timestamp converted to UTC."""

        if self.taken_at is None:
            return None
        if self.taken_at.tzinfo is None:
            return self.taken_at.replace(tzinfo=UTC)
        return self.taken_at.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class ReviewPosition:
    """Stores persisted progress for a folder."""

    index: int
    anchor_date: Optional[datetime]
