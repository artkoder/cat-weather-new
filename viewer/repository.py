"""In-memory photo repository with efficient date-based lookup."""

from __future__ import annotations

from bisect import bisect_left
from datetime import datetime, timezone
from typing import Iterable, List, Sequence

from .models import PhotoItem


UTC = timezone.utc
_SENTINEL_MAX = datetime.max.replace(tzinfo=UTC)


def _normalized(photo: PhotoItem) -> datetime:
    taken = photo.normalized_taken_at()
    if taken is None:
        return _SENTINEL_MAX
    return taken


class PhotoRepository:
    """Stores photos sorted by their capture timestamps."""

    def __init__(self, photos: Iterable[PhotoItem]):
        self._photos: List[PhotoItem] = sorted(
            photos,
            key=lambda photo: (_normalized(photo), photo.id),
        )

    @property
    def photos(self) -> Sequence[PhotoItem]:
        """Return the ordered photo sequence."""

        return self._photos

    def find_index_at_or_after(self, date: datetime) -> int:
        """Return the index of the first photo captured at or after ``date``.

        Items without a capture timestamp are treated as very old entries and
        are consequently sorted to the end of the ordered list.
        """

        if date.tzinfo is None:
            raise ValueError("date must be timezone-aware")

        normalized_target = date.astimezone(UTC)
        normalized_dates = [_normalized(photo) for photo in self._photos]
        pos = bisect_left(normalized_dates, normalized_target)
        if not self._photos:
            return 0
        return min(pos, len(self._photos) - 1)
