from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest

from viewer import PhotoItem, PhotoRepository, ReviewProgressStore, folder_id_from_uri

UTC = timezone.utc


def _dt(days: int) -> datetime:
    return datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=days)


def test_repository_orders_photos_and_finds_index() -> None:
    repo = PhotoRepository(
        [
            PhotoItem("b", _dt(10)),
            PhotoItem("c", None),
            PhotoItem("a", _dt(0)),
        ]
    )

    # Sorted ascending with undated entries at the end.
    assert [photo.id for photo in repo.photos] == ["a", "b", "c"]

    assert repo.find_index_at_or_after(_dt(-5)) == 0
    assert repo.find_index_at_or_after(_dt(0)) == 0
    assert repo.find_index_at_or_after(_dt(5)) == 1
    assert repo.find_index_at_or_after(_dt(25)) == 2


@pytest.mark.parametrize("anchor_days", [None, 5])
def test_progress_store_round_trip(tmp_path: Path, anchor_days: int | None) -> None:
    storage = tmp_path / "progress.json"
    store = ReviewProgressStore(storage)
    folder_id = folder_id_from_uri("content://tree/folder")

    anchor = _dt(anchor_days) if anchor_days is not None else None
    store.save_position(folder_id, 3, anchor)

    loaded = store.load_position(folder_id)
    assert loaded is not None
    assert loaded.index == 3
    if anchor_days is None:
        assert loaded.anchor_date is None
    else:
        assert loaded.anchor_date == anchor

    store.clear(folder_id)
    assert store.load_position(folder_id) is None


def test_progress_store_rejects_invalid_input(tmp_path: Path) -> None:
    store = ReviewProgressStore(tmp_path / "progress.json")
    folder_id = folder_id_from_uri("content://tree/folder")

    with pytest.raises(ValueError):
        store.save_position(folder_id, -1, None)

    with pytest.raises(ValueError):
        naive_date = datetime(2024, 1, 1)
        store.save_position(folder_id, 0, naive_date)
