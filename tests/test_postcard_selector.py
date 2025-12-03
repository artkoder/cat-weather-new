from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from zoneinfo import ZoneInfo

from data_access import DataAccess
from asset_selectors.postcard import (
    _MIN_SCORE as POSTCARD_SELECTOR_MIN_SCORE,
    select_postcard_asset,
)


@pytest.fixture
def data(tmp_path: Path) -> DataAccess:
    db_path = tmp_path / "postcard.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE assets (
            id TEXT PRIMARY KEY,
            payload_json TEXT,
            created_at TEXT NOT NULL,
            source TEXT,
            postcard_score INTEGER,
            captured_at TEXT,
            photo_doy INTEGER
        );

        CREATE TABLE vision_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id TEXT NOT NULL,
            result_json TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE rubrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE,
            title TEXT,
            description TEXT,
            config TEXT,
            created_at TEXT,
            updated_at TEXT
        );

        CREATE TABLE posts_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id INTEGER NOT NULL,
            message_id INTEGER NOT NULL,
            asset_id TEXT,
            rubric_id INTEGER,
            metadata TEXT,
            published_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(rubric_id) REFERENCES rubrics(id)
        );
        """
    )
    conn.execute(
        "INSERT INTO rubrics (code, title, created_at, updated_at) VALUES ('postcard', 'Postcard', datetime('now'), datetime('now'))"
    )
    conn.commit()
    return DataAccess(conn)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _add_asset(
    data: DataAccess,
    *,
    asset_id: str,
    score: int,
    created_at: datetime,
    captured_at: datetime | None,
    photo_doy: int | None,
    city: str | None = None,
    region: str | None = None,
    last_used_at: datetime | None = None,
    last_used_history: list[datetime] | None = None,
    tags: list[str] | None = None,
) -> None:
    payload: dict[str, object] = {}
    if city:
        payload["city"] = city
    if region:
        payload["region"] = region
    if last_used_history:
        payload["postcard_last_used_at"] = [_iso(moment) for moment in last_used_history]
    elif last_used_at:
        payload["postcard_last_used_at"] = _iso(last_used_at)
    if tags is not None:
        payload["tags"] = tags
    data.conn.execute(
        """
        INSERT INTO assets (id, payload_json, created_at, source, postcard_score, captured_at, photo_doy)
        VALUES (?, ?, ?, 'telegram', ?, ?, ?)
        """,
        (
            asset_id,
            json.dumps(payload, ensure_ascii=False),
            _iso(created_at),
            score,
            _iso(captured_at) if captured_at else None,
            photo_doy,
        ),
    )
    data.conn.commit()


def test_selects_best_recent_asset(data: DataAccess) -> None:
    tz = ZoneInfo("Europe/Kaliningrad")
    now = datetime(2024, 6, 5, 12, 0, tzinfo=tz)
    doy_now = now.timetuple().tm_yday

    _add_asset(
        data,
        asset_id="asset-a",
        score=10,
        created_at=now - timedelta(days=1),
        captured_at=now - timedelta(days=1),
        photo_doy=doy_now,
        city="Kaliningrad",
    )
    _add_asset(
        data,
        asset_id="asset-b",
        score=10,
        created_at=now - timedelta(days=5),
        captured_at=now - timedelta(days=5),
        photo_doy=doy_now,
    )
    _add_asset(
        data,
        asset_id="asset-c",
        score=8,
        created_at=now - timedelta(days=1),
        captured_at=now - timedelta(days=1),
        photo_doy=doy_now,
    )

    asset = select_postcard_asset(data, now=now)
    assert asset is not None
    assert asset.id == "asset-a"


def test_uses_stale_asset_when_no_fresh_available(data: DataAccess) -> None:
    tz = ZoneInfo("Europe/Kaliningrad")
    now = datetime(2024, 7, 10, 9, 0, tzinfo=tz)
    doy_now = now.timetuple().tm_yday

    _add_asset(
        data,
        asset_id="asset-old",
        score=10,
        created_at=now - timedelta(days=6),
        captured_at=now - timedelta(days=6),
        photo_doy=doy_now,
    )
    _add_asset(
        data,
        asset_id="asset-lower",
        score=8,
        created_at=now - timedelta(days=1),
        captured_at=now - timedelta(days=1),
        photo_doy=doy_now,
    )

    asset = select_postcard_asset(data, now=now)
    assert asset is not None
    assert asset.id == "asset-old"


def test_logs_season_fallback_when_no_match(
    data: DataAccess, caplog: pytest.LogCaptureFixture
) -> None:
    tz = ZoneInfo("Europe/Kaliningrad")
    now = datetime(2024, 2, 1, 8, 0, tzinfo=tz)
    far_doy = ((now.timetuple().tm_yday + 200 - 1) % 366) + 1

    _add_asset(
        data,
        asset_id="asset-season",
        score=10,
        created_at=now - timedelta(days=1),
        captured_at=now - timedelta(days=1),
        photo_doy=far_doy,
        region="Amber",
    )

    caplog.set_level(logging.INFO)
    asset = select_postcard_asset(data, now=now)
    assert asset is not None

    season_logs = [
        record.getMessage()
        for record in caplog.records
        if "POSTCARD_RUBRIC season" in record.getMessage()
    ]
    assert season_logs, "Season log should be present"
    assert "fallback=True" in season_logs[-1]


def test_repeat_guard_skips_recently_used_assets(data: DataAccess) -> None:
    tz = ZoneInfo("Europe/Kaliningrad")
    now = datetime(2024, 8, 20, 18, 0, tzinfo=tz)
    doy_now = now.timetuple().tm_yday

    _add_asset(
        data,
        asset_id="asset-stale",
        score=10,
        created_at=now - timedelta(days=2),
        captured_at=now - timedelta(days=2),
        photo_doy=doy_now,
        last_used_at=now - timedelta(days=1),
    )
    _add_asset(
        data,
        asset_id="asset-fresh",
        score=10,
        created_at=now - timedelta(days=2),
        captured_at=now - timedelta(days=2),
        photo_doy=doy_now,
        last_used_at=now - timedelta(days=10),
    )

    asset = select_postcard_asset(data, now=now)
    assert asset is not None
    assert asset.id == "asset-fresh"


def test_repeat_guard_broadens_to_out_of_season_assets(data: DataAccess) -> None:
    tz = ZoneInfo("Europe/Kaliningrad")
    now = datetime(2024, 3, 15, 9, 0, tzinfo=tz)
    doy_now = now.timetuple().tm_yday
    far_doy = ((doy_now + 200 - 1) % 366) + 1

    _add_asset(
        data,
        asset_id="asset-in-season-recent",
        score=10,
        created_at=now - timedelta(days=1),
        captured_at=now - timedelta(days=1),
        photo_doy=doy_now,
        last_used_at=now - timedelta(days=1),
    )
    _add_asset(
        data,
        asset_id="asset-out-of-season-available",
        score=10,
        created_at=now - timedelta(days=2),
        captured_at=now - timedelta(days=2),
        photo_doy=far_doy,
    )

    asset = select_postcard_asset(data, now=now)
    assert asset is not None
    assert asset.id == "asset-out-of-season-available"


def test_repeat_guard_falls_back_to_lower_score_when_recent_high_score_used(
    data: DataAccess,
) -> None:
    tz = ZoneInfo("Europe/Kaliningrad")
    now = datetime(2024, 9, 15, 10, 0, tzinfo=tz)
    doy_now = now.timetuple().tm_yday

    _add_asset(
        data,
        asset_id="asset-recent-top",
        score=10,
        created_at=now - timedelta(days=2),
        captured_at=now - timedelta(days=2),
        photo_doy=doy_now,
        last_used_at=now - timedelta(days=1),
    )
    _add_asset(
        data,
        asset_id="asset-lower-fresh",
        score=8,
        created_at=now - timedelta(days=1),
        captured_at=now - timedelta(days=1),
        photo_doy=doy_now,
        last_used_at=None,
    )

    asset = select_postcard_asset(data, now=now)
    assert asset is not None
    assert asset.id == "asset-lower-fresh"


def test_test_mode_randomizes_selection(data: DataAccess, monkeypatch: pytest.MonkeyPatch) -> None:
    tz = ZoneInfo("Europe/Kaliningrad")
    now = datetime(2024, 8, 25, 12, 0, tzinfo=tz)
    doy_now = now.timetuple().tm_yday

    _add_asset(
        data,
        asset_id="asset-priority",
        score=10,
        created_at=now - timedelta(days=1),
        captured_at=now - timedelta(days=1),
        photo_doy=doy_now,
        last_used_at=now - timedelta(days=10),
    )
    _add_asset(
        data,
        asset_id="asset-random",
        score=10,
        created_at=now - timedelta(days=2),
        captured_at=now - timedelta(days=2),
        photo_doy=doy_now,
        last_used_at=now - timedelta(days=9),
    )

    def fake_time_choice(items, _moment):
        assert len(items) == 2
        for candidate in items:
            if candidate.asset_id == "asset-random":
                return candidate
        raise AssertionError("asset-random not found in candidates")

    monkeypatch.setattr("asset_selectors.postcard._time_random_choice", fake_time_choice)

    asset = select_postcard_asset(data, now=now, test=True)
    assert asset is not None
    assert asset.id == "asset-random"


def test_test_mode_random_pool_expands_when_only_one_fresh_candidate(
    data: DataAccess, monkeypatch: pytest.MonkeyPatch
) -> None:
    tz = ZoneInfo("Europe/Kaliningrad")
    now = datetime(2024, 10, 1, 9, 0, tzinfo=tz)
    doy_now = now.timetuple().tm_yday

    _add_asset(
        data,
        asset_id="asset-fresh",
        score=10,
        created_at=now - timedelta(days=1),
        captured_at=now - timedelta(days=1),
        photo_doy=doy_now,
        last_used_at=now - timedelta(days=30),
    )
    _add_asset(
        data,
        asset_id="asset-stale-a",
        score=10,
        created_at=now - timedelta(days=5),
        captured_at=now - timedelta(days=5),
        photo_doy=doy_now,
        last_used_at=now - timedelta(days=40),
    )
    _add_asset(
        data,
        asset_id="asset-stale-b",
        score=10,
        created_at=now - timedelta(days=6),
        captured_at=now - timedelta(days=6),
        photo_doy=doy_now,
        last_used_at=now - timedelta(days=45),
    )

    def fake_time_choice(items, _moment):
        ids = {candidate.asset_id for candidate in items}
        assert ids == {"asset-fresh", "asset-stale-a", "asset-stale-b"}
        for candidate in items:
            if candidate.asset_id == "asset-stale-b":
                return candidate
        raise AssertionError("asset-stale-b not found in candidates")

    monkeypatch.setattr("asset_selectors.postcard._time_random_choice", fake_time_choice)

    asset = select_postcard_asset(data, now=now, test=True)
    assert asset is not None
    assert asset.id == "asset-stale-b"


def test_returns_none_when_scores_below_threshold(data: DataAccess) -> None:
    tz = ZoneInfo("Europe/Kaliningrad")
    now = datetime(2024, 9, 5, 10, 0, tzinfo=tz)
    doy_now = now.timetuple().tm_yday

    _add_asset(
        data,
        asset_id="asset-low",
        score=POSTCARD_SELECTOR_MIN_SCORE - 1,
        created_at=now - timedelta(days=1),
        captured_at=now - timedelta(days=1),
        photo_doy=doy_now,
    )

    asset = select_postcard_asset(data, now=now)
    assert asset is None


def test_tag_strict_window_skips_high_overlap(data: DataAccess) -> None:
    tz = ZoneInfo("Europe/Kaliningrad")
    now = datetime(2024, 7, 10, 12, 0, tzinfo=tz)
    doy_now = now.timetuple().tm_yday

    _add_asset(
        data,
        asset_id="asset-recent-tag",
        score=10,
        created_at=now - timedelta(days=2),
        captured_at=now - timedelta(days=2),
        photo_doy=doy_now,
        last_used_history=[now - timedelta(days=1)],
        tags=["sea", "sunset"],
    )
    _add_asset(
        data,
        asset_id="asset-high-overlap",
        score=10,
        created_at=now - timedelta(days=1),
        captured_at=now - timedelta(days=1),
        photo_doy=doy_now,
        tags=["sea", "sunset", "pier"],
    )
    _add_asset(
        data,
        asset_id="asset-clear",
        score=10,
        created_at=now - timedelta(days=1),
        captured_at=now - timedelta(days=1),
        photo_doy=doy_now,
        tags=["forest"],
    )

    asset = select_postcard_asset(data, now=now)
    assert asset is not None
    assert asset.id == "asset-clear"


def test_tag_soft_window_downranks_candidates(data: DataAccess) -> None:
    tz = ZoneInfo("Europe/Kaliningrad")
    now = datetime(2024, 7, 20, 8, 0, tzinfo=tz)
    doy_now = now.timetuple().tm_yday

    _add_asset(
        data,
        asset_id="asset-soft-history",
        score=10,
        created_at=now - timedelta(days=10),
        captured_at=now - timedelta(days=10),
        photo_doy=doy_now,
        last_used_history=[now - timedelta(days=5)],
        tags=["cat", "museum"],
    )
    _add_asset(
        data,
        asset_id="asset-penalized",
        score=10,
        created_at=now - timedelta(days=1),
        captured_at=now - timedelta(days=1),
        photo_doy=doy_now,
        tags=["cat", "river"],
    )
    _add_asset(
        data,
        asset_id="asset-clean",
        score=10,
        created_at=now - timedelta(days=2),
        captured_at=now - timedelta(days=2),
        photo_doy=doy_now,
        tags=["garden"],
    )

    asset = select_postcard_asset(data, now=now)
    assert asset is not None
    assert asset.id == "asset-clean"
