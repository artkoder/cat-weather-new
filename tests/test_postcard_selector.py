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
        """
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
) -> None:
    payload: dict[str, object] = {}
    if city:
        payload["city"] = city
    if region:
        payload["region"] = region
    if last_used_at:
        payload["last_used_at"] = _iso(last_used_at)
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
