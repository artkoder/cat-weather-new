import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_access import DataAccess


def _setup_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE assets (
            id TEXT PRIMARY KEY,
            upload_id TEXT,
            file_ref TEXT,
            content_type TEXT,
            sha256 TEXT,
            width INTEGER,
            height INTEGER,
            exif_json TEXT,
            labels_json TEXT,
            tg_message_id TEXT,
            payload_json TEXT,
            created_at TEXT,
            source TEXT,
            shot_at_utc INTEGER,
            shot_doy INTEGER,
            photo_doy INTEGER,
            photo_wave REAL,
            sky_visible TEXT,
            captured_at TEXT,
            doy INTEGER,
            daypart TEXT,
            vision_wave_score REAL,
            vision_wave_conf REAL,
            vision_sky_bucket TEXT,
            wave_score_0_10 REAL,
            wave_conf REAL,
            sky_code TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE vision_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id TEXT,
            provider TEXT,
            status TEXT,
            category TEXT,
            arch_view TEXT,
            photo_weather TEXT,
            flower_varieties TEXT,
            confidence REAL,
            result_json TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    return conn


def test_create_asset_populates_capture_fields() -> None:
    conn = _setup_connection()
    data = DataAccess(conn)

    capture = datetime(2024, 5, 10, 6, 15, tzinfo=timezone.utc)
    exif = {"DateTimeOriginal": capture.strftime("%Y:%m:%d %H:%M:%S")}
    local_dt = capture.astimezone(ZoneInfo("Europe/Kaliningrad"))
    shot_doy = local_dt.timetuple().tm_yday

    asset_id = data.create_asset(
        upload_id="u1",
        file_ref="file-1",
        content_type="image/jpeg",
        sha256="sha-1",
        width=800,
        height=600,
        exif=exif,
        labels=None,
        tg_message_id=None,
        tg_chat_id=None,
        source="mobile",
        shot_at_utc=int(capture.timestamp()),
        shot_doy=shot_doy,
    )

    row = conn.execute(
        "SELECT captured_at, doy, daypart FROM assets WHERE id=?",
        (asset_id,),
    ).fetchone()
    assert row is not None
    assert row["captured_at"] == capture.isoformat()
    assert row["doy"] == shot_doy
    assert row["daypart"] == "morning"


def test_fetch_sea_candidates_uses_capture_metadata() -> None:
    conn = _setup_connection()
    data = DataAccess(conn)

    now = datetime.now(timezone.utc)

    def _create_sea_asset(idx: int, delta_days: int) -> str:
        capture = now - timedelta(days=delta_days)
        exif = {"DateTimeOriginal": capture.strftime("%Y:%m:%d %H:%M:%S")}
        local_dt = capture.astimezone(ZoneInfo("Europe/Kaliningrad"))
        shot_doy = local_dt.timetuple().tm_yday
        asset_id = data.create_asset(
            upload_id=f"u{idx}",
            file_ref=f"file-{idx}",
            content_type="image/jpeg",
            sha256=f"sha-{idx}",
            width=640,
            height=480,
            exif=exif,
            labels=None,
            tg_message_id=None,
            tg_chat_id=None,
            source="mobile",
            shot_at_utc=int(capture.timestamp()),
            shot_doy=shot_doy,
        )
        data.update_asset(
            asset_id,
            vision_category="sea",
            vision_results={
                "tags": ["sea"],
                "is_sea": True,
                "sea_wave_score": 2.0,
                "photo_sky": "sunny",
            },
            photo_wave=2.0,
            sky_visible=True,
            rubric_id=1,
        )
        return asset_id

    older_id = _create_sea_asset(1, 10)
    recent_id = _create_sea_asset(2, 1)

    candidates = data.fetch_sea_candidates(1, limit=10)
    assert len(candidates) == 2

    bonuses = {cand["asset"].id: cand["age_bonus"] for cand in candidates}
    assert bonuses[older_id] >= bonuses[recent_id]

    for cand in candidates:
        asset = cand["asset"]
        assert asset.captured_at is not None
        assert cand["captured_at"] == asset.captured_at
        assert cand["capture_daypart"] in {"morning", "day", "evening", "night"}
        assert cand["capture_doy"] == asset.doy

    for asset_id, daypart in {
        cand["asset"].id: cand["asset"].daypart for cand in candidates
    }.items():
        assert daypart in {"morning", "day", "evening", "night"}


def test_wave_sky_metrics_columns() -> None:
    """Test that new wave_score_0_10, wave_conf, sky_code columns are persisted."""
    conn = _setup_connection()
    data = DataAccess(conn)

    asset_id = data.create_asset(
        upload_id="u1",
        file_ref="file-1",
        content_type="image/jpeg",
        sha256="sha-1",
        width=800,
        height=600,
        exif=None,
        labels=None,
        tg_message_id=None,
        tg_chat_id=None,
        source="mobile",
    )

    data.update_asset(
        asset_id,
        wave_score_0_10=7.5,
        wave_conf=0.92,
        sky_code="partly_cloudy",
    )

    row = conn.execute(
        "SELECT wave_score_0_10, wave_conf, sky_code FROM assets WHERE id=?",
        (asset_id,),
    ).fetchone()
    assert row is not None
    assert row["wave_score_0_10"] == 7.5
    assert row["wave_conf"] == 0.92
    assert row["sky_code"] == "partly_cloudy"


def test_update_asset_stores_region_payload() -> None:
    conn = _setup_connection()
    data = DataAccess(conn)

    asset_id = data.create_asset(
        upload_id="region-u1",
        file_ref="file-region",
        content_type="image/jpeg",
        sha256="sha-region",
        width=640,
        height=480,
        exif=None,
        labels=None,
        tg_message_id=None,
        tg_chat_id=None,
        source="mobile",
    )

    data.update_asset(
        asset_id,
        city="Светлогорск",
        region="Калининградская область",
        country="Россия",
    )

    asset = data.get_asset(asset_id)
    assert asset is not None
    assert asset.city == "Светлогорск"
    assert asset.region == "Калининградская область"
    assert asset.country == "Россия"
