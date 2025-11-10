"""Test that wave_score_0_10 and wave_conf are saved during ingestion."""

import sqlite3
from pathlib import Path

import pytest

from data_access import DataAccess


@pytest.fixture
def data(tmp_path: Path) -> DataAccess:
    """Create a test database with minimal schema for testing."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Create minimal schema needed for testing save_asset
    conn.executescript(
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
            created_at TEXT NOT NULL,
            source TEXT,
            shot_at_utc INTEGER,
            shot_doy INTEGER,
            photo_doy INTEGER,
            photo_wave REAL,
            sky_visible TEXT,
            wave_score_0_10 REAL,
            wave_conf REAL,
            sky_code TEXT,
            vision_wave_score REAL,
            vision_wave_conf REAL,
            vision_sky_bucket TEXT,
            captured_at TEXT,
            doy INTEGER,
            daypart TEXT
        );

        CREATE TABLE vision_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id TEXT NOT NULL,
            provider TEXT,
            status TEXT,
            category TEXT,
            arch_view TEXT,
            photo_weather TEXT,
            flower_varieties TEXT,
            confidence REAL,
            result_json TEXT,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()

    return DataAccess(conn)


def test_save_asset_with_wave_score_and_conf(data: DataAccess) -> None:
    """Test that save_asset properly saves wave_score_0_10 and wave_conf."""
    # Create an asset with wave score and confidence
    asset_id = data.save_asset(
        channel_id=123,
        message_id=456,
        template=None,
        hashtags=None,
        tg_chat_id=123,
        caption="Test sea photo",
        kind=None,
        wave_score_0_10=3.0,
        wave_conf=0.85,
    )

    # Retrieve the asset and verify the columns are populated
    asset = data.get_asset(asset_id)

    assert asset is not None
    assert asset.wave_score_0_10 == 3.0
    assert asset.wave_conf == 0.85


def test_save_asset_without_wave_data(data: DataAccess) -> None:
    """Test that save_asset works without wave data (backwards compatibility)."""
    # Create an asset without wave score
    asset_id = data.save_asset(
        channel_id=123,
        message_id=789,
        template=None,
        hashtags=None,
        tg_chat_id=123,
        caption="Test photo without waves",
        kind=None,
    )

    # Retrieve the asset and verify it was created successfully
    asset = data.get_asset(asset_id)

    assert asset is not None
    assert asset.wave_score_0_10 is None
    assert asset.wave_conf is None


def test_save_asset_updates_wave_data(data: DataAccess) -> None:
    """Test that save_asset can update existing asset's wave data."""
    # Create an asset without wave data
    asset_id = data.save_asset(
        channel_id=123,
        message_id=999,
        template=None,
        hashtags=None,
        tg_chat_id=123,
        caption="Test photo",
        kind=None,
    )

    # Verify no wave data initially
    asset = data.get_asset(asset_id)
    assert asset.wave_score_0_10 is None
    assert asset.wave_conf is None

    # Update with wave data (simulates processing after upload)
    data.save_asset(
        channel_id=123,
        message_id=999,
        template=None,
        hashtags=None,
        tg_chat_id=123,
        caption="Test photo with waves",
        kind=None,
        wave_score_0_10=5.0,
        wave_conf=0.92,
    )

    # Verify wave data is now saved
    asset = data.get_asset(asset_id)
    assert asset.wave_score_0_10 == 5.0
    assert asset.wave_conf == 0.92
