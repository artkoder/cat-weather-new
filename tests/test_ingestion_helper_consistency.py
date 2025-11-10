"""Tests for ingestion helper consistency - ensuring vision metrics flow through update_asset."""

import os
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_access import DataAccess


@pytest.fixture
def temp_db(tmp_path: Path) -> sqlite3.Connection:
    """Create a temporary database connection for testing."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Create minimal schema needed for DataAccess
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
            vision_wave_score REAL,
            vision_wave_conf REAL,
            vision_sky_bucket TEXT,
            wave_score_0_10 REAL,
            wave_conf REAL,
            sky_code TEXT,
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
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    conn.commit()

    return conn


class TestIngestionHelperConsistency:
    """Test that vision metrics are properly plumbed through update_asset."""

    def test_update_asset_parses_vision_wave_score(self, temp_db: sqlite3.Connection) -> None:
        """Test that update_asset parses and stores wave score from vision_results."""
        data = DataAccess(temp_db)

        # Create a test asset
        asset_id = data.create_asset(
            upload_id="test-upload",
            file_ref="test-file",
            content_type="image/jpeg",
            sha256="abc123",
            width=800,
            height=600,
            exif=None,
            labels=None,
            tg_message_id=123,
            tg_chat_id=456,
            source="test",
        )

        # Update with vision results containing wave score
        vision_results = {
            "status": "ok",
            "provider": "gpt-4o-mini",
            "weather": {"sea": {"wave_score": 7.5, "confidence": 0.92}},
        }
        data.update_asset(asset_id, vision_results=vision_results)

        # Verify the columns were updated
        asset = data.get_asset(asset_id)
        assert asset is not None
        assert asset.vision_wave_score == 7.5
        assert asset.vision_wave_conf == 0.92

    def test_update_asset_parses_vision_sky_bucket(self, temp_db: sqlite3.Connection) -> None:
        """Test that update_asset parses and stores sky bucket from vision_results."""
        data = DataAccess(temp_db)

        asset_id = data.create_asset(
            upload_id="test-upload-2",
            file_ref="test-file-2",
            content_type="image/jpeg",
            sha256="def456",
            width=800,
            height=600,
            exif=None,
            labels=None,
            tg_message_id=124,
            tg_chat_id=457,
            source="test",
        )

        vision_results = {
            "status": "ok",
            "provider": "gpt-4o-mini",
            "weather": {"sky": {"bucket": "partly_cloudy"}},
        }
        data.update_asset(asset_id, vision_results=vision_results)

        asset = data.get_asset(asset_id)
        assert asset is not None
        assert asset.vision_sky_bucket == "partly_cloudy"

    def test_update_asset_parses_both_wave_and_sky(self, temp_db: sqlite3.Connection) -> None:
        """Test that update_asset parses both wave and sky metrics together."""
        data = DataAccess(temp_db)

        asset_id = data.create_asset(
            upload_id="test-upload-3",
            file_ref="test-file-3",
            content_type="image/jpeg",
            sha256="ghi789",
            width=800,
            height=600,
            exif=None,
            labels=None,
            tg_message_id=125,
            tg_chat_id=458,
            source="test",
        )

        vision_results = {
            "status": "ok",
            "provider": "gpt-4o-mini",
            "weather": {
                "sea": {"wave_score": 3.2, "confidence": 0.88},
                "sky": {"bucket": "overcast"},
            },
        }
        data.update_asset(asset_id, vision_results=vision_results)

        asset = data.get_asset(asset_id)
        assert asset is not None
        assert asset.vision_wave_score == 3.2
        assert asset.vision_wave_conf == 0.88
        assert asset.vision_sky_bucket == "overcast"

    def test_update_asset_with_legacy_sea_wave_score_dict(
        self, temp_db: sqlite3.Connection
    ) -> None:
        """Test parsing legacy sea_wave_score dict format."""
        data = DataAccess(temp_db)

        asset_id = data.create_asset(
            upload_id="test-upload-4",
            file_ref="test-file-4",
            content_type="image/jpeg",
            sha256="jkl012",
            width=800,
            height=600,
            exif=None,
            labels=None,
            tg_message_id=126,
            tg_chat_id=459,
            source="test",
        )

        vision_results = {
            "status": "ok",
            "sea_wave_score": {"value": 5.5, "confidence": 0.75},
        }
        data.update_asset(asset_id, vision_results=vision_results)

        asset = data.get_asset(asset_id)
        assert asset is not None
        assert asset.vision_wave_score == 5.5
        assert asset.vision_wave_conf == 0.75

    def test_update_asset_with_textual_wave_score(self, temp_db: sqlite3.Connection) -> None:
        """Test parsing wave score from textual result."""
        data = DataAccess(temp_db)

        asset_id = data.create_asset(
            upload_id="test-upload-5",
            file_ref="test-file-5",
            content_type="image/jpeg",
            sha256="mno345",
            width=800,
            height=600,
            exif=None,
            labels=None,
            tg_message_id=127,
            tg_chat_id=460,
            source="test",
        )

        vision_results = {
            "status": "ok",
            "result_text": "Погода: хорошая\nВолнение моря: 6.5/10 (conf=0.85)\nВидимость: отличная",
        }
        data.update_asset(asset_id, vision_results=vision_results)

        asset = data.get_asset(asset_id)
        assert asset is not None
        assert asset.vision_wave_score == 6.5
        assert asset.vision_wave_conf == 0.85

    def test_update_asset_no_vision_metrics(self, temp_db: sqlite3.Connection) -> None:
        """Test that update_asset handles vision results without wave/sky metrics gracefully."""
        data = DataAccess(temp_db)

        asset_id = data.create_asset(
            upload_id="test-upload-6",
            file_ref="test-file-6",
            content_type="image/jpeg",
            sha256="pqr678",
            width=800,
            height=600,
            exif=None,
            labels=None,
            tg_message_id=128,
            tg_chat_id=461,
            source="test",
        )

        vision_results = {
            "status": "ok",
            "provider": "gpt-4o-mini",
            "caption": "A beautiful landscape",
            "tags": ["nature", "landscape"],
        }
        data.update_asset(asset_id, vision_results=vision_results)

        # Should not crash, metrics should remain None
        asset = data.get_asset(asset_id)
        assert asset is not None
        assert asset.vision_wave_score is None
        assert asset.vision_wave_conf is None
        assert asset.vision_sky_bucket is None

    def test_update_asset_overwrites_existing_metrics(self, temp_db: sqlite3.Connection) -> None:
        """Test that update_asset overwrites existing vision metrics."""
        data = DataAccess(temp_db)

        asset_id = data.create_asset(
            upload_id="test-upload-7",
            file_ref="test-file-7",
            content_type="image/jpeg",
            sha256="stu901",
            width=800,
            height=600,
            exif=None,
            labels=None,
            tg_message_id=129,
            tg_chat_id=462,
            source="test",
        )

        # First update
        vision_results_1 = {
            "status": "ok",
            "weather": {"sea": {"wave_score": 2.0, "confidence": 0.5}},
        }
        data.update_asset(asset_id, vision_results=vision_results_1)

        asset = data.get_asset(asset_id)
        assert asset is not None
        assert asset.vision_wave_score == 2.0
        assert asset.vision_wave_conf == 0.5

        # Second update with different values
        vision_results_2 = {
            "status": "ok",
            "weather": {"sea": {"wave_score": 8.5, "confidence": 0.95}},
        }
        data.update_asset(asset_id, vision_results=vision_results_2)

        asset = data.get_asset(asset_id)
        assert asset is not None
        assert asset.vision_wave_score == 8.5
        assert asset.vision_wave_conf == 0.95

    def test_update_asset_preserves_manual_overrides(self, temp_db: sqlite3.Connection) -> None:
        """Test that manual column updates are preserved when passing vision_results."""
        data = DataAccess(temp_db)

        asset_id = data.create_asset(
            upload_id="test-upload-8",
            file_ref="test-file-8",
            content_type="image/jpeg",
            sha256="vwx234",
            width=800,
            height=600,
            exif=None,
            labels=None,
            tg_message_id=130,
            tg_chat_id=463,
            source="test",
        )

        # Manual update with explicit wave score
        data.update_asset(asset_id, vision_wave_score=9.0, vision_wave_conf=1.0)

        asset = data.get_asset(asset_id)
        assert asset is not None
        assert asset.vision_wave_score == 9.0
        assert asset.vision_wave_conf == 1.0

        # Now update with vision_results containing different values
        # The parsed values should overwrite the manual ones
        vision_results = {
            "status": "ok",
            "weather": {"sea": {"wave_score": 4.5, "confidence": 0.8}},
        }
        data.update_asset(asset_id, vision_results=vision_results)

        asset = data.get_asset(asset_id)
        assert asset is not None
        assert asset.vision_wave_score == 4.5
        assert asset.vision_wave_conf == 0.8
