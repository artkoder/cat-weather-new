"""Tests for backfill_wave_metrics migration function."""
import json
import os
import sqlite3
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_access import DataAccess
from weather_migration import backfill_wave_metrics


@pytest.mark.integration
def test_backfill_wave_metrics_from_vision(tmp_path):
    """Test backfilling vision wave/sky metrics from vision_results."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    
    conn.executescript("""
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
            updated_at TEXT NOT NULL,
            FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
        );
    """)
    
    vision_json_1 = {
        "weather": {
            "sea": {"wave_score": 2.5, "confidence": 0.90},
            "sky": {"bucket": "clear"}
        }
    }
    
    conn.execute(
        """
        INSERT INTO assets (
            id, created_at, source, payload_json
        ) VALUES (?, ?, ?, ?)
        """,
        ("asset1", "2024-01-15T10:00:00", "telegram", "{}")
    )
    
    conn.execute(
        """
        INSERT INTO vision_results (
            asset_id, provider, status, result_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "asset1",
            "openai",
            "success",
            json.dumps(vision_json_1),
            "2024-01-15T10:05:00",
            "2024-01-15T10:05:00"
        )
    )
    
    vision_json_2 = {
        "sea_state": {"score": 7.0, "confidence": 0.85},
        "sky_bucket": "overcast"
    }
    
    conn.execute(
        """
        INSERT INTO assets (
            id, created_at, source, payload_json
        ) VALUES (?, ?, ?, ?)
        """,
        ("asset2", "2024-01-16T10:00:00", "telegram", "{}")
    )
    
    conn.execute(
        """
        INSERT INTO vision_results (
            asset_id, provider, status, result_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "asset2",
            "openai",
            "success",
            json.dumps(vision_json_2),
            "2024-01-16T10:05:00",
            "2024-01-16T10:05:00"
        )
    )
    
    conn.execute(
        """
        INSERT INTO assets (
            id, created_at, source, payload_json,
            vision_wave_score, vision_wave_conf, vision_sky_bucket
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "asset3",
            "2024-01-17T10:00:00",
            "telegram",
            "{}",
            3.5,
            0.88,
            "partly_cloudy"
        )
    )
    
    conn.commit()
    
    stats = backfill_wave_metrics(conn, dry_run=False)
    
    assert stats["updated"] == 2
    assert stats["unchanged"] == 1
    
    data = DataAccess(conn)
    
    asset1 = data.get_asset("asset1")
    assert asset1 is not None
    assert asset1.vision_wave_score == 2.5
    assert asset1.vision_wave_conf == 0.90
    assert asset1.vision_sky_bucket == "clear"
    
    asset2 = data.get_asset("asset2")
    assert asset2 is not None
    assert asset2.vision_wave_score == 7.0
    assert asset2.vision_wave_conf == 0.85
    assert asset2.vision_sky_bucket == "overcast"
    
    asset3 = data.get_asset("asset3")
    assert asset3 is not None
    assert asset3.vision_wave_score == 3.5
    assert asset3.vision_wave_conf == 0.88
    assert asset3.vision_sky_bucket == "partly_cloudy"
    
    conn.close()


@pytest.mark.integration
def test_backfill_wave_metrics_dry_run(tmp_path):
    """Test backfill in dry-run mode doesn't modify data."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    
    conn.executescript("""
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
            updated_at TEXT NOT NULL,
            FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
        );
    """)
    
    vision_json = {
        "weather": {
            "sea": {"wave_score": 4.0, "confidence": 0.75},
            "sky": {"bucket": "cloudy"}
        }
    }
    
    conn.execute(
        """
        INSERT INTO assets (
            id, created_at, source, payload_json
        ) VALUES (?, ?, ?, ?)
        """,
        ("asset_dry", "2024-01-18T10:00:00", "telegram", "{}")
    )
    
    conn.execute(
        """
        INSERT INTO vision_results (
            asset_id, provider, status, result_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "asset_dry",
            "openai",
            "success",
            json.dumps(vision_json),
            "2024-01-18T10:05:00",
            "2024-01-18T10:05:00"
        )
    )
    
    conn.commit()
    
    stats = backfill_wave_metrics(conn, dry_run=True)
    
    assert stats["updated"] == 1
    
    data = DataAccess(conn)
    asset = data.get_asset("asset_dry")
    assert asset is not None
    assert asset.vision_wave_score is None
    assert asset.vision_wave_conf is None
    assert asset.vision_sky_bucket is None
    
    conn.close()


@pytest.mark.integration
def test_backfill_wave_metrics_partial_update(tmp_path):
    """Test backfill only updates missing fields."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    
    conn.executescript("""
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
            updated_at TEXT NOT NULL,
            FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
        );
    """)
    
    vision_json = {
        "weather": {
            "sea": {"wave_score": 5.5, "confidence": 0.80},
            "sky": {"bucket": "partly_cloudy"}
        }
    }
    
    conn.execute(
        """
        INSERT INTO assets (
            id, created_at, source, payload_json, vision_wave_score
        ) VALUES (?, ?, ?, ?, ?)
        """,
        ("asset_partial", "2024-01-19T10:00:00", "telegram", "{}", 5.5)
    )
    
    conn.execute(
        """
        INSERT INTO vision_results (
            asset_id, provider, status, result_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "asset_partial",
            "openai",
            "success",
            json.dumps(vision_json),
            "2024-01-19T10:05:00",
            "2024-01-19T10:05:00"
        )
    )
    
    conn.commit()
    
    stats = backfill_wave_metrics(conn, dry_run=False)
    
    assert stats["updated"] == 1
    
    data = DataAccess(conn)
    asset = data.get_asset("asset_partial")
    assert asset is not None
    assert asset.vision_wave_score == 5.5
    assert asset.vision_wave_conf == 0.80
    assert asset.vision_sky_bucket == "partly_cloudy"
    
    conn.close()
