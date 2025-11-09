"""Integration tests for sea assets CSV dump."""
import csv
import json
import os
import sqlite3
import sys
from io import StringIO

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_access import DataAccess


@pytest.mark.integration
def test_dump_sea_assets_csv_basic(tmp_path):
    """Test basic CSV dump with various data scenarios."""
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
    
    asset1_payload = {
        "city": "Калининград",
        "latitude": 54.71,
        "longitude": 20.51,
        "vision_confidence": 0.95,
        "vision_photo_weather": "sunny",
        "local_path": "/data/images/asset1.jpg",
        "last_used_at": "2024-01-15T10:00:00"
    }
    
    vision1_json = {
        "weather": {
            "sea": {
                "wave_score": 3.5,
                "confidence": 0.88
            },
            "sky": {
                "bucket": "partly_cloudy"
            }
        }
    }
    
    conn.execute(
        """
        INSERT INTO assets (
            id, upload_id, file_ref, content_type, created_at, source,
            payload_json, exif_json, doy, daypart,
            vision_wave_score, vision_wave_conf, vision_sky_bucket
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "asset1",
            None,
            "file123",
            "image/jpeg",
            "2024-01-15T08:00:00",
            "telegram",
            json.dumps(asset1_payload),
            '{"Make": "Canon"}',
            15,
            "morning",
            3.5,
            0.88,
            "partly_cloudy"
        )
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
            json.dumps(vision1_json),
            "2024-01-15T08:05:00",
            "2024-01-15T08:05:00"
        )
    )
    
    asset2_payload = {
        "city": "Зеленоградск",
        "latitude": 54.96,
        "longitude": 20.48,
        "vision_confidence": 0.82
    }
    
    conn.execute(
        """
        INSERT INTO assets (
            id, upload_id, file_ref, content_type, created_at, source,
            payload_json, photo_wave, doy, daypart
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "asset2",
            None,
            "file456",
            "image/jpeg",
            "2024-01-16T14:00:00",
            "telegram",
            json.dumps(asset2_payload),
            5.0,
            16,
            "day"
        )
    )
    
    conn.commit()
    
    data = DataAccess(conn)
    csv_content = data.dump_sea_assets_csv()
    
    assert csv_content is not None
    assert len(csv_content) > 0
    
    reader = csv.DictReader(StringIO(csv_content))
    rows = list(reader)
    
    assert len(rows) == 2
    
    headers = reader.fieldnames
    assert headers is not None
    expected_headers = [
        "asset_id",
        "created_at",
        "uses_count",
        "city",
        "lat",
        "lon",
        "vision_confidence",
        "vision_photo_weather",
        "sky_bucket",
        "daypart",
        "wave_score_0_10",
        "wave_conf",
        "exif_present",
        "doy",
        "local_path",
        "vision_json"
    ]
    assert list(headers) == expected_headers
    
    row1 = rows[0]
    assert row1["asset_id"] == "asset2"
    assert row1["city"] == "Зеленоградск"
    assert row1["lat"] == "54.96"
    assert row1["lon"] == "20.48"
    assert row1["vision_confidence"] == "0.82"
    assert row1["daypart"] == "day"
    assert row1["wave_score_0_10"] == "5.0"
    assert row1["wave_conf"] == ""
    assert row1["exif_present"] == "0"
    assert row1["doy"] == "16"
    assert row1["uses_count"] == "0"
    
    row2 = rows[1]
    assert row2["asset_id"] == "asset1"
    assert row2["city"] == "Калининград"
    assert row2["lat"] == "54.71"
    assert row2["lon"] == "20.51"
    assert row2["vision_confidence"] == "0.95"
    assert row2["vision_photo_weather"] == "sunny"
    assert row2["sky_bucket"] == "partly_cloudy"
    assert row2["daypart"] == "morning"
    assert row2["wave_score_0_10"] == "3.5"
    assert row2["wave_conf"] == "0.88"
    assert row2["exif_present"] == "1"
    assert row2["doy"] == "15"
    assert row2["local_path"] == "/data/images/asset1.jpg"
    assert row2["uses_count"] == "1"
    
    vision_json_parsed = json.loads(row2["vision_json"])
    assert vision_json_parsed["weather"]["sea"]["wave_score"] == 3.5
    
    conn.close()


@pytest.mark.integration
def test_dump_sea_assets_csv_fallback_to_legacy(tmp_path):
    """Test CSV dump falls back to legacy photo_wave when vision columns are empty."""
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
    
    asset_payload = {"city": "Test City"}
    vision_json = {
        "sea_wave_score": 6.5,
        "sky_bucket": "clear"
    }
    
    conn.execute(
        """
        INSERT INTO assets (
            id, upload_id, file_ref, content_type, created_at, source,
            payload_json, photo_wave, doy
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "asset_legacy",
            None,
            "file789",
            "image/jpeg",
            "2024-01-17T12:00:00",
            "telegram",
            json.dumps(asset_payload),
            4.2,
            17
        )
    )
    
    conn.execute(
        """
        INSERT INTO vision_results (
            asset_id, provider, status, result_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "asset_legacy",
            "openai",
            "success",
            json.dumps(vision_json),
            "2024-01-17T12:05:00",
            "2024-01-17T12:05:00"
        )
    )
    
    conn.commit()
    
    data = DataAccess(conn)
    csv_content = data.dump_sea_assets_csv()
    
    reader = csv.DictReader(StringIO(csv_content))
    rows = list(reader)
    
    assert len(rows) == 1
    row = rows[0]
    
    assert row["asset_id"] == "asset_legacy"
    assert row["wave_score_0_10"] == "6.5"
    assert row["sky_bucket"] == "clear"
    
    conn.close()


@pytest.mark.integration
def test_dump_sea_assets_csv_empty_db(tmp_path):
    """Test CSV dump with no assets returns only headers."""
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
    
    conn.commit()
    
    data = DataAccess(conn)
    csv_content = data.dump_sea_assets_csv()
    
    lines = csv_content.strip().split("\n")
    assert len(lines) == 1
    assert "asset_id" in lines[0]
    assert "wave_score_0_10" in lines[0]
    
    conn.close()
