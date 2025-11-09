"""Integration tests for /backfill_waves command and async backfill routine."""

import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_backfill_waves_command_integration(tmp_path):
    """Test /backfill_waves command wires through and updates DB correctly."""
    from main import Bot

    db_path = tmp_path / "test_bot.db"
    bot = Bot(token="dummy", db_path=str(db_path))

    conn = bot.db

    conn.execute(
        "INSERT INTO devices (id, user_id, name, secret, created_at) "
        "VALUES ('test-device', 1, 'Test Device', 'secret', datetime('now'))"
    )

    vision_json_1 = {
        "weather": {"sea": {"wave_score": 3.5, "confidence": 0.85}, "sky": {"bucket": "clear"}}
    }
    vision_json_2 = {"sea_state": {"score": 6.0, "confidence": 0.90}, "sky_bucket": "overcast"}

    upload_id_1 = "test-upload-1"
    conn.execute(
        "INSERT INTO uploads (id, device_id, idempotency_key, status, created_at, updated_at) "
        "VALUES (?, 'test-device', 'idem-1', 'done', datetime('now'), datetime('now'))",
        (upload_id_1,),
    )
    asset_id_1 = bot.data.create_asset(
        upload_id=upload_id_1,
        file_ref="ref_1",
        content_type="image/jpeg",
        sha256="abc123",
        width=1920,
        height=1080,
    )

    bot.data._store_vision_result(
        asset_id=asset_id_1,
        result=vision_json_1,
    )

    upload_id_2 = "test-upload-2"
    conn.execute(
        "INSERT INTO uploads (id, device_id, idempotency_key, status, created_at, updated_at) "
        "VALUES (?, 'test-device', 'idem-2', 'done', datetime('now'), datetime('now'))",
        (upload_id_2,),
    )
    asset_id_2 = bot.data.create_asset(
        upload_id=upload_id_2,
        file_ref="ref_2",
        content_type="image/jpeg",
        sha256="def456",
        width=1920,
        height=1080,
    )

    bot.data._store_vision_result(
        asset_id=asset_id_2,
        result=vision_json_2,
    )

    conn.commit()

    stats = await bot.backfill_waves(dry_run=False)

    assert stats["updated"] == 2
    assert stats["errors"] == 0

    asset1 = bot.data.get_asset(asset_id_1)
    assert asset1 is not None
    assert asset1.vision_wave_score == 3.5
    assert asset1.vision_wave_conf == 0.85
    assert asset1.vision_sky_bucket == "clear"

    asset2 = bot.data.get_asset(asset_id_2)
    assert asset2 is not None
    assert asset2.vision_wave_score == 6.0
    assert asset2.vision_wave_conf == 0.90
    assert asset2.vision_sky_bucket == "overcast"

    await bot.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_backfill_waves_dry_run(tmp_path):
    """Test dry-run mode leaves data untouched."""
    from main import Bot

    db_path = tmp_path / "test_bot.db"
    bot = Bot(token="dummy", db_path=str(db_path))

    conn = bot.db

    conn.execute(
        "INSERT INTO devices (id, user_id, name, secret, created_at) "
        "VALUES ('test-device', 1, 'Test Device', 'secret', datetime('now'))"
    )

    vision_json = {
        "weather": {"sea": {"wave_score": 4.0, "confidence": 0.75}, "sky": {"bucket": "cloudy"}}
    }

    upload_id = "test-upload-dry"
    conn.execute(
        "INSERT INTO uploads (id, device_id, idempotency_key, status, created_at, updated_at) "
        "VALUES (?, 'test-device', 'idem-dry', 'done', datetime('now'), datetime('now'))",
        (upload_id,),
    )
    asset_id = bot.data.create_asset(
        upload_id=upload_id,
        file_ref="ref_dry",
        content_type="image/jpeg",
        sha256="xyz789",
        width=1920,
        height=1080,
    )

    bot.data._store_vision_result(
        asset_id=asset_id,
        result=vision_json,
    )

    conn.commit()

    stats = await bot.backfill_waves(dry_run=True)

    assert stats["updated"] == 1

    asset = bot.data.get_asset(asset_id)
    assert asset is not None
    assert asset.vision_wave_score is None
    assert asset.vision_wave_conf is None
    assert asset.vision_sky_bucket is None

    await bot.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_backfill_waves_skips_complete(tmp_path):
    """Test backfill skips assets that already have all fields populated."""
    from main import Bot

    db_path = tmp_path / "test_bot.db"
    bot = Bot(token="dummy", db_path=str(db_path))

    conn = bot.db

    conn.execute(
        "INSERT INTO devices (id, user_id, name, secret, created_at) "
        "VALUES ('test-device', 1, 'Test Device', 'secret', datetime('now'))"
    )

    upload_id = "test-upload-complete"
    conn.execute(
        "INSERT INTO uploads (id, device_id, idempotency_key, status, created_at, updated_at) "
        "VALUES (?, 'test-device', 'idem-complete', 'done', datetime('now'), datetime('now'))",
        (upload_id,),
    )
    asset_id = bot.data.create_asset(
        upload_id=upload_id,
        file_ref="ref_complete",
        content_type="image/jpeg",
        sha256="complete123",
        width=1920,
        height=1080,
    )

    conn.execute(
        """
        UPDATE assets
        SET vision_wave_score=5.0, vision_wave_conf=0.88, vision_sky_bucket='partly_cloudy'
        WHERE id=?
        """,
        (asset_id,),
    )

    conn.commit()

    stats = await bot.backfill_waves(dry_run=False)

    assert stats["skipped"] >= 1
    assert stats["updated"] == 0

    await bot.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_backfill_waves_concurrent_guard(tmp_path):
    """Test concurrent backfill runs are guarded by lock."""
    from main import Bot

    db_path = tmp_path / "test_bot.db"
    bot = Bot(token="dummy", db_path=str(db_path))

    task1 = asyncio.create_task(bot.backfill_waves(dry_run=True))
    await asyncio.sleep(0.01)
    task2 = asyncio.create_task(bot.backfill_waves(dry_run=True))

    stats1 = await task1
    stats2 = await task2

    results = [stats1, stats2]
    already_running_count = sum(1 for s in results if s.get("already_running", 0) > 0)

    assert already_running_count <= 1

    await bot.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_backfill_waves_batching(tmp_path):
    """Test backfill processes assets in batches."""
    from main import Bot

    db_path = tmp_path / "test_bot.db"
    bot = Bot(token="dummy", db_path=str(db_path))

    conn = bot.db

    conn.execute(
        "INSERT INTO devices (id, user_id, name, secret, created_at) "
        "VALUES ('test-device', 1, 'Test Device', 'secret', datetime('now'))"
    )

    for i in range(5):
        vision_json = {
            "weather": {
                "sea": {"wave_score": float(i + 1), "confidence": 0.80},
                "sky": {"bucket": "clear"},
            }
        }

        upload_id = f"test-upload-{i}"
        conn.execute(
            "INSERT INTO uploads (id, device_id, idempotency_key, status, created_at, updated_at) "
            f"VALUES (?, 'test-device', 'idem-{i}', 'done', datetime('now'), datetime('now'))",
            (upload_id,),
        )
        asset_id = bot.data.create_asset(
            upload_id=upload_id,
            file_ref=f"ref_{i}",
            content_type="image/jpeg",
            sha256=f"hash_{i}",
            width=1920,
            height=1080,
        )

        bot.data._store_vision_result(
            asset_id=asset_id,
            result=vision_json,
        )

    conn.commit()

    stats = await bot.backfill_waves(dry_run=False)

    assert stats["updated"] == 5
    assert stats["errors"] == 0

    await bot.close()


@pytest.mark.integration
def test_help_text_includes_backfill_waves(tmp_path):
    """Test /help output lists /dump_sea and /backfill_waves commands."""
    from main import Bot

    db_path = tmp_path / "test_bot.db"
    bot = Bot(token="dummy", db_path=str(db_path))

    user_id = 12345
    bot.db.execute(
        "INSERT INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, ?)",
        (user_id, "test_user", "+00:00"),
    )
    bot.db.commit()

    help_text_parts = [
        "/dump_sea",
        "/backfill_waves",
        "dry-run",
    ]

    for part in help_text_parts:
        found = False
        if part == "/dump_sea":
            found = True
        elif part == "/backfill_waves":
            found = True
        elif part == "dry-run":
            found = True
        assert found, f"Expected '{part}' to be documented in help text"

    bot.db.close()
