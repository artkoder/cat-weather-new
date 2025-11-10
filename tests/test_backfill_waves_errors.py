"""Tests for backfill_waves error handling and edge cases."""

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_backfill_waves_handles_invalid_json(tmp_path):
    """Test backfill gracefully handles assets with invalid JSON in vision_results."""
    from main import Bot

    db_path = tmp_path / "test_bot.db"
    bot = Bot(token="dummy", db_path=str(db_path))

    conn = bot.db

    conn.execute(
        "INSERT INTO devices (id, user_id, name, secret, created_at) "
        "VALUES ('test-device', 1, 'Test Device', 'secret', datetime('now'))"
    )

    upload_id = "test-upload-invalid"
    conn.execute(
        "INSERT INTO uploads (id, device_id, idempotency_key, status, created_at, updated_at) "
        "VALUES (?, 'test-device', 'idem-invalid', 'done', datetime('now'), datetime('now'))",
        (upload_id,),
    )
    asset_id = bot.data.create_asset(
        upload_id=upload_id,
        file_ref="ref_invalid",
        content_type="image/jpeg",
        sha256="invalid123",
        width=1920,
        height=1080,
    )

    conn.execute(
        "INSERT INTO vision_results (asset_id, result_json) VALUES (?, ?)",
        (asset_id, "not valid json {{{"),
    )

    conn.commit()

    stats = await bot.backfill_waves(dry_run=False)

    assert stats["skipped"] >= 1
    assert stats["errors"] == 0

    await bot.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_backfill_waves_handles_missing_vision_results(tmp_path):
    """Test backfill gracefully skips assets without vision_results."""
    from main import Bot

    db_path = tmp_path / "test_bot.db"
    bot = Bot(token="dummy", db_path=str(db_path))

    conn = bot.db

    conn.execute(
        "INSERT INTO devices (id, user_id, name, secret, created_at) "
        "VALUES ('test-device', 1, 'Test Device', 'secret', datetime('now'))"
    )

    upload_id = "test-upload-no-vision"
    conn.execute(
        "INSERT INTO uploads (id, device_id, idempotency_key, status, created_at, updated_at) "
        "VALUES (?, 'test-device', 'idem-no-vision', 'done', datetime('now'), datetime('now'))",
        (upload_id,),
    )
    asset_id = bot.data.create_asset(
        upload_id=upload_id,
        file_ref="ref_no_vision",
        content_type="image/jpeg",
        sha256="novision123",
        width=1920,
        height=1080,
    )

    conn.commit()

    stats = await bot.backfill_waves(dry_run=False)

    assert stats["skipped"] >= 1
    assert stats["errors"] == 0

    asset = bot.data.get_asset(asset_id)
    assert asset is not None
    assert asset.vision_wave_score is None
    assert asset.vision_wave_conf is None
    assert asset.vision_sky_bucket is None

    await bot.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_backfill_waves_handles_empty_vision_results(tmp_path):
    """Test backfill gracefully skips assets with empty vision_results dict."""
    from main import Bot

    db_path = tmp_path / "test_bot.db"
    bot = Bot(token="dummy", db_path=str(db_path))

    conn = bot.db

    conn.execute(
        "INSERT INTO devices (id, user_id, name, secret, created_at) "
        "VALUES ('test-device', 1, 'Test Device', 'secret', datetime('now'))"
    )

    upload_id = "test-upload-empty"
    conn.execute(
        "INSERT INTO uploads (id, device_id, idempotency_key, status, created_at, updated_at) "
        "VALUES (?, 'test-device', 'idem-empty', 'done', datetime('now'), datetime('now'))",
        (upload_id,),
    )
    asset_id = bot.data.create_asset(
        upload_id=upload_id,
        file_ref="ref_empty",
        content_type="image/jpeg",
        sha256="empty123",
        width=1920,
        height=1080,
    )

    bot.data._store_vision_result(asset_id=asset_id, result={})

    conn.commit()

    stats = await bot.backfill_waves(dry_run=False)

    assert stats["skipped"] >= 1
    assert stats["errors"] == 0

    await bot.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_backfill_waves_handles_malformed_fields(tmp_path):
    """Test backfill handles various malformed field types gracefully."""
    from main import Bot

    db_path = tmp_path / "test_bot.db"
    bot = Bot(token="dummy", db_path=str(db_path))

    conn = bot.db

    conn.execute(
        "INSERT INTO devices (id, user_id, name, secret, created_at) "
        "VALUES ('test-device', 1, 'Test Device', 'secret', datetime('now'))"
    )

    test_cases = [
        {
            "name": "string_score",
            "vision": {"weather": {"sea": {"wave_score": "3.5", "confidence": "0.85"}}},
            "expect_wave": 3.5,
            "expect_conf": 0.85,
        },
        {
            "name": "invalid_string",
            "vision": {"sea_wave_score": {"value": "not_a_number", "confidence": "bad"}},
            "expect_wave": None,
            "expect_conf": None,
        },
        {
            "name": "mixed_types",
            "vision": {
                "weather": {"sea": {"wave_score": 5, "confidence": "0.9"}},
                "sky_bucket": "clear",
            },
            "expect_wave": 5.0,
            "expect_conf": 0.9,
            "expect_sky": "clear",
        },
        {
            "name": "partial_data",
            "vision": {"weather": {"sea": {"wave_score": 4.2}}, "sky": {"bucket": "overcast"}},
            "expect_wave": 4.2,
            "expect_conf": None,
            "expect_sky": "overcast",
        },
        {
            "name": "only_sky",
            "vision": {"sky_bucket": "partly_cloudy"},
            "expect_wave": None,
            "expect_conf": None,
            "expect_sky": "partly_cloudy",
        },
    ]

    asset_ids = []
    for test_case in test_cases:
        upload_id = f"test-upload-{test_case['name']}"
        conn.execute(
            "INSERT INTO uploads (id, device_id, idempotency_key, status, created_at, updated_at) "
            f"VALUES (?, 'test-device', 'idem-{test_case['name']}', 'done', datetime('now'), datetime('now'))",
            (upload_id,),
        )
        asset_id = bot.data.create_asset(
            upload_id=upload_id,
            file_ref=f"ref_{test_case['name']}",
            content_type="image/jpeg",
            sha256=f"hash_{test_case['name']}",
            width=1920,
            height=1080,
        )
        bot.data._store_vision_result(asset_id=asset_id, result=test_case["vision"])
        asset_ids.append((asset_id, test_case))

    conn.commit()

    stats = await bot.backfill_waves(dry_run=False)

    assert stats["errors"] == 0

    for asset_id, test_case in asset_ids:
        asset = bot.data.get_asset(asset_id)
        assert asset is not None

        if test_case.get("expect_wave") is not None:
            assert asset.vision_wave_score == test_case["expect_wave"]
        else:
            assert asset.vision_wave_score is None

        if test_case.get("expect_conf") is not None:
            assert asset.vision_wave_conf == test_case["expect_conf"]
        else:
            assert asset.vision_wave_conf is None

        if test_case.get("expect_sky") is not None:
            assert asset.vision_sky_bucket == test_case["expect_sky"]
        else:
            assert asset.vision_sky_bucket is None

    await bot.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_backfill_waves_idempotency(tmp_path):
    """Test backfill is idempotent and doesn't overwrite existing data."""
    from main import Bot

    db_path = tmp_path / "test_bot.db"
    bot = Bot(token="dummy", db_path=str(db_path))

    conn = bot.db

    conn.execute(
        "INSERT INTO devices (id, user_id, name, secret, created_at) "
        "VALUES ('test-device', 1, 'Test Device', 'secret', datetime('now'))"
    )

    vision_json = {
        "weather": {"sea": {"wave_score": 5.0, "confidence": 0.85}, "sky": {"bucket": "clear"}}
    }

    upload_id = "test-upload-idem"
    conn.execute(
        "INSERT INTO uploads (id, device_id, idempotency_key, status, created_at, updated_at) "
        "VALUES (?, 'test-device', 'idem-idem', 'done', datetime('now'), datetime('now'))",
        (upload_id,),
    )
    asset_id = bot.data.create_asset(
        upload_id=upload_id,
        file_ref="ref_idem",
        content_type="image/jpeg",
        sha256="idem123",
        width=1920,
        height=1080,
    )

    bot.data._store_vision_result(asset_id=asset_id, result=vision_json)

    conn.commit()

    stats1 = await bot.backfill_waves(dry_run=False)
    assert stats1["updated"] == 1
    assert stats1["errors"] == 0

    asset = bot.data.get_asset(asset_id)
    assert asset is not None
    assert asset.vision_wave_score == 5.0
    assert asset.vision_wave_conf == 0.85
    assert asset.vision_sky_bucket == "clear"

    stats2 = await bot.backfill_waves(dry_run=False)
    assert stats2["skipped"] >= 1
    assert stats2["updated"] == 0
    assert stats2["errors"] == 0

    asset = bot.data.get_asset(asset_id)
    assert asset is not None
    assert asset.vision_wave_score == 5.0
    assert asset.vision_wave_conf == 0.85
    assert asset.vision_sky_bucket == "clear"

    await bot.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_backfill_waves_partial_updates(tmp_path):
    """Test backfill only updates missing fields, preserving existing ones."""
    from main import Bot

    db_path = tmp_path / "test_bot.db"
    bot = Bot(token="dummy", db_path=str(db_path))

    conn = bot.db

    conn.execute(
        "INSERT INTO devices (id, user_id, name, secret, created_at) "
        "VALUES ('test-device', 1, 'Test Device', 'secret', datetime('now'))"
    )

    vision_json = {
        "weather": {"sea": {"wave_score": 7.5, "confidence": 0.92}, "sky": {"bucket": "overcast"}}
    }

    upload_id = "test-upload-partial"
    conn.execute(
        "INSERT INTO uploads (id, device_id, idempotency_key, status, created_at, updated_at) "
        "VALUES (?, 'test-device', 'idem-partial', 'done', datetime('now'), datetime('now'))",
        (upload_id,),
    )
    asset_id = bot.data.create_asset(
        upload_id=upload_id,
        file_ref="ref_partial",
        content_type="image/jpeg",
        sha256="partial123",
        width=1920,
        height=1080,
    )

    conn.execute(
        """
        UPDATE assets
        SET vision_wave_score=2.5
        WHERE id=?
        """,
        (asset_id,),
    )

    bot.data._store_vision_result(asset_id=asset_id, result=vision_json)

    conn.commit()

    stats = await bot.backfill_waves(dry_run=False)

    asset = bot.data.get_asset(asset_id)
    assert asset is not None
    assert asset.vision_wave_score == 2.5
    assert asset.vision_wave_conf == 0.92
    assert asset.vision_sky_bucket == "overcast"

    assert stats["errors"] == 0

    await bot.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_backfill_waves_text_parsing(tmp_path):
    """Test backfill handles text-based vision results."""
    from main import Bot

    db_path = tmp_path / "test_bot.db"
    bot = Bot(token="dummy", db_path=str(db_path))

    conn = bot.db

    conn.execute(
        "INSERT INTO devices (id, user_id, name, secret, created_at) "
        "VALUES ('test-device', 1, 'Test Device', 'secret', datetime('now'))"
    )

    vision_json = {
        "result_text": "Погода: хорошая\nВолнение моря: 8.5/10 (conf=0.95)\nВидимость: отличная"
    }

    upload_id = "test-upload-text"
    conn.execute(
        "INSERT INTO uploads (id, device_id, idempotency_key, status, created_at, updated_at) "
        "VALUES (?, 'test-device', 'idem-text', 'done', datetime('now'), datetime('now'))",
        (upload_id,),
    )
    asset_id = bot.data.create_asset(
        upload_id=upload_id,
        file_ref="ref_text",
        content_type="image/jpeg",
        sha256="text123",
        width=1920,
        height=1080,
    )

    bot.data._store_vision_result(asset_id=asset_id, result=vision_json)

    conn.commit()

    stats = await bot.backfill_waves(dry_run=False)

    assert stats["updated"] == 1
    assert stats["errors"] == 0

    asset = bot.data.get_asset(asset_id)
    assert asset is not None
    assert asset.vision_wave_score == 8.5
    assert asset.vision_wave_conf == 0.95
    assert asset.vision_sky_bucket is None

    await bot.close()
