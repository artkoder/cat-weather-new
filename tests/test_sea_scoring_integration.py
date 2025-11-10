"""Integration tests for sea scoring with guard rules and logging."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import Bot
from tests.fixtures.sea import create_sea_asset, create_stub_image, seed_sea_environment


class CapturingOpenAI:
    def __init__(self) -> None:
        self.api_key = "dummy"
        self.calls: list[dict[str, Any]] = []

    async def chat_completion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)

        class MockChoice:
            def __init__(self) -> None:
                self.message = type("obj", (object,), {"content": "Calm sea test caption"})()

        class MockResponse:
            def __init__(self) -> None:
                self.choices = [MockChoice()]
                self.usage = None

        return MockResponse()


@pytest.mark.asyncio
async def test_calm_seas_integration_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """Integration test: calm conditions should select calm assets over stormy ones."""

    caplog.set_level(logging.INFO)

    bot = Bot("dummy", str(tmp_path / "test_calm_integration.db"))

    # Set calm conditions: wave = 0.1m (wave_score = 0)
    seed_sea_environment(
        bot,
        sea_id=1,
        sea_lat=54.95,
        sea_lon=20.2,
        wave=0.1,
        water_temp=15.0,
        city_id=101,
        city_name="TestCity",
        wind_speed=3.0,
        cloud_cover=15.0,
    )

    rubric = bot.data.get_rubric_by_code("sea")
    assert rubric is not None
    updated_config = dict(rubric.config or {})
    updated_config.update(
        {
            "enabled": True,
            "channel_id": -999,
            "test_channel_id": -999,
            "sea_id": 1,
        }
    )
    bot.data.save_rubric_config("sea", updated_config)

    image_path = create_stub_image(tmp_path, "test.jpg")
    today_doy = datetime.utcnow().timetuple().tm_yday

    # Create calm asset with wave_score=0
    calm_0_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=1,
        file_name="calm_0.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=0,
        photo_sky="sunny",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, photo_wave=? WHERE id=?", (today_doy, 0.0, calm_0_id)
    )

    # Create calm asset with wave_score=1
    calm_1_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=2,
        file_name="calm_1.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=1,
        photo_sky="sunny",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, photo_wave=? WHERE id=?", (today_doy, 0.2, calm_1_id)
    )

    # Create stormy asset with wave_score=6
    stormy_6_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=3,
        file_name="stormy_6.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=6,
        photo_sky="mostly_cloudy",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, photo_wave=? WHERE id=?", (today_doy, 1.2, stormy_6_id)
    )

    # Create very stormy asset with wave_score=8
    stormy_8_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=4,
        file_name="stormy_8.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=8,
        photo_sky="overcast",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, photo_wave=? WHERE id=?", (today_doy, 1.6, stormy_8_id)
    )

    # Create asset with NULL wave_score
    null_wave_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=5,
        file_name="null_wave.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=None,
        photo_sky="sunny",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, photo_wave=? WHERE id=?", (today_doy, None, null_wave_id)
    )

    bot.db.commit()

    capturing_openai = CapturingOpenAI()
    monkeypatch.setattr(bot, "openai", capturing_openai)

    result = await bot._publish_sea(
        rubric=rubric,
        channel_id=-999,
        test=True,
    )

    assert result is True

    # Parse logs
    sea_logs = [rec for rec in caplog.records if "SEA_RUBRIC" in rec.message]
    assert len(sea_logs) > 0, "Should have SEA_RUBRIC logs"

    # Check calm_guard log exists
    calm_guard_logs = [rec for rec in sea_logs if "calm_guard" in rec.message]
    assert len(calm_guard_logs) > 0, "Should have calm_guard log"

    # Verify calm guard filtered stormy assets
    calm_guard_log = calm_guard_logs[0].message
    assert "active=True" in calm_guard_log
    # Check that stormy assets were filtered (IDs should be in the log)
    assert str(stormy_6_id) in calm_guard_log or str(stormy_8_id) in calm_guard_log

    # Check top5 logs for wave metrics
    top5_logs = [rec for rec in sea_logs if "top5:" in rec.message]
    assert len(top5_logs) > 0, "Should have top5 logs"

    for top5_log in top5_logs:
        msg = top5_log.message
        # Check required fields (note: wave_photo and delta may be omitted if None)
        assert "wave_target=" in msg, f"top5 log should contain wave_target: {msg}"
        assert "penalties=" in msg, f"top5 log should contain penalties: {msg}"
        assert "total_score=" in msg, f"top5 log should contain total_score: {msg}"
        assert "rank=" in msg, f"top5 log should contain rank: {msg}"

        # Verify wave_target is low (0 or 1 for calm conditions)
        if "wave_target=0" in msg or "wave_target=1" in msg:
            # For calm targets, wave_photo should be low if present
            if "wave_photo=0" in msg or "wave_photo=1" in msg or "wave_photo=2" in msg:
                # Delta should be <= 2 for reasonable matches
                import re

                delta_match = re.search(r"delta=(\d+(?:\.\d+)?)", msg)
                if delta_match:
                    delta_value = float(delta_match.group(1))
                    assert delta_value <= 2.0, f"Delta should be <= 2 for calm match: {delta_value}"

    # Check selected log
    selected_logs = [rec for rec in sea_logs if "SEA_RUBRIC selected" in rec.message]
    assert len(selected_logs) == 1, "Should have exactly one selected log"

    selected_log = selected_logs[0].message
    assert "wave_target=" in selected_log
    assert "total_score=" in selected_log
    assert "reason=" in selected_log
    # Note: wave_photo and delta may be omitted if None

    # Verify selected asset is calm (wave_score 0 or 1)
    import re

    asset_id_match = re.search(r"asset_id=([a-f0-9\-]+)", selected_log)
    assert asset_id_match, "Should have asset_id in selected log"
    selected_asset_id = asset_id_match.group(1)
    calm_ids = [str(calm_0_id), str(calm_1_id), str(null_wave_id)]
    assert (
        selected_asset_id in calm_ids
    ), f"Selected asset should be calm (one of {calm_ids}), got {selected_asset_id}"


@pytest.mark.asyncio
async def test_storm_conditions_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """Regression test: storm conditions should work as before."""

    caplog.set_level(logging.INFO)

    bot = Bot("dummy", str(tmp_path / "test_storm_regression.db"))

    # Set storm conditions: wave = 1.6m (wave_score = 8)
    seed_sea_environment(
        bot,
        sea_id=1,
        sea_lat=54.95,
        sea_lon=20.2,
        wave=1.6,
        water_temp=12.0,
        city_id=101,
        city_name="TestCity",
        wind_speed=12.0,
        cloud_cover=70.0,
    )

    rubric = bot.data.get_rubric_by_code("sea")
    assert rubric is not None
    updated_config = dict(rubric.config or {})
    updated_config.update(
        {
            "enabled": True,
            "channel_id": -999,
            "test_channel_id": -999,
            "sea_id": 1,
        }
    )
    bot.data.save_rubric_config("sea", updated_config)

    image_path = create_stub_image(tmp_path, "test.jpg")
    today_doy = datetime.utcnow().timetuple().tm_yday

    # Create stormy asset with wave_score=8
    stormy_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=1,
        file_name="stormy.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=8,
        photo_sky="mostly_cloudy",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, photo_wave=? WHERE id=?", (today_doy, 1.6, stormy_id)
    )

    # Create calm asset (should be deprioritized in storm conditions)
    calm_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=2,
        file_name="calm.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=0,
        photo_sky="sunny",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, photo_wave=? WHERE id=?", (today_doy, 0.0, calm_id)
    )

    bot.db.commit()

    capturing_openai = CapturingOpenAI()
    monkeypatch.setattr(bot, "openai", capturing_openai)

    result = await bot._publish_sea(
        rubric=rubric,
        channel_id=-999,
        test=True,
    )

    assert result is True

    # Parse logs
    sea_logs = [rec for rec in caplog.records if "SEA_RUBRIC" in rec.message]

    # Check weather log shows storm conditions
    weather_logs = [rec for rec in sea_logs if "SEA_RUBRIC weather" in rec.message]
    assert len(weather_logs) > 0
    weather_log = weather_logs[0].message
    assert "wave_target_score=8" in weather_log or "wave_target_score=7" in weather_log

    # Check that calm_guard is NOT active for storm conditions
    calm_guard_logs = [rec for rec in sea_logs if "calm_guard" in rec.message]
    # Should be empty or not active
    if calm_guard_logs:
        for log in calm_guard_logs:
            assert "active=False" in log.message or "active=True" not in log.message

    # Verify selection works
    selected_logs = [rec for rec in sea_logs if "SEA_RUBRIC selected" in rec.message]
    assert len(selected_logs) == 1


@pytest.mark.asyncio
async def test_mixed_pool_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """Regression test: mixed pool with NULL wave_score should work without crash."""

    caplog.set_level(logging.INFO)

    bot = Bot("dummy", str(tmp_path / "test_mixed_pool.db"))

    # Set calm conditions
    seed_sea_environment(
        bot,
        sea_id=1,
        sea_lat=54.95,
        sea_lon=20.2,
        wave=0.2,
        water_temp=14.0,
        city_id=101,
        city_name="TestCity",
        wind_speed=5.0,
        cloud_cover=20.0,
    )

    rubric = bot.data.get_rubric_by_code("sea")
    assert rubric is not None
    updated_config = dict(rubric.config or {})
    updated_config.update(
        {
            "enabled": True,
            "channel_id": -999,
            "test_channel_id": -999,
            "sea_id": 1,
        }
    )
    bot.data.save_rubric_config("sea", updated_config)

    image_path = create_stub_image(tmp_path, "test.jpg")
    today_doy = datetime.utcnow().timetuple().tm_yday

    # Create calm asset
    calm_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=1,
        file_name="calm.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=1,
        photo_sky="sunny",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, photo_wave=? WHERE id=?", (today_doy, 0.2, calm_id)
    )

    # Create multiple assets with NULL wave_score (old photos)
    for i in range(3):
        null_id = create_sea_asset(
            bot,
            rubric_id=rubric.id,
            message_id=10 + i,
            file_name=f"old_{i}.jpg",
            local_path=image_path,
            tags=["sea"],
            sea_wave_score=None,
            photo_sky="partly_cloudy",
            sky_visible=True,
        )
        bot.db.execute(
            "UPDATE assets SET shot_doy=?, photo_wave=? WHERE id=?", (today_doy, None, null_id)
        )

    bot.db.commit()

    capturing_openai = CapturingOpenAI()
    monkeypatch.setattr(bot, "openai", capturing_openai)

    # Should not crash with mixed pool
    result = await bot._publish_sea(
        rubric=rubric,
        channel_id=-999,
        test=True,
    )

    assert result is True

    # Verify logs are structured
    sea_logs = [rec for rec in caplog.records if "SEA_RUBRIC" in rec.message]
    assert len(sea_logs) > 0

    selected_logs = [rec for rec in sea_logs if "SEA_RUBRIC selected" in rec.message]
    assert len(selected_logs) == 1


@pytest.mark.asyncio
async def test_seasonal_filter_with_guard_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """Regression test: seasonal filter and guard rules should work together."""

    caplog.set_level(logging.INFO)

    bot = Bot("dummy", str(tmp_path / "test_seasonal_guard.db"))

    # Set calm conditions
    seed_sea_environment(
        bot,
        sea_id=1,
        sea_lat=54.95,
        sea_lon=20.2,
        wave=0.1,
        water_temp=15.0,
        city_id=101,
        city_name="TestCity",
        wind_speed=4.0,
        cloud_cover=10.0,
    )

    rubric = bot.data.get_rubric_by_code("sea")
    assert rubric is not None
    updated_config = dict(rubric.config or {})
    updated_config.update(
        {
            "enabled": True,
            "channel_id": -999,
            "test_channel_id": -999,
            "sea_id": 1,
        }
    )
    bot.data.save_rubric_config("sea", updated_config)

    image_path = create_stub_image(tmp_path, "test.jpg")
    today_doy = datetime.utcnow().timetuple().tm_yday

    # Create in-season calm asset
    in_season_calm_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=1,
        file_name="in_season_calm.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=0,
        photo_sky="sunny",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, photo_wave=? WHERE id=?",
        (today_doy, 0.0, in_season_calm_id),
    )

    # Create out-of-season stormy asset (should be filtered by season, not guard)
    out_season_doy = (today_doy + 100) % 365 + 1
    out_season_stormy_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=2,
        file_name="out_season_stormy.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=7,
        photo_sky="overcast",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, photo_wave=? WHERE id=?",
        (out_season_doy, 1.4, out_season_stormy_id),
    )

    # Create in-season stormy asset (should be filtered by calm guard)
    in_season_stormy_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=3,
        file_name="in_season_stormy.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=6,
        photo_sky="mostly_cloudy",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, photo_wave=? WHERE id=?",
        (today_doy, 1.2, in_season_stormy_id),
    )

    bot.db.commit()

    capturing_openai = CapturingOpenAI()
    monkeypatch.setattr(bot, "openai", capturing_openai)

    result = await bot._publish_sea(
        rubric=rubric,
        channel_id=-999,
        test=True,
    )

    assert result is True

    # Parse logs
    sea_logs = [rec for rec in caplog.records if "SEA_RUBRIC" in rec.message]

    # Check season filter was applied
    season_logs = [rec for rec in sea_logs if "SEA_RUBRIC season" in rec.message]
    assert len(season_logs) > 0

    # Check calm guard was applied
    calm_guard_logs = [rec for rec in sea_logs if "calm_guard" in rec.message]
    assert len(calm_guard_logs) > 0

    # Verify in-season calm asset was selected
    selected_logs = [rec for rec in sea_logs if "SEA_RUBRIC selected" in rec.message]
    assert len(selected_logs) == 1
    selected_log = selected_logs[0].message
    assert f"asset_id={in_season_calm_id}" in selected_log
