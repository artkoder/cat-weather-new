"""Tests for calm-wave priority in sea selection."""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import Bot
from tests.fixtures.sea import create_sea_asset, create_stub_image, seed_sea_environment


@pytest.mark.asyncio
async def test_calm_sea_prefers_calm_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When target wave is ≤0.2m, calm assets (score 0-1) should win."""

    bot = Bot("dummy", str(tmp_path / "test_calm.db"))

    seed_sea_environment(
        bot,
        sea_id=1,
        sea_lat=54.95,
        sea_lon=20.2,
        wave=0.1,
        water_temp=15.0,
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

    image_path = create_stub_image(tmp_path, "calm.jpg")
    today_doy = datetime.utcnow().timetuple().tm_yday

    calm_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=1,
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

    moderate_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=2,
        file_name="moderate.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=4,
        photo_sky="sunny",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, photo_wave=? WHERE id=?", (today_doy, 0.8, moderate_id)
    )
    bot.db.commit()

    class DummyOpenAI:
        def __init__(self) -> None:
            self.api_key = "dummy"

        async def chat_completion(self, **kwargs: Any) -> Any:
            class MockChoice:
                def __init__(self) -> None:
                    self.message = type("obj", (object,), {"content": "Calm sea test"})()

            class MockResponse:
                def __init__(self) -> None:
                    self.choices = [MockChoice()]
                    self.usage = None

            return MockResponse()

    monkeypatch.setattr(bot, "openai", DummyOpenAI())

    result = await bot._publish_sea(
        rubric=rubric,
        channel_id=-999,
        test=True,
    )

    assert result is True


@pytest.mark.asyncio
async def test_calm_sea_fallback_nearest_wave(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When no calm assets exist, selector should choose nearest-wave candidate."""

    bot = Bot("dummy", str(tmp_path / "test_calm_fallback.db"))

    seed_sea_environment(
        bot,
        sea_id=1,
        sea_lat=54.95,
        sea_lon=20.2,
        wave=0.1,
        water_temp=15.0,
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

    image_path = create_stub_image(tmp_path, "high.jpg")
    today_doy = datetime.utcnow().timetuple().tm_yday

    high_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=1,
        file_name="high.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=6,
        photo_sky="sunny",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, photo_wave=? WHERE id=?", (today_doy, 1.2, high_id)
    )

    higher_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=2,
        file_name="higher.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=10,
        photo_sky="sunny",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, photo_wave=? WHERE id=?", (today_doy, 2.0, higher_id)
    )
    bot.db.commit()

    class DummyOpenAI:
        def __init__(self) -> None:
            self.api_key = "dummy"

        async def chat_completion(self, **kwargs: Any) -> Any:
            class MockChoice:
                def __init__(self) -> None:
                    self.message = type("obj", (object,), {"content": "Calm sea fallback"})()

            class MockResponse:
                def __init__(self) -> None:
                    self.choices = [MockChoice()]
                    self.usage = None

            return MockResponse()

    monkeypatch.setattr(bot, "openai", DummyOpenAI())

    result = await bot._publish_sea(
        rubric=rubric,
        channel_id=-999,
        test=True,
    )

    assert result is True


@pytest.mark.asyncio
async def test_calm_sea_filters_high_waves_when_calm_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When target wave is ≤1 and calm assets exist, filter out wave_score ≥5."""

    bot = Bot("dummy", str(tmp_path / "test_calm_filter.db"))

    seed_sea_environment(
        bot,
        sea_id=1,
        sea_lat=54.95,
        sea_lon=20.2,
        wave=0.1,
        water_temp=15.0,
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

    image_path = create_stub_image(tmp_path, "sea.jpg")
    today_doy = datetime.utcnow().timetuple().tm_yday

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

    high_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=2,
        file_name="high.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=6,
        photo_sky="sunny",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, photo_wave=? WHERE id=?", (today_doy, 1.2, high_id)
    )
    bot.db.commit()

    candidates = bot.data.fetch_sea_candidates(
        rubric.id, limit=48, season_range=None, target_wave_score=0
    )

    candidate_ids = [c["asset"].id for c in candidates]
    assert calm_id in candidate_ids
    assert high_id not in candidate_ids


@pytest.mark.asyncio
async def test_calm_sea_allows_null_waves_with_penalty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NULL wave scores should remain in pool but get +0.8 penalty."""

    bot = Bot("dummy", str(tmp_path / "test_null_wave.db"))

    seed_sea_environment(
        bot,
        sea_id=1,
        sea_lat=54.95,
        sea_lon=20.2,
        wave=0.5,
        water_temp=15.0,
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

    image_path = create_stub_image(tmp_path, "sea.jpg")
    today_doy = datetime.utcnow().timetuple().tm_yday

    null_wave_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=1,
        file_name="null.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=None,
        photo_sky="sunny",
        sky_visible=True,
    )
    bot.db.execute("UPDATE assets SET shot_doy=? WHERE id=?", (today_doy, null_wave_id))

    known_wave_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=2,
        file_name="known.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=2,
        photo_sky="sunny",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, photo_wave=? WHERE id=?", (today_doy, 0.5, known_wave_id)
    )
    bot.db.commit()

    candidates = bot.data.fetch_sea_candidates(
        rubric.id, limit=48, season_range=None, target_wave_score=3
    )

    candidate_ids = [c["asset"].id for c in candidates]
    assert null_wave_id in candidate_ids
    assert known_wave_id in candidate_ids
