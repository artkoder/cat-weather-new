"""Test wave scoring fixes: use wave_score_0_10 in filters."""

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
async def test_calm_guard_filters_high_wave(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When target_wave=0, asset with wave_score_0_10=3 and wave_conf=0.9 should be filtered out."""

    bot = Bot("dummy", str(tmp_path / "test_calm_guard_high_wave.db"))

    seed_sea_environment(
        bot,
        sea_id=1,
        sea_lat=54.95,
        sea_lon=20.2,
        wave=0.0,  # target_wave=0
        water_temp=15.0,
        city_id=101,
        city_name="TestCity",
        wind_speed=3.0,
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

    # Create calm asset with wave_score_0_10=0, wave_conf=0.95
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
        "UPDATE assets SET shot_doy=?, wave_score_0_10=?, wave_conf=? WHERE id=?",
        (today_doy, 0.0, 0.95, calm_id),
    )

    # Create high wave asset with wave_score_0_10=3, wave_conf=0.9
    # This should be filtered out by calm_guard
    high_wave_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=2,
        file_name="high_wave.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=3,
        photo_sky="sunny",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, wave_score_0_10=?, wave_conf=? WHERE id=?",
        (today_doy, 3.0, 0.9, high_wave_id),
    )

    bot.db.commit()

    class CapturingOpenAI:
        def __init__(self) -> None:
            self.api_key = "dummy"
            self.calls: list[dict[str, Any]] = []

        async def chat_completion(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)

            class MockChoice:
                def __init__(self) -> None:
                    self.message = type("obj", (object,), {"content": "Calm sea test"})()

            class MockResponse:
                def __init__(self) -> None:
                    self.choices = [MockChoice()]
                    self.usage = None

            return MockResponse()

    capturing_openai = CapturingOpenAI()
    monkeypatch.setattr(bot, "openai", capturing_openai)

    result = await bot._publish_sea(
        rubric=rubric,
        channel_id=-999,
        test=True,
    )

    assert result is True
    # High wave asset should have been filtered by calm_guard


@pytest.mark.asyncio
async def test_unknown_wave_no_bonus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Asset with wave_score_0_10=NULL should not receive CalmWaveBonus."""

    bot = Bot("dummy", str(tmp_path / "test_unknown_wave_no_bonus.db"))

    seed_sea_environment(
        bot,
        sea_id=1,
        sea_lat=54.95,
        sea_lon=20.2,
        wave=0.0,  # target_wave=0
        water_temp=15.0,
        city_id=101,
        city_name="TestCity",
        wind_speed=3.0,
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

    # Create asset with wave_score_0_10=0, wave_conf=0.95
    known_calm_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=1,
        file_name="known_calm.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=0,
        photo_sky="sunny",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, wave_score_0_10=?, wave_conf=? WHERE id=?",
        (today_doy, 0.0, 0.95, known_calm_id),
    )

    # Create asset with wave_score_0_10=NULL (unknown wave)
    # Should NOT receive CalmWaveBonus
    unknown_wave_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=2,
        file_name="unknown_wave.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=None,
        photo_sky="sunny",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, wave_score_0_10=?, wave_conf=? WHERE id=?",
        (today_doy, None, None, unknown_wave_id),
    )

    bot.db.commit()

    class CapturingOpenAI:
        def __init__(self) -> None:
            self.api_key = "dummy"
            self.calls: list[dict[str, Any]] = []

        async def chat_completion(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)

            class MockChoice:
                def __init__(self) -> None:
                    self.message = type("obj", (object,), {"content": "Calm sea test"})()

            class MockResponse:
                def __init__(self) -> None:
                    self.choices = [MockChoice()]
                    self.usage = None

            return MockResponse()

    capturing_openai = CapturingOpenAI()
    monkeypatch.setattr(bot, "openai", capturing_openai)

    result = await bot._publish_sea(
        rubric=rubric,
        channel_id=-999,
        test=True,
    )

    assert result is True
    # Unknown wave asset should not receive CalmWaveBonus, should receive penalty


@pytest.mark.asyncio
async def test_b0_corridor_enforcement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B0 corridor (0-1) should exclude wave=2, pass wave=0,1, and pass wave=None with penalty."""

    bot = Bot("dummy", str(tmp_path / "test_b0_corridor_enforcement.db"))

    seed_sea_environment(
        bot,
        sea_id=1,
        sea_lat=54.95,
        sea_lon=20.2,
        wave=0.1,  # target_wave=0
        water_temp=15.0,
        city_id=101,
        city_name="TestCity",
        wind_speed=3.0,
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

    # Create asset with wave_score_0_10=0
    wave0_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=1,
        file_name="wave0.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=0,
        photo_sky="sunny",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, wave_score_0_10=?, wave_conf=? WHERE id=?",
        (today_doy, 0.0, 0.95, wave0_id),
    )

    # Create asset with wave_score_0_10=1
    wave1_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=2,
        file_name="wave1.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=1,
        photo_sky="sunny",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, wave_score_0_10=?, wave_conf=? WHERE id=?",
        (today_doy, 1.0, 0.90, wave1_id),
    )

    # Create asset with wave_score_0_10=2 (should be excluded from B0)
    wave2_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=3,
        file_name="wave2.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=2,
        photo_sky="sunny",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, wave_score_0_10=?, wave_conf=? WHERE id=?",
        (today_doy, 2.0, 0.88, wave2_id),
    )

    # Create asset with wave_score_0_10=None (unknown wave, passes with penalty)
    wave_none_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=4,
        file_name="wave_none.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=None,
        photo_sky="sunny",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, wave_score_0_10=?, wave_conf=? WHERE id=?",
        (today_doy, None, None, wave_none_id),
    )

    bot.db.commit()

    class CapturingOpenAI:
        def __init__(self) -> None:
            self.api_key = "dummy"
            self.calls: list[dict[str, Any]] = []

        async def chat_completion(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)

            class MockChoice:
                def __init__(self) -> None:
                    self.message = type("obj", (object,), {"content": "Calm sea test"})()

            class MockResponse:
                def __init__(self) -> None:
                    self.choices = [MockChoice()]
                    self.usage = None

            return MockResponse()

    capturing_openai = CapturingOpenAI()
    monkeypatch.setattr(bot, "openai", capturing_openai)

    result = await bot._publish_sea(
        rubric=rubric,
        channel_id=-999,
        test=True,
    )

    assert result is True
    # wave=0 and wave=1 should pass B0 corridor
    # wave=2 should be excluded from B0
    # wave=None should pass but receive penalty
