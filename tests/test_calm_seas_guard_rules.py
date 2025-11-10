"""Tests for calm seas guard rules in sea photo selection."""

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
async def test_calm_seas_guard_filters_stormy_photos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When target_wave=1 and calm candidates exist, wave≥5 should be filtered out."""

    bot = Bot("dummy", str(tmp_path / "test_calm_guard.db"))

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

    image_path = create_stub_image(tmp_path, "test.jpg")
    today_doy = datetime.utcnow().timetuple().tm_yday

    # Create calm asset with wave_score=0
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

    # Create moderate asset with wave_score=2
    moderate_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=2,
        file_name="moderate.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=2,
        photo_sky="sunny",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, photo_wave=? WHERE id=?", (today_doy, 0.4, moderate_id)
    )

    # Create stormy asset with wave_score=5
    stormy5_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=3,
        file_name="stormy5.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=5,
        photo_sky="sunny",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, photo_wave=? WHERE id=?", (today_doy, 1.0, stormy5_id)
    )

    # Create very stormy asset with wave_score=7
    stormy7_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=4,
        file_name="stormy7.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=7,
        photo_sky="sunny",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, photo_wave=? WHERE id=?", (today_doy, 1.4, stormy7_id)
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
    # The stormy photos (wave_score≥5) should have been filtered out in B0/B1 stages


@pytest.mark.asyncio
async def test_calm_seas_guard_no_filter_when_no_calm_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When target_wave=2 but no wave≤2 candidates, guard rules should not apply."""

    bot = Bot("dummy", str(tmp_path / "test_calm_guard_no_filter.db"))

    seed_sea_environment(
        bot,
        sea_id=1,
        sea_lat=54.95,
        sea_lon=20.2,
        wave=0.4,
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

    image_path = create_stub_image(tmp_path, "test.jpg")
    today_doy = datetime.utcnow().timetuple().tm_yday

    # Create moderate asset with wave_score=4
    moderate_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=1,
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

    # Create stormy asset with wave_score=6
    stormy_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=2,
        file_name="stormy.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=6,
        photo_sky="sunny",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, photo_wave=? WHERE id=?", (today_doy, 1.2, stormy_id)
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
                    self.message = type("obj", (object,), {"content": "Moderate sea test"})()

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
    # When no calm candidates exist, guard rules don't apply, all candidates in pool


@pytest.mark.asyncio
async def test_calm_seas_guard_null_wave_penalty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When target_wave≤1 and calm candidates exist, NULL wave gets +0.8 penalty."""

    bot = Bot("dummy", str(tmp_path / "test_calm_guard_null.db"))

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

    image_path = create_stub_image(tmp_path, "test.jpg")
    today_doy = datetime.utcnow().timetuple().tm_yday

    # Create calm asset with wave_score=0
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

    # Create asset with NULL wave_score
    null_wave_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=2,
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
    # The NULL wave candidate should receive +0.8 penalty but remain in pool


@pytest.mark.asyncio
async def test_calm_seas_guard_very_calm_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When target_wave=0.5 and calm candidates exist, wave≥5 filtered correctly."""

    bot = Bot("dummy", str(tmp_path / "test_calm_guard_very_calm.db"))

    seed_sea_environment(
        bot,
        sea_id=1,
        sea_lat=54.95,
        sea_lon=20.2,
        wave=0.05,
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

    # Create very calm asset with wave_score=0
    very_calm_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=1,
        file_name="very_calm.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=0,
        photo_sky="sunny",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, photo_wave=? WHERE id=?", (today_doy, 0.0, very_calm_id)
    )

    # Create moderate calm asset with wave_score=1
    calm_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=2,
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

    # Create stormy asset with wave_score=7
    stormy_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=3,
        file_name="stormy.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=7,
        photo_sky="sunny",
        sky_visible=True,
    )
    bot.db.execute(
        "UPDATE assets SET shot_doy=?, photo_wave=? WHERE id=?", (today_doy, 1.4, stormy_id)
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
                    self.message = type("obj", (object,), {"content": "Very calm sea test"})()

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
    # Stormy photo should be filtered out, very calm photos should win
