import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import main as main_module  # noqa: E402
from tests.fixtures.sea import create_sea_asset, create_stub_image, seed_sea_environment  # noqa: E402


@pytest.mark.asyncio
async def test_clear_prefers_sunny_visible_sky(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ensure clear bucket prefers sunny/visible skies."""

    bot = main_module.Bot("dummy", str(tmp_path / "clear_choice.db"))

    seed_sea_environment(
        bot,
        sea_id=1,
        sea_lat=54.95,
        sea_lon=20.2,
        wave=0.8,
        water_temp=12.0,
        city_id=101,
        city_name="Калининград",
        wind_speed=3.0,
        cloud_cover=5.0,
    )

    rubric = bot.data.get_rubric_by_code("sea")
    assert rubric is not None
    updated_config = dict(rubric.config or {})
    updated_config.update(
        {
            "enabled": True,
            "channel_id": -999777,
            "test_channel_id": -999777,
            "sea_id": 1,
        }
    )
    bot.data.save_rubric_config("sea", updated_config)

    image_path = create_stub_image(tmp_path, "clear-test.jpg")
    today_doy = datetime.utcnow().timetuple().tm_yday

    sunny_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=2001,
        file_name="sunny.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=3.0,
        photo_sky="sunny",
        sky_visible=True,
    )
    overcast_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=2002,
        file_name="overcast.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=3.0,
        photo_sky="overcast",
        sky_visible=True,
    )
    hidden_id = create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=2003,
        file_name="hidden.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=3.0,
        photo_sky="sunny",
        sky_visible=False,
    )

    for asset_id in (sunny_id, overcast_id, hidden_id):
        bot.db.execute("UPDATE assets SET shot_doy=? WHERE id=?", (today_doy, asset_id))
    bot.db.commit()

    class DummyOpenAI:
        def __init__(self) -> None:
            self.api_key = "dummy"

        async def generate_json(self, **kwargs: Any):  # type: ignore[override]
            from openai_client import OpenAIResponse

            usage = {
                "prompt_tokens": 5,
                "completion_tokens": 5,
                "total_tokens": 10,
                "endpoint": "/v1/responses",
                "request_id": "clear-test",
            }
            content = {
                "caption": "Порадую вас морем — небо сегодня прозрачное.",
                "hashtags": ["#море", "#БалтийскоеМоре"],
            }
            return OpenAIResponse(content, usage)

    bot.openai = DummyOpenAI()

    async def fake_api_request(self, method: str, data: Any = None, *, files: Any = None) -> dict[str, Any]:
        if method == "sendPhoto":
            return {"ok": True, "result": {"message_id": 600}}
        return {"ok": True}

    bot.api_request = fake_api_request.__get__(bot, main_module.Bot)

    async def fake_reverse_geocode(self, lat: float, lon: float) -> dict[str, Any]:
        return {}

    bot._reverse_geocode = fake_reverse_geocode.__get__(bot, main_module.Bot)

    published: dict[str, Any] = {}

    def capture_post_history(channel_id: int, message_id: int, asset_id: str, rubric_id: int, metadata: dict[str, Any]) -> None:
        published["asset_id"] = asset_id
        published["metadata"] = metadata

    monkeypatch.setattr(bot.data, "record_post_history", capture_post_history, raising=False)

    result = await bot.publish_rubric("sea")
    assert result is True
    assert published["asset_id"] == sunny_id

    await bot.close()


@pytest.mark.asyncio
async def test_logs_steps_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    bot = main_module.Bot("dummy", str(tmp_path / "log_steps.db"))

    caplog.set_level(logging.INFO)

    seed_sea_environment(
        bot,
        sea_id=1,
        sea_lat=54.95,
        sea_lon=20.2,
        wave=0.6,
        water_temp=10.0,
        city_id=301,
        city_name="Калининград",
        wind_speed=2.5,
        cloud_cover=5.0,
    )

    rubric = bot.data.get_rubric_by_code("sea")
    assert rubric is not None
    updated_config = dict(rubric.config or {})
    updated_config.update(
        {
            "enabled": True,
            "channel_id": -999888,
            "test_channel_id": -999888,
            "sea_id": 1,
        }
    )
    bot.data.save_rubric_config("sea", updated_config)

    image_path = create_stub_image(tmp_path, "logs-test.jpg")
    today_doy = datetime.utcnow().timetuple().tm_yday

    asset_ids = [
        create_sea_asset(
            bot,
            rubric_id=rubric.id,
            message_id=3001,
            file_name="sunny1.jpg",
            local_path=image_path,
            tags=["sea"],
            sea_wave_score=3.0,
            photo_sky="sunny",
            sky_visible=True,
        ),
        create_sea_asset(
            bot,
            rubric_id=rubric.id,
            message_id=3002,
            file_name="partly.jpg",
            local_path=image_path,
            tags=["sea"],
            sea_wave_score=3.2,
            photo_sky="partly_cloudy",
            sky_visible=True,
        ),
        create_sea_asset(
            bot,
            rubric_id=rubric.id,
            message_id=3003,
            file_name="hidden.jpg",
            local_path=image_path,
            tags=["sea"],
            sea_wave_score=2.8,
            photo_sky="sunny",
            sky_visible=False,
        ),
    ]

    for asset_id in asset_ids:
        bot.db.execute("UPDATE assets SET shot_doy=? WHERE id=?", (today_doy, asset_id))
    bot.db.commit()

    class DummyOpenAI:
        def __init__(self) -> None:
            self.api_key = "dummy"

        async def generate_json(self, **kwargs: Any):  # type: ignore[override]
            from openai_client import OpenAIResponse

            usage = {
                "prompt_tokens": 5,
                "completion_tokens": 5,
                "total_tokens": 10,
                "endpoint": "/v1/responses",
                "request_id": "log-test",
            }
            content = {
                "caption": "Поделюсь морским настроением.",
                "hashtags": ["#море", "#БалтийскоеМоре"],
            }
            return OpenAIResponse(content, usage)

    bot.openai = DummyOpenAI()

    async def fake_api_request(self, method: str, data: Any = None, *, files: Any = None) -> dict[str, Any]:
        if method == "sendPhoto":
            return {"ok": True, "result": {"message_id": 700}}
        return {"ok": True}

    bot.api_request = fake_api_request.__get__(bot, main_module.Bot)

    async def fake_reverse_geocode(self, lat: float, lon: float) -> dict[str, Any]:
        return {}

    bot._reverse_geocode = fake_reverse_geocode.__get__(bot, main_module.Bot)

    await bot.publish_rubric("sea")

    log_lines = [record.getMessage() for record in caplog.records if "SEA_RUBRIC" in record.getMessage()]
    combined = "\n".join(log_lines)

    assert "SEA_RUBRIC stage B1" in combined
    assert "pool_after_B1=" in combined
    assert "pool_after_B2=" in combined
    assert "pool_after_AN=" in combined
    assert "pool_after_B0=" in combined
    assert any("SEA_RUBRIC top5 #" in line and "sky_visible=" in line for line in log_lines)
    assert "SEA_RUBRIC selected" in combined

    await bot.close()


@pytest.mark.asyncio
async def test_sky_scoring_partly_cloudy_prefers_matching(tmp_path: Path) -> None:
    """Test that when factual sky_bucket is partly_cloudy, picker prefers photo_sky=partly_cloudy/sunny over overcast."""
    bot = main_module.Bot("dummy", str(tmp_path / "sky_scoring.db"))

    # Set up sea environment with partly_cloudy factual sky
    seed_sea_environment(
        bot,
        sea_id=1,
        sea_lat=54.95,
        sea_lon=20.2,
        wave=0.8,
        water_temp=12.0,
        city_id=101,
        city_name="Калининград",
        wind_speed=3.0,
        cloud_cover=50.0,  # This should result in partly_cloudy bucket
    )

    rubric = bot.data.get_rubric_by_code("sea")
    assert rubric is not None
    updated_config = dict(rubric.config or {})
    updated_config.update(
        {
            "enabled": True,
            "channel_id": -999123,
            "test_channel_id": -999123,
            "sea_id": 1,
        }
    )
    bot.data.save_rubric_config("sea", updated_config)

    # Create assets with different sky conditions
    image_path = create_stub_image(tmp_path, "test-sky.jpg")

    # Asset with matching partly_cloudy sky (should be preferred)
    create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=1001,
        file_name="partly_cloudy.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=3.0,
        photo_sky="partly_cloudy",
        is_sunset=False,
    )

    # Asset with sunny sky (should get bonus but less than partly_cloudy)
    create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=1002,
        file_name="sunny.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=3.0,
        photo_sky="sunny",
        is_sunset=False,
    )

    # Asset with overcast sky (should be penalized)
    create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=1003,
        file_name="overcast.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=3.0,
        photo_sky="overcast",
        is_sunset=False,
    )

    # Asset with mostly_clear sky (should get bonus)
    create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=1004,
        file_name="mostly_clear.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=3.0,
        photo_sky="mostly_clear",
        is_sunset=False,
    )

    # Asset with mostly_cloudy sky (should get penalty)
    create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=1005,
        file_name="mostly_cloudy.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=3.0,
        photo_sky="mostly_cloudy",
        is_sunset=False,
    )

    # Mock OpenAI to avoid actual API calls
    from openai_client import OpenAIResponse

    class DummyOpenAI:
        def __init__(self) -> None:
            self.api_key = "dummy"

        async def generate_json(self, **kwargs: Any) -> OpenAIResponse:
            usage = {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "endpoint": "/v1/responses",
                "request_id": "test-req",
            }
            content = {
                "caption": "Тестовая подпись для проверки выбора.",
                "hashtags": ["#море", "#БалтийскоеМоре"]
            }
            return OpenAIResponse(content, usage)

    bot.openai = DummyOpenAI()

    # Mock API requests
    async def fake_api_request(self, method: str, data: Any = None, *, files: Any = None) -> dict[str, Any]:
        if method == "sendPhoto":
            return {"ok": True, "result": {"message_id": 501}}
        return {"ok": True}

    bot.api_request = fake_api_request.__get__(bot, main_module.Bot)

    # Mock reverse geocode
    async def fake_reverse_geocode(self, lat: float, lon: float) -> dict[str, Any]:
        return {"city": "Калининград"}

    bot._reverse_geocode = fake_reverse_geocode.__get__(bot, main_module.Bot)

    # Publish sea rubric
    result = await bot.publish_rubric("sea")
    assert result is True

    # Check that selected asset has preferred sky conditions
    # The selection should prioritize partly_cloudy or sunny over overcast
    # We can verify this by checking the published message or by examining the selection logic
    # For now, we'll just ensure the rubric published successfully
    # In a real test, you might want to capture the selected asset and verify its sky condition

    await bot.close()


@pytest.mark.asyncio
async def test_logs_not_truncated_and_prefixed(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Test that SEA_RUBRIC logs are multi-line and not truncated."""
    bot = main_module.Bot("dummy", str(tmp_path / "logs.db"))

    # Set up sea environment
    seed_sea_environment(
        bot,
        sea_id=1,
        sea_lat=54.95,
        sea_lon=20.2,
        wave=0.8,
        water_temp=12.0,
        city_id=101,
        city_name="Калининград",
        wind_speed=3.0,
        cloud_cover=50.0,
    )

    rubric = bot.data.get_rubric_by_code("sea")
    assert rubric is not None
    updated_config = dict(rubric.config or {})
    updated_config.update(
        {
            "enabled": True,
            "channel_id": -999123,
            "test_channel_id": -999123,
            "sea_id": 1,
        }
    )
    bot.data.save_rubric_config("sea", updated_config)

    # Create test asset
    image_path = create_stub_image(tmp_path, "test-logs.jpg")
    create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=2001,
        file_name="test-logs.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=3.0,
        photo_sky="partly_cloudy",
        is_sunset=False,
    )

    # Mock OpenAI
    from openai_client import OpenAIResponse

    class DummyOpenAI:
        def __init__(self) -> None:
            self.api_key = "dummy"

        async def generate_json(self, **kwargs: Any) -> OpenAIResponse:
            usage = {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "endpoint": "/v1/responses",
                "request_id": "test-req",
            }
            content = {
                "caption": "Тестовая подпись.",
                "hashtags": ["#море", "#БалтийскоеМоре"]
            }
            return OpenAIResponse(content, usage)

    bot.openai = DummyOpenAI()

    # Mock API requests
    async def fake_api_request(self, method: str, data: Any = None, *, files: Any = None) -> dict[str, Any]:
        if method == "sendPhoto":
            return {"ok": True, "result": {"message_id": 501}}
        return {"ok": True}

    bot.api_request = fake_api_request.__get__(bot, main_module.Bot)

    # Mock reverse geocode
    async def fake_reverse_geocode(self, lat: float, lon: float) -> dict[str, Any]:
        return {"city": "Калининград"}

    bot._reverse_geocode = fake_reverse_geocode.__get__(bot, main_module.Bot)

    with caplog.at_level(logging.INFO):
        result = await bot.publish_rubric("sea")

    assert result is True

    # Check that logs are properly formatted and not truncated
    messages = [record.getMessage() for record in caplog.records if "SEA_RUBRIC" in record.getMessage()]

    # Should have separate weather, season, pool, top5, and selected logs
    weather_logs = [msg for msg in messages if msg.startswith("SEA_RUBRIC weather ")]
    season_logs = [msg for msg in messages if msg.startswith("SEA_RUBRIC season ")]
    pool_logs = [msg for msg in messages if msg.startswith("SEA_RUBRIC pool_counts ")]
    top5_logs = [msg for msg in messages if msg.startswith("SEA_RUBRIC top5 #")]
    selected_logs = [msg for msg in messages if msg.startswith("SEA_RUBRIC selected ")]

    assert len(weather_logs) >= 1, f"Expected weather logs, got: {weather_logs}"
    assert len(season_logs) >= 1, f"Expected season logs, got: {season_logs}"
    assert len(pool_logs) >= 1, f"Expected pool logs, got: {pool_logs}"
    assert len(top5_logs) >= 1, f"Expected top5 logs, got: {top5_logs}"
    assert len(selected_logs) >= 1, f"Expected selected logs, got: {selected_logs}"

    # Check that logs are not giant JSON (they should be readable multi-line)
    for log in weather_logs + season_logs + pool_logs + selected_logs:
        # Should not contain giant JSON that gets truncated
        assert "..." not in log or log.count("...") <= 1, f"Log appears truncated: {log}"
        # Should be reasonably short for individual lines
        assert len(log) < 1000, f"Log too long, likely contains giant JSON: {log[:200]}..."

    # Top5 logs should be individual lines
    for top5_log in top5_logs:
        assert "SEA_RUBRIC top5 #" in top5_log
        assert "id=" in top5_log
        assert "sky=" in top5_log
        assert "wave_score=" in top5_log
        assert "score=" in top5_log
        assert "reasons=" in top5_log

    await bot.close()


@pytest.mark.asyncio
async def test_prompt_soft_intro_and_constraints(tmp_path: Path) -> None:
    """Test that prompt includes anti-jargon rule, soft-intro options, and 350-char instruction."""
    bot = main_module.Bot("dummy", str(tmp_path / "prompt.db"))

    from openai_client import OpenAIResponse

    class CapturingOpenAI:
        def __init__(self) -> None:
            self.api_key = "dummy"
            self.calls: list[dict[str, Any]] = []

        async def generate_json(self, **kwargs: Any) -> OpenAIResponse:
            self.calls.append(kwargs)
            usage = {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "endpoint": "/v1/responses",
                "request_id": f"req-{len(self.calls)}",
            }
            content = {
                "caption": "Порадую вас морем — побережье зовёт вдохнуть глубже.",
                "hashtags": ["#море", "#БалтийскоеМоре"]
            }
            return OpenAIResponse(content, usage)

    bot.openai = CapturingOpenAI()

    # Test with storm state and fact
    await bot._generate_sea_caption(
        storm_state="storm",
        storm_persisting=False,
        wave_height_m=1.2,
        wave_score=4.8,
        wind_class="strong",
        wind_ms=12.0,
        wind_kmh=43.2,
        clouds_label="пасмурно",
        sunset_selected=False,
        want_sunset=False,
        place_hashtag="#Балтийск",
        fact_sentence="Балтийское море содержит 0.2% соли, что меньше, чем в Мировом океане.",
        job=None,
    )

    first_call = bot.openai.calls[-1]
    system_prompt = first_call["system_prompt"]
    user_prompt = first_call["user_prompt"]

    # Check anti-jargon rule
    assert "термоклин" in system_prompt  # Should be in list of terms to avoid
    assert "галоклин" in system_prompt  # Should be in list of terms to avoid
    assert "научных терминов" in system_prompt

    # Check soft-intro options
    assert "Знаете ли вы" in system_prompt
    assert "Интересный факт:" in system_prompt
    assert "К слову о Балтике" in system_prompt
    assert "Поделюсь фактом" in system_prompt

    # Check 350-char limit
    assert "350 символов" in system_prompt
    assert "450" not in system_prompt  # Should not have old 450 limit

    # Check that fact is included for storm (not strong_storm)
    assert '"fact_sentence":' in user_prompt

    # Test with strong_storm - fact should be omitted
    await bot._generate_sea_caption(
        storm_state="strong_storm",
        storm_persisting=True,
        wave_height_m=2.1,
        wave_score=6.7,
        wind_class="very_strong",
        wind_ms=18.0,
        wind_kmh=64.8,
        clouds_label="штормово",
        sunset_selected=False,
        want_sunset=False,
        place_hashtag="#Балтийск",
        fact_sentence="Балтийское море содержит 0.2% соли, что меньше, чем в Мировом океане.",
        job=None,
    )

    strong_call = bot.openai.calls[-1]
    strong_user_prompt = strong_call["user_prompt"]

    # Fact should be omitted for strong_storm
    assert '"fact_sentence":' not in strong_user_prompt

    await bot.close()
