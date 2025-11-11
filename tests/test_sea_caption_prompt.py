import logging
import os
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import main as main_module  # noqa: E402
from openai_client import OpenAIResponse  # noqa: E402
from tests.fixtures.sea import (  # noqa: E402
    create_sea_asset,
    create_stub_image,
    seed_sea_environment,
)


class CapturingOpenAI:
    def __init__(self) -> None:
        self.api_key = "fake-key"
        self.calls: list[dict[str, Any]] = []

    async def generate_json(self, **kwargs) -> OpenAIResponse:  # type: ignore[override]
        self.calls.append(kwargs)
        usage = {
            "prompt_tokens": 5,
            "completion_tokens": 10,
            "total_tokens": 15,
            "endpoint": "/v1/responses",
            "request_id": f"req-{len(self.calls)}",
        }
        content = {
            "caption": "Порадую вас морем — волны зовут на прогулку.",
            "hashtags": ["морем", "БалтийскоеМоре"],
        }
        return OpenAIResponse(content, usage)


class DummySupabase:
    async def insert_token_usage(self, *args: Any, **kwargs: Any) -> tuple[bool, Any, Any]:
        return False, None, "disabled"

    async def aclose(self) -> None:
        return None


async def async_record_usage_noop(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
    return None


@pytest.mark.asyncio
async def test_sea_caption_prompt_fact_numbers_and_limit_instruction(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        main_module.Bot, "_record_openai_usage", async_record_usage_noop, raising=False
    )
    bot = main_module.Bot("dummy", str(tmp_path / "prompt.db"))
    bot.supabase = DummySupabase()
    bot.openai = CapturingOpenAI()

    fact_text = "В 1998 году берег отвоевал 3 км² песка."

    await bot._generate_sea_caption(
        storm_state="storm",
        storm_persisting=False,
        wave_height_m=1.2,
        wave_score=6,
        wind_class="strong",
        wind_ms=12.0,
        wind_kmh=43.2,
        clouds_label="пасмурно",
        sunset_selected=False,
        want_sunset=False,
        place_hashtag="#Балтийск",
        fact_sentence=fact_text,
        job=None,
    )
    first_call = bot.openai.calls[-1]
    system_prompt = first_call["system_prompt"]
    user_prompt = first_call["user_prompt"]
    assert "350" in system_prompt
    assert "числа/названия/термины" in system_prompt
    assert '"fact_sentence":' in user_prompt
    assert fact_text in user_prompt
    assert '"place_hashtag": "#Балтийск"' in user_prompt

    await bot._generate_sea_caption(
        storm_state="strong_storm",
        storm_persisting=True,
        wave_height_m=2.1,
        wave_score=10,
        wind_class="very_strong",
        wind_ms=18.0,
        wind_kmh=64.8,
        clouds_label="штормово",
        sunset_selected=False,
        want_sunset=False,
        place_hashtag="#Балтийск",
        fact_sentence=fact_text,
        job=None,
    )
    strong_call = bot.openai.calls[-1]
    strong_user_prompt = strong_call["user_prompt"]
    assert '"fact_sentence":' not in strong_user_prompt

    bot.db.close()


@pytest.mark.asyncio
async def test_sea_logging_prefixes(
    monkeypatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        main_module.Bot, "_record_openai_usage", async_record_usage_noop, raising=False
    )

    async def fake_reverse_geocode(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"city": "Зеленоградск"}

    async def fake_api_request(self, method: str, data: Any = None, *, files: Any = None) -> dict[str, Any]:  # type: ignore[override]
        if method == "sendPhoto":
            return {"ok": True, "result": {"message_id": 501}}
        return {"ok": True}

    monkeypatch.setattr(main_module.Bot, "_reverse_geocode", fake_reverse_geocode, raising=False)
    monkeypatch.setattr(main_module.Bot, "api_request", fake_api_request, raising=False)

    bot = main_module.Bot("dummy", str(tmp_path / "logging.db"))
    bot.supabase = DummySupabase()
    bot.openai = CapturingOpenAI()

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
    rubric = bot.data.get_rubric_by_code("sea")
    assert rubric is not None

    seed_sea_environment(
        bot,
        sea_id=1,
        sea_lat=54.95,
        sea_lon=20.2,
        wave=0.4,
        water_temp=9.5,
        city_id=101,
        city_name="Зеленоградск",
        wind_speed=5.0,
    )

    image_path = create_stub_image(tmp_path, "sea-log.jpg")
    create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=401,
        file_name="sea-log.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=2,
        photo_sky="sunny",
        is_sunset=True,
    )

    with caplog.at_level(logging.INFO):
        result = await bot.publish_rubric("sea")
    assert result is True

    messages = [record.getMessage() for record in caplog.records]
    assert any(msg.startswith("SEA_RUBRIC weather") for msg in messages)
    assert any(msg.startswith("SEA_RUBRIC season") for msg in messages)
    assert any(msg.startswith("SEA_RUBRIC pool after") for msg in messages)
    assert any(msg.startswith("SEA_RUBRIC top5:") for msg in messages)
    assert any(msg.startswith("SEA_RUBRIC facts choose") for msg in messages)
    assert any(msg.startswith("SEA_RUBRIC selected") for msg in messages)

    bot.db.close()
