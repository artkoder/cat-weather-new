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
async def test_sea_prompt_contains_hard_limit_phrase(
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
    assert "ЖЁСТКОЕ ОГРАНИЧЕНИЕ" in system_prompt
    assert "700 символов" in system_prompt
    assert "900 символов" in system_prompt
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


@pytest.mark.asyncio
async def test_sea_caption_trim_applies_at_990(
    monkeypatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        main_module.Bot, "_record_openai_usage", async_record_usage_noop, raising=False
    )

    long_text = "A" * 1200
    captured_requests: list[dict[str, Any]] = []
    send_calls: list[dict[str, Any]] = []

    async def fake_generate(
        self, *args: Any, **kwargs: Any
    ) -> tuple[str, list[str], dict[str, Any]]:
        return long_text, ["#БалтийскоеМоре"], {"request_id": "trim-req"}

    async def fake_api_request(
        self,
        method: str,
        data: Any = None,
        *,
        files: Any = None,
    ) -> dict[str, Any]:
        record = {"method": method, "data": data, "files": files}
        captured_requests.append(record)
        if method == "sendPhoto":
            send_calls.append(record)
            return {"ok": True, "result": {"message_id": 777}}
        return {"ok": True}

    monkeypatch.setattr(
        main_module.Bot,
        "_generate_sea_caption_with_timeout",
        fake_generate,
        raising=False,
    )
    monkeypatch.setattr(main_module.Bot, "api_request", fake_api_request, raising=False)

    bot = main_module.Bot("dummy", str(tmp_path / "trim.db"))
    bot.supabase = DummySupabase()

    rubric = bot.data.get_rubric_by_code("sea")
    assert rubric is not None
    updated_config = dict(rubric.config or {})
    updated_config.update(
        {
            "enabled": True,
            "channel_id": -999321,
            "test_channel_id": -999321,
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

    image_path = create_stub_image(tmp_path, "sea-trim.jpg")
    create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=402,
        file_name="sea-trim.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=2,
        photo_sky="sunny",
        is_sunset=True,
    )

    with caplog.at_level(logging.INFO):
        result = await bot.publish_rubric("sea")
    assert result is True

    methods = [entry["method"] for entry in captured_requests]
    assert "sendPhoto" in methods
    assert send_calls
    assert len(send_calls) == 1
    send_payload = send_calls[0]["data"]
    assert isinstance(send_payload, dict)
    caption = send_payload.get("caption")
    assert isinstance(caption, str)

    trim_records = [
        record
        for record in caplog.records
        if "SEA_RUBRIC caption_trim applied" in record.getMessage()
    ]
    assert trim_records
    trim_record = trim_records[0]
    assert isinstance(trim_record.args, tuple)
    original_len, final_len = trim_record.args
    assert original_len > final_len
    assert final_len <= 990
    assert len(caption) == final_len

    bot.db.close()
