import json
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


class RoutingOpenAI:
    def __init__(self) -> None:
        self.api_key = "fake-key"
        self.calls: list[dict[str, Any]] = []

    async def generate_json(self, **kwargs: Any) -> OpenAIResponse:  # type: ignore[override]
        self.calls.append(kwargs)
        usage = {
            "prompt_tokens": 5,
            "completion_tokens": 10,
            "total_tokens": 15,
            "endpoint": "/v1/responses",
            "request_id": f"req-{len(self.calls)}",
        }
        content = {
            "caption": "Порадую вас морем — побережье зовёт на прогулку.",
            "hashtags": ["котопогода", "БалтийскоеМоре", "мореветка"],
        }
        return OpenAIResponse(content, usage)


class DummySupabase:
    async def insert_token_usage(self, *args: Any, **kwargs: Any) -> tuple[bool, Any, Any]:
        return False, None, "disabled"

    async def aclose(self) -> None:
        return None


async def async_noop(*_args: Any, **_kwargs: Any) -> None:
    return None


@pytest.mark.asyncio
async def test_sea_hashtags_excludes_kotopogoda(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(main_module.Bot, "_record_openai_usage", async_noop, raising=False)

    bot = main_module.Bot("dummy", str(tmp_path / "sea-routing-hashtag.db"))
    bot.supabase = DummySupabase()
    bot.openai = RoutingOpenAI()

    rubric = bot.data.get_rubric_by_code("sea")
    assert rubric is not None
    updated_config = dict(rubric.config or {})
    updated_config.update(
        {
            "enabled": True,
            "channel_id": -900001,
            "test_channel_id": -900002,
            "sea_id": 1,
        }
    )
    bot.data.save_rubric_config("sea", updated_config)
    rubric = bot.data.get_rubric_by_code("sea")
    assert rubric is not None

    seed_sea_environment(
        bot,
        sea_id=1,
        wave=0.4,
        water_temp=9.0,
        city_id=101,
        city_name="Зеленоградск",
        wind_speed=5.0,
    )

    image_path = create_stub_image(tmp_path, "sea-hashtag.jpg")
    create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=701,
        file_name="sea-hashtag.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=2,
        photo_sky="sunny",
        is_sunset=False,
    )

    send_calls: list[dict[str, Any]] = []

    async def capture_api_request(
        self: main_module.Bot, method: str, data: Any = None, *, files: Any = None
    ) -> dict[str, Any]:  # type: ignore[override]
        if method == "sendPhoto":
            send_calls.append({"method": method, "data": data, "files": files})
            return {"ok": True, "result": {"message_id": 501}}
        return {"ok": True, "result": {}}

    monkeypatch.setattr(main_module.Bot, "api_request", capture_api_request, raising=False)

    result = await bot.publish_rubric("sea", test=True, initiator_id=12345)
    assert result is True
    assert send_calls, "sendPhoto call was not captured"

    caption = send_calls[0]["data"]["caption"]
    assert "#котопогода" not in caption

    row = bot.db.execute("SELECT metadata FROM posts_history ORDER BY id DESC LIMIT 1").fetchone()
    assert row is not None
    metadata = json.loads(row["metadata"])
    hashtags = metadata.get("hashtags", [])
    assert all("#котопогода" not in str(tag) for tag in hashtags)

    bot.db.close()


@pytest.mark.asyncio
async def test_publish_sea_routes_to_prod_chat_when_is_prod_true(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(main_module.Bot, "_record_openai_usage", async_noop, raising=False)

    bot = main_module.Bot("dummy", str(tmp_path / "sea-routing-prod.db"))
    bot.supabase = DummySupabase()
    bot.openai = RoutingOpenAI()

    rubric = bot.data.get_rubric_by_code("sea")
    assert rubric is not None
    prod_channel = -910001
    test_channel = -910002
    updated_config = dict(rubric.config or {})
    updated_config.update(
        {
            "enabled": True,
            "channel_id": prod_channel,
            "test_channel_id": test_channel,
            "sea_id": 1,
        }
    )
    bot.data.save_rubric_config("sea", updated_config)
    rubric = bot.data.get_rubric_by_code("sea")
    assert rubric is not None

    seed_sea_environment(
        bot,
        sea_id=1,
        wave=0.5,
        water_temp=8.0,
        city_id=201,
        city_name="Балтийск",
        wind_speed=6.0,
    )

    image_path = create_stub_image(tmp_path, "sea-prod.jpg")
    create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=801,
        file_name="sea-prod.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=3,
        photo_sky="partly_cloudy",
        is_sunset=False,
    )

    send_calls: list[dict[str, Any]] = []

    async def capture_api_request(
        self: main_module.Bot, method: str, data: Any = None, *, files: Any = None
    ) -> dict[str, Any]:  # type: ignore[override]
        if method == "sendPhoto":
            send_calls.append({"method": method, "data": data, "files": files})
            return {"ok": True, "result": {"message_id": 601}}
        return {"ok": True, "result": {}}

    monkeypatch.setattr(main_module.Bot, "api_request", capture_api_request, raising=False)

    result = await bot.publish_rubric("sea", test=False, initiator_id=54321)
    assert result is True
    assert send_calls, "sendPhoto call was not captured"
    assert send_calls[0]["data"]["chat_id"] == prod_channel

    bot.db.close()


@pytest.mark.asyncio
async def test_publish_sea_routes_to_test_chat_when_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(main_module.Bot, "_record_openai_usage", async_noop, raising=False)

    bot = main_module.Bot("dummy", str(tmp_path / "sea-routing-test.db"))
    bot.supabase = DummySupabase()
    bot.openai = RoutingOpenAI()

    rubric = bot.data.get_rubric_by_code("sea")
    assert rubric is not None
    prod_channel = -920001
    test_channel = -920002
    updated_config = dict(rubric.config or {})
    updated_config.update(
        {
            "enabled": True,
            "channel_id": prod_channel,
            "test_channel_id": test_channel,
            "sea_id": 1,
        }
    )
    bot.data.save_rubric_config("sea", updated_config)
    rubric = bot.data.get_rubric_by_code("sea")
    assert rubric is not None

    seed_sea_environment(
        bot,
        sea_id=1,
        wave=0.3,
        water_temp=7.5,
        city_id=301,
        city_name="Янтарный",
        wind_speed=4.5,
    )

    image_path = create_stub_image(tmp_path, "sea-test.jpg")
    create_sea_asset(
        bot,
        rubric_id=rubric.id,
        message_id=901,
        file_name="sea-test.jpg",
        local_path=image_path,
        tags=["sea"],
        sea_wave_score=1,
        photo_sky="mostly_clear",
        is_sunset=False,
    )

    send_calls: list[dict[str, Any]] = []

    async def capture_api_request(
        self: main_module.Bot, method: str, data: Any = None, *, files: Any = None
    ) -> dict[str, Any]:  # type: ignore[override]
        if method == "sendPhoto":
            send_calls.append({"method": method, "data": data, "files": files})
            return {"ok": True, "result": {"message_id": 701}}
        return {"ok": True, "result": {}}

    monkeypatch.setattr(main_module.Bot, "api_request", capture_api_request, raising=False)

    result = await bot.publish_rubric("sea", test=True, initiator_id=11111)
    assert result is True
    assert send_calls, "sendPhoto call was not captured"
    assert send_calls[0]["data"]["chat_id"] == test_channel

    bot.db.close()
