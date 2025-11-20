import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import main as main_module  # noqa: E402
from caption_gen import POSTCARD_PREFIX  # noqa: E402
from main import LOVE_COLLECTION_LINK  # noqa: E402
from openai_client import OpenAIResponse  # noqa: E402

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "dummy")


class DummySupabase:
    async def insert_token_usage(self, *args: Any, **kwargs: Any) -> tuple[bool, Any, Any]:
        return False, None, "disabled"

    async def aclose(self) -> None:
        return None


async def async_noop(*_args: Any, **_kwargs: Any) -> None:
    return None


class DummyPostcardOpenAI:
    def __init__(self, caption: str) -> None:
        self.api_key = "test-key"
        self._response = OpenAIResponse(
            {
                "caption": caption,
                "hashtags": ["#Светлогорск", "#БалтийскоеМоре"],
            },
            {
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "total_tokens": 15,
            },
        )

    async def generate_json(self, **kwargs: Any) -> OpenAIResponse:  # type: ignore[override]
        return self._response


def _create_postcard_asset(
    bot: main_module.Bot, *, city: str = "Светлогорск", region: str = "Калининградская область"
) -> str:
    asset_id = bot.data.create_asset(
        upload_id="upload-1",
        file_ref="photo-file",
        content_type="image/jpeg",
        sha256="sha-test",
        width=1080,
        height=1350,
        tg_message_id="1:1",
        tg_chat_id=1,
        source="telegram",
    )
    payload_row = bot.db.execute(
        "SELECT payload_json FROM assets WHERE id=?", (asset_id,)
    ).fetchone()
    payload = json.loads(payload_row["payload_json"] or "{}")
    payload.update({"city": city, "region": region, "tg_chat_id": 1, "message_id": 1})
    now = datetime.now(UTC)
    bot.db.execute(
        "UPDATE assets SET payload_json=?, postcard_score=?, captured_at=?, photo_doy=? WHERE id=?",
        (
            json.dumps(payload, ensure_ascii=False),
            5,
            now.isoformat(),
            now.timetuple().tm_yday,
            asset_id,
        ),
    )
    bot.db.commit()
    return asset_id


@pytest.mark.asyncio
async def test_postcard_publish_routes_to_prod_channel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(main_module.Bot, "_record_openai_usage", async_noop, raising=False)

    bot = main_module.Bot("dummy", str(tmp_path / "postcard-prod.db"))
    bot.supabase = DummySupabase()
    bot.openai = DummyPostcardOpenAI(f"{POSTCARD_PREFIX}над Балтикой — мягкий свет")

    rubric = bot.data.get_rubric_by_code("postcard")
    assert rubric is not None
    prod_channel = -900501
    test_channel = -900502
    config = dict(rubric.config or {})
    config.update(
        {
            "enabled": True,
            "channel_id": prod_channel,
            "test_channel_id": test_channel,
            "postcard_region_hashtag": "#КалининградскаяОбласть",
            "postcard_stopwords": ["клише"],
        }
    )
    bot.data.save_rubric_config("postcard", config)
    _create_postcard_asset(bot)

    send_calls: list[dict[str, Any]] = []

    async def capture_api_request(
        self: main_module.Bot, method: str, data: Any = None, *, files: Any = None
    ) -> dict[str, Any]:  # type: ignore[override]
        if method == "sendPhoto":
            send_calls.append({"data": data, "files": files})
            return {"ok": True, "result": {"message_id": 777}}
        return {"ok": True, "result": {}}

    monkeypatch.setattr(main_module.Bot, "api_request", capture_api_request, raising=False)

    result = await bot.publish_rubric("postcard", test=False)

    assert result is True
    assert send_calls
    payload = send_calls[0]["data"]
    assert payload["chat_id"] == prod_channel
    caption = payload["caption"]
    assert caption.startswith(POSTCARD_PREFIX)
    assert LOVE_COLLECTION_LINK in caption
    assert payload["parse_mode"] == "HTML"

    bot.db.close()


@pytest.mark.asyncio
async def test_postcard_publish_routes_to_test_channel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(main_module.Bot, "_record_openai_usage", async_noop, raising=False)

    bot = main_module.Bot("dummy", str(tmp_path / "postcard-test.db"))
    bot.supabase = DummySupabase()
    bot.openai = DummyPostcardOpenAI(f"{POSTCARD_PREFIX}в Янтарном — вечернее золото")

    rubric = bot.data.get_rubric_by_code("postcard")
    assert rubric is not None
    prod_channel = -910601
    test_channel = -910602
    config = dict(rubric.config or {})
    config.update(
        {
            "enabled": True,
            "channel_id": prod_channel,
            "test_channel_id": test_channel,
            "postcard_region_hashtag": "#КалининградскаяОбласть",
            "postcard_stopwords": [],
        }
    )
    bot.data.save_rubric_config("postcard", config)
    _create_postcard_asset(bot, city="Янтарный")

    send_calls: list[dict[str, Any]] = []

    async def capture_api_request(
        self: main_module.Bot, method: str, data: Any = None, *, files: Any = None
    ) -> dict[str, Any]:  # type: ignore[override]
        if method == "sendPhoto":
            send_calls.append({"data": data, "files": files})
            return {"ok": True, "result": {"message_id": 888}}
        return {"ok": True, "result": {}}

    monkeypatch.setattr(main_module.Bot, "api_request", capture_api_request, raising=False)

    result = await bot.publish_rubric("postcard", test=True)

    assert result is True
    assert send_calls
    payload = send_calls[0]["data"]
    assert payload["chat_id"] == test_channel
    caption = payload["caption"]
    assert caption.startswith(POSTCARD_PREFIX)
    assert LOVE_COLLECTION_LINK in caption

    bot.db.close()
