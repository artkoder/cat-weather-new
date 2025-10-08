import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_access import Asset
from flowers_patterns import load_flowers_knowledge
from main import Bot
from openai_client import OpenAIResponse


def _make_asset(asset_id: int, city: str, varieties: list[str]) -> Asset:
    return Asset(
        id=asset_id,
        channel_id=1,
        tg_chat_id=1,
        message_id=asset_id,
        origin="test",
        caption_template=None,
        caption=None,
        hashtags=None,
        categories=[],
        kind="photo",
        file_id=f"file-{asset_id}",
        file_unique_id=f"uniq-{asset_id}",
        file_name=f"flower-{asset_id}.jpg",
        mime_type="image/jpeg",
        file_size=None,
        width=1080,
        height=1080,
        duration=None,
        recognized_message_id=None,
        exif_present=False,
        latitude=None,
        longitude=None,
        city=city,
        country="Россия",
        author_user_id=None,
        author_username=None,
        sender_chat_id=None,
        via_bot_id=None,
        forward_from_user=None,
        forward_from_chat=None,
        local_path=None,
        metadata=None,
        vision_results=None,
        rubric_id=None,
        vision_category="flowers",
        vision_arch_view=None,
        vision_photo_weather=None,
        vision_flower_varieties=varieties,
        vision_confidence=None,
        vision_caption=None,
    )


@pytest.mark.asyncio
async def test_flowers_loader_and_plan_deterministic(monkeypatch, tmp_path):
    kb = load_flowers_knowledge()
    assert kb.patterns, "patterns should be loaded"
    assert kb.colors, "colors should be loaded"
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    bot.data.upsert_rubric("flowers", "Flowers", config={"enabled": True})
    rubric = bot.data.get_rubric_by_code("flowers")
    asset = _make_asset(1, "Калининград", ["rose", "tulip"])

    class FixedDatetime(datetime):
        @classmethod
        def utcnow(cls):  # type: ignore[override]
            return datetime(2024, 5, 17, 6, 0, 0)

        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            base = cls.utcnow()
            if tz is None:
                return base
            return base.replace(tzinfo=timezone.utc).astimezone(tz)

    import main as main_module

    monkeypatch.setattr(main_module, "datetime", FixedDatetime)

    plan_first = bot._build_flowers_plan(
        rubric,
        [asset],
        weather_block=None,
        channel_id=-100,
    )
    plan_second = bot._build_flowers_plan(
        rubric,
        [asset],
        weather_block=None,
        channel_id=-100,
    )
    assert plan_first["pattern_ids"] == plan_second["pattern_ids"]
    assert "weather_focus" in plan_first["pattern_ids"]
    await bot.close()


@pytest.mark.asyncio
async def test_flowers_generation_skips_banned_words(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    bot.data.upsert_rubric("flowers", "Flowers", config={"enabled": True})
    rubric = bot.data.get_rubric_by_code("flowers")
    asset = _make_asset(2, "Калининград", ["hydrangea"])

    calls: list[dict[str, object]] = []

    class DummyOpenAI:
        def __init__(self) -> None:
            self.api_key = "test"

        async def generate_json(self, **kwargs):  # type: ignore[override]
            calls.append(kwargs)
            if len(calls) == 1:
                return OpenAIResponse(
                    {"greeting": "Доброе утро, у нас скидка!", "hashtags": ["#котопогода"]},
                    {
                        "prompt_tokens": 5,
                        "completion_tokens": 5,
                        "total_tokens": 10,
                        "request_id": "req-1",
                        "endpoint": "/v1/responses",
                    },
                )
            return OpenAIResponse(
                {"greeting": "Доброе утро, делимся уютом", "hashtags": ["#котопогода", "#цветы"]},
                {
                    "prompt_tokens": 6,
                    "completion_tokens": 7,
                    "total_tokens": 13,
                    "request_id": "req-2",
                    "endpoint": "/v1/responses",
                },
            )

    bot.openai = DummyOpenAI()

    greeting, hashtags, plan = await bot._generate_flowers_copy(
        rubric,
        [asset],
        channel_id=-200,
    )

    assert len(calls) == 2
    assert "скидка" not in greeting.lower()
    assert hashtags == ["#котопогода", "#цветы"]
    assert isinstance(plan, dict)
    await bot.close()
