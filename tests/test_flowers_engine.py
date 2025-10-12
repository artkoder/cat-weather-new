import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_access import Asset
from flowers_patterns import load_flowers_knowledge
from main import Bot
from openai_client import OpenAIResponse


def _make_asset(asset_id: int, city: str, varieties: list[str]) -> Asset:
    payload = {
        "channel_id": 1,
        "tg_chat_id": 1,
        "message_id": asset_id,
        "origin": "test",
        "caption_template": None,
        "caption": None,
        "hashtags": None,
        "kind": "photo",
        "file_id": f"file-{asset_id}",
        "file_unique_id": f"uniq-{asset_id}",
        "file_name": f"flower-{asset_id}.jpg",
        "mime_type": "image/jpeg",
        "file_size": None,
        "duration": None,
        "latitude": None,
        "longitude": None,
        "city": city,
        "country": "Россия",
        "metadata": None,
    }
    labels_json = json.dumps([])
    return Asset(
        id=str(asset_id),
        upload_id=None,
        file_ref=f"file-{asset_id}",
        content_type="image/jpeg",
        sha256=None,
        width=1080,
        height=1080,
        exif_json=None,
        labels_json=labels_json,
        tg_message_id=f"1:{asset_id}",
        payload_json=json.dumps(payload, ensure_ascii=False),
        created_at=datetime.utcnow().isoformat(),
        exif=None,
        labels=json.loads(labels_json),
        payload=payload,
        legacy_values={},
        _vision_category="flowers",
        _vision_flower_varieties=[str(v) for v in varieties] if varieties else [],
    )


def test_flowers_loader_banned_words_contains_new_entries():
    kb = load_flowers_knowledge()
    assert "шепчет" in kb.banned_words
    assert "громче" in kb.banned_words


@pytest.mark.asyncio
async def test_flowers_loader_and_plan_deterministic(monkeypatch, tmp_path):
    kb = load_flowers_knowledge()
    assert kb.patterns, "patterns should be loaded"
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

    plan_first, meta_first = bot._build_flowers_plan(
        rubric,
        [asset],
        weather_block=None,
        channel_id=-100,
    )
    plan_second, meta_second = bot._build_flowers_plan(
        rubric,
        [asset],
        weather_block=None,
        channel_id=-100,
    )
    assert meta_first["pattern_ids"] == meta_second["pattern_ids"]
    assert "weather_focus" in meta_first["pattern_ids"]
    assert "color_palette" not in meta_first["pattern_ids"]
    assert len(plan_first["patterns"]) == len(meta_first["pattern_ids"])
    await bot.close()


@pytest.mark.asyncio
async def test_flowers_plan_uses_detected_colors(monkeypatch, tmp_path):
    kb = load_flowers_knowledge()
    assert kb.patterns, "patterns should be loaded"
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    bot.flowers_kb = kb
    bot.data.upsert_rubric("flowers", "Flowers", config={"enabled": True})
    rubric = bot.data.get_rubric_by_code("flowers")
    asset = _make_asset(10, "Светлогорск", ["rose"])
    asset.vision_results = {
        "colors": {
            "palettes": [
                {
                    "title": "Рассветная гамма",
                    "descriptors": ["персиковый отблеск", "янтарный свет"],
                }
            ]
        }
    }
    asset.vision_caption = "Цвета: персиковый отблеск, янтарный свет"

    class FixedDatetime(datetime):
        @classmethod
        def utcnow(cls):  # type: ignore[override]
            return datetime(2024, 5, 19, 6, 0, 0)

        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            base = cls.utcnow()
            if tz is None:
                return base
            return base.replace(tzinfo=timezone.utc).astimezone(tz)

    import main as main_module

    monkeypatch.setattr(main_module, "datetime", FixedDatetime)

    assert rubric is not None
    plan, meta = bot._build_flowers_plan(
        rubric,
        [asset],
        weather_block=None,
        channel_id=-150,
    )
    assert "color_palette" in meta["pattern_ids"]
    color_instruction = ""
    for pattern in plan["patterns"]:
        if pattern.get("id") == "color_palette":
            color_instruction = pattern.get("instruction") or ""
            break

    assert color_instruction
    assert "Рассветная гамма" in color_instruction
    assert "персиковый отблеск" in color_instruction
    assert "янтарный свет" in color_instruction
    await bot.close()


@pytest.mark.asyncio
async def test_flowers_plan_skips_recent_duplicate_pattern(monkeypatch, tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    bot.data.upsert_rubric("flowers", "Flowers", config={"enabled": True})
    rubric = bot.data.get_rubric_by_code("flowers")
    asset = _make_asset(3, "Москва", ["rose", "tulip"])

    class FixedDatetime(datetime):
        @classmethod
        def utcnow(cls):  # type: ignore[override]
            return datetime(2024, 5, 18, 6, 0, 0)

        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            base = cls.utcnow()
            if tz is None:
                return base
            return base.replace(tzinfo=timezone.utc).astimezone(tz)

    import main as main_module

    monkeypatch.setattr(main_module, "datetime", FixedDatetime)

    metadata_recent = {
        "pattern_ids": ["weather_focus", "color_palette"],
    }
    metadata_previous = {
        "pattern_ids": ["weather_focus", "micro_engagement_question"],
    }
    assert rubric is not None
    bot.data.record_post_history(1, 99, None, rubric.id, metadata_previous)
    bot.data.record_post_history(1, 100, None, rubric.id, metadata_recent)

    plan, meta = bot._build_flowers_plan(
        rubric,
        [asset],
        weather_block=None,
        channel_id=-200,
    )

    assert meta["pattern_ids"] == ["weather_focus", "texture_focus"]
    assert len(plan["patterns"]) == 2
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

    greeting, hashtags, plan, plan_meta = await bot._generate_flowers_copy(
        rubric,
        [asset],
        channel_id=-200,
    )

    assert len(calls) == 2
    assert "скидка" not in greeting.lower()
    assert hashtags == ["#котопогода", "#цветы"]
    assert isinstance(plan, dict)
    assert isinstance(plan_meta, dict)
    await bot.close()
