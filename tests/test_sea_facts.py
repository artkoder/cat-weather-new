import html
import json
import os
import random
import types
from datetime import datetime, timedelta

import pytest

from facts.loader import Fact
from main import Bot
from openai_client import OpenAIResponse
from tests.fixtures.sea import create_sea_asset, create_stub_image, seed_sea_environment


@pytest.mark.asyncio
async def test_facts_skipped_on_strong_storm(tmp_path):
    os.environ.setdefault("TELEGRAM_BOT_TOKEN", "dummy")
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    try:
        now = datetime.utcnow()
        text, fact_id, info = bot._prepare_sea_fact(
            sea_id=1,
            storm_state="strong_storm",
            enable_facts=True,
            now=now,
        )
        assert text is None
        assert fact_id is None
        assert info.get("reason") == "strong_storm"
    finally:
        await bot.close()


@pytest.mark.asyncio
async def test_facts_nonrepeating(tmp_path, monkeypatch):
    os.environ.setdefault("TELEGRAM_BOT_TOKEN", "dummy")
    bot = Bot("dummy", str(tmp_path / "facts.sqlite"))
    try:
        facts_pool = [Fact(id=f"fact-{idx}", text=f"Факт про Балтику {idx}") for idx in range(3)]
        monkeypatch.setattr("main.load_baltic_facts", lambda: facts_pool)

        base_time = datetime(2024, 1, 1, 9, 0, 0)
        ids = {fact.id for fact in facts_pool}

        _, fact_id1, info1 = bot._prepare_sea_fact(
            sea_id=1,
            storm_state="calm",
            enable_facts=True,
            now=base_time,
            rng=random.Random(0),
        )
        assert fact_id1 in ids
        assert info1["weights"][fact_id1] == pytest.approx(1.0)

        next_time = base_time + timedelta(days=1)
        _, fact_id2, info2 = bot._prepare_sea_fact(
            sea_id=1,
            storm_state="calm",
            enable_facts=True,
            now=next_time,
            rng=random.Random(1),
        )
        assert fact_id2 in ids
        assert fact_id2 != fact_id1
        assert fact_id1 not in info2["candidates"]

        window_time = base_time + timedelta(days=8)
        _, fact_id3, info3 = bot._prepare_sea_fact(
            sea_id=1,
            storm_state="calm",
            enable_facts=True,
            now=window_time,
            rng=random.Random(2),
        )
        assert fact_id3 in ids
        assert info3["weights"].get(fact_id1) == pytest.approx(0.5)

        usage_map = bot.data.get_fact_usage_map()
        assert usage_map[fact_id1][0] >= 1
        assert usage_map[fact_id2][0] >= 1

        day_start = int(base_time.timestamp()) // 86400
        day_next = int(next_time.timestamp()) // 86400
        day_far = int(window_time.timestamp()) // 86400
        rollout_entries = bot.data.get_fact_rollout_range(day_start, end_day=day_far)
        rollout_map = {day: fid for day, fid in rollout_entries}
        assert rollout_map.get(day_start) == fact_id1
        assert rollout_map.get(day_next) == fact_id2
    finally:
        await bot.close()


class StormPersistOpenAI:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.api_key = "stub"

    async def generate_json(self, **kwargs):
        user_prompt = kwargs.get("user_prompt", "")
        assert '"storm_persisting": true' in user_prompt
        self.calls.append(kwargs)
        caption = "Продолжает штормить на море — волны всё ещё упрямы."
        hashtags = ["котопогода", "море", "БалтийскоеМоре"]
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "endpoint": "/v1/storm",
            "request_id": "storm-persist",
        }
        return OpenAIResponse({"caption": caption, "hashtags": hashtags}, usage)


class FactOnceOpenAI:
    def __init__(self) -> None:
        self.payloads: list[dict] = []
        self.api_key = "stub"

    async def generate_json(self, **kwargs):
        user_prompt = kwargs.get("user_prompt", "")
        payload_line = user_prompt.splitlines()[1]
        payload = json.loads(payload_line)
        self.payloads.append(payload)
        fact_sentence = payload.get("fact_sentence")
        base_caption = "Порадую закатом над морем — берег сегодня нежный."
        rephrase = "Балтика этим делится каждый день." if fact_sentence else ""
        caption = base_caption
        if rephrase:
            caption = f"{base_caption} {rephrase}"
        hashtags = ["котопогода", "море", "БалтийскоеМоре"]
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "endpoint": "/v1/fact",
            "request_id": "fact-once",
        }
        return OpenAIResponse({"caption": caption, "hashtags": hashtags}, usage)


@pytest.mark.asyncio
async def test_storm_persisting_flag(tmp_path):
    os.environ.setdefault("TELEGRAM_BOT_TOKEN", "dummy")
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    try:
        rubric = bot.data.get_rubric_by_code("sea")
        assert rubric is not None
        channel_id = -12345
        updated_config = dict(rubric.config or {})
        updated_config.update(
            {
                "enabled": True,
                "sea_id": 1,
                "channel_id": channel_id,
                "test_channel_id": channel_id,
                "enable_facts": True,
            }
        )
        bot.data.save_rubric_config("sea", updated_config)
        rubric = bot.data.get_rubric_by_code("sea")
        assert rubric is not None

        seed_sea_environment(bot, sea_id=1, wave=1.2, wind_speed=10.0)
        image_path = create_stub_image(tmp_path, "storm.jpg")
        create_sea_asset(
            bot,
            rubric_id=rubric.id,
            message_id=501,
            file_name="storm.jpg",
            local_path=image_path,
            tags=["storm"],
            sea_wave_score=8.0,
            photo_sky="mostly_cloudy",
            is_sunset=False,
        )

        bot.openai = StormPersistOpenAI()
        requests: list[dict] = []

        async def fake_api_request(self, method, data=None, files=None):
            requests.append({"method": method, "data": data, "files": files})
            return {"ok": True, "result": {"message_id": 900}}

        bot.api_request = types.MethodType(fake_api_request, bot)

        # Insert yesterday's storm post metadata
        bot.data.record_post_history(
            channel_id,
            400,
            None,
            rubric.id,
            {"storm_state": "storm", "sea_id": 1},
        )
        post_id_row = bot.db.execute("SELECT id FROM posts_history ORDER BY id DESC LIMIT 1").fetchone()
        assert post_id_row is not None
        post_id = post_id_row[0]
        yesterday = (datetime.utcnow() - timedelta(days=1, hours=1)).isoformat()
        bot.db.execute(
            "UPDATE posts_history SET published_at=?, created_at=? WHERE id=?",
            (yesterday, yesterday, post_id),
        )
        bot.db.commit()

        assert await bot._publish_sea(rubric, channel_id) is True
        assert requests, "Expected sendPhoto call"
        caption_html = requests[0]["data"]["caption"]
        caption_text = html.unescape(caption_html)
        assert "Продолжает штормить" in caption_text

        row = bot.db.execute("SELECT metadata FROM posts_history ORDER BY id DESC LIMIT 1").fetchone()
        assert row is not None
        metadata = json.loads(row[0])
        assert metadata.get("storm_persisting") is True
        assert bot.openai.calls, "OpenAI should be invoked"
    finally:
        await bot.close()


@pytest.mark.asyncio
async def test_caption_includes_fact_sentence_once(tmp_path):
    os.environ.setdefault("TELEGRAM_BOT_TOKEN", "dummy")
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    try:
        rubric = bot.data.get_rubric_by_code("sea")
        assert rubric is not None
        channel_id = -67890
        updated_config = dict(rubric.config or {})
        updated_config.update(
            {
                "enabled": True,
                "sea_id": 1,
                "channel_id": channel_id,
                "test_channel_id": channel_id,
                "enable_facts": True,
            }
        )
        bot.data.save_rubric_config("sea", updated_config)
        rubric = bot.data.get_rubric_by_code("sea")
        assert rubric is not None

        seed_sea_environment(bot, sea_id=1, wave=0.3, cloud_cover=10.0)
        image_path = create_stub_image(tmp_path, "sunset.jpg")
        create_sea_asset(
            bot,
            rubric_id=rubric.id,
            message_id=601,
            file_name="sunset.jpg",
            local_path=image_path,
            tags=["sunset"],
            sea_wave_score=1.0,
            photo_sky="sunny",
            is_sunset=True,
        )

        fact_stub = FactOnceOpenAI()
        bot.openai = fact_stub
        requests: list[dict] = []

        async def fake_api_request(self, method, data=None, files=None):
            requests.append({"method": method, "data": data, "files": files})
            return {"ok": True, "result": {"message_id": 901}}

        bot.api_request = types.MethodType(fake_api_request, bot)

        assert await bot._publish_sea(rubric, channel_id) is True
        assert requests, "Expected sendPhoto call"
        caption_html = requests[0]["data"]["caption"]
        caption_text = html.unescape(caption_html)
        rephrase = "Балтика этим делится каждый день."
        assert caption_text.count(rephrase) == 1
        assert "#море #БалтийскоеМоре" in caption_text
        assert "📂 Полюбить 39" in caption_text

        row = bot.db.execute("SELECT metadata FROM posts_history ORDER BY id DESC LIMIT 1").fetchone()
        assert row is not None
        metadata = json.loads(row[0])
        assert metadata.get("fact_id")
        assert metadata.get("fact_text")
        assert fact_stub.payloads
        fact_payload = fact_stub.payloads[0]
        assert metadata["fact_text"] == fact_payload.get("fact_sentence")
    finally:
        await bot.close()
