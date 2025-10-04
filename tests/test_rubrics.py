import json
import os
import sys
from datetime import datetime

from typing import Any

import pytest
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import Bot
from openai_client import OpenAIResponse

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "dummy")


def _insert_rubric(bot: Bot, code: str, config: dict, rubric_id: int = 1) -> None:
    now = datetime.utcnow().isoformat()
    bot.db.execute(
        """
        INSERT INTO rubrics (id, code, title, description, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(code) DO UPDATE SET description=excluded.description, updated_at=excluded.updated_at
        """,
        (
            rubric_id,
            code,
            code.title(),
            json.dumps(config),
            now,
            now,
        ),
    )
    bot.db.commit()


@pytest.mark.asyncio
async def test_rubric_scheduler_enqueues_jobs(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "tz": "+00:00",
        "schedules": [
            {"time": "00:00", "channel_id": -100}
        ],
        "assets": {"categories": ["flowers"], "min": 4, "max": 4},
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    await bot.process_rubric_schedule(reference=datetime.utcnow())
    row = bot.db.execute(
        "SELECT name, status, payload FROM jobs_queue WHERE name='publish_rubric'"
    ).fetchone()
    assert row is not None
    payload = json.loads(row["payload"])
    assert payload["rubric_code"] == "flowers"
    assert payload["channel_id"] == -100
    assert payload["tz_offset"] == "+00:00"
    await bot.close()


@pytest.mark.asyncio
async def test_rubric_scheduler_respects_timezone(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "schedules": [
            {"time": "10:00", "channel_id": -900, "tz": "+03:00"}
        ],
    }
    _insert_rubric(bot, "guess_arch", config, rubric_id=2)
    reference = datetime(2024, 3, 1, 6, 0, 0)
    expected = bot._compute_next_rubric_run(  # type: ignore[attr-defined]
        time_str="10:00", tz_offset="+03:00", days=None, reference=reference
    )
    await bot.process_rubric_schedule(reference=reference)
    row = bot.db.execute(
        "SELECT payload FROM jobs_queue WHERE name='publish_rubric'"
    ).fetchone()
    assert row is not None
    payload = json.loads(row["payload"])
    assert payload["rubric_code"] == "guess_arch"
    assert payload["channel_id"] == -900
    assert payload["scheduled_at"] == expected.isoformat()
    assert payload["tz_offset"] == "+03:00"
    await bot.close()


@pytest.mark.asyncio
async def test_enqueue_rubric_manual_and_test_channels(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "channel_id": -111,
        "test_channel_id": -222,
    }
    _insert_rubric(bot, "flowers", config, rubric_id=5)

    job_id = bot.enqueue_rubric("flowers")
    row = bot.db.execute(
        "SELECT status, payload FROM jobs_queue WHERE id=?", (job_id,)
    ).fetchone()
    assert row is not None
    payload = json.loads(row["payload"])
    assert payload["schedule_key"] == "manual"
    assert payload["channel_id"] == -111
    assert not payload.get("test")
    assert payload["tz_offset"] == "+00:00"
    assert row["status"] == "queued"

    test_job_id = bot.enqueue_rubric("flowers", test=True)
    test_row = bot.db.execute(
        "SELECT status, payload FROM jobs_queue WHERE id=?", (test_job_id,)
    ).fetchone()
    assert test_row is not None
    test_payload = json.loads(test_row["payload"])
    assert test_payload["schedule_key"] == "manual-test"
    assert test_payload["channel_id"] == -222
    assert test_payload["test"] is True
    assert test_payload["tz_offset"] == "+00:00"
    assert test_row["status"] == "queued"
    await bot.close()


@pytest.mark.asyncio
async def test_publish_flowers_removes_assets(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "assets": {"min": 4, "max": 6},
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    now = datetime.utcnow().isoformat()
    for idx in range(4):
        metadata = {"date": now}
        file_meta = {"file_id": f"file{idx}"}
        asset_id = bot.data.save_asset(
            -2000,
            100 + idx,
            None,
            "",
            tg_chat_id=-2000,
            caption="",
            kind="photo",
            file_meta=file_meta,
            metadata=metadata,
            categories=["flowers"],
            rubric_id=1,
        )
        bot.data.update_asset(
            asset_id,
            vision_category="flowers",
            vision_photo_weather="солнечно",
            city=f"Город {idx}",
        )

    calls = []

    async def fake_api(method, data=None, *, files=None):
        calls.append({"method": method, "data": data, "files": files})
        if method == "sendMediaGroup":
            return {"ok": True, "result": [{"message_id": 42}]}
        return {"ok": True}

    bot.api_request = fake_api  # type: ignore
    ok = await bot.publish_rubric("flowers", channel_id=-500)
    assert ok
    assert calls and calls[0]["method"] == "sendMediaGroup"
    delete_calls = [call for call in calls if call["method"] == "deleteMessage"]
    assert len(delete_calls) == 4
    remaining = bot.db.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    assert remaining == 0
    history = bot.db.execute("SELECT metadata FROM posts_history").fetchone()
    assert history is not None
    meta = json.loads(history["metadata"])
    assert meta["rubric_code"] == "flowers"
    assert meta["asset_ids"]
    assert meta["greeting"]
    await bot.close()


@pytest.mark.asyncio
async def test_publish_guess_arch_with_overlays(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    storage = tmp_path / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    bot.asset_storage = storage
    config = {
        "enabled": True,
        "assets": {"min": 4, "max": 4},
        "weather_city": "Kaliningrad",
    }
    _insert_rubric(bot, "guess_arch", config, rubric_id=2)
    bot.db.execute(
        "INSERT OR IGNORE INTO cities (id, name, lat, lon) VALUES (1, 'Kaliningrad', 0, 0)"
    )
    bot.db.execute(
        """
        INSERT OR REPLACE INTO weather_cache_hour (city_id, timestamp, temperature, weather_code, wind_speed, is_day)
        VALUES (1, ?, 12.5, 3, 5.4, 1)
        """,
        (datetime.utcnow().isoformat(),),
    )
    bot.db.commit()
    now = datetime.utcnow().isoformat()
    for idx in range(4):
        image_path = tmp_path / f"asset_{idx}.jpg"
        Image.new("RGB", (400, 300), color=(idx * 40, 10, 10)).save(image_path)
        metadata = {"date": now}
        file_meta = {"file_id": f"gfile{idx}"}
        asset_id = bot.data.save_asset(
            -3000,
            500 + idx,
            None,
            "",
            tg_chat_id=-3000,
            caption="",
            kind="photo",
            file_meta=file_meta,
            metadata=metadata,
            categories=["photo_weather"],
            rubric_id=2,
        )
        bot.data.update_asset(asset_id, local_path=str(image_path))
        bot.data.update_asset(
            asset_id,
            vision_category="architecture",
            vision_arch_view="фасад",
            vision_photo_weather="пасмурно",
        )

    calls = []

    async def fake_api(method, data=None, *, files=None):
        calls.append({"method": method, "data": data, "files": files})
        if method == "sendMediaGroup":
            return {"ok": True, "result": [{"message_id": 99}]}
        return {"ok": True}

    bot.api_request = fake_api  # type: ignore
    ok = await bot.publish_rubric("guess_arch", channel_id=-777)
    assert ok
    assert calls and calls[0]["files"]
    media_payload = calls[0]["data"]["media"]
    assert len(media_payload) == 4
    remaining = bot.db.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    assert remaining == 0
    history = bot.db.execute("SELECT metadata FROM posts_history").fetchone()
    assert history is not None
    meta = json.loads(history["metadata"])
    assert meta["rubric_code"] == "guess_arch"
    assert "weather" in meta
    assert meta["caption"]
    # ensure overlays cleaned up
    numbered_exists = any(storage.glob("*_numbered_*.png"))
    assert not numbered_exists
    delete_calls = [call for call in calls if call["method"] == "deleteMessage"]
    assert len(delete_calls) == 4
    await bot.close()


@pytest.mark.asyncio
async def test_generate_flowers_uses_gpt4o(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {"enabled": True}
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    rubric = bot.data.get_rubric_by_code("flowers")

    calls: list[dict[str, Any]] = []

    class DummyOpenAI:
        def __init__(self) -> None:
            self.api_key = "test"

        async def generate_json(self, **kwargs):  # type: ignore[override]
            calls.append(kwargs)
            return OpenAIResponse(
                {"greeting": "Доброе утро", "hashtags": ["котопогода", "цветы"]},
                10,
                15,
                25,
                "req-1",
            )

    bot.openai = DummyOpenAI()

    greeting, hashtags = await bot._generate_flowers_copy(rubric, ["Москва"], 4)

    assert greeting == "Доброе утро"
    assert hashtags == ["котопогода", "цветы"]
    assert calls, "generate_json was not called"
    request = calls[0]
    assert request["model"] == "gpt-4o"
    assert 0.9 <= request["temperature"] <= 1.1
    assert request["top_p"] == 0.9

    await bot.close()


@pytest.mark.asyncio
async def test_generate_flowers_retries_on_duplicate(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {"enabled": True}
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    rubric = bot.data.get_rubric_by_code("flowers")
    bot.data.record_post_history(1, 1, None, rubric.id, {
        "rubric_code": "flowers",
        "greeting": "Доброе утро",
        "hashtags": ["#котопогода"],
    })

    calls: list[dict[str, Any]] = []

    class DummyOpenAI:
        def __init__(self) -> None:
            self.api_key = "test"

        async def generate_json(self, **kwargs):  # type: ignore[override]
            calls.append(kwargs)
            if len(calls) == 1:
                return OpenAIResponse(
                    {"greeting": "Доброе утро", "hashtags": ["#котопогода"]},
                    5,
                    5,
                    10,
                    "req-first",
                )
            return OpenAIResponse(
                {"greeting": "Привет, друзья", "hashtags": ["#котопогода", "#цветы"]},
                6,
                7,
                13,
                "req-second",
            )

    bot.openai = DummyOpenAI()

    greeting, hashtags = await bot._generate_flowers_copy(rubric, [], 4)

    assert greeting == "Привет, друзья"
    assert hashtags == ["#котопогода", "#цветы"]
    assert len(calls) >= 2

    await bot.close()


@pytest.mark.asyncio
async def test_generate_guess_arch_uses_gpt4o(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {"enabled": True}
    _insert_rubric(bot, "guess_arch", config, rubric_id=2)
    rubric = bot.data.get_rubric_by_code("guess_arch")

    calls: list[dict[str, Any]] = []

    class DummyOpenAI:
        def __init__(self) -> None:
            self.api_key = "test"

        async def generate_json(self, **kwargs):  # type: ignore[override]
            calls.append(kwargs)
            return OpenAIResponse(
                {"caption": "Тест", "hashtags": ["угадай", "архитектура"]},
                8,
                9,
                17,
                "req-3",
            )

    bot.openai = DummyOpenAI()

    caption, hashtags = await bot._generate_guess_arch_copy(rubric, 4, None)

    assert caption == "Тест"
    assert hashtags == ["угадай", "архитектура"]
    assert calls
    request = calls[0]
    assert request["model"] == "gpt-4o"
    assert 0.9 <= request["temperature"] <= 1.1
    assert request["top_p"] == 0.9

    await bot.close()

