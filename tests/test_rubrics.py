import html
import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

from typing import Any

import pytest
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import data_access
from data_access import Asset, Rubric
from jobs import Job
from main import Bot, FLOWERS_PREVIEW_MAX_LENGTH
from openai_client import OpenAIResponse

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "dummy")


FLOWERS_FOOTER_LINK = '<a href="https://t.me/addlist/sW-rkrslxqo1NTVi">📂 Полюбить 39</a>'


def _insert_rubric(bot: Bot, code: str, config: dict, rubric_id: int = 1) -> None:
    bot.data.upsert_rubric(code, code.title(), config=config)


def _seed_weather(bot: Bot) -> None:
    bot.db.execute(
        "INSERT OR IGNORE INTO cities (id, name, lat, lon) VALUES (?, ?, ?, ?)",
        (1, "Kaliningrad", 54.7104, 20.4522),
    )
    timestamp = datetime.utcnow().isoformat()
    today = datetime.utcnow().date().isoformat()
    yesterday = (datetime.utcnow() - timedelta(days=1)).date().isoformat()
    bot.db.execute(
        "INSERT OR REPLACE INTO weather_cache_hour (city_id, timestamp, temperature, weather_code, wind_speed, is_day) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (1, timestamp, 12.0, 1, 3.0, 1),
    )
    bot.db.execute(
        "INSERT OR REPLACE INTO weather_cache_day (city_id, day, temperature, weather_code, wind_speed) "
        "VALUES (?, ?, ?, ?, ?)",
        (1, today, 12.0, 1, 3.0),
    )
    bot.db.execute(
        "INSERT OR REPLACE INTO weather_cache_day (city_id, day, temperature, weather_code, wind_speed) "
        "VALUES (?, ?, ?, ?, ?)",
        (1, yesterday, 9.0, 2, 5.0),
    )
    bot.db.execute(
        "INSERT OR IGNORE INTO seas (id, name, lat, lon) VALUES (?, ?, ?, ?)",
        (1, "Балтика", 54.95, 20.2),
    )
    bot.db.execute(
        "INSERT OR REPLACE INTO sea_cache (sea_id, updated, current, morning, day, evening, night, wave, morning_wave, day_wave, evening_wave, night_wave) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (1, timestamp, 8.5, None, None, None, None, 0.4, None, None, None, None),
    )
    bot.db.commit()


def _seed_two_day_hourly_weather(bot: Bot) -> None:
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    today = now.date()
    yesterday = (now - timedelta(days=1)).date()
    city_id = 1
    bot.db.execute(
        "INSERT OR IGNORE INTO cities (id, name, lat, lon) VALUES (?, ?, ?, ?)",
        (city_id, "Kaliningrad", 54.7104, 20.4522),
    )

    for day_date in (yesterday, today):
        base_dt = datetime.combine(day_date, datetime.min.time())
        is_today = day_date == today
        for hour in range(24):
            ts = base_dt + timedelta(hours=hour)
            if is_today:
                if 6 <= hour < 12:
                    code = 0  # sunny
                    temp = 12.0 + hour * 0.1
                    wind = 3.2
                elif 12 <= hour < 18:
                    code = 3  # overcast
                    temp = 15.0 + hour * 0.1
                    wind = 4.4
                elif 18 <= hour < 24:
                    code = 45  # fog
                    temp = 9.0 + hour * 0.1
                    wind = 2.6
                else:
                    code = 0
                    temp = 10.0 + hour * 0.05
                    wind = 2.2
            else:
                if 6 <= hour < 12:
                    code = 61  # rain
                    temp = 6.0 + hour * 0.1
                    wind = 5.4
                elif 12 <= hour < 18:
                    code = 71  # snow
                    temp = 1.0 + hour * 0.1
                    wind = 6.0
                elif 18 <= hour < 24:
                    code = 2  # partly cloudy
                    temp = 3.5 + hour * 0.1
                    wind = 3.6
                else:
                    code = 61
                    temp = 4.5 + hour * 0.05
                    wind = 4.8
            is_day = 1 if 6 <= hour < 22 else 0
            bot.db.execute(
                "INSERT OR REPLACE INTO weather_cache_hour (city_id, timestamp, temperature, weather_code, wind_speed, is_day) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    city_id,
                    ts.isoformat(),
                    temp,
                    code,
                    wind,
                    is_day,
                ),
            )

    bot.db.execute(
        "INSERT OR REPLACE INTO weather_cache_day (city_id, day, temperature, weather_code, wind_speed) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            city_id,
            today.isoformat(),
            14.0,
            3,
            4.0,
        ),
    )
    bot.db.execute(
        "INSERT OR REPLACE INTO weather_cache_day (city_id, day, temperature, weather_code, wind_speed) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            city_id,
            yesterday.isoformat(),
            5.5,
            61,
            5.2,
        ),
    )
    bot.db.commit()


def _seed_city(
    bot: Bot,
    *,
    city_id: int,
    name: str,
    lat: float = 0.0,
    lon: float = 0.0,
    hourly_temperature: float = 10.0,
    hourly_wind: float = 2.0,
    today_temperature: float = 9.0,
    yesterday_temperature: float = 6.0,
    today_wind: float = 3.0,
    yesterday_wind: float = 2.0,
) -> None:
    timestamp = datetime.utcnow().isoformat()
    today = datetime.utcnow().date().isoformat()
    yesterday = (datetime.utcnow() - timedelta(days=1)).date().isoformat()
    bot.db.execute(
        "INSERT OR IGNORE INTO cities (id, name, lat, lon) VALUES (?, ?, ?, ?)",
        (city_id, name, lat, lon),
    )
    bot.db.execute(
        "INSERT OR REPLACE INTO weather_cache_hour (city_id, timestamp, temperature, weather_code, wind_speed, is_day) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (city_id, timestamp, hourly_temperature, 1, hourly_wind, 1),
    )
    bot.db.execute(
        "INSERT OR REPLACE INTO weather_cache_day (city_id, day, temperature, weather_code, wind_speed) "
        "VALUES (?, ?, ?, ?, ?)",
        (city_id, today, today_temperature, 1, today_wind),
    )
    bot.db.execute(
        "INSERT OR REPLACE INTO weather_cache_day (city_id, day, temperature, weather_code, wind_speed) "
        "VALUES (?, ?, ?, ?, ?)",
        (city_id, yesterday, yesterday_temperature, 2, yesterday_wind),
    )
    bot.db.commit()


@pytest.mark.asyncio
async def test_build_flowers_plan_uses_single_rotating_pattern(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {"enabled": True, "assets": {"min": 1, "max": 1}}
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    _seed_weather(bot)

    today = datetime.utcnow().date().isoformat()
    yesterday = (datetime.utcnow() - timedelta(days=1)).date().isoformat()
    bot.db.execute(
        "UPDATE weather_cache_day SET temperature=?, wind_speed=? WHERE city_id=? AND day=?",
        (
            json.dumps({"temperature_2m_mean": 4.0, "temperature_2m_max": 8.0}),
            json.dumps({"wind_speed_10m_max": 4.8}),
            1,
            today,
        ),
    )
    bot.db.execute(
        "UPDATE weather_cache_day SET temperature=?, wind_speed=? WHERE city_id=? AND day=?",
        (
            json.dumps({"temperature_2m_mean": 9.0}),
            json.dumps({"wind_speed_10m_mean": 6.0}),
            1,
            yesterday,
        ),
    )
    bot.db.commit()

    rubric = bot.data.get_rubric_by_code("flowers")
    assert rubric is not None
    weather_block = bot._compose_flowers_weather_block(["Kaliningrad"], rubric)
    assert weather_block is not None
    city_snapshot = weather_block.get("city")
    assert city_snapshot is not None
    summary = weather_block.get("trend_summary")
    assert (
        summary
        == "свежесть бодрит — около 4°C и ветер стал спокойнее — около 5 м/с"
    )
    assert city_snapshot["trend_temperature_value"] == 4.0
    assert city_snapshot["trend_wind_value"] == 4.8
    assert city_snapshot["trend_temperature_previous_value"] == 9.0
    assert city_snapshot["trend_wind_previous_value"] == 6.0
    assert "12°C" not in summary
    assert "12°C" in (city_snapshot.get("detail") or "")
    assert "ветер 3 м/с" in (city_snapshot.get("detail") or "")

    await bot.close()


@pytest.mark.asyncio
async def test_flowers_weather_block_uses_configured_city(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "assets": {"min": 1, "max": 1},
        "weather_city": "Minsk",
        "weather_city_id": 2,
    }
    _insert_rubric(bot, "flowers", config, rubric_id=3)
    _seed_weather(bot)
    _seed_city(
        bot,
        city_id=2,
        name="Minsk",
        lat=53.9,
        lon=27.5667,
        hourly_temperature=5.0,
        hourly_wind=1.5,
        today_temperature=7.0,
        yesterday_temperature=4.0,
        today_wind=2.2,
        yesterday_wind=3.8,
    )

    rubric = bot.data.get_rubric_by_code("flowers")
    assert rubric is not None
    weather_block = bot._compose_flowers_weather_block([], rubric)
    assert weather_block is not None
    city_snapshot = weather_block.get("city")
    assert city_snapshot is not None
    assert city_snapshot.get("name") == "Minsk"
    today_metrics = weather_block.get("today")
    assert isinstance(today_metrics, dict)
    assert today_metrics.get("temperature") == pytest.approx(7.0)
    assert today_metrics.get("wind_speed") == pytest.approx(2.2)
    yesterday_metrics = weather_block.get("yesterday")
    assert isinstance(yesterday_metrics, dict)
    assert yesterday_metrics.get("temperature") == pytest.approx(4.0)
    assert yesterday_metrics.get("wind_speed") == pytest.approx(3.8)

    await bot.close()


@pytest.mark.asyncio
async def test_flowers_prompt_includes_dayparts_for_two_days(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    _insert_rubric(bot, "flowers", {"enabled": True, "assets": {"min": 0, "max": 2}}, rubric_id=7)
    _seed_two_day_hourly_weather(bot)

    rubric = bot.data.get_rubric_by_code("flowers")
    assert rubric is not None

    weather_block = bot._compose_flowers_weather_block([], rubric)
    assert weather_block is not None

    plan, plan_meta = bot._build_flowers_plan(rubric, [], weather_block, channel_id=None)
    today_parts = plan["weather"]["today"].get("parts", {})
    yesterday_parts = plan["weather"]["yesterday"].get("parts", {})
    assert today_parts == {
        "morning": "солнечно",
        "day": "пасмурно",
        "evening": "туман",
    }
    assert yesterday_parts == {
        "morning": "дождь",
        "day": "снег",
        "evening": "переменная облачность",
    }

    prompt_payload = bot._build_flowers_prompt_payload(plan, plan_meta)
    user_prompt = prompt_payload["user_prompt"]
    assert '"today"' in user_prompt
    assert '"yesterday"' in user_prompt
    for snippet in (
        '"morning": "солнечно"',
        '"day": "пасмурно"',
        '"evening": "туман"',
        '"morning": "дождь"',
        '"day": "снег"',
        '"evening": "переменная облачность"',
    ):
        assert snippet in user_prompt

    await bot.close()


@pytest.mark.asyncio
async def test_rubric_deletion_disabled(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    with pytest.raises(NotImplementedError):
        bot.data.delete_rubric("flowers")

    await bot.close()


@pytest.mark.asyncio
async def test_default_rubrics_created_on_boot(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    rubrics = {rubric.code: rubric for rubric in bot.data.list_rubrics()}
    assert {"flowers", "guess_arch"}.issubset(rubrics.keys())
    for code in ("flowers", "guess_arch"):
        config = rubrics[code].config
        assert config.get("enabled") is False
        assert config.get("schedules") == []
    await bot.close()


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
async def test_vision_job_assigns_rubric_id(tmp_path):
    bot = Bot("test-token", str(tmp_path / "db.sqlite"))
    bot.openai.api_key = "test"  # type: ignore[assignment]

    flowers_rubric = bot.data.get_rubric_by_code("flowers")
    assert flowers_rubric is not None

    image_path = tmp_path / "flower.jpg"
    Image.new("RGB", (32, 32), color="red").save(image_path)

    file_meta = {"file_id": "file123", "file_unique_id": "uniq123", "mime_type": "image/jpeg"}
    asset_id = bot.data.save_asset(
        -1500,
        10,
        None,
        "",
        tg_chat_id=-1500,
        caption="",
        kind="photo",
        file_meta=file_meta,
        metadata={},
    )
    bot.data.update_asset(asset_id, local_path=str(image_path))

    async def fake_api_request(method, data=None, *, files=None):
        if method == "copyMessage":
            return {"ok": True, "result": {"message_id": 600}}
        return {"ok": True}

    async def fake_record_usage(*args, **kwargs):
        return None

    async def fake_classify_image(*, model, system_prompt, user_prompt, image_path, schema):
        return OpenAIResponse(
            content={
                "framing": "wide",
                "architecture_close_up": False,
                "architecture_wide": False,
                "weather_image": "sunny",
                "season_guess": "summer",
                "arch_style": None,
                "caption": "Field of tulips",
                "objects": ["tulips"],
                "landmarks": [],
                "tags": ["flowers"],
                "arch_view": False,
                "is_outdoor": True,
                "guess_country": None,
                "guess_city": None,
                "location_confidence": 0.5,
                "safety": {"nsfw": False, "reason": "безопасно"},
                "category": "flowers",
            },
            usage={
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "request_id": "req-flowers",
            },
        )

    bot.api_request = fake_api_request  # type: ignore[assignment]
    bot._record_openai_usage = fake_record_usage  # type: ignore[assignment]
    bot.openai.classify_image = fake_classify_image  # type: ignore[assignment]

    now = datetime.utcnow()
    job = Job(
        id=1,
        name="vision",
        payload={"asset_id": asset_id, "tz_offset": "+00:00"},
        status="queued",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=now,
        updated_at=now,
    )

    await bot._job_vision_locked(job)

    updated_asset = bot.data.get_asset(asset_id)
    assert updated_asset is not None
    assert updated_asset.vision_category == "flowers"
    assert updated_asset.rubric_id == flowers_rubric.id

    assets = bot.data.fetch_assets_by_vision_category("flowers", rubric_id=flowers_rubric.id)
    assert any(asset.id == asset_id for asset in assets)

    await bot.close()


@pytest.mark.asyncio
async def test_vision_job_handles_singular_flower_tag(tmp_path):
    bot = Bot("test-token", str(tmp_path / "db.sqlite"))
    bot.openai.api_key = "test"  # type: ignore[assignment]

    flowers_rubric = bot.data.get_rubric_by_code("flowers")
    assert flowers_rubric is not None

    image_path = tmp_path / "flower.jpg"
    Image.new("RGB", (32, 32), color="red").save(image_path)

    file_meta = {
        "file_id": "file-singular",
        "file_unique_id": "uniq-singular",
        "mime_type": "image/jpeg",
    }
    asset_id = bot.data.save_asset(
        -1600,
        20,
        None,
        "",
        tg_chat_id=-1600,
        caption="",
        kind="photo",
        file_meta=file_meta,
        metadata={},
    )
    bot.data.update_asset(asset_id, local_path=str(image_path))

    async def fake_api_request(method, data=None, *, files=None):
        if method == "copyMessage":
            return {"ok": True, "result": {"message_id": 601}}
        return {"ok": True}

    async def fake_record_usage(*args, **kwargs):
        return None

    async def fake_classify_image(*, model, system_prompt, user_prompt, image_path, schema):
        return OpenAIResponse(
            content={
                "framing": "medium",
                "architecture_close_up": False,
                "architecture_wide": False,
                "weather_image": "sunny",
                "season_guess": "summer",
                "arch_style": None,
                "caption": "A single rose",
                "objects": ["rose"],
                "landmarks": [],
                "tags": ["flower"],
                "arch_view": False,
                "is_outdoor": True,
                "guess_country": None,
                "guess_city": None,
                "location_confidence": 0.6,
                "safety": {"nsfw": False, "reason": "безопасно"},
                "category": "flower",
            },
            usage={
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "request_id": "req-flower",
            },
        )

    bot.api_request = fake_api_request  # type: ignore[assignment]
    bot._record_openai_usage = fake_record_usage  # type: ignore[assignment]
    bot.openai.classify_image = fake_classify_image  # type: ignore[assignment]

    now = datetime.utcnow()
    job = Job(
        id=2,
        name="vision",
        payload={"asset_id": asset_id, "tz_offset": "+00:00"},
        status="queued",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=now,
        updated_at=now,
    )

    await bot._job_vision_locked(job)

    updated_asset = bot.data.get_asset(asset_id)
    assert updated_asset is not None
    assert updated_asset.vision_category == "flowers"
    assert updated_asset.rubric_id == flowers_rubric.id
    assert updated_asset.vision_flower_varieties == ["rose"]

    await bot.close()


@pytest.mark.asyncio
async def test_fetch_assets_includes_singular_flower_category(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    asset_id = bot.data.save_asset(
        -2000,
        50,
        None,
        None,
        tg_chat_id=-2000,
        caption="",
        kind="photo",
        categories=["flowers"],
    )

    bot.db.execute(
        "UPDATE assets SET vision_category=?, vision_flower_varieties=?, updated_at=? WHERE id=?",
        ("flower", json.dumps(["rose"]), datetime.utcnow().isoformat(), asset_id),
    )
    bot.db.commit()

    assets = bot.data.fetch_assets_by_vision_category("flowers")
    assert any(asset.id == asset_id for asset in assets)
    matched = next(asset for asset in assets if asset.id == asset_id)
    assert matched.vision_category == "flowers"
    assert matched.vision_flower_varieties == ["rose"]

    await bot.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("asset_count", "expected_method"),
    [
        (1, "sendPhoto"),
        (2, "sendMediaGroup"),
        (3, "sendMediaGroup"),
    ],
)
async def test_publish_flowers_varied_asset_counts(tmp_path, asset_count, expected_method):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "assets": {"min": asset_count, "max": asset_count},
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    _seed_weather(bot)
    now = datetime.utcnow().isoformat()
    file_ids: list[str] = []
    for idx in range(asset_count):
        metadata = {"date": now}
        file_id = f"file{idx}"
        file_ids.append(file_id)
        file_meta = {"file_id": file_id}
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

    calls: list[dict[str, Any]] = []

    async def fake_api(method, data=None, *, files=None):
        calls.append({"method": method, "data": data, "files": files})
        if method == "sendPhoto":
            return {"ok": True, "result": {"message_id": 42}}
        if method == "sendMediaGroup":
            return {"ok": True, "result": [{"message_id": 42}]}
        return {"ok": True}

    bot.api_request = fake_api  # type: ignore[assignment]
    ok = await bot.publish_rubric("flowers", channel_id=-500)
    assert ok
    send_call = next(
        call for call in calls if call["method"] in {"sendPhoto", "sendMediaGroup"}
    )
    assert send_call["method"] == expected_method
    data = send_call["data"]
    assert data is not None
    if expected_method == "sendPhoto":
        assert data["photo"] == file_ids[0]
        caption = data.get("caption", "")
        assert data.get("parse_mode") == "HTML"
    else:
        media_payload = data["media"]
        assert isinstance(media_payload, list)
        assert len(media_payload) == asset_count
        caption = media_payload[0].get("caption", "")
        assert media_payload[0].get("parse_mode") == "HTML"
    delete_calls = [call for call in calls if call["method"] == "deleteMessage"]
    assert len(delete_calls) == asset_count
    remaining = bot.db.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    assert remaining == 0
    history = bot.db.execute("SELECT metadata FROM posts_history").fetchone()
    assert history is not None
    meta = json.loads(history["metadata"])
    assert meta["rubric_code"] == "flowers"
    assert meta["asset_ids"]
    greeting = str(meta.get("greeting") or "").strip()
    assert greeting
    assert meta.get("pattern_ids"), "pattern ids should be stored in metadata"
    assert isinstance(meta.get("plan"), dict)
    cities_meta = list(meta.get("cities") or [])
    city_hashtags: list[str] = []
    for city in cities_meta:
        tag = bot._normalize_city_hashtag(city)
        assert tag
        city_hashtags.append(tag)
    hashtags_combined = list(meta.get("hashtags") or [])
    assert hashtags_combined[: len(city_hashtags)] == city_hashtags
    trailing_only = hashtags_combined[len(city_hashtags) :]
    weather_meta = meta.get("weather")
    assert isinstance(weather_meta, dict)
    weather_today = str(meta.get("weather_today_line") or "").strip()
    weather_yesterday = str(meta.get("weather_yesterday_line") or "").strip()
    assert weather_today
    assert "Сегодня" in weather_today
    assert "Вчера" in weather_yesterday
    today_metrics = weather_meta.get("today")
    yesterday_metrics = weather_meta.get("yesterday")
    assert isinstance(today_metrics, dict)
    assert isinstance(yesterday_metrics, dict)
    assert today_metrics.get("temperature") is not None
    assert yesterday_metrics.get("temperature") is not None
    assert meta.get("weather_line") == weather_today
    preview_parts = [greeting]
    if city_hashtags:
        preview_parts.append(" ".join(city_hashtags))
    if trailing_only:
        preview_parts.append(" ".join(trailing_only))
    expected_preview = "\n\n".join(preview_parts)
    expected_publish = html.escape(expected_preview) + "\n\n" + FLOWERS_FOOTER_LINK
    assert caption == expected_publish
    assert weather_today not in expected_preview
    assert html.escape(weather_today) not in caption
    assert FLOWERS_FOOTER_LINK in caption
    await bot.close()


@pytest.mark.asyncio
async def test_flowers_preview_notifies_when_not_enough_assets(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "channel_id": -500,
        "assets": {"min": 5, "max": 6},
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)

    calls: list[tuple[str, dict[str, Any] | None]] = []

    async def fake_api(method, data=None, *, files=None):  # type: ignore[override]
        calls.append((method, data))
        return {"ok": True}

    bot.api_request = fake_api  # type: ignore[assignment]

    payload = {
        "rubric_code": "flowers",
        "channel_id": -500,
        "initiator_id": 777,
        "test": False,
    }
    now = datetime.utcnow()
    job = Job(
        id=1,
        name="publish_rubric",
        payload=payload,
        status="queued",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=now,
        updated_at=now,
    )

    await bot._job_publish_rubric(job)

    send_calls = [call for call in calls if call[0] == "sendMessage"]
    assert send_calls, "Expected operator notification"
    message = send_calls[0][1]
    assert message is not None
    assert message["chat_id"] == 777
    assert "минимальн" in message["text"].lower()
    assert "5" in message["text"]

    ok = await bot.publish_rubric("flowers", channel_id=-500)
    assert not ok

    await bot.close()


@pytest.mark.asyncio
async def test_flowers_preview_single_photo_paths(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "channel_id": -500,
        "test_channel_id": -600,
        "assets": {"min": 1, "max": 1},
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    _seed_weather(bot)

    now = datetime.utcnow().isoformat()
    file_meta = {"file_id": "single-file"}
    asset_id = bot.data.save_asset(
        -2100,
        915,
        None,
        "",
        tg_chat_id=-2100,
        caption="",
        kind="photo",
        file_meta=file_meta,
        metadata={"date": now},
        categories=["flowers"],
        rubric_id=1,
    )
    bot.data.update_asset(
        asset_id,
        vision_category="flowers",
        vision_photo_weather="солнечно",
        city="Москва",
    )

    bot.db.execute(
        "INSERT OR REPLACE INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, '+00:00')",
        (1234, "tester"),
    )
    bot.db.commit()

    calls: list[tuple[str, dict[str, Any] | None]] = []
    posted_link: dict[str, str] = {}

    async def fake_api(method, data=None, *, files=None):  # type: ignore[override]
        calls.append((method, data))
        if method == "sendPhoto":
            if data and data.get("chat_id") == 1234:
                return {"ok": True, "result": {"message_id": 50}}
            if data and data.get("chat_id") == -500:
                posted_link["url"] = bot.post_url(-500, 75)
                return {"ok": True, "result": {"message_id": 75}}
        if method == "sendMessage":
            return {"ok": True, "result": {"message_id": 200}}
        if method in {"deleteMessage", "editMessageText", "answerCallbackQuery"}:
            return {"ok": True}
        return {"ok": True}

    bot.api_request = fake_api  # type: ignore[assignment]

    await bot.publish_rubric("flowers", channel_id=-500, initiator_id=1234)

    state = bot.pending_flowers_previews.get(1234)
    assert state is not None
    assert state.get("media_message_ids") == [50]
    preview_send = [call for call in calls if call[0] == "sendPhoto" and call[1] and call[1].get("chat_id") == 1234]
    assert preview_send, "Expected preview to use sendPhoto"
    preview_payload = preview_send[0][1]
    assert preview_payload is not None
    assert preview_payload.get("parse_mode") is None

    preview_caption = str(state.get("preview_caption") or "")
    publish_caption = str(state.get("publish_caption") or "")
    publish_mode = state.get("publish_parse_mode")
    assert preview_payload.get("caption") == preview_caption
    assert FLOWERS_FOOTER_LINK not in preview_caption
    assert FLOWERS_FOOTER_LINK in publish_caption
    assert publish_caption.endswith(FLOWERS_FOOTER_LINK)
    assert publish_mode == "HTML"
    weather_today = str(state.get("weather_today_line") or "")
    weather_yesterday = str(state.get("weather_yesterday_line") or "")
    assert weather_today
    assert weather_today not in preview_caption
    assert html.escape(weather_today) not in publish_caption
    assert "Вчера:" in weather_yesterday
    assert state.get("weather_line") == weather_today
    if preview_caption:
        assert publish_caption.startswith(html.escape(preview_caption.strip()))

    await bot._handle_flowers_preview_callback(1234, "send_main", {"id": "cb-photo"})

    final_send = [call for call in calls if call[0] == "sendPhoto" and call[1] and call[1].get("chat_id") == -500]
    assert final_send, "Expected finalize to use sendPhoto"
    final_payload = final_send[0][1]
    assert final_payload is not None
    assert final_payload.get("caption") == publish_caption
    assert final_payload.get("parse_mode") == "HTML"
    confirmations = [
        data
        for method, data in calls
        if method == "sendMessage" and data and data.get("chat_id") == 1234
    ]
    assert confirmations, "Expected confirmation messages to be sent to operator"
    expected_link = posted_link.get("url")
    assert expected_link is not None
    assert any(expected_link in str(msg.get("text") or "") for msg in confirmations)
    assert bot.pending_flowers_previews.get(1234) is None
    history = bot.db.execute("SELECT metadata FROM posts_history").fetchone()
    assert history is not None
    meta = json.loads(history["metadata"])
    assert meta.get("weather_today_line") == weather_today
    assert meta.get("weather_yesterday_line") == weather_yesterday
    assert meta.get("weather_line") == weather_today
    assert isinstance(meta.get("weather"), dict)
    assert meta.get("pattern_ids"), "pattern ids should be persisted"
    assert isinstance(meta.get("plan"), dict)
    remaining = bot.db.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    assert remaining == 0

    await bot.close()


@pytest.mark.asyncio
async def test_flowers_preview_document_media_paths(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "channel_id": -500,
        "test_channel_id": -600,
        "assets": {"min": 1, "max": 1},
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    _seed_weather(bot)

    now = datetime.utcnow().isoformat()
    file_meta = {"file_id": "doc-file", "file_name": "flower.pdf"}
    asset_id = bot.data.save_asset(
        -2100,
        916,
        None,
        "",
        tg_chat_id=-2100,
        caption="",
        kind="document",
        file_meta=file_meta,
        metadata={"date": now},
        categories=["flowers"],
        rubric_id=1,
    )
    bot.data.update_asset(
        asset_id,
        vision_category="flowers",
        vision_photo_weather="солнечно",
        city="Москва",
    )

    bot.db.execute(
        "INSERT OR REPLACE INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, '+00:00')",
        (1234, "tester"),
    )
    bot.db.commit()

    calls: list[tuple[str, dict[str, Any] | None]] = []
    posted_link: dict[str, str] = {}

    async def fake_api(method, data=None, *, files=None):  # type: ignore[override]
        calls.append((method, data))
        if method == "sendDocument":
            if data and data.get("chat_id") == 1234:
                return {"ok": True, "result": {"message_id": 60}}
            if data and data.get("chat_id") == -500:
                posted_link["url"] = bot.post_url(-500, 85)
                return {"ok": True, "result": {"message_id": 85}}
        if method == "sendMessage":
            return {"ok": True, "result": {"message_id": 210}}
        if method in {"deleteMessage", "editMessageText", "answerCallbackQuery"}:
            return {"ok": True}
        return {"ok": True}

    bot.api_request = fake_api  # type: ignore[assignment]

    await bot.publish_rubric("flowers", channel_id=-500, initiator_id=1234)

    state = bot.pending_flowers_previews.get(1234)
    assert state is not None
    assert state.get("media_message_ids") == [60]
    assert state.get("asset_kinds") == ["document"]

    preview_send = [
        call for call in calls if call[0] == "sendDocument" and call[1] and call[1].get("chat_id") == 1234
    ]
    assert preview_send, "Expected preview to use sendDocument"
    preview_payload = preview_send[0][1]
    assert preview_payload is not None
    assert preview_payload.get("parse_mode") is None

    preview_caption = str(state.get("preview_caption") or "")
    publish_caption = str(state.get("publish_caption") or "")
    publish_mode = state.get("publish_parse_mode")
    assert preview_payload.get("caption") == preview_caption
    assert FLOWERS_FOOTER_LINK not in preview_caption
    assert FLOWERS_FOOTER_LINK in publish_caption
    assert publish_caption.endswith(FLOWERS_FOOTER_LINK)
    assert publish_mode == "HTML"
    weather_today = str(state.get("weather_today_line") or "")
    weather_yesterday = str(state.get("weather_yesterday_line") or "")
    assert weather_today
    assert weather_today not in preview_caption
    assert html.escape(weather_today) not in publish_caption
    assert "Вчера:" in weather_yesterday
    assert state.get("weather_line") == weather_today
    if preview_caption:
        assert publish_caption.startswith(html.escape(preview_caption.strip()))

    await bot._handle_flowers_preview_callback(1234, "send_main", {"id": "cb-doc"})

    final_send = [
        call for call in calls if call[0] == "sendDocument" and call[1] and call[1].get("chat_id") == -500
    ]
    assert final_send, "Expected finalize to use sendDocument"
    final_payload = final_send[0][1]
    assert final_payload is not None
    assert final_payload.get("caption") == publish_caption
    assert final_payload.get("parse_mode") == "HTML"
    confirmations = [
        data
        for method, data in calls
        if method == "sendMessage" and data and data.get("chat_id") == 1234
    ]
    assert confirmations, "Expected confirmation messages to be sent to operator"
    expected_link = posted_link.get("url")
    assert expected_link is not None
    assert any(expected_link in str(msg.get("text") or "") for msg in confirmations)
    assert bot.pending_flowers_previews.get(1234) is None
    remaining = bot.db.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    assert remaining == 0

    await bot.close()


@pytest.mark.asyncio
async def test_flowers_preview_document_with_image_filename(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "channel_id": -500,
        "test_channel_id": -600,
        "assets": {"min": 1, "max": 1},
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    _seed_weather(bot)

    now = datetime.utcnow().isoformat()
    file_meta = {"file_id": "doc-photo", "file_name": "flower.jpg"}
    asset_id = bot.data.save_asset(
        -2100,
        917,
        None,
        "",
        tg_chat_id=-2100,
        caption="",
        kind="document",
        file_meta=file_meta,
        metadata={"date": now},
        categories=["flowers"],
        rubric_id=1,
    )
    bot.data.update_asset(
        asset_id,
        vision_category="flowers",
        vision_photo_weather="солнечно",
        city="Москва",
    )

    bot.db.execute(
        "INSERT OR REPLACE INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, '+00:00')",
        (1234, "tester"),
    )
    bot.db.commit()

    call_log: list[dict[str, Any]] = []
    multipart_calls: list[dict[str, Any]] = []
    posted_link: dict[str, str] = {}

    async def fake_api(method, data=None, *, files=None):  # type: ignore[override]
        call_log.append({"method": method, "data": data, "files": files})
        if method == "getFile":
            return {"ok": True, "result": {"file_path": "downloads/flower.jpg"}}
        if method == "sendPhoto":
            chat_id = data.get("chat_id") if isinstance(data, dict) else None
            message_id = 905 if chat_id == 1234 else 915
            if chat_id == -500:
                posted_link["url"] = bot.post_url(-500, message_id)
            return {"ok": True, "result": {"message_id": message_id}}
        if method == "sendMessage":
            return {
                "ok": True,
                "result": {"message_id": 320 if data and data.get("chat_id") == -500 else 310},
            }
        if method in {"deleteMessage", "editMessageText", "answerCallbackQuery"}:
            return {"ok": True}
        return {"ok": True}

    async def fake_api_multipart(method, data=None, *, files=None):  # type: ignore[override]
        multipart_calls.append({"method": method, "data": data, "files": files})
        photo_payload = {
            "file_id": "photo-new",
            "file_unique_id": "uniq-photo-new",
            "width": 800,
            "height": 600,
            "file_size": 2048,
        }
        message_id = 610 if data and data.get("chat_id") == 1234 else 960
        return {
            "ok": True,
            "result": {"message_id": message_id, "photo": [photo_payload]},
        }

    async def fake_download(file_id, dest_path):  # type: ignore[override]
        dest = Path(dest_path)
        dest.write_bytes(b"stub")
        return dest

    def fake_prepare(local_path):  # type: ignore[override]
        path = Path(local_path)
        if not path.exists():
            path.write_bytes(b"stub")
        return path, None, "flower.jpg", "image/jpeg", "original", "image/jpeg", 4

    bot.api_request = fake_api  # type: ignore[assignment]
    bot.api_request_multipart = fake_api_multipart  # type: ignore[assignment]
    bot._download_file = fake_download  # type: ignore[assignment]
    bot._prepare_photo_for_upload = fake_prepare  # type: ignore[assignment]

    await bot.publish_rubric("flowers", channel_id=-500, initiator_id=1234)

    state = bot.pending_flowers_previews.get(1234)
    assert state is not None
    assert state.get("media_message_ids") == [610]
    assert state.get("asset_kinds") == ["photo"]
    assert state.get("file_ids") == ["photo-new"]

    preview_caption = str(state.get("preview_caption") or "")
    publish_caption = str(state.get("publish_caption") or "")
    publish_mode = state.get("publish_parse_mode")
    assert FLOWERS_FOOTER_LINK not in preview_caption
    assert FLOWERS_FOOTER_LINK in publish_caption
    assert publish_caption.endswith(FLOWERS_FOOTER_LINK)
    assert publish_mode == "HTML"
    weather_today = str(state.get("weather_today_line") or "")
    weather_yesterday = str(state.get("weather_yesterday_line") or "")
    assert weather_today
    assert weather_today not in preview_caption
    assert html.escape(weather_today) not in publish_caption
    assert "Вчера:" in weather_yesterday
    assert state.get("weather_line") == weather_today
    if preview_caption:
        assert publish_caption.startswith(html.escape(preview_caption.strip()))

    preview_photos = [
        call
        for call in multipart_calls
        if call["method"] == "sendPhoto" and call["data"] and call["data"].get("chat_id") == 1234
    ]
    assert preview_photos, "Expected preview to upload via sendPhoto"
    preview_payload = preview_photos[0]["data"]
    assert preview_payload is not None
    assert preview_payload.get("parse_mode") is None
    assert preview_payload.get("caption") == preview_caption

    updated_asset = bot.data.get_asset(asset_id)
    assert updated_asset is not None
    assert updated_asset.kind == "photo"
    assert updated_asset.file_id == "photo-new"
    assert updated_asset.metadata.get("original_document_file_id") == "doc-photo"

    await bot._handle_flowers_preview_callback(1234, "send_main", {"id": "cb-photo"})

    final_photos = [
        call
        for call in call_log
        if call["method"] == "sendPhoto" and call["data"] and call["data"].get("chat_id") == -500
    ]
    assert final_photos, "Expected finalize to use sendPhoto"
    final_payload = final_photos[0]["data"]
    assert final_payload["photo"] == "photo-new"
    assert final_payload.get("caption") == publish_caption
    assert final_payload.get("parse_mode") == "HTML"
    confirmations = [
        call["data"]
        for call in call_log
        if call["method"] == "sendMessage"
        and call["data"]
        and call["data"].get("chat_id") == 1234
    ]
    assert confirmations, "Expected confirmation messages to be sent to operator"
    expected_link = posted_link.get("url")
    assert expected_link is not None
    assert any(expected_link in str(msg.get("text") or "") for msg in confirmations)

    assert bot.pending_flowers_previews.get(1234) is None
    remaining = bot.db.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    assert remaining == 0

    await bot.close()


@pytest.mark.asyncio
async def test_flowers_preview_reuses_converted_photo_id(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "channel_id": -500,
        "test_channel_id": -600,
        "assets": {"min": 1, "max": 1},
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    _seed_weather(bot)

    file_meta = {
        "file_id": "photo_cached",
        "file_unique_id": "uniq_photo",
        "mime_type": "image/jpeg",
    }
    asset_id = bot.data.save_asset(
        -2100,
        999,
        None,
        "",
        tg_chat_id=-2100,
        caption="",
        kind="photo",
        file_meta=file_meta,
        metadata={"original_document_file_id": "doc_cached"},
        categories=["flowers"],
        rubric_id=1,
    )
    bot.data.update_asset(
        asset_id,
        vision_category="flowers",
        vision_photo_weather="солнечно",
        city="Калининград",
    )

    bot.db.execute(
        "INSERT OR REPLACE INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, '+00:00')",
        (1234, "tester"),
    )
    bot.db.commit()

    calls: list[dict[str, Any]] = []
    multipart_calls: list[dict[str, Any]] = []
    posted_link: dict[str, str] = {}

    async def fake_api(method, data=None, *, files=None):  # type: ignore[override]
        calls.append({"method": method, "data": data, "files": files})
        if method == "sendPhoto":
            chat_id = data.get("chat_id") if isinstance(data, dict) else None
            message_id = 70 if chat_id == 1234 else 95
            if chat_id == -500:
                posted_link["url"] = bot.post_url(-500, message_id)
            return {"ok": True, "result": {"message_id": message_id}}
        if method == "sendMessage":
            return {"ok": True, "result": {"message_id": 205}}
        if method in {"deleteMessage", "editMessageText", "answerCallbackQuery"}:
            return {"ok": True}
        return {"ok": True}

    async def fake_api_multipart(method, data=None, *, files=None):  # type: ignore[override]
        multipart_calls.append({"method": method, "data": data, "files": files})
        return {"ok": True, "result": {"message_id": 880}}

    bot.api_request = fake_api  # type: ignore[assignment]
    bot.api_request_multipart = fake_api_multipart  # type: ignore[assignment]

    await bot.publish_rubric("flowers", channel_id=-500, initiator_id=1234)

    state = bot.pending_flowers_previews.get(1234)
    assert state is not None

    preview_caption = str(state.get("preview_caption") or "")
    publish_caption = str(state.get("publish_caption") or "")
    publish_mode = state.get("publish_parse_mode")
    assert FLOWERS_FOOTER_LINK not in preview_caption
    assert FLOWERS_FOOTER_LINK in publish_caption
    assert publish_caption.endswith(FLOWERS_FOOTER_LINK)
    assert publish_mode == "HTML"
    weather_today = str(state.get("weather_today_line") or "")
    weather_yesterday = str(state.get("weather_yesterday_line") or "")
    assert weather_today
    assert weather_today not in preview_caption
    assert html.escape(weather_today) not in publish_caption
    assert "Вчера:" in weather_yesterday
    assert state.get("weather_line") == weather_today
    if preview_caption:
        assert publish_caption.startswith(html.escape(preview_caption.strip()))

    preview_photos = [
        call
        for call in calls
        if call["method"] == "sendPhoto" and call["data"] and call["data"].get("chat_id") == 1234
    ]
    assert preview_photos, "Preview should be sent with cached photo file_id"
    assert preview_photos[0]["files"] is None, "Preview must reuse Telegram file id"
    preview_payload = preview_photos[0]["data"]
    assert preview_payload.get("parse_mode") is None
    assert preview_payload.get("caption") == preview_caption
    assert not multipart_calls, "Cached photo should not trigger multipart uploads"

    await bot._handle_flowers_preview_callback(1234, "send_main", {"id": "cb-reuse"})

    final_photos = [
        call
        for call in calls
        if call["method"] == "sendPhoto" and call["data"] and call["data"].get("chat_id") == -500
    ]
    assert final_photos, "Final publication should reuse cached photo file_id"
    assert final_photos[0]["files"] is None
    final_payload = final_photos[0]["data"]
    assert final_payload.get("caption") == publish_caption
    assert final_payload.get("parse_mode") == "HTML"
    confirmations = [
        call["data"]
        for call in calls
        if call["method"] == "sendMessage"
        and call["data"]
        and call["data"].get("chat_id") == 1234
    ]
    assert confirmations, "Expected confirmation messages to be sent to operator"
    expected_link = posted_link.get("url")
    assert expected_link is not None
    assert any(expected_link in str(msg.get("text") or "") for msg in confirmations)
    assert bot.pending_flowers_previews.get(1234) is None
    assert not multipart_calls

    await bot.close()


@pytest.mark.asyncio
async def test_flowers_asset_selection_random(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "assets": {"min": 4, "max": 4},
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    now = datetime.utcnow().isoformat()
    for idx in range(10):
        file_meta = {"file_id": f"rf{idx}"}
        asset_id = bot.data.save_asset(
            -2100,
            300 + idx,
            None,
            "",
            tg_chat_id=-2100,
            caption="",
            kind="photo",
            file_meta=file_meta,
            metadata={"date": now},
            categories=["flowers"],
            rubric_id=1,
        )
        bot.data.update_asset(
            asset_id,
            vision_category="flowers",
            vision_photo_weather="солнечно",
            city=f"Город {idx}",
        )

    selections = [
        [
            asset.id
            for asset in bot.data.fetch_assets_by_vision_category(
                "flowers",
                rubric_id=1,
                limit=4,
                random_order=True,
            )
        ]
        for _ in range(3)
    ]
    assert all(len(selection) == 4 for selection in selections)
    assert len({tuple(selection) for selection in selections}) > 1
    await bot.close()


@pytest.mark.asyncio
async def test_publish_flowers_uses_distinct_assets(tmp_path, monkeypatch):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "assets": {"min": 4, "max": 4},
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    _seed_weather(bot)
    now = datetime.utcnow().isoformat()
    for idx in range(8):
        file_meta = {"file_id": f"rf{idx}"}
        asset_id = bot.data.save_asset(
            -2100,
            400 + idx,
            None,
            "",
            tg_chat_id=-2100,
            caption="",
            kind="photo",
            file_meta=file_meta,
            metadata={"date": now},
            categories=["flowers"],
            rubric_id=1,
        )
        bot.data.update_asset(
            asset_id,
            vision_category="flowers",
            vision_photo_weather="солнечно",
            city=f"Город {idx}",
        )

    used_media: list[list[str]] = []

    async def fake_api(method, data=None, *, files=None):
        if method == "sendMediaGroup":
            assert data is not None
            used_media.append([item["media"] for item in data["media"]])
            return {"ok": True, "result": [{"message_id": len(used_media)}]}
        return {"ok": True}

    async def fake_cleanup(assets, *, extra_paths=None):
        return None

    shuffle_calls = {"count": 0}

    def rotating_shuffle(seq):
        if not seq:
            return
        shuffle_calls["count"] += 1
        shift = shuffle_calls["count"] % len(seq)
        if shift:
            seq[:] = seq[shift:] + seq[:shift]

    monkeypatch.setattr(data_access.random, "shuffle", rotating_shuffle)
    bot.api_request = fake_api  # type: ignore[assignment]
    monkeypatch.setattr(bot, "_cleanup_assets", fake_cleanup)

    ok_first = await bot.publish_rubric("flowers", channel_id=-700)
    ok_second = await bot.publish_rubric("flowers", channel_id=-700)

    assert ok_first and ok_second
    assert len(used_media) == 2
    assert used_media[0] != used_media[1]

    await bot.close()


@pytest.mark.asyncio
async def test_flowers_preview_service_block(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    state = {
        "preview_caption": "Черновик",
        "weather_today_line": "Калининград: Сегодня: переменная облачность, температура 6°C, ветер 3 м/с",
        "weather_yesterday_line": "Калининград: Вчера: пасмурно, температура 4°C, ветер 5 м/с",
        "weather_line": "Калининград: Сегодня: переменная облачность, температура 6°C, ветер 3 м/с",
        "instructions": "Добавь тёплый тон",
        "channel_id": -500,
        "test_channel_id": -600,
        "default_channel_id": -500,
        "default_channel_type": "main",
        "plan": {
            "patterns": [
                {"id": "p-1", "instruction": "Упомяни бутоны", "photo_dependent": True},
                {"id": "p-2", "instruction": "Добавь цитату", "photo_dependent": False},
            ],
            "weather": {
                "cities": "Калининград",
                "today": {
                    "temperature": 6.0,
                    "wind_speed": 3.2,
                    "condition": "переменная облачность",
                    "parts": {
                        "morning": "ясно",
                        "day": "переменная облачность",
                        "evening": "пасмурно",
                    },
                },
                "yesterday": {
                    "temperature": 4.0,
                    "wind_speed": 5.1,
                    "condition": "пасмурно",
                    "parts": {
                        "morning": "морось",
                        "day": "пасмурно",
                        "evening": "дождь",
                    },
                },
                "sea": {"temperature": 9.0, "wave": 0.4},
            },
            "flowers": [{"name": "ирисы"}, {"name": "тюльпаны"}],
            "previous_text": "Учора говорили о тепле",
            "instructions": "Добавь тёплый тон",
        },
        "plan_meta": {
            "pattern_ids": ["p-1", "p-2"],
            "banned_words": ["реклама"],
            "length": {"min": 420, "max": 520},
            "cities": ["Калининград"],
        },
        "weather_block": {
            "city": {"name": "Калининград"},
            "cities": "Калининград",
            "today": {
                "temperature": 6.0,
                "wind_speed": 3.2,
                "condition": "переменная облачность",
                "parts": {
                    "morning": "ясно",
                    "day": "переменная облачность",
                    "evening": "пасмурно",
                },
            },
            "yesterday": {
                "temperature": 4.0,
                "wind_speed": 5.1,
                "condition": "пасмурно",
                "parts": {
                    "morning": "морось",
                    "day": "пасмурно",
                    "evening": "дождь",
                },
            },
        },
        "weather_details": {
            "city": {"name": "Калининград"},
            "sea": {"temperature": 9.0, "wave": 0.4, "description": "море спокойное"},
            "today": {
                "temperature": 6.0,
                "wind_speed": 3.2,
                "condition": "переменная облачность",
                "parts": {
                    "morning": "ясно",
                    "day": "переменная облачность",
                    "evening": "пасмурно",
                },
            },
            "yesterday": {
                "temperature": 4.0,
                "wind_speed": 5.1,
                "condition": "пасмурно",
                "parts": {
                    "morning": "морось",
                    "day": "пасмурно",
                    "evening": "дождь",
                },
            },
        },
        "previous_main_post_text": "Вчерашний текст",
    }

    prompt_payload = bot._build_flowers_prompt_payload(state["plan"], state["plan_meta"])
    assert prompt_payload["prompt_length"] <= 2000
    state["serialized_plan"] = prompt_payload["serialized_plan"]
    state["plan_system_prompt"] = prompt_payload["system_prompt"]
    state["plan_user_prompt"] = prompt_payload["user_prompt"]
    state["plan_request_text"] = prompt_payload["request_text"]
    state["plan_prompt"] = prompt_payload["user_prompt"]
    state["plan_prompt_length"] = prompt_payload["prompt_length"]
    state["plan_prompt_fallback"] = prompt_payload["used_fallback"]

    text = bot._render_flowers_preview_text(state)

    assert "Служебно (длина" in text
    escaped_system = html.escape(state["plan_system_prompt"])
    escaped_user = html.escape(state["plan_user_prompt"])
    assert (
        "<blockquote expandable=\"true\">"
        f"<b>System prompt</b>:\n{escaped_system}"
        "\n\n"
        f"<b>User prompt</b>:\n{escaped_user}"
        "</blockquote>"
        in text
    )
    assert "Шаблоны:" not in text
    assert (
        "Погода сегодня: Калининград: Сегодня: переменная облачность, температура 6°C, ветер 3 м/с"
        in text
    )
    assert (
        "Погода вчера: Калининград: Вчера: пасмурно, температура 4°C, ветер 5 м/с"
        in text
    )
    assert "Море: вода 9°C, волна 0.4 м" in text
    assert "Предыдущая публикация: Вчерашний текст" in text

    await bot.close()


@pytest.mark.asyncio
async def test_flowers_prompt_contains_raw_weather_json(tmp_path, monkeypatch):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    plan = {
        "patterns": [
            {
                "id": "weather_focus",
                "instruction": "Напомни про сравнение",
                "kind": "weather",
                "photo_dependent": False,
            }
        ],
        "weather": {
            "cities": "Калининград",
            "today": {
                "temperature": 6.0,
                "wind_speed": 3.0,
                "condition": "дождь",
                "parts": {
                    "morning": "солнечно",
                    "day": "пасмурно",
                    "evening": "туман",
                },
            },
            "yesterday": {
                "temperature": 4.0,
                "wind_speed": 5.0,
                "condition": "дождь",
                "parts": {
                    "morning": "ливень",
                    "day": "снег",
                    "evening": "ветер",
                },
            },
        },
        "flowers": [
            {"name": "роза"},
            {"name": "тюльпан"},
        ],
        "previous_text": "",
        "instructions": "",
        "photo_context": [
            {
                "flowers": ["роза", "тюльпан"],
                "hints": [
                    "Весенний букет в вазе",
                    "Тёплый солнечный луч на столе",
                ],
                "location": "Калининград",
            },
            {
                "location": "Светлогорск",
            },
        ],
    }
    plan_meta = {
        "pattern_ids": ["weather_focus"],
        "banned_words": [],
        "length": {"min": 420, "max": 520},
        "cities": ["Калининград"],
    }

    prompt_payload = bot._build_flowers_prompt_payload(plan, plan_meta)
    user_prompt = prompt_payload["user_prompt"]

    assert "Сырые данные погоды" in user_prompt
    assert '"today"' in user_prompt
    assert '"temperature": 6.0' in user_prompt
    assert '"yesterday"' in user_prompt
    assert '"condition": "дождь"' in user_prompt
    assert '"parts"' in user_prompt
    assert '"morning": "солнечно"' in user_prompt
    assert '"day": "пасмурно"' in user_prompt
    assert '"evening": "туман"' in user_prompt
    assert '"morning": "ливень"' in user_prompt
    assert '"day": "снег"' in user_prompt
    assert '"evening": "ветер"' in user_prompt
    assert '"trend_indicators"' not in user_prompt
    assert '"trend_strings"' not in user_prompt
    assert "Цветы на фото (распознаны): роза, тюльпан" in user_prompt
    assert "Фотографии:" in user_prompt
    assert "Фото 1: Цветы: роза, тюльпан" in user_prompt
    assert "Подсказки: Весенний букет в вазе; Тёплый солнечный луч на столе" in user_prompt
    assert "Локация: Калининград" in user_prompt
    assert "Фото 2: Локация: Светлогорск" in user_prompt
    comparison_rule = (
        "Сравнивай сегодня и вчера, опираясь на сырые погодные данные; если изменения ощутимые, коротко озвучь их"
    )
    photo_timeliness_rule = (
        "Помни: фотографии сделаны заранее, поэтому не описывай их как отражение сегодняшней погоды."
    )
    assert comparison_rule in user_prompt
    assert photo_timeliness_rule in user_prompt
    assert "positive_intro" not in user_prompt
    assert "trend_summary" not in user_prompt
    assert "Утро радует" not in user_prompt
    assert any(comparison_rule in rule for rule in prompt_payload["rules"])
    assert any(
        photo_timeliness_rule in rule for rule in prompt_payload["rules"]
    )
    assert any(
        "Пиши естественно, как живой человек" in rule
        for rule in prompt_payload["rules"]
    )
    assert any(
        "Игнорируй микроколебания погоды" in rule
        for rule in prompt_payload["rules"]
    )
    assert any(
        "Сопоставляй температуру, ветер и другие показатели" in rule
        for rule in prompt_payload["rules"]
    )
    assert any(
        "Приветствие опирай на реальные фото" in rule
        for rule in prompt_payload["rules"]
    )
    system_timeliness_clause = (
        "Фотографии сделаны заранее, поэтому не описывай их как отражение сегодняшней погоды."
    )
    assert system_timeliness_clause in prompt_payload["system_prompt"]

    short_plan = deepcopy(plan)
    short_plan["weather"]["today"]["parts"] = {}
    short_plan["weather"]["yesterday"]["parts"] = {}

    short_payload = bot._build_flowers_prompt_payload(short_plan, plan_meta)
    assert not short_payload["used_fallback"]
    short_prompt = short_payload["user_prompt"]
    assert "Фотографии:" in short_prompt
    assert "Фото 1: Цветы: роза, тюльпан" in short_prompt
    assert "Подсказки: Весенний букет в вазе; Тёплый солнечный луч на столе" in short_prompt
    assert "Фото 2: Локация: Светлогорск" in short_prompt

    fake_photo_context = [{"flowers": ["ромашка"], "hints": ["Лёгкий ветер"], "location": "Балтика"}]

    class DummyAsset:
        def __init__(self, city: str):
            self.city = city

    def fake_extract(self, assets, weather_block, *, seed_rng):
        return {
            "has_photo": True,
            "flowers": [],
            "flower_ids": [],
            "palettes": [],
            "photo_context": fake_photo_context,
            "weather": None,
            "season": None,
            "season_description": None,
            "holiday": None,
            "wisdom": None,
            "engagement": None,
        }

    monkeypatch.setattr(Bot, "_extract_flower_features", fake_extract)
    monkeypatch.setattr(
        Bot,
        "_flowers_recent_pattern_ids",
        lambda self, rubric, limit=14: (set(), set(), []),
    )
    bot.flowers_kb = None

    rubric = Rubric(id=1, code="flowers", title="Цветы", description=None, config={})
    built_plan, _ = bot._build_flowers_plan(
        rubric,
        [DummyAsset("Балтийск")],
        {},
        channel_id=None,
        instructions=None,
    )

    assert built_plan["photo_context"] == fake_photo_context

    await bot.close()


@pytest.mark.asyncio
async def test_flowers_prompt_fallback_contains_human_tone(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    long_instruction = "Очень длинное описание " * 120
    plan = {
        "patterns": [
            {
                "id": f"p-{idx}",
                "instruction": long_instruction,
                "photo_dependent": False,
            }
            for idx in range(60)
        ],
        "weather": {},
        "flowers": [],
        "previous_text": "",
        "instructions": "",
    }
    plan_meta = {
        "pattern_ids": [f"p-{idx}" for idx in range(60)],
        "banned_words": [],
        "length": {"min": 420, "max": 520},
        "cities": [],
    }

    prompt_payload = bot._build_flowers_prompt_payload(plan, plan_meta)

    assert prompt_payload["used_fallback"]
    assert "Пиши естественно, как живой человек" in prompt_payload["user_prompt"]

    await bot.close()


@pytest.mark.asyncio
async def test_flowers_prompt_preserves_all_photo_descriptions(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    photo_context = [
        {
            "flowers": [f"цветок {idx}", f"деталь {idx}"],
            "hints": [
                f"подсказка {idx}-{hint}" for hint in range(1, 8)
            ],
            "location": f"Локация {idx}",
        }
        for idx in range(1, 7)
    ]

    plan = {
        "patterns": [
            {
                "id": "p-short",
                "instruction": "Сделай приветствие солнечным",
                "photo_dependent": False,
            },
            {
                "id": "p-photo",
                "instruction": "Отметь детали на снимках",
                "photo_dependent": True,
            },
        ],
        "weather": {},
        "flowers": [],
        "previous_text": "",
        "instructions": "",
        "photo_context": photo_context,
    }
    plan_meta = {
        "pattern_ids": ["p-short", "p-photo"],
        "banned_words": [],
        "length": {"min": 420, "max": 520},
        "cities": ["Калининград"],
    }

    primary_payload = bot._build_flowers_prompt_payload(plan, plan_meta)
    assert not primary_payload["used_fallback"]
    primary_lines = [
        line
        for line in primary_payload["user_prompt"].splitlines()
        if line.startswith("Фото ")
    ]
    assert len(primary_lines) == 6
    for idx in range(1, 7):
        matching_line = next(
            (line for line in primary_lines if line.startswith(f"Фото {idx}:")),
            "",
        )
        assert matching_line
        assert f"Локация {idx}" in matching_line
        assert "Подсказки:" in matching_line

    long_instruction = "Очень длинное описание " * 200
    fallback_plan = deepcopy(plan)
    fallback_plan["patterns"] = [
        {
            "id": f"p-{idx}",
            "instruction": long_instruction,
            "photo_dependent": False,
        }
        for idx in range(90)
    ]
    fallback_meta = deepcopy(plan_meta)
    fallback_meta["pattern_ids"] = [f"p-{idx}" for idx in range(90)]

    fallback_payload = bot._build_flowers_prompt_payload(fallback_plan, fallback_meta)
    assert fallback_payload["used_fallback"]
    fallback_lines = [
        line
        for line in fallback_payload["user_prompt"].splitlines()
        if line.startswith("Фото ")
    ]
    assert len(fallback_lines) == 6
    for idx in range(1, 7):
        matching_line = next(
            (line for line in fallback_lines if line.startswith(f"Фото {idx}:")),
            "",
        )
        assert matching_line
        assert f"Локация {idx}" in matching_line
        assert "Подсказки:" in matching_line

    await bot.close()


@pytest.mark.asyncio
async def test_flowers_preview_instructions_escaped(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    instructions = "Используй <script> & проверь"
    state: dict[str, Any] = {
        "preview_caption": "",
        "weather_today_line": "Ясно",
        "weather_yesterday_line": "Дождливо",
        "instructions": instructions,
    }

    text = bot._render_flowers_preview_text(state)

    assert "Инструкции оператора:" in text
    assert "<blockquote expandable=\"true\">" in text
    assert "&lt;script&gt;" in text
    assert "&amp; проверь" in text
    assert "<script>" not in text

    await bot.close()


@pytest.mark.asyncio
async def test_flowers_preview_previous_text_html_escaped(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    snippet = '<b>Важное объявление</b> и <a href="https://example.com">ссылка</a>'
    previous_text = " ".join([snippet for _ in range(200)])

    state: dict[str, Any] = {
        "preview_caption": "",
        "weather_today_line": "Ясно",
        "weather_yesterday_line": "Пасмурно",
        "previous_main_post_text": previous_text,
    }

    text = bot._render_flowers_preview_text(state)

    assert len(text) <= FLOWERS_PREVIEW_MAX_LENGTH
    assert "Предыдущая публикация:" in text
    assert "&lt;b&gt;Важное объявление&lt;/b&gt;" in text
    assert "&lt;a href=&quot;https://example.com&quot;&gt;ссылка&lt;/a&gt;" in text
    assert "<b>" not in text
    assert "<a href=\"https://example.com\"" not in text
    assert "…" in text

    await bot.close()


@pytest.mark.asyncio
async def test_flowers_preview_truncates_long_payload(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    long_instruction = " ".join(["очень" for _ in range(800)])
    pattern_instruction = "Описание паттерна " + ("очень длинное предложение " * 20)
    patterns = [
        {
            "id": f"p-{idx}",
            "instruction": pattern_instruction,
            "photo_dependent": idx % 2 == 0,
            "kind": "detail" if idx % 3 == 0 else "",
        }
        for idx in range(60)
    ]
    plan_weather = {
        "cities": ["Калининград", "Балтийск", "Советск"],
        "today": {
            "temperature": 12.5,
            "wind_speed": 6.7,
            "condition": "пасмурно",
        },
        "yesterday": {
            "temperature": 10.2,
            "wind_speed": 8.5,
            "condition": "дождь",
        },
    }
    weather_details = {
        "city": {"name": "Калининград"},
        "sea": {
            "temperature": 7.2,
            "wave": 1.4,
            "description": "Море играет",
            "detail": "Балтика волнуется",
        },
        "today": {
            "temperature": 12.5,
            "wind_speed": 6.7,
            "condition": "пасмурно",
        },
        "yesterday": {
            "temperature": 10.2,
            "wind_speed": 8.5,
            "condition": "дождь",
        },
    }
    long_system = "System " + ("prompt " * 200)
    long_user = "User " + ("prompt " * 200)
    previous_text = " ".join(["Предыдущая" for _ in range(400)])

    state = {
        "preview_caption": "",
        "weather_today_line": "Солнечно" + (" ☀️" * 50),
        "weather_yesterday_line": "Дождливо" + (" 💧" * 50),
        "weather_line": "",
        "instructions": long_instruction,
        "channel_id": -500,
        "test_channel_id": -600,
        "default_channel_id": -500,
        "default_channel_type": "main",
        "plan": {
            "patterns": patterns,
            "weather": plan_weather,
        },
        "plan_system_prompt": long_system,
        "plan_user_prompt": long_user,
        "plan_prompt_length": 9000,
        "plan_prompt_fallback": True,
        "weather_details": weather_details,
        "previous_main_post_text": previous_text,
    }

    text = bot._render_flowers_preview_text(state)

    assert len(text) <= FLOWERS_PREVIEW_MAX_LENGTH
    assert "Погода сегодня: Солнечно" in text
    assert "Погода вчера: Дождливо" in text
    assert "Выберите действие" in text
    assert "Инструкции оператора" in text
    assert "…" in text
    assert "<blockquote expandable=\"true\">" in text

    await bot.close()


@pytest.mark.asyncio
async def test_flowers_preview_weather_lines_escaped(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    today_raw = "Лёгкий дождь & ветер < 5 м/с"
    yesterday_raw = "Ясно <туман> & солнце"
    state = {
        "preview_caption": "",
        "weather_today_line": today_raw,
        "weather_yesterday_line": yesterday_raw,
        "weather_line": "",
        "instructions": "",
        "channel_id": -500,
        "test_channel_id": -600,
        "default_channel_id": -500,
        "default_channel_type": "main",
        "plan": {},
    }

    text = bot._render_flowers_preview_text(state)

    assert len(text) <= FLOWERS_PREVIEW_MAX_LENGTH
    assert "Погода сегодня: Лёгкий дождь & ветер < 5 м/с" not in text
    assert "Погода вчера: Ясно <туман> & солнце" not in text
    assert "Погода сегодня: Лёгкий дождь &amp; ветер &lt; 5 м/с" in text
    assert "Погода вчера: Ясно &lt;туман&gt; &amp; солнце" in text

    second_text = bot._render_flowers_preview_text(state)
    assert second_text == text
    assert state.get("weather_today_line_html") == html.escape(today_raw)
    assert state.get("weather_yesterday_line_html") == html.escape(yesterday_raw)

    await bot.close()


@pytest.mark.asyncio
async def test_flowers_preview_regenerate_and_finalize(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "channel_id": -500,
        "test_channel_id": -600,
        "assets": {"min": 4, "max": 4},
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    _seed_weather(bot)

    now = datetime.utcnow().isoformat()
    for idx in range(4):
        file_meta = {"file_id": f"pv{idx}"}
        asset_id = bot.data.save_asset(
            -2100,
            900 + idx,
            None,
            "",
            tg_chat_id=-2100,
            caption="",
            kind="photo",
            file_meta=file_meta,
            metadata={"date": now},
            categories=["flowers"],
            rubric_id=1,
        )
        bot.data.update_asset(
            asset_id,
            vision_category="flowers",
            vision_photo_weather="солнечно",
            city=f"Город {idx}",
        )

    bot.db.execute(
        "INSERT OR REPLACE INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, '+00:00')",
        (1234, "tester"),
    )
    bot.db.commit()

    api_calls: list[tuple[str, dict[str, Any] | None]] = []
    posted_link: dict[str, str] = {}
    message_counter = {"value": 200}
    openai_calls: list[dict[str, Any]] = []
    document_uploads: list[
        tuple[dict[str, Any] | None, dict[str, tuple[str, bytes]] | None]
    ] = []

    class DummyOpenAI:
        def __init__(self) -> None:
            self.api_key = "test"

        async def generate_json(self, **kwargs):  # type: ignore[override]
            openai_calls.append(kwargs)
            return OpenAIResponse(
                {"greeting": "Доброе утро, команда!", "hashtags": ["котопогода", "цветы"]},
                {
                    "prompt_tokens": 150,
                    "completion_tokens": 200,
                    "total_tokens": 350,
                    "request_id": "req-flowers-preview",
                    "endpoint": "/v1/responses",
                },
            )

    bot.openai = DummyOpenAI()

    async def fake_api(method, data=None, *, files=None):  # type: ignore[override]
        api_calls.append((method, data))
        if method == "sendMediaGroup":
            if data and data.get("chat_id") == 1234:
                return {"ok": True, "result": [{"message_id": 10}, {"message_id": 11}]}
            posted_link["url"] = bot.post_url(-500, 210)
            return {"ok": True, "result": [{"message_id": 210}]}
        if method == "sendDocument":
            document_uploads.append((data, files))
            return {"ok": True, "result": {"message_id": 310}}
        if method == "sendMessage":
            message_counter["value"] += 1
            return {"ok": True, "result": {"message_id": message_counter["value"]}}
        if method in {"editMessageText", "deleteMessage", "answerCallbackQuery"}:
            return {"ok": True}
        return {"ok": True}

    bot.api_request = fake_api  # type: ignore[assignment]

    await bot.publish_rubric("flowers", channel_id=-500, initiator_id=1234)

    state = bot.pending_flowers_previews.get(1234)
    assert state is not None
    assert state.get("caption_message_id")
    weather_today_initial = str(state.get("weather_today_line") or "")
    weather_yesterday_initial = str(state.get("weather_yesterday_line") or "")
    assert weather_today_initial
    assert "Вчера:" in weather_yesterday_initial

    await bot._handle_flowers_preview_callback(1234, "regen_caption", {"id": "cb1"})
    assert any(call[0] == "editMessageText" for call in api_calls)

    await bot._handle_flowers_preview_callback(1234, "instruction", {"id": "cb2"})
    prompt_id = state.get("instruction_prompt_id")
    assert isinstance(prompt_id, int)
    instruction_message = {
        "message_id": 999,
        "from": {"id": 1234, "username": "tester"},
        "chat": {"id": 1234},
        "text": "Добавь смайлы",
        "reply_to_message": {"message_id": prompt_id},
    }
    await bot.handle_message(instruction_message)
    state = bot.pending_flowers_previews.get(1234)
    assert state is not None
    assert state.get("instructions") == "Добавь смайлы"
    assert not state.get("awaiting_instruction")

    preview_caption = str(state.get("preview_caption") or "").strip()
    publish_caption = str(state.get("publish_caption") or "")
    assert FLOWERS_FOOTER_LINK not in preview_caption
    assert FLOWERS_FOOTER_LINK in publish_caption
    assert state.get("publish_parse_mode") == "HTML"
    weather_today = str(state.get("weather_today_line") or "")
    weather_yesterday = str(state.get("weather_yesterday_line") or "")
    assert weather_today
    assert weather_today not in preview_caption
    assert html.escape(weather_today) not in publish_caption
    assert weather_yesterday == weather_yesterday_initial
    assert state.get("weather_line") == weather_today
    summary_updates = [
        data for method, data in api_calls if method == "editMessageText" and data
    ]
    assert summary_updates, "Ожидалось обновление текста предпросмотра"
    assert summary_updates[-1].get("parse_mode") == "HTML"
    summary_text = str(summary_updates[-1].get("text") or "")
    if preview_caption:
        assert preview_caption not in summary_text
    assert "Подпись на медиа показана выше" in summary_text
    assert "Инструкции оператора" in summary_text
    assert "Добавь смайлы" in summary_text
    assert "Доступные каналы:" in summary_text
    assert "📣 -500" in summary_text
    assert "🧪 -600" in summary_text
    assert "Служебно (длина" in summary_text
    plan_system_prompt = str(state.get("plan_system_prompt") or "")
    plan_user_prompt = str(state.get("plan_user_prompt") or "")
    plan_request_text = str(state.get("plan_request_text") or "")
    assert plan_system_prompt
    assert plan_user_prompt
    assert plan_system_prompt in plan_request_text
    assert plan_user_prompt in plan_request_text
    assert openai_calls, "generate_json should have been called for preview"
    assert openai_calls[-1]["system_prompt"] == plan_system_prompt
    assert openai_calls[-1]["user_prompt"] == plan_user_prompt
    plan_prompt_text = str(state.get("plan_prompt") or "")
    assert plan_prompt_text == plan_user_prompt
    assert len(plan_prompt_text) <= 2000
    prompt_length = state.get("plan_prompt_length")
    assert isinstance(prompt_length, int)
    assert prompt_length <= 2000
    escaped_system = html.escape(plan_system_prompt)
    escaped_user = html.escape(plan_user_prompt)
    expected_block = (
        "<blockquote expandable=\"true\">"
        f"<b>System prompt</b>:\n{escaped_system}"
        "\n\n"
        f"<b>User prompt</b>:\n{escaped_user}"
        "</blockquote>"
    )
    assert expected_block in summary_text
    assert "Добавь смайлы" in plan_prompt_text
    assert html.escape("Добавь смайлы") in escaped_user
    serialized_plan_text = str(state.get("serialized_plan") or "")
    assert "Добавь смайлы" in serialized_plan_text
    assert "Шаблоны:" not in summary_text
    assert f"Погода сегодня: {weather_today}" in summary_text
    assert f"Погода вчера: {weather_yesterday}" in summary_text
    assert "Предыдущая публикация: не публиковалось" in summary_text

    preview_messages = [
        data
        for method, data in api_calls
        if method == "sendMessage"
        and data
        and "Выберите действие на клавиатуре" in str(data.get("text") or "")
    ]
    assert preview_messages
    assert preview_messages[-1].get("parse_mode") == "HTML"
    reply_markup = preview_messages[-1].get("reply_markup")
    assert isinstance(reply_markup, dict)
    keyboard = reply_markup.get("inline_keyboard")
    assert isinstance(keyboard, list)
    assert any(
        isinstance(button, dict)
        and button.get("callback_data") == "flowers_preview:download_prompt"
        for row in keyboard
        if isinstance(row, list)
        for button in row
    ), "Ожидалась кнопка скачивания инструкции"

    await bot._handle_flowers_preview_callback(1234, "download_prompt", {"id": "cb-download"})
    assert document_uploads, "Ожидалось, что промпт будет отправлен документом"
    download_data, download_files = document_uploads[-1]
    assert download_data is not None
    assert download_data.get("chat_id") == 1234
    assert download_files is not None
    document_entry = download_files.get("document") if download_files else None
    assert document_entry is not None
    filename, payload = document_entry
    assert filename == "flowers_prompt.txt"
    plan_request_text = str(state.get("plan_request_text") or "")
    assert payload == plan_request_text.encode("utf-8")
    download_confirmations = [
        data
        for method, data in api_calls
        if method == "answerCallbackQuery"
        and data
        and data.get("text") == "Промпт отправлен."
    ]
    assert download_confirmations, "Ожидалось подтверждение отправки промпта"

    await bot._handle_flowers_preview_callback(1234, "send_main", {"id": "cb3"})
    confirmations = [
        data
        for method, data in api_calls
        if method == "sendMessage" and data and data.get("chat_id") == 1234
    ]
    assert confirmations, "Expected confirmation messages to be sent to operator"
    expected_link = posted_link.get("url")
    assert expected_link is not None
    assert any(expected_link in str(msg.get("text") or "") for msg in confirmations)
    assert bot.pending_flowers_previews.get(1234) is None
    remaining = bot.db.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    assert remaining == 0
    history = bot.db.execute("SELECT metadata FROM posts_history").fetchone()
    assert history is not None
    meta = json.loads(history["metadata"])
    assert meta["asset_ids"]
    assert meta["test"] is False
    assert meta.get("weather_today_line") == weather_today
    assert meta.get("weather_yesterday_line") == weather_yesterday
    assert meta.get("weather_line") == weather_today
    assert meta.get("pattern_ids")
    assert isinstance(meta.get("plan"), dict)

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
async def test_guess_arch_asset_selection_random(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "assets": {"min": 4, "max": 4},
    }
    _insert_rubric(bot, "guess_arch", config, rubric_id=2)
    now = datetime.utcnow().isoformat()
    for idx in range(12):
        file_meta = {"file_id": f"ra{idx}"}
        asset_id = bot.data.save_asset(
            -3200,
            800 + idx,
            None,
            "",
            tg_chat_id=-3200,
            caption="",
            kind="photo",
            file_meta=file_meta,
            metadata={"date": now},
            categories=["architecture"],
            rubric_id=2,
        )
        bot.data.update_asset(
            asset_id,
            vision_category="architecture",
            vision_arch_view="фасад",
            vision_photo_weather="пасмурно",
        )

    selections = [
        [
            asset.id
            for asset in bot.data.fetch_assets_by_vision_category(
                "architecture",
                rubric_id=2,
                limit=4,
                require_arch_view=True,
                random_order=True,
            )
        ]
        for _ in range(3)
    ]
    assert all(len(selection) == 4 for selection in selections)
    assert len({tuple(selection) for selection in selections}) > 1
    await bot.close()


@pytest.mark.asyncio
async def test_generate_flowers_uses_gpt_4o(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {"enabled": True}
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    rubric = bot.data.get_rubric_by_code("flowers")
    _seed_weather(bot)
    weather_block = bot._compose_flowers_weather_block(["Baltiisk"])  # type: ignore[attr-defined]

    calls: list[dict[str, Any]] = []

    class DummyOpenAI:
        def __init__(self) -> None:
            self.api_key = "test"

        async def generate_json(self, **kwargs):  # type: ignore[override]
            calls.append(kwargs)
            return OpenAIResponse(
                {"greeting": "Доброе утро", "hashtags": ["котопогода", "цветы"]},
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 15,
                    "total_tokens": 25,
                    "request_id": "req-1",
                    "endpoint": "/v1/responses",
                },
            )

    bot.openai = DummyOpenAI()

    asset = Asset(
        id=1,
        channel_id=1,
        tg_chat_id=1,
        message_id=1,
        origin="test",
        caption_template=None,
        caption=None,
        hashtags=None,
        categories=[],
        kind="photo",
        file_id="file-1",
        file_unique_id="uniq-1",
        file_name="flower.jpg",
        mime_type="image/jpeg",
        file_size=None,
        width=1080,
        height=1350,
        duration=None,
        recognized_message_id=None,
        exif_present=False,
        latitude=None,
        longitude=None,
        city="Baltiisk",
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
        vision_flower_varieties=["rose"],
        vision_confidence=None,
        vision_caption=None,
    )

    greeting, hashtags, plan, plan_meta = await bot._generate_flowers_copy(
        rubric,
        [asset],
        channel_id=-100,
        weather_block=weather_block,
    )

    assert greeting == "Доброе утро"
    assert hashtags == ["котопогода", "цветы"]
    assert isinstance(plan, dict)
    assert isinstance(plan_meta, dict)
    assert plan_meta.get("pattern_ids"), "pattern ids should be present in metadata"
    assert calls, "generate_json was not called"
    request = calls[0]
    assert request["model"] == "gpt-4o"
    assert 0.9 <= request["temperature"] <= 1.1
    assert request["top_p"] == 0.9
    user_prompt = request["user_prompt"]
    assert "Контекст" in user_prompt
    if "Паттерны" in user_prompt:
        assert "Правила" in user_prompt
    else:
        assert "Основные идеи" in user_prompt
        assert "Пиши естественно" in user_prompt
    assert len(user_prompt) <= 2000
    assert weather_block["city"] is not None
    assert weather_block["city"]["name"].casefold() == "kaliningrad"
    assert isinstance(weather_block.get("today"), dict)
    assert isinstance(weather_block.get("yesterday"), dict)

    await bot.close()


@pytest.mark.asyncio
async def test_generate_flowers_retries_on_banned_cliches(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {"enabled": True}
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    rubric = bot.data.get_rubric_by_code("flowers")
    assert rubric is not None

    asset = Asset(
        id=3,
        channel_id=1,
        tg_chat_id=1,
        message_id=3,
        origin="test",
        caption_template=None,
        caption=None,
        hashtags=None,
        categories=[],
        kind="photo",
        file_id="file-3",
        file_unique_id="uniq-3",
        file_name="flower-3.jpg",
        mime_type="image/jpeg",
        file_size=None,
        width=1024,
        height=1024,
        duration=None,
        recognized_message_id=None,
        exif_present=False,
        latitude=None,
        longitude=None,
        city="Калининград",
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
        vision_flower_varieties=["rose"],
        vision_confidence=None,
        vision_caption=None,
    )

    calls: list[dict[str, Any]] = []

    class DummyOpenAI:
        def __init__(self) -> None:
            self.api_key = "test"

        async def generate_json(self, **kwargs):  # type: ignore[override]
            calls.append(kwargs)
            if len(calls) == 1:
                return OpenAIResponse(
                    {
                        "greeting": "Какой прекрасный и волшебный день!",
                        "hashtags": ["#котопогода", "#цветы"],
                    },
                    {
                        "prompt_tokens": 7,
                        "completion_tokens": 9,
                        "total_tokens": 16,
                        "request_id": "req-banned-1",
                        "endpoint": "/v1/responses",
                    },
                )
            return OpenAIResponse(
                {
                    "greeting": "Доброе утро, делимся уютом без клише",
                    "hashtags": ["#котопогода", "#цветы", "#уют"],
                },
                {
                    "prompt_tokens": 6,
                    "completion_tokens": 8,
                    "total_tokens": 14,
                    "request_id": "req-banned-2",
                    "endpoint": "/v1/responses",
                },
            )

    bot.openai = DummyOpenAI()

    greeting, hashtags, plan, plan_meta = await bot._generate_flowers_copy(
        rubric,
        [asset],
        channel_id=-300,
    )

    assert len(calls) == 2
    assert "прекрасн" not in greeting.casefold()
    assert isinstance(plan, dict)
    assert isinstance(plan_meta, dict)
    banned_words = plan_meta.get("banned_words") or []
    assert {"прекрасный", "волшебный", "неповторимый", "самый-самый"}.issubset(
        set(banned_words)
    )
    for word in ["прекрасный", "волшебный", "неповторимый", "самый-самый"]:
        sample = f"Это {word} букет!"
        assert bot._flowers_contains_banned_word(sample, banned_words)
    await bot.close()


@pytest.mark.asyncio
async def test_generate_flowers_retries_on_duplicate(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {"enabled": True}
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    rubric = bot.data.get_rubric_by_code("flowers")
    bot.data.record_post_history(1, 1, None, rubric.id, {
        "rubric_code": "flowers",
        "greeting": "Тёплый кот мурчит радостно",
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
                    {
                        "greeting": "Тёплый кот мурчит у окна",
                        "hashtags": ["#котопогода"],
                    },
                    {
                        "prompt_tokens": 5,
                        "completion_tokens": 5,
                        "total_tokens": 10,
                        "request_id": "req-first",
                        "endpoint": "/v1/responses",
                    },
                )
            return OpenAIResponse(
                {
                    "greeting": "Солнечный привет, друзья",
                    "hashtags": ["#котопогода", "#цветы"],
                },
                {
                    "prompt_tokens": 6,
                    "completion_tokens": 7,
                    "total_tokens": 13,
                    "request_id": "req-second",
                    "endpoint": "/v1/responses",
                },
            )

    bot.openai = DummyOpenAI()

    asset = Asset(
        id=2,
        channel_id=1,
        tg_chat_id=1,
        message_id=2,
        origin="test",
        caption_template=None,
        caption=None,
        hashtags=None,
        categories=[],
        kind="photo",
        file_id="file-2",
        file_unique_id="uniq-2",
        file_name="flower2.jpg",
        mime_type="image/jpeg",
        file_size=None,
        width=1080,
        height=1080,
        duration=None,
        recognized_message_id=None,
        exif_present=False,
        latitude=None,
        longitude=None,
        city="Калининград",
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
        vision_flower_varieties=["tulip"],
        vision_confidence=None,
        vision_caption=None,
    )

    greeting, hashtags, plan, plan_meta = await bot._generate_flowers_copy(
        rubric,
        [asset],
        channel_id=-200,
    )

    assert greeting == "Солнечный привет, друзья"
    assert hashtags == ["#котопогода", "#цветы"]
    assert len(calls) >= 2
    # Первая попытка должна быть отклонена из-за повторяющейся биграммы
    assert bot._jaccard_similarity(
        "Тёплый кот мурчит радостно",
        "Тёплый кот мурчит у окна",
    ) >= 0.4
    assert isinstance(calls[0], dict)
    assert calls[0]["schema"] == {
        "type": "object",
        "properties": {
            "greeting": {"type": "string"},
            "hashtags": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
            },
        },
        "required": ["greeting", "hashtags"],
    }
    assert isinstance(plan, dict)
    assert isinstance(plan_meta, dict)

    await bot.close()


@pytest.mark.asyncio
async def test_generate_guess_arch_uses_gpt_4o(tmp_path):
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
                {
                    "prompt_tokens": 8,
                    "completion_tokens": 9,
                    "total_tokens": 17,
                    "request_id": "req-3",
                    "endpoint": "/v1/responses",
                },
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


@pytest.mark.asyncio
async def test_rubrics_overview_lists_configs(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    message_counter = 0
    calls: list[tuple[str, dict[str, Any] | None]] = []

    async def fake_api(method, data=None, *, files=None):
        nonlocal message_counter
        calls.append((method, data))
        if method == "sendMessage":
            message_counter += 1
            chat_id = data.get("chat_id") if isinstance(data, dict) else None
            return {
                "ok": True,
                "result": {"message_id": message_counter, "chat": {"id": chat_id}},
            }
        if method == "editMessageText":
            return {
                "ok": True,
                "result": {
                    "message_id": data.get("message_id") if isinstance(data, dict) else None,
                    "chat": {"id": data.get("chat_id") if isinstance(data, dict) else None},
                },
            }
        return {"ok": True}

    bot.api_request = fake_api  # type: ignore
    await bot.start()

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})
    rubrics = bot.data.list_rubrics()
    assert {r.code for r in rubrics} == {"flowers", "guess_arch"}
    bot.data.upsert_rubric(
        "flowers",
        "Flowers",
        config={
            "enabled": True,
            "channel_id": -100,
            "test_channel_id": -200,
            "tz": "+03:00",
            "days": ["mon", "wed"],
            "schedules": [
                {"time": "09:00", "tz": "+03:00", "days": ["mon"], "channel_id": -100},
            ],
        },
    )

    calls.clear()
    await bot.handle_update({"message": {"text": "/rubrics", "from": {"id": 1}}})
    send_calls = [item for item in calls if item[0] == "sendMessage"]
    assert len(send_calls) == 3
    dashboard_method, dashboard_data = send_calls[0]
    assert dashboard_method == "sendMessage"
    assert dashboard_data is not None
    assert "Карточки рубрик" in dashboard_data.get("text", "")
    dashboard_keyboard = dashboard_data.get("reply_markup", {}).get("inline_keyboard", [])
    assert dashboard_keyboard and dashboard_keyboard[0][0]["text"] == "Обновить карточки"

    flowers_message = send_calls[1][1]
    assert flowers_message is not None
    assert "flowers" in flowers_message.get("text", "").lower()
    flowers_keyboard = flowers_message["reply_markup"]["inline_keyboard"]
    assert any(
        btn.get("callback_data") == "rubric_toggle:flowers"
        for row in flowers_keyboard
        for btn in row
    )
    assert bot.rubric_overview_messages[1]["flowers"]["message_id"] is not None

    guess_message = send_calls[2][1]
    assert guess_message is not None
    assert "guess_arch" in guess_message.get("text", "")

    calls.clear()
    await bot.handle_update({"message": {"text": "/rubrics", "from": {"id": 1}}})
    edit_calls = [item for item in calls if item[0] == "editMessageText"]
    assert len(edit_calls) == 3
    await bot.close()


@pytest.mark.asyncio
async def test_rubric_channel_and_schedule_edit_flow(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    message_counter = 0
    calls: list[tuple[str, dict[str, Any] | None]] = []

    async def fake_api(method, data=None, *, files=None):
        nonlocal message_counter
        calls.append((method, data))
        if method == "sendMessage":
            message_counter += 1
            chat_id = data.get("chat_id") if isinstance(data, dict) else None
            return {
                "ok": True,
                "result": {"message_id": message_counter, "chat": {"id": chat_id}},
            }
        if method in {"editMessageText", "editMessageReplyMarkup"}:
            return {
                "ok": True,
                "result": {
                    "message_id": data.get("message_id") if isinstance(data, dict) else None,
                    "chat": {"id": data.get("chat_id") if isinstance(data, dict) else None},
                },
            }
        return {"ok": True}

    bot.api_request = fake_api  # type: ignore
    await bot.start()

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})
    bot.db.execute("INSERT INTO channels (chat_id, title) VALUES (?, ?)", (-100, "Main"))
    bot.db.execute("INSERT INTO channels (chat_id, title) VALUES (?, ?)", (-200, "Test"))
    bot.db.execute("INSERT INTO channels (chat_id, title) VALUES (?, ?)", (-300, "ArchMain"))
    bot.db.execute("INSERT INTO channels (chat_id, title) VALUES (?, ?)", (-400, "ArchTest"))
    bot.db.commit()

    await bot.handle_update({"message": {"text": "/rubrics", "from": {"id": 1}}})
    flowers_info = bot.rubric_overview_messages[1]["flowers"]
    guess_info = bot.rubric_overview_messages[1]["guess_arch"]
    flowers_message = {"chat": {"id": 1}, "message_id": flowers_info["message_id"]}
    guess_message = {"chat": {"id": 1}, "message_id": guess_info["message_id"]}

    await bot.handle_update(
        {"callback_query": {"id": "f0", "from": {"id": 1}, "data": "rubric_toggle:flowers", "message": flowers_message}}
    )
    await bot.handle_update(
        {"callback_query": {"id": "g0", "from": {"id": 1}, "data": "rubric_toggle:guess_arch", "message": guess_message}}
    )

    await bot.handle_update(
        {"callback_query": {"id": "f1", "from": {"id": 1}, "data": "rubric_channel:flowers:main", "message": flowers_message}}
    )
    await bot.handle_update(
        {"callback_query": {"id": "f2", "from": {"id": 1}, "data": "rubric_channel_set:-100", "message": flowers_message}}
    )
    await bot.handle_update(
        {"callback_query": {"id": "f3", "from": {"id": 1}, "data": "rubric_channel:flowers:test", "message": flowers_message}}
    )
    await bot.handle_update(
        {"callback_query": {"id": "f4", "from": {"id": 1}, "data": "rubric_channel_set:-200", "message": flowers_message}}
    )

    await bot.handle_update(
        {"callback_query": {"id": "f5", "from": {"id": 1}, "data": "rubric_sched_add:flowers", "message": flowers_message}}
    )
    wizard_message = bot.pending[1]["rubric_input"]["message"]
    wizard_callback = {"id": "f6", "from": {"id": 1}, "message": wizard_message}
    await bot.handle_update({"callback_query": {**wizard_callback, "data": "rubric_sched_time"}})
    await bot.handle_update({"callback_query": {**wizard_callback, "data": "rubric_sched_hour:9"}})
    await bot.handle_update({"callback_query": {**wizard_callback, "data": "rubric_sched_minute:30"}})
    await bot.handle_update({"callback_query": {**wizard_callback, "data": "rubric_sched_days"}})
    await bot.handle_update({"callback_query": {**wizard_callback, "data": "rubric_sched_day:mon"}})
    await bot.handle_update({"callback_query": {**wizard_callback, "data": "rubric_sched_day:tue"}})
    await bot.handle_update({"callback_query": {**wizard_callback, "data": "rubric_sched_days_done"}})
    await bot.handle_update({"callback_query": {**wizard_callback, "data": "rubric_sched_save"}})

    await bot.handle_update(
        {"callback_query": {"id": "f7", "from": {"id": 1}, "data": "rubric_sched_toggle:flowers:0", "message": flowers_message}}
    )

    await bot.handle_update(
        {"callback_query": {"id": "g1", "from": {"id": 1}, "data": "rubric_channel:guess_arch:main", "message": guess_message}}
    )
    await bot.handle_update(
        {"callback_query": {"id": "g2", "from": {"id": 1}, "data": "rubric_channel_set:-300", "message": guess_message}}
    )
    await bot.handle_update(
        {"callback_query": {"id": "g3", "from": {"id": 1}, "data": "rubric_channel:guess_arch:test", "message": guess_message}}
    )
    await bot.handle_update(
        {"callback_query": {"id": "g4", "from": {"id": 1}, "data": "rubric_channel_set:-400", "message": guess_message}}
    )
    await bot.handle_update(
        {"callback_query": {"id": "g5", "from": {"id": 1}, "data": "rubric_sched_add:guess_arch", "message": guess_message}}
    )
    g_wizard_message = bot.pending[1]["rubric_input"]["message"]
    g_wizard_callback = {"id": "g6", "from": {"id": 1}, "message": g_wizard_message}
    await bot.handle_update({"callback_query": {**g_wizard_callback, "data": "rubric_sched_time"}})
    await bot.handle_update({"callback_query": {**g_wizard_callback, "data": "rubric_sched_hour:10"}})
    await bot.handle_update({"callback_query": {**g_wizard_callback, "data": "rubric_sched_minute:15"}})
    await bot.handle_update({"callback_query": {**g_wizard_callback, "data": "rubric_sched_days"}})
    await bot.handle_update({"callback_query": {**g_wizard_callback, "data": "rubric_sched_day:wed"}})
    await bot.handle_update({"callback_query": {**g_wizard_callback, "data": "rubric_sched_days_done"}})
    await bot.handle_update({"callback_query": {**g_wizard_callback, "data": "rubric_sched_save"}})

    await bot.handle_update(
        {"callback_query": {"id": "f8", "from": {"id": 1}, "data": "rubric_toggle:flowers", "message": flowers_message}}
    )
    await bot.handle_update(
        {"callback_query": {"id": "g7", "from": {"id": 1}, "data": "rubric_toggle:guess_arch", "message": guess_message}}
    )

    flowers_config = bot.data.get_rubric_config("flowers")
    assert flowers_config is not None
    assert flowers_config.get("channel_id") == -100
    assert flowers_config.get("test_channel_id") == -200
    assert flowers_config.get("enabled") is False
    assert flowers_config.get("schedules") and flowers_config["schedules"][0]["time"] == "09:30"
    assert flowers_config["schedules"][0]["enabled"] is False
    assert flowers_config["schedules"][0]["days"] == ["mon", "tue"]

    guess_config = bot.data.get_rubric_config("guess_arch")
    assert guess_config is not None
    assert guess_config.get("channel_id") == -300
    assert guess_config.get("test_channel_id") == -400
    assert guess_config.get("enabled") is False
    assert guess_config.get("schedules") and guess_config["schedules"][0]["time"] == "10:15"
    assert guess_config["schedules"][0]["days"] == ["wed"]

    await bot.close()


@pytest.mark.asyncio
async def test_overlay_size_stays_within_expected_ratio(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    custom_overlay_path = tmp_path / "custom_overlay.png"
    Image.new("RGBA", (640, 320), (255, 0, 0, 255)).save(custom_overlay_path)

    cases = [
        (custom_overlay_path, (1600, 1200)),
        (tmp_path / "missing_overlay.png", (600, 900)),
        (tmp_path / "missing_overlay.png", (240, 960)),
    ]

    for overlay_path, base_size in cases:
        overlay = bot._load_overlay_image(overlay_path, 1, base_size)
        min_side = min(base_size)
        ratio = overlay.width / min_side
        assert overlay.size[0] == overlay.size[1]
        assert 0.10 <= ratio <= 0.16

    await bot.close()



@pytest.mark.asyncio
async def test_overlay_offset_respects_safe_zone(tmp_path, monkeypatch):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    overlays_dir = tmp_path / "overlays"
    overlays_dir.mkdir()

    captured_offsets: list[tuple[int, int] | None] = []
    original_paste = Image.Image.paste

    def capture_paste(self, im, box=None, mask=None):  # type: ignore[override]
        captured_offsets.append(box)
        return original_paste(self, im, box, mask)

    monkeypatch.setattr(Image.Image, "paste", capture_paste)

    def make_asset(idx: int, size: tuple[int, int]) -> Asset:
        path = tmp_path / f"base_{idx}.png"
        Image.new("RGB", size, (255, 255, 255)).save(path)
        return Asset(
            id=idx,
            channel_id=1,
            tg_chat_id=1,
            message_id=idx,
            origin="test",
            caption_template=None,
            caption=None,
            hashtags=None,
            categories=[],
            kind="photo",
            file_id=f"file-{idx}",
            file_unique_id=f"unique-{idx}",
            file_name=path.name,
            mime_type="image/png",
            file_size=None,
            width=size[0],
            height=size[1],
            duration=None,
            recognized_message_id=None,
            exif_present=False,
            latitude=None,
            longitude=None,
            city=None,
            country=None,
            author_user_id=None,
            author_username=None,
            sender_chat_id=None,
            via_bot_id=None,
            forward_from_user=None,
            forward_from_chat=None,
            local_path=str(path),
            metadata=None,
            vision_results=None,
            rubric_id=None,
            vision_category=None,
            vision_arch_view=None,
            vision_photo_weather=None,
            vision_flower_varieties=None,
            vision_confidence=None,
            vision_caption=None,
        )

    large_asset = make_asset(1, (1280, 720))
    captured_offsets.clear()
    bot._overlay_number(large_asset, 1, {"overlays_dir": str(overlays_dir)})
    assert captured_offsets[-1] == (24, 24)

    small_asset = make_asset(2, (320, 640))
    captured_offsets.clear()
    bot._overlay_number(small_asset, 2, {"overlays_dir": str(overlays_dir)})
    assert captured_offsets[-1] == (12, 12)

    await bot.close()


@pytest.mark.asyncio
async def test_rubric_publish_callback_success(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    bot.db.execute(
        "INSERT OR REPLACE INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, ?)",
        (1, "boss", "+00:00"),
    )
    bot.db.commit()
    _insert_rubric(bot, "flowers", {"enabled": True, "channel_id": -1}, rubric_id=10)

    api_calls: list[tuple[str, dict | None, Any]] = []

    async def fake_api(method, payload=None, *, files=None):
        api_calls.append((method, payload, files))
        return {"ok": True}

    bot.api_request = fake_api  # type: ignore[assignment]

    recorded: dict[str, Any] = {}

    def fake_enqueue(
        code: str,
        *,
        test: bool = False,
        channel_id: int | None = None,
        initiator_id: int | None = None,
        instructions: str | None = None,
    ) -> int:
        recorded["code"] = code
        recorded["test"] = test
        recorded["channel_id"] = channel_id
        recorded["initiator_id"] = initiator_id
        recorded["instructions"] = instructions
        return 314

    bot.enqueue_rubric = fake_enqueue  # type: ignore[assignment]

    message = {"message_id": 5, "chat": {"id": 1}}
    await bot.handle_update(
        {
            "callback_query": {
                "id": "cb-confirm",
                "from": {"id": 1},
                "data": "rubric_publish_confirm:flowers:prod",
                "message": message,
            }
        }
    )
    assert bot.rubric_pending_runs[(1, "flowers")] == "prod"
    api_calls.clear()

    await bot.handle_update(
        {
            "callback_query": {
                "id": "cb-success",
                "from": {"id": 1},
                "data": "rubric_publish_execute:flowers:prod",
                "message": message,
            }
        }
    )

    assert recorded["code"] == "flowers"
    assert recorded["test"] is False
    assert bot.rubric_pending_runs == {}

    ack_payloads = [payload for method, payload, _ in api_calls if method == "answerCallbackQuery"]
    assert any(payload and payload.get("callback_query_id") == "cb-success" for payload in ack_payloads)
    assert any(payload and "Задача поставлена" in payload.get("text", "") for payload in ack_payloads)

    send_payloads = [payload for method, payload, _ in api_calls if method == "sendMessage"]
    assert send_payloads, "callback should send confirmation message"
    confirmation = send_payloads[-1]
    assert confirmation["chat_id"] == 1
    assert "✅ Рабочая публикация рубрики flowers" in confirmation["text"]
    assert "#314" in confirmation["text"]

    await bot.close()


@pytest.mark.asyncio
async def test_rubric_publish_callback_error(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    bot.db.execute(
        "INSERT OR REPLACE INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, ?)",
        (1, "boss", "+00:00"),
    )
    bot.db.commit()
    _insert_rubric(bot, "flowers", {"enabled": True, "test_channel_id": -2}, rubric_id=11)

    api_calls: list[tuple[str, dict | None, Any]] = []

    async def fake_api(method, payload=None, *, files=None):
        api_calls.append((method, payload, files))
        return {"ok": True}

    bot.api_request = fake_api  # type: ignore[assignment]

    def failing_enqueue(
        code: str,
        *,
        test: bool = False,
        channel_id: int | None = None,
        initiator_id: int | None = None,
        instructions: str | None = None,
    ) -> int:
        raise ValueError("нет подходящих ассетов")

    bot.enqueue_rubric = failing_enqueue  # type: ignore[assignment]

    message = {"message_id": 6, "chat": {"id": 1}}
    await bot.handle_update(
        {
            "callback_query": {
                "id": "cb-error-confirm",
                "from": {"id": 1},
                "data": "rubric_publish_confirm:flowers:test",
                "message": message,
            }
        }
    )
    assert bot.rubric_pending_runs[(1, "flowers")] == "test"
    api_calls.clear()

    await bot.handle_update(
        {
            "callback_query": {
                "id": "cb-error",
                "from": {"id": 1},
                "data": "rubric_publish_execute:flowers:test",
                "message": message,
            }
        }
    )

    ack_payloads = [payload for method, payload, _ in api_calls if method == "answerCallbackQuery"]
    assert any(payload and payload.get("callback_query_id") == "cb-error" for payload in ack_payloads)
    assert any(payload and payload.get("show_alert") is True for payload in ack_payloads)
    assert bot.rubric_pending_runs == {}

    send_payloads = [payload for method, payload, _ in api_calls if method == "sendMessage"]
    assert send_payloads, "callback should send error details"
    error_message = send_payloads[-1]
    assert error_message["chat_id"] == 1
    assert "⚠️ тестовая публикация рубрики flowers не запущена" in error_message["text"]
    assert "нет подходящих ассетов" in error_message["text"]

    await bot.close()


@pytest.mark.asyncio
async def test_rubric_publish_cancel_resets_pending(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    bot.db.execute(
        "INSERT OR REPLACE INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, ?)",
        (1, "boss", "+00:00"),
    )
    bot.db.commit()
    _insert_rubric(bot, "flowers", {"enabled": True, "channel_id": -1}, rubric_id=12)

    api_calls: list[tuple[str, dict | None, Any]] = []

    async def fake_api(method, payload=None, *, files=None):
        api_calls.append((method, payload, files))
        return {"ok": True}

    bot.api_request = fake_api  # type: ignore[assignment]

    invoked = {"called": False}

    def fail_if_called(*args, **kwargs):
        invoked["called"] = True
        return 0

    bot.enqueue_rubric = fail_if_called  # type: ignore[assignment]

    message = {"message_id": 7, "chat": {"id": 1}}
    await bot.handle_update(
        {
            "callback_query": {
                "id": "cb-cancel-confirm",
                "from": {"id": 1},
                "data": "rubric_publish_confirm:flowers:prod",
                "message": message,
            }
        }
    )
    assert bot.rubric_pending_runs[(1, "flowers")] == "prod"
    api_calls.clear()

    await bot.handle_update(
        {
            "callback_query": {
                "id": "cb-cancel",
                "from": {"id": 1},
                "data": "rubric_publish_cancel:flowers",
                "message": message,
            }
        }
    )

    assert bot.rubric_pending_runs == {}
    assert invoked["called"] is False
    ack_payloads = [payload for method, payload, _ in api_calls if method == "answerCallbackQuery"]
    assert any(payload and payload.get("callback_query_id") == "cb-cancel" for payload in ack_payloads)

    await bot.close()
