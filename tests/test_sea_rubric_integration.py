import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from aiohttp.test_utils import TestServer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import main as main_module
from openai_client import OpenAIResponse
from tests.fixtures.sea import create_sea_asset, create_stub_image, seed_sea_environment


class FakeOpenAI:
    def __init__(self) -> None:
        self.api_key = "fake-key"
        self.calls: list[dict[str, Any]] = []

    async def generate_json(self, **kwargs) -> OpenAIResponse:  # type: ignore[override]
        self.calls.append(kwargs)
        prompt = kwargs.get("user_prompt", "")
        if '"storm_state": "calm"' in prompt:
            caption = "Порадую закатом над морем — воздух тёплый и тихий."
            hashtags = ["морем"]
        elif '"wind_class": "strong"' in prompt:
            caption = "Сегодня шторм на море — ветер сбивающий с ног и россыпь пены."
            hashtags = ["БалтийскоеМоре"]
        elif '"wind_class": "very_strong"' in prompt:
            caption = "Сегодня шторм на море. Ураганный ветер кружит песок.\n" + (
                "Ураганный ветер кружит песок. " * 80
            )
            hashtags = ["моря"]
        else:
            caption = "Море сегодня спокойное."
            hashtags = []
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "endpoint": "/v1/responses",
            "request_id": f"req-{len(self.calls)}",
        }
        return OpenAIResponse({"caption": caption, "hashtags": hashtags}, usage)


@pytest.mark.asyncio
async def test_sea_rubric_end_to_end(monkeypatch, tmp_path):
    webhook_url = "https://example.com/base"
    db_path = Path("/tmp/test_bot.db")
    if db_path.exists():
        db_path.unlink()

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    monkeypatch.setenv("WEBHOOK_URL", webhook_url)
    monkeypatch.setenv("4O_API_KEY", "dummy-token")
    monkeypatch.setattr(main_module, "DB_PATH", str(db_path))

    async def noop(*_args, **_kwargs):
        return None

    async def bot_noop(self, *_args, **_kwargs):
        return None

    monkeypatch.setattr(main_module, "ensure_webhook", noop)
    monkeypatch.setattr(main_module.Bot, "run_openai_health_check", bot_noop)
    monkeypatch.setattr(main_module.Bot, "collect_weather", bot_noop)
    monkeypatch.setattr(main_module.Bot, "collect_sea", bot_noop)
    monkeypatch.setattr(main_module.Bot, "process_weather_channels", bot_noop)
    monkeypatch.setattr(main_module.Bot, "process_rubric_schedule", bot_noop)

    requests_log: list[dict[str, Any]] = []
    expected_webhook = webhook_url.rstrip("/") + "/webhook"

    async def fake_api_request(self, method: str, data: Any = None, *, files: Any = None) -> dict[str, Any]:
        entry = {"method": method, "data": data, "files": files}
        requests_log.append(entry)
        if method == "getWebhookInfo":
            return {"ok": True, "result": {"url": ""}}
        if method == "setWebhook":
            assert data and data.get("url") == expected_webhook
            return {"ok": True, "result": {"ok": True}}
        if method == "sendPhoto":
            assert files and "photo" in files
            assert data and data.get("parse_mode") == "HTML"
            counter = sum(1 for item in requests_log if item["method"] == "sendPhoto")
            return {"ok": True, "result": {"message_id": 100 + counter}}
        if method == "deleteMessage":
            return {"ok": True}
        return {"ok": True}

    monkeypatch.setattr(main_module.Bot, "api_request", fake_api_request, raising=False)

    async def fake_reverse_geocode(self, lat: float, lon: float) -> dict[str, Any]:
        if abs(lat - 54.9604) < 0.2 and abs(lon - 20.4721) < 0.2:
            return {"city": "Зеленоградск"}
        return {}

    monkeypatch.setattr(main_module.Bot, "_reverse_geocode", fake_reverse_geocode, raising=False)

    app = main_module.create_app()
    async with TestServer(app) as server:
        bot = server.app["bot"]
        bot.openai = FakeOpenAI()

        def fetch_history() -> list[dict[str, Any]]:
            rows = bot.db.execute("SELECT metadata FROM posts_history ORDER BY id").fetchall()
            result: list[dict[str, Any]] = []
            for row in rows:
                raw = row["metadata"] if "metadata" in row.keys() else row[0]
                result.append(json.loads(raw) if raw else {})
            return result

        rubric = bot.data.get_rubric_by_code("sea")
        assert rubric is not None
        assert rubric.title == "Море / Закат на море"
        assert rubric.config.get("sea_id") == 1

        sea_channel = -900123
        updated_config = dict(rubric.config or {})
        updated_config.update(
            {
                "enabled": True,
                "channel_id": sea_channel,
                "test_channel_id": sea_channel,
                "sea_id": 1,
                "schedules": [
                    {
                        "time": "08:45",
                        "tz": "+03:00",
                        "days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
                        "channel_id": sea_channel,
                        "enabled": True,
                    }
                ],
            }
        )
        bot.data.save_rubric_config("sea", updated_config)
        rubric = bot.data.get_rubric_by_code("sea")
        assert rubric is not None
        assert rubric.config.get("sea_id") == 1
        assert rubric.config.get("schedules")
        assert rubric.config["schedules"][0]["time"] == "08:45"

        seed_sea_environment(
            bot,
            sea_id=1,
            sea_lat=54.95,
            sea_lon=20.2,
            wave=0.3,
            water_temp=9.0,
            city_id=101,
            city_name="Зеленоградск",
            city_lat=54.9604,
            city_lon=20.4721,
            wind_speed=5.0,
        )

        sunset_file = create_stub_image(tmp_path, "sunset.jpg")
        storm_file = create_stub_image(tmp_path, "storm.jpg")
        heavy_storm_file = create_stub_image(tmp_path, "storm-heavy.jpg")

        sunset_id = create_sea_asset(
            bot,
            rubric_id=rubric.id,
            message_id=201,
            file_name="sunset.jpg",
            local_path=sunset_file,
            tags=["sunset", "sea"],
            latitude=54.9604,
            longitude=20.4721,
            sea_wave_score=1.5,
            photo_sky="sunny",
            is_sunset=True,
        )
        sunset_asset = bot.data.get_asset(sunset_id)
        assert sunset_asset is not None
        assert "закат" not in {c.lower() for c in sunset_asset.categories}
        bot.data.update_asset_categories_merge(sunset_id, ["закат"])
        sunset_asset = bot.data.get_asset(sunset_id)
        assert sunset_asset is not None
        assert "закат" in {c.lower() for c in sunset_asset.categories}

        storm_id = create_sea_asset(
            bot,
            rubric_id=rubric.id,
            message_id=202,
            file_name="storm.jpg",
            local_path=storm_file,
            tags=["storm", "waves"],
            sea_wave_score=8.0,
            photo_sky="mostly_cloudy",
            is_sunset=False,
        )
        heavy_storm_id = create_sea_asset(
            bot,
            rubric_id=rubric.id,
            message_id=203,
            file_name="storm-heavy.jpg",
            local_path=heavy_storm_file,
            tags=["sea"],
            categories=["Шторм"],
            sea_wave_score=9.5,
            photo_sky="overcast",
            is_sunset=False,
        )

        assert await bot.publish_rubric("sea") is True
        send_calls = [entry for entry in requests_log if entry["method"] == "sendPhoto"]
        assert len(send_calls) == 1
        calm_caption = send_calls[0]["data"]["caption"]
        assert "закат" in calm_caption.lower()
        assert "шторм" not in calm_caption.lower()
        assert "#море #БалтийскоеМоре #Зеленоградск" in calm_caption
        assert "📂 Полюбить 39" in calm_caption
        assert len(calm_caption) < 1000
        calm_meta = fetch_history()[-1]
        assert calm_meta["storm_state"] == "calm"
        assert calm_meta["wind_class"] is None
        assert calm_meta["place_hashtag"] == "#Зеленоградск"
        assert calm_meta["sunset_selected"] is True
        assert calm_meta["wind_speed_ms"] == pytest.approx(5.0)
        assert sunset_id in calm_meta["asset_ids"]
        assert bot.data.get_asset(sunset_id) is None

        bot.db.execute(
            "UPDATE sea_cache SET wave=?, updated=? WHERE sea_id=?",
            (1.6, datetime.utcnow().isoformat(), 1),
        )
        bot.db.execute(
            "UPDATE sea_conditions SET wave_height_m=?, wind_speed_10m_ms=?, wind_speed_10m_kmh=?, wind_gusts_10m_ms=?, wind_gusts_10m_kmh=?, wind_units=?, wind_gusts_units=?, wind_time_ref=?, cloud_cover_pct=?, updated=? WHERE sea_id=?",
            (1.6, 12.0, 43.2, 15.0, 54.0, "m/s", "km/h", datetime.utcnow().isoformat(), 65.0, datetime.utcnow().isoformat(), 1),
        )
        bot.db.execute(
            "UPDATE weather_cache_hour SET wind_speed=?, timestamp=? WHERE city_id=?",
            (12.0, datetime.utcnow().isoformat(), 101),
        )
        bot.db.commit()

        assert await bot.publish_rubric("sea") is True
        send_calls = [entry for entry in requests_log if entry["method"] == "sendPhoto"]
        assert len(send_calls) == 2
        storm_caption = send_calls[1]["data"]["caption"]
        assert "шторм" in storm_caption.lower()
        lowered = storm_caption.lower()
        assert ("сбивающ" in lowered) or ("ураган" in lowered)
        assert "#море #БалтийскоеМоре" in storm_caption
        assert "#Зеленоградск" not in storm_caption
        storm_meta = fetch_history()[-1]
        assert storm_meta["storm_state"] == "strong_storm"
        assert storm_meta["wind_class"] == "very_strong"
        assert storm_meta["place_hashtag"] is None
        assert storm_meta["wind_speed_ms"] == pytest.approx(12.0)
        # Verify enhanced wind data is preserved in metadata
        assert storm_meta.get("wind_speed_kmh") == pytest.approx(43.2)
        assert storm_meta.get("wind_gust_ms") == pytest.approx(15.0)
        assert storm_meta.get("wind_gust_kmh") == pytest.approx(54.0)
        assert storm_meta.get("wind_units") == "m/s"
        assert storm_meta.get("wind_gust_units") == "km/h"
        assert storm_meta.get("cloud_cover_pct") == 65.0
        assert bot.data.get_asset(storm_id) is None

        bot.db.execute(
            "UPDATE sea_conditions SET wave_height_m=?, wind_speed_10m_ms=?, wind_speed_10m_kmh=?, wind_gusts_10m_ms=?, wind_gusts_10m_kmh=?, wind_units=?, wind_gusts_units=?, wind_time_ref=?, cloud_cover_pct=?, updated=? WHERE sea_id=?",
            (2.2, 16.0, 57.6, 20.0, 72.0, "m/s", "km/h", datetime.utcnow().isoformat(), 85.0, datetime.utcnow().isoformat(), 1),
        )
        bot.db.execute(
            "UPDATE weather_cache_hour SET wind_speed=?, timestamp=? WHERE city_id=?",
            (16.0, datetime.utcnow().isoformat(), 101),
        )
        bot.db.commit()

        assert await bot.publish_rubric("sea") is True
        send_calls = [entry for entry in requests_log if entry["method"] == "sendPhoto"]
        assert len(send_calls) == 3
        heavy_caption = send_calls[2]["data"]["caption"]
        assert "шторм" in heavy_caption.lower()
        assert "ураган" in heavy_caption.lower()
        assert len(heavy_caption) < 1000
        assert "#море #БалтийскоеМоре" in heavy_caption
        assert "📂 Полюбить 39" in heavy_caption
        heavy_meta = fetch_history()[-1]
        assert heavy_meta["wind_class"] == "very_strong"
        assert heavy_meta["storm_state"] == "strong_storm"
        assert heavy_meta["wind_speed_ms"] == pytest.approx(16.0)
        # Verify enhanced wind data is preserved in metadata for heavy storm
        assert heavy_meta.get("wind_speed_kmh") == pytest.approx(57.6)
        assert heavy_meta.get("wind_gust_ms") == pytest.approx(20.0)
        assert heavy_meta.get("wind_gust_kmh") == pytest.approx(72.0)
        assert heavy_meta.get("wind_units") == "m/s"
        assert heavy_meta.get("wind_gust_units") == "km/h"
        assert heavy_meta.get("cloud_cover_pct") == 85.0
        assert bot.data.get_asset(heavy_storm_id) is None

        for call in send_calls:
            files = call["files"]
            assert files and "photo" in files
            photo_payload = files["photo"]
            assert isinstance(photo_payload, tuple)
            assert photo_payload[1]

        history = fetch_history()
        assert len(history) == 3
        assert bot.openai.calls and len(bot.openai.calls) == 3

    if db_path.exists():
        db_path.unlink()
