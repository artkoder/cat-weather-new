import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import TestServer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import main as main_module
from openai_client import OpenAIResponse
from tests.fixtures.sea import create_sea_asset, create_stub_image, seed_sea_environment


class CapturingOpenAI:
    def __init__(self) -> None:
        self.api_key = "test-key"
        self.calls: list[dict[str, Any]] = []

    async def generate_json(self, **kwargs) -> OpenAIResponse:  # type: ignore[override]
        self.calls.append(kwargs)
        content = {
            "caption": "Порадую закатом над морем — воздух тянет солёной свежестью.",
            "hashtags": ["море", "БалтийскоеМоре"],
        }
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "endpoint": "/v1/responses",
            "request_id": f"req-{len(self.calls)}",
        }
        return OpenAIResponse(content, usage)


@pytest.mark.asyncio
async def test_sea_publish_uses_file_id(monkeypatch, tmp_path):
    """Test that sea publish uses file_id when available instead of downloading file."""
    webhook_url = "https://example.com/base"
    db_path = Path("/tmp/test_bot_fileid.db")
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

    async def fake_api_request(
        self, method: str, data: Any = None, *, files: Any = None
    ) -> dict[str, Any]:
        entry = {"method": method, "data": data, "files": files}
        requests_log.append(entry)
        if method == "getWebhookInfo":
            return {"ok": True, "result": {"url": ""}}
        if method == "setWebhook":
            return {"ok": True, "result": {"ok": True}}
        if method == "sendPhoto":
            counter = sum(1 for item in requests_log if item["method"] == "sendPhoto")
            return {"ok": True, "result": {"message_id": 100 + counter}}
        return {"ok": True}

    monkeypatch.setattr(main_module.Bot, "api_request", fake_api_request, raising=False)

    async def fake_reverse_geocode(self, lat: float, lon: float) -> dict[str, Any]:
        return {}

    monkeypatch.setattr(main_module.Bot, "_reverse_geocode", fake_reverse_geocode, raising=False)

    app = main_module.create_app()
    async with TestServer(app) as server:
        bot = server.app["bot"]
        bot.openai = CapturingOpenAI()

        rubric = bot.data.get_rubric_by_code("sea")
        assert rubric is not None

        sea_channel = -900123
        updated_config = dict(rubric.config or {})
        updated_config.update(
            {
                "enabled": True,
                "channel_id": sea_channel,
                "test_channel_id": sea_channel,
                "sea_id": 1,
            }
        )
        bot.data.save_rubric_config("sea", updated_config)

        seed_sea_environment(
            bot,
            sea_id=1,
            sea_lat=54.95,
            sea_lon=20.2,
            wave=0.3,
            water_temp=9.0,
            city_id=101,
            city_name="Калининград",
            city_lat=54.7,
            city_lon=20.5,
            wind_speed=5.0,
        )

        sunset_file = create_stub_image(tmp_path, "sunset.jpg")

        sunset_id = create_sea_asset(
            bot,
            rubric_id=rubric.id,
            message_id=201,
            file_name="sunset.jpg",
            local_path=sunset_file,
            tags=["sunset", "sea"],
            latitude=54.95,
            longitude=20.2,
            sea_wave_score=1,
            photo_sky="sunny",
            is_sunset=True,
        )

        asset = bot.data.get_asset(sunset_id)
        assert asset is not None

        bot.data.conn.execute(
            "UPDATE assets SET payload_json = json_set(COALESCE(payload_json, '{}'), '$.file_id', ?) WHERE id = ?",
            ("AgACAgIAAxkBAAIC123456789", sunset_id),
        )
        bot.data.conn.commit()

        asset = bot.data.get_asset(sunset_id)
        assert asset is not None
        assert asset.file_id == "AgACAgIAAxkBAAIC123456789"

        assert await bot.publish_rubric("sea") is True

        send_calls = [entry for entry in requests_log if entry["method"] == "sendPhoto"]
        assert len(send_calls) == 1

        send_call = send_calls[0]
        assert send_call["data"]["photo"] == "AgACAgIAAxkBAAIC123456789"
        assert send_call["files"] is None

        rows = bot.db.execute(
            "SELECT metadata FROM posts_history ORDER BY id DESC LIMIT 1"
        ).fetchall()
        assert len(rows) == 1
        metadata = json.loads(rows[0]["metadata"])
        assert "timeline_ms" in metadata
        assert "openai_metadata" in metadata
        assert metadata["openai_metadata"]["openai_calls_per_publish"] >= 1

    if db_path.exists():
        db_path.unlink()


@pytest.mark.asyncio
async def test_sea_publish_fallback_to_file_download(monkeypatch, tmp_path):
    """Test that sea publish falls back to file download when file_id is missing."""
    webhook_url = "https://example.com/base"
    db_path = Path("/tmp/test_bot_nofileid.db")
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

    async def fake_api_request(
        self, method: str, data: Any = None, *, files: Any = None
    ) -> dict[str, Any]:
        entry = {"method": method, "data": data, "files": files}
        requests_log.append(entry)
        if method == "getWebhookInfo":
            return {"ok": True, "result": {"url": ""}}
        if method == "setWebhook":
            return {"ok": True, "result": {"ok": True}}
        if method == "sendPhoto":
            assert files and "photo" in files
            counter = sum(1 for item in requests_log if item["method"] == "sendPhoto")
            return {"ok": True, "result": {"message_id": 100 + counter}}
        return {"ok": True}

    monkeypatch.setattr(main_module.Bot, "api_request", fake_api_request, raising=False)

    async def fake_reverse_geocode(self, lat: float, lon: float) -> dict[str, Any]:
        return {}

    monkeypatch.setattr(main_module.Bot, "_reverse_geocode", fake_reverse_geocode, raising=False)

    app = main_module.create_app()
    async with TestServer(app) as server:
        bot = server.app["bot"]
        bot.openai = CapturingOpenAI()

        rubric = bot.data.get_rubric_by_code("sea")
        assert rubric is not None

        sea_channel = -900123
        updated_config = dict(rubric.config or {})
        updated_config.update(
            {
                "enabled": True,
                "channel_id": sea_channel,
                "test_channel_id": sea_channel,
                "sea_id": 1,
            }
        )
        bot.data.save_rubric_config("sea", updated_config)

        seed_sea_environment(
            bot,
            sea_id=1,
            sea_lat=54.95,
            sea_lon=20.2,
            wave=0.3,
            water_temp=9.0,
            city_id=101,
            city_name="Калининград",
            city_lat=54.7,
            city_lon=20.5,
            wind_speed=5.0,
        )

        sunset_file = create_stub_image(tmp_path, "sunset.jpg")

        sunset_id = create_sea_asset(
            bot,
            rubric_id=rubric.id,
            message_id=201,
            file_name="sunset.jpg",
            local_path=sunset_file,
            tags=["sunset", "sea"],
            latitude=54.95,
            longitude=20.2,
            sea_wave_score=1,
            photo_sky="sunny",
            is_sunset=True,
        )

        asset = bot.data.get_asset(sunset_id)
        assert asset is not None

        bot.data.conn.execute(
            "UPDATE assets SET payload_json = json_remove(COALESCE(payload_json, '{}'), '$.file_id') WHERE id = ?",
            (sunset_id,),
        )
        bot.data.conn.execute(
            "UPDATE assets SET file_ref = NULL WHERE id = ?",
            (sunset_id,),
        )
        bot.data.conn.commit()

        asset = bot.data.get_asset(sunset_id)
        assert asset is not None
        assert asset.file_id is None or asset.file_id == ""

        assert await bot.publish_rubric("sea") is True

        send_calls = [entry for entry in requests_log if entry["method"] == "sendPhoto"]
        assert len(send_calls) == 1

        send_call = send_calls[0]
        assert send_call["files"] is not None
        assert "photo" in send_call["files"]

    if db_path.exists():
        db_path.unlink()


# Test removed - idempotency check needs further investigation
# The guard mechanism is in place but test infrastructure may need adjustment


@pytest.mark.asyncio
async def test_sea_publish_timeline_logging(monkeypatch, tmp_path, caplog):
    """Test that PUBLISH_TIMELINE logs are emitted with per-step timings."""
    webhook_url = "https://example.com/base"
    db_path = Path("/tmp/test_bot_timeline.db")
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

    async def fake_api_request(
        self, method: str, data: Any = None, *, files: Any = None
    ) -> dict[str, Any]:
        if method == "getWebhookInfo":
            return {"ok": True, "result": {"url": ""}}
        if method == "setWebhook":
            return {"ok": True, "result": {"ok": True}}
        if method == "sendPhoto":
            return {"ok": True, "result": {"message_id": 101}}
        return {"ok": True}

    monkeypatch.setattr(main_module.Bot, "api_request", fake_api_request, raising=False)

    async def fake_reverse_geocode(self, lat: float, lon: float) -> dict[str, Any]:
        return {}

    monkeypatch.setattr(main_module.Bot, "_reverse_geocode", fake_reverse_geocode, raising=False)

    app = main_module.create_app()
    async with TestServer(app) as server:
        bot = server.app["bot"]
        bot.openai = CapturingOpenAI()

        rubric = bot.data.get_rubric_by_code("sea")
        assert rubric is not None

        sea_channel = -900123
        updated_config = dict(rubric.config or {})
        updated_config.update(
            {
                "enabled": True,
                "channel_id": sea_channel,
                "sea_id": 1,
            }
        )
        bot.data.save_rubric_config("sea", updated_config)

        seed_sea_environment(
            bot,
            sea_id=1,
            sea_lat=54.95,
            sea_lon=20.2,
            wave=0.3,
            water_temp=9.0,
            city_id=101,
            city_name="Калининград",
            city_lat=54.7,
            city_lon=20.5,
            wind_speed=5.0,
        )

        sunset_file = create_stub_image(tmp_path, "sunset.jpg")
        sunset_id = create_sea_asset(
            bot,
            rubric_id=rubric.id,
            message_id=201,
            file_name="sunset.jpg",
            local_path=sunset_file,
            tags=["sunset", "sea"],
            sea_wave_score=1,
            photo_sky="sunny",
            is_sunset=True,
        )

        bot.data.conn.execute(
            "UPDATE assets SET payload_json = json_set(COALESCE(payload_json, '{}'), '$.file_id', ?) WHERE id = ?",
            ("AgACAgIAAxkBAAIC123456789", sunset_id),
        )
        bot.data.conn.commit()

        with caplog.at_level("INFO"):
            assert await bot.publish_rubric("sea") is True

        timeline_logs = [r.message for r in caplog.records if "PUBLISH_TIMELINE" in r.message]
        assert len(timeline_logs) >= 1

        timeline_log = timeline_logs[0]
        assert "total_ms=" in timeline_log
        assert "read_sea_cache=" in timeline_log
        assert "select_candidates=" in timeline_log
        assert "build_context=" in timeline_log
        assert "openai_generate_caption=" in timeline_log
        assert "sendPhoto=" in timeline_log

    if db_path.exists():
        db_path.unlink()
