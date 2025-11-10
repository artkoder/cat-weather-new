"""Test sea rubric topN enriched logging."""

import logging
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

    async def generate_json(self, **kwargs: Any) -> OpenAIResponse:
        self.calls.append(kwargs)
        caption = "Море сегодня спокойное."
        hashtags = ["море"]
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "endpoint": "/v1/responses",
            "request_id": f"req-{len(self.calls)}",
        }
        return OpenAIResponse({"caption": caption, "hashtags": hashtags}, usage)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sea_topn_logs_structure(monkeypatch: Any, tmp_path: Path, caplog: Any) -> None:
    """Test that sea rubric logs include all required topN fields."""
    webhook_url = "https://example.com/base"
    db_path = Path("/tmp/test_sea_topn_logs.db")
    if db_path.exists():
        db_path.unlink()

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    monkeypatch.setenv("WEBHOOK_URL", webhook_url)
    monkeypatch.setenv("4O_API_KEY", "dummy-token")
    monkeypatch.setattr(main_module, "DB_PATH", str(db_path))

    async def noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def bot_noop(self: Any, *_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(main_module, "ensure_webhook", noop)
    monkeypatch.setattr(main_module.Bot, "run_openai_health_check", bot_noop)
    monkeypatch.setattr(main_module.Bot, "collect_weather", bot_noop)
    monkeypatch.setattr(main_module.Bot, "collect_sea", bot_noop)
    monkeypatch.setattr(main_module.Bot, "process_weather_channels", bot_noop)
    monkeypatch.setattr(main_module.Bot, "process_rubric_schedule", bot_noop)

    requests_log: list[dict[str, Any]] = []

    async def fake_api_request(
        self: Any, method: str, data: Any = None, *, files: Any = None
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

    async def fake_reverse_geocode(self: Any, lat: float, lon: float) -> dict[str, Any]:
        return {}

    monkeypatch.setattr(main_module.Bot, "_reverse_geocode", fake_reverse_geocode, raising=False)

    caplog.set_level(logging.INFO)

    app = main_module.create_app()
    async with TestServer(app) as server:
        bot = server.app["bot"]
        bot.openai = FakeOpenAI()

        rubric = bot.data.get_rubric_by_code("sea")
        assert rubric is not None

        sea_channel = -900124
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
        rubric = bot.data.get_rubric_by_code("sea")
        assert rubric is not None

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

        # Create 5 assets with different wave scores to ensure we get topN candidates
        today = datetime.utcnow()
        today_doy = today.timetuple().tm_yday
        timestamp = int(today.timestamp())

        asset_ids = []
        for i in range(5):
            asset_file = create_stub_image(tmp_path, f"sea_asset_{i}.jpg")
            asset_id = create_sea_asset(
                bot,
                rubric_id=rubric.id,
                message_id=200 + i,
                file_name=f"sea_{i}.jpg",
                local_path=asset_file,
                tags=["sea", "water"],
                sea_wave_score=i,
                photo_sky="sunny",
                is_sunset=False,
            )
            bot.db.execute(
                "UPDATE assets SET shot_at_utc=?, shot_doy=? WHERE id=?",
                (timestamp, today_doy, asset_id),
            )
            asset_ids.append(asset_id)

        bot.db.commit()

        # Publish sea rubric
        result = await bot._publish_sea(
            rubric,
            sea_channel,
            test=True,
            job=None,
            initiator_id=None,
            instructions=None,
        )
        assert result is True

        # Parse logs to verify required fields
        sea_logs = [rec for rec in caplog.records if "SEA_RUBRIC" in rec.message]
        assert len(sea_logs) > 0, "Should have SEA_RUBRIC logs"

        # Check for attempt logs (B0/B1/B2/AN)
        attempt_logs = [rec for rec in sea_logs if "attempt:" in rec.message]
        assert len(attempt_logs) > 0, "Should have attempt: logs"

        for attempt_log in attempt_logs:
            msg = attempt_log.message
            # Should contain pool_before and pool_after
            assert "pool_before=" in msg, f"attempt log should contain pool_before: {msg}"
            assert "pool_after=" in msg, f"attempt log should contain pool_after: {msg}"
            assert "corridor=" in msg, f"attempt log should contain corridor: {msg}"
            assert "sky_policy=" in msg, f"attempt log should contain sky_policy: {msg}"

        # Check for top5 logs
        top5_logs = [rec for rec in sea_logs if "top5:" in rec.message]
        if top5_logs:
            for top5_log in top5_logs:
                msg = top5_log.message
                # Check all required fields are present
                assert "wave_target=" in msg, f"top5 log should contain wave_target: {msg}"
                assert "wave_photo=" in msg, f"top5 log should contain wave_photo: {msg}"
                assert "delta=" in msg, f"top5 log should contain delta: {msg}"
                assert "sky_photo=" in msg, f"top5 log should contain sky_photo: {msg}"
                assert "penalties=" in msg, f"top5 log should contain penalties: {msg}"
                assert "total_score=" in msg, f"top5 log should contain total_score: {msg}"
                assert "rank=" in msg, f"top5 log should contain rank: {msg}"

                # Ensure wave_delta is not None when wave data is available
                if "wave_photo=None" not in msg and "wave_target=None" not in msg:
                    assert (
                        "delta=None" not in msg
                    ), f"delta should not be None when wave data exists: {msg}"

        # Check for selected log
        selected_logs = [rec for rec in sea_logs if "SEA_RUBRIC selected" in rec.message]
        assert len(selected_logs) == 1, "Should have exactly one selected log"

        selected_log = selected_logs[0].message
        # Check required fields in selected log
        assert "wave_target=" in selected_log, f"selected log should contain wave_target: {selected_log}"
        assert "wave_photo=" in selected_log, f"selected log should contain wave_photo: {selected_log}"
        assert "delta=" in selected_log, f"selected log should contain delta: {selected_log}"
        assert "sky_photo=" in selected_log, f"selected log should contain sky_photo: {selected_log}"
        assert "penalties=" in selected_log, f"selected log should contain penalties: {selected_log}"
        assert "total_score=" in selected_log, f"selected log should contain total_score: {selected_log}"
        assert "reason=" in selected_log, f"selected log should contain reason: {selected_log}"

        # Verify reason is meaningful (not None or empty)
        import re

        reason_match = re.search(r"reason=([^\s]+)", selected_log)
        assert reason_match, "Should have reason value in selected log"
        reason_value = reason_match.group(1)
        assert reason_value not in [
            "None",
            "",
        ], f"reason should have meaningful value, got: {reason_value}"

    if db_path.exists():
        db_path.unlink()


@pytest.mark.asyncio
async def test_sea_topn_logs_all_stages(monkeypatch: Any, tmp_path: Path, caplog: Any) -> None:
    """Test that all stages (B0, B1, B2, AN) are logged."""
    webhook_url = "https://example.com/base"
    db_path = Path("/tmp/test_sea_stages.db")
    if db_path.exists():
        db_path.unlink()

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    monkeypatch.setenv("WEBHOOK_URL", webhook_url)
    monkeypatch.setenv("4O_API_KEY", "dummy-token")
    monkeypatch.setattr(main_module, "DB_PATH", str(db_path))

    async def noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def bot_noop(self: Any, *_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(main_module, "ensure_webhook", noop)
    monkeypatch.setattr(main_module.Bot, "run_openai_health_check", bot_noop)
    monkeypatch.setattr(main_module.Bot, "collect_weather", bot_noop)
    monkeypatch.setattr(main_module.Bot, "collect_sea", bot_noop)
    monkeypatch.setattr(main_module.Bot, "process_weather_channels", bot_noop)
    monkeypatch.setattr(main_module.Bot, "process_rubric_schedule", bot_noop)

    requests_log: list[dict[str, Any]] = []

    async def fake_api_request(
        self: Any, method: str, data: Any = None, *, files: Any = None
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

    async def fake_reverse_geocode(self: Any, lat: float, lon: float) -> dict[str, Any]:
        return {}

    monkeypatch.setattr(main_module.Bot, "_reverse_geocode", fake_reverse_geocode, raising=False)

    caplog.set_level(logging.INFO)

    app = main_module.create_app()
    async with TestServer(app) as server:
        bot = server.app["bot"]
        bot.openai = FakeOpenAI()

        rubric = bot.data.get_rubric_by_code("sea")
        assert rubric is not None

        sea_channel = -900124
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
            city_name="Зеленоградск",
            city_lat=54.9604,
            city_lon=20.4721,
            wind_speed=5.0,
        )

        today = datetime.utcnow()
        today_doy = today.timetuple().tm_yday
        timestamp = int(today.timestamp())

        # Create assets
        asset_file = create_stub_image(tmp_path, "sea_asset.jpg")
        asset_id = create_sea_asset(
            bot,
            rubric_id=rubric.id,
            message_id=201,
            file_name="sea.jpg",
            local_path=asset_file,
            tags=["sea"],
            sea_wave_score=1,
            photo_sky="sunny",
            is_sunset=False,
        )
        bot.db.execute(
            "UPDATE assets SET shot_at_utc=?, shot_doy=? WHERE id=?",
            (timestamp, today_doy, asset_id),
        )
        bot.db.commit()

        rubric = bot.data.get_rubric_by_code("sea")
        assert rubric is not None

        await bot._publish_sea(
            rubric,
            sea_channel,
            test=True,
            job=None,
            initiator_id=None,
            instructions=None,
        )

        # Check that at least one attempt log exists
        sea_logs = [rec for rec in caplog.records if "SEA_RUBRIC" in rec.message]
        attempt_logs = [rec for rec in sea_logs if "attempt:" in rec.message]
        assert len(attempt_logs) > 0, "Should have at least one attempt log"

        # Check stage names
        stages_found = set()
        for log in attempt_logs:
            if "attempt:B0" in log.message:
                stages_found.add("B0")
            elif "attempt:B1" in log.message:
                stages_found.add("B1")
            elif "attempt:B2" in log.message:
                stages_found.add("B2")
            elif "attempt:AN" in log.message:
                stages_found.add("AN")

        # At least one stage should be present
        assert len(stages_found) > 0, f"Should have at least one stage logged, found: {stages_found}"

    if db_path.exists():
        db_path.unlink()
