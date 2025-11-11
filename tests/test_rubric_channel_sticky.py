"""Tests for rubric channel stickiness fix."""

import json
import os
import sys
from datetime import datetime, timedelta

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import Bot

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "dummy")


def _insert_rubric(bot: Bot, code: str, config: dict, rubric_id: int = 1) -> None:
    bot.data.upsert_rubric(code, code.title(), config=config)


@pytest.mark.asyncio
async def test_channel_change_deletes_future_jobs(tmp_path):
    """Test that changing a rubric's channel deletes future queued jobs."""
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "channel_id": -100,
        "test_channel_id": -200,
        "schedules": [{"time": "10:00", "enabled": True}],
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    await bot.process_rubric_schedule(reference=datetime.utcnow())
    jobs_before = bot.db.execute(
        "SELECT COUNT(*) as cnt FROM jobs_queue WHERE name='publish_rubric' AND status='delayed'"
    ).fetchone()
    assert jobs_before["cnt"] > 0
    rubric = bot.data.get_rubric_by_code("flowers")
    assert rubric is not None
    new_config = rubric.config.copy()
    new_config["channel_id"] = -300
    bot.data.save_rubric_config("flowers", new_config)
    deleted = bot._delete_future_rubric_jobs("flowers", "prod_channel_changed")
    jobs_after = bot.db.execute(
        "SELECT COUNT(*) as cnt FROM jobs_queue WHERE name='publish_rubric' AND status='delayed'"
    ).fetchone()
    assert jobs_after["cnt"] == 0
    assert deleted > 0
    await bot.close()


@pytest.mark.asyncio
async def test_channel_resolved_at_execution_time(tmp_path):
    """Test that channel is resolved at job execution time, not enqueue time."""
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "channel_id": -100,
        "test_channel_id": -200,
        "schedules": [{"time": "10:00", "enabled": True}],
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    await bot.process_rubric_schedule(reference=datetime.utcnow())
    job_row = bot.db.execute(
        "SELECT id, payload FROM jobs_queue WHERE name='publish_rubric' LIMIT 1"
    ).fetchone()
    assert job_row is not None
    payload = json.loads(job_row["payload"])
    assert "slot_channel_id" not in payload or payload.get("slot_channel_id") is None
    rubric = bot.data.get_rubric_by_code("flowers")
    assert rubric is not None
    new_config = rubric.config.copy()
    new_config["channel_id"] = -300
    bot.data.save_rubric_config("flowers", new_config)
    await bot.close()


@pytest.mark.asyncio
async def test_manual_test_jobs_unaffected_by_cleanup(tmp_path):
    """Test that manual test jobs are not deleted when channel changes."""
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "channel_id": -100,
        "test_channel_id": -200,
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    bot.enqueue_rubric("flowers", test=True, channel_id=-200)
    manual_jobs_before = bot.db.execute(
        "SELECT COUNT(*) as cnt FROM jobs_queue WHERE name='publish_rubric' AND json_extract(payload, '$.schedule_key') LIKE 'manual%'"
    ).fetchone()
    assert manual_jobs_before["cnt"] > 0
    rubric = bot.data.get_rubric_by_code("flowers")
    assert rubric is not None
    new_config = rubric.config.copy()
    new_config["channel_id"] = -300
    bot.data.save_rubric_config("flowers", new_config)
    deleted = bot._delete_future_rubric_jobs("flowers", "prod_channel_changed")
    manual_jobs_after = bot.db.execute(
        "SELECT COUNT(*) as cnt FROM jobs_queue WHERE name='publish_rubric' AND json_extract(payload, '$.schedule_key') LIKE 'manual%'"
    ).fetchone()
    assert manual_jobs_after["cnt"] == manual_jobs_before["cnt"]
    assert deleted == 0
    await bot.close()


@pytest.mark.asyncio
async def test_slot_channel_override_preserved(tmp_path):
    """Test that slot-specific channel overrides are preserved in payload."""
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "channel_id": -100,
        "schedules": [{"time": "10:00", "enabled": True, "channel_id": -500}],
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    await bot.process_rubric_schedule(reference=datetime.utcnow())
    job_row = bot.db.execute(
        "SELECT payload FROM jobs_queue WHERE name='publish_rubric' LIMIT 1"
    ).fetchone()
    assert job_row is not None
    payload = json.loads(job_row["payload"])
    assert payload.get("slot_channel_id") == -500
    await bot.close()


@pytest.mark.asyncio
async def test_disable_rubric_deletes_future_jobs(tmp_path):
    """Test that disabling a rubric deletes all future jobs."""
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "channel_id": -100,
        "schedules": [{"time": "10:00", "enabled": True}],
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    await bot.process_rubric_schedule(reference=datetime.utcnow())
    jobs_before = bot.db.execute(
        "SELECT COUNT(*) as cnt FROM jobs_queue WHERE name='publish_rubric'"
    ).fetchone()
    assert jobs_before["cnt"] > 0
    rubric = bot.data.get_rubric_by_code("flowers")
    assert rubric is not None
    new_config = rubric.config.copy()
    new_config["enabled"] = False
    bot.data.save_rubric_config("flowers", new_config)
    deleted = bot._delete_future_rubric_jobs("flowers", "rubric_disabled")
    jobs_after = bot.db.execute(
        "SELECT COUNT(*) as cnt FROM jobs_queue WHERE name='publish_rubric'"
    ).fetchone()
    assert jobs_after["cnt"] == 0
    assert deleted > 0
    await bot.close()


@pytest.mark.asyncio
async def test_scheduler_no_channel_in_normal_payload(tmp_path):
    """Test that scheduler doesn't include channel_id in payload (unless slot override)."""
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "channel_id": -100,
        "schedules": [{"time": "10:00", "enabled": True}],
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    await bot.process_rubric_schedule(reference=datetime.utcnow())
    job_row = bot.db.execute(
        "SELECT payload FROM jobs_queue WHERE name='publish_rubric' LIMIT 1"
    ).fetchone()
    assert job_row is not None
    payload = json.loads(job_row["payload"])
    assert payload.get("rubric_code") == "flowers"
    assert payload.get("schedule_key") is not None
    assert "channel_id" not in payload
    assert "slot_channel_id" not in payload
    assert payload.get("slot_index") == 0
    await bot.close()


@pytest.mark.asyncio
async def test_schedule_key_without_channel(tmp_path):
    """Test that schedule key doesn't include channel for normal schedules."""
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "channel_id": -100,
        "schedules": [{"time": "10:00", "enabled": True}],
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    await bot.process_rubric_schedule(reference=datetime.utcnow())
    job_row = bot.db.execute(
        "SELECT payload FROM jobs_queue WHERE name='publish_rubric' LIMIT 1"
    ).fetchone()
    assert job_row is not None
    payload = json.loads(job_row["payload"])
    schedule_key = payload.get("schedule_key")
    assert schedule_key is not None
    assert schedule_key.startswith("flowers:")
    assert "-100" not in schedule_key
    await bot.close()
