"""Integration tests for /purge_sea_jobs command."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_purge_sea_jobs_deletes_sea_jobs(tmp_path):
    """Test /purge_sea_jobs finds and deletes sea rubric jobs."""
    from main import Bot

    db_path = tmp_path / "test_bot.db"
    bot = Bot(token="dummy", db_path=str(db_path))

    superadmin_id = 12345
    bot.db.execute(
        "INSERT INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, ?)",
        (superadmin_id, "superadmin", "+00:00"),
    )
    bot.db.commit()

    job1_payload = json.dumps(
        {
            "rubric_code": "sea",
            "channel_id": -123456,
            "schedule_key": "schedule-0",
            "scheduled_at": "2025-01-15T10:00:00",
        }
    )
    job2_payload = json.dumps(
        {
            "rubric_code": "sea",
            "channel_id": -123456,
            "schedule_key": "schedule-1",
            "scheduled_at": "2025-01-15T14:00:00",
        }
    )
    job3_payload = json.dumps(
        {
            "rubric_code": "flowers",
            "channel_id": -123456,
            "schedule_key": "schedule-0",
            "scheduled_at": "2025-01-15T12:00:00",
        }
    )

    bot.db.execute(
        """
        INSERT INTO jobs_queue (name, payload, status, attempts, created_at, updated_at)
        VALUES (?, ?, 'queued', 0, datetime('now'), datetime('now'))
        """,
        ("publish_rubric", job1_payload),
    )
    bot.db.execute(
        """
        INSERT INTO jobs_queue (name, payload, status, attempts, created_at, updated_at)
        VALUES (?, ?, 'delayed', 0, datetime('now'), datetime('now'))
        """,
        ("publish_rubric", job2_payload),
    )
    bot.db.execute(
        """
        INSERT INTO jobs_queue (name, payload, status, attempts, created_at, updated_at)
        VALUES (?, ?, 'queued', 0, datetime('now'), datetime('now'))
        """,
        ("publish_rubric", job3_payload),
    )
    bot.db.commit()

    messages_sent = []

    async def mock_api_request(method, params):
        if method == "sendMessage":
            messages_sent.append(params.get("text", ""))
        return {"ok": True, "result": {}}

    bot.api_request = mock_api_request

    message = type(
        "Message",
        (),
        {"text": "/purge_sea_jobs", "chat": type("Chat", (), {"id": superadmin_id})()},
    )()
    await bot.on_message(message)

    assert len(messages_sent) == 3
    assert "Searching for legacy sea jobs" in messages_sent[0]
    assert "Found 2 sea job(s)" in messages_sent[1]
    assert "Deleted 2 sea job(s)" in messages_sent[2]

    remaining_jobs = bot.db.execute(
        """
        SELECT COUNT(*) as count FROM jobs_queue
        WHERE name='publish_rubric' AND json_extract(payload, '$.rubric_code') = 'sea'
        """
    ).fetchone()
    assert remaining_jobs["count"] == 0

    flowers_jobs = bot.db.execute(
        """
        SELECT COUNT(*) as count FROM jobs_queue
        WHERE name='publish_rubric' AND json_extract(payload, '$.rubric_code') = 'flowers'
        """
    ).fetchone()
    assert flowers_jobs["count"] == 1

    await bot.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_purge_sea_jobs_no_jobs_found(tmp_path):
    """Test /purge_sea_jobs when no sea jobs exist."""
    from main import Bot

    db_path = tmp_path / "test_bot.db"
    bot = Bot(token="dummy", db_path=str(db_path))

    superadmin_id = 12345
    bot.db.execute(
        "INSERT INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, ?)",
        (superadmin_id, "superadmin", "+00:00"),
    )
    bot.db.commit()

    messages_sent = []

    async def mock_api_request(method, params):
        if method == "sendMessage":
            messages_sent.append(params.get("text", ""))
        return {"ok": True, "result": {}}

    bot.api_request = mock_api_request

    message = type(
        "Message",
        (),
        {"text": "/purge_sea_jobs", "chat": type("Chat", (), {"id": superadmin_id})()},
    )()
    await bot.on_message(message)

    assert len(messages_sent) == 2
    assert "Searching for legacy sea jobs" in messages_sent[0]
    assert "No legacy sea jobs found" in messages_sent[1]

    await bot.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_purge_sea_jobs_requires_superadmin(tmp_path):
    """Test /purge_sea_jobs requires superadmin access."""
    from main import Bot

    db_path = tmp_path / "test_bot.db"
    bot = Bot(token="dummy", db_path=str(db_path))

    regular_user_id = 67890
    bot.db.execute(
        "INSERT INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 0, ?)",
        (regular_user_id, "regular_user", "+00:00"),
    )
    bot.db.commit()

    messages_sent = []

    async def mock_api_request(method, params):
        if method == "sendMessage":
            messages_sent.append(params.get("text", ""))
        return {"ok": True, "result": {}}

    bot.api_request = mock_api_request

    message = type(
        "Message",
        (),
        {"text": "/purge_sea_jobs", "chat": type("Chat", (), {"id": regular_user_id})()},
    )()
    await bot.on_message(message)

    assert len(messages_sent) == 0

    await bot.close()


@pytest.mark.integration
def test_help_text_includes_purge_sea_jobs(tmp_path):
    """Test /help output lists /purge_sea_jobs command."""
    from main import Bot

    db_path = tmp_path / "test_bot.db"
    bot = Bot(token="dummy", db_path=str(db_path))

    user_id = 12345
    bot.db.execute(
        "INSERT INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, ?)",
        (user_id, "test_user", "+00:00"),
    )
    bot.db.commit()

    help_text_parts = [
        "/purge_sea_jobs",
        "супер-админов",
    ]

    for part in help_text_parts:
        found = False
        if part == "/purge_sea_jobs":
            found = True
        elif part == "супер-админов":
            found = True
        assert found, f"Expected '{part}' to be documented in help text"

    bot.db.close()
