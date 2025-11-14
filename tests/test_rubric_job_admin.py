"""Tests for rubric job classification and admin commands."""

import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_access import JobRecord
from main import Bot


def _insert_rubric_job(
    bot: Bot,
    *,
    available_at: datetime | None,
    status: str,
    payload: dict[str, Any],
) -> int:
    """Insert a publish_rubric job used by admin command tests."""

    now_iso = datetime.utcnow().isoformat()
    cursor = bot.db.execute(
        """
        INSERT INTO jobs_queue (name, payload, status, attempts, available_at, last_error, created_at, updated_at)
        VALUES (?, ?, ?, 0, ?, NULL, ?, ?)
        """,
        (
            "publish_rubric",
            json.dumps(payload),
            status,
            available_at.isoformat() if available_at else None,
            now_iso,
            now_iso,
        ),
    )
    bot.db.commit()
    return int(cursor.lastrowid)


@pytest.mark.asyncio
async def test_classify_rubric_job_variants(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    now = datetime.utcnow()

    direct = JobRecord(
        id=1,
        name="publish_rubric",
        payload={"rubric": "Sea"},
        status="queued",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=now,
        updated_at=now,
    )
    assert bot.classify_rubric_job(direct) == "sea"

    nested = JobRecord(
        id=2,
        name="publish_rubric",
        payload={"metadata": {"rubric_name": "Flowers"}},
        status="delayed",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=now,
        updated_at=now,
    )
    assert bot.classify_rubric_job(nested) == "flowers"

    flowers = bot.data.get_rubric_by_code("flowers")
    assert flowers is not None
    by_id = JobRecord(
        id=3,
        name="publish_rubric",
        payload={"rubric_id": flowers.id},
        status="delayed",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=now,
        updated_at=now,
    )
    assert bot.classify_rubric_job(by_id) == "flowers"

    schedule = JobRecord(
        id=4,
        name="publish_rubric",
        payload={"schedule_key": "sea:slot:2130"},
        status="delayed",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=now,
        updated_at=now,
    )
    assert bot.classify_rubric_job(schedule) == "sea"

    other = JobRecord(
        id=5,
        name="vision",
        payload={},
        status="queued",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=now,
        updated_at=now,
    )
    assert bot.classify_rubric_job(other) is None

    await bot.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rubric_jobs_command_filters_and_timezone(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    user_id = 111
    bot.db.execute(
        "INSERT INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 0, ?)",
        (user_id, "operator", "+00:00"),
    )
    bot.db.commit()

    now = datetime.utcnow()
    sea_time = now + timedelta(hours=1)
    flowers_time = now + timedelta(hours=4)
    stale_time = now - timedelta(hours=3)

    sea_id = _insert_rubric_job(
        bot,
        available_at=sea_time,
        status="delayed",
        payload={
            "rubric": "sea",
            "rubric_code": "sea",
            "schedule_key": "sea:morning",
            "scheduled_at": sea_time.isoformat(),
        },
    )
    _insert_rubric_job(
        bot,
        available_at=flowers_time,
        status="delayed",
        payload={
            "rubric": "flowers",
            "rubric_code": "flowers",
            "schedule_key": "flowers:evening",
            "scheduled_at": flowers_time.isoformat(),
        },
    )
    stale_id = _insert_rubric_job(
        bot,
        available_at=stale_time,
        status="delayed",
        payload={
            "rubric": "sea",
            "rubric_code": "sea",
            "schedule_key": "sea:legacy",
            "scheduled_at": (stale_time + timedelta(minutes=10)).isoformat(),
        },
    )

    messages: list[str] = []

    async def mock_api_request(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "sendMessage":
            messages.append(params.get("text", ""))
        return {"ok": True, "result": {}}

    bot.api_request = mock_api_request

    sea_message = type(
        "Message",
        (),
        {
            "text": "/rubric_jobs rubric=sea window=48h",
            "chat": type("Chat", (), {"id": user_id})(),
        },
    )()
    await bot.on_message(sea_message)

    assert len(messages) == 1
    sea_report = messages.pop()
    assert "rubric=sea" in sea_report
    assert f"#{sea_id}" in sea_report
    assert f"#{stale_id}" in sea_report
    assert "(overdue)" in sea_report
    assert "flowers" not in sea_report
    assert "+02:00" in sea_report  # Europe/Kaliningrad timezone

    all_message = type(
        "Message",
        (),
        {"text": "/rubric_jobs", "chat": type("Chat", (), {"id": user_id})()},
    )()
    await bot.on_message(all_message)

    assert len(messages) == 1
    all_report = messages.pop()
    assert "rubric=all" in all_report
    assert "flowers" in all_report
    assert "sea" in all_report

    await bot.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cancel_rubric_jobs_dry_run_and_confirm(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    superadmin_id = 222
    bot.db.execute(
        "INSERT INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, ?)",
        (superadmin_id, "boss", "+00:00"),
    )
    bot.db.commit()

    now = datetime.utcnow()
    for offset in range(2):
        _insert_rubric_job(
            bot,
            available_at=now + timedelta(hours=offset + 1),
            status="delayed",
            payload={
                "rubric": "sea",
                "rubric_code": "sea",
                "schedule_key": f"sea:auto:{offset}",
                "scheduled_at": (now + timedelta(hours=offset + 1)).isoformat(),
            },
        )

    messages: list[str] = []

    async def mock_api_request(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "sendMessage":
            messages.append(params.get("text", ""))
        return {"ok": True, "result": {}}

    bot.api_request = mock_api_request

    dry_message = type(
        "Message",
        (),
        {
            "text": "/cancel_rubric_jobs rubric=sea",
            "chat": type("Chat", (), {"id": superadmin_id})(),
        },
    )()
    await bot.on_message(dry_message)

    assert len(messages) == 1
    dry_report = messages.pop()
    assert "Режим предварительного просмотра" in dry_report
    count_after_dry = bot.db.execute(
        "SELECT COUNT(*) AS cnt FROM jobs_queue WHERE name='publish_rubric'"
    ).fetchone()
    assert count_after_dry["cnt"] == 2

    confirm_message = type(
        "Message",
        (),
        {
            "text": "/cancel_rubric_jobs rubric=sea confirm=true",
            "chat": type("Chat", (), {"id": superadmin_id})(),
        },
    )()
    await bot.on_message(confirm_message)

    assert len(messages) == 1
    confirm_report = messages.pop()
    assert "Удалено 2 задач(и)." in confirm_report
    remaining = bot.db.execute(
        "SELECT COUNT(*) AS cnt FROM jobs_queue WHERE name='publish_rubric'"
    ).fetchone()
    assert remaining["cnt"] == 0

    await bot.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_help_mentions_rubric_job_commands(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    superadmin_id = 999
    bot.db.execute(
        "INSERT INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, ?)",
        (superadmin_id, "admin", "+00:00"),
    )
    bot.db.commit()

    messages: list[str] = []

    async def mock_api_request(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "sendMessage":
            messages.append(params.get("text", ""))
        return {"ok": True, "result": {}}

    bot.api_request = mock_api_request

    help_message = type(
        "Message",
        (),
        {"text": "/help", "chat": type("Chat", (), {"id": superadmin_id})()},
    )()
    await bot.on_message(help_message)

    joined = "\n".join(messages)
    assert "/rubric_jobs" in joined
    assert "/cancel_rubric_jobs" in joined
    assert "/purge_sea_jobs" in joined

    await bot.close()
