"""Integration tests for rubric purge command behaviour."""

import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import Bot


def _insert_rubric_job(
    bot: Bot,
    *,
    available_at: datetime | None,
    status: str,
    payload: dict[str, Any],
) -> int:
    """Insert a publish_rubric job with the provided payload."""

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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_purge_sea_jobs_keeps_canonical_by_default(tmp_path):
    """By default the purge keeps the canonical sea job and removes stale ones."""

    bot = Bot("dummy", str(tmp_path / "test_bot.db"))

    superadmin_id = 12345
    bot.db.execute(
        "INSERT INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, ?)",
        (superadmin_id, "superadmin", "+00:00"),
    )
    bot.db.commit()

    now = datetime.utcnow()
    canonical_time = now + timedelta(hours=3)
    stale_time = now - timedelta(hours=2)

    canonical_payload = {
        "rubric": "sea",
        "rubric_code": "sea",
        "schedule_key": "sea:primary",
        "scheduled_at": canonical_time.isoformat(),
    }
    stale_payload = {
        "rubric": "sea",
        "rubric_code": "sea",
        "schedule_key": "legacy",
        "scheduled_at": (stale_time + timedelta(minutes=15)).isoformat(),
    }

    canonical_id = _insert_rubric_job(
        bot,
        available_at=canonical_time,
        status="delayed",
        payload=canonical_payload,
    )
    stale_id = _insert_rubric_job(
        bot,
        available_at=stale_time,
        status="delayed",
        payload=stale_payload,
    )

    messages: list[str] = []

    async def mock_api_request(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "sendMessage":
            messages.append(params.get("text", ""))
        return {"ok": True, "result": {}}

    bot.api_request = mock_api_request

    message = type(
        "Message",
        (),
        {"text": "/purge_sea_jobs", "chat": type("Chat", (), {"id": superadmin_id})()},
    )()
    await bot.on_message(message)

    assert len(messages) == 2
    assert "Сохраняем 1 актуальную задачу" in messages[1]
    assert f"#{canonical_id}" in messages[1]
    assert "Удалено 1 задач(и)." in messages[1]
    remaining = bot.db.execute(
        "SELECT id FROM jobs_queue WHERE name='publish_rubric' ORDER BY id"
    ).fetchall()
    assert [row["id"] for row in remaining] == [canonical_id]
    removed = bot.db.execute(
        "SELECT COUNT(*) AS cnt FROM jobs_queue WHERE id=?", (stale_id,)
    ).fetchone()
    assert removed["cnt"] == 0

    await bot.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_purge_sea_jobs_keep_false_removes_all(tmp_path):
    """When keep=false is passed all sea jobs are removed."""

    bot = Bot("dummy", str(tmp_path / "test_bot.db"))

    superadmin_id = 12345
    bot.db.execute(
        "INSERT INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, ?)",
        (superadmin_id, "superadmin", "+00:00"),
    )
    bot.db.commit()

    now = datetime.utcnow()
    future_time = now + timedelta(hours=1)

    for offset in (0, 2):
        _insert_rubric_job(
            bot,
            available_at=future_time + timedelta(minutes=offset * 5),
            status="delayed",
            payload={
                "rubric": "sea",
                "rubric_code": "sea",
                "schedule_key": f"sea:{offset}",
                "scheduled_at": (future_time + timedelta(minutes=offset * 5)).isoformat(),
            },
        )

    messages: list[str] = []

    async def mock_api_request(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "sendMessage":
            messages.append(params.get("text", ""))
        return {"ok": True, "result": {}}

    bot.api_request = mock_api_request

    message = type(
        "Message",
        (),
        {
            "text": "/purge_sea_jobs keep=false",
            "chat": type("Chat", (), {"id": superadmin_id})(),
        },
    )()
    await bot.on_message(message)

    assert len(messages) == 2
    assert "Удалено 2 задач(и)." in messages[1]
    remaining = bot.db.execute(
        "SELECT COUNT(*) AS cnt FROM jobs_queue WHERE name='publish_rubric'"
    ).fetchone()
    assert remaining["cnt"] == 0

    await bot.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_purge_sea_jobs_no_jobs_found(tmp_path):
    """If no sea jobs exist the command reports success."""

    bot = Bot("dummy", str(tmp_path / "test_bot.db"))

    superadmin_id = 12345
    bot.db.execute(
        "INSERT INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, ?)",
        (superadmin_id, "superadmin", "+00:00"),
    )
    bot.db.commit()

    messages: list[str] = []

    async def mock_api_request(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "sendMessage":
            messages.append(params.get("text", ""))
        return {"ok": True, "result": {}}

    bot.api_request = mock_api_request

    message = type(
        "Message",
        (),
        {"text": "/purge_sea_jobs", "chat": type("Chat", (), {"id": superadmin_id})()},
    )()
    await bot.on_message(message)

    assert messages == ["🔍 Ищу задачи моря...", "✓ Задачи моря не найдены."]

    await bot.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_purge_sea_jobs_requires_superadmin(tmp_path):
    """Superadmin privileges are required to purge sea jobs."""

    bot = Bot("dummy", str(tmp_path / "test_bot.db"))

    regular_user_id = 67890
    bot.db.execute(
        "INSERT INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 0, ?)",
        (regular_user_id, "regular", "+00:00"),
    )
    bot.db.commit()

    messages: list[str] = []

    async def mock_api_request(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "sendMessage":
            messages.append(params.get("text", ""))
        return {"ok": True, "result": {}}

    bot.api_request = mock_api_request

    message = type(
        "Message",
        (),
        {"text": "/purge_sea_jobs", "chat": type("Chat", (), {"id": regular_user_id})()},
    )()
    await bot.on_message(message)

    assert messages == []

    await bot.close()
