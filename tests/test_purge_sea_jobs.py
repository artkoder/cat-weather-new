"""Tests for the /purge_sea_jobs admin command."""

import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import Bot

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "dummy")


def _insert_rubric_job(
    bot: Bot,
    *,
    status: str,
    schedule_key: str | None,
    rubric_code: str = "sea",
    available_at: datetime | None = None,
) -> None:
    now_iso = datetime.utcnow().isoformat()
    payload: dict[str, Any] = {
        "rubric": rubric_code,
        "rubric_code": rubric_code,
    }
    if schedule_key is not None:
        payload["schedule_key"] = schedule_key
    bot.db.execute(
        """
        INSERT INTO jobs_queue (
            name,
            payload,
            status,
            attempts,
            available_at,
            last_error,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, 0, ?, NULL, ?, ?)
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


@pytest.mark.asyncio
async def test_purge_sea_jobs_deletes_future_jobs(tmp_path):
    bot = Bot("dummy", str(tmp_path / "test_bot.db"))
    try:
        superadmin_id = 12345
        bot.db.execute(
            "INSERT INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, ?)",
            (superadmin_id, "superadmin", "+00:00"),
        )

        now = datetime.utcnow()
        _insert_rubric_job(bot, status="queued", schedule_key="sea:morning", available_at=None)
        _insert_rubric_job(
            bot,
            status="delayed",
            schedule_key="sea:evening",
            available_at=now + timedelta(hours=1),
        )
        _insert_rubric_job(bot, status="queued", schedule_key="manual")
        _insert_rubric_job(bot, status="delayed", schedule_key="manual-test")
        _insert_rubric_job(
            bot,
            status="queued",
            schedule_key="flowers:slot",
            rubric_code="flowers",
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

        assert messages == ["SEA: удалено плановых задач publish_rubric: 2"]

        remaining = bot.db.execute(
            """
            SELECT json_extract(payload, '$.schedule_key') AS schedule_key,
                   json_extract(payload, '$.rubric_code') AS rubric_code,
                   status
            FROM jobs_queue
            ORDER BY id
            """
        ).fetchall()
        schedule_keys = [row["schedule_key"] for row in remaining]
        assert schedule_keys.count("manual") == 1
        assert schedule_keys.count("manual-test") == 1
        assert any(row["rubric_code"] == "flowers" for row in remaining)
        assert all(
            (row["rubric_code"] != "sea")
            or (row["schedule_key"] in {"manual", "manual-test"})
            for row in remaining
        )
    finally:
        await bot.close()


@pytest.mark.asyncio
async def test_purge_sea_jobs_requires_superadmin(tmp_path):
    bot = Bot("dummy", str(tmp_path / "test_bot.db"))
    try:
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
    finally:
        await bot.close()
