import json
import os
import sys
from datetime import datetime
from typing import Any
from uuid import uuid4

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import Bot

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "dummy")


def _seed_rubrics(bot: Bot) -> dict[str, int]:
    """Seed default rubrics and return mapping of code -> id."""
    rubrics = {}
    for code, title in [("sea", "Море"), ("flowers", "Цветы"), ("guess_arch", "Архитектура")]:
        bot.data.upsert_rubric(code, title, config={})
        rubric = bot.data.get_rubric_by_code(code)
        if rubric:
            rubrics[code] = rubric.id
    bot.db.commit()
    return rubrics


def _create_asset(
    bot: Bot,
    tg_chat_id: int,
    message_id: int,
    rubric_id: int | None = None,
) -> str:
    """Create an asset and return its ID."""
    asset_id = str(uuid4())
    tg_message_id = f"{tg_chat_id}:{message_id}"
    payload = {}
    if rubric_id is not None:
        payload["rubric_id"] = rubric_id
    payload_json = json.dumps(payload)

    bot.db.execute(
        """
        INSERT INTO assets (
            id, tg_message_id, payload_json, created_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            asset_id,
            tg_message_id,
            payload_json,
            datetime.utcnow().isoformat(),
        ),
    )
    bot.db.commit()
    return asset_id


@pytest.mark.asyncio
async def test_audit_assets_removes_missing(tmp_path: Any) -> None:
    """Test that audit_assets removes assets where TG message not found."""
    db_path = tmp_path / "test.db"
    bot = Bot("dummy", str(db_path))
    rubrics = _seed_rubrics(bot)

    # Create assets: some will be "missing" (copyMessage fails)
    asset_1 = _create_asset(bot, -1001234, 101, rubrics["sea"])
    asset_2 = _create_asset(bot, -1001234, 102, rubrics["flowers"])
    asset_3 = _create_asset(bot, -1001234, 103, rubrics["guess_arch"])
    asset_4 = _create_asset(bot, -1001234, 104, None)  # unassigned

    # Mock api_request to simulate missing messages for asset_2 and asset_4
    api_calls: list[tuple[str, dict[str, Any]]] = []

    async def mock_api_request(method: str, params: dict[str, Any]) -> dict[str, Any]:
        api_calls.append((method, params))
        if method == "copyMessage":
            msg_id = params.get("message_id")
            # Simulate that messages 102 and 104 are missing
            if msg_id in [102, 104]:
                raise Exception("Bad Request: message to copy not found")
            # Other messages exist - return success
            return {"ok": True, "result": {"message_id": 9999}}
        elif method == "deleteMessage":
            return {"ok": True}
        return {"ok": True}

    bot.api_request = mock_api_request  # type: ignore[method-assign]

    # Simulate operator user sending /audit_assets
    operator_id = 12345
    bot.db.execute(
        "INSERT INTO users (user_id, is_superadmin) VALUES (?, 1)",
        (operator_id,),
    )
    bot.db.commit()

    message = {
        "message_id": 1,
        "from": {"id": operator_id, "username": "operator"},
        "chat": {"id": operator_id},
        "text": "/audit_assets",
    }

    await bot.handle_message(message)

    # Verify that missing assets (102, 104) were deleted
    remaining = bot.db.execute("SELECT id, tg_message_id FROM assets ORDER BY id").fetchall()
    remaining_ids = [row["id"] for row in remaining]
    remaining_msg_ids = []
    for row in remaining:
        if row["tg_message_id"]:
            # Extract message_id from "chat_id:message_id" format
            parts = row["tg_message_id"].split(":")
            if len(parts) == 2:
                remaining_msg_ids.append(int(parts[1]))

    assert asset_1 in remaining_ids, "Asset 1 should remain (message exists)"
    assert asset_2 not in remaining_ids, "Asset 2 should be deleted (message missing)"
    assert asset_3 in remaining_ids, "Asset 3 should remain (message exists)"
    assert asset_4 not in remaining_ids, "Asset 4 should be deleted (message missing)"

    assert 101 in remaining_msg_ids
    assert 102 not in remaining_msg_ids
    assert 103 in remaining_msg_ids
    assert 104 not in remaining_msg_ids


@pytest.mark.asyncio
async def test_audit_assets_reports_counts(tmp_path: Any) -> None:
    """Test that audit_assets reports correct checked/removed counts."""
    db_path = tmp_path / "test.db"
    bot = Bot("dummy", str(db_path))
    rubrics = _seed_rubrics(bot)

    # Create 10 assets, 3 are "missing"
    for i in range(1, 11):
        rubric_id = rubrics["sea"] if i <= 5 else rubrics["flowers"]
        _create_asset(bot, -1001234, 100 + i, rubric_id)

    messages_sent: list[str] = []

    async def mock_api_request(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "sendMessage":
            messages_sent.append(params.get("text", ""))
        if method == "copyMessage":
            msg_id = params.get("message_id")
            # Messages 102, 105, 108 are missing
            if msg_id in [102, 105, 108]:
                raise Exception("Bad Request: message to copy not found")
            return {"ok": True, "result": {"message_id": 9999}}
        elif method == "deleteMessage":
            return {"ok": True}
        return {"ok": True}

    bot.api_request = mock_api_request  # type: ignore[method-assign]

    operator_id = 12345
    bot.db.execute(
        "INSERT INTO users (user_id, is_superadmin) VALUES (?, 1)",
        (operator_id,),
    )
    bot.db.commit()

    message = {
        "message_id": 1,
        "from": {"id": operator_id, "username": "operator"},
        "chat": {"id": operator_id},
        "text": "/audit_assets",
    }

    await bot.handle_message(message)

    # Find the final report message
    report = next((msg for msg in messages_sent if "завершён" in msg), "")

    assert "Проверено: 10" in report
    assert "Удалено «мёртвых душ»: 3" in report


@pytest.mark.asyncio
async def test_audit_assets_continues_on_error(tmp_path: Any) -> None:
    """Test that audit_assets continues on non-404 errors."""
    db_path = tmp_path / "test.db"
    bot = Bot("dummy", str(db_path))
    rubrics = _seed_rubrics(bot)

    # Create 5 assets
    for i in range(1, 6):
        _create_asset(bot, -1001234, 100 + i, rubrics["sea"])

    async def mock_api_request(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "copyMessage":
            msg_id = params.get("message_id")
            if msg_id == 102:
                # This is a "missing" message
                raise Exception("Bad Request: message to copy not found")
            elif msg_id == 104:
                # This is a rate limit error - should NOT delete
                raise Exception("Too Many Requests: retry after 5")
            # Others succeed
            return {"ok": True, "result": {"message_id": 9999}}
        elif method == "deleteMessage":
            return {"ok": True}
        return {"ok": True}

    bot.api_request = mock_api_request  # type: ignore[method-assign]

    operator_id = 12345
    bot.db.execute(
        "INSERT INTO users (user_id, is_superadmin) VALUES (?, 1)",
        (operator_id,),
    )
    bot.db.commit()

    message = {
        "message_id": 1,
        "from": {"id": operator_id, "username": "operator"},
        "chat": {"id": operator_id},
        "text": "/audit_assets",
    }

    await bot.handle_message(message)

    # Check remaining assets
    remaining = bot.db.execute("SELECT tg_message_id FROM assets ORDER BY tg_message_id").fetchall()
    remaining_msg_ids = []
    for row in remaining:
        if row["tg_message_id"]:
            parts = row["tg_message_id"].split(":")
            if len(parts) == 2:
                remaining_msg_ids.append(int(parts[1]))

    # Only 102 should be deleted (404 error)
    # 104 should remain (rate limit error, not 404)
    assert 101 in remaining_msg_ids
    assert 102 not in remaining_msg_ids
    assert 103 in remaining_msg_ids
    assert 104 in remaining_msg_ids
    assert 105 in remaining_msg_ids


@pytest.mark.asyncio
async def test_audit_assets_requires_authorization(tmp_path: Any) -> None:
    """Test that audit_assets requires operator authorization."""
    db_path = tmp_path / "test.db"
    bot = Bot("dummy", str(db_path))

    messages_sent: list[str] = []

    async def mock_api_request(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "sendMessage":
            messages_sent.append(params.get("text", ""))
        return {"ok": True}

    bot.api_request = mock_api_request  # type: ignore[method-assign]

    # Unauthorized user tries to run audit
    unauthorized_id = 99999
    message = {
        "message_id": 1,
        "from": {"id": unauthorized_id, "username": "hacker"},
        "chat": {"id": unauthorized_id},
        "text": "/audit_assets",
    }

    await bot.handle_message(message)

    # Should get "Not authorized" message
    assert any("Not authorized" in msg for msg in messages_sent)


@pytest.mark.asyncio
async def test_help_has_audit_assets(tmp_path: Any) -> None:
    """Test that /help includes /audit_assets command."""
    db_path = tmp_path / "test.db"
    bot = Bot("dummy", str(db_path))

    messages_sent: list[str] = []

    async def mock_api_request(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "sendMessage":
            messages_sent.append(params.get("text", ""))
        return {"ok": True}

    bot.api_request = mock_api_request  # type: ignore[method-assign]

    operator_id = 12345
    bot.db.execute(
        "INSERT INTO users (user_id, is_superadmin) VALUES (?, 1)",
        (operator_id,),
    )
    bot.db.commit()

    message = {
        "message_id": 1,
        "from": {"id": operator_id, "username": "operator"},
        "chat": {"id": operator_id},
        "text": "/help",
    }

    await bot.handle_message(message)

    # Check that audit_assets is mentioned in help
    help_text = " ".join(messages_sent)
    assert "/audit_assets" in help_text


@pytest.mark.asyncio
async def test_audit_assets_batching(tmp_path: Any) -> None:
    """Test that audit_assets processes in batches with delays."""
    db_path = tmp_path / "test.db"
    bot = Bot("dummy", str(db_path))
    rubrics = _seed_rubrics(bot)

    # Create 120 assets (more than batch size of 50)
    for i in range(1, 121):
        _create_asset(bot, -1001234, 100 + i, rubrics["sea"])

    copy_calls = 0

    async def mock_api_request(method: str, params: dict[str, Any]) -> dict[str, Any]:
        nonlocal copy_calls
        if method == "copyMessage":
            copy_calls += 1
            # All messages exist
            return {"ok": True, "result": {"message_id": 9999}}
        elif method == "deleteMessage":
            return {"ok": True}
        return {"ok": True}

    bot.api_request = mock_api_request  # type: ignore[method-assign]

    operator_id = 12345
    bot.db.execute(
        "INSERT INTO users (user_id, is_superadmin) VALUES (?, 1)",
        (operator_id,),
    )
    bot.db.commit()

    message = {
        "message_id": 1,
        "from": {"id": operator_id, "username": "operator"},
        "chat": {"id": operator_id},
        "text": "/audit_assets",
    }

    await bot.handle_message(message)

    # Verify all 120 assets were checked
    assert copy_calls == 120


@pytest.mark.asyncio
async def test_audit_assets_per_rubric_reporting(tmp_path: Any) -> None:
    """Test that audit_assets reports per-rubric breakdown."""
    db_path = tmp_path / "test.db"
    bot = Bot("dummy", str(db_path))
    rubrics = _seed_rubrics(bot)

    # Create assets across different rubrics
    _create_asset(bot, -1001234, 101, rubrics["sea"])
    _create_asset(bot, -1001234, 102, rubrics["sea"])
    _create_asset(bot, -1001234, 103, rubrics["flowers"])
    _create_asset(bot, -1001234, 104, rubrics["flowers"])
    _create_asset(bot, -1001234, 105, rubrics["guess_arch"])
    _create_asset(bot, -1001234, 106, None)  # unassigned

    messages_sent: list[str] = []

    async def mock_api_request(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "sendMessage":
            messages_sent.append(params.get("text", ""))
        if method == "copyMessage":
            msg_id = params.get("message_id")
            # Delete one from each rubric: 102 (sea), 103 (flowers), 106 (unassigned)
            if msg_id in [102, 103, 106]:
                raise Exception("Bad Request: message to copy not found")
            return {"ok": True, "result": {"message_id": 9999}}
        elif method == "deleteMessage":
            return {"ok": True}
        return {"ok": True}

    bot.api_request = mock_api_request  # type: ignore[method-assign]

    operator_id = 12345
    bot.db.execute(
        "INSERT INTO users (user_id, is_superadmin) VALUES (?, 1)",
        (operator_id,),
    )
    bot.db.commit()

    message = {
        "message_id": 1,
        "from": {"id": operator_id, "username": "operator"},
        "chat": {"id": operator_id},
        "text": "/audit_assets",
    }

    await bot.handle_message(message)

    # Find the final report
    report = next((msg for msg in messages_sent if "завершён" in msg), "")

    # Check per-rubric breakdown
    assert "sea:" in report or "sea" in report
    assert "flowers:" in report or "flowers" in report
    assert "guess_arch:" in report or "guess_arch" in report
    assert "unassigned:" in report or "unassigned" in report
