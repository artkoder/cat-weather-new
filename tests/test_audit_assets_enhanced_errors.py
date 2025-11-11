"""Test enhanced error pattern detection for audit_assets."""

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
    for code, title in [("sea", "Море"), ("flowers", "Цветы")]:
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
async def test_audit_assets_various_error_patterns(tmp_path: Any) -> None:
    """Test that audit_assets detects various Telegram error messages."""
    db_path = tmp_path / "test.db"
    bot = Bot("dummy", str(db_path))
    rubrics = _seed_rubrics(bot)

    # Create assets with different error patterns
    asset_1 = _create_asset(bot, -1001234, 101, rubrics["sea"])  # message_id_invalid
    asset_2 = _create_asset(bot, -1001234, 102, rubrics["flowers"])  # message to get not found
    asset_3 = _create_asset(bot, -1001234, 103, rubrics["sea"])  # chat not found
    asset_4 = _create_asset(bot, -1001234, 104, rubrics["flowers"])  # exists

    # Mock api_request to return different error types
    async def mock_api_request(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "copyMessage":
            msg_id = params.get("message_id")
            if msg_id == 101:
                raise Exception("Bad Request: MESSAGE_ID_INVALID")
            elif msg_id == 102:
                raise Exception("Bad Request: message to get not found")
            elif msg_id == 103:
                raise Exception("Bad Request: chat not found")
            elif msg_id == 104:
                # This message exists
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

    # Verify that assets with various error patterns were deleted
    remaining = bot.db.execute("SELECT id FROM assets ORDER BY id").fetchall()
    remaining_ids = [row["id"] for row in remaining]

    assert asset_1 not in remaining_ids, "Asset 1 should be deleted (MESSAGE_ID_INVALID)"
    assert asset_2 not in remaining_ids, "Asset 2 should be deleted (message to get not found)"
    assert asset_3 not in remaining_ids, "Asset 3 should be deleted (chat not found)"
    assert asset_4 in remaining_ids, "Asset 4 should remain (message exists)"


@pytest.mark.asyncio
async def test_audit_assets_logging(tmp_path: Any) -> None:
    """Test that audit_assets logs detailed information during checks."""
    db_path = tmp_path / "test.db"
    bot = Bot("dummy", str(db_path))
    rubrics = _seed_rubrics(bot)

    # Create one existing and one missing asset
    _create_asset(bot, -1001234, 101, rubrics["sea"])
    _create_asset(bot, -1001234, 102, rubrics["flowers"])

    log_messages: list[str] = []

    # Capture log messages
    import logging

    class LogCapture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            log_messages.append(record.getMessage())

    handler = LogCapture()
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)

    async def mock_api_request(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "copyMessage":
            msg_id = params.get("message_id")
            if msg_id == 102:
                raise Exception("Bad Request: message not found")
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

    # Check that we have the expected log messages
    checking_logs = [msg for msg in log_messages if "ASSETS_AUDIT_CHECKING" in msg]
    exists_logs = [msg for msg in log_messages if "ASSETS_AUDIT_EXISTS" in msg]
    failed_logs = [msg for msg in log_messages if "ASSETS_AUDIT_COPY_FAILED" in msg]
    dead_soul_logs = [msg for msg in log_messages if "ASSETS_AUDIT_DEAD_SOUL" in msg]
    deleted_logs = [msg for msg in log_messages if "ASSETS_AUDIT_DELETED" in msg]

    assert len(checking_logs) == 2, "Should log checking for both assets"
    assert len(exists_logs) == 1, "Should log exists for one asset"
    assert len(failed_logs) == 1, "Should log failed copy for one asset"
    assert len(dead_soul_logs) == 1, "Should log dead soul for one asset"
    assert len(deleted_logs) == 1, "Should log deletion for one asset"

    # Clean up
    logging.getLogger().removeHandler(handler)
