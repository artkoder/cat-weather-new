import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import main as main_module
from tests.fixtures.sea import create_sea_asset


@pytest.mark.asyncio
async def test_sea_inventory_report_counts_correctly(monkeypatch, tmp_path):
    """Test that _send_sea_inventory_report correctly counts sea assets using payload_json."""
    db_path = tmp_path / "test_inventory.db"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    monkeypatch.setenv("WEBHOOK_URL", "https://example.com")
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

    messages_sent: list[dict[str, Any]] = []

    async def capture_api_request(
        self, method: str, data: Any = None, *, files: Any = None
    ) -> dict[str, Any]:
        if method == "sendMessage":
            messages_sent.append(data)
        return {"ok": True, "result": {}}

    monkeypatch.setattr(main_module.Bot, "api_request", capture_api_request, raising=False)

    bot = main_module.Bot(token="dummy", webhook_url="https://example.com")

    # Create a sea rubric
    bot.data.db.execute(
        """
        INSERT INTO rubrics (code, name, channel_id, config_json) 
        VALUES (?, ?, ?, ?)
        """,
        ("sea", "Море", -1001234567890, "{}"),
    )
    bot.data.db.commit()
    rubric_id = bot.data.db.execute("SELECT id FROM rubrics WHERE code='sea'").fetchone()["id"]

    # Create test sea assets with different sky buckets and wave scores
    assets_data = [
        {"sky": "clear", "wave": 0},
        {"sky": "clear", "wave": 1},
        {"sky": "mostly_clear", "wave": 2},
        {"sky": "partly_cloudy", "wave": 3},
        {"sky": "partly_cloudy", "wave": 3},
        {"sky": "mostly_cloudy", "wave": 5},
        {"sky": "overcast", "wave": 7},
    ]

    for idx, asset_data in enumerate(assets_data, 1):
        bot.data.db.execute(
            """
            INSERT INTO assets (
                id, rubric_id, file_id, file_unique_id, asset_type, 
                payload_json, vision_sky_bucket, wave_score_0_10
            ) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"asset_{idx}",
                rubric_id,
                f"file_{idx}",
                f"unique_{idx}",
                "photo",
                json.dumps({"vision_category": "sea"}),
                asset_data["sky"],
                asset_data["wave"],
            ),
        )

    bot.data.db.commit()

    # Call the inventory report function
    await bot._send_sea_inventory_report(is_prod=True, initiator_id=12345)

    # Verify a message was sent
    assert len(messages_sent) == 1
    message = messages_sent[0]
    assert message["chat_id"] == 12345

    # Verify the report contains correct data
    report_text = message["text"]
    assert "🗂 Остатки фото «Море»: 7" in report_text
    assert "Солнечно: 2" in report_text
    assert "Преимущественно ясно: 1" in report_text
    assert "Переменная облачность: 2" in report_text
    assert "Преимущественно облачно: 1" in report_text
    assert "Пасмурно: 1" in report_text

    # Verify wave scores
    assert "0/10 (штиль): 1" in report_text
    assert "1/10: 1" in report_text
    assert "2/10: 1" in report_text
    assert "3/10: 2" in report_text
    assert "5/10: 1" in report_text
    assert "7/10: 1" in report_text


@pytest.mark.asyncio
async def test_sea_inventory_report_zero_when_no_assets(monkeypatch, tmp_path):
    """Test that inventory report shows zeros when no sea assets exist."""
    db_path = tmp_path / "test_empty_inventory.db"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    monkeypatch.setenv("WEBHOOK_URL", "https://example.com")
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

    messages_sent: list[dict[str, Any]] = []

    async def capture_api_request(
        self, method: str, data: Any = None, *, files: Any = None
    ) -> dict[str, Any]:
        if method == "sendMessage":
            messages_sent.append(data)
        return {"ok": True, "result": {}}

    monkeypatch.setattr(main_module.Bot, "api_request", capture_api_request, raising=False)

    bot = main_module.Bot(token="dummy", webhook_url="https://example.com")

    # Call the inventory report function without any assets
    await bot._send_sea_inventory_report(is_prod=True, initiator_id=12345)

    # Verify a message was sent
    assert len(messages_sent) == 1
    message = messages_sent[0]

    # Verify the report shows zero total
    report_text = message["text"]
    assert "🗂 Остатки фото «Море»: 0" in report_text
    assert "Солнечно: 0" in report_text


@pytest.mark.asyncio
async def test_sea_inventory_report_ignores_non_sea_assets(monkeypatch, tmp_path):
    """Test that inventory report only counts sea assets, not other categories."""
    db_path = tmp_path / "test_filter_inventory.db"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    monkeypatch.setenv("WEBHOOK_URL", "https://example.com")
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

    messages_sent: list[dict[str, Any]] = []

    async def capture_api_request(
        self, method: str, data: Any = None, *, files: Any = None
    ) -> dict[str, Any]:
        if method == "sendMessage":
            messages_sent.append(data)
        return {"ok": True, "result": {}}

    monkeypatch.setattr(main_module.Bot, "api_request", capture_api_request, raising=False)

    bot = main_module.Bot(token="dummy", webhook_url="https://example.com")

    # Create a rubric
    bot.data.db.execute(
        """
        INSERT INTO rubrics (code, name, channel_id, config_json) 
        VALUES (?, ?, ?, ?)
        """,
        ("flowers", "Цветы", -1001234567890, "{}"),
    )
    bot.data.db.commit()
    rubric_id = bot.data.db.execute("SELECT id FROM rubrics WHERE code='flowers'").fetchone()[
        "id"
    ]

    # Create sea asset
    bot.data.db.execute(
        """
        INSERT INTO assets (
            id, rubric_id, file_id, file_unique_id, asset_type, 
            payload_json, vision_sky_bucket, wave_score_0_10
        ) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "sea_asset_1",
            rubric_id,
            "file_1",
            "unique_1",
            "photo",
            json.dumps({"vision_category": "sea"}),
            "clear",
            0,
        ),
    )

    # Create flowers asset (should be ignored)
    bot.data.db.execute(
        """
        INSERT INTO assets (
            id, rubric_id, file_id, file_unique_id, asset_type, 
            payload_json, vision_sky_bucket, wave_score_0_10
        ) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "flowers_asset_1",
            rubric_id,
            "file_2",
            "unique_2",
            "photo",
            json.dumps({"vision_category": "flowers"}),
            None,
            None,
        ),
    )

    bot.data.db.commit()

    # Call the inventory report function
    await bot._send_sea_inventory_report(is_prod=True, initiator_id=12345)

    # Verify only sea asset is counted
    assert len(messages_sent) == 1
    report_text = messages_sent[0]["text"]
    assert "🗂 Остатки фото «Море»: 1" in report_text
    assert "Солнечно: 1" in report_text
