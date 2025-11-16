import os
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import main as main_module


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

    bot = main_module.Bot(token="dummy", db_path=str(db_path))

    rubric_id = bot.db.execute("SELECT id FROM rubrics WHERE code='sea'").fetchone()["id"]

    assets_data = [
        {"sky": "clear", "wave": 0},
        {"sky": "clear", "wave": 1},
        {"sky": "mostly_clear", "wave": 2},
        {"sky": "partly_cloudy", "wave": 3},
        {"sky": "partly_cloudy", "wave": 3},
        {"sky": "mostly_cloudy", "wave": 5},
        {"sky": "overcast", "wave": 7},
    ]

    channel_id = -1001234567890
    for idx, asset_data in enumerate(assets_data, 1):
        file_meta = {
            "file_id": f"file_{idx}",
            "file_unique_id": f"unique_{idx}",
            "file_name": f"test_{idx}.jpg",
            "mime_type": "image/jpeg",
            "width": 1920,
            "height": 1080,
        }
        asset_id = bot.data.save_asset(
            channel_id,
            1000 + idx,
            template=None,
            hashtags=None,
            tg_chat_id=channel_id,
            caption=None,
            kind="photo",
            file_meta=file_meta,
            rubric_id=rubric_id,
            wave_score_0_10=asset_data["wave"],
        )
        vision_payload = {
            "is_sea": True,
            "sea_wave_score": asset_data["wave"],
            "photo_sky": asset_data["sky"],
        }
        bot.data.update_asset(
            asset_id,
            vision_category="sea",
            vision_results=vision_payload,
            vision_sky_bucket=asset_data["sky"],
        )

    bot.db.commit()

    total, sky_counts, _ = bot._compute_sea_inventory_stats()
    assert total == len(assets_data)
    assert sum(sky_counts.values()) == total
    assert sky_counts["clear"] == 2
    assert sky_counts["mostly_clear"] == 1
    assert sky_counts["partly_cloudy"] == 2
    assert sky_counts["mostly_cloudy"] == 1
    assert sky_counts["overcast"] == 1

    await bot._send_sea_inventory_report(is_prod=True, initiator_id=12345)

    assert len(messages_sent) == 1
    message = messages_sent[0]
    assert message["chat_id"] == 12345

    report_text = message["text"]
    assert "🗂 Остатки фото «Море»: 7" in report_text
    assert "Солнечно: 2" in report_text
    assert "Преимущественно ясно: 1" in report_text
    assert "Переменная облачность: 2" in report_text
    assert "Преимущественно облачно: 1" in report_text
    assert "Пасмурно: 1" in report_text

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

    bot = main_module.Bot(token="dummy", db_path=str(db_path))

    await bot._send_sea_inventory_report(is_prod=True, initiator_id=12345)

    assert len(messages_sent) == 1
    message = messages_sent[0]
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

    bot = main_module.Bot(token="dummy", db_path=str(db_path))

    sea_rubric_id = bot.db.execute("SELECT id FROM rubrics WHERE code='sea'").fetchone()["id"]
    flowers_rubric_id = bot.db.execute("SELECT id FROM rubrics WHERE code='flowers'").fetchone()[
        "id"
    ]

    channel_id = -1001234567890

    file_meta = {
        "file_id": "file_1",
        "file_unique_id": "unique_1",
        "file_name": "sea_test.jpg",
        "mime_type": "image/jpeg",
        "width": 1920,
        "height": 1080,
    }
    asset_id = bot.data.save_asset(
        channel_id,
        1001,
        template=None,
        hashtags=None,
        tg_chat_id=channel_id,
        caption=None,
        kind="photo",
        file_meta=file_meta,
        rubric_id=sea_rubric_id,
        wave_score_0_10=0,
    )
    vision_payload = {
        "is_sea": True,
        "sea_wave_score": 0,
        "photo_sky": "clear",
    }
    bot.data.update_asset(
        asset_id,
        vision_category="sea",
        vision_results=vision_payload,
        vision_sky_bucket="clear",
    )

    file_meta = {
        "file_id": "file_2",
        "file_unique_id": "unique_2",
        "file_name": "flowers_test.jpg",
        "mime_type": "image/jpeg",
        "width": 1920,
        "height": 1080,
    }
    asset_id = bot.data.save_asset(
        channel_id,
        1002,
        template=None,
        hashtags=None,
        tg_chat_id=channel_id,
        caption=None,
        kind="photo",
        file_meta=file_meta,
        rubric_id=flowers_rubric_id,
    )
    vision_payload = {
        "is_flower": True,
    }
    bot.data.update_asset(
        asset_id,
        vision_category="flowers",
        vision_results=vision_payload,
    )

    bot.db.commit()

    await bot._send_sea_inventory_report(is_prod=True, initiator_id=12345)

    assert len(messages_sent) == 1
    report_text = messages_sent[0]["text"]
    assert "🗂 Остатки фото «Море»: 1" in report_text
    assert "Солнечно: 1" in report_text


@pytest.mark.asyncio
async def test_sea_inventory_bucket_mapping_from_cloud_cover(monkeypatch, tmp_path):
    """Ensure cloud cover values bucket correctly and totals align with counts."""
    db_path = tmp_path / "sea_inventory_clouds.db"
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

    bot = main_module.Bot(token="dummy", db_path=str(db_path))

    sea_rubric_id = bot.db.execute("SELECT id FROM rubrics WHERE code='sea'").fetchone()["id"]

    channel_id = -100555666777
    cloud_samples = [0, 15, 40, 55, 70, 95]
    expected_counts = {
        "clear": 2,
        "mostly_clear": 1,
        "partly_cloudy": 1,
        "mostly_cloudy": 1,
        "overcast": 1,
    }

    for idx, cloud_pct in enumerate(cloud_samples, 1):
        file_meta = {
            "file_id": f"cloud_file_{idx}",
            "file_unique_id": f"cloud_unique_{idx}",
            "file_name": f"cloud_{idx}.jpg",
            "mime_type": "image/jpeg",
            "width": 1600,
            "height": 900,
        }
        asset_id = bot.data.save_asset(
            channel_id,
            2000 + idx,
            template=None,
            hashtags=None,
            tg_chat_id=channel_id,
            caption=None,
            kind="photo",
            file_meta=file_meta,
            rubric_id=sea_rubric_id,
            wave_score_0_10=idx % 10,
        )
        vision_payload = {
            "is_sea": True,
            "weather": {"sky": {"cloud_cover_pct": cloud_pct}},
        }
        bot.data.update_asset(
            asset_id,
            vision_category="sea",
            vision_results=vision_payload,
        )

    bot.db.commit()

    total, sky_counts, _ = bot._compute_sea_inventory_stats()
    assert total == len(cloud_samples)
    assert sum(sky_counts.values()) == total
    for bucket, expected in expected_counts.items():
        assert sky_counts[bucket] == expected

    await bot._send_sea_inventory_report(is_prod=True, initiator_id=98765)

    assert len(messages_sent) == 1
    text = messages_sent[0]["text"]
    assert f"🗂 Остатки фото «Море»: {len(cloud_samples)}" in text

    label_map = {
        "clear": "Солнечно",
        "mostly_clear": "Преимущественно ясно",
        "partly_cloudy": "Переменная облачность",
        "mostly_cloudy": "Преимущественно облачно",
        "overcast": "Пасмурно",
    }
    for bucket, label in label_map.items():
        expected = expected_counts.get(bucket, 0)
        assert f"{label}: {expected}" in text
