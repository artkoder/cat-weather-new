import json
import sqlite3
from pathlib import Path

import pytest

from data_access import DataAccess


def _load_schema(conn: sqlite3.Connection) -> None:
    asset_channel_path = Path(__file__).resolve().parents[1] / "migrations" / "0004_asset_channel.sql"
    conn.executescript(asset_channel_path.read_text(encoding="utf-8"))
    schema_path = Path(__file__).resolve().parents[1] / "migrations" / "0012_core_schema.sql"
    conn.executescript(schema_path.read_text(encoding="utf-8"))
    upgrade_path = Path(__file__).resolve().parents[1] / "migrations" / "0014_split_asset_channels.sql"
    conn.executescript(upgrade_path.read_text(encoding="utf-8"))


@pytest.fixture
def db_connection():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _load_schema(conn)
    yield conn
    conn.close()


def test_update_asset_persists_vision_results(db_connection):
    data = DataAccess(db_connection)
    asset_id = data.save_asset(
        channel_id=1,
        message_id=10,
        template=None,
        hashtags=None,
        tg_chat_id=1,
        caption=None,
        kind="photo",
    )

    payload = {
        "status": "ok",
        "provider": "test-model",
        "category": "architecture",
        "arch_view": "небоскрёб",
        "photo_weather": "солнечно",
        "flower_varieties": ["роза"],
        "confidence": 0.87,
    }

    data.update_asset(
        asset_id,
        vision_results=payload,
        vision_category=payload["category"],
        vision_arch_view=payload["arch_view"],
        vision_photo_weather=payload["photo_weather"],
        vision_flower_varieties=payload["flower_varieties"],
        vision_confidence=payload["confidence"],
    )

    row = db_connection.execute(
        "SELECT provider, status, category, arch_view, photo_weather, flower_varieties, confidence, result_json "
        "FROM vision_results WHERE asset_id=? ORDER BY id DESC LIMIT 1",
        (asset_id,),
    ).fetchone()

    assert row is not None
    assert row["provider"] == "test-model"
    assert row["status"] == "ok"
    assert row["category"] == "architecture"
    assert row["arch_view"] == "небоскрёб"
    assert row["photo_weather"] == "солнечно"
    assert json.loads(row["flower_varieties"]) == ["роза"]
    assert pytest.approx(row["confidence"], rel=1e-6) == 0.87
    stored_payload = json.loads(row["result_json"])
    assert stored_payload["category"] == "architecture"

    asset = data.get_asset(asset_id)
    assert asset is not None
    assert asset.vision_category == "architecture"
    assert asset.vision_photo_weather == "солнечно"
    assert asset.vision_results == payload

    skip_payload = {"status": "skipped"}
    data.update_asset(asset_id, vision_results=skip_payload)
    skipped_row = db_connection.execute(
        "SELECT status, provider FROM vision_results WHERE asset_id=? ORDER BY id DESC LIMIT 1",
        (asset_id,),
    ).fetchone()
    assert skipped_row["status"] == "skipped"
    assert skipped_row["provider"] is None
