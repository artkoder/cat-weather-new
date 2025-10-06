import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

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
        "primary_scene": "architecture",
        "guess_country": "Россия",
        "guess_city": "Калининград",
        "arch_view": True,
        "weather": {"label": "cloudy", "description": "пасмурно"},
        "objects": ["роза"],
        "tags": ["architecture", "flowers"],
        "safety": {"nsfw": False, "violence": False, "self_harm": False, "hate": False},
        "notes": "",
        "category": "architecture",
        "photo_weather": "cloudy",
        "flower_varieties": ["роза"],
        "confidence": 0.87,
    }

    data.update_asset(
        asset_id,
        vision_results=payload,
        vision_category=payload["category"],
        vision_arch_view="yes",
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
    assert row["arch_view"] == "yes"
    assert row["photo_weather"] == "пасмурно"
    assert json.loads(row["flower_varieties"]) == ["роза"]
    assert pytest.approx(row["confidence"], rel=1e-6) == 0.87
    stored_payload = json.loads(row["result_json"])
    assert stored_payload["primary_scene"] == "architecture"

    asset = data.get_asset(asset_id)
    assert asset is not None
    assert asset.vision_category == "architecture"
    assert asset.vision_photo_weather == "cloudy"
    assert asset.vision_results == payload

    skip_payload = {"status": "skipped"}
    data.update_asset(asset_id, vision_results=skip_payload)
    skipped_row = db_connection.execute(
        "SELECT status, provider FROM vision_results WHERE asset_id=? ORDER BY id DESC LIMIT 1",
        (asset_id,),
    ).fetchone()
    assert skipped_row["status"] == "skipped"
    assert skipped_row["provider"] is None


def test_asset_vision_schema_definition():
    from main import ASSET_VISION_V1_SCHEMA

    expected = {
        "type": "object",
        "title": "asset_vision_v1",
        "description": (
            "Структурированное описание фото для классификации рубрик, "
            "угадывания города и оценки безопасности."
        ),
        "additionalProperties": False,
        "properties": {
            "primary_scene": {
                "type": "string",
                "description": "Короткое описание основного сюжета (на русском языке).",
                "minLength": 1,
            },
            "guess_country": {
                "type": ["string", "null"],
                "description": "Предполагаемая страна, если есть контекст.",
            },
            "guess_city": {
                "type": ["string", "null"],
                "description": "Предполагаемый город, если распознаётся.",
            },
            "arch_view": {
                "type": "boolean",
                "description": "Присутствует ли в кадре архитектурный ракурс (здания, фасады, панорамы).",
            },
            "weather": {
                "type": "object",
                "description": "Погода, которую можно определить по фото.",
                "additionalProperties": False,
                "properties": {
                    "label": {
                        "type": "string",
                        "description": (
                            "Краткая машинно-читаемая метка: indoor, sunny, cloudy, rainy, snowy, foggy, stormy, twilight, night."
                        ),
                        "minLength": 1,
                    },
                    "description": {
                        "type": "string",
                        "description": "Небольшое текстовое описание погоды на русском.",
                        "minLength": 1,
                    },
                },
                "required": ["label", "description"],
            },
            "objects": {
                "type": "array",
                "description": (
                    "Список заметных объектов или деталей. Если присутствуют цветы, укажи их вид."
                ),
                "items": {"type": "string", "minLength": 1},
                "default": [],
            },
            "tags": {
                "type": "array",
                "description": (
                    "Набор тегов (на английском в нижнем регистре) для downstream-логики: architecture, flowers, people, animals и т.п."
                ),
                "items": {"type": "string", "minLength": 1},
                "default": [],
            },
            "safety": {
                "type": "object",
                "description": "Флаги чувствительного контента (True если категория потенциально нарушает правила).",
                "additionalProperties": False,
                "properties": {
                    "nsfw": {"type": "boolean"},
                    "violence": {"type": "boolean"},
                    "self_harm": {"type": "boolean"},
                    "hate": {"type": "boolean"},
                },
                "required": ["nsfw", "violence", "self_harm", "hate"],
            },
            "notes": {
                "type": "string",
                "description": "Дополнительные наблюдения, если они нужны редакции.",
                "default": "",
            },
        },
        "required": [
            "primary_scene",
            "guess_country",
            "guess_city",
            "arch_view",
            "weather",
            "objects",
            "tags",
            "safety",
        ],
    }

    assert ASSET_VISION_V1_SCHEMA == expected
