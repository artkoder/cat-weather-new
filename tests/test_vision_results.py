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
        "arch_view": True,
        "caption": "Архитектурный фасад",
        "objects": ["роза"],
        "is_outdoor": True,
        "guess_country": "Россия",
        "guess_city": "Калининград",
        "location_confidence": 0.82,
        "landmarks": ["Кафедральный собор"],
        "tags": ["architecture", "flowers", "cloudy"],
        "framing": "wide shot",
        "architecture_close_up": False,
        "architecture_wide": True,
        "weather_image": "cloudy",
        "season_guess": "spring",
        "arch_style": "gothic",
        "safety": {"nsfw": False, "reason": "безопасно"},
        "category": "architecture",
        "photo_weather": "cloudy",
        "photo_weather_display": "пасмурно",
        "flower_varieties": ["роза"],
    }

    data.update_asset(
        asset_id,
        vision_results=payload,
        vision_category=payload["category"],
        vision_arch_view="yes",
        vision_photo_weather=payload["photo_weather"],
        vision_confidence=payload["location_confidence"],
        vision_flower_varieties=payload["flower_varieties"],
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
    assert row["confidence"] == pytest.approx(0.82)
    stored_payload = json.loads(row["result_json"])
    assert stored_payload["caption"] == "Архитектурный фасад"

    asset = data.get_asset(asset_id)
    assert asset is not None
    assert asset.vision_category == "architecture"
    assert asset.vision_photo_weather == "cloudy"
    assert asset.vision_confidence == pytest.approx(0.82)
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
            "arch_view": {
                "type": "boolean",
                "description": "Присутствует ли в кадре архитектурный ракурс (здания, фасады, панорамы).",
            },
            "caption": {
                "type": "string",
                "description": "Короткое описание основного сюжета (на русском языке).",
                "minLength": 1,
            },
            "objects": {
                "type": "array",
                "description": (
                    "Список заметных объектов или деталей. Если присутствуют цветы, укажи их вид."
                ),
                "items": {"type": "string", "minLength": 1},
                "default": [],
            },
            "is_outdoor": {
                "type": "boolean",
                "description": "True, если сцена снята на улице (иначе — в помещении).",
            },
            "guess_country": {
                "type": ["string", "null"],
                "description": "Предполагаемая страна, если есть контекст.",
            },
            "guess_city": {
                "type": ["string", "null"],
                "description": "Предполагаемый город, если распознаётся.",
            },
            "location_confidence": {
                "type": "number",
                "description": "Числовая уверенность в локации (0 — нет уверенности, 1 — полностью уверен).",
                "minimum": 0,
                "maximum": 1,
            },
            "landmarks": {
                "type": "array",
                "description": "Имена распознанных достопримечательностей или ориентиров.",
                "items": {"type": "string", "minLength": 1},
                "default": [],
            },
            "tags": {
                "type": "array",
                "description": (
                    "Набор тегов (на английском в нижнем регистре) для downstream-логики: architecture, flowers, people, animals и т.п."
                ),
                "items": {"type": "string", "minLength": 1},
                "minItems": 3,
                "maxItems": 12,
                "default": [],
            },
            "framing": {
                "type": "string",
                "description": "Кадровка/ракурс снимка (например, close-up, medium shot, wide shot).",
                "minLength": 1,
            },
            "architecture_close_up": {
                "type": "boolean",
                "description": "Есть ли крупный план архитектурных деталей.",
            },
            "architecture_wide": {
                "type": "boolean",
                "description": "Есть ли широкий архитектурный план или панорама.",
            },
            "weather_image": {
                "type": "string",
                "description": "Краткое описание погодных условий на фото (на английском).",
                "minLength": 1,
            },
            "season_guess": {
                "type": ["string", "null"],
                "description": "Предполагаемый сезон (spring, summer, autumn, winter) или null, если неясно.",
            },
            "arch_style": {
                "type": ["string", "null"],
                "description": "Предполагаемый архитектурный стиль, если распознан.",
            },
            "safety": {
                "type": "object",
                "description": "Информация о чувствительном контенте: nsfw и краткая причина.",
                "additionalProperties": False,
                "properties": {
                    "nsfw": {"type": "boolean"},
                    "reason": {
                        "type": "string",
                        "description": "Краткое пояснение статуса безопасности (на русском).",
                        "minLength": 1,
                    },
                },
                "required": ["nsfw", "reason"],
            },
        },
        "required": [
            "arch_view",
            "caption",
            "objects",
            "is_outdoor",
            "guess_country",
            "guess_city",
            "location_confidence",
            "landmarks",
            "tags",
            "framing",
            "architecture_close_up",
            "architecture_wide",
            "weather_image",
            "season_guess",
            "safety",
        ],
    }

    assert ASSET_VISION_V1_SCHEMA == expected
