import json
import importlib.util
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_access import DataAccess
from main import Bot


class DummyResponse:
    def __init__(self, status: int, payload: dict[str, Any]):
        self.status = status
        self._payload = payload

    async def __aenter__(self) -> "DummyResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def json(self) -> dict[str, Any]:
        return self._payload

    async def text(self) -> str:
        return json.dumps(self._payload)


class DummySession:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.requests: list[tuple[str, dict[str, str], dict[str, str]]] = []

    def post(
        self,
        url: str,
        *,
        data: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> DummyResponse:
        stored_data = data or {}
        stored_headers = headers or {}
        self.requests.append((url, stored_data, stored_headers))
        return DummyResponse(200, self.payload)


def _load_schema(conn: sqlite3.Connection) -> None:
    asset_channel_path = Path(__file__).resolve().parents[1] / "migrations" / "0004_asset_channel.sql"
    conn.executescript(asset_channel_path.read_text(encoding="utf-8"))
    schema_path = Path(__file__).resolve().parents[1] / "migrations" / "0012_core_schema.sql"
    conn.executescript(schema_path.read_text(encoding="utf-8"))
    upgrade_path = Path(__file__).resolve().parents[1] / "migrations" / "0014_split_asset_channels.sql"
    conn.executescript(upgrade_path.read_text(encoding="utf-8"))
    uploader_path = Path(__file__).resolve().parents[1] / "migrations" / "0018_uploader_init.py"
    spec = importlib.util.spec_from_file_location("_migration_0018_uploader_init", uploader_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "run"):
            module.run(conn)
    assets_uploads_path = Path(__file__).resolve().parents[1] / "migrations" / "0020_assets_uploads.py"
    spec_uploads = importlib.util.spec_from_file_location("_migration_0020_assets_uploads", assets_uploads_path)
    if spec_uploads and spec_uploads.loader:
        module_uploads = importlib.util.module_from_spec(spec_uploads)
        spec_uploads.loader.exec_module(module_uploads)
        if hasattr(module_uploads, "run"):
            module_uploads.run(conn)
    mobile_sources_path = Path(__file__).resolve().parents[1] / "migrations" / "0021_mobile_sources.py"
    spec_sources = importlib.util.spec_from_file_location("_migration_0021_mobile_sources", mobile_sources_path)
    if spec_sources and spec_sources.loader:
        module_sources = importlib.util.module_from_spec(spec_sources)
        spec_sources.loader.exec_module(module_sources)
        if hasattr(module_sources, "run"):
            module_sources.run(conn)


@pytest.fixture
def db_connection():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _load_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def bot_instance(tmp_path):
    db_path = tmp_path / "bot.db"
    bot = Bot("test-token", str(db_path))
    try:
        yield bot
    finally:
        bot.db.close()


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
        "tags": ["architecture", "flowers", "overcast"],
        "framing": "wide",
        "architecture_close_up": False,
        "architecture_wide": True,
        "weather_image": "overcast",
        "season_guess": "spring",
        "arch_style": {"label": "gothic", "confidence": 0.9},
        "safety": {"nsfw": False, "reason": "безопасно"},
        "category": "architecture",
        "photo_weather": "overcast",
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
    assert stored_payload["arch_style"]["confidence"] == pytest.approx(0.9)

    asset = data.get_asset(asset_id)
    assert asset is not None
    assert asset.vision_category == "architecture"
    assert asset.vision_photo_weather == "overcast"
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


def test_update_asset_persists_null_confidence(db_connection):
    data = DataAccess(db_connection)
    asset_id = data.save_asset(
        channel_id=1,
        message_id=11,
        template=None,
        hashtags=None,
        tg_chat_id=1,
        caption=None,
        kind="photo",
    )

    payload = {
        "status": "ok",
        "provider": "test-model",
        "arch_style": {"label": "art_deco", "confidence": None},
    }

    data.update_asset(asset_id, vision_results=payload)

    row = db_connection.execute(
        "SELECT result_json FROM vision_results WHERE asset_id=? ORDER BY id DESC LIMIT 1",
        (asset_id,),
    ).fetchone()

    assert row is not None
    stored_payload = json.loads(row["result_json"])
    assert stored_payload["arch_style"]["confidence"] is None

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
                "description": (
                "Кадровка/ракурс снимка. Используй один из вариантов: close_up, medium, wide."
            ),
            "enum": [
                "close_up",
                "medium",
                "wide",
            ],
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
                "description": (
                "Краткое описание погодных условий на фото (на английском). Выбирай из категорий: sunny, partly_cloudy, overcast, rain, snow, fog, night."
            ),
            "enum": [
                "sunny",
                "partly_cloudy",
                "overcast",
                "rain",
                "snow",
                "fog",
                "night",
            ],
            },
            "season_guess": {
                "type": ["string", "null"],
                "description": "Предполагаемый сезон (spring, summer, autumn, winter) или null, если неясно.",
                "enum": ["spring", "summer", "autumn", "winter", None],
            },
            "arch_style": {
                "type": ["object", "null"],
                "description": (
                    "Предполагаемый архитектурный стиль. Либо null, либо объект с label (строка) и confidence (число 0..1)."
                ),
                "additionalProperties": False,
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "Название архитектурного стиля (на английском).",
                        "minLength": 1,
                    },
                    "confidence": {
                        "type": ["number", "null"],
                        "description": "Уверенность в определении стиля (0 — неизвестно, 1 — уверен).",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
                "required": ["label", "confidence"],
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


@pytest.mark.asyncio
async def test_marine_synonym_appends_sea_tag(bot_instance):
    asset_id = bot_instance.data.save_asset(
        channel_id=99,
        message_id=1,
        template=None,
        hashtags=None,
        tg_chat_id=1,
        caption=None,
        kind="photo",
    )

    asset = bot_instance.data.get_asset(asset_id)
    assert asset is not None
    tags = ["ocean", "overcast"]

    await bot_instance._maybe_append_marine_tag(asset, tags)

    assert "sea" in tags

    payload = {"tags": tags}
    bot_instance.data.update_asset(asset_id, vision_results=payload)

    stored_asset = bot_instance.data.get_asset(asset_id)
    assert stored_asset is not None
    assert stored_asset.vision_results is not None
    assert "sea" in stored_asset.vision_results.get("tags", [])


@pytest.mark.asyncio
async def test_marine_lookup_appends_sea_tag(bot_instance):
    bot_instance.session = DummySession({"elements": [{"type": "way", "id": 1}]})

    asset_id = bot_instance.data.save_asset(
        channel_id=42,
        message_id=2,
        template=None,
        hashtags=None,
        tg_chat_id=1,
        caption=None,
        kind="photo",
    )
    bot_instance.data.update_asset(asset_id, latitude=54.1234, longitude=20.1234)

    asset = bot_instance.data.get_asset(asset_id)
    assert asset is not None
    tags = ["architecture"]

    await bot_instance._maybe_append_marine_tag(asset, tags)

    assert "sea" in tags

    payload = {"tags": tags}
    bot_instance.data.update_asset(asset_id, vision_results=payload)

    stored_asset = bot_instance.data.get_asset(asset_id)
    assert stored_asset is not None
    assert stored_asset.vision_results is not None
    assert "sea" in stored_asset.vision_results.get("tags", [])

    assert bot_instance.session.requests
    url, data, _headers = bot_instance.session.requests[0]
    assert "overpass" in url
    assert "around:250" in data.get("data", "")
