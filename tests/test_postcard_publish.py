import json
import logging
import os
import sys
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageChops

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import caption_gen  # noqa: E402
import main as main_module  # noqa: E402
import postcard_watermark  # noqa: E402
from caption_gen import POSTCARD_OPENING_CHOICES  # noqa: E402
from geo_context import GeoContext  # noqa: E402
from main import LOVE_COLLECTION_LINK, POSTCARD_MIN_SCORE  # noqa: E402
from openai_client import OpenAIResponse  # noqa: E402
from postcard_watermark import WATERMARK_PATH  # noqa: E402

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "dummy")


class DummySupabase:
    async def insert_token_usage(self, *args: Any, **kwargs: Any) -> tuple[bool, Any, Any]:
        return False, None, "disabled"

    async def aclose(self) -> None:
        return None


async def async_noop(*_args: Any, **_kwargs: Any) -> None:
    return None


class DummyPostcardOpenAI:
    def __init__(self, sentence: str) -> None:
        self.api_key = "test-key"
        self._response = OpenAIResponse(
            {
                "caption": sentence,
                "hashtags": ["#Светлогорск", "#вид"],
            },
            {
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "total_tokens": 15,
            },
        )

    async def generate_json(self, **kwargs: Any) -> OpenAIResponse:  # type: ignore[override]
        return self._response


def _create_postcard_asset(
    bot: main_module.Bot,
    *,
    city: str = "Светлогорск",
    region: str = "Калининградская область",
    postcard_score: int = 10,
    latitude: float | None = None,
    longitude: float | None = None,
    vision_tags: list[str] | None = None,
) -> str:
    asset_id = bot.data.create_asset(
        upload_id="upload-1",
        file_ref="photo-file",
        content_type="image/jpeg",
        sha256="sha-test",
        width=1080,
        height=1350,
        tg_message_id="1:1",
        tg_chat_id=1,
        source="telegram",
    )
    payload_row = bot.db.execute(
        "SELECT payload_json FROM assets WHERE id=?", (asset_id,)
    ).fetchone()
    payload = json.loads(payload_row["payload_json"] or "{}")
    payload.update({"city": city, "region": region, "tg_chat_id": 1, "message_id": 1})
    if latitude is not None:
        payload["latitude"] = latitude
    if longitude is not None:
        payload["longitude"] = longitude
    if vision_tags:
        vision_payload = payload.get("vision_results")
        if not isinstance(vision_payload, dict):
            vision_payload = {}
        existing_tags = vision_payload.get("tags")
        merged_tags: list[str] = list(existing_tags) if isinstance(existing_tags, list) else []
        for tag in vision_tags:
            if tag not in merged_tags:
                merged_tags.append(tag)
        vision_payload["tags"] = merged_tags
        payload["vision_results"] = vision_payload
    now = datetime.now(UTC)
    bot.db.execute(
        "UPDATE assets SET payload_json=?, postcard_score=?, captured_at=?, photo_doy=? WHERE id=?",
        (
            json.dumps(payload, ensure_ascii=False),
            postcard_score,
            now.isoformat(),
            now.timetuple().tm_yday,
            asset_id,
        ),
    )
    bot.db.commit()

    image_path = Path(bot.asset_storage) / f"{asset_id}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    placeholder = Image.new("RGB", (1080, 1350), color=(64, 64, 64))
    try:
        placeholder.save(image_path, format="JPEG")
    finally:
        placeholder.close()
    bot.data.update_asset(asset_id, local_path=str(image_path))
    return asset_id


@pytest.mark.asyncio
async def test_postcard_publish_routes_to_prod_channel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(main_module.Bot, "_record_openai_usage", async_noop, raising=False)

    bot = main_module.Bot("dummy", str(tmp_path / "postcard-prod.db"))
    bot.supabase = DummySupabase()
    bot.openai = DummyPostcardOpenAI("Это Светлогорск — мягкий свет над Балтикой.")

    rubric = bot.data.get_rubric_by_code("postcard")
    assert rubric is not None
    prod_channel = -900501
    test_channel = -900502
    config = dict(rubric.config or {})
    config.update(
        {
            "enabled": True,
            "channel_id": prod_channel,
            "test_channel_id": test_channel,
            "postcard_region_hashtag": "#КалининградскаяОбласть",
            "postcard_stopwords": ["клише"],
        }
    )
    bot.data.save_rubric_config("postcard", config)
    _create_postcard_asset(bot)

    send_calls: list[dict[str, Any]] = []
    _message_calls: list[dict[str, Any]] = []

    async def capture_api_request(
        self: main_module.Bot, method: str, data: Any = None, *, files: Any = None
    ) -> dict[str, Any]:  # type: ignore[override]
        if method == "sendPhoto":
            send_calls.append({"data": data, "files": files})
            return {"ok": True, "result": {"message_id": 777}}
        if method == "sendMessage":
            _message_calls.append(data or {})
            return {"ok": True, "result": {}}
        return {"ok": True, "result": {}}

    monkeypatch.setattr(main_module.Bot, "api_request", capture_api_request, raising=False)

    result = await bot.publish_rubric("postcard", test=False)

    assert result is True
    assert send_calls
    payload = send_calls[0]["data"]
    assert payload["chat_id"] == prod_channel
    caption = payload["caption"]
    assert caption.startswith(POSTCARD_OPENING_CHOICES)
    assert LOVE_COLLECTION_LINK in caption
    assert payload["parse_mode"] == "HTML"

    bot.db.close()


@pytest.mark.asyncio
async def test_postcard_prod_cleanup_removes_asset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(main_module.Bot, "_record_openai_usage", async_noop, raising=False)

    bot = main_module.Bot("dummy", str(tmp_path / "postcard-cleanup.db"))
    bot.supabase = DummySupabase()
    bot.openai = DummyPostcardOpenAI("Морской берег и город в лучах заката.")

    rubric = bot.data.get_rubric_by_code("postcard")
    assert rubric is not None
    config = dict(rubric.config or {})
    config.update(
        {
            "enabled": True,
            "channel_id": -940010,
            "test_channel_id": -940011,
            "postcard_region_hashtag": "#КалининградскаяОбласть",
            "postcard_stopwords": [],
        }
    )
    bot.data.save_rubric_config("postcard", config)

    asset_id = _create_postcard_asset(bot)
    asset_path = Path(bot.asset_storage) / f"{asset_id}.jpg"
    assert asset_path.exists()

    send_calls: list[dict[str, Any]] = []

    async def capture_api_request(
        self: main_module.Bot, method: str, data: Any = None, *, files: Any = None
    ) -> dict[str, Any]:  # type: ignore[override]
        if method == "sendPhoto":
            send_calls.append(data or {})
            return {"ok": True, "result": {"message_id": 77}}
        return {"ok": True, "result": {"message_id": 1}}

    delete_calls: list[str] = []

    async def fake_delete_asset_message(self: main_module.Bot, asset: Any) -> None:  # type: ignore[override]
        delete_calls.append(str(getattr(asset, "id", "")))

    monkeypatch.setattr(main_module.Bot, "api_request", capture_api_request, raising=False)
    monkeypatch.setattr(main_module.Bot, "_delete_asset_message", fake_delete_asset_message, raising=False)

    caplog.clear()
    with caplog.at_level(logging.INFO):
        result = await bot.publish_rubric("postcard", test=False)

    assert result is True
    assert send_calls
    assert bot.data.get_asset(asset_id) is None
    assert bot.db.execute("SELECT 1 FROM assets WHERE id=?", (asset_id,)).fetchone() is None
    assert not asset_path.exists()
    assert delete_calls == [str(asset_id)]

    messages = [record.getMessage() for record in caplog.records]
    assert any("POSTCARD_RUBRIC prod_cleanup_success" in msg for msg in messages)

    bot.db.close()


@pytest.mark.asyncio
async def test_postcard_publish_routes_to_test_channel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(main_module.Bot, "_record_openai_usage", async_noop, raising=False)

    bot = main_module.Bot("dummy", str(tmp_path / "postcard-test.db"))
    bot.supabase = DummySupabase()
    bot.openai = DummyPostcardOpenAI("Это Янтарный — вечернее золото на набережной.")

    rubric = bot.data.get_rubric_by_code("postcard")
    assert rubric is not None
    prod_channel = -910601
    test_channel = -910602
    config = dict(rubric.config or {})
    config.update(
        {
            "enabled": True,
            "channel_id": prod_channel,
            "test_channel_id": test_channel,
            "postcard_region_hashtag": "#КалининградскаяОбласть",
            "postcard_stopwords": [],
        }
    )
    bot.data.save_rubric_config("postcard", config)
    _create_postcard_asset(bot, city="Янтарный")

    send_calls: list[dict[str, Any]] = []
    _message_calls: list[dict[str, Any]] = []

    async def capture_api_request(
        self: main_module.Bot, method: str, data: Any = None, *, files: Any = None
    ) -> dict[str, Any]:  # type: ignore[override]
        if method == "sendPhoto":
            send_calls.append({"data": data, "files": files})
            return {"ok": True, "result": {"message_id": 888}}
        if method == "sendMessage":
            _message_calls.append(data or {})
            return {"ok": True, "result": {}}
        return {"ok": True, "result": {}}

    monkeypatch.setattr(main_module.Bot, "api_request", capture_api_request, raising=False)

    result = await bot.publish_rubric("postcard", test=True)

    assert result is True
    assert send_calls
    payload = send_calls[0]["data"]
    assert payload["chat_id"] == test_channel
    caption = payload["caption"]
    assert caption.startswith(POSTCARD_OPENING_CHOICES)
    assert LOVE_COLLECTION_LINK in caption

    bot.db.close()


@pytest.mark.asyncio
async def test_postcard_publish_renders_expected_markdown_caption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(main_module.Bot, "_record_openai_usage", async_noop, raising=False)

    bot = main_module.Bot("dummy", str(tmp_path / "postcard-md.db"))
    bot.supabase = DummySupabase()
    bot.openai = None

    rubric = bot.data.get_rubric_by_code("postcard")
    assert rubric is not None
    config = dict(rubric.config or {})
    config.update(
        {
            "enabled": True,
            "channel_id": -910701,
            "test_channel_id": -910702,
            "postcard_region_hashtag": "#КалининградскаяОбласть",
            "postcard_stopwords": [],
        }
    )
    bot.data.save_rubric_config("postcard", config)
    _create_postcard_asset(bot)

    async def fake_generate_postcard_caption(*_args: Any, **_kwargs: Any) -> tuple[str, list[str]]:
        base = f"{POSTCARD_OPENING_CHOICES[0]} Мягкий свет над водой."
        return f"{base}\n\n{LOVE_COLLECTION_LINK}", ["#один", "#два"]

    monkeypatch.setattr(
        caption_gen,
        "generate_postcard_caption",
        fake_generate_postcard_caption,
        raising=False,
    )

    send_calls: list[dict[str, Any]] = []
    _message_calls: list[dict[str, Any]] = []

    async def capture_api_request(
        self: main_module.Bot, method: str, data: Any = None, *, files: Any = None
    ) -> dict[str, Any]:  # type: ignore[override]
        if method == "sendPhoto":
            send_calls.append({"data": data, "files": files})
            return {"ok": True, "result": {"message_id": 7777}}
        if method == "sendMessage":
            _message_calls.append(data or {})
            return {"ok": True, "result": {}}
        return {"ok": True, "result": {}}

    monkeypatch.setattr(main_module.Bot, "api_request", capture_api_request, raising=False)

    result = await bot.publish_rubric("postcard", test=True)

    assert result is True
    assert send_calls
    payload = send_calls[0]["data"]
    expected_body = f"{POSTCARD_OPENING_CHOICES[0]} Мягкий свет над водой."
    expected_hashtags = "#один #два"
    expected_caption = f"{expected_body}\n\n{expected_hashtags}\n\n{LOVE_COLLECTION_LINK}"
    assert payload["caption"] == expected_caption
    assert payload["parse_mode"] == "HTML"

    bot.db.close()


@pytest.mark.asyncio
async def test_postcard_publish_applies_watermark_to_photo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if not WATERMARK_PATH.is_file():
        pytest.skip("LoveKaliningrad watermark asset is required for this test.")

    monkeypatch.setattr(main_module.Bot, "_record_openai_usage", async_noop, raising=False)

    bot = main_module.Bot("dummy", str(tmp_path / "postcard-watermark.db"))
    bot.supabase = DummySupabase()
    bot.openai = DummyPostcardOpenAI("Это Зеленоградск — лёгкое утро у моря.")

    rubric = bot.data.get_rubric_by_code("postcard")
    assert rubric is not None
    prod_channel = -910701
    test_channel = -910702
    config = dict(rubric.config or {})
    config.update(
        {
            "enabled": True,
            "channel_id": prod_channel,
            "test_channel_id": test_channel,
            "postcard_region_hashtag": "#КалининградскаяОбласть",
            "postcard_stopwords": ["клише"],
        }
    )
    bot.data.save_rubric_config("postcard", config)

    def _make_local_asset(image_path: Path, color: tuple[int, int, int]) -> str:
        image = Image.new("RGB", (1200, 1600), color=color)
        image.save(image_path, format="JPEG")
        image.close()
        asset_id = _create_postcard_asset(bot)
        bot.data.update_asset(asset_id, local_path=str(image_path))
        return asset_id

    first_asset_path = tmp_path / "postcard-first.jpg"
    first_asset_id = _make_local_asset(first_asset_path, (32, 64, 96))

    send_calls: list[dict[str, Any]] = []
    _message_calls: list[dict[str, Any]] = []

    async def capture_api_request(
        self: main_module.Bot, method: str, data: Any = None, *, files: Any = None
    ) -> dict[str, Any]:  # type: ignore[override]
        if method == "sendPhoto":
            send_calls.append({"data": data, "files": files})
            return {"ok": True, "result": {"message_id": len(send_calls)}}
        if method == "sendMessage":
            _message_calls.append(data or {})
            return {"ok": True, "result": {}}
        return {"ok": True, "result": {}}

    monkeypatch.setattr(main_module.Bot, "api_request", capture_api_request, raising=False)

    baseline_image: Image.Image | None = None
    watermarked_image: Image.Image | None = None

    try:
        with monkeypatch.context() as ctx:
            ctx.setattr(
                postcard_watermark,
                "add_love_kaliningrad_watermark",
                lambda img: img.copy(),
                raising=False,
            )
            assert await bot.publish_rubric("postcard", test=True) is True

        # Mark the first asset as used so it is not selected again.
        # The first run consumes (deletes) the local file, so selecting it again would fail.
        bot.data.mark_assets_used([first_asset_id])
        # In test mode, reuse logic is skipped, so we also lower the score to ensure the new asset is preferred.
        bot.db.execute("UPDATE assets SET postcard_score=5 WHERE id=?", (first_asset_id,))
        bot.db.commit()

        assert send_calls, "Expected baseline sendPhoto call"

        def _photo_bytes(index: int) -> bytes:
            entry = send_calls[index]
            files = entry["files"] or {}
            photo_tuple = files.get("photo")
            assert photo_tuple is not None
            return photo_tuple[1]

        def _image_from_bytes(payload: bytes) -> Image.Image:
            buffer = BytesIO(payload)
            with Image.open(buffer) as img:
                copy = img.convert("RGB")
            buffer.close()
            return copy

        baseline_image = _image_from_bytes(_photo_bytes(0))

        second_asset_path = tmp_path / "postcard-second.jpg"
        _make_local_asset(second_asset_path, (48, 96, 144))

        assert await bot.publish_rubric("postcard", test=True) is True
        assert len(send_calls) >= 2
        watermarked_image = _image_from_bytes(_photo_bytes(1))

        diff = ImageChops.difference(baseline_image, watermarked_image).convert("L")
        width, height = diff.size
        region_height = max(10, height // 5)
        region_width = max(10, width // 4)
        left = max(0, (width - region_width) // 2)
        bottom_region = diff.crop((left, height - region_height, left + region_width, height))
        extrema = bottom_region.getextrema()
        assert extrema is not None
        assert extrema[1] > 0, "Bottom-center region should change after watermarking"
    finally:
        if baseline_image is not None:
            baseline_image.close()
        if watermarked_image is not None:
            watermarked_image.close()
        bot.db.close()


@pytest.mark.asyncio
async def test_postcard_publish_sends_inventory_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(main_module.Bot, "_record_openai_usage", async_noop, raising=False)

    bot = main_module.Bot("dummy", str(tmp_path / "postcard-inventory.db"))
    bot.supabase = DummySupabase()
    bot.openai = DummyPostcardOpenAI("Это Пионерский — золото на горизонте.")

    rubric = bot.data.get_rubric_by_code("postcard")
    assert rubric is not None
    rubric_title = rubric.title or "Открыточный вид"
    config = dict(rubric.config or {})
    config.update(
        {
            "enabled": True,
            "channel_id": -920001,
            "test_channel_id": -920002,
            "postcard_region_hashtag": "#КалининградскаяОбласть",
            "postcard_stopwords": [],
        }
    )
    bot.data.save_rubric_config("postcard", config)
    _create_postcard_asset(bot)

    send_calls: list[dict[str, Any]] = []
    message_calls: list[dict[str, Any]] = []

    async def capture_api_request(
        self: main_module.Bot, method: str, data: Any = None, *, files: Any = None
    ) -> dict[str, Any]:  # type: ignore[override]
        if method == "sendPhoto":
            send_calls.append({"data": data, "files": files})
            return {"ok": True, "result": {"message_id": 1010}}
        if method == "sendMessage":
            message_calls.append(data or {})
            return {"ok": True, "result": {}}
        return {"ok": True, "result": {}}

    monkeypatch.setattr(main_module.Bot, "api_request", capture_api_request, raising=False)

    result = await bot.publish_rubric("postcard", test=True, initiator_id=4321)

    assert result is True
    assert send_calls
    assert len(message_calls) == 1
    message = message_calls[0]
    assert message["chat_id"] == 4321
    text = message["text"]
    assert f"🗂 Остатки фото «{rubric_title}»: 1" in text
    assert "Открыточность (7–10)" in text
    for score in range(POSTCARD_MIN_SCORE, 10):
        assert f"• {score}/10: 0 ⚠️ мало" in text
    assert "• 10/10: 1 ⚠️ мало" in text

    bot.db.close()


@pytest.mark.asyncio
async def test_postcard_prod_publish_decrements_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(main_module.Bot, "_record_openai_usage", async_noop, raising=False)

    bot = main_module.Bot("dummy", str(tmp_path / "postcard-prod-inventory.db"))
    bot.supabase = DummySupabase()
    bot.openai = DummyPostcardOpenAI("Это Гвардейск — туманный рассвет.")

    rubric = bot.data.get_rubric_by_code("postcard")
    assert rubric is not None
    rubric_title = rubric.title or "Открыточный вид"
    config = dict(rubric.config or {})
    config.update(
        {
            "enabled": True,
            "channel_id": -925001,
            "test_channel_id": -925002,
            "postcard_region_hashtag": "#КалининградскаяОбласть",
            "postcard_stopwords": [],
        }
    )
    bot.data.save_rubric_config("postcard", config)
    _create_postcard_asset(bot)

    send_calls: list[dict[str, Any]] = []
    message_calls: list[dict[str, Any]] = []

    async def capture_api_request(
        self: main_module.Bot, method: str, data: Any = None, *, files: Any = None
    ) -> dict[str, Any]:  # type: ignore[override]
        if method == "sendPhoto":
            send_calls.append({"data": data, "files": files})
            return {"ok": True, "result": {"message_id": 2025}}
        if method == "sendMessage":
            message_calls.append(data or {})
            return {"ok": True, "result": {}}
        return {"ok": True, "result": {}}

    monkeypatch.setattr(main_module.Bot, "api_request", capture_api_request, raising=False)

    result = await bot.publish_rubric("postcard", test=False, initiator_id=5555)

    assert result is True
    assert send_calls
    assert message_calls
    report_messages = [entry for entry in message_calls if "text" in entry]
    assert report_messages
    report_text = report_messages[-1]["text"]
    assert f"🗂 Остатки фото «{rubric_title}»: 0" in report_text
    assert "• 10/10: 0" in report_text

    bot.db.close()


def test_postcard_inventory_ignores_other_rubrics(tmp_path: Path) -> None:
    bot = main_module.Bot("dummy", str(tmp_path / "postcard-cross-rubric.db"))

    postcard_rubric = bot.data.get_rubric_by_code("postcard")
    other_rubric = bot.data.get_rubric_by_code("flowers")
    assert postcard_rubric is not None
    assert other_rubric is not None

    asset_id = _create_postcard_asset(bot)

    total_before, score_counts_before, _ = bot._compute_postcard_inventory_stats()
    assert total_before == 1
    assert score_counts_before.get(10, 0) == 1

    bot.data.mark_assets_used([asset_id], rubric_code=other_rubric.code)
    bot.data.record_post_history(1234, 2001, asset_id, other_rubric.id, {"rubric_code": other_rubric.code})

    total_other_rubric, score_counts_other, _ = bot._compute_postcard_inventory_stats()
    assert total_other_rubric == 1
    assert score_counts_other.get(10, 0) == 1

    bot.data.mark_assets_used([asset_id], rubric_code="postcard")
    bot.data.record_post_history(1234, 2002, asset_id, postcard_rubric.id, {"rubric_code": "postcard"})

    total_after_postcard, score_counts_after, _ = bot._compute_postcard_inventory_stats()
    assert total_after_postcard == 0
    assert score_counts_after.get(10, 0) == 0

    bot.db.close()


@pytest.mark.asyncio
async def test_postcard_publish_notifies_when_no_high_score_assets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(main_module.Bot, "_record_openai_usage", async_noop, raising=False)

    bot = main_module.Bot("dummy", str(tmp_path / "postcard-empty.db"))
    bot.supabase = DummySupabase()
    bot.openai = DummyPostcardOpenAI("Это Балтийск — пустая набережная.")

    rubric = bot.data.get_rubric_by_code("postcard")
    assert rubric is not None
    config = dict(rubric.config or {})
    config.update(
        {
            "enabled": True,
            "channel_id": -930001,
            "test_channel_id": -930002,
            "postcard_region_hashtag": "#КалининградскаяОбласть",
            "postcard_stopwords": [],
        }
    )
    bot.data.save_rubric_config("postcard", config)
    _create_postcard_asset(bot, postcard_score=POSTCARD_MIN_SCORE - 1)

    send_calls: list[dict[str, Any]] = []
    message_calls: list[dict[str, Any]] = []

    async def capture_api_request(
        self: main_module.Bot, method: str, data: Any = None, *, files: Any = None
    ) -> dict[str, Any]:  # type: ignore[override]
        if method == "sendPhoto":
            send_calls.append({"data": data, "files": files})
            return {"ok": True, "result": {"message_id": 1}}
        if method == "sendMessage":
            message_calls.append(data or {})
            return {"ok": True, "result": {}}
        return {"ok": True, "result": {}}

    monkeypatch.setattr(main_module.Bot, "api_request", capture_api_request, raising=False)

    result = await bot.publish_rubric("postcard", test=True, initiator_id=9876)

    assert result is True
    assert not send_calls
    assert len(message_calls) == 1
    alert = message_calls[0]
    assert alert["chat_id"] == 9876
    text = alert["text"]
    assert "пропущена" in text
    assert f"{POSTCARD_MIN_SCORE}–10" in text

    bot.db.close()


@pytest.mark.asyncio
async def test_postcard_logging_exposes_geo_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(main_module.Bot, "_record_openai_usage", async_noop, raising=False)

    bot = main_module.Bot("dummy", str(tmp_path / "postcard-logs.db"))
    bot.supabase = DummySupabase()
    bot.openai = DummyPostcardOpenAI("Порадую вас красивым видом Светлогорска. Балтика рядом.")

    rubric = bot.data.get_rubric_by_code("postcard")
    assert rubric is not None
    config = dict(rubric.config or {})
    config.update(
        {
            "enabled": True,
            "channel_id": -940001,
            "test_channel_id": -940002,
            "postcard_region_hashtag": "#КалининградскаяОбласть",
            "postcard_stopwords": ["клише"],
        }
    )
    bot.data.save_rubric_config("postcard", config)

    geo_stub = GeoContext(
        main_place="Светлогорск",
        water_name="Балтийское море",
        water_kind="sea",
        national_park="Куршская коса",
        location_mode="inside_city",
        coastline_distance_m=120.0,
        water_candidates=tuple(),
        water_decision_reason="sea_tag_coastline",
        has_water_hint=True,
    )

    async def fake_geo_builder(
        *,
        lat: float | None,
        lon: float | None,
        asset_city: str | None,
        asset_tags: list[str] | None,
        asset_id: str | None = None,
    ) -> GeoContext:
        return geo_stub

    monkeypatch.setattr(caption_gen, "build_geo_context_for_asset", fake_geo_builder)

    _create_postcard_asset(
        bot,
        city="Светлогорск",
        postcard_score=9,
        latitude=54.7104,
        longitude=20.4522,
        vision_tags=["water", "sea", "закат"],
    )

    send_calls: list[dict[str, Any]] = []

    async def capture_api_request(
        self: main_module.Bot, method: str, data: Any = None, *, files: Any = None
    ) -> dict[str, Any]:  # type: ignore[override]
        if method == "sendPhoto":
            send_calls.append(data or {})
            return {"ok": True, "result": {"message_id": 2024}}
        if method == "sendMessage":
            return {"ok": True, "result": {}}
        return {"ok": True, "result": {}}

    monkeypatch.setattr(main_module.Bot, "api_request", capture_api_request, raising=False)

    caplog.clear()
    with caplog.at_level(logging.INFO):
        result = await bot.publish_rubric("postcard", test=True)

    assert result is True
    assert send_calls

    messages = [record.getMessage() for record in caplog.records]
    assert any("POSTCARD_RUBRIC geo_decision" in msg for msg in messages)
    assert any("POSTCARD_RUBRIC context_payload" in msg for msg in messages)
    assert any("POSTCARD_CAPTION response" in msg for msg in messages)
    assert any("POSTCARD_RUBRIC final_caption" in msg for msg in messages)

    bot.db.close()
