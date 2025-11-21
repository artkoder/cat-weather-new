import json
import os
import sys
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageChops

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import main as main_module  # noqa: E402
import postcard_watermark  # noqa: E402
from caption_gen import POSTCARD_OPENING_CHOICES  # noqa: E402
from main import LOVE_COLLECTION_LINK  # noqa: E402
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

    async def capture_api_request(
        self: main_module.Bot, method: str, data: Any = None, *, files: Any = None
    ) -> dict[str, Any]:  # type: ignore[override]
        if method == "sendPhoto":
            send_calls.append({"data": data, "files": files})
            return {"ok": True, "result": {"message_id": 777}}
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

    async def capture_api_request(
        self: main_module.Bot, method: str, data: Any = None, *, files: Any = None
    ) -> dict[str, Any]:  # type: ignore[override]
        if method == "sendPhoto":
            send_calls.append({"data": data, "files": files})
            return {"ok": True, "result": {"message_id": 888}}
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
    _make_local_asset(first_asset_path, (32, 64, 96))

    send_calls: list[dict[str, Any]] = []

    async def capture_api_request(
        self: main_module.Bot, method: str, data: Any = None, *, files: Any = None
    ) -> dict[str, Any]:  # type: ignore[override]
        if method == "sendPhoto":
            send_calls.append({"data": data, "files": files})
            return {"ok": True, "result": {"message_id": len(send_calls)}}
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
