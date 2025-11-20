import json
import re
from datetime import UTC, datetime

import pytest

from caption_gen import POSTCARD_OPENING_CHOICES, POSTCARD_RUBRIC_HASHTAG, generate_postcard_caption
from data_access import Asset
from openai_client import OpenAIResponse
from main import LOVE_COLLECTION_LINK


def _make_asset(
    *,
    asset_id: str = "1",
    city: str | None = "Светлогорск",
    region: str | None = "Калининградская область",
    country: str = "Россия",
    tags: list[str] | None = None,
    vision_results: dict[str, object] | None = None,
) -> Asset:
    payload: dict[str, object] = {
        "channel_id": 1,
        "tg_chat_id": 1,
        "message_id": 1,
        "origin": "test",
        "city": city,
        "country": country,
    }
    if region is not None:
        payload["region"] = region
    vision_payload: dict[str, object] = {}
    if isinstance(vision_results, dict):
        vision_payload.update(vision_results)
    if tags:
        merged_tags: list[str] = []
        existing_tags = vision_payload.get("tags")
        if isinstance(existing_tags, list):
            merged_tags.extend(existing_tags)
        for tag in tags:
            if tag not in merged_tags:
                merged_tags.append(tag)
        vision_payload["tags"] = merged_tags
    if vision_payload:
        payload["vision_results"] = vision_payload
    payload_json = json.dumps(payload, ensure_ascii=False)
    return Asset(
        id=asset_id,
        upload_id=None,
        file_ref=f"file-{asset_id}",
        content_type="image/jpeg",
        sha256=None,
        width=1080,
        height=1080,
        exif_json=None,
        labels_json="[]",
        tg_message_id="1:1",
        payload_json=payload_json,
        created_at=datetime.now(UTC).isoformat(),
        exif=None,
        labels=json.loads("[]"),
        payload=payload,
        legacy_values={},
    )


class DummyOpenAI:
    def __init__(self, responses: list[OpenAIResponse]):
        self.api_key = "test-key"
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    async def generate_json(self, **kwargs) -> OpenAIResponse:  # type: ignore[override]
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


@pytest.mark.asyncio
async def test_postcard_caption_with_location_and_love_block() -> None:
    asset = _make_asset(tags=["море", "закат"])
    response = OpenAIResponse(
        {
            "sentence": "Это Светлогорск — закат подсвечивает променад у воды.",
            "hashtags": ["#Светлогорск", "#котопогода", "#БалтийскоеМоре", "#море", "#закат"],
        },
        {
            "prompt_tokens": 5,
            "completion_tokens": 10,
            "total_tokens": 15,
        },
    )
    client = DummyOpenAI([response])

    caption, hashtags = await generate_postcard_caption(
        client,
        asset,
        region_hashtag="#КалининградскаяОбласть",
        stopwords=[],
    )

    parts = caption.split("\n\n")
    text_part = parts[0]
    assert text_part.startswith(POSTCARD_OPENING_CHOICES)
    assert "Светлогор" in text_part
    assert not re.search(r"[A-Za-z]", text_part)
    assert LOVE_COLLECTION_LINK in caption
    assert caption.strip().endswith(LOVE_COLLECTION_LINK)
    normalized_tags = {tag.casefold() for tag in hashtags}
    assert POSTCARD_RUBRIC_HASHTAG in hashtags
    assert "#калининградскаяобласть" in normalized_tags
    assert "#светлогорск" in normalized_tags
    assert "#котопогода" not in normalized_tags
    assert "#балтийскоморе" not in normalized_tags
    assert all("море" not in tag for tag in normalized_tags)
    assert 2 <= len(hashtags) <= 4


@pytest.mark.asyncio
async def test_postcard_caption_filters_stopwords() -> None:
    asset = _make_asset(tags=["город"])
    bad = OpenAIResponse(
        {
            "sentence": "Это Светлогорск — волшебный вечер у воды.",
            "hashtags": ["#город", "#вечер"],
        },
        {
            "prompt_tokens": 3,
            "completion_tokens": 5,
            "total_tokens": 8,
        },
    )
    good = OpenAIResponse(
        {
            "sentence": "Это Светлогорск — вечерний свет ложится на улицы.",
            "hashtags": ["#БалтийскаяКоса", "#вид"],
        },
        {
            "prompt_tokens": 3,
            "completion_tokens": 6,
            "total_tokens": 9,
        },
    )
    client = DummyOpenAI([bad, good])

    caption, hashtags = await generate_postcard_caption(
        client,
        asset,
        region_hashtag="#КалининградскаяОбласть",
        stopwords=["волшебный"],
    )

    assert len(client.calls) == 2
    assert "волшеб" not in caption.casefold()
    assert LOVE_COLLECTION_LINK in caption
    assert caption.strip().endswith(LOVE_COLLECTION_LINK)
    assert any(tag.casefold() == "#калининградскаяобласть".casefold() for tag in hashtags)


@pytest.mark.asyncio
async def test_postcard_caption_fallback_without_openai() -> None:
    asset = _make_asset(city=None, region="Куршская коса", tags=["дюны"])

    caption, hashtags = await generate_postcard_caption(
        None,
        asset,
        region_hashtag="#КалининградскаяОбласть",
        stopwords=[],
    )

    text_part = caption.split("\n\n")[0]
    assert text_part.startswith(POSTCARD_OPENING_CHOICES)
    assert "Куршская коса" in text_part
    assert LOVE_COLLECTION_LINK in caption
    assert caption.strip().endswith(LOVE_COLLECTION_LINK)
    assert any(tag.casefold() == "#калининградскаяобласть".casefold() for tag in hashtags)
    assert POSTCARD_RUBRIC_HASHTAG in hashtags
    assert 2 <= len(hashtags) <= 4


@pytest.mark.asyncio
async def test_postcard_caption_strips_latin_words() -> None:
    asset = _make_asset(tags=["лес"])
    response = OpenAIResponse(
        {
            "sentence": "Это Светлогорск — calm water и сосны рядом.",
            "hashtags": ["#Светлогорск"],
        },
        {
            "prompt_tokens": 4,
            "completion_tokens": 8,
            "total_tokens": 12,
        },
    )
    client = DummyOpenAI([response])

    caption, _ = await generate_postcard_caption(
        client,
        asset,
        region_hashtag="#КалининградскаяОбласть",
        stopwords=[],
    )

    text_part = caption.split("\n\n")[0]
    assert "calm" not in text_part.lower()
    assert "water" not in text_part.lower()
    assert text_part.startswith(POSTCARD_OPENING_CHOICES)


@pytest.mark.asyncio
async def test_postcard_caption_keeps_sea_hashtags_for_sea_scene() -> None:
    asset = _make_asset(
        city="Зеленоградск",
        tags=["море"],
        vision_results={"sea_wave_score": {"value": 2}},
    )
    response = OpenAIResponse(
        {
            "sentence": "Это Зеленоградск — море сегодня спокойно тянется к берегу.",
            "hashtags": ["#море", "#морскойбриз"],
        },
        {
            "prompt_tokens": 5,
            "completion_tokens": 9,
            "total_tokens": 14,
        },
    )
    client = DummyOpenAI([response])

    _, hashtags = await generate_postcard_caption(
        client,
        asset,
        region_hashtag="#КалининградскаяОбласть",
        stopwords=[],
    )

    assert any(tag.casefold().startswith("#море") for tag in hashtags)
