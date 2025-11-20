import json
from datetime import UTC, datetime

import pytest

from caption_gen import POSTCARD_PREFIX, generate_postcard_caption
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
    if tags:
        payload["vision_results"] = {"tags": tags}
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
            "caption": f"{POSTCARD_PREFIX}Светлогорском — тихий кадр с янтарным небом.",
            "hashtags": ["#Светлогорск", "#Балтика"],
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

    assert caption.startswith(POSTCARD_PREFIX)
    assert "Светлогорск" in caption
    assert LOVE_COLLECTION_LINK in caption
    assert caption.strip().endswith(LOVE_COLLECTION_LINK)
    assert any(tag.casefold() == "#калининградскаяобласть".casefold() for tag in hashtags)
    assert any(tag.casefold() == "#светлогорск".casefold() for tag in hashtags)
    assert 3 <= len(hashtags) <= 5


@pytest.mark.asyncio
async def test_postcard_caption_filters_stopwords() -> None:
    asset = _make_asset(tags=["город"])
    bad = OpenAIResponse(
        {
            "caption": f"{POSTCARD_PREFIX}видом города — волшебный вечер.",
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
            "caption": f"{POSTCARD_PREFIX}Светлогорска — тихая линия горизонта вдохновляет.",
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

    assert caption.startswith(POSTCARD_PREFIX)
    assert "Куршская коса" in caption
    assert LOVE_COLLECTION_LINK in caption
    assert caption.strip().endswith(LOVE_COLLECTION_LINK)
    assert any(tag.casefold() == "#калининградскаяобласть".casefold() for tag in hashtags)
    assert 3 <= len(hashtags) <= 5
