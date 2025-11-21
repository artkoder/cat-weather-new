import json
import re
from datetime import UTC, datetime, timedelta

import pytest

import caption_gen
from caption_gen import POSTCARD_OPENING_CHOICES, POSTCARD_RUBRIC_HASHTAG, generate_postcard_caption
from data_access import Asset
from openai_client import OpenAIResponse
from main import LOVE_COLLECTION_LINK
from zoneinfo import ZoneInfo

KALININGRAD_TZ = ZoneInfo("Europe/Kaliningrad")


def _make_asset(
    *,
    asset_id: str = "1",
    city: str | None = "Светлогорск",
    region: str | None = "Калининградская область",
    country: str = "Россия",
    tags: list[str] | None = None,
    vision_results: dict[str, object] | None = None,
    postcard_score: int | None = 9,
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
        postcard_score=postcard_score,
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


def _set_postcard_now(monkeypatch: pytest.MonkeyPatch, current: datetime) -> None:
    monkeypatch.setattr(caption_gen, "_now_kaliningrad", lambda: current, raising=False)


def _dummy_postcard_client(caption_text: str) -> DummyOpenAI:
    response = OpenAIResponse(
        {
            "caption": caption_text,
            "hashtags": ["#Светлогорск", "#вид", "#детали"],
        },
        {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
    )
    return DummyOpenAI([response])


def _extract_prompt_payload(prompt_text: str) -> dict[str, object]:
    marker = "Контекст сцены (JSON):\n"
    start = prompt_text.index(marker) + len(marker)
    end_marker = "\n\nСформируй JSON"
    end = prompt_text.index(end_marker, start)
    payload_text = prompt_text[start:end]
    return json.loads(payload_text)


@pytest.mark.asyncio
async def test_postcard_caption_with_location_and_love_block() -> None:
    asset = _make_asset(tags=["море", "закат"])
    response = OpenAIResponse(
        {
            "caption": "Порадую вас красивым видом Светлогорска. Закат подсвечивает променад у воды.",
            "hashtags": [
                "#Светлогорск",
                "#котопогода",
                "#БалтийскоеМоре",
                "#закат",
                "#открыточныйвид",
            ],
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
    assert 3 <= len(hashtags) <= 5


@pytest.mark.asyncio
async def test_postcard_caption_filters_stopwords() -> None:
    asset = _make_asset(tags=["город"], postcard_score=10)
    bad = OpenAIResponse(
        {
            "caption": "Порадую вас красивым видом Светлогорска. Волшебный вечер у воды.",
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
            "caption": "Порадую вас красивым видом Светлогорска. Вечерний свет ложится на улицы.",
            "hashtags": ["#БалтийскаяКоса", "#вид", "#улицы"],
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
    normalized_tags = {tag.casefold() for tag in hashtags}
    assert "#калининградскаяобласть" in normalized_tags
    assert 3 <= len(hashtags) <= 5


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
    normalized_tags = {tag.casefold() for tag in hashtags}
    assert "#калининградскаяобласть" in normalized_tags
    assert POSTCARD_RUBRIC_HASHTAG in hashtags
    assert 3 <= len(hashtags) <= 5


@pytest.mark.asyncio
async def test_postcard_caption_strips_latin_words() -> None:
    asset = _make_asset(tags=["лес"])
    response = OpenAIResponse(
        {
            "caption": "Порадую вас красивым видом Светлогорска. Calm light и сосны рядом.",
            "hashtags": ["#Светлогорск", "#лес"],
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
    assert "light" not in text_part.lower()
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
            "caption": "Порадую вас красивым видом Зеленоградска. Море сегодня спокойно тянется к берегу.",
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
    assert 3 <= len(hashtags) <= 5


@pytest.mark.asyncio
async def test_postcard_caption_removes_forbidden_hashtags_and_keeps_region() -> None:
    asset = _make_asset(postcard_score=8)
    response = OpenAIResponse(
        {
            "caption": "Порадую вас красивым видом Светлогорска. Тихая улица ведёт к воде.",
            "hashtags": ["#котопогода", "#открыточныйвид", "#река"],
        },
        {
            "prompt_tokens": 4,
            "completion_tokens": 7,
            "total_tokens": 11,
        },
    )
    client = DummyOpenAI([response])

    _, hashtags = await generate_postcard_caption(
        client,
        asset,
        region_hashtag="#КалининградскаяОбласть",
        stopwords=[],
    )

    normalized_tags = {tag.casefold() for tag in hashtags}
    assert "#котопогода" not in normalized_tags
    assert POSTCARD_RUBRIC_HASHTAG not in hashtags
    assert "#калининградскаяобласть" in normalized_tags
    assert 3 <= len(hashtags) <= 5


@pytest.mark.asyncio
async def test_postcard_caption_handles_postcard_score_threshold() -> None:
    high_asset = _make_asset(asset_id="high", postcard_score=9)
    low_asset = _make_asset(asset_id="low", postcard_score=8)
    high_response = OpenAIResponse(
        {
            "caption": "Порадую вас красивым видом Светлогорска. Закат мягко подсвечивает улицы.",
            "hashtags": ["#Светлогорск", "#закат"],
        },
        {
            "prompt_tokens": 4,
            "completion_tokens": 7,
            "total_tokens": 11,
        },
    )
    low_response = OpenAIResponse(
        {
            "caption": "Порадую вас красивым видом Светлогорска. Домики отражаются в воде.",
            "hashtags": ["#Светлогорск", "#открыточныйвид"],
        },
        {
            "prompt_tokens": 4,
            "completion_tokens": 7,
            "total_tokens": 11,
        },
    )

    high_client = DummyOpenAI([high_response])
    _, high_hashtags = await generate_postcard_caption(
        high_client,
        high_asset,
        region_hashtag="#КалининградскаяОбласть",
        stopwords=[],
    )
    assert POSTCARD_RUBRIC_HASHTAG in high_hashtags

    low_client = DummyOpenAI([low_response])
    _, low_hashtags = await generate_postcard_caption(
        low_client,
        low_asset,
        region_hashtag="#КалининградскаяОбласть",
        stopwords=[],
    )
    assert POSTCARD_RUBRIC_HASHTAG not in low_hashtags


@pytest.mark.asyncio
async def test_postcard_prompt_includes_bird_info() -> None:
    asset = _make_asset(
        asset_id="birds",
        tags=["море"],
        vision_results={"tags": ["лебедь", "вода"]},
    )
    response = OpenAIResponse(
        {
            "caption": "Порадую вас красивым видом Светлогорска. Лебедь выводит дорожку на тихой воде.",
            "hashtags": ["#Светлогорск", "#лебедь", "#вода"],
        },
        {
            "prompt_tokens": 4,
            "completion_tokens": 8,
            "total_tokens": 12,
        },
    )
    client = DummyOpenAI([response])

    await generate_postcard_caption(
        client,
        asset,
        region_hashtag="#КалининградскаяОбласть",
        stopwords=[],
    )

    assert client.calls, "OpenAI client should be called"
    prompt = str(client.calls[0]["user_prompt"])
    assert "has_birds: true" in prompt
    assert "bird_tags: лебедь" in prompt


@pytest.mark.asyncio
async def test_postcard_caption_omits_season_line_for_recent_photo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2024, 8, 1, tzinfo=KALININGRAD_TZ)
    _set_postcard_now(monkeypatch, now)
    asset = _make_asset()
    asset.captured_at = (now - timedelta(days=30)).isoformat()
    client = _dummy_postcard_client(
        "Порадую вас красивым видом Светлогорска. Тёплое солнце ложится на улицы."
    )

    caption, _ = await generate_postcard_caption(
        client,
        asset,
        region_hashtag="#КалининградскаяОбласть",
        stopwords=[],
    )

    assert not re.search(r"(весна|лето|осень|зима) \d{4}", caption)


@pytest.mark.asyncio
async def test_postcard_caption_includes_season_line_for_old_summer_photo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2025, 10, 1, tzinfo=KALININGRAD_TZ)
    _set_postcard_now(monkeypatch, now)
    asset = _make_asset()
    asset.captured_at = datetime(2025, 7, 10, tzinfo=UTC).isoformat()
    client = _dummy_postcard_client(
        "Порадую вас красивым видом Светлогорска. Тёплый воздух подчёркивает линии набережной."
    )

    caption, _ = await generate_postcard_caption(
        client,
        asset,
        region_hashtag="#КалининградскаяОбласть",
        stopwords=[],
    )

    parts = caption.split("\n\n")
    assert "лето 2025" in parts
    assert parts[-1] == LOVE_COLLECTION_LINK


@pytest.mark.asyncio
async def test_postcard_caption_includes_season_line_for_old_autumn_photo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2025, 1, 10, tzinfo=KALININGRAD_TZ)
    _set_postcard_now(monkeypatch, now)
    asset = _make_asset()
    asset.captured_at = datetime(2024, 10, 5, tzinfo=UTC).isoformat()
    client = _dummy_postcard_client(
        "Порадую вас красивым видом Светлогорска. Спокойные дома прячутся в мягком свете."
    )

    caption, _ = await generate_postcard_caption(
        client,
        asset,
        region_hashtag="#КалининградскаяОбласть",
        stopwords=[],
    )

    assert "осень 2024" in caption.split("\n\n")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("capture_date", "now_value", "expected"),
    [
        (
            datetime(2023, 12, 20, tzinfo=UTC),
            datetime(2024, 3, 1, tzinfo=KALININGRAD_TZ),
            "зима 2023/2024",
        ),
        (
            datetime(2024, 1, 15, tzinfo=UTC),
            datetime(2024, 4, 1, tzinfo=KALININGRAD_TZ),
            "зима 2023/2024",
        ),
    ],
)
async def test_postcard_caption_includes_winter_season_line(
    monkeypatch: pytest.MonkeyPatch,
    capture_date: datetime,
    now_value: datetime,
    expected: str,
) -> None:
    _set_postcard_now(monkeypatch, now_value)
    asset = _make_asset()
    asset.captured_at = capture_date.isoformat()
    client = _dummy_postcard_client(
        "Порадую вас красивым видом Светлогорска. Лёгкий холод подчёркивает спокойствие улиц."
    )

    caption, _ = await generate_postcard_caption(
        client,
        asset,
        region_hashtag="#КалининградскаяОбласть",
        stopwords=[],
    )

    assert expected in caption.split("\n\n")


@pytest.mark.asyncio
async def test_postcard_caption_uses_created_at_when_captured_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2024, 9, 1, tzinfo=KALININGRAD_TZ)
    _set_postcard_now(monkeypatch, now)
    asset = _make_asset()
    asset.captured_at = None
    asset.created_at = datetime(2024, 5, 1, tzinfo=UTC).isoformat()
    client = _dummy_postcard_client(
        "Порадую вас красивым видом Светлогорска. Солнечные дорожки ведут к воде."
    )

    caption, _ = await generate_postcard_caption(
        client,
        asset,
        region_hashtag="#КалининградскаяОбласть",
        stopwords=[],
    )

    assert "весна 2024" in caption.split("\n\n")


@pytest.mark.asyncio
async def test_postcard_caption_skips_season_line_without_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2024, 8, 1, tzinfo=KALININGRAD_TZ)
    _set_postcard_now(monkeypatch, now)
    asset = _make_asset()
    asset.captured_at = None
    asset.created_at = ""
    client = _dummy_postcard_client(
        "Порадую вас красивым видом Светлогорска. Ровные линии домов поддерживают тишину."
    )

    caption, _ = await generate_postcard_caption(
        client,
        asset,
        region_hashtag="#КалининградскаяОбласть",
        stopwords=[],
    )

    assert not re.search(r"(весна|лето|осень|зима) \d{4}", caption)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("captured_at", "created_at", "now_value", "expected"),
    [
        (
            datetime(2024, 7, 10, tzinfo=UTC),
            None,
            datetime(2024, 7, 1, tzinfo=KALININGRAD_TZ),
            False,
        ),
        (
            datetime(2024, 1, 15, tzinfo=UTC),
            None,
            datetime(2024, 7, 1, tzinfo=KALININGRAD_TZ),
            True,
        ),
        (
            datetime(2024, 12, 20, tzinfo=UTC),
            None,
            datetime(2025, 1, 5, tzinfo=KALININGRAD_TZ),
            False,
        ),
        (
            datetime(2024, 12, 20, tzinfo=UTC),
            None,
            datetime(2025, 5, 10, tzinfo=KALININGRAD_TZ),
            True,
        ),
        (
            None,
            datetime(2024, 1, 15, tzinfo=UTC),
            datetime(2024, 7, 1, tzinfo=KALININGRAD_TZ),
            True,
        ),
        (None, None, datetime(2024, 7, 1, tzinfo=KALININGRAD_TZ), False),
    ],
)
async def test_postcard_is_out_of_season_flag_in_prompt(
    monkeypatch: pytest.MonkeyPatch,
    captured_at: datetime | None,
    created_at: datetime | None,
    now_value: datetime,
    expected: bool,
) -> None:
    _set_postcard_now(monkeypatch, now_value)
    asset = _make_asset()
    asset.captured_at = captured_at.isoformat() if captured_at else None
    asset.created_at = created_at.isoformat() if created_at else None
    client = _dummy_postcard_client(
        "Порадую вас красивым видом Светлогорска. Свет лёгким эхом скользит по домам."
    )

    await generate_postcard_caption(
        client,
        asset,
        region_hashtag="#КалининградскаяОбласть",
        stopwords=[],
    )

    assert client.calls, "OpenAI client should receive at least one call"
    prompt_payload = _extract_prompt_payload(str(client.calls[0]["user_prompt"]))
    assert prompt_payload.get("is_out_of_season") is expected


@pytest.mark.asyncio
async def test_postcard_prompt_in_season_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2024, 7, 1, tzinfo=KALININGRAD_TZ)
    _set_postcard_now(monkeypatch, now)
    asset = _make_asset()
    asset.captured_at = datetime(2024, 7, 2, tzinfo=UTC).isoformat()
    client = _dummy_postcard_client(
        "Порадую вас красивым видом Светлогорска. Лучи утреннего солнца бегут по плитке."
    )

    await generate_postcard_caption(
        client,
        asset,
        region_hashtag="#КалининградскаяОбласть",
        stopwords=[],
    )

    assert client.calls, "OpenAI client should receive at least one call"
    system_prompt = str(client.calls[0]["system_prompt"]).casefold()
    assert "прошедшем времени" not in system_prompt
    assert "вспом" not in system_prompt


@pytest.mark.asyncio
async def test_postcard_prompt_out_of_season_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2024, 7, 1, tzinfo=KALININGRAD_TZ)
    _set_postcard_now(monkeypatch, now)
    asset = _make_asset()
    asset.captured_at = datetime(2024, 1, 15, tzinfo=UTC).isoformat()
    client = _dummy_postcard_client(
        "Порадую вас красивым видом Светлогорска. Лёгкий иней остался на плитке набережной."
    )

    await generate_postcard_caption(
        client,
        asset,
        region_hashtag="#КалининградскаяОбласть",
        stopwords=[],
    )

    assert client.calls, "OpenAI client should receive at least one call"
    system_prompt = str(client.calls[0]["system_prompt"])  # keep original case
    assert "прошедшем времени" in system_prompt
    assert "Вспом" in system_prompt
