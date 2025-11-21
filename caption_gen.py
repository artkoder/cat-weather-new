from __future__ import annotations

import json
import logging
import random
import re
import time
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from data_access import Asset
from openai_client import OpenAIClient

if TYPE_CHECKING:  # pragma: no cover
    from jobs import Job
    from main import Bot

POSTCARD_OPENING_CHOICES = (
    "Порадую вас открыточным видом.",
    "Порадую вас красивым видом",
)
POSTCARD_PREFIX = POSTCARD_OPENING_CHOICES[1]
POSTCARD_RUBRIC_HASHTAG = "#открыточныйвид"
POSTCARD_DEFAULT_HASHTAGS = ["#Балтика", "#КрасивыйВид"]
POSTCARD_MIN_HASHTAGS = 3
POSTCARD_HASHTAG_LIMIT = 5
POSTCARD_BANNED_TAGS = {
    "#котопогода",
    "#Котопогода",
    "#БалтийскоеМоре",
    "#балтийскоеморе",
}
POSTCARD_BANNED_TAG_KEYS = {tag.lstrip("#").casefold() for tag in POSTCARD_BANNED_TAGS}
POSTCARD_ADDITIONAL_STOP_PHRASES = (
    "кадр, который хочется пересматривать снова",
    "делает настроение особенным",
    "наполняет особым настроением",
    "момент, который хочется сохранить навсегда",
    "идеальное место",
    "атмосферный",
    "атмосферная",
    "атмосферное",
    "магия момента",
    "уникальный",
    "уникальная",
    "уникальное",
    "особенное место, где хочется остаться навсегда",
)
POSTCARD_MARINE_KEYWORDS = ("море", "морск", "sea", "ocean", "coast", "shore", "beach")
POSTCARD_RUBRIC_TAG_THRESHOLD = 9
POSTCARD_BIRD_TAGS = (
    "bird",
    "birds",
    "swan",
    "swans",
    "duck",
    "ducks",
    "seagull",
    "seagulls",
    "goose",
    "geese",
    "лебедь",
    "лебеди",
    "утка",
    "утки",
    "чайка",
    "чайки",
    "гусь",
    "гуси",
    "журавль",
    "журавли",
    "аист",
    "аисты",
    "баклан",
    "бакланы",
    "птица",
    "птицы",
)
POSTCARD_BIRD_TAG_KEYS = {tag.casefold() for tag in POSTCARD_BIRD_TAGS}
_LATIN_WORD_PATTERN = re.compile(r"[A-Za-z]")
_POSTCARD_COMMON_STOPWORDS: tuple[str, ...] | None = None
_POSTCARD_TZ = ZoneInfo("Europe/Kaliningrad")
_POSTCARD_SEASON_AGE_THRESHOLD_DAYS = 60


def _now_kaliningrad() -> datetime:
    return datetime.now(tz=_POSTCARD_TZ)


def _parse_asset_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        normalized = text
        if normalized.upper().endswith("Z"):
            normalized = f"{normalized[:-1]}+00:00"
        match = re.match(r"(.*)([+-]\d{2})(\d{2})$", normalized)
        if match and ":" not in match.group(3):
            normalized = f"{match.group(1)}{match.group(2)}:{match.group(3)}"
        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError:
            try:
                timestamp = float(text)
            except (TypeError, ValueError):
                return None
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_POSTCARD_TZ)


def _resolve_photo_datetime(asset: Asset) -> datetime | None:
    candidates = (getattr(asset, "captured_at", None), getattr(asset, "created_at", None))
    for candidate in candidates:
        dt = _parse_asset_datetime(candidate)
        if dt is not None:
            return dt
    return None


def _format_postcard_season_label(photo_date: datetime) -> str:
    month = photo_date.month
    year = photo_date.year
    if month in (3, 4, 5):
        return f"весна {year}"
    if month in (6, 7, 8):
        return f"лето {year}"
    if month in (9, 10, 11):
        return f"осень {year}"
    if month == 12:
        return f"зима {year}/{year + 1}"
    # month is 1 or 2
    return f"зима {year - 1}/{year}"


def _resolve_postcard_season_line(asset: Asset) -> str | None:
    photo_dt = _resolve_photo_datetime(asset)
    if photo_dt is None:
        return None
    today = _now_kaliningrad().date()
    age_days = (today - photo_dt.date()).days
    if age_days <= _POSTCARD_SEASON_AGE_THRESHOLD_DAYS:
        return None
    return _format_postcard_season_label(photo_dt)


def _append_season_line(text: str, season_line: str | None) -> str:
    cleaned = text.strip()
    if not season_line:
        return cleaned
    if cleaned:
        return f"{cleaned}\n\n{season_line}"
    return season_line


def _attach_link_block(text: str, link_block: str) -> str:
    cleaned_text = text.strip()
    cleaned_link = (link_block or "").strip()
    if not cleaned_link:
        return cleaned_text
    if cleaned_text:
        return f"{cleaned_text}\n\n{cleaned_link}"
    return cleaned_link


def _sanitize_prompt_leaks(text: str) -> str:
    from main import sanitize_prompt_leaks  # avoid circular import at module load

    return sanitize_prompt_leaks(text)


def _build_link_block() -> str:
    from main import build_rubric_link_block  # avoid circular import at module load

    return build_rubric_link_block("postcard")


def _normalize_hashtag_candidate(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(" ", "")
    if not text:
        return None
    if not text.startswith("#"):
        text = f"#{text.lstrip('#')}"
    return text


def _normalize_city_hashtag(value: str | None) -> str | None:
    if not value:
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if not text:
        return None
    text = text.replace("ё", "е").replace("Ё", "Е")
    cleaned = re.sub(r"[^\w]+", "", text, flags=re.UNICODE)
    if not cleaned:
        return None
    return f"#{cleaned}"


def _deduplicate_hashtags(tags: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        normalized = _normalize_hashtag_candidate(tag)
        if not normalized:
            continue
        key = normalized.lstrip("#").lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _get_postcard_common_stopwords() -> tuple[str, ...]:
    global _POSTCARD_COMMON_STOPWORDS
    if _POSTCARD_COMMON_STOPWORDS is not None:
        return _POSTCARD_COMMON_STOPWORDS
    try:
        from flowers_patterns import load_flowers_knowledge

        kb = load_flowers_knowledge()
        words = sorted(kb.banned_words or [])
        _POSTCARD_COMMON_STOPWORDS = tuple(words)
    except Exception:
        logging.debug("POSTCARD_CAPTION common_stopwords_unavailable", exc_info=True)
        _POSTCARD_COMMON_STOPWORDS = ()
    return _POSTCARD_COMMON_STOPWORDS


def _collect_postcard_stopwords(stopwords: Sequence[str]) -> list[str]:
    pool: list[str] = []
    sources: tuple[Sequence[str] | tuple[str, ...], ...] = (
        stopwords,
        _get_postcard_common_stopwords(),
        POSTCARD_ADDITIONAL_STOP_PHRASES,
    )
    for source in sources:
        for item in source:
            text = str(item or "").strip()
            if text:
                pool.append(text)
    dedup: dict[str, str] = {}
    for word in pool:
        key = word.casefold()
        if key and key not in dedup:
            dedup[key] = word
    return list(dedup.values())


def _contains_banned_word(text: str, banned_words: Iterable[str]) -> bool:
    if not banned_words:
        return False
    lowered = text.casefold()
    tokens = {token.strip() for token in re.split(r"\W+", lowered) if token.strip()}
    for word in banned_words:
        normalized = str(word or "").strip().casefold()
        if not normalized:
            continue
        if normalized in tokens:
            return True
        pattern = re.escape(normalized)
        if re.search(rf"(?<!\w){pattern}(?!\w)", lowered):
            return True
    return False


def _build_postcard_opening(location: _LocationInfo) -> str:
    label = (location.city or location.region or "").strip()
    if label:
        opening = f"Порадую вас красивым видом {label}."
        return _sanitize_sentence(opening)
    return POSTCARD_OPENING_CHOICES[0]


def _sanitize_sentence(text: str) -> str:
    if not text:
        return ""
    cleaned = _sanitize_prompt_leaks(str(text).strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _ensure_sentence_punctuation(value: str) -> str:
    if not value:
        return ""
    if value[-1] not in ".!?…":
        return f"{value}."
    return value


def _sanitize_postcard_caption_text(text: str) -> str:
    if not text:
        return ""
    cleaned = _sanitize_prompt_leaks(str(text).strip())
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _remove_latin_words(text: str) -> str:
    if not text:
        return ""
    parts = re.split(r"(\s+)", text)
    filtered: list[str] = []
    for part in parts:
        if not part:
            continue
        if part.isspace() or part.startswith("#"):
            filtered.append(part)
            continue
        if _LATIN_WORD_PATTERN.search(part):
            continue
        filtered.append(part)
    rebuilt = "".join(filtered)
    rebuilt = re.sub(r"\s{2,}", " ", rebuilt)
    return rebuilt.strip()


def _location_value_in_text(text: str, candidate: str) -> bool:
    normalized_text = text.casefold()
    normalized_candidate = candidate.casefold().strip()
    if not normalized_candidate:
        return False
    if normalized_candidate in normalized_text:
        return True
    tokens = [token for token in re.split(r"[\s-]+", normalized_candidate) if token]
    for token in tokens:
        if len(token) < 4:
            continue
        if token in normalized_text:
            return True
        if len(token) > 5 and token[:-1] in normalized_text:
            return True
        if len(token) > 6 and token[:-2] in normalized_text:
            return True
    return False


def _mentions_location(text: str, location: _LocationInfo) -> bool:
    normalized_text = text.casefold()
    candidates = [location.city, location.region, location.display]
    for candidate in candidates:
        if candidate and _location_value_in_text(normalized_text, candidate):
            return True
    return False


def _looks_like_marine_tag(tag: str) -> bool:
    normalized = str(tag or "").strip()
    if not normalized:
        return False
    key = normalized.lstrip("#").casefold()
    return any(keyword in key for keyword in POSTCARD_MARINE_KEYWORDS)


def _is_sea_scene(asset: Asset) -> bool:
    results = asset.vision_results or {}
    if isinstance(results, dict):
        if bool(results.get("is_sea")):
            return True
        sea_wave = results.get("sea_wave_score")
        value: float | None = None
        if isinstance(sea_wave, dict):
            value = sea_wave.get("value")  # type: ignore[assignment]
        else:
            value = sea_wave  # type: ignore[assignment]
        try:
            if value is not None and float(value) >= 0:
                return True
        except (TypeError, ValueError):
            pass

    return False


@dataclass(slots=True)
class _LocationInfo:
    display: str
    city: str | None
    region: str | None
    country: str | None


def _resolve_location(asset: Asset) -> _LocationInfo:
    city = (asset.city or "").strip() or None
    region = getattr(asset, "region", None)
    if isinstance(region, str):
        region = region.strip() or None
    country = (asset.country or "").strip() or None
    display = city or region or country or "Калининградская область"
    return _LocationInfo(display=display, city=city, region=region, country=country)


def _collect_semantic_tags(asset: Asset) -> list[str]:
    tags: list[str] = []
    results = asset.vision_results or {}
    raw_tags = results.get("tags") if isinstance(results, dict) else None
    if isinstance(raw_tags, (list, tuple, set)):
        for tag in raw_tags:
            text = str(tag or "").strip()
            if text and text not in tags:
                tags.append(text)
    if isinstance(results, dict):
        if results.get("is_sunset") and "закат" not in tags:
            tags.append("закат")
        if results.get("photo_weather_display"):
            weather = str(results.get("photo_weather_display") or "").strip()
            if weather and weather not in tags:
                tags.append(weather)
    return tags[:8]


def _collect_bird_tags(asset: Asset) -> list[str]:
    results = asset.vision_results or {}
    raw_tags = results.get("tags") if isinstance(results, dict) else None
    if not isinstance(raw_tags, (list, tuple, set)):
        return []
    found: list[str] = []
    seen: set[str] = set()
    for tag in raw_tags:
        text = str(tag or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in POSTCARD_BIRD_TAG_KEYS and key not in seen:
            seen.add(key)
            found.append(text)
    return found


def _finalize_postcard_hashtags(
    candidate_tags: Iterable[str],
    region_hashtag: str | None,
    city_hashtag: str | None,
    *,
    is_sea_scene: bool,
    include_rubric_tag: bool,
    fallback_keywords: Sequence[str] | None = None,
) -> list[str]:
    prepared = _deduplicate_hashtags(candidate_tags)
    filtered: list[str] = []
    rubric_key = POSTCARD_RUBRIC_HASHTAG.lstrip("#").casefold()
    for tag in prepared:
        if not tag:
            continue
        key = tag.lstrip("#").casefold()
        if key in POSTCARD_BANNED_TAG_KEYS:
            continue
        if key == rubric_key and not include_rubric_tag:
            continue
        if not is_sea_scene and _looks_like_marine_tag(tag):
            continue
        filtered.append(tag)
    normalized_region_tag = _normalize_hashtag_candidate(region_hashtag)
    normalized_city_tag = _normalize_hashtag_candidate(city_hashtag)
    region_value = normalized_region_tag or (
        region_hashtag.strip() if isinstance(region_hashtag, str) else None
    )
    city_value = normalized_city_tag or (
        city_hashtag.strip() if isinstance(city_hashtag, str) else None
    )
    region_required_key: str | None = None
    if region_value:
        filtered.append(region_value)
        region_required_key = region_value.casefold()
    if city_value:
        filtered.append(city_value)
    if include_rubric_tag:
        filtered.append(POSTCARD_RUBRIC_HASHTAG)
    combined = _deduplicate_hashtags(filtered)
    fallback_candidates: list[str] = []
    if fallback_keywords:
        for keyword in fallback_keywords:
            candidate = _normalize_hashtag_candidate(keyword)
            if not candidate:
                continue
            key = candidate.lstrip("#").casefold()
            if key in POSTCARD_BANNED_TAG_KEYS:
                continue
            if key == rubric_key and not include_rubric_tag:
                continue
            if not is_sea_scene and _looks_like_marine_tag(candidate):
                continue
            if candidate not in combined:
                fallback_candidates.append(candidate)
    while len(combined) < POSTCARD_MIN_HASHTAGS and fallback_candidates:
        combined.append(fallback_candidates.pop(0))
    if len(combined) < POSTCARD_MIN_HASHTAGS:
        for fallback in POSTCARD_DEFAULT_HASHTAGS:
            candidate = _normalize_hashtag_candidate(fallback) or fallback
            key = candidate.lstrip("#").casefold()
            if key in POSTCARD_BANNED_TAG_KEYS:
                continue
            if key == rubric_key and not include_rubric_tag:
                continue
            if candidate not in combined:
                combined.append(candidate)
            if len(combined) >= POSTCARD_MIN_HASHTAGS:
                break
    required_keys: set[str] = set()
    if region_required_key:
        required_keys.add(region_required_key)
    if include_rubric_tag:
        required_keys.add(POSTCARD_RUBRIC_HASHTAG.casefold())
    return _limit_postcard_hashtags(combined, required_keys)


def _limit_postcard_hashtags(tags: list[str], required_keys: set[str]) -> list[str]:
    if len(tags) <= POSTCARD_HASHTAG_LIMIT:
        return tags
    limited: list[str] = []
    optional_indices: list[int] = []
    present_required: set[str] = set()
    for tag in tags:
        key = tag.casefold()
        is_required = key in required_keys
        if len(limited) < POSTCARD_HASHTAG_LIMIT:
            limited.append(tag)
            if is_required:
                present_required.add(key)
            else:
                optional_indices.append(len(limited) - 1)
            continue
        if is_required and key not in present_required and optional_indices:
            replace_index = optional_indices.pop()
            limited[replace_index] = tag
            present_required.add(key)
    return limited


def _postcard_fallback_sentence(location: _LocationInfo, semantic_tags: Sequence[str]) -> str:
    detail = None
    for tag in semantic_tags:
        text = str(tag or "").strip()
        if text:
            detail = text
            break
    label = location.display or "Калининградская область"
    fragment: str
    lowered = detail.casefold() if isinstance(detail, str) else ""
    if lowered and "закат" in lowered:
        fragment = "закат мягко подсвечивает горизонт"
    elif lowered and "море" in lowered:
        fragment = "тихая вода подчёркивает простор"
    elif lowered and "город" in lowered:
        fragment = "городские линии звучат спокойно"
    elif detail:
        fragment = f"вид с акцентом на {detail}"
    else:
        fragment = "спокойный вид для прогулки"
    sentence = f"Это {label} — {fragment}."
    return _sanitize_sentence(sentence)


async def generate_postcard_caption(
    openai: OpenAIClient | None,
    asset: Asset,
    region_hashtag: str,
    stopwords: Sequence[str],
) -> tuple[str, list[str]]:
    location = _resolve_location(asset)
    semantic_tags = _collect_semantic_tags(asset)
    bird_tags = _collect_bird_tags(asset)
    has_birds = bool(bird_tags)
    is_sea_scene = _is_sea_scene(asset)
    season_line = _resolve_postcard_season_line(asset)
    region_tag = _normalize_hashtag_candidate(region_hashtag)
    city_tag = _normalize_city_hashtag(location.city)
    banned_words = _collect_postcard_stopwords(stopwords)
    score_value = asset.postcard_score if isinstance(asset.postcard_score, int) else None
    include_rubric_tag = bool(
        score_value is not None and score_value >= POSTCARD_RUBRIC_TAG_THRESHOLD
    )
    context_payload: dict[str, Any] = {
        "location": location.display,
        "city": location.city,
        "region": location.region,
        "country": location.country,
        "tags": semantic_tags,
        "postcard_score": score_value,
        "location_case_hint": "prepositional",
        "has_birds": has_birds,
        "is_sea_scene": is_sea_scene,
    }
    extra_hint = asset.vision_results or {}
    if isinstance(extra_hint, dict):
        if extra_hint.get("weather_final_display"):
            context_payload["weather"] = extra_hint.get("weather_final_display")
        if extra_hint.get("is_sunset"):
            context_payload["is_sunset"] = True
    if bird_tags:
        context_payload["bird_tags"] = bird_tags
    payload_text = json.dumps(context_payload, ensure_ascii=False, separators=(",", ": "))
    stopwords_text = ", ".join(banned_words) if banned_words else ""
    region_prompt = region_tag or _normalize_hashtag_candidate(region_hashtag)
    if not region_prompt and isinstance(region_hashtag, str):
        region_prompt = region_hashtag.strip()
    if not region_prompt:
        region_prompt = "#КалининградскаяОбласть"
    system_prompt_lines = [
        "Ты — голос проекта «Котопогода» и готовишь подпись для рубрики «Открыточный вид».",
        "Собери подпись из 2–3 коротких предложений (до 250 символов) в спокойном, человеческом тоне.",
        "Первая фраза строго одна из двух: «Порадую вас открыточным видом.» или «Порадую вас красивым видом <локации>.».",
        "Если в данных есть город или регион, используй вторую формулу и поставь название в подходящем падеже (например, «в Калининградской области», «в Зеленоградске»).",
        "После первой фразы добавь 1–2 коротких предложения с конкретными деталями сцены без пафоса и канцелярита.",
        "Пиши только по-русски, не используй английские слова, эмодзи, хэштеги или ссылки внутри текста.",
        "Обязательно назови указанную локацию (город или регион) в корректной форме.",
        "Сохраняй дружелюбный и лаконичный тон, избегай рекламных штампов.",
    ]
    if stopwords_text:
        system_prompt_lines.append(f"Запрещены стоп-слова и клише: {stopwords_text}.")
    system_prompt_lines.append(
        "Если has_birds=true или перечислены bird_tags, мягко упомяни заметных птиц (лебеди, утки, чайки и т. п.), если это органично."
    )
    system_prompt_lines.append(
        f"Хэштеги возвращай отдельным массивом: всего 3–5 тегов, обязательно включай региональный тег {region_prompt} и 2–4 смысловых."
    )
    if city_tag:
        system_prompt_lines.append(
            "Если есть city, добавь отдельный хэштег с его названием в формате #Город."
        )
    system_prompt_lines.append("Хэштеги #котопогода и дубли запрещены.")
    if include_rubric_tag:
        system_prompt_lines.append(
            f"При высоком postcard_score добавь тег {POSTCARD_RUBRIC_HASHTAG} (только в массиве hashtags)."
        )
    else:
        system_prompt_lines.append(
            f"Не используй тег {POSTCARD_RUBRIC_HASHTAG} при текущем postcard_score."
        )
    system_prompt_lines.append("Ссылку «Полюбить 39» добавит система — не упоминай её в тексте.")
    system_prompt = "\n".join(system_prompt_lines)
    bird_info_lines = [f"has_birds: {'true' if has_birds else 'false'}."]
    if bird_tags:
        bird_info_lines.append(f"bird_tags: {', '.join(bird_tags)}.")
    bird_info = " ".join(bird_info_lines)
    user_prompt = (
        "Контекст сцены (JSON):\n"
        f"{payload_text}\n\n"
        'Сформируй JSON {"caption":"...","hashtags":["#..."]}.\n'
        "Caption — полная подпись по правилам system prompt (без хэштегов, ссылок и «Полюбить 39»).\n"
        "Hashtags — 3–5 тегов с символом #, без #котопогода и повторов. Обязательно включи региональный тег "
        f"{region_prompt} и подбери 2–4 смысловых тега по сцене.\n"
    )
    if city_tag:
        user_prompt += "Если указан city, добавь отдельный тег с названием города (#Город).\n"
    if include_rubric_tag:
        user_prompt += f"Добавь тег {POSTCARD_RUBRIC_HASHTAG} в массив hashtags.\n"
    else:
        user_prompt += f"Не используй тег {POSTCARD_RUBRIC_HASHTAG}.\n"
    user_prompt += f"\nПодсказка о птицах:\n{bird_info}"
    schema = {
        "type": "object",
        "properties": {
            "caption": {"type": "string"},
            "hashtags": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 5,
            },
        },
        "required": ["caption", "hashtags"],
    }
    region_value_for_tags = region_prompt
    default_tags = _finalize_postcard_hashtags(
        [],
        region_value_for_tags,
        city_tag,
        is_sea_scene=is_sea_scene,
        include_rubric_tag=include_rubric_tag,
        fallback_keywords=semantic_tags,
    )
    link_block = _build_link_block()
    if not openai or not getattr(openai, "api_key", None):
        fallback_sentence = _postcard_fallback_sentence(location, semantic_tags)
        opening = _build_postcard_opening(location)
        combined = _remove_latin_words(f"{opening} {fallback_sentence}".strip())
        caption_body = _append_season_line(combined, season_line)
        caption_with_block = _attach_link_block(caption_body, link_block)
        return caption_with_block.strip(), default_tags
    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            logging.info(
                "POSTCARD_CAPTION request model=%s attempt=%s/%s", "gpt-4o", attempt, attempts
            )
            response = await openai.generate_json(
                model="gpt-4o",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=schema,
                temperature=0.95,
                top_p=0.9,
            )
        except Exception:
            logging.exception("POSTCARD_CAPTION openai_error attempt=%s", attempt)
            continue
        if not response or not isinstance(response.content, dict):
            logging.warning("POSTCARD_CAPTION invalid_response attempt=%s", attempt)
            continue
        hashtag_raw = response.content.get("hashtags")
        if not isinstance(hashtag_raw, list):
            hashtag_raw = []
        caption_raw = response.content.get("caption")
        if not isinstance(caption_raw, str):
            caption_raw = response.content.get("sentence")
        if not isinstance(caption_raw, str):
            logging.warning("POSTCARD_CAPTION missing_caption attempt=%s", attempt)
            continue
        caption_text = _sanitize_postcard_caption_text(caption_raw)
        if not caption_text:
            logging.info("POSTCARD_CAPTION empty_caption attempt=%s", attempt)
            continue
        caption_text = re.sub(r"#\S+", "", caption_text).strip()
        caption_text = _remove_latin_words(caption_text)
        caption_text = re.sub(r"\s{2,}", " ", caption_text).strip()
        if not caption_text:
            logging.info("POSTCARD_CAPTION empty_after_latin attempt=%s", attempt)
            continue
        if caption_text[-1] not in ".!?…":
            caption_text = f"{caption_text}."
        opening_sentence = re.split(r"(?<=[.!?…])\s+", caption_text)[0]
        lowered_opening = opening_sentence.casefold()
        if not any(
            lowered_opening.startswith(prefix.casefold()) for prefix in POSTCARD_OPENING_CHOICES
        ):
            logging.info("POSTCARD_CAPTION invalid_opening attempt=%s", attempt)
            continue
        if banned_words and _contains_banned_word(caption_text, banned_words):
            logging.info("POSTCARD_CAPTION banned_word attempt=%s", attempt)
            continue
        if (location.city or location.region) and not _mentions_location(caption_text, location):
            logging.info("POSTCARD_CAPTION missing_location attempt=%s", attempt)
            continue
        hashtags = _finalize_postcard_hashtags(
            hashtag_raw,
            region_value_for_tags,
            city_tag,
            is_sea_scene=is_sea_scene,
            include_rubric_tag=include_rubric_tag,
            fallback_keywords=semantic_tags,
        )
        caption_with_season = _append_season_line(caption_text, season_line)
        caption_with_block = _attach_link_block(caption_with_season, link_block)
        return caption_with_block.strip(), hashtags
    logging.warning("POSTCARD_CAPTION fallback_used")
    fallback_sentence = _postcard_fallback_sentence(location, semantic_tags)
    opening = _build_postcard_opening(location)
    combined = _remove_latin_words(f"{opening} {fallback_sentence}".strip())
    caption_body = _append_season_line(combined, season_line)
    caption_with_block = _attach_link_block(caption_body, link_block)
    return caption_with_block.strip(), default_tags


async def generate_sea_caption(
    bot: Bot,
    *,
    storm_state: str,
    storm_persisting: bool,
    wave_height_m: float | None,
    wave_score: float,
    wind_class: str | None,
    wind_ms: float | None,
    wind_kmh: float | None,
    clouds_label: str,
    sunset_selected: bool,
    want_sunset: bool,
    place_hashtag: str | None,
    fact_sentence: str | None,
    now_local_iso: str | None = None,
    day_part: str | None = None,
    tz_name: str | None = None,
    job: Job | None = None,
) -> tuple[str, list[str]]:
    default_hashtags = bot._default_hashtags("sea")
    leads = [
        "А вы знали",
        "Знаете ли вы",
        "Интересный факт:",
        "Это интересно:",
        "Кстати:",
        "К слову о Балтике,",
        "Теперь вы будете знать:",
        "Небольшой факт:",
        "Поделюсь фактом:",
        "К слову",
    ]

    def fallback_caption() -> str:
        if storm_state == "strong_storm":
            opening = "Сегодня сильный шторм на море — волны гремят у самого берега."
        elif storm_state == "storm":
            if storm_persisting:
                opening = "Продолжает штормить на море — волны всё ещё бьют о берег."
            else:
                opening = "Сегодня шторм на море — волны упрямо разбиваются о кромку."
        else:
            opening = (
                "Порадую закатом над морем — побережье дышит теплом."
                if sunset_selected
                else "Порадую вас морем — тихий берег и ровный плеск."
            )
        second_para = ""
        if fact_sentence:
            lead = random.choice(leads)
            fact_text = fact_sentence.strip()
            if lead.endswith(":") or lead.endswith(","):
                second_para = f"{lead} {fact_text}"
            else:
                second_para = f"{lead}, {fact_text}"
        if second_para:
            return f"{opening}\n\n{second_para}"
        if wind_class == "very_strong":
            return f"{opening} Ветер срывает шапки на набережной."
        if wind_class == "strong":
            return f"{opening} Ветер ощутимо тянет к морю."
        if storm_state == "calm":
            return f"{opening} На побережье спокойно и хочется задержаться."
        return opening

    if not bot.openai or not bot.openai.api_key:
        raw_fallback = fallback_caption()
        cleaned = bot.strip_header(raw_fallback)
        fallback_text = cleaned.strip() if cleaned else raw_fallback.strip()
        return fallback_text, default_hashtags

    day_part_instruction = ""
    if day_part:
        day_part_instruction = (
            "Учитывай локальное время публикации: now_local_iso, day_part (morning|day|evening|night), tz_name. "
            "Пиши уместно текущему времени суток, но не упоминай время явно. "
            "Избегай приветствий и пожеланий, не соответствующих моменту (например, «пусть ваш день будет…», если уже вечер/ночь). "
            "Сохраняй естественный, дружелюбный тон. "
        )

    system_prompt = (
        "Ты редактор телеграм-канала о Балтийском море.\n\n"
        "# Форма и тон\n"
        "• Пиши ровно 2 абзаца: короткое вступление + факт.\n"
        "• Вступление — 1–2 коротких предложения, спокойно и по-дружески, без канцелярита; допустим один уместный эмодзи.\n"
        "• Факт — на тему Балтики. Начни с подводки (одну из: "
        + ", ".join(f'"{lead}"' for lead in leads)
        + ") и естественно перефразируй переданный факт без искажения смысла.\n"
        "• При перефразировании сохраняй числа/названия/термины корректными: не меняй значения, единицы, топонимы и термины.\n"
        "• Избегай штампов и клише: «дышит», «шепчет», «манит», «ласкает», «величавый», «бурлит», «одаряет», «нежный бриз» и т. п.\n"
        "• Не используй узкоспециальный жаргон и заумные слова (избегай научных терминов): «термоклин», «галоклин», «денситерм», «бенталь», «синоптическая обстановка», "
        "«фронтальная зона», «инфильтрация», «морфодинамика», и канцеляризмы.\n"
        f"• {day_part_instruction}"
        "• Проверь орфографию и пунктуацию по правилам русского языка. Абзацы разделяй одной пустой строкой.\n"
        "• ЖЁСТКОЕ ОГРАНИЧЕНИЕ: основной текст (без хэштегов и ссылки) — не более 700 символов; НИКОГДА не превышай 900 символов.\n"
    )
    wind_label = wind_class if wind_class in {"strong", "very_strong"} else "none"
    prompt_payload: dict[str, Any] = {
        "storm_state": storm_state,
        "storm_persisting": storm_persisting,
        "wave_height_m": round(wave_height_m, 2) if wave_height_m is not None else None,
        "wave_score": round(wave_score, 2),
        "wind_class": wind_label,
        "wind_ms": round(wind_ms, 1) if wind_ms is not None else None,
        "wind_kmh": round(wind_kmh, 1) if wind_kmh is not None else None,
        "clouds_label": clouds_label,
        "sunset_selected": sunset_selected,
        "want_sunset": want_sunset,
        "blog_tone": True,
    }
    if place_hashtag:
        prompt_payload["place_hashtag"] = place_hashtag
    if storm_state != "strong_storm" and fact_sentence:
        prompt_payload["fact_sentence"] = fact_sentence
    if now_local_iso:
        prompt_payload["now_local_iso"] = now_local_iso
    if day_part:
        prompt_payload["day_part"] = day_part
    if tz_name:
        prompt_payload["tz_name"] = tz_name
    payload_text = json.dumps(prompt_payload, ensure_ascii=False, separators=(",", ": "))
    user_prompt = (
        "Параметры сцены (JSON):\n"
        f"{payload_text}\n\n"
        "Собери подпись как два абзаца.\n\n"
        "Вступление: 1–2 коротких предложения, можно очень кратко («Порадую вас морем»). Допустим один эмодзи только здесь. Не перечисляй погодные параметры и числа.\n\n"
        "Интересный факт: начни одной подводкой из списка (или естественной эквивалентной, явно помечающей переход к факту) и передай смысл fact_sentence естественно и без искажений, одним предложением.\n\n"
        "Лимиты: вступление ≤220; общий ≤350 (или ≤400, если fact_sentence длинный). Без восклицательных знаков и риторических вопросов.\n\n"
        "Абзацы разделены одной пустой строкой.\n\n"
        "Если place_hashtag задан — включи его в массив hashtags.\n"
        "Не вставляй хэштеги в caption; верни их только в массиве hashtags.\n\n"
        'Верни JSON: {"caption":"<два абзаца>","hashtags":[...]}\n\n'
        "Перед возвратом проверь логичность и целостность текста: очевидный переход ко второму абзацу, связность с Балтикой/сценой, корректная пунктуация."
    )
    schema = {
        "type": "object",
        "properties": {
            "caption": {"type": "string"},
            "hashtags": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
            },
        },
        "required": ["caption", "hashtags"],
    }
    attempts = 3
    for attempt in range(1, attempts + 1):
        temperature = bot._creative_temperature()
        attempt_start = time.perf_counter()
        try:
            logging.info(
                "Запрос генерации текста для sea: модель=%s temperature=%.2f top_p=0.9 попытка %s/%s",
                "gpt-4o",
                temperature,
                attempt,
                attempts,
            )
            bot._enforce_openai_limit(job, "gpt-4o")
            response = await bot.openai.generate_json(
                model="gpt-4o",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=schema,
                temperature=temperature,
                top_p=0.9,
            )
            attempt_latency = (time.perf_counter() - attempt_start) * 1000
        except Exception:
            attempt_latency = (time.perf_counter() - attempt_start) * 1000
            logging.exception(
                "Failed to generate sea caption (attempt %s) latency_ms=%.1f",
                attempt,
                attempt_latency,
            )
            response = None
        if response:
            await bot._record_openai_usage("gpt-4o", response, job=job)
            finish_reason = "completed"
            if hasattr(response, "meta") and response.meta and isinstance(response.meta, dict):
                finish_reason = response.meta.get("finish_reason", "completed")
            logging.info(
                "SEA_RUBRIC openai_response attempt=%d latency_ms=%.1f finish_reason=%s source=llm",
                attempt,
                attempt_latency,
                finish_reason,
            )
        if not response or not isinstance(response.content, dict):
            logging.warning(
                "SEA_RUBRIC json_parse_error attempt=%d (response missing or not dict)", attempt
            )
            continue
        if "raw" in response.content and "caption" not in response.content:
            logging.warning(
                "SEA_RUBRIC json_parse_error attempt=%d (OpenAI returned raw text, not JSON)",
                attempt,
            )
            continue
        caption_raw = response.content.get("caption")
        raw_hashtags = response.content.get("hashtags")
        if not caption_raw or not isinstance(caption_raw, str):
            logging.warning(
                "SEA_RUBRIC caption_missing_or_invalid attempt=%d (caption not in response or not string)",
                attempt,
            )
            continue
        if not raw_hashtags or not isinstance(raw_hashtags, list):
            logging.warning(
                "SEA_RUBRIC hashtags_missing_or_invalid attempt=%d (using default hashtags)",
                attempt,
            )
            raw_hashtags = []
        cleaned_caption = bot.strip_header(caption_raw)
        caption = cleaned_caption.strip() if cleaned_caption else caption_raw.strip()
        caption, hashtags = bot._build_final_sea_caption(caption, raw_hashtags)
        if not caption:
            logging.warning(
                "SEA_RUBRIC empty_caption_error attempt=%d (caption empty after processing)",
                attempt,
            )
            continue
        paragraphs = [p.strip() for p in caption.split("\n\n") if p.strip()]
        paragraph_count = len(paragraphs)
        if paragraph_count != 2:
            logging.warning(
                "SEA_RUBRIC caption_structure expected 2 paragraphs, got %d", paragraph_count
            )
        if paragraph_count >= 2:
            second_para = paragraphs[1]
            has_lead = any(lead in second_para for lead in leads)
            if not has_lead:
                logging.warning("SEA_RUBRIC caption_leads no standard lead found in paragraph 2")
        caption_length = len(caption)
        if caption_length > 400:
            logging.warning("SEA_RUBRIC caption_length %d exceeds soft limit 400", caption_length)
        if paragraph_count >= 2:
            emoji_pattern = r"[\U0001F300-\U0001F9FF]|[\U0001F600-\U0001F64F]|[\U0001F680-\U0001F6FF]|[\U00002600-\U000027BF]"
            if re.search(emoji_pattern, paragraphs[1]):
                logging.warning(
                    "SEA_RUBRIC caption_emoji found in paragraph 2 (expected only in para 1)"
                )
        if bot._is_duplicate_rubric_copy("sea", "caption", caption, hashtags):
            logging.info(
                "Получен повторяющийся текст для рубрики sea, пробуем снова (%s/%s)",
                attempt,
                attempts,
            )
            continue
        logging.info("SEA_RUBRIC caption_accepted attempt=%d source=llm", attempt)
        return caption, hashtags

    logging.warning(
        "SEA_RUBRIC openai_fallback reason=caption_generation_failed source=fallback attempts=%d",
        attempts,
    )
    raw_fallback = fallback_caption()
    cleaned = bot.strip_header(raw_fallback)
    fallback_text = cleaned.strip() if cleaned else raw_fallback.strip()
    fallback_text = _sanitize_prompt_leaks(fallback_text)
    return fallback_text, default_hashtags


async def generate_sea_caption_with_timeout(
    bot: Bot,
    *,
    storm_state: str,
    storm_persisting: bool,
    wave_height_m: float | None,
    wave_score: float,
    wind_class: str | None,
    wind_ms: float | None,
    wind_kmh: float | None,
    clouds_label: str,
    sunset_selected: bool,
    want_sunset: bool,
    place_hashtag: str | None,
    fact_sentence: str | None,
    now_local_iso: str | None = None,
    day_part: str | None = None,
    tz_name: str | None = None,
    job: Job | None = None,
) -> tuple[str, list[str], dict[str, Any]]:
    import asyncio

    openai_metadata: dict[str, Any] = {
        "openai_calls_per_publish": 0,
        "duration_ms": 0,
        "tokens": 0,
        "retries": 0,
        "timeout_hit": 0,
        "fallback": 0,
    }
    openai_deadline = 90.0
    per_attempt_timeout = 60.0
    max_retries = 1
    backoff_delays = [1.5, 2.0]
    global_start = time.perf_counter()
    for retry_idx in range(max_retries + 1):
        elapsed_global = time.perf_counter() - global_start
        if elapsed_global >= openai_deadline:
            openai_metadata["timeout_hit"] = 1
            openai_metadata["fallback"] = 1
            logging.warning(
                "SEA_RUBRIC OPENAI_CALL timeout=global_deadline elapsed_ms=%.1f",
                elapsed_global * 1000,
            )
            break
        remaining_time = min(per_attempt_timeout, openai_deadline - elapsed_global)
        attempt_start = time.perf_counter()
        try:
            caption_task = asyncio.create_task(
                generate_sea_caption(
                    bot,
                    storm_state=storm_state,
                    storm_persisting=storm_persisting,
                    wave_height_m=wave_height_m,
                    wave_score=wave_score,
                    wind_class=wind_class,
                    wind_ms=wind_ms,
                    wind_kmh=wind_kmh,
                    clouds_label=clouds_label,
                    sunset_selected=sunset_selected,
                    want_sunset=want_sunset,
                    place_hashtag=place_hashtag,
                    fact_sentence=fact_sentence,
                    now_local_iso=now_local_iso,
                    day_part=day_part,
                    tz_name=tz_name,
                    job=job,
                )
            )
            caption, hashtags = await asyncio.wait_for(caption_task, timeout=remaining_time)
            attempt_duration = (time.perf_counter() - attempt_start) * 1000
            openai_metadata["openai_calls_per_publish"] += 1
            openai_metadata["duration_ms"] = round(attempt_duration, 2)
            openai_metadata["retries"] = retry_idx
            logging.info(
                "SEA_RUBRIC OPENAI_CALL success attempt=%d duration_ms=%.1f retries=%d",
                openai_metadata["openai_calls_per_publish"],
                openai_metadata["duration_ms"],
                openai_metadata["retries"],
            )
            return caption, hashtags, openai_metadata
        except TimeoutError:
            attempt_duration = (time.perf_counter() - attempt_start) * 1000
            openai_metadata["openai_calls_per_publish"] += 1
            openai_metadata["timeout_hit"] = 1
            logging.warning(
                "SEA_RUBRIC OPENAI_CALL timeout attempt=%d duration_ms=%.1f timeout_sec=%.1f",
                openai_metadata["openai_calls_per_publish"],
                attempt_duration,
                remaining_time,
            )
            if retry_idx < max_retries:
                openai_metadata["retries"] = retry_idx + 1
                backoff_delay = backoff_delays[min(retry_idx, len(backoff_delays) - 1)]
                logging.info("SEA_RUBRIC OPENAI_CALL retry backoff_sec=%.1f", backoff_delay)
                await asyncio.sleep(backoff_delay)
            else:
                openai_metadata["fallback"] = 1
                break
        except Exception:
            attempt_duration = (time.perf_counter() - attempt_start) * 1000
            openai_metadata["openai_calls_per_publish"] += 1
            logging.exception(
                "SEA_RUBRIC OPENAI_CALL error attempt=%d duration_ms=%.1f",
                openai_metadata["openai_calls_per_publish"],
                attempt_duration,
            )
            if retry_idx < max_retries:
                openai_metadata["retries"] = retry_idx + 1
                backoff_delay = backoff_delays[min(retry_idx, len(backoff_delays) - 1)]
                logging.info("SEA_RUBRIC OPENAI_CALL retry backoff_sec=%.1f", backoff_delay)
                await asyncio.sleep(backoff_delay)
            else:
                openai_metadata["fallback"] = 1
                break
    openai_metadata["fallback"] = 1
    total_duration = (time.perf_counter() - global_start) * 1000
    openai_metadata["duration_ms"] = round(total_duration, 2)
    logging.info(
        "SEA_RUBRIC OPENAI_FALLBACK total_duration_ms=%.1f calls=%d retries=%d timeout_hit=%d source=fallback",
        openai_metadata["duration_ms"],
        openai_metadata["openai_calls_per_publish"],
        openai_metadata["retries"],
        openai_metadata["timeout_hit"],
    )
    leads_fallback = [
        "А вы знали",
        "Знаете ли вы",
        "Интересный факт:",
        "Это интересно:",
        "Кстати:",
        "К слову о Балтике,",
        "Теперь вы будете знать:",
        "Небольшой факт:",
        "Поделюсь фактом:",
        "К слову",
    ]
    if storm_state == "strong_storm":
        fallback_opening = "Сегодня сильный шторм на море — волны гремят у самого берега."
    elif storm_state == "storm":
        if storm_persisting:
            fallback_opening = "Продолжает штормить на море — волны всё ещё бьют о берег."
        else:
            fallback_opening = "Сегодня шторм на море — волны упрямо разбиваются о кромку."
    else:
        fallback_opening = (
            "Порадую закатом над морем — побережье дышит теплом."
            if sunset_selected
            else "Порадую вас морем — тихий берег и ровный плеск."
        )
    second_para_fallback = ""
    if fact_sentence:
        lead_fallback = random.choice(leads_fallback)
        fact_text_fallback = fact_sentence.strip()
        if lead_fallback.endswith(":") or lead_fallback.endswith(","):
            second_para_fallback = f"{lead_fallback} {fact_text_fallback}"
        else:
            second_para_fallback = f"{lead_fallback}, {fact_text_fallback}"
    if second_para_fallback:
        fallback_caption = f"{fallback_opening}\n\n{second_para_fallback}"
    else:
        if wind_class == "very_strong":
            fallback_caption = f"{fallback_opening} Ветер срывает шапки на набережной."
        elif wind_class == "strong":
            fallback_caption = f"{fallback_opening} Ветер ощутимо тянет к морю."
        elif storm_state == "calm":
            fallback_caption = f"{fallback_opening} На побережье спокойно и хочется задержаться."
        else:
            fallback_caption = fallback_opening
    fallback_caption = _sanitize_prompt_leaks(fallback_caption)
    return fallback_caption, bot._default_hashtags("sea"), openai_metadata
