from __future__ import annotations

import json
import logging
import random
import re
import time
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from data_access import Asset
from openai_client import OpenAIClient

if TYPE_CHECKING:  # pragma: no cover
    from jobs import Job
    from main import Bot

POSTCARD_PREFIX = "Порадую вас красивым видом "
POSTCARD_DEFAULT_HASHTAGS = ["#котопогода", "#БалтийскоеМоре", "#открыточныйвид"]


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


def _finalize_postcard_hashtags(
    candidate_tags: Iterable[str],
    region_hashtag: str | None,
    city_hashtag: str | None,
) -> list[str]:
    prepared = list(candidate_tags)
    if region_hashtag:
        prepared.append(region_hashtag)
    if city_hashtag:
        prepared.append(city_hashtag)
    prepared.extend(POSTCARD_DEFAULT_HASHTAGS)
    finalized = _deduplicate_hashtags(prepared)
    if len(finalized) < 3:
        extras = [tag for tag in POSTCARD_DEFAULT_HASHTAGS if tag not in finalized]
        finalized.extend(extras)
        finalized = _deduplicate_hashtags(finalized)
    if len(finalized) > 5:
        finalized = finalized[:5]
    return finalized


def _postcard_fallback_caption(location_label: str, semantic_tags: Sequence[str]) -> str:
    detail = semantic_tags[0] if semantic_tags else "мягкого света"
    return (
        f"{POSTCARD_PREFIX}{location_label} — кадр, который хочется пересматривать снова. "
        f"Тут {detail} делает настроение особенным."
    )


def _sanitize_and_prefix(text: str) -> str:
    cleaned = _sanitize_prompt_leaks(text.strip())
    if not cleaned.startswith(POSTCARD_PREFIX):
        cleaned = f"{POSTCARD_PREFIX}{cleaned.lstrip()}"
    return cleaned


async def generate_postcard_caption(
    openai: OpenAIClient | None,
    asset: Asset,
    region_hashtag: str,
    stopwords: Sequence[str],
) -> tuple[str, list[str]]:
    location = _resolve_location(asset)
    semantic_tags = _collect_semantic_tags(asset)
    region_tag = _normalize_hashtag_candidate(region_hashtag)
    city_tag = _normalize_city_hashtag(location.city)
    banned_words = [str(word or "").strip() for word in stopwords if str(word or "").strip()]
    context_payload: dict[str, Any] = {
        "location": location.display,
        "city": location.city,
        "region": location.region,
        "country": location.country,
        "tags": semantic_tags,
        "postcard_score": asset.postcard_score,
    }
    extra_hint = asset.vision_results or {}
    if isinstance(extra_hint, dict) and extra_hint.get("weather_final_display"):
        context_payload["weather"] = extra_hint.get("weather_final_display")
    payload_text = json.dumps(context_payload, ensure_ascii=False, separators=(",", ": "))
    stopwords_text = ", ".join(banned_words) if banned_words else ""
    system_prompt_lines = [
        "Ты — голос проекта «Котопогода» и описываешь рубрику «Открыточный вид».",
        "Пиши образно, но конкретно: максимум 200–300 символов, 1–2 коротких предложения.",
        "Тон — спокойный и дружелюбный, никаких канцеляризмов или рекламных клише.",
        f"Строго начинай текст с фразы «{POSTCARD_PREFIX}».",
        "Обязательно упоминай переданный населённый пункт или регион.",
        "Не добавляй хэштеги и ссылки в самом тексте — только описания.",
    ]
    if stopwords_text:
        system_prompt_lines.append(f"Запрещены стоп-слова и клише: {stopwords_text}.")
    system_prompt_lines.append("Соблюдай орфографию и избегай повторов.")
    system_prompt = "\n".join(system_prompt_lines)
    user_prompt = (
        "Контекст (JSON):\n"
        f"{payload_text}\n\n"
        "Собери подпись о сцене.\n"
        f"1. Начни ровно с «{POSTCARD_PREFIX}» и продолжи описание вида.\n"
        "2. Укажи, где находится зритель (город, регион или страна).\n"
        "3. Добавь 1–2 детали из подсказок (море, закат, городские мотивы и т. д.).\n"
        "4. Подчеркни настроение кадра без штампов.\n"
        "5. Заверши лёгким приглашением насладиться моментом.\n\n"
        "Отдельно верни 3–5 хэштегов с символом #. Обязательно включи региональный хештег,"
        f" например {region_tag or region_hashtag}. Если знаешь город — добавь его хэштег.\n\n"
        'Формат ответа: {"caption": "<текст>", "hashtags": ["#…"]}.'
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
    normalized_region_tag = region_tag or _normalize_hashtag_candidate(region_hashtag)
    default_tags = _finalize_postcard_hashtags([], normalized_region_tag, city_tag)
    if not openai or not getattr(openai, "api_key", None):
        fallback = _postcard_fallback_caption(location.display, semantic_tags)
        caption = _sanitize_and_prefix(fallback)
        caption_with_block = f"{caption}\n\n{_build_link_block()}"
        return caption_with_block, default_tags

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
        caption_raw = response.content.get("caption")
        hashtag_raw = response.content.get("hashtags")
        if not isinstance(caption_raw, str):
            logging.warning("POSTCARD_CAPTION missing_caption attempt=%s", attempt)
            continue
        if not isinstance(hashtag_raw, list):
            hashtag_raw = []
        caption = _sanitize_and_prefix(caption_raw)
        if location.display.casefold() not in caption.casefold():
            logging.info("POSTCARD_CAPTION missing_location attempt=%s", attempt)
            continue
        if _contains_banned_word(caption, banned_words):
            logging.info("POSTCARD_CAPTION banned_word attempt=%s", attempt)
            continue
        hashtags = _finalize_postcard_hashtags(hashtag_raw, normalized_region_tag, city_tag)
        caption_with_block = f"{caption}\n\n{_build_link_block()}"
        return caption_with_block.strip(), hashtags

    logging.warning("POSTCARD_CAPTION fallback_used")
    fallback = _postcard_fallback_caption(location.display, semantic_tags)
    caption = _sanitize_and_prefix(fallback)
    caption_with_block = f"{caption}\n\n{_build_link_block()}"
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
