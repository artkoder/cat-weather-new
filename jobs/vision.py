from __future__ import annotations

import gc
import io
import json
import logging
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil

from data_access import Asset
from jobs import Job

if TYPE_CHECKING:
    from main import Bot


async def handle_vision(bot: "Bot", job: Job) -> None:
    async with bot._vision_semaphore:
        await run_vision(bot, job)


async def run_vision(bot: "Bot", job: Job) -> None:
    def _utf16_length(text: str) -> int:
        return len(text.encode("utf-16-le")) // 2

    start_time = datetime.utcnow()
    asset_id = job.payload.get("asset_id") if job.payload else None
    asset_vision_schema = bot._asset_vision_schema
    framing_allowed_values = bot._framing_allowed_values
    framing_alias_map = bot._framing_alias_map
    season_translations = bot._season_translations
    weather_tag_translations = bot._weather_tag_translations
    assets_debug_exif = bot._assets_debug_exif
    temp_cleanup_paths: set[str] = set()
    final_cleanup_paths: set[str] = set()

    def _register_cleanup(path: str, *, temp: bool = False) -> None:
        if not path:
            return
        if temp:
            temp_cleanup_paths.add(path)
            final_cleanup_paths.discard(path)
        elif path not in temp_cleanup_paths:
            final_cleanup_paths.add(path)

    logging.info("Starting vision job %s for asset %s", job.id, asset_id)
    vision_success = False
    cleanup_storage = False
    storage_cleanup_key: str | None = None

    def _mark_vision_failure(
        message: str,
        *,
        stage: str,
        error: Exception | None = None,
    ) -> None:
        nonlocal cleanup_storage
        log_error = f"{type(error).__name__}: {error}" if error else message
        logging.error(
            "VISION_PROCESSING_FAILED job=%s asset=%s stage=%s error=%s",
            job.id,
            asset_id,
            stage,
            log_error,
        )
        if asset_id is not None:
            try:
                error_details = {
                    "status": "error",
                    "error": message,
                    "stage": stage,
                }
                if error is not None:
                    error_details["error_type"] = type(error).__name__
                    error_details["error_message"] = str(error)
                bot.data.update_asset(
                    asset_id,
                    vision_results=error_details,
                    local_path=None,
                )
            except Exception:
                logging.exception("Failed to persist vision failure for asset %s", asset_id)
        cleanup_storage = True

    try:
        if not asset_id:
            logging.warning("Vision job %s missing asset_id", job.id)
            return
        asset = bot.data.get_asset(asset_id)
        if not asset:
            logging.warning("Asset %s missing for vision", asset_id)
            return
        file_id = asset.file_id
        if not file_id:
            logging.warning("Asset %s has no file for vision", asset_id)
            return

        file_meta = {
            "file_id": asset.file_id,
            "file_unique_id": asset.file_unique_id,
            "file_name": asset.file_name,
            "mime_type": asset.mime_type,
            "duration": asset.duration,
            "file_size": asset.file_size,
            "width": asset.width,
            "height": asset.height,
        }
        storage_key = str(asset.file_ref) if asset.file_ref else None
        origin_value = (asset.origin or "").lower()
        source_value = (asset.source or "").lower() if getattr(asset, "source", None) else ""
        if storage_key and ("mobile" in {origin_value, source_value}):
            storage_cleanup_key = storage_key
        local_path = asset.local_path if asset.local_path else None
        if local_path:
            local_path = str(local_path)
            if os.path.exists(local_path):
                logging.info(
                    "Vision job %s using cached file %s for asset %s",
                    job.id,
                    local_path,
                    asset_id,
                )
                _register_cleanup(local_path, temp=False)
            else:
                logging.debug(
                    "Vision job %s cached path missing for asset %s path=%s",
                    job.id,
                    asset_id,
                    local_path,
                )
                local_path = None
        if not local_path and storage_key:
            storage_download = await bot._download_from_storage(key=storage_key)
            if storage_download:
                local_path = str(storage_download.path)
                _register_cleanup(local_path, temp=storage_download.cleanup)
                logging.info(
                    "Vision job %s fetched asset %s from storage key=%s path=%s cleanup=%s",
                    job.id,
                    asset_id,
                    storage_key,
                    local_path,
                    storage_download.cleanup,
                )
        if not local_path and not bot.dry_run:
            target_path = bot._build_local_file_path(asset_id, file_meta)
            downloaded_path = await bot._download_file(file_id, target_path)
            if downloaded_path:
                local_path = str(downloaded_path)
                _register_cleanup(local_path, temp=True)
                logging.info(
                    "Vision job %s downloaded asset %s to %s",
                    job.id,
                    asset_id,
                    local_path,
                )
        if bot.openai and not bot.openai.api_key:
            bot.openai.refresh_api_key()
        if bot.dry_run or not bot.openai or not bot.openai.api_key:
            if bot.dry_run:
                logging.info(
                    "Vision job %s skipped for asset %s: dry run enabled",
                    job.id,
                    asset_id,
                )
            else:
                logging.warning(
                    "Vision job %s skipped for asset %s: OpenAI key missing",
                    job.id,
                    asset_id,
                )
            bot.data.update_asset(
                asset_id, vision_results={"status": "skipped"}, local_path=None
            )
            return
        process = psutil.Process(os.getpid())

        def log_rss(stage: str) -> None:
            try:
                rss = process.memory_info().rss // (1024 * 1024)
                logging.info("MEM rss=%sMB stage=%s", rss, stage)
            except Exception:
                logging.debug("Failed to capture RSS at stage=%s", stage)

        if not local_path or not os.path.exists(local_path):
            raise RuntimeError(f"Local file for asset {asset_id} not found")

        system_prompt = (
            "Ты ассистент проекта Котопогода. Проанализируй изображение и верни JSON, строго соответствующий схеме asset_vision_v1. "
            "Структура включает arch_view (boolean), caption (строка на русском), objects (массив строк), is_outdoor (boolean), guess_country/guess_city (строка или null), "
            "location_confidence (число 0..1), landmarks (массив строк), tags (3-12 элементов в нижнем регистре), framing, архитектурные признаки, погодное описание, сезон и безопасность. "
            "Поле framing обязательно и принимает только close_up, medium, wide. "
            "weather_image описывает нюансы погоды и выбирается из sunny, partly_cloudy, overcast, rain, snow, fog, night. "
            "season_guess — spring, summer, autumn, winter или null. arch_style либо null, либо объект с label (название стиля на английском) и confidence (0..1). "
            "В objects перечисляй заметные элементы, цветы называй видами. В tags используй английские слова в нижнем регистре и обязательно включай погодный тег. "
            "Поле safety содержит nsfw:boolean и reason:string, где reason всегда непустая строка на русском. "
            "Дополнительно определи, есть ли море, океан, пляж или береговая линия — поле is_sea. "
            "Если is_sea=true, оцени sea_wave_score по шкале 0..10 (0 — гладь, 10 — шквал), укажи photo_sky одной из категорий sunny/partly_cloudy/mostly_cloudy/overcast/night/unknown и выставь is_sunset=true, когда заметен закат. "
            "Если моря нет, sea_wave_score ставь null, но всё равно классифицируй photo_sky по видимому небу. "
            "Обязательно укажи sky_visible=true, если на фото видно небо (даже частично), иначе sky_visible=false. Если небо не видно или неясно, ставь photo_sky=unknown. "
            "Также оцени postcard_score (1 — обычное фото, 5 — идеальная открытка) с учётом композиции, света и художественности кадра."
        )
        user_prompt = (
            "Опиши сцену, перечисли объекты, теги, достопримечательности, архитектуру и безопасность фото. Укажи кадровку (framing), "
            "наличие архитектуры крупным планом и панорам, погодный тег (weather_image), сезон и стиль архитектуры (если можно). "
            "Если локация неочевидна, ставь guess_country/guess_city = null и указывай низкую числовую уверенность. "
            "Отдельно отметь, присутствует ли море/океан/пляж (is_sea), оцени sea_wave_score 0..10, классифицируй небо photo_sky и укажи is_sunset для закатных кадров. "
            "Обязательно определи, видно ли небо на фото (sky_visible), и если небо не видно или неясно, используй photo_sky=unknown. "
            "Также оцени postcard_score (1..5) и возвращай значение даже при низкой художественности кадра."
        )
        bot._enforce_openai_limit(job, "gpt-4o-mini")
        logging.info(
            "Vision job %s classifying asset %s using gpt-4o-mini from %s",
            job.id,
            asset_id,
            local_path,
        )
        try:
            response = await bot.openai.classify_image(
                model="gpt-4o-mini",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_path=Path(local_path),
                schema=asset_vision_schema,
            )
        except Exception as exc:
            _mark_vision_failure(
                "Vision model request failed",
                stage="request",
                error=exc,
            )
            return
        logging.info(
            "Vision job %s received response from model for asset %s",
            job.id,
            asset_id,
        )
        log_rss("after_openai")
        gc.collect()
        if response is None:
            logging.warning(
                "Vision job %s for asset %s returned no response",
                job.id,
                asset_id,
            )
            bot.data.update_asset(asset_id, vision_results={"status": "skipped"})
            cleanup_storage = True
            gc.collect()
            return
        result = response.content
        if not isinstance(result, dict):
            _mark_vision_failure(
                "Invalid response from vision model",
                stage="parse_result",
            )
            return
        framing_raw = result.get("framing")
        framing: str | None = None
        if isinstance(framing_raw, str):
            framing = re.sub(r"[\s\-]+", "_", framing_raw.strip().lower()) or None
        elif framing_raw is not None:
            framing = (
                re.sub(
                    r"[\s\-]+",
                    "_",
                    str(framing_raw).strip().lower(),
                )
                or None
            )
        if not framing:
            _mark_vision_failure(
                "Invalid response from vision model: missing framing",
                stage="parse_result",
            )
            return
        if framing not in framing_allowed_values:
            alias = framing_alias_map.get(framing)
            if alias in framing_allowed_values:
                framing = alias
            else:
                _mark_vision_failure(
                    "Invalid response from vision model: unknown framing",
                    stage="parse_result",
                )
                return
        architecture_close_up_raw = result.get("architecture_close_up")
        architecture_close_up = (
            bool(architecture_close_up_raw)
            if isinstance(architecture_close_up_raw, bool)
            else str(architecture_close_up_raw).strip().lower() in {"1", "true", "yes", "да"}
        )
        architecture_wide_raw = result.get("architecture_wide")
        architecture_wide = (
            bool(architecture_wide_raw)
            if isinstance(architecture_wide_raw, bool)
            else str(architecture_wide_raw).strip().lower() in {"1", "true", "yes", "да"}
        )
        weather_image_raw = result.get("weather_image")
        weather_image: str | None = None
        if isinstance(weather_image_raw, str):
            weather_image = (
                re.sub(
                    r"[\s\-]+",
                    "_",
                    weather_image_raw.strip().lower(),
                )
                or None
            )
        elif weather_image_raw is not None:
            weather_image = (
                re.sub(
                    r"[\s\-]+",
                    "_",
                    str(weather_image_raw).strip().lower(),
                )
                or None
            )
        if not weather_image:
            _mark_vision_failure(
                "Invalid response from vision model: missing weather_image",
                stage="parse_result",
            )
            return
        normalized_weather = bot._normalize_weather_enum(weather_image)
        if not normalized_weather:
            _mark_vision_failure(
                "Invalid response from vision model: unknown weather_image",
                stage="parse_result",
            )
            return
        weather_image = normalized_weather
        season_guess_raw = result.get("season_guess")
        if isinstance(season_guess_raw, str):
            season_guess = bot._normalize_season(season_guess_raw)
        elif season_guess_raw is None:
            season_guess = None
        else:
            season_guess = bot._normalize_season(str(season_guess_raw))
        arch_style_raw = result.get("arch_style")
        arch_style: dict[str, Any] | None
        if isinstance(arch_style_raw, dict):
            label_raw = arch_style_raw.get("label")
            if isinstance(label_raw, str):
                label = label_raw.strip()
            elif label_raw is None:
                label = ""
            else:
                label = str(label_raw).strip()
            if label:
                confidence_value: float | None = None
                confidence_raw = arch_style_raw.get("confidence")
                if isinstance(confidence_raw, (int, float)):
                    confidence_value = float(confidence_raw)
                elif isinstance(confidence_raw, str):
                    try:
                        confidence_value = float(confidence_raw.strip())
                    except ValueError:
                        confidence_value = None
                if confidence_value is not None:
                    confidence_value = min(max(confidence_value, 0.0), 1.0)
                arch_style = {"label": label, "confidence": confidence_value}
            else:
                arch_style = None
        elif isinstance(arch_style_raw, str):
            label = arch_style_raw.strip()
            arch_style = {"label": label, "confidence": None} if label else None
        else:
            arch_style = None
        usage = response.usage if isinstance(response.usage, dict) else {}
        caption = str(result.get("caption", "")).strip()
        guess_country_raw = result.get("guess_country")
        guess_city_raw = result.get("guess_city")
        if isinstance(guess_country_raw, str):
            guess_country = guess_country_raw.strip() or None
        elif guess_country_raw is None:
            guess_country = None
        else:
            guess_country = str(guess_country_raw).strip() or None
        if isinstance(guess_city_raw, str):
            guess_city = guess_city_raw.strip() or None
        elif guess_city_raw is None:
            guess_city = None
        else:
            guess_city = str(guess_city_raw).strip() or None
        arch_view_raw = result.get("arch_view")
        arch_view = (
            bool(arch_view_raw)
            if isinstance(arch_view_raw, bool)
            else str(arch_view_raw).strip().lower() in {"1", "true", "yes", "да"}
        )
        is_outdoor_raw = result.get("is_outdoor")
        is_outdoor = (
            bool(is_outdoor_raw)
            if isinstance(is_outdoor_raw, bool)
            else str(is_outdoor_raw).strip().lower() in {"1", "true", "yes", "да"}
        )
        is_sea_raw = result.get("is_sea")
        is_sea = (
            bool(is_sea_raw)
            if isinstance(is_sea_raw, bool)
            else str(is_sea_raw).strip().lower() in {"1", "true", "yes", "да"}
        )
        photo_sky_raw = result.get("photo_sky")
        photo_sky_result: str | None = None
        if isinstance(photo_sky_raw, str):
            photo_sky_result = photo_sky_raw.strip() or None
        elif photo_sky_raw is not None:
            photo_sky_result = str(photo_sky_raw).strip() or None
        is_sunset_raw = result.get("is_sunset")
        is_sunset = (
            bool(is_sunset_raw)
            if isinstance(is_sunset_raw, bool)
            else str(is_sunset_raw).strip().lower() in {"1", "true", "yes", "да"}
        )
        sky_visible_raw = result.get("sky_visible")
        sky_visible = (
            bool(sky_visible_raw)
            if isinstance(sky_visible_raw, bool)
            else str(sky_visible_raw).strip().lower() in {"1", "true", "yes", "да"}
        )
        postcard_score_value = Asset._normalize_postcard_score(result.get("postcard_score"))
        raw_objects = result.get("objects")
        objects: list[str] = []
        if isinstance(raw_objects, list):
            seen_objects: set[str] = set()
            for item in raw_objects:
                text = str(item).strip()
                if not text or text in seen_objects:
                    continue
                seen_objects.add(text)
                objects.append(text)
        raw_landmarks = result.get("landmarks")
        landmarks: list[str] = []
        if isinstance(raw_landmarks, list):
            seen_landmarks: set[str] = set()
            for item in raw_landmarks:
                text = str(item).strip()
                normalized = text.lower()
                if not text or normalized in seen_landmarks:
                    continue
                seen_landmarks.add(normalized)
                landmarks.append(text)
        raw_tags = result.get("tags")
        tags: list[str] = []
        if isinstance(raw_tags, list):
            seen_tags: set[str] = set()
            for tag in raw_tags:
                text = str(tag).strip().lower()
                if not text or text in seen_tags:
                    continue
                seen_tags.add(text)
                tags.append(text)
        if weather_image and weather_image not in tags:
            tags.append(weather_image)
        if architecture_close_up and "architecture_close_up" not in tags:
            tags.append("architecture_close_up")
        if architecture_wide and "architecture_wide" not in tags:
            tags.append("architecture_wide")
        if postcard_score_value is not None and postcard_score_value >= 3:
            if "postcard" not in tags:
                tags.append("postcard")
        await bot._maybe_append_marine_tag(asset, tags)
        metadata_dict = asset.metadata if isinstance(asset.metadata, dict) else {}
        capture_datetime: datetime | None = None
        capture_time_display: str | None = None
        timestamp_keys = [
            "exif_datetime_best",
            "exif_datetime_original",
            "exif_datetime",
            "exif_datetime_digitized",
        ]
        for ts_key in timestamp_keys:
            raw_value = metadata_dict.get(ts_key)
            if raw_value is None:
                continue
            if isinstance(raw_value, (list, tuple)):
                candidate = next(
                    (str(item).strip() for item in raw_value if str(item).strip()),
                    "",
                )
            else:
                candidate = str(raw_value).strip()
            if not candidate or candidate.lower() == "none":
                continue
            try:
                parsed_dt = datetime.strptime(candidate, "%Y:%m:%d %H:%M:%S")
            except ValueError:
                parsed_dt = None
            if parsed_dt:
                capture_datetime = parsed_dt
                capture_time_display = parsed_dt.strftime("%Y-%m-%d %H:%M")
            else:
                capture_time_display = candidate
            break
        exif_month = bot._extract_month_from_metadata(metadata_dict)
        if exif_month is None and local_path and os.path.exists(local_path):
            exif_month = bot._extract_exif_month(local_path)
        season_from_exif = bot._season_from_month(exif_month)
        season_final = bot._normalize_season(season_from_exif or season_guess)
        season_final_display = season_translations.get(season_final) if season_final else None
        fallback_weather = bot._normalize_weather_enum(weather_image)
        model_weather: str | None = None
        model_weather_display: str | None = None
        for tag_value in tags:
            normalized_tag = bot._normalize_weather_enum(tag_value)
            if not normalized_tag:
                continue
            translated = weather_tag_translations.get(normalized_tag)
            if translated:
                model_weather = normalized_tag
                model_weather_display = translated
                break
        if not model_weather and fallback_weather:
            model_weather = fallback_weather
            model_weather_display = weather_tag_translations.get(fallback_weather)
        metadata_weather = bot._extract_weather_enum_from_metadata(metadata_dict)
        weather_final = metadata_weather or model_weather or fallback_weather
        weather_final = bot._normalize_weather_enum(weather_final)
        weather_final_display = bot._weather_display(weather_final)
        if not weather_final_display and weather_final:
            weather_final_display = weather_final
        if weather_final and weather_final not in tags:
            tags.append(weather_final)
        photo_weather = weather_final or model_weather
        photo_weather_display: str | None = weather_final_display
        if not photo_weather_display and model_weather_display:
            photo_weather_display = model_weather_display
        if not photo_weather_display and photo_weather:
            photo_weather_display = photo_weather
        supabase_meta = {
            "asset_id": asset_id,
            "channel_id": asset.channel_id,
            "architecture_close_up": architecture_close_up,
            "architecture_wide": architecture_wide,
            "weather_final": photo_weather,
            "weather_final_display": photo_weather_display,
            "season_final": season_final,
            "season_final_display": season_final_display,
        }
        if arch_style:
            supabase_meta["arch_style"] = arch_style
        success, payload, error = await bot.supabase.insert_token_usage(
            bot="kotopogoda",
            model="gpt-4o-mini",
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            request_id=response.request_id,
            endpoint=usage.get("endpoint") or "/v1/responses",
            meta=supabase_meta,
        )
        log_context = {
            "log_token_usage": payload,
            "weather_final": photo_weather,
            "season_final": season_final,
        }
        if arch_style:
            log_context["arch_style"] = arch_style
        if success:
            logging.info("Supabase token usage insert succeeded", extra=log_context)
        else:
            if error == "disabled":
                logging.debug(
                    "Supabase client disabled; token usage skipped", extra=log_context
                )
            elif error:
                logging.error(
                    "Supabase token usage insert failed: %s", error, extra=log_context
                )
            else:
                logging.error("Supabase token usage insert failed", extra=log_context)
        safety_raw = result.get("safety")
        nsfw_flag = False
        safety_reason: str | None = None
        if isinstance(safety_raw, dict):
            nsfw_value = safety_raw.get("nsfw")
            if isinstance(nsfw_value, bool):
                nsfw_flag = nsfw_value
            elif nsfw_value is not None:
                nsfw_flag = str(nsfw_value).strip().lower() in {"1", "true", "yes", "да"}
            reason_raw = safety_raw.get("reason")
            if isinstance(reason_raw, str):
                safety_reason = reason_raw.strip() or None
            elif reason_raw is not None:
                safety_reason = str(reason_raw).strip() or None
        if not safety_reason:
            safety_reason = "обнаружен чувствительный контент" if nsfw_flag else "безопасно"
        location_confidence_raw = result.get("location_confidence")
        location_confidence: float | None = None
        if isinstance(location_confidence_raw, (int, float)):
            location_confidence = float(location_confidence_raw)
        elif isinstance(location_confidence_raw, str):
            try:
                location_confidence = float(location_confidence_raw.strip())
            except ValueError:
                location_confidence = None
        if location_confidence is not None:
            location_confidence = min(max(location_confidence, 0.0), 1.0)
        if not caption:
            _mark_vision_failure(
                "Invalid response from vision model",
                stage="parse_result",
            )
            return
        category = bot._derive_primary_scene(caption, tags)
        # Force sea category when is_sea=true, regardless of heuristics
        if is_sea:
            category = "sea"
        rubric_id = bot._resolve_rubric_id_for_category(category)
        flower_varieties: list[str] = []
        normalized_tag_set = {tag.lower() for tag in tags if tag}
        if normalized_tag_set.intersection({"flowers", "flower"}):
            flower_varieties = [obj for obj in objects if obj]

        sea_wave_score_data: dict[str, Any] | None = None
        is_sea_asset = is_sea or (
            category == "sea" or normalized_tag_set.intersection({"sea", "ocean"})
        )
        if (
            is_sea_asset
            and bot.openai
            and bot.openai.api_key
            and local_path
            and os.path.exists(local_path)
        ):
            try:
                sea_wave_prompt = (
                    "Analyze the sea/ocean in this image and return a JSON with sea wave intensity score. "
                    "Score criteria: 0 = calm/flat, 1-3 = small waves, 4-6 = moderate waves/storm, "
                    "7-8 = strong storm (many whitecaps), 9-10 = very strong storm (massive whitecaps, foam, spray everywhere). "
                    'Evaluate only the sea/ocean visible. Return: {"sea_wave_score": 0-10 (integer), "confidence": 0.0-1.0 (float)}'
                )
                sea_wave_schema = {
                    "type": "object",
                    "properties": {
                        "sea_wave_score": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 10,
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                    "required": ["sea_wave_score", "confidence"],
                }
                logging.info(
                    "Vision job %s calling 4o-mini for sea_wave_score on asset %s",
                    job.id,
                    asset_id,
                )
                bot._enforce_openai_limit(job, "gpt-4o-mini")
                sea_wave_response = await bot.openai.classify_image(
                    model="gpt-4o-mini",
                    system_prompt="You are analyzing sea/ocean conditions. Return only the requested JSON.",
                    user_prompt=sea_wave_prompt,
                    image_path=Path(local_path),
                    schema=sea_wave_schema,
                    schema_name="sea_wave_score_v1",
                )
                if sea_wave_response and isinstance(sea_wave_response.content, dict):
                    wave_score_raw = sea_wave_response.content.get("sea_wave_score")
                    confidence_raw = sea_wave_response.content.get("confidence")
                    if isinstance(wave_score_raw, int) and isinstance(
                        confidence_raw, (int, float)
                    ):
                        wave_score = max(0, min(10, wave_score_raw))
                        confidence = max(0.0, min(1.0, float(confidence_raw)))
                        sea_wave_score_data = {
                            "value": wave_score,
                            "confidence": confidence,
                            "model": "gpt-4o-mini",
                        }
                        logging.info(
                            "Sea wave score for asset %s: score=%s conf=%.2f",
                            asset_id,
                            wave_score,
                            confidence,
                        )
                        await bot._record_openai_usage(
                            "gpt-4o-mini",
                            sea_wave_response,
                            job=job,
                            record_supabase=True,
                        )
            except Exception:
                logging.exception(
                    "Failed to get sea_wave_score for asset %s, continuing without it",
                    asset_id,
                )

        location_parts: list[str] = []
        existing_lower: set[str] = set()
        if asset.city:
            location_parts.append(asset.city)
            existing_lower.add(asset.city.lower())
        if asset.country and asset.country.lower() not in existing_lower:
            location_parts.append(asset.country)
            existing_lower.add(asset.country.lower())
        if guess_city and guess_city.lower() not in existing_lower:
            location_parts.insert(0, guess_city)
            existing_lower.add(guess_city.lower())
        if guess_country and guess_country.lower() not in existing_lower:
            location_parts.append(guess_country)
            existing_lower.add(guess_country.lower())
        exif_coords: tuple[float, float] | None = None
        if asset.latitude is not None and asset.longitude is not None:
            try:
                exif_coords = (float(asset.latitude), float(asset.longitude))
            except (TypeError, ValueError):
                exif_coords = None
        if not exif_coords and local_path and os.path.exists(local_path):
            exif_retry = bot._extract_gps(local_path)
            if exif_retry:
                exif_coords = exif_retry
        confidence_display: str | None = None
        if location_confidence is not None and math.isfinite(location_confidence):
            confidence_percent = int(round(location_confidence * 100))
            confidence_percent = max(0, min(100, confidence_percent))
            confidence_display = f"{confidence_percent}%"
        caption_lines = [f"Распознано: {caption}"]
        if exif_coords:
            exif_lat, exif_lon = exif_coords
            exif_address: dict[str, Any] | None = None
            try:
                exif_address = await bot._reverse_geocode(exif_lat, exif_lon)
            except Exception:
                logging.exception(
                    "Reverse geocode failed for EXIF coordinates of asset %s",
                    asset_id,
                )
                exif_address = {}

            fallback_text: str | None = None
            if isinstance(exif_address, dict):
                fallback_value = exif_address.get("fallback")
                if isinstance(fallback_value, str):
                    fallback_text = fallback_value

            formatted_exif, dedupe_values, has_osm_components = (
                bot._format_exif_address_caption(
                    exif_address if isinstance(exif_address, dict) else None,
                    exif_lat,
                    exif_lon,
                )
            )

            for value in dedupe_values:
                existing_lower.add(value)

            if formatted_exif and formatted_exif not in caption_lines:
                caption_lines.append(formatted_exif)

            if fallback_text and not has_osm_components and fallback_text not in caption_lines:
                caption_lines.append(fallback_text)
        if location_parts:
            location_line = ", ".join(location_parts)
            if confidence_display:
                location_line += f" (уверенность: {confidence_display})"
            caption_lines.append("Локация: " + location_line)
        elif confidence_display:
            caption_lines.append(f"Уверенность в локации: {confidence_display}")
        if photo_weather_display:
            caption_lines.append(f"Обстановка: {photo_weather_display}")
        caption_lines.append(f"На улице: {'да' if is_outdoor else 'нет'}")
        caption_lines.append(f"Архитектура: {'да' if arch_view else 'нет'}")
        season_caption_display = season_final_display or "неизвестно"
        weather_caption_display = photo_weather_display or "неизвестно"
        capture_display_value = capture_time_display or "неизвестно"
        caption_lines.append(f"Время съёмки: {capture_display_value}")
        caption_lines.append(f"Погода: {weather_caption_display}")
        caption_lines.append(f"Сезон: {season_caption_display}")
        if postcard_score_value is not None and postcard_score_value >= 3:
            caption_lines.append(f"Открыточность: {postcard_score_value}/5")
        if arch_style and arch_style.get("label"):
            confidence_value = arch_style.get("confidence")
            style_line = f"Стиль: {arch_style['label']}"
            confidence_note: str
            if isinstance(confidence_value, (int, float)) and math.isfinite(confidence_value):
                confidence_float = float(confidence_value)
                confidence_pct = int(round(confidence_float * 100))
                confidence_pct = max(0, min(100, confidence_pct))
                if confidence_float >= 0.4:
                    confidence_note = f"(≈{confidence_pct}%)"
                else:
                    confidence_note = f"(низкая уверенность ≈{confidence_pct}%)"
            else:
                confidence_note = "(уверенность неизвестна)"
            caption_lines.append(f"{style_line} {confidence_note}".strip())
        if landmarks:
            caption_lines.append("Ориентиры: " + ", ".join(landmarks))
        if flower_varieties:
            caption_lines.append("Цветы: " + ", ".join(flower_varieties))
        flower_set = set(flower_varieties)
        remaining_objects = [obj for obj in objects if obj not in flower_set]
        if remaining_objects:
            caption_lines.append("Объекты: " + ", ".join(remaining_objects))
        if tags:
            caption_lines.append("Теги: " + ", ".join(tags))
        if sea_wave_score_data:
            wave_val = sea_wave_score_data["value"]
            wave_conf = sea_wave_score_data["confidence"]
            caption_lines.append(f"Волнение моря: {wave_val}/10 (conf={wave_conf:.2f})")
        if nsfw_flag:
            caption_lines.append(
                "⚠️ Чувствительный контент: " + (safety_reason or "потенциально NSFW")
            )

        attribution_line = "Адрес: OSM/Nominatim"
        if attribution_line not in caption_lines:
            caption_lines.append(attribution_line)

        caption_text = "\n".join(line for line in caption_lines if line)
        caption_entities: list[dict[str, Any]] | None = None
        if caption_text:
            caption_entities = [
                {
                    "type": "expandable_blockquote",
                    "offset": 0,
                    "length": _utf16_length(caption_text),
                }
            ]
        location_log_parts: list[str] = []
        if guess_city:
            location_log_parts.append(guess_city)
        if guess_country and (not guess_city or guess_country.lower() != guess_city.lower()):
            location_log_parts.append(guess_country)
        location_log = ", ".join(location_log_parts) or "-"
        confidence_log = (
            f"{location_confidence:.3f}"
            if location_confidence is not None and math.isfinite(location_confidence)
            else "-"
        )
        request_id = response.request_id if response else None
        logging.info(
            "VISION_RESULT asset=%s model=%s request_id=%s description=%s location=%s confidence=%s caption_len=%s weather=%s season=%s style=%s",
            asset_id,
            "gpt-4o-mini",
            request_id or "-",
            caption,
            location_log,
            confidence_log,
            len(caption_text),
            photo_weather or "-",
            season_final or "-",
            arch_style["label"] if arch_style and arch_style.get("label") else "-",
        )
        result_payload = {
            "status": "ok",
            "provider": "gpt-4o-mini",
            "arch_view": arch_view,
            "caption": caption,
            "objects": objects,
            "is_outdoor": is_outdoor,
            "guess_country": guess_country,
            "guess_city": guess_city,
            "location_confidence": location_confidence,
            "landmarks": landmarks,
            "tags": tags,
            "framing": framing,
            "architecture_close_up": architecture_close_up,
            "architecture_wide": architecture_wide,
            "weather_image": weather_image,
            "season_guess": season_guess,
            "season_final": season_final,
            "season_final_display": season_final_display,
            "arch_style": arch_style,
            "safety": {"nsfw": nsfw_flag, "reason": safety_reason},
            "category": category,
            "photo_weather": photo_weather,
            "photo_weather_display": photo_weather_display,
            "weather_final": photo_weather,
            "weather_final_display": photo_weather_display,
            "flower_varieties": flower_varieties,
            "is_sea": is_sea,
            "photo_sky": photo_sky_result,
            "sky_visible": sky_visible,
            "is_sunset": is_sunset,
            "postcard_score": postcard_score_value,
        }
        if rubric_id is not None:
            result_payload["rubric_id"] = rubric_id
        if sea_wave_score_data:
            result_payload["sea_wave_score"] = sea_wave_score_data
        logging.info(
            "Vision job %s classified asset %s: scene=%s, arch=%s, tags=%s, weather_tag=%s",
            job.id,
            asset_id,
            caption,
            arch_view,
            ", ".join(tags) if tags else "-",
            photo_weather or "-",
        )
        await bot._record_openai_usage(
            "gpt-4o-mini",
            response,
            job=job,
            record_supabase=False,
        )
        logging.info(
            "OpenAI request_id=%s usage in/out/total=%s/%s/%s",
            request_id or "-",
            response.prompt_tokens if response and response.prompt_tokens is not None else "-",
            (
                response.completion_tokens
                if response and response.completion_tokens is not None
                else "-"
            ),
            response.total_tokens if response and response.total_tokens is not None else "-",
        )
        delete_original_after_post = False
        if asset.kind == "photo":
            method_used = "copyMessage"
            copy_payload: dict[str, Any] = {
                "chat_id": asset.channel_id,
                "from_chat_id": asset.channel_id,
                "message_id": asset.message_id,
                "caption": caption_text or None,
            }
            if caption_entities:
                copy_payload["caption_entities"] = caption_entities
            resp = await bot.api_request("copyMessage", copy_payload)
            if resp.get("ok"):
                delete_original_after_post = True
            else:
                logging.error(
                    "Vision job %s failed to copy message for asset %s: %s",
                    job.id,
                    asset_id,
                    resp,
                )
                fallback_method = "sendPhoto" if asset.kind == "photo" else "sendDocument"
                file_field = "photo" if fallback_method == "sendPhoto" else "document"
                fallback_payload: dict[str, Any] = {
                    "chat_id": asset.channel_id,
                    file_field: file_id,
                    "caption": caption_text or None,
                }
                if caption_entities:
                    fallback_payload["caption_entities"] = caption_entities
                resp = await bot.api_request(fallback_method, fallback_payload)
                method_used = fallback_method
                if resp.get("ok"):
                    delete_original_after_post = fallback_method == "sendPhoto"
                else:
                    logging.error(
                        "Vision job %s failed to publish result for asset %s via %s: %s",
                        job.id,
                        asset_id,
                        fallback_method,
                        resp,
                    )
                    raise RuntimeError(f"Failed to publish vision result: {resp}")
        elif bot._is_convertible_image_document(asset):
            if not local_path or not os.path.exists(local_path):
                if not bot.dry_run:
                    target_path = bot._build_local_file_path(asset_id, file_meta)
                    downloaded_path = await bot._download_file(file_id, target_path)
                    if downloaded_path:
                        local_path = str(downloaded_path)
                        cleanup_paths.append(local_path)
            if not local_path or not os.path.exists(local_path):
                raise RuntimeError("Unable to load asset for conversion")

            resp: dict[str, Any] | None = None
            publish_mode = "original"
            log_rss("before_sendPhoto")
            try:
                resp, publish_mode = await bot._publish_as_photo(
                    asset.channel_id,
                    local_path,
                    caption_text or None,
                    caption_entities=caption_entities,
                )
            finally:
                log_rss("after_sendPhoto")
                gc.collect()
            method_used = "sendPhoto"
            delete_original_after_post = True
            logging.info(
                "Vision job %s published document asset %s via sendPhoto (%s)",
                job.id,
                asset_id,
                publish_mode,
            )
            if resp and resp.get("ok"):
                result_payload = resp.get("result")
                photo_sizes = None
                if isinstance(result_payload, dict):
                    photo_sizes = result_payload.get("photo")
                photo_meta = bot._extract_photo_file_meta(photo_sizes)
                if photo_meta and photo_meta.get("file_id"):
                    bot.data.update_asset(
                        asset_id,
                        kind="photo",
                        file_meta=photo_meta,
                        metadata={"original_document_file_id": file_id},
                    )
                    new_file_id = photo_meta.get("file_id")
                    asset.payload["kind"] = "photo"
                    if new_file_id is not None:
                        asset.payload["file_id"] = new_file_id
                    file_unique = photo_meta.get("file_unique_id")
                    if file_unique is not None:
                        asset.payload["file_unique_id"] = file_unique
                    mime_type = photo_meta.get("mime_type")
                    if mime_type is not None:
                        asset.payload["mime_type"] = mime_type
                    file_size = photo_meta.get("file_size")
                    if file_size is not None:
                        asset.payload["file_size"] = file_size
                    width_value = photo_meta.get("width")
                    height_value = photo_meta.get("height")
                    asset.width = Asset._to_int(width_value)
                    asset.height = Asset._to_int(height_value)
                else:
                    logging.warning(
                        "Vision job %s missing photo metadata in response for asset %s: %s",
                        job.id,
                        asset_id,
                        resp,
                    )
            if not resp or not resp.get("ok"):
                logging.error(
                    "Vision job %s failed to publish converted photo for asset %s: %s",
                    job.id,
                    asset_id,
                    resp,
                )
                fallback_doc_payload: dict[str, Any] = {
                    "chat_id": asset.channel_id,
                    "document": file_id,
                    "caption": caption_text or None,
                }
                if caption_entities:
                    fallback_doc_payload["caption_entities"] = caption_entities
                resp = await bot.api_request("sendDocument", fallback_doc_payload)
                method_used = "sendDocument"
                delete_original_after_post = False
                if not resp.get("ok"):
                    logging.error(
                        "Vision job %s failed to publish result for asset %s via sendDocument: %s",
                        job.id,
                        asset_id,
                        resp,
                    )
                    raise RuntimeError(f"Failed to publish vision result: {resp}")
        else:
            document_payload: dict[str, Any] = {
                "chat_id": asset.channel_id,
                "document": file_id,
                "caption": caption_text or None,
            }
            if caption_entities:
                document_payload["caption_entities"] = caption_entities
            resp = await bot.api_request("sendDocument", document_payload)
            method_used = "sendDocument"
            if not resp.get("ok"):
                logging.error(
                    "Vision job %s failed to publish result for asset %s via sendDocument: %s",
                    job.id,
                    asset_id,
                    resp,
                )
                raise RuntimeError(f"Failed to publish vision result: {resp}")
        new_mid = resp.get("result", {}).get("message_id") if resp.get("result") else None
        logging.info(
            "Vision job %s posted classification for asset %s via %s: message_id=%s",
            job.id,
            asset_id,
            method_used,
            new_mid,
        )
        weather_display_log = (
            weather_final_display or photo_weather_display or photo_weather or "-"
        )
        weather_source_log: str | None
        if metadata_weather:
            weather_source_log = "metadata"
        elif model_weather:
            weather_source_log = "model"
        elif fallback_weather:
            weather_source_log = "fallback"
        else:
            weather_source_log = None
        if weather_source_log:
            weather_display_log = f"{weather_display_log} ({weather_source_log})"
        season_display_log = season_final_display or season_final or "-"
        arch_style_label = arch_style.get("label") if isinstance(arch_style, dict) else None
        arch_style_confidence = (
            arch_style.get("confidence") if isinstance(arch_style, dict) else None
        )
        arch_confidence_log = (
            f"{float(arch_style_confidence):.3f}"
            if isinstance(arch_style_confidence, (int, float))
            else "-"
        )
        logging.info(
            "VISION: framing=%s weather=%s season=%s arch_style=%s arch_confidence=%s",
            framing,
            weather_display_log,
            season_display_log,
            arch_style_label or "-",
            arch_confidence_log,
        )
        asset_update_kwargs = {
            "recognized_message_id": new_mid,
            "vision_results": result_payload,
            "vision_category": category,
            "vision_arch_view": "yes" if arch_view else "",
            "vision_photo_weather": photo_weather,
            "vision_confidence": location_confidence,
            "vision_flower_varieties": flower_varieties,
            "vision_caption": caption_text,
            "local_path": None,
        }
        if postcard_score_value is not None:
            asset_update_kwargs["postcard_score"] = postcard_score_value
        if rubric_id is not None:
            asset_update_kwargs["rubric_id"] = rubric_id
        bot.data.update_asset(asset_id, **asset_update_kwargs)
        if tags and any(t in {"sunset", "закат", "golden hour"} for t in tags):
            try:
                bot.data.update_asset_categories_merge(asset_id, ["закат"])
            except Exception:
                logging.exception("Failed to add закат category to asset %s", asset_id)
        if assets_debug_exif and not bot.dry_run and new_mid:
            try:
                debug_path: str | None = (
                    local_path if local_path and os.path.exists(local_path) else None
                )
                if not debug_path:
                    target_path = bot._build_local_file_path(asset_id, file_meta)
                    downloaded_path = await bot._download_file(file_id, target_path)
                    if downloaded_path:
                        debug_path = str(downloaded_path)
                        _register_cleanup(debug_path, temp=True)
                if debug_path and os.path.exists(debug_path):
                    exif_payload = bot._extract_exif_full(debug_path)
                    exif_json = json.dumps(exif_payload, ensure_ascii=False, indent=2)
                    message_text = (
                        f"EXIF (raw)\n```json\n{exif_json}\n```" if exif_json else "EXIF (raw)"
                    )
                    exif_bytes = exif_json.encode("utf-8") if exif_json else b""
                    if len(message_text) <= 3500:
                        await bot.api_request(
                            "sendMessage",
                            {
                                "chat_id": asset.channel_id,
                                "text": message_text,
                                "reply_to_message_id": new_mid,
                            },
                        )
                    else:
                        buffer = io.BytesIO(exif_bytes)
                        await bot.api_request(
                            "sendDocument",
                            {
                                "chat_id": asset.channel_id,
                                "caption": "EXIF (raw)",
                                "reply_to_message_id": new_mid,
                            },
                            files={"document": ("exif.json", buffer.getvalue())},
                        )
            except Exception:
                logging.exception("Failed to publish EXIF debug for asset %s", asset_id)
        if delete_original_after_post and not bot.dry_run and new_mid and asset.message_id:
            logging.info(
                "Vision job %s deleting original document message %s for asset %s",
                job.id,
                asset.message_id,
                asset_id,
            )
            delete_resp = await bot.api_request(
                "deleteMessage",
                {"chat_id": asset.channel_id, "message_id": asset.message_id},
            )
            if not delete_resp.get("ok"):
                logging.error(
                    "Vision job %s failed to delete original message %s for asset %s: %s",
                    job.id,
                    asset.message_id,
                    asset_id,
                    delete_resp,
                )
        cleanup_storage = True
        vision_success = True
    finally:
        cleanup_targets: set[str] = set()
        cleanup_targets.update(temp_cleanup_paths)
        cleanup_targets.update(final_cleanup_paths)
        for path in cleanup_targets:
            logging.info(
                "Vision job %s cleanup removing %s for asset %s",
                job.id,
                path,
                asset_id,
            )
            bot._remove_file(path)
        if cleanup_storage and storage_cleanup_key:
            await bot._delete_storage_entry(key=storage_cleanup_key)
        duration = (datetime.utcnow() - start_time).total_seconds()
        logging.info(
            "Vision job %s for asset %s completed in %.2fs",
            job.id,
            asset_id,
            duration,
        )
