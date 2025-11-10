"""Wave scoring utilities for sea photo selection."""

from __future__ import annotations

import math
import re
from typing import Any


def wave_m_to_score(meters: float | int | str | None) -> int:
    """Map wave height in meters to an integer score on a 0..10 scale."""

    if meters is None:
        return 0
    try:
        value = float(meters)
    except (TypeError, ValueError):
        return 0
    if value <= 0.0:
        return 0
    step = 0.2
    epsilon = 1e-9
    score = math.floor((value + epsilon) / step)
    return max(0, min(10, score))


def _to_float(value: Any) -> float | None:
    """Convert value to float safely."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def parse_wave_score_from_vision(
    vision_json: dict[str, Any] | None,
) -> tuple[float | None, float | None]:
    """Extract wave score and confidence from various vision JSON layouts.

    Supports multiple structured JSON formats:
    - weather.sea.wave_score with weather.sea.confidence
    - sea_state.score with sea_state.confidence
    - sea_wave_score.value with sea_wave_score.confidence (legacy dict)
    - sea_wave_score as scalar (legacy)

    Also supports textual fallbacks:
    - «Волнение моря: X/10» (without confidence)
    - «Волнение моря: X/10 (conf=Y)» (with confidence)

    Returns: (wave_score, confidence) tuple, both can be None
    """
    if not vision_json:
        return None, None

    wave_score: float | None = None
    wave_conf: float | None = None

    # Try weather.sea layout
    if "weather" in vision_json and isinstance(vision_json["weather"], dict):
        weather = vision_json["weather"]
        if "sea" in weather and isinstance(weather["sea"], dict):
            sea_data = weather["sea"]
            wave_score = _to_float(sea_data.get("wave_score"))
            wave_conf = _to_float(sea_data.get("confidence"))

    # Try sea_state layout
    if wave_score is None and "sea_state" in vision_json:
        sea_state = vision_json["sea_state"]
        if isinstance(sea_state, dict):
            wave_score = _to_float(sea_state.get("score"))
            wave_conf = _to_float(sea_state.get("confidence"))

    # Try legacy sea_wave_score dict layout
    if wave_score is None and "sea_wave_score" in vision_json:
        raw_wave = vision_json["sea_wave_score"]
        if isinstance(raw_wave, dict):
            wave_score = _to_float(raw_wave.get("value"))
            wave_conf = _to_float(raw_wave.get("confidence"))
        else:
            wave_score = _to_float(raw_wave)

    # Try textual fallback with optional confidence
    # Check both result_text and caption fields for backward compatibility
    text_to_parse: str | None = None
    if wave_score is None and "result_text" in vision_json:
        text_to_parse = str(vision_json["result_text"])
    elif wave_score is None and "caption" in vision_json:
        text_to_parse = str(vision_json["caption"])
    
    if text_to_parse:
        # Try pattern with confidence: «Волнение моря: X/10 (conf=Y)»
        match = re.search(
            r"Волнение\s+моря:\s*(\d+(?:\.\d+)?)\s*/\s*10\s*\(\s*conf\s*=\s*(\d+(?:\.\d+)?)\s*\)",
            text_to_parse,
        )
        if match:
            wave_score = _to_float(match.group(1))
            wave_conf = _to_float(match.group(2))
        else:
            # Fallback to pattern without confidence: «Волнение моря: X/10»
            match = re.search(r"Волнение\s+моря:\s*(\d+(?:\.\d+)?)\s*/\s*10", text_to_parse)
            if match:
                wave_score = _to_float(match.group(1))

    return wave_score, wave_conf


def parse_sky_bucket_from_vision(vision_json: dict[str, Any] | None) -> str | None:
    """Extract sky bucket code from vision JSON.

    Supports multiple layouts:
    - weather.sky.bucket
    - sky.bucket
    - sky_bucket (direct)

    Returns: sky bucket string (e.g., "clear", "partly_cloudy", "overcast") or None
    """
    if not vision_json:
        return None

    # Try weather.sky.bucket layout
    if "weather" in vision_json and isinstance(vision_json["weather"], dict):
        weather = vision_json["weather"]
        if "sky" in weather and isinstance(weather["sky"], dict):
            bucket = weather["sky"].get("bucket")
            if bucket:
                return str(bucket)

    # Try direct sky_bucket field
    if "sky_bucket" in vision_json:
        bucket = vision_json["sky_bucket"]
        if bucket:
            return str(bucket)

    # Try sky.bucket layout
    if "sky" in vision_json and isinstance(vision_json["sky"], dict):
        bucket = vision_json["sky"].get("bucket")
        if bucket:
            return str(bucket)

    return None
