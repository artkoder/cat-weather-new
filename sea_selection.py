"""Helpers for sea photo selection scoring and configuration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

_SKY_POSITIVE_KEYWORDS = {
    "sky",
    "clear_sky",
    "clearsky",
    "sunny",
    "clouds",
    "cloud",
    "partly_cloudy",
    "partlycloudy",
    "blue_sky",
    "blue_sky_background",
}

_SKY_NEGATIVE_KEYWORDS = {
    "indoor",
    "interior",
    "inside",
    "room",
    "studio",
    "ceiling",
}


def _normalize_token(token: str) -> str:
    text = token.strip().lower()
    if not text:
        return ""
    return text.replace(" ", "_")


def infer_sky_visible(tags: Iterable[str]) -> bool | None:
    """Infer whether the sky is visible based on recognition tags.

    Returns ``True`` if a sky-related tag is present, ``False`` if an
    explicit indoor/interior tag is found, and ``None`` otherwise.
    """

    positive = False
    negative = False
    for raw in tags:
        token = _normalize_token(raw)
        if not token:
            continue
        if token in _SKY_NEGATIVE_KEYWORDS:
            negative = True
        if token in _SKY_POSITIVE_KEYWORDS:
            positive = True
    if positive:
        return True
    if negative:
        return False
    return None


@dataclass(frozen=True)
class StageConfig:
    name: str
    wave_tolerance: float
    wave_penalty_rate: float
    require_visible_sky: bool
    allow_unknown_sky: bool
    allow_false_sky: bool
    require_allowed_sky: bool
    season_required: bool
    season_penalty: float
    allow_missing_wave: bool
    unknown_sky_penalty: float
    mismatch_penalty: float
    match_bonus: float
    calm_wave_cap: float | None = None
    unknown_wave_penalty: float = 0.5
    season_mismatch_extra: float = 0.0


STAGE_CONFIGS: Mapping[str, StageConfig] = {
    "B0": StageConfig(
        name="B0",
        wave_tolerance=1.0,
        wave_penalty_rate=1.2,
        require_visible_sky=True,
        allow_unknown_sky=False,
        allow_false_sky=False,
        require_allowed_sky=True,
        season_required=True,
        season_penalty=2.5,
        allow_missing_wave=False,
        unknown_sky_penalty=0.0,
        mismatch_penalty=3.0,
        match_bonus=1.5,
    ),
    "B1": StageConfig(
        name="B1",
        wave_tolerance=1.8,
        wave_penalty_rate=1.0,
        require_visible_sky=False,
        allow_unknown_sky=True,
        allow_false_sky=False,
        require_allowed_sky=False,
        season_required=True,
        season_penalty=2.0,
        allow_missing_wave=False,
        unknown_sky_penalty=0.8,
        mismatch_penalty=2.5,
        match_bonus=1.2,
        unknown_wave_penalty=0.7,
    ),
    "B2": StageConfig(
        name="B2",
        wave_tolerance=2.5,
        wave_penalty_rate=0.9,
        require_visible_sky=False,
        allow_unknown_sky=True,
        allow_false_sky=False,
        require_allowed_sky=False,
        season_required=False,
        season_penalty=1.2,
        allow_missing_wave=True,
        unknown_sky_penalty=0.6,
        mismatch_penalty=1.8,
        match_bonus=1.0,
        unknown_wave_penalty=0.5,
        season_mismatch_extra=0.8,
    ),
    "AN": StageConfig(
        name="AN",
        wave_tolerance=3.5,
        wave_penalty_rate=0.7,
        require_visible_sky=False,
        allow_unknown_sky=True,
        allow_false_sky=True,
        require_allowed_sky=False,
        season_required=False,
        season_penalty=0.4,
        allow_missing_wave=True,
        unknown_sky_penalty=0.4,
        mismatch_penalty=1.2,
        match_bonus=0.8,
        calm_wave_cap=4.0,
        unknown_wave_penalty=0.4,
        season_mismatch_extra=0.4,
    ),
}


def calc_wave_penalty(photo_wave: float | None, target_wave: float, stage: StageConfig) -> float:
    """Compute penalty for wave mismatch for a given stage."""

    if photo_wave is None:
        if stage.allow_missing_wave:
            return stage.unknown_wave_penalty
        return stage.unknown_wave_penalty + stage.wave_penalty_rate

    delta = abs(photo_wave - target_wave)
    overshoot = max(0.0, delta - stage.wave_tolerance)
    return overshoot * stage.wave_penalty_rate


_ALLOWED_NEIGHBOURS = {
    "sunny": {"mostly_clear", "partly_cloudy"},
    "mostly_clear": {"sunny", "partly_cloudy"},
    "partly_cloudy": {"sunny", "mostly_clear", "mostly_cloudy"},
    "mostly_cloudy": {"partly_cloudy", "overcast"},
    "overcast": {"mostly_cloudy"},
}


def sky_similarity(photo_sky: str | None, allowed: set[str]) -> str:
    if not photo_sky:
        return "none"
    normalized = _normalize_token(photo_sky)
    if normalized in allowed:
        return "match"
    for candidate in allowed:
        neighbours = _ALLOWED_NEIGHBOURS.get(candidate, set())
        if normalized in neighbours:
            return "close"
    return "mismatch"
