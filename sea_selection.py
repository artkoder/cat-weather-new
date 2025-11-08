"""Helpers for sea photo selection scoring and configuration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedSky:
    """Normalized representation of sky conditions for selection logic."""

    daypart: str
    weather_tag: str

    def token(self) -> str:
        return f"{self.daypart}:{self.weather_tag}"

    def __str__(self) -> str:  # pragma: no cover - used for logging readability
        return self.token()

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
    outside_corridor_multiplier: float = 0.0
    calm_wave_penalty: float = 0.0
    false_sky_penalty: float = 0.0
    strict_unknown_sky_penalty: float = 0.0
    required_sky_penalty: float = 0.0
    required_sky_unknown_penalty: float = 0.0
    visible_sky_bonus: float = 0.0


STAGE_CONFIGS: Mapping[str, StageConfig] = {
    "B0": StageConfig(
        name="B0",
        wave_tolerance=1.0,
        wave_penalty_rate=1.3,
        require_visible_sky=True,
        allow_unknown_sky=False,
        allow_false_sky=False,
        require_allowed_sky=True,
        season_required=True,
        season_penalty=2.4,
        allow_missing_wave=False,
        unknown_sky_penalty=0.6,
        mismatch_penalty=2.8,
        match_bonus=1.6,
        calm_wave_cap=3.0,
        unknown_wave_penalty=0.8,
        season_mismatch_extra=0.6,
        outside_corridor_multiplier=3.2,
        calm_wave_penalty=2.2,
        false_sky_penalty=3.5,
        strict_unknown_sky_penalty=1.4,
        required_sky_penalty=3.0,
        required_sky_unknown_penalty=2.2,
        visible_sky_bonus=0.5,
    ),
    "B1": StageConfig(
        name="B1",
        wave_tolerance=1.8,
        wave_penalty_rate=1.05,
        require_visible_sky=False,
        allow_unknown_sky=True,
        allow_false_sky=False,
        require_allowed_sky=False,
        season_required=True,
        season_penalty=1.8,
        allow_missing_wave=False,
        unknown_sky_penalty=0.7,
        mismatch_penalty=2.4,
        match_bonus=1.25,
        calm_wave_cap=4.2,
        unknown_wave_penalty=0.7,
        season_mismatch_extra=0.5,
        outside_corridor_multiplier=1.8,
        calm_wave_penalty=1.5,
        false_sky_penalty=2.6,
        strict_unknown_sky_penalty=0.8,
        required_sky_penalty=2.2,
        required_sky_unknown_penalty=1.4,
        visible_sky_bonus=0.3,
    ),
    "B2": StageConfig(
        name="B2",
        wave_tolerance=2.5,
        wave_penalty_rate=0.95,
        require_visible_sky=False,
        allow_unknown_sky=True,
        allow_false_sky=False,
        require_allowed_sky=False,
        season_required=False,
        season_penalty=1.0,
        allow_missing_wave=True,
        unknown_sky_penalty=0.6,
        mismatch_penalty=1.9,
        match_bonus=1.05,
        calm_wave_cap=5.0,
        unknown_wave_penalty=0.5,
        season_mismatch_extra=0.7,
        outside_corridor_multiplier=1.2,
        calm_wave_penalty=1.0,
        false_sky_penalty=1.6,
        strict_unknown_sky_penalty=0.5,
        required_sky_penalty=1.5,
        required_sky_unknown_penalty=1.1,
        visible_sky_bonus=0.15,
    ),
    "AN": StageConfig(
        name="AN",
        wave_tolerance=3.5,
        wave_penalty_rate=0.75,
        require_visible_sky=False,
        allow_unknown_sky=True,
        allow_false_sky=True,
        require_allowed_sky=False,
        season_required=False,
        season_penalty=0.6,
        allow_missing_wave=True,
        unknown_sky_penalty=0.5,
        mismatch_penalty=1.3,
        match_bonus=0.85,
        calm_wave_cap=5.5,
        unknown_wave_penalty=0.45,
        season_mismatch_extra=0.5,
        outside_corridor_multiplier=0.8,
        calm_wave_penalty=0.6,
        false_sky_penalty=0.0,
        strict_unknown_sky_penalty=0.0,
        required_sky_penalty=0.8,
        required_sky_unknown_penalty=0.4,
        visible_sky_bonus=0.0,
    ),
}


def calc_wave_penalty(
    photo_wave: float | int | None, target_wave: float | int, stage: StageConfig
) -> float:
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


def sky_similarity(photo_sky: NormalizedSky | None, allowed: set[NormalizedSky]) -> str:
    if not photo_sky or photo_sky.weather_tag in {"", "unknown"}:
        return "none"
    if photo_sky in allowed:
        return "match"
    for candidate in allowed:
        if candidate.daypart != photo_sky.daypart:
            continue
        neighbours = _ALLOWED_NEIGHBOURS.get(candidate.weather_tag, set())
        if photo_sky.weather_tag in neighbours:
            return "close"
    if (
        photo_sky.daypart == "evening"
        and photo_sky.weather_tag in {"sunny", "mostly_clear"}
        and any(
            candidate.daypart == "day"
            and candidate.weather_tag in {"sunny", "mostly_clear"}
            for candidate in allowed
        )
    ):
        return "match"
    return "mismatch"
