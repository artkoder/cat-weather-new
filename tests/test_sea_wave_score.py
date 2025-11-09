import os
import sys
from datetime import date

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import (
    bucket_clouds,
    classify_wind_kph,
    compatible_skies,
    compute_season_window,
    season_match,
)
from utils_wave import wave_m_to_score


@pytest.mark.parametrize(
    "wave,expected",
    [
        (None, 0),
        (0.0, 0),
        (0.05, 0),
        (0.2, 1),
        (0.21, 1),
        (0.39, 1),
        (0.4, 2),
        (0.8, 4),
        (1.0, 5),
        (1.5, 7),
        (1.6, 8),
        (2.0, 10),
        (2.8, 10),
        ("bad", 0),
    ],
)
def test_wave_m_to_score(wave: float | None, expected: int) -> None:
    assert wave_m_to_score(wave) == expected


@pytest.mark.parametrize(
    "speed,expected",
    [
        (10.0, None),
        (24.9, None),
        (25.0, "strong"),
        (34.9, "strong"),
        (35.0, "very_strong"),
        (48.0, "very_strong"),
        (None, None),
    ],
)
def test_classify_wind_kph(speed: float | None, expected: str | None) -> None:
    assert classify_wind_kph(speed) == expected


@pytest.mark.parametrize(
    "cloud_pct,expected",
    [
        (5.0, "clear"),
        (30.0, "mostly_clear"),
        (50.0, "partly_cloudy"),
        (70.0, "mostly_cloudy"),
        (95.0, "overcast"),
        (None, None),
    ],
)
def test_bucket_clouds(cloud_pct: float | None, expected: str | None) -> None:
    assert bucket_clouds(cloud_pct) == expected


@pytest.mark.parametrize(
    "bucket,daypart,expected",
    [
        (
            "clear",
            "day",
            {
                "day:sunny",
                "day:mostly_clear",
                "day:partly_cloudy",
                "evening:sunny",
                "evening:mostly_clear",
            },
        ),
        (
            "mostly_clear",
            "day",
            {
                "day:sunny",
                "day:mostly_clear",
                "day:partly_cloudy",
                "evening:sunny",
                "evening:mostly_clear",
            },
        ),
        (
            "partly_cloudy",
            "day",
            {
                "day:sunny",
                "day:mostly_clear",
                "day:partly_cloudy",
                "day:mostly_cloudy",
            },
        ),
        (
            "mostly_cloudy",
            "day",
            {"day:mostly_cloudy", "day:overcast"},
        ),
        (
            "overcast",
            "day",
            {"day:mostly_cloudy", "day:overcast"},
        ),
        (
            "unknown",
            "night",
            {
                "night:sunny",
                "night:mostly_clear",
                "night:partly_cloudy",
                "night:mostly_cloudy",
                "night:overcast",
            },
        ),
    ],
)
def test_compatible_skies(bucket: str, daypart: str, expected: set[str]) -> None:
    result = {sky.token() for sky in compatible_skies(bucket, daypart)}
    assert result == expected


def test_compute_season_window_and_match() -> None:
    window = compute_season_window(date(2024, 6, 1))
    assert "summer" in window
    assert "spring" in window
    assert "winter" not in window
    assert season_match("summer", window) is True
    assert season_match("winter", window) is False
