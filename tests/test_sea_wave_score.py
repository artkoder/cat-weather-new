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
    map_wave_height_to_score,
    season_match,
)


@pytest.mark.parametrize(
    "wave,expected",
    [
        (0.0, 0.0),
        (0.5, 3.0),
        (1.0, 5.0),
        (1.5, 7.0),
        (2.0, 8.5),
        (3.0, 10.0),
        (0.25, 1.5),
        (1.25, 6.0),
    ],
)
def test_map_wave_height_to_score(wave: float, expected: float) -> None:
    assert map_wave_height_to_score(wave) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "speed,expected",
    [
        (10.0, "normal"),
        (24.9, "normal"),
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
    "bucket,expected",
    [
        ("clear", {"sunny", "partly_cloudy"}),
        ("mostly_clear", {"sunny", "partly_cloudy"}),
        ("partly_cloudy", {"sunny", "partly_cloudy", "mostly_cloudy"}),
        ("mostly_cloudy", {"mostly_cloudy", "overcast"}),
        ("overcast", {"mostly_cloudy", "overcast"}),
        ("night", {"sunny", "partly_cloudy", "mostly_cloudy", "overcast"}),
    ],
)
def test_compatible_skies(bucket: str, expected: set[str]) -> None:
    assert compatible_skies(bucket) == expected


def test_compute_season_window_and_match() -> None:
    window = compute_season_window(date(2024, 6, 1))
    assert "summer" in window
    assert "spring" in window
    assert "winter" not in window
    assert season_match("summer", window) is True
    assert season_match("winter", window) is False
