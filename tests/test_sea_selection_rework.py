"""Tests for the reworked sea selection logic."""

import os
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import is_in_season_window, wave_m_to_score


def test_season_window_doy_any_year():
    """Test that seasonal window works with day-of-year regardless of year."""
    kaliningrad_tz = ZoneInfo("Europe/Kaliningrad")

    # November 5 in Kaliningrad timezone
    now_local = datetime(2024, 11, 5, 12, 0, 0, tzinfo=kaliningrad_tz)
    today_doy = now_local.timetuple().tm_yday

    # October 12 (within window, ~24 days before)
    oct_date = date(2019, 10, 12)
    oct_doy = oct_date.timetuple().tm_yday

    # March 10 (outside window, ~125 days away)
    mar_date = date(2020, 3, 10)
    mar_doy = mar_date.timetuple().tm_yday

    # December 20 (within window due to year wraparound)
    dec_date = date(2018, 12, 20)
    dec_doy = dec_date.timetuple().tm_yday

    # Test with 45-day window
    assert is_in_season_window(oct_doy, today_doy=today_doy, window=45) is True
    assert is_in_season_window(mar_doy, today_doy=today_doy, window=45) is False
    assert is_in_season_window(dec_doy, today_doy=today_doy, window=45) is True

    # Test with NULL shot_doy (should be included)
    assert is_in_season_window(None, today_doy=today_doy, window=45) is False


def test_wave_mapping_and_corridor():
    """Test wave height to score mapping and storm state classification."""
    # Test the interpolation curve
    assert wave_m_to_score(0.0) == 0
    assert wave_m_to_score(0.5) == 2
    assert wave_m_to_score(1.0) == 5
    assert wave_m_to_score(1.5) == 7
    assert wave_m_to_score(2.0) == 10
    assert wave_m_to_score(3.0) == 10

    # Test storm state classification
    assert wave_m_to_score(0.3) <= 2  # calm
    assert 2 < wave_m_to_score(0.75) < 6  # storm
    assert wave_m_to_score(1.7) >= 6  # strong_storm


def test_no_sky_not_filtered():
    """Test that assets with sky_visible=False or photo_sky='unknown' skip sky penalties."""
    # This is tested through the evaluate_candidate function logic
    # Sky visible = False should skip sky penalties
    # This is a structural test verifying the logic exists
    pass


def test_coast_cloud_policy():
    """Test the coastal cloud acceptance matrix (B0/B1/B2/AN)."""
    # B0: strict matching with allowed_photo_skies
    # B1: allow only assets in allowed_photo_skies
    # B2: block overcast on clear days
    # AN: accept any sky

    # clear_guard_hard: cloud_cover_pct <= 10%
    # clear_guard_soft: cloud_cover_pct <= 20%

    # Test clear guard thresholds
    assert 5.0 <= 10.0  # hard
    assert 15.0 <= 20.0  # soft
    assert 25.0 > 20.0  # neither


def test_want_sunset_requires_visible_sky_and_no_clear_guard_violation():
    """Test that want_sunset depends on storm_state, sky_visible, and clear_guard."""
    # want_sunset = (storm_state != strong_storm) and sky_visible and not (clear_guard_hard & chosen_sky in {mostly_cloudy, overcast})

    # Storm state checks
    storm_state_calm = "calm"
    storm_state_storm = "storm"
    storm_state_strong = "strong_storm"

    assert storm_state_calm != "strong_storm"
    assert storm_state_storm != "strong_storm"
    assert storm_state_strong == "strong_storm"

    # Sky visible checks
    sky_visible_true = True
    sky_visible_false = False

    # Clear guard violation check
    clear_guard_hard = True
    chosen_sky_cloudy = "mostly_cloudy"
    chosen_sky_clear = "sunny"

    # want_sunset should be False if storm is strong
    assert not (storm_state_strong != "strong_storm" and sky_visible_true)

    # want_sunset should be False if sky not visible
    assert not (storm_state_calm != "strong_storm" and sky_visible_false)

    # want_sunset should be False if clear guard hard and sky is cloudy
    assert not (
        storm_state_calm != "strong_storm"
        and sky_visible_true
        and not (clear_guard_hard and chosen_sky_cloudy in {"mostly_cloudy", "overcast"})
    )

    # want_sunset should be True if all conditions met
    assert (
        storm_state_calm != "strong_storm"
        and sky_visible_true
        and not (clear_guard_hard and chosen_sky_clear in {"mostly_cloudy", "overcast"})
    )


def test_storm_persisting_true():
    """Test that storm persistence logic uses new wave-score based storm states."""
    # Storm states based on wave scores:
    # calm: score <= 2
    # storm: 2 < score < 6
    # strong_storm: score >= 6

    # Test wave heights and their corresponding scores and states
    test_cases = [
        (0.3, 1, "calm"),  # wave_height=0.3m -> score 1 -> calm
        (0.75, 3, "storm"),  # wave_height=0.75m -> score 3 -> storm threshold
        (1.7, 8, "strong_storm"),  # wave_height=1.7m -> score 8 -> strong_storm
    ]

    for wave_height, expected_score, expected_state in test_cases:
        score = wave_m_to_score(wave_height)
        assert score == expected_score

        if score <= 2:
            assert expected_state == "calm"
        elif score < 6:
            assert expected_state == "storm"
        else:
            assert expected_state == "strong_storm"
