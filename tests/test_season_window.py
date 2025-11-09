"""Tests for the seasonal window filter (day-of-year based)."""

import os
import sys
from datetime import date, datetime, timedelta

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import is_in_season_window


def test_season_window_same_day():
    """Photo taken on the same day should always match."""
    today_doy = 150  # May 30 in non-leap year
    assert is_in_season_window(150, today_doy=today_doy, window=45) is True


def test_season_window_within_range():
    """Photo within ±45 days should match."""
    today_doy = 150  # May 30
    # 45 days before: day 105 (April 15)
    assert is_in_season_window(105, today_doy=today_doy, window=45) is True
    # 45 days after: day 195 (July 14)
    assert is_in_season_window(195, today_doy=today_doy, window=45) is True
    # 30 days before
    assert is_in_season_window(120, today_doy=today_doy, window=45) is True
    # 30 days after
    assert is_in_season_window(180, today_doy=today_doy, window=45) is True


def test_season_window_outside_range():
    """Photo outside the ±45 day window should not match."""
    today_doy = 150  # May 30
    # 46 days before: day 104
    assert is_in_season_window(104, today_doy=today_doy, window=45) is False
    # 46 days after: day 196
    assert is_in_season_window(196, today_doy=today_doy, window=45) is False
    # Much earlier: day 50 (Feb 19)
    assert is_in_season_window(50, today_doy=today_doy, window=45) is False
    # Much later: day 250 (Sept 7)
    assert is_in_season_window(250, today_doy=today_doy, window=45) is False


def test_season_window_cross_year_jan_to_dec():
    """Window crossing New Year: early January should match late December."""
    today_doy = 5  # January 5
    # December 20 of previous year: day 354 (365 - 11)
    # Distance from 5 to 354: forward = (354-5) % 365 = 349, backward = (5-354) % 365 = 16
    # min(349, 16) = 16, which is <= 45
    assert is_in_season_window(354, today_doy=today_doy, window=45) is True
    # December 25: day 359
    # Distance: forward = 354, backward = 11
    assert is_in_season_window(359, today_doy=today_doy, window=45) is True
    # November 20: day 324 (in non-leap year)
    # Distance from 5: forward = 319, backward = 46
    # This is outside the window (min distance is 46)
    assert is_in_season_window(324, today_doy=today_doy, window=45) is False


def test_season_window_cross_year_dec_to_jan():
    """Window crossing New Year: late December should match early January."""
    today_doy = 355  # December 21
    # January 5 of next year: day 5
    # Distance from 355 to 5: forward = (5-355) % 365 = 15, backward = (355-5) % 365 = 350
    # min(15, 350) = 15, which is <= 45
    assert is_in_season_window(5, today_doy=today_doy, window=45) is True
    # January 15: day 15
    # Distance: forward = 25, backward = 340
    assert is_in_season_window(15, today_doy=today_doy, window=45) is True
    # February 10: day 41
    # Distance: forward = 51, backward = 314
    # min is 51, which is > 45
    assert is_in_season_window(41, today_doy=today_doy, window=45) is False


def test_season_window_wraparound_early_year():
    """Test wraparound for early year dates."""
    today_doy = 10  # January 10
    # December 1: day 335
    # Distance: forward = 325, backward = 40
    assert is_in_season_window(335, today_doy=today_doy, window=45) is True
    # November 20: day 324
    # Distance: forward = 314, backward = 51
    # min is 51, outside window
    assert is_in_season_window(324, today_doy=today_doy, window=45) is False


def test_season_window_wraparound_late_year():
    """Test wraparound for late year dates."""
    today_doy = 360  # December 26
    # January 20: day 20
    # Distance from 360: forward = 25, backward = 340
    assert is_in_season_window(20, today_doy=today_doy, window=45) is True
    # February 15: day 46
    # Distance: forward = 51, backward = 314
    # min is 51, outside window
    assert is_in_season_window(46, today_doy=today_doy, window=45) is False


def test_season_window_leap_day_normalization():
    """Leap day (366) should be treated as day 365 for non-leap year comparison."""
    today_doy = 60  # March 1 in non-leap year
    # Shot on leap day (Feb 29): day 366 → normalized to 365
    # Distance from 60 to 365: forward = 305, backward = 60
    # min is 60, which is > 45
    assert is_in_season_window(366, today_doy=today_doy, window=45) is False

    # Today is Dec 20 (day 354)
    today_doy = 354
    # Shot on leap day: day 366 → 365
    # Distance: forward = 11, backward = 354
    assert is_in_season_window(366, today_doy=today_doy, window=45) is True


def test_season_window_none_shot_doy():
    """Photos without shot_doy should return False."""
    today_doy = 150
    assert is_in_season_window(None, today_doy=today_doy, window=45) is False


def test_season_window_invalid_shot_doy():
    """Invalid shot_doy values should return False."""
    today_doy = 150
    assert is_in_season_window(0, today_doy=today_doy, window=45) is False
    assert is_in_season_window(367, today_doy=today_doy, window=45) is False
    assert is_in_season_window(-10, today_doy=today_doy, window=45) is False


def test_season_window_boundary_cases():
    """Test exact boundary cases."""
    today_doy = 100
    # Exactly 45 days before: day 55
    assert is_in_season_window(55, today_doy=today_doy, window=45) is True
    # Exactly 45 days after: day 145
    assert is_in_season_window(145, today_doy=today_doy, window=45) is True
    # 46 days before: day 54
    assert is_in_season_window(54, today_doy=today_doy, window=45) is False
    # 46 days after: day 146
    assert is_in_season_window(146, today_doy=today_doy, window=45) is False


def test_season_window_custom_window_size():
    """Test with different window sizes."""
    today_doy = 100
    # With window=10
    assert is_in_season_window(90, today_doy=today_doy, window=10) is True
    assert is_in_season_window(110, today_doy=today_doy, window=10) is True
    assert is_in_season_window(89, today_doy=today_doy, window=10) is False
    assert is_in_season_window(111, today_doy=today_doy, window=10) is False

    # With window=90
    assert is_in_season_window(10, today_doy=today_doy, window=90) is True
    assert is_in_season_window(190, today_doy=today_doy, window=90) is True
    assert is_in_season_window(9, today_doy=today_doy, window=90) is False
    assert is_in_season_window(191, today_doy=today_doy, window=90) is False


def test_season_window_opposite_side_of_year():
    """Photos on the opposite side of the year should not match."""
    # Today is June 1 (day ~152)
    today_doy = 152
    # December 1 (day 335) is ~183 days away
    # Distance: forward = 183, backward = 182
    assert is_in_season_window(335, today_doy=today_doy, window=45) is False

    # Today is December 1 (day 335)
    today_doy = 335
    # June 1 (day 152) is ~183 days away
    assert is_in_season_window(152, today_doy=today_doy, window=45) is False


def test_season_window_real_dates():
    """Test with real date scenarios."""
    # November 5 (typically day 309 in non-leap year)
    nov_5 = date(2023, 11, 5)
    today_doy = nov_5.timetuple().tm_yday  # 309

    # October 12 (typically day 285)
    oct_12 = date(2023, 10, 12)
    shot_doy_oct = oct_12.timetuple().tm_yday  # 285
    # Distance: 309 - 285 = 24 days
    assert is_in_season_window(shot_doy_oct, today_doy=today_doy, window=45) is True

    # March 10 (typically day 69)
    mar_10 = date(2023, 3, 10)
    shot_doy_mar = mar_10.timetuple().tm_yday  # 69
    # Distance from 309: forward = 125, backward = 240
    # min is 125, which is > 45
    assert is_in_season_window(shot_doy_mar, today_doy=today_doy, window=45) is False


def test_season_window_new_year_transition_real_dates():
    """Test New Year transition with real dates."""
    # January 5 (day 5)
    jan_5 = date(2024, 1, 5)
    today_doy = jan_5.timetuple().tm_yday  # 5

    # December 20 of previous year (day 355 in leap year, 354 in non-leap)
    # For 2023 (non-leap): day 354
    dec_20 = date(2023, 12, 20)
    shot_doy_dec = dec_20.timetuple().tm_yday  # 354
    # Distance from 5: forward = 349, backward = 16
    assert is_in_season_window(shot_doy_dec, today_doy=today_doy, window=45) is True


def test_season_window_string_shot_doy():
    """Test that string shot_doy is handled correctly."""
    today_doy = 150
    # Valid string that can be converted to int
    assert is_in_season_window("150", today_doy=today_doy, window=45) is True  # type: ignore
    assert is_in_season_window("140", today_doy=today_doy, window=45) is True  # type: ignore
    # Invalid string
    assert is_in_season_window("invalid", today_doy=today_doy, window=45) is False  # type: ignore
