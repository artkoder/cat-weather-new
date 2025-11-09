"""Tests for utils_wave module."""

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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
def test_wave_m_to_score_basic(wave: float | None, expected: int) -> None:
    """Test basic wave height to score conversion with 0.2m steps."""
    assert wave_m_to_score(wave) == expected


@pytest.mark.parametrize(
    "wave,expected",
    [
        (0.0, 0),
        (0.1, 0),
        (0.2, 1),
        (0.3, 1),
        (0.4, 2),
        (0.6, 3),
    ],
)
def test_wave_m_to_score_calm_range(wave: float, expected: int) -> None:
    """Test calm wave range (0-0.6m) scoring."""
    assert wave_m_to_score(wave) == expected


def test_wave_m_to_score_corridor_boundaries() -> None:
    """Test that scores align with corridor tolerances."""
    assert wave_m_to_score(0.0) == 0
    assert wave_m_to_score(0.2) == 1
    assert wave_m_to_score(0.4) == 2
    assert wave_m_to_score(0.6) == 3
    assert wave_m_to_score(1.0) == 5
    assert wave_m_to_score(2.0) == 10
