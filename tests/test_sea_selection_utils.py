import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sea_selection import STAGE_CONFIGS, calc_wave_penalty, infer_sky_visible


def test_infer_sky_visible_positive_tags() -> None:
    assert infer_sky_visible(["Sea", "Blue Sky"]) is True


def test_infer_sky_visible_negative_tags() -> None:
    assert infer_sky_visible(["indoor", "cat"]) is False


def test_infer_sky_visible_unknown_tags() -> None:
    assert infer_sky_visible(["boat", "sand"]) is None


def test_calc_wave_penalty_within_tolerance() -> None:
    stage = STAGE_CONFIGS["B0"]
    assert calc_wave_penalty(3, 2, stage) == pytest.approx(0.0)


def test_calc_wave_penalty_outside_tolerance() -> None:
    stage = STAGE_CONFIGS["B1"]
    penalty = calc_wave_penalty(8, 3, stage)
    expected = max(0.0, abs(8 - 3) - stage.wave_tolerance) * stage.wave_penalty_rate
    assert penalty == pytest.approx(expected)


def test_calc_wave_penalty_missing_allowed() -> None:
    stage = STAGE_CONFIGS["B2"]
    assert calc_wave_penalty(None, 4, stage) == pytest.approx(stage.unknown_wave_penalty)
