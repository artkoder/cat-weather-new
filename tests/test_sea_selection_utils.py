import pytest

from sea_selection import STAGE_CONFIGS, calc_wave_penalty, infer_sky_visible


def test_infer_sky_visible_positive_tags() -> None:
    assert infer_sky_visible(["Sea", "Blue Sky"]) is True


def test_infer_sky_visible_negative_tags() -> None:
    assert infer_sky_visible(["indoor", "cat"]) is False


def test_infer_sky_visible_unknown_tags() -> None:
    assert infer_sky_visible(["boat", "sand"]) is None


def test_calc_wave_penalty_within_tolerance() -> None:
    stage = STAGE_CONFIGS["B0"]
    assert calc_wave_penalty(1.5, 1.4, stage) == pytest.approx(0.0)


def test_calc_wave_penalty_outside_tolerance() -> None:
    stage = STAGE_CONFIGS["B1"]
    penalty = calc_wave_penalty(4.0, 1.5, stage)
    expected = max(0.0, abs(4.0 - 1.5) - stage.wave_tolerance) * stage.wave_penalty_rate
    assert penalty == pytest.approx(expected)


def test_calc_wave_penalty_missing_allowed() -> None:
    stage = STAGE_CONFIGS["B2"]
    assert calc_wave_penalty(None, 2.0, stage) == pytest.approx(stage.unknown_wave_penalty)
