import os
import sys
from datetime import datetime, timedelta

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_access import DataAccess


def test_compute_age_bonus_defaults_to_two_days() -> None:
    assert DataAccess.compute_age_bonus(None) == pytest.approx(2.0)


def test_compute_age_bonus_handles_future_timestamp() -> None:
    future = datetime.utcnow() + timedelta(days=1)
    assert DataAccess.compute_age_bonus(future.isoformat()) == pytest.approx(0.5)


def test_compute_age_bonus_scales_with_days() -> None:
    nine_days_ago = datetime.utcnow() - timedelta(days=9)
    assert DataAccess.compute_age_bonus(nine_days_ago.isoformat()) == pytest.approx(3.0)


def test_compute_age_bonus_partial_days() -> None:
    three_days_ago = datetime.utcnow() - timedelta(days=3)
    assert DataAccess.compute_age_bonus(three_days_ago.isoformat()) == pytest.approx(1.0)
