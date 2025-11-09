"""Wave scoring utilities for sea photo selection."""

from __future__ import annotations

import math


def wave_m_to_score(meters: float | int | str | None) -> int:
    """Map wave height in meters to an integer score on a 0..10 scale."""

    if meters is None:
        return 0
    try:
        value = float(meters)
    except (TypeError, ValueError):
        return 0
    if value <= 0.0:
        return 0
    step = 0.2
    epsilon = 1e-9
    score = math.floor((value + epsilon) / step)
    return max(0, min(10, score))
