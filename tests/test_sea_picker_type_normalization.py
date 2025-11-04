"""
Test suite for sea picker type normalization fixes.

Tests edge cases where sorting keys contain mixed types (str, int, float, datetime).
"""
import json
import os
import sys
from datetime import datetime
from unittest.mock import Mock

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import Bot


def test_safe_int_handles_string():
    """Test that _safe_int converts string to int."""
    bot = Bot(token="dummy", db_path=":memory:")

    assert bot._safe_int("42") == 42
    assert bot._safe_int("7") == 7
    assert bot._safe_int("0") == 0
    assert bot._safe_int("-5") == -5
    assert bot._safe_int("3.14") == 3  # String float gets converted


def test_safe_int_handles_none():
    """Test that _safe_int handles None gracefully."""
    bot = Bot(token="dummy", db_path=":memory:")

    assert bot._safe_int(None) == 0
    assert bot._safe_int(None, default=42) == 42


def test_safe_int_handles_invalid_string():
    """Test that _safe_int returns default for invalid strings."""
    bot = Bot(token="dummy", db_path=":memory:")

    assert bot._safe_int("invalid") == 0
    assert bot._safe_int("abc", default=99) == 99
    assert bot._safe_int("") == 0


def test_safe_int_handles_various_types():
    """Test that _safe_int handles various types."""
    bot = Bot(token="dummy", db_path=":memory:")

    assert bot._safe_int(42) == 42
    assert bot._safe_int(3.14) == 3
    assert bot._safe_int(True) == 1
    assert bot._safe_int(False) == 0


def test_safe_float_handles_string():
    """Test that _safe_float converts string to float."""
    bot = Bot(token="dummy", db_path=":memory:")

    assert bot._safe_float("3.14") == 3.14
    assert bot._safe_float("7") == 7.0
    assert bot._safe_float("0.0") == 0.0
    assert bot._safe_float("-2.5") == -2.5


def test_safe_float_handles_none():
    """Test that _safe_float handles None gracefully."""
    bot = Bot(token="dummy", db_path=":memory:")

    assert bot._safe_float(None) is None
    assert bot._safe_float(None, default=1.5) == 1.5


def test_safe_float_handles_invalid_string():
    """Test that _safe_float returns default for invalid strings."""
    bot = Bot(token="dummy", db_path=":memory:")

    assert bot._safe_float("invalid") is None
    assert bot._safe_float("abc", default=9.9) == 9.9
    assert bot._safe_float("") is None


def test_safe_float_handles_various_types():
    """Test that _safe_float handles various types."""
    bot = Bot(token="dummy", db_path=":memory:")

    assert bot._safe_float(42) == 42.0
    assert bot._safe_float(3.14) == 3.14


def test_parse_datetime_iso_valid():
    """Test that _parse_datetime_iso parses valid ISO strings."""
    bot = Bot(token="dummy", db_path=":memory:")

    result = bot._parse_datetime_iso("2023-01-15T10:30:00")
    assert isinstance(result, datetime)
    assert result.year == 2023
    assert result.month == 1
    assert result.day == 15

    result2 = bot._parse_datetime_iso("2025-11-04T08:50:00")
    assert isinstance(result2, datetime)
    assert result2.year == 2025


def test_parse_datetime_iso_invalid():
    """Test that _parse_datetime_iso returns datetime.min for invalid input."""
    bot = Bot(token="dummy", db_path=":memory:")

    assert bot._parse_datetime_iso(None) == datetime.min
    assert bot._parse_datetime_iso("") == datetime.min
    assert bot._parse_datetime_iso("invalid") == datetime.min
    assert bot._parse_datetime_iso("not-a-date") == datetime.min


def test_get_asset_wave_score_with_string_value():
    """Test that wave score extraction handles string values."""
    bot = Bot(token="dummy", db_path=":memory:")

    asset = Mock()
    asset.id = 1
    asset.vision_results = {
        "sea_wave_score": {"value": "7", "confidence": 0.85}
    }

    result = bot._get_asset_wave_score_with_fallback(asset)
    assert result == 7


def test_get_asset_wave_score_with_int_value():
    """Test that wave score extraction handles int values."""
    bot = Bot(token="dummy", db_path=":memory:")

    asset = Mock()
    asset.id = 1
    asset.vision_results = {
        "sea_wave_score": {"value": 7, "confidence": 0.85}
    }

    result = bot._get_asset_wave_score_with_fallback(asset)
    assert result == 7


def test_get_asset_wave_score_with_float_value():
    """Test that wave score extraction handles float values."""
    bot = Bot(token="dummy", db_path=":memory:")

    asset = Mock()
    asset.id = 1
    asset.vision_results = {
        "sea_wave_score": {"value": 7.8, "confidence": 0.85}
    }

    result = bot._get_asset_wave_score_with_fallback(asset)
    assert result == 7


def test_pick_sea_asset_with_string_last_used_at():
    """Test that sea picker handles string last_used_at."""
    bot = Bot(token="dummy", db_path=":memory:")

    asset1 = Mock()
    asset1.id = "101"  # String ID
    asset1.vision_results = {
        "sea_wave_score": {"value": "5"},
        "tags": ["sea"]
    }
    asset1.categories = None
    asset1.payload = {"last_used_at": "2023-01-01T00:00:00"}

    asset2 = Mock()
    asset2.id = 102  # Int ID
    asset2.vision_results = {
        "sea_wave_score": {"value": 5},
        "tags": ["sea"]
    }
    asset2.categories = None
    asset2.payload = {"last_used_at": "2024-01-01T00:00:00"}

    # Should prefer asset1 because it's older (LRU)
    selected = bot._pick_sea_asset(
        [asset1, asset2],
        desired_wave_score=5,
        desired_sky=None
    )
    assert selected.id == "101"


def test_pick_sea_asset_with_empty_last_used_at():
    """Test that sea picker handles missing or empty last_used_at."""
    bot = Bot(token="dummy", db_path=":memory:")

    asset1 = Mock()
    asset1.id = 1
    asset1.vision_results = {"sea_wave_score": {"value": 5}, "tags": ["sea"]}
    asset1.categories = None
    asset1.payload = {}  # No last_used_at

    asset2 = Mock()
    asset2.id = 2
    asset2.vision_results = {"sea_wave_score": {"value": 5}, "tags": ["sea"]}
    asset2.categories = None
    asset2.payload = {"last_used_at": ""}  # Empty last_used_at

    asset3 = Mock()
    asset3.id = 3
    asset3.vision_results = {"sea_wave_score": {"value": 5}, "tags": ["sea"]}
    asset3.categories = None
    asset3.payload = None  # No payload

    # Should work without errors and pick one deterministically
    selected = bot._pick_sea_asset(
        [asset1, asset2, asset3],
        desired_wave_score=5,
        desired_sky=None
    )
    assert selected is not None


def test_pick_sea_asset_with_invalid_last_used_at():
    """Test that sea picker handles invalid datetime strings."""
    bot = Bot(token="dummy", db_path=":memory:")

    asset1 = Mock()
    asset1.id = 1
    asset1.vision_results = {"sea_wave_score": {"value": 5}, "tags": ["sea"]}
    asset1.categories = None
    asset1.payload = {"last_used_at": "not-a-date"}

    asset2 = Mock()
    asset2.id = 2
    asset2.vision_results = {"sea_wave_score": {"value": 5}, "tags": ["sea"]}
    asset2.categories = None
    asset2.payload = {"last_used_at": "2024-01-01T00:00:00"}

    # Should work without errors - both assets can be compared
    selected = bot._pick_sea_asset(
        [asset1, asset2],
        desired_wave_score=5,
        desired_sky=None
    )
    # asset1 will have float('inf') (treat as never used), asset2 has a valid date
    # asset1 with inf is treated as "never used before" which means it should be preferred
    # (higher priority for unused assets in LRU)
    assert selected.id == 1


def test_pick_sea_asset_mixed_type_consistency():
    """
    Test that sea picker handles mixed types consistently.

    This is the regression test for the TypeError bug where sorting keys
    contained incompatible types.
    """
    bot = Bot(token="dummy", db_path=":memory:")

    # Asset with all string values
    asset_str = Mock()
    asset_str.id = "201"
    asset_str.vision_results = {
        "sea_wave_score": {"value": "7"},
        "tags": ["sea", "sunset"]
    }
    asset_str.categories = ["Закат"]
    asset_str.payload = {"last_used_at": "2023-11-04T08:50:00"}

    # Asset with all numeric values
    asset_num = Mock()
    asset_num.id = 202
    asset_num.vision_results = {
        "sea_wave_score": {"value": 6},
        "tags": ["sea"]
    }
    asset_num.categories = None
    asset_num.payload = {"last_used_at": "2023-11-05T08:50:00"}

    # Asset with None/missing values
    asset_none = Mock()
    asset_none.id = 203
    asset_none.vision_results = {"tags": ["sea", "waves"]}
    asset_none.categories = None
    asset_none.payload = {}

    # This should not raise TypeError about comparing str and float
    selected = bot._pick_sea_asset(
        [asset_str, asset_num, asset_none],
        desired_wave_score=7,
        desired_sky=None
    )

    assert selected is not None
    # asset_str should win because it matches wave score better
    assert selected.id == "201"


def test_pick_sea_asset_deterministic_with_same_scores():
    """Test that sea picker is deterministic when assets have identical scores."""
    bot = Bot(token="dummy", db_path=":memory:")

    asset1 = Mock()
    asset1.id = 100
    asset1.vision_results = {"sea_wave_score": {"value": 5}, "tags": ["sea"]}
    asset1.categories = None
    asset1.payload = {}

    asset2 = Mock()
    asset2.id = 200
    asset2.vision_results = {"sea_wave_score": {"value": 5}, "tags": ["sea"]}
    asset2.categories = None
    asset2.payload = {}

    # When all else is equal (same scores, both have inf LRU), the tiebreaker is -asset_id
    # Since sort key uses -asset_id, -200 < -100, so asset1 (id=100) wins in max()
    # This is because max() with negative values prefers the less negative (higher absolute value)
    selected = bot._pick_sea_asset(
        [asset1, asset2],
        desired_wave_score=5,
        desired_sky=None
    )

    # Actually, max() prefers larger tuple values at each position
    # When comparing (-100) vs (-200), -100 > -200, so asset1 wins
    assert selected.id == 100


def test_pick_sea_asset_preserves_existing_behavior():
    """
    Test that type normalization doesn't break existing behavior.

    This mirrors the test_pick_sea_asset_prioritizes_tags_and_categories test.
    """
    bot = Bot(token="dummy", db_path=":memory:")

    def build_asset(asset_id, tags=None, categories=None, last_used=None):
        asset = Mock()
        asset.id = asset_id
        asset.vision_results = {"tags": list(tags or [])}
        asset.categories = categories
        asset.payload = {}
        if last_used:
            asset.payload["last_used_at"] = last_used
        return asset

    sunset_tag = build_asset(101, tags=["sunset", "sea"], last_used="2023-01-01T00:00:00")
    sunset_category = build_asset(102, categories=["Закат"], last_used=datetime.utcnow().isoformat())

    # Calm seas should prefer sunset tag over sunset category
    selected = bot._pick_sea_asset(
        [sunset_category, sunset_tag],
        desired_wave_score=2,
        desired_sky=None
    )
    assert selected.id == 101
