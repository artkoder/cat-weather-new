"""
Test suite for sea picker type normalization fixes.

Tests edge cases where sorting keys contain mixed types (str, int, float, datetime).
"""
import json
import logging
import os
import sys
from datetime import datetime
from unittest.mock import Mock

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import main as main_module
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
    # asset1 falls back to timestamp 0.0 (treated as the oldest), asset2 has a valid date
    # Older assets have higher priority in the LRU component, so asset1 should be preferred.
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


@pytest.mark.parametrize("desired_wave_score", ["7", 7, 7.0, None])
@pytest.mark.parametrize("asset_wave_score", ["7", 7, 7.0, None])
@pytest.mark.parametrize("last_used_at", ["2025-11-04T08:50:00", "", None])
def test_pick_sea_asset_handles_mixed_numeric_inputs(
    desired_wave_score, asset_wave_score, last_used_at
):
    bot = Bot(token="dummy", db_path=":memory:")

    asset = Mock()
    asset.id = "101"
    vision_results = {"tags": ["sea"]}
    if asset_wave_score is not None:
        vision_results["sea_wave_score"] = {"value": asset_wave_score}
    asset.vision_results = vision_results
    asset.categories = None
    if last_used_at is None:
        asset.payload = None
    else:
        asset.payload = {"last_used_at": last_used_at}
    selected = bot._pick_sea_asset(
        [asset],
        desired_wave_score=desired_wave_score,
        desired_sky=None
    )
    assert selected is asset


def test_pick_sea_asset_fallback_to_id_on_exception(monkeypatch, caplog):
    bot = Bot(token="dummy", db_path=":memory:")

    asset = Mock()
    asset.id = "bad"
    asset.vision_results = {"sea_wave_score": {"value": "5"}, "tags": ["sea"]}
    asset.categories = None
    asset.payload = {"last_used_at": "2024-01-01T00:00:00"}

    def boom(*_args, **_kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(bot, "_extract_asset_sky", boom)

    with caplog.at_level(logging.ERROR):
        selected = bot._pick_sea_asset(
            [asset],
            desired_wave_score="5",
            desired_sky=None
        )

    assert selected is asset
    assert any(
        "Failed to build sea picker key" in record.message for record in caplog.records
    )


def test_pick_sea_asset_debug_logging_numeric_types(monkeypatch, caplog):
    bot = Bot(token="dummy", db_path=":memory:")

    monkeypatch.setattr(main_module, "DEBUG_SEA_PICK", True)

    asset_primary = Mock()
    asset_primary.id = "101"
    asset_primary.vision_results = {
        "sea_wave_score": {"value": "7"},
        "tags": ["sea", "sunset", "sunny"],
    }
    asset_primary.categories = ["Закат"]
    asset_primary.payload = {"last_used_at": "2025-11-04T08:50:00"}

    asset_secondary = Mock()
    asset_secondary.id = 202
    asset_secondary.vision_results = {
        "sea_wave_score": {"value": 5},
        "tags": ["sea", "waves"],
    }
    asset_secondary.categories = None
    asset_secondary.payload = {"last_used_at": "2024-11-04T08:50:00"}

    with caplog.at_level(logging.INFO):
        selected = bot._pick_sea_asset(
            [asset_primary, asset_secondary],
            desired_wave_score="2",
            desired_sky="sunny"
        )

    assert selected is asset_primary
    log_records = [
        record for record in caplog.records if "sea_picker_debug" in record.message
    ]
    assert log_records, "Expected diagnostic log when DEBUG_SEA_PICK is enabled"
    debug_payload = log_records[-1].args[-1]
    assert isinstance(debug_payload, list)
    assert debug_payload, "Expected candidate payloads in debug log"

    for entry in debug_payload:
        assert entry["types"] == ("int", "int", "float", "float", "float", "int")
        assert isinstance(entry["lru_ts"], float)
        assert isinstance(entry["wave_penalty"], float)
        assert isinstance(entry["key"], tuple)
        assert isinstance(entry["id"], int)

    primary_entry = next(entry for entry in debug_payload if entry["id"] == 101)
    assert isinstance(primary_entry["asset_score"], float)
    assert primary_entry["has_sunset_tag"] == 1
    assert primary_entry["has_sunset_category"] == 1
    assert isinstance(primary_entry["desired_wave_score"], (float, type(None)))
    if isinstance(primary_entry["desired_wave_score"], float):
        assert primary_entry["desired_wave_score"] == pytest.approx(2.0)



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

    # When all else is equal (same scores, both fall back to an LRU timestamp of 0.0),
    # the tiebreaker is the -asset_id component. Because -200 < -100, asset1 (id=100) wins.
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
