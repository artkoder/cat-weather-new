"""Tests for vision result parsers."""

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_access import DataAccess
from utils_wave import parse_sky_bucket_from_vision, parse_wave_score_from_vision


class TestVisionParsers:
    def test_parse_wave_score_weather_sea_layout(self):
        """Test parsing wave score from weather.sea.wave_score layout."""
        vision_json = {"weather": {"sea": {"wave_score": 3.5, "confidence": 0.85}}}
        wave_score, wave_conf = DataAccess._parse_wave_score_from_vision(vision_json)
        assert wave_score == 3.5
        assert wave_conf == 0.85

    def test_parse_wave_score_sea_state_layout(self):
        """Test parsing wave score from sea_state layout."""
        vision_json = {"sea_state": {"score": 7.2, "confidence": 0.92}}
        wave_score, wave_conf = DataAccess._parse_wave_score_from_vision(vision_json)
        assert wave_score == 7.2
        assert wave_conf == 0.92

    def test_parse_wave_score_legacy_dict_layout(self):
        """Test parsing wave score from legacy sea_wave_score dict layout."""
        vision_json = {"sea_wave_score": {"value": 5.0, "confidence": 0.78}}
        wave_score, wave_conf = DataAccess._parse_wave_score_from_vision(vision_json)
        assert wave_score == 5.0
        assert wave_conf == 0.78

    def test_parse_wave_score_legacy_scalar_layout(self):
        """Test parsing wave score from legacy sea_wave_score scalar layout."""
        vision_json = {"sea_wave_score": 4.5}
        wave_score, wave_conf = DataAccess._parse_wave_score_from_vision(vision_json)
        assert wave_score == 4.5
        assert wave_conf is None

    def test_parse_wave_score_from_text(self):
        """Test parsing wave score from Russian text result."""
        vision_json = {"result_text": "Погода: хорошая\nВолнение моря: 2.5/10\nВидимость: отличная"}
        wave_score, wave_conf = DataAccess._parse_wave_score_from_vision(vision_json)
        assert wave_score == 2.5
        assert wave_conf is None

    def test_parse_wave_score_missing_data(self):
        """Test parsing wave score when no data is present."""
        vision_json = {"some_other_field": "value"}
        wave_score, wave_conf = DataAccess._parse_wave_score_from_vision(vision_json)
        assert wave_score is None
        assert wave_conf is None

    def test_parse_wave_score_none_input(self):
        """Test parsing wave score with None input."""
        wave_score, wave_conf = DataAccess._parse_wave_score_from_vision(None)
        assert wave_score is None
        assert wave_conf is None

    def test_parse_sky_bucket_weather_layout(self):
        """Test parsing sky bucket from weather.sky.bucket layout."""
        vision_json = {"weather": {"sky": {"bucket": "partly_cloudy"}}}
        sky_bucket = DataAccess._parse_sky_bucket_from_vision(vision_json)
        assert sky_bucket == "partly_cloudy"

    def test_parse_sky_bucket_direct_layout(self):
        """Test parsing sky bucket from direct sky_bucket field."""
        vision_json = {"sky_bucket": "overcast"}
        sky_bucket = DataAccess._parse_sky_bucket_from_vision(vision_json)
        assert sky_bucket == "overcast"

    def test_parse_sky_bucket_sky_dict_layout(self):
        """Test parsing sky bucket from sky.bucket layout."""
        vision_json = {"sky": {"bucket": "clear"}}
        sky_bucket = DataAccess._parse_sky_bucket_from_vision(vision_json)
        assert sky_bucket == "clear"

    def test_parse_sky_bucket_missing_data(self):
        """Test parsing sky bucket when no data is present."""
        vision_json = {"some_other_field": "value"}
        sky_bucket = DataAccess._parse_sky_bucket_from_vision(vision_json)
        assert sky_bucket is None

    def test_parse_sky_bucket_none_input(self):
        """Test parsing sky bucket with None input."""
        sky_bucket = DataAccess._parse_sky_bucket_from_vision(None)
        assert sky_bucket is None


class TestUtilsWaveParsers:
    """Test centralized parsing functions in utils_wave module."""

    def test_parse_wave_score_with_confidence_from_text(self):
        """Test parsing wave score with confidence from Russian text."""
        vision_json = {
            "result_text": "Погода: хорошая\nВолнение моря: 7.5/10 (conf=0.92)\nВидимость: отличная"
        }
        wave_score, wave_conf = parse_wave_score_from_vision(vision_json)
        assert wave_score == 7.5
        assert wave_conf == 0.92

    def test_parse_wave_score_with_confidence_from_text_spaces(self):
        """Test parsing with various spacing patterns."""
        vision_json = {"result_text": "Волнение моря:  3.2  /  10  ( conf = 0.88 )"}
        wave_score, wave_conf = parse_wave_score_from_vision(vision_json)
        assert wave_score == 3.2
        assert wave_conf == 0.88

    def test_parse_wave_score_integer_values(self):
        """Test parsing integer wave scores."""
        vision_json = {"sea_wave_score": {"value": 8, "confidence": 1}}
        wave_score, wave_conf = parse_wave_score_from_vision(vision_json)
        assert wave_score == 8.0
        assert wave_conf == 1.0

    def test_parse_wave_score_mixed_types(self):
        """Test parsing with mixed int/float types."""
        vision_json = {"weather": {"sea": {"wave_score": 4, "confidence": 0.75}}}
        wave_score, wave_conf = parse_wave_score_from_vision(vision_json)
        assert wave_score == 4.0
        assert wave_conf == 0.75

    def test_parse_wave_score_string_numbers(self):
        """Test parsing wave score from string representations."""
        vision_json = {"sea_state": {"score": "6.5", "confidence": "0.88"}}
        wave_score, wave_conf = parse_wave_score_from_vision(vision_json)
        assert wave_score == 6.5
        assert wave_conf == 0.88

    def test_parse_wave_score_invalid_string(self):
        """Test parsing with invalid string values."""
        vision_json = {"sea_wave_score": {"value": "invalid", "confidence": "bad"}}
        wave_score, wave_conf = parse_wave_score_from_vision(vision_json)
        assert wave_score is None
        assert wave_conf is None

    def test_parse_wave_score_empty_dict(self):
        """Test parsing from empty dict."""
        vision_json = {}
        wave_score, wave_conf = parse_wave_score_from_vision(vision_json)
        assert wave_score is None
        assert wave_conf is None

    def test_parse_wave_score_partial_confidence(self):
        """Test parsing when only score is present."""
        vision_json = {"sea_wave_score": {"value": 5.0}}
        wave_score, wave_conf = parse_wave_score_from_vision(vision_json)
        assert wave_score == 5.0
        assert wave_conf is None

    def test_parse_wave_score_fallback_priority(self):
        """Test that structured formats take priority over text."""
        vision_json = {
            "weather": {"sea": {"wave_score": 3.0, "confidence": 0.9}},
            "result_text": "Волнение моря: 8.0/10 (conf=0.5)",
        }
        wave_score, wave_conf = parse_wave_score_from_vision(vision_json)
        assert wave_score == 3.0
        assert wave_conf == 0.9

    def test_parse_sky_bucket_weather_sky_layout(self):
        """Test parsing sky bucket from weather.sky.bucket."""
        vision_json = {"weather": {"sky": {"bucket": "clear"}}}
        sky_bucket = parse_sky_bucket_from_vision(vision_json)
        assert sky_bucket == "clear"

    def test_parse_sky_bucket_sky_layout(self):
        """Test parsing sky bucket from sky.bucket."""
        vision_json = {"sky": {"bucket": "partly_cloudy"}}
        sky_bucket = parse_sky_bucket_from_vision(vision_json)
        assert sky_bucket == "partly_cloudy"

    def test_parse_sky_bucket_direct(self):
        """Test parsing sky bucket from direct field."""
        vision_json = {"sky_bucket": "overcast"}
        sky_bucket = parse_sky_bucket_from_vision(vision_json)
        assert sky_bucket == "overcast"

    def test_parse_sky_bucket_priority(self):
        """Test that weather.sky.bucket takes priority."""
        vision_json = {
            "weather": {"sky": {"bucket": "clear"}},
            "sky_bucket": "overcast",
            "sky": {"bucket": "partly_cloudy"},
        }
        sky_bucket = parse_sky_bucket_from_vision(vision_json)
        assert sky_bucket == "clear"

    def test_parse_sky_bucket_empty(self):
        """Test parsing when sky bucket is empty string."""
        vision_json = {"sky_bucket": ""}
        sky_bucket = parse_sky_bucket_from_vision(vision_json)
        assert sky_bucket is None

    def test_parse_sky_bucket_none_value(self):
        """Test parsing when sky bucket is None."""
        vision_json = {"sky_bucket": None}
        sky_bucket = parse_sky_bucket_from_vision(vision_json)
        assert sky_bucket is None

    def test_parse_wave_score_from_caption_field(self):
        """Test parsing wave score from caption field (backward compatibility)."""
        vision_json = {"caption": "Морской пейзаж. Волнение моря: 5/10 (conf=0.80)"}
        wave_score, wave_conf = parse_wave_score_from_vision(vision_json)
        assert wave_score == 5.0
        assert wave_conf == 0.80

    def test_parse_wave_score_from_caption_without_confidence(self):
        """Test parsing wave score from caption field without confidence."""
        vision_json = {"caption": "Красивый вид. Волнение моря: 3/10"}
        wave_score, wave_conf = parse_wave_score_from_vision(vision_json)
        assert wave_score == 3.0
        assert wave_conf is None

    def test_parse_wave_score_prefers_result_text_over_caption(self):
        """Test that result_text is preferred over caption when both present."""
        vision_json = {
            "result_text": "Волнение моря: 8/10 (conf=0.95)",
            "caption": "Волнение моря: 3/10 (conf=0.50)",
        }
        wave_score, wave_conf = parse_wave_score_from_vision(vision_json)
        assert wave_score == 8.0
        assert wave_conf == 0.95

    def test_consistency_between_utils_and_dataaccess(self):
        """Test that DataAccess methods delegate to utils_wave correctly."""
        test_cases = [
            {"weather": {"sea": {"wave_score": 5.5, "confidence": 0.85}}},
            {"sea_state": {"score": 3.2, "confidence": 0.75}},
            {"sea_wave_score": {"value": 7, "confidence": 0.9}},
            {"sea_wave_score": 4.5},
            {"result_text": "Волнение моря: 6.0/10 (conf=0.88)"},
            {"result_text": "Волнение моря: 2.5/10"},
            {"caption": "Волнение моря: 5/10 (conf=0.80)"},
        ]
        for test_case in test_cases:
            utils_result = parse_wave_score_from_vision(test_case)
            data_access_result = DataAccess._parse_wave_score_from_vision(test_case)
            assert utils_result == data_access_result

        sky_test_cases = [
            {"weather": {"sky": {"bucket": "clear"}}},
            {"sky": {"bucket": "partly_cloudy"}},
            {"sky_bucket": "overcast"},
        ]
        for test_case in sky_test_cases:
            utils_result = parse_sky_bucket_from_vision(test_case)
            data_access_result = DataAccess._parse_sky_bucket_from_vision(test_case)
            assert utils_result == data_access_result
