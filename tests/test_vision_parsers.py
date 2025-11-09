"""Tests for vision result parsers."""

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_access import DataAccess


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
