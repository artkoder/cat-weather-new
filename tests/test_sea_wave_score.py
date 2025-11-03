import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import main as main_module


@pytest.mark.parametrize(
    "wave,expected",
    [
        (0.0, 0),
        (0.25, 1),
        (0.5, 2),
        (0.75, 3),
        (1.0, 4),
        (1.25, 6),
        (1.5, 7),
        (1.75, 8),
        (2.0, 9),
        (2.5, 10),
        (3.0, 10),
    ],
)
def test_interpolate_wave_to_score(wave, expected):
    bot = main_module.Bot(token="dummy", db_path=":memory:")
    result = bot._interpolate_wave_to_score(wave)
    assert result == expected


def test_extract_asset_sky_sunny():
    bot = main_module.Bot(token="dummy", db_path=":memory:")
    vision_results = {"tags": ["sunny", "clear", "bright"]}
    result = bot._extract_asset_sky(vision_results)
    assert result == "sunny"


def test_extract_asset_sky_cloudy():
    bot = main_module.Bot(token="dummy", db_path=":memory:")
    vision_results = {"tags": ["cloudy", "overcast"]}
    result = bot._extract_asset_sky(vision_results)
    assert result == "cloudy"


def test_extract_asset_sky_cloudy_priority_over_sunny():
    bot = main_module.Bot(token="dummy", db_path=":memory:")
    vision_results = {"tags": ["sunny", "cloudy", "rain"]}
    result = bot._extract_asset_sky(vision_results)
    assert result == "cloudy"


def test_extract_asset_sky_from_weather_image():
    bot = main_module.Bot(token="dummy", db_path=":memory:")
    vision_results = {"tags": ["sea"], "weather_image": "sunny"}
    result = bot._extract_asset_sky(vision_results)
    assert result == "sunny"


def test_extract_asset_sky_none():
    bot = main_module.Bot(token="dummy", db_path=":memory:")
    vision_results = {"tags": ["sea", "waves"]}
    result = bot._extract_asset_sky(vision_results)
    assert result is None


def test_get_asset_wave_score_with_direct_value():
    bot = main_module.Bot(token="dummy", db_path=":memory:")
    asset = Mock()
    asset.id = 1
    asset.vision_results = {
        "sea_wave_score": {"value": 7, "confidence": 0.85, "model": "gpt-4o-mini"}
    }
    result = bot._get_asset_wave_score_with_fallback(asset)
    assert result == 7


def test_get_asset_wave_score_fallback_whitecaps():
    bot = main_module.Bot(token="dummy", db_path=":memory:")
    asset = Mock()
    asset.id = 1
    asset.vision_results = {"tags": ["sea", "whitecaps", "foam"]}
    result = bot._get_asset_wave_score_with_fallback(asset)
    assert 8 <= result <= 9


def test_get_asset_wave_score_fallback_storm():
    bot = main_module.Bot(token="dummy", db_path=":memory:")
    asset = Mock()
    asset.id = 1
    asset.vision_results = {"tags": ["sea", "storm"]}
    result = bot._get_asset_wave_score_with_fallback(asset)
    assert 6 <= result <= 7


def test_get_asset_wave_score_fallback_waves():
    bot = main_module.Bot(token="dummy", db_path=":memory:")
    asset = Mock()
    asset.id = 1
    asset.vision_results = {"tags": ["sea", "waves"]}
    result = bot._get_asset_wave_score_with_fallback(asset)
    assert 4 <= result <= 5


def test_get_asset_wave_score_fallback_default():
    bot = main_module.Bot(token="dummy", db_path=":memory:")
    asset = Mock()
    asset.id = 1
    asset.vision_results = {"tags": ["sea"]}
    result = bot._get_asset_wave_score_with_fallback(asset)
    assert 1 <= result <= 2


def test_pick_sea_asset_by_wave_score():
    bot = main_module.Bot(token="dummy", db_path=":memory:")
    
    asset_calm = Mock()
    asset_calm.id = 1
    asset_calm.vision_results = {
        "sea_wave_score": {"value": 1, "confidence": 0.9},
        "tags": ["sea"],
    }
    asset_calm.categories = None
    asset_calm.payload = {}
    
    asset_stormy = Mock()
    asset_stormy.id = 2
    asset_stormy.vision_results = {
        "sea_wave_score": {"value": 9, "confidence": 0.9},
        "tags": ["sea", "storm"],
    }
    asset_stormy.categories = None
    asset_stormy.payload = {}
    
    asset_moderate = Mock()
    asset_moderate.id = 3
    asset_moderate.vision_results = {
        "sea_wave_score": {"value": 5, "confidence": 0.9},
        "tags": ["sea", "waves"],
    }
    asset_moderate.categories = None
    asset_moderate.payload = {}
    
    selected = bot._pick_sea_asset(
        [asset_calm, asset_stormy, asset_moderate],
        desired_wave_score=8,
        desired_sky=None,
    )
    assert selected.id == 2


def test_pick_sea_asset_by_sky_match():
    bot = main_module.Bot(token="dummy", db_path=":memory:")
    
    asset_sunny = Mock()
    asset_sunny.id = 1
    asset_sunny.vision_results = {
        "sea_wave_score": {"value": 5, "confidence": 0.9},
        "tags": ["sea", "sunny"],
    }
    asset_sunny.categories = None
    asset_sunny.payload = {}
    
    asset_cloudy = Mock()
    asset_cloudy.id = 2
    asset_cloudy.vision_results = {
        "sea_wave_score": {"value": 5, "confidence": 0.9},
        "tags": ["sea", "cloudy"],
    }
    asset_cloudy.categories = None
    asset_cloudy.payload = {}
    
    selected = bot._pick_sea_asset(
        [asset_sunny, asset_cloudy],
        desired_wave_score=5,
        desired_sky="cloudy",
    )
    assert selected.id == 2


def test_pick_sea_asset_sunset_bonus_calm():
    bot = main_module.Bot(token="dummy", db_path=":memory:")
    
    asset_regular = Mock()
    asset_regular.id = 1
    asset_regular.vision_results = {
        "sea_wave_score": {"value": 2, "confidence": 0.9},
        "tags": ["sea"],
    }
    asset_regular.categories = None
    asset_regular.payload = {}
    
    asset_sunset = Mock()
    asset_sunset.id = 2
    asset_sunset.vision_results = {
        "sea_wave_score": {"value": 2, "confidence": 0.9},
        "tags": ["sea", "sunset"],
    }
    asset_sunset.categories = None
    asset_sunset.payload = {}
    
    selected = bot._pick_sea_asset(
        [asset_regular, asset_sunset],
        desired_wave_score=2,
        desired_sky=None,
    )
    assert selected.id == 2


def test_pick_sea_asset_storm_bonus():
    bot = main_module.Bot(token="dummy", db_path=":memory:")
    
    asset_regular = Mock()
    asset_regular.id = 1
    asset_regular.vision_results = {
        "sea_wave_score": {"value": 7, "confidence": 0.9},
        "tags": ["sea"],
    }
    asset_regular.categories = None
    asset_regular.payload = {}
    
    asset_storm = Mock()
    asset_storm.id = 2
    asset_storm.vision_results = {
        "sea_wave_score": {"value": 7, "confidence": 0.9},
        "tags": ["sea", "storm", "whitecaps"],
    }
    asset_storm.categories = None
    asset_storm.payload = {}
    
    selected = bot._pick_sea_asset(
        [asset_regular, asset_storm],
        desired_wave_score=7,
        desired_sky=None,
    )
    assert selected.id == 2
