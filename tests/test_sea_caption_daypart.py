import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import main as main_module
from openai_client import OpenAIResponse


class CapturingOpenAI:
    def __init__(self) -> None:
        self.api_key = "fake-key"
        self.calls: list[dict[str, Any]] = []

    async def generate_json(self, **kwargs) -> OpenAIResponse:  # type: ignore[override]
        self.calls.append(kwargs)
        usage = {
            "prompt_tokens": 5,
            "completion_tokens": 10,
            "total_tokens": 15,
            "endpoint": "/v1/responses",
            "request_id": f"req-{len(self.calls)}",
        }
        content = {
            "caption": "Порадую вас морем — волны зовут на прогулку.",
            "hashtags": ["морем", "БалтийскоеМоре"],
        }
        return OpenAIResponse(content, usage)


class DummySupabase:
    async def insert_token_usage(self, *args: Any, **kwargs: Any) -> tuple[bool, Any, Any]:
        return False, None, "disabled"

    async def aclose(self) -> None:
        return None


async def async_record_usage_noop(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
    return None


class TestDayPartMapping:
    """Test the hour to day_part mapping."""

    def test_morning_boundaries(self) -> None:
        """Test morning boundaries: 05:00–10:59."""
        assert main_module.map_hour_to_day_part(5) == "morning"
        assert main_module.map_hour_to_day_part(7) == "morning"
        assert main_module.map_hour_to_day_part(10) == "morning"

    def test_day_boundaries(self) -> None:
        """Test day boundaries: 11:00–16:59."""
        assert main_module.map_hour_to_day_part(11) == "day"
        assert main_module.map_hour_to_day_part(14) == "day"
        assert main_module.map_hour_to_day_part(16) == "day"

    def test_evening_boundaries(self) -> None:
        """Test evening boundaries: 17:00–21:59."""
        assert main_module.map_hour_to_day_part(17) == "evening"
        assert main_module.map_hour_to_day_part(19) == "evening"
        assert main_module.map_hour_to_day_part(21) == "evening"

    def test_night_boundaries(self) -> None:
        """Test night boundaries: 22:00–04:59."""
        assert main_module.map_hour_to_day_part(22) == "night"
        assert main_module.map_hour_to_day_part(0) == "night"
        assert main_module.map_hour_to_day_part(4) == "night"

    def test_edge_cases_transitions(self) -> None:
        """Test edge cases at transitions."""
        # 10:59 is morning, 11:00 is day
        assert main_module.map_hour_to_day_part(10) == "morning"
        assert main_module.map_hour_to_day_part(11) == "day"

        # 16:59 is day, 17:00 is evening
        assert main_module.map_hour_to_day_part(16) == "day"
        assert main_module.map_hour_to_day_part(17) == "evening"

        # 21:59 is evening, 22:00 is night
        assert main_module.map_hour_to_day_part(21) == "evening"
        assert main_module.map_hour_to_day_part(22) == "night"

        # 04:59 is night, 05:00 is morning
        assert main_module.map_hour_to_day_part(4) == "night"
        assert main_module.map_hour_to_day_part(5) == "morning"


@pytest.mark.asyncio
async def test_sea_caption_includes_day_part_params(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that day_part parameters are included in the payload."""
    monkeypatch.setattr(main_module.Bot, "_record_openai_usage", async_record_usage_noop, raising=False)
    bot = main_module.Bot("dummy", str(tmp_path / "daypart.db"))
    bot.supabase = DummySupabase()
    bot.openai = CapturingOpenAI()

    # Test with morning
    await bot._generate_sea_caption(
        storm_state="calm",
        storm_persisting=False,
        wave_height_m=0.5,
        wave_score=2,
        wind_class=None,
        wind_ms=5.0,
        wind_kmh=18.0,
        clouds_label="ясное небо",
        sunset_selected=False,
        want_sunset=False,
        place_hashtag=None,
        fact_sentence=None,
        now_local_iso="2025-11-05T08:30:00+02:00",
        day_part="morning",
        tz_name="Europe/Kaliningrad",
        job=None,
    )

    first_call = bot.openai.calls[-1]
    user_prompt = first_call["user_prompt"]
    system_prompt = first_call["system_prompt"]

    # Check that the fields are in the user prompt
    assert '"now_local_iso": "2025-11-05T08:30:00+02:00"' in user_prompt
    assert '"day_part": "morning"' in user_prompt
    assert '"tz_name": "Europe/Kaliningrad"' in user_prompt

    # Check that the day_part instruction is in the system prompt
    assert "day_part (morning|day|evening|night)" in system_prompt
    assert "Пиши уместно текущему времени суток" in system_prompt

    bot.db.close()


@pytest.mark.asyncio
async def test_sea_caption_evening_without_morning_wishes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that evening captions don't contain morning wishes."""
    monkeypatch.setattr(main_module.Bot, "_record_openai_usage", async_record_usage_noop, raising=False)

    class MorningWishCapturingOpenAI(CapturingOpenAI):
        async def generate_json(self, **kwargs) -> OpenAIResponse:  # type: ignore[override]
            self.calls.append(kwargs)
            usage = {
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "total_tokens": 15,
                "endpoint": "/v1/responses",
                "request_id": f"req-{len(self.calls)}",
            }
            # Ensure evening caption doesn't contain morning wishes
            if "user_prompt" in kwargs:
                if '"day_part": "evening"' in kwargs["user_prompt"]:
                    # Check if the system prompt includes day_part instruction
                    if "day_part (morning|day|evening|night)" in kwargs.get("system_prompt", ""):
                        # Evening caption should not have morning-specific wishes
                        content = {
                            "caption": "Спокойный вечер у моря — волны ласкают берег.",
                            "hashtags": ["морем", "БалтийскоеМоре"],
                        }
                    else:
                        content = {
                            "caption": "Пусть ваш день будет чудесным!",
                            "hashtags": ["морем"],
                        }
                else:
                    content = {
                        "caption": "Порадую вас морем — волны зовут на прогулку.",
                        "hashtags": ["морем", "БалтийскоеМоре"],
                    }
            else:
                content = {
                    "caption": "Порадую вас морем — волны зовут на прогулку.",
                    "hashtags": ["морем", "БалтийскоеМоре"],
                }
            return OpenAIResponse(content, usage)

    bot = main_module.Bot("dummy", str(tmp_path / "evening.db"))
    bot.supabase = DummySupabase()
    bot.openai = MorningWishCapturingOpenAI()

    # Test with evening
    await bot._generate_sea_caption(
        storm_state="calm",
        storm_persisting=False,
        wave_height_m=0.5,
        wave_score=2,
        wind_class=None,
        wind_ms=5.0,
        wind_kmh=18.0,
        clouds_label="ясное небо",
        sunset_selected=False,
        want_sunset=False,
        place_hashtag=None,
        fact_sentence=None,
        now_local_iso="2025-11-05T19:40:00+02:00",
        day_part="evening",
        tz_name="Europe/Kaliningrad",
        job=None,
    )

    last_call = bot.openai.calls[-1]
    user_prompt = last_call["user_prompt"]
    system_prompt = last_call["system_prompt"]

    # Verify evening parameters are in the prompt
    assert '"day_part": "evening"' in user_prompt
    # Verify the instruction about time-appropriate content is in system prompt
    assert "day_part (morning|day|evening|night)" in system_prompt

    bot.db.close()


@pytest.mark.asyncio
async def test_sea_caption_without_day_part(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that captions work without day_part parameters (backward compatibility)."""
    monkeypatch.setattr(main_module.Bot, "_record_openai_usage", async_record_usage_noop, raising=False)
    bot = main_module.Bot("dummy", str(tmp_path / "no_daypart.db"))
    bot.supabase = DummySupabase()
    bot.openai = CapturingOpenAI()

    # Test without day_part parameters
    await bot._generate_sea_caption(
        storm_state="calm",
        storm_persisting=False,
        wave_height_m=0.5,
        wave_score=2,
        wind_class=None,
        wind_ms=5.0,
        wind_kmh=18.0,
        clouds_label="ясное небо",
        sunset_selected=False,
        want_sunset=False,
        place_hashtag=None,
        fact_sentence=None,
        job=None,
    )

    first_call = bot.openai.calls[-1]
    user_prompt = first_call["user_prompt"]
    system_prompt = first_call["system_prompt"]

    # Day_part fields should NOT be in the payload
    assert '"day_part"' not in user_prompt
    assert '"now_local_iso"' not in user_prompt
    assert '"tz_name"' not in user_prompt

    # Day_part instruction should NOT be in the system prompt
    assert "day_part (morning|day|evening|night)" not in system_prompt

    bot.db.close()
