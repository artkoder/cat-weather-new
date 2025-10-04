import os
import sys
from datetime import datetime, date, timezone

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import Bot  # noqa: E402


@pytest.mark.asyncio
async def test_next_usage_reset_uses_local_midnight(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    reference = datetime(2024, 6, 1, 20, 0, 0)
    tz_offset = "+03:00"
    reset = bot._next_usage_reset(now=reference, tz_offset=tz_offset)  # type: ignore[attr-defined]
    assert reset == datetime(2024, 6, 1, 21, 5)
    local_reset = reset.replace(tzinfo=timezone.utc).astimezone(  # type: ignore[attr-defined]
        bot._parse_tz_offset(tz_offset)  # type: ignore[attr-defined]
    )
    assert local_reset.hour == 0
    assert local_reset.minute == 5
    assert local_reset.date() == date(2024, 6, 2)
    await bot.close()


@pytest.mark.asyncio
async def test_token_usage_total_respects_timezone(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    entries = [
        ("2024-06-01T20:59:59", 5),
        ("2024-06-01T21:00:00", 7),
        ("2024-06-02T20:59:59", 11),
        ("2024-06-02T21:00:00", 13),
    ]
    for ts, total in entries:
        bot.data.log_token_usage(
            "gpt-4o",
            total,
            None,
            total,
            timestamp=ts,
        )
    total = bot.data.get_daily_token_usage_total(
        day=date(2024, 6, 2), models={"gpt-4o"}, tz_offset="+03:00"
    )
    assert total == 18
    # Default timezone fallback should use global TZ_OFFSET when tz is None
    default_total = bot.data.get_daily_token_usage_total(models={"gpt-4o"})
    assert isinstance(default_total, int)
    await bot.close()
