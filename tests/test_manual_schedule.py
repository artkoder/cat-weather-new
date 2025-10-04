import os
import sys
import pytest
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from main import Bot

os.environ.setdefault('TELEGRAM_BOT_TOKEN', 'dummy')

@pytest.mark.asyncio
async def test_manual_run_does_not_block_schedule(tmp_path):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    bot.set_weather_assets_channel(-100)
    calls = []

    async def dummy(method, data=None):
        calls.append(method)
        return {'ok': True}

    bot.api_request = dummy  # type: ignore
    bot.add_asset(1, '', 't1')
    bot.add_asset(2, '', 't2')
    # schedule channel for current time
    bot.add_weather_channel(-200, (datetime.utcnow() - timedelta(minutes=1)).strftime('%H:%M'))
    # manual run first
    await bot.publish_weather(-200, None, record=False)
    await bot.process_weather_channels()
    copy_count = sum(1 for c in calls if c == 'copyMessage')
    assert copy_count == 2
    await bot.close()

