import os
import pytest
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from main import Bot

os.environ.setdefault('TELEGRAM_BOT_TOKEN', 'dummy')

@pytest.mark.asyncio
async def test_asset_selection(tmp_path):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    bot.add_asset(1, '#дождь', 'cap')
    bot.add_asset(2, '', 'cap2')
    a = bot.next_asset({'#дождь'})
    assert a['message_id'] == 1
    a2 = bot.next_asset(None)
    assert a2['message_id'] == 2
    await bot.close()

@pytest.mark.asyncio
async def test_render_date(tmp_path):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    tomorrow = datetime.utcnow().date() + timedelta(days=1)
    tpl = 'date {next-day-date} {next-day-month}'
    result = bot._render_template(tpl)
    assert str(tomorrow.day) in result
    await bot.close()

@pytest.mark.asyncio
async def test_weather_scheduler_publish(tmp_path):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    bot.set_asset_channel(-100)
    calls = []
    async def dummy(method, data=None):
        calls.append((method, data))
        return {'ok': True}
    bot.api_request = dummy  # type: ignore
    bot.add_asset(1, '', 'hi')
    bot.add_weather_channel(-100, (datetime.utcnow() + timedelta(minutes=-1)).strftime('%H:%M'))
    await bot.process_weather_channels()
    assert any(c[0]=='copyMessage' for c in calls)
    await bot.close()

@pytest.mark.asyncio
async def test_handle_asset_message(tmp_path):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    bot.set_asset_channel(-100123)
    msg = {
        'message_id': 10,
        'chat': {'id': -100123},
        'caption': '#котопогода #дождь cap'
    }
    await bot.handle_message(msg)
    a = bot.next_asset({'#дождь'})
    assert a['message_id'] == 10
    await bot.close()
