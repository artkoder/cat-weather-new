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


@pytest.mark.asyncio

async def test_edit_asset(tmp_path):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    bot.set_asset_channel(-100123)
    msg = {
        'message_id': 11,
        'chat': {'id': -100123},
        'caption': '#котопогода old'
    }
    await bot.handle_message(msg)
    edit = {
        'message_id': 11,
        'chat': {'id': -100123},
        'caption': '#котопогода #новый new'
    }
    await bot.handle_edited_message(edit)
    a = bot.next_asset({'#новый'})
    assert a and a['message_id'] == 11
    await bot.close()


@pytest.mark.asyncio

async def test_template_russian_and_period(tmp_path):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    # insert cached weather and sea data
    bot.db.execute(
        "INSERT INTO weather_cache_hour (city_id, timestamp, temperature, weather_code, wind_speed, is_day)"
        " VALUES (1, ?, 20.0, 1, 5.0, 1)",
        (datetime.utcnow().isoformat(),),
    )
    bot.db.execute(

        "INSERT INTO weather_cache_period (city_id, updated, morning_temp, morning_code, morning_wind, day_temp, day_code, day_wind, evening_temp, evening_code, evening_wind, night_temp, night_code, night_wind)"
        " VALUES (1, ?, 21.0, 1, 4.0, 22.0, 2, 5.0, 23.0, 3, 6.0, 24.0, 4, 7.0)",
        (datetime.utcnow().isoformat(),),
    )
    bot.db.execute(

        "INSERT INTO sea_cache (sea_id, updated, current, morning, day, evening, night)"
        " VALUES (1, ?, 15.0, 15.1, 15.2, 15.3, 15.4)",
        (datetime.utcnow().isoformat(),),
    )
    bot.db.commit()
    tpl = '{next-day-date} {next-day-month} {1|nm-temp} {1|nd-seatemperature}'
    result = bot._render_template(tpl)

    assert '15.' in result and '21\u00B0C' in result
    months = ['января','февраля','марта','апреля','мая','июня','июля','августа','сентября','октября','ноября','декабря']
    assert any(m in result for m in months)
    await bot.close()


@pytest.mark.asyncio
async def test_seastorm_render(tmp_path):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))

    bot.db.execute(
        "INSERT INTO sea_cache (sea_id, updated, current, morning, day, evening, night, wave, morning_wave, day_wave, evening_wave, night_wave)"
        " VALUES (1, ?, 15.0, 15.1, 15.2, 15.3, 15.4, 0.2, 0.4, 0.6, 1.6, 0.3)",
        (datetime.utcnow().isoformat(),),
    )
    bot.db.commit()

    assert bot._render_template('{1|seastorm}') == '\U0001F30A 15.0\u00B0C'
    assert bot._render_template('{1|nd-seastorm}') == '\U0001F30A шторм'
    assert bot._render_template('{1|ny-seastorm}') == '\U0001F30A сильный шторм'
    await bot.close()

