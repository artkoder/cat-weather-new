import os
import sys
from datetime import datetime, timedelta
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from main import Bot

os.environ.setdefault('TELEGRAM_BOT_TOKEN', 'dummy')

@pytest.mark.asyncio
async def test_amber_notification(tmp_path):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    # set up sea and channel
    bot.db.execute("INSERT INTO seas (id, name, lat, lon) VALUES (1, 'sea', 0, 0)")
    bot.db.execute("INSERT INTO channels (chat_id, title) VALUES (-100, 'ch')")
    bot.db.commit()
    bot.set_amber_sea(1)
    bot.db.execute("INSERT INTO amber_channels (channel_id) VALUES (-100)")
    start = datetime.utcnow() - timedelta(hours=2)
    bot.db.execute("UPDATE amber_state SET storm_start=?, active=1 WHERE sea_id=1", (start.isoformat(),))
    bot.db.execute(
        "INSERT INTO sea_cache (sea_id, updated, current, morning, day, evening, night, wave, morning_wave, day_wave, evening_wave, night_wave) VALUES (1, ?, 15, 15,15,15,15, 0.4,0,0,0,0)",
        (datetime.utcnow().isoformat(),),
    )
    bot.db.commit()
    calls = []
    async def dummy(method, data=None):
        calls.append((method, data))
        return {'ok': True}
    bot.api_request = dummy  # type: ignore
    await bot.check_amber()
    assert any(c[0] == 'sendMessage' for c in calls)
    await bot.close()


@pytest.mark.asyncio
async def test_amber_wave_string(tmp_path):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    bot.db.execute("INSERT INTO seas (id, name, lat, lon) VALUES (1, 'sea', 0, 0)")
    bot.db.commit()
    bot.set_amber_sea(1)
    bot.db.execute(
        "INSERT INTO sea_cache (sea_id, updated, current, morning, day, evening, night, wave, morning_wave, day_wave, evening_wave, night_wave) "
        "VALUES (1, ?, 15, 15,15,15,15, '1.6',0,0,0,0)",
        (datetime.utcnow().isoformat(),),
    )
    bot.db.commit()
    await bot.check_amber()
    state = bot.db.execute('SELECT storm_start, active FROM amber_state WHERE sea_id=1').fetchone()
    assert state['active'] == 1
    assert state['storm_start'] is not None
    await bot.close()


@pytest.mark.asyncio
async def test_amber_respects_config_timezone(tmp_path, monkeypatch):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    bot.db.execute("INSERT INTO seas (id, name, lat, lon) VALUES (1, 'sea', 0, 0)")
    bot.db.execute("INSERT INTO channels (chat_id, title) VALUES (-100, 'ch')")
    bot.db.commit()
    bot.set_amber_sea(1)
    bot.db.execute("INSERT INTO amber_channels (channel_id) VALUES (-100)")
    bot.data.upsert_rubric('weather', 'Weather', config={'tz': '+02:00'})
    start = datetime(2024, 1, 1, 4, 0, 0)
    bot.db.execute(
        "UPDATE amber_state SET storm_start=?, active=1 WHERE sea_id=1",
        (start.isoformat(),),
    )
    bot.db.execute(
        "INSERT INTO sea_cache (sea_id, updated, current, morning, day, evening, night, wave, "
        "morning_wave, day_wave, evening_wave, night_wave) VALUES (1, ?, 15, 15,15,15,15, 0.4,0,0,0,0)",
        (datetime.utcnow().isoformat(),),
    )
    bot.db.commit()

    fixed_now = datetime(2024, 1, 1, 6, 0, 0)

    class FixedDatetime(datetime):
        @classmethod
        def utcnow(cls):
            return fixed_now

    monkeypatch.setattr(sys.modules['main'], 'datetime', FixedDatetime)

    calls = []

    async def dummy(method, data=None):
        calls.append((method, data))
        return {'ok': True}

    bot.api_request = dummy  # type: ignore
    await bot.check_amber()

    assert calls
    message = next((payload for method, payload in calls if method == 'sendMessage'), None)
    assert message is not None
    text = message['text']
    assert '06:00 01.01.2024' in text
    assert '08:00 01.01.2024' in text
    await bot.close()
