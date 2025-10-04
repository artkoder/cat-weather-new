import json
import os
import pytest
import sys
import sqlite3
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from main import Bot, Job

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

async def test_photo_triggers_ingest_and_vision(tmp_path, caplog):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    bot.set_asset_channel(-100123)
    message = {
        'message_id': 21,
        'date': int(datetime.utcnow().timestamp()),
        'chat': {'id': -100123},
        'caption': '#котопогода свежий кадр',
        'from': {'id': 777, 'username': 'catlover'},
        'photo': [
            {'file_id': 'small', 'file_unique_id': 'uniq_small', 'file_size': 10, 'width': 90, 'height': 90},
            {'file_id': 'large', 'file_unique_id': 'uniq_large', 'file_size': 20, 'width': 1920, 'height': 1080},
        ],
    }
    with caplog.at_level('INFO'):
        await bot.handle_message(message)
    rows = bot.db.execute(
        "SELECT id, name, payload FROM jobs_queue ORDER BY id"
    ).fetchall()
    assert rows and rows[0]['name'] == 'ingest'
    payload = json.loads(rows[0]['payload'])
    asset_id = payload['asset_id']
    asset = bot.data.get_asset(asset_id)
    assert asset.kind == 'photo'
    assert asset.file_id == 'large'
    assert asset.author_user_id == 777
    await bot.handle_message(message)
    ingest_jobs = bot.db.execute(
        "SELECT COUNT(*) FROM jobs_queue WHERE name='ingest'"
    ).fetchone()[0]
    assert ingest_jobs == 1
    job = Job(
        id=rows[0]['id'],
        name='ingest',
        payload=payload,
        status='queued',
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    with caplog.at_level('INFO'):
        await bot._job_ingest(job)
    vision_rows = bot.db.execute(
        "SELECT name, payload FROM jobs_queue WHERE name='vision'"
    ).fetchall()
    assert vision_rows
    vision_payload = json.loads(vision_rows[0]['payload'])
    assert vision_payload['asset_id'] == asset_id
    ingest_messages = [rec.message for rec in caplog.records if 'Scheduled ingest job' in rec.message]
    ingest_log = bool(ingest_messages)
    vision_log = any('queued for vision job' in rec.message for rec in caplog.records)
    assert ingest_log and vision_log
    assert any('reason=new_message' in msg for msg in ingest_messages)
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


def test_strip_header():
    assert Bot.strip_header('🌊 16°C∙text') == 'text'
    assert Bot.strip_header('prefix ∙ data') == 'prefix ∙ data'



@pytest.mark.asyncio
async def test_migrate_legacy_weather_channels(tmp_path):
    db_path = tmp_path / 'db.sqlite'
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE weather_publish_channels ("
        " channel_id INTEGER PRIMARY KEY,"
        " post_time TEXT NOT NULL,"
        " last_published_at TEXT"
        ")"
    )
    conn.execute(
        "INSERT INTO weather_publish_channels (channel_id, post_time, last_published_at) VALUES (?, ?, ?)",
        (-100500, '10:30', '2024-01-01T09:00:00')
    )
    conn.commit()
    conn.close()

    bot = Bot('dummy', str(db_path))
    channels = bot.list_weather_channels()
    assert channels and channels[0]['channel_id'] == -100500
    assert channels[0]['post_time'] == '10:30'
    assert channels[0]['last_published_at'] == '2024-01-01T09:00:00'
    assert (
        bot.db.execute("SELECT name FROM sqlite_master WHERE name='weather_publish_channels'").fetchone()
        is None
    )
    await bot.close()
