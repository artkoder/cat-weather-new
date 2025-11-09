import json
import os
import sys
from datetime import date, datetime, timedelta

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from main import Bot

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "dummy")


@pytest.mark.asyncio
async def test_add_list_delete_city(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    calls = []

    async def dummy(method, data=None):
        calls.append((method, data))
        return {"ok": True}

    bot.api_request = dummy  # type: ignore
    await bot.start()

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})

    await bot.handle_update({"message": {"text": "/addcity Paris 48.85 2.35", "from": {"id": 1}}})
    cur = bot.db.execute("SELECT name FROM cities")
    row = cur.fetchone()
    assert row and row["name"] == "Paris"

    await bot.handle_update({"message": {"text": "/cities", "from": {"id": 1}}})
    last = calls[-1]
    assert last[0] == "sendMessage"

    assert "48.850000" in last[1]["text"] and "2.350000" in last[1]["text"]

    cb = last[1]["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
    cid = int(cb.split(":")[1])

    await bot.handle_update(
        {
            "callback_query": {
                "from": {"id": 1},
                "data": cb,
                "message": {"chat": {"id": 1}, "message_id": 10},
                "id": "q",
            }
        }
    )
    cur = bot.db.execute("SELECT * FROM cities WHERE id=?", (cid,))
    assert cur.fetchone() is None
    assert any(c[0] == "editMessageReplyMarkup" for c in calls)

    await bot.close()


@pytest.mark.asyncio
async def test_add_list_delete_sea(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    calls = []

    async def dummy(method, data=None):
        calls.append((method, data))
        return {"ok": True}

    bot.api_request = dummy  # type: ignore
    await bot.start()

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})
    await bot.handle_update({"message": {"text": "/addsea Baltic 54 20", "from": {"id": 1}}})
    cur = bot.db.execute("SELECT name FROM seas")
    row = cur.fetchone()
    assert row and row["name"] == "Baltic"

    await bot.handle_update({"message": {"text": "/seas", "from": {"id": 1}}})
    last = calls[-1]
    assert last[0] == "sendMessage"
    assert "54.000000" in last[1]["text"] and "20.000000" in last[1]["text"]

    cb = last[1]["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
    sid = int(cb.split(":")[1])

    await bot.handle_update(
        {
            "callback_query": {
                "from": {"id": 1},
                "data": cb,
                "message": {"chat": {"id": 1}, "message_id": 10},
                "id": "q",
            }
        }
    )
    cur = bot.db.execute("SELECT * FROM seas WHERE id=?", (sid,))
    assert cur.fetchone() is None
    assert any(c[0] == "editMessageReplyMarkup" for c in calls)

    await bot.close()


@pytest.mark.asyncio
async def test_collect_and_report_weather(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    api_calls = []

    async def dummy(method, data=None):
        api_calls.append((method, data))
        return {"ok": True}

    bot.api_request = dummy  # type: ignore

    async def fetch_dummy(lat, lon):
        return {
            "current": {
                "temperature_2m": 10.0,
                "weather_code": 1,
                "wind_speed_10m": 3.0,
                "is_day": 1,
            }
        }

    bot.fetch_open_meteo = fetch_dummy  # type: ignore

    await bot.start()

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})
    bot.db.execute("INSERT INTO cities (id, name, lat, lon) VALUES (1, 'Paris', 48.85, 2.35)")
    bot.db.commit()

    await bot.collect_weather()

    cur = bot.db.execute("SELECT temperature, weather_code FROM weather_cache_hour WHERE city_id=1")
    row = cur.fetchone()
    assert row and row["temperature"] == 10.0 and row["weather_code"] == 1

    await bot.handle_update({"message": {"text": "/weather", "from": {"id": 1}}})
    assert api_calls[-1][0] == "sendMessage"
    assert "Paris" in api_calls[-1][1]["text"]

    await bot.close()


@pytest.mark.asyncio
async def test_weather_upsert(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    async def fetch1(lat, lon):
        return {
            "current": {
                "temperature_2m": 1.0,
                "weather_code": 1,
                "wind_speed_10m": 1.0,
                "is_day": 1,
            }
        }

    async def fetch2(lat, lon):
        return {
            "current": {
                "temperature_2m": 2.0,
                "weather_code": 1,
                "wind_speed_10m": 1.0,
                "is_day": 1,
            }
        }

    bot.fetch_open_meteo = fetch1  # type: ignore

    await bot.start()

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})
    bot.db.execute("INSERT INTO cities (id, name, lat, lon) VALUES (1, 'Paris', 48.85, 2.35)")
    bot.db.commit()

    await bot.collect_weather()

    bot.fetch_open_meteo = fetch2  # type: ignore
    await bot.collect_weather()

    cur = bot.db.execute("SELECT temperature FROM weather_cache_day WHERE city_id=1")
    row = cur.fetchone()
    assert row and row["temperature"] == 2.0

    await bot.close()


@pytest.mark.asyncio
async def test_weather_now_forces_fetch(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    api_calls = []

    async def dummy(method, data=None):
        api_calls.append((method, data))
        return {"ok": True}

    bot.api_request = dummy  # type: ignore

    count = 0

    async def fetch_dummy(lat, lon):
        nonlocal count
        count += 1
        return {
            "current": {
                "temperature_2m": 5.0,
                "weather_code": 2,
                "wind_speed_10m": 1.0,
                "is_day": 1,
            }
        }

    bot.fetch_open_meteo = fetch_dummy  # type: ignore

    await bot.start()

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})
    bot.db.execute("INSERT INTO cities (id, name, lat, lon) VALUES (1, 'Rome', 41.9, 12.5)")
    bot.db.commit()

    await bot.handle_update({"message": {"text": "/weather now", "from": {"id": 1}}})
    assert count == 1
    assert api_calls[-1][0] == "sendMessage"

    await bot.close()


@pytest.mark.asyncio
async def test_weather_retry_logic(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    count = 0

    async def fetch_fail(lat, lon):
        nonlocal count
        count += 1
        return None

    bot.fetch_open_meteo = fetch_fail  # type: ignore

    await bot.start()

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})
    bot.db.execute("INSERT INTO cities (id, name, lat, lon) VALUES (1, 'Rome', 41.9, 12.5)")
    bot.db.commit()

    await bot.collect_weather()
    assert count == 1
    # second call within a minute should not trigger another request
    await bot.collect_weather()
    assert count == 1

    # pretend one minute passed
    attempts, ts = bot.failed_fetches[1]
    bot.failed_fetches[1] = (attempts, ts - timedelta(minutes=1, seconds=1))
    await bot.collect_weather()
    assert count == 2

    # set three attempts in last second, should skip
    bot.failed_fetches[1] = (3, datetime.utcnow())
    await bot.collect_weather()
    assert count == 2

    # after thirty minutes allowed again
    bot.failed_fetches[1] = (3, datetime.utcnow() - timedelta(minutes=31))
    await bot.collect_weather()
    assert count == 3

    await bot.close()


@pytest.mark.asyncio
async def test_register_weather_post(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    api_calls = []

    async def dummy(method, data=None):
        api_calls.append((method, data))
        if method == "forwardMessage":

            return {
                "ok": True,
                "result": {
                    "message_id": 99,
                    "text": "orig",
                    "reply_markup": {"inline_keyboard": [[{"text": "b", "url": "u"}]]},
                },
            }

        return {"ok": True, "result": {"message_id": 1}}

    bot.api_request = dummy  # type: ignore

    async def fetch_dummy(lat, lon):

        return {
            "current": {
                "temperature_2m": 15.0,
                "weather_code": 1,
                "wind_speed_10m": 2.0,
                "is_day": 1,
            }
        }

    bot.fetch_open_meteo = fetch_dummy  # type: ignore

    await bot.start()

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})
    await bot.handle_update({"message": {"text": "/addcity Paris 48.85 2.35", "from": {"id": 1}}})

    await bot.handle_update(
        {
            "message": {
                "text": "/regweather https://t.me/c/123/5 Paris {1|temperature}",
                "from": {"id": 1},
            }
        }
    )

    cur = bot.db.execute(
        "SELECT chat_id, message_id, template, base_text, base_caption, reply_markup FROM weather_posts"
    )

    row = cur.fetchone()
    assert row and row["chat_id"] == -100123 and row["message_id"] == 5
    assert row["template"] == "Paris {1|temperature}"
    assert row["base_text"] == "orig"

    assert row["base_caption"] is None
    assert json.loads(row["reply_markup"])["inline_keyboard"][0][0]["text"] == "b"

    await bot.collect_weather()
    assert any(c[0] == "editMessageText" for c in api_calls)
    payload = [c[1] for c in api_calls if c[0] == "editMessageText"][0]
    assert payload["reply_markup"]["inline_keyboard"][0][0]["url"] == "u"

    await bot.handle_update({"message": {"text": "/weatherposts update", "from": {"id": 1}}})

    assert api_calls[-2][0] == "editMessageText"
    msg = api_calls[-1]
    assert msg[0] == "sendMessage"
    assert "https://t.me/c/123/5" in msg[1]["text"]
    assert "15.0" in msg[1]["text"]

    await bot.close()


@pytest.mark.asyncio
async def test_register_weather_post_caption(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    api_calls = []

    async def dummy(method, data=None):
        api_calls.append((method, data))
        if method == "forwardMessage":

            return {
                "ok": True,
                "result": {
                    "message_id": 99,
                    "caption": "orig cap",
                    "reply_markup": {"inline_keyboard": [[{"text": "b2", "url": "u2"}]]},
                },
            }

        return {"ok": True, "result": {"message_id": 1}}

    bot.api_request = dummy  # type: ignore

    async def fetch_dummy(lat, lon):

        return {
            "current": {
                "temperature_2m": 15.0,
                "weather_code": 1,
                "wind_speed_10m": 2.0,
                "is_day": 1,
            }
        }

    bot.fetch_open_meteo = fetch_dummy  # type: ignore

    await bot.start()

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})
    await bot.handle_update({"message": {"text": "/addcity Paris 48.85 2.35", "from": {"id": 1}}})

    await bot.handle_update(
        {
            "message": {
                "text": "/regweather https://t.me/c/123/5 Paris {1|temperature}",
                "from": {"id": 1},
            }
        }
    )

    cur = bot.db.execute("SELECT base_text, base_caption, reply_markup FROM weather_posts")
    row = cur.fetchone()
    assert row["base_text"] is None
    assert row["base_caption"] == "orig cap"

    assert json.loads(row["reply_markup"])["inline_keyboard"][0][0]["text"] == "b2"

    await bot.collect_weather()
    assert any(c[0] == "editMessageCaption" for c in api_calls)
    payload = [c[1] for c in api_calls if c[0] == "editMessageCaption"][0]
    assert payload["reply_markup"]["inline_keyboard"][0][0]["url"] == "u2"

    await bot.close()


@pytest.mark.asyncio
async def test_regweather_strips_header(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    async def dummy(method, data=None):
        if method == "forwardMessage":
            return {
                "ok": True,
                "result": {"message_id": 99, "text": "old\u2219orig"},
            }
        return {"ok": True, "result": {"message_id": 1}}

    bot.api_request = dummy  # type: ignore

    await bot.start()
    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})

    await bot.handle_update(
        {"message": {"text": "/regweather https://t.me/c/1/1 t {1|temperature}", "from": {"id": 1}}}
    )

    row = bot.db.execute("SELECT base_text FROM weather_posts").fetchone()
    assert row["base_text"] == "orig"

    await bot.close()


@pytest.mark.asyncio
async def test_night_clear_emoji(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    api_calls = []

    async def dummy(method, data=None):
        api_calls.append((method, data))
        return {"ok": True}

    bot.api_request = dummy  # type: ignore

    async def fetch_dummy(lat, lon):
        return {
            "current": {
                "temperature_2m": 11.0,
                "weather_code": 0,
                "wind_speed_10m": 1.0,
                "is_day": 0,
            }
        }

    bot.fetch_open_meteo = fetch_dummy  # type: ignore

    await bot.start()
    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})
    bot.db.execute("INSERT INTO cities (id, name, lat, lon) VALUES (1, 'Night', 0.0, 0.0)")
    bot.db.commit()

    await bot.collect_weather()

    cur = bot.db.execute("SELECT weather_code, is_day FROM weather_cache_hour WHERE city_id=1")
    row = cur.fetchone()
    assert row["weather_code"] == 0 and row["is_day"] == 0

    await bot.handle_update({"message": {"text": "/weather", "from": {"id": 1}}})
    # last sendMessage should include moon emoji U+1F319
    assert "\U0001f319" in api_calls[-1][1]["text"]

    await bot.close()


@pytest.mark.asyncio
async def test_add_sea_and_template(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    api_calls = []

    async def dummy(method, data=None):
        api_calls.append((method, data))
        if method == "forwardMessage":
            return {"ok": True, "result": {"message_id": 99, "text": "orig"}}
        return {"ok": True, "result": {"message_id": 1}}

    bot.api_request = dummy  # type: ignore

    async def fetch_sea(lat, lon):
        tomorrow = date.today() + timedelta(days=1)
        times = [
            datetime.utcnow().isoformat(),
            datetime.combine(tomorrow, datetime.min.time()).isoformat(),
            datetime.combine(tomorrow, datetime.min.time().replace(hour=6)).isoformat(),
            datetime.combine(tomorrow, datetime.min.time().replace(hour=12)).isoformat(),
            datetime.combine(tomorrow, datetime.min.time().replace(hour=18)).isoformat(),
        ]
        temps = [19.0, 20.0, 21.0, 22.0, 23.0]
        return {"hourly": {"water_temperature": temps, "time": times}}

    bot.fetch_open_meteo_sea = fetch_sea  # type: ignore

    await bot.start()
    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})
    await bot.handle_update({"message": {"text": "/addsea Black 40 30", "from": {"id": 1}}})
    cur = bot.db.execute("SELECT name FROM seas")
    assert cur.fetchone()["name"] == "Black"

    await bot.handle_update(
        {
            "message": {
                "text": "/regweather https://t.me/c/1/1 {1|seatemperature}",
                "from": {"id": 1},
            }
        }
    )
    await bot.collect_sea()
    assert any(c[0] == "editMessageText" for c in api_calls)
    text = [c[1]["text"] for c in api_calls if c[0] == "editMessageText"][0]

    assert "\U0001f30a" in text and "19.0\u00b0C" in text

    await bot.close()


@pytest.mark.asyncio
async def test_add_sea_comma_coords(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    async def dummy(method, data=None):
        return {"ok": True}

    bot.api_request = dummy  # type: ignore

    await bot.start()
    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})
    await bot.handle_update({"message": {"text": "/addsea Baltic 54.1, 19.2", "from": {"id": 1}}})

    cur = bot.db.execute("SELECT lat, lon FROM seas WHERE name='Baltic'")
    row = cur.fetchone()
    assert round(row["lat"], 1) == 54.1 and round(row["lon"], 1) == 19.2

    await bot.close()


@pytest.mark.asyncio
async def test_weather_lists_sea(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    api_calls = []

    async def dummy(method, data=None):
        api_calls.append((method, data))
        return {"ok": True}

    bot.api_request = dummy  # type: ignore

    async def fetch_city(lat, lon):
        return {
            "current": {
                "temperature_2m": 12.0,
                "weather_code": 1,
                "wind_speed_10m": 3.0,
                "is_day": 1,
            }
        }

    tomorrow = date.today() + timedelta(days=1)
    times = [
        datetime.utcnow().isoformat(),
        datetime.combine(tomorrow, datetime.min.time()).isoformat(),
        datetime.combine(tomorrow, datetime.min.time().replace(hour=6)).isoformat(),
        datetime.combine(tomorrow, datetime.min.time().replace(hour=12)).isoformat(),
        datetime.combine(tomorrow, datetime.min.time().replace(hour=18)).isoformat(),
    ]
    temps = [19.0, 20.0, 21.0, 22.0, 23.0]

    async def fetch_sea(lat, lon):
        return {"hourly": {"water_temperature": temps, "time": times}}

    bot.fetch_open_meteo = fetch_city  # type: ignore
    bot.fetch_open_meteo_sea = fetch_sea  # type: ignore

    await bot.start()
    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})
    await bot.handle_update({"message": {"text": "/addcity Paris 48 2", "from": {"id": 1}}})
    await bot.handle_update({"message": {"text": "/addsea Baltic 40 30", "from": {"id": 1}}})

    await bot.collect_weather()
    await bot.collect_sea()

    await bot.handle_update({"message": {"text": "/weather", "from": {"id": 1}}})
    msg = api_calls[-1]
    assert msg[0] == "sendMessage"
    assert "Paris" in msg[1]["text"] and "Baltic" in msg[1]["text"]
    assert "\U0001f30a" in msg[1]["text"]

    await bot.close()


@pytest.mark.asyncio
async def test_weather_now_fetches_sea(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    async def dummy(method, data=None):
        return {"ok": True}

    bot.api_request = dummy  # type: ignore

    count_city = 0

    async def fetch_city(lat, lon):
        nonlocal count_city
        count_city += 1
        return {
            "current": {
                "temperature_2m": 5.0,
                "weather_code": 1,
                "wind_speed_10m": 1.0,
                "is_day": 1,
            }
        }

    count_sea = 0
    tomorrow = date.today() + timedelta(days=1)
    times = [
        datetime.utcnow().isoformat(),
        datetime.combine(tomorrow, datetime.min.time()).isoformat(),
        datetime.combine(tomorrow, datetime.min.time().replace(hour=6)).isoformat(),
        datetime.combine(tomorrow, datetime.min.time().replace(hour=12)).isoformat(),
        datetime.combine(tomorrow, datetime.min.time().replace(hour=18)).isoformat(),
    ]
    temps = [18.0, 19.0, 20.0, 21.0, 22.0]

    async def fetch_sea(lat, lon):
        nonlocal count_sea
        count_sea += 1
        return {"hourly": {"water_temperature": temps, "time": times}}

    bot.fetch_open_meteo = fetch_city  # type: ignore
    bot.fetch_open_meteo_sea = fetch_sea  # type: ignore

    await bot.start()
    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})
    bot.db.execute("INSERT INTO cities (id, name, lat, lon) VALUES (1, 'Rome', 1, 1)")
    bot.db.execute("INSERT INTO seas (id, name, lat, lon) VALUES (1, 'Med', 2, 2)")
    bot.db.commit()

    await bot.handle_update({"message": {"text": "/weather now", "from": {"id": 1}}})
    assert count_city == 1 and count_sea == 1

    await bot.close()


@pytest.mark.asyncio
async def test_remove_weather_post(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    api_calls = []

    async def dummy(method, data=None):
        api_calls.append((method, data))
        if method == "forwardMessage":
            return {"ok": True, "result": {"message_id": 99, "text": "orig"}}
        return {"ok": True}

    bot.api_request = dummy  # type: ignore

    await bot.start()
    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})
    await bot.handle_update({"message": {"text": "/addcity Paris 0 0", "from": {"id": 1}}})
    await bot.handle_update(
        {
            "message": {
                "text": "/regweather https://t.me/c/123/5 t {1|temperature}",
                "from": {"id": 1},
            }
        }
    )

    bot.db.execute(
        "INSERT INTO weather_cache_hour (city_id, timestamp, temperature, weather_code, wind_speed, is_day)"
        " VALUES (1, ?, 15.0, 1, 3.0, 1)",
        (datetime.utcnow().isoformat(),),
    )
    bot.db.commit()

    await bot.update_weather_posts()
    api_calls.clear()

    await bot.handle_update({"message": {"text": "/weatherposts", "from": {"id": 1}}})
    cb = api_calls[-1][1]["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
    assert cb.startswith("wpost_del:")

    await bot.handle_update(
        {
            "callback_query": {
                "from": {"id": 1},
                "data": cb,
                "message": {"chat": {"id": 1}, "message_id": 10},
                "id": "q",
            }
        }
    )

    assert bot.db.execute("SELECT COUNT(*) FROM weather_posts").fetchone()[0] == 0
    assert any(c[0] == "editMessageText" for c in api_calls)

    await bot.close()
