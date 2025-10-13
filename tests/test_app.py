import logging
import os
import re
import sys
from typing import Any

import pytest
from aiohttp import web
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from main import create_app, Bot
from data_access import create_device, insert_upload, set_upload_status

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "dummy")
os.environ.setdefault("WEBHOOK_URL", "https://example.com")

@pytest.mark.asyncio
async def test_startup_cleanup():
    app = create_app()

    async def dummy(method, data=None):
        return {"ok": True}

    app['bot'].api_request = dummy  # type: ignore

    runner = web.AppRunner(app)
    await runner.setup()
    await runner.cleanup()

@pytest.mark.asyncio
async def test_registration_queue(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    calls = []

    async def dummy(method, data=None):
        calls.append((method, data))
        return {"ok": True}

    bot.api_request = dummy  # type: ignore
    await bot.start()

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})
    row = bot.get_user(1)
    assert row and row["is_superadmin"] == 1

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 2}}})
    assert bot.is_pending(2)

    # reject user 2 and ensure they cannot re-register
    bot.reject_user(2)
    await bot.handle_update({"message": {"text": "/start", "from": {"id": 2}}})
    assert bot.is_rejected(2)
    assert not bot.is_pending(2)
    assert calls[-1][0] == 'sendMessage'
    assert calls[-1][1]['text'] == 'Access denied by administrator'

    await bot.close()


@pytest.mark.asyncio
async def test_superadmin_user_management(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    calls = []

    async def dummy(method, data=None):
        calls.append((method, data))
        return {"ok": True}

    bot.api_request = dummy  # type: ignore
    await bot.start()

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})
    await bot.handle_update({"message": {"text": "/start", "from": {"id": 2}}})
    await bot.handle_update({"message": {"text": "/pending", "from": {"id": 1}}})
    assert bot.is_pending(2)
    pending_msg = calls[-1]
    assert pending_msg[0] == 'sendMessage'
    assert pending_msg[1]['reply_markup']['inline_keyboard'][0][0]['callback_data'] == 'approve:2'
    assert 'tg://user?id=2' in pending_msg[1]['text']
    assert pending_msg[1]['parse_mode'] == 'Markdown'

    await bot.handle_update({"message": {"text": "/approve 2", "from": {"id": 1}}})
    assert bot.get_user(2)
    assert not bot.is_pending(2)

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 3}}})
    await bot.handle_update({"message": {"text": "/reject 3", "from": {"id": 1}}})
    assert not bot.is_pending(3)
    assert not bot.get_user(3)

    await bot.handle_update({"message": {"text": "/remove_user 2", "from": {"id": 1}}})
    assert not bot.get_user(2)

    await bot.close()


@pytest.mark.asyncio
async def test_list_users_links(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    calls = []

    async def dummy(method, data=None):
        calls.append((method, data))
        return {"ok": True}

    bot.api_request = dummy  # type: ignore
    await bot.start()

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1, "username": "admin"}}})
    await bot.handle_update({"message": {"text": "/start", "from": {"id": 2, "username": "user"}}})
    bot.approve_user(2)

    await bot.handle_update({"message": {"text": "/list_users", "from": {"id": 1}}})
    msg = calls[-1][1]
    assert msg['parse_mode'] == 'Markdown'
    assert 'tg://user?id=1' in msg['text']
    assert 'tg://user?id=2' in msg['text']

    await bot.close()


@pytest.mark.asyncio
async def test_set_timezone(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    calls = []

    async def dummy(method, data=None):
        calls.append((method, data))
        return {"ok": True}

    bot.api_request = dummy  # type: ignore
    await bot.start()

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})
    await bot.handle_update({"message": {"text": "/tz +03:00", "from": {"id": 1}}})

    cur = bot.db.execute("SELECT tz_offset FROM users WHERE user_id=1")
    row = cur.fetchone()
    assert row["tz_offset"] == "+03:00"

    await bot.close()


@pytest.mark.asyncio
async def test_channel_tracking(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    calls = []

    async def dummy(method, data=None):
        calls.append((method, data))
        return {"ok": True}

    bot.api_request = dummy  # type: ignore
    await bot.start()

    # register superadmin
    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})

    # bot added to channel
    await bot.handle_update({
        "my_chat_member": {
            "chat": {"id": -100, "title": "Chan"},
            "new_chat_member": {"status": "administrator"}
        }
    })
    cur = bot.db.execute('SELECT title FROM channels WHERE chat_id=?', (-100,))
    row = cur.fetchone()
    assert row and row["title"] == "Chan"

    await bot.handle_update({"message": {"text": "/channels", "from": {"id": 1}}})
    assert calls[-1][1]["text"] == "Chan (-100)"

    # non-admin cannot list channels
    await bot.handle_update({"message": {"text": "/start", "from": {"id": 2}}})
    await bot.handle_update({"message": {"text": "/channels", "from": {"id": 2}}})
    assert calls[-1][1]["text"] == "Not authorized"

    # bot removed from channel
    await bot.handle_update({
        "my_chat_member": {
            "chat": {"id": -100, "title": "Chan"},
            "new_chat_member": {"status": "left"}
        }
    })
    cur = bot.db.execute('SELECT * FROM channels WHERE chat_id=?', (-100,))
    assert cur.fetchone() is None

    await bot.close()


@pytest.mark.asyncio
async def test_pair_command_generates_token(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    calls = []

    async def dummy(method, data=None):
        calls.append((method, data))
        return {"ok": True}

    bot.api_request = dummy  # type: ignore
    await bot.start()

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})
    calls.clear()

    await bot.handle_update({"message": {"text": "/pair", "from": {"id": 1}}})
    assert calls[-1][0] == "sendMessage"
    assert "Код для привязки" in calls[-1][1]["text"]

    row = bot.db.execute(
        "SELECT code, used_at FROM pairing_tokens WHERE user_id=?", (1,)
    ).fetchone()
    assert row is not None
    assert row["used_at"] is None
    initial_code = row["code"]

    calls.clear()
    await bot.handle_update({"message": {"text": "/pair", "from": {"id": 1}}})
    assert calls[-1][0] == "sendMessage"
    assert "Активный код" in calls[-1][1]["text"]
    assert initial_code in calls[-1][1]["text"]

    query = {
        "id": "regen-1",
        "from": {"id": 1},
        "data": "pairing_regen",
        "message": {"message_id": 42, "chat": {"id": 1}},
    }
    await bot.handle_callback(query)

    assert any(method == "answerCallbackQuery" for method, _ in calls)
    new_row = bot.db.execute(
        "SELECT code FROM pairing_tokens WHERE user_id=?", (1,)
    ).fetchone()
    assert new_row is not None
    assert new_row["code"] != initial_code

    await bot.close()


@pytest.mark.asyncio
async def test_mobile_command_and_callbacks(tmp_path, caplog):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    calls: list[tuple[str, Any]] = []
    multipart_calls: list[tuple[str, Any, Any]] = []

    async def dummy_api(method, data=None):
        calls.append((method, data))
        return {"ok": True, "result": {}}

    async def dummy_multipart(method, data=None, *, files=None):
        multipart_calls.append((method, data, files))
        return {"ok": True, "result": {"message_id": 77}}

    bot.api_request = dummy_api  # type: ignore
    bot.api_request_multipart = dummy_multipart  # type: ignore
    await bot.start()
    caplog.set_level(logging.INFO)

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})

    with bot.db:
        create_device(
            bot.db,
            device_id="dev-1",
            user_id=1,
            name="Office Pixel",
            secret="secret-1",
        )
        create_device(
            bot.db,
            device_id="dev-2",
            user_id=1,
            name="Backup Phone",
            secret="secret-2",
        )

    calls.clear()
    multipart_calls.clear()

    caplog.clear()
    await bot.handle_update({"message": {"text": "/mobile", "from": {"id": 1}}})

    assert multipart_calls, "sendPhoto should be used for the QR card"
    method, payload, files = multipart_calls[-1]
    assert method == "sendPhoto"
    assert payload["chat_id"] == 1
    assert "Код" in payload["caption"]
    keyboard = payload["reply_markup"]["inline_keyboard"]
    callbacks = [btn["callback_data"] for row in keyboard for btn in row]
    assert "pair:new" in callbacks
    assert any(cb == "dev:revoke:dev-1" for cb in callbacks)
    assert any(cb == "dev:rotate:dev-1" for cb in callbacks)
    assert "photo" in files
    photo_entry = files["photo"]
    assert hasattr(photo_entry[1], "read")

    mobile_logs = [rec for rec in caplog.records if rec.message == "MOBILE_PAIR_UI"]
    assert mobile_logs, "Expected MOBILE_PAIR_UI log entry"
    first_mobile_log = mobile_logs[-1]
    assert first_mobile_log.user_id == 1
    assert first_mobile_log.has_devices is True

    row = bot.db.execute(
        "SELECT code FROM pairing_tokens WHERE user_id=?", (1,)
    ).fetchone()
    assert row is not None
    initial_code = row["code"]
    assert first_mobile_log.code_len == len(initial_code)

    caplog.clear()
    await bot.handle_update({"message": {"text": "/mobile", "from": {"id": 1}}})

    reuse_logs = [rec for rec in caplog.records if rec.message == "MOBILE_PAIR_UI"]
    assert reuse_logs, "Expected MOBILE_PAIR_UI log when reusing token"
    reuse_log = reuse_logs[-1]
    assert reuse_log.user_id == 1
    assert reuse_log.has_devices is True
    assert reuse_log.code_len == len(initial_code)

    row = bot.db.execute(
        "SELECT code FROM pairing_tokens WHERE user_id=?", (1,)
    ).fetchone()
    assert row is not None
    assert row["code"] == initial_code

    multipart_calls.clear()
    await bot.handle_callback(
        {
            "id": "cb-new",
            "from": {"id": 1},
            "data": "pair:new",
            "message": {"message_id": 77, "chat": {"id": 1}},
        }
    )
    assert multipart_calls and multipart_calls[-1][0] == "editMessageMedia"
    row = bot.db.execute(
        "SELECT code FROM pairing_tokens WHERE user_id=?", (1,)
    ).fetchone()
    assert row and row["code"] != initial_code

    calls.clear()
    multipart_calls.clear()
    await bot.handle_callback(
        {
            "id": "cb-revoke",
            "from": {"id": 1},
            "data": "dev:revoke:dev-1",
            "message": {"message_id": 77, "chat": {"id": 1}},
        }
    )
    assert any(
        method == "answerCallbackQuery" and data["text"] == "Устройство отозвано"
        for method, data in calls
    )
    revoked_row = bot.db.execute(
        "SELECT revoked_at FROM devices WHERE id=?",
        ("dev-1",),
    ).fetchone()
    assert revoked_row and revoked_row["revoked_at"]

    old_secret = bot.db.execute(
        "SELECT secret FROM devices WHERE id=?",
        ("dev-2",),
    ).fetchone()["secret"]

    calls.clear()
    multipart_calls.clear()
    await bot.handle_callback(
        {
            "id": "cb-rotate",
            "from": {"id": 1},
            "data": "dev:rotate:dev-2",
            "message": {"message_id": 77, "chat": {"id": 1}},
        }
    )
    assert any(
        method == "answerCallbackQuery"
        and data["text"].startswith("Новый секрет")
        for method, data in calls
    )
    new_secret = bot.db.execute(
        "SELECT secret FROM devices WHERE id=?",
        ("dev-2",),
    ).fetchone()["secret"]
    assert new_secret != old_secret

    await bot.close()


@pytest.mark.asyncio
async def test_mobile_stats_command(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    calls: list[tuple[str, Any]] = []

    async def dummy_api(method, data=None):
        calls.append((method, data))
        return {"ok": True, "result": {}}

    bot.api_request = dummy_api  # type: ignore
    await bot.start()

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})

    with bot.db:
        create_device(
            bot.db,
            device_id="dev-1",
            user_id=1,
            name="Primary Phone",
            secret="secret-1",
        )
        create_device(
            bot.db,
            device_id="dev-2",
            user_id=1,
            name="Backup",
            secret="secret-2",
        )

        def _record(device_id: str, key: str, days: int) -> None:
            upload_id = insert_upload(
                bot.db,
                id=f"{device_id}-{key}",
                device_id=device_id,
                idempotency_key=f"key-{device_id}-{key}",
                file_ref=f"file-{key}",
            )
            set_upload_status(bot.db, id=upload_id, status="processing")
            set_upload_status(bot.db, id=upload_id, status="done")
            stamp = (datetime.utcnow() - timedelta(days=days)).isoformat()
            bot.db.execute(
                "UPDATE uploads SET created_at=?, updated_at=? WHERE id=?",
                (stamp, stamp, upload_id),
            )

        _record("dev-1", "today", 0)
        _record("dev-1", "recent", 3)
        _record("dev-1", "month", 20)
        _record("dev-2", "old", 40)

    calls.clear()
    await bot.handle_update({"message": {"text": "/mobile_stats", "from": {"id": 1}}})

    assert calls, "mobile stats should send a message"
    method, payload = calls[-1]
    assert method == "sendMessage"
    text = payload["text"]
    assert "Total: 4" in text
    assert "Today: 1" in text
    assert "7d: 2" in text
    assert "30d: 3" in text
    assert "Top devices" in text
    assert "Primary Phone" in text and "— 3" in text
    assert "Backup" in text and "— 1" in text

    await bot.close()


@pytest.mark.asyncio
async def test_schedule_flow(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    calls = []

    async def dummy(method, data=None):
        calls.append((method, data))
        return {"ok": True}

    bot.api_request = dummy  # type: ignore
    await bot.start()

    # register superadmin
    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})

    # bot added to two channels
    await bot.handle_update({
        "my_chat_member": {
            "chat": {"id": -100, "title": "Chan1"},
            "new_chat_member": {"status": "administrator"}
        }
    })
    await bot.handle_update({
        "my_chat_member": {
            "chat": {"id": -101, "title": "Chan2"},
            "new_chat_member": {"status": "administrator"}
        }
    })

    # forward a message to schedule
    await bot.handle_update({
        "message": {
            "forward_from_chat": {"id": 500},
            "forward_from_message_id": 7,
            "from": {"id": 1}
        }
    })
    assert calls[-1][1]["reply_markup"]["inline_keyboard"][-1][0]["callback_data"] == "chdone"

    # select channels and finish
    await bot.handle_update({"callback_query": {"from": {"id": 1}, "data": "addch:-100", "id": "q"}})
    await bot.handle_update({"callback_query": {"from": {"id": 1}, "data": "addch:-101", "id": "q"}})
    await bot.handle_update({"callback_query": {"from": {"id": 1}, "data": "chdone", "id": "q"}})

    time_str = (datetime.now() + timedelta(minutes=5)).strftime("%H:%M")
    await bot.handle_update({"message": {"text": time_str, "from": {"id": 1}}})
    assert any(c[0] == "forwardMessage" for c in calls)

    cur = bot.db.execute("SELECT target_chat_id FROM schedule ORDER BY target_chat_id")
    rows = [r["target_chat_id"] for r in cur.fetchall()]
    assert rows == [-101, -100] or rows == [-100, -101]

    # list schedules
    await bot.handle_update({"message": {"text": "/scheduled", "from": {"id": 1}}})
    forward_calls = [c for c in calls if c[0] == "forwardMessage"]
    assert forward_calls
    last_msg = [c for c in calls if c[0] == "sendMessage" and c[1].get("reply_markup")][-1]
    assert "cancel" in last_msg[1]["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
    assert re.search(r"\d{2}:\d{2} \d{2}\.\d{2}\.\d{4}", last_msg[1]["text"])
    assert "Chan1" in last_msg[1]["text"] or "Chan2" in last_msg[1]["text"]

    # cancel first schedule
    cur = bot.db.execute("SELECT id FROM schedule ORDER BY id")
    sid = cur.fetchone()["id"]
    await bot.handle_update({"callback_query": {"from": {"id": 1}, "data": f"cancel:{sid}", "id": "c"}})
    cur = bot.db.execute("SELECT * FROM schedule WHERE id=?", (sid,))
    assert cur.fetchone() is None

    await bot.close()


@pytest.mark.asyncio
async def test_scheduler_process_due(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    calls = []

    async def dummy(method, data=None):
        calls.append((method, data))
        return {"ok": True}

    bot.api_request = dummy  # type: ignore
    await bot.start()

    # register superadmin
    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})

    due_time = (datetime.utcnow() - timedelta(seconds=1)).isoformat()
    bot.add_schedule(500, 5, {-100}, due_time)

    await bot.process_due()

    cur = bot.db.execute("SELECT sent FROM schedule")
    row = cur.fetchone()
    assert row["sent"] == 1
    assert calls[-1][0] == "forwardMessage"

    await bot.close()


@pytest.mark.asyncio
async def test_add_button(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    calls = []

    forward_resps = [
        {
            "ok": True,
            "result": {
                "message_id": 42,
                "reply_markup": {"inline_keyboard": [[{"text": "old", "url": "u"}]]},
            },
        },
        {
            "ok": True,
            "result": {
                "message_id": 42,
                "reply_markup": {
                    "inline_keyboard": [[{"text": "old", "url": "u"}, {"text": "btn", "url": "https://example.com"}]]
                },
            },
        },
    ]
    count = 0

    async def dummy(method, data=None):
        nonlocal count
        calls.append((method, data))
        if method == "getChat":
            return {"ok": True, "result": {"id": -100123}}
        if method == "forwardMessage":
            resp = forward_resps[count]
            count += 1
            return resp
        return {"ok": True}

    bot.api_request = dummy  # type: ignore
    await bot.start()

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})

    await bot.handle_update({
        "message": {
            "text": "/addbutton https://t.me/c/123/5 btn https://example.com",
            "from": {"id": 1},
        }
    })
    edit_calls = [c for c in calls if c[0] == "editMessageReplyMarkup"]
    assert len(edit_calls) == 1
    assert len(edit_calls[0][1]["reply_markup"]["inline_keyboard"]) == 2

    await bot.handle_update({
        "message": {
            "text": "/addbutton https://t.me/c/123/5 ask locals https://example.com",
            "from": {"id": 1},
        }
    })

    # check that button text with spaces is parsed correctly
    edit_calls = [c for c in calls if c[0] == "editMessageReplyMarkup"]
    assert len(edit_calls) == 2
    payload = edit_calls[-1][1]
    assert len(payload["reply_markup"]["inline_keyboard"]) == 3
    assert payload["reply_markup"]["inline_keyboard"][2][0]["text"] == "ask locals"

    await bot.close()


@pytest.mark.asyncio
async def test_delete_button(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    calls = []

    async def dummy(method, data=None):
        calls.append((method, data))
        return {"ok": True}

    bot.api_request = dummy  # type: ignore
    await bot.start()

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})

    await bot.handle_update({
        "message": {
            "text": "/delbutton https://t.me/c/123/5",
            "from": {"id": 1},
        }
    })

    assert calls[-1][0] == "editMessageReplyMarkup"
    assert calls[-1][1]["reply_markup"] == {}

    await bot.close()


@pytest.mark.asyncio
async def test_add_weather_button(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    calls = []
    async def dummy(method, data=None):
        calls.append((method, data))
        if method == "forwardMessage":
            return {
                "ok": True,
                "result": {"message_id": 11, "reply_markup": {"inline_keyboard": []}},
            }
        return {"ok": True}

    bot.api_request = dummy  # type: ignore
    bot.set_latest_weather_post(-100, 7)
    await bot.start()


    bot.db.execute("INSERT INTO cities (id, name, lat, lon) VALUES (1, 'c', 0, 0)")
    bot.db.execute(
        "INSERT INTO weather_cache_hour (city_id, timestamp, temperature, weather_code, wind_speed, is_day) VALUES (1, ?, 15.0, 1, 3, 1)",
        (datetime.utcnow().isoformat(),),
    )
    bot.db.commit()

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})

    await bot.handle_update({
        "message": {

            "text": "/addweatherbutton https://t.me/c/123/5 K. {1|temperature}",

            "from": {"id": 1},
        }
    })

    assert any(c[0] == "editMessageReplyMarkup" for c in calls)
    payload = [c[1] for c in calls if c[0] == "editMessageReplyMarkup"][0]

    assert len(payload["reply_markup"]["inline_keyboard"]) == 1
    assert payload["reply_markup"]["inline_keyboard"][0][0]["url"].endswith("/7")

    assert "\u00B0C" in payload["reply_markup"]["inline_keyboard"][0][0]["text"]

    calls.clear()
    await bot.update_weather_buttons()
    up_payload = [c[1] for c in calls if c[0] == "editMessageReplyMarkup"][0]

    assert len(up_payload["reply_markup"]["inline_keyboard"]) == 1
    assert "\u00B0C" in up_payload["reply_markup"]["inline_keyboard"][0][0]["text"]


    await bot.close()


@pytest.mark.asyncio
async def test_rubric_channel_picker_flow(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    calls: list[tuple[str, dict[str, Any]]] = []

    async def dummy(method, data=None):
        calls.append((method, data))
        return {"ok": True}

    bot.api_request = dummy  # type: ignore
    await bot.start()

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})

    bot.db.execute("INSERT INTO channels (chat_id, title) VALUES (?, ?)", (-100, "Main"))
    bot.db.execute("INSERT INTO channels (chat_id, title) VALUES (?, ?)", (-200, "Second"))
    bot.db.commit()
    bot.data.upsert_rubric("news", "News", config={"enabled": True})

    message = {"message_id": 10, "chat": {"id": 1}}
    callback_base = {"id": "cb1", "from": {"id": 1}, "message": message}

    await bot.handle_update(
        {
            "callback_query": {
                **callback_base,
                "data": "rubric_channel:news:main",
            }
        }
    )

    assert 1 in bot.pending and bot.pending[1]["rubric_input"]["mode"] == "channel_picker"
    assert any(call[0] == "editMessageText" for call in calls)

    await bot.handle_update(
        {
            "callback_query": {
                **callback_base,
                "data": "rubric_channel_set:-100",
            }
        }
    )

    config = bot.data.get_rubric_config("news")
    assert config["channel_id"] == -100
    assert 1 not in bot.pending
    assert any(call[0] == "editMessageText" for call in calls)

    await bot.close()


@pytest.mark.asyncio
async def test_rubric_schedule_wizard_flow(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    calls: list[tuple[str, dict[str, Any]]] = []

    async def dummy(method, data=None):
        calls.append((method, data))
        return {"ok": True}

    bot.api_request = dummy  # type: ignore
    await bot.start()

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})

    bot.db.execute("INSERT INTO channels (chat_id, title) VALUES (?, ?)", (-100, "Main"))
    bot.db.commit()
    bot.data.upsert_rubric("news", "News", config={"enabled": True})

    message = {"message_id": 20, "chat": {"id": 1}}
    callback_base = {"id": "cb2", "from": {"id": 1}, "message": message}

    await bot.handle_update(
        {
            "callback_query": {
                **callback_base,
                "data": "rubric_sched_add:news",
            }
        }
    )

    assert bot.pending[1]["rubric_input"]["mode"] == "schedule_wizard"

    await bot.handle_update(
        {
            "callback_query": {
                **callback_base,
                "data": "rubric_sched_time",
            }
        }
    )
    await bot.handle_update(
        {
            "callback_query": {
                **callback_base,
                "data": "rubric_sched_hour:9",
            }
        }
    )
    await bot.handle_update(
        {
            "callback_query": {
                **callback_base,
                "data": "rubric_sched_minute:30",
            }
        }
    )

    await bot.handle_update(
        {
            "callback_query": {
                **callback_base,
                "data": "rubric_sched_days",
            }
        }
    )
    await bot.handle_update(
        {
            "callback_query": {
                **callback_base,
                "data": "rubric_sched_day:mon",
            }
        }
    )
    await bot.handle_update(
        {
            "callback_query": {
                **callback_base,
                "data": "rubric_sched_day:wed",
            }
        }
    )
    await bot.handle_update(
        {
            "callback_query": {
                **callback_base,
                "data": "rubric_sched_days_done",
            }
        }
    )

    await bot.handle_update(
        {
            "callback_query": {
                **callback_base,
                "data": "rubric_sched_channel",
            }
        }
    )
    await bot.handle_update(
        {
            "callback_query": {
                **callback_base,
                "data": "rubric_channel_set:-100",
            }
        }
    )

    await bot.handle_update(
        {
            "callback_query": {
                **callback_base,
                "data": "rubric_sched_toggle_enabled",
            }
        }
    )

    await bot.handle_update(
        {
            "callback_query": {
                **callback_base,
                "data": "rubric_sched_save",
            }
        }
    )

    config = bot.data.get_rubric_config("news")
    schedule = config["schedules"][0]
    assert schedule["time"] == "09:30"
    assert schedule["days"] == ["mon", "wed"]
    assert schedule.get("channel_id") == -100
    assert schedule.get("enabled") is False

    await bot.handle_update(
        {
            "callback_query": {
                **callback_base,
                "data": "rubric_sched_edit:news:0",
            }
        }
    )
    await bot.handle_update(
        {
            "callback_query": {
                **callback_base,
                "data": "rubric_sched_time",
            }
        }
    )
    await bot.handle_update(
        {
            "callback_query": {
                **callback_base,
                "data": "rubric_sched_hour:10",
            }
        }
    )
    await bot.handle_update(
        {
            "callback_query": {
                **callback_base,
                "data": "rubric_sched_minute:0",
            }
        }
    )
    await bot.handle_update(
        {
            "callback_query": {
                **callback_base,
                "data": "rubric_sched_toggle_enabled",
            }
        }
    )
    await bot.handle_update(
        {
            "callback_query": {
                **callback_base,
                "data": "rubric_sched_save",
            }
        }
    )

    config = bot.data.get_rubric_config("news")
    schedule = config["schedules"][0]
    assert schedule["time"] == "10:00"
    assert schedule.get("enabled") is True

    await bot.handle_update(
        {
            "callback_query": {
                **callback_base,
                "data": "rubric_sched_toggle:news:0",
            }
        }
    )

    config = bot.data.get_rubric_config("news")
    assert config["schedules"][0]["enabled"] is False

    await bot.handle_update(
        {
            "callback_query": {
                **callback_base,
                "data": "rubric_sched_del:news:0",
            }
        }
    )

    config = bot.data.get_rubric_config("news")
    assert config.get("schedules") == []

    await bot.close()


@pytest.mark.asyncio
async def test_delbutton_clears_weather_record(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    calls = []

    async def dummy(method, data=None):
        calls.append((method, data))
        if method == "forwardMessage":
            return {
                "ok": True,
                "result": {"message_id": 5, "reply_markup": {"inline_keyboard": []}},
            }
        return {"ok": True}

    bot.api_request = dummy  # type: ignore
    bot.set_latest_weather_post(-100, 7)
    await bot.start()

    bot.db.execute("INSERT INTO cities (id, name, lat, lon) VALUES (1, 'c', 0, 0)")
    bot.db.execute(
        "INSERT INTO weather_cache_hour (city_id, timestamp, temperature, weather_code, wind_speed, is_day) VALUES (1, ?, 15.0, 1, 3, 1)",
        (datetime.utcnow().isoformat(),),
    )
    bot.db.commit()

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})

    await bot.handle_update({
        "message": {
            "text": "/addweatherbutton https://t.me/c/123/5 K. {1|temperature}",
            "from": {"id": 1},
        }
    })

    assert bot.db.execute("SELECT COUNT(*) FROM weather_link_posts").fetchone()[0] == 1

    await bot.handle_update({
        "message": {
            "text": "/delbutton https://t.me/c/123/5",
            "from": {"id": 1},
        }
    })

    assert bot.db.execute("SELECT COUNT(*) FROM weather_link_posts").fetchone()[0] == 0
    assert calls[-1][0] == "editMessageReplyMarkup"
    assert calls[-1][1]["reply_markup"] == {}


    await bot.close()


@pytest.mark.asyncio
async def test_multiple_weather_buttons_same_row(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    calls = []

    async def dummy(method, data=None):
        calls.append((method, data))
        if method == "forwardMessage":
            return {
                "ok": True,
                "result": {"message_id": 5, "reply_markup": {"inline_keyboard": []}},
            }
        return {"ok": True}

    bot.api_request = dummy  # type: ignore
    bot.set_latest_weather_post(-100, 7)
    await bot.start()

    bot.db.execute("INSERT INTO cities (id, name, lat, lon) VALUES (1, 'c', 0, 0)")
    bot.db.execute(
        "INSERT INTO weather_cache_hour (city_id, timestamp, temperature, weather_code, wind_speed, is_day) VALUES (1, ?, 15.0, 1, 3, 1)",
        (datetime.utcnow().isoformat(),),
    )
    bot.db.commit()

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})

    await bot.handle_update(
        {
            "message": {
                "text": "/addweatherbutton https://t.me/c/123/5 A {1|temperature}",
                "from": {"id": 1},
            }
        }
    )

    calls.clear()
    await bot.handle_update(
        {
            "message": {
                "text": "/addweatherbutton https://t.me/c/123/5 B {1|temperature}",
                "from": {"id": 1},
            }
        }
    )

    payload = [c[1] for c in calls if c[0] == "editMessageReplyMarkup"][0]
    assert len(payload["reply_markup"]["inline_keyboard"]) == 1
    assert len(payload["reply_markup"]["inline_keyboard"][0]) == 2


    await bot.close()
