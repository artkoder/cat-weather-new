import os
import re
import sys
import pytest
from aiohttp import web
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from main import create_app, Bot

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

    async def dummy(method, data=None):
        calls.append((method, data))
        if method == "getChat":
            return {"ok": True, "result": {"id": -100123}}
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

    assert any(c[0] == "editMessageReplyMarkup" for c in calls)

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
    assert payload["reply_markup"]["inline_keyboard"][0][0]["text"] == "ask locals"

    await bot.close()
