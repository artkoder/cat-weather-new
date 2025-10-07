import json
import os
import sys
from datetime import datetime
from pathlib import Path

from typing import Any

import pytest
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import data_access
from data_access import Asset
from jobs import Job
from main import Bot
from openai_client import OpenAIResponse

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "dummy")


def _insert_rubric(bot: Bot, code: str, config: dict, rubric_id: int = 1) -> None:
    bot.data.upsert_rubric(code, code.title(), config=config)


@pytest.mark.asyncio
async def test_rubric_deletion_disabled(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    with pytest.raises(NotImplementedError):
        bot.data.delete_rubric("flowers")

    await bot.close()


@pytest.mark.asyncio
async def test_default_rubrics_created_on_boot(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    rubrics = {rubric.code: rubric for rubric in bot.data.list_rubrics()}
    assert {"flowers", "guess_arch"}.issubset(rubrics.keys())
    for code in ("flowers", "guess_arch"):
        config = rubrics[code].config
        assert config.get("enabled") is False
        assert config.get("schedules") == []
    await bot.close()


@pytest.mark.asyncio
async def test_rubric_scheduler_enqueues_jobs(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "tz": "+00:00",
        "schedules": [
            {"time": "00:00", "channel_id": -100}
        ],
        "assets": {"categories": ["flowers"], "min": 4, "max": 4},
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    await bot.process_rubric_schedule(reference=datetime.utcnow())
    row = bot.db.execute(
        "SELECT name, status, payload FROM jobs_queue WHERE name='publish_rubric'"
    ).fetchone()
    assert row is not None
    payload = json.loads(row["payload"])
    assert payload["rubric_code"] == "flowers"
    assert payload["channel_id"] == -100
    assert payload["tz_offset"] == "+00:00"
    await bot.close()


@pytest.mark.asyncio
async def test_rubric_scheduler_respects_timezone(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "schedules": [
            {"time": "10:00", "channel_id": -900, "tz": "+03:00"}
        ],
    }
    _insert_rubric(bot, "guess_arch", config, rubric_id=2)
    reference = datetime(2024, 3, 1, 6, 0, 0)
    expected = bot._compute_next_rubric_run(  # type: ignore[attr-defined]
        time_str="10:00", tz_offset="+03:00", days=None, reference=reference
    )
    await bot.process_rubric_schedule(reference=reference)
    row = bot.db.execute(
        "SELECT payload FROM jobs_queue WHERE name='publish_rubric'"
    ).fetchone()
    assert row is not None
    payload = json.loads(row["payload"])
    assert payload["rubric_code"] == "guess_arch"
    assert payload["channel_id"] == -900
    assert payload["scheduled_at"] == expected.isoformat()
    assert payload["tz_offset"] == "+03:00"
    await bot.close()


@pytest.mark.asyncio
async def test_enqueue_rubric_manual_and_test_channels(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "channel_id": -111,
        "test_channel_id": -222,
    }
    _insert_rubric(bot, "flowers", config, rubric_id=5)

    job_id = bot.enqueue_rubric("flowers")
    row = bot.db.execute(
        "SELECT status, payload FROM jobs_queue WHERE id=?", (job_id,)
    ).fetchone()
    assert row is not None
    payload = json.loads(row["payload"])
    assert payload["schedule_key"] == "manual"
    assert payload["channel_id"] == -111
    assert not payload.get("test")
    assert payload["tz_offset"] == "+00:00"
    assert row["status"] == "queued"

    test_job_id = bot.enqueue_rubric("flowers", test=True)
    test_row = bot.db.execute(
        "SELECT status, payload FROM jobs_queue WHERE id=?", (test_job_id,)
    ).fetchone()
    assert test_row is not None
    test_payload = json.loads(test_row["payload"])
    assert test_payload["schedule_key"] == "manual-test"
    assert test_payload["channel_id"] == -222
    assert test_payload["test"] is True
    assert test_payload["tz_offset"] == "+00:00"
    assert test_row["status"] == "queued"
    await bot.close()


@pytest.mark.asyncio
async def test_vision_job_assigns_rubric_id(tmp_path):
    bot = Bot("test-token", str(tmp_path / "db.sqlite"))
    bot.openai.api_key = "test"  # type: ignore[assignment]

    flowers_rubric = bot.data.get_rubric_by_code("flowers")
    assert flowers_rubric is not None

    image_path = tmp_path / "flower.jpg"
    Image.new("RGB", (32, 32), color="red").save(image_path)

    file_meta = {"file_id": "file123", "file_unique_id": "uniq123", "mime_type": "image/jpeg"}
    asset_id = bot.data.save_asset(
        -1500,
        10,
        None,
        "",
        tg_chat_id=-1500,
        caption="",
        kind="photo",
        file_meta=file_meta,
        metadata={},
    )
    bot.data.update_asset(asset_id, local_path=str(image_path))

    async def fake_api_request(method, data=None, *, files=None):
        if method == "copyMessage":
            return {"ok": True, "result": {"message_id": 600}}
        return {"ok": True}

    async def fake_record_usage(*args, **kwargs):
        return None

    async def fake_classify_image(*, model, system_prompt, user_prompt, image_path, schema):
        return OpenAIResponse(
            content={
                "framing": "wide",
                "architecture_close_up": False,
                "architecture_wide": False,
                "weather_image": "sunny",
                "season_guess": "summer",
                "arch_style": None,
                "caption": "Field of tulips",
                "objects": ["tulips"],
                "landmarks": [],
                "tags": ["flowers"],
                "arch_view": False,
                "is_outdoor": True,
                "guess_country": None,
                "guess_city": None,
                "location_confidence": 0.5,
                "safety": {"nsfw": False, "reason": "безопасно"},
                "category": "flowers",
            },
            usage={
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "request_id": "req-flowers",
            },
        )

    bot.api_request = fake_api_request  # type: ignore[assignment]
    bot._record_openai_usage = fake_record_usage  # type: ignore[assignment]
    bot.openai.classify_image = fake_classify_image  # type: ignore[assignment]

    now = datetime.utcnow()
    job = Job(
        id=1,
        name="vision",
        payload={"asset_id": asset_id, "tz_offset": "+00:00"},
        status="queued",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=now,
        updated_at=now,
    )

    await bot._job_vision_locked(job)

    updated_asset = bot.data.get_asset(asset_id)
    assert updated_asset is not None
    assert updated_asset.vision_category == "flowers"
    assert updated_asset.rubric_id == flowers_rubric.id

    assets = bot.data.fetch_assets_by_vision_category("flowers", rubric_id=flowers_rubric.id)
    assert any(asset.id == asset_id for asset in assets)

    await bot.close()


@pytest.mark.asyncio
async def test_vision_job_handles_singular_flower_tag(tmp_path):
    bot = Bot("test-token", str(tmp_path / "db.sqlite"))
    bot.openai.api_key = "test"  # type: ignore[assignment]

    flowers_rubric = bot.data.get_rubric_by_code("flowers")
    assert flowers_rubric is not None

    image_path = tmp_path / "flower.jpg"
    Image.new("RGB", (32, 32), color="red").save(image_path)

    file_meta = {
        "file_id": "file-singular",
        "file_unique_id": "uniq-singular",
        "mime_type": "image/jpeg",
    }
    asset_id = bot.data.save_asset(
        -1600,
        20,
        None,
        "",
        tg_chat_id=-1600,
        caption="",
        kind="photo",
        file_meta=file_meta,
        metadata={},
    )
    bot.data.update_asset(asset_id, local_path=str(image_path))

    async def fake_api_request(method, data=None, *, files=None):
        if method == "copyMessage":
            return {"ok": True, "result": {"message_id": 601}}
        return {"ok": True}

    async def fake_record_usage(*args, **kwargs):
        return None

    async def fake_classify_image(*, model, system_prompt, user_prompt, image_path, schema):
        return OpenAIResponse(
            content={
                "framing": "medium",
                "architecture_close_up": False,
                "architecture_wide": False,
                "weather_image": "sunny",
                "season_guess": "summer",
                "arch_style": None,
                "caption": "A single rose",
                "objects": ["rose"],
                "landmarks": [],
                "tags": ["flower"],
                "arch_view": False,
                "is_outdoor": True,
                "guess_country": None,
                "guess_city": None,
                "location_confidence": 0.6,
                "safety": {"nsfw": False, "reason": "безопасно"},
                "category": "flower",
            },
            usage={
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "request_id": "req-flower",
            },
        )

    bot.api_request = fake_api_request  # type: ignore[assignment]
    bot._record_openai_usage = fake_record_usage  # type: ignore[assignment]
    bot.openai.classify_image = fake_classify_image  # type: ignore[assignment]

    now = datetime.utcnow()
    job = Job(
        id=2,
        name="vision",
        payload={"asset_id": asset_id, "tz_offset": "+00:00"},
        status="queued",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=now,
        updated_at=now,
    )

    await bot._job_vision_locked(job)

    updated_asset = bot.data.get_asset(asset_id)
    assert updated_asset is not None
    assert updated_asset.vision_category == "flowers"
    assert updated_asset.rubric_id == flowers_rubric.id
    assert updated_asset.vision_flower_varieties == ["rose"]

    await bot.close()


@pytest.mark.asyncio
async def test_fetch_assets_includes_singular_flower_category(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    asset_id = bot.data.save_asset(
        -2000,
        50,
        None,
        None,
        tg_chat_id=-2000,
        caption="",
        kind="photo",
        categories=["flowers"],
    )

    bot.db.execute(
        "UPDATE assets SET vision_category=?, vision_flower_varieties=?, updated_at=? WHERE id=?",
        ("flower", json.dumps(["rose"]), datetime.utcnow().isoformat(), asset_id),
    )
    bot.db.commit()

    assets = bot.data.fetch_assets_by_vision_category("flowers")
    assert any(asset.id == asset_id for asset in assets)
    matched = next(asset for asset in assets if asset.id == asset_id)
    assert matched.vision_category == "flowers"
    assert matched.vision_flower_varieties == ["rose"]

    await bot.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("asset_count", "expected_method"),
    [
        (1, "sendPhoto"),
        (2, "sendMediaGroup"),
        (3, "sendMediaGroup"),
    ],
)
async def test_publish_flowers_varied_asset_counts(tmp_path, asset_count, expected_method):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "assets": {"min": asset_count, "max": asset_count},
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    now = datetime.utcnow().isoformat()
    file_ids: list[str] = []
    for idx in range(asset_count):
        metadata = {"date": now}
        file_id = f"file{idx}"
        file_ids.append(file_id)
        file_meta = {"file_id": file_id}
        asset_id = bot.data.save_asset(
            -2000,
            100 + idx,
            None,
            "",
            tg_chat_id=-2000,
            caption="",
            kind="photo",
            file_meta=file_meta,
            metadata=metadata,
            categories=["flowers"],
            rubric_id=1,
        )
        bot.data.update_asset(
            asset_id,
            vision_category="flowers",
            vision_photo_weather="солнечно",
            city=f"Город {idx}",
        )

    calls: list[dict[str, Any]] = []

    async def fake_api(method, data=None, *, files=None):
        calls.append({"method": method, "data": data, "files": files})
        if method == "sendPhoto":
            return {"ok": True, "result": {"message_id": 42}}
        if method == "sendMediaGroup":
            return {"ok": True, "result": [{"message_id": 42}]}
        return {"ok": True}

    bot.api_request = fake_api  # type: ignore[assignment]
    ok = await bot.publish_rubric("flowers", channel_id=-500)
    assert ok
    send_call = next(
        call for call in calls if call["method"] in {"sendPhoto", "sendMediaGroup"}
    )
    assert send_call["method"] == expected_method
    data = send_call["data"]
    assert data is not None
    if expected_method == "sendPhoto":
        assert data["photo"] == file_ids[0]
    else:
        media_payload = data["media"]
        assert isinstance(media_payload, list)
        assert len(media_payload) == asset_count
    delete_calls = [call for call in calls if call["method"] == "deleteMessage"]
    assert len(delete_calls) == asset_count
    remaining = bot.db.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    assert remaining == 0
    history = bot.db.execute("SELECT metadata FROM posts_history").fetchone()
    assert history is not None
    meta = json.loads(history["metadata"])
    assert meta["rubric_code"] == "flowers"
    assert meta["asset_ids"]
    assert meta["greeting"]
    await bot.close()


@pytest.mark.asyncio
async def test_flowers_preview_notifies_when_not_enough_assets(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "channel_id": -500,
        "assets": {"min": 5, "max": 6},
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)

    calls: list[tuple[str, dict[str, Any] | None]] = []

    async def fake_api(method, data=None, *, files=None):  # type: ignore[override]
        calls.append((method, data))
        return {"ok": True}

    bot.api_request = fake_api  # type: ignore[assignment]

    payload = {
        "rubric_code": "flowers",
        "channel_id": -500,
        "initiator_id": 777,
        "test": False,
    }
    now = datetime.utcnow()
    job = Job(
        id=1,
        name="publish_rubric",
        payload=payload,
        status="queued",
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=now,
        updated_at=now,
    )

    await bot._job_publish_rubric(job)

    send_calls = [call for call in calls if call[0] == "sendMessage"]
    assert send_calls, "Expected operator notification"
    message = send_calls[0][1]
    assert message is not None
    assert message["chat_id"] == 777
    assert "минимальн" in message["text"].lower()
    assert "5" in message["text"]

    ok = await bot.publish_rubric("flowers", channel_id=-500)
    assert not ok

    await bot.close()


@pytest.mark.asyncio
async def test_flowers_preview_single_photo_paths(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "channel_id": -500,
        "test_channel_id": -600,
        "assets": {"min": 1, "max": 1},
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)

    now = datetime.utcnow().isoformat()
    file_meta = {"file_id": "single-file"}
    asset_id = bot.data.save_asset(
        -2100,
        915,
        None,
        "",
        tg_chat_id=-2100,
        caption="",
        kind="photo",
        file_meta=file_meta,
        metadata={"date": now},
        categories=["flowers"],
        rubric_id=1,
    )
    bot.data.update_asset(
        asset_id,
        vision_category="flowers",
        vision_photo_weather="солнечно",
        city="Москва",
    )

    bot.db.execute(
        "INSERT OR REPLACE INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, '+00:00')",
        (1234, "tester"),
    )
    bot.db.commit()

    calls: list[tuple[str, dict[str, Any] | None]] = []

    async def fake_api(method, data=None, *, files=None):  # type: ignore[override]
        calls.append((method, data))
        if method == "sendPhoto":
            if data and data.get("chat_id") == 1234:
                return {"ok": True, "result": {"message_id": 50}}
            if data and data.get("chat_id") == -500:
                return {"ok": True, "result": {"message_id": 75}}
        if method == "sendMessage":
            return {"ok": True, "result": {"message_id": 200}}
        if method in {"deleteMessage", "editMessageText", "answerCallbackQuery"}:
            return {"ok": True}
        return {"ok": True}

    bot.api_request = fake_api  # type: ignore[assignment]

    await bot.publish_rubric("flowers", channel_id=-500, initiator_id=1234)

    state = bot.pending_flowers_previews.get(1234)
    assert state is not None
    assert state.get("media_message_ids") == [50]
    preview_send = [call for call in calls if call[0] == "sendPhoto" and call[1] and call[1].get("chat_id") == 1234]
    assert preview_send, "Expected preview to use sendPhoto"

    await bot._handle_flowers_preview_callback(1234, "send_main", {"id": "cb-photo"})

    final_send = [call for call in calls if call[0] == "sendPhoto" and call[1] and call[1].get("chat_id") == -500]
    assert final_send, "Expected finalize to use sendPhoto"
    assert bot.pending_flowers_previews.get(1234) is None
    remaining = bot.db.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    assert remaining == 0

    await bot.close()


@pytest.mark.asyncio
async def test_flowers_preview_document_media_paths(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "channel_id": -500,
        "test_channel_id": -600,
        "assets": {"min": 1, "max": 1},
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)

    now = datetime.utcnow().isoformat()
    file_meta = {"file_id": "doc-file", "file_name": "flower.pdf"}
    asset_id = bot.data.save_asset(
        -2100,
        916,
        None,
        "",
        tg_chat_id=-2100,
        caption="",
        kind="document",
        file_meta=file_meta,
        metadata={"date": now},
        categories=["flowers"],
        rubric_id=1,
    )
    bot.data.update_asset(
        asset_id,
        vision_category="flowers",
        vision_photo_weather="солнечно",
        city="Москва",
    )

    bot.db.execute(
        "INSERT OR REPLACE INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, '+00:00')",
        (1234, "tester"),
    )
    bot.db.commit()

    calls: list[tuple[str, dict[str, Any] | None]] = []

    async def fake_api(method, data=None, *, files=None):  # type: ignore[override]
        calls.append((method, data))
        if method == "sendDocument":
            if data and data.get("chat_id") == 1234:
                return {"ok": True, "result": {"message_id": 60}}
            if data and data.get("chat_id") == -500:
                return {"ok": True, "result": {"message_id": 85}}
        if method == "sendMessage":
            return {"ok": True, "result": {"message_id": 210}}
        if method in {"deleteMessage", "editMessageText", "answerCallbackQuery"}:
            return {"ok": True}
        return {"ok": True}

    bot.api_request = fake_api  # type: ignore[assignment]

    await bot.publish_rubric("flowers", channel_id=-500, initiator_id=1234)

    state = bot.pending_flowers_previews.get(1234)
    assert state is not None
    assert state.get("media_message_ids") == [60]
    assert state.get("asset_kinds") == ["document"]

    preview_send = [
        call for call in calls if call[0] == "sendDocument" and call[1] and call[1].get("chat_id") == 1234
    ]
    assert preview_send, "Expected preview to use sendDocument"

    await bot._handle_flowers_preview_callback(1234, "send_main", {"id": "cb-doc"})

    final_send = [
        call for call in calls if call[0] == "sendDocument" and call[1] and call[1].get("chat_id") == -500
    ]
    assert final_send, "Expected finalize to use sendDocument"
    assert bot.pending_flowers_previews.get(1234) is None
    remaining = bot.db.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    assert remaining == 0

    await bot.close()


@pytest.mark.asyncio
async def test_flowers_preview_document_with_image_filename(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "channel_id": -500,
        "test_channel_id": -600,
        "assets": {"min": 1, "max": 1},
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)

    now = datetime.utcnow().isoformat()
    file_meta = {"file_id": "doc-photo", "file_name": "flower.jpg"}
    asset_id = bot.data.save_asset(
        -2100,
        917,
        None,
        "",
        tg_chat_id=-2100,
        caption="",
        kind="document",
        file_meta=file_meta,
        metadata={"date": now},
        categories=["flowers"],
        rubric_id=1,
    )
    bot.data.update_asset(
        asset_id,
        vision_category="flowers",
        vision_photo_weather="солнечно",
        city="Москва",
    )

    bot.db.execute(
        "INSERT OR REPLACE INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, '+00:00')",
        (1234, "tester"),
    )
    bot.db.commit()

    call_log: list[dict[str, Any]] = []
    multipart_calls: list[dict[str, Any]] = []

    async def fake_api(method, data=None, *, files=None):  # type: ignore[override]
        call_log.append({"method": method, "data": data, "files": files})
        if method == "getFile":
            return {"ok": True, "result": {"file_path": "downloads/flower.jpg"}}
        if method == "sendPhoto":
            return {"ok": True, "result": {"message_id": 915}}
        if method == "sendMessage":
            return {
                "ok": True,
                "result": {"message_id": 320 if data and data.get("chat_id") == -500 else 310},
            }
        if method in {"deleteMessage", "editMessageText", "answerCallbackQuery"}:
            return {"ok": True}
        return {"ok": True}

    async def fake_api_multipart(method, data=None, *, files=None):  # type: ignore[override]
        multipart_calls.append({"method": method, "data": data, "files": files})
        photo_payload = {
            "file_id": "photo-new",
            "file_unique_id": "uniq-photo-new",
            "width": 800,
            "height": 600,
            "file_size": 2048,
        }
        message_id = 610 if data and data.get("chat_id") == 1234 else 960
        return {
            "ok": True,
            "result": {"message_id": message_id, "photo": [photo_payload]},
        }

    async def fake_download(file_id, dest_path):  # type: ignore[override]
        dest = Path(dest_path)
        dest.write_bytes(b"stub")
        return dest

    def fake_prepare(local_path):  # type: ignore[override]
        path = Path(local_path)
        if not path.exists():
            path.write_bytes(b"stub")
        return path, None, "flower.jpg", "image/jpeg", "original", "image/jpeg", 4

    bot.api_request = fake_api  # type: ignore[assignment]
    bot.api_request_multipart = fake_api_multipart  # type: ignore[assignment]
    bot._download_file = fake_download  # type: ignore[assignment]
    bot._prepare_photo_for_upload = fake_prepare  # type: ignore[assignment]

    await bot.publish_rubric("flowers", channel_id=-500, initiator_id=1234)

    state = bot.pending_flowers_previews.get(1234)
    assert state is not None
    assert state.get("media_message_ids") == [610]
    assert state.get("asset_kinds") == ["photo"]
    assert state.get("file_ids") == ["photo-new"]

    preview_photos = [
        call
        for call in multipart_calls
        if call["method"] == "sendPhoto" and call["data"] and call["data"].get("chat_id") == 1234
    ]
    assert preview_photos, "Expected preview to upload via sendPhoto"

    updated_asset = bot.data.get_asset(asset_id)
    assert updated_asset is not None
    assert updated_asset.kind == "photo"
    assert updated_asset.file_id == "photo-new"
    assert updated_asset.metadata.get("original_document_file_id") == "doc-photo"

    await bot._handle_flowers_preview_callback(1234, "send_main", {"id": "cb-photo"})

    final_photos = [
        call
        for call in call_log
        if call["method"] == "sendPhoto" and call["data"] and call["data"].get("chat_id") == -500
    ]
    assert final_photos, "Expected finalize to use sendPhoto"
    assert final_photos[0]["data"]["photo"] == "photo-new"

    assert bot.pending_flowers_previews.get(1234) is None
    remaining = bot.db.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    assert remaining == 0

    await bot.close()


@pytest.mark.asyncio
async def test_flowers_preview_reuses_converted_photo_id(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "channel_id": -500,
        "test_channel_id": -600,
        "assets": {"min": 1, "max": 1},
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)

    file_meta = {
        "file_id": "photo_cached",
        "file_unique_id": "uniq_photo",
        "mime_type": "image/jpeg",
    }
    asset_id = bot.data.save_asset(
        -2100,
        999,
        None,
        "",
        tg_chat_id=-2100,
        caption="",
        kind="photo",
        file_meta=file_meta,
        metadata={"original_document_file_id": "doc_cached"},
        categories=["flowers"],
        rubric_id=1,
    )
    bot.data.update_asset(
        asset_id,
        vision_category="flowers",
        vision_photo_weather="солнечно",
        city="Калининград",
    )

    bot.db.execute(
        "INSERT OR REPLACE INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, '+00:00')",
        (1234, "tester"),
    )
    bot.db.commit()

    calls: list[dict[str, Any]] = []
    multipart_calls: list[dict[str, Any]] = []

    async def fake_api(method, data=None, *, files=None):  # type: ignore[override]
        calls.append({"method": method, "data": data, "files": files})
        if method == "sendPhoto":
            chat_id = data.get("chat_id") if isinstance(data, dict) else None
            message_id = 70 if chat_id == 1234 else 95
            return {"ok": True, "result": {"message_id": message_id}}
        if method == "sendMessage":
            return {"ok": True, "result": {"message_id": 205}}
        if method in {"deleteMessage", "editMessageText", "answerCallbackQuery"}:
            return {"ok": True}
        return {"ok": True}

    async def fake_api_multipart(method, data=None, *, files=None):  # type: ignore[override]
        multipart_calls.append({"method": method, "data": data, "files": files})
        return {"ok": True, "result": {"message_id": 880}}

    bot.api_request = fake_api  # type: ignore[assignment]
    bot.api_request_multipart = fake_api_multipart  # type: ignore[assignment]

    await bot.publish_rubric("flowers", channel_id=-500, initiator_id=1234)

    state = bot.pending_flowers_previews.get(1234)
    assert state is not None

    preview_photos = [
        call
        for call in calls
        if call["method"] == "sendPhoto" and call["data"] and call["data"].get("chat_id") == 1234
    ]
    assert preview_photos, "Preview should be sent with cached photo file_id"
    assert preview_photos[0]["files"] is None, "Preview must reuse Telegram file id"
    assert not multipart_calls, "Cached photo should not trigger multipart uploads"

    await bot._handle_flowers_preview_callback(1234, "send_main", {"id": "cb-reuse"})

    final_photos = [
        call
        for call in calls
        if call["method"] == "sendPhoto" and call["data"] and call["data"].get("chat_id") == -500
    ]
    assert final_photos, "Final publication should reuse cached photo file_id"
    assert final_photos[0]["files"] is None
    assert bot.pending_flowers_previews.get(1234) is None
    assert not multipart_calls

    await bot.close()


@pytest.mark.asyncio
async def test_flowers_asset_selection_random(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "assets": {"min": 4, "max": 4},
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    now = datetime.utcnow().isoformat()
    for idx in range(10):
        file_meta = {"file_id": f"rf{idx}"}
        asset_id = bot.data.save_asset(
            -2100,
            300 + idx,
            None,
            "",
            tg_chat_id=-2100,
            caption="",
            kind="photo",
            file_meta=file_meta,
            metadata={"date": now},
            categories=["flowers"],
            rubric_id=1,
        )
        bot.data.update_asset(
            asset_id,
            vision_category="flowers",
            vision_photo_weather="солнечно",
            city=f"Город {idx}",
        )

    selections = [
        [
            asset.id
            for asset in bot.data.fetch_assets_by_vision_category(
                "flowers",
                rubric_id=1,
                limit=4,
                random_order=True,
            )
        ]
        for _ in range(3)
    ]
    assert all(len(selection) == 4 for selection in selections)
    assert len({tuple(selection) for selection in selections}) > 1
    await bot.close()


@pytest.mark.asyncio
async def test_publish_flowers_uses_distinct_assets(tmp_path, monkeypatch):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "assets": {"min": 4, "max": 4},
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    now = datetime.utcnow().isoformat()
    for idx in range(8):
        file_meta = {"file_id": f"rf{idx}"}
        asset_id = bot.data.save_asset(
            -2100,
            400 + idx,
            None,
            "",
            tg_chat_id=-2100,
            caption="",
            kind="photo",
            file_meta=file_meta,
            metadata={"date": now},
            categories=["flowers"],
            rubric_id=1,
        )
        bot.data.update_asset(
            asset_id,
            vision_category="flowers",
            vision_photo_weather="солнечно",
            city=f"Город {idx}",
        )

    used_media: list[list[str]] = []

    async def fake_api(method, data=None, *, files=None):
        if method == "sendMediaGroup":
            assert data is not None
            used_media.append([item["media"] for item in data["media"]])
            return {"ok": True, "result": [{"message_id": len(used_media)}]}
        return {"ok": True}

    async def fake_cleanup(assets, *, extra_paths=None):
        return None

    shuffle_calls = {"count": 0}

    def rotating_shuffle(seq):
        if not seq:
            return
        shuffle_calls["count"] += 1
        shift = shuffle_calls["count"] % len(seq)
        if shift:
            seq[:] = seq[shift:] + seq[:shift]

    monkeypatch.setattr(data_access.random, "shuffle", rotating_shuffle)
    bot.api_request = fake_api  # type: ignore[assignment]
    monkeypatch.setattr(bot, "_cleanup_assets", fake_cleanup)

    ok_first = await bot.publish_rubric("flowers", channel_id=-700)
    ok_second = await bot.publish_rubric("flowers", channel_id=-700)

    assert ok_first and ok_second
    assert len(used_media) == 2
    assert used_media[0] != used_media[1]

    await bot.close()


@pytest.mark.asyncio
async def test_flowers_preview_regenerate_and_finalize(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "channel_id": -500,
        "test_channel_id": -600,
        "assets": {"min": 4, "max": 4},
    }
    _insert_rubric(bot, "flowers", config, rubric_id=1)

    now = datetime.utcnow().isoformat()
    for idx in range(4):
        file_meta = {"file_id": f"pv{idx}"}
        asset_id = bot.data.save_asset(
            -2100,
            900 + idx,
            None,
            "",
            tg_chat_id=-2100,
            caption="",
            kind="photo",
            file_meta=file_meta,
            metadata={"date": now},
            categories=["flowers"],
            rubric_id=1,
        )
        bot.data.update_asset(
            asset_id,
            vision_category="flowers",
            vision_photo_weather="солнечно",
            city=f"Город {idx}",
        )

    bot.db.execute(
        "INSERT OR REPLACE INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, '+00:00')",
        (1234, "tester"),
    )
    bot.db.commit()

    api_calls: list[tuple[str, dict[str, Any] | None]] = []
    message_counter = {"value": 200}

    async def fake_api(method, data=None, *, files=None):  # type: ignore[override]
        api_calls.append((method, data))
        if method == "sendMediaGroup":
            if data and data.get("chat_id") == 1234:
                return {"ok": True, "result": [{"message_id": 10}, {"message_id": 11}]}
            return {"ok": True, "result": [{"message_id": 210}]}
        if method == "sendMessage":
            message_counter["value"] += 1
            return {"ok": True, "result": {"message_id": message_counter["value"]}}
        if method in {"editMessageText", "deleteMessage", "answerCallbackQuery"}:
            return {"ok": True}
        return {"ok": True}

    bot.api_request = fake_api  # type: ignore[assignment]

    await bot.publish_rubric("flowers", channel_id=-500, initiator_id=1234)

    state = bot.pending_flowers_previews.get(1234)
    assert state is not None
    assert state.get("caption_message_id")

    await bot._handle_flowers_preview_callback(1234, "regen_caption", {"id": "cb1"})
    assert any(call[0] == "editMessageText" for call in api_calls)

    await bot._handle_flowers_preview_callback(1234, "instruction", {"id": "cb2"})
    prompt_id = state.get("instruction_prompt_id")
    assert isinstance(prompt_id, int)
    instruction_message = {
        "message_id": 999,
        "from": {"id": 1234, "username": "tester"},
        "chat": {"id": 1234},
        "text": "Добавь смайлы",
        "reply_to_message": {"message_id": prompt_id},
    }
    await bot.handle_message(instruction_message)
    state = bot.pending_flowers_previews.get(1234)
    assert state is not None
    assert state.get("instructions") == "Добавь смайлы"
    assert not state.get("awaiting_instruction")

    caption_text = str(state.get("caption") or "").strip()
    summary_updates = [
        data for method, data in api_calls if method == "editMessageText" and data
    ]
    assert summary_updates, "Ожидалось обновление текста предпросмотра"
    summary_text = str(summary_updates[-1].get("text") or "")
    if caption_text:
        assert caption_text not in summary_text
    assert "Подпись на медиа показана выше" in summary_text
    assert "Инструкции оператора" in summary_text
    assert "Добавь смайлы" in summary_text
    assert "Доступные каналы:" in summary_text
    assert "📣 -500" in summary_text
    assert "🧪 -600" in summary_text

    await bot._handle_flowers_preview_callback(1234, "send_main", {"id": "cb3"})
    assert bot.pending_flowers_previews.get(1234) is None
    remaining = bot.db.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    assert remaining == 0
    history = bot.db.execute("SELECT metadata FROM posts_history").fetchone()
    assert history is not None
    meta = json.loads(history["metadata"])
    assert meta["asset_ids"]
    assert meta["test"] is False

    await bot.close()


@pytest.mark.asyncio
async def test_publish_guess_arch_with_overlays(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    storage = tmp_path / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    bot.asset_storage = storage
    config = {
        "enabled": True,
        "assets": {"min": 4, "max": 4},
        "weather_city": "Kaliningrad",
    }
    _insert_rubric(bot, "guess_arch", config, rubric_id=2)
    bot.db.execute(
        "INSERT OR IGNORE INTO cities (id, name, lat, lon) VALUES (1, 'Kaliningrad', 0, 0)"
    )
    bot.db.execute(
        """
        INSERT OR REPLACE INTO weather_cache_hour (city_id, timestamp, temperature, weather_code, wind_speed, is_day)
        VALUES (1, ?, 12.5, 3, 5.4, 1)
        """,
        (datetime.utcnow().isoformat(),),
    )
    bot.db.commit()
    now = datetime.utcnow().isoformat()
    for idx in range(4):
        image_path = tmp_path / f"asset_{idx}.jpg"
        Image.new("RGB", (400, 300), color=(idx * 40, 10, 10)).save(image_path)
        metadata = {"date": now}
        file_meta = {"file_id": f"gfile{idx}"}
        asset_id = bot.data.save_asset(
            -3000,
            500 + idx,
            None,
            "",
            tg_chat_id=-3000,
            caption="",
            kind="photo",
            file_meta=file_meta,
            metadata=metadata,
            categories=["photo_weather"],
            rubric_id=2,
        )
        bot.data.update_asset(asset_id, local_path=str(image_path))
        bot.data.update_asset(
            asset_id,
            vision_category="architecture",
            vision_arch_view="фасад",
            vision_photo_weather="пасмурно",
        )

    calls = []

    async def fake_api(method, data=None, *, files=None):
        calls.append({"method": method, "data": data, "files": files})
        if method == "sendMediaGroup":
            return {"ok": True, "result": [{"message_id": 99}]}
        return {"ok": True}

    bot.api_request = fake_api  # type: ignore
    ok = await bot.publish_rubric("guess_arch", channel_id=-777)
    assert ok
    assert calls and calls[0]["files"]
    media_payload = calls[0]["data"]["media"]
    assert len(media_payload) == 4
    remaining = bot.db.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    assert remaining == 0
    history = bot.db.execute("SELECT metadata FROM posts_history").fetchone()
    assert history is not None
    meta = json.loads(history["metadata"])
    assert meta["rubric_code"] == "guess_arch"
    assert "weather" in meta
    assert meta["caption"]
    # ensure overlays cleaned up
    numbered_exists = any(storage.glob("*_numbered_*.png"))
    assert not numbered_exists
    delete_calls = [call for call in calls if call["method"] == "deleteMessage"]
    assert len(delete_calls) == 4
    await bot.close()


@pytest.mark.asyncio
async def test_guess_arch_asset_selection_random(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {
        "enabled": True,
        "assets": {"min": 4, "max": 4},
    }
    _insert_rubric(bot, "guess_arch", config, rubric_id=2)
    now = datetime.utcnow().isoformat()
    for idx in range(12):
        file_meta = {"file_id": f"ra{idx}"}
        asset_id = bot.data.save_asset(
            -3200,
            800 + idx,
            None,
            "",
            tg_chat_id=-3200,
            caption="",
            kind="photo",
            file_meta=file_meta,
            metadata={"date": now},
            categories=["architecture"],
            rubric_id=2,
        )
        bot.data.update_asset(
            asset_id,
            vision_category="architecture",
            vision_arch_view="фасад",
            vision_photo_weather="пасмурно",
        )

    selections = [
        [
            asset.id
            for asset in bot.data.fetch_assets_by_vision_category(
                "architecture",
                rubric_id=2,
                limit=4,
                require_arch_view=True,
                random_order=True,
            )
        ]
        for _ in range(3)
    ]
    assert all(len(selection) == 4 for selection in selections)
    assert len({tuple(selection) for selection in selections}) > 1
    await bot.close()


@pytest.mark.asyncio
async def test_generate_flowers_uses_gpt_4o(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {"enabled": True}
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    rubric = bot.data.get_rubric_by_code("flowers")

    calls: list[dict[str, Any]] = []

    class DummyOpenAI:
        def __init__(self) -> None:
            self.api_key = "test"

        async def generate_json(self, **kwargs):  # type: ignore[override]
            calls.append(kwargs)
            return OpenAIResponse(
                {"greeting": "Доброе утро", "hashtags": ["котопогода", "цветы"]},
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 15,
                    "total_tokens": 25,
                    "request_id": "req-1",
                    "endpoint": "/v1/responses",
                },
            )

    bot.openai = DummyOpenAI()

    greeting, hashtags = await bot._generate_flowers_copy(rubric, ["Москва"], 4)

    assert greeting == "Доброе утро"
    assert hashtags == ["котопогода", "цветы"]
    assert calls, "generate_json was not called"
    request = calls[0]
    assert request["model"] == "gpt-4o"
    assert 0.9 <= request["temperature"] <= 1.1
    assert request["top_p"] == 0.9

    await bot.close()


@pytest.mark.asyncio
async def test_generate_flowers_retries_on_duplicate(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {"enabled": True}
    _insert_rubric(bot, "flowers", config, rubric_id=1)
    rubric = bot.data.get_rubric_by_code("flowers")
    bot.data.record_post_history(1, 1, None, rubric.id, {
        "rubric_code": "flowers",
        "greeting": "Доброе утро",
        "hashtags": ["#котопогода"],
    })

    calls: list[dict[str, Any]] = []

    class DummyOpenAI:
        def __init__(self) -> None:
            self.api_key = "test"

        async def generate_json(self, **kwargs):  # type: ignore[override]
            calls.append(kwargs)
            if len(calls) == 1:
                return OpenAIResponse(
                    {"greeting": "Доброе утро", "hashtags": ["#котопогода"]},
                    {
                        "prompt_tokens": 5,
                        "completion_tokens": 5,
                        "total_tokens": 10,
                        "request_id": "req-first",
                        "endpoint": "/v1/responses",
                    },
                )
            return OpenAIResponse(
                {"greeting": "Привет, друзья", "hashtags": ["#котопогода", "#цветы"]},
                {
                    "prompt_tokens": 6,
                    "completion_tokens": 7,
                    "total_tokens": 13,
                    "request_id": "req-second",
                    "endpoint": "/v1/responses",
                },
            )

    bot.openai = DummyOpenAI()

    greeting, hashtags = await bot._generate_flowers_copy(rubric, [], 4)

    assert greeting == "Привет, друзья"
    assert hashtags == ["#котопогода", "#цветы"]
    assert len(calls) >= 2

    await bot.close()


@pytest.mark.asyncio
async def test_generate_guess_arch_uses_gpt_4o(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    config = {"enabled": True}
    _insert_rubric(bot, "guess_arch", config, rubric_id=2)
    rubric = bot.data.get_rubric_by_code("guess_arch")

    calls: list[dict[str, Any]] = []

    class DummyOpenAI:
        def __init__(self) -> None:
            self.api_key = "test"

        async def generate_json(self, **kwargs):  # type: ignore[override]
            calls.append(kwargs)
            return OpenAIResponse(
                {"caption": "Тест", "hashtags": ["угадай", "архитектура"]},
                {
                    "prompt_tokens": 8,
                    "completion_tokens": 9,
                    "total_tokens": 17,
                    "request_id": "req-3",
                    "endpoint": "/v1/responses",
                },
            )

    bot.openai = DummyOpenAI()

    caption, hashtags = await bot._generate_guess_arch_copy(rubric, 4, None)

    assert caption == "Тест"
    assert hashtags == ["угадай", "архитектура"]
    assert calls
    request = calls[0]
    assert request["model"] == "gpt-4o"
    assert 0.9 <= request["temperature"] <= 1.1
    assert request["top_p"] == 0.9

    await bot.close()


@pytest.mark.asyncio
async def test_rubrics_overview_lists_configs(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    message_counter = 0
    calls: list[tuple[str, dict[str, Any] | None]] = []

    async def fake_api(method, data=None, *, files=None):
        nonlocal message_counter
        calls.append((method, data))
        if method == "sendMessage":
            message_counter += 1
            chat_id = data.get("chat_id") if isinstance(data, dict) else None
            return {
                "ok": True,
                "result": {"message_id": message_counter, "chat": {"id": chat_id}},
            }
        if method == "editMessageText":
            return {
                "ok": True,
                "result": {
                    "message_id": data.get("message_id") if isinstance(data, dict) else None,
                    "chat": {"id": data.get("chat_id") if isinstance(data, dict) else None},
                },
            }
        return {"ok": True}

    bot.api_request = fake_api  # type: ignore
    await bot.start()

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})
    rubrics = bot.data.list_rubrics()
    assert {r.code for r in rubrics} == {"flowers", "guess_arch"}
    bot.data.upsert_rubric(
        "flowers",
        "Flowers",
        config={
            "enabled": True,
            "channel_id": -100,
            "test_channel_id": -200,
            "tz": "+03:00",
            "days": ["mon", "wed"],
            "schedules": [
                {"time": "09:00", "tz": "+03:00", "days": ["mon"], "channel_id": -100},
            ],
        },
    )

    calls.clear()
    await bot.handle_update({"message": {"text": "/rubrics", "from": {"id": 1}}})
    send_calls = [item for item in calls if item[0] == "sendMessage"]
    assert len(send_calls) == 3
    dashboard_method, dashboard_data = send_calls[0]
    assert dashboard_method == "sendMessage"
    assert dashboard_data is not None
    assert "Карточки рубрик" in dashboard_data.get("text", "")
    dashboard_keyboard = dashboard_data.get("reply_markup", {}).get("inline_keyboard", [])
    assert dashboard_keyboard and dashboard_keyboard[0][0]["text"] == "Обновить карточки"

    flowers_message = send_calls[1][1]
    assert flowers_message is not None
    assert "flowers" in flowers_message.get("text", "").lower()
    flowers_keyboard = flowers_message["reply_markup"]["inline_keyboard"]
    assert any(
        btn.get("callback_data") == "rubric_toggle:flowers"
        for row in flowers_keyboard
        for btn in row
    )
    assert bot.rubric_overview_messages[1]["flowers"]["message_id"] is not None

    guess_message = send_calls[2][1]
    assert guess_message is not None
    assert "guess_arch" in guess_message.get("text", "")

    calls.clear()
    await bot.handle_update({"message": {"text": "/rubrics", "from": {"id": 1}}})
    edit_calls = [item for item in calls if item[0] == "editMessageText"]
    assert len(edit_calls) == 3
    await bot.close()


@pytest.mark.asyncio
async def test_rubric_channel_and_schedule_edit_flow(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    message_counter = 0
    calls: list[tuple[str, dict[str, Any] | None]] = []

    async def fake_api(method, data=None, *, files=None):
        nonlocal message_counter
        calls.append((method, data))
        if method == "sendMessage":
            message_counter += 1
            chat_id = data.get("chat_id") if isinstance(data, dict) else None
            return {
                "ok": True,
                "result": {"message_id": message_counter, "chat": {"id": chat_id}},
            }
        if method in {"editMessageText", "editMessageReplyMarkup"}:
            return {
                "ok": True,
                "result": {
                    "message_id": data.get("message_id") if isinstance(data, dict) else None,
                    "chat": {"id": data.get("chat_id") if isinstance(data, dict) else None},
                },
            }
        return {"ok": True}

    bot.api_request = fake_api  # type: ignore
    await bot.start()

    await bot.handle_update({"message": {"text": "/start", "from": {"id": 1}}})
    bot.db.execute("INSERT INTO channels (chat_id, title) VALUES (?, ?)", (-100, "Main"))
    bot.db.execute("INSERT INTO channels (chat_id, title) VALUES (?, ?)", (-200, "Test"))
    bot.db.execute("INSERT INTO channels (chat_id, title) VALUES (?, ?)", (-300, "ArchMain"))
    bot.db.execute("INSERT INTO channels (chat_id, title) VALUES (?, ?)", (-400, "ArchTest"))
    bot.db.commit()

    await bot.handle_update({"message": {"text": "/rubrics", "from": {"id": 1}}})
    flowers_info = bot.rubric_overview_messages[1]["flowers"]
    guess_info = bot.rubric_overview_messages[1]["guess_arch"]
    flowers_message = {"chat": {"id": 1}, "message_id": flowers_info["message_id"]}
    guess_message = {"chat": {"id": 1}, "message_id": guess_info["message_id"]}

    await bot.handle_update(
        {"callback_query": {"id": "f0", "from": {"id": 1}, "data": "rubric_toggle:flowers", "message": flowers_message}}
    )
    await bot.handle_update(
        {"callback_query": {"id": "g0", "from": {"id": 1}, "data": "rubric_toggle:guess_arch", "message": guess_message}}
    )

    await bot.handle_update(
        {"callback_query": {"id": "f1", "from": {"id": 1}, "data": "rubric_channel:flowers:main", "message": flowers_message}}
    )
    await bot.handle_update(
        {"callback_query": {"id": "f2", "from": {"id": 1}, "data": "rubric_channel_set:-100", "message": flowers_message}}
    )
    await bot.handle_update(
        {"callback_query": {"id": "f3", "from": {"id": 1}, "data": "rubric_channel:flowers:test", "message": flowers_message}}
    )
    await bot.handle_update(
        {"callback_query": {"id": "f4", "from": {"id": 1}, "data": "rubric_channel_set:-200", "message": flowers_message}}
    )

    await bot.handle_update(
        {"callback_query": {"id": "f5", "from": {"id": 1}, "data": "rubric_sched_add:flowers", "message": flowers_message}}
    )
    wizard_message = bot.pending[1]["rubric_input"]["message"]
    wizard_callback = {"id": "f6", "from": {"id": 1}, "message": wizard_message}
    await bot.handle_update({"callback_query": {**wizard_callback, "data": "rubric_sched_time"}})
    await bot.handle_update({"callback_query": {**wizard_callback, "data": "rubric_sched_hour:9"}})
    await bot.handle_update({"callback_query": {**wizard_callback, "data": "rubric_sched_minute:30"}})
    await bot.handle_update({"callback_query": {**wizard_callback, "data": "rubric_sched_days"}})
    await bot.handle_update({"callback_query": {**wizard_callback, "data": "rubric_sched_day:mon"}})
    await bot.handle_update({"callback_query": {**wizard_callback, "data": "rubric_sched_day:tue"}})
    await bot.handle_update({"callback_query": {**wizard_callback, "data": "rubric_sched_days_done"}})
    await bot.handle_update({"callback_query": {**wizard_callback, "data": "rubric_sched_save"}})

    await bot.handle_update(
        {"callback_query": {"id": "f7", "from": {"id": 1}, "data": "rubric_sched_toggle:flowers:0", "message": flowers_message}}
    )

    await bot.handle_update(
        {"callback_query": {"id": "g1", "from": {"id": 1}, "data": "rubric_channel:guess_arch:main", "message": guess_message}}
    )
    await bot.handle_update(
        {"callback_query": {"id": "g2", "from": {"id": 1}, "data": "rubric_channel_set:-300", "message": guess_message}}
    )
    await bot.handle_update(
        {"callback_query": {"id": "g3", "from": {"id": 1}, "data": "rubric_channel:guess_arch:test", "message": guess_message}}
    )
    await bot.handle_update(
        {"callback_query": {"id": "g4", "from": {"id": 1}, "data": "rubric_channel_set:-400", "message": guess_message}}
    )
    await bot.handle_update(
        {"callback_query": {"id": "g5", "from": {"id": 1}, "data": "rubric_sched_add:guess_arch", "message": guess_message}}
    )
    g_wizard_message = bot.pending[1]["rubric_input"]["message"]
    g_wizard_callback = {"id": "g6", "from": {"id": 1}, "message": g_wizard_message}
    await bot.handle_update({"callback_query": {**g_wizard_callback, "data": "rubric_sched_time"}})
    await bot.handle_update({"callback_query": {**g_wizard_callback, "data": "rubric_sched_hour:10"}})
    await bot.handle_update({"callback_query": {**g_wizard_callback, "data": "rubric_sched_minute:15"}})
    await bot.handle_update({"callback_query": {**g_wizard_callback, "data": "rubric_sched_days"}})
    await bot.handle_update({"callback_query": {**g_wizard_callback, "data": "rubric_sched_day:wed"}})
    await bot.handle_update({"callback_query": {**g_wizard_callback, "data": "rubric_sched_days_done"}})
    await bot.handle_update({"callback_query": {**g_wizard_callback, "data": "rubric_sched_save"}})

    await bot.handle_update(
        {"callback_query": {"id": "f8", "from": {"id": 1}, "data": "rubric_toggle:flowers", "message": flowers_message}}
    )
    await bot.handle_update(
        {"callback_query": {"id": "g7", "from": {"id": 1}, "data": "rubric_toggle:guess_arch", "message": guess_message}}
    )

    flowers_config = bot.data.get_rubric_config("flowers")
    assert flowers_config is not None
    assert flowers_config.get("channel_id") == -100
    assert flowers_config.get("test_channel_id") == -200
    assert flowers_config.get("enabled") is False
    assert flowers_config.get("schedules") and flowers_config["schedules"][0]["time"] == "09:30"
    assert flowers_config["schedules"][0]["enabled"] is False
    assert flowers_config["schedules"][0]["days"] == ["mon", "tue"]

    guess_config = bot.data.get_rubric_config("guess_arch")
    assert guess_config is not None
    assert guess_config.get("channel_id") == -300
    assert guess_config.get("test_channel_id") == -400
    assert guess_config.get("enabled") is False
    assert guess_config.get("schedules") and guess_config["schedules"][0]["time"] == "10:15"
    assert guess_config["schedules"][0]["days"] == ["wed"]

    await bot.close()


@pytest.mark.asyncio
async def test_overlay_size_stays_within_expected_ratio(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    custom_overlay_path = tmp_path / "custom_overlay.png"
    Image.new("RGBA", (640, 320), (255, 0, 0, 255)).save(custom_overlay_path)

    cases = [
        (custom_overlay_path, (1600, 1200)),
        (tmp_path / "missing_overlay.png", (600, 900)),
        (tmp_path / "missing_overlay.png", (240, 960)),
    ]

    for overlay_path, base_size in cases:
        overlay = bot._load_overlay_image(overlay_path, 1, base_size)
        min_side = min(base_size)
        ratio = overlay.width / min_side
        assert overlay.size[0] == overlay.size[1]
        assert 0.10 <= ratio <= 0.16

    await bot.close()



@pytest.mark.asyncio
async def test_overlay_offset_respects_safe_zone(tmp_path, monkeypatch):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    overlays_dir = tmp_path / "overlays"
    overlays_dir.mkdir()

    captured_offsets: list[tuple[int, int] | None] = []
    original_paste = Image.Image.paste

    def capture_paste(self, im, box=None, mask=None):  # type: ignore[override]
        captured_offsets.append(box)
        return original_paste(self, im, box, mask)

    monkeypatch.setattr(Image.Image, "paste", capture_paste)

    def make_asset(idx: int, size: tuple[int, int]) -> Asset:
        path = tmp_path / f"base_{idx}.png"
        Image.new("RGB", size, (255, 255, 255)).save(path)
        return Asset(
            id=idx,
            channel_id=1,
            tg_chat_id=1,
            message_id=idx,
            origin="test",
            caption_template=None,
            caption=None,
            hashtags=None,
            categories=[],
            kind="photo",
            file_id=f"file-{idx}",
            file_unique_id=f"unique-{idx}",
            file_name=path.name,
            mime_type="image/png",
            file_size=None,
            width=size[0],
            height=size[1],
            duration=None,
            recognized_message_id=None,
            exif_present=False,
            latitude=None,
            longitude=None,
            city=None,
            country=None,
            author_user_id=None,
            author_username=None,
            sender_chat_id=None,
            via_bot_id=None,
            forward_from_user=None,
            forward_from_chat=None,
            local_path=str(path),
            metadata=None,
            vision_results=None,
            rubric_id=None,
            vision_category=None,
            vision_arch_view=None,
            vision_photo_weather=None,
            vision_flower_varieties=None,
            vision_confidence=None,
            vision_caption=None,
        )

    large_asset = make_asset(1, (1280, 720))
    captured_offsets.clear()
    bot._overlay_number(large_asset, 1, {"overlays_dir": str(overlays_dir)})
    assert captured_offsets[-1] == (24, 24)

    small_asset = make_asset(2, (320, 640))
    captured_offsets.clear()
    bot._overlay_number(small_asset, 2, {"overlays_dir": str(overlays_dir)})
    assert captured_offsets[-1] == (12, 12)

    await bot.close()


@pytest.mark.asyncio
async def test_rubric_publish_callback_success(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    bot.db.execute(
        "INSERT OR REPLACE INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, ?)",
        (1, "boss", "+00:00"),
    )
    bot.db.commit()
    _insert_rubric(bot, "flowers", {"enabled": True, "channel_id": -1}, rubric_id=10)

    api_calls: list[tuple[str, dict | None, Any]] = []

    async def fake_api(method, payload=None, *, files=None):
        api_calls.append((method, payload, files))
        return {"ok": True}

    bot.api_request = fake_api  # type: ignore[assignment]

    recorded: dict[str, Any] = {}

    def fake_enqueue(
        code: str,
        *,
        test: bool = False,
        channel_id: int | None = None,
        initiator_id: int | None = None,
        instructions: str | None = None,
    ) -> int:
        recorded["code"] = code
        recorded["test"] = test
        recorded["channel_id"] = channel_id
        recorded["initiator_id"] = initiator_id
        recorded["instructions"] = instructions
        return 314

    bot.enqueue_rubric = fake_enqueue  # type: ignore[assignment]

    message = {"message_id": 5, "chat": {"id": 1}}
    await bot.handle_update(
        {
            "callback_query": {
                "id": "cb-confirm",
                "from": {"id": 1},
                "data": "rubric_publish_confirm:flowers:prod",
                "message": message,
            }
        }
    )
    assert bot.rubric_pending_runs[(1, "flowers")] == "prod"
    api_calls.clear()

    await bot.handle_update(
        {
            "callback_query": {
                "id": "cb-success",
                "from": {"id": 1},
                "data": "rubric_publish_execute:flowers:prod",
                "message": message,
            }
        }
    )

    assert recorded["code"] == "flowers"
    assert recorded["test"] is False
    assert bot.rubric_pending_runs == {}

    ack_payloads = [payload for method, payload, _ in api_calls if method == "answerCallbackQuery"]
    assert any(payload and payload.get("callback_query_id") == "cb-success" for payload in ack_payloads)
    assert any(payload and "Задача поставлена" in payload.get("text", "") for payload in ack_payloads)

    send_payloads = [payload for method, payload, _ in api_calls if method == "sendMessage"]
    assert send_payloads, "callback should send confirmation message"
    confirmation = send_payloads[-1]
    assert confirmation["chat_id"] == 1
    assert "✅ Рабочая публикация рубрики flowers" in confirmation["text"]
    assert "#314" in confirmation["text"]

    await bot.close()


@pytest.mark.asyncio
async def test_rubric_publish_callback_error(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    bot.db.execute(
        "INSERT OR REPLACE INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, ?)",
        (1, "boss", "+00:00"),
    )
    bot.db.commit()
    _insert_rubric(bot, "flowers", {"enabled": True, "test_channel_id": -2}, rubric_id=11)

    api_calls: list[tuple[str, dict | None, Any]] = []

    async def fake_api(method, payload=None, *, files=None):
        api_calls.append((method, payload, files))
        return {"ok": True}

    bot.api_request = fake_api  # type: ignore[assignment]

    def failing_enqueue(
        code: str,
        *,
        test: bool = False,
        channel_id: int | None = None,
        initiator_id: int | None = None,
        instructions: str | None = None,
    ) -> int:
        raise ValueError("нет подходящих ассетов")

    bot.enqueue_rubric = failing_enqueue  # type: ignore[assignment]

    message = {"message_id": 6, "chat": {"id": 1}}
    await bot.handle_update(
        {
            "callback_query": {
                "id": "cb-error-confirm",
                "from": {"id": 1},
                "data": "rubric_publish_confirm:flowers:test",
                "message": message,
            }
        }
    )
    assert bot.rubric_pending_runs[(1, "flowers")] == "test"
    api_calls.clear()

    await bot.handle_update(
        {
            "callback_query": {
                "id": "cb-error",
                "from": {"id": 1},
                "data": "rubric_publish_execute:flowers:test",
                "message": message,
            }
        }
    )

    ack_payloads = [payload for method, payload, _ in api_calls if method == "answerCallbackQuery"]
    assert any(payload and payload.get("callback_query_id") == "cb-error" for payload in ack_payloads)
    assert any(payload and payload.get("show_alert") is True for payload in ack_payloads)
    assert bot.rubric_pending_runs == {}

    send_payloads = [payload for method, payload, _ in api_calls if method == "sendMessage"]
    assert send_payloads, "callback should send error details"
    error_message = send_payloads[-1]
    assert error_message["chat_id"] == 1
    assert "⚠️ тестовая публикация рубрики flowers не запущена" in error_message["text"]
    assert "нет подходящих ассетов" in error_message["text"]

    await bot.close()


@pytest.mark.asyncio
async def test_rubric_publish_cancel_resets_pending(tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    bot.db.execute(
        "INSERT OR REPLACE INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, ?)",
        (1, "boss", "+00:00"),
    )
    bot.db.commit()
    _insert_rubric(bot, "flowers", {"enabled": True, "channel_id": -1}, rubric_id=12)

    api_calls: list[tuple[str, dict | None, Any]] = []

    async def fake_api(method, payload=None, *, files=None):
        api_calls.append((method, payload, files))
        return {"ok": True}

    bot.api_request = fake_api  # type: ignore[assignment]

    invoked = {"called": False}

    def fail_if_called(*args, **kwargs):
        invoked["called"] = True
        return 0

    bot.enqueue_rubric = fail_if_called  # type: ignore[assignment]

    message = {"message_id": 7, "chat": {"id": 1}}
    await bot.handle_update(
        {
            "callback_query": {
                "id": "cb-cancel-confirm",
                "from": {"id": 1},
                "data": "rubric_publish_confirm:flowers:prod",
                "message": message,
            }
        }
    )
    assert bot.rubric_pending_runs[(1, "flowers")] == "prod"
    api_calls.clear()

    await bot.handle_update(
        {
            "callback_query": {
                "id": "cb-cancel",
                "from": {"id": 1},
                "data": "rubric_publish_cancel:flowers",
                "message": message,
            }
        }
    )

    assert bot.rubric_pending_runs == {}
    assert invoked["called"] is False
    ack_payloads = [payload for method, payload, _ in api_calls if method == "answerCallbackQuery"]
    assert any(payload and payload.get("callback_query_id") == "cb-cancel" for payload in ack_payloads)

    await bot.close()
