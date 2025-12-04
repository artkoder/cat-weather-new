import json
import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import Bot


@pytest.mark.asyncio
async def test_download_ocr_payload_success(monkeypatch, tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    calls: list[tuple[str, dict]] = []

    async def fake_api_request(method, data, files=None):
        calls.append((method, data))
        if method == "copyMessage":
            return {"ok": True, "result": {"message_id": 111}}
        return {"ok": True, "result": {}}

    async def fake_download_file(file_id, dest_path):
        path = tmp_path / "ocr.json"
        path.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
        return path

    monkeypatch.setattr(bot, "api_request", fake_api_request)
    monkeypatch.setattr(
        bot, "_collect_asset_metadata", lambda payload: {"file_meta": {"file_id": "file-1"}}
    )
    monkeypatch.setattr(bot, "_download_file", fake_download_file)

    result = await bot._download_ocr_payload(channel_id=1, message_id=2, target_chat_id=3)

    assert result.payload == {"foo": "bar"}
    assert result.error is None
    assert result.stage == "success"
    assert any(call[0] == "copyMessage" for call in calls)
    await bot.close()


@pytest.mark.asyncio
async def test_download_ocr_payload_fallback_copy(monkeypatch, tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    bot._raw_answer_scans_channel_id = "99"
    calls: list[tuple[str, dict]] = []

    async def fake_api_request(method, data, files=None):
        calls.append((method, data))
        if method == "copyMessage":
            if str(data.get("chat_id")) != str(bot._raw_answer_scans_channel_id):
                return {"ok": False, "result": {}}
            return {"ok": True, "result": {"message_id": 777}}
        return {"ok": True, "result": {}}

    async def fake_download_file(file_id, dest_path):
        path = tmp_path / "ocr_fallback.json"
        path.write_text(json.dumps({"page": 1}), encoding="utf-8")
        return path

    monkeypatch.setattr(bot, "api_request", fake_api_request)
    monkeypatch.setattr(
        bot, "_collect_asset_metadata", lambda payload: {"file_meta": {"file_id": "file-2"}}
    )
    monkeypatch.setattr(bot, "_download_file", fake_download_file)

    result = await bot._download_ocr_payload(channel_id=10, message_id=20, target_chat_id=30)

    assert result.payload == {"page": 1}
    assert result.error is None
    assert result.stage == "success"
    assert any(
        call[1].get("chat_id") == bot._raw_answer_scans_channel_id
        for call in calls
        if call[0] == "copyMessage"
    )
    await bot.close()


@pytest.mark.asyncio
async def test_download_ocr_payload_parse_error(monkeypatch, tmp_path):
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))

    async def fake_api_request(method, data, files=None):
        if method == "copyMessage":
            return {"ok": True, "result": {"message_id": 333}}
        return {"ok": True, "result": {}}

    async def fake_download_file(file_id, dest_path):
        path = tmp_path / "ocr_invalid.json"
        path.write_text("not json", encoding="utf-8")
        return path

    monkeypatch.setattr(bot, "api_request", fake_api_request)
    monkeypatch.setattr(
        bot, "_collect_asset_metadata", lambda payload: {"file_meta": {"file_id": "file-3"}}
    )
    monkeypatch.setattr(bot, "_download_file", fake_download_file)

    result = await bot._download_ocr_payload(channel_id=5, message_id=6, target_chat_id=7)

    assert result.payload is None
    assert result.raw_bytes
    assert result.error == "parse_error"
    assert result.stage == "parse_error"
    await bot.close()
