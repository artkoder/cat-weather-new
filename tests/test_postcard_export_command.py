from __future__ import annotations

import csv
import io
import json
import os
import sys
from typing import Any

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import main as main_module


def _insert_asset(
    conn,
    *,
    asset_id: str,
    postcard_score: int | None,
    payload: dict[str, Any] | None,
    created_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO assets (id, payload_json, created_at, postcard_score)
        VALUES (?, ?, ?, ?)
        """,
        (
            asset_id,
            json.dumps(payload or {}, ensure_ascii=False),
            created_at,
            postcard_score,
        ),
    )


@pytest.mark.asyncio
async def test_postcard_photos_db_command_exports_scores(monkeypatch, tmp_path):
    db_path = tmp_path / "postcard.db"
    bot = main_module.Bot("dummy", str(db_path))

    timestamp = "2024-11-23T12:00:00+00:00"
    _insert_asset(
        bot.db,
        asset_id="asset-seven",
        postcard_score=7,
        payload={"city": "A"},
        created_at=timestamp,
    )
    _insert_asset(
        bot.db,
        asset_id="asset-eight",
        postcard_score=8,
        payload={"city": "B"},
        created_at=timestamp,
    )
    _insert_asset(
        bot.db,
        asset_id="asset-vision",
        postcard_score=None,
        payload={"city": "Vision"},
        created_at=timestamp,
    )
    _insert_asset(
        bot.db,
        asset_id="asset-low",
        postcard_score=5,
        payload={"city": "Low"},
        created_at=timestamp,
    )

    bot.db.execute(
        """
        INSERT INTO vision_results (
            asset_id,
            provider,
            status,
            category,
            arch_view,
            photo_weather,
            flower_varieties,
            confidence,
            result_json,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "asset-vision",
            "openai",
            "done",
            None,
            None,
            None,
            None,
            None,
            json.dumps({"postcard_score": 8, "tags": ["postcard"]}, ensure_ascii=False),
            timestamp,
            timestamp,
        ),
    )
    bot.db.commit()

    user_id = 4242
    bot.db.execute(
        "INSERT INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, ?)",
        (user_id, "tester", "+00:00"),
    )
    bot.db.commit()

    api_calls: list[tuple[str, dict[str, Any] | None, dict[str, tuple[str, bytes]] | None]] = []

    async def capture_api_request(
        self,
        method: str,
        data: dict[str, Any] | None = None,
        *,
        files: dict[str, tuple[str, bytes]] | None = None,
    ) -> dict[str, Any]:
        api_calls.append((method, data, files))
        return {"ok": True, "result": {}}

    monkeypatch.setattr(main_module.Bot, "api_request", capture_api_request, raising=False)

    message = {
        "message_id": 1,
        "from": {"id": user_id, "username": "tester"},
        "chat": {"id": user_id},
        "text": "/postcard_photos_db",
    }

    await bot.handle_message(message)

    send_docs = [call for call in api_calls if call[0] == "sendDocument"]
    assert len(send_docs) == 1
    _, payload, files = send_docs[0]
    assert payload is not None
    assert files is not None
    assert payload["chat_id"] == user_id
    assert "document" in files
    filename, file_bytes = files["document"]
    assert filename.startswith("postcard_photos_")
    assert filename.endswith(".csv")

    csv_reader = csv.DictReader(io.StringIO(file_bytes.decode("utf-8")))
    rows = list(csv_reader)
    assert len(rows) == 4
    asset_ids = {row["asset_id"] for row in rows}
    assert asset_ids == {"asset-seven", "asset-eight", "asset-vision", "asset-low"}
    vision_row = next(row for row in rows if row["asset_id"] == "asset-vision")
    assert vision_row["resolved_postcard_score"] == "8"
    assert vision_row["score_source"] == "vision"
    low_row = next(row for row in rows if row["asset_id"] == "asset-low")
    assert low_row["resolved_postcard_score"] == "5"
    assert low_row["score_source"] == "column"

    await bot.close()


@pytest.mark.asyncio
async def test_postcard_photos_db_command_supports_pretty(monkeypatch, tmp_path):
    db_path = tmp_path / "postcard_pretty.db"
    bot = main_module.Bot("dummy", str(db_path))

    timestamp = "2024-11-23T12:00:00+00:00"
    _insert_asset(
        bot.db,
        asset_id="asset-pretty",
        postcard_score=7,
        payload={"city": "Pretty", "extra": {"nested": 1}},
        created_at=timestamp,
    )
    bot.db.commit()

    user_id = 777
    bot.db.execute(
        "INSERT INTO users (user_id, username, is_superadmin, tz_offset) VALUES (?, ?, 1, ?)",
        (user_id, "pretty", "+00:00"),
    )
    bot.db.commit()

    api_calls: list[tuple[str, dict[str, Any] | None, dict[str, tuple[str, bytes]] | None]] = []

    async def capture_api_request(
        self,
        method: str,
        data: dict[str, Any] | None = None,
        *,
        files: dict[str, tuple[str, bytes]] | None = None,
    ) -> dict[str, Any]:
        api_calls.append((method, data, files))
        return {"ok": True, "result": {}}

    monkeypatch.setattr(main_module.Bot, "api_request", capture_api_request, raising=False)

    message = {
        "message_id": 2,
        "from": {"id": user_id, "username": "pretty"},
        "chat": {"id": user_id},
        "text": "/postcard_photos_db pretty",
    }

    await bot.handle_message(message)

    send_docs = [call for call in api_calls if call[0] == "sendDocument"]
    assert len(send_docs) == 1
    _, _, files = send_docs[0]
    assert files and "document" in files
    file_bytes = files["document"][1]
    csv_reader = csv.DictReader(io.StringIO(file_bytes.decode("utf-8")))
    rows = list(csv_reader)
    assert len(rows) == 1
    payload_dump = rows[0]["payload_json"]
    assert "\n" in payload_dump  # pretty flag should introduce indentation

    await bot.close()
