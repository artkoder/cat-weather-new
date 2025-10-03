import os
import sqlite3
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_access import DataAccess
from main import Bot, apply_migrations


def _prepare_legacy_db(path: str, *, channel_id: int = 123, message_id: int = 456) -> None:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS asset_channel (
            channel_id INTEGER PRIMARY KEY
        );
        DELETE FROM asset_channel;
        INSERT INTO asset_channel (channel_id) VALUES ({channel});
        CREATE TABLE IF NOT EXISTS asset_images (
            message_id INTEGER PRIMARY KEY,
            hashtags TEXT,
            template TEXT,
            used_at TEXT
        );
        DELETE FROM asset_images;
        INSERT INTO asset_images (message_id, hashtags, template, used_at)
        VALUES ({message}, '#sun #Cat', 'Legacy caption', NULL);
        """.format(channel=channel_id, message=message_id)
    )
    conn.commit()
    conn.close()


def test_sql_migration_transfers_legacy_assets(tmp_path):
    db_path = tmp_path / "db.sqlite"
    _prepare_legacy_db(str(db_path))

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    apply_migrations(conn)

    data = DataAccess(conn)
    asset = data.get_next_asset({"#sun"})

    assert asset is not None
    assert asset.message_id == 456
    assert asset.channel_id == 123
    assert set(asset.categories) == {"#sun", "#Cat"}
    assert asset.caption_template == "Legacy caption"

    table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='asset_images'"
    ).fetchone()
    assert table is None

    conn.close()


@pytest.mark.asyncio
async def test_bot_publish_weather_uses_migrated_legacy_assets(tmp_path):
    db_path = tmp_path / "db.sqlite"
    _prepare_legacy_db(str(db_path), channel_id=321, message_id=654)

    bot = Bot("dummy", str(db_path))

    calls: list[tuple[str, dict | None]] = []

    async def dummy_request(method: str, data: dict | None = None):
        calls.append((method, data))
        if method == "copyMessage":
            return {"ok": True, "result": {"message_id": 111}}
        return {"ok": True}

    bot.api_request = dummy_request  # type: ignore[assignment]

    ok = await bot.publish_weather(999, {"#sun"})

    assert ok is True
    assert calls
    method, payload = calls[0]
    assert method == "copyMessage"
    assert payload is not None
    assert payload["from_chat_id"] == 321
    assert payload["message_id"] == 654

    await bot.close()
