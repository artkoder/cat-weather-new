import logging
import os
import random
import sys
import types
from datetime import datetime

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_access import Asset, Rubric  # noqa: E402
from main import Bot  # noqa: E402


@pytest.mark.asyncio
async def test_publish_new_year_filters_missing_tag(tmp_path, monkeypatch, caplog):
    bot = Bot("test-token", str(tmp_path / "db.sqlite"))
    bot.asset_storage = tmp_path

    asset = Asset(
        id="asset-1",
        upload_id=None,
        file_ref="file-ref",
        content_type="image/jpeg",
        sha256=None,
        width=10,
        height=10,
        exif_json=None,
        labels_json=None,
        tg_message_id=None,
        payload_json=None,
        created_at=datetime.utcnow().isoformat(),
        payload={"tags": ["winter", "snow"]},
    )

    monkeypatch.setattr(bot, "_select_new_year_assets", lambda *, limit, test: [asset])

    notifications: list[tuple[str | None, int | None, int]] = []

    async def fake_notify(self, *, rubric_title, initiator_id, requested_count):
        notifications.append((rubric_title, initiator_id, requested_count))

    bot._notify_new_year_no_inventory = types.MethodType(fake_notify, bot)  # type: ignore[assignment]

    ensure_called = False

    async def fake_ensure(_asset):  # type: ignore[unused-argument]
        nonlocal ensure_called
        ensure_called = True
        return None, False

    bot._ensure_asset_source = fake_ensure  # type: ignore[assignment]

    monkeypatch.setattr(random, "randint", lambda a, b: a)

    rubric = Rubric(
        id=1,
        code="new_year",
        title="Новогодняя рубрика",
        description=None,
        config={"assets": {"min": 1, "max": 1}},
    )

    caplog.set_level(logging.INFO)
    result = await bot._publish_new_year(
        rubric,
        channel_id=123,
        test=True,
        initiator_id=42,
    )

    assert result is True
    assert notifications == [(rubric.title, 42, 1)]
    assert "skip_missing_new_year_tag" in caplog.text
    assert ensure_called is False
