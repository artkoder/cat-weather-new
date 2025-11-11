"""Test prod vs test channel resolution.

This test verifies the fix for the bug where rubric publications
always went to test_channel regardless of whether prod or test
publish button was clicked.
"""

import json
import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import Bot

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "dummy")


@pytest.mark.asyncio
async def test_prod_publish_uses_prod_channel(tmp_path):
    """Test that prod publication (test=False) uses prod_channel, not test_channel."""
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    
    config = {
        "enabled": True,
        "channel_id": -1002,  # Prod channel
        "test_channel_id": -1001,  # Test channel (different from prod!)
    }
    bot.data.upsert_rubric("flowers", "Flowers", config=config)
    
    # Enqueue a PROD publication (test=False)
    job_id = bot.enqueue_rubric("flowers", test=False)
    
    job_row = bot.db.execute(
        "SELECT payload FROM jobs_queue WHERE id=?", (job_id,)
    ).fetchone()
    payload = json.loads(job_row["payload"])
    
    # Assert that prod channel is used, not test channel
    assert payload["channel_id"] == -1002, (
        f"Prod publish should use channel_id (-1002), not test_channel_id (-1001). "
        f"Got: {payload['channel_id']}"
    )
    assert payload["test"] is False
    assert payload["schedule_key"] == "manual"
    
    await bot.close()


@pytest.mark.asyncio
async def test_test_publish_uses_test_channel(tmp_path):
    """Test that test publication (test=True) uses test_channel, not prod_channel."""
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    
    config = {
        "enabled": True,
        "channel_id": -1002,  # Prod channel
        "test_channel_id": -1001,  # Test channel (different from prod!)
    }
    bot.data.upsert_rubric("flowers", "Flowers", config=config)
    
    # Enqueue a TEST publication (test=True)
    job_id = bot.enqueue_rubric("flowers", test=True)
    
    job_row = bot.db.execute(
        "SELECT payload FROM jobs_queue WHERE id=?", (job_id,)
    ).fetchone()
    payload = json.loads(job_row["payload"])
    
    # Assert that test channel is used, not prod channel
    assert payload["channel_id"] == -1001, (
        f"Test publish should use test_channel_id (-1001), not channel_id (-1002). "
        f"Got: {payload['channel_id']}"
    )
    assert payload["test"] is True
    assert payload["schedule_key"] == "manual-test"
    
    await bot.close()


@pytest.mark.asyncio
async def test_changing_prod_channel_affects_only_prod_publishes(tmp_path):
    """Test that changing prod_channel doesn't affect test publications."""
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    
    config = {
        "enabled": True,
        "channel_id": -1002,
        "test_channel_id": -1001,
    }
    bot.data.upsert_rubric("flowers", "Flowers", config=config)
    
    # Prod publish before change
    job_id_1 = bot.enqueue_rubric("flowers", test=False)
    payload_1 = json.loads(
        bot.db.execute("SELECT payload FROM jobs_queue WHERE id=?", (job_id_1,)).fetchone()[
            "payload"
        ]
    )
    assert payload_1["channel_id"] == -1002
    
    # Change prod channel
    config["channel_id"] = -2002
    bot.data.save_rubric_config("flowers", config)
    
    # Prod publish after change should use new channel
    job_id_2 = bot.enqueue_rubric("flowers", test=False)
    payload_2 = json.loads(
        bot.db.execute("SELECT payload FROM jobs_queue WHERE id=?", (job_id_2,)).fetchone()[
            "payload"
        ]
    )
    assert payload_2["channel_id"] == -2002, (
        f"Prod publish should use new channel_id (-2002), got {payload_2['channel_id']}"
    )
    
    # Test publish should still use original test channel
    job_id_3 = bot.enqueue_rubric("flowers", test=True)
    payload_3 = json.loads(
        bot.db.execute("SELECT payload FROM jobs_queue WHERE id=?", (job_id_3,)).fetchone()[
            "payload"
        ]
    )
    assert payload_3["channel_id"] == -1001, (
        f"Test publish should still use test_channel_id (-1001), got {payload_3['channel_id']}"
    )
    
    await bot.close()


@pytest.mark.asyncio
async def test_changing_test_channel_affects_only_test_publishes(tmp_path):
    """Test that changing test_channel doesn't affect prod publications."""
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    
    config = {
        "enabled": True,
        "channel_id": -1002,
        "test_channel_id": -1001,
    }
    bot.data.upsert_rubric("flowers", "Flowers", config=config)
    
    # Test publish before change
    job_id_1 = bot.enqueue_rubric("flowers", test=True)
    payload_1 = json.loads(
        bot.db.execute("SELECT payload FROM jobs_queue WHERE id=?", (job_id_1,)).fetchone()[
            "payload"
        ]
    )
    assert payload_1["channel_id"] == -1001
    
    # Change test channel
    config["test_channel_id"] = -2001
    bot.data.save_rubric_config("flowers", config)
    
    # Test publish after change should use new channel
    job_id_2 = bot.enqueue_rubric("flowers", test=True)
    payload_2 = json.loads(
        bot.db.execute("SELECT payload FROM jobs_queue WHERE id=?", (job_id_2,)).fetchone()[
            "payload"
        ]
    )
    assert payload_2["channel_id"] == -2001, (
        f"Test publish should use new test_channel_id (-2001), got {payload_2['channel_id']}"
    )
    
    # Prod publish should still use original prod channel
    job_id_3 = bot.enqueue_rubric("flowers", test=False)
    payload_3 = json.loads(
        bot.db.execute("SELECT payload FROM jobs_queue WHERE id=?", (job_id_3,)).fetchone()[
            "payload"
        ]
    )
    assert payload_3["channel_id"] == -1002, (
        f"Prod publish should still use channel_id (-1002), got {payload_3['channel_id']}"
    )
    
    await bot.close()
