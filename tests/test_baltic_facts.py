import asyncio
import hashlib
import math
import os
import random
from datetime import datetime, timedelta

import pytest

from facts.loader import Fact, load_baltic_facts
from main import Bot


def test_baltic_facts_parse_md():
    os.environ.setdefault("TELEGRAM_BOT_TOKEN", "dummy")
    facts = load_baltic_facts()
    assert isinstance(facts, list)
    assert len(facts) >= 30
    for fact in facts:
        assert isinstance(fact, Fact)
        expected_id = hashlib.sha1(fact.text.encode("utf-8")).hexdigest()
        assert fact.id == expected_id


@pytest.mark.asyncio
async def test_baltic_facts_no_repeat_in_window(tmp_path):
    os.environ.setdefault("TELEGRAM_BOT_TOKEN", "dummy")
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    try:
        facts = load_baltic_facts()[:20]
        rng = random.Random(42)
        window_days = max(7, min(21, math.ceil(len(facts) * 0.6)))
        start = datetime(2024, 1, 1)
        seen: set[str] = set()
        for offset in range(window_days):
            now = start + timedelta(days=offset)
            fact, info = bot._select_baltic_fact(facts, now=now, rng=rng)
            assert fact is not None
            assert fact.id not in seen
            seen.add(fact.id)
        assert len(seen) == window_days
    finally:
        await bot.close()


@pytest.mark.asyncio
async def test_baltic_facts_weights_prefer_rare(tmp_path):
    os.environ.setdefault("TELEGRAM_BOT_TOKEN", "dummy")
    bot = Bot("dummy", str(tmp_path / "db.sqlite"))
    try:
        facts = load_baltic_facts()[:2]
        assert len(facts) == 2
        common_fact, rare_fact = facts
        # Simulate previous usage for the common fact far in the past
        for day in range(5):
            bot.data.record_fact_selection(common_fact.id, now_ts=day * 86400, day_utc=day)
        # Choose on a future day so rollout window ignores the synthetic past
        now = datetime(2024, 5, 1)
        fact, info = bot._select_baltic_fact(facts, now=now, rng=random.Random(7))
        weights = info.get("weights") or {}
        assert weights[rare_fact.id] > weights[common_fact.id]
        assert fact is not None
        assert fact.id == rare_fact.id
    finally:
        await bot.close()
