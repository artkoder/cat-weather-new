from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from jobs import Job

if TYPE_CHECKING:
    from main import Bot


async def handle_publish_rubric(bot: Bot, job: Job) -> None:
    payload = job.payload or {}
    code = payload.get("rubric_code")
    if not code:
        logging.warning("Rubric job %s missing code", job.id)
        return
    test_mode = bool(payload.get("test"))
    schedule_key = payload.get("schedule_key")
    scheduled_at = payload.get("scheduled_at")
    old_payload_channel = payload.get("channel_id")
    resolved_channel: int | None = None
    if schedule_key and not schedule_key.startswith("manual"):
        rubric = bot.data.get_rubric_by_code(code)
        if rubric:
            config = rubric.config or {}
            slot_channel_id = payload.get("slot_channel_id")
            if slot_channel_id:
                resolved_channel = slot_channel_id
            elif test_mode:
                resolved_channel = config.get("test_channel_id")
            else:
                resolved_channel = config.get("channel_id")
            if old_payload_channel and old_payload_channel != resolved_channel:
                logging.info(
                    "Channel resolved at execution: rubric=%s, old_payload_channel=%s, resolved=%s",
                    code,
                    old_payload_channel,
                    resolved_channel,
                )
            elif not old_payload_channel and resolved_channel:
                logging.debug(
                    "Channel resolved at execution: rubric=%s, resolved=%s",
                    code,
                    resolved_channel,
                )
    else:
        resolved_channel = old_payload_channel
        logging.info(
            "_job_publish_rubric (manual): rubric=%s, test_mode=%s, "
            "schedule_key=%s, payload_channel=%s, resolved=%s",
            code,
            test_mode,
            schedule_key,
            old_payload_channel,
            resolved_channel,
        )
    success = await publish_rubric(
        bot,
        code,
        channel_id=resolved_channel,
        test=test_mode,
        job=job,
        initiator_id=payload.get("initiator_id"),
        instructions=payload.get("instructions"),
    )
    if success and schedule_key and scheduled_at:
        try:
            run_at = datetime.fromisoformat(scheduled_at)
        except ValueError:
            run_at = datetime.utcnow()
        bot.data.mark_rubric_run(code, schedule_key, run_at)
    if not success:
        raise RuntimeError(f"Failed to publish rubric {code}")


async def publish_rubric(
    bot: Bot,
    code: str,
    channel_id: int | None = None,
    *,
    test: bool = False,
    job: Job | None = None,
    initiator_id: int | None = None,
    instructions: str | None = None,
) -> bool:
    rubric = bot.data.get_rubric_by_code(code)
    if not rubric:
        logging.warning("Rubric %s not found", code)
        return False
    config = rubric.config or {}
    target = channel_id
    channel_source = "provided" if channel_id is not None else "config"
    if target is None:
        prod_channel = config.get("channel_id")
        test_channel = config.get("test_channel_id")
        target = test_channel if test else prod_channel
    logging.info(
        "publish_rubric: rubric=%s, test=%s, prod_channel=%s, "
        "test_channel=%s, channel_source=%s, resolved=%s",
        code,
        test,
        config.get("channel_id"),
        config.get("test_channel_id"),
        channel_source,
        target,
    )
    if target is None:
        logging.warning("Rubric %s missing channel configuration", code)
        return False
    handler = getattr(bot, f"_publish_{code}", None)
    if not handler:
        logging.warning("No handler defined for rubric %s", code)
        return False
    return await handler(
        rubric,
        int(target),
        test=test,
        job=job,
        initiator_id=initiator_id,
        instructions=instructions,
    )
