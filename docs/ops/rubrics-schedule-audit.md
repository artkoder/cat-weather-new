# Rubric Scheduler Audit – SEA Duplicates

_Date: 2025-11-16_

## 1. Objective and scope
- Reconstruct how rubric schedules are persisted and executed.
- Explain why a previously deleted SEA schedule kept firing (duplicate posts around 21:30 Kaliningrad time).
- Catalogue existing admin tooling and time-zone handling.
- Produce actionable remediation options (with and without an execution "safety lock").

## 2. Implementation inventory

### 2.1 Scheduler stack
- **Job runner**: custom `JobQueue` (`jobs.py`, ~L43) backed by the `jobs_queue` table (`migrations/0012_core_schema.sql`). Columns include `id`, `name`, `payload`, `status`, `attempts`, `available_at`, `last_error`, `created_at`, `updated_at`.
- **Persistence helpers**: `DataAccess.enqueue_job/list_jobs/delete_job` (`data_access.py`, L3283-L3392) expose CRUD for the queue.
- **Scheduler loop**: `Bot.schedule_loop()` (`main.py`, L18973-L18990) wakes every `SCHED_INTERVAL_SEC` (default 30 s) to run `process_rubric_schedule()` and enqueue rubric jobs.
- **Rubric state**: `rubrics` table stores JSON configs; `rubric_schedule_state` table (`migrations/0008_rubric_scheduler.sql`) tracks `next_run_at`/`last_run_at` per `(rubric_code, schedule_key)`.

### 2.2 Rubric scheduling flow
1. `process_rubric_schedule()` (`main.py`, L10022-L10080) loads every enabled rubric/config.
2. Each schedule entry is normalised (config default, fallbacks for channel and `tz`). `schedule_key` is derived on the fly as `slot_channel_id:{idx}:{time}` or `rubric.code:{idx}:{time}` when no explicit key exists.
3. State bootstrap: if no `rubric_schedule_state` row exists or it is in the past, `_compute_next_rubric_run()` recalculates the next UTC slot using the stored string `tz_offset` (`main.py`, L11606-L11630). The function uses a fixed offset (`±HH:MM`), not IANA zones, so daylight shifts are ignored.
4. Duplicate guard: `_rubric_job_exists()` (`main.py`, L10081-L10093) performs a `SELECT` for a job whose payload has the same `rubric_code` **and** `schedule_key`. If none is found, the scheduler inserts a new `publish_rubric` job via `self.jobs.enqueue(...)` without `dedupe=True`.
5. Job payload always contains `rubric_code`, `schedule_key`, `scheduled_at` (UTC ISO string), optional `slot_channel_id`, plus `tz_offset`. Channel resolution is deferred until execution (`_job_publish_rubric`, L11416-L11477).

### 2.3 Admin tooling
- `/rubric_jobs ...` commands use `_find_rubric_jobs()` / `_identify_canonical_rubric_jobs()` (`main.py`, L10332-L10460) to list, preview, and delete entries. Rubric attribution relies on heuristics in `classify_rubric_job()` (L10216-L10330).
- `/purge_sea_jobs [keep=false]` (`main.py`, L7300-L7369) filters jobs classified as `sea`, keeps one canonical job (matching the current config’s expected key/time) unless `keep=false`, and deletes the rest via `DataAccess.delete_job()`.
- Schedule editing wizards (`main.py`, L9722-L9959) call `DataAccess.add_rubric_schedule`, `update_rubric_schedule`, and `remove_rubric_schedule` (L3170-L3205). None of these routines trigger queue cleanup. `remove_rubric_schedule` deletes `rubric_schedule_state` rows only when an explicit `key` field exists in the removed schedule, which is not the default.

## 3. Data inspection
The point-in-time queue snapshot collected during the audit lives at [`docs/ops/rubrics_pending_jobs.json`](./rubrics_pending_jobs.json). Both entries stem from the reproduction (see §4):

| job_id | status | scheduled_at (UTC) | schedule_key   | rubric |
|--------|--------|--------------------|----------------|--------|
| 1      | queued | 2024-11-16T19:30:00 | `sea:0:21:30` | sea    |
| 2      | queued | 2024-11-16T17:30:00 | `sea:0:19:30` | sea    |

`available_at` is `NULL` because the historical reference date used for the reproduction placed both slots in the past; in production the column is a future timestamp.

Additional inspection showed that `rubric_schedule_state` retained both derived keys even after the schedule was deleted:
```sql
SELECT rubric_code, schedule_key, next_run_at
FROM rubric_schedule_state;
-- → (sea, sea:0:21:30, 2024-11-16T19:30:00)
--   (sea, sea:0:19:30, 2024-11-16T17:30:00)
```

## 4. Reproduction and logs
Reproduced the duplicate job scenario with a disposable SQLite database (`/tmp/rubric_audit.db`) using the public API:

```bash
.venv/bin/python <<'PY'
import asyncio, json
from datetime import datetime
from main import Bot

async def run():
    bot = Bot("dummy", "/tmp/rubric_audit.db")
    reference = datetime(2024, 11, 16, 16, 0)

    # 1. Add schedule A @21:30
    bot.data.save_rubric_config("sea", {
        "enabled": True,
        "channel_id": 1001,
        "test_channel_id": 2002,
        "tz": "+02:00",
        "schedules": [{"time": "21:30", "tz": "+02:00", "enabled": True}],
    })
    await bot.process_rubric_schedule(reference=reference)

    # 2. Edit the same entry to 19:30 (schedule B)
    config = bot.data.get_rubric_config("sea")
    config["schedules"][0]["time"] = "19:30"
    bot.data.save_rubric_config("sea", config)
    await bot.process_rubric_schedule(reference=reference)

    # 3. Delete the schedule
    bot.data.remove_rubric_schedule("sea", 0)
    await bot.process_rubric_schedule(reference=reference)

    rows = bot.db.execute("SELECT id, payload FROM jobs_queue ORDER BY id").fetchall()
    for row in rows:
        payload = json.loads(row["payload"])
        print(row["id"], payload["schedule_key"], payload["scheduled_at"])

    await bot.close()

asyncio.run(run())
PY
```

Output (abridged):
```
Enqueued job 1 (publish_rubric) … schedule_key=sea:0:21:30
Enqueued job 2 (publish_rubric) … schedule_key=sea:0:19:30
1 sea:0:21:30 2024-11-16T19:30:00
2 sea:0:19:30 2024-11-16T17:30:00
```
Both jobs remain in `jobs_queue` even after the schedule is removed from the rubric config.

## 5. Findings
1. **Derived schedule keys are unstable.** Keys default to `rubric.code:{index}:{time}`. Editing time or channel changes the derived key, so `_rubric_job_exists` no longer matches the previous job.
2. **Schedule CRUD never clears the queue.** `DataAccess.update_rubric_schedule` and `remove_rubric_schedule` save the new config but leave existing queued jobs untouched. `remove_rubric_schedule` attempts to delete schedule-state rows, but only when a literal `key` field existed in the removed entry, which is uncommon.
3. **Queue dedupe is disabled.** `JobQueue.enqueue` can skip exact duplicates (`dedupe=True`), but the scheduler never enables it. Because the payload changes (`schedule_key` / `scheduled_at`), each edit enqueues a fresh job.
4. **Admin purge is reactive.** `/purge_sea_jobs` can clean duplicates but requires manual intervention and relies on the same derived key heuristics. If config has no schedules, the command keeps whichever job is closest to "now", but duplicates still slip through until someone runs it.
5. **Execution safety is best-effort.** `_sea_publish_guard` prevents the exact same sea/storm-state from publishing twice within ~60 seconds, but does not guard against jobs spaced minutes apart (as in the duplicate schedule scenario).

### Root cause of the 21:30 ghost job
- Original SEA slot at 21:30 was scheduled with key `sea:0:21:30`.
- When ops changed the slot to a different time, the scheduler derived a new key, scheduled a new job, but left the `sea:0:21:30` job untouched.
- Deleting the schedule removed it from the config but, lacking a persisted key, `remove_rubric_schedule` could not locate and delete the matching queue row or `rubric_schedule_state` entry. The orphan job kept executing on its original cadence.

## 6. Remediation plan

### 6.1 Minimal fix (no new safety lock)
- **Persist canonical schedule IDs.** When adding a schedule, stamp it with a stable `key` (UUID or hash). `update_rubric_schedule` must preserve the key; `process_rubric_schedule` should prefer persisted keys.
- **Atomic replace on edit/delete.** For config mutations, fetch previous schedule data, compute the relevant keys, and call `_delete_future_rubric_jobs(rubric_code, reason, keys)` (extend helper to accept a key whitelist). Also clear matching `rubric_schedule_state` rows.
- **Enable payload dedupe.** When enqueuing from `process_rubric_schedule`, pass `dedupe=True` to prevent exact duplicates if a task is retried without changes.

_Risk / impact_: touching rubric config serializers and scheduler logic. Requires regression coverage for all rubric types (sea, flowers, guess_arch) and admin flows. Minimal schema change (only config JSON). No downtime.

_Testing_: unit tests around `_compute_next_rubric_run`, new helper to delete by key, and admin command integration tests to ensure duplicates disappear after edits.

### 6.2 Robust fix (with safety lock)
- **Startup reconciliation.** On bot start, scan `jobs_queue` grouped by `(rubric_code, schedule_key)` and drop extras compared to the canonical plan computed from config/state.
- **Schema-backed schedule registry.** Introduce a dedicated `rubric_schedules` table (id, rubric_code, key, config, channel override, tz). Reference this table from `rubric_schedule_state` and job payloads for true foreign-key stability.
- **Execution-time idempotency lock.** Before publishing, insert into a new `rubric_job_locks` table keyed by `(rubric_code, schedule_key, scheduled_at_date)`; if the row exists, skip publishing/log duplication. This is the "safety lock" requested.
- **Enhanced admin tooling.** Extend `/purge_*` to accept a dry-run diff between canonical plan and queue contents; expose `rubric_schedule_state` mismatches for visibility.

_Risk / impact_: requires migrations, coordinated deployment (DB schema upgrade), and thorough rollback plan. Adds storage overhead but provides operational guarantees.

_Testing_: integration tests simulating schedule churn, restart reconciliation, and concurrent worker execution; load-test safety lock under intentionally duplicated jobs.

## 7. Verification checklist (staging & production)
1. Capture `jobs_queue` snapshot before and after schedule edits/deletions; confirm no orphaned entries remain.
2. Exercise schedule wizard: create, edit time, swap channels, delete. Ensure queue mirrors config within one scheduler tick.
3. Run `/purge_sea_jobs` in preview mode to confirm report matches actual queue state.
4. Trigger duplicate job injection (manually enqueue same payload) and verify safety lock (when implemented) suppresses the second execution.
5. Observe scheduler logs for `Deleted future rubric jobs` entries whenever config changes.
6. Monitor `rubric_schedule_state` for stale keys after deletions; entries should vanish alongside queue rows.
7. Smoke-test manual rubric runs (`enqueue_rubric`, manual publish) to ensure new locking logic does not suppress legitimate runs.

## 8. References
- `jobs.py` – JobQueue implementation and dedupe logic.
- `main.py` – `process_rubric_schedule`, `_rubric_job_exists`, admin commands.
- `data_access.py` – Rubric config CRUD, job queue helpers.
- `docs/ops/rubrics_pending_jobs.json` – Audit snapshot of queued rubric jobs.
