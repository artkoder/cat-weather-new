from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Dict

import sqlite3

JobHandler = Callable[["Job"], Awaitable[None]]


class JobDelayed(Exception):
    """Signal that a job should be retried later without counting as failure."""

    def __init__(self, available_at: datetime, reason: str):
        super().__init__(reason)
        self.available_at = available_at
        self.reason = reason


@dataclass
class Job:
    id: int
    name: str
    payload: dict[str, Any]
    status: str
    attempts: int
    available_at: datetime | None
    last_error: str | None
    created_at: datetime
    updated_at: datetime


class JobQueue:
    """Simple persistent job queue backed by SQLite."""

    STATUSES = {"queued", "running", "done", "failed", "delayed"}

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        concurrency: int = 1,
        poll_interval: float = 1.0,
        max_attempts: int = 5,
    ) -> None:
        self.conn = connection
        self.conn.row_factory = sqlite3.Row
        self.concurrency = max(1, concurrency)
        self.poll_interval = poll_interval
        self.max_attempts = max_attempts
        self.handlers: Dict[str, JobHandler] = {}
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._loader: asyncio.Task | None = None
        self._running = False
        self._inflight: set[int] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def register_handler(self, name: str, handler: JobHandler) -> None:
        self.handlers[name] = handler

    def metrics(self) -> dict[str, int]:
        """Return lightweight queue counters for health checks."""
        rows = self.conn.execute(
            """
            SELECT status, COUNT(*) as count
            FROM jobs_queue
            WHERE status IN ('queued', 'delayed', 'running', 'failed')
            GROUP BY status
            """
        ).fetchall()
        counts = {str(row["status"]): int(row["count"]) for row in rows}
        pending = counts.get("queued", 0) + counts.get("delayed", 0)
        active = counts.get("running", 0)
        failed = counts.get("failed", 0)
        return {"pending": pending, "active": active, "failed": failed}

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop = asyncio.get_running_loop()
        await self._load_due_jobs()
        self._loader = asyncio.create_task(self._loader_loop())
        for _ in range(self.concurrency):
            self._workers.append(asyncio.create_task(self._worker_loop()))

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._loader:
            self._loader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loader
        for _ in self._workers:
            await self._queue.put(None)  # type: ignore[arg-type]
        for task in self._workers:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._workers.clear()
        self._loader = None
        self._inflight.clear()

    async def _loader_loop(self) -> None:
        try:
            while self._running:
                await self._load_due_jobs()
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            pass

    async def _worker_loop(self) -> None:
        try:
            while self._running:
                job = await self._queue.get()
                if job is None:
                    break
                await self._execute_job(job)
                self._queue.task_done()
        except asyncio.CancelledError:
            pass

    async def _execute_job(self, job: Job) -> None:
        handler = self.handlers.get(job.name)
        if not handler:
            logging.error("No handler registered for job %s", job.name)
            await self._mark_failed(job, "handler missing")
            self._inflight.discard(job.id)
            return
        start = datetime.utcnow().isoformat()
        self.conn.execute(
            "UPDATE jobs_queue SET status=?, updated_at=? WHERE id=?",
            ("running", start, job.id),
        )
        self.conn.commit()
        start_time = time.perf_counter()
        attempt_number = job.attempts + 1
        logging.info(
            "Starting job %s (%s) attempt %s", job.id, job.name, attempt_number
        )
        try:
            await handler(job)
        except JobDelayed as delayed:
            now = datetime.utcnow().isoformat()
            self.conn.execute(
                """
                UPDATE jobs_queue
                SET status=?, available_at=?, last_error=?, updated_at=?
                WHERE id=?
                """,
                (
                    "delayed",
                    delayed.available_at.isoformat(),
                    delayed.reason,
                    now,
                    job.id,
                ),
            )
            self.conn.commit()
        except Exception as exc:  # pragma: no cover - defensive
            logging.exception("Job %s failed", job.id)
            await self._handle_failure(job, str(exc))
        else:
            self.conn.execute(
                "UPDATE jobs_queue SET status=?, last_error=NULL, updated_at=? WHERE id=?",
                ("done", datetime.utcnow().isoformat(), job.id),
            )
            self.conn.commit()
            duration = time.perf_counter() - start_time
            logging.info(
                "Completed job %s (%s) in %.3f seconds", job.id, job.name, duration
            )
        finally:
            self._inflight.discard(job.id)

    async def _handle_failure(self, job: Job, error: str) -> None:
        attempts = job.attempts + 1
        if attempts >= self.max_attempts:
            await self._mark_failed(job, error, attempts)
            return
        delay_seconds = min(3600, 2 ** attempts)
        available_at = datetime.utcnow() + timedelta(seconds=delay_seconds)
        self.conn.execute(
            """
            UPDATE jobs_queue
            SET status=?, attempts=?, available_at=?, last_error=?, updated_at=?
            WHERE id=?
            """,
            (
                "delayed",
                attempts,
                available_at.isoformat(),
                error,
                datetime.utcnow().isoformat(),
                job.id,
            ),
        )
        self.conn.commit()

    async def _mark_failed(self, job: Job, error: str, attempts: int | None = None) -> None:
        self.conn.execute(
            "UPDATE jobs_queue SET status=?, attempts=?, last_error=?, updated_at=? WHERE id=?",
            (
                "failed",
                attempts if attempts is not None else job.attempts + 1,
                error,
                datetime.utcnow().isoformat(),
                job.id,
            ),
        )
        self.conn.commit()

    async def _load_due_jobs(self) -> None:
        now = datetime.utcnow().isoformat()
        rows = self.conn.execute(
            """
            SELECT * FROM jobs_queue
            WHERE status IN ('queued', 'delayed')
              AND (available_at IS NULL OR available_at <= ?)
            ORDER BY created_at, id
            """,
            (now,),
        ).fetchall()
        for row in rows:
            job_id = int(row["id"])
            if job_id in self._inflight:
                continue
            status = row["status"]
            if status == "delayed":
                self.conn.execute(
                    "UPDATE jobs_queue SET status=?, updated_at=? WHERE id=?",
                    ("queued", datetime.utcnow().isoformat(), job_id),
                )
                self.conn.commit()
            job = self._row_to_job(row)
            self._inflight.add(job_id)
            await self._queue.put(job)
            logging.debug(
                "Queued job %s (%s) for worker", job.id, job.name
            )

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        payload = row["payload"]
        data = json.loads(payload) if payload else {}
        available = (
            datetime.fromisoformat(row["available_at"])
            if row["available_at"]
            else None
        )
        return Job(
            id=row["id"],
            name=row["name"],
            payload=data,
            status=row["status"],
            attempts=row["attempts"],
            available_at=available,
            last_error=row["last_error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def enqueue(
        self,
        name: str,
        payload: dict[str, Any] | None = None,
        *,
        run_at: datetime | None = None,
        dedupe: bool = False,
    ) -> int:
        if name not in self.handlers:
            logging.warning("Enqueuing job %s without registered handler", name)
        now = datetime.utcnow()
        available_at = run_at if run_at else now
        status = "queued" if available_at <= now else "delayed"
        payload_data = dict(payload) if payload else {}
        payload_json = json.dumps(payload_data, sort_keys=True)
        if dedupe:
            statuses = ("queued", "running", "delayed")
            placeholders = ", ".join("?" for _ in statuses)
            rows = self.conn.execute(
                f"""
                SELECT id, payload
                FROM jobs_queue
                WHERE name = ?
                  AND status IN ({placeholders})
                """,
                (name, *statuses),
            ).fetchall()
            for row in rows:
                try:
                    existing_payload = json.loads(row["payload"]) if row["payload"] else {}
                except json.JSONDecodeError:
                    continue
                if existing_payload == payload_data:
                    job_id = int(row["id"])
                    logging.debug(
                        "Skipping duplicate job %s for payload %s (existing id=%s)",
                        name,
                        payload_data,
                        job_id,
                    )
                    return job_id
        cur = self.conn.execute(
            """
            INSERT INTO jobs_queue (name, payload, status, attempts, available_at, created_at, updated_at)
            VALUES (?, ?, ?, 0, ?, ?, ?)
            """,
            (
                name,
                payload_json,
                status,
                available_at.isoformat() if status == "delayed" else None,
                now.isoformat(),
                now.isoformat(),
            ),
        )
        self.conn.commit()
        job_id = int(cur.lastrowid)
        logging.info(
            "Enqueued job %s (%s) with status %s", job_id, name, status
        )
        if status == "queued" and self._running:
            row = self.conn.execute(
                "SELECT * FROM jobs_queue WHERE id=?", (job_id,)
            ).fetchone()
            if row:
                job = self._row_to_job(row)
                self._inflight.add(job_id)
                loop = self._loop or asyncio.get_running_loop()
                loop.call_soon(self._queue.put_nowait, job)
                logging.debug(
                    "Dispatched job %s (%s) to worker queue", job.id, job.name
                )
        return job_id
