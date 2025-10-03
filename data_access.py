from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import sqlite3


@dataclass
class Asset:
    id: int
    channel_id: int
    message_id: int
    caption_template: str | None
    hashtags: str | None
    categories: list[str]
    recognized_message_id: int | None
    metadata: dict[str, Any] | None
    vision_results: dict[str, Any] | None
    latitude: float | None
    longitude: float | None
    city: str | None
    country: str | None


@dataclass
class WeatherJob:
    id: int
    channel_id: int
    post_time: str
    run_at: datetime
    last_run_at: datetime | None
    failures: int
    last_error: str | None


class DataAccess:
    """High level helpers for working with the SQLite database."""

    def __init__(self, connection: sqlite3.Connection):
        self.conn = connection
        self.conn.row_factory = sqlite3.Row
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id INTEGER NOT NULL,
                message_id INTEGER NOT NULL UNIQUE,
                caption_template TEXT,
                hashtags TEXT,
                categories TEXT,
                recognized_message_id INTEGER,
                metadata TEXT,
                vision_results TEXT,
                latitude REAL,
                longitude REAL,
                city TEXT,
                country TEXT,
                last_used_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS asset_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id INTEGER NOT NULL,
                message_id INTEGER NOT NULL,
                asset_id INTEGER,
                schedule_id INTEGER,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS weather_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id INTEGER NOT NULL UNIQUE,
                post_time TEXT NOT NULL,
                run_at TEXT NOT NULL,
                last_run_at TEXT,
                failures INTEGER DEFAULT 0,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                payload TEXT,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                available_at TEXT,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                job_name TEXT,
                job_id INTEGER,
                asset_id INTEGER,
                created_at TEXT NOT NULL
            )
            """
        )
        # ensure indexes for frequent lookups
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_assets_message ON assets(message_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_queue_status ON jobs_queue(status, available_at)"
        )
        self.conn.commit()

    @staticmethod
    def _serialize_categories(hashtags: str | None, categories: Iterable[str] | None) -> str:
        if categories is not None:
            cats = list(dict.fromkeys(categories))
        elif hashtags:
            cats = [tag.strip() for tag in hashtags.split() if tag.strip()]
        else:
            cats = []
        return json.dumps(cats)

    def save_asset(
        self,
        channel_id: int,
        message_id: int,
        template: str | None,
        hashtags: str | None,
        *,
        metadata: dict[str, Any] | None = None,
        categories: Iterable[str] | None = None,
    ) -> int:
        """Insert or update asset metadata."""

        now = datetime.utcnow().isoformat()
        cats_json = self._serialize_categories(hashtags, categories)
        metadata_json = json.dumps(metadata) if metadata is not None else None
        cur = self.conn.execute(
            """
            INSERT INTO assets (
                channel_id, message_id, caption_template, hashtags, categories,
                metadata, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(message_id) DO UPDATE SET
                channel_id=excluded.channel_id,
                caption_template=excluded.caption_template,
                hashtags=excluded.hashtags,
                categories=excluded.categories,
                metadata=COALESCE(excluded.metadata, assets.metadata),
                updated_at=excluded.updated_at
            """,
            (
                channel_id,
                message_id,
                template,
                hashtags,
                cats_json,
                metadata_json,
                now,
                now,
            ),
        )
        self.conn.commit()
        if cur.lastrowid:
            return int(cur.lastrowid)
        row = self.get_asset_by_message(message_id)
        return row.id if row else 0

    def update_asset(
        self,
        asset_id: int,
        *,
        template: str | None = None,
        hashtags: str | None = None,
        metadata: dict[str, Any] | None = None,
        recognized_message_id: int | None = None,
        vision_results: dict[str, Any] | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        city: str | None = None,
        country: str | None = None,
    ) -> None:
        """Update selected asset fields while preserving unset values."""

        row = self.get_asset(asset_id)
        if not row:
            logging.warning("Attempted to update missing asset %s", asset_id)
            return
        values: dict[str, Any] = {}
        if template is not None:
            values["caption_template"] = template
        if hashtags is not None:
            values["hashtags"] = hashtags
            values["categories"] = self._serialize_categories(hashtags, None)
        if metadata is not None:
            existing = row.metadata or {}
            merged = existing.copy()
            merged.update(metadata)
            values["metadata"] = json.dumps(merged)
        if recognized_message_id is not None:
            values["recognized_message_id"] = recognized_message_id
        if vision_results is not None:
            values["vision_results"] = json.dumps(vision_results)
        if latitude is not None:
            values["latitude"] = latitude
        if longitude is not None:
            values["longitude"] = longitude
        if city is not None:
            values["city"] = city
        if country is not None:
            values["country"] = country
        if not values:
            return
        assignments = ", ".join(f"{k} = ?" for k in values)
        params: list[Any] = list(values.values())
        params.append(asset_id)
        sql = f"UPDATE assets SET {assignments}, updated_at=? WHERE id=?"
        params.insert(-1, datetime.utcnow().isoformat())
        self.conn.execute(sql, params)
        self.conn.commit()

    def get_asset(self, asset_id: int) -> Asset | None:
        row = self.conn.execute(
            "SELECT * FROM assets WHERE id=?", (asset_id,)
        ).fetchone()
        return self._asset_from_row(row) if row else None

    def get_asset_by_message(self, message_id: int) -> Asset | None:
        row = self.conn.execute(
            "SELECT * FROM assets WHERE message_id=?", (message_id,)
        ).fetchone()
        return self._asset_from_row(row) if row else None

    def _asset_from_row(self, row: sqlite3.Row | None) -> Asset | None:
        if not row:
            return None
        categories = json.loads(row["categories"] or "[]")
        metadata = json.loads(row["metadata"] or "null")
        vision = json.loads(row["vision_results"] or "null")
        return Asset(
            id=row["id"],
            channel_id=row["channel_id"],
            message_id=row["message_id"],
            caption_template=row["caption_template"],
            hashtags=row["hashtags"],
            categories=categories,
            recognized_message_id=row["recognized_message_id"],
            metadata=metadata,
            vision_results=vision,
            latitude=row["latitude"],
            longitude=row["longitude"],
            city=row["city"],
            country=row["country"],
        )

    def get_next_asset(self, tags: set[str] | None) -> Asset | None:
        now_iso = datetime.utcnow().isoformat()
        rows = self.conn.execute(
            "SELECT * FROM assets ORDER BY COALESCE(last_used_at, created_at) ASC"
        ).fetchall()
        normalized = {t.lower() for t in tags} if tags else None
        for row in rows:
            asset = self._asset_from_row(row)
            if not asset:
                continue
            if normalized:
                asset_tags = {t.lower() for t in asset.categories}
                if not asset_tags.intersection(normalized):
                    continue
            self.conn.execute(
                "UPDATE assets SET last_used_at=?, updated_at=? WHERE id=?",
                (now_iso, now_iso, asset.id),
            )
            self.conn.commit()
            return asset
        return None

    def record_post_history(
        self,
        channel_id: int,
        message_id: int,
        asset_id: int | None,
        schedule_id: int | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """
            INSERT INTO asset_history (
                channel_id, message_id, asset_id, schedule_id, metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                channel_id,
                message_id,
                asset_id,
                schedule_id,
                json.dumps(metadata) if metadata is not None else None,
                now,
            ),
        )
        self.conn.commit()

    def upsert_weather_job(self, channel_id: int, post_time: str, next_run: datetime) -> None:
        now = datetime.utcnow().isoformat()
        run_iso = next_run.isoformat()
        self.conn.execute(
            """
            INSERT INTO weather_jobs (channel_id, post_time, run_at, last_run_at, failures, last_error, created_at, updated_at)
            VALUES (?, ?, ?, NULL, 0, NULL, ?, ?)
            ON CONFLICT(channel_id) DO UPDATE SET
                post_time=excluded.post_time,
                run_at=excluded.run_at,
                updated_at=excluded.updated_at
            """,
            (channel_id, post_time, run_iso, now, now),
        )
        self.conn.commit()

    def remove_weather_job(self, channel_id: int) -> None:
        self.conn.execute("DELETE FROM weather_jobs WHERE channel_id=?", (channel_id,))
        self.conn.commit()

    def list_weather_jobs(self) -> list[WeatherJob]:
        rows = self.conn.execute(
            "SELECT * FROM weather_jobs ORDER BY channel_id"
        ).fetchall()
        return [self._weather_job_from_row(r) for r in rows]

    def _weather_job_from_row(self, row: sqlite3.Row) -> WeatherJob:
        return WeatherJob(
            id=row["id"],
            channel_id=row["channel_id"],
            post_time=row["post_time"],
            run_at=datetime.fromisoformat(row["run_at"]),
            last_run_at=datetime.fromisoformat(row["last_run_at"])
            if row["last_run_at"]
            else None,
            failures=row["failures"] or 0,
            last_error=row["last_error"],
        )

    def due_weather_jobs(self, now: datetime) -> list[WeatherJob]:
        rows = self.conn.execute(
            "SELECT * FROM weather_jobs WHERE run_at <= ? ORDER BY run_at",
            (now.isoformat(),),
        ).fetchall()
        return [self._weather_job_from_row(r) for r in rows]

    def mark_weather_job_run(self, job_id: int, next_run: datetime) -> None:
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """
            UPDATE weather_jobs
            SET run_at=?, last_run_at=?, failures=0, last_error=NULL, updated_at=?
            WHERE id=?
            """,
            (next_run.isoformat(), now, now, job_id),
        )
        self.conn.commit()

    def record_weather_job_failure(self, job_id: int, reason: str) -> None:
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """
            UPDATE weather_jobs
            SET failures=COALESCE(failures, 0) + 1, last_error=?, updated_at=?
            WHERE id=?
            """,
            (reason, now, job_id),
        )
        self.conn.commit()

    def log_ai_usage(
        self,
        model: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        *,
        job_name: str | None = None,
        job_id: int | None = None,
        asset_id: int | None = None,
    ) -> None:
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """
            INSERT INTO ai_usage (
                model, prompt_tokens, completion_tokens, total_tokens,
                job_name, job_id, asset_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (model, prompt_tokens, completion_tokens, total_tokens, job_name, job_id, asset_id, now),
        )
        self.conn.commit()
