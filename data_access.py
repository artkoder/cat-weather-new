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
    rubric_id: int | None
    vision_category: str | None
    vision_arch_view: str | None
    vision_photo_weather: str | None
    vision_flower_varieties: list[str] | None
    vision_confidence: float | None


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
        rubric_id: int | None = None,
    ) -> int:
        """Insert or update asset metadata."""

        now = datetime.utcnow().isoformat()
        cats_json = self._serialize_categories(hashtags, categories)
        metadata_json = json.dumps(metadata) if metadata is not None else None
        cur = self.conn.execute(
            """
            INSERT INTO assets (
                channel_id,
                message_id,
                caption_template,
                hashtags,
                categories,
                metadata,
                rubric_id,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(message_id) DO UPDATE SET
                channel_id=excluded.channel_id,
                caption_template=excluded.caption_template,
                hashtags=excluded.hashtags,
                categories=excluded.categories,
                metadata=COALESCE(excluded.metadata, assets.metadata),
                rubric_id=COALESCE(excluded.rubric_id, assets.rubric_id),
                updated_at=excluded.updated_at
            """,
            (
                channel_id,
                message_id,
                template,
                hashtags,
                cats_json,
                metadata_json,
                rubric_id,
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
        rubric_id: int | None = None,
        vision_category: str | None = None,
        vision_arch_view: str | None = None,
        vision_photo_weather: str | None = None,
        vision_flower_varieties: list[str] | None = None,
        vision_confidence: float | None = None,
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
        if rubric_id is not None:
            values["rubric_id"] = rubric_id
        if latitude is not None:
            values["latitude"] = latitude
        if longitude is not None:
            values["longitude"] = longitude
        if city is not None:
            values["city"] = city
        if country is not None:
            values["country"] = country
        if vision_category is not None:
            values["vision_category"] = vision_category
        if vision_arch_view is not None:
            values["vision_arch_view"] = vision_arch_view
        if vision_photo_weather is not None:
            values["vision_photo_weather"] = vision_photo_weather
        if vision_flower_varieties is not None:
            values["vision_flower_varieties"] = json.dumps(vision_flower_varieties)
        if vision_confidence is not None:
            values["vision_confidence"] = vision_confidence
        if not values:
            return
        assignments = ", ".join(f"{k} = ?" for k in values)
        params: list[Any] = list(values.values())
        params.append(asset_id)
        sql = f"UPDATE assets SET {assignments}, updated_at=? WHERE id=?"
        params.insert(-1, datetime.utcnow().isoformat())
        self.conn.execute(sql, params)
        if vision_results is not None:
            self._store_vision_result(row.id, vision_results)
        self.conn.commit()

    def get_asset(self, asset_id: int) -> Asset | None:
        return self._fetch_asset("a.id = ?", (asset_id,))

    def get_asset_by_message(self, message_id: int) -> Asset | None:
        return self._fetch_asset("a.message_id = ?", (message_id,))

    def _fetch_asset(self, where_clause: str, params: Iterable[Any]) -> Asset | None:
        query = f"""
            SELECT a.*, vr.result_json AS vision_payload
            FROM assets a
            LEFT JOIN vision_results vr
              ON vr.asset_id = a.id
             AND vr.id = (
                 SELECT id FROM vision_results
                 WHERE asset_id = a.id
                 ORDER BY created_at DESC, id DESC
                 LIMIT 1
             )
            WHERE {where_clause}
            LIMIT 1
        """
        row = self.conn.execute(query, tuple(params)).fetchone()
        return self._asset_from_row(row)

    def _asset_from_row(self, row: sqlite3.Row | None) -> Asset | None:
        if not row:
            return None
        categories = json.loads(row["categories"] or "[]")
        metadata = json.loads(row["metadata"] or "null")
        raw_vision: str | None
        if "vision_payload" in row.keys():
            raw_vision = row["vision_payload"]
        else:
            raw_vision = self._load_latest_vision_json(int(row["id"]))
        vision = json.loads(raw_vision) if raw_vision else None
        vision_category = row["vision_category"] if "vision_category" in row.keys() else None
        vision_arch_view = row["vision_arch_view"] if "vision_arch_view" in row.keys() else None
        vision_photo_weather = (
            row["vision_photo_weather"] if "vision_photo_weather" in row.keys() else None
        )
        raw_flowers = (
            row["vision_flower_varieties"] if "vision_flower_varieties" in row.keys() else None
        )
        flower_varieties = None
        if raw_flowers:
            try:
                flower_varieties = json.loads(raw_flowers)
            except json.JSONDecodeError:
                flower_varieties = [raw_flowers]
        vision_confidence = (
            row["vision_confidence"] if "vision_confidence" in row.keys() else None
        )
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
            rubric_id=row["rubric_id"] if "rubric_id" in row.keys() else None,
            vision_category=vision_category,
            vision_arch_view=vision_arch_view,
            vision_photo_weather=vision_photo_weather,
            vision_flower_varieties=flower_varieties,
            vision_confidence=vision_confidence,
        )

    def get_next_asset(self, tags: set[str] | None) -> Asset | None:
        now_iso = datetime.utcnow().isoformat()
        rows = self.conn.execute(
            """
            SELECT a.*, vr.result_json AS vision_payload
            FROM assets a
            LEFT JOIN vision_results vr
              ON vr.asset_id = a.id
             AND vr.id = (
                 SELECT id FROM vision_results
                 WHERE asset_id = a.id
                 ORDER BY created_at DESC, id DESC
                 LIMIT 1
             )
            ORDER BY COALESCE(a.last_used_at, a.created_at) ASC, a.id ASC
            """
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

    def _store_vision_result(self, asset_id: int, result: Any) -> None:
        payload = json.dumps(result) if not isinstance(result, str) else result
        provider = result.get("provider") if isinstance(result, dict) else None
        status = result.get("status") if isinstance(result, dict) else None
        category = result.get("category") if isinstance(result, dict) else None
        arch_view = result.get("arch_view") if isinstance(result, dict) else None
        photo_weather = result.get("photo_weather") if isinstance(result, dict) else None
        flowers_raw: Any | None = result.get("flower_varieties") if isinstance(result, dict) else None
        if isinstance(flowers_raw, list):
            flowers_json = json.dumps(flowers_raw)
        elif flowers_raw is None:
            flowers_json = None
        else:
            flowers_json = json.dumps([flowers_raw])
        confidence = result.get("confidence") if isinstance(result, dict) else None
        now = datetime.utcnow().isoformat()
        self.conn.execute(
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
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asset_id,
                provider,
                status,
                category,
                arch_view,
                photo_weather,
                flowers_json,
                confidence,
                payload,
                now,
                now,
            ),
        )

    def _load_latest_vision_json(self, asset_id: int) -> str | None:
        row = self.conn.execute(
            """
            SELECT result_json FROM vision_results
            WHERE asset_id=?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (asset_id,),
        ).fetchone()
        if not row:
            return None
        return row["result_json"]

    def record_post_history(
        self,
        channel_id: int,
        message_id: int,
        asset_id: int | None,
        rubric_id: int | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        now = datetime.utcnow().isoformat()
        resolved_rubric = rubric_id
        if resolved_rubric is None and asset_id is not None:
            asset = self.get_asset(asset_id)
            if asset:
                resolved_rubric = asset.rubric_id
        self.conn.execute(
            """
            INSERT INTO posts_history (
                channel_id, message_id, asset_id, rubric_id, metadata, published_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                channel_id,
                message_id,
                asset_id,
                resolved_rubric,
                json.dumps(metadata) if metadata is not None else None,
                now,
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
            INSERT INTO token_usage (
                model, prompt_tokens, completion_tokens, total_tokens,
                job_name, job_id, asset_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (model, prompt_tokens, completion_tokens, total_tokens, job_name, job_id, asset_id, now),
        )
        self.conn.commit()
