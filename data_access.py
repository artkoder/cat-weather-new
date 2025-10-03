from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, date
from typing import Any, Iterable, Iterator, Sequence

import sqlite3


@dataclass
class Asset:
    id: int
    channel_id: int
    tg_chat_id: int
    message_id: int
    caption_template: str | None
    caption: str | None
    hashtags: str | None
    categories: list[str]
    kind: str | None
    file_id: str | None
    file_unique_id: str | None
    file_name: str | None
    mime_type: str | None
    file_size: int | None
    width: int | None
    height: int | None
    duration: int | None
    recognized_message_id: int | None
    exif_present: bool
    latitude: float | None
    longitude: float | None
    city: str | None
    country: str | None
    author_user_id: int | None
    author_username: str | None
    sender_chat_id: int | None
    via_bot_id: int | None
    forward_from_user: int | None
    forward_from_chat: int | None
    local_path: str | None
    metadata: dict[str, Any] | None
    vision_results: dict[str, Any] | None
    rubric_id: int | None
    vision_category: str | None
    vision_arch_view: str | None
    vision_photo_weather: str | None
    vision_flower_varieties: list[str] | None
    vision_confidence: float | None
    vision_caption: str | None


@dataclass
class WeatherJob:
    id: int
    channel_id: int
    post_time: str
    run_at: datetime
    last_run_at: datetime | None
    failures: int
    last_error: str | None


@dataclass
class Rubric:
    id: int
    code: str
    title: str
    description: str | None
    config: dict[str, Any]


@dataclass
class RubricScheduleState:
    rubric_code: str
    schedule_key: str
    next_run_at: datetime | None
    last_run_at: datetime | None


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

    @staticmethod
    def _prepare_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
        if not metadata:
            return None
        removable = {
            "chat_id",
            "message_id",
            "caption",
            "kind",
            "author_user_id",
            "author_username",
            "sender_chat_id",
            "via_bot_id",
            "forward_from_user",
            "forward_from_chat",
            "file",
            "local_path",
            "exif_present",
            "vision_caption",
        }
        cleaned = {k: v for k, v in metadata.items() if k not in removable}
        return cleaned or None

    def save_asset(
        self,
        channel_id: int,
        message_id: int,
        template: str | None,
        hashtags: str | None,
        *,
        tg_chat_id: int,
        caption: str | None,
        kind: str | None,
        file_meta: dict[str, Any] | None = None,
        author_user_id: int | None = None,
        author_username: str | None = None,
        sender_chat_id: int | None = None,
        via_bot_id: int | None = None,
        forward_from_user: int | None = None,
        forward_from_chat: int | None = None,
        metadata: dict[str, Any] | None = None,
        categories: Iterable[str] | None = None,
        rubric_id: int | None = None,
    ) -> int:
        """Insert or update asset metadata."""

        now = datetime.utcnow().isoformat()
        cats_json = self._serialize_categories(hashtags, categories)
        cleaned_metadata = self._prepare_metadata(metadata)
        metadata_json = json.dumps(cleaned_metadata) if cleaned_metadata else None
        file_meta = file_meta or {}
        cur = self.conn.execute(
            """
            INSERT INTO assets (
                channel_id,
                tg_chat_id,
                message_id,
                caption_template,
                caption,
                hashtags,
                categories,
                kind,
                file_id,
                file_unique_id,
                file_name,
                mime_type,
                file_size,
                width,
                height,
                duration,
                author_user_id,
                author_username,
                sender_chat_id,
                via_bot_id,
                forward_from_user,
                forward_from_chat,
                metadata,
                rubric_id,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tg_chat_id, message_id) DO UPDATE SET
                channel_id=excluded.channel_id,
                caption_template=excluded.caption_template,
                caption=excluded.caption,
                hashtags=excluded.hashtags,
                categories=excluded.categories,
                kind=excluded.kind,
                file_id=excluded.file_id,
                file_unique_id=excluded.file_unique_id,
                file_name=excluded.file_name,
                mime_type=excluded.mime_type,
                file_size=excluded.file_size,
                width=excluded.width,
                height=excluded.height,
                duration=excluded.duration,
                author_user_id=excluded.author_user_id,
                author_username=excluded.author_username,
                sender_chat_id=excluded.sender_chat_id,
                via_bot_id=excluded.via_bot_id,
                forward_from_user=excluded.forward_from_user,
                forward_from_chat=excluded.forward_from_chat,
                metadata=COALESCE(excluded.metadata, assets.metadata),
                rubric_id=COALESCE(excluded.rubric_id, assets.rubric_id),
                updated_at=excluded.updated_at
            """,
            (
                channel_id,
                tg_chat_id,
                message_id,
                template,
                caption,
                hashtags,
                cats_json,
                kind,
                file_meta.get("file_id"),
                file_meta.get("file_unique_id"),
                file_meta.get("file_name"),
                file_meta.get("mime_type"),
                file_meta.get("file_size"),
                file_meta.get("width"),
                file_meta.get("height"),
                file_meta.get("duration"),
                author_user_id,
                author_username,
                sender_chat_id,
                via_bot_id,
                forward_from_user,
                forward_from_chat,
                metadata_json,
                rubric_id,
                now,
                now,
            ),
        )
        self.conn.commit()
        if cur.lastrowid:
            return int(cur.lastrowid)
        row = self.get_asset_by_message(tg_chat_id, message_id)
        return row.id if row else 0

    def update_asset(
        self,
        asset_id: int,
        *,
        template: str | None = None,
        caption: str | None = None,
        hashtags: str | None = None,
        kind: str | None = None,
        file_meta: dict[str, Any] | None = None,
        author_user_id: int | None = None,
        author_username: str | None = None,
        sender_chat_id: int | None = None,
        via_bot_id: int | None = None,
        forward_from_user: int | None = None,
        forward_from_chat: int | None = None,
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
        exif_present: bool | None = None,
        local_path: str | None = None,
        vision_caption: str | None = None,
    ) -> None:
        """Update selected asset fields while preserving unset values."""

        row = self.get_asset(asset_id)
        if not row:
            logging.warning("Attempted to update missing asset %s", asset_id)
            return
        values: dict[str, Any] = {}
        if template is not None:
            values["caption_template"] = template
        if caption is not None:
            values["caption"] = caption
        if hashtags is not None:
            values["hashtags"] = hashtags
            values["categories"] = self._serialize_categories(hashtags, None)
        if kind is not None:
            values["kind"] = kind
        if file_meta is not None:
            values["file_id"] = file_meta.get("file_id")
            values["file_unique_id"] = file_meta.get("file_unique_id")
            values["file_name"] = file_meta.get("file_name")
            values["mime_type"] = file_meta.get("mime_type")
            values["file_size"] = file_meta.get("file_size")
            values["width"] = file_meta.get("width")
            values["height"] = file_meta.get("height")
            values["duration"] = file_meta.get("duration")
        if author_user_id is not None:
            values["author_user_id"] = author_user_id
        if author_username is not None:
            values["author_username"] = author_username
        if sender_chat_id is not None:
            values["sender_chat_id"] = sender_chat_id
        if via_bot_id is not None:
            values["via_bot_id"] = via_bot_id
        if forward_from_user is not None:
            values["forward_from_user"] = forward_from_user
        if forward_from_chat is not None:
            values["forward_from_chat"] = forward_from_chat
        if metadata is not None:
            existing = row.metadata or {}
            merged = existing.copy()
            merged.update(metadata)
            cleaned = self._prepare_metadata(merged)
            if cleaned is not None:
                values["metadata"] = json.dumps(cleaned)
            elif row.metadata is not None:
                values["metadata"] = None
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
        if exif_present is not None:
            values["exif_present"] = int(bool(exif_present))
        if local_path is not None:
            values["local_path"] = local_path
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
        if vision_caption is not None:
            values["vision_caption"] = vision_caption
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

    def get_asset_by_message(self, tg_chat_id: int, message_id: int) -> Asset | None:
        return self._fetch_asset(
            "a.tg_chat_id = ? AND a.message_id = ?",
            (tg_chat_id, message_id),
        )

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
        raw_exif = row["exif_present"] if "exif_present" in row.keys() else 0
        try:
            exif_value = int(raw_exif)
        except (TypeError, ValueError):
            exif_value = 1 if str(raw_exif).lower() in {"true", "1"} else 0
        return Asset(
            id=row["id"],
            channel_id=row["channel_id"],
            tg_chat_id=row["tg_chat_id"],
            message_id=row["message_id"],
            caption_template=row["caption_template"],
            caption=row["caption"],
            hashtags=row["hashtags"],
            categories=categories,
            kind=row["kind"],
            file_id=row["file_id"],
            file_unique_id=row["file_unique_id"],
            file_name=row["file_name"],
            mime_type=row["mime_type"],
            file_size=row["file_size"],
            width=row["width"],
            height=row["height"],
            duration=row["duration"],
            recognized_message_id=row["recognized_message_id"],
            exif_present=bool(exif_value),
            latitude=row["latitude"],
            longitude=row["longitude"],
            city=row["city"],
            country=row["country"],
            author_user_id=row["author_user_id"],
            author_username=row["author_username"],
            sender_chat_id=row["sender_chat_id"],
            via_bot_id=row["via_bot_id"],
            forward_from_user=row["forward_from_user"],
            forward_from_chat=row["forward_from_chat"],
            local_path=row["local_path"],
            metadata=metadata,
            vision_results=vision,
            rubric_id=row["rubric_id"] if "rubric_id" in row.keys() else None,
            vision_category=vision_category,
            vision_arch_view=vision_arch_view,
            vision_photo_weather=vision_photo_weather,
            vision_flower_varieties=flower_varieties,
            vision_confidence=vision_confidence,
            vision_caption=row["vision_caption"] if "vision_caption" in row.keys() else None,
        )

    def iter_assets(self, *, rubric_id: int | None = None) -> Iterator[Asset]:
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
        for row in rows:
            asset = self._asset_from_row(row)
            if not asset:
                continue
            if rubric_id is not None and asset.rubric_id != rubric_id:
                continue
            yield asset

    def fetch_assets_by_categories(
        self,
        categories: Sequence[str],
        *,
        rubric_id: int | None = None,
        limit: int | None = None,
    ) -> list[Asset]:
        normalized = {c.lower().lstrip("#") for c in categories if c}
        if not normalized:
            return []
        results: list[Asset] = []
        for asset in self.iter_assets(rubric_id=rubric_id):
            asset_tags = {t.lower().lstrip("#") for t in asset.categories}
            if not asset_tags.intersection(normalized):
                continue
            results.append(asset)
            if limit is not None and len(results) >= limit:
                break
        return results

    def delete_assets(self, asset_ids: Sequence[int]) -> None:
        if not asset_ids:
            return
        placeholders = ",".join("?" for _ in asset_ids)
        params = tuple(int(a) for a in asset_ids)
        self.conn.execute(
            f"DELETE FROM vision_results WHERE asset_id IN ({placeholders})",
            params,
        )
        self.conn.execute(
            f"DELETE FROM assets WHERE id IN ({placeholders})",
            params,
        )
        self.conn.commit()

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

    def log_token_usage(
        self,
        model: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        *,
        job_id: int | None = None,
        request_id: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        recorded_at = timestamp or datetime.utcnow().isoformat()
        total = total_tokens
        if total is None and (prompt_tokens is not None or completion_tokens is not None):
            total = (prompt_tokens or 0) + (completion_tokens or 0)
        self.conn.execute(
            """
            INSERT INTO token_usage (
                model,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                job_id,
                request_id,
                timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model,
                prompt_tokens,
                completion_tokens,
                total,
                job_id,
                request_id,
                recorded_at,
            ),
        )
        self.conn.commit()

    def get_daily_token_usage_total(
        self,
        *,
        day: date | None = None,
        models: Iterable[str] | None = None,
    ) -> int:
        target_day = (day or datetime.utcnow().date()).isoformat()
        params: list[Any] = [target_day]
        query = (
            "SELECT COALESCE(SUM(COALESCE(total_tokens, 0)), 0) AS total "
            "FROM token_usage WHERE date(timestamp) = date(?)"
        )
        model_list = list(models) if models else []
        if model_list:
            placeholders = ", ".join("?" for _ in model_list)
            query += f" AND model IN ({placeholders})"
            params.extend(model_list)
        row = self.conn.execute(query, params).fetchone()
        value = row["total"] if row else 0
        try:
            return int(value or 0)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return 0

    def _rubric_from_row(self, row: sqlite3.Row) -> Rubric:
        description = row["description"]
        config: dict[str, Any]
        if description:
            try:
                parsed = json.loads(description)
                config = parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                config = {}
        else:
            config = {}
        return Rubric(
            id=row["id"],
            code=row["code"],
            title=row["title"],
            description=row["description"],
            config=config,
        )

    def list_rubrics(self) -> list[Rubric]:
        rows = self.conn.execute(
            "SELECT * FROM rubrics ORDER BY id"
        ).fetchall()
        return [self._rubric_from_row(r) for r in rows]

    def get_rubric_by_code(self, code: str) -> Rubric | None:
        row = self.conn.execute(
            "SELECT * FROM rubrics WHERE code=?",
            (code,),
        ).fetchone()
        if not row:
            return None
        return self._rubric_from_row(row)

    def get_rubric_schedule_state(
        self, rubric_code: str, schedule_key: str
    ) -> RubricScheduleState | None:
        row = self.conn.execute(
            """
            SELECT rubric_code, schedule_key, next_run_at, last_run_at
            FROM rubric_schedule_state
            WHERE rubric_code=? AND schedule_key=?
            LIMIT 1
            """,
            (rubric_code, schedule_key),
        ).fetchone()
        if not row:
            return None
        next_run = (
            datetime.fromisoformat(row["next_run_at"])
            if row["next_run_at"]
            else None
        )
        last_run = (
            datetime.fromisoformat(row["last_run_at"])
            if row["last_run_at"]
            else None
        )
        return RubricScheduleState(
            rubric_code=row["rubric_code"],
            schedule_key=row["schedule_key"],
            next_run_at=next_run,
            last_run_at=last_run,
        )

    def set_rubric_schedule_state(
        self,
        rubric_code: str,
        schedule_key: str,
        *,
        next_run_at: datetime | None,
        last_run_at: datetime | None = None,
    ) -> None:
        now = datetime.utcnow().isoformat()
        next_iso = next_run_at.isoformat() if next_run_at else None
        last_iso = last_run_at.isoformat() if last_run_at else None
        self.conn.execute(
            """
            INSERT INTO rubric_schedule_state (
                rubric_code, schedule_key, next_run_at, last_run_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(rubric_code, schedule_key) DO UPDATE SET
                next_run_at=excluded.next_run_at,
                last_run_at=COALESCE(excluded.last_run_at, rubric_schedule_state.last_run_at),
                updated_at=excluded.updated_at
            """,
            (rubric_code, schedule_key, next_iso, last_iso, now, now),
        )
        self.conn.commit()

    def mark_rubric_run(
        self, rubric_code: str, schedule_key: str, run_at: datetime
    ) -> None:
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """
            INSERT INTO rubric_schedule_state (
                rubric_code, schedule_key, next_run_at, last_run_at, created_at, updated_at
            ) VALUES (?, ?, NULL, ?, ?, ?)
            ON CONFLICT(rubric_code, schedule_key) DO UPDATE SET
                last_run_at=excluded.last_run_at,
                updated_at=excluded.updated_at
            """,
            (
                rubric_code,
                schedule_key,
                run_at.isoformat(),
                now,
                now,
            ),
        )
        self.conn.commit()
