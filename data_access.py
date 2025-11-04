from __future__ import annotations

import importlib.util
import json
import logging
import random
import secrets
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

_UNSET = object()

import sqlite3

PAIRING_TOKEN_TTL_SECONDS = 600
NONCE_TTL_SECONDS = 600
UPLOAD_IDEMPOTENCY_TTL_SECONDS = 24 * 3600


@dataclass
class Asset:
    id: str
    upload_id: str | None
    file_ref: str | None
    content_type: str | None
    sha256: str | None
    width: int | None
    height: int | None
    exif_json: str | None
    labels_json: str | None
    tg_message_id: str | None
    payload_json: str | None
    created_at: str
    source: str | None = None
    exif: dict[str, Any] | None = None
    labels: Any | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    legacy_values: dict[str, Any] = field(default_factory=dict)
    _vision_results: dict[str, Any] | None = None
    _vision_category: str | None = None
    _vision_arch_view: str | None = None
    _vision_photo_weather: str | None = None
    _vision_flower_varieties: list[str] | None = None
    _vision_confidence: float | None = None
    _vision_caption: str | None = None
    _local_path: str | None = None
    _rubric_id: int | None = None


    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_int(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return int(text)
            except ValueError:
                try:
                    return int(float(text))
                except ValueError:
                    return None
        return None

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                return None
        return None

    @staticmethod
    def _to_bool(value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"", "none"}:
                return None
            if text in {"true", "1", "yes", "y"}:
                return True
            if text in {"false", "0", "no", "n"}:
                return False
        return None

    @staticmethod
    def _ensure_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return [value]
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        return [value]

    def _resolve(self, key: str, default: Any | None = None) -> Any | None:
        if key in self.legacy_values:
            return self.legacy_values.get(key)
        return self.payload.get(key, default)

    def _tg_parts(self) -> tuple[int | None, int | None]:
        chat = self._resolve("tg_chat_id")
        msg = self._resolve("message_id")
        chat_id = self._to_int(chat)
        message_id = self._to_int(msg)
        if chat_id is not None or message_id is not None:
            return chat_id, message_id
        if self.tg_message_id:
            raw = str(self.tg_message_id)
            if ":" in raw:
                chat_text, message_text = raw.split(":", 1)
                chat_id = self._to_int(chat_text)
                message_id = self._to_int(message_text)
            else:
                chat_id = None
                message_id = self._to_int(raw)
        return chat_id, message_id

    @property
    def channel_id(self) -> int | None:
        return self._to_int(self._resolve("channel_id"))

    @property
    def tg_chat_id(self) -> int | None:
        chat, _ = self._tg_parts()
        return chat

    @property
    def message_id(self) -> int | None:
        _, message = self._tg_parts()
        return message

    @property
    def origin(self) -> str | None:
        value = self._resolve("origin")
        if value is None:
            return "weather"
        return str(value)

    @property
    def caption_template(self) -> str | None:
        value = self._resolve("caption_template")
        return str(value) if value is not None else None

    @property
    def caption(self) -> str | None:
        value = self._resolve("caption")
        return str(value) if value is not None else None

    @property
    def hashtags(self) -> str | None:
        value = self._resolve("hashtags")
        return str(value) if value is not None else None

    @property
    def categories(self) -> list[str]:
        labels_source = self.labels
        values: list[Any] = []
        if isinstance(labels_source, list):
            values = labels_source
        elif isinstance(labels_source, dict):
            raw = labels_source.get("categories") or labels_source.get("labels")
            if isinstance(raw, list):
                values = raw
            elif raw is not None:
                values = [raw]
        if not values:
            raw_payload = self._resolve("categories")
            if isinstance(raw_payload, list):
                values = raw_payload
            elif isinstance(raw_payload, str):
                try:
                    parsed = json.loads(raw_payload)
                    if isinstance(parsed, list):
                        values = parsed
                    else:
                        values = [raw_payload]
                except json.JSONDecodeError:
                    values = [v.strip() for v in raw_payload.split(",") if v.strip()]
            elif raw_payload is not None:
                values = [raw_payload]
        normalized: list[str] = []
        seen: set[str] = set()
        for item in values:
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized

    @property
    def kind(self) -> str | None:
        value = self._resolve("kind")
        if value is not None:
            return str(value)
        return None

    @property
    def file_id(self) -> str | None:
        value = self._resolve("file_id")
        if value is None and self.file_ref is not None:
            return str(self.file_ref)
        return str(value) if value is not None else None

    @property
    def file_unique_id(self) -> str | None:
        value = self._resolve("file_unique_id")
        return str(value) if value is not None else None

    @property
    def file_name(self) -> str | None:
        value = self._resolve("file_name")
        return str(value) if value is not None else None

    @property
    def mime_type(self) -> str | None:
        value = self._resolve("mime_type") or self.content_type
        return str(value) if value is not None else None

    @property
    def file_size(self) -> int | None:
        return self._to_int(self._resolve("file_size"))

    @property
    def duration(self) -> int | None:
        return self._to_int(self._resolve("duration"))

    @property
    def recognized_message_id(self) -> int | None:
        value = self._resolve("recognized_message_id")
        if value is None and self.payload:
            value = self.payload.get("recognized_message_id")
        return self._to_int(value)

    @property
    def exif_present(self) -> bool:
        value = self._resolve("exif_present")
        bool_value = self._to_bool(value)
        if bool_value is not None:
            return bool_value
        return bool(self.exif)

    @property
    def latitude(self) -> float | None:
        return self._to_float(self._resolve("latitude"))

    @property
    def longitude(self) -> float | None:
        return self._to_float(self._resolve("longitude"))

    @property
    def city(self) -> str | None:
        value = self._resolve("city")
        return str(value) if value is not None else None

    @property
    def country(self) -> str | None:
        value = self._resolve("country")
        return str(value) if value is not None else None

    @property
    def author_user_id(self) -> int | None:
        return self._to_int(self._resolve("author_user_id"))

    @property
    def author_username(self) -> str | None:
        value = self._resolve("author_username")
        return str(value) if value is not None else None

    @property
    def sender_chat_id(self) -> int | None:
        return self._to_int(self._resolve("sender_chat_id"))

    @property
    def via_bot_id(self) -> int | None:
        return self._to_int(self._resolve("via_bot_id"))

    @property
    def forward_from_user(self) -> int | None:
        return self._to_int(self._resolve("forward_from_user"))

    @property
    def forward_from_chat(self) -> int | None:
        return self._to_int(self._resolve("forward_from_chat"))

    @property
    def local_path(self) -> str | None:
        if self._local_path is not None:
            return self._local_path
        value = self._resolve("local_path")
        return str(value) if value is not None else None

    @property
    def metadata(self) -> dict[str, Any] | None:
        value = self._resolve("metadata")
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, dict):
                return parsed
        return None

    @property
    def vision_results(self) -> dict[str, Any] | None:
        if self._vision_results is not None:
            return self._vision_results
        value = self._resolve("vision_results")
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, dict):
                return parsed
        return None

    @vision_results.setter
    def vision_results(self, value: dict[str, Any] | None) -> None:
        self._vision_results = value

    @property
    def rubric_id(self) -> int | None:
        if self._rubric_id is not None:
            return self._rubric_id
        return self._to_int(self._resolve("rubric_id"))

    @property
    def vision_category(self) -> str | None:
        if self._vision_category is not None:
            return self._vision_category
        value = self._resolve("vision_category")
        return str(value) if value is not None else None

    @property
    def vision_arch_view(self) -> str | None:
        if self._vision_arch_view is not None:
            return self._vision_arch_view
        value = self._resolve("vision_arch_view")
        return str(value) if value is not None else None

    @property
    def vision_photo_weather(self) -> str | None:
        if self._vision_photo_weather is not None:
            return self._vision_photo_weather
        value = self._resolve("vision_photo_weather")
        return str(value) if value is not None else None

    @property
    def vision_flower_varieties(self) -> list[str] | None:
        if self._vision_flower_varieties is not None:
            return self._vision_flower_varieties
        value = self._resolve("vision_flower_varieties")
        if value is None:
            return None
        entries = self._ensure_list(value)
        return [str(item) for item in entries]

    @property
    def vision_confidence(self) -> float | None:
        if self._vision_confidence is not None:
            return self._vision_confidence
        return self._to_float(self._resolve("vision_confidence"))

    @property
    def vision_caption(self) -> str | None:
        if self._vision_caption is not None:
            return self._vision_caption
        value = self._resolve("vision_caption")
        return str(value) if value is not None else None

    @vision_caption.setter
    def vision_caption(self, value: str | None) -> None:
        self._vision_caption = value


@dataclass
class DeviceUploadStats:
    device_id: str
    name: str | None
    count: int


@dataclass(frozen=True)
class MobileUploadStats:
    total: int
    today: int
    seven_days: int
    thirty_days: int
    top_devices: list[DeviceUploadStats]


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


@dataclass
class JobRecord:
    id: int
    name: str
    payload: dict[str, Any]
    status: str
    attempts: int
    available_at: datetime | None
    last_error: str | None
    created_at: datetime
    updated_at: datetime


class DataAccess:
    """High level helpers for working with the SQLite database."""

    _VISION_CATEGORY_GROUPS: dict[str, set[str]] = {
        "flowers": {"flowers", "flower"},
    }

    def __init__(self, connection: sqlite3.Connection):
        self.conn = connection
        self.conn.row_factory = sqlite3.Row
        self._ensure_assets_payload_schema()
        self.conn.row_factory = sqlite3.Row

    def _table_info(self, table: str) -> list[sqlite3.Row]:
        try:
            cursor = self.conn.execute(f"PRAGMA table_info('{table}')")
        except sqlite3.OperationalError:
            return []
        return cursor.fetchall()

    @staticmethod
    def _column_name(column: sqlite3.Row) -> str:
        try:
            return str(column["name"])  # type: ignore[index]
        except (KeyError, TypeError):
            return str(column[1])

    def _has_column(self, table: str, column: str) -> bool:
        for entry in self._table_info(table):
            if self._column_name(entry) == column:
                return True
        return False

    def _ensure_assets_payload_schema(self) -> None:
        columns = self._table_info("assets")
        if not columns:
            return
        column_names = {self._column_name(col) for col in columns}
        if {"payload_json", "tg_message_id"}.issubset(column_names):
            return
        migration_path = Path(__file__).resolve().parent / "migrations" / "0020_assets_uploads.py"
        if not migration_path.exists():
            return
        spec = importlib.util.spec_from_file_location(
            "_migration_0020_assets_uploads", migration_path
        )
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "run"):
            module.run(self.conn)
            self.conn.commit()

    def create_asset(
        self,
        *,
        upload_id: str,
        file_ref: str,
        content_type: str | None,
        sha256: str,
        width: int | None,
        height: int | None,
        exif: dict[str, Any] | None = None,
        labels: dict[str, Any] | None = None,
        tg_message_id: str | int | None = None,
        tg_chat_id: int | None = None,
        source: str = "mobile",
    ) -> str:
        """Create a new asset entry tied to an upload and return its UUID."""

        asset_id = str(uuid4())
        now = datetime.utcnow().isoformat()
        exif_json = self._encode_exif_blob(exif)
        labels_json = self._encode_json_blob(labels)
        chat_id_value = Asset._to_int(tg_chat_id)
        message_id_value: int | None = None
        identifier: str | None = None

        if isinstance(tg_message_id, str):
            text = tg_message_id.strip()
            if text:
                identifier = text
                if ":" in text:
                    chat_text, message_text = text.split(":", 1)
                    if chat_id_value is None:
                        chat_id_value = Asset._to_int(chat_text)
                    message_id_value = Asset._to_int(message_text)
                else:
                    message_id_value = Asset._to_int(text)
        elif tg_message_id is not None:
            message_id_value = Asset._to_int(tg_message_id)
            identifier = str(tg_message_id)

        if chat_id_value is not None and message_id_value is not None:
            identifier = self._build_tg_message_identifier(chat_id_value, message_id_value)
        elif identifier is None and message_id_value is not None:
            identifier = str(message_id_value)

        payload_map: dict[str, Any] = {}
        if chat_id_value is not None:
            payload_map["tg_chat_id"] = chat_id_value
        if message_id_value is not None:
            payload_map["message_id"] = message_id_value
        payload_json = json.dumps(payload_map, ensure_ascii=False) if payload_map else None

        self.conn.execute(
            """
            INSERT INTO assets (
                id,
                upload_id,
                file_ref,
                content_type,
                sha256,
                width,
                height,
                exif_json,
                labels_json,
                tg_message_id,
                payload_json,
                created_at,
                source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asset_id,
                upload_id,
                file_ref,
                content_type,
                sha256,
                width,
                height,
                exif_json,
                labels_json,
                identifier,
                payload_json,
                now,
                source,
            ),
        )
        self.conn.commit()
        return asset_id

    def insert_uploaded_asset(
        self,
        *,
        upload_id: str,
        file_ref: str,
        content_type: str | None,
        sha256: str,
        width: int | None,
        height: int | None,
        exif: dict[str, Any] | None = None,
        labels: dict[str, Any] | None = None,
        tg_message_id: str | int | None = None,
        tg_chat_id: int | None = None,
        source: str = "mobile",
    ) -> str:
        """Compatibility wrapper for legacy callers."""

        return self.create_asset(
            upload_id=upload_id,
            file_ref=file_ref,
            content_type=content_type,
            sha256=sha256,
            width=width,
            height=height,
            exif=exif,
            labels=labels,
            tg_message_id=tg_message_id,
            tg_chat_id=tg_chat_id,
            source=source,
        )

    def get_mobile_upload_stats(self, *, limit: int = 5) -> MobileUploadStats:
        """Return aggregate counters for mobile uploads."""

        counters = self.conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN datetime(created_at) >= datetime('now', 'start of day') THEN 1 ELSE 0 END) AS today,
                SUM(CASE WHEN datetime(created_at) >= datetime('now', '-6 days', 'start of day') THEN 1 ELSE 0 END) AS seven_days,
                SUM(CASE WHEN datetime(created_at) >= datetime('now', '-29 days', 'start of day') THEN 1 ELSE 0 END) AS thirty_days
            FROM uploads
            WHERE source='mobile' AND status='done'
            """
        ).fetchone()

        def _value(row: sqlite3.Row | tuple[Any, ...] | None, key: str | int) -> int:
            if not row:
                return 0
            try:
                if isinstance(row, sqlite3.Row):
                    raw = row[key]  # type: ignore[index]
                else:
                    raw = row[key]  # type: ignore[index]
            except (KeyError, TypeError, IndexError):
                return 0
            if raw is None:
                return 0
            try:
                return int(raw)
            except (TypeError, ValueError):
                return 0

        total = _value(counters, "total")
        today = _value(counters, "today")
        seven = _value(counters, "seven_days")
        thirty = _value(counters, "thirty_days")

        device_rows = self.conn.execute(
            """
            SELECT u.device_id, d.name, COUNT(*) AS uploads
            FROM uploads u
            LEFT JOIN devices d ON d.id = u.device_id
            WHERE u.source='mobile' AND u.status='done'
            GROUP BY u.device_id
            ORDER BY uploads DESC, u.device_id
            LIMIT ?
            """,
            (max(0, int(limit)),),
        ).fetchall()

        devices: list[DeviceUploadStats] = []
        for row in device_rows:
            device_id = str(row["device_id"]) if isinstance(row, sqlite3.Row) else str(row[0])
            name_value: str | None
            if isinstance(row, sqlite3.Row):
                raw_name = row["name"]
                uploads_count = row["uploads"]
            else:
                raw_name = row[1]
                uploads_count = row[2]
            name_value = str(raw_name) if raw_name is not None else None
            try:
                count_value = int(uploads_count)
            except (TypeError, ValueError):
                count_value = 0
            devices.append(
                DeviceUploadStats(
                    device_id=device_id,
                    name=name_value,
                    count=count_value,
                )
            )

        return MobileUploadStats(
            total=total,
            today=today,
            seven_days=seven,
            thirty_days=thirty,
            top_devices=devices,
        )

    @staticmethod
    def _collect_categories(
        hashtags: str | None, categories: Iterable[str] | None
    ) -> list[str]:
        if categories is not None:
            candidates = list(dict.fromkeys(categories))
        elif hashtags:
            candidates = [tag.strip() for tag in hashtags.split() if tag.strip()]
        else:
            candidates = []
        normalized: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized

    @staticmethod
    def _decode_payload_blob(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return dict(raw)
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return {}
            if isinstance(parsed, dict):
                return dict(parsed)
        return {}

    @staticmethod
    def _encode_payload_blob(payload: dict[str, Any]) -> str | None:
        return DataAccess._encode_json_blob(payload, sort_keys=True)

    @staticmethod
    def _encode_json_blob(payload: Any, *, sort_keys: bool = False) -> str | None:
        if payload is None:
            return None
        if isinstance(payload, (dict, list, tuple, set)) and not payload:
            return None
        safe_payload = DataAccess._make_json_safe(payload)
        try:
            return json.dumps(safe_payload, ensure_ascii=False, sort_keys=sort_keys)
        except TypeError:
            fallback = DataAccess._make_json_safe(str(safe_payload))
            return json.dumps(fallback, ensure_ascii=False, sort_keys=sort_keys)

    @staticmethod
    def _encode_exif_blob(payload: Any) -> str | None:
        if payload is None:
            return None
        safe_payload = DataAccess._make_json_safe(payload)
        try:
            return json.dumps(safe_payload, ensure_ascii=False)
        except TypeError:
            fallback = DataAccess._make_json_safe(str(safe_payload))
            return json.dumps(fallback, ensure_ascii=False)

    @staticmethod
    def _try_ratio_tuple(value: tuple[Any, ...]) -> float | None:
        if len(value) != 2:
            return None
        numerator, denominator = value
        try:
            numerator_value = float(numerator)
            denominator_value = float(denominator)
        except (TypeError, ValueError):
            return None
        if denominator_value == 0:
            return numerator_value
        return numerator_value / denominator_value

    @staticmethod
    def _make_json_safe(value: Any) -> Any:
        def _coerce(obj: Any) -> Any:
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            if isinstance(obj, (bytes, bytearray, memoryview)):
                return bytes(obj).hex()
            if isinstance(obj, dict):
                return {str(key): _coerce(val) for key, val in obj.items()}
            if isinstance(obj, tuple):
                ratio = DataAccess._try_ratio_tuple(obj)
                if ratio is not None:
                    return ratio
            if isinstance(obj, (list, tuple, set)):
                return [_coerce(item) for item in obj]
            numerator = getattr(obj, "numerator", None)
            denominator = getattr(obj, "denominator", None)
            if numerator is not None and denominator is not None:
                try:
                    numerator_value = float(numerator)
                    denominator_value = float(denominator)
                    if denominator_value != 0:
                        return numerator_value / denominator_value
                    return numerator_value
                except Exception:
                    return str(obj)
            isoformat = getattr(obj, "isoformat", None)
            if callable(isoformat):
                try:
                    return isoformat()
                except Exception:
                    pass
            as_float = getattr(obj, "__float__", None)
            if callable(as_float):
                try:
                    return float(obj)
                except Exception:
                    pass
            return str(obj)

        return _coerce(value)

    @staticmethod
    def _update_payload_map(
        payload: dict[str, Any], updates: dict[str, Any | object]
    ) -> dict[str, Any]:
        for key, value in updates.items():
            if value is _UNSET:
                continue
            if value is None:
                payload.pop(key, None)
            else:
                payload[key] = value
        return payload

    @staticmethod
    def _build_tg_message_identifier(
        tg_chat_id: int | None, message_id: int | None
    ) -> str | None:
        if tg_chat_id is None and message_id is None:
            return None
        if tg_chat_id is not None and message_id is not None:
            return f"{tg_chat_id}:{message_id}"
        value = tg_chat_id if tg_chat_id is not None else message_id
        return str(value) if value is not None else None

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

    @classmethod
    def _normalize_vision_category(cls, category: str | None) -> str | None:
        if category is None:
            return None
        normalized = category.strip().lower()
        if not normalized:
            return normalized
        for canonical, variants in cls._VISION_CATEGORY_GROUPS.items():
            if normalized in variants:
                return canonical
        return normalized

    @classmethod
    def _vision_category_variants(cls, category: str) -> set[str]:
        canonical = cls._normalize_vision_category(category)
        if canonical is None:
            return set()
        variants = cls._VISION_CATEGORY_GROUPS.get(canonical)
        if variants:
            return set(variants)
        return {canonical}

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
        origin: str = "weather",
        source: str = "telegram",
    ) -> str:
        """Insert or update asset metadata."""

        now = datetime.utcnow().isoformat()
        categories_list = self._collect_categories(hashtags, categories)
        cleaned_metadata = self._prepare_metadata(metadata)
        file_meta = file_meta or {}

        existing = self.get_asset_by_message(tg_chat_id, message_id)
        if categories_list:
            labels_json = json.dumps(categories_list, ensure_ascii=False)
        elif existing:
            labels_json = existing.labels_json
        else:
            labels_json = None
        payload = dict(existing.payload) if existing else {}
        if "created_at" not in payload:
            payload["created_at"] = existing.created_at if existing else now
        payload_updates: dict[str, Any | object] = {
            "channel_id": channel_id,
            "tg_chat_id": tg_chat_id,
            "message_id": message_id,
            "origin": origin,
            "caption_template": template,
            "caption": caption,
            "hashtags": hashtags,
            "categories": categories_list,
            "kind": kind,
            "author_user_id": author_user_id,
            "author_username": author_username,
            "sender_chat_id": sender_chat_id,
            "via_bot_id": via_bot_id,
            "forward_from_user": forward_from_user,
            "forward_from_chat": forward_from_chat,
            "rubric_id": rubric_id,
            "updated_at": now,
        }
        payload = self._update_payload_map(payload, payload_updates)
        if cleaned_metadata is not None:
            payload["metadata"] = cleaned_metadata
        else:
            payload.pop("metadata", None)

        file_size = Asset._to_int(file_meta.get("file_size"))
        duration = Asset._to_int(file_meta.get("duration"))
        file_meta_updates: dict[str, Any | object] = {}
        if "file_id" in file_meta:
            file_meta_updates["file_id"] = file_meta.get("file_id")
        if "file_unique_id" in file_meta:
            file_meta_updates["file_unique_id"] = file_meta.get("file_unique_id")
        if "file_name" in file_meta:
            file_meta_updates["file_name"] = file_meta.get("file_name")
        if "mime_type" in file_meta:
            file_meta_updates["mime_type"] = file_meta.get("mime_type")
        if "file_size" in file_meta:
            file_meta_updates["file_size"] = file_size
        if "duration" in file_meta:
            file_meta_updates["duration"] = duration
        if file_meta_updates:
            payload = self._update_payload_map(payload, file_meta_updates)

        payload_json = self._encode_payload_blob(payload)

        file_ref = (
            file_meta.get("file_ref")
            or file_meta.get("file_id")
            or (existing.file_ref if existing else None)
        )
        content_type = file_meta.get("mime_type") or (existing.content_type if existing else None)
        sha256 = file_meta.get("sha256") or (existing.sha256 if existing else None)
        width = Asset._to_int(file_meta.get("width"))
        if width is None and existing:
            width = existing.width
        height = Asset._to_int(file_meta.get("height"))
        if height is None and existing:
            height = existing.height

        exif_json_value = file_meta.get("exif_json")
        if exif_json_value is None and "exif" in file_meta:
            exif_json_value = self._encode_exif_blob(file_meta["exif"])
        if exif_json_value is None and existing:
            exif_json_value = existing.exif_json

        tg_identifier = self._build_tg_message_identifier(tg_chat_id, message_id)

        if existing:
            self.conn.execute(
                """
                UPDATE assets
                   SET file_ref=?,
                       content_type=?,
                       sha256=?,
                       width=?,
                       height=?,
                       exif_json=?,
                       labels_json=?,
                       tg_message_id=?,
                       payload_json=?
                 WHERE id=?
                """,
                (
                    file_ref,
                    content_type,
                    sha256,
                    width,
                    height,
                    exif_json_value,
                    labels_json,
                    tg_identifier,
                    payload_json,
                    existing.id,
                ),
            )
            self.conn.commit()
            return existing.id

        asset_id = str(uuid4())
        payload["created_at"] = payload.get("created_at", now)
        payload_json = self._encode_payload_blob(payload)
        self.conn.execute(
            """
            INSERT INTO assets (
                id,
                upload_id,
                file_ref,
                content_type,
                sha256,
                width,
                height,
                exif_json,
                labels_json,
                tg_message_id,
                payload_json,
                created_at,
                source
            ) VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asset_id,
                file_ref,
                content_type,
                sha256,
                width,
                height,
                exif_json_value,
                labels_json,
                tg_identifier,
                payload_json,
                now,
                source,
            ),
        )
        self.conn.commit()
        return asset_id

    def update_recognized_message(
        self, asset_id: str | int, *, message_id: int | None
    ) -> None:
        """Store the Telegram message that acknowledged the asset."""

        asset = self.get_asset(str(asset_id))
        if not asset:
            logging.warning("Attempted to update missing asset %s", asset_id)
            return
        payload = dict(asset.payload)
        if message_id is None:
            payload.pop("recognized_message_id", None)
        else:
            payload["recognized_message_id"] = int(message_id)
        payload["updated_at"] = datetime.utcnow().isoformat()
        payload_json = self._encode_payload_blob(payload)
        self.conn.execute(
            "UPDATE assets SET payload_json=? WHERE id=?",
            (payload_json, asset.id),
        )
        self.conn.commit()

    def update_asset(
        self,
        asset_id: str | int,
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
        local_path: str | None | object = _UNSET,
        vision_caption: str | None = None,
        origin: str | None = None,
    ) -> None:
        """Update selected asset fields while preserving unset values."""

        row = self.get_asset(str(asset_id))
        if not row:
            logging.warning("Attempted to update missing asset %s", asset_id)
            return
        payload = dict(row.payload)
        payload_updates: dict[str, Any | object] = {}
        columns: dict[str, Any] = {}
        column_dirty = False

        if template is not None:
            payload_updates["caption_template"] = template
        if caption is not None:
            payload_updates["caption"] = caption
        if hashtags is not None:
            categories_list = self._collect_categories(hashtags, None)
            payload_updates["hashtags"] = hashtags
            payload_updates["categories"] = categories_list
            columns["labels_json"] = json.dumps(categories_list, ensure_ascii=False)
            column_dirty = True
        if kind is not None:
            payload_updates["kind"] = kind
        if file_meta is not None:
            fm = file_meta or {}
            file_ref_value = fm.get("file_ref") or fm.get("file_id") or row.file_ref
            columns["file_ref"] = file_ref_value
            content_type_value = fm.get("mime_type") or row.content_type
            columns["content_type"] = content_type_value
            sha256_value = fm.get("sha256") or row.sha256
            columns["sha256"] = sha256_value
            width_value = Asset._to_int(fm.get("width"))
            if width_value is None:
                width_value = row.width
            columns["width"] = width_value
            height_value = Asset._to_int(fm.get("height"))
            if height_value is None:
                height_value = row.height
            columns["height"] = height_value
            exif_json_value = fm.get("exif_json")
            if exif_json_value is None and "exif" in fm:
                exif_json_value = self._encode_exif_blob(fm["exif"])
            if exif_json_value is None:
                exif_json_value = row.exif_json
            columns["exif_json"] = exif_json_value
            column_dirty = True
            file_size_value = Asset._to_int(fm.get("file_size"))
            duration_value = Asset._to_int(fm.get("duration"))
            if "file_id" in fm:
                payload_updates["file_id"] = fm.get("file_id")
            if "file_unique_id" in fm:
                payload_updates["file_unique_id"] = fm.get("file_unique_id")
            if "file_name" in fm:
                payload_updates["file_name"] = fm.get("file_name")
            if "mime_type" in fm:
                payload_updates["mime_type"] = fm.get("mime_type")
            if "file_size" in fm:
                payload_updates["file_size"] = file_size_value
            if "duration" in fm:
                payload_updates["duration"] = duration_value
        if author_user_id is not None:
            payload_updates["author_user_id"] = author_user_id
        if author_username is not None:
            payload_updates["author_username"] = author_username
        if sender_chat_id is not None:
            payload_updates["sender_chat_id"] = sender_chat_id
        if via_bot_id is not None:
            payload_updates["via_bot_id"] = via_bot_id
        if forward_from_user is not None:
            payload_updates["forward_from_user"] = forward_from_user
        if forward_from_chat is not None:
            payload_updates["forward_from_chat"] = forward_from_chat
        if metadata is not None:
            existing = row.metadata or {}
            merged = existing.copy()
            merged.update(metadata)
            cleaned = self._prepare_metadata(merged)
            payload_updates["metadata"] = cleaned
        if recognized_message_id is not None:
            payload_updates["recognized_message_id"] = recognized_message_id
        if rubric_id is not None:
            payload_updates["rubric_id"] = rubric_id
        if latitude is not None:
            payload_updates["latitude"] = latitude
        if longitude is not None:
            payload_updates["longitude"] = longitude
        if city is not None:
            payload_updates["city"] = city
        if country is not None:
            payload_updates["country"] = country
        if exif_present is not None:
            payload_updates["exif_present"] = bool(exif_present)
        if local_path is not _UNSET:
            payload_updates["local_path"] = (
                str(local_path) if local_path is not None else None
            )
        if vision_category is not None:
            payload_updates["vision_category"] = self._normalize_vision_category(
                vision_category
            )
        if vision_arch_view is not None:
            payload_updates["vision_arch_view"] = vision_arch_view
        if vision_photo_weather is not None:
            payload_updates["vision_photo_weather"] = vision_photo_weather
        if vision_flower_varieties is not None:
            payload_updates["vision_flower_varieties"] = [
                str(item) for item in vision_flower_varieties
            ]
        if vision_confidence is not None:
            payload_updates["vision_confidence"] = vision_confidence
        if vision_caption is not None:
            payload_updates["vision_caption"] = vision_caption
        if origin is not None:
            payload_updates["origin"] = origin

        payload_dirty = bool(payload_updates)
        if column_dirty or payload_dirty:
            payload_updates["updated_at"] = datetime.utcnow().isoformat()
            payload = self._update_payload_map(payload, payload_updates)
            columns["payload_json"] = self._encode_payload_blob(payload)
            column_dirty = True

        performed_write = False
        if column_dirty:
            assignments = ", ".join(f"{k} = ?" for k in columns)
            params: list[Any] = list(columns.values())
            params.append(row.id)
            self.conn.execute(
                f"UPDATE assets SET {assignments} WHERE id=?",
                params,
            )
            performed_write = True
        if vision_results is not None:
            self._store_vision_result(row.id, vision_results)
            performed_write = True
        if performed_write:
            self.conn.commit()

    def update_asset_categories_merge(
        self, asset_id: str | int, to_add: Iterable[str]
    ) -> None:
        row = self.get_asset(str(asset_id))
        if not row:
            logging.warning("Attempted to update categories for missing asset %s", asset_id)
            return
        payload = dict(row.payload)
        current_categories = payload.get("categories")
        if not isinstance(current_categories, list):
            current_categories = []
        existing = [str(c).strip() for c in current_categories if str(c).strip()]
        existing_lower = {c.lower() for c in existing}
        merged = list(existing)
        for item in to_add:
            normalized = str(item).strip()
            if not normalized:
                continue
            if normalized.lower() not in existing_lower:
                merged.append(normalized)
                existing_lower.add(normalized.lower())
        payload["categories"] = merged
        payload["updated_at"] = datetime.utcnow().isoformat()
        payload_json = self._encode_payload_blob(payload)
        labels_json = json.dumps(merged, ensure_ascii=False)
        self.conn.execute(
            "UPDATE assets SET payload_json = ?, labels_json = ? WHERE id = ?",
            (payload_json, labels_json, str(asset_id)),
        )
        self.conn.commit()

    def get_asset(self, asset_id: str | int) -> Asset | None:
        return self._fetch_asset("a.id = ?", (str(asset_id),))

    def get_asset_by_message(self, tg_chat_id: int, message_id: int) -> Asset | None:
        identifier = self._build_tg_message_identifier(tg_chat_id, message_id)
        if identifier is None:
            return None
        return self._fetch_asset("a.tg_message_id = ?", (identifier,))

    def is_recognized_message(self, tg_chat_id: int | None, message_id: int | None) -> bool:
        """Check if the pair matches a previously recognized asset message."""

        if tg_chat_id is None or message_id is None:
            return False
        row = self.conn.execute(
            """
            SELECT 1
              FROM assets
             WHERE json_extract(payload_json, '$.recognized_message_id') = ?
               AND COALESCE(
                       json_extract(payload_json, '$.tg_chat_id'),
                       json_extract(payload_json, '$.channel_id')
                   ) = ?
             LIMIT 1
            """,
            (int(message_id), int(tg_chat_id)),
        ).fetchone()
        return row is not None

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

        keys = row.keys()
        row_dict = {key: row[key] for key in keys}

        def _parse_json(raw: Any) -> Any | None:
            if raw is None:
                return None
            if isinstance(raw, (dict, list)):
                return raw
            if isinstance(raw, str):
                text = raw.strip()
                if not text:
                    return None
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return None
            return None

        width = Asset._to_int(row_dict.get("width"))
        height = Asset._to_int(row_dict.get("height"))
        exif_json = row_dict.get("exif_json")
        exif_data = _parse_json(exif_json)
        if exif_data is not None and not isinstance(exif_data, dict):
            exif_data = None
        labels_json = row_dict.get("labels_json")
        labels_data = _parse_json(labels_json)
        if labels_data is None and isinstance(labels_json, str) and labels_json.strip():
            try:
                labels_data = json.loads(labels_json)
            except json.JSONDecodeError:
                labels_data = labels_json
        payload_json = row_dict.get("payload_json")
        payload_data = self._decode_payload_blob(payload_json)
        created_at_raw = row_dict.get("created_at")
        created_at = str(created_at_raw) if created_at_raw is not None else datetime.utcnow().isoformat()
        raw_vision: str | None
        if "vision_payload" in keys:
            raw_payload_value = row_dict.get("vision_payload")
            raw_vision = str(raw_payload_value) if raw_payload_value is not None else None
        else:
            raw_vision = self._load_latest_vision_json(row_dict.get("id"))
        vision = None
        if raw_vision:
            try:
                vision = json.loads(raw_vision)
            except (TypeError, json.JSONDecodeError):
                vision = None
        vision_category_raw = payload_data.get("vision_category")
        vision_category = None
        if vision_category_raw is not None:
            vision_category = self._normalize_vision_category(str(vision_category_raw))
        vision_arch_view = payload_data.get("vision_arch_view")
        if vision_arch_view is not None:
            vision_arch_view = str(vision_arch_view)
        vision_photo_weather = payload_data.get("vision_photo_weather")
        if vision_photo_weather is not None:
            vision_photo_weather = str(vision_photo_weather)
        flower_varieties: list[str] | None = None
        raw_flowers = payload_data.get("vision_flower_varieties")
        if raw_flowers is not None:
            entries = Asset._ensure_list(raw_flowers)
            flower_varieties = [str(item) for item in entries]
        vision_confidence = Asset._to_float(payload_data.get("vision_confidence"))
        local_path_value = payload_data.get("local_path")
        local_path = str(local_path_value) if local_path_value is not None else None
        rubric_id = Asset._to_int(payload_data.get("rubric_id"))

        tg_message_id_raw = row_dict.get("tg_message_id")
        tg_message_id = None
        if tg_message_id_raw is not None:
            tg_message_id = str(tg_message_id_raw)

        payload_json_text: str | None
        if payload_json is None or isinstance(payload_json, str):
            payload_json_text = payload_json
        else:
            payload_json_text = json.dumps(payload_json, ensure_ascii=False)

        return Asset(
            id=str(row_dict.get("id")),
            upload_id=row_dict.get("upload_id"),
            file_ref=row_dict.get("file_ref"),
            content_type=row_dict.get("content_type"),
            sha256=row_dict.get("sha256"),
            width=width,
            height=height,
            exif_json=exif_json,
            labels_json=labels_json,
            tg_message_id=tg_message_id,
            payload_json=payload_json_text,
            created_at=created_at,
            source=(
                str(row_dict.get("source"))
                if row_dict.get("source") is not None
                else None
            ),
            exif=exif_data,
            labels=labels_data,
            payload=payload_data,
            legacy_values={},
            _vision_results=vision,
            _vision_category=vision_category,
            _vision_arch_view=vision_arch_view,
            _vision_photo_weather=vision_photo_weather,
            _vision_flower_varieties=flower_varieties,
            _vision_confidence=vision_confidence,
            _vision_caption=(
                str(payload_data.get("vision_caption"))
                if payload_data.get("vision_caption") is not None
                else None
            ),
            _local_path=local_path,
            _rubric_id=rubric_id,
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
            ORDER BY COALESCE(json_extract(a.payload_json, '$.last_used_at'), a.created_at) ASC, a.id ASC
            """
        ).fetchall()
        for row in rows:
            asset = self._asset_from_row(row)
            if not asset:
                continue
            if rubric_id is not None and asset.rubric_id != rubric_id:
                continue
            yield asset

    def _fetch_assets(
        self,
        *,
        rubric_id: int | None = None,
        limit: int | None = None,
        where: Sequence[str] | None = None,
        params: Sequence[Any] | None = None,
        random_order: bool = False,
        mark_used: bool = False,
    ) -> list[Asset]:
        conditions = list(where or [])
        query_params: list[Any] = list(params or [])
        if rubric_id is not None:
            conditions.append("json_extract(a.payload_json, '$.rubric_id') = ?")
            query_params.append(rubric_id)
        sql = (
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
            """
        )
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY COALESCE(json_extract(a.payload_json, '$.last_used_at'), a.created_at) ASC, a.id ASC"
        apply_limit = limit if limit is not None and not random_order else None
        if apply_limit is not None:
            sql += " LIMIT ?"
            query_params.append(apply_limit)
        rows = self.conn.execute(sql, query_params).fetchall()
        assets: list[Asset] = []
        for row in rows:
            asset = self._asset_from_row(row)
            if not asset:
                continue
            assets.append(asset)
        if random_order and len(assets) > 1:
            random.shuffle(assets)
        if limit is not None:
            assets = assets[:limit]
        if mark_used:
            self.mark_assets_used(asset.id for asset in assets)
        return assets

    def fetch_assets_by_vision_category(
        self,
        category: str,
        *,
        rubric_id: int | None = None,
        limit: int | None = None,
        require_arch_view: bool = False,
        random_order: bool = False,
        mark_used: bool = False,
    ) -> list[Asset]:
        if not category:
            return []
        variants = sorted(self._vision_category_variants(category))
        if not variants:
            normalized = self._normalize_vision_category(category)
            if not normalized:
                return []
            variants = [normalized]
        placeholders = ",".join("?" for _ in variants)
        where = [
            f"LOWER(COALESCE(json_extract(a.payload_json, '$.vision_category'), '')) IN ({placeholders})"
        ]
        params: list[Any] = [variant.lower() for variant in variants]
        if require_arch_view:
            where.append(
                "TRIM(COALESCE(json_extract(a.payload_json, '$.vision_arch_view'), '')) != ''"
            )
        return self._fetch_assets(
            rubric_id=rubric_id,
            limit=limit,
            where=where,
            params=params,
            random_order=random_order,
            mark_used=mark_used,
        )

    @staticmethod
    def compute_age_bonus(last_used_at: str | None, now: datetime | None = None) -> float:
        reference = now or datetime.utcnow()
        if not last_used_at:
            return 2.0
        try:
            timestamp = datetime.fromisoformat(last_used_at)
        except Exception:
            return 1.0
        delta = reference - timestamp
        if delta.total_seconds() <= 0:
            return 0.5
        days = delta.total_seconds() / 86400.0
        return max(0.5, min(3.0, days / 3.0))

    def fetch_sea_candidates(
        self,
        rubric_id: int,
        *,
        limit: int = 48,
    ) -> list[dict[str, Any]]:
        assets = self.fetch_assets_by_vision_category(
            "sea",
            rubric_id=rubric_id,
            limit=limit,
            random_order=False,
            mark_used=False,
        )
        now = datetime.utcnow()
        candidates: list[dict[str, Any]] = []
        for asset in assets:
            vision = asset.vision_results or {}
            raw_wave = vision.get("sea_wave_score")
            if isinstance(raw_wave, dict):
                raw_wave = raw_wave.get("value")
            try:
                wave_score = float(raw_wave) if raw_wave is not None else None
            except (TypeError, ValueError):
                wave_score = None
            photo_sky_raw = vision.get("photo_sky")
            photo_sky = str(photo_sky_raw).strip().lower() if isinstance(photo_sky_raw, str) else None
            is_sunset = bool(vision.get("is_sunset"))
            season_guess_raw = vision.get("season_guess")
            season_guess = (
                str(season_guess_raw).strip().lower()
                if isinstance(season_guess_raw, str) and season_guess_raw.strip()
                else None
            )
            tags_raw = vision.get("tags")
            if isinstance(tags_raw, list):
                tags = {str(tag).strip().lower() for tag in tags_raw if str(tag).strip()}
            else:
                tags = set()
            payload_data = asset.payload if isinstance(asset.payload, dict) else {}
            last_used_value = payload_data.get("last_used_at") if payload_data else None
            last_used_dt: datetime | None
            if isinstance(last_used_value, str) and last_used_value.strip():
                try:
                    last_used_dt = datetime.fromisoformat(last_used_value)
                except ValueError:
                    last_used_dt = None
            else:
                last_used_dt = None
            age_bonus = self.compute_age_bonus(last_used_value if isinstance(last_used_value, str) else None, now=now)
            candidates.append(
                {
                    "asset": asset,
                    "wave_score": wave_score,
                    "photo_sky": photo_sky,
                    "is_sunset": is_sunset,
                    "season_guess": season_guess,
                    "tags": tags,
                    "last_used_at": last_used_dt,
                    "age_bonus": age_bonus,
                }
            )
        return candidates

    def mark_assets_used(self, asset_ids: Iterable[str | int]) -> None:

        ids = [str(asset_id) for asset_id in asset_ids]
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"SELECT id, payload_json FROM assets WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        if not rows:
            return
        now_iso = datetime.utcnow().isoformat()
        for row in rows:
            payload = self._decode_payload_blob(row["payload_json"])
            payload["last_used_at"] = now_iso
            payload["updated_at"] = now_iso
            payload_json = self._encode_payload_blob(payload)
            self.conn.execute(
                "UPDATE assets SET payload_json=? WHERE id=?",
                (payload_json, row["id"]),
            )
        self.conn.commit()

    def delete_assets(self, asset_ids: Sequence[str | int]) -> None:
        if not asset_ids:
            return
        placeholders = ",".join("?" for _ in asset_ids)
        params = tuple(str(a) for a in asset_ids)
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
            ORDER BY COALESCE(json_extract(a.payload_json, '$.last_used_at'), a.created_at) ASC, a.id ASC
            """
        ).fetchall()
        normalized = {t.lower() for t in tags} if tags else None
        for row in rows:
            asset = self._asset_from_row(row)
            if not asset:
                continue
            if asset.origin and asset.origin.lower() != "weather":
                continue
            if normalized:
                asset_tags = {t.lower() for t in asset.categories}
                if not asset_tags.intersection(normalized):
                    continue
            now_iso = datetime.utcnow().isoformat()
            asset.payload["last_used_at"] = now_iso
            asset.payload["updated_at"] = now_iso
            self.mark_assets_used([asset.id])
            return asset
        return None

    def _store_vision_result(self, asset_id: str | int, result: Any) -> None:
        payload = json.dumps(result) if not isinstance(result, str) else result
        provider = result.get("provider") if isinstance(result, dict) else None
        status = result.get("status") if isinstance(result, dict) else None
        category = None
        arch_view = None
        photo_weather = None
        flowers_raw: Any | None = None
        confidence = None
        if isinstance(result, dict):
            raw_tags = result.get("tags")
            normalized_tags: list[str] = []
            if isinstance(raw_tags, list):
                seen: set[str] = set()
                for tag in raw_tags:
                    text = str(tag).strip().lower()
                    if not text or text in seen:
                        continue
                    seen.add(text)
                    normalized_tags.append(text)
            normalized_tag_set = set(normalized_tags)
            category = (
                result.get("category")
                or result.get("caption")
                or result.get("primary_scene")
            )
            if not category and normalized_tags:
                category = normalized_tags[0]
            is_sea_flag = bool(result.get("is_sea"))
            if not is_sea_flag and normalized_tag_set.intersection(
                {"sea", "ocean", "beach", "coast", "shore", "shoreline", "seaside", "coastal"}
            ):
                is_sea_flag = True
            if is_sea_flag:
                category = "sea"
            category = self._normalize_vision_category(category)
            if category == "sunset":
                category = "sea" if is_sea_flag else None
            arch_view_value = result.get("arch_view")
            if isinstance(arch_view_value, bool):
                arch_view = "yes" if arch_view_value else ""
            elif arch_view_value is not None:
                arch_view = str(arch_view_value)
            weather_info = result.get("weather")
            if isinstance(weather_info, dict):
                label = weather_info.get("label")
                description = weather_info.get("description")
                photo_weather = (
                    str(description).strip() or str(label).strip()
                    if description or label
                    else None
                )
            if not photo_weather and result.get("photo_weather_display") is not None:
                photo_weather = (
                    str(result.get("photo_weather_display")).strip() or None
                )
            if not photo_weather and result.get("photo_weather") is not None:
                photo_weather = str(result.get("photo_weather")).strip() or None
            flowers_raw = result.get("flower_varieties")
            if flowers_raw is None:
                if normalized_tag_set.intersection({"flowers", "flower"}):
                    objects = result.get("objects")
                    if isinstance(objects, list):
                        extracted: list[str] = []
                        for entry in objects:
                            if isinstance(entry, str):
                                value = entry.strip()
                            elif isinstance(entry, dict):
                                value = str(
                                    entry.get("label")
                                    or entry.get("name")
                                    or ""
                                ).strip()
                            else:
                                value = ""
                            if value:
                                extracted.append(value)
                        flowers_raw = extracted or None
            raw_confidence = result.get("confidence")
            if isinstance(raw_confidence, (int, float)):
                confidence = float(raw_confidence)
            elif isinstance(raw_confidence, str):
                try:
                    confidence = float(raw_confidence)
                except ValueError:
                    confidence = None
            if confidence is None:
                raw_location_confidence = result.get("location_confidence")
                if isinstance(raw_location_confidence, (int, float)):
                    confidence = float(raw_location_confidence)
                elif isinstance(raw_location_confidence, str):
                    try:
                        confidence = float(raw_location_confidence)
                    except ValueError:
                        confidence = None
                if confidence is not None:
                    confidence = min(max(confidence, 0.0), 1.0)
        if isinstance(flowers_raw, list):
            flowers_json = json.dumps(flowers_raw)
        elif flowers_raw is None:
            flowers_json = None
        else:
            flowers_json = json.dumps([flowers_raw])
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
                str(asset_id),
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

    def _load_latest_vision_json(self, asset_id: str | int | None) -> str | None:
        if asset_id is None:
            return None
        row = self.conn.execute(
            """
            SELECT result_json FROM vision_results
            WHERE asset_id=?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (str(asset_id),),
        ).fetchone()
        if not row:
            return None
        return row["result_json"]

    def record_post_history(
        self,
        channel_id: int,
        message_id: int,
        asset_id: str | int | None,
        rubric_id: int | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        now = datetime.utcnow().isoformat()
        resolved_rubric = rubric_id
        if resolved_rubric is None and asset_id is not None:
            asset = self.get_asset(asset_id)
            if asset:
                resolved_rubric = asset.rubric_id
        asset_id_value = str(asset_id) if asset_id is not None else None
        self.conn.execute(
            """
            INSERT INTO posts_history (
                channel_id, message_id, asset_id, rubric_id, metadata, published_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                channel_id,
                message_id,
                asset_id_value,
                resolved_rubric,
                json.dumps(metadata) if metadata is not None else None,
                now,
                now,
            ),
        )
        self.conn.commit()

    def get_recent_rubric_metadata(self, rubric_code: str, limit: int = 5) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT ph.metadata
            FROM posts_history AS ph
            JOIN rubrics AS r ON ph.rubric_id = r.id
            WHERE r.code = ?
            ORDER BY ph.published_at DESC, ph.id DESC
            LIMIT ?
            """,
            (rubric_code, limit),
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            raw = row["metadata"]
            if not raw:
                result.append({})
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {}
            result.append(parsed if isinstance(parsed, dict) else {})
        return result

    def get_recent_rubric_pattern_ids(self, rubric_code: str, limit: int = 14) -> list[list[str]]:
        rows = self.conn.execute(
            """
            SELECT ph.metadata
            FROM posts_history AS ph
            JOIN rubrics AS r ON ph.rubric_id = r.id
            WHERE r.code = ?
            ORDER BY ph.published_at DESC, ph.id DESC
            LIMIT ?
            """,
            (rubric_code, limit),
        ).fetchall()
        result: list[list[str]] = []
        for row in rows:
            raw = row["metadata"]
            if not raw:
                result.append([])
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                result.append([])
                continue
            patterns_raw = payload.get("pattern_ids") or payload.get("patterns")
            if isinstance(patterns_raw, list):
                normalized = [str(item) for item in patterns_raw if str(item).strip()]
            else:
                normalized = []
            result.append(normalized)
        return result

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

    @staticmethod
    def _tz_offset_delta(tz_offset: str | None) -> timedelta:
        offset = (tz_offset or "+00:00").strip()
        sign = 1
        if offset.startswith("-"):
            sign = -1
            offset = offset[1:]
        elif offset.startswith("+"):
            offset = offset[1:]
        hours_str, _, minutes_str = offset.partition(":")
        try:
            hours = int(hours_str or "0")
            minutes = int(minutes_str or "0")
        except ValueError:
            return timedelta()
        return sign * timedelta(hours=hours, minutes=minutes)

    def get_daily_token_usage_total(
        self,
        *,
        day: date | None = None,
        models: Iterable[str] | None = None,
        tz_offset: str | None = None,
    ) -> int:
        offset = self._tz_offset_delta(tz_offset)
        if day is None:
            local_now = datetime.utcnow() + offset
            target_day = local_now.date()
        else:
            target_day = day
        start_local = datetime.combine(target_day, datetime.min.time())
        start_utc = start_local - offset
        end_utc = start_utc + timedelta(days=1)
        params: list[Any] = [start_utc.isoformat(), end_utc.isoformat()]
        query = (
            "SELECT COALESCE(SUM(COALESCE(total_tokens, 0)), 0) AS total "
            "FROM token_usage WHERE timestamp >= ? AND timestamp < ?"
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

    def upsert_rubric(
        self,
        code: str,
        title: str,
        *,
        config: dict[str, Any] | None = None,
    ) -> Rubric:
        now = datetime.utcnow().isoformat()
        payload = json.dumps(config or {}) if config else None
        self.conn.execute(
            """
            INSERT INTO rubrics (code, title, description, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(code) DO UPDATE SET
                title=excluded.title,
                description=excluded.description,
                updated_at=excluded.updated_at
            """,
            (code, title, payload, now, now),
        )
        self.conn.commit()
        rubric = self.get_rubric_by_code(code)
        if not rubric:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to load rubric {code}")
        return rubric

    def save_rubric_config(self, code: str, config: dict[str, Any]) -> None:
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """
            UPDATE rubrics
            SET description=?, updated_at=?
            WHERE code=?
            """,
            (json.dumps(config), now, code),
        )
        self.conn.commit()

    def get_rubric_config(self, code: str) -> dict[str, Any] | None:
        rubric = self.get_rubric_by_code(code)
        if not rubric:
            return None
        return rubric.config or {}

    @staticmethod
    def _normalize_schedules(config: dict[str, Any]) -> list[dict[str, Any]]:
        schedules = config.get("schedules") or config.get("schedule") or []
        if isinstance(schedules, dict):
            schedules = [schedules]
        elif not isinstance(schedules, list):
            schedules = []
        normalized: list[dict[str, Any]] = []
        for item in schedules:
            if isinstance(item, dict):
                normalized.append(dict(item))
        config["schedules"] = normalized
        config.pop("schedule", None)
        return normalized

    @staticmethod
    def _prepare_schedule_payload(schedule: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(schedule)
        days = prepared.get("days")
        if isinstance(days, set):
            prepared["days"] = sorted(days)
        elif isinstance(days, tuple):
            prepared["days"] = list(days)
        elif days is None:
            prepared.pop("days", None)
        channel_id = prepared.get("channel_id")
        if channel_id in {"", None}:
            prepared.pop("channel_id", None)
        else:
            try:
                prepared["channel_id"] = int(channel_id)
            except (TypeError, ValueError):
                prepared.pop("channel_id", None)
        enabled = prepared.get("enabled")
        if enabled is not None:
            prepared["enabled"] = bool(enabled)
        time_value = prepared.get("time")
        if time_value is not None:
            prepared["time"] = str(time_value)
        tz_value = prepared.get("tz")
        if tz_value is not None:
            prepared["tz"] = str(tz_value)
        return prepared

    def add_rubric_schedule(self, code: str, schedule: dict[str, Any]) -> list[dict[str, Any]]:
        config = self.get_rubric_config(code) or {}
        schedules = self._normalize_schedules(config)
        schedules.append(self._prepare_schedule_payload(schedule))
        self.save_rubric_config(code, config)
        return schedules

    def update_rubric_schedule(
        self,
        code: str,
        index: int,
        schedule: dict[str, Any],
    ) -> list[dict[str, Any]]:
        config = self.get_rubric_config(code) or {}
        schedules = self._normalize_schedules(config)
        if not 0 <= index < len(schedules):
            raise IndexError("Schedule index out of range")
        schedules[index] = self._prepare_schedule_payload(schedule)
        self.save_rubric_config(code, config)
        return schedules

    def remove_rubric_schedule(self, code: str, index: int) -> bool:
        config = self.get_rubric_config(code) or {}
        schedules = self._normalize_schedules(config)
        if not 0 <= index < len(schedules):
            return False
        schedule = schedules.pop(index)
        self.save_rubric_config(code, config)
        key = schedule.get("key")
        if key:
            self.conn.execute(
                "DELETE FROM rubric_schedule_state WHERE rubric_code=? AND schedule_key=?",
                (code, key),
            )
            self.conn.commit()
        return True

    def delete_rubric(self, code: str) -> bool:
        """Rubrics are immutable in production deployments."""

        raise NotImplementedError(
            "Deleting rubrics is disabled to protect fixed configurations"
        )

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

    # --- Job queue helpers -------------------------------------------------

    def enqueue_job(
        self,
        name: str,
        payload: dict[str, Any] | None,
        *,
        available_at: datetime | None = None,
    ) -> int:
        """Insert a new job into the queue using the normalized schema."""

        now = datetime.utcnow().isoformat()
        encoded_payload = json.dumps(payload or {})
        available = available_at.isoformat() if available_at else None
        cur = self.conn.execute(
            """
            INSERT INTO jobs_queue (
                name, payload, status, attempts, available_at, last_error, created_at, updated_at
            ) VALUES (?, ?, 'queued', 0, ?, NULL, ?, ?)
            """,
            (name, encoded_payload, available, now, now),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def fetch_due_jobs(self, *, limit: int = 50) -> list[JobRecord]:
        """Return queued jobs that are ready for execution."""

        now = datetime.utcnow().isoformat()
        rows = self.conn.execute(
            """
            SELECT *
            FROM jobs_queue
            WHERE status IN ('queued', 'delayed')
              AND (available_at IS NULL OR available_at <= ?)
            ORDER BY available_at IS NOT NULL, available_at, id
            LIMIT ?
            """,
            (now, limit),
        ).fetchall()
        return [self._job_from_row(row) for row in rows]

    def get_job(self, job_id: int) -> JobRecord | None:
        row = self.conn.execute(
            "SELECT * FROM jobs_queue WHERE id=?",
            (job_id,),
        ).fetchone()
        if not row:
            return None
        return self._job_from_row(row)

    def update_job_status(
        self,
        job_id: int,
        *,
        status: str,
        attempts: int | None = None,
        available_at: datetime | None | object = _UNSET,
        last_error: str | None | object = _UNSET,
    ) -> None:
        """Update queue bookkeeping after workers change a job state."""

        now = datetime.utcnow().isoformat()
        clauses = ["status=?", "updated_at=?"]
        params: list[Any] = [status, now]
        if attempts is not None:
            clauses.append("attempts=?")
            params.append(attempts)
        if available_at is not _UNSET:
            if available_at is not None:
                clauses.append("available_at=?")
                params.append(available_at.isoformat())
            else:
                clauses.append("available_at=NULL")
        if last_error is not _UNSET:
            if last_error is not None:
                clauses.append("last_error=?")
                params.append(last_error)
            else:
                clauses.append("last_error=NULL")
        params.append(job_id)
        sql = f"UPDATE jobs_queue SET {', '.join(clauses)} WHERE id=?"
        self.conn.execute(sql, params)
        self.conn.commit()

    def delete_job(self, job_id: int) -> None:
        self.conn.execute("DELETE FROM jobs_queue WHERE id=?", (job_id,))
        self.conn.commit()

    def _job_from_row(self, row: sqlite3.Row) -> JobRecord:
        payload_raw = row["payload"]
        try:
            payload = json.loads(payload_raw) if payload_raw else {}
        except json.JSONDecodeError:
            payload = {}
        available_at = (
            datetime.fromisoformat(row["available_at"])
            if row["available_at"]
            else None
        )
        created_at = datetime.fromisoformat(row["created_at"])
        updated_at = datetime.fromisoformat(row["updated_at"])
        return JobRecord(
            id=row["id"],
            name=row["name"],
            payload=payload if isinstance(payload, dict) else {},
            status=row["status"],
            attempts=row["attempts"] or 0,
            available_at=available_at,
            last_error=row["last_error"],
            created_at=created_at,
            updated_at=updated_at,
        )


def create_device(
    conn: sqlite3.Connection,
    *,
    device_id: str,
    user_id: int,
    name: str,
    secret: str,
) -> None:
    """Insert or update a device record."""

    if not secret:
        raise ValueError("Device secret must be a non-empty string")
    now = datetime.utcnow().isoformat()
    conn.execute(
        """
        INSERT INTO devices (id, user_id, name, secret, created_at, last_seen_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            user_id=excluded.user_id,
            name=excluded.name,
            secret=excluded.secret,
            last_seen_at=excluded.last_seen_at,
            revoked_at=NULL
        """,
        (device_id, user_id, name, secret, now, now),
    )


def list_user_devices(
    conn: sqlite3.Connection,
    *,
    user_id: int,
) -> list[dict[str, Any]]:
    """Return devices attached to the user ordered by creation time."""

    rows = conn.execute(
        """
        SELECT id, name, created_at, last_seen_at, revoked_at
        FROM devices
        WHERE user_id=?
        ORDER BY created_at DESC, id
        """,
        (user_id,),
    ).fetchall()
    devices: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, sqlite3.Row):
            entry = {
                "id": str(row["id"]),
                "name": str(row["name"] or ""),
                "created_at": str(row["created_at"] or ""),
                "last_seen_at": str(row["last_seen_at"] or ""),
                "revoked_at": str(row["revoked_at"] or ""),
            }
        else:
            created_at = row[2] if len(row) > 2 else None
            last_seen_at = row[3] if len(row) > 3 else None
            revoked_at = row[4] if len(row) > 4 else None
            entry = {
                "id": str(row[0]),
                "name": str(row[1] or ""),
                "created_at": str(created_at or ""),
                "last_seen_at": str(last_seen_at or ""),
                "revoked_at": str(revoked_at or ""),
            }
        devices.append(entry)
    return devices


def revoke_device(
    conn: sqlite3.Connection,
    *,
    device_id: str,
    expected_user_id: int | None = None,
) -> bool:
    """Mark the device secret as revoked."""

    if expected_user_id is not None:
        row = conn.execute(
            "SELECT user_id FROM devices WHERE id=?", (device_id,)
        ).fetchone()
        if not row:
            return False
        if isinstance(row, sqlite3.Row):
            owner = int(row["user_id"])
        else:
            owner = int(row[0])
        if owner != expected_user_id:
            return False
    now = datetime.utcnow().isoformat()
    updated = conn.execute(
        """
        UPDATE devices
        SET revoked_at=?
        WHERE id=? AND revoked_at IS NULL
        """,
        (now, device_id),
    )
    return updated.rowcount > 0


def rotate_device_secret(
    conn: sqlite3.Connection,
    *,
    device_id: str,
    expected_user_id: int | None = None,
) -> tuple[str, int, str] | None:
    """Generate a new secret for the device and clear revoked status."""

    row = conn.execute(
        "SELECT user_id, name FROM devices WHERE id=?",
        (device_id,),
    ).fetchone()
    if not row:
        return None
    if isinstance(row, sqlite3.Row):
        user_id = int(row["user_id"])
        name = str(row["name"] or "")
    else:
        user_id = int(row[0])
        name = str(row[1] or "")
    if expected_user_id is not None and user_id != expected_user_id:
        return None
    secret = secrets.token_hex(32)
    create_device(
        conn,
        device_id=device_id,
        user_id=user_id,
        name=name,
        secret=secret,
    )
    return secret, user_id, name


def get_device(conn: sqlite3.Connection, *, device_id: str) -> sqlite3.Row | None:
    """Return a device row or None if missing."""

    cur = conn.execute("SELECT * FROM devices WHERE id=?", (device_id,))
    return cur.fetchone()


def get_device_secret(
    conn: sqlite3.Connection, *, device_id: str
) -> tuple[str, str | None] | None:
    """Return the shared secret and revoked_at timestamp for a device."""

    cur = conn.execute(
        "SELECT secret, revoked_at FROM devices WHERE id=?",
        (device_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    if isinstance(row, sqlite3.Row):
        secret = str(row["secret"])
        revoked = str(row["revoked_at"]) if row["revoked_at"] else None
    else:
        secret = str(row[0])
        revoked = str(row[1]) if row[1] else None
    return secret, revoked


def register_nonce(
    conn: sqlite3.Connection,
    *,
    device_id: str,
    nonce: str,
    ttl_seconds: int = NONCE_TTL_SECONDS,
) -> bool:
    """Persist a nonce for anti-replay checks.

    Returns False if the nonce already exists for the device within the TTL window.
    """

    now = datetime.utcnow()
    now_iso = now.isoformat()
    cur = conn.execute(
        """
        SELECT 1 FROM nonces
        WHERE device_id=? AND value=? AND expires_at>?""",
        (device_id, nonce, now_iso),
    )
    if cur.fetchone():
        return False

    expires_at = (now + timedelta(seconds=max(ttl_seconds, 0))).isoformat()
    conn.execute(
        """
        INSERT INTO nonces (id, device_id, value, created_at, expires_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (secrets.token_hex(8), device_id, nonce, now_iso, expires_at),
    )
    conn.commit()
    return True


def touch_device(conn: sqlite3.Connection, *, device_id: str) -> bool:
    """Update the last_seen_at timestamp for a device."""

    now = datetime.utcnow().isoformat()
    cur = conn.execute(
        "UPDATE devices SET last_seen_at=? WHERE id=?",
        (now, device_id),
    )
    return cur.rowcount > 0


def create_pairing_token(
    conn: sqlite3.Connection,
    *,
    code: str,
    user_id: int,
    device_name: str,
    ttl_sec: int = PAIRING_TOKEN_TTL_SECONDS,
) -> None:
    """Create or replace a pairing token for attaching a device."""

    now = datetime.utcnow()
    ttl = max(int(ttl_sec), 0)
    expires_at = (now + timedelta(seconds=ttl)).isoformat()
    conn.execute(
        """
        INSERT INTO pairing_tokens (code, user_id, device_name, created_at, expires_at, used_at)
        VALUES (?, ?, ?, ?, ?, NULL)
        ON CONFLICT(code) DO UPDATE SET
            user_id=excluded.user_id,
            device_name=excluded.device_name,
            created_at=excluded.created_at,
            expires_at=excluded.expires_at,
            used_at=NULL
        """,
        (code, user_id, device_name, now.isoformat(), expires_at),
    )


def consume_pairing_token(
    conn: sqlite3.Connection,
    *,
    code: str,
) -> tuple[int, str] | None:
    """Mark a pairing token as used and return its payload."""

    cur = conn.execute(
        "SELECT user_id, device_name, expires_at, used_at FROM pairing_tokens WHERE code=?",
        (code,),
    )
    row = cur.fetchone()
    if not row:
        return None
    if isinstance(row, sqlite3.Row):
        user_id = int(row["user_id"])
        device_name = str(row["device_name"])
        expires_at_raw = row["expires_at"]
        used_at_raw = row["used_at"]
    else:
        user_id = int(row[0])
        device_name = str(row[1])
        expires_at_raw = row[2]
        used_at_raw = row[3]
    if used_at_raw:
        return None
    now = datetime.utcnow()
    try:
        expires_at = datetime.fromisoformat(str(expires_at_raw))
    except ValueError:
        expires_at = now
    if expires_at <= now:
        return None
    updated = conn.execute(
        """
        UPDATE pairing_tokens
        SET used_at=?
        WHERE code=? AND used_at IS NULL AND expires_at>?
        """,
        (now.isoformat(), code, now.isoformat()),
    )
    if updated.rowcount == 0:
        return None
    return user_id, device_name


def insert_upload(
    conn: sqlite3.Connection,
    *,
    id: str,
    device_id: str,
    idempotency_key: str,
    file_ref: str | None = None,
    source: str = "mobile",
    gps_redacted_by_client: bool = False,
) -> str:
    """Insert an upload row respecting idempotency."""

    now = datetime.utcnow().isoformat()
    gps_flag = 1 if gps_redacted_by_client else 0
    try:
        conn.execute(
            """
            INSERT INTO uploads (
                id,
                device_id,
                idempotency_key,
                status,
                error,
                file_ref,
                asset_id,
                source,
                gps_redacted_by_client,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, 'queued', NULL, ?, NULL, ?, ?, ?, ?)
            """,
            (id, device_id, idempotency_key, file_ref, source, gps_flag, now, now),
        )
        return id
    except sqlite3.IntegrityError as exc:
        message = str(exc)
        if (
            "uploads.device_id" in message and "idempotency_key" in message
        ) or "uq_uploads_device_idempotency" in message:
            cur = conn.execute(
                "SELECT id FROM uploads WHERE device_id=? AND idempotency_key=?",
                (device_id, idempotency_key),
            )
            existing = cur.fetchone()
            if not existing:
                raise
            if isinstance(existing, sqlite3.Row):
                return str(existing["id"])
            return str(existing[0])
        raise


def get_upload(
    conn: sqlite3.Connection,
    *,
    device_id: str,
    upload_id: str,
) -> dict[str, str | None] | None:
    cur = conn.execute(
        """
        SELECT id, status, error, asset_id
        FROM uploads
        WHERE id=? AND device_id=?
        """,
        (upload_id, device_id),
    )
    row = cur.fetchone()
    if not row:
        return None
    if isinstance(row, sqlite3.Row):
        error_value = row["error"]
        error_text = str(error_value) if error_value is not None else None
        return {
            "id": str(row["id"]),
            "status": str(row["status"]),
            "error": error_text,
            "asset_id": str(row["asset_id"]) if row["asset_id"] is not None else None,
        }
    error_value = row[2]
    error_text = str(error_value) if error_value is not None else None
    asset_value = row[3] if len(row) > 3 else None
    asset_text = str(asset_value) if asset_value is not None else None
    return {
        "id": str(row[0]),
        "status": str(row[1]),
        "error": error_text,
        "asset_id": asset_text,
    }


def get_recognition_channel_id(conn: sqlite3.Connection) -> int | None:
    """Return the configured recognition channel identifier or ``None`` if missing."""

    cur = conn.execute("SELECT channel_id FROM recognition_channel LIMIT 1")
    row = cur.fetchone()
    if not row:
        return None
    if isinstance(row, sqlite3.Row):
        value = row["channel_id"]
    else:
        value = row[0]
    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        return None


def get_asset_channel_id(conn: sqlite3.Connection) -> int | None:
    """Return the configured asset channel identifier or ``None`` if missing."""

    cur = conn.execute("SELECT channel_id FROM asset_channel LIMIT 1")
    row = cur.fetchone()
    if not row:
        return None
    if isinstance(row, sqlite3.Row):
        value = row["channel_id"]
    else:
        value = row[0]
    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        return None


def set_upload_status(
    conn: sqlite3.Connection,
    *,
    id: str,
    status: str,
    error: str | None = None,
) -> None:
    """Update the status of an upload enforcing valid transitions."""

    valid_statuses = {"queued", "processing", "failed", "done"}
    if status not in valid_statuses:
        raise ValueError(f"Unsupported upload status: {status}")
    cur = conn.execute("SELECT status FROM uploads WHERE id=?", (id,))
    row = cur.fetchone()
    if not row:
        raise ValueError(f"Upload {id} not found")
    if isinstance(row, sqlite3.Row):
        current = str(row["status"])
    else:
        current = str(row[0])
    if current == status:
        if status in {"done", "failed"}:
            return
        return
    transitions = {
        "queued": {"processing", "failed"},
        "processing": {"done", "failed"},
        "done": {"processing"},
        "failed": {"processing"},
    }
    allowed = transitions.get(current)
    if not allowed or status not in allowed:
        raise ValueError(f"Invalid status transition {current!r} -> {status!r}")
    now = datetime.utcnow().isoformat()
    if status == "failed":
        conn.execute(
            "UPDATE uploads SET status=?, error=?, updated_at=? WHERE id=?",
            (status, error, now, id),
        )
    else:
        conn.execute(
            "UPDATE uploads SET status=?, error=NULL, updated_at=? WHERE id=?",
            (status, now, id),
        )


def link_upload_asset(
    conn: sqlite3.Connection,
    *,
    upload_id: str,
    asset_id: str,
) -> None:
    """Associate an upload row with a persisted asset."""

    cur = conn.execute("SELECT asset_id FROM uploads WHERE id=?", (upload_id,))
    row = cur.fetchone()
    if not row:
        raise ValueError(f"Upload {upload_id} not found")
    if isinstance(row, sqlite3.Row):
        existing = row["asset_id"]
    else:
        existing = row[0]
    if existing and str(existing) != asset_id:
        raise ValueError(f"Upload {upload_id} already linked to asset {existing}")
    now = datetime.utcnow().isoformat()
    conn.execute(
        "UPDATE uploads SET asset_id=?, updated_at=? WHERE id=?",
        (asset_id, now, upload_id),
    )


def fetch_upload_record(
    conn: sqlite3.Connection, *, upload_id: str
) -> dict[str, Any] | None:
    cur = conn.execute(
        """
        SELECT id, device_id, status, error, file_ref, asset_id, source, gps_redacted_by_client, created_at, updated_at
        FROM uploads
        WHERE id=?
        """,
        (upload_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    if isinstance(row, sqlite3.Row):
        return {
            "id": str(row["id"]),
            "device_id": str(row["device_id"]),
            "status": str(row["status"]),
            "error": row["error"],
            "file_ref": row["file_ref"],
            "asset_id": str(row["asset_id"]) if row["asset_id"] is not None else None,
            "source": str(row["source"]) if row["source"] is not None else None,
            "gps_redacted_by_client": bool(row["gps_redacted_by_client"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
    return {
        "id": str(row[0]),
        "device_id": str(row[1]),
        "status": str(row[2]),
        "error": row[3],
        "file_ref": row[4],
        "asset_id": str(row[5]) if row[5] is not None else None,
        "source": str(row[6]) if row[6] is not None else None,
        "gps_redacted_by_client": bool(row[7]),
        "created_at": row[8],
        "updated_at": row[9],
    }
