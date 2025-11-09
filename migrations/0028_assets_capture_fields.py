"""Add capture metadata columns to assets and backfill values."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - zoneinfo may be missing in some environments
    ZoneInfo = None  # type: ignore[assignment]


LOCAL_TZ = ZoneInfo("Europe/Kaliningrad") if ZoneInfo else timezone(timedelta(hours=2))


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM pragma_table_info(?) WHERE name=?",
        (table, column),
    ).fetchone()
    if row is not None:
        return True
    cursor = conn.execute(f"PRAGMA table_info('{table}')")
    for entry in cursor.fetchall():
        name = entry[1] if len(entry) > 1 else entry["name"]
        if str(name) == column:
            return True
    return False


def _index_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _safe_json_loads(value: str | None) -> dict[str, object] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _extract_capture_datetime(exif: dict[str, object] | None) -> str | None:
    if not exif:
        return None
    for key in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
        raw = exif.get(key)
        if not raw:
            continue
        text = str(raw).strip()
        if not text:
            continue
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(text, fmt)
            except ValueError:
                continue
            return dt.replace(tzinfo=timezone.utc).isoformat()
        return text
    return None


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                parsed = datetime.strptime(text, fmt)
            except ValueError:
                continue
            return parsed.replace(tzinfo=timezone.utc)
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _determine_daypart(dt: datetime) -> str:
    local_dt = dt.astimezone(LOCAL_TZ)
    hour = local_dt.hour
    if 5 <= hour < 11:
        return "morning"
    if 11 <= hour < 17:
        return "day"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def _compute_capture_fields(
    *,
    exif: dict[str, object] | None,
    shot_at_utc: int | None,
    shot_doy: int | None,
    existing_captured_at: str | None,
    existing_doy: int | None,
    existing_daypart: str | None,
) -> tuple[str | None, int | None, str | None]:
    captured_at = _extract_capture_datetime(exif)
    parsed_dt = _parse_datetime(captured_at)
    if parsed_dt is None and shot_at_utc is not None:
        try:
            parsed_dt = datetime.fromtimestamp(int(shot_at_utc), tz=timezone.utc)
        except (OSError, OverflowError, ValueError, TypeError):
            parsed_dt = None
        else:
            if captured_at is None:
                captured_at = parsed_dt.isoformat()
    if captured_at is None:
        captured_at = existing_captured_at
    if parsed_dt is None and captured_at:
        parsed_dt = _parse_datetime(captured_at)

    doy_value: int | None = None
    daypart_value: str | None = None
    if parsed_dt is not None:
        local_dt = parsed_dt.astimezone(LOCAL_TZ)
        doy_value = local_dt.timetuple().tm_yday
        daypart_value = _determine_daypart(parsed_dt)
    if doy_value is None and shot_doy is not None:
        try:
            candidate = int(shot_doy)
        except (TypeError, ValueError):
            candidate = None
        else:
            if 1 <= candidate <= 366:
                doy_value = candidate
    if doy_value is None and isinstance(existing_doy, int):
        if 1 <= existing_doy <= 366:
            doy_value = existing_doy
    if daypart_value is None and existing_daypart:
        text = existing_daypart.strip().lower()
        if text:
            daypart_value = text
    return captured_at, doy_value, daypart_value


def run(conn: sqlite3.Connection) -> None:
    if not _column_exists(conn, "assets", "captured_at"):
        conn.execute("ALTER TABLE assets ADD COLUMN captured_at TEXT")
    if not _column_exists(conn, "assets", "doy"):
        conn.execute("ALTER TABLE assets ADD COLUMN doy INTEGER")
    if not _column_exists(conn, "assets", "daypart"):
        conn.execute("ALTER TABLE assets ADD COLUMN daypart TEXT")

    if _column_exists(conn, "assets", "doy") and not _index_exists(conn, "idx_assets_doy"):
        conn.execute("CREATE INDEX IF NOT EXISTS idx_assets_doy ON assets(doy)")
    if _column_exists(conn, "assets", "daypart") and not _index_exists(conn, "idx_assets_daypart"):
        conn.execute("CREATE INDEX IF NOT EXISTS idx_assets_daypart ON assets(daypart)")
    if _column_exists(conn, "assets", "vision_weather_tag") and not _index_exists(
        conn, "idx_assets_vision_weather"
    ):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_assets_vision_weather ON assets(vision_weather_tag)"
        )
    if _column_exists(conn, "assets", "sea_wave_score") and not _index_exists(
        conn, "idx_assets_sea_wave_score"
    ):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_assets_sea_wave_score ON assets(sea_wave_score)"
        )

    rows = conn.execute(
        "SELECT id, exif_json, shot_at_utc, shot_doy, captured_at, doy, daypart FROM assets"
    ).fetchall()
    for row in rows:
        exif = _safe_json_loads(row["exif_json"])
        shot_at = row["shot_at_utc"]
        shot_doy = row["shot_doy"]
        existing_captured = row["captured_at"]
        existing_doy = row["doy"]
        existing_daypart = row["daypart"]
        capture_iso, capture_doy, capture_daypart = _compute_capture_fields(
            exif=exif,
            shot_at_utc=shot_at,
            shot_doy=shot_doy,
            existing_captured_at=existing_captured if existing_captured else None,
            existing_doy=existing_doy if isinstance(existing_doy, int) else None,
            existing_daypart=existing_daypart if isinstance(existing_daypart, str) else None,
        )
        assignments: list[str] = []
        params: list[object] = []
        if capture_iso and not existing_captured:
            assignments.append("captured_at=?")
            params.append(capture_iso)
        if capture_doy is not None and (existing_doy is None or existing_doy != capture_doy):
            assignments.append("doy=?")
            params.append(int(capture_doy))
        if capture_daypart and not existing_daypart:
            assignments.append("daypart=?")
            params.append(capture_daypart)
        if assignments:
            params.append(row["id"])
            conn.execute(
                f"UPDATE assets SET {', '.join(assignments)} WHERE id=?",
                params,
            )
    conn.commit()
