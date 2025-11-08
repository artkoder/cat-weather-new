from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from data_access import Asset, DataAccess
from sea_selection import infer_sky_visible


KALININGRAD_TZ = ZoneInfo("Europe/Kaliningrad")
DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "weather.db"


def _load_json(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logging.debug("Failed to parse JSON: %s", text[:80])
            return None
    return None


def _normalize_tags(values: Iterable[Any]) -> set[str]:
    normalized: set[str] = set()
    for value in values:
        text = str(value).strip().lower().replace(" ", "_")
        if text:
            normalized.add(text)
    return normalized


def _parse_exif_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    candidates = [str(value)]
    text = str(value)
    if ":" in text[:10]:
        candidates.append(text.replace(":", "-", 2))
    parsed: datetime | None = None
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            parsed = None
        if parsed is not None:
            break
    if parsed is None:
        formats = (
            "%Y:%m:%d %H:%M:%S",
            "%Y:%m:%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
        )
        for fmt in formats:
            try:
                parsed = datetime.strptime(text, fmt)
            except ValueError:
                continue
            else:
                break
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=KALININGRAD_TZ)
    return parsed.astimezone(KALININGRAD_TZ)
from zoneinfo import ZoneInfo


def _parse_offset(offset: str) -> timedelta:
    sign = -1 if offset.startswith("-") else 1
    hours, minutes = offset.lstrip("+-").split(":")
    return timedelta(minutes=sign * (int(hours) * 60 + int(minutes)))


def _next_weather_run(post_time: str, offset: str, reference: datetime | None = None) -> datetime:
    reference_dt = reference or datetime.utcnow()
    tz_delta = _parse_offset(offset)
    local_ref = reference_dt + tz_delta
    hour, minute = map(int, post_time.split(":"))
    candidate = local_ref.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= local_ref:
        candidate += timedelta(days=1)
    return candidate - tz_delta


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    return cur.fetchone() is not None


def migrate_weather_publish_channels(
    conn: sqlite3.Connection, *, tz_offset: str | None = None, reference: datetime | None = None
) -> bool:
    """Backfill legacy weather channels into weather_jobs.

    Returns True if any migration was performed (rows moved or table dropped).
    """

    if not _table_exists(conn, "weather_publish_channels"):
        return False

    rows = conn.execute(
        "SELECT channel_id, post_time, last_published_at FROM weather_publish_channels"
    ).fetchall()

    tz = tz_offset or os.getenv("TZ_OFFSET", "+00:00")
    now = datetime.utcnow().isoformat()
    for row in rows:
        channel_id, post_time, last_published = row
        if not post_time:
            continue
        run_at = _next_weather_run(post_time, tz, reference=reference).isoformat()
        created_at = last_published or now
        conn.execute(
            """
            INSERT INTO weather_jobs (
                channel_id, post_time, run_at, last_run_at, failures, last_error, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 0, NULL, ?, ?)
            ON CONFLICT(channel_id) DO UPDATE SET
                post_time=excluded.post_time,
                run_at=excluded.run_at,
                last_run_at=COALESCE(excluded.last_run_at, weather_jobs.last_run_at),
                failures=0,
                last_error=NULL,
                updated_at=excluded.updated_at
            """,
            (
                channel_id,
                post_time,
                run_at,
                last_published,
                created_at,
                now,
            ),
        )

    conn.execute("DROP TABLE IF EXISTS weather_publish_channels")
    return True


def fill_photo_doy(conn: sqlite3.Connection, *, dry_run: bool = False) -> dict[str, int]:
    stats = {"updated": 0, "already_set": 0, "unknown": 0}
    original_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, photo_doy, shot_doy, shot_at_utc, exif_json FROM assets"
        ).fetchall()
    finally:
        conn.row_factory = original_factory
    for row in rows:
        asset_id = row["id"]
        current_doy = Asset._to_int(row["photo_doy"])
        if current_doy is not None and 1 <= current_doy <= 366:
            stats["already_set"] += 1
            continue

        new_doy: int | None = None
        exif_data = _load_json(row["exif_json"])
        if isinstance(exif_data, dict):
            for key in ("DateTimeOriginal", "DateTime", "DateTimeDigitized"):
                candidate = _parse_exif_datetime(exif_data.get(key))
                if candidate is not None:
                    new_doy = candidate.timetuple().tm_yday
                    break
        if new_doy is None and row["shot_at_utc"] is not None:
            try:
                shot_dt = datetime.fromtimestamp(int(row["shot_at_utc"]), tz=timezone.utc)
            except (TypeError, ValueError, OSError):
                shot_dt = None
            if shot_dt is not None:
                new_doy = shot_dt.astimezone(KALININGRAD_TZ).timetuple().tm_yday
        if new_doy is None:
            shot_doy = Asset._to_int(row["shot_doy"])
            if shot_doy is not None and 1 <= shot_doy <= 366:
                new_doy = shot_doy

        if new_doy is None:
            stats["unknown"] += 1
            continue

        stats["updated"] += 1
        if not dry_run:
            conn.execute(
                "UPDATE assets SET photo_doy=? WHERE id=?",
                (int(new_doy), str(asset_id)),
            )

    if not dry_run:
        conn.commit()
    return stats


def recalc_sky_visible(conn: sqlite3.Connection, *, dry_run: bool = False) -> dict[str, int]:
    data = DataAccess(conn)
    stats = {"updated": 0, "unchanged": 0}
    for asset in data.iter_assets():
        vision = asset.vision_results or {}
        tags_raw = vision.get("tags")
        normalized_tags = _normalize_tags(tags_raw or []) if tags_raw else set()
        inferred = infer_sky_visible(normalized_tags)
        if inferred is None:
            sky_raw = vision.get("sky_visible")
            if isinstance(sky_raw, bool):
                inferred = sky_raw
        if inferred is True:
            normalized_value = "true"
        elif inferred is False:
            normalized_value = "false"
        else:
            normalized_value = "unknown"
        current_value = asset.sky_visible_hint or "unknown"
        if normalized_value == current_value:
            stats["unchanged"] += 1
            continue
        stats["updated"] += 1
        if not dry_run:
            conn.execute(
                "UPDATE assets SET sky_visible=? WHERE id=?",
                (normalized_value, str(asset.id)),
            )
    if not dry_run:
        conn.commit()
    return stats


def backfill_photo_wave(conn: sqlite3.Connection, *, dry_run: bool = False) -> dict[str, int]:
    data = DataAccess(conn)
    stats = {"updated": 0, "unchanged": 0, "missing": 0}
    for asset in data.iter_assets():
        if asset.photo_wave is not None:
            stats["unchanged"] += 1
            continue
        vision = asset.vision_results or {}
        wave_raw = vision.get("sea_wave_score")
        if isinstance(wave_raw, dict):
            wave_raw = wave_raw.get("value")
        wave_val = Asset._to_float(wave_raw)
        if wave_val is None:
            stats["missing"] += 1
            continue
        stats["updated"] += 1
        if not dry_run:
            conn.execute(
                "UPDATE assets SET photo_wave=? WHERE id=?",
                (float(wave_val), str(asset.id)),
            )
    if not dry_run:
        conn.commit()
    return stats


def _ensure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Weather photo metadata migrations")
    parser.add_argument(
        "--database",
        default=str(DEFAULT_DB_PATH),
        help="Path to the SQLite database (default: %(default)s)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes")
    parser.add_argument("--fill-doy", action="store_true", help="Backfill assets.photo_doy")
    parser.add_argument(
        "--recalc-sky-visible",
        action="store_true",
        dest="recalc_sky_visible_flag",
        help="Recompute assets.sky_visible from vision tags",
    )
    parser.add_argument(
        "--backfill-wave",
        action="store_true",
        help="Backfill assets.photo_wave from vision results",
    )
    args = parser.parse_args()

    if not any((args.fill_doy, args.recalc_sky_visible_flag, args.backfill_wave)):
        parser.error("No action specified. Use --fill-doy, --recalc-sky-visible, or --backfill-wave.")

    _ensure_logging()

    db_path = Path(args.database)
    if not db_path.exists():
        logging.error("Database %s does not exist", db_path)
        raise SystemExit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        if args.fill_doy:
            stats = fill_photo_doy(conn, dry_run=args.dry_run)
            logging.info(
                "fill_photo_doy updated=%s already_set=%s unknown=%s",
                stats["updated"],
                stats["already_set"],
                stats["unknown"],
            )
        if args.recalc_sky_visible_flag:
            stats = recalc_sky_visible(conn, dry_run=args.dry_run)
            logging.info(
                "recalc_sky_visible updated=%s unchanged=%s",
                stats["updated"],
                stats["unchanged"],
            )
        if args.backfill_wave:
            stats = backfill_photo_wave(conn, dry_run=args.dry_run)
            logging.info(
                "backfill_photo_wave updated=%s unchanged=%s missing=%s",
                stats["updated"],
                stats["unchanged"],
                stats["missing"],
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
