from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

try:  # pragma: no cover - fallback for limited environments
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]

from data_access import Asset, DataAccess

logger = logging.getLogger(__name__)

_MIN_SCORE = 7
_CANDIDATE_LIMIT = 200
_FRESHNESS_WINDOW = timedelta(days=3)
_SEASON_WINDOW_DAYS = 30
_REPEAT_GUARD_WINDOW = timedelta(days=7)
_DAYS_IN_YEAR = 366
_MIN_DATETIME = datetime(1970, 1, 1, tzinfo=timezone.utc)

if ZoneInfo is not None:
    KALININGRAD_TZ = ZoneInfo("Europe/Kaliningrad")
else:  # pragma: no cover - fallback on systems without TZ database
    KALININGRAD_TZ = timezone(timedelta(hours=2))


@dataclass
class _Candidate:
    asset_id: str
    score: int
    captured_raw: str | None
    created_raw: str
    photo_doy: int | None
    city: str | None
    region: str | None
    last_used_raw: str | None
    captured_dt: datetime | None
    created_dt: datetime
    last_used_dt: datetime | None

    @property
    def effective_dt(self) -> datetime:
        return self.captured_dt or self.created_dt or _MIN_DATETIME


def select_postcard_asset(data: DataAccess, *, now: datetime) -> Asset | None:
    """Choose the best postcard asset based on score, freshness and season."""

    now_utc = _ensure_aware(now)
    now_local = now_utc.astimezone(KALININGRAD_TZ)
    doy_now = now_local.timetuple().tm_yday

    max_score_row = data.conn.execute(
        """
        SELECT MAX(postcard_score) AS max_score
        FROM assets
        WHERE postcard_score IS NOT NULL AND postcard_score >= ?
        """,
        (_MIN_SCORE,),
    ).fetchone()
    max_score = (
        int(max_score_row["max_score"])
        if max_score_row and max_score_row["max_score"] is not None
        else None
    )
    if max_score is None:
        logger.info("POSTCARD_RUBRIC score_pool empty threshold=%s", _MIN_SCORE)
        return None

    rows = data.conn.execute(
        """
        SELECT
            id,
            postcard_score,
            captured_at,
            created_at,
            photo_doy,
            json_extract(payload_json, '$.last_used_at') AS last_used_at,
            json_extract(payload_json, '$.city') AS city,
            json_extract(payload_json, '$.region') AS region
        FROM assets
        WHERE postcard_score = ?
        ORDER BY captured_at DESC, created_at DESC, id ASC
        LIMIT ?
        """,
        (max_score, _CANDIDATE_LIMIT),
    ).fetchall()
    candidates = [_build_candidate(row) for row in rows]
    if not candidates:
        logger.info(
            "POSTCARD_RUBRIC score_pool empty_rows max_score=%s threshold=%s",
            max_score,
            _MIN_SCORE,
        )
        return None

    logger.info(
        "POSTCARD_RUBRIC score_pool threshold=%s max_score=%s candidate_count=%s limit=%s",
        _MIN_SCORE,
        max_score,
        len(candidates),
        _CANDIDATE_LIMIT,
    )

    fresh_threshold = now_utc - _FRESHNESS_WINDOW
    fresh_candidates = [c for c in candidates if c.effective_dt >= fresh_threshold]
    logger.info(
        "POSTCARD_RUBRIC freshness window_days=%s fresh=%s stale=%s",
        _FRESHNESS_WINDOW.days,
        len(fresh_candidates),
        len(candidates) - len(fresh_candidates),
    )
    working_set = fresh_candidates if fresh_candidates else candidates
    if not fresh_candidates:
        logger.info("POSTCARD_RUBRIC freshness fallback_to_stale count=%s", len(working_set))

    in_season = [c for c in working_set if _matches_season(c.photo_doy, doy_now)]
    season_candidates = in_season if in_season else working_set
    logger.info(
        "POSTCARD_RUBRIC season doy_now=%s window_days=%s in_window=%s out_window=%s fallback=%s",
        doy_now,
        _SEASON_WINDOW_DAYS,
        len(in_season),
        len(working_set) - len(in_season),
        not in_season,
    )

    repeat_threshold = now_utc - _REPEAT_GUARD_WINDOW
    non_recent = [
        c for c in season_candidates if not c.last_used_dt or c.last_used_dt < repeat_threshold
    ]
    repeat_candidates = non_recent if non_recent else season_candidates
    logger.info(
        "POSTCARD_RUBRIC repeat_guard window_days=%s skipped=%s fallback=%s",
        _REPEAT_GUARD_WINDOW.days,
        len(season_candidates) - len(non_recent),
        not non_recent,
    )

    if not repeat_candidates:
        logger.info("POSTCARD_RUBRIC selection_empty max_score=%s", max_score)
        return None

    repeat_candidates.sort(key=_sort_key)
    best = repeat_candidates[0]

    freshness_hours: float | None = None
    try:
        delta = now_utc - best.effective_dt
        freshness_hours = max(delta.total_seconds() / 3600.0, 0.0)
    except Exception:  # pragma: no cover - defensive against invalid datetimes
        freshness_hours = None

    recently_used = bool(best.last_used_dt and best.last_used_dt >= repeat_threshold)
    logger.info(
        "POSTCARD_RUBRIC selected asset_id=%s score=%s captured_at=%s photo_doy=%s city=%s region=%s "
        "freshness_hours=%s reused_recently=%s",
        best.asset_id,
        best.score,
        best.captured_raw or best.created_raw,
        best.photo_doy,
        best.city or "",
        best.region or "",
        f"{freshness_hours:.2f}" if freshness_hours is not None else "unknown",
        recently_used,
    )

    asset = data.get_asset(best.asset_id)
    if asset is None:
        logger.warning("POSTCARD_RUBRIC asset_missing asset_id=%s", best.asset_id)
    return asset


def _ensure_aware(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _build_candidate(row: Any) -> _Candidate:
    asset_id = str(row["id"])
    score = int(row["postcard_score"])
    captured_raw = _to_str(row["captured_at"])
    created_raw = _to_str(row["created_at"]) or _MIN_DATETIME.isoformat()
    photo_doy = _normalize_doy(row["photo_doy"])
    city = _normalize_optional_str(row["city"])
    region = _normalize_optional_str(row["region"])
    last_used_raw = _to_str(row["last_used_at"])
    captured_dt = _parse_iso(captured_raw)
    created_dt = _parse_iso(created_raw) or _MIN_DATETIME
    last_used_dt = _parse_iso(last_used_raw)
    return _Candidate(
        asset_id=asset_id,
        score=score,
        captured_raw=captured_raw,
        created_raw=created_raw,
        photo_doy=photo_doy,
        city=city,
        region=region,
        last_used_raw=last_used_raw,
        captured_dt=captured_dt,
        created_dt=created_dt,
        last_used_dt=last_used_dt,
    )


def _matches_season(photo_doy: int | None, now_doy: int) -> bool:
    if photo_doy is None:
        return False
    delta = abs(photo_doy - now_doy)
    wrapped = _DAYS_IN_YEAR - delta
    return min(delta, wrapped) <= _SEASON_WINDOW_DAYS


def _sort_key(candidate: _Candidate) -> tuple[float, str]:
    reference = candidate.effective_dt
    timestamp = reference.timestamp() if reference else float("-inf")
    return (-timestamp, candidate.asset_id)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_doy(raw: Any) -> int | None:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    if 1 <= value <= _DAYS_IN_YEAR:
        return value
    return None


def _normalize_optional_str(value: Any) -> str | None:
    text = _to_str(value)
    if text:
        return text
    return None


def _to_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
