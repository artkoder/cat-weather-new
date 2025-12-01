from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
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
_TAG_STRICT_WINDOW = timedelta(days=3)
_TAG_SOFT_WINDOW = timedelta(days=7)
_TAG_STRICT_MAX_OVERLAP = 1
_TAG_SOFT_MAX_OVERLAP = 0
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
    tags: frozenset[str]
    last_used_history: tuple[datetime, ...]
    captured_dt: datetime | None
    created_dt: datetime
    last_used_dt: datetime | None

    @property
    def effective_dt(self) -> datetime:
        return self.captured_dt or self.created_dt or _MIN_DATETIME


def _time_random_choice(candidates: Sequence[_Candidate], moment: datetime) -> _Candidate:
    if not candidates:
        raise ValueError("Candidate pool must not be empty")
    moment_aware = _ensure_aware(moment)
    timestamp_key = f"{moment_aware.timestamp():.6f}"
    iso_key = moment_aware.isoformat()
    payload = f"{timestamp_key}-{iso_key}-{len(candidates)}".encode()
    digest = hashlib.sha256(payload).digest()
    index = int.from_bytes(digest[:8], "big") % len(candidates)
    return candidates[index]


def select_postcard_asset(
    data: DataAccess,
    *,
    now: datetime,
    test: bool = False,
) -> Asset | None:
    """Choose the best postcard asset based on score, freshness and season.

    When ``test`` is True, selection ignores repeat guard checks and randomizes across the
    available pool to better sample inventory without burning real candidates.
    """

    now_utc = _ensure_aware(now)
    now_local = now_utc.astimezone(KALININGRAD_TZ)
    doy_now = now_local.timetuple().tm_yday

    score_rows = data.conn.execute(
        """
        SELECT DISTINCT postcard_score AS score
        FROM assets
        WHERE postcard_score IS NOT NULL AND postcard_score >= ?
        ORDER BY score DESC
        """,
        (_MIN_SCORE,),
    ).fetchall()
    scores = [int(row["score"]) for row in score_rows if row["score"] is not None]
    if not scores:
        logger.info("POSTCARD_RUBRIC score_pool empty threshold=%s", _MIN_SCORE)
        return None

    best: _Candidate | None = None
    selection_notes: list[str] = []
    repeat_threshold = now_utc - _REPEAT_GUARD_WINDOW

    for score in scores:
        rows = data.conn.execute(
            """
            SELECT
                id,
                postcard_score,
                captured_at,
                created_at,
                photo_doy,
                json_extract(payload_json, '$.tags') AS tags,
                json_extract(payload_json, '$.last_used_at') AS last_used_at,
                json_extract(payload_json, '$.city') AS city,
                json_extract(payload_json, '$.region') AS region
            FROM assets
            WHERE postcard_score = ?
            ORDER BY captured_at DESC, created_at DESC, id ASC
            LIMIT ?
            """,
            (score, _CANDIDATE_LIMIT),
        ).fetchall()
        candidates = [_build_candidate(row) for row in rows]
        if not candidates:
            logger.info(
                "POSTCARD_RUBRIC score_pool empty_rows current_score=%s threshold=%s",
                score,
                _MIN_SCORE,
            )
            continue

        logger.info(
            "POSTCARD_RUBRIC score_pool threshold=%s current_score=%s candidate_count=%s limit=%s",
            _MIN_SCORE,
            score,
            len(candidates),
            _CANDIDATE_LIMIT,
        )

        chosen, notes = _select_candidate(
            candidates=candidates,
            now_utc=now_utc,
            doy_now=doy_now,
            repeat_threshold=repeat_threshold,
            test=test,
        )
        if chosen:
            best = chosen
            selection_notes = notes
            if score != scores[0]:
                selection_notes.append("score_fallback")
            break

    if best is None:
        logger.info(
            "POSTCARD_RUBRIC selection_empty max_score=%s min_score=%s",
            scores[0],
            _MIN_SCORE,
        )
        return None

    selection_notes_str = ",".join(selection_notes) if selection_notes else "direct"

    freshness_hours: float | None = None
    try:
        delta = now_utc - best.effective_dt
        freshness_hours = max(delta.total_seconds() / 3600.0, 0.0)
    except Exception:  # pragma: no cover - defensive against invalid datetimes
        freshness_hours = None

    season_match = _matches_season(best.photo_doy, doy_now)
    is_fresh_choice = bool(best.effective_dt and best.effective_dt >= now_utc - _FRESHNESS_WINDOW)
    captured_display = best.captured_raw or best.created_raw

    logger.info(
        "POSTCARD_RUBRIC selection asset_id=%s score=%s captured_at=%s created_at=%s photo_doy=%s doy_now=%s city=%s region=%s "
        "freshness_hours=%s reused_recently=%s season_match=%s fresh_window=%s notes=%s",
        best.asset_id,
        best.score,
        captured_display,
        best.created_raw,
        best.photo_doy,
        doy_now,
        best.city or "",
        best.region or "",
        f"{freshness_hours:.2f}" if freshness_hours is not None else "unknown",
        bool(best.last_used_dt and best.last_used_dt >= repeat_threshold),
        season_match,
        is_fresh_choice,
        selection_notes_str,
    )

    asset = data.get_asset(best.asset_id)
    if asset is None:
        logger.warning("POSTCARD_RUBRIC asset_missing asset_id=%s", best.asset_id)
    return asset


def _select_candidate(
    *,
    candidates: Sequence[_Candidate],
    now_utc: datetime,
    doy_now: int,
    repeat_threshold: datetime,
    test: bool,
) -> tuple[_Candidate | None, list[str]]:
    strict_tag_cutoff = now_utc - _TAG_STRICT_WINDOW
    soft_tag_cutoff = now_utc - _TAG_SOFT_WINDOW
    strict_recent_tags = _collect_recent_tags(candidates, strict_tag_cutoff)
    soft_recent_tags = _collect_recent_tags(candidates, soft_tag_cutoff)

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

    non_recent = [
        c for c in season_candidates if not c.last_used_dt or c.last_used_dt < repeat_threshold
    ]
    logger.info(
        "POSTCARD_RUBRIC repeat_guard window_days=%s skipped=%s fallback=%s",
        _REPEAT_GUARD_WINDOW.days,
        len(season_candidates) - len(non_recent),
        not non_recent,
    )

    selection_notes: list[str] = []
    if not fresh_candidates:
        selection_notes.append("stale_pool")
    if not in_season:
        selection_notes.append("season_fallback")
    if not non_recent:
        selection_notes.append("repeat_guard_fallback")

    repeat_candidates = non_recent
    if not non_recent:
        expanded_non_recent = [
            c for c in working_set if not c.last_used_dt or c.last_used_dt < repeat_threshold
        ]
        if expanded_non_recent:
            repeat_candidates = expanded_non_recent
            selection_notes.append("repeat_guard_broadened_pool")
        elif not test:
            logger.info("POSTCARD_RUBRIC repeat_guard_exhausted retry_lower_score=True")
            selection_notes.append("repeat_guard_exhausted")
            return None, selection_notes

    repeat_candidates = repeat_candidates if repeat_candidates else season_candidates

    tag_penalties: dict[str, int] = {}
    tag_filtered: list[_Candidate] = []
    skipped_tag_overlap = 0
    for candidate in repeat_candidates:
        strict_overlap = len(candidate.tags & strict_recent_tags)
        soft_overlap = len(candidate.tags & soft_recent_tags)
        if strict_overlap > _TAG_STRICT_MAX_OVERLAP:
            skipped_tag_overlap += 1
            continue
        penalty = max(soft_overlap - _TAG_SOFT_MAX_OVERLAP, 0)
        if penalty:
            tag_penalties[candidate.asset_id] = penalty
        tag_filtered.append(candidate)

    if skipped_tag_overlap:
        selection_notes.append("tag_overlap_filter")

    if not tag_filtered:
        selection_notes.append("tag_overlap_exhausted")
        return None, selection_notes

    if test:
        random_pool = tag_filtered if tag_filtered else season_candidates
        if not random_pool or (len(random_pool) < 2 and len(candidates) > len(random_pool)):
            random_pool = candidates
            selection_notes.append("test_random_full_pool")
        best = _time_random_choice(random_pool, now_utc)
        selection_notes.append("test_time_random_pool")
    else:
        tag_filtered.sort(key=lambda cand: _sort_key(cand, penalties=tag_penalties))
        best = tag_filtered[0]
        if tag_penalties.get(best.asset_id):
            selection_notes.append("tag_penalty")

    return best, selection_notes


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
    tags = _parse_tags(row["tags"])
    last_used_raw = _to_str(row["last_used_at"])
    last_used_history = _parse_last_used_history(row["last_used_at"])
    captured_dt = _parse_iso(captured_raw)
    created_dt = _parse_iso(created_raw) or _MIN_DATETIME
    last_used_dt = _latest_dt(last_used_history) or _parse_iso(last_used_raw)
    return _Candidate(
        asset_id=asset_id,
        score=score,
        captured_raw=captured_raw,
        created_raw=created_raw,
        photo_doy=photo_doy,
        city=city,
        region=region,
        last_used_raw=last_used_raw,
        tags=tags,
        last_used_history=last_used_history,
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


def _sort_key(candidate: _Candidate, *, penalties: dict[str, int]) -> tuple[int, float, str]:
    penalty = penalties.get(candidate.asset_id, 0)
    reference = candidate.effective_dt
    timestamp = reference.timestamp() if reference else float("-inf")
    return (penalty, -timestamp, candidate.asset_id)


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


def _parse_tags(raw: Any) -> frozenset[str]:
    if raw is None:
        return frozenset()
    parsed: Any = raw
    if isinstance(raw, str):
        raw_text = raw.strip()
        if not raw_text:
            return frozenset()
        try:
            parsed = json.loads(raw_text)
        except Exception:
            parsed = raw_text
    if isinstance(parsed, str):
        return frozenset({parsed})
    if isinstance(parsed, (list, tuple, set, frozenset)):
        normalized = {_to_str(item) for item in parsed}
        return frozenset(filter(None, normalized))
    return frozenset()


def _parse_last_used_history(raw: Any) -> tuple[datetime, ...]:
    if raw is None:
        return ()
    parsed: Any = raw
    if isinstance(raw, str):
        raw_text = raw.strip()
        if not raw_text:
            return ()
        try:
            parsed = json.loads(raw_text)
        except Exception:
            parsed = raw_text
    if isinstance(parsed, (list, tuple, set)):
        history = [_parse_iso(_to_str(item)) for item in parsed]
        return tuple(sorted(filter(None, history)))
    single = _parse_iso(_to_str(parsed))
    return (single,) if single else ()


def _latest_dt(history: Sequence[datetime]) -> datetime | None:
    if not history:
        return None
    try:
        return max(history)
    except Exception:
        return None


def _collect_recent_tags(candidates: Sequence[_Candidate], cutoff: datetime) -> set[str]:
    recent_tags: set[str] = set()
    for candidate in candidates:
        if not candidate.tags:
            continue
        for moment in candidate.last_used_history:
            if moment >= cutoff:
                recent_tags.update(candidate.tags)
                break
    return recent_tags
