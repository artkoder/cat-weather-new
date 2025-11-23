"""Utilities for exporting postcard assets into CSV files."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterator

_QUERY_WITH_VISION = """
WITH latest_vision AS (
    SELECT
        vr.asset_id,
        vr.result_json,
        json_extract(vr.result_json, '$.postcard_score') AS vision_score
    FROM vision_results AS vr
    WHERE vr.id = (
        SELECT vr2.id
        FROM vision_results AS vr2
        WHERE vr2.asset_id = vr.asset_id
        ORDER BY COALESCE(vr2.updated_at, vr2.created_at) DESC,
                 vr2.id DESC
        LIMIT 1
    )
)
SELECT
    a.id AS asset_id,
    a.created_at,
    a.postcard_score AS column_score,
    lv.vision_score,
    json_extract(a.payload_json, '$.last_used_at') AS last_used_at,
    a.payload_json,
    lv.result_json
FROM assets AS a
LEFT JOIN latest_vision AS lv ON lv.asset_id = a.id
ORDER BY a.created_at DESC,
         a.id ASC
"""

_QUERY_WITHOUT_VISION = """
SELECT
    a.id AS asset_id,
    a.created_at,
    a.postcard_score AS column_score,
    NULL AS vision_score,
    json_extract(a.payload_json, '$.last_used_at') AS last_used_at,
    a.payload_json,
    NULL AS result_json
FROM assets AS a
ORDER BY a.created_at DESC,
         a.id ASC
"""


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cursor = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND lower(name)=lower(?) LIMIT 1",
        (name,),
    )
    try:
        return cursor.fetchone() is not None
    finally:
        cursor.close()


def _iter_asset_rows(conn: sqlite3.Connection) -> Iterator[sqlite3.Row]:
    if _table_exists(conn, "vision_results"):
        query = _QUERY_WITH_VISION
    else:
        query = _QUERY_WITHOUT_VISION
    cursor = conn.execute(query)
    try:
        for row in cursor:
            yield row
    finally:
        cursor.close()


def _normalize_score(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        candidate = int(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            candidate = int(text)
        except ValueError:
            try:
                candidate = int(float(text))
            except ValueError:
                return None
    return candidate


def _resolve_score(column_score: Any, vision_score: Any) -> tuple[int | None, str]:
    column_value = _normalize_score(column_score)
    vision_value = _normalize_score(vision_score)
    if column_value is not None:
        return column_value, "column"
    if vision_value is not None:
        return vision_value, "vision"
    return None, ""


def _format_json(value: Any, pretty: bool) -> str:
    if value is None:
        return ""
    text = str(value)
    if not pretty:
        return text
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError, ValueError):
        return text
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True, indent=2)


def _prepare_record(row: sqlite3.Row, *, pretty_json: bool) -> dict[str, Any]:
    resolved_score, score_source = _resolve_score(row["column_score"], row["vision_score"])
    return {
        "asset_id": row["asset_id"],
        "created_at": row["created_at"],
        "resolved_postcard_score": resolved_score,
        "score_source": score_source,
        "column_postcard_score": row["column_score"],
        "vision_postcard_score": row["vision_score"],
        "last_used_at": row["last_used_at"],
        "payload_json": _format_json(row["payload_json"], pretty_json),
        "vision_result_json": _format_json(row["result_json"], pretty_json),
    }


def export_high_score_assets(
    conn: sqlite3.Connection,
    *,
    output_path: Path,
    pretty_json: bool = False,
) -> int:
    """Export postcard assets into a CSV file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "asset_id",
        "created_at",
        "resolved_postcard_score",
        "score_source",
        "column_postcard_score",
        "vision_postcard_score",
        "last_used_at",
        "payload_json",
        "vision_result_json",
    ]
    row_count = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in _iter_asset_rows(conn):
            writer.writerow(_prepare_record(row, pretty_json=pretty_json))
            row_count += 1
    return row_count
