"""Export postcard-ready assets (scores 7 and 8) into a CSV dump."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

TARGET_SCORES: tuple[int, int] = (7, 8)
SCORE_FILTER = ", ".join(str(score) for score in TARGET_SCORES)
DEFAULT_DB_FALLBACK = "/data/bot.db"
DEFAULT_OUTPUT = "high_score_postcards_dump.csv"

_QUERY_WITH_VISION = f"""
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
WHERE
    CAST(a.postcard_score AS INTEGER) IN ({SCORE_FILTER})
    OR CAST(lv.vision_score AS INTEGER) IN ({SCORE_FILTER})
ORDER BY a.created_at DESC,
         a.id ASC
"""

_QUERY_WITHOUT_VISION = f"""
SELECT
    a.id AS asset_id,
    a.created_at,
    a.postcard_score AS column_score,
    NULL AS vision_score,
    json_extract(a.payload_json, '$.last_used_at') AS last_used_at,
    a.payload_json,
    NULL AS result_json
FROM assets AS a
WHERE CAST(a.postcard_score AS INTEGER) IN ({SCORE_FILTER})
ORDER BY a.created_at DESC,
         a.id ASC
"""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dump all postcard-ready assets with score 7 or 8. "
            "Scores are resolved both from the assets table and the latest vision result."
        )
    )
    parser.add_argument(
        "--db-path",
        help=("Path to the SQLite database. Defaults to DB_PATH env or /data/bot.db."),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=(
            "Destination CSV file (defaults to high_score_postcards_dump.csv in the CWD)."
        ),
    )
    parser.add_argument(
        "--pretty-json",
        action="store_true",
        help=("Pretty-print JSON blobs for readability before writing them to the CSV."),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _resolve_db_path(candidate: str | None) -> str:
    if candidate:
        return candidate
    env_value = os.getenv("DB_PATH")
    if env_value:
        return env_value
    return DEFAULT_DB_FALLBACK


def _connect(db_path: str) -> sqlite3.Connection:
    uri = db_path.startswith("file:")
    conn = sqlite3.connect(db_path, uri=uri)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cursor = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND lower(name)=lower(?) LIMIT 1",
        (name,),
    )
    try:
        return cursor.fetchone() is not None
    finally:
        cursor.close()


def _iter_high_score_rows(conn: sqlite3.Connection) -> Iterator[sqlite3.Row]:
    query = _QUERY_WITH_VISION if _table_exists(conn, "vision_results") else _QUERY_WITHOUT_VISION
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
        for row in _iter_high_score_rows(conn):
            writer.writerow(_prepare_record(row, pretty_json=pretty_json))
            row_count += 1
    return row_count


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    db_path = _resolve_db_path(args.db_path)
    output_path = Path(args.output).expanduser().resolve()
    try:
        conn = _connect(db_path)
    except sqlite3.Error as exc:
        print(f"Failed to open database at {db_path}: {exc}")
        return 1
    try:
        row_count = export_high_score_assets(
            conn,
            output_path=output_path,
            pretty_json=bool(args.pretty_json),
        )
    finally:
        conn.close()
    print(f"Exported {row_count} record(s) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
