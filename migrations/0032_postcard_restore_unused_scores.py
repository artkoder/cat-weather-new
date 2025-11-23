from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

TARGET_SCORES: tuple[int, ...] = (7, 8)


def _decode_payload(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, (bytes, bytearray)):
        try:
            text = raw.decode("utf-8")
        except Exception:
            text = raw.decode("utf-8", errors="ignore")
    else:
        text = str(raw)
    text = text.strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, dict):
        return dict(payload)
    return {}


def _encode_payload(payload: dict[str, Any]) -> str | None:
    if not payload:
        return None
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def run(conn: sqlite3.Connection) -> None:
    placeholders = ",".join("?" for _ in TARGET_SCORES)
    query = f"""
        SELECT id, payload_json
        FROM assets
        WHERE postcard_score IN ({placeholders})
    """
    rows = conn.execute(query, TARGET_SCORES).fetchall()
    updated = 0
    for row in rows:
        payload = _decode_payload(row["payload_json"])
        if "last_used_at" not in payload:
            continue
        payload.pop("last_used_at", None)
        payload["updated_at"] = datetime.utcnow().isoformat()
        payload_json = _encode_payload(payload)
        conn.execute(
            "UPDATE assets SET payload_json=? WHERE id=?",
            (payload_json, row["id"]),
        )
        updated += 1
    if updated:
        conn.commit()
