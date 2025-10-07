import json
from datetime import datetime


DEFAULT_RUBRICS = {
    "flowers": {
        "title": "Цветы",
        "config": {
            "enabled": False,
            "schedules": [],
            "assets": {"min": 1, "max": 6, "categories": ["flowers"]},
        },
    },
    "guess_arch": {
        "title": "Угадай архитектуру",
        "config": {
            "enabled": False,
            "schedules": [],
            "assets": {"min": 4, "max": 4, "categories": ["architecture"]},
            "weather_city": "Kaliningrad",
            "overlays_dir": "overlays",
        },
    },
}


def _coerce_row_value(row, key, index):
    if row is None:
        return None
    if hasattr(row, "keys") and key in row.keys():
        return row[key]
    return row[index]


def run(conn):
    now = datetime.utcnow().isoformat()
    for code, payload in DEFAULT_RUBRICS.items():
        title = payload.get("title") or code.title()
        config = payload.get("config") or {}
        row = conn.execute(
            "SELECT title, description FROM rubrics WHERE code=?",
            (code,),
        ).fetchone()
        if row is None:
            conn.execute(
                """
                INSERT INTO rubrics (code, title, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (code, title, json.dumps(config), now, now),
            )
            continue
        existing_title = _coerce_row_value(row, "title", 0)
        description = _coerce_row_value(row, "description", 1)
        update_needed = False
        try:
            current_config = json.loads(description) if description else {}
            if not isinstance(current_config, dict):
                current_config = {}
        except json.JSONDecodeError:
            current_config = {}
        if not current_config:
            current_config = config
            update_needed = True
        if not existing_title:
            existing_title = title
            update_needed = True
        if update_needed:
            conn.execute(
                """
                UPDATE rubrics
                SET title=?, description=?, updated_at=?
                WHERE code=?
                """,
                (existing_title, json.dumps(current_config), now, code),
            )
