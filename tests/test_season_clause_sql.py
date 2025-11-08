import os
import sqlite3
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_access import DataAccess  # noqa: E402


def test_season_clause_parentheses() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE assets (id INTEGER PRIMARY KEY, payload_json TEXT, shot_doy INTEGER)"
    )

    def _insert(asset_id: int, category: str, shot_doy: int | None) -> None:
        payload = f'{{"vision_category": "{category}"}}'
        conn.execute(
            "INSERT INTO assets (id, payload_json, shot_doy) VALUES (?, ?, ?)",
            (asset_id, payload, shot_doy),
        )

    _insert(1, "sea", 300)
    _insert(2, "sea", None)
    _insert(3, "sea", 50)
    _insert(4, "land", None)
    conn.commit()

    clause, params = DataAccess.build_season_clause(250, 320)

    assert clause.startswith("(") and clause.endswith(")")
    assert "OR shot_doy IS NULL" in clause

    sql = (
        "SELECT id FROM assets WHERE json_extract(payload_json, '$.vision_category')='sea' "
        f"AND {clause} ORDER BY id"
    )
    rows = conn.execute(sql, params).fetchall()
    selected_ids = [row[0] for row in rows]

    assert selected_ids == [1, 2]

    conn.close()
