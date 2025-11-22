import types

import pytest

import osm_utils


@pytest.mark.asyncio
async def test_find_water_body_detects_sea(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def fake_query(query: str, timeout: int = 25) -> dict[str, object]:
        calls.append(query)
        if "coastline" in query:
            return {"elements": [{"type": "way"}]}
        return {"elements": []}

    monkeypatch.setattr(osm_utils, "overpass_query", fake_query)

    info = await osm_utils.find_water_body(54.7, 20.5)

    assert info is not None
    assert info.kind == "sea"
    assert info.name_ru == "Балтийское море"
    assert len(calls) == 1  # lagoon query is skipped because coastline matched


@pytest.mark.asyncio
async def test_find_water_body_returns_nearest_named_element(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries: list[str] = []
    responses = types.SimpleNamespace(step=0)

    async def fake_query(query: str, timeout: int = 25) -> dict[str, object]:
        queries.append(query)
        if responses.step == 0:
            responses.step += 1
            return {"elements": []}
        return {
            "elements": [
                {
                    "center": {"lat": 54.7052, "lon": 20.412},
                    "tags": {"name:ru": "Куршский залив", "water": "lagoon"},
                },
                {
                    "center": {"lat": 54.71, "lon": 20.45},
                    "tags": {"name": "Синявинское озеро / Sinyavino", "water": "lake"},
                },
            ]
        }

    monkeypatch.setattr(osm_utils, "overpass_query", fake_query)

    info = await osm_utils.find_water_body(54.705, 20.41)

    assert info is not None
    assert info.kind == "lagoon"
    assert info.name_ru == "Куршский залив"
    assert len(queries) == 2


@pytest.mark.asyncio
async def test_find_national_park_matches_known_names(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_query(query: str, timeout: int = 25) -> dict[str, object]:
        return {
            "elements": [
                {"tags": {"name:ru": "Национальный парк \u00abКуршская коса\u00bb"}},
                {"tags": {"name": "random area"}},
            ]
        }

    monkeypatch.setattr(osm_utils, "overpass_query", fake_query)

    info = await osm_utils.find_national_park(54.98, 20.5)

    assert info is not None
    assert info.short_name == "Куршская коса"
    assert info.hashtag == "#КуршскаяКоса"


@pytest.mark.asyncio
async def test_find_nearest_settlement_returns_closest(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_query(query: str, timeout: int = 25) -> dict[str, object]:
        return {
            "elements": [
                {
                    "center": {"lat": 54.65, "lon": 20.20},
                    "tags": {"name:ru": "Балтийск"},
                },
                {
                    "center": {"lat": 54.71, "lon": 20.45},
                    "tags": {"name": "Светлогорск"},
                },
            ]
        }

    monkeypatch.setattr(osm_utils, "overpass_query", fake_query)

    info = await osm_utils.find_nearest_settlement(54.70, 20.40, radius_m=3000)

    assert info is not None
    assert info.name == "Светлогорск"
    assert info.distance_m < 4000
