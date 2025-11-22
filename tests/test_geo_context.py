from __future__ import annotations

from types import SimpleNamespace

import pytest

import geo_context


@pytest.fixture()
def stub_overpass(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    state = SimpleNamespace(
        park=None,
        settlement=None,
        settlement_calls=[],
        coastline_elements=[],
        water_elements=[],
        overpass_queries=[],
    )

    async def fake_find_national_park(lat: float, lon: float):  # type: ignore[override]
        return state.park

    async def fake_find_nearest_settlement(lat: float, lon: float, radius_m: int):  # type: ignore[override]
        state.settlement_calls.append(radius_m)
        return state.settlement

    async def fake_overpass_query(query: str, timeout: int = 25):  # type: ignore[override]
        state.overpass_queries.append(query)
        if '"natural"="coastline"' in query:
            return {"elements": list(state.coastline_elements)}
        if '"natural"="water"' in query or '["water"]' in query:
            return {"elements": list(state.water_elements)}
        return {"elements": []}

    monkeypatch.setattr(geo_context, "find_national_park", fake_find_national_park)
    monkeypatch.setattr(geo_context, "find_nearest_settlement", fake_find_nearest_settlement)
    monkeypatch.setattr(geo_context, "overpass_query", fake_overpass_query)
    return state


@pytest.mark.asyncio
async def test_build_geo_context_prefers_asset_city(stub_overpass: SimpleNamespace) -> None:
    result = await geo_context.build_geo_context_for_asset(
        lat=54.7,
        lon=20.5,
        asset_city="Светлогорск",
        asset_tags=["forest"],
    )

    assert result.main_place == "Светлогорск"
    assert stub_overpass.settlement_calls == []


@pytest.mark.asyncio
async def test_build_geo_context_uses_settlement_within_park(stub_overpass: SimpleNamespace) -> None:
    stub_overpass.park = SimpleNamespace(short_name="Куршская коса")
    stub_overpass.settlement = SimpleNamespace(name="Морское")

    result = await geo_context.build_geo_context_for_asset(
        lat=54.9,
        lon=20.0,
        asset_city=None,
        asset_tags=["water"],
    )

    assert result.main_place == "Морское"
    assert result.national_park == "Куршская коса"
    assert stub_overpass.settlement_calls == [500]


@pytest.mark.asyncio
async def test_build_geo_context_skips_water_without_tags(stub_overpass: SimpleNamespace) -> None:
    await geo_context.build_geo_context_for_asset(
        lat=54.7,
        lon=20.5,
        asset_city=None,
        asset_tags=["forest"],
    )

    assert stub_overpass.overpass_queries == []


@pytest.mark.asyncio
async def test_detects_baltic_sea_when_coastline_and_sea_tag(stub_overpass: SimpleNamespace) -> None:
    stub_overpass.coastline_elements = [
        {"center": {"lat": 54.7005, "lon": 20.5001}},
    ]

    result = await geo_context.build_geo_context_for_asset(
        lat=54.7,
        lon=20.5,
        asset_city=None,
        asset_tags=["sea", "water"],
    )

    assert result.water_kind == "sea"
    assert result.water_name == "Балтийское море"


@pytest.mark.asyncio
async def test_prefers_nearest_water_over_sea(stub_overpass: SimpleNamespace) -> None:
    stub_overpass.coastline_elements = [
        {"center": {"lat": 54.702, "lon": 20.52}},
    ]
    stub_overpass.water_elements = [
        {
            "center": {"lat": 54.7001, "lon": 20.5001},
            "tags": {"name:ru": "Синявинское озеро", "water": "lake"},
        }
    ]

    result = await geo_context.build_geo_context_for_asset(
        lat=54.7,
        lon=20.5,
        asset_city=None,
        asset_tags=["sea", "water"],
    )

    assert result.water_kind == "lake"
    assert result.water_name == "Синявинское озеро"


@pytest.mark.asyncio
async def test_prefers_named_water_without_sea_tag(stub_overpass: SimpleNamespace) -> None:
    stub_overpass.coastline_elements = [
        {"center": {"lat": 54.7001, "lon": 20.5001}},
    ]
    stub_overpass.water_elements = [
        {
            "center": {"lat": 54.704, "lon": 20.54},
            "tags": {"name": "Преголя", "water": "river"},
        }
    ]

    result = await geo_context.build_geo_context_for_asset(
        lat=54.7,
        lon=20.5,
        asset_city=None,
        asset_tags=["water"],
    )

    assert result.water_kind == "river"
    assert result.water_name == "Преголя"


@pytest.mark.asyncio
async def test_returns_none_when_no_water_detected(stub_overpass: SimpleNamespace) -> None:
    result = await geo_context.build_geo_context_for_asset(
        lat=54.7,
        lon=20.5,
        asset_city=None,
        asset_tags=["water"],
    )

    assert result.water_kind is None
    assert result.water_name is None
