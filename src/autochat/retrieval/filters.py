"""Metadata filters -> exact, parameterized Cosmos SQL on the `cars` container.

Filters never touch the vector query (the emulator applies WHERE after the approximate top-K). They resolve to
car ids first; retrieval then fans out one vector query per car partition (see vector.py).

Null semantics:
- price / 0-100 / makes / powertrain / body type: strict (a car without the value does not match).
- seats / range: lenient on unknown (no review states seat count; dropping every car would be wrong).
"""

from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from autochat.ingestion.schemas import BodyType, Powertrain, Tag

SortField = Literal[
    "price_ils",
    "accel_0_100_s",
    "power_hp",
    "torque_nm",
    "top_speed_kmh",
    "range_real_km",
    "range_wltp_km",
    "dc_charge_kw",
    "battery_kwh",
    "trunk_l",
    "fuel_km_per_l",
    "energy_kwh_per_100km",
    "review_date",
]
SORT_FIELDS: tuple[str, ...] = get_args(SortField)

# A user asking for "petrol" also means mild hybrids; "hybrid" covers every electrified ICE.
POWERTRAIN_GROUPS: dict[str, list[str]] = {
    "ice": ["ice", "mhev"],
    "hybrid": ["hybrid", "mhev", "phev"],
    "mhev": ["mhev"],
    "phev": ["phev"],
    "ev": ["ev"],
}
BODY_GROUPS: dict[str, list[str]] = {"suv": ["suv", "crossover"], "crossover": ["crossover", "suv"]}


class SearchFilters(BaseModel):
    """Hard constraints extracted from the conversation. Empty = no constraint."""

    # Set by code from resolved car names, never by the planner (hidden from the LLM schema).
    car_ids: SkipJsonSchema[list[str]] = Field(default_factory=list)
    makes: list[str] = Field(default_factory=list, description="Manufacturer names in English, e.g. 'MG', 'Audi'")
    powertrains: list[Powertrain] = Field(default_factory=list)
    body_types: list[BodyType] = Field(default_factory=list)
    min_price_ils: int | None = None
    max_price_ils: int | None = None
    min_seats: int | None = None
    min_range_km: int | None = Field(None, description="Electric range (real-world if reported, else WLTP)")
    max_accel_0_100_s: float | None = None
    tags: list[Tag] = Field(default_factory=list, description="Match ANY of these suitability tags")

    def is_empty(self) -> bool:
        return not any(v not in (None, [], "") for v in self.model_dump().values())


def expand_powertrains(values: list[str]) -> list[str]:
    out: dict[str, None] = {}
    for v in values:
        for x in POWERTRAIN_GROUPS.get(v, [v]):
            out.setdefault(x, None)
    return list(out)


def expand_body_types(values: list[str]) -> list[str]:
    out: dict[str, None] = {}
    for v in values:
        for x in BODY_GROUPS.get(v, [v]):
            out.setdefault(x, None)
    return list(out)


def build_where(f: SearchFilters) -> tuple[str, list[dict[str, Any]]]:
    """Return (where_clause_or_empty, parameters). Only fixed SQL fragments; every value is a parameter."""
    clauses: list[str] = []
    params: list[dict[str, Any]] = []

    def param(name: str, value: Any) -> str:
        params.append({"name": f"@{name}", "value": value})
        return f"@{name}"

    if f.car_ids:
        clauses.append(f"ARRAY_CONTAINS({param('car_ids', f.car_ids)}, c.car_id)")
    if f.makes:
        clauses.append(f"ARRAY_CONTAINS({param('makes', [m.strip().lower() for m in f.makes])}, LOWER(c.make))")
    if f.powertrains:
        clauses.append(f"ARRAY_CONTAINS({param('powertrains', expand_powertrains(f.powertrains))}, c.powertrain)")
    if f.body_types:
        clauses.append(f"ARRAY_CONTAINS({param('body_types', expand_body_types(f.body_types))}, c.body_type)")
    if f.min_price_ils is not None:
        clauses.append(f"c.price_ils >= {param('min_price', f.min_price_ils)}")
    if f.max_price_ils is not None:
        clauses.append(f"c.price_ils <= {param('max_price', f.max_price_ils)}")
    if f.min_seats is not None:
        clauses.append(f"(NOT IS_NUMBER(c.seats) OR c.seats >= {param('min_seats', f.min_seats)})")
    if f.min_range_km is not None:
        p = param("min_range", f.min_range_km)
        clauses.append(f"(c.range_real_km >= {p} OR (NOT IS_NUMBER(c.range_real_km) AND c.range_wltp_km >= {p}))")
    if f.max_accel_0_100_s is not None:
        clauses.append(f"c.accel_0_100_s <= {param('max_accel', f.max_accel_0_100_s)}")
    if f.tags:
        ors = [f"ARRAY_CONTAINS(c.tags, {param(f'tag{i}', t)})" for i, t in enumerate(f.tags)]
        clauses.append("(" + " OR ".join(ors) + ")")
    return " AND ".join(clauses), params


def build_query(f: SearchFilters) -> tuple[str, list[dict[str, Any]]]:
    where, params = build_where(f)
    return "SELECT * FROM c" + (f" WHERE {where}" if where else ""), params


def _num(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def matches_filters(car: dict[str, Any], f: SearchFilters) -> bool:
    """Python twin of `build_where` (same null semantics) for the in-memory catalog and tests."""
    price, accel = car.get("price_ils"), car.get("accel_0_100_s")
    seats, real, wltp = car.get("seats"), car.get("range_real_km"), car.get("range_wltp_km")
    checks = [
        not f.car_ids or car.get("car_id") in f.car_ids,
        not f.makes or (car.get("make") or "").lower() in {m.strip().lower() for m in f.makes},
        not f.powertrains or car.get("powertrain") in expand_powertrains(f.powertrains),
        not f.body_types or car.get("body_type") in expand_body_types(f.body_types),
        f.min_price_ils is None or (_num(price) and price >= f.min_price_ils),
        f.max_price_ils is None or (_num(price) and price <= f.max_price_ils),
        f.min_seats is None or not _num(seats) or seats >= f.min_seats,
        f.min_range_km is None
        or (_num(real) and real >= f.min_range_km)
        or (not _num(real) and _num(wltp) and wltp >= f.min_range_km),
        f.max_accel_0_100_s is None or (_num(accel) and accel <= f.max_accel_0_100_s),
        not f.tags or bool(set(car.get("tags") or []) & set(f.tags)),
    ]
    return all(checks)


def _clean(doc: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in doc.items() if not k.startswith("_")}


def sort_cars(cars: list[dict[str, Any]], sort_by: str, descending: bool = False) -> list[dict[str, Any]]:
    """Sort by a numeric field; cars missing the value always go last."""
    if sort_by not in SORT_FIELDS:
        raise ValueError(f"cannot sort by {sort_by!r}")
    known = [c for c in cars if c.get(sort_by) is not None]
    unknown = [c for c in cars if c.get(sort_by) is None]
    return sorted(known, key=lambda c: c[sort_by], reverse=descending) + unknown


async def find_cars(
    filters: SearchFilters | None = None,
    *,
    sort_by: SortField | None = None,
    descending: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Exact filter + sort on `cars` (SQL pushdown, capped). Thin wrapper over `CosmosCarCatalog.find`."""
    from autochat.retrieval.catalog import CosmosCarCatalog

    return await CosmosCarCatalog().find(filters, sort_by=sort_by, descending=descending, limit=limit)


async def resolve_car_ids(filters: SearchFilters | None) -> list[str]:
    return [c["car_id"] for c in await find_cars(filters)]
