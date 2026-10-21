"""Car catalog: the only way the agent reads car records, so nothing depends on the number of cars.

Every read is bounded (`MAX_CANDIDATES`) and pushed down to Cosmos as parameterized SQL:
    get(ids)              records by id
    find(filters, sort)   exact filter + sort + TOP n (cars missing the sort value come last)
    by_name_keys(keys)    exact name-key lookup for the resolver
    by_makes(makes)       the cars of the named makes (bounded)
    scope()               fixed-size summary for prompts (count, vocabularies, price range), cached with a TTL
    page(q, offset, n)    browse/search for the API and UI

`InMemoryCarCatalog` implements the same protocol over a list (tests, offline tools).
Emulator notes (8.0 probe): `null` sorts first in ORDER BY ASC and MIN/MAX see nulls, hence the explicit
`IS_DEFINED/NOT IS_NULL` and `IS_NUMBER` guards.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from autochat.names import compact, ensure_name_keys
from autochat.retrieval.filters import SORT_FIELDS, SearchFilters, SortField, build_where, matches_filters, sort_cars

MAX_CANDIDATES = 50
SCOPE_TTL_S = 300.0


class MakeName(BaseModel):
    make: str
    make_he: str | None = None


class CatalogScope(BaseModel):
    """What the catalog covers, in a size that does not grow with the number of cars."""

    count: int = 0
    makes: list[MakeName] = Field(default_factory=list)
    powertrains: list[str] = Field(default_factory=list)
    body_types: list[str] = Field(default_factory=list)
    price_min_ils: int | None = None
    price_max_ils: int | None = None

    def make_vocab(self) -> dict[str, str]:
        """compact make key (en + he) -> canonical English make."""
        vocab: dict[str, str] = {}
        for m in self.makes:
            for name in (m.make, m.make_he):
                if k := compact(name):
                    vocab.setdefault(k, m.make)
        return vocab

    def describe(self) -> str:
        price = (
            f"prices ₪{self.price_min_ils:,}-₪{self.price_max_ils:,}"
            if self.price_min_ils is not None and self.price_max_ils is not None
            else "prices vary"
        )
        return (
            f"{self.count} reviewed cars from {len(self.makes)} makes; "
            f"powertrains: {', '.join(self.powertrains) or 'n/a'}; "
            f"body types: {', '.join(self.body_types) or 'n/a'}; {price}"
        )


@runtime_checkable
class CarCatalog(Protocol):
    async def get(self, car_ids: list[str]) -> list[dict[str, Any]]: ...

    async def find(
        self,
        filters: SearchFilters | None = None,
        *,
        sort_by: SortField | None = None,
        descending: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]: ...

    async def by_name_keys(self, keys: list[str]) -> list[dict[str, Any]]: ...

    async def by_makes(self, makes: list[str]) -> list[dict[str, Any]]: ...

    async def scope(self) -> CatalogScope: ...

    async def page(self, q: str = "", offset: int = 0, limit: int = 50) -> tuple[list[dict[str, Any]], int]: ...


def _cap(limit: int | None) -> int:
    return max(1, min(limit or MAX_CANDIDATES, MAX_CANDIDATES))


def _clean(doc: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in doc.items() if not k.startswith("_")}


def _in_order(cars: list[dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    by_id = {c["car_id"]: c for c in cars}
    return [by_id[i] for i in dict.fromkeys(ids) if i in by_id]


def _name_sort(cars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(cars, key=lambda c: ((c.get("make") or "").lower(), (c.get("model") or "").lower(), c["car_id"]))


# --------------------------------------------------------------------------- in-memory
class InMemoryCarCatalog:
    def __init__(self, cars: list[dict[str, Any]] | dict[str, dict[str, Any]]) -> None:
        items = list(cars.values()) if isinstance(cars, dict) else list(cars)
        self._cars = {c["car_id"]: {**c, "name_keys": ensure_name_keys(c)} for c in items}

    async def get(self, car_ids: list[str]) -> list[dict[str, Any]]:
        return _in_order(list(self._cars.values()), car_ids)

    async def find(
        self,
        filters: SearchFilters | None = None,
        *,
        sort_by: SortField | None = None,
        descending: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        f = filters or SearchFilters()
        out = [c for c in self._cars.values() if matches_filters(c, f)]
        out = sort_cars(out, sort_by, descending) if sort_by else sorted(out, key=lambda c: c["car_id"])
        return out[: _cap(limit)]

    async def by_name_keys(self, keys: list[str]) -> list[dict[str, Any]]:
        wanted = set(keys)
        return [c for c in self._cars.values() if wanted & set(c["name_keys"])][:MAX_CANDIDATES]

    async def by_makes(self, makes: list[str]) -> list[dict[str, Any]]:
        wanted = {m.lower() for m in makes}
        return [c for c in self._cars.values() if (c.get("make") or "").lower() in wanted][:MAX_CANDIDATES]

    async def scope(self) -> CatalogScope:
        cars = list(self._cars.values())
        prices = [c["price_ils"] for c in cars if isinstance(c.get("price_ils"), int | float)]
        makes = {c["make"]: c.get("make_he") for c in cars if c.get("make")}
        return CatalogScope(
            count=len(cars),
            makes=[MakeName(make=m, make_he=he) for m, he in sorted(makes.items())],
            powertrains=sorted({c["powertrain"] for c in cars if c.get("powertrain")}),
            body_types=sorted({c["body_type"] for c in cars if c.get("body_type")}),
            price_min_ils=min(prices) if prices else None,
            price_max_ils=max(prices) if prices else None,
        )

    async def page(self, q: str = "", offset: int = 0, limit: int = 50) -> tuple[list[dict[str, Any]], int]:
        key = compact(q)
        hits = [c for c in self._cars.values() if not key or any(key in k for k in c["name_keys"])]
        return _name_sort(hits)[offset : offset + limit], len(hits)


# --------------------------------------------------------------------------- Cosmos
class CosmosCarCatalog:
    """Parameterized SQL on the `cars` container. `TOP`/`OFFSET`/`LIMIT` are int literals; the sort field is
    whitelisted (SORT_FIELDS), every value is a parameter."""

    def __init__(self, scope_ttl_s: float = SCOPE_TTL_S) -> None:
        self._ttl = scope_ttl_s
        self._scope: tuple[float, CatalogScope] | None = None

    @staticmethod
    async def _query(sql: str, params: list[dict[str, Any]] | None = None) -> list[Any]:
        from autochat.db.cosmos import cars_container

        items = cars_container().query_items(sql, parameters=params or [])
        return [_clean(d) if isinstance(d, dict) else d async for d in items]

    async def get(self, car_ids: list[str]) -> list[dict[str, Any]]:
        ids = list(dict.fromkeys(car_ids))[:MAX_CANDIDATES]
        if not ids:
            return []
        sql = "SELECT * FROM c WHERE ARRAY_CONTAINS(@ids, c.car_id)"
        return _in_order(await self._query(sql, [{"name": "@ids", "value": ids}]), ids)

    async def find(
        self,
        filters: SearchFilters | None = None,
        *,
        sort_by: SortField | None = None,
        descending: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        where, params = build_where(filters or SearchFilters())
        n = _cap(limit)
        if not sort_by:
            sql = f"SELECT TOP {n} * FROM c" + (f" WHERE {where}" if where else "") + " ORDER BY c.car_id"
            return await self._query(sql, params)
        if sort_by not in SORT_FIELDS:
            raise ValueError(f"cannot sort by {sort_by!r}")
        has = f"(IS_DEFINED(c.{sort_by}) AND NOT IS_NULL(c.{sort_by}))"
        known_where = " AND ".join(x for x in (where, has) if x)
        order = "DESC" if descending else "ASC"
        rows = await self._query(f"SELECT TOP {n} * FROM c WHERE {known_where} ORDER BY c.{sort_by} {order}", params)
        if len(rows) < n:  # cars without the value go last
            missing_where = " AND ".join(x for x in (where, f"NOT {has}") if x)
            rows += await self._query(
                f"SELECT TOP {n - len(rows)} * FROM c WHERE {missing_where} ORDER BY c.car_id", params
            )
        return rows

    async def by_name_keys(self, keys: list[str]) -> list[dict[str, Any]]:
        if not keys:
            return []
        sql = (
            f"SELECT TOP {MAX_CANDIDATES} * FROM c "
            "WHERE EXISTS(SELECT VALUE k FROM k IN c.name_keys WHERE ARRAY_CONTAINS(@keys, k))"
        )
        return await self._query(sql, [{"name": "@keys", "value": sorted(set(keys))}])

    async def by_makes(self, makes: list[str]) -> list[dict[str, Any]]:
        if not makes:
            return []
        sql = f"SELECT TOP {MAX_CANDIDATES} * FROM c WHERE ARRAY_CONTAINS(@makes, LOWER(c.make)) ORDER BY c.car_id"
        return await self._query(sql, [{"name": "@makes", "value": sorted({m.lower() for m in makes})}])

    async def scope(self) -> CatalogScope:
        now = time.monotonic()
        if self._scope and now - self._scope[0] < self._ttl:
            return self._scope[1]
        count = await self._query("SELECT VALUE COUNT(1) FROM c")
        makes = await self._query("SELECT DISTINCT c.make, c.make_he FROM c WHERE IS_STRING(c.make)")
        pts = await self._query("SELECT DISTINCT VALUE c.powertrain FROM c WHERE IS_STRING(c.powertrain)")
        bodies = await self._query("SELECT DISTINCT VALUE c.body_type FROM c WHERE IS_STRING(c.body_type)")
        lo = await self._query("SELECT VALUE MIN(c.price_ils) FROM c WHERE IS_NUMBER(c.price_ils)")
        hi = await self._query("SELECT VALUE MAX(c.price_ils) FROM c WHERE IS_NUMBER(c.price_ils)")
        by_make = {m["make"]: m.get("make_he") for m in makes}
        scope = CatalogScope(
            count=int(count[0]) if count else 0,
            makes=[MakeName(make=m, make_he=he) for m, he in sorted(by_make.items())],
            powertrains=sorted(pts),
            body_types=sorted(bodies),
            price_min_ils=int(lo[0]) if lo and lo[0] is not None else None,
            price_max_ils=int(hi[0]) if hi and hi[0] is not None else None,
        )
        self._scope = (now, scope)
        return scope

    async def page(self, q: str = "", offset: int = 0, limit: int = 50) -> tuple[list[dict[str, Any]], int]:
        offset, limit = max(0, int(offset)), max(1, min(int(limit), 200))
        key = compact(q)
        where = "WHERE EXISTS(SELECT VALUE k FROM k IN c.name_keys WHERE CONTAINS(k, @q))" if key else ""
        params = [{"name": "@q", "value": key}] if key else []
        total = await self._query(f"SELECT VALUE COUNT(1) FROM c {where}", params)
        rows = await self._query(
            f"SELECT * FROM c {where} ORDER BY c.make OFFSET {offset} LIMIT {limit}",
            params,
        )
        return _name_sort(rows), int(total[0]) if total else 0
