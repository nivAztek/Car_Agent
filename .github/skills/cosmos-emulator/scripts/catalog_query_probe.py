"""Probe the emulator for the query shapes an N-car catalog needs (read-only, real `autochat` containers).

Usage:
    .venv\\Scripts\\python.exe .github/skills/cosmos-emulator/scripts/catalog_query_probe.py

Checks: cross-partition COUNT / MIN / MAX / DISTINCT, ORDER BY <number> + TOP (and where cars missing the field
land), ARRAY_CONTAINS against a string array, EXISTS over an array, OFFSET/LIMIT paging, CONTAINS search, and the
recall of a GLOBAL vector TOP-K (no partition_key) against the exact per-partition fan-out. Query vectors are
stored chunk embeddings, so no model calls are made.
"""

from __future__ import annotations

import asyncio
import os
import random
import sys
import time
from typing import Any

from azure.cosmos.aio import CosmosClient

EMULATOR_KEY = "C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw=="
ENDPOINT = os.environ.get("COSMOS_ENDPOINT", "http://localhost:8081")
DB = os.environ.get("COSMOS_DATABASE", "autochat")
VEC = "SELECT TOP {k} c.id, c.car_id, VectorDistance(c.embedding, @qv) AS s FROM c ORDER BY VectorDistance(c.embedding, @qv)"


async def q(container: Any, sql: str, params: list[dict] | None = None, pk: str | None = None) -> list[Any]:
    kw: dict[str, Any] = {"parameters": params or []}
    if pk is not None:
        kw["partition_key"] = pk
    return [r async for r in container.query_items(sql, **kw)]


async def check(name: str, container: Any, sql: str, params: list[dict] | None = None) -> list[Any] | None:
    t = time.perf_counter()
    try:
        rows = await q(container, sql, params)
    except Exception as exc:  # noqa: BLE001 - this is a capability probe
        print(f"FAIL  {name}: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
        return None
    shown = rows if len(rows) <= 8 else [*rows[:8], f"... +{len(rows) - 8}"]
    print(f"OK    {name} ({(time.perf_counter() - t) * 1000:.0f} ms): {shown}")
    return rows


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    async with CosmosClient(ENDPOINT, credential=EMULATOR_KEY) as client:
        db = client.get_database_client(DB)
        cars, chunks = db.get_container_client("cars"), db.get_container_client("chunks")

        print("--- cars: aggregates")
        await check("COUNT", cars, "SELECT VALUE COUNT(1) FROM c")
        await check("MIN/MAX price", cars, "SELECT MIN(c.price_ils) AS lo, MAX(c.price_ils) AS hi FROM c")
        await check("DISTINCT powertrain", cars, "SELECT DISTINCT VALUE c.powertrain FROM c")
        await check("DISTINCT body_type", cars, "SELECT DISTINCT VALUE c.body_type FROM c")
        await check("COUNT with filter", cars, "SELECT VALUE COUNT(1) FROM c WHERE c.price_ils <= @p", [{"name": "@p", "value": 200000}])

        print("--- cars: ORDER BY + TOP (Kia EV9 has no price)")
        await check("TOP 3 ORDER BY price ASC", cars, "SELECT TOP 3 c.car_id, c.price_ils FROM c ORDER BY c.price_ils ASC")
        await check("TOP 3 ORDER BY price DESC", cars, "SELECT TOP 3 c.car_id, c.price_ils FROM c ORDER BY c.price_ils DESC")
        await check("ALL ORDER BY price ASC", cars, "SELECT c.car_id, c.price_ils FROM c ORDER BY c.price_ils ASC")
        await check(
            "TOP 3 IS_NUMBER + ORDER BY price ASC",
            cars,
            "SELECT TOP 3 c.car_id, c.price_ils FROM c WHERE IS_NUMBER(c.price_ils) ORDER BY c.price_ils ASC",
        )
        await check(
            "TOP 2 filter + ORDER BY range DESC",
            cars,
            "SELECT TOP 2 c.car_id, c.range_real_km FROM c WHERE c.powertrain = 'ev' AND IS_NUMBER(c.range_real_km) "
            "ORDER BY c.range_real_km DESC",
        )
        await check(
            "ORDER BY accel ASC (some null)",
            cars,
            "SELECT c.car_id, c.accel_0_100_s FROM c ORDER BY c.accel_0_100_s ASC",
        )

        print("--- cars: name lookup")
        await check("ARRAY_CONTAINS(c.aliases, @a)", cars, "SELECT VALUE c.car_id FROM c WHERE ARRAY_CONTAINS(c.aliases, @a)", [{"name": "@a", "value": "mg s6"}])
        await check(
            "EXISTS(k IN c.aliases WHERE ARRAY_CONTAINS(@keys, k))",
            cars,
            "SELECT VALUE c.car_id FROM c WHERE EXISTS(SELECT VALUE k FROM k IN c.aliases WHERE ARRAY_CONTAINS(@keys, k))",
            [{"name": "@keys", "value": ["gv80", "tesla model 3", "איון ht"]}],
        )
        await check(
            "ARRAY_CONTAINS(@ids, c.car_id) projection",
            cars,
            "SELECT c.car_id, c.make, c.model FROM c WHERE ARRAY_CONTAINS(@ids, c.car_id)",
            [{"name": "@ids", "value": ["mg-s6", "aion-ht"]}],
        )
        await check(
            "LOWER(c.make) IN list",
            cars,
            "SELECT VALUE c.car_id FROM c WHERE ARRAY_CONTAINS(@m, LOWER(c.make))",
            [{"name": "@m", "value": ["audi", "kia"]}],
        )

        print("--- cars: paging + search")
        await check("OFFSET 0 LIMIT 3 ORDER BY car_id", cars, "SELECT VALUE c.car_id FROM c ORDER BY c.car_id OFFSET 0 LIMIT 3")
        await check("OFFSET 3 LIMIT 3 ORDER BY car_id", cars, "SELECT VALUE c.car_id FROM c ORDER BY c.car_id OFFSET 3 LIMIT 3")
        await check(
            "CONTAINS(LOWER(make+model), @q)",
            cars,
            "SELECT VALUE c.car_id FROM c WHERE CONTAINS(LOWER(CONCAT(c.make, ' ', c.model)), @q)",
            [{"name": "@q", "value": "elan"}],
        )
        await check(
            "CONTAINS(c.make_he, @q)",
            cars,
            "SELECT VALUE c.car_id FROM c WHERE CONTAINS(c.make_he, @q) OR CONTAINS(c.model_he, @q)",
            [{"name": "@q", "value": "אלנטרה"}],
        )

        print("--- chunks: point read + IN projection")
        await check(
            "ARRAY_CONTAINS(@ids, c.id) text projection",
            chunks,
            "SELECT c.id, LENGTH(c.text_en) AS n FROM c WHERE ARRAY_CONTAINS(@ids, c.id)",
            [{"name": "@ids", "value": ["mg-s6:card", "aion-ht:card"]}],
        )

        print("--- chunks: global vector TOP-K vs exact per-partition fan-out")
        car_ids = await q(cars, "SELECT VALUE c.car_id FROM c")
        sample = await q(chunks, "SELECT c.id, c.embedding FROM c")
        random.seed(7)
        queries = random.sample(sample, 20)
        for k in (5, 10, 20):
            recalls, t_global, t_fan = [], 0.0, 0.0
            for doc in queries:
                params = [{"name": "@qv", "value": doc["embedding"]}]
                t = time.perf_counter()
                parts = await asyncio.gather(*(q(chunks, VEC.format(k=k), params, pk=c) for c in car_ids))
                t_fan += time.perf_counter() - t
                exact = sorted((r for p in parts for r in p), key=lambda r: -r["s"])[:k]
                t = time.perf_counter()
                glob = await q(chunks, VEC.format(k=k), params)
                t_global += time.perf_counter() - t
                recalls.append(len({r["id"] for r in exact} & {r["id"] for r in glob}) / k)
            print(
                f"TOP {k:>2}: global recall@k mean={sum(recalls) / len(recalls):.3f} min={min(recalls):.2f} "
                f"| avg ms global={t_global / len(queries) * 1000:.0f} fan-out({len(car_ids)})={t_fan / len(queries) * 1000:.0f}"
            )
        doc = queries[0]
        await check(
            "global vector + text projection (TOP 5)",
            chunks,
            "SELECT TOP 5 c.id, c.car_id, c.section_type, LENGTH(c.text_en) AS n, VectorDistance(c.embedding, @qv) AS s "
            "FROM c ORDER BY VectorDistance(c.embedding, @qv)",
            [{"name": "@qv", "value": doc["embedding"]}],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
