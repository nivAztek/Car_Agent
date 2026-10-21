"""Investigate recall of filtered VectorDistance queries on the emulator (DiskANN + WHERE).

Usage:
    .venv\\Scripts\\python.exe .github/skills/cosmos-emulator/scripts/filter_recall_probe.py

Inserts N docs over 3 cars, then compares, for a filter that matches M docs:
  - count of docs matching the filter (no vector)
  - results of filtered vector query for several TOP values
  - results when scoping by partition key (single-partition query) instead of WHERE
"""

from __future__ import annotations

import asyncio
import math
import os
import random
import sys
import uuid

from azure.cosmos import PartitionKey
from azure.cosmos.aio import CosmosClient

EMULATOR_KEY = "C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw=="
ENDPOINT = os.environ.get("COSMOS_ENDPOINT", "http://localhost:8081")
DB = f"autochat_recall_{uuid.uuid4().hex[:6]}"
DIMS = 64
N = 240  # ~ size of the real corpus


def unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


async def run(container, sql: str, params=None, pk=None) -> list[dict]:
    kwargs = {"parameters": params or []}
    if pk is not None:
        kwargs["partition_key"] = pk
    return [r async for r in container.query_items(sql, **kwargs)]


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    random.seed(11)
    cars = [
        ("mg-s6", "ev", 168888),
        ("aion-ht", "ev", 179990),
        ("audi-rs3-fl", "ice", 528000),
        ("genesis-gv80-2026", "ice", 575990),
        ("citroen-c3-2026", "ice", 114990),
        ("hyundai-elantra-n-manual", "ice", 224900),
        ("link-and-co-01-2026", "phev", 198900),
        ("long-term-report-kia-ev9-4", "ev", 0),
    ]
    async with CosmosClient(ENDPOINT, credential=EMULATOR_KEY) as client:
        try:
            db = await client.create_database_if_not_exists(DB)
            c = await db.create_container_if_not_exists(
                id="chunks",
                partition_key=PartitionKey(path="/car_id"),
                vector_embedding_policy={
                    "vectorEmbeddings": [
                        {"path": "/embedding", "dataType": "float32", "distanceFunction": "cosine", "dimensions": DIMS}
                    ]
                },
                indexing_policy={
                    "indexingMode": "consistent",
                    "automatic": True,
                    "includedPaths": [{"path": "/*"}],
                    "excludedPaths": [{"path": '/"_etag"/?'}, {"path": "/embedding/*"}],
                    "vectorIndexes": [{"path": "/embedding", "type": "diskANN"}],
                },
            )
            for i in range(N):
                car_id, pt, price = cars[i % len(cars)]
                await c.upsert_item(
                    {
                        "id": f"{car_id}:{i:03d}",
                        "car_id": car_id,
                        "powertrain": pt,
                        "price_ils": price,
                        "half": (i // len(cars)) % 2,
                        "embedding": unit([random.uniform(-1, 1) for _ in range(DIMS)]),
                    }
                )
            qv = unit([random.uniform(-1, 1) for _ in range(DIMS)])
            vp = [{"name": "@qv", "value": qv}]

            matching = await run(c, "SELECT VALUE COUNT(1) FROM c WHERE c.car_id = 'mg-s6'")
            print(f"docs matching car_id='mg-s6': {matching[0]}")
            ev = await run(c, "SELECT VALUE COUNT(1) FROM c WHERE c.powertrain = 'ev'")
            print(f"docs matching powertrain='ev': {ev[0]}  (3 of 8 cars)")

            for top in (5, 10, 30):
                base = f"SELECT TOP {top} c.id, c.car_id FROM c {{w}} ORDER BY VectorDistance(c.embedding, @qv)"
                r_none = await run(c, base.format(w=""), vp)
                r_where_car = await run(c, base.format(w="WHERE c.car_id = 'mg-s6'"), vp)
                r_pk = await run(c, base.format(w=""), vp, pk="mg-s6")
                r_arr = await run(
                    c,
                    base.format(w="WHERE ARRAY_CONTAINS(@ids, c.car_id)"),
                    vp + [{"name": "@ids", "value": ["mg-s6", "aion-ht"]}],
                )
                r_pt = await run(c, base.format(w="WHERE c.powertrain = 'ev'"), vp)
                r_price = await run(c, base.format(w="WHERE c.price_ils <= 200000"), vp)
                r_pk_where = await run(c, base.format(w="WHERE c.half = 1"), vp, pk="mg-s6")
                print(
                    f"TOP {top:>2}: no-filter={len(r_none):>2}  where car={len(r_where_car):>2}  "
                    f"partition_key={len(r_pk):>2}  array_contains(2 cars)={len(r_arr):>2}  "
                    f"powertrain=ev={len(r_pt):>2}  price<=200k={len(r_price):>2}  "
                    f"pk+where(half of 30)={len(r_pk_where):>2}"
                )
        finally:
            await client.delete_database(DB)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
