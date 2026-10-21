"""End-to-end smoke test of the emulator features autochat relies on. Uses a throwaway DB.

Checks:
  1. connect + create DB/container with vector policy + DiskANN index (8 dims)
  2. upsert docs with metadata + embeddings
  3. VectorDistance query with parameterized vector  (records whether it works)
  4. VectorDistance query with inlined vector literal (fallback)
  5. metadata pre-filter (ARRAY_CONTAINS + range) combined with vector ordering
  6. confirms full-text functions are NOT available (expected failure on emulator)
  7. LangGraph CosmosDBSaver round-trip (put + get checkpoint)
  8. drops the throwaway DB

Usage:
    .venv\\Scripts\\python.exe .github/skills/cosmos-emulator/scripts/cosmos_smoke.py
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
KEY = os.environ.get("COSMOS_KEY", EMULATOR_KEY)
DB = f"autochat_smoke_{uuid.uuid4().hex[:6]}"
DIMS = 8

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def unit(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


async def checkpointer_roundtrip() -> None:
    try:
        from langchain_azure_cosmosdb import CosmosDBSaver
        from langgraph.checkpoint.base import empty_checkpoint
    except ImportError as exc:
        record("checkpointer import", False, str(exc))
        return
    try:
        async with CosmosDBSaver.from_conn_info(
            endpoint=ENDPOINT, key=KEY, database_name=DB, container_name="checkpoints"
        ) as saver:
            cfg = {"configurable": {"thread_id": "smoke-thread", "checkpoint_ns": ""}}
            ckpt = empty_checkpoint()
            new_cfg = await saver.aput(cfg, ckpt, {"source": "input", "step": 0}, {})
            got = await saver.aget_tuple(new_cfg)
            record("checkpointer put/get", got is not None and got.checkpoint["id"] == ckpt["id"])
    except Exception as exc:  # noqa: BLE001
        record("checkpointer put/get", False, f"{type(exc).__name__}: {exc}"[:300])


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    random.seed(7)
    async with CosmosClient(ENDPOINT, credential=KEY) as client:
        try:
            db = await client.create_database_if_not_exists(DB)
            container = await db.create_container_if_not_exists(
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
            record("create container w/ vector policy + diskANN", True)

            cars = [("mg-s6", "ev", 168888), ("aion-ht", "ev", 179990), ("audi-rs3-fl", "ice", 528000)]
            for i in range(30):
                car_id, pt, price = cars[i % 3]
                await container.upsert_item(
                    {
                        "id": f"{car_id}:{i:02d}:0",
                        "car_id": car_id,
                        "powertrain": pt,
                        "price_ils": price,
                        "text_he": "טקסט בדיקה",
                        "text_en": f"test chunk {i}",
                        "embedding": unit([random.uniform(-1, 1) for _ in range(DIMS)]),
                    }
                )
            record("upsert 30 docs", True)

            qv = unit([random.uniform(-1, 1) for _ in range(DIMS)])

            # 3. parameterized vector
            q_param = (
                "SELECT TOP 5 c.id, VectorDistance(c.embedding, @qv) AS score "
                "FROM c ORDER BY VectorDistance(c.embedding, @qv)"
            )
            try:
                rows = [r async for r in container.query_items(q_param, parameters=[{"name": "@qv", "value": qv}])]
                record("vector query (parameterized @qv)", len(rows) == 5, f"top={rows[0]['score']:.4f}")
            except Exception as exc:  # noqa: BLE001
                record("vector query (parameterized @qv)", False, f"{type(exc).__name__} -> use inline literal")

            # 4. inline literal
            lit = "[" + ",".join(f"{x:.6f}" for x in qv) + "]"
            q_inline = (
                f"SELECT TOP 5 c.id, VectorDistance(c.embedding, {lit}) AS score "
                f"FROM c ORDER BY VectorDistance(c.embedding, {lit})"
            )
            rows = [r async for r in container.query_items(q_inline)]
            record("vector query (inline literal)", len(rows) == 5)

            # 5. metadata pre-filter + vector order
            q_filter = (
                "SELECT TOP 5 c.id, c.car_id, c.price_ils, VectorDistance(c.embedding, @qv) AS score FROM c "
                "WHERE ARRAY_CONTAINS(@car_ids, c.car_id) AND c.price_ils <= @max_price AND c.powertrain = @pt "
                "ORDER BY VectorDistance(c.embedding, @qv)"
            )
            params = [
                {"name": "@qv", "value": qv},
                {"name": "@car_ids", "value": ["mg-s6", "audi-rs3-fl"]},
                {"name": "@max_price", "value": 200000},
                {"name": "@pt", "value": "ev"},
            ]
            try:
                rows = [r async for r in container.query_items(q_filter, parameters=params)]
            except Exception:  # noqa: BLE001 - retry with inline vector
                q_filter_inline = q_filter.replace("@qv", lit)
                rows = [r async for r in container.query_items(q_filter_inline, parameters=params[1:])]
            ok = bool(rows) and all(r["car_id"] == "mg-s6" for r in rows)
            record("filtered vector query", ok, f"n={len(rows)} cars={sorted({r['car_id'] for r in rows})}")

            # 6. full-text should be unsupported on emulator
            try:
                _ = [
                    r
                    async for r in container.query_items(
                        "SELECT TOP 3 c.id FROM c ORDER BY RANK FullTextScore(c.text_en, 'test')"
                    )
                ]
                record("full-text NOT supported (expected)", False, "full-text worked -> emulator gained support?")
            except Exception as exc:  # noqa: BLE001
                record("full-text NOT supported (expected)", True, type(exc).__name__)

            # 7. checkpointer
            await checkpointer_roundtrip()
        finally:
            try:
                await client.delete_database(DB)
                print(f"cleaned up {DB}")
            except Exception:  # noqa: BLE001
                pass

    failed = [r for r in results if not r[1] and "parameterized" not in r[0]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks ok")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
