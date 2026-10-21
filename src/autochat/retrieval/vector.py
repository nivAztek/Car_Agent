"""Per-car-partition vector search (exact on the emulator; no WHERE on vector queries)."""

from __future__ import annotations

import asyncio
from typing import Any

# TOP must be an int literal; it is formatted from a validated int, never from user input.
VECTOR_SQL = (
    "SELECT TOP {k} c.id, c.car_id, VectorDistance(c.embedding, @qv) AS score "
    "FROM c ORDER BY VectorDistance(c.embedding, @qv)"
)


async def _search_partition(container: Any, qv: list[float], car_id: str, k: int) -> list[tuple[str, float]]:
    sql = VECTOR_SQL.format(k=int(k))
    rows = container.query_items(sql, parameters=[{"name": "@qv", "value": qv}], partition_key=car_id)
    return [(r["id"], float(r["score"])) async for r in rows]


async def vector_search(qv: list[float], car_ids: list[str], k_per_car: int) -> list[tuple[str, float]]:
    """Fan out one TOP-k query per car partition in parallel; return (chunk_id, cosine similarity) best-first."""
    from autochat.db.cosmos import chunks_container

    if k_per_car < 1 or not car_ids:
        return []
    container = chunks_container()
    results = await asyncio.gather(*(_search_partition(container, qv, cid, k_per_car) for cid in car_ids))
    merged = [hit for part in results for hit in part]
    merged.sort(key=lambda h: -h[1])
    return merged
