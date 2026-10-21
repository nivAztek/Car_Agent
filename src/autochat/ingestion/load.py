"""Upsert CarRecords and embedded Chunks into Cosmos; delete stale chunks of re-ingested cars."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from azure.cosmos.exceptions import CosmosHttpResponseError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from autochat.db.cosmos import bootstrap, cars_container, chunks_container
from autochat.ingestion.schemas import CarRecord, Chunk

log = logging.getLogger(__name__)

CONCURRENCY = 8


def _retryable(exc: BaseException) -> bool:
    return isinstance(exc, CosmosHttpResponseError) and ((exc.status_code or 0) == 429 or (exc.status_code or 0) >= 500)


@retry(
    retry=retry_if_exception(_retryable),
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(multiplier=1, max=10),
    reraise=True,
)
async def _upsert(container: Any, body: dict[str, Any]) -> None:
    await container.upsert_item(body)


async def _existing_chunk_ids(car_id: str) -> set[str]:
    query = "SELECT c.id FROM c WHERE c.car_id = @car_id"
    items = chunks_container().query_items(
        query, parameters=[{"name": "@car_id", "value": car_id}], partition_key=car_id
    )
    return {row["id"] async for row in items}


async def load(cars: list[CarRecord], chunks: list[Chunk], vectors: dict[str, list[float]]) -> dict[str, int]:
    await bootstrap()
    missing = [c.id for c in chunks if c.id not in vectors]
    if missing:
        raise RuntimeError(f"{len(missing)} chunks have no embedding, e.g. {missing[:3]}")
    sem = asyncio.Semaphore(CONCURRENCY)

    async def put(container: Any, body: dict[str, Any]) -> None:
        async with sem:
            await _upsert(container, body)

    cars_c, chunks_c = cars_container(), chunks_container()
    await asyncio.gather(*(put(cars_c, car.model_dump(mode="json")) for car in cars))
    await asyncio.gather(
        *(put(chunks_c, chunk.model_dump(mode="json") | {"embedding": vectors[chunk.id]}) for chunk in chunks)
    )

    deleted = 0
    for car in cars:
        keep = {c.id for c in chunks if c.car_id == car.car_id}
        stale = await _existing_chunk_ids(car.car_id) - keep
        for chunk_id in stale:
            await chunks_c.delete_item(chunk_id, partition_key=car.car_id)
            deleted += 1
    if deleted:
        log.info("deleted %d stale chunks", deleted)
    return {"cars": len(cars), "chunks": len(chunks), "deleted": deleted}
