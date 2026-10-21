"""Async Cosmos DB client singleton + container bootstrap (emulator-compatible).

Verified emulator facts (see cosmos-emulator skill): DiskANN only, vector policies are immutable,
`create_container` occasionally fails with a transient PostgresError(E57014) -> retried here.
"""

from __future__ import annotations

import logging
from typing import Any

from azure.cosmos import PartitionKey
from azure.cosmos.aio import ContainerProxy, CosmosClient, DatabaseProxy
from azure.cosmos.exceptions import CosmosHttpResponseError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from autochat.config import get_settings

log = logging.getLogger(__name__)

_client: CosmosClient | None = None


def get_client() -> CosmosClient:
    """One CosmosClient per process (call `close_client()` on shutdown)."""
    global _client
    if _client is None:
        s = get_settings()
        _client = CosmosClient(s.cosmos_endpoint, credential=s.cosmos_key.get_secret_value())
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None


def database() -> DatabaseProxy:
    return get_client().get_database_client(get_settings().cosmos_database)


def cars_container() -> ContainerProxy:
    return database().get_container_client(get_settings().cosmos_cars_container)


def chunks_container() -> ContainerProxy:
    return database().get_container_client(get_settings().cosmos_chunks_container)


def container_specs(dims: int | None = None) -> dict[str, dict[str, Any]]:
    s = get_settings()
    dims = dims or s.embed_dimensions
    return {
        s.cosmos_cars_container: {"partition_key": PartitionKey(path="/car_id")},
        s.cosmos_chunks_container: {
            "partition_key": PartitionKey(path="/car_id"),
            "vector_embedding_policy": {
                "vectorEmbeddings": [
                    {"path": "/embedding", "dataType": "float32", "distanceFunction": "cosine", "dimensions": dims}
                ]
            },
            "indexing_policy": {
                "indexingMode": "consistent",
                "automatic": True,
                "includedPaths": [{"path": "/*"}],
                "excludedPaths": [{"path": '/"_etag"/?'}, {"path": "/embedding/*"}],
                "vectorIndexes": [{"path": "/embedding", "type": "diskANN"}],
            },
        },
        # checkpoints container is created by langchain_azure_cosmosdb.CosmosDBSaver (PK /partition_key)
    }


def _transient(exc: BaseException) -> bool:
    return isinstance(exc, CosmosHttpResponseError) and (exc.status_code or 0) >= 500


@retry(
    retry=retry_if_exception(_transient),
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(multiplier=1, max=10),
    reraise=True,
)
async def bootstrap() -> None:
    """Create database + cars/chunks containers if missing (idempotent)."""
    db = await get_client().create_database_if_not_exists(get_settings().cosmos_database)
    for name, spec in container_specs().items():
        await db.create_container_if_not_exists(id=name, **spec)
        log.debug("container ready: %s", name)
