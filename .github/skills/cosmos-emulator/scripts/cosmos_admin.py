"""Cosmos DB admin for autochat: bootstrap / reset / stats.

Usage (repo root):
    .venv\\Scripts\\python.exe .github/skills/cosmos-emulator/scripts/cosmos_admin.py bootstrap
    .venv\\Scripts\\python.exe .github/skills/cosmos-emulator/scripts/cosmos_admin.py reset     # drop + recreate cars/chunks
    .venv\\Scripts\\python.exe .github/skills/cosmos-emulator/scripts/cosmos_admin.py stats

Reads COSMOS_* and EMBED_DIMENSIONS from .env (falls back to emulator defaults).
The container definitions here are the single source of truth - copy `container_specs()` into
src/autochat/db/cosmos.py.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from azure.cosmos import PartitionKey, exceptions
from azure.cosmos.aio import CosmosClient

REPO_ROOT = Path(__file__).resolve().parents[4]
EMULATOR_KEY = "C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw=="


def load_env() -> None:
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def cfg(name: str, default: str) -> str:
    return os.environ.get(name) or default


def container_specs(dims: int) -> dict[str, dict]:
    return {
        cfg("COSMOS_CARS_CONTAINER", "cars"): {
            "partition_key": PartitionKey(path="/car_id"),
        },
        cfg("COSMOS_CHUNKS_CONTAINER", "chunks"): {
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


async def bootstrap(client: CosmosClient, db_name: str, dims: int) -> None:
    db = await client.create_database_if_not_exists(db_name)
    for name, spec in container_specs(dims).items():
        await db.create_container_if_not_exists(id=name, **spec)
        print(f"ok container {db_name}/{name}")


async def reset(client: CosmosClient, db_name: str, dims: int) -> None:
    db = await client.create_database_if_not_exists(db_name)
    for name in container_specs(dims):
        try:
            await db.delete_container(name)
            print(f"dropped {db_name}/{name}")
        except exceptions.CosmosResourceNotFoundError:
            pass
    await bootstrap(client, db_name, dims)


async def stats(client: CosmosClient, db_name: str) -> None:
    db = client.get_database_client(db_name)
    try:
        await db.read()
    except exceptions.CosmosResourceNotFoundError:
        print(f"database '{db_name}' does not exist yet - run: cosmos_admin.py bootstrap (or ingest)")
        return
    async for c in db.list_containers():
        container = db.get_container_client(c["id"])
        count = [x async for x in container.query_items("SELECT VALUE COUNT(1) FROM c")]
        # emulator does not echo vectorEmbeddingPolicy; indexingPolicy.vectorIndexes is returned
        vec = [f"{v['path']}:{v['type']}" for v in (c.get("indexingPolicy") or {}).get("vectorIndexes", [])]
        print(f"{c['id']:<14} docs={count[0] if count else '?':<6} pk={c['partitionKey']['paths']} vector_idx={vec}")


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["bootstrap", "reset", "stats"])
    args = ap.parse_args()
    load_env()
    endpoint = cfg("COSMOS_ENDPOINT", "http://localhost:8081")
    key = cfg("COSMOS_KEY", EMULATOR_KEY)
    db_name = cfg("COSMOS_DATABASE", "autochat")
    dims = int(cfg("EMBED_DIMENSIONS", "1536"))
    async with CosmosClient(endpoint, credential=key) as client:
        if args.action == "bootstrap":
            await bootstrap(client, db_name, dims)
        elif args.action == "reset":
            await reset(client, db_name, dims)
        else:
            await stats(client, db_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
