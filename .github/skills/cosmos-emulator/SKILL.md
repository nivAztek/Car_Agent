---
name: cosmos-emulator
description: "Run and use the Azure Cosmos DB for NoSQL vNext emulator (Docker) for autochat: start/stop/wipe, container + vector policies, filtered vector query patterns, LangGraph checkpointer, verified emulator limitations. Use when touching src/autochat/db or src/autochat/retrieval, when queries return too few results, or when Cosmos errors appear."
---

# cosmos-emulator

## Scripts (run from repo root)

| Script | Purpose |
|---|---|
| `scripts/start_emulator.ps1 [-Stop] [-Wipe] [-TimeoutSec N]` | Starts Docker Desktop if needed, then `docker compose up -d cosmos`, then polls `http://localhost:8080/ready`. |
| `scripts/cosmos_admin.py bootstrap\|reset\|stats` | Idempotent container creation, drop + recreate (`reset`, needed after a dims change), and doc counts. `container_specs()` is the source of truth for policies. |
| `scripts/cosmos_smoke.py` | 7-check E2E test on a throwaway DB: vector policy, upsert, parameterized and literal vectors, filtered query, full-text unsupported, `CosmosDBSaver` round-trip. |
| `scripts/filter_recall_probe.py` | Measures how many rows filtered vector queries return. Re-run if the emulator image is updated. |
| `scripts/catalog_query_probe.py` | Read-only probe of the N-car query shapes on the real containers: aggregates, `ORDER BY`+`TOP` with missing values, name lookups, paging, global vector recall vs. fan-out. No model calls. |

```powershell
powershell -ExecutionPolicy Bypass -File .github/skills/cosmos-emulator/scripts/start_emulator.ps1
.venv\Scripts\python.exe .github/skills/cosmos-emulator/scripts/cosmos_smoke.py
```

Endpoint `http://localhost:8081`. Data Explorer `http://localhost:1234`. Data persists in the named Docker volume `autochat-cosmos-data` (`-Wipe` drops it). **Do not bind-mount a Windows folder to `/data`**: every write then costs ~1.5 s versus ~0.1 s on a named volume (measured), and LangGraph checkpointing does ~20 sequential upserts per turn. The first image pull takes about 7 minutes.

## Verified emulator behavior (2026-10-16, image `vnext-latest`, azure-cosmos 4.17.1)

| Capability | Result |
|---|---|
| Vector policy + `diskANN` index | ✅ works |
| `VectorDistance(c.embedding, @qv)` with a **parameterized** vector | ✅ works (the inline literal also works) |
| `FullTextScore` / `FullTextContains` / `RRF` | ❌ `CosmosHttpResponseError`. **Never use on the emulator.** |
| `CosmosDBSaver.from_conn_info(...)` (LangGraph checkpointer) | ✅ put/get round-trip. Creates its own container with PK `/partition_key` |
| **`WHERE` + `ORDER BY VectorDistance`** | ⚠️ **The filter is applied AFTER the approximate top-K.** With 240 docs and `TOP 5 ... WHERE c.car_id='mg-s6'` (30 matching), it returned **0**. `TOP 30` returned 2. The same happens inside a single partition. |
| `partition_key=<car_id>` on `query_items` (no `WHERE`) | ✅ **Exact.** `TOP k` returns k rows |
| Occasional `InternalServerError PostgresError(E57014)` on create container | Transient (query canceled). Retry with backoff |
| `container.read()` / `list_containers()` omit `vectorEmbeddingPolicy` | Policy IS applied (vector queries work). Check `indexingPolicy.vectorIndexes` instead |
| Range and composite indexes | Accepted but ignored (no optimization). Fine at this scale |
| `ARRAY_CONTAINS`, `LOWER`, `IS_NUMBER`, `OR`, `??` (coalesce) in `WHERE` on `cars` | ✅ all work (verified 2026-10-17) |
| Reserved words as property names (`order`, `value`, ...) | ❌ `SC1001 Syntax error`. Use `c["order"]` |

### Catalog query shapes (verified 2026-10-21, `catalog_query_probe.py`, 8 cars / 110 chunks)

| Query | Result |
|---|---|
| Cross-partition `COUNT(1)`, `DISTINCT VALUE`, `COUNT` with `WHERE` | ✅ |
| `MIN(c.price_ils)` when one car has `price_ils: null` | ⚠️ returns `null`. Guard with `WHERE IS_NUMBER(c.x)` |
| `ORDER BY c.x ASC` + `TOP` cross-partition | ✅, but **`null` sorts first** (Kia EV9 "cheapest"). Always add `WHERE IS_NUMBER(c.x)` to a sort; append cars missing the value in code if needed |
| `ARRAY_CONTAINS(c.arr, @v)`, `EXISTS(SELECT VALUE k FROM k IN c.arr WHERE ARRAY_CONTAINS(@keys, k))` | ✅ (name-key lookup in one query) |
| `ORDER BY c.car_id OFFSET @o LIMIT @l`, `CONTAINS(LOWER(CONCAT(...)), @q)`, `CONTAINS` on Hebrew fields | ✅ |
| **Global vector `TOP k` (no `partition_key`, no `WHERE`)** vs. exact per-car fan-out | ✅ recall@5/10/20 = **1.00** on 20 queries; 0.2–0.6 s vs. 2.4–3.0 s for the 8-way fan-out. Use it for open (all-car / many-car) questions and post-filter by car in code. A `WHERE` on the vector query is still forbidden |

### ⇒ Required retrieval pattern ("filter-then-partition-fanout")

1. **Resolve metadata filters to car ids first** with an exact, non-vector query on `cars`:
   ```sql
   SELECT VALUE c.car_id FROM c WHERE c.price_ils <= @max_price AND ARRAY_CONTAINS(@powertrains, c.powertrain)
   ```
2. **Vector search per car partition, in parallel** (`asyncio.gather`), with no `WHERE` on the vector query:
   ```python
   sql = (
       "SELECT TOP {k} c.id, c.car_id, c.section_type, c.heading, c.header, c.text_he, c.text_en, c.url, "
       "VectorDistance(c.embedding, @qv) AS score FROM c ORDER BY VectorDistance(c.embedding, @qv)"
   )
   rows = [
       r
       async for r in chunks.query_items(
           sql.format(k=k_per_car), parameters=[{"name": "@qv", "value": qv}], partition_key=car_id
       )
   ]
   ```
   `k` is an int literal (TOP can't be parameterized). Use `k_per_car = 6` for single-car questions and 3–4 for multi-car ones.
3. Merge all rows and sort by `score` (cosine **similarity**, higher is better). Apply chunk-level filters such as `section_type` **in Python** after over-fetching.
4. BM25 (in-app) runs over the same candidate car set. Fuse with RRF (see the `autochat-agent` skill).

With 8 cars this is at most 8 small queries. It is correct on both the emulator and Azure. On Azure, this one function can be swapped for a single query with `WHERE` + `FullTextScore` + `RRF`.

## Container definitions (database `autochat`)

| Container | PK | Policy |
|---|---|---|
| `cars` | `/car_id` | default indexing |
| `chunks` | `/car_id` | vector `/embedding` float32 cosine `EMBED_DIMENSIONS`; `vectorIndexes: diskANN`; exclude `/embedding/*` |
| `checkpoints` | `/partition_key` | created by `CosmosDBSaver` |

Vector policies are **immutable**. After changing dims or the path, run `cosmos_admin.py reset` and then re-run ingestion `load`.

## Python SDK patterns

```python
from azure.cosmos.aio import CosmosClient

client = CosmosClient(settings.cosmos_endpoint, credential=settings.cosmos_key)  # one per process
db = client.get_database_client(settings.cosmos_database)
chunks = db.get_container_client("chunks")
await chunks.upsert_item(doc)
car = await db.get_container_client("cars").read_item(item=car_id, partition_key=car_id)  # cars use id == car_id
await client.close()  # in lifespan shutdown
```

Checkpointer (FastAPI lifespan):
```python
from contextlib import AsyncExitStack
from langchain_azure_cosmosdb import CosmosDBSaver

stack = AsyncExitStack()
saver = await stack.enter_async_context(
    CosmosDBSaver.from_conn_info(
        endpoint=s.cosmos_endpoint,
        key=s.cosmos_key,
        database_name=s.cosmos_database,
        container_name=s.cosmos_checkpoints_container,
    )
)
graph = builder.compile(checkpointer=saver)
# shutdown: await stack.aclose()
```

## Troubleshooting
- `Connection refused :8081`: run `start_emulator.ps1`. Check `docker compose logs cosmos`.
- Fewer results than `TOP k`: you used `WHERE` with a vector `ORDER BY`. Switch to the partition fan-out above.
- HTTP 500 on large result sets: raise `QUERY_BUFFER_SIZE_KB` in `docker-compose.yml` (already 16384). Also project only the needed fields and never `SELECT *` (embeddings are large).
- Dimension mismatch on upsert or query: the embedding length must equal the container's policy dims. Run `reset`.
