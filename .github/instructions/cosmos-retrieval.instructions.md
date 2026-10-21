---
applyTo: "src/autochat/db/**,src/autochat/retrieval/**"
description: "Cosmos DB emulator + hybrid retrieval rules"
---
# Cosmos DB & retrieval rules

Load the `cosmos-emulator` skill for policies, query templates, and the smoke test.

- One async `CosmosClient` per process, created in FastAPI lifespan or the CLI `main`, and closed on shutdown. Emulator endpoint is `http://localhost:8081` (HTTP mode, so no TLS settings are needed).
- Container bootstrap is idempotent (`create_container_if_not_exists`):
  - `cars`: PK `/car_id`.
  - `chunks`: PK `/car_id`, vector policy `{path:/embedding, dataType:float32, distanceFunction:cosine, dimensions:EMBED_DIMENSIONS}`, `vectorIndexes:[{path:/embedding, type:diskANN}]`, exclude `/embedding/*` from regular indexing.
  - `checkpoints`: created by `CosmosDBSaver.from_conn_info(...)` (PK `/partition_key`, managed by the library).
- **No full-text functions on the emulator.** Hybrid = vector top-k (Cosmos) + BM25 top-k (in-app) + weighted RRF (Python, k=60, weights vector 2 : bm25 1).
- **Never combine `WHERE` with `ORDER BY VectorDistance`.** On the emulator the filter runs *after* the approximate top-K, so results silently drop to 0–2 rows (verified). Use **filter-then-partition-fanout**:
  1. Run an exact SQL query on `cars` with the `SearchFilters` to get the `car_ids`.
  2. Run one vector query **per car with `partition_key=car_id`** and no `WHERE`, in parallel with `asyncio.gather`.
  3. Merge by score, then apply chunk-level filters (`section_type`) in Python after over-fetching.
- Vector query: `SELECT TOP {k} <projected fields>, VectorDistance(c.embedding, @qv) AS score FROM c ORDER BY VectorDistance(c.embedding, @qv)`. `TOP` must be an int literal. Parameterized `@qv` works on the emulator. Never `SELECT *` (it returns embeddings).
- Build `cars` filters from the `SearchFilters` Pydantic model into `(where_sql, parameters)`. Never interpolate user strings. Use `ARRAY_CONTAINS(@values, c.field)` for multi-value filters.
- Filter fields (`make`, `body_type`, `powertrain`, `price_ils`, `model_year`, `section_type`) are also **denormalized onto every chunk**, for display and for the future Azure native-filter path.
- Diversity: when several cars are targeted, cap at `max_per_car` chunks (default 3), and always include the car's `verdict` chunk when recommending.
- Return `RetrievedChunk` models: `id, car_id, section_type, heading, text_he, text_en, url, vector_score, bm25_score, rrf_score`.
