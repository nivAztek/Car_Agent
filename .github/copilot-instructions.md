# Copilot instructions — autochat (Automotive Review Chatbot POC)

Take-home POC: a grounded chatbot that helps users understand and compare 8 cars from Hebrew road-test articles on auto.co.il. The task brief is in [task.md](../task.md) and the architecture and phase plan are in [PLAN.md](../PLAN.md). **Read PLAN.md before any non-trivial change.**

## Stack (fixed — do not swap without asking)
- **LangGraph** for the conversation graph, **FastAPI** for the API (SSE streaming), **Streamlit** for the UI.
- **Azure Cosmos DB for NoSQL vNext emulator** (Docker, `http://localhost:8081`) stores documents, vectors, and LangGraph checkpoints.
- **Microsoft Foundry** provides the chat model and `text-embedding-3-large` (1536 dims) through `langchain-azure-ai`.
- **MLflow 3** (local sqlite `mlflow.db`, UI on :5000) handles tracing, user feedback and GenAI evaluation. Judges are Foundry deployments via `openai:/<deployment>`.
- Parsing: `selectolax` site adapter, with a `trafilatura` generic fallback. **No OCR and no LLM-as-HTML-parser**: the data is text in the DOM (verified), and the LLM only structures clean text.
- Python 3.13 in `.venv`. Install packages with `python -m uv pip install --python .venv\Scripts\python.exe <pkg>`. Run with `.venv\Scripts\python.exe`. Shell is **Windows PowerShell 5.1**: chain commands with `;`, never `&&`.

## Hard constraints (verified; don't rediscover)
1. **The emulator has NO full-text / BM25 search.** Never write `FullTextScore`, `FullTextContains`, or `ORDER BY RANK RRF(...)` against it. Hybrid search = Cosmos `VectorDistance` + in-app `rank_bm25` + RRF in Python. Keep native hybrid search only behind the `Retriever` interface as the Azure swap path.
2. **Never put `WHERE` on a vector query.** On the emulator the filter runs *after* the approximate top-K: `TOP 5 ... WHERE car_id=X` returned 0 rows (verified). Instead, resolve metadata filters to `car_ids` on the `cars` container, then fan out one vector query per car with `partition_key=car_id` (exact), merge, and post-filter in Python.
3. **Vector policies are immutable.** Changing `EMBED_DIMENSIONS` or the vector path requires dropping and recreating the containers (`scripts/dev.ps1 reset-db`).
4. **The emulator only optimizes DiskANN vector indexes.** Range and composite indexes are accepted but ignored. That's fine at about 200 docs. Don't add `flat` or `quantizedFlat` (unsupported).
5. **Articles are Hebrew.** Store `text_he` and an LLM-made `text_en` for each chunk, embed both, and answer in the user's language (`he`/`en`). Render Hebrew with per-block bidi CSS (`unicode-bidi: plaintext`, see `ui/components.py`).
6. **Grounding:** answers use only retrieved chunks and `cars` records, with `[n]` citations that map to (car, section, url). If a car or fact is not in the corpus, say so. Never invent specs.
7. **Structured beats generated:** spec-table values (price, range, 0-100, ...) override LLM-extracted values. Every LLM-extracted field needs an evidence quote or must be `null`. Kia EV9 has **no** spec table (it's a long-term report), so all its fields come from the LLM.
8. **Secrets** come only from `.env` through `autochat.config.Settings`. Never hard-code keys and never print them. The emulator key in `.env.example` is the public, well-known key.

## Layout
```
src/autochat/{config.py, llm.py, observability.py, db/, ingestion/{sources/,...}, retrieval/, agent/, api/, ui/}
data/urls.txt  data/raw/*.html (cached)  data/processed/{cars.json, chunks.jsonl}
eval/golden.jsonl  scripts/dev.ps1  tests/
```

## Workflows → use the skills in `.github/skills/`
| Task | Skill |
|---|---|
| Scraping, parsing, extraction, chunking, translation, embedding, loading | `autochat-ingestion` |
| Start/reset emulator, container policies, vector query patterns | `cosmos-emulator` |
| Foundry chat/embedding clients, structured output, connectivity checks | `foundry-models` |
| LangGraph state/nodes/graph, FastAPI endpoints, Streamlit UI | `autochat-agent` |
| Golden set, retrieval and answer evaluation, example transcripts | `autochat-eval` |
| MLflow tracing, `run_turn` wrapper, feedback, judges, MLflow UI | `mlflow-observability` |

## Everyday commands (`scripts/dev.ps1 <task>`)
`up` / `down` / `logs` (whole stack in Docker: cosmos, init, mlflow, api, ui from one `Dockerfile`; ports 8501/8000/5000, so stop it before running the local `api`/`ui`/`mlflow`) · `setup` · `emulator` · `smoke` (Cosmos + Foundry + MLflow) · `probe` · `ingest` · `reset-db` · `api` · `ui` · `mlflow` · `test` · `eval [--judge]` · `lint`

## Test order (always: local first, Docker last)
1. **Local terminal first.** Develop and verify every change in the `.venv`: `scripts/dev.ps1 lint` and `test`, then `emulator` (starts only the `cosmos` service) plus `smoke`, `chat`, `retrieve` or `eval` against the local `api`/`ui`. Debug here, never inside the containers.
2. **Docker only after the local checks pass and the feature is finished.** Then `scripts/dev.ps1 up` (rebuilds the image) and re-run the relevant checks against :8000/:8501 (`/health`, one chat turn, `eval`). A failure that only appears in Docker is a packaging or config problem (paths, env overrides, ports, healthchecks), so fix it in `Dockerfile`/`docker-compose.yml`, not in the app logic.
3. The local stack and the Docker stack share ports 8000/8501/5000. Run `scripts/dev.ps1 down` before starting the local `api`/`ui`/`mlflow`, and stop the local ones before `up`.

## Conventions
- Type-hinted, async-first (`azure.cosmos.aio`, `ainvoke`/`astream`), Pydantic v2 models for every boundary (LLM structured output, API, Cosmos docs).
- One `CosmosClient` singleton per process. Parameterized SQL only, with no f-strings on user input. `TOP {k}` must be an int literal, and the parameterized `@qv` vector works on the emulator.
- Keep LLM calls few. `analyze_turn` is a single structured call per user turn.
- Every chat turn goes through `autochat.agent.runtime.run_turn()` (MLflow trace, session = `thread_id`).
- Keep changes small and runnable. After each phase, run `scripts/dev.ps1 test`, plus `smoke` or `eval` where relevant.
- Don't create extra markdown docs. README.md is the only submission doc (approach, architecture, trade-offs, example interaction).
