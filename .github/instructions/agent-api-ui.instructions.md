---
applyTo: "src/autochat/agent/**,src/autochat/api/**,src/autochat/ui/**"
description: "LangGraph agent, FastAPI and Streamlit rules"
---
# Agent / API / UI rules

Load the `autochat-agent` skill for the graph contract, prompts, and API schema.

## LangGraph
- State is a `TypedDict` with `messages: Annotated[list[AnyMessage], add_messages]`. Other keys are plain and overwritten per turn, except `user_profile`, which is merged by `profile.merge_profile`.
- Nodes are `async def node(state) -> dict` and return **partial updates only**. No side effects except retrieval reads.
- Run exactly one structured LLM call in `analyze_turn` (the `TurnAnalysis` model) to get intent, standalone query, mentioned cars, filters, preference deltas, and language.
- Routing uses pure functions over state (no LLM). The retrieval correction loop is capped at `retries <= 1`. Set `recursion_limit=25` on invoke.
- Checkpointer: `CosmosDBSaver.from_conn_info(endpoint, key, database_name, container_name)` (async context manager) is entered once in FastAPI lifespan. Fall back to `InMemorySaver` if Cosmos is unavailable, and log a warning.
- `thread_id` comes from the client. A new conversation means a new uuid4.
- Every turn goes through `runtime.run_turn()` (MLflow `chat_turn` span, session = `thread_id`). Never call `graph.astream`/`ainvoke` directly from the API, CLI or eval. Retrieval output is exposed as a RETRIEVER span. Never put secrets in span inputs or tags. See the `mlflow-observability` skill.

## Prompts
- Keep them in `agent/prompts.py` as module constants. The system prompt enforces: grounded-only, `[n]` citations, the user's language, the "not covered by the reviews" fallback, and at most one follow-up preference question.
- Context blocks are numbered: `[n] {make model} — {heading}\n{text in answer language}\n(source: url)`.

## FastAPI
- `POST /chat` returns SSE events `token` (text delta), `sources` (list), `profile` (UserProfile), `done` (`{trace_id}`). `POST /chat/invoke` returns the same payload as JSON, including `trace_id` (non-streaming, used by eval and tests). `POST /feedback {trace_id, value, comment?}` logs a human assessment to MLflow.
- `GET /threads/{id}/profile`, `GET /cars?q=&offset=&limit=` (paged, `X-Total-Count`), `GET /cars/{car_id}`, `GET /health` (checks Cosmos and Foundry).
- Use lifespan for `setup_mlflow()`, the Cosmos client, BM25 index, LLM clients, compiled graph, and checkpointer. Enable CORS for `http://localhost:8501`.

## Streamlit
- Layering: `ui/client.py` (httpx + SSE, no Streamlit) → `ui/state.py` (session state, typed `Message`) → `ui/components.py` (pure formatters + thin renderers, no API calls) → `ui/app.py` (page flow only). Keep that direction of imports.
- Keep `thread_id` in `st.session_state`, mirrored to `?thread=` so a refresh resumes from the checkpoint. Stream the SSE `ChatStream` into `st.write_stream`. Show a "Sources" expander, a "How I answered" expander (steps) and a 👍/👎 `st.feedback` (sends `POST /feedback` with the answer's `trace_id`) for each answer. The sidebar has the inferred profile, the car catalog and a "New conversation" button.
- Hebrew/mixed text: per-block bidi through the static `BIDI_CSS` (`unicode-bidi: plaintext; text-align: start`). Model text is plain Markdown. **Never** render model or user text with `unsafe_allow_html`.
