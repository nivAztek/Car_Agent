---
name: autochat-agent
description: "Design and implement the autochat LangGraph conversation graph (state, nodes, routing, preference inference, hybrid retrieval fusion, grounded generation), the FastAPI SSE API and the Streamlit UI. Use when working in src/autochat/agent, src/autochat/retrieval/hybrid.py, src/autochat/api or src/autochat/ui, or when debugging a conversation."
---

# autochat-agent

## Scripts (repo root)

| Script | Purpose |
|---|---|
| `scripts/chat_cli.py [--api URL] [--script file.txt] [--thread ID]` | Chat in the terminal. In-process graph by default, or `--api` to hit FastAPI SSE. `--script` replays one user line per line (useful for transcripts). Prints sources and the profile after each turn. |
| `scripts/graph_mermaid.py` | Prints the compiled graph as Mermaid, for the README and for checking wiring. |
| `scripts/retrieval_check.py [--k 5] [--show 3] [-q "query"]` (`dev.ps1 retrieve`) | Live check: 7 metadata-filter cases (exit 1 on failure) + 14 queries, hit@k for vector-only / BM25-only / hybrid. Baseline (2026-10-17): vector 14/14, BM25 11/14, hybrid 14/14. **Re-run after changing retrieval, chunking or the tokenizer.** |

## Contracts the code must expose (as built)
- `autochat.agent.graph.build_graph(checkpointer) -> CompiledStateGraph`, with `context_schema=AgentContext`. `STREAM_NODES = {generate, respond_direct, ask_clarifying}`.
- `autochat.agent.runtime.get_runtime(*, checkpointer=None)`: an async context manager that yields `(graph, ctx: AgentContext)`. It builds a `CosmosCarCatalog` (warms its cached `scope()`) and the chunk store, builds the Foundry clients, enters `CosmosDBSaver.from_conn_info` (falls back to `InMemorySaver` with a warning), and closes the Cosmos client on exit.
- `autochat.agent.runtime.run_turn(graph, *, thread_id, message, user=None, on_token=None, context=None) -> {answer, sources, profile, intent, language, steps, trace_id}`: the **only** way to run a turn. It opens the MLflow `chat_turn` AGENT span only when tracing is enabled and passes `context=ctx` to `astream`. It forwards only `AIMessageChunk` tokens from STREAM_NODES; the final `AIMessage` a node writes to state is streamed by LangGraph too and must be skipped, or the answer prints twice. `on_token` may be sync or async. `steps` holds the executor notes.
- Call `autochat.observability.setup_mlflow()` once per process before building the graph.

## Files
`state.py` (AgentState, TURN_RESET, AgentContext) · `profile.py` (UserProfile, ProfileDelta + `evidence`, `validate_delta`, `merge_profile`, `needs_clarification`, `next_missing_slot`) · `schemas.py` (TurnAnalysis, PlanStep, Source, MAX_STEPS=4) · `tools.py` (StructuredTools, `rank_cars`, `execute_plan`, `relax_plan`, `resolve_plan`, `useful_query`, `default_plan`, `ensure_search_step`) · `prompts.py` · `nodes.py` · `graph.py` · `runtime.py`. Outside the agent: `autochat/names.py` (normalize, `car_name_keys`), `retrieval/catalog.py` (`CarCatalog`: `get`, `find`, `by_name_keys`, `by_makes`, `scope`, `page`; `CosmosCarCatalog`, `InMemoryCarCatalog`, `MAX_CANDIDATES=50`), `retrieval/resolver.py` (`resolve_mentions` → resolved / ambiguous / unknown + same-make suggestions; `ref_key`, `is_focus_ref`). Tests: `tests/test_agent.py`, `tests/test_catalog_resolver.py` (fakes: retriever, `InMemoryCarCatalog(CARS)`, planner LLM via RunnableLambda, `GenericFakeChatModel` for streaming).

## State
Persistent across turns: `messages` (add_messages, **never deleted**), `summary`, `summarized_upto`, `user_profile` (dict), `clarified`, and `focus_car_ids` (cars cited in the last answer, in citation order, ≤ `MAX_FOCUS=6`; the planner refers to them as `#1`, `#2`, …).
Per turn (`analyze_turn` returns `**TURN_RESET`): `language`, `intent`, `standalone_query`, `mentioned_car_ids`, `unknown_cars`, `unknown_suggestions`, `ambiguous_cars`, `car_refs` (mention / `#n` → car_id), `profile_delta`, `profile_dropped`, `plan`, `plan_trace`, `exec_notes`, `candidate_car_ids`, `retrieved` (chunk dicts in citation order), `answer`, and `sources` (only the cited `[n]`).
Dependencies are not in state: they live in `AgentContext(retriever, catalog, fast_llm, answer_llm)`, accessed via `runtime.context`. **No prompt ever contains the car list**; prompts get a fixed-size `{scope}` line (count, makes, powertrains, price range) and the small focus list.

`UserProfile`: `budget_max_ils, budget_min_ils, use_cases[], passengers, powertrain_pref[], body_pref[], priorities[] (most important first), must_haves[], dealbreakers[], home_charging, cars_liked[], cars_rejected[]`.
`merge_profile(old, delta)`: scalars are replaced only when the new value is not None. Lists are unioned, or replaced when listed in `replace_lists`. New priorities move to the front. A car moves between liked and rejected.
**`validate_delta(delta, message, resolve_car)`** runs before the merge (liked/rejected cars arrive as names and are resolved through the turn's `car_refs`):
- Each field needs an `evidence` quote that is ≥80% verbatim from the last user message.
- `FIELD_CUES` require an explicit cue word for powertrain, body and liked/rejected cars: "I can charge at home" is not an EV preference, and "compare X and Y" is not liking them.
- `replace_lists` needs a change-of-mind cue (`CHANGE_CUE`).
- Liked/rejected cars must be among this turn's mentioned cars.

Verified live: gpt-4.1-mini copies the known profile into the delta and over-infers. The validator removes this (gpt-5.5 is clean but adds ~4 s per turn).

## Graph (planner → code executor)

```
START → analyze_turn → update_profile → route (pure)
  general_chat | out_of_scope | ambiguous cars | (unknown cars only, not recommend) → respond_direct
  recommend & no mentioned cars & not clarified & needs_clarification(profile) → ask_clarifying (sets clarified)
  otherwise → execute_plan → generate
respond_direct | ask_clarifying | generate → compact_history → END
```

- **analyze_turn** makes one `fast_llm.with_structured_output(TurnAnalysis)` call.
  - Inputs: scope line, focus list (`#n` → name), profile summary, summary, and the unsummarized history. Output includes `car_mentions` (verbatim names) and `focus_refs`; plan steps reference cars by name or `#n`.
  - Names are resolved in code by `resolve_mentions` (exact `name_keys` → fuzzy within the named make only, same digits → unknown). An unknown model is never mapped to a same-make car ("Audi Q5" stays unknown and gets the RS3 as a suggestion). Several close matches → `ambiguous_cars` → one question in `respond_direct`.
  - Language comes from `detect_language()` (Hebrew vs Latin character ratio), not from the model.
  - On LLM failure it degrades to `car_question` + the raw message.
- **execute_plan** makes no LLM calls:
  1. `resolve_plan` (car refs → ids via `car_refs`, make filters canonicalized via `scope().make_vocab()`, degenerate queries dropped by `useful_query`, ≤4 steps).
  2. `default_plan` if the plan is empty.
  3. Inject `mentioned_car_ids` if search steps have none.
  4. `ensure_search_step`.
  5. Run the plan.
  6. If any step is empty: `relax_plan` (find_cars: budget +10%, drop range/0-100/tags/body; rank: `relax=True` ignores profile constraints; search: drop `section_types`), then run once more.
  7. `order_context` groups excerpts by car in plan order, cards kept, cap 14.
- **Tools** (`make_tools(...)` returns StructuredTools, which show up as TOOL spans):
  - `find_cars` returns `{car_id, label}` (SQL filter + `ORDER BY` + `TOP` in Cosmos, capped at `MAX_CANDIDATES`).
  - `rank_by_profile` with no candidates first calls `catalog.find(profile_filters(profile))`, then uses hard constraints (budget ×1.1, powertrain, body, rejected) and the score `Σ w_i·aspect` (weights 1.0/0.7/0.5/…; priorities first, then `USE_CASE_PRIORITIES`; unknown aspect = 3), +0.5 per use-case tag and +1 if liked.
  - `search_reviews` returns chunk dicts (with cards).
  - A car-selecting step that comes back empty stops the plan (`needs_replan`).
- **generate** uses the chat model and streams. Inputs: the numbered context in the answer language, planner notes (internal), the last 4 messages, the summary, and `FOLLOWUP_HINT` (recommend only, `next_missing_slot`) and `UNKNOWN_CARS_HINT`. `build_sources` parses `[n]`, `[1, 3]` and `[2-4]`.
- **compact_history** is a no-op unless the window since `summarized_upto` has more than 16 messages or ~6k tokens. It then summarizes everything except the last 6 messages into `summary` (fast model). A failure is logged and retried on the next turn.

## Hybrid fusion (in-app) — implemented in `src/autochat/retrieval/`

API (as built):
- `filters.SearchFilters` (car_ids, makes, powertrains, body_types, min/max_price_ils, min_seats, min_range_km, max_accel_0_100_s, tags) → `build_query()` → `find_cars(filters, sort_by=, descending=, limit=)` returns plain car dicts (exact SQL on `cars`, sort in Python, unknown values last). `"ice"` expands to `ice+mhev`, `"hybrid"` to `hybrid+mhev+phev`, `"suv"` to `suv+crossover`. Price, 0-100, make, powertrain and body type are strict on null. Seats and range are lenient (no review states seats).
- `bm25.ChunkStore.from_cosmos()` holds all chunk docs (no embeddings) plus the BM25 index. When the search targets specific cars, BM25 **drops those cars' name tokens** from the query; otherwise name-stuffed FAQ answers outrank the relevant section.
- `vector.vector_search(qv, car_ids, k_per_car)` returns only `(id, score)`; hits are hydrated from the store.
- `hybrid.HybridRetriever(store).retrieve(query, car_ids=None, *, section_types=None, top_n=None, max_per_car=None, include_cards=None, query_vector=None) -> list[RetrievedChunk]`. Defaults: k_per_car 6/4/3, top_n 8 (1 car) else min(12, 4n), max_per_car 4 (≤3 cars) / 3. `include_cards` (default when ≤4 cars) appends missing `car_card` chunks with `pinned=True`. Unknown car ids are dropped. A `section_types` filter that empties the list falls back to unfiltered.
- The RETRIEVER span (`hybrid_retrieve`) is opened **only inside an active trace**, so scripts and tests make no stray traces.

```python
def rrf(rank_lists: list[list[str]], weights: list[float], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranks, w in zip(rank_lists, weights, strict=True):
        for r, doc_id in enumerate(ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + r + 1)
    return scores


# weights: vector 2.0, bm25 1.0 ; then cap max_per_car and take top N (8 for single car, 12 for compare)
```
BM25: `rank_bm25.BM25Okapi` over `tokenize(header + text_en + text_he)`. Build it once at startup from Cosmos (`SELECT c.id, c.car_id, c.header, c.text_en, c.text_he FROM c`). Tokenizer: lowercase, split on `\W+`, and strip single Hebrew prefix letters `[והבלמשכ]` from tokens longer than 3 characters.

## Prompts (agent/prompts.py)
- `ANALYZE_SYSTEM`: intent definitions, profile rules (explicit only, evidence quotes, never copy the known profile), and plan rules:
  - recommend: `[find_cars] → rank_by_profile(3) → search_reviews`.
  - compare and named cars: search with `car_ids` and `use_previous=false`.
  - Catalog superlatives cover the whole catalog, with no profile filters.
  - Every plan ends with search.
- `ANSWER_SYSTEM` = `GROUNDING_RULES` (excerpts only, `[n]`, specs from the Car card, "reviews do not cover", user's language, never mention planner notes) + `INTENT_INSTRUCTIONS[intent]` (compare: verdict line, then a table, then trade-off bullets).
- `DIRECT_SITUATIONS`: for out-of-scope and unknown cars the reply must say "not covered by the reviews". The eval's abstain regex depends on this wording.
- `SUMMARY_SYSTEM`: car-shopping-focused, ≤150 words.

## FastAPI (api/main.py, api/schemas.py) — as built
- `create_app(runtime_factory=None, *, tracing=True, health_checks=None)`; `app = create_app()` for uvicorn. Lifespan enters `get_runtime()` once and stores `graph`, `ctx`, `busy` (thread_ids with a running turn) and `tasks` on `app.state`.
- `POST /chat` with body `{thread_id, message, user?}` returns SSE via `sse_starlette.EventSourceResponse` (ping 15 s):
  `event: token data: {"t": "..."}` … `event: sources data: [...]` · `event: profile data: {...}` · `event: done data: {"trace_id": "tr-...", "intent", "language", "steps"}`. On failure: `event: error data: {"detail": <generic text>}`.
  Implementation: `start_turn` reserves the thread (409 if busy), runs `run_turn(..., on_token=queue.put_nowait)` in its own task and the SSE generator drains the queue. The task is never cancelled by a client disconnect, so checkpoints are never half-written.
- `POST /chat/invoke` returns `ChatResponse` `{answer, sources, profile, intent, language, steps, trace_id}` (non-streaming; used by eval). The turn task is `asyncio.shield`ed.
- `POST /feedback` with body `{trace_id, value: bool, comment?, user?}` calls `observability.log_user_feedback` in a thread and returns 204. Returns 404 for an unknown trace and 503 when tracing is off.
- `GET /threads/{id}/profile` returns a `UserProfile` (defaults for unknown threads). `GET /threads/{id}/history` returns `{thread_id, summary, messages:[{role, content}]}`.
- `GET /cars?q=&offset=&limit=` (`CarSummary` page sorted by make/model, `limit` ≤ 200, total in `X-Total-Count`), `GET /cars/{car_id}` (`catalog.get`; full record minus `adapter`, `source_sha`, `dropped`, `name_keys`, `_*`), `GET /health` (Cosmos + Foundry checks → 200/503; `cars` = scope count).
- `thread_id` pattern `^[A-Za-z0-9._-]{1,128}$` on the body and path (CosmosDBSaver rejects its key separator). JSON events use `ensure_ascii=False` (Hebrew).
- Tests: `tests/test_api.py` with `create_app(fake_factory, tracing=False, health_checks={...})`. The factory yields `(build_graph(InMemorySaver()), make_ctx(...))` from `tests/fakes.py`.

## Streamlit (src/autochat/ui) — as built
- `client.py`:
  - `ApiClient(base_url, http=None)` with `chat()` → `ChatStream`, plus `cars()`, `profile()`, `history()` and `feedback()`.
  - `parse_sse()`; `ApiError` / `ThreadBusyError` (409).
  - `default_client()` reads `API_BASE_URL`; tests monkeypatch it.
  - `ChatStream` yields tokens; afterwards `.sources`, `.profile`, `.trace_id`, `.intent`, `.language` and `.steps` are filled.
- `state.py`:
  - `Message` dataclass; `init(client_factory)`; `new_conversation()`; `cars()` (cached per session); queued example prompts.
  - `thread_id` is mirrored to `?thread=`, and a reload resumes from `/threads/{id}/history` + `/profile`.
- `components.py`:
  - `BIDI_CSS`; pure helpers `linkify_citations`, `profile_rows`, `budget_label`, `car_line`, `humanize`.
  - `render_message`, `render_sidebar`, `render_examples`.
- `app.py`: init → sidebar → history → prompt.
  - `answer()` shows a spinner until the first token, then `st.write_stream`.
  - It stores the `Message` and calls `st.rerun()`, so the sidebar profile and feedback widgets render from state.
  - A 409 shows a "still finishing" warning.
- SSE `done` carries `{trace_id, intent, language, steps}`.
- Tests (`tests/test_ui.py`):
  - The client runs against `fakes.fake_api()` (the FastAPI app on fakes, via `TestClient`).
  - Pure formatters are unit-tested.
  - `AppTest` covers the chat turn, resume from the URL, and the API-down error.

## Debug checklist
0. Open the turn's trace in MLflow (`scripts/dev.ps1 mlflow`; the CLI prints `trace: tr-...`).
   - Expected spans: `chat_turn` (AGENT) → `analyze_turn` → `execute_plan` → TOOL spans (`find_cars` / `rank_by_profile` / `search_reviews` → RETRIEVER `hybrid_retrieve`) → `generate` → `compact_history`.
   - Typical turn is 8–20 s: planner 3–9 s, retrieval ~1.5 s, streamed answer 2–12 s, checkpoint writes ~1 s.
   - Turn much slower than its spans add up to? The gap is checkpoint I/O or event-loop stalls (untraced). Check that the emulator uses the named volume (not a bind mount), and time `graph.checkpointer.aput`/`aput_writes`.
1. Wrong intent or plan? Look at the `analyze_turn` span output (`plan`, `profile_delta`, `profile_dropped`), then at `exec_notes` (`out["steps"]` in the CLI result).
2. Is `retrieved` empty? Check the plan's `car_ids`, the find_cars filters (try the same `SearchFilters` in `dev.ps1 retrieve -q`), and the REPLAN notes.
3. Hallucinated a spec? Check that the `car_card` chunk was in context and that the number appears in it.
4. Profile not persisting? Same `thread_id`? Is the checkpointer `CosmosDBSaver` (no "in-memory checkpoints" warning)? Was the field dropped by `validate_delta` (see `profile_dropped`)?
5. Live scripted conversations live in `eval/scripts/*.txt` (`chat_cli.py --script ... --out eval/transcripts/...md`).
