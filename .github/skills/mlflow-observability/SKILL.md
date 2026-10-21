---
name: mlflow-observability
description: "Local MLflow 3 tracing, feedback and GenAI evaluation for autochat: setup_mlflow, LangGraph autolog, RETRIEVER spans, session (thread) tagging, user thumbs feedback, mlflow.genai.evaluate with code scorers and Foundry-backed LLM judges, local MLflow UI. Use when touching src/autochat/observability.py, the run_turn wrapper, eval scoring, or when traces/judges misbehave."
---

# mlflow-observability

MLflow **3.17.1** runs fully locally: sqlite store `mlflow.db` at the repo root, UI on `http://localhost:5000`, no Databricks.
The code is in [src/autochat/observability.py](../../../src/autochat/observability.py).

## Scripts and commands (repo root)

| Command | Purpose |
|---|---|
| `scripts/dev.ps1 mlflow` | Starts the MLflow UI and tracking server (`mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000`). |
| `scripts/dev.ps1 smoke-mlflow` | Runs `scripts/mlflow_smoke.py`: a mini LangGraph traced end to end (streaming `run_turn`, RETRIEVER span, session tag, feedback) plus `genai.evaluate` with Foundry judges. 14/14 pass. |
| `scripts/dev.ps1 eval [--judge]` | Golden set → MLflow run in experiment `autochat-eval` (see `autochat-eval` skill). |

Experiments: `autochat` (app and CLI traces), `autochat-eval` (eval runs), and `autochat-smoke`.

## Verified facts (2026-10-17, mlflow 3.17.1, Python 3.13, Windows)

| Fact | Consequence |
|---|---|
| `mlflow.langchain.autolog()` traces async LangGraph, **but manual `@mlflow.trace`/`start_span` spans inside nodes are NOT nested** unless `run_tracer_inline=True`. | `setup_mlflow()` always uses `autolog(run_tracer_inline=True)`. |
| Traces are exported **asynchronously**. `get_trace(id)` right after a run returns `None`, and `log_feedback` raises "trace not found". | Use `mlflow.get_trace(id, flush=True)`. Call `mlflow.flush_trace_async_logging()` before `log_feedback` (done in `log_user_feedback`). |
| Judges accept `model="openai:/<deployment>"` and read `OPENAI_API_KEY`/`OPENAI_BASE_URL`. Foundry `/openai/v1` is OpenAI-compatible. litellm is **not** needed (and not installed). | `judge_model_uri()` bridges `FOUNDRY_*` → `OPENAI_*` (setdefault) and defaults to `FOUNDRY_CHAT_DEPLOYMENT` (`MLFLOW_JUDGE_DEPLOYMENT` overrides; the fast model misjudged Hebrew refusals). |
| Verified working judges with gpt-4.1-mini: `RelevanceToQuery`, `RetrievalGroundedness`, `Correctness` (needs `expected_facts`/`expected_response`), `Guidelines(name=, guidelines=)`. | Used in `run_eval.py --judge`. |
| `RetrievalGroundedness`/`RetrievalRelevance` read **top-level RETRIEVER spans** (RETRIEVER spans nested in another RETRIEVER span are ignored) whose output is `list[Document]` (`page_content`, `metadata.doc_uri`). | The executor node wraps the whole plan in `obs.evidence_span()` (`plan_evidence`), output = `obs.evidence_documents(selection_notes, final_context)`: catalog query results first (what makes "cheapest" true), then the numbered excerpts. The per-search `hybrid_retrieve` spans nest inside it. |
| `mlflow.genai.evaluate` defaults (`MLFLOW_GENAI_EVAL_MAX_WORKERS=10` × `MAX_SCORER_WORKERS=10`) with 4 LLM judges: Foundry throttling + sqlite "database is locked" → every judge times out after 300 s and the metric means are computed from 1–3 rows. | `run_eval.py` sets 4 × 2 (setdefault) and calls `log_to_mlflow` after `asyncio.run()` returns. Always check the "Failure summary" line, not only the means. |
| Guidelines judges on a trace see the **whole root-span output** (sources JSON, steps). | Wrap them in a `@scorer` that passes `inputs={"message": ...}, outputs=answer_text`. A `@scorer` returning `None` logs no assessment (used to skip relevance/groundedness on abstain cases). |
| `mlflow.update_current_trace(session_id=..., user=..., tags=...)` works and is searchable: `search_traces(filter_string="metadata.\`mlflow.trace.session\` = 'x'")`. | session = LangGraph `thread_id`, so multi-turn conversations group in the UI's Sessions view. |
| `mlflow.log_feedback(trace_id, name, value, rationale, source=AssessmentSource("HUMAN", user))` works. | `POST /feedback` → `log_user_feedback()`; the UI shows thumbs per trace. |
| Code scorers (`@scorer`) can return `list[Feedback]` and receive nested dict `expectations`. | The `golden` scorer emits `check_*` and `golden_pass` (selftest verifies `golden_pass/mean=0.5`). |
| `genai.evaluate(data=[{"trace": t, "expectations": {...}}])` scores **pre-captured traces**, with no `predict_fn`. | The async graph is run by our own runner; MLflow only scores. This avoids the sync `predict_fn` + async Cosmos loop problem. |
| The sqlite store stays open for the whole process (file locked on Windows). | Temp stores need `TemporaryDirectory(ignore_cleanup_errors=True)`. |
| Telemetry is on by default. | `MLFLOW_DISABLE_TELEMETRY=true` (set by `setup_mlflow` and `dev.ps1 mlflow`). |
| A `from __future__ import annotations` module with a LangGraph `TypedDict` state must import its annotation types at **module scope**. | Otherwise you get `NameError` in `StateGraph(...)`. |

## Integration contract (Phase 4/5)

```python
# autochat/agent/runtime.py
import mlflow
from mlflow.entities import SpanType
from autochat import observability as obs


async def run_turn(graph, *, thread_id: str, message: str, user: str | None = None, on_token=None) -> dict:
    """The ONE entry point for CLI, API (/chat and /chat/invoke) and eval. Root span = chat_turn."""
    with mlflow.start_span(name="chat_turn", span_type=SpanType.AGENT) as span:  # start_span: graph not serialized
        span.set_inputs({"message": message, "thread_id": thread_id})
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
        final: dict = {}
        async for mode, chunk in graph.astream(
            {"messages": [HumanMessage(message)]}, config, stream_mode=["messages", "values"]
        ):
            if mode == "messages":
                msg, meta = chunk
                if on_token and msg.content and meta.get("langgraph_node") in STREAM_NODES:
                    on_token(msg.content)
            else:
                final = chunk
        result = {"answer": ..., "sources": [...], "profile": ..., "intent": final.get("intent")}
        obs.tag_current_turn(thread_id, user=user, intent=result["intent"], language=final.get("language"))
        span.set_outputs(result)
        return {**result, "trace_id": span.trace_id}
```
When `AUTOCHAT_TRACING=false`, `mlflow.tracing.disable()` makes `start_span` a no-op (`trace_id` may be `None`; callers must tolerate that).

- **Retrieval**: wrap the fused result in `@mlflow.trace(span_type=SpanType.RETRIEVER, name="hybrid_retrieve")` returning `obs.retriever_documents(chunks)`. Keep the raw chunks in state separately.
- **FastAPI**: call `setup_mlflow()` in lifespan. The SSE `done` event carries `{"trace_id": ...}`; `/chat/invoke` returns `trace_id`. `POST /feedback {trace_id, value: bool, comment?}` → `obs.log_user_feedback`.
  For `/chat`, run `run_turn(..., on_token=queue.put_nowait)` in a task and drain the queue into SSE, so the span context stays in one task (design; verify in Phase 5 that the trace is complete).
- **Streamlit**: show 👍/👎 (`st.feedback("thumbs")`) under each answer → `POST /feedback` with that answer's `trace_id`.
- **Never** put secrets in span inputs or tags. Span inputs are the message and thread_id only.

## What to look at in the UI
- **Traces** tab: per-turn tree `chat_turn → LangGraph nodes → analyze_turn LLM call → plan_evidence (RETRIEVER) → tools → hybrid_retrieve → generate LLM`, with latency and tokens per span.
- **Sessions**: whole conversations by `thread_id`, useful to debug preference inference across turns.
- **Evaluation runs** (experiment `autochat-eval`): compare runs side by side (`golden_pass/mean`, per-check means, judge means, `pass_rate_<type>`, `latency_p50_s`). Click a failed row to see its trace.

## Troubleshooting
| Symptom | Fix |
|---|---|
| Retriever span missing in the trace | `autolog(run_tracer_inline=True)`; make sure the retriever function is called inside the node, not in a thread pool. |
| `get_trace` returns `None` / feedback "trace not found" | Flush (see above). |
| Judge 401/404 | `OPENAI_BASE_URL` must be the `/openai/v1` endpoint; the deployment name goes after `openai:/`. A stray `OPENAI_API_KEY` already in the shell wins (setdefault). |
| `retrieval_groundedness` errors | The trace has no RETRIEVER span (e.g. abstain or general_chat turn). `run_eval.py` skips those rows (no assessment). |
| Every judge "timed out after 300 seconds" | Concurrency too high for Foundry + sqlite; see the eval concurrency row above. |
| UI empty | The server must point at the same store: run `dev.ps1 mlflow` from the repo root (it uses `sqlite:///mlflow.db`). |
