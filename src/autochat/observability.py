"""MLflow 3 tracing + evaluation helpers for autochat (fully local, sqlite tracking store).

Verified with mlflow 3.17.1 (see .github/skills/mlflow-observability/SKILL.md):
- `mlflow.langchain.autolog(run_tracer_inline=True)` is REQUIRED so manual `@mlflow.trace` spans
  (retriever, turn wrapper) nest inside async LangGraph `ainvoke`/`astream` traces.
- Traces are exported asynchronously: flush before reading a trace back or logging feedback on it.
- LLM judges use `openai:/<deployment>` + OPENAI_API_KEY/OPENAI_BASE_URL -> Foundry `/openai/v1` (no litellm).

Everything here is a no-op when AUTOCHAT_TRACING=false.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKING_URI = f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}"
DEFAULT_EXPERIMENT = "autochat"

_state = {"configured": False, "enabled": False}


def tracing_enabled() -> bool:
    return os.environ.get("AUTOCHAT_TRACING", "true").strip().lower() in {"1", "true", "yes", "on"}


def setup_mlflow(
    experiment: str | None = None,
    *,
    tracking_uri: str | None = None,
    autolog: bool = True,
) -> bool:
    """Configure MLflow once per process. Returns True if tracing is active.

    Call from FastAPI lifespan, the chat CLI and the eval runner *before* building the graph.
    """
    import mlflow
    from dotenv import load_dotenv

    # MLflow + judge bridge read os.environ (pydantic-settings does not export); never override real env vars
    load_dotenv(REPO_ROOT / ".env", override=False)
    if not tracing_enabled():
        mlflow.tracing.disable()
        _state.update(configured=True, enabled=False)
        return False
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    mlflow.set_tracking_uri(tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI)
    mlflow.set_experiment(experiment or os.environ.get("MLFLOW_EXPERIMENT_NAME") or DEFAULT_EXPERIMENT)
    if autolog and not _state["configured"]:
        mlflow.langchain.autolog(run_tracer_inline=True, silent=True)
    _state.update(configured=True, enabled=True)
    return True


def is_enabled() -> bool:
    return bool(_state["enabled"])


def judge_model_uri(
    *,
    endpoint: str | None = None,
    api_key: str | None = None,
    deployment: str | None = None,
) -> str:
    """Return an MLflow judge model URI that routes to Foundry.

    MLflow's native OpenAI provider reads OPENAI_API_KEY / OPENAI_BASE_URL; the Foundry `/openai/v1`
    endpoint is OpenAI-compatible, so we set those (only if not already set) from FOUNDRY_* values.
    Defaults to the chat deployment: the fast model misjudged Hebrew answers (it invented claims in a
    one-line refusal), and a judge should be at least as strong as the model it grades.
    """
    endpoint = (endpoint or os.environ.get("FOUNDRY_ENDPOINT", "")).rstrip("/")
    api_key = api_key or os.environ.get("FOUNDRY_API_KEY", "")
    deployment = (
        deployment
        or os.environ.get("MLFLOW_JUDGE_DEPLOYMENT")
        or os.environ.get("FOUNDRY_CHAT_DEPLOYMENT")
        or os.environ.get("FOUNDRY_FAST_DEPLOYMENT")
    )
    if not (endpoint and api_key and deployment):
        raise RuntimeError("Judge needs FOUNDRY_ENDPOINT, FOUNDRY_API_KEY and a judge/chat/fast deployment.")
    os.environ.setdefault("OPENAI_API_KEY", api_key)
    os.environ.setdefault("OPENAI_BASE_URL", endpoint)
    return f"openai:/{deployment}"


def tag_current_turn(thread_id: str, *, user: str | None = None, **tags: Any) -> None:
    """Attach session (= LangGraph thread_id), user and tags (intent, language, ...) to the active trace."""
    if not is_enabled():
        return
    import mlflow

    clean = {k: str(v) for k, v in tags.items() if v is not None}
    try:
        mlflow.update_current_trace(session_id=thread_id, user=user, tags=clean or None)
    except Exception:  # noqa: BLE001 - observability must never break a chat turn
        pass


def last_trace_id() -> str | None:
    if not is_enabled():
        return None
    import mlflow

    return mlflow.get_last_active_trace_id()


def log_user_feedback(trace_id: str, value: bool, *, comment: str | None = None, user: str = "anonymous") -> None:
    """Thumbs up/down from the UI -> human assessment on the trace (visible in the MLflow UI)."""
    if not is_enabled():
        return
    import mlflow
    from mlflow.entities import AssessmentSource

    mlflow.flush_trace_async_logging()  # the trace may still be in the async export queue
    mlflow.log_feedback(
        trace_id=trace_id,
        name="user_feedback",
        value=value,
        rationale=comment,
        source=AssessmentSource(source_type="HUMAN", source_id=user),
    )


def retriever_documents(chunks: Iterable[Mapping[str, Any]]) -> list[Any]:
    """Convert retrieved chunk dicts into LangChain Documents for a RETRIEVER span output.

    RetrievalGroundedness / RetrievalRelevance read `page_content` + `metadata.doc_uri` from RETRIEVER spans.
    """
    from langchain_core.documents import Document

    docs = []
    for c in chunks:
        text = c.get("text_en") or c.get("text") or c.get("text_he") or ""
        meta = {
            "doc_uri": c.get("url", ""),
            "chunk_id": c.get("id", ""),
            "car_id": c.get("car_id", ""),
            "section_type": c.get("section_type", ""),
        }
        docs.append(Document(page_content=text, metadata=meta))
    return docs


@contextlib.contextmanager
def evidence_span(inputs: dict[str, Any]) -> Iterator[Any]:
    """Top-level RETRIEVER span around the plan executor. The per-search `hybrid_retrieve` spans nest inside it,
    so RetrievalGroundedness judges the answer against everything the answer model saw (catalog selections +
    excerpts) instead of one search at a time. Yields None outside an active trace."""
    try:
        import mlflow
        from mlflow.entities import SpanType
    except ImportError:  # pragma: no cover
        yield None
        return
    if mlflow.get_current_active_span() is None:
        yield None
        return
    with mlflow.start_span(name="plan_evidence", span_type=SpanType.RETRIEVER) as span:
        span.set_inputs(inputs)
        yield span


def evidence_documents(selections: Iterable[str], chunks: Iterable[Mapping[str, Any]]) -> list[Any]:
    """Catalog query results (exact SQL over all reviewed cars: what makes 'cheapest' or 'quickest' true) as
    documents, followed by the numbered excerpts."""
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content=f"Catalog query over all reviewed cars (exact database result): {s}",
            metadata={"doc_uri": "catalog://cars", "chunk_id": f"selection-{i}"},
        )
        for i, s in enumerate(selections, 1)
    ]
    return docs + retriever_documents(chunks)
