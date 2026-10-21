"""Process runtime: dependencies + checkpointer + compiled graph, and the traced `run_turn` wrapper.

Every chat turn (API, CLI, eval) goes through `run_turn` -> one MLflow `chat_turn` AGENT span, session = thread_id.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from langchain_core.messages import AIMessageChunk, HumanMessage

from autochat import observability as obs
from autochat.agent.graph import STREAM_NODES, build_graph
from autochat.agent.state import AgentContext

log = logging.getLogger(__name__)

RECURSION_LIMIT = 25
# Modules openai/mlflow import lazily inside the first LLM call / trace export. The imports are blocking and cost
# seconds each, stalling the event loop (measured up to 9 s on turn 1), so pay for them once in get_runtime().
WARM_IMPORTS = ("openai.resources.beta", "openai.types.beta", "mlflow.store.artifact.artifact_repo")
_current: dict[str, AgentContext | None] = {"ctx": None}


def current_context() -> AgentContext:
    ctx = _current["ctx"]
    if ctx is None:
        raise RuntimeError("No agent runtime: use `async with get_runtime()` or pass context= to run_turn")
    return ctx


def set_context(ctx: AgentContext | None) -> None:
    _current["ctx"] = ctx


def warm_imports() -> None:
    import importlib

    for name in WARM_IMPORTS:
        with contextlib.suppress(Exception):
            importlib.import_module(name)


async def build_context() -> AgentContext:
    """Create the retriever, the car catalog (bounded Cosmos reads, nothing preloaded) and the Foundry clients."""
    from autochat.llm import chat_llm
    from autochat.retrieval import CosmosCarCatalog, build_retriever

    retriever = await build_retriever()
    catalog = CosmosCarCatalog()
    await catalog.scope()  # warm the cached scope + fail fast if Cosmos is down
    return AgentContext(
        retriever=retriever,
        catalog=catalog,
        fast_llm=chat_llm("fast", temperature=0.0),
        answer_llm=chat_llm("chat"),
    )


async def open_checkpointer(stack: contextlib.AsyncExitStack) -> Any:
    """CosmosDBSaver on the `checkpoints` container; InMemorySaver (warning) if Cosmos is unavailable."""
    from langgraph.checkpoint.memory import InMemorySaver

    from autochat.config import get_settings

    s = get_settings()
    try:
        from langchain_azure_cosmosdb import CosmosDBSaver

        return await stack.enter_async_context(
            CosmosDBSaver.from_conn_info(
                endpoint=s.cosmos_endpoint,
                key=s.cosmos_key.get_secret_value() or None,
                database_name=s.cosmos_database,
                container_name=s.cosmos_checkpoints_container,
            )
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("CosmosDBSaver unavailable (%s); using in-memory checkpoints", type(exc).__name__)
        return InMemorySaver()


@contextlib.asynccontextmanager
async def get_runtime(*, checkpointer: Any = None) -> AsyncIterator[tuple[Any, AgentContext]]:
    """`async with get_runtime() as (graph, ctx):` - used by FastAPI lifespan, the CLI and the eval runner."""
    from autochat.db.cosmos import close_client

    async with contextlib.AsyncExitStack() as stack:
        try:
            await asyncio.to_thread(warm_imports)
            ctx = await build_context()
            saver = checkpointer if checkpointer is not None else await open_checkpointer(stack)
            graph = build_graph(saver)
            set_context(ctx)
            yield graph, ctx
        finally:
            set_context(None)
            await close_client()


def _turn_span() -> Any:
    if not obs.is_enabled():
        return contextlib.nullcontext(None)
    import mlflow
    from mlflow.entities import SpanType

    return mlflow.start_span(name="chat_turn", span_type=SpanType.AGENT)


async def run_turn(
    graph: Any,
    *,
    thread_id: str,
    message: str,
    user: str | None = None,
    on_token: Callable[[str], Any] | None = None,
    context: AgentContext | None = None,
) -> dict[str, Any]:
    """Run one user turn. Streams answer tokens to `on_token`; returns
    {answer, sources, profile, intent, language, steps, trace_id}."""
    ctx = context or current_context()
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    with _turn_span() as span:
        if span is not None:
            span.set_inputs({"message": message, "thread_id": thread_id})
        final: dict[str, Any] = {}
        async for mode, chunk in graph.astream(
            {"messages": [HumanMessage(message)]}, config, context=ctx, stream_mode=["messages", "values"]
        ):
            if mode == "messages":
                msg, meta = chunk
                # only LLM token chunks; the final AIMessage a node writes to state is streamed too -> skip it
                if (
                    on_token
                    and isinstance(msg, AIMessageChunk)
                    and meta.get("langgraph_node") in STREAM_NODES
                    and msg.content
                ):
                    res = on_token(str(msg.content))
                    if hasattr(res, "__await__"):
                        await res
            else:
                final = chunk
        result = {
            "answer": final.get("answer", ""),
            "sources": final.get("sources", []),
            "profile": final.get("user_profile", {}),
            "intent": final.get("intent"),
            "language": final.get("language"),
            "steps": final.get("exec_notes", []),
        }
        trace_id = None
        if span is not None:
            obs.tag_current_turn(thread_id, user=user, intent=result["intent"], language=result["language"])
            span.set_outputs({k: result[k] for k in ("answer", "sources", "profile", "intent", "steps")})
            trace_id = span.trace_id
    return {**result, "trace_id": trace_id}
