"""Smoke test MLflow 3 tracing + GenAI evaluation against Foundry, fully local.

Validates the exact patterns autochat uses (src/autochat/observability.py):
  1. setup_mlflow(): sqlite tracking store + experiment + langchain autolog(run_tracer_inline=True)
  2. the `run_turn` contract: mlflow.start_span("chat_turn") around one async LangGraph `astream`
     (tokens -> on_token callback), inputs={message, thread_id}, outputs={answer, sources}, session=thread_id
  3. a @mlflow.trace(span_type=RETRIEVER) step nested in the same trace (Documents with doc_uri)
  4. human feedback (thumbs up) logged on the trace
  5. mlflow.genai.evaluate on the captured trace: code scorer + Foundry judges
     (RelevanceToQuery, RetrievalGroundedness, Correctness, Guidelines) via `openai:/<deployment>`

Usage:
    scripts/dev.ps1 smoke-mlflow     (then: scripts/dev.ps1 mlflow -> http://localhost:5000, experiment autochat-smoke)
Never prints secrets.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import TypedDict

from langchain_core.documents import Document  # module scope: LangGraph resolves state type hints

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

DOCS = [
    {
        "id": "mg-s6#range",
        "car_id": "mg-s6",
        "url": "https://www.auto.co.il/mg-s6",
        "section_type": "range",
        "text_en": "MG S6 EV: WLTP range 530 km; in our real-world test about 430 km. DC charging up to 144 kW.",
    },
    {
        "id": "mg-s6#price",
        "car_id": "mg-s6",
        "url": "https://www.auto.co.il/mg-s6",
        "section_type": "price",
        "text_en": "MG S6 price in Israel: 168,888 NIS.",
    },
]


def load_env() -> None:
    env = REPO_ROOT / ".env"
    if not env.exists():
        sys.exit("Missing .env")
    for line in env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    load_env()
    import mlflow
    from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
    from langgraph.graph import END, START, StateGraph
    from mlflow.entities import SpanType
    from mlflow.genai.scorers import Correctness, Guidelines, RelevanceToQuery, RetrievalGroundedness, scorer

    from autochat import observability as obs

    if not obs.setup_mlflow("autochat-smoke"):
        sys.exit("AUTOCHAT_TRACING is disabled - enable it to run this smoke test.")
    judge = obs.judge_model_uri()
    print(f"mlflow {mlflow.__version__} tracking={mlflow.get_tracking_uri()} judge={judge}")

    llm = AzureAIOpenAIApiChatModel(
        endpoint=os.environ["FOUNDRY_ENDPOINT"].rstrip("/"),
        credential=os.environ["FOUNDRY_API_KEY"],
        model=os.environ["FOUNDRY_FAST_DEPLOYMENT"],
        temperature=0,
    )

    class S(TypedDict, total=False):
        question: str
        docs: list[Document]
        answer: str

    @mlflow.trace(span_type=SpanType.RETRIEVER, name="hybrid_retrieve")
    def retrieve_docs(query: str) -> list[Document]:
        return obs.retriever_documents(DOCS)

    async def retrieve(state: S) -> S:
        return {"docs": retrieve_docs(state["question"])}

    async def generate(state: S) -> S:
        ctx = "\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(state["docs"]))
        msg = await llm.ainvoke(
            f"Answer ONLY from the context, cite [n].\nContext:\n{ctx}\n\nQuestion: {state['question']}"
        )
        return {"answer": str(msg.content)}

    g = StateGraph(S)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    graph = g.compile()

    # Same shape as autochat.agent.runtime.run_turn: start_span (not @trace, so `graph` is not serialized),
    # one astream() for token streaming via on_token, outputs = {answer, sources, trace_id}.
    async def run_turn(graph, *, thread_id: str, message: str, user: str | None = None, on_token=None) -> dict:
        with mlflow.start_span(name="chat_turn", span_type=SpanType.AGENT) as span:
            span.set_inputs({"message": message, "thread_id": thread_id})
            obs.tag_current_turn(thread_id, user=user, intent="car_question", language="en")
            final: dict = {}
            async for mode, chunk in graph.astream({"question": message}, stream_mode=["messages", "values"]):
                if mode == "messages":
                    msg, meta = chunk
                    if meta.get("langgraph_node") == "generate" and msg.content and on_token:
                        on_token(str(msg.content))
                else:
                    final = chunk
            sources = [{"n": i + 1, **d.metadata} for i, d in enumerate(final.get("docs", []))]
            result = {"answer": final.get("answer", ""), "sources": sources}
            span.set_outputs(result)
            return {**result, "trace_id": span.trace_id}

    failures = 0

    def ok(name: str, cond: bool, detail: str = "") -> None:
        nonlocal failures
        failures += 0 if cond else 1
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    tokens: list[str] = []
    out = await run_turn(
        graph,
        thread_id="smoke-thread-1",
        message="What is the real-world range of the MG S6?",
        user="smoke",
        on_token=tokens.append,
    )
    ok("tokens streamed", len(tokens) > 1, f"chunks={len(tokens)}")
    trace_id = out["trace_id"]
    ok("trace captured", bool(trace_id) and trace_id == obs.last_trace_id(), f"id={trace_id}")
    trace = mlflow.get_trace(trace_id, flush=True)  # async export -> flush before reading back
    ok("trace readable", trace is not None)
    if trace is None:
        return 1
    root = trace.data.spans[0]
    kinds = {s.span_type for s in trace.data.spans}
    ok("root span = chat_turn", root.name == "chat_turn", f"inputs={list((root.inputs or {}).keys())}")
    ok("retriever span nested", SpanType.RETRIEVER in kinds, f"span types={sorted(map(str, kinds))}")
    ok("chat model span nested", SpanType.CHAT_MODEL in kinds)
    ok("session tag", trace.info.trace_metadata.get("mlflow.trace.session") == "smoke-thread-1")
    ok("answer", "430" in out["answer"], repr(out["answer"][:120]))

    obs.log_user_feedback(trace_id, True, comment="smoke thumbs up", user="smoke")
    fb = [a.name for a in mlflow.get_trace(trace_id, flush=True).info.assessments]
    ok("user feedback logged", "user_feedback" in fb)

    @scorer
    def must_contain(outputs, expectations) -> bool:
        text = str(outputs.get("answer", outputs) if isinstance(outputs, dict) else outputs)
        return all(s.lower() in text.lower() for s in expectations.get("must_contain", []))

    result = mlflow.genai.evaluate(
        data=[
            {
                "trace": trace,
                "expectations": {
                    "must_contain": ["430"],
                    "expected_facts": ["The MG S6 real-world range is about 430 km."],
                },
            }
        ],
        scorers=[
            must_contain,
            RelevanceToQuery(model=judge),
            RetrievalGroundedness(model=judge),
            Correctness(model=judge),
            Guidelines(
                name="cites_sources",
                model=judge,
                guidelines="Every factual claim about a car is followed by a [n] citation.",
            ),
        ],
    )
    metrics = {k: round(float(v), 3) for k, v in sorted(result.metrics.items()) if v is not None}
    print("metrics:", metrics)
    for m in ["must_contain", "relevance_to_query", "retrieval_groundedness", "correctness", "cites_sources"]:
        ok(f"metric {m}", f"{m}/mean" in metrics, str(metrics.get(f"{m}/mean")))
    print(f"eval run_id={result.run_id}")
    print("\nALL OK" if failures == 0 else f"\n{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
