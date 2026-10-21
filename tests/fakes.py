"""Shared offline fakes for agent and API tests (no Cosmos, no Foundry)."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from autochat.agent.graph import build_graph
from autochat.agent.schemas import PlanStep, TurnAnalysis
from autochat.agent.state import AgentContext
from autochat.retrieval.catalog import InMemoryCarCatalog
from autochat.retrieval.hybrid import RetrievedChunk

CARS: dict[str, dict[str, Any]] = {
    "ev-cheap": {
        "car_id": "ev-cheap",
        "make": "MG",
        "model": "S6",
        "powertrain": "ev",
        "body_type": "suv",
        "price_ils": 168_888,
        "range_real_km": 430,
        "tags": ["family", "budget"],
        "aspect_scores": {"value": 5, "comfort": 3, "practicality": 4, "fun": 2},
    },
    "ev-big": {
        "car_id": "ev-big",
        "make": "Kia",
        "model": "EV9",
        "powertrain": "ev",
        "body_type": "suv",
        "price_ils": None,
        "range_real_km": 480,
        "tags": ["family", "long_trips"],
        "aspect_scores": {"value": 3, "comfort": 5, "practicality": 5, "fun": 2},
    },
    "petrol-fun": {
        "car_id": "petrol-fun",
        "make": "Hyundai",
        "model": "Elantra N",
        "powertrain": "ice",
        "body_type": "sedan",
        "price_ils": 224_900,
        "tags": ["enthusiast"],
        "aspect_scores": {"value": 4, "comfort": 2, "practicality": 3, "fun": 5},
    },
    "small-mhev": {
        "car_id": "small-mhev",
        "make": "Citroen",
        "model": "C3",
        "powertrain": "mhev",
        "body_type": "hatchback",
        "price_ils": 114_990,
        "tags": ["city", "budget"],
        "aspect_scores": {"value": 5, "comfort": 4, "practicality": 3, "fun": 2},
    },
}


def chunk(car_id: str, kind: str, **kw: Any) -> RetrievedChunk:
    c = CARS[car_id]
    return RetrievedChunk(
        id=f"{car_id}:{kind}",
        car_id=car_id,
        chunk_type="car_card" if kind == "card" else "section",
        section_type="car_card" if kind == "card" else kind,
        heading=kind,
        heading_en=kind.title(),
        text_he="טקסט",
        text_en=f"{c['make']} {c['model']} {kind} text",
        url=f"https://x/{car_id}",
        make=c["make"],
        model=c["model"],
        **kw,
    )


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def retrieve(self, query: str, car_ids: list[str] | None = None, **kw: Any) -> list[RetrievedChunk]:
        targets = car_ids or list(CARS)
        self.calls.append({"query": query, "car_ids": targets, **kw})
        return [chunk(c, "verdict") for c in targets] + [chunk(c, "card", pinned=True) for c in targets]


class FakeFastLLM:
    """Planner/summary/direct model: queued TurnAnalysis objects for structured output, text otherwise."""

    def __init__(self, analyses: list[TurnAnalysis], text: str = "summary text") -> None:
        self.analyses = list(analyses)
        self.text = text
        self.prompts: list[Any] = []

    def with_structured_output(self, schema: Any) -> Any:
        async def _next(prompt: Any) -> TurnAnalysis:
            self.prompts.append(prompt)
            return self.analyses.pop(0)

        return RunnableLambda(_next)

    async def ainvoke(self, prompt: Any) -> AIMessage:
        self.prompts.append(prompt)
        return AIMessage(self.text)


def make_ctx(analyses: list[TurnAnalysis], answers: list[str]) -> AgentContext:
    return AgentContext(
        retriever=FakeRetriever(),
        catalog=InMemoryCarCatalog(CARS),
        fast_llm=FakeFastLLM(analyses),
        answer_llm=GenericFakeChatModel(messages=iter([AIMessage(a) for a in answers])),
    )


# --------------------------------------------------------------------------- fake API (FastAPI app on fakes)
ANSWER = "The MG S6 fits your budget [1]. How many people usually ride?"


def recommend_analysis() -> TurnAnalysis:
    return TurnAnalysis(
        language="en",
        intent="recommend",
        standalone_query="Recommend an EV under 180k",
        profile_delta={
            "budget_max_ils": 180_000,
            "powertrain_pref": ["ev"],
            "evidence": [
                {"field": "budget_max_ils", "quote": "under 180k"},
                {"field": "powertrain_pref", "quote": "an EV"},
            ],
        },
        plan=[PlanStep(tool="rank_by_profile", limit=1), PlanStep(tool="search_reviews")],
    )


def runtime_factory(ctx: AgentContext) -> Any:
    """Stand-in for `get_runtime`: in-memory checkpointer + the given fake context."""

    @contextlib.asynccontextmanager
    async def factory() -> AsyncIterator[tuple[Any, AgentContext]]:
        yield build_graph(InMemorySaver()), ctx

    return factory


def fake_api(ctx: AgentContext | None = None, **kw: Any) -> TestClient:
    """FastAPI app on fakes (tracing off). Use as a context manager so the lifespan runs."""
    from autochat.api.main import create_app

    ctx = ctx or make_ctx([recommend_analysis()], [ANSWER])
    kw.setdefault("health_checks", {})
    return TestClient(create_app(runtime_factory(ctx), tracing=False, **kw))
