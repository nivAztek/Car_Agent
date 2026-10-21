"""LangGraph state (checkpointed per thread) and the per-process runtime context (injected, not checkpointed)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from autochat.retrieval.catalog import CarCatalog
from autochat.retrieval.hybrid import Retriever

MAX_FOCUS = 6


class AgentState(TypedDict, total=False):
    # --- persisted across turns (checkpointer)
    messages: Annotated[list[AnyMessage], add_messages]  # full transcript, never deleted
    summary: str  # rolling summary of messages[:summarized_upto]
    summarized_upto: int
    user_profile: dict[str, Any]  # UserProfile.model_dump()
    clarified: bool  # a clarifying question was already asked in this thread (ask at most once)
    focus_car_ids: list[str]  # cars of the last answer, in the order shown (max MAX_FOCUS): '#1', 'it', ...
    # --- per turn (overwritten every turn)
    language: str
    intent: str
    standalone_query: str
    mentioned_car_ids: list[str]  # resolved in code from car_mentions + focus_refs
    unknown_cars: list[str]  # mentions that are not reviewed cars
    unknown_suggestions: list[str]  # reviewed cars of the same make as an unknown mention
    ambiguous_cars: dict[str, list[str]]  # mention -> candidate car names (ask which one)
    car_refs: dict[str, list[str]]  # ref_key -> car_ids, for the plan and the profile delta (debug/trace)
    profile_delta: dict[str, Any]
    profile_dropped: list[str]  # delta fields rejected for lack of a verbatim quote (debug/trace)
    plan: list[dict[str, Any]]  # PlanStep dumps
    plan_trace: list[dict[str, Any]]  # StepResult dumps (what the executor did)
    exec_notes: list[str]  # replan notes, shown to the answer model
    candidate_car_ids: list[str]
    retrieved: list[dict[str, Any]]  # RetrievedChunk dumps, in citation order
    answer: str
    sources: list[dict[str, Any]]


# Keys reset at the start of every turn (the checkpointer would otherwise carry them over).
TURN_RESET: dict[str, Any] = {
    "plan": [],
    "plan_trace": [],
    "exec_notes": [],
    "candidate_car_ids": [],
    "retrieved": [],
    "answer": "",
    "sources": [],
    "unknown_cars": [],
    "unknown_suggestions": [],
    "ambiguous_cars": {},
    "car_refs": {},
    "mentioned_car_ids": [],
    "profile_delta": {},
    "profile_dropped": [],
}


@dataclass
class AgentContext:
    """Runtime dependencies (LangGraph `context_schema`). Fakes are injected in tests."""

    retriever: Retriever
    catalog: CarCatalog  # bounded reads only; the agent never holds the full car list
    fast_llm: Any = None  # planner, direct replies, summaries
    answer_llm: Any = None  # grounded answers (streamed)
    settings: dict[str, Any] = field(default_factory=dict)
