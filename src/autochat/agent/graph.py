"""Conversation graph: plan (1 LLM call) -> code executor -> grounded answer -> optional history compaction.

START -> analyze_turn -> update_profile -> route
    route -> respond_direct | ask_clarifying | execute_plan -> generate
    (all answering nodes) -> compact_history -> END
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from autochat.agent.nodes import (
    analyze_turn,
    ask_clarifying,
    compact_history,
    execute_plan,
    generate,
    respond_direct,
    update_profile,
)
from autochat.agent.profile import UserProfile, needs_clarification
from autochat.agent.state import AgentContext, AgentState

Route = Literal["respond_direct", "ask_clarifying", "execute_plan"]
STREAM_NODES = frozenset({"generate", "respond_direct", "ask_clarifying"})


def route(state: AgentState) -> Route:
    """Pure routing over the analyzed turn (no LLM)."""
    intent = state.get("intent", "car_question")
    if intent in ("general_chat", "out_of_scope"):
        return "respond_direct"
    if state.get("ambiguous_cars"):  # ask which car before answering about the wrong one
        return "respond_direct"
    if state.get("unknown_cars") and not state.get("mentioned_car_ids") and intent != "recommend":
        return "respond_direct"
    if (
        intent == "recommend"
        and not state.get("mentioned_car_ids")
        and not state.get("clarified")
        and needs_clarification(UserProfile.model_validate(state.get("user_profile") or {}))
    ):
        return "ask_clarifying"
    return "execute_plan"


def build_graph(checkpointer: Any = None) -> Any:
    g = StateGraph(AgentState, context_schema=AgentContext)
    g.add_node("analyze_turn", analyze_turn)
    g.add_node("update_profile", update_profile)
    g.add_node("execute_plan", execute_plan)
    g.add_node("generate", generate)
    g.add_node("respond_direct", respond_direct)
    g.add_node("ask_clarifying", ask_clarifying)
    g.add_node("compact_history", compact_history)

    g.add_edge(START, "analyze_turn")
    g.add_edge("analyze_turn", "update_profile")
    g.add_conditional_edges("update_profile", route, ["respond_direct", "ask_clarifying", "execute_plan"])
    g.add_edge("execute_plan", "generate")
    for node in STREAM_NODES:
        g.add_edge(node, "compact_history")
    g.add_edge("compact_history", END)
    return g.compile(checkpointer=checkpointer)
