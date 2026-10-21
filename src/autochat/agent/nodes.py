"""Graph nodes. Async, partial state updates only; routing lives in graph.py as pure functions.

LLM calls per turn: analyze_turn (1 structured call, fast model) + one answering call (generate / respond_direct /
ask_clarifying). compact_history adds one fast call only when the unsummarized window grows too large.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.runtime import Runtime
from tenacity import AsyncRetrying

from autochat import observability as obs
from autochat.agent import prompts as P
from autochat.agent import tools as T
from autochat.agent.profile import UserProfile, merge_profile, next_missing_slot, profile_summary, validate_delta
from autochat.agent.schemas import PlanStep, Source, TurnAnalysis
from autochat.agent.state import MAX_FOCUS, TURN_RESET, AgentContext, AgentState
from autochat.llm import RETRY_KW
from autochat.retrieval.resolver import ref_key, resolve_mentions

log = logging.getLogger(__name__)

HEBREW = re.compile(r"[\u0590-\u05FF]")
LATIN = re.compile(r"[A-Za-z]")
CITATION = re.compile(r"\[(\d+(?:\s*[,\u2013-]\s*\d+)*)\]")
MAX_CONTEXT_CHUNKS = 14
ANSWER_HISTORY = 4
# compact_history: summarize when the unsummarized window exceeds either limit; keep the last KEEP verbatim.
COMPACT_MAX_MESSAGES = 16
COMPACT_MAX_TOKENS = 6000
COMPACT_KEEP = 6


# --------------------------------------------------------------------------- helpers
def detect_language(text: str) -> str:
    he, en = len(HEBREW.findall(text)), len(LATIN.findall(text))
    return "he" if he and he >= en * 0.5 else "en"


def last_user_text(messages: list[AnyMessage]) -> str:
    for m in reversed(messages):
        if m.type == "human":
            return str(m.content)
    return ""


def cited_numbers(answer: str) -> list[int]:
    nums: list[int] = []
    for group in CITATION.findall(answer):
        parts = [int(x) for x in re.findall(r"\d+", group)]
        if re.search(r"[\u2013-]", group) and len(parts) == 2 and 0 < parts[1] - parts[0] <= 10:
            parts = list(range(parts[0], parts[1] + 1))
        nums.extend(parts)
    return list(dict.fromkeys(nums))


def build_sources(answer: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sources actually cited in the answer, keeping the [n] numbering of the context block."""
    out = []
    for n in cited_numbers(answer):
        if 1 <= n <= len(chunks):
            c = chunks[n - 1]
            out.append(
                Source(
                    n=n,
                    car_id=c["car_id"],
                    car=f"{c.get('make', '')} {c.get('model', '')}".strip(),
                    heading=c.get("heading_en") or c.get("heading", ""),
                    section_type=c.get("section_type", ""),
                    url=c.get("url", ""),
                ).model_dump()
            )
    return sorted(out, key=lambda s: s["n"])


def order_context(chunks: list[dict[str, Any]], car_order: list[str], limit: int = MAX_CONTEXT_CHUNKS) -> list[dict]:
    """Group excerpts by car (plan order), ranked excerpts before pinned cards, capped."""
    rank = {c: i for i, c in enumerate(car_order)}
    ordered = sorted(enumerate(chunks), key=lambda t: (rank.get(t[1]["car_id"], 99), t[1].get("pinned", False), t[0]))
    picked = [c for _, c in ordered]
    if len(picked) <= limit:
        return picked
    # keep every car's card, trim ranked excerpts from the end
    cards = [c for c in picked if c.get("chunk_type") == "car_card"]
    others = [c for c in picked if c.get("chunk_type") != "car_card"][: max(0, limit - len(cards))]
    keep = {id(c) for c in cards + others}
    return [c for c in picked if id(c) in keep]


async def _ainvoke_retry(runnable: Any, payload: Any) -> Any:
    async for attempt in AsyncRetrying(**RETRY_KW):
        with attempt:
            return await runnable.ainvoke(payload)
    raise RuntimeError("unreachable")


def _profile(state: AgentState) -> UserProfile:
    return UserProfile.model_validate(state.get("user_profile") or {})


def _car_refs(analysis: TurnAnalysis) -> list[str]:
    """Every car reference the planner wrote: mentions, plan step cars, liked/rejected cars."""
    d = analysis.profile_delta
    steps = [r for s in analysis.plan for r in s.cars]
    return list(dict.fromkeys([*analysis.car_mentions, *steps, *d.cars_liked, *d.cars_rejected]))


def focus_from_answer(answer: str, chunks: list[dict[str, Any]], fallback: list[str]) -> list[str]:
    """Cars in the order the answer cites them (what the user saw), else the plan's cars."""
    cited = [chunks[n - 1]["car_id"] for n in cited_numbers(answer) if 1 <= n <= len(chunks)]
    return list(dict.fromkeys(cited or fallback))[:MAX_FOCUS]


# --------------------------------------------------------------------------- nodes
async def analyze_turn(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Planner: one structured call -> intent, standalone query, car names, profile delta and a typed tool plan.
    The prompt holds a fixed-size catalog scope + the previous answer's cars; names are resolved here in code."""
    ctx = runtime.context
    messages = state.get("messages", [])
    message = last_user_text(messages)
    upto = min(state.get("summarized_upto", 0), max(len(messages) - 1, 0))
    focus_cars = await ctx.catalog.get(state.get("focus_car_ids", [])[:MAX_FOCUS])
    scope = await ctx.catalog.scope()
    prompt = [
        SystemMessage(P.ANALYZE_SYSTEM.format(scope=scope.describe(), max_steps=P.MAX_STEPS)),
        HumanMessage(
            P.ANALYZE_USER.format(
                profile=profile_summary(_profile(state)),
                summary=state.get("summary") or "(none)",
                focus=P.focus_block(focus_cars),
                history=P.history_block(messages[upto:-1], limit=COMPACT_MAX_MESSAGES),
                message=message,
            )
        ),
    ]
    try:
        analysis: TurnAnalysis = await _ainvoke_retry(ctx.fast_llm.with_structured_output(TurnAnalysis), prompt)
    except Exception:  # noqa: BLE001 - degrade to a plain grounded search instead of failing the turn
        log.exception("analyze_turn failed; falling back to a plain search")
        analysis = TurnAnalysis(language="en", intent="car_question", standalone_query=message)

    # names -> car_ids in code (bounded catalog lookups); '#n' -> the n-th focus car
    resolved = await resolve_mentions(_car_refs(analysis), ctx.catalog)
    refs: dict[str, list[str]] = {k: r.car_ids for k, r in resolved.items() if r.status == "resolved"}
    for i, car in enumerate(focus_cars, 1):
        refs[f"#{i}"] = [car["car_id"]]

    def resolve_car(ref: str) -> list[str]:
        return refs.get(ref_key(ref), [])

    mention_res = [resolved[k] for k in dict.fromkeys(ref_key(m) for m in analysis.car_mentions) if k in resolved]
    focus_ids = [cid for i in analysis.focus_refs for cid in refs.get(f"#{i}", [])]
    mentioned = list(dict.fromkeys([*(c for r in mention_res for c in resolve_car(r.mention)), *focus_ids]))
    unknown = [r for r in mention_res if r.status == "unknown"]
    ambiguous = {r.mention: r.names for r in mention_res if r.status == "ambiguous"}

    delta, dropped = validate_delta(analysis.profile_delta, message, resolve_car)
    if dropped:
        log.info("profile fields without a verbatim quote dropped: %s", dropped)
    plan = T.resolve_plan(analysis.plan, resolve_car, scope.make_vocab())
    return {
        **TURN_RESET,
        "language": detect_language(message),  # deterministic beats the model for this
        "intent": analysis.intent,
        "standalone_query": analysis.standalone_query or message,
        "mentioned_car_ids": mentioned,
        "unknown_cars": [r.mention for r in unknown],
        "unknown_suggestions": list(dict.fromkeys(s for r in unknown for s in r.suggestions)),
        "ambiguous_cars": ambiguous,
        "car_refs": refs,
        "profile_delta": delta.model_dump(exclude_defaults=True),
        "profile_dropped": dropped,
        "plan": [s.model_dump(exclude_defaults=True) for s in plan],
    }


async def update_profile(state: AgentState) -> dict[str, Any]:
    return {"user_profile": merge_profile(state.get("user_profile"), state.get("profile_delta")).model_dump()}


async def execute_plan(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Executor (no LLM): run the typed plan through the tools; one deterministic relax-and-retry if empty."""
    ctx = runtime.context
    intent = state.get("intent", "car_question")
    query = state.get("standalone_query") or last_user_text(state.get("messages", []))
    mentioned = state.get("mentioned_car_ids", [])
    plan = [PlanStep.model_validate(s) for s in state.get("plan", [])][: P.MAX_STEPS]
    if not plan:
        plan = T.default_plan(intent, mentioned, query)
    if mentioned and all(s.tool == "search_reviews" and not s.car_ids for s in plan):
        plan = [s.model_copy(update={"car_ids": mentioned}) for s in plan]  # planner forgot the named cars
    plan = T.ensure_search_step(plan, query)

    tools = T.make_tools(retriever=ctx.retriever, catalog=ctx.catalog, profile=_profile(state))
    with obs.evidence_span({"query": query, "plan": [s.model_dump(exclude_defaults=True) for s in plan]}) as span:
        result = await T.execute_plan(plan, tools, standalone_query=query)
        notes = [s.note for s in result.steps]
        if result.needs_replan:
            plan, relax_notes = T.relax_plan(plan, result.steps)
            retry = await T.execute_plan(plan, tools, standalone_query=query, relax=True)
            notes += [f"REPLAN: {n}" for n in relax_notes] + [s.note for s in retry.steps]
            if retry.chunks or retry.car_ids:
                result = retry

        chunks = order_context(result.chunks, result.car_ids)
        if span is not None:
            # exactly what generate sees: the catalog selections (planner notes) + the numbered excerpts
            selections = [n for n in notes if not n.startswith("search_reviews")]
            span.set_outputs(obs.evidence_documents(selections, chunks))
    return {
        "plan": [s.model_dump(exclude_defaults=True) for s in plan],
        "plan_trace": [s.as_dict() for s in result.steps],
        "exec_notes": notes,
        "candidate_car_ids": result.car_ids,
        "retrieved": chunks,
    }


async def generate(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Grounded answer over the numbered excerpts (streamed by the caller via stream_mode='messages')."""
    ctx = runtime.context
    language = state.get("language", "en")
    intent = state.get("intent", "car_question")
    profile = _profile(state)
    chunks = state.get("retrieved", [])
    messages = state.get("messages", [])

    slot = next_missing_slot(profile) if intent == "recommend" else None
    extra = [P.FOLLOWUP_HINT.format(slot=slot)] if slot else []
    if state.get("unknown_cars"):
        extra.append(P.UNKNOWN_CARS_HINT.format(cars=", ".join(state["unknown_cars"])))
    prompt = [
        SystemMessage(
            P.ANSWER_SYSTEM.format(
                scope=(await ctx.catalog.scope()).describe(),
                language_name=P.LANGUAGE_NAMES.get(language, "English"),
                intent_instructions=P.INTENT_INSTRUCTIONS.get(intent, P.INTENT_INSTRUCTIONS["car_question"]),
            )
        ),
        HumanMessage(
            P.ANSWER_USER.format(
                profile=profile_summary(profile),
                followup="\n".join(extra),
                notes="\n".join(f"- {n}" for n in state.get("exec_notes", [])) or "(none)",
                context=P.context_block(chunks, language),
                summary=state.get("summary") or "(none)",
                history=P.history_block(messages[:-1], limit=ANSWER_HISTORY),
                standalone_query=state.get("standalone_query", ""),
                message=last_user_text(messages),
            )
        ),
    ]
    msg = await ctx.answer_llm.ainvoke(prompt)
    answer = str(msg.content).strip()
    focus = focus_from_answer(answer, chunks, state.get("candidate_car_ids", []))
    return {
        "answer": answer,
        "sources": build_sources(answer, chunks),
        "focus_car_ids": focus or state.get("focus_car_ids", []),
        "messages": [AIMessage(answer)],
    }


async def respond_direct(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Greetings, out-of-scope, cars that are not reviewed, or an ambiguous name. No retrieval, no car facts."""
    ctx = runtime.context
    unknown = state.get("unknown_cars", [])
    ambiguous = state.get("ambiguous_cars", {})
    intent = state.get("intent", "general_chat")
    if intent in ("general_chat", "out_of_scope") or not (unknown or ambiguous):
        situation = P.DIRECT_SITUATIONS.get(intent, P.DIRECT_SITUATIONS["general_chat"])
    elif ambiguous:
        options = "; ".join(f"'{m}' -> {' / '.join(names)}" for m, names in ambiguous.items())
        situation = P.DIRECT_SITUATIONS["ambiguous_cars"].format(options=options)
    else:
        suggestions = state.get("unknown_suggestions", [])
        hint = P.SUGGEST_SAME_MAKE.format(cars=", ".join(suggestions)) if suggestions else P.SUGGEST_GENERIC
        situation = P.DIRECT_SITUATIONS["unknown_cars"].format(cars=", ".join(unknown), suggestions=hint)
    prompt = [
        SystemMessage(
            P.DIRECT_SYSTEM.format(
                scope=(await ctx.catalog.scope()).describe(),
                language_name=P.LANGUAGE_NAMES.get(state.get("language", "en"), "English"),
                situation=situation,
            )
        ),
        *state.get("messages", [])[-ANSWER_HISTORY:],
    ]
    msg = await ctx.fast_llm.ainvoke(prompt)
    answer = str(msg.content).strip()
    return {"answer": answer, "sources": [], "messages": [AIMessage(answer)]}


async def ask_clarifying(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Recommendation requested with no preference anchor: ask exactly one question (once per thread)."""
    ctx = runtime.context
    slot = next_missing_slot(_profile(state)) or "main needs"
    prompt = [
        SystemMessage(
            P.CLARIFY_SYSTEM.format(
                scope=(await ctx.catalog.scope()).describe(),
                language_name=P.LANGUAGE_NAMES.get(state.get("language", "en"), "English"),
                slot=slot,
            )
        ),
        *state.get("messages", [])[-ANSWER_HISTORY:],
    ]
    msg = await ctx.fast_llm.ainvoke(prompt)
    answer = str(msg.content).strip()
    return {"answer": answer, "sources": [], "clarified": True, "messages": [AIMessage(answer)]}


def needs_compaction(messages: list[AnyMessage], upto: int) -> bool:
    window = messages[upto:]
    return len(window) > COMPACT_MAX_MESSAGES or count_tokens_approximately(window) > COMPACT_MAX_TOKENS


async def compact_history(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Rolling summary of everything except the last COMPACT_KEEP messages. The transcript is never deleted
    (the UI and checkpoints keep it); only the planner/answer prompts switch to summary + recent window."""
    messages = state.get("messages", [])
    upto = state.get("summarized_upto", 0)
    if not needs_compaction(messages, upto):
        return {}
    new_upto = max(upto, len(messages) - COMPACT_KEEP)
    if new_upto <= upto:
        return {}
    block = "\n".join(f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}" for m in messages[upto:new_upto])
    prompt = [
        SystemMessage(P.SUMMARY_SYSTEM),
        HumanMessage(P.SUMMARY_USER.format(summary=state.get("summary") or "(empty)", messages=block)),
    ]
    try:
        msg = await _ainvoke_retry(runtime.context.fast_llm, prompt)
    except Exception:  # noqa: BLE001 - keep the longer window rather than fail the (already answered) turn
        log.exception("compact_history failed; will retry next turn")
        return {}
    return {"summary": str(msg.content).strip(), "summarized_upto": new_upto}
