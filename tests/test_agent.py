"""Offline agent tests: profile merge, ranking, plan executor + replan, routing, citations, compaction, full graph."""

from __future__ import annotations

from fakes import CARS, FakeRetriever, chunk, make_ctx
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from autochat.agent import nodes
from autochat.agent.graph import build_graph, route
from autochat.agent.profile import (
    UserProfile,
    merge_profile,
    needs_clarification,
    next_missing_slot,
    validate_delta,
)
from autochat.agent.runtime import run_turn
from autochat.agent.schemas import PlanStep, TurnAnalysis
from autochat.agent.tools import (
    ensure_search_step,
    execute_plan,
    make_tools,
    profile_filters,
    rank_cars,
    relax_plan,
    resolve_plan,
)
from autochat.retrieval.catalog import InMemoryCarCatalog
from autochat.retrieval.filters import SearchFilters
from autochat.retrieval.resolver import ref_key


# --------------------------------------------------------------------------- profile
def test_merge_profile_union_replace_and_priority_order() -> None:
    p = merge_profile(None, {"budget_max_ils": 200_000, "powertrain_pref": ["ev"], "priorities": ["comfort"]})
    p = merge_profile(p, {"priorities": ["value"], "cars_liked": ["ev-cheap"]})
    assert p.priorities == ["value", "comfort"]
    p = merge_profile(
        p, {"powertrain_pref": ["ice"], "replace_lists": ["powertrain_pref"], "cars_rejected": ["ev-cheap"]}
    )
    assert p.powertrain_pref == ["ice"]
    assert p.budget_max_ils == 200_000  # untouched scalar survives
    assert p.cars_rejected == ["ev-cheap"] and p.cars_liked == []


def test_validate_delta_requires_verbatim_quote() -> None:
    msg = "I can charge at home. Which would you recommend? Budget is 200k"
    delta = {
        "home_charging": True,
        "powertrain_pref": ["ev"],
        "budget_max_ils": 200_000,
        "cars_liked": ["ev-cheap"],
        "evidence": [
            {"field": "home_charging", "quote": "I can charge at home"},
            {"field": "powertrain_pref", "quote": "I want an electric car"},  # not in the message
            {"field": "budget_max_ils", "quote": "budget is 200k"},  # case-insensitive
            {"field": "cars_liked", "quote": "I can charge at home"},  # quoted, but no liking cue
        ],
    }
    clean, dropped = validate_delta(delta, msg, resolve_car=lambda ref: [])
    assert clean.home_charging is True and clean.budget_max_ils == 200_000
    assert clean.powertrain_pref == [] and clean.cars_liked == []
    assert dropped == ["powertrain_pref", "cars_liked"]
    # liked cars are names mapped to ids in code; names that do not resolve are dropped
    liked = {"cars_liked": ["the MG", "Tesla"], "evidence": [{"field": "cars_liked", "quote": "I like the MG"}]}
    clean, _ = validate_delta(liked, "I like the MG and the Tesla", lambda r: {"the MG": ["ev-cheap"]}.get(r, []))
    assert clean.cars_liked == ["ev-cheap"]
    # verbatim but no explicit cue: home charging is not a powertrain preference
    clean, dropped = validate_delta(
        {"powertrain_pref": ["ev"], "evidence": [{"field": "powertrain_pref", "quote": "I can charge at home"}]}, msg
    )
    assert clean.powertrain_pref == [] and dropped == ["powertrain_pref"]
    clean, _ = validate_delta(
        {"powertrain_pref": ["ev"], "evidence": [{"field": "powertrain_pref", "quote": "רכב חשמלי"}]},
        "אני רוצה רכב חשמלי",
    )
    assert clean.powertrain_pref == ["ev"]
    # a budget without any figure in the quote is a guess ("cheapest electric car" is not a budget)
    clean, dropped = validate_delta(
        {"budget_max_ils": 114_990, "evidence": [{"field": "budget_max_ils", "quote": "the cheapest electric car"}]},
        "I want the cheapest electric car",
    )
    assert clean.budget_max_ils is None and dropped == ["budget_max_ils"]
    clean, _ = validate_delta(
        {"budget_max_ils": 250_000, "evidence": [{"field": "budget_max_ils", "quote": "עד 250 אלף"}]},
        "התקציב שלי עד 250 אלף",
    )
    assert clean.budget_max_ils == 250_000
    # replace_lists only with an explicit change-of-mind cue
    ev = {"field": "powertrain_pref", "quote": "petrol is fine"}
    change = {"powertrain_pref": ["ice"], "replace_lists": ["powertrain_pref"], "evidence": [ev]}
    assert validate_delta(change, "Actually petrol is fine")[0].replace_lists == ["powertrain_pref"]
    assert validate_delta(change, "petrol is fine too")[0].replace_lists == []


def test_clarification_and_next_slot() -> None:
    assert needs_clarification(UserProfile())
    assert not needs_clarification(UserProfile(powertrain_pref=["ev"]))
    assert next_missing_slot(UserProfile()) == "budget"
    fam = UserProfile(budget_max_ils=1, use_cases=["family"])
    assert next_missing_slot(fam) == "how many people usually ride"


# --------------------------------------------------------------------------- ranking
def test_rank_respects_budget_and_priorities() -> None:
    profile = UserProfile(budget_max_ils=180_000, priorities=["comfort"], use_cases=["city"])
    ranked = rank_cars(list(CARS.values()), profile)
    ids = [c["car_id"] for c, _, _ in ranked]
    assert ids == ["small-mhev", "ev-cheap"]  # Kia (no price) and Elantra (over budget) excluded
    relaxed = [c["car_id"] for c, _, _ in rank_cars(list(CARS.values()), profile, relax=True)]
    assert "ev-big" in relaxed  # comfort 5 wins once constraints are relaxed


# --------------------------------------------------------------------------- executor
async def run(plan: list[PlanStep], profile: UserProfile | None = None, relax: bool = False):
    retriever = FakeRetriever()
    tools = make_tools(retriever=retriever, catalog=InMemoryCarCatalog(CARS), profile=profile or UserProfile())
    res = await execute_plan(plan, tools, standalone_query="q", relax=relax)
    return res, retriever


async def test_executor_chains_find_rank_search() -> None:
    plan = [
        PlanStep(tool="find_cars", filters=SearchFilters(powertrains=["ev"])),
        PlanStep(tool="rank_by_profile", limit=1),
        PlanStep(tool="search_reviews", query="comfort"),
    ]
    res, retriever = await run(plan, UserProfile(priorities=["comfort"]))
    assert res.steps[0].car_ids == ["ev-big", "ev-cheap"]
    assert res.steps[1].car_ids == ["ev-big"]
    assert retriever.calls[0]["car_ids"] == ["ev-big"]
    assert not res.needs_replan and {c["car_id"] for c in res.chunks} == {"ev-big"}


async def test_superlative_sort_and_empty_triggers_replan() -> None:
    plan = [
        PlanStep(
            tool="find_cars",
            filters=SearchFilters(powertrains=["ev"]),
            sort_by="range_real_km",
            descending=True,
            limit=1,
        ),
        PlanStep(tool="search_reviews"),
    ]
    res, _ = await run(plan)
    assert res.steps[0].car_ids == ["ev-big"]

    tight = [PlanStep(tool="find_cars", filters=SearchFilters(max_price_ils=110_000)), PlanStep(tool="search_reviews")]
    res, retriever = await run(tight)
    assert res.needs_replan and len(res.steps) == 1 and not retriever.calls  # stops before searching
    relaxed, notes = relax_plan(tight, res.steps)
    assert relaxed[0].filters.max_price_ils == 121_000 and notes
    res2, _ = await run(relaxed, relax=True)
    assert res2.steps[0].car_ids == ["small-mhev"] and res2.chunks


async def test_rank_without_candidates_pushes_profile_constraints_to_catalog() -> None:
    profile = UserProfile(budget_max_ils=180_000, powertrain_pref=["ev"])
    assert profile_filters(profile).max_price_ils == 198_000
    res, _ = await run([PlanStep(tool="rank_by_profile", limit=3), PlanStep(tool="search_reviews")], profile)
    assert res.steps[0].car_ids == ["ev-cheap"]  # Kia (no price) and non-EVs never leave the catalog


def test_resolve_plan_and_ensure_search() -> None:
    refs = {ref_key("MG S6"): ["ev-cheap"], "#2": ["ev-big"]}
    plan = [PlanStep(tool="rank_by_profile", cars=["MG S6", "Tesla 3", "#2"])] * 6
    clean = resolve_plan(plan, lambda r: refs.get(ref_key(r), []))
    assert len(clean) == 4 and clean[0].car_ids == ["ev-cheap", "ev-big"]
    final = ensure_search_step(clean, "q")
    assert len(final) == 4 and final[-1].tool == "search_reviews"
    junk = PlanStep(tool="search_reviews", query="overview, design, verdict, faq")
    good = PlanStep(tool="search_reviews", query="how is the interior design?")
    assert [s.query for s in resolve_plan([junk, good], lambda r: [])] == ["", "how is the interior design?"]
    find = PlanStep(tool="find_cars", filters=SearchFilters(makes=["Citroën", "קיה", "Tesla"]))
    vocab = {"citroen": "Citroen", "קיה": "Kia"}
    assert resolve_plan([find], lambda r: [], vocab)[0].filters.makes == ["Citroen", "Kia", "Tesla"]


def test_planner_schema_hides_car_ids() -> None:
    from langchain_core.utils.function_calling import convert_to_openai_tool

    params = str(convert_to_openai_tool(TurnAnalysis)["function"]["parameters"])
    assert "car_mentions" in params and "focus_refs" in params
    assert "car_ids" not in params  # ids are resolved in code, never written by the model


# --------------------------------------------------------------------------- routing + helpers
def test_route() -> None:
    assert route({"intent": "general_chat"}) == "respond_direct"
    assert route({"intent": "car_question", "unknown_cars": ["Tesla Model 3"]}) == "respond_direct"
    assert route({"intent": "compare", "unknown_cars": ["Tesla"], "mentioned_car_ids": ["ev-cheap"]}) == "execute_plan"
    assert route({"intent": "car_question", "ambiguous_cars": {"the Hyundai": ["A", "B"]}}) == "respond_direct"
    assert route({"intent": "recommend", "user_profile": {}}) == "ask_clarifying"
    assert route({"intent": "recommend", "user_profile": {}, "clarified": True}) == "execute_plan"
    assert route({"intent": "recommend", "user_profile": {"budget_max_ils": 1}}) == "execute_plan"


def test_citations_and_language() -> None:
    assert nodes.cited_numbers("a [1] b [3, 2] c [4-6] d [1]") == [1, 3, 2, 4, 5, 6]
    chunks = [chunk("ev-cheap", "verdict").model_dump(), chunk("ev-cheap", "card").model_dump()]
    sources = nodes.build_sources("x [2] y [9]", chunks)
    assert [s["n"] for s in sources] == [2] and sources[0]["car_id"] == "ev-cheap"
    assert nodes.detect_language("איזה רכב חשמלי הכי טוב ל-MG?") == "he"
    assert nodes.detect_language("Which EV is best?") == "en"


def test_order_context_keeps_cards() -> None:
    chunks = [chunk(c, k).model_dump() for c in CARS for k in ("verdict", "design", "card")]
    out = nodes.order_context(chunks, ["petrol-fun"], limit=6)
    assert len(out) == 6 and out[0]["car_id"] == "petrol-fun"
    assert sum(c["chunk_type"] == "car_card" for c in out) == 4


async def test_compact_history_summarizes_but_keeps_transcript() -> None:
    msgs = [HumanMessage(f"m{i}") if i % 2 == 0 else AIMessage(f"a{i}") for i in range(18)]
    assert nodes.needs_compaction(msgs, 0) and not nodes.needs_compaction(msgs, 10)

    class R:
        context = make_ctx([], [])

    out = await nodes.compact_history({"messages": msgs, "summarized_upto": 0}, R())  # type: ignore[arg-type]
    assert out == {"summary": "summary text", "summarized_upto": 12}
    assert "m0" in str(R.context.fast_llm.prompts[0][1].content)


# --------------------------------------------------------------------------- full graph (fakes, in-memory)
async def test_graph_two_turns_profile_persists_and_streams() -> None:
    analyses = [
        TurnAnalysis(
            language="en",
            intent="recommend",
            standalone_query="Recommend an EV under 180k",
            profile_delta={
                "budget_max_ils": 180_000,
                "powertrain_pref": ["ev"],
                "body_pref": ["suv"],  # no quote -> dropped
                "evidence": [
                    {"field": "budget_max_ils", "quote": "under 180k"},
                    {"field": "powertrain_pref", "quote": "an EV"},
                ],
            },
            plan=[PlanStep(tool="rank_by_profile", limit=2), PlanStep(tool="search_reviews")],
        ),
        TurnAnalysis(
            language="en",
            intent="car_question",
            standalone_query="Tell me about the Tesla Model 3",
            car_mentions=["Tesla Model 3"],
        ),
    ]
    ctx = make_ctx(analyses, ["The MG S6 fits your budget [1] and is good value [2]. How many people ride?"])
    ctx.fast_llm.text = "I don't have a review of the Tesla Model 3."
    graph = build_graph(InMemorySaver())
    tokens: list[str] = []
    out = await run_turn(graph, thread_id="t1", message="I want an EV under 180k", context=ctx, on_token=tokens.append)
    assert out["intent"] == "recommend" and len(tokens) > 1
    assert "".join(tokens).strip() == out["answer"]  # streamed once (the final state message is not re-emitted)
    assert out["profile"]["budget_max_ils"] == 180_000
    assert out["profile"]["body_pref"] == []
    assert {s["car_id"] for s in out["sources"]} == {"ev-cheap"}
    assert any("rank_by_profile" in s for s in out["steps"])

    out2 = await run_turn(graph, thread_id="t1", message="What about a Tesla Model 3?", context=ctx)
    assert out2["sources"] == [] and "Tesla" in out2["answer"]
    assert out2["profile"]["powertrain_pref"] == ["ev"]  # persisted via the checkpointer
    state = (await graph.aget_state({"configurable": {"thread_id": "t1"}})).values
    assert len(state["messages"]) == 4 and state["retrieved"] == []  # per-turn keys reset


def test_evidence_documents_put_catalog_selections_first():
    from autochat.observability import evidence_documents, evidence_span

    docs = evidence_documents(
        ["find_cars(sorted by price_ils asc, top 1) -> ev-cheap"], [chunk("ev-cheap", "card").model_dump()]
    )
    assert docs[0].page_content.startswith("Catalog query over all reviewed cars")
    assert "ev-cheap" in docs[0].page_content and docs[0].metadata["doc_uri"] == "catalog://cars"
    assert docs[1].metadata["chunk_id"] == "ev-cheap:card"
    with evidence_span({"query": "q"}) as span:  # no active trace -> no stray root trace
        assert span is None
