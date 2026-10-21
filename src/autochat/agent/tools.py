"""Planner tools (LangChain StructuredTools -> TOOL spans in MLflow) and the code-only plan executor.

The planner (`analyze_turn`) writes a typed plan; the executor runs it with no LLM calls:
    find_cars       exact metadata filter + sort on the `cars` container (via the catalog, capped)
    rank_by_profile soft scoring of candidates against the inferred user profile; without explicit candidates the
                    pool is the catalog filtered by the profile's hard constraints (SQL pushdown, capped)
    search_reviews  hybrid retrieval (vector + BM25) of review chunks for the current cars
Each step's resulting cars feed the next step (`use_previous`).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_args

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from autochat.agent.profile import USE_CASE_TAGS, UserProfile
from autochat.agent.schemas import MAX_STEPS, PlanStep, SectionType
from autochat.names import compact
from autochat.retrieval.catalog import MAX_CANDIDATES, CarCatalog
from autochat.retrieval.filters import SearchFilters, SortField, expand_body_types, expand_powertrains
from autochat.retrieval.hybrid import Retriever

log = logging.getLogger(__name__)

PRIORITY_WEIGHTS = (1.0, 0.7, 0.5, 0.35, 0.25, 0.2, 0.15, 0.1)
USE_CASE_PRIORITIES: dict[str, list[str]] = {
    "family": ["practicality", "comfort"],
    "performance": ["fun", "performance"],
    "commute": ["efficiency", "comfort", "value"],
    "long_trips": ["comfort", "efficiency"],
    "city": ["value", "efficiency"],
    "offroad_light": ["practicality"],
}
BUDGET_TOLERANCE = 1.10
NEUTRAL_SCORE = 3
MAX_QUERY_CHARS = 300
SECTION_NAMES = frozenset(get_args(SectionType))


@dataclass
class StepResult:
    step: int
    tool: str
    args: dict[str, Any]
    car_ids: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    note: str = ""
    empty: bool = False

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class ExecutionResult:
    steps: list[StepResult]
    car_ids: list[str]  # every car the plan touched, in first-seen order
    chunks: list[dict[str, Any]]
    needs_replan: bool


# --------------------------------------------------------------------------- tool argument schemas
class FindCarsArgs(BaseModel):
    filters: SearchFilters = Field(default_factory=SearchFilters)
    sort_by: SortField | None = None
    descending: bool = False
    limit: int | None = None


class RankArgs(BaseModel):
    car_ids: list[str] = Field(default_factory=list, description="Candidates; empty = all cars")
    limit: int = 3
    relax_constraints: bool = False


class SearchArgs(BaseModel):
    query: str
    car_ids: list[str] = Field(default_factory=list, description="Empty = all cars")
    section_types: list[SectionType] = Field(default_factory=list)


# --------------------------------------------------------------------------- profile ranking (pure)
def profile_weights(profile: UserProfile) -> dict[str, float]:
    order = list(profile.priorities)
    for uc in profile.use_cases:
        order += [p for p in USE_CASE_PRIORITIES.get(uc, []) if p not in order]
    return {p: PRIORITY_WEIGHTS[min(i, len(PRIORITY_WEIGHTS) - 1)] for i, p in enumerate(dict.fromkeys(order))}


def passes_profile(car: dict[str, Any], profile: UserProfile) -> bool:
    if car["car_id"] in profile.cars_rejected:
        return False
    if profile.budget_max_ils is not None:
        price = car.get("price_ils")
        if price is None or price > profile.budget_max_ils * BUDGET_TOLERANCE:
            return False
    if profile.powertrain_pref and car.get("powertrain") not in expand_powertrains(profile.powertrain_pref):
        return False
    return not (profile.body_pref and car.get("body_type") not in expand_body_types(profile.body_pref))


def profile_filters(profile: UserProfile) -> SearchFilters:
    """The profile's hard constraints as a catalog filter (same semantics as `passes_profile`)."""
    budget = profile.budget_max_ils
    return SearchFilters(
        max_price_ils=int(budget * BUDGET_TOLERANCE) if budget is not None else None,
        powertrains=list(profile.powertrain_pref),
        body_types=list(profile.body_pref),
    )


def score_car(car: dict[str, Any], profile: UserProfile) -> tuple[float, list[str]]:
    """Σ priority weight × aspect score (1-5, unknown = 3) + 0.5 per matching use-case tag + 1 if liked."""
    weights = profile_weights(profile)
    aspects = car.get("aspect_scores") or {}
    score = sum(w * aspects.get(a, NEUTRAL_SCORE) for a, w in weights.items())
    reasons = [f"{a} {aspects[a]}/5" for a in weights if a in aspects and aspects[a] >= 4]
    tags = set(car.get("tags") or [])
    for uc in profile.use_cases:
        if USE_CASE_TAGS.get(uc) in tags:
            score += 0.5
            reasons.append(f"suits {uc}")
    if car["car_id"] in profile.cars_liked:
        score += 1.0
        reasons.append("you liked it")
    return round(score, 2), reasons


def rank_cars(
    cars: list[dict[str, Any]], profile: UserProfile, limit: int = 3, relax: bool = False
) -> list[tuple[dict[str, Any], float, list[str]]]:
    pool = cars if relax else [c for c in cars if passes_profile(c, profile)]
    pool = [c for c in pool if c["car_id"] not in profile.cars_rejected]
    scored = [(c, *score_car(c, profile)) for c in pool]
    scored.sort(key=lambda t: (-t[1], t[0].get("price_ils") or 10**9))
    return scored[:limit]


# --------------------------------------------------------------------------- formatting helpers
def car_label(car: dict[str, Any]) -> str:
    price = f"₪{car['price_ils']:,}" if car.get("price_ils") else "price not stated"
    return f"{car['car_id']} ({car.get('make', '')} {car.get('model', '')}, {car.get('powertrain') or '?'}, {price})"


def describe_filters(f: SearchFilters, sort_by: str | None, descending: bool, limit: int | None) -> str:
    parts = [f"{k}={v}" for k, v in f.model_dump(exclude_defaults=True).items()]
    if sort_by:
        parts.append(f"sorted by {sort_by} {'desc' if descending else 'asc'}")
    if limit:
        parts.append(f"top {limit}")
    return ", ".join(parts) or "all cars"


# --------------------------------------------------------------------------- tools
def make_tools(
    *,
    retriever: Retriever,
    catalog: CarCatalog,
    profile: UserProfile,
) -> dict[str, StructuredTool]:
    async def find_cars(
        filters: SearchFilters | dict | None = None,
        sort_by: SortField | None = None,
        descending: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        f = SearchFilters.model_validate(filters or {})
        found = await catalog.find(f, sort_by=sort_by, descending=descending, limit=limit)
        return [{"car_id": c["car_id"], "label": car_label(c)} for c in found]

    async def rank_by_profile(
        car_ids: list[str] | None = None, limit: int = 3, relax_constraints: bool = False
    ) -> list[dict[str, Any]]:
        if car_ids:
            pool = await catalog.get(car_ids)
        else:
            hard = SearchFilters() if relax_constraints else profile_filters(profile)
            pool = await catalog.find(hard, limit=MAX_CANDIDATES)
        ranked = rank_cars(pool, profile, limit=limit, relax=relax_constraints)
        return [{"car_id": c["car_id"], "label": car_label(c), "score": s, "why": r} for c, s, r in ranked]

    async def search_reviews(
        query: str, car_ids: list[str] | None = None, section_types: list[str] | None = None
    ) -> list[dict[str, Any]]:
        chunks = await retriever.retrieve(
            query, car_ids or None, section_types=section_types or None, include_cards=True
        )
        return [c.model_dump() for c in chunks]

    return {
        "find_cars": StructuredTool.from_function(
            coroutine=find_cars,
            name="find_cars",
            args_schema=FindCarsArgs,
            description="Exact metadata filter and sort over the reviewed cars (price, powertrain, body, range...).",
        ),
        "rank_by_profile": StructuredTool.from_function(
            coroutine=rank_by_profile,
            name="rank_by_profile",
            args_schema=RankArgs,
            description="Rank cars for the user's inferred profile (budget, powertrain, priorities, use cases).",
        ),
        "search_reviews": StructuredTool.from_function(
            coroutine=search_reviews,
            name="search_reviews",
            args_schema=SearchArgs,
            description="Hybrid search (vector + keyword) over review excerpts of the given cars.",
        ),
    }


# --------------------------------------------------------------------------- plan helpers
def useful_query(query: str) -> bool:
    """The planner sometimes writes section names ('overview, design, ...') instead of a question: use the
    standalone query then."""
    words = [w for w in re.split(r"[\s,;]+", query.lower()) if w]
    if not words or len(query) > MAX_QUERY_CHARS:
        return False
    return sum(w in SECTION_NAMES for w in words) / len(words) < 0.5


def resolve_plan(
    plan: list[PlanStep], resolve_car: Callable[[str], list[str]], make_vocab: dict[str, str] | None = None
) -> list[PlanStep]:
    """Cap the plan and map each step's car refs (names / '#n') to car_ids; unresolvable refs are dropped.
    Make filters are mapped to the catalog's spelling ('Citroën', 'קיה' -> 'Citroen', 'Kia'); unknown makes stay
    (they match nothing, which is the truthful answer). Planner filters never carry car ids."""
    vocab = make_vocab or {}
    out = []
    for step in plan[:MAX_STEPS]:
        ids = list(dict.fromkeys([*step.car_ids, *(cid for ref in step.cars for cid in resolve_car(ref))]))
        makes = list(dict.fromkeys(vocab.get(compact(m), m) for m in step.filters.makes))
        update = {
            "car_ids": ids,
            "filters": step.filters.model_copy(update={"car_ids": [], "makes": makes}),
            "query": step.query if useful_query(step.query) else "",
        }
        out.append(step.model_copy(update=update))
    return out


def default_plan(intent: str, mentioned: list[str], query: str) -> list[PlanStep]:
    if intent == "recommend":
        return [
            PlanStep(tool="rank_by_profile", car_ids=mentioned, limit=3),
            PlanStep(tool="search_reviews", query=query),
        ]
    return [PlanStep(tool="search_reviews", car_ids=mentioned, use_previous=False, query=query)]


def ensure_search_step(plan: list[PlanStep], query: str) -> list[PlanStep]:
    """Every answer needs citable evidence: end with a search over the plan's cars."""
    if plan and plan[-1].tool == "search_reviews":
        return plan
    return [*plan[: MAX_STEPS - 1], PlanStep(tool="search_reviews", query=query)]


def relax_plan(plan: list[PlanStep], results: list[StepResult]) -> tuple[list[PlanStep], list[str]]:
    """Deterministic replanner: loosen the step(s) that came back empty. Returns (new_plan, notes)."""
    empty = {r.step for r in results if r.empty}
    notes: list[str] = []
    new = []
    for i, step in enumerate(plan):
        if i not in empty:
            new.append(step)
            continue
        if step.tool == "find_cars":
            f = step.filters
            update: dict[str, Any] = {"min_range_km": None, "max_accel_0_100_s": None, "tags": [], "body_types": []}
            if f.max_price_ils:
                update["max_price_ils"] = int(f.max_price_ils * BUDGET_TOLERANCE)
            relaxed = f.model_copy(update=update)
            notes.append(
                f"No car matched [{describe_filters(f, None, False, None)}]; "
                f"relaxed to [{describe_filters(relaxed, None, False, None)}]"
            )
            new.append(step.model_copy(update={"filters": relaxed}))
        elif step.tool == "rank_by_profile":
            notes.append("No car meets all profile constraints (budget/powertrain/body); showing the closest matches")
            new.append(step)  # re-run with relax=True
        else:
            notes.append("No excerpts matched the focused search; searched all sections")
            new.append(step.model_copy(update={"section_types": []}))
    return new, notes


async def execute_plan(
    plan: list[PlanStep],
    tools: dict[str, StructuredTool],
    *,
    standalone_query: str,
    relax: bool = False,
) -> ExecutionResult:
    """Run the plan in order (no LLM). Stops early when a car-selecting step returns nothing."""
    steps: list[StepResult] = []
    current: list[str] = []
    touched: dict[str, None] = {}
    chunks: dict[str, dict[str, Any]] = {}

    for i, step in enumerate(plan):
        prev = current if step.use_previous else []
        ids = list(dict.fromkeys([*step.car_ids, *prev]))
        if step.tool == "find_cars":
            f = step.filters
            if ids and not f.car_ids:
                f = f.model_copy(update={"car_ids": ids})
            args = {
                "filters": f.model_dump(exclude_defaults=True),
                "sort_by": step.sort_by,
                "descending": step.descending,
                "limit": step.limit,
            }
            found = await tools["find_cars"].ainvoke(args)
            res = StepResult(
                i,
                step.tool,
                args,
                car_ids=[c["car_id"] for c in found],
                note="; ".join(c["label"] for c in found)
                or f"no car matched [{describe_filters(f, step.sort_by, step.descending, step.limit)}]",
            )
            res.note = f"find_cars({describe_filters(f, step.sort_by, step.descending, step.limit)}) -> {res.note}"
        elif step.tool == "rank_by_profile":
            args = {"car_ids": ids, "limit": step.limit or 3, "relax_constraints": relax}
            ranked = await tools["rank_by_profile"].ainvoke(args)
            res = StepResult(i, step.tool, args, car_ids=[c["car_id"] for c in ranked])
            res.note = "rank_by_profile -> " + (
                "; ".join(f"{c['label']} score {c['score']} ({', '.join(c['why']) or 'no standout'})" for c in ranked)
                or "no car meets the profile constraints"
            )
        else:
            target = ids or None
            args = {
                "query": step.query or standalone_query,
                "car_ids": target or [],
                "section_types": list(step.section_types),
            }
            found_chunks = await tools["search_reviews"].ainvoke(args)
            for c in found_chunks:
                chunks.setdefault(c["id"], c)
            res = StepResult(i, step.tool, args, car_ids=target or [], chunk_ids=[c["id"] for c in found_chunks])
            res.note = f"search_reviews('{args['query']}') -> {len(found_chunks)} excerpts"
        res.empty = not (res.chunk_ids if step.tool == "search_reviews" else res.car_ids)
        steps.append(res)
        if step.tool != "search_reviews":
            current = res.car_ids
        for c in res.car_ids:
            touched.setdefault(c, None)
        for c in chunks.values():
            touched.setdefault(c["car_id"], None)
        if res.empty and step.tool != "search_reviews":
            break  # later steps depend on these cars

    needs_replan = any(s.empty for s in steps)
    return ExecutionResult(steps=steps, car_ids=list(touched), chunks=list(chunks.values()), needs_replan=needs_replan)
