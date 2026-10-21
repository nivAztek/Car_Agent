"""Structured outputs of the planner (`analyze_turn`) and the answer source model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from autochat.agent.profile import ProfileDelta
from autochat.retrieval.filters import SearchFilters, SortField

Intent = Literal["recommend", "compare", "car_question", "general_chat", "out_of_scope"]
Language = Literal["he", "en"]
ToolName = Literal["find_cars", "rank_by_profile", "search_reviews"]
SectionType = Literal[
    "overview",
    "value",
    "design",
    "cockpit_tech",
    "interior_space",
    "trunk",
    "performance",
    "efficiency_range",
    "ride_handling",
    "equipment_safety",
    "verdict",
    "faq",
]
MAX_STEPS = 4


class PlanStep(BaseModel):
    """One tool call. Steps run in order; each step's cars feed the next one."""

    tool: ToolName
    cars: list[str] = Field(
        default_factory=list, description="Cars for this step: names from car_mentions, or '#n' for focus car n"
    )
    # Resolved from `cars` in code (the planner never sees car ids).
    car_ids: SkipJsonSchema[list[str]] = Field(default_factory=list)
    use_previous: bool = Field(True, description="rank/search: also use the cars returned by the previous step")
    filters: SearchFilters = Field(default_factory=SearchFilters, description="find_cars: hard constraints")
    sort_by: SortField | None = Field(None, description="find_cars: numeric field to sort by")
    descending: bool = Field(False, description="find_cars: sort high->low (e.g. biggest range, most power)")
    limit: int | None = Field(None, ge=1, le=8, description="find_cars / rank_by_profile: keep the top N")
    query: str = Field("", description="search_reviews: English search query (default: the standalone query)")
    section_types: list[SectionType] = Field(default_factory=list, description="search_reviews: optional focus")


class TurnAnalysis(BaseModel):
    language: Language = Field(description="Language of the LAST user message")
    intent: Intent
    standalone_query: str = Field(description="The last message rewritten as a self-contained English request")
    car_mentions: list[str] = Field(
        default_factory=list,
        description="Every specific car (make and/or model) named in the LAST message, copied verbatim in the "
        "user's spelling and language, e.g. 'Toyota Corolla', 'הקורולה', 'Tesla Model 3'. "
        "Not categories ('electric SUV').",
    )
    focus_refs: list[int] = Field(
        default_factory=list,
        description="Numbers of focus cars the LAST message refers to indirectly ('it', 'the second one', 'both')",
    )
    profile_delta: ProfileDelta = Field(default_factory=ProfileDelta)
    plan: list[PlanStep] = Field(default_factory=list, description=f"0-{MAX_STEPS} tool steps; empty for chat")


class Source(BaseModel):
    n: int
    car_id: str
    car: str
    heading: str
    section_type: str
    url: str
