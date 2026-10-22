"""User preference profile: inferred gradually, persisted per thread (checkpointer), merged, never overwritten."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from autochat.ingestion.schemas import BodyType, Powertrain

UseCase = Literal["commute", "family", "long_trips", "performance", "offroad_light", "city"]
Priority = Literal["comfort", "performance", "practicality", "tech", "quality", "value", "efficiency", "fun"]
ListField = Literal[
    "use_cases",
    "powertrain_pref",
    "body_pref",
    "priorities",
    "must_haves",
    "dealbreakers",
    "cars_liked",
    "cars_rejected",
]

LIST_FIELDS: tuple[str, ...] = (
    "use_cases",
    "powertrain_pref",
    "body_pref",
    "priorities",
    "must_haves",
    "dealbreakers",
    "cars_liked",
    "cars_rejected",
)
USE_CASE_TAGS: dict[str, str] = {
    "commute": "commuter",
    "family": "family",
    "long_trips": "long_trips",
    "performance": "enthusiast",
    "offroad_light": "offroad_light",
    "city": "city",
}


class UserProfile(BaseModel):
    budget_max_ils: int | None = Field(None, description="Upper budget in ILS ('around 200k' -> 200000)")
    budget_min_ils: int | None = None
    use_cases: list[UseCase] = Field(default_factory=list)
    passengers: int | None = Field(None, description="Usual number of occupants incl. driver")
    powertrain_pref: list[Powertrain] = Field(default_factory=list, description="ev | ice | hybrid | phev | mhev")
    body_pref: list[BodyType] = Field(default_factory=list)
    priorities: list[Priority] = Field(default_factory=list, description="Most important first")
    must_haves: list[str] = Field(default_factory=list, description="Short English phrases")
    dealbreakers: list[str] = Field(default_factory=list, description="Short English phrases")
    home_charging: bool | None = None
    cars_liked: list[str] = Field(default_factory=list, description="car_ids the user liked")
    cars_rejected: list[str] = Field(default_factory=list, description="car_ids the user rejected")


class FieldQuote(BaseModel):
    field: str = Field(description="Profile field name, e.g. 'budget_max_ils', 'powertrain_pref'")
    quote: str = Field(description="VERBATIM words from the LAST user message that state this field")


class ProfileDelta(UserProfile):
    """Only what the LAST user message adds or changes. Empty when it reveals nothing."""

    cars_liked: list[str] = Field(
        default_factory=list, description="Cars the user says they like: names from car_mentions or '#n'"
    )
    cars_rejected: list[str] = Field(
        default_factory=list, description="Cars the user rules out: names from car_mentions or '#n'"
    )
    replace_lists: list[ListField] = Field(
        default_factory=list,
        description="List fields the user CHANGED their mind about (replace instead of append), "
        "e.g. 'actually petrol is fine' -> ['powertrain_pref']",
    )
    evidence: list[FieldQuote] = Field(
        default_factory=list, description="One verbatim quote per field you set; fields without a quote are dropped"
    )


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# Fields the small planner model tends to over-infer: the quote must also contain an explicit cue.
# ("I can charge at home" is not a powertrain preference; "compare X and Y" is not liking them.)
FIELD_CUES: dict[str, re.Pattern[str]] = {
    # a budget needs a figure: "cheapest electric car" made the planner guess a budget (the lowest price)
    "budget_max_ils": re.compile(r"\d|thousand|million|אלף|מיליון", re.I),
    "budget_min_ils": re.compile(r"\d|thousand|million|אלף|מיליון", re.I),
    "powertrain_pref": re.compile(
        r"electric|\bevs?\b|hybrid|plug|phev|petrol|gasoline|gas\b|diesel|benzin|fuel|"
        r"חשמל|היבריד|הייבריד|פלאג|בנזין|דיזל|דלק",
        re.I,
    ),
    "body_pref": re.compile(
        r"suv|crossover|sedan|hatch|wagon|estate|coupe|convertible|mpv|minivan|pickup|"
        r"פנאי|ג'יפ|גיפ|סדאן|האצ'בק|סטיישן|קופה|מיניוואן|טנדר",
        re.I,
    ),
    "cars_liked": re.compile(r"like|love|prefer|favou?r|great|best|interest|אוהב|מעדיף|אהבתי|מוצא חן|מעניין", re.I),
    "cars_rejected": re.compile(
        r"not|n't|no\b|hate|dislike|rule|skip|too |avoid|without|לא|בלי|פחות|מוותר|יקר מדי",
        re.I,
    ),
}
# replace_lists (overwrite instead of append) only when the user signals a change of mind.
CHANGE_CUE = re.compile(
    r"actually|instead|change[ds]? my mind|rather|no longer|any ?more|on second thought|"
    r"בעצם|במקום|התחרטתי|דווקא|כבר לא|שיניתי",
    re.I,
)


def quote_in(quote: str, message: str, min_overlap: float = 0.8) -> bool:
    """Quote is (almost) verbatim from the message: normalized substring, or >= 80% of its words present."""
    q, m = _words(quote), _words(message)
    if not q:
        return False
    if " ".join(q) in " ".join(m):
        return True
    present = set(m)
    return sum(w in present for w in q) / len(q) >= min_overlap


def validate_delta(
    delta: ProfileDelta | dict | None, message: str, resolve_car: Callable[[str], list[str]] | None = None
) -> tuple[ProfileDelta, list[str]]:
    """Drop every field the planner set without a verbatim quote from the user's last message (the same
    'evidence or null' rule as ingestion). Liked/rejected cars are names/'#n' refs mapped to car_ids by
    `resolve_car` (refs that do not resolve to a reviewed car are dropped).
    Returns (clean_delta, dropped_field_names)."""
    d = ProfileDelta.model_validate(delta or {})
    quoted = {
        e.field
        for e in d.evidence
        if quote_in(e.quote, message) and (e.field not in FIELD_CUES or FIELD_CUES[e.field].search(e.quote))
    }
    data = d.model_dump(exclude={"evidence"})
    dropped: list[str] = []
    for name in UserProfile.model_fields:
        value = data[name]
        if value in (None, []):
            continue
        if name not in quoted:
            data[name] = [] if name in LIST_FIELDS else None
            dropped.append(name)
        elif name in ("cars_liked", "cars_rejected") and resolve_car is not None:
            data[name] = list(dict.fromkeys(cid for ref in value for cid in resolve_car(ref)))
    data["replace_lists"] = [f for f in data["replace_lists"] if f in quoted] if CHANGE_CUE.search(message) else []
    return ProfileDelta.model_validate(data), dropped


def _union(old: list[Any], new: list[Any]) -> list[Any]:
    return list(dict.fromkeys([*old, *new]))


def merge_profile(old: UserProfile | dict | None, delta: ProfileDelta | dict | None) -> UserProfile:
    """Scalars: replaced when the delta value is not None. Lists: unioned in order, or replaced if listed in
    `replace_lists`. Priorities just stated move to the front. Liked/rejected cars move between lists."""
    base = UserProfile.model_validate(old or {})
    if delta is None:
        return base
    d = ProfileDelta.model_validate(delta)
    data = base.model_dump()
    for name in UserProfile.model_fields:
        value = getattr(d, name)
        if name in LIST_FIELDS:
            if name in d.replace_lists:
                data[name] = list(dict.fromkeys(value))
            elif name == "priorities":
                data[name] = _union(value, data[name])
            else:
                data[name] = _union(data[name], value)
        elif value is not None:
            data[name] = value
    data["cars_liked"] = [c for c in data["cars_liked"] if c not in d.cars_rejected]
    data["cars_rejected"] = [c for c in data["cars_rejected"] if c not in d.cars_liked]
    return UserProfile.model_validate(data)


def needs_clarification(profile: UserProfile) -> bool:
    """Recommend without ANY anchor (budget, use, passengers, powertrain, body, priority, must-have) -> ask first."""
    return (
        profile.budget_max_ils is None
        and profile.passengers is None
        and not any(
            [
                profile.use_cases,
                profile.priorities,
                profile.powertrain_pref,
                profile.body_pref,
                profile.must_haves,
                profile.cars_liked,
            ]
        )
    )


def next_missing_slot(profile: UserProfile) -> str | None:
    """The single most valuable thing to ask next (budget > use case > passengers > powertrain)."""
    if profile.budget_max_ils is None:
        return "budget"
    if not profile.use_cases:
        return "main use (commuting, family, long trips, fun driving, city)"
    if "family" in profile.use_cases and profile.passengers is None:
        return "how many people usually ride"
    if not profile.powertrain_pref:
        return "powertrain preference (electric, plug-in hybrid, petrol)"
    return None


def profile_summary(profile: UserProfile) -> str:
    parts = [f"{k}={v}" for k, v in profile.model_dump(exclude_defaults=True).items()]
    return ", ".join(parts) or "(nothing known yet)"
