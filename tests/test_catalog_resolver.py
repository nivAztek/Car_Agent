"""Car catalog + name resolver (offline, in-memory). The names mirror the real corpus (data/processed/cars.json)."""

from __future__ import annotations

import pytest
from fakes import CARS, make_ctx
from langgraph.checkpoint.memory import InMemorySaver

from autochat.agent.graph import build_graph
from autochat.agent.runtime import run_turn
from autochat.agent.schemas import PlanStep, TurnAnalysis
from autochat.ingestion.schemas import CarRecord
from autochat.names import car_name_keys, compact, normalize
from autochat.retrieval.catalog import MAX_CANDIDATES, InMemoryCarCatalog
from autochat.retrieval.filters import SearchFilters
from autochat.retrieval.resolver import ref_key, resolve_mentions


def car(car_id: str, make: str, model: str, make_he: str, model_he: str, **kw: object) -> dict:
    aliases = [make.lower(), model.lower(), f"{make} {model}".lower(), car_id.replace("-", " ")]
    return {"car_id": car_id, "make": make, "model": model, "make_he": make_he, "model_he": model_he,
            "aliases": aliases, **kw}  # fmt: skip


REAL = [
    car("aion-ht", "GAC", "Aion HT", "GAC", "איון HT", powertrain="ev", price_ils=169_990),
    car("audi-rs3-fl", "Audi", "RS3", "אודי", "RS3", powertrain="ice", price_ils=575_990),
    car("citroen-c3-2026", "Citroen", "C3", "סיטרואן", "C3", powertrain="mhev", trim="מקס", price_ils=114_990),
    car("genesis-gv80-2026", "Genesis", "GV80", "ג'נסיס", "GV80", powertrain="phev", price_ils=None),
    car("hyundai-elantra-n-manual", "Hyundai", "Elantra N", "יונדאי", "אלנטרה N", powertrain="ice"),
    car("link-and-co-01-2026", "Lynk & Co", "01", "לינק & קו", "01", powertrain="phev"),
    car("long-term-report-kia-ev9-4", "Kia", "EV9", "קיה", "EV9", trim="GT ליין", powertrain="ev"),
    car("mg-s6", "MG", "S6", "MG", "S6", powertrain="ev", price_ils=168_888),
]


@pytest.fixture
def catalog() -> InMemoryCarCatalog:
    return InMemoryCarCatalog(REAL)


async def resolve(catalog: InMemoryCarCatalog, mention: str):
    return (await resolve_mentions([mention], catalog))[ref_key(mention)]


# --------------------------------------------------------------------------- names
def test_normalize_and_keys() -> None:
    assert normalize("Lynk & Co 01") == normalize("lynk and co 01") == "lynk co 01"
    assert compact("ג׳נסיס GV80") == compact("ג'נסיס gv-80") == "גנסיסgv80"
    keys = car_name_keys(REAL[5])
    assert "lynkco01" in keys and "01" not in keys  # numeric-only model is not a key on its own
    assert "audi" not in car_name_keys(REAL[1])  # never the make alone
    rec = CarRecord(id="x", car_id="x", url="u", title="t", make="Kia", model="EV9", make_he="קיה", model_he="EV9")
    assert {"kiaev9", "קיהev9", "ev9"} <= set(rec.name_keys)  # derived on every record


# --------------------------------------------------------------------------- resolver
@pytest.mark.parametrize(
    ("mention", "car_id"),
    [
        ("Kia EV9", "long-term-report-kia-ev9-4"),
        ("the Kia", "long-term-report-kia-ev9-4"),
        ("הקיה", "long-term-report-kia-ev9-4"),
        ("the new Kia EV9 GT-Line", "long-term-report-kia-ev9-4"),
        ("the S6", "mg-s6"),
        ("Lynk & Co 01", "link-and-co-01-2026"),
        ("Link and Co 01", "link-and-co-01-2026"),
        ("לינק אנד קו", "link-and-co-01-2026"),
        ("Genesys GV80", "genesis-gv80-2026"),
        ("ג׳נסיס", "genesis-gv80-2026"),
        ("Hyundia Elantra N", "hyundai-elantra-n-manual"),
        ("האלנטרה N", "hyundai-elantra-n-manual"),
        ("Audi RS 3", "audi-rs3-fl"),
        ("Citroën C3", "citroen-c3-2026"),
        ("סיטרואן C3 החדשה", "citroen-c3-2026"),
    ],
)
async def test_resolves_spelling_hebrew_and_typos(catalog: InMemoryCarCatalog, mention: str, car_id: str) -> None:
    r = await resolve(catalog, mention)
    assert (r.status, r.car_ids) == ("resolved", [car_id])


@pytest.mark.parametrize(
    ("mention", "suggestions"),
    [
        ("Audi Q5", ["Audi RS3"]),  # same brand, different model: never mapped
        ("Audi S6", ["Audi RS3"]),  # model key of another make (MG S6) must not win
        ("Kia EV6", ["Kia EV9"]),  # digits differ
        ("Citroen C3 Aircross", ["Citroen C3"]),  # extra model word
        ("Tesla Model 3", []),
        ("electric SUV", []),
    ],
)
async def test_unknown_cars_are_not_mapped(catalog: InMemoryCarCatalog, mention: str, suggestions: list[str]) -> None:
    r = await resolve(catalog, mention)
    assert (r.status, r.car_ids, r.suggestions) == ("unknown", [], suggestions)


async def test_ambiguous_make_lists_candidates() -> None:
    cat = InMemoryCarCatalog([*REAL, car("hyundai-ioniq-5", "Hyundai", "Ioniq 5", "יונדאי", "איוניק 5")])
    r = await resolve(cat, "the Hyundai")
    assert r.status == "ambiguous" and r.names == ["Hyundai Elantra N", "Hyundai Ioniq 5"]
    assert (await resolve(cat, "Ioniq 5")).car_ids == ["hyundai-ioniq-5"]


# --------------------------------------------------------------------------- catalog
async def test_catalog_find_sort_missing_last_and_cap(catalog: InMemoryCarCatalog) -> None:
    cheap = await catalog.find(SearchFilters(), sort_by="price_ils", limit=3)
    assert [c["car_id"] for c in cheap] == ["citroen-c3-2026", "mg-s6", "aion-ht"]
    every = await catalog.find(SearchFilters(), sort_by="price_ils", descending=True)
    assert every[0]["car_id"] == "audi-rs3-fl" and every[-1].get("price_ils") is None
    big = InMemoryCarCatalog([car(f"c{i}", "Make", f"M{i}", "", "") for i in range(MAX_CANDIDATES + 10)])
    assert len(await big.find()) == MAX_CANDIDATES


async def test_scope_is_fixed_size_and_names_no_car(catalog: InMemoryCarCatalog) -> None:
    scope = await catalog.scope()
    text = scope.describe()
    assert scope.count == 8 and "8 reviewed cars from 8 makes" in text
    assert not any(c["model"] in text for c in REAL)
    assert scope.make_vocab()["קיה"] == "Kia"


async def test_catalog_page(catalog: InMemoryCarCatalog) -> None:
    rows, total = await catalog.page("", 0, 3)
    assert total == 8 and [c["make"] for c in rows] == ["Audi", "Citroen", "GAC"]
    rows, total = await catalog.page("קיה", 0, 10)
    assert total == 1 and rows[0]["car_id"] == "long-term-report-kia-ev9-4"


# --------------------------------------------------------------------------- graph: focus refs, no catalog in prompts
async def test_focus_refs_and_prompts_without_catalog() -> None:
    analyses = [
        TurnAnalysis(
            language="en",
            intent="compare",
            standalone_query="Compare the MG S6 and the Kia EV9",
            car_mentions=["MG S6", "Kia EV9"],
            plan=[PlanStep(tool="search_reviews", cars=["MG S6", "Kia EV9"], use_previous=False)],
        ),
        TurnAnalysis(
            language="en",
            intent="car_question",
            standalone_query="Tell me more about the Kia EV9",
            focus_refs=[1],
            plan=[PlanStep(tool="search_reviews")],
        ),
    ]
    # context order: [1] ev-cheap verdict, [2] ev-cheap card, [3] ev-big verdict, [4] ev-big card
    ctx = make_ctx(analyses, ["The Kia is roomier [3]; the MG is cheaper [1].", "It is comfortable [1]."])
    graph = build_graph(InMemorySaver())
    await run_turn(graph, thread_id="f1", message="MG S6 or Kia EV9?", context=ctx)
    state = (await graph.aget_state({"configurable": {"thread_id": "f1"}})).values
    assert state["focus_car_ids"] == ["ev-big", "ev-cheap"]  # order the answer cited them

    await run_turn(graph, thread_id="f1", message="Tell me more about the first one", context=ctx)
    assert ctx.retriever.calls[-1]["car_ids"] == ["ev-big"]
    planner_user = str(ctx.fast_llm.prompts[1][1].content)
    assert "#1 Kia EV9" in planner_user and "#2 MG S6" in planner_user
    for prompt in ctx.fast_llm.prompts:
        system = str(prompt[0].content)
        assert not any(c["model"] in system or c["car_id"] in system for c in CARS.values())
