"""Offline retrieval tests: filters SQL, tokenizer, BM25 store, RRF and the hybrid retriever with fakes."""

from __future__ import annotations

import pytest

from autochat.retrieval.bm25 import ChunkStore, tokenize
from autochat.retrieval.filters import SearchFilters, build_query, expand_powertrains, sort_cars
from autochat.retrieval.hybrid import HybridRetriever, cap_per_car, rrf

# --------------------------------------------------------------------------- filters


def test_empty_filters_select_all() -> None:
    assert SearchFilters().is_empty()
    assert build_query(SearchFilters()) == ("SELECT * FROM c", [])


def test_build_query_is_fully_parameterized() -> None:
    evil = "x') OR 1=1 --"
    f = SearchFilters(
        makes=[evil],
        powertrains=["ice"],
        max_price_ils=230_000,
        min_range_km=400,
        tags=["family", "budget"],
        min_seats=7,
    )
    sql, params = build_query(f)
    assert evil.lower() not in sql and "230000" not in sql and "400" not in sql
    values = {p["name"]: p["value"] for p in params}
    assert values["@makes"] == [evil.lower()]
    assert values["@powertrains"] == ["ice", "mhev"]  # petrol includes mild hybrids
    assert values["@max_price"] == 230_000
    assert values["@tag0"] == "family" and values["@tag1"] == "budget"
    assert "NOT IS_NUMBER(c.seats)" in sql  # unknown seat count does not exclude a car
    assert sql.count("@min_range") == 2


def test_expand_powertrains() -> None:
    assert expand_powertrains(["hybrid", "ev"]) == ["hybrid", "mhev", "phev", "ev"]
    assert expand_powertrains(["phev"]) == ["phev"]


def test_sort_cars_puts_unknown_last() -> None:
    cars = [{"car_id": "a", "price_ils": 200}, {"car_id": "b", "price_ils": None}, {"car_id": "c", "price_ils": 100}]
    assert [c["car_id"] for c in sort_cars(cars, "price_ils")] == ["c", "a", "b"]
    assert [c["car_id"] for c in sort_cars(cars, "price_ils", descending=True)] == ["a", "c", "b"]
    with pytest.raises(ValueError, match="cannot sort"):
        sort_cars(cars, "id; DROP")


# --------------------------------------------------------------------------- tokenizer + BM25


def test_tokenize_hebrew_and_english() -> None:
    assert tokenize("The trunk is 450 litres") == ["trunk", "450", "litres"]
    assert tokenize('ק"מ') == ["קמ"]  # gershayim removed
    assert tokenize("והמטען") == ["המטען"]  # one prefix letter stripped
    assert tokenize("של את") == []  # stopwords


def _doc(doc_id: str, car_id: str, text: str, section_type: str = "general", make: str = "", model: str = "") -> dict:
    return {
        "id": doc_id,
        "car_id": car_id,
        "chunk_type": "car_card" if doc_id.endswith(":card") else "section",
        "section_type": section_type,
        "heading": doc_id,
        "heading_en": doc_id,
        "header": f"{make} {model}",
        "text_he": "",
        "text_en": text,
        "url": f"https://x/{car_id}",
        "make": make,
        "model": model,
    }


DOCS = [
    _doc("a:00:0", "a", "Alpha trunk is huge, 600 litres of trunk space", "trunk", "Alpha", "One"),
    _doc("a:01:0", "a", "Alpha One ride comfort is soft", "ride_handling", "Alpha", "One"),
    _doc("a:faq:00", "a", "Alpha One Alpha One Alpha One price question", "faq", "Alpha", "One"),
    _doc("a:card", "a", "Car card Alpha One price 100", "car_card", "Alpha", "One"),
    _doc("b:00:0", "b", "Beta engine is loud and fast", "performance", "Beta", "Two"),
    _doc("b:01:0", "b", "Beta trunk is small", "trunk", "Beta", "Two"),
    _doc("b:card", "b", "Car card Beta Two", "car_card", "Beta", "Two"),
    _doc("c:00:0", "c", "Gamma comfort and trunk", "ride_handling", "Gamma", "Three"),
    _doc("c:card", "c", "Car card Gamma", "car_card", "Gamma", "Three"),
]


def test_bm25_restricts_to_cars() -> None:
    store = ChunkStore(DOCS)
    hits = store.bm25_search("trunk", ["b"])
    assert hits and all(store.docs[i]["car_id"] == "b" for i, _ in hits)


def test_bm25_ignores_target_car_name() -> None:
    store = ChunkStore(DOCS)
    # Without name stripping the name-stuffed FAQ would win; restricted to car a, "alpha one" carries no signal.
    assert store.bm25_search("Alpha One comfort", ["a"])[0][0] == "a:01:0"
    # Across all cars the name is a useful signal and is kept.
    assert store.bm25_search("Alpha", None)[0][0].startswith("a:")


# --------------------------------------------------------------------------- fusion + hybrid


def test_rrf_weights_and_ranks() -> None:
    scores = rrf([["x", "y"], ["y", "z"]], [2.0, 1.0], k=60)
    assert scores["x"] == pytest.approx(2 / 61)
    assert scores["y"] == pytest.approx(2 / 62 + 1 / 61)
    assert scores["z"] == pytest.approx(1 / 62)
    assert max(scores, key=scores.get) == "y"  # type: ignore[arg-type]


def test_cap_per_car() -> None:
    assert cap_per_car(["a1", "a2", "b1", "a3"], lambda i: i[0], 2) == ["a1", "a2", "b1"]


def _fake_vector(order: list[str]):
    calls: list[tuple[list[str], int]] = []

    async def vector_fn(qv: list[float], car_ids: list[str], k: int) -> list[tuple[str, float]]:
        calls.append((list(car_ids), k))
        hits = [i for i in order if i.split(":")[0] in car_ids]
        per_car: dict[str, int] = {}
        out = []
        for i in hits:
            car = i.split(":")[0]
            if per_car.get(car, 0) < k:
                per_car[car] = per_car.get(car, 0) + 1
                out.append((i, 1.0 - 0.01 * len(out)))
        return out

    return vector_fn, calls


async def _embed(_: str) -> list[float]:
    return [0.0]


async def test_hybrid_single_car_and_pinned_card() -> None:
    vector_fn, calls = _fake_vector(["a:00:0", "a:01:0", "a:faq:00"])
    r = HybridRetriever(ChunkStore(DOCS), embed_fn=_embed, vector_fn=vector_fn)
    out = await r.retrieve("trunk", ["a"])
    assert calls == [(["a"], 6)]  # k_per_car = 6 for one car
    assert out[0].id == "a:00:0" and out[0].vector_rank == 1 and out[0].bm25_rank == 1
    assert out[-1].id == "a:card" and out[-1].pinned


async def test_hybrid_unknown_car_ids_are_dropped() -> None:
    vector_fn, calls = _fake_vector([])
    r = HybridRetriever(ChunkStore(DOCS), embed_fn=_embed, vector_fn=vector_fn)
    assert await r.retrieve("trunk", ["tesla-model-3"]) == []
    assert calls == []


async def test_hybrid_section_filter_and_cap() -> None:
    vector_fn, _ = _fake_vector(["a:00:0", "a:01:0", "b:01:0", "b:00:0", "c:00:0"])
    r = HybridRetriever(ChunkStore(DOCS), embed_fn=_embed, vector_fn=vector_fn)
    out = await r.retrieve("trunk", ["a", "b", "c"], section_types=["trunk"], include_cards=False)
    assert {c.id for c in out} == {"a:00:0", "b:01:0"}
    out = await r.retrieve("trunk", None, max_per_car=1, include_cards=False)
    assert len({c.car_id for c in out}) == len(out)


async def test_hybrid_section_filter_falls_back_when_empty() -> None:
    vector_fn, _ = _fake_vector(["b:00:0"])
    r = HybridRetriever(ChunkStore(DOCS), embed_fn=_embed, vector_fn=vector_fn)
    out = await r.retrieve("engine", ["b"], section_types=["verdict"], include_cards=False)
    assert out and out[0].id == "b:00:0"
