"""Retrieval sanity check against the live Cosmos emulator + Foundry embeddings.

    .venv\\Scripts\\python.exe .github/skills/autochat-agent/scripts/retrieval_check.py [--k 5] [--show 3] [-q "query"]

1. Metadata filter cases (exact SQL on `cars`) -> must match expected car ids (exit 1 on failure).
2. ~14 queries: hit@k for vector-only, BM25-only and hybrid (RRF). A hit = a top-k chunk from an expected car
   whose text contains one of the expected keywords (any keyword if none given).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field

from autochat.db.cosmos import close_client
from autochat.retrieval import ChunkStore, HybridRetriever, SearchFilters, find_cars
from autochat.retrieval.hybrid import RetrievedChunk, embed_query

ALL = None
EV_IDS = ["mg-s6", "aion-ht", "long-term-report-kia-ev9-4"]


@dataclass
class Case:
    query: str
    car_ids: list[str] | None  # what the agent would pass (mentioned cars / filter result); None = all cars
    expect_cars: list[str]
    keywords: list[str] = field(default_factory=list)


CASES = [
    Case("How big is the trunk of the Citroen C3?", ["citroen-c3-2026"], ["citroen-c3-2026"], ["trunk", "מטען"]),
    Case("RS3 engine sound", ["audi-rs3-fl"], ["audi-rs3-fl"], ["sound", "exhaust", "cylinder"]),
    Case("What are the downsides of the Audi RS3?", ["audi-rs3-fl"], ["audi-rs3-fl"], ["cons", "downside", "חסרונות"]),
    Case("איך הנוחות של לינק אנד קו 01 בעיר?", ["link-and-co-01-2026"], ["link-and-co-01-2026"], ["comfort", "נוחות"]),
    Case(
        "What did the long-term report say about living with the Kia EV9?",
        ["long-term-report-kia-ev9-4"],
        ["long-term-report-kia-ev9-4"],
    ),
    Case(
        "Is the Elantra N comfortable for daily driving?",
        ["hyundai-elantra-n-manual"],
        ["hyundai-elantra-n-manual"],
        ["comfort", "daily", "everyday"],
    ),
    Case("What real-world range did MG S6 achieve?", ["mg-s6"], ["mg-s6"], ["430", "420", "440"]),
    Case("What is the DC fast-charging power of the Aion HT?", ["aion-ht"], ["aion-ht"], ["163"]),
    Case(
        "What fuel consumption did the Genesis GV80 get in the test?",
        ["genesis-gv80-2026"],
        ["genesis-gv80-2026"],
        ["8.1"],
    ),
    Case("מה הטווח בפועל של איון HT?", ["aion-ht"], ["aion-ht"], ["370"]),
    Case(
        "Compare the MG S6 and the Aion HT on real range and charging",
        ["mg-s6", "aion-ht"],
        ["mg-s6", "aion-ht"],
        ["range", "charg"],
    ),
    Case("Which EV charges fastest?", EV_IDS, EV_IDS, ["kw", "charg"]),
    Case("Which car is the most fun to drive?", ALL, ["hyundai-elantra-n-manual", "audi-rs3-fl"], ["fun", "כיף"]),
    Case(
        "Which car has the biggest trunk for a family?",
        ALL,
        ["mg-s6", "genesis-gv80-2026", "long-term-report-kia-ev9-4", "link-and-co-01-2026", "aion-ht"],
        ["trunk", "מטען"],
    ),
]

FILTER_CASES: list[tuple[str, SearchFilters, dict, list[str]]] = [
    ("EV under 175k", SearchFilters(powertrains=["ev"], max_price_ils=175_000), {}, ["mg-s6"]),
    (
        "petrol under 230k",
        SearchFilters(powertrains=["ice"], max_price_ils=230_000),
        {},
        ["citroen-c3-2026", "hyundai-elantra-n-manual"],
    ),
    ("cheapest EV (sorted)", SearchFilters(powertrains=["ev"]), {"sort_by": "price_ils", "limit": 1}, ["mg-s6"]),
    ("quickest 0-100 (sorted)", SearchFilters(), {"sort_by": "accel_0_100_s", "limit": 1}, ["audi-rs3-fl"]),
    (
        "SUV, real range >= 400",
        SearchFilters(body_types=["suv"], min_range_km=400),
        {},
        ["long-term-report-kia-ev9-4", "mg-s6"],
    ),
    ("make Kia/MG", SearchFilters(makes=["kia", "MG"]), {}, ["long-term-report-kia-ev9-4", "mg-s6"]),
    ("enthusiast tag", SearchFilters(tags=["enthusiast"]), {}, ["audi-rs3-fl", "hyundai-elantra-n-manual"]),
]


def is_hit(chunks: list[RetrievedChunk], case: Case) -> bool:
    for c in chunks:
        text = f"{c.heading} {c.heading_en} {c.text_en} {c.text_he}".lower()
        if c.car_id in case.expect_cars and (not case.keywords or any(k.lower() in text for k in case.keywords)):
            return True
    return False


def row(c: RetrievedChunk) -> str:
    v = f"v#{c.vector_rank}" if c.vector_rank else "v- "
    b = f"b#{c.bm25_rank}" if c.bm25_rank else "b- "
    pin = " (card pinned)" if c.pinned else ""
    return f"      {c.rrf_score:.4f} {v:>5} {b:>5}  {c.id:36} {c.section_type:16} {c.heading_en[:45]}{pin}"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="hit@k")
    ap.add_argument("--show", type=int, default=3, help="print top-N hybrid results per query (0 = none)")
    ap.add_argument("-q", "--query", help="run one ad-hoc query over all cars and print the top results")
    args = ap.parse_args()
    failed = 0
    try:
        store = await ChunkStore.from_cosmos()
        hybrid = HybridRetriever(store)
        if args.query:
            for c in await hybrid.retrieve(args.query, top_n=args.k * 2, include_cards=False):
                print(row(c))
            return 0

        print("== metadata filters (exact SQL on cars)")
        for name, filters, kw, expected in FILTER_CASES:
            got = [c["car_id"] for c in await find_cars(filters, **kw)]
            ok = sorted(got) == sorted(expected)
            failed += not ok
            print(f"  {'OK  ' if ok else 'FAIL'} {name:26} -> {got}")

        modes = {
            "vector": HybridRetriever(store, vector_weight=1.0, bm25_weight=0.0),
            "bm25": HybridRetriever(store, vector_weight=0.0, bm25_weight=1.0),
            "hybrid": hybrid,
        }
        hits = dict.fromkeys(modes, 0)
        print(f"\n== retrieval hit@{args.k}   (vector | bm25 | hybrid)")
        for case in CASES:
            qv = await embed_query(case.query)
            marks = []
            top: list[RetrievedChunk] = []
            for name, retriever in modes.items():
                if name == "bm25":
                    store_hits = store.bm25_search(case.query, case.car_ids or store.car_ids, k=args.k)
                    chunks = [await _hydrate(retriever, i) for i, _ in store_hits]
                else:
                    chunks = await retriever.retrieve(
                        case.query, case.car_ids, top_n=args.k, include_cards=False, query_vector=qv
                    )
                ok = is_hit(chunks[: args.k], case)
                hits[name] += ok
                marks.append("Y" if ok else ".")
                if name == "hybrid":
                    top = chunks
            print(f"  {' | '.join(f'{m:^6}' for m in marks)}  {case.query}")
            for c in top[: args.show]:
                print(row(c))
        n = len(CASES)
        print("\n  " + "   ".join(f"{name}={hits[name]}/{n}" for name in modes))
    finally:
        await close_client()
    return 1 if failed else 0


async def _hydrate(retriever: HybridRetriever, chunk_id: str) -> RetrievedChunk:
    return retriever._hit(chunk_id, None, None, 0.0)  # noqa: SLF001 - display only


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
