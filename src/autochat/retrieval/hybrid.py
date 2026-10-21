"""Hybrid retrieval: Cosmos vector search (per-car partition fan-out) + in-app BM25, fused with weighted RRF.

Swap point: anything implementing `Retriever` can replace `HybridRetriever`. On Azure Cosmos DB (not the
emulator), it can be one query with `ORDER BY RANK RRF(VectorDistance(...), FullTextScore(...))`.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Awaitable, Callable, Iterator
from typing import Any, Protocol

from pydantic import BaseModel
from tenacity import AsyncRetrying

from autochat.retrieval.bm25 import ChunkStore

log = logging.getLogger(__name__)

RRF_K = 60
VECTOR_WEIGHT = 2.0
BM25_WEIGHT = 1.0

EmbedFn = Callable[[str], Awaitable[list[float]]]
VectorFn = Callable[[list[float], list[str], int], Awaitable[list[tuple[str, float]]]]


class RetrievedChunk(BaseModel):
    id: str
    car_id: str
    chunk_type: str
    section_type: str
    heading: str
    heading_en: str = ""
    header: str = ""
    text_he: str
    text_en: str = ""
    url: str
    make: str = ""
    model: str = ""
    vector_score: float | None = None
    vector_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    pinned: bool = False  # added because the caller asked for the car's card, not because it ranked


class Retriever(Protocol):
    async def retrieve(
        self,
        query: str,
        car_ids: list[str] | None = None,
        *,
        section_types: list[str] | None = None,
        top_n: int | None = None,
        max_per_car: int | None = None,
        include_cards: bool | None = None,
        query_vector: list[float] | None = None,
    ) -> list[RetrievedChunk]: ...


def rrf(rank_lists: list[list[str]], weights: list[float], k: int = RRF_K) -> dict[str, float]:
    """Weighted reciprocal rank fusion: score(d) = sum_i w_i / (k + rank_i(d)), rank starting at 1."""
    scores: dict[str, float] = {}
    for ranks, w in zip(rank_lists, weights, strict=True):
        for r, doc_id in enumerate(ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + r + 1)
    return scores


def default_k_per_car(n_cars: int) -> int:
    return 6 if n_cars == 1 else 4 if n_cars <= 3 else 3


def default_top_n(n_cars: int) -> int:
    return 8 if n_cars == 1 else min(12, 4 * n_cars)


def default_max_per_car(n_cars: int, top_n: int) -> int:
    return top_n if n_cars == 1 else 4 if n_cars <= 3 else 3


def cap_per_car(ranked: list[str], car_of: Callable[[str], str], max_per_car: int) -> list[str]:
    counts: dict[str, int] = {}
    out = []
    for doc_id in ranked:
        car = car_of(doc_id)
        if counts.get(car, 0) < max_per_car:
            counts[car] = counts.get(car, 0) + 1
            out.append(doc_id)
    return out


async def embed_query(text: str) -> list[float]:
    from autochat.llm import RETRY_KW, embeddings

    async for attempt in AsyncRetrying(**RETRY_KW):
        with attempt:
            return await embeddings().aembed_query(text)
    raise RuntimeError("unreachable")


@contextlib.contextmanager
def _retriever_span(inputs: dict[str, Any]) -> Iterator[Any]:
    """RETRIEVER span inside an active trace (run_turn / eval); no stray root traces from scripts or tests."""
    try:
        import mlflow
        from mlflow.entities import SpanType
    except ImportError:  # pragma: no cover
        yield None
        return
    if mlflow.get_current_active_span() is None:
        yield None
        return
    with mlflow.start_span(name="hybrid_retrieve", span_type=SpanType.RETRIEVER) as span:
        span.set_inputs(inputs)
        yield span


class HybridRetriever:
    def __init__(
        self,
        store: ChunkStore,
        embed_fn: EmbedFn | None = None,
        vector_fn: VectorFn | None = None,
        vector_weight: float = VECTOR_WEIGHT,
        bm25_weight: float = BM25_WEIGHT,
    ) -> None:
        if vector_fn is None:
            from autochat.retrieval.vector import vector_search

            vector_fn = vector_search
        self.store = store
        self.embed_fn = embed_fn or embed_query
        self.vector_fn = vector_fn
        self.weights = [vector_weight, bm25_weight]

    async def retrieve(
        self,
        query: str,
        car_ids: list[str] | None = None,
        *,
        section_types: list[str] | None = None,
        top_n: int | None = None,
        max_per_car: int | None = None,
        include_cards: bool | None = None,
        query_vector: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        """Rank chunks of `car_ids` (default: all cars) for `query`.

        section_types: keep only these section types (car cards are still added if include_cards).
        include_cards: append each target car's `car_card` chunk if it did not rank (default: when <= 4 cars).
        """
        known = set(self.store.car_ids)
        targets = [c for c in dict.fromkeys(car_ids or self.store.car_ids) if c in known]
        inputs = {"query": query, "car_ids": targets, "section_types": section_types or []}
        with _retriever_span(inputs) as span:
            result = await self._retrieve(
                query, targets, section_types, top_n, max_per_car, include_cards, query_vector
            )
            if span is not None:
                from autochat.observability import retriever_documents

                span.set_outputs(retriever_documents([c.model_dump() for c in result]))
        return result

    async def _retrieve(
        self,
        query: str,
        targets: list[str],
        section_types: list[str] | None,
        top_n: int | None,
        max_per_car: int | None,
        include_cards: bool | None,
        query_vector: list[float] | None,
    ) -> list[RetrievedChunk]:
        if not targets:
            return []
        n = len(targets)
        k_per_car = default_k_per_car(n) * (2 if section_types else 1)
        top_n = top_n or default_top_n(n)
        max_per_car = max_per_car or default_max_per_car(n, top_n)
        include_cards = n <= 4 if include_cards is None else include_cards

        qv = query_vector or await self.embed_fn(query)
        vec_hits = [(i, s) for i, s in await self.vector_fn(qv, targets, k_per_car) if i in self.store.docs]
        bm_hits = self.store.bm25_search(query, targets, k=k_per_car * n)
        vec_rank = {doc_id: (r + 1, s) for r, (doc_id, s) in enumerate(vec_hits)}
        bm_rank = {doc_id: (r + 1, s) for r, (doc_id, s) in enumerate(bm_hits)}

        fused = rrf([[i for i, _ in vec_hits], [i for i, _ in bm_hits]], self.weights)
        ranked = sorted(fused, key=lambda i: -fused[i])
        if section_types:
            wanted = set(section_types)
            filtered = [i for i in ranked if self.store.docs[i]["section_type"] in wanted]
            if filtered:
                ranked = filtered
            else:
                log.info("no chunks with section_type in %s; keeping unfiltered ranking", sorted(wanted))
        ranked = cap_per_car(ranked, lambda i: self.store.docs[i]["car_id"], max_per_car)[:top_n]

        out = [self._hit(i, vec_rank.get(i), bm_rank.get(i), fused[i]) for i in ranked]
        if include_cards:
            present = set(ranked)
            for car_id in targets:
                card = self.store.card(car_id)
                if card and card["id"] not in present:
                    out.append(
                        self._hit(
                            card["id"],
                            vec_rank.get(card["id"]),
                            bm_rank.get(card["id"]),
                            fused.get(card["id"], 0.0),
                            pinned=True,
                        )
                    )
        return out

    def _hit(
        self,
        doc_id: str,
        vec: tuple[int, float] | None,
        bm: tuple[int, float] | None,
        score: float,
        pinned: bool = False,
    ) -> RetrievedChunk:
        doc = self.store.docs[doc_id]
        fields = {k: doc.get(k) for k in RetrievedChunk.model_fields if k in doc and doc.get(k) is not None}
        return RetrievedChunk(
            **fields,
            vector_rank=vec[0] if vec else None,
            vector_score=round(vec[1], 4) if vec else None,
            bm25_rank=bm[0] if bm else None,
            bm25_score=round(bm[1], 3) if bm else None,
            rrf_score=round(score, 5),
            pinned=pinned,
        )


async def build_retriever() -> HybridRetriever:
    """Load the chunk store from Cosmos and return a ready retriever (call once per process)."""
    return HybridRetriever(await ChunkStore.from_cosmos())
