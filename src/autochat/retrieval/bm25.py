"""In-memory chunk store + BM25 keyword index (the emulator has no full-text search).

Loaded once per process from Cosmos (~110 docs, no embeddings). It also hydrates vector hits, so the vector
query only needs to return ids and scores.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any

from rank_bm25 import BM25Okapi

log = logging.getLogger(__name__)

STORE_FIELDS = (
    "id",
    "car_id",
    "chunk_type",
    "section_type",
    "aspects",
    "order",
    "heading",
    "heading_en",
    "header",
    "text_he",
    "text_en",
    "url",
    "make",
    "model",
    "model_year",
    "body_type",
    "powertrain",
    "price_ils",
)
LOAD_SQL = "SELECT " + ", ".join(f'c["{f}"]' for f in STORE_FIELDS) + " FROM c"  # quoted: `order` is reserved
CARS_SQL = "SELECT c.car_id, c.make_he, c.model_he, c.aliases FROM c"

_GERESH = re.compile(r"(?<=[\u05d0-\u05ea])[\"'\u05f3\u05f4](?=[\u05d0-\u05ea])")  # ק"מ -> קמ, כ"ס -> כס
_TOKEN = re.compile(r"\w+", re.UNICODE)
_HEBREW = re.compile(r"^[\u05d0-\u05ea]+$")
_HE_PREFIX = "והבלמשכ"
EN_STOPWORDS = frozenset(
    "a an and are as at be but by do does did for from has have how i in is it its me my of on or our so than that "
    "the their them then there these they this to was we were what when which who why will with you your".split()
)
HE_STOPWORDS = frozenset("של את על עם זה זו הוא היא גם לא כי אם או מה איך יש אין כל רק עוד כמו אבל".split())


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; strip ONE Hebrew prefix letter from Hebrew tokens longer than 3 chars."""
    text = _GERESH.sub("", text.lower())
    out: list[str] = []
    for tok in _TOKEN.findall(text):
        if tok in EN_STOPWORDS or tok in HE_STOPWORDS:
            continue
        if len(tok) > 3 and tok[0] in _HE_PREFIX and _HEBREW.match(tok):
            tok = tok[1:]
        out.append(tok)
    return out


def doc_text(doc: dict[str, Any]) -> str:
    return "\n".join(str(doc.get(f) or "") for f in ("header", "heading_en", "text_en", "text_he"))


class ChunkStore:
    """All chunk docs by id, plus a BM25 index over header + English + Hebrew text.

    `car_names` (car_id -> names/aliases) lets BM25 ignore a car's own name tokens when the search is already
    restricted to that car: inside one car's chunks the name carries no signal, and FAQ answers that repeat the
    name would otherwise outrank the relevant section.
    """

    def __init__(self, docs: Iterable[dict[str, Any]], car_names: dict[str, list[str]] | None = None) -> None:
        self.docs: dict[str, dict[str, Any]] = {d["id"]: d for d in docs}
        self._ids: list[str] = list(self.docs)
        corpus = [tokenize(doc_text(self.docs[i])) for i in self._ids]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self.car_ids: list[str] = sorted({d["car_id"] for d in self.docs.values()})
        names: dict[str, set[str]] = {c: set() for c in self.car_ids}
        for d in self.docs.values():
            names[d["car_id"]].update(tokenize(f"{d.get('make') or ''} {d.get('model') or ''}"))
        for car_id, aliases in (car_names or {}).items():
            if car_id in names:
                for alias in aliases:
                    names[car_id].update(tokenize(alias))
        self.name_tokens: dict[str, frozenset[str]] = {c: frozenset(t) for c, t in names.items()}

    def __len__(self) -> int:
        return len(self.docs)

    def get(self, chunk_id: str) -> dict[str, Any] | None:
        return self.docs.get(chunk_id)

    def card(self, car_id: str) -> dict[str, Any] | None:
        return self.docs.get(f"{car_id}:card")

    def bm25_search(self, query: str, car_ids: Iterable[str] | None = None, k: int = 20) -> list[tuple[str, float]]:
        """Top-k (chunk_id, bm25 score) within car_ids; zero-score docs are not returned."""
        allowed = set(car_ids) if car_ids is not None else None
        tokens = tokenize(query)
        if allowed is not None and allowed != set(self.car_ids):
            names = frozenset().union(*(self.name_tokens.get(c, frozenset()) for c in allowed))
            tokens = [t for t in tokens if t not in names]
        if not tokens or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokens)
        hits = [
            (cid, float(s))
            for cid, s in zip(self._ids, scores, strict=True)
            if s > 0 and (allowed is None or self.docs[cid]["car_id"] in allowed)
        ]
        hits.sort(key=lambda h: -h[1])
        return hits[:k]

    @classmethod
    async def from_cosmos(cls) -> ChunkStore:
        from autochat.db.cosmos import cars_container, chunks_container

        docs = [d async for d in chunks_container().query_items(LOAD_SQL)]
        car_names: dict[str, list[str]] = {}
        async for car in cars_container().query_items(CARS_SQL):
            names = [car.get("make_he"), car.get("model_he"), *(car.get("aliases") or [])]
            car_names[car["car_id"]] = [n for n in names if n]
        store = cls(docs, car_names)
        log.info("chunk store: %d chunks, %d cars", len(store), len(store.car_ids))
        return store
