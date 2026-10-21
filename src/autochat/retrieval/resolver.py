"""Resolve the car names a user wrote ("the Kia", "האיוניק", "Lynk 01", "Audi Q5") to catalog car_ids in code.

The planner only copies names verbatim (`car_mentions`); it never sees the catalog. Resolution order:
1. exact `name_keys` match on any token span (Hebrew prefixes stripped), longest span wins. A make named in the
   mention must agree with the car's make, and leftover words must be generic / a year / the car's trim
   ("Citroen C3 Aircross" is NOT the C3, "Audi S6" is NOT the MG S6);
2. a make is named (exact or fuzzy, e.g. "Genesys", "Link & Co") -> make only: its single car, or ambiguous;
   otherwise fuzzy model match among that make's cars only, with identical digits ("EV6" is never the EV9);
3. otherwise unknown (with same-make suggestions when a reviewed make was named).
Cost: one `by_name_keys` query + one `by_makes` query (only if a make was named) + the cached scope.
"""

from __future__ import annotations

import difflib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

from autochat.names import digits, display_name, is_generic, model_keys, ngram_keys, normalize, spans
from autochat.retrieval.catalog import CarCatalog

Status = Literal["resolved", "ambiguous", "unknown"]
MAX_TOKENS = 8
MAKE_SPAN_MAX = 3
FUZZY_CUTOFF = 0.8
FUZZY_MIN_LEN = 4
MAX_AMBIGUOUS = 5
MAX_SUGGESTIONS = 3
FOCUS_REF = re.compile(r"#\s*(\d+)")


@dataclass
class MentionResolution:
    mention: str
    status: Status
    car_ids: list[str] = field(default_factory=list)  # resolved: one id; ambiguous: the candidates
    names: list[str] = field(default_factory=list)  # display names of car_ids
    suggestions: list[str] = field(default_factory=list)  # unknown: reviewed cars of the same make


def ref_key(ref: str) -> str:
    """Lookup key for a car reference: '#2' for focus refs, else the normalized name."""
    m = FOCUS_REF.fullmatch(ref.strip())
    return f"#{m.group(1)}" if m else normalize(ref)


def is_focus_ref(ref: str) -> bool:
    return bool(FOCUS_REF.fullmatch(ref.strip()))


@dataclass
class _Parsed:
    mention: str
    tokens: list[str]
    spans: list[tuple[int, int, set[str]]]

    @classmethod
    def of(cls, mention: str) -> _Parsed:
        tokens = normalize(mention).split()[:MAX_TOKENS]
        return cls(mention, tokens, [(i, j, ngram_keys(tokens, i, j)) for i, j in spans(len(tokens))])


def _detect_makes(p: _Parsed, vocab: dict[str, str]) -> tuple[set[str], set[int]]:
    """Makes named in the mention (exact first, then fuzzy on short spans) and the token positions they cover."""
    makes: set[str] = set()
    covered: set[int] = set()
    for i, j, keys in p.spans:
        if j - i > MAKE_SPAN_MAX or covered & set(range(i, j)):
            continue
        hit = next((vocab[k] for k in sorted(keys) if k in vocab), None)
        if hit:
            makes.add(hit)
            covered |= set(range(i, j))
    fuzzy_vocab = [k for k in vocab if len(k) >= FUZZY_MIN_LEN]
    for i, j, keys in p.spans:
        if j - i > 2 or covered & set(range(i, j)):
            continue
        for k in sorted(keys):
            if len(k) < FUZZY_MIN_LEN or digits(k):
                continue
            close = difflib.get_close_matches(k, fuzzy_vocab, n=1, cutoff=FUZZY_CUTOFF)
            if close:
                makes.add(vocab[close[0]])
                covered |= set(range(i, j))
                break
    return makes, covered


def _extra_tokens_ok(tokens: list[str], car: dict[str, Any]) -> bool:
    own = set(normalize(car.get("trim")).split())
    for alias in car.get("aliases") or []:
        own |= set(normalize(alias).split())
    return all(is_generic(t) or t in own for t in tokens)


def _exact(
    p: _Parsed, by_key: dict[str, list[dict[str, Any]]], makes: set[str], make_cov: set[int]
) -> list[dict[str, Any]]:
    best: dict[str, tuple[int, set[int], dict[str, Any]]] = {}
    for i, j, keys in p.spans:
        for k in keys:
            for car in by_key.get(k, []):
                if makes and car.get("make") not in makes:
                    continue
                prev = best.get(car["car_id"])
                if prev is None or j - i > prev[0]:
                    best[car["car_id"]] = (j - i, set(range(i, j)), car)
    valid: list[tuple[int, dict[str, Any]]] = []
    for n, cov, car in best.values():
        rest = [p.tokens[t] for t in range(len(p.tokens)) if t not in cov | make_cov]
        if _extra_tokens_ok(rest, car):
            valid.append((n, car))
    if not valid:
        return []
    top = max(n for n, _ in valid)
    return [car for n, car in valid if n == top]


def _fuzzy_in_make(p: _Parsed, make_cars: list[dict[str, Any]], make_cov: set[int]) -> list[dict[str, Any]] | None:
    """None = the mention is the make alone; [] = no model matched."""
    rest = [t for i, t in enumerate(p.tokens) if i not in make_cov and not is_generic(t)]
    if not rest:
        return None
    rest_keys = ngram_keys(rest, 0, len(rest))
    scored: list[tuple[float, dict[str, Any]]] = []
    for car in make_cars:
        score = 0.0
        for mk in model_keys(car):
            for rk in rest_keys:
                if rk == mk:
                    score = 1.0
                elif len(rk) >= FUZZY_MIN_LEN and digits(rk) == digits(mk):
                    score = max(score, difflib.SequenceMatcher(None, rk, mk).ratio())
        if score >= FUZZY_CUTOFF:
            scored.append((score, car))
    if not scored:
        return []
    top = max(s for s, _ in scored)
    return [car for s, car in scored if s == top]


def _result(mention: str, cars: list[dict[str, Any]]) -> MentionResolution:
    cars = sorted(cars, key=lambda c: c["car_id"])[:MAX_AMBIGUOUS]
    status: Status = "resolved" if len(cars) == 1 else "ambiguous"
    return MentionResolution(mention, status, [c["car_id"] for c in cars], [display_name(c) for c in cars])


async def resolve_mentions(mentions: list[str], catalog: CarCatalog) -> dict[str, MentionResolution]:
    """Resolve names (not '#n' focus refs) -> {ref_key(mention): MentionResolution}."""
    unique = {ref_key(m): m for m in mentions if m and m.strip() and not is_focus_ref(m)}
    parsed = {k: _Parsed.of(m) for k, m in unique.items() if normalize(m)}
    if not parsed:
        return {}
    all_keys = sorted({k for p in parsed.values() for _, _, keys in p.spans for k in keys})
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for car in await catalog.by_name_keys(all_keys):
        for k in car.get("name_keys") or []:
            by_key[k].append(car)

    vocab = (await catalog.scope()).make_vocab()
    detected = {k: _detect_makes(p, vocab) for k, p in parsed.items()}
    named_makes = sorted({m for makes, _ in detected.values() for m in makes})
    make_cars = await catalog.by_makes(named_makes) if named_makes else []

    out: dict[str, MentionResolution] = {}
    for key, p in parsed.items():
        makes, make_cov = detected[key]
        exact = _exact(p, by_key, makes, make_cov)
        if exact:
            out[key] = _result(p.mention, exact)
            continue
        of_make = [c for c in make_cars if c.get("make") in makes]
        fuzzy = _fuzzy_in_make(p, of_make, make_cov) if makes else []
        if fuzzy is None:  # the make alone ("the Kia")
            fuzzy = of_make
        if fuzzy:
            out[key] = _result(p.mention, fuzzy)
        else:
            out[key] = MentionResolution(
                p.mention, "unknown", suggestions=[display_name(c) for c in of_make[:MAX_SUGGESTIONS]]
            )
    return out
