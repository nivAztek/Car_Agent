"""Car-name normalization shared by ingestion (`CarRecord.name_keys`) and the name resolver (user mentions).

A key is the normalized name with spaces removed: "Lynk & Co 01" -> "lynkco01", "ג'נסיס GV80" -> "גנסיסgv80".
Lives outside `autochat.retrieval` / `autochat.ingestion` so both can import it without a cycle.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Any

QUOTES = re.compile(r"[\"'`´‘’׳״]")
NON_WORD = re.compile(r"[\W_]+")
HEBREW = re.compile(r"[\u0590-\u05FF]")
DIGITS = re.compile(r"\d+")
YEAR = re.compile(r"(19|20)\d\d")
STOP_TOKENS = frozenset({"and", "אנד"})
# Hebrew one/two-letter prefixes glued to a word: "הקיה", "לאודי", "שהג'נסיס".
HEBREW_PREFIXES = ("וה", "שה", "מה", "לה", "בה", "כש", "ה", "ו", "ב", "ל", "מ", "ש", "כ")
# Words that may surround a car name without changing which car it is ("the new MG S6 manual").
GENERIC_TOKENS = frozenset(
    {
        "the", "a", "new", "my", "this", "that", "car", "model", "review", "test", "drive", "road", "version",
        "ev", "electric", "hybrid", "manual", "automatic", "auto", "suv", "sedan", "hatchback", "wagon",
        "ה", "של", "את", "רכב", "מכונית", "הרכב", "המכונית", "חדש", "חדשה", "החדש", "החדשה", "דגם", "הדגם",
        "מבחן", "חשמלי", "חשמלית", "היברידי", "ידני", "ידנית", "אוטומטי", "אוטומטית",
        # trim words: a trim of a reviewed model is still that model ("Kia EV9 GT-Line")
        "gt", "line", "sport", "premium", "luxury", "plus", "pro", "limited", "base", "comfort", "design",
        "edition", "launch", "awd", "fwd", "rwd", "4x4", "inspire", "prestige", "executive", "ליין", "פרימיום",
    }
)  # fmt: skip


def normalize(text: str | None) -> str:
    """Lowercase, strip accents/niqqud, drop quotes/geresh, '&' and 'and', punctuation -> space, collapse spaces."""
    t = unicodedata.normalize("NFKD", text or "")
    t = unicodedata.normalize("NFKC", "".join(ch for ch in t if not unicodedata.combining(ch))).lower()
    t = QUOTES.sub("", t).replace("&", " ")
    t = NON_WORD.sub(" ", t)
    return " ".join(w for w in t.split() if w not in STOP_TOKENS)


def compact(text: str | None) -> str:
    return normalize(text).replace(" ", "")


def token_variants(token: str) -> list[str]:
    """The token plus Hebrew-prefix-stripped forms ("הקיה" -> ["הקיה", "קיה"])."""
    out = [token]
    if HEBREW.match(token):
        out += [token[len(p) :] for p in HEBREW_PREFIXES if token.startswith(p) and len(token) - len(p) >= 2]
    return list(dict.fromkeys(out))


def is_generic(token: str) -> bool:
    return any(v in GENERIC_TOKENS for v in token_variants(token)) or bool(YEAR.fullmatch(token))


def digits(key: str) -> list[str]:
    return DIGITS.findall(key)


def _distinctive(key: str) -> bool:
    """Model-only keys must contain a letter: '01' alone is not a car name, 'c3' / 's6' / 'ev9' are."""
    return len(key) >= 2 and any(ch.isalpha() for ch in key)


def make_keys(car: dict[str, Any]) -> list[str]:
    return [k for k in dict.fromkeys(compact(car.get(f)) for f in ("make", "make_he")) if k]


def model_keys(car: dict[str, Any]) -> list[str]:
    return [k for k in dict.fromkeys(compact(car.get(f)) for f in ("model", "model_he")) if k]


def car_name_keys(car: dict[str, Any]) -> list[str]:
    """Exact-match keys for a car: make+model in every en/he combination, distinctive model-only keys and
    aliases. Never the make alone (it would map 'Audi Q5' to any Audi)."""
    makes, models = make_keys(car), model_keys(car)
    keys: dict[str, None] = {}
    for mk in makes:
        for md in models:
            keys.setdefault(mk + md, None)
    for md in models:
        if _distinctive(md) and md not in makes:
            keys.setdefault(md, None)
    for alias in car.get("aliases") or []:
        k = compact(alias)
        if len(k) >= 4 and k not in makes and _distinctive(k):
            keys.setdefault(k, None)
    return sorted(keys)


def ensure_name_keys(car: dict[str, Any]) -> list[str]:
    return list(car.get("name_keys") or car_name_keys(car))


def display_name(car: dict[str, Any]) -> str:
    return f"{car.get('make', '')} {car.get('model', '')}".strip()


def ngram_keys(tokens: list[str], i: int, j: int) -> set[str]:
    """Compact keys for tokens[i:j] over every Hebrew-prefix variant combination."""
    keys = {""}
    for tok in tokens[i:j]:
        keys = {k + v for k in keys for v in token_variants(tok)}
    return keys


def spans(n_tokens: int, max_n: int = 4) -> Iterable[tuple[int, int]]:
    """Token spans, longest first."""
    for n in range(min(n_tokens, max_n), 0, -1):
        for i in range(n_tokens - n + 1):
            yield i, i + n
