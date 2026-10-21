"""Shared HTML helpers for source adapters."""

from __future__ import annotations

import re

from selectolax.lexbor import LexborHTMLParser


def clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()


def meta(tree: LexborHTMLParser, key: str) -> str | None:
    node = tree.css_first(f'meta[property="{key}"]') or tree.css_first(f'meta[name="{key}"]')
    return node.attributes.get("content") if node else None


_ISO_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_DMY_RE = re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b")


def find_date(tree: LexborHTMLParser) -> str | None:
    """Return an ISO date: article:published_time meta, then the first <time>, then None."""
    candidates = [meta(tree, "article:published_time")]
    first_time = tree.css_first("time")
    if first_time is not None:
        candidates += [first_time.attributes.get("datetime"), clean(first_time.text())]
    for cand in candidates:
        if not cand:
            continue
        if m := _ISO_RE.search(cand):
            return "-".join(m.groups())
        if m := _DMY_RE.search(cand):
            d, mo, y = m.groups()
            return f"{y}-{int(mo):02d}-{int(d):02d}"
    return None
