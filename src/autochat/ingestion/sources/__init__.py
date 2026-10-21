"""Source adapters: turn raw HTML into a `ParsedArticle`.

Add a new site by writing an adapter with `name`, `matches(url)` and `parse(html, url, slug)` and registering it in
`ADAPTERS`. Any URL that no adapter matches, or whose site parse fails the quality gate, goes to the generic
trafilatura adapter.
"""

from __future__ import annotations

import logging
from typing import Protocol

from autochat.ingestion.schemas import ParsedArticle
from autochat.ingestion.sources.auto_co_il import AutoCoIlAdapter
from autochat.ingestion.sources.generic import GenericAdapter

log = logging.getLogger(__name__)

MIN_BODY_CHARS = 1500


class SourceAdapter(Protocol):
    name: str

    def matches(self, url: str) -> bool: ...

    def parse(self, html: str, url: str, slug: str) -> ParsedArticle: ...


ADAPTERS: list[SourceAdapter] = [AutoCoIlAdapter()]
GENERIC = GenericAdapter()


def passes_quality_gate(article: ParsedArticle) -> bool:
    return bool(article.sections) and article.body_chars >= MIN_BODY_CHARS


def parse_html(html: str, url: str, slug: str) -> ParsedArticle:
    """Use the matching site adapter. Fall back to generic if its output is too thin, keeping any site-only
    structured data (spec table, pros/cons, FAQ), which trafilatura cannot recover."""
    for adapter in ADAPTERS:
        if not adapter.matches(url):
            continue
        article = adapter.parse(html, url, slug)
        if passes_quality_gate(article):
            return article
        log.warning("%s: %s adapter output thin (%d chars) -> generic fallback", slug, adapter.name, article.body_chars)
        generic = GENERIC.parse(html, url, slug)
        return generic.model_copy(
            update={
                "adapter": f"{adapter.name}+generic",
                "specs_raw": article.specs_raw or generic.specs_raw,
                "pros": article.pros or generic.pros,
                "cons": article.cons or generic.cons,
                "faq": article.faq or generic.faq,
                "title": article.title or generic.title,
                "review_date": article.review_date or generic.review_date,
            }
        )
    return GENERIC.parse(html, url, slug)


__all__ = ["ADAPTERS", "GENERIC", "SourceAdapter", "parse_html", "passes_quality_gate"]
