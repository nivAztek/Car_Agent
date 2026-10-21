"""Generic fallback adapter (any article site) built on trafilatura main-content extraction.

It removes boilerplate well but loses div-based spec tables, so the site adapter is preferred when one exists.
"""

from __future__ import annotations

import re

import trafilatura
from selectolax.lexbor import LexborHTMLParser

from autochat.ingestion.schemas import ParsedArticle, Section
from autochat.ingestion.sources._html import clean, find_date, meta

_HEADING_RE = re.compile(r"^#{1,4}\s+(.*)$")
_BOLD_LINE_RE = re.compile(r"^\*\*(.{1,90})\*\*$")
INTRO_HEADING = "פתיח"


def markdown_sections(markdown: str) -> list[Section]:
    sections: list[Section] = []
    heading, kind, paras = INTRO_HEADING, "intro", []

    def push() -> None:
        if paras:
            sections.append(Section(heading=heading, kind=kind, paragraphs=list(paras)))

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if m := _HEADING_RE.match(line):
            push()
            heading, kind, paras = clean(m.group(1)), "md", []
        elif m := _BOLD_LINE_RE.match(line):
            push()
            heading, kind, paras = clean(m.group(1)), "bold", []
        else:
            text = clean(re.sub(r"[*_`]+", "", line.lstrip("-•> ")))
            if text:
                paras.append(text)
    push()
    return sections


class GenericAdapter:
    name = "generic"

    def matches(self, url: str) -> bool:  # noqa: ARG002 - matches everything
        return True

    def parse(self, html: str, url: str, slug: str) -> ParsedArticle:
        markdown = (
            trafilatura.extract(
                html, url=url, output_format="markdown", include_tables=True, include_comments=False, favor_recall=True
            )
            or ""
        )
        tree = LexborHTMLParser(html)
        h1 = tree.css_first("h1")
        return ParsedArticle(
            slug=slug,
            url=url,
            adapter=self.name,
            title=clean(meta(tree, "og:title")) or (clean(h1.text()) if h1 else slug),
            subtitle=clean(meta(tree, "og:description")) or None,
            review_date=find_date(tree),
            sections=markdown_sections(markdown),
        )
