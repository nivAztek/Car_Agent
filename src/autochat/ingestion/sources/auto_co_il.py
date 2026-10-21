"""auto.co.il site adapter (selectors verified on all 8 articles; see the autochat-ingestion skill)."""

from __future__ import annotations

from collections.abc import Iterator
from urllib.parse import urlparse

from selectolax.lexbor import LexborHTMLParser, LexborNode

from autochat.ingestion.schemas import FaqItem, ParsedArticle, Section
from autochat.ingestion.sources._html import clean, find_date, meta

BODY_SELECTOR = "div.sections__items .text-wrapper"
BLOCK_TAGS = ("p", "ul", "ol", "blockquote", "table")
CONTAINER_TAGS = ("div", "section", "figure", "span")
SKIP_CONTAINERS = ".pros-cons-table, .vehicle-table"
BOILERPLATE_H2 = ("הצטרפו לוואטסאפ", "כתבות אחרונות", "כתבות באותם נושאים", "מותגים מתחרים", "המומחה האובייקטיבי")
INTRO_HEADING = "פתיח"


def _walk_blocks(node: LexborNode) -> Iterator[LexborNode]:
    for child in node.iter():
        if child.tag in CONTAINER_TAGS and not child.css_matches(SKIP_CONTAINERS):
            yield from _walk_blocks(child)
        else:
            yield child


def _is_bold_subheading(node: LexborNode) -> bool:
    """Pages without <h2> (long-term reports, some tests) use a short all-bold <p> as a subheading."""
    if node.tag != "p":
        return False
    strong = node.css_first("strong, b")
    if strong is None:
        return False
    full, bold = clean(node.text()), clean(strong.text())
    return 0 < len(full) <= 90 and len(bold) >= 0.8 * len(full)


class AutoCoIlAdapter:
    name = "auto_co_il"

    def matches(self, url: str) -> bool:
        host = urlparse(url).hostname or ""
        return host == "auto.co.il" or host.endswith(".auto.co.il")

    # ------------------------------------------------------------------ parts
    @staticmethod
    def spec_table(tree: LexborHTMLParser) -> dict[str, str]:
        specs: dict[str, str] = {}
        for item in tree.css(".vehicle-table__item, .table-item"):
            title = item.css_first(".table-item__title")
            value = item.css_first(".table-item__value")
            if title and value:
                key = clean(title.text())
                if key and key not in specs:
                    specs[key] = clean(value.text())
        return specs

    @staticmethod
    def pros_cons(tree: LexborHTMLParser) -> tuple[list[str], list[str]]:
        pros: list[str] = []
        cons: list[str] = []
        for col in tree.css(".pros-cons-table__col"):
            title_node = col.css_first(".pros-cons-table__title")
            title = clean(title_node.text()) if title_node else ""
            items = [clean(li.text()) for li in col.css("li")]
            items = [i for i in items if i]
            if "יתרונות" in title:
                pros = items
            elif "חסרונות" in title:
                cons = items
        return pros, cons

    @staticmethod
    def sections(tree: LexborHTMLParser) -> list[Section]:
        result: list[Section] = []
        heading, kind, paras = INTRO_HEADING, "intro", []

        def push() -> None:
            if paras:
                result.append(Section(heading=heading, kind=kind, paragraphs=list(paras)))

        for wrapper in tree.css(BODY_SELECTOR):
            for node in _walk_blocks(wrapper):
                if node.tag in ("h2", "h3"):
                    text = clean(node.text())
                    if not text or any(b in text for b in BOILERPLATE_H2):
                        continue
                    push()
                    heading, kind, paras = text, node.tag, []
                elif _is_bold_subheading(node):
                    push()
                    heading, kind, paras = clean(node.text()), "bold", []
                elif node.tag in BLOCK_TAGS:
                    text = clean(node.text())
                    if text:
                        paras.append(text)
        push()
        return result

    @staticmethod
    def faq(tree: LexborHTMLParser) -> list[FaqItem]:
        items: list[FaqItem] = []
        for item in tree.css("div.faq__container details.spollers__item"):
            q = item.css_first("summary")
            a = item.css_first(".spollers__body")
            if q and a and clean(q.text()) and clean(a.text()):
                items.append(FaqItem(q=clean(q.text()), a=clean(a.text())))
        return items

    # ------------------------------------------------------------------ main
    def parse(self, html: str, url: str, slug: str) -> ParsedArticle:
        tree = LexborHTMLParser(html)
        h1 = tree.css_first("h1")
        pros, cons = self.pros_cons(tree)
        return ParsedArticle(
            slug=slug,
            url=url,
            adapter=self.name,
            title=clean(meta(tree, "og:title")) or (clean(h1.text()) if h1 else slug),
            subtitle=clean(meta(tree, "og:description")) or None,
            review_date=find_date(tree),
            specs_raw=self.spec_table(tree),
            pros=pros,
            cons=cons,
            faq=self.faq(tree),
            sections=self.sections(tree),
        )
