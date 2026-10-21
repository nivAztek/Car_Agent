"""Dump the DOM skeleton of a cached article to find content containers.

Usage:
    .venv\\Scripts\\python.exe .github/skills/autochat-ingestion/scripts/inspect_dom.py long-term-report-kia-ev9-4 [--min-chars 400]

Prints every element whose *direct* paragraph text is large, with its tag/class path,
so you can identify the article body container and heading tags on unusual pages.
Also prints the tag/class of every heading (h1-h4) in document order.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from selectolax.lexbor import LexborHTMLParser

RAW_DIR = Path(__file__).resolve().parents[4] / "data" / "raw"


def path_of(node) -> str:
    parts = []
    while node is not None and node.tag not in ("html", "-undef"):
        cls = (node.attributes.get("class") or "").split()
        parts.append(node.tag + ("." + ".".join(cls[:2]) if cls else ""))
        node = node.parent
    return " > ".join(reversed(parts[:6]))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--min-chars", type=int, default=400)
    args = ap.parse_args()

    html = (RAW_DIR / f"{args.slug}.html").read_text(encoding="utf-8")
    tree = LexborHTMLParser(html)
    for bad in tree.css("script, style, noscript, svg"):
        bad.decompose()

    print("== headings ==")
    for h in tree.css("h1, h2, h3, h4"):
        txt = re.sub(r"\s+", " ", h.text()).strip()
        if txt:
            print(f"{h.tag:<3} {path_of(h.parent)[-70:]:<70} | {txt[:70]}")

    print("\n== containers holding many <p> ==")
    counter: Counter[str] = Counter()
    sizes: dict[str, int] = {}
    for p in tree.css("p"):
        parent = p.parent
        if parent is None:
            continue
        key = path_of(parent)
        counter[key] += 1
        sizes[key] = sizes.get(key, 0) + len(p.text(strip=True))
    for key, n in counter.most_common(8):
        if sizes[key] >= args.min_chars:
            print(f"{n:>3} p, {sizes[key]:>6} chars  {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
