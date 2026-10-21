"""Probe auto.co.il article pages and report their structure.

Usage (from repo root):
    .venv\\Scripts\\python.exe .github/skills/autochat-ingestion/scripts/probe_articles.py            # all URLs in data/urls.txt
    .venv\\Scripts\\python.exe .github/skills/autochat-ingestion/scripts/probe_articles.py --slug mg-s6
    .venv\\Scripts\\python.exe .github/skills/autochat-ingestion/scripts/probe_articles.py --refresh  # ignore cache

What it does:
  * Downloads each article (browser User-Agent, cached to data/raw/<slug>.html).
  * Prints: og:title, publish date candidates, H2 section headings, spec table (label -> value),
    pros/cons, FAQ count, and body text length.
  * Writes a JSON summary to data/raw/_probe_summary.json so selectors can be compared across pages.

Use this BEFORE changing parse.py and whenever a page parses with missing fields.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx
from selectolax.lexbor import LexborHTMLParser as HTMLParser
from selectolax.lexbor import LexborNode as Node

REPO_ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = REPO_ROOT / "data" / "raw"
URLS_FILE = REPO_ROOT / "data" / "urls.txt"

DEFAULT_URLS = [
    "https://www.auto.co.il/articles/test-drives/road-tests/citroen-c3-2026/",
    "https://www.auto.co.il/articles/test-drives/road-tests/audi-rs3-fl/",
    "https://www.auto.co.il/articles/test-drives/road-tests/long-term-report-kia-ev9-4/",
    "https://www.auto.co.il/articles/test-drives/road-tests/mg-s6/",
    "https://www.auto.co.il/articles/test-drives/road-tests/hyundai-elantra-n-manual/",
    "https://www.auto.co.il/articles/test-drives/road-tests/aion-ht/",
    "https://www.auto.co.il/articles/test-drives/road-tests/link-and-co-01-2026/",
    "https://www.auto.co.il/articles/test-drives/road-tests/genesis-gv80-2026/",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0 Safari/537.36"
    ),
    "Accept-Language": "he-IL,he;q=0.9,en;q=0.8",
}

BOILERPLATE_H2 = ("הצטרפו לוואטסאפ", "כתבות אחרונות", "כתבות באותם נושאים", "מותגים מתחרים", "המומחה האובייקטיבי")


def slug_of(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


def load_urls() -> list[str]:
    if URLS_FILE.exists():
        urls = [
            u.strip() for u in URLS_FILE.read_text(encoding="utf-8").splitlines() if u.strip() and not u.startswith("#")
        ]
        if urls:
            return urls
    return DEFAULT_URLS


def fetch(url: str, refresh: bool) -> str:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{slug_of(url)}.html"
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=30) as client:
        resp = client.get(url)
        resp.raise_for_status()
        if "auto.co.il" not in str(resp.url):
            raise RuntimeError(f"Redirected away from site: {resp.url}")
        path.write_text(resp.text, encoding="utf-8")
        time.sleep(1.0)  # be polite
        return resp.text


def clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()


def meta(tree: HTMLParser, key: str) -> str | None:
    node = tree.css_first(f'meta[property="{key}"]') or tree.css_first(f'meta[name="{key}"]')
    return node.attributes.get("content") if node else None


def date_candidates(tree: HTMLParser, html: str) -> list[str]:
    found: list[str] = []
    for key in ("article:published_time", "article:modified_time", "date", "pubdate"):
        val = meta(tree, key)
        if val:
            found.append(f"{key}={val}")
    for node in tree.css("time"):
        found.append(f"time={node.attributes.get('datetime') or clean(node.text())}")
    found += [f"regex={m}" for m in re.findall(r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b", html)[:3]]
    return found


def spec_table(tree: HTMLParser) -> dict[str, str]:
    specs: dict[str, str] = {}
    for item in tree.css(".vehicle-table__item, .table-item"):
        title = item.css_first(".table-item__title")
        value = item.css_first(".table-item__value")
        if title and value:
            key = clean(title.text())
            if key and key not in specs:
                specs[key] = clean(value.text())
    return specs


def pros_cons(tree: HTMLParser) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for col in tree.css(".pros-cons-table__col"):
        title = col.css_first(".pros-cons-table__title")
        name = clean(title.text()) if title else f"col{len(out)}"
        items = [clean(li.text()) for li in col.css("li")] or [clean(col.text()).replace(name, "", 1).strip()]
        out[name] = [i for i in items if i]
    return out


BODY_SELECTOR = "div.sections__items .text-wrapper"  # verified on all 8 pages
BLOCK_TAGS = ("p", "ul", "ol", "blockquote", "table")


def walk_blocks(node: Node):
    """Yield headings/blocks in document order, descending only into layout containers (div/section/figure)."""
    for child in node.iter():
        if child.tag in ("div", "section", "figure", "span") and not child.css_matches(
            ".pros-cons-table, .vehicle-table"
        ):
            yield from walk_blocks(child)
        else:
            yield child


def _is_bold_subheading(node: Node) -> bool:
    """Pages without <h2> (long-term reports, some tests) use a short <p><strong>..</strong></p> as subheading."""
    if node.tag != "p":
        return False
    strong = node.css_first("strong, b")
    if strong is None:
        return False
    full, bold = clean(node.text()), clean(strong.text())
    return 0 < len(full) <= 90 and len(bold) >= 0.8 * len(full)


def sections(tree: HTMLParser) -> list[dict]:
    """Walk body text blocks in document order; start a new section on h2/h3 or a bold-only short paragraph."""
    result: list[dict] = []
    current = {"heading": "פתיח", "paras": [], "kind": "intro"}

    def push() -> None:
        text = "\n".join(current["paras"])
        if text:
            result.append(
                {"heading": current["heading"], "kind": current["kind"], "chars": len(text), "preview": text[:120]}
            )

    for wrapper in tree.css(BODY_SELECTOR):
        for node in walk_blocks(wrapper):
            if node.tag in ("h2", "h3"):
                heading = clean(node.text())
                if not heading or any(b in heading for b in BOILERPLATE_H2):
                    continue
                push()
                current = {"heading": heading, "paras": [], "kind": node.tag}
            elif _is_bold_subheading(node):
                push()
                current = {"heading": clean(node.text()), "paras": [], "kind": "bold"}
            elif node.tag in BLOCK_TAGS:
                txt = clean(node.text())
                if txt:
                    current["paras"].append(txt)
    push()
    return result


def faq(tree: HTMLParser) -> list[dict]:
    """FAQ accordion: div.faq__container details.spollers__item > summary (question) + .spollers__body (answer)."""
    items = []
    for item in tree.css("div.faq__container details.spollers__item"):
        q = item.css_first("summary")
        a = item.css_first(".spollers__body")
        if q:
            items.append({"q": clean(q.text()), "a": clean(a.text()) if a else ""})
    return items


def probe(url: str, refresh: bool) -> dict:
    html = fetch(url, refresh)
    tree = HTMLParser(html)
    h1 = tree.css_first("h1")
    secs = sections(tree)
    return {
        "slug": slug_of(url),
        "url": url,
        "og_title": meta(tree, "og:title"),
        "og_description": meta(tree, "og:description"),
        "h1": clean(h1.text()) if h1 else None,
        "dates": date_candidates(tree, html),
        "specs": spec_table(tree),
        "pros_cons": pros_cons(tree),
        "faq": faq(tree),
        "sections": secs,
        "body_chars": sum(s["chars"] for s in secs),
        "html_bytes": len(html),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", help="probe only the URL with this slug")
    ap.add_argument("--refresh", action="store_true", help="re-download even if cached")
    ap.add_argument("--quiet", action="store_true", help="only print the summary table")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")  # Hebrew output on Windows consoles
    urls = [u for u in load_urls() if not args.slug or slug_of(u) == args.slug]
    results = []
    for url in urls:
        try:
            r = probe(url, args.refresh)
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            print(f"!! {slug_of(url)}: {exc}")
            continue
        results.append(r)
        if not args.quiet:
            print(f"\n=== {r['slug']} ===")
            print(f"title: {r['og_title']}")
            print(f"dates: {r['dates'][:3]}")
            print("specs:")
            for k, v in r["specs"].items():
                print(f"  {k} = {v}")
            print(f"pros/cons: {json.dumps(r['pros_cons'], ensure_ascii=False)[:300]}")
            print(f"faq items: {len(r['faq'])}")
            print("sections:")
            for s in r["sections"]:
                print(f"  [{s['chars']:>5}] ({s['kind']}) {s['heading']}")
            for f in r["faq"][:3]:
                print(f"  FAQ: {f['q'][:80]}")

    print("\nslug                              specs  sections  faq  body_chars")
    for r in results:
        print(f"{r['slug']:<34}{len(r['specs']):>5}{len(r['sections']):>10}{len(r['faq']):>5}{r['body_chars']:>12}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "_probe_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
