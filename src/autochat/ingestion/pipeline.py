"""Ingestion CLI: scrape -> parse -> structure -> chunk -> translate -> embed -> load.

    python -m autochat.ingestion.pipeline                       # full run (incremental, cached)
    python -m autochat.ingestion.pipeline --stage chunk         # stop after a stage
    python -m autochat.ingestion.pipeline --only mg-s6 --force  # re-extract one car with the LLM
    python -m autochat.ingestion.pipeline --url https://www.auto.co.il/...   # add a URL and ingest

Incremental: manifest.json keeps a content sha per car. The LLM extraction runs only for new/changed content
(or --force). Translations and embeddings are content-hash cached, so unchanged chunks cost nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autochat.config import get_settings
from autochat.ingestion.chunk import build_chunks
from autochat.ingestion.extract import PROMPT_VERSION, extract_car, merge
from autochat.ingestion.schemas import CarExtraction, CarRecord, Chunk, ParsedArticle
from autochat.ingestion.scrape import add_url, fetch_all, load_urls, slug_of
from autochat.ingestion.sources import parse_html

log = logging.getLogger("autochat.ingestion")

STAGES = ("scrape", "parse", "structure", "chunk", "translate", "embed", "load")
EXTRACT_CONCURRENCY = 3


# --------------------------------------------------------------------------- paths + io
def _p() -> dict[str, Path]:
    d = get_settings().processed_dir
    return {
        "dir": d,
        "parsed": d / "parsed",
        "extractions": d / "extractions",
        "cars": d / "cars.json",
        "chunks": d / "chunks.jsonl",
        "manifest": d / "manifest.json",
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def content_sha(article: ParsedArticle) -> str:
    """Hash of the parsed content (not raw HTML, which changes with ads/scripts on every fetch)."""
    payload = json.dumps(article.model_dump(exclude={"source_sha", "adapter"}), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_cars() -> list[CarRecord]:
    return [CarRecord.model_validate(c) for c in _read_json(_p()["cars"], [])]


def load_chunks() -> list[Chunk]:
    path = _p()["chunks"]
    if not path.exists():
        return []
    return [Chunk.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_chunks(chunks: list[Chunk]) -> None:
    path = _p()["chunks"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(c.model_dump_json() for c in chunks) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- stages
def stage_parse(urls: list[str], raw: dict[str, Path]) -> list[ParsedArticle]:
    articles = []
    for url in urls:
        slug = slug_of(url)
        article = parse_html(raw[slug].read_text(encoding="utf-8"), url, slug)
        article = article.model_copy(update={"source_sha": content_sha(article)})
        _write_json(_p()["parsed"] / f"{slug}.json", article.model_dump(mode="json"))
        log.info(
            "parsed %-28s adapter=%-10s sections=%2d faq=%2d specs=%2d chars=%d",
            slug,
            article.adapter,
            len(article.sections),
            len(article.faq),
            len(article.specs_raw),
            article.body_chars,
        )
        articles.append(article)
    return articles


async def _fill_english_lists(car: CarRecord) -> CarRecord:
    from autochat.ingestion.translate import translate_list

    update: dict[str, list[str]] = {}
    if car.pros_he and not car.pros_en:
        update["pros_en"] = await translate_list(car.pros_he)
    if car.cons_he and not car.cons_en:
        update["cons_en"] = await translate_list(car.cons_he)
    return car.model_copy(update=update) if update else car


async def stage_structure(articles: list[ParsedArticle], manifest: dict[str, Any], force: bool) -> list[CarRecord]:
    """LLM extraction for new/changed articles; deterministic re-merge from the cached extraction otherwise."""
    sem = asyncio.Semaphore(EXTRACT_CONCURRENCY)

    async def one(article: ParsedArticle) -> CarRecord:
        entry = manifest.get(article.slug, {})
        ex_path = _p()["extractions"] / f"{article.slug}.json"
        cached = _read_json(ex_path, None)
        fresh = (
            cached is not None
            and entry.get("content_sha256") == article.source_sha
            and cached.get("prompt_version") == PROMPT_VERSION
        )
        if fresh and not force:
            record = merge(article, CarExtraction.model_validate(cached["extraction"]))
            log.info("structure %-24s cached extraction", article.slug)
        else:
            async with sem:
                log.info("structure %-24s LLM extraction ...", article.slug)
                record, extraction = await extract_car(article)
            _write_json(ex_path, {"prompt_version": PROMPT_VERSION, "extraction": extraction.model_dump(mode="json")})
        return await _fill_english_lists(record)

    return list(await asyncio.gather(*(one(a) for a in articles)))


def stage_chunk(articles: list[ParsedArticle], cars: list[CarRecord], previous: list[Chunk]) -> list[Chunk]:
    by_id = {c.car_id: c for c in cars}
    old = {c.id: c for c in previous}
    chunks: list[Chunk] = []
    for article in articles:
        for chunk in build_chunks(article, by_id[article.slug]):
            prev = old.get(chunk.id)  # keep translations of unchanged chunks
            same = prev is not None and (prev.heading, prev.text_he) == (chunk.heading, chunk.text_he)
            if prev and same and chunk.chunk_type != "car_card":
                chunk = chunk.model_copy(update={"heading_en": prev.heading_en, "text_en": prev.text_en})
            chunks.append(chunk)
    return chunks


def _summary(cars: list[CarRecord], chunks: list[Chunk]) -> None:
    for car in cars:
        n = sum(1 for c in chunks if c.car_id == car.car_id)
        price = f"₪{car.price_ils:,}" if car.price_ils else "-"
        log.info(
            "  %-26s %-10s %-8s %-5s %-9s %3d chunks  table=%d llm=%d dropped=%d",
            car.car_id,
            car.make,
            car.model,
            car.powertrain or "?",
            price,
            n,
            sum(1 for v in car.field_sources.values() if v == "table"),
            sum(1 for v in car.field_sources.values() if v == "llm"),
            len(car.dropped),
        )


# --------------------------------------------------------------------------- main
async def run(
    stage: str = "load",
    force: bool = False,
    refresh: bool = False,
    only: list[str] | None = None,
    new_urls: list[str] | None = None,
) -> dict[str, Any]:
    paths = _p()
    for url in new_urls or []:
        if add_url(url):
            log.info("added %s to urls.txt", url)
    urls = load_urls()
    if only:
        urls = [u for u in urls if slug_of(u) in set(only)]
    if not urls:
        raise SystemExit("no URLs to ingest (data/urls.txt is empty or --only matched nothing)")
    stop = STAGES.index(stage)
    result: dict[str, Any] = {"urls": len(urls)}

    raw = await fetch_all(urls, refresh=refresh)
    if stop == 0:
        return result
    articles = stage_parse(urls, raw)
    if stop == 1:
        return result

    manifest: dict[str, Any] = _read_json(paths["manifest"], {})
    cars = await stage_structure(articles, manifest, force)
    # keep cars not selected by --only
    selected = {a.slug for a in articles}
    all_cars = [c for c in load_cars() if c.car_id not in selected] + cars
    all_cars.sort(key=lambda c: c.car_id)
    _write_json(paths["cars"], [c.model_dump(mode="json") for c in all_cars])
    now = datetime.now(UTC).isoformat(timespec="seconds")
    for a in articles:
        manifest[a.slug] = manifest.get(a.slug, {}) | {
            "url": a.url,
            "slug": a.slug,
            "content_sha256": a.source_sha,
            "adapter": a.adapter,
            "extract_prompt": PROMPT_VERSION,
            "structured_at": now,
        }
    _write_json(paths["manifest"], manifest)
    if stop == 2:
        return result | {"cars": len(all_cars)}

    previous = load_chunks()
    chunks = stage_chunk(articles, cars, previous)
    all_chunks = [c for c in previous if c.car_id not in selected] + chunks
    _write_chunks(all_chunks)
    result |= {"cars": len(all_cars), "chunks": len(all_chunks)}
    if stop == 3:
        _summary(all_cars, all_chunks)
        return result

    from autochat.ingestion.translate import translate_chunks

    all_chunks = await translate_chunks(all_chunks)
    _write_chunks(all_chunks)
    untranslated = [c.id for c in all_chunks if not c.text_en]
    result["untranslated"] = len(untranslated)
    if stop == 4:
        _summary(all_cars, all_chunks)
        return result

    from autochat.ingestion.embed import embed_chunks

    vectors = await embed_chunks(all_chunks)
    result["embedded"] = len(vectors)
    if stop == 5:
        _summary(all_cars, all_chunks)
        return result

    from autochat.db.cosmos import close_client
    from autochat.ingestion.load import load

    try:
        result |= {"loaded": await load(all_cars, all_chunks, vectors)}
    finally:
        await close_client()
    for slug in manifest:
        manifest[slug]["loaded_at"] = now
    _write_json(paths["manifest"], manifest)
    _summary(all_cars, all_chunks)
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="autochat.ingestion.pipeline", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stage", choices=STAGES, default="load", help="stop after this stage (default: load)")
    ap.add_argument("--force", action="store_true", help="re-run the LLM extraction even if content is unchanged")
    ap.add_argument("--refresh", action="store_true", help="re-download HTML even if cached")
    ap.add_argument("--only", nargs="+", metavar="SLUG", help="limit to these car slugs")
    ap.add_argument("--url", nargs="+", default=[], help="add URL(s) to data/urls.txt, then ingest")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=get_settings().log_level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    for noisy in ("httpx", "azure", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    only = args.only or ([slug_of(u) for u in args.url] if args.url else None)
    result = asyncio.run(run(args.stage, args.force, args.refresh, only, args.url))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
