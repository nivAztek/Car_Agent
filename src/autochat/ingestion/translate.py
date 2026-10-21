"""Hebrew -> English translation of chunks (fast model, structured output, content-hash cache).

The English text is embedded alongside the Hebrew (cross-lingual recall) and fed to the in-app BM25 so
English queries match. Cache: data/processed/cache/translations.json  {sha256(heading|text_he): {...}}.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from tenacity import AsyncRetrying

from autochat.config import get_settings
from autochat.ingestion.schemas import Chunk
from autochat.llm import RETRY_KW, chat_llm

log = logging.getLogger(__name__)

PROMPT_VERSION = "translate-v1"
BATCH_SIZE = 5
CONCURRENCY = 4

SYSTEM_PROMPT = """You translate Israeli Hebrew car-review text into natural, faithful English.
Rules:
- Translate everything; do not summarize, add, or omit facts. Keep every number, unit and currency (₪) exactly.
- Keep brand and model names in their usual English form (e.g. "לינק אנד קו" -> "Lynk & Co", "קיה" -> "Kia").
- Car jargon: כ"ס = hp, קג"מ = kgm, קוט"ש = kWh, קמ"ש = km/h, ק"מ = km, נ"מ = Nm.
- Return one item per input item with the same id."""


class TranslatedItem(BaseModel):
    id: str
    heading_en: str = Field(description="English translation of the heading")
    text_en: str = Field(description="Full English translation of the text")


class TranslationBatch(BaseModel):
    items: list[TranslatedItem]


class TranslatedList(BaseModel):
    items_en: list[str] = Field(description="English translation of each input phrase, same order and count")


def _key(heading: str, text: str) -> str:
    return hashlib.sha256(f"{PROMPT_VERSION}|{heading}|{text}".encode()).hexdigest()


def cache_path() -> Path:
    return get_settings().processed_dir / "cache" / "translations.json"


def load_cache() -> dict[str, dict[str, str]]:
    path = cache_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_cache(cache: dict[str, dict[str, str]]) -> None:
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")


async def _translate_batch(batch: list[Chunk]) -> dict[str, TranslatedItem]:
    structured = chat_llm("fast").with_structured_output(TranslationBatch)
    payload = [{"id": c.id, "heading": c.heading, "text": c.text_he} for c in batch]
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(json.dumps(payload, ensure_ascii=False))]
    async for attempt in AsyncRetrying(**RETRY_KW):
        with attempt:
            result = await structured.ainvoke(messages)
    if not isinstance(result, TranslationBatch):
        result = TranslationBatch.model_validate(result)
    return {item.id: item for item in result.items}


async def translate_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Fill heading_en/text_en for chunks that lack them (car_card chunks are rendered bilingual already)."""
    cache = load_cache()
    todo: list[Chunk] = []
    out: dict[str, Chunk] = {}
    for c in chunks:
        if c.text_en:
            out[c.id] = c
            continue
        hit = cache.get(_key(c.heading, c.text_he))
        if hit:
            out[c.id] = c.model_copy(update=hit)
        else:
            todo.append(c)

    sem = asyncio.Semaphore(CONCURRENCY)

    async def run(batch: list[Chunk]) -> None:
        async with sem:
            got = await _translate_batch(batch)
        for c in batch:
            item = got.get(c.id)
            if item is None and len(batch) > 1:  # model dropped an item: retry it alone
                async with sem:
                    item = (await _translate_batch([c])).get(c.id)
            if item is None or not item.text_en.strip():
                log.warning("translation missing for %s", c.id)
                out[c.id] = c
                continue
            update = {"heading_en": item.heading_en.strip(), "text_en": item.text_en.strip()}
            cache[_key(c.heading, c.text_he)] = update
            out[c.id] = c.model_copy(update=update)

    batches = [todo[i : i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    if batches:
        log.info("translating %d chunks in %d batches (%d cached)", len(todo), len(batches), len(chunks) - len(todo))
        try:
            await asyncio.gather(*(run(b) for b in batches))
        finally:
            save_cache(cache)
    return [out[c.id] for c in chunks]


async def translate_list(items_he: list[str]) -> list[str]:
    """Translate short phrases (pros/cons) when the extraction did not return aligned English."""
    if not items_he:
        return []
    cache = load_cache()
    key = _key("list", "\n".join(items_he))
    if key in cache:
        return json.loads(cache[key]["items"])
    structured = chat_llm("fast").with_structured_output(TranslatedList)
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(json.dumps(items_he, ensure_ascii=False))]
    async for attempt in AsyncRetrying(**RETRY_KW):
        with attempt:
            result = await structured.ainvoke(messages)
    if not isinstance(result, TranslatedList):
        result = TranslatedList.model_validate(result)
    if len(result.items_en) != len(items_he):
        log.warning("list translation count mismatch (%d vs %d)", len(result.items_en), len(items_he))
        return []
    cache[key] = {"items": json.dumps(result.items_en, ensure_ascii=False)}
    save_cache(cache)
    return result.items_en
