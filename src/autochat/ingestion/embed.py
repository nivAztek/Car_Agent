"""Embed chunks with the Foundry embedding deployment (cached by content hash + model + dims)."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from tenacity import AsyncRetrying

from autochat.config import get_settings
from autochat.ingestion.schemas import Chunk
from autochat.llm import RETRY_KW, embeddings

log = logging.getLogger(__name__)

BATCH_SIZE = 16


def _key(text: str) -> str:
    s = get_settings()
    return hashlib.sha256(f"{s.foundry_embed_deployment}|{s.embed_dimensions}|{text}".encode()).hexdigest()


def cache_path() -> Path:
    return get_settings().processed_dir / "cache" / "embeddings.jsonl"


def load_cache() -> dict[str, list[float]]:
    path = cache_path()
    if not path.exists():
        return {}
    out: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["key"]] = row["embedding"]
    return out


def _append_cache(rows: dict[str, list[float]]) -> None:
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for key, vec in rows.items():
            f.write(json.dumps({"key": key, "embedding": vec}) + "\n")


async def embed_texts(texts: list[str]) -> list[list[float]]:
    dims = get_settings().embed_dimensions
    vectors: list[list[float]] = []
    model = embeddings()
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        async for attempt in AsyncRetrying(**RETRY_KW):
            with attempt:
                got = await model.aembed_documents(batch)
        for vec in got:
            if len(vec) != dims:
                raise RuntimeError(f"embedding has {len(vec)} dims, expected {dims} (EMBED_DIMENSIONS)")
        vectors.extend(got)
    return vectors


async def embed_chunks(chunks: list[Chunk]) -> dict[str, list[float]]:
    """Return chunk.id -> embedding of chunk.embed_text(), using and filling the cache."""
    cache = load_cache()
    keys = {c.id: _key(c.embed_text()) for c in chunks}
    missing = [c for c in chunks if keys[c.id] not in cache]
    if missing:
        log.info("embedding %d chunks (%d cached)", len(missing), len(chunks) - len(missing))
        vectors = await embed_texts([c.embed_text() for c in missing])
        new = {keys[c.id]: v for c, v in zip(missing, vectors, strict=True)}
        _append_cache(new)
        cache.update(new)
    return {c.id: cache[keys[c.id]] for c in chunks}
