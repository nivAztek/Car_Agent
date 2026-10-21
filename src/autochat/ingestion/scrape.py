"""Download article HTML with a polite, cached, host-checked async client."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from urllib.parse import urlparse

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from autochat.config import get_settings

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0 Safari/537.36"
    ),
    "Accept-Language": "he-IL,he;q=0.9,en;q=0.8",
}
POLITE_DELAY_S = 1.0


def slug_of(url: str) -> str:
    """Last non-empty path segment. It is also the car_id and Cosmos partition key."""
    return urlparse(url).path.rstrip("/").rsplit("/", 1)[-1].lower()


def load_urls(path: Path | None = None) -> list[str]:
    path = path or get_settings().urls_file
    if not path.exists():
        return []
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    seen: dict[str, None] = {}
    for line in lines:
        if line and not line.startswith("#"):
            seen.setdefault(line, None)
    return list(seen)


def add_url(url: str, path: Path | None = None) -> bool:
    """Append a URL to urls.txt if new. Returns True if added."""
    path = path or get_settings().urls_file
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"Not an http(s) URL: {url!r}")
    if url in load_urls(path):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    sep = "" if not existing or existing.endswith("\n") else "\n"
    path.write_text(f"{existing}{sep}{url}\n", encoding="utf-8")
    return True


def raw_path(slug: str) -> Path:
    return get_settings().raw_dir / f"{slug}.html"


@retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=8),
    reraise=True,
)
async def _get(client: httpx.AsyncClient, url: str) -> str:
    resp = await client.get(url)
    if resp.status_code >= 500:
        resp.raise_for_status()
    if resp.status_code >= 400:
        raise RuntimeError(f"{url} -> HTTP {resp.status_code}")
    requested, final = urlparse(url).hostname or "", resp.url.host or ""
    if final.removeprefix("www.") != requested.removeprefix("www."):
        raise RuntimeError(f"{url} redirected off-site to {resp.url}")
    return resp.text


async def fetch_all(urls: list[str], refresh: bool = False) -> dict[str, Path]:
    """Download (or reuse cached) HTML for every URL. Returns slug -> cached file path."""
    raw_dir = get_settings().raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True, timeout=30) as client:
        for url in urls:
            slug = slug_of(url)
            path = raw_path(slug)
            if path.exists() and not refresh:
                out[slug] = path
                continue
            log.info("fetch %s", url)
            html = await _get(client, url)
            path.write_text(html, encoding="utf-8")
            out[slug] = path
            await asyncio.sleep(POLITE_DELAY_S)
    return out
