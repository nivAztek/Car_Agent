"""Smoke test Microsoft Foundry deployments used by autochat (via langchain-azure-ai).

Checks:
  1. chat completion on FOUNDRY_CHAT_DEPLOYMENT (Hebrew in -> answer)
  2. structured output (Pydantic) on FOUNDRY_FAST_DEPLOYMENT (falls back to CHAT)
  3. embeddings on FOUNDRY_EMBED_DEPLOYMENT with EMBED_DIMENSIONS; verifies vector length + HE/EN similarity
  4. token streaming (astream) on the chat deployment

Usage:
    .venv\\Scripts\\python.exe .github/skills/foundry-models/scripts/foundry_smoke.py
Requires .env (see .env.example). Never prints the API key.
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[4]


def load_env() -> None:
    env = REPO_ROOT / ".env"
    if not env.exists():
        sys.exit("Missing .env - copy .env.example to .env and fill FOUNDRY_* values.")
    for line in env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def credential():
    key = os.environ.get("FOUNDRY_API_KEY", "").strip()
    if key:
        return key
    from azure.identity import DefaultAzureCredential  # Entra ID fallback

    return DefaultAzureCredential()


class CarIntent(BaseModel):
    intent: str = Field(description="one of: recommend, compare, car_question, general_chat, out_of_scope")
    language: str = Field(description="he or en")
    mentioned_cars: list[str] = Field(default_factory=list)
    max_price_ils: int | None = None


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    load_env()
    from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
    from langchain_azure_ai.embeddings import AzureAIOpenAIApiEmbeddingsModel

    endpoint = os.environ.get("FOUNDRY_ENDPOINT", "").rstrip("/")
    chat_dep = os.environ.get("FOUNDRY_CHAT_DEPLOYMENT", "")
    fast_dep = os.environ.get("FOUNDRY_FAST_DEPLOYMENT", "") or chat_dep
    embed_dep = os.environ.get("FOUNDRY_EMBED_DEPLOYMENT", "")
    dims = int(os.environ.get("EMBED_DIMENSIONS", "1536"))
    if not (endpoint and chat_dep and embed_dep):
        sys.exit("FOUNDRY_ENDPOINT, FOUNDRY_CHAT_DEPLOYMENT and FOUNDRY_EMBED_DEPLOYMENT are required.")
    if not endpoint.endswith("/openai/v1"):
        print(f"WARN endpoint does not end with /openai/v1 - it is used verbatim as base_url: {endpoint}")
    if os.environ.get("AZURE_AI_PROJECT_ENDPOINT"):
        print("WARN AZURE_AI_PROJECT_ENDPOINT is set in the environment and may take precedence.")

    cred = credential()
    failures = 0

    def ok(name: str, cond: bool, detail: str = "") -> None:
        nonlocal failures
        failures += 0 if cond else 1
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    # 1. chat
    chat = AzureAIOpenAIApiChatModel(endpoint=endpoint, credential=cred, model=chat_dep)
    t = time.perf_counter()
    try:
        resp = await chat.ainvoke("ענה במשפט אחד בעברית: מה זה טווח WLTP ברכב חשמלי?")
        ok("chat", bool(resp.content), f"{chat_dep} {time.perf_counter() - t:.1f}s -> {str(resp.content)[:90]!r}")
    except Exception as exc:  # noqa: BLE001
        ok("chat", False, f"{type(exc).__name__}: {str(exc)[:300]}")

    # 2. structured output
    fast = AzureAIOpenAIApiChatModel(endpoint=endpoint, credential=cred, model=fast_dep)
    try:
        structured = fast.with_structured_output(CarIntent)
        out = await structured.ainvoke("I have up to 200k shekels, compare the MG S6 and Aion HT for me")
        ok(
            "structured output",
            isinstance(out, CarIntent) and out.intent == "compare",
            f"{fast_dep} -> {out.model_dump() if isinstance(out, CarIntent) else out}",
        )
    except Exception as exc:  # noqa: BLE001
        ok("structured output", False, f"{type(exc).__name__}: {str(exc)[:300]}")

    # 3. embeddings
    emb = AzureAIOpenAIApiEmbeddingsModel(endpoint=endpoint, credential=cred, model=embed_dep, dimensions=dims)
    try:
        he, en, other = await emb.aembed_documents(
            ["תא המטען של הרכב גדול ומרווח", "The car's trunk is large and spacious", "The engine sounds amazing"]
        )
        ok("embedding dims", len(he) == dims, f"{embed_dep} len={len(he)} expected={dims}")
        s_same, s_diff = cosine(he, en), cosine(he, other)
        ok("cross-lingual similarity", s_same > s_diff, f"he~en={s_same:.3f} he~other={s_diff:.3f}")
    except Exception as exc:  # noqa: BLE001
        ok("embeddings", False, f"{type(exc).__name__}: {str(exc)[:300]}")

    # 4. streaming
    try:
        n = 0
        async for _ in chat.astream("Count from 1 to 5, comma separated."):
            n += 1
        ok("streaming", n > 1, f"chunks={n}")
    except Exception as exc:  # noqa: BLE001
        ok("streaming", False, f"{type(exc).__name__}: {str(exc)[:300]}")

    print("\nALL OK" if failures == 0 else f"\n{failures} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
