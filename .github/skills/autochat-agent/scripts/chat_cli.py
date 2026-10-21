"""Terminal chat for autochat - in-process graph (default) or via the FastAPI SSE endpoint.

Usage (repo root):
    .venv\\Scripts\\python.exe .github/skills/autochat-agent/scripts/chat_cli.py
    .venv\\Scripts\\python.exe .github/skills/autochat-agent/scripts/chat_cli.py --api http://localhost:8000
    .venv\\Scripts\\python.exe .github/skills/autochat-agent/scripts/chat_cli.py --script eval/transcripts/family_he.txt --out eval/transcripts/family_he.md

In-process mode requires `autochat.agent.runtime.get_runtime()` (async context manager yielding (graph, deps)).
Commands while chatting: /new (new thread), /profile (show profile), /quit
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))


async def turn_via_api(api: str, thread_id: str, message: str) -> dict:
    import httpx

    answer, sources, profile = [], [], {}
    event = None
    async with httpx.AsyncClient(timeout=180) as client:
        async with client.stream("POST", f"{api}/chat", json={"thread_id": thread_id, "message": message}) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data = json.loads(line[5:].strip() or "{}")
                    if event == "token":
                        print(data.get("t", ""), end="", flush=True)
                        answer.append(data.get("t", ""))
                    elif event == "sources":
                        sources = data
                    elif event == "profile":
                        profile = data
    print()
    return {"answer": "".join(answer), "sources": sources, "profile": profile}


async def turn_in_process(graph, thread_id: str, message: str) -> dict:
    """Uses the traced runtime wrapper (MLflow trace per turn, session = thread_id)."""
    from autochat.agent.runtime import run_turn  # type: ignore[import-not-found]

    out = await run_turn(
        graph, thread_id=thread_id, message=message, user="cli", on_token=lambda t: print(t, end="", flush=True)
    )
    print()
    if out.get("trace_id"):
        print(f"  trace: {out['trace_id']}")
    return out


def dump(obj) -> str:
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json", exclude_none=True)
    if isinstance(obj, list):
        obj = [o.model_dump(mode="json") if hasattr(o, "model_dump") else o for o in obj]
    return json.dumps(obj, ensure_ascii=False)


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", help="FastAPI base URL; omit for in-process graph")
    ap.add_argument("--script", help="text file, one user message per line")
    ap.add_argument("--out", help="write a Markdown transcript here")
    ap.add_argument("--thread", default=None)
    args = ap.parse_args()

    thread_id = args.thread or str(uuid.uuid4())
    scripted = (
        [ln.strip() for ln in Path(args.script).read_text(encoding="utf-8").splitlines() if ln.strip()]
        if args.script
        else None
    )
    transcript: list[str] = []
    last_profile: dict = {}

    async def run(turn_fn) -> None:
        nonlocal thread_id, last_profile
        queue = list(scripted) if scripted is not None else None
        while True:
            if queue is not None:
                if not queue:
                    break
                msg = queue.pop(0)
                print(f"\nyou> {msg}")
            else:
                msg = input("\nyou> ").strip()
            if not msg:
                continue
            if msg == "/quit":
                break
            if msg == "/new":
                thread_id = str(uuid.uuid4())
                print(f"(new thread {thread_id})")
                continue
            if msg == "/profile":
                print(dump(last_profile))
                continue
            print("bot> ", end="")
            out = await turn_fn(thread_id, msg)
            last_profile = out["profile"]
            print(f"  sources: {dump(out['sources'])[:600]}")
            print(f"  profile: {dump(out['profile'])[:400]}")
            transcript.append(f"**User:** {msg}\n\n**Bot:** {out['answer']}\n")
            if out["sources"]:
                transcript.append(
                    "<details><summary>Sources</summary>\n\n```json\n" + dump(out["sources"]) + "\n```\n</details>\n"
                )

    if args.api:
        await run(lambda t, m: turn_via_api(args.api.rstrip("/"), t, m))
    else:
        from autochat.agent.runtime import get_runtime  # type: ignore[import-not-found]
        from autochat.observability import setup_mlflow  # type: ignore[import-not-found]

        setup_mlflow()  # traces -> experiment MLFLOW_EXPERIMENT_NAME (default "autochat")
        async with get_runtime() as (graph, _deps):
            await run(lambda t, m: turn_in_process(graph, t, m))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(transcript), encoding="utf-8")
        print(f"transcript -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
