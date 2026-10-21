"""Print the compiled autochat LangGraph as Mermaid (no LLM/DB calls; uses an in-memory checkpointer).

Usage:
    .venv\\Scripts\\python.exe .github/skills/autochat-agent/scripts/graph_mermaid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    from autochat.agent.graph import build_graph  # type: ignore[import-not-found]
    from langgraph.checkpoint.memory import InMemorySaver

    graph = build_graph(InMemorySaver())
    print(graph.get_graph().draw_mermaid())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
