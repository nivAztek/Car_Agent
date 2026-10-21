---
applyTo: "**/*.py"
description: "Python conventions for the autochat project"
---
# Python conventions

- Target Python 3.13. Use `from __future__ import annotations`, full type hints, and `X | None` (not `Optional`).
- Use Pydantic v2 for data models (`model_validate`, `model_dump(mode="json")`). Load settings only through `autochat.config.get_settings()` (cached `pydantic-settings`, reads `.env`).
- Code is async-first. Use `azure.cosmos.aio.CosmosClient`, `await llm.ainvoke(...)`, `graph.astream(...)`. CLI entry points wrap with `asyncio.run(main())`.
- On Windows, scripts that print Hebrew must call `sys.stdout.reconfigure(encoding="utf-8")`. Read and write files with `encoding="utf-8"` and use `json.dumps(..., ensure_ascii=False)`.
- Retries: use `tenacity` (`stop_after_attempt(4)`, `wait_exponential_jitter`) around network calls to Foundry and Cosmos. Respect 429 retry-after.
- Logging: use `logging.getLogger(__name__)`, never `print`, inside `src/`. Scripts may print.
- No secrets in code, logs, or exceptions.
- Tests: `pytest` + `pytest-asyncio` (auto mode). Unit tests must not need network. Use cached HTML in `data/raw/` and fake LLM/embeddings.
- Lint with `ruff` (line length 120).
