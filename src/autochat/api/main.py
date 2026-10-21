"""FastAPI app: `uvicorn autochat.api.main:app --app-dir src` (see `scripts/dev.ps1 api`).

Endpoints
- POST /chat            SSE: `token` {"t"} ... -> `sources` -> `profile` -> `done` {trace_id, intent, language, steps}
                        | `error` {"detail"}
- POST /chat/invoke     same payload as JSON (eval, tests)
- POST /feedback        thumbs up/down -> MLflow human assessment on the trace (204)
- GET  /threads/{id}/profile | /threads/{id}/history
- GET  /cars?q=&offset=&limit= (paged; total in X-Total-Count) | /cars/{car_id}
- GET  /health          Cosmos + Foundry checks (503 when degraded)

A turn runs in its own task, so a client disconnect never cancels a half-written checkpoint; at most one turn
per thread runs at a time (409 otherwise).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Path, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse

from autochat import observability as obs
from autochat.agent.profile import UserProfile
from autochat.agent.runtime import get_runtime, run_turn
from autochat.api.schemas import (
    THREAD_ID_PATTERN,
    CarSummary,
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    HealthResponse,
    HistoryMessage,
    ThreadHistory,
)

log = logging.getLogger(__name__)

RuntimeFactory = Callable[[], contextlib.AbstractAsyncContextManager[tuple[Any, Any]]]
HealthCheck = Callable[[], Awaitable[dict[str, Any]]]

CORS_ORIGINS = ["http://localhost:8501", "http://127.0.0.1:8501"]
INTERNAL_CAR_FIELDS = {"adapter", "source_sha", "dropped", "name_keys"}
CARS_PAGE_MAX = 200
HEALTH_TIMEOUT_S = 10.0
FOUNDRY_HEALTH_TTL_S = 60.0
SSE_PING_S = 15
GENERIC_ERROR = "The assistant failed to answer this message. Please try again."

ThreadId = Annotated[str, Path(pattern=THREAD_ID_PATTERN)]


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


# --------------------------------------------------------------------------- health checks


async def _cosmos_check() -> dict[str, Any]:
    from autochat.db.cosmos import cars_container

    items = [x async for x in cars_container().query_items("SELECT VALUE COUNT(1) FROM c")]
    return {"cars_docs": items[0] if items else 0}


def _cached(check: HealthCheck, ttl_s: float) -> HealthCheck:
    """Cache a successful result for `ttl_s` (avoid paying for an embedding call on every probe)."""
    memo: dict[str, Any] = {"at": 0.0, "value": None}

    async def run() -> dict[str, Any]:
        if memo["value"] is not None and time.monotonic() - memo["at"] < ttl_s:
            return {**memo["value"], "cached": True}
        value = await check()
        memo.update(at=time.monotonic(), value=value)
        return value

    return run


async def _foundry_check() -> dict[str, Any]:
    from autochat.llm import embeddings

    vec = await embeddings().aembed_query("ping")
    return {"embedding_dims": len(vec)}


def default_health_checks() -> dict[str, HealthCheck]:
    return {"cosmos": _cosmos_check, "foundry": _cached(_foundry_check, FOUNDRY_HEALTH_TTL_S)}


async def _run_check(check: HealthCheck) -> dict[str, Any]:
    try:
        detail = await asyncio.wait_for(check(), HEALTH_TIMEOUT_S)
        return {"ok": True, **detail}
    except Exception as exc:  # noqa: BLE001 - report, never raise from /health
        return {"ok": False, "error": type(exc).__name__}


# --------------------------------------------------------------------------- app


def create_app(
    runtime_factory: RuntimeFactory | None = None,
    *,
    tracing: bool = True,
    health_checks: dict[str, HealthCheck] | None = None,
) -> FastAPI:
    """Build the app. Tests pass a fake `runtime_factory` (yielding `(graph, ctx)`) and `tracing=False`."""
    factory = runtime_factory or get_runtime
    checks = health_checks if health_checks is not None else default_health_checks()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.tracing = obs.setup_mlflow() if tracing else False
        app.state.busy = set()  # thread_ids with a running turn
        app.state.tasks = set()  # strong refs to running turn tasks
        async with factory() as (graph, ctx):
            app.state.graph, app.state.ctx = graph, ctx
            scope = await ctx.catalog.scope()
            log.info("autochat API ready: %d cars, checkpointer=%s", scope.count, type(graph.checkpointer).__name__)
            try:
                yield
            finally:
                pending = list(app.state.tasks)
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

    app = FastAPI(title="autochat", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    def start_turn(request: Request, body: ChatRequest) -> tuple[asyncio.Task[dict[str, Any]], asyncio.Queue]:
        """Reserve the thread and run the turn in a background task that streams into a queue."""
        state = request.app.state
        if body.thread_id in state.busy:
            raise HTTPException(409, "A turn is already running for this thread_id")
        state.busy.add(body.thread_id)
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        async def turn() -> dict[str, Any]:
            try:
                result = await run_turn(
                    state.graph,
                    thread_id=body.thread_id,
                    message=body.message,
                    user=body.user,
                    on_token=lambda t: queue.put_nowait(("token", t)),
                    context=state.ctx,
                )
                queue.put_nowait(("result", result))
                return result
            except asyncio.CancelledError:
                queue.put_nowait(("error", GENERIC_ERROR))
                raise
            except Exception:
                log.exception("turn failed (thread_id=%s)", body.thread_id)
                queue.put_nowait(("error", GENERIC_ERROR))
                raise
            finally:
                state.busy.discard(body.thread_id)

        task = asyncio.create_task(turn(), name=f"turn:{body.thread_id}")
        state.tasks.add(task)
        task.add_done_callback(state.tasks.discard)
        task.add_done_callback(lambda t: t.cancelled() or t.exception())  # mark exception as retrieved
        return task, queue

    def to_response(result: dict[str, Any]) -> ChatResponse:
        return ChatResponse.model_validate({**result, "profile": result.get("profile") or {}})

    @app.post("/chat")
    async def chat(request: Request, body: ChatRequest) -> EventSourceResponse:
        _, queue = start_turn(request, body)

        async def events() -> AsyncIterator[dict[str, str]]:
            while True:
                kind, payload = await queue.get()
                if kind == "token":
                    yield {"event": "token", "data": _dumps({"t": payload})}
                elif kind == "result":
                    resp = to_response(payload)
                    yield {"event": "sources", "data": _dumps([s.model_dump() for s in resp.sources])}
                    yield {"event": "profile", "data": _dumps(resp.profile.model_dump())}
                    done = resp.model_dump(include={"trace_id", "intent", "language", "steps"})
                    yield {"event": "done", "data": _dumps(done)}
                    return
                else:
                    yield {"event": "error", "data": _dumps({"detail": payload})}
                    return

        return EventSourceResponse(events(), ping=SSE_PING_S)

    @app.post("/chat/invoke", response_model=ChatResponse)
    async def chat_invoke(request: Request, body: ChatRequest) -> ChatResponse:
        task, _ = start_turn(request, body)
        try:
            result = await asyncio.shield(task)  # a dropped client must not cancel the turn
        except Exception as exc:
            raise HTTPException(500, GENERIC_ERROR) from exc
        return to_response(result)

    async def thread_values(request: Request, thread_id: str) -> dict[str, Any]:
        snap = await request.app.state.graph.aget_state({"configurable": {"thread_id": thread_id}})
        return dict(snap.values or {})

    @app.get("/threads/{thread_id}/profile", response_model=UserProfile)
    async def thread_profile(request: Request, thread_id: ThreadId) -> UserProfile:
        values = await thread_values(request, thread_id)
        return UserProfile.model_validate(values.get("user_profile") or {})

    @app.get("/threads/{thread_id}/history", response_model=ThreadHistory)
    async def thread_history(request: Request, thread_id: ThreadId) -> ThreadHistory:
        values = await thread_values(request, thread_id)
        roles = {"human": "user", "ai": "assistant"}
        messages = [
            HistoryMessage(role=roles[m.type], content=str(m.content))
            for m in values.get("messages", [])
            if m.type in roles and m.content
        ]
        return ThreadHistory(thread_id=thread_id, summary=values.get("summary") or "", messages=messages)

    @app.get("/cars", response_model=list[CarSummary])
    async def list_cars(
        request: Request,
        response: Response,
        q: Annotated[str, Query(max_length=80)] = "",
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=CARS_PAGE_MAX)] = 50,
    ) -> list[CarSummary]:
        cars, total = await request.app.state.ctx.catalog.page(q, offset, limit)
        response.headers["X-Total-Count"] = str(total)
        return [CarSummary.model_validate(c) for c in cars]

    @app.get("/cars/{car_id}")
    async def get_car(request: Request, car_id: str) -> dict[str, Any]:
        found = await request.app.state.ctx.catalog.get([car_id])
        if not found:
            raise HTTPException(404, f"Unknown car_id: {car_id}")
        return {k: v for k, v in found[0].items() if k not in INTERNAL_CAR_FIELDS and not k.startswith("_")}

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> JSONResponse:
        names = list(checks)
        results = await asyncio.gather(*(_run_check(checks[n]) for n in names))
        report = dict(zip(names, results, strict=True))
        ok = all(r["ok"] for r in results)
        state = request.app.state
        try:
            n_cars = (await state.ctx.catalog.scope()).count
        except Exception:  # noqa: BLE001 - the cosmos check already reports the failure
            n_cars = 0
        body = HealthResponse(
            status="ok" if ok else "degraded",
            checks=report,
            checkpointer=type(state.graph.checkpointer).__name__,
            cars=n_cars,
            tracing=bool(state.tracing),
        )
        return JSONResponse(body.model_dump(), status_code=200 if ok else 503)

    @app.post("/feedback", status_code=204)
    async def feedback(request: Request, body: FeedbackRequest) -> Response:
        if not request.app.state.tracing:
            raise HTTPException(503, "Tracing is disabled; feedback cannot be recorded")
        try:
            await asyncio.to_thread(
                obs.log_user_feedback, body.trace_id, body.value, comment=body.comment, user=body.user or "anonymous"
            )
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "error_code", "")
            if code == "RESOURCE_DOES_NOT_EXIST":
                raise HTTPException(404, "Unknown trace_id") from exc
            log.warning("feedback failed for %s: %s", body.trace_id, type(exc).__name__)
            raise HTTPException(400, "Feedback could not be recorded") from exc
        return Response(status_code=204)

    return app


app = create_app()
