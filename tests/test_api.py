"""FastAPI tests: fake runtime (in-memory graph + offline fakes), no Cosmos/Foundry/MLflow."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fakes import ANSWER, CARS, fake_api, make_ctx
from fastapi.testclient import TestClient

from autochat.agent.state import AgentContext
from autochat.api import main as api_main
from autochat.api.main import _cached
from autochat.retrieval.catalog import InMemoryCarCatalog

THREAD = "t-api-1"


async def ok_check() -> dict[str, Any]:
    return {"detail": "fine"}


async def failing_check() -> dict[str, Any]:
    raise ConnectionError("down")


def make_client(ctx: AgentContext | None = None, **kw: Any) -> TestClient:
    kw.setdefault("health_checks", {"cosmos": ok_check})
    return fake_api(ctx, **kw)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with make_client() as c:
        yield c


def parse_sse(lines: Iterator[str]) -> list[tuple[str, Any]]:
    events: list[tuple[str, Any]] = []
    event = None
    for line in lines:
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and event:
            events.append((event, json.loads(line.split(":", 1)[1].strip())))
            event = None
    return events


# --------------------------------------------------------------------------- chat
def test_chat_sse_event_order_and_payloads(client: TestClient) -> None:
    body = {"thread_id": THREAD, "message": "I want an EV under 180k"}
    with client.stream("POST", "/chat", json=body) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(r.iter_lines())

    kinds = [k for k, _ in events]
    assert kinds[-3:] == ["sources", "profile", "done"]
    assert set(kinds[:-3]) == {"token"} and len(kinds) > 4
    assert "".join(d["t"] for k, d in events if k == "token").strip() == ANSWER
    sources = events[-3][1]
    assert sources and {s["car_id"] for s in sources} == {"ev-cheap"}
    assert events[-2][1]["budget_max_ils"] == 180_000
    done = events[-1][1]
    assert done["trace_id"] is None  # tracing disabled in tests
    assert done["intent"] == "recommend" and done["language"] == "en"
    assert any("rank_by_profile" in s for s in done["steps"])


def test_chat_invoke_then_thread_state(client: TestClient) -> None:
    r = client.post("/chat/invoke", json={"thread_id": THREAD, "message": "I want an EV under 180k"})
    assert r.status_code == 200
    out = r.json()
    assert out["answer"] == ANSWER and out["intent"] == "recommend" and out["language"] == "en"
    assert out["profile"]["powertrain_pref"] == ["ev"]
    assert any("rank_by_profile" in s for s in out["steps"])

    profile = client.get(f"/threads/{THREAD}/profile").json()
    assert profile["budget_max_ils"] == 180_000
    hist = client.get(f"/threads/{THREAD}/history").json()
    assert [m["role"] for m in hist["messages"]] == ["user", "assistant"]
    assert hist["messages"][1]["content"] == ANSWER


def test_unknown_thread_has_default_profile_and_empty_history(client: TestClient) -> None:
    assert client.get("/threads/never-seen/profile").json()["budget_max_ils"] is None
    assert client.get("/threads/never-seen/history").json()["messages"] == []


@pytest.mark.parametrize("thread_id", ["bad$id", "has space", "x" * 129, ""])
def test_chat_rejects_bad_thread_id(client: TestClient, thread_id: str) -> None:
    r = client.post("/chat/invoke", json={"thread_id": thread_id, "message": "hi"})
    assert r.status_code == 422


def test_bad_path_thread_id_rejected(client: TestClient) -> None:
    assert client.get("/threads/bad$id/profile").status_code == 422


def test_empty_message_rejected(client: TestClient) -> None:
    assert client.post("/chat", json={"thread_id": THREAD, "message": ""}).status_code == 422


def test_busy_thread_returns_409(client: TestClient) -> None:
    client.app.state.busy.add(THREAD)
    try:
        assert client.post("/chat", json={"thread_id": THREAD, "message": "hi"}).status_code == 409
        assert client.post("/chat/invoke", json={"thread_id": THREAD, "message": "hi"}).status_code == 409
    finally:
        client.app.state.busy.discard(THREAD)


def test_turn_failure_emits_error_event_and_frees_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(*_: Any, **__: Any) -> dict[str, Any]:
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(api_main, "run_turn", boom)
    with make_client() as c:
        with c.stream("POST", "/chat", json={"thread_id": THREAD, "message": "hi"}) as r:
            events = parse_sse(r.iter_lines())
        assert events == [("error", {"detail": api_main.GENERIC_ERROR})]
        r2 = c.post("/chat/invoke", json={"thread_id": THREAD, "message": "hi"})
        assert r2.status_code == 500 and "secret" not in r2.text
        assert c.app.state.busy == set()


# --------------------------------------------------------------------------- cars
def test_cars_list_and_detail(client: TestClient) -> None:
    r = client.get("/cars")
    cars = r.json()
    assert {c["car_id"] for c in cars} == set(CARS) and r.headers["X-Total-Count"] == str(len(CARS))
    assert [c["make"] for c in cars] == sorted(c["make"] for c in cars)
    car = client.get("/cars/ev-cheap").json()
    assert car["model"] == "S6" and car["aspect_scores"]["value"] == 5 and "name_keys" not in car
    assert client.get("/cars/nope").status_code == 404


def test_cars_paging_and_search(client: TestClient) -> None:
    page = client.get("/cars", params={"offset": 1, "limit": 2})
    assert len(page.json()) == 2 and page.headers["X-Total-Count"] == str(len(CARS))
    found = client.get("/cars", params={"q": "elantra"})
    assert [c["car_id"] for c in found.json()] == ["petrol-fun"] and found.headers["X-Total-Count"] == "1"
    assert client.get("/cars", params={"limit": 1000}).status_code == 422


def test_car_detail_hides_internal_fields() -> None:
    ctx = make_ctx([], [])
    car_x = {**CARS["ev-cheap"], "car_id": "x", "adapter": "a", "source_sha": "s", "dropped": [], "_rid": "r"}
    ctx.catalog = InMemoryCarCatalog([car_x])
    with make_client(ctx) as c:
        car = c.get("/cars/x").json()
    assert not {"adapter", "source_sha", "dropped", "_rid", "name_keys"} & set(car)


# --------------------------------------------------------------------------- health / feedback / cors
def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["checks"]["cosmos"] == {"ok": True, "detail": "fine"}
    assert body["checkpointer"] == "InMemorySaver" and body["cars"] == len(CARS) and body["tracing"] is False


def test_health_degraded_hides_error_message() -> None:
    with make_client(health_checks={"cosmos": ok_check, "foundry": failing_check}) as c:
        r = c.get("/health")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "degraded"
    assert body["checks"]["foundry"] == {"ok": False, "error": "ConnectionError"}


async def test_cached_check_reuses_success_only() -> None:
    calls = {"n": 0}

    async def flaky() -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError
        return {"n": calls["n"]}

    cached = _cached(flaky, ttl_s=60)
    with pytest.raises(ConnectionError):
        await cached()
    assert await cached() == {"n": 2}
    assert await cached() == {"n": 2, "cached": True}
    assert calls["n"] == 2


def test_feedback_requires_tracing(client: TestClient) -> None:
    r = client.post("/feedback", json={"trace_id": "tr-abc", "value": True})
    assert r.status_code == 503
    assert client.post("/feedback", json={"trace_id": "bad id", "value": True}).status_code == 422


def test_cors_allows_streamlit_origin(client: TestClient) -> None:
    r = client.options(
        "/chat",
        headers={"Origin": "http://localhost:8501", "Access-Control-Request-Method": "POST"},
    )
    assert r.headers.get("access-control-allow-origin") == "http://localhost:8501"
