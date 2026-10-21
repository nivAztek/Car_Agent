"""UI tests: SSE parsing, the API client against the FastAPI app on fakes, formatters, and a headless app run."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fakes import ANSWER, fake_api, make_ctx, recommend_analysis
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from autochat.ui import client as ui_client
from autochat.ui.client import ApiClient, ApiError, ChatStream, SSEEvent, ThreadBusyError, parse_sse
from autochat.ui.components import budget_label, car_line, linkify_citations, profile_rows

APP = Path(__file__).resolve().parents[1] / "src" / "autochat" / "ui" / "app.py"
THREAD = "t-ui-1"


@pytest.fixture
def api_http() -> Iterator[TestClient]:
    with fake_api(make_ctx([recommend_analysis()] * 3, [ANSWER] * 3)) as http:
        yield http


@pytest.fixture
def api(api_http: TestClient) -> ApiClient:
    return ApiClient(str(api_http.base_url), http=api_http)


# --------------------------------------------------------------------------- SSE + stream
def test_parse_sse_handles_pings_multiline_and_missing_trailing_blank() -> None:
    lines = [
        ": ping",
        "event: token",
        'data: {"t": "a"}',
        "",
        "event: sources",
        "data: [1,",
        "data: 2]",
        "",
        "data: {}",
    ]
    assert list(parse_sse(lines)) == [
        SSEEvent("token", {"t": "a"}),
        SSEEvent("sources", [1, 2]),
        SSEEvent("message", {}),
    ]


def test_chat_stream_collects_metadata_and_raises_on_error() -> None:
    events = [
        SSEEvent("token", {"t": "Hi "}),
        SSEEvent("token", {"t": "there"}),
        SSEEvent("sources", [{"n": 1}]),
        SSEEvent("profile", {"budget_max_ils": 1}),
        SSEEvent("done", {"trace_id": "tr-1", "intent": "chat", "language": "en", "steps": ["s"]}),
    ]
    stream = ChatStream(events)
    assert list(stream) == ["Hi ", "there"]
    assert (stream.answer, stream.trace_id, stream.steps, stream.done) == ("Hi there", "tr-1", ["s"], True)
    with pytest.raises(ApiError, match="boom"):
        list(ChatStream([SSEEvent("error", {"detail": "boom"})]))


# --------------------------------------------------------------------------- client <-> real API app (fakes)
def test_client_chat_roundtrip(api: ApiClient) -> None:
    stream = api.chat(THREAD, "I want an EV under 180k", user="test")
    assert "".join(stream).strip() == ANSWER
    assert stream.done and stream.intent == "recommend" and stream.language == "en"
    assert {s["car_id"] for s in stream.sources} == {"ev-cheap"}
    assert stream.profile["budget_max_ils"] == 180_000

    assert api.profile(THREAD)["powertrain_pref"] == ["ev"]
    assert [m["role"] for m in api.history(THREAD)["messages"]] == ["user", "assistant"]
    assert len(api.cars()) == 4


def test_client_errors(api: ApiClient, api_http: TestClient) -> None:
    api_http.app.state.busy.add(THREAD)
    with pytest.raises(ThreadBusyError):
        list(api.chat(THREAD, "hi"))
    with pytest.raises(ApiError) as exc:
        api.feedback("tr-abc", True)
    assert exc.value.status == 503  # tracing is off in the fake API


def test_client_unreachable() -> None:
    api = ApiClient("http://127.0.0.1:9")
    with pytest.raises(ApiError, match="unreachable"):
        api.cars()
    with pytest.raises(ApiError, match="unreachable"):
        list(api.chat(THREAD, "hi"))


# --------------------------------------------------------------------------- formatters
def test_linkify_citations_only_known_numbers() -> None:
    sources = [{"n": 1, "url": "https://x/1"}, {"n": 3, "url": "https://x/3"}]
    out = linkify_citations("A [1] B [2] C [3][1]", sources)
    assert out == "A [\\[1\\]](https://x/1) B [2] C [\\[3\\]](https://x/3)[\\[1\\]](https://x/1)"


def test_profile_rows_and_budget() -> None:
    assert budget_label({}) is None
    assert budget_label({"budget_max_ils": 200_000}) == "up to ₪200,000"
    assert budget_label({"budget_min_ils": 150_000, "budget_max_ils": 200_000}) == "₪150,000 – ₪200,000"
    rows = profile_rows(
        {"budget_max_ils": 200_000, "powertrain_pref": ["ev", "phev"], "home_charging": False, "cars_liked": ["mg-s6"]},
        {"mg-s6": "MG S6"},
    )
    assert rows == [
        ("Budget", ["up to ₪200,000"]),
        ("Powertrain", ["Electric", "Plug-in hybrid"]),
        ("Home charging", ["No"]),
        ("Liked", ["MG S6"]),
    ]


def test_car_line() -> None:
    car = {"make": "MG", "model": "S6", "powertrain": "ev", "price_ils": 168_888, "range_real_km": 430, "url": "u"}
    assert car_line(car) == "**MG S6** · Electric · ₪168,888 · 430 km real range · [review](u)"
    assert car_line({"make": "Kia", "model": "EV9"}) == "**Kia EV9** · price n/a"


# --------------------------------------------------------------------------- headless app
def run_app(monkeypatch: pytest.MonkeyPatch, api: ApiClient, **query: str) -> AppTest:
    monkeypatch.setattr(ui_client, "default_client", lambda: api)
    at = AppTest.from_file(str(APP), default_timeout=30)
    for k, v in query.items():
        at.query_params[k] = v
    return at.run()


def markdown_text(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_app_chat_turn_updates_history_and_profile(monkeypatch: pytest.MonkeyPatch, api: ApiClient) -> None:
    at = run_app(monkeypatch, api)
    assert not at.exception
    assert at.title[0].value == "autochat"
    assert len(at.button) > 3  # new conversation + example prompts

    at.chat_input[0].set_value("I want an EV under 180k").run()
    assert not at.exception and not at.error
    msgs = at.session_state["messages"]
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[1].content.strip() == ANSWER and msgs[1].sources
    assert "[\\[1\\]](https://x/ev-cheap)" in markdown_text(at)  # citation linked to its source
    assert "up to ₪180,000" in markdown_text(at)  # sidebar profile chip


def test_app_resumes_thread_from_url(monkeypatch: pytest.MonkeyPatch, api: ApiClient) -> None:
    list(api.chat(THREAD, "I want an EV under 180k"))
    at = run_app(monkeypatch, api, thread=THREAD)
    assert not at.exception
    assert at.session_state["thread_id"] == THREAD
    assert [m.role for m in at.session_state["messages"]] == ["user", "assistant"]
    assert at.session_state["profile"]["budget_max_ils"] == 180_000


def test_app_shows_error_when_api_down(monkeypatch: pytest.MonkeyPatch) -> None:
    at = run_app(monkeypatch, ApiClient("http://127.0.0.1:9"))
    assert not at.exception
    assert "dev.ps1 api" in at.error[0].value
