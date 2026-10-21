"""HTTP client for the autochat API (sync httpx; no Streamlit imports, so it is unit-testable and reusable)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx

STREAM_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0)  # read = max gap between SSE events
REQUEST_TIMEOUT = httpx.Timeout(15.0)


class ApiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class ThreadBusyError(ApiError):
    """409: a previous turn on this thread is still running (e.g. the page was interrupted mid-answer)."""


@dataclass(frozen=True)
class SSEEvent:
    event: str
    data: Any


def parse_sse(lines: Iterable[str]) -> Iterator[SSEEvent]:
    """Minimal SSE parser: `event:` + one or more `data:` lines, dispatched on a blank line; `:` lines are pings."""
    event, data = "message", []
    for line in lines:
        if not line:
            if data:
                yield SSEEvent(event, json.loads("\n".join(data)))
            event, data = "message", []
        elif line.startswith(":"):
            continue
        else:
            name, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if name == "event":
                event = value
            elif name == "data":
                data.append(value)
    if data:  # stream closed without a trailing blank line
        yield SSEEvent(event, json.loads("\n".join(data)))


@dataclass
class ChatStream:
    """Iterate to get answer tokens (feed it to `st.write_stream`); afterwards the turn metadata is filled in."""

    events: Iterable[SSEEvent]
    answer: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    profile: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None
    intent: str | None = None
    language: str | None = None
    steps: list[str] = field(default_factory=list)
    done: bool = False

    def __iter__(self) -> Iterator[str]:
        for ev in self.events:
            if ev.event == "token":
                token = ev.data.get("t", "")
                self.answer += token
                yield token
            elif ev.event == "sources":
                self.sources = ev.data
            elif ev.event == "profile":
                self.profile = ev.data
            elif ev.event == "done":
                self.done = True
                self.trace_id = ev.data.get("trace_id")
                self.intent = ev.data.get("intent")
                self.language = ev.data.get("language")
                self.steps = ev.data.get("steps") or []
            elif ev.event == "error":
                raise ApiError(ev.data.get("detail", "The assistant failed to answer."))


class ApiClient:
    """`http` lets tests inject e.g. FastAPI's `TestClient` or an `httpx.MockTransport` client."""

    def __init__(self, base_url: str, *, http: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = http or httpx.Client(base_url=self.base_url, timeout=REQUEST_TIMEOUT)

    # ------------------------------------------------------------------ chat
    def chat(self, thread_id: str, message: str, *, user: str | None = None) -> ChatStream:
        return ChatStream(self._stream_events({"thread_id": thread_id, "message": message, "user": user}))

    def _stream_events(self, body: dict[str, Any]) -> Iterator[SSEEvent]:
        try:
            with self._http.stream("POST", "/chat", json=body, timeout=STREAM_TIMEOUT) as resp:
                if resp.status_code != 200:
                    resp.read()
                    _raise_for(resp)
                yield from parse_sse(resp.iter_lines())
        except httpx.HTTPError as exc:
            raise ApiError(f"API unreachable at {self.base_url} ({type(exc).__name__})") from exc

    # ------------------------------------------------------------------ reads
    def cars(self) -> list[dict[str, Any]]:
        return self._get("/cars")

    def profile(self, thread_id: str) -> dict[str, Any]:
        return self._get(f"/threads/{thread_id}/profile")

    def history(self, thread_id: str) -> dict[str, Any]:
        return self._get(f"/threads/{thread_id}/history")

    def feedback(self, trace_id: str, value: bool, *, comment: str | None = None) -> None:
        self._request("POST", "/feedback", json={"trace_id": trace_id, "value": value, "comment": comment})

    def _get(self, path: str) -> Any:
        return self._request("GET", path).json()

    def _request(self, method: str, path: str, **kw: Any) -> httpx.Response:
        try:
            resp = self._http.request(method, path, **kw)
        except httpx.HTTPError as exc:
            raise ApiError(f"API unreachable at {self.base_url} ({type(exc).__name__})") from exc
        _raise_for(resp)
        return resp


def _raise_for(resp: httpx.Response) -> None:
    if resp.is_success:
        return
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        detail = resp.text
    cls = ThreadBusyError if resp.status_code == 409 else ApiError
    raise cls(str(detail), status=resp.status_code)


def default_client() -> ApiClient:
    """Client for `API_BASE_URL` (settings). Tests monkeypatch this factory."""
    from autochat.config import get_settings

    return ApiClient(get_settings().api_base_url)
