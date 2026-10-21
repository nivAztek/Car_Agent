"""Session state for the Streamlit app: typed chat messages, the API client and conversation lifecycle.

The thread id is mirrored in the URL (`?thread=...`), so a page refresh resumes the conversation from the
server-side checkpoint (history + profile) instead of losing it.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import streamlit as st

from autochat.ui.client import ApiClient, ApiError

THREAD_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")  # same rule as the API


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    trace_id: str | None = None


def init(client_factory: Callable[[], ApiClient]) -> None:
    """Idempotent per rerun: create the client once per session and start or resume a conversation."""
    ss = st.session_state
    if "api" not in ss:
        ss.api = client_factory()
    if "thread_id" in ss:
        return
    requested = st.query_params.get("thread")
    if requested and THREAD_ID_RE.fullmatch(requested) and _resume(requested):
        return
    new_conversation()


def api() -> ApiClient:
    return st.session_state.api


def thread_id() -> str:
    return st.session_state.thread_id


def messages() -> list[Message]:
    return st.session_state.messages


def profile() -> dict[str, Any]:
    return st.session_state.profile


def set_profile(value: dict[str, Any]) -> None:
    st.session_state.profile = value


def new_conversation() -> None:
    _set_thread(str(uuid.uuid4()), [], {})


def cars() -> list[dict[str, Any]]:
    """Car catalog, fetched once per session (raises ApiError if the API is down)."""
    ss = st.session_state
    if "cars" not in ss:
        ss.cars = api().cars()
    return ss.cars


def queue_prompt(text: str) -> None:
    """Used by example-prompt buttons; consumed by `take_queued_prompt` on the next run."""
    st.session_state.queued_prompt = text


def take_queued_prompt() -> str | None:
    return st.session_state.pop("queued_prompt", None)


def _resume(tid: str) -> bool:
    try:
        history = api().history(tid)
        prof = api().profile(tid) if history["messages"] else {}
    except ApiError:
        return False
    if not history["messages"]:
        return False
    msgs = [Message(role=m["role"], content=m["content"]) for m in history["messages"]]
    _set_thread(tid, msgs, prof)
    return True


def _set_thread(tid: str, msgs: list[Message], prof: dict[str, Any]) -> None:
    ss = st.session_state
    ss.thread_id, ss.messages, ss.profile = tid, msgs, prof
    st.query_params["thread"] = tid
