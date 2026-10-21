"""Rendering helpers. Pure formatters (tested) + thin Streamlit render functions; no API calls here."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import streamlit as st

from autochat.ui.state import Message

# Mixed Hebrew/English answers: every paragraph, list item and table cell picks its own direction from its first
# strong character (like dir="auto", but per block, and it works while tokens stream). Static CSS only:
# model text is rendered as plain Markdown, never as raw HTML.
BIDI_CSS = """
<style>
[data-testid="stChatMessage"] :is(p, li, h1, h2, h3, h4, td, th, blockquote),
[data-testid="stSidebar"] :is(p, li),
[data-testid="stChatInput"] textarea {
  unicode-bidi: plaintext;
  text-align: start;
}
</style>
"""

EXAMPLE_PROMPTS = [
    "We're a family of 5 with a budget of about ₪200k. Which car fits us?",
    "Compare the Genesis GV80 and the Kia EV9 for long family trips",
    "Which car has the quickest 0-100 km/h?",
    "מה המחיר של סיטרואן C3 ומה החסרונות שלה?",
    "איך הנוחות של MG S6 בנסיעה בעיר?",
]

POWERTRAIN_LABELS = {
    "ev": "Electric",
    "ice": "Petrol",
    "hybrid": "Hybrid",
    "phev": "Plug-in hybrid",
    "mhev": "Mild hybrid",
}

_CITATION = re.compile(r"\[(\d+)\]")


# --------------------------------------------------------------------------- pure formatters
def format_ils(value: int | None) -> str:
    return f"₪{value:,}" if value is not None else "n/a"


def humanize(token: str) -> str:
    return POWERTRAIN_LABELS.get(token) or token.replace("_", " ").capitalize()


def linkify_citations(text: str, sources: list[dict[str, Any]]) -> str:
    """`[3]` -> `[[3]](url)` for citations that map to a source; unknown numbers are left untouched."""
    urls = {s["n"]: s["url"] for s in sources if s.get("url")}
    return _CITATION.sub(lambda m: f"[\\[{m[1]}\\]]({urls[int(m[1])]})" if int(m[1]) in urls else m[0], text)


def budget_label(profile: dict[str, Any]) -> str | None:
    lo, hi = profile.get("budget_min_ils"), profile.get("budget_max_ils")
    if lo and hi:
        return f"{format_ils(lo)} – {format_ils(hi)}"
    if hi:
        return f"up to {format_ils(hi)}"
    if lo:
        return f"from {format_ils(lo)}"
    return None


def profile_rows(profile: dict[str, Any], car_names: dict[str, str]) -> list[tuple[str, list[str]]]:
    """Inferred preferences as (label, chips) rows, skipping empty fields."""
    charging = profile.get("home_charging")
    rows: list[tuple[str, list[str]]] = [
        ("Budget", [b] if (b := budget_label(profile)) else []),
        ("Passengers", [str(profile["passengers"])] if profile.get("passengers") else []),
        ("Use", [humanize(x) for x in profile.get("use_cases", [])]),
        ("Powertrain", [humanize(x) for x in profile.get("powertrain_pref", [])]),
        ("Body", [humanize(x) for x in profile.get("body_pref", [])]),
        ("Priorities", [humanize(x) for x in profile.get("priorities", [])]),
        ("Must have", list(profile.get("must_haves", []))),
        ("Deal-breakers", list(profile.get("dealbreakers", []))),
        ("Home charging", [] if charging is None else ["Yes" if charging else "No"]),
        ("Liked", [car_names.get(c, c) for c in profile.get("cars_liked", [])]),
        ("Rejected", [car_names.get(c, c) for c in profile.get("cars_rejected", [])]),
    ]
    return [(label, chips) for label, chips in rows if chips]


def car_line(car: dict[str, Any]) -> str:
    parts = [f"**{car['make']} {car['model']}**"]
    if car.get("powertrain"):
        parts.append(humanize(car["powertrain"]))
    parts.append(format_ils(car.get("price_ils")) if car.get("price_ils") else "price n/a")
    if car.get("range_real_km"):
        parts.append(f"{car['range_real_km']} km real range")
    if car.get("url"):
        parts.append(f"[review]({car['url']})")
    return " · ".join(parts)


def _chip(text: str) -> str:
    return ":blue-badge[" + re.sub(r"[\[\]]", "", text) + "]"


# --------------------------------------------------------------------------- Streamlit renderers
def inject_css() -> None:
    st.markdown(BIDI_CSS, unsafe_allow_html=True)


def render_message(
    msg: Message,
    *,
    feedback_key: str | None = None,
    on_feedback: Callable[[str, Message], None] | None = None,
) -> None:
    with st.chat_message(msg.role):
        if msg.role == "user":
            st.markdown(msg.content)
            return
        st.markdown(linkify_citations(msg.content, msg.sources))
        render_turn_details(msg)
        if msg.trace_id and feedback_key and on_feedback:
            st.feedback("thumbs", key=feedback_key, on_change=on_feedback, args=(feedback_key, msg))


def render_turn_details(msg: Message) -> None:
    if msg.sources:
        with st.expander(f"Sources ({len(msg.sources)})", icon=":material/menu_book:"):
            for s in msg.sources:
                st.markdown(f"[{s['n']}] **{s['car']}** — [{s['heading']}]({s['url']})")
    if msg.steps:
        with st.expander("How I answered", icon=":material/route:"):
            for step in msg.steps:
                st.text(step)


def render_examples(on_pick: Callable[[str], None]) -> None:
    st.caption("Try one of these, or ask anything about the reviewed cars:")
    for i, text in enumerate(EXAMPLE_PROMPTS):
        st.button(text, key=f"example-{i}", on_click=on_pick, args=(text,))


def render_sidebar(
    *,
    profile: dict[str, Any],
    cars: list[dict[str, Any]],
    thread_id: str,
    on_new_conversation: Callable[[], None],
) -> None:
    car_names = {c["car_id"]: f"{c['make']} {c['model']}" for c in cars}
    st.button("New conversation", icon=":material/add_comment:", on_click=on_new_conversation, width="stretch")
    st.subheader("What I know about you")
    rows = profile_rows(profile, car_names)
    if not rows:
        st.caption("Tell me your budget, who rides with you and how you drive, and I'll keep track here.")
    for label, chips in rows:
        st.markdown(f"**{label}:** " + " ".join(_chip(c) for c in chips))
    with st.expander(f"Reviewed cars ({len(cars)})", icon=":material/directions_car:"):
        for car in cars:
            st.markdown(car_line(car))
    st.caption(f"Conversation `{thread_id[:8]}`")
