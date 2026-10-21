"""autochat Streamlit app: `scripts/dev.ps1 ui` (needs the API: `scripts/dev.ps1 api`).

Flow per run: init session -> sidebar -> history -> (new prompt? stream the answer, store it, rerun).
"""

from __future__ import annotations

from collections.abc import Iterator

import streamlit as st

from autochat.ui import client, components, state
from autochat.ui.client import ApiError, ThreadBusyError
from autochat.ui.state import Message


def send_feedback(widget_key: str, msg: Message) -> None:
    value = st.session_state.get(widget_key)
    if value is None or not msg.trace_id:
        return
    try:
        state.api().feedback(msg.trace_id, value == 1)
        st.toast("Thanks for the feedback!", icon=":material/thumb_up:")
    except ApiError as exc:
        st.toast(f"Feedback was not recorded: {exc}", icon=":material/error:")


def answer(prompt: str) -> None:
    """Stream one turn into an assistant bubble; on success store it and rerun (refreshes sidebar + feedback)."""
    msgs = state.messages()
    msgs.append(Message("user", prompt))
    components.render_message(msgs[-1])
    with st.chat_message("assistant"):
        stream = state.api().chat(state.thread_id(), prompt, user="ui")
        try:
            with st.spinner("Reading the reviews...", show_time=True):
                tokens = iter(stream)
                first = next(tokens, "")  # planner + retrieval happen before the first token
            st.write_stream(_prepend(first, tokens))
        except ThreadBusyError:
            msgs.pop()
            st.warning("Still finishing the previous answer. Please try again in a few seconds.")
            return
        except ApiError as exc:
            msgs.pop()
            st.error(str(exc))
            return
    msgs.append(Message("assistant", stream.answer, stream.sources, stream.steps, stream.trace_id))
    state.set_profile(stream.profile)
    st.rerun()


def _prepend(first: str, rest: Iterator[str]) -> Iterator[str]:
    if first:
        yield first
    yield from rest


def main() -> None:
    st.set_page_config(page_title="autochat · car reviews", page_icon=":material/directions_car:")
    components.inject_css()
    state.init(client.default_client)

    try:
        cars = state.cars()
    except ApiError as exc:
        st.error(f"{exc}. Start the API with `scripts/dev.ps1 api`, then reload this page.")
        st.stop()

    with st.sidebar:
        components.render_sidebar(
            profile=state.profile(),
            cars=cars,
            thread_id=state.thread_id(),
            on_new_conversation=state.new_conversation,
        )

    st.title("autochat")
    st.caption(
        f"Ask about the {len(cars)} cars reviewed on auto.co.il, in English or Hebrew. Answers cite the reviews."
    )

    thread = state.thread_id()
    for i, msg in enumerate(state.messages()):
        components.render_message(msg, feedback_key=f"fb-{thread}-{i}", on_feedback=send_feedback)

    prompt = st.chat_input("Ask about the reviewed cars...") or state.take_queued_prompt()
    if prompt:
        answer(prompt)
    elif not state.messages():
        components.render_examples(on_pick=state.queue_prompt)


main()
