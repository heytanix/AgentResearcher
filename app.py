"""Streamlit UI for AgentResearcher."""

import streamlit as st

from agent import run_research_loop
from tools.notes import clear_notes

st.set_page_config(page_title="AgentResearcher", page_icon="🔎", layout="wide")

with st.sidebar:
    st.header("How this works")
    st.markdown(
        "AgentResearcher is a **ReAct** agent: on each turn it *Reasons* about what to do "
        "next, then *Acts* by calling one tool, observes the result, and repeats — up to "
        "10 steps — until it decides it has enough to write a report."
    )

    st.subheader("The 4 tools")
    st.markdown(
        "- **web_search** — searches the web and returns a short list of results "
        "(title, url, snippet). Cheap and fast; used to discover sources.\n"
        "- **fetch_page** — fetches one exact URL and returns the full extracted page "
        "text. Used when a snippet isn't enough detail, e.g. for comparisons or deep "
        "technical explanations (the agent is required to fetch at least one full "
        "source before compiling a report on those kinds of topics).\n"
        "- **save_note** — records one finding plus its source URL to the agent's "
        "working memory for this session. Notes are the only material the final "
        "report is built from.\n"
        "- **compile_report** — compiles all saved notes into the final structured "
        "markdown report. Called once, at the end, and only after the agent has "
        "assessed whether its notes are strongly on-topic or only tangential."
    )

    st.caption(
        "Model: gemini-flash-lite-latest · Search/fetch: Tavily · "
        "Built with google-genai native function calling."
    )

st.title("🔎 AgentResearcher")
st.caption("Give it a topic. Watch it search, read, take notes, and write you a report.")

topic = st.text_input(
    "Research topic",
    placeholder="e.g. Compare the memory architectures of Mem0 and Letta in technical detail",
)
run_clicked = st.button("Research", type="primary", disabled=not topic.strip())

if run_clicked:
    clear_notes()

    st.subheader("Reasoning trace")
    trace_placeholder = st.empty()
    step_lines: list[str] = []

    def _on_step(step: dict) -> None:
        step_lines.append(
            f"**{len(step_lines) + 1}. `{step['tool']}`** "
            f"— args: `{step['args']}`\n\n"
            f"&nbsp;&nbsp;&nbsp;&nbsp;→ {step['summary']}"
        )
        trace_placeholder.markdown("\n\n".join(step_lines))

    with st.spinner("Researching..."):
        try:
            final_report, reasoning_trace = run_research_loop(
                topic.strip(), on_step=_on_step
            )
        except Exception as e:
            st.error(f"Research failed: {e}")
            final_report = None

    if final_report:
        st.subheader("Final report")
        st.markdown(final_report)
