"""Core ReAct loop for AgentResearcher."""

import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from tools.config import get_api_key
from tools.fetch_page import fetch_page
from tools.notes import get_notes, save_note
from tools.report import compile_report
from tools.web_search import web_search

load_dotenv()

# gemini-flash-lite-latest is a deliberate choice, not a placeholder: gemini-3.6-flash's
# free-tier quota (20 requests/day) is exhausted by a single research loop or two, which
# makes it unusable for a public-facing demo. The lite model has a separate, higher quota
# and is fast enough for this tool-calling workload.
MODEL_NAME = "gemini-flash-lite-latest"

TECHNICAL_TOPIC_KEYWORDS = [
    "compare",
    "comparison",
    " vs ",
    " vs. ",
    "versus",
    "how does",
    "how do",
    "architecture",
    "deep dive",
    "in technical detail",
    "internals",
    "under the hood",
]


def topic_requires_fetch(topic: str) -> bool:
    """Detect via keyword matching whether a topic demands a comparison or deep technical
    explanation, for which search snippets alone are considered insufficient."""
    topic_lower = f" {topic.lower()} "
    return any(keyword in topic_lower for keyword in TECHNICAL_TOPIC_KEYWORDS)


SYSTEM_INSTRUCTION = """You are a research agent. Given a topic, research it step by step:

1. Use web_search to find relevant sources for the topic (and for follow-up sub-questions
   that come up as you learn more).
2. When a search result looks especially promising, use fetch_page to read the full page
   content rather than relying on the short snippet.
3. Whenever you learn a concrete, useful fact, use save_note to record it as a short,
   self-contained finding, along with the exact source_url it came from.
4. Once you have gathered enough notes to cover the topic well, call compile_report exactly
   once to produce the final markdown report. Do not call compile_report until you have
   saved several notes.

If the topic asks for a detailed comparison, a deep technical explanation, or "how does X
work", search snippets alone are NOT sufficient. You must call fetch_page on at least one
full source before saving your notes on that comparison/explanation and calling
compile_report — snippets are too shallow for that level of detail. This is enforced: if
you call compile_report on such a topic without having called fetch_page first, the call
will be rejected and you will be told to fetch a source before trying again.

Before calling compile_report, assess how well your notes actually match the topic:
- If your notes are strongly and directly on-topic, call compile_report with no
  coverage_note (or an empty one).
- If your notes are only tangentially related, sparse, or come from sources that don't
  directly address the exact topic (e.g. you had to broaden the search or settle for
  adjacent subject matter), you MUST pass a coverage_note to compile_report that plainly
  says so, e.g. "Limited direct research exists on this exact topic; the following are the
  closest related sources found." Do not present tangential findings as if they were a
  confident, direct match for the topic.

Be efficient: don't search redundantly, and don't fetch pages that don't add new information.
"""

FUNCTION_DECLARATIONS = [
    types.FunctionDeclaration(
        name="web_search",
        description=(
            "Search the web for a query and get back a short list of results, each with a "
            "title, url, and a short content snippet. Use this to discover sources on the "
            "topic or to answer a specific sub-question. Does NOT return full page text — "
            "use fetch_page on a promising url for that."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
            },
            "required": ["query"],
        },
    ),
    types.FunctionDeclaration(
        name="fetch_page",
        description=(
            "Fetch a single web page by its exact URL and return its full extracted text "
            "content. Use this only after web_search, on a url from the search results, when "
            "the short snippet isn't enough detail. Do not invent URLs."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The exact URL to fetch, taken from prior web_search results.",
                },
            },
            "required": ["url"],
        },
    ),
    types.FunctionDeclaration(
        name="save_note",
        description=(
            "Save one concrete research finding to the agent's working memory (notes list) "
            "for later use in the final report. Call this every time you learn something "
            "worth keeping. This does not search or fetch anything, it only records text "
            "and a source you already have."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "A short, self-contained finding.",
                },
                "source_url": {
                    "type": "string",
                    "description": (
                        "The exact URL this finding came from (from a prior web_search or "
                        "fetch_page result). Required — every finding must be traceable to a "
                        "source."
                    ),
                },
            },
            "required": ["text", "source_url"],
        },
    ),
    types.FunctionDeclaration(
        name="compile_report",
        description=(
            "Compile all saved notes into the final structured markdown research report. "
            "Call this exactly ONCE, only after you have saved enough notes to thoroughly "
            "cover the topic. This ends the research loop."
        ),
        parameters={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The research topic the report is about.",
                },
                "coverage_note": {
                    "type": "string",
                    "description": (
                        "Leave empty if the saved notes are strongly and directly on-topic. "
                        "If the notes are only tangentially related, sparse, or come from "
                        "sources that don't directly address the exact topic, state that "
                        "plainly here, e.g. 'Limited direct research exists on this exact "
                        "topic; the following are the closest related sources found.'"
                    ),
                },
            },
            "required": ["topic"],
        },
    ),
]

TOOLS = [types.Tool(function_declarations=FUNCTION_DECLARATIONS)]


def get_client() -> genai.Client:
    api_key = get_api_key("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=api_key)


def _generate_with_retry(
    client: genai.Client, contents: list, config: types.GenerateContentConfig, max_retries: int = 5
):
    """Call generate_content, retrying on rate-limit (429) errors with backoff."""
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=MODEL_NAME, contents=contents, config=config
            )
        except errors.ClientError as e:
            if e.code == 429 and attempt < max_retries - 1:
                delay = 30
                details = getattr(e, "details", None) or {}
                for d in details.get("error", {}).get("details", []):
                    if d.get("@type", "").endswith("RetryInfo"):
                        retry_delay = d.get("retryDelay", "")
                        if retry_delay.endswith("s"):
                            delay = float(retry_delay[:-1]) + 2
                time.sleep(delay)
                continue
            raise


def _missing_args_error(name: str, missing: list[str]) -> tuple[dict, str, bool]:
    summary = f"rejected: {name} call was missing required arg(s) {missing}"
    return (
        {
            "status": "rejected",
            "reason": (
                f"Your {name} call was missing required argument(s): {', '.join(missing)}. "
                "Call it again with all required arguments filled in."
            ),
        },
        summary,
        False,
    )


def _execute_tool(name: str, args: dict, topic: str, has_fetched: bool) -> tuple[dict, str, bool]:
    """Execute a tool call and return (response_payload, short_summary, is_final).

    is_final is True only when compile_report actually ran and the loop should stop.
    A tool call missing required arguments (occasionally emitted malformed by the model)
    is reported back to the model as a rejection rather than crashing the loop.
    """
    if name == "web_search":
        if not args.get("query"):
            return _missing_args_error(name, ["query"])
        results = web_search(args["query"])
        summary = f"{len(results)} result(s) for '{args['query']}'"
        return {"results": results}, summary, False

    if name == "fetch_page":
        if not args.get("url"):
            return _missing_args_error(name, ["url"])
        content = fetch_page(args["url"])
        summary = f"fetched {len(content)} chars from {args['url']}"
        return {"content": content}, summary, False

    if name == "save_note":
        missing = [f for f in ("text", "source_url") if not args.get(f)]
        if missing:
            return _missing_args_error(name, missing)
        save_note(args["text"], args["source_url"])
        summary = f"saved note: {args['text'][:80]}"
        return {"status": "saved", "total_notes": len(get_notes())}, summary, False

    if name == "compile_report":
        if not args.get("topic"):
            return _missing_args_error(name, ["topic"])
        report_topic = args.get("topic", topic)

        if topic_requires_fetch(report_topic) and not has_fetched:
            summary = "rejected: topic needs fetch_page before compiling"
            return (
                {
                    "status": "rejected",
                    "reason": (
                        "This topic asks for a comparison or deep technical explanation, "
                        "which requires fetch_page on at least one full source before "
                        "compiling the report. Call fetch_page on a promising URL from your "
                        "prior search results, then try compile_report again."
                    ),
                },
                summary,
                False,
            )

        coverage_note = args.get("coverage_note", "")
        report = compile_report(get_notes(), report_topic, coverage_note=coverage_note)
        summary = f"compiled report ({len(get_notes())} notes)"
        if coverage_note:
            summary += " [flagged tangential coverage]"
        return {"report": report}, summary, True

    raise ValueError(f"Unknown tool: {name}")


def run_research_loop(
    topic: str, max_steps: int = 10, on_step=None
) -> tuple[str, list[dict]]:
    """Run the ReAct loop: reason, call tools (search/fetch/notes), then compile a report.

    If given, on_step(step_dict) is called immediately after each tool call executes,
    so callers (e.g. a UI) can stream progress instead of waiting for the final result.

    Returns (final_report, trace) where trace is a list of step dicts, each with keys
    "tool", "args", and "summary".
    """
    client = get_client()
    trace: list[dict] = []
    has_fetched = False
    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=f"Research topic: {topic}")])
    ]
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=TOOLS,
    )

    for _ in range(max_steps):
        response = _generate_with_retry(client, contents, config)
        candidate = response.candidates[0]
        contents.append(candidate.content)

        function_calls = [
            part.function_call for part in candidate.content.parts if part.function_call
        ]

        if not function_calls:
            # Model responded with plain text instead of a tool call; nudge it to continue
            # or stop if nothing more can be done.
            break

        response_parts = []
        for call in function_calls:
            name = call.name
            args = dict(call.args or {})
            result_payload, summary, is_final = _execute_tool(name, args, topic, has_fetched)

            if name == "fetch_page":
                has_fetched = True

            step = {"tool": name, "args": args, "summary": summary}
            trace.append(step)
            if on_step:
                on_step(step)

            response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=name,
                        response=result_payload,
                    )
                )
            )

            if is_final:
                return result_payload["report"], trace

        contents.append(types.Content(role="user", parts=response_parts))

    # Ran out of steps without an explicit compile_report call; compile from whatever
    # notes were gathered so the loop still produces a usable report.
    report = compile_report(get_notes(), topic)
    final_step = {
        "tool": "compile_report",
        "args": {"topic": topic},
        "summary": f"compiled report after hitting max_steps ({len(get_notes())} notes)",
    }
    trace.append(final_step)
    if on_step:
        on_step(final_step)
    return report, trace


if __name__ == "__main__":
    test_topic = "recent approaches to LLM agent memory"

    print(f"Researching: {test_topic}\n")
    print("=== Reasoning trace ===")

    step_count = 0

    def _print_step(step: dict) -> None:
        global step_count
        step_count += 1
        print(f"{step_count}. [{step['tool']}] args={step['args']}")
        print(f"   -> {step['summary']}")

    final_report, reasoning_trace = run_research_loop(test_topic, on_step=_print_step)

    print("\n=== Final report ===\n")
    print(final_report)
