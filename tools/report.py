def compile_report(notes: list[dict], topic: str, coverage_note: str = "") -> str:
    """Format accumulated notes into a structured markdown report with a sources list.

    Each note is a dict with "text" and "source_url" keys (see tools.notes.save_note).
    coverage_note is an optional caveat (e.g. "sources are tangential, not direct hits")
    that gets rendered as a "Coverage note" callout before the Summary section.
    """
    sources: list[str] = []
    for note in notes:
        url = note.get("source_url", "")
        if url and url not in sources:
            sources.append(url)

    lines = [f"# Research Report: {topic}", ""]

    if coverage_note:
        lines += ["## Coverage note", "", coverage_note, ""]

    lines += ["## Summary", ""]

    if notes:
        lines.append(
            f"This report compiles {len(notes)} research note(s) gathered while investigating {topic}."
        )
    else:
        lines.append(f"No notes were gathered while investigating {topic}.")

    lines += ["", "## Findings", ""]
    if notes:
        for note in notes:
            source_suffix = f" (Source: {note['source_url']})" if note.get("source_url") else ""
            lines.append(f"- {note['text']}{source_suffix}")
    else:
        lines.append("- No findings recorded.")

    lines += ["", "## Sources", ""]
    if sources:
        for i, url in enumerate(sources, 1):
            lines.append(f"{i}. {url}")
    else:
        lines.append("No sources recorded.")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    fake_notes = [
        {
            "text": "LLM agents combine reasoning and tool use in a loop (ReAct pattern).",
            "source_url": "https://example.com/react-paper",
        },
        {
            "text": "Popular tools for agents include web search, code execution, and file access.",
            "source_url": "https://example.com/agent-tools",
        },
        {
            "text": "Gemini and Claude both support native function calling for tool use.",
            "source_url": "https://example.com/function-calling",
        },
    ]
    print(compile_report(fake_notes, "LLM Agent Tool Use"))
    print(
        compile_report(
            fake_notes,
            "Underwater Basket Weaving in LLM Agents",
            coverage_note=(
                "Limited direct research exists on this exact topic; the following are "
                "the closest related sources found."
            ),
        )
    )
