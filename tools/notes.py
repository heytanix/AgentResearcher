"""In-memory research notes acting as the agent's working memory for a session."""

_notes: list[dict] = []


def save_note(text: str, source_url: str) -> None:
    _notes.append({"text": text, "source_url": source_url})


def get_notes() -> list[dict]:
    return list(_notes)


def clear_notes() -> None:
    _notes.clear()


if __name__ == "__main__":
    save_note(
        "LLM agents use a ReAct loop of reasoning and acting.",
        "https://example.com/react-paper",
    )
    save_note(
        "Tool use lets agents call search, fetch, and other APIs mid-reasoning.",
        "https://example.com/agent-tools",
    )
    save_note(
        "Working memory (notes) persists context across loop steps.",
        "https://example.com/agent-memory",
    )
    for note in get_notes():
        print(note)
