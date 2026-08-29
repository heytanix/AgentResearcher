from tavily import TavilyClient

from tools.config import get_api_key

_client = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        api_key = get_api_key("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is not set")
        _client = TavilyClient(api_key=api_key)
    return _client


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web via Tavily and return top results.

    Each result dict has keys: title, url, content.
    """
    response = _get_client().search(query=query, max_results=max_results)
    results = []
    for item in response.get("results", []):
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            }
        )
    return results


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    results = web_search("LLM agent tool use")
    for r in results:
        print(r["title"])
        print(r["url"])
        print(r["content"][:200])
        print("---")
