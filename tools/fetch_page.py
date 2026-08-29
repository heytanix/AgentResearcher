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


def fetch_page(url: str) -> str:
    """Fetch a URL and return its full extracted page content."""
    response = _get_client().extract(urls=[url])
    results = response.get("results", [])
    if not results:
        failed = response.get("failed_results", [])
        if failed:
            error = failed[0].get("error", "unknown error")
            raise RuntimeError(f"Failed to fetch {url}: {error}")
        return ""
    return results[0].get("raw_content", "")


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    load_dotenv()

    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    content = fetch_page(test_url)
    print(content[:1000])
    print(f"\n[total length: {len(content)} chars]")
