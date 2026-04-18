"""LangChain tools for article agents."""

from duckduckgo_search import DDGS
from langchain_core.tools import tool


@tool("internet_search_DDGO", return_direct=False)
def internet_search_DDGO(query: str) -> str:
    """
    Searches the internet using DuckDuckGo.
    Args:
        query: Search query

    Returns:
        Search result
    """

    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=5)]
    if not results:
        return "No results found."
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        href = r.get("href", "")
        body = r.get("body", "")
        lines.append(f"{i}. {title}\n   {href}\n   {body}")
    return "\n\n".join(lines)
