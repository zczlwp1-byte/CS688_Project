"""
tools/scraper.py — Web mining layer.

Searches the web for news articles on a topic (DuckDuckGo, no API key needed)
and scrapes the article text with requests + BeautifulSoup.
"""

from __future__ import annotations

import requests
from bs4 import BeautifulSoup

try:
    from ddgs import DDGS  # package name changed from duckduckgo_search
except ImportError:
    from duckduckgo_search import DDGS

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/125.0 Safari/537.36")
}


def search_news(topic: str, max_results: int = 5) -> list[dict]:
    """Search DuckDuckGo News for a topic.

    Returns a list of {"title", "url", "date", "source", "excerpt"}.
    """
    results = []
    with DDGS() as ddgs:
        for r in ddgs.news(topic, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "date": r.get("date", ""),
                "source": r.get("source", ""),
                "excerpt": r.get("body", ""),
            })
    return results


def fetch_article(url: str, timeout: int = 15) -> str:
    """Download a web page and extract readable article text.

    Strategy: strip script/style/nav tags, then join all <p> paragraphs.
    Simple but works on most news sites.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        return f"[fetch error] {exc}"

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    # Drop very short fragments (menus, cookie banners, captions).
    paragraphs = [p for p in paragraphs if len(p) > 60]
    return "\n".join(paragraphs)


def collect_articles(topic: str, max_results: int = 5) -> list[dict]:
    """Search a topic, scrape each result, return articles with full text.

    Articles whose scrape failed or returned too little text keep the
    search-result excerpt as a fallback body.
    """
    articles = search_news(topic, max_results=max_results)
    for art in articles:
        text = fetch_article(art["url"])
        if text.startswith("[fetch error]") or len(text) < 200:
            text = art["excerpt"]  # fallback to the search snippet
        art["text"] = text
    return [a for a in articles if a.get("text")]
