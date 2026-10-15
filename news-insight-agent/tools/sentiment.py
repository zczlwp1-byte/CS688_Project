"""
tools/sentiment.py — Sentiment analysis + keyword extraction over articles.

Sentiment: VADER (lexicon-based, fast, no model download).
Keywords: simple TF scoring with stopword removal (transparent and dependency-free);
the LLM can also be asked for keywords via the agent for comparison.
"""

from __future__ import annotations

import re
from collections import Counter

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

_STOPWORDS = set("""
a an the and or but if then else for while of to in on at by with from as is are
was were be been being it its this that these those he she they we you i not no
yes his her their our your my me him them us do does did done have has had having
will would can could should may might must about into over under more most other
some such only own same so than too very just also there here when where why how
all any both each few once during before after above below up down out off again
said says new one two three
""".split())


def sentiment_score(text: str) -> dict:
    """VADER polarity scores for one text: {neg, neu, pos, compound}."""
    return _analyzer.polarity_scores(text)


def label(compound: float) -> str:
    """Standard VADER thresholds."""
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def analyze_articles(articles: list[dict]) -> list[dict]:
    """Score each article; returns [{title, compound, label}, ...]."""
    results = []
    for art in articles:
        score = sentiment_score(art["text"][:5000])  # cap length for speed
        results.append({
            "title": art.get("title", "untitled"),
            "compound": score["compound"],
            "label": label(score["compound"]),
        })
    return results


def summarize_sentiment(articles: list[dict]) -> str:
    """Human-readable one-paragraph summary the agent can return as text."""
    results = analyze_articles(articles)
    if not results:
        return "No articles to analyze."
    counts = Counter(r["label"] for r in results)
    avg = sum(r["compound"] for r in results) / len(results)
    lines = [
        f"Analyzed {len(results)} articles: "
        f"{counts.get('positive', 0)} positive, "
        f"{counts.get('neutral', 0)} neutral, "
        f"{counts.get('negative', 0)} negative. "
        f"Average compound score: {avg:+.3f} ({label(avg)})."
    ]
    for r in results:
        lines.append(f"- {r['label'].upper():8s} ({r['compound']:+.2f}) {r['title']}")
    return "\n".join(lines)


def extract_keywords(articles: list[dict], top_n: int = 15) -> list[tuple[str, int]]:
    """Top-N frequent content words across all article texts."""
    text = " ".join(a["text"] for a in articles).lower()
    words = re.findall(r"[a-z][a-z\-']{2,}", text)
    words = [w for w in words if w not in _STOPWORDS]
    return Counter(words).most_common(top_n)
