"""RSS feed parser for financial news."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import feedparser

from src.config import NEWS_FEEDS


@dataclass
class NewsItem:
    source: str
    title: str
    url: str
    published: str
    summary: str


def fetch_feed(feed_url: str, source_name: str, limit: int = 20) -> list[NewsItem]:
    parsed = feedparser.parse(feed_url)
    items = []
    for entry in parsed.entries[:limit]:
        items.append(
            NewsItem(
                source=source_name,
                title=str(entry.get("title", "")).strip(),
                url=str(entry.get("link", "")),
                published=str(entry.get("published", "")),
                summary=str(entry.get("summary", ""))[:400],
            )
        )
    return items


def fetch_all_feeds(limit_per_feed: int = 15) -> list[NewsItem]:
    """Pull headlines from every configured feed and return them combined."""
    all_items = []
    for name, url in NEWS_FEEDS.items():
        try:
            all_items.extend(fetch_feed(url, name, limit=limit_per_feed))
        except Exception as e:
            print(f"  warning: failed to fetch {name}: {e}")
    # De-dup by title (some publishers cross-post)
    seen = set()
    deduped = []
    for item in all_items:
        if item.title and item.title not in seen:
            seen.add(item.title)
            deduped.append(item)
    return deduped
