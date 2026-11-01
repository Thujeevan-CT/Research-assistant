from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse, urlunparse


def normalize_url(url: str) -> str:
    url = url.strip()
    try:
        p = urlparse(url)
    except ValueError:
        return url.lower().rstrip("/")
    host = (p.hostname or "").lower()
    path = p.path.rstrip("/") or ""
    return urlunparse((p.scheme.lower(), host, path, "", "", ""))


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@dataclass
class PageEvidence:
    title: str = ""
    texts: list[str] = field(default_factory=list)
    fetched: bool = False


class EvidenceStore:
    def __init__(self) -> None:
        self._pages: dict[str, PageEvidence] = {}

    def ingest_tool_result(self, tool_name: str, result_str: str) -> None:
        if tool_name == "web_search":
            try:
                data = json.loads(result_str)
            except json.JSONDecodeError:
                return
            if not isinstance(data, dict):
                return
            for item in data.get("results", []):
                url = normalize_url(item.get("url") or "")
                if not url:
                    continue
                page = self._pages.setdefault(url, PageEvidence())
                page.title = item.get("title") or page.title
                snippet = item.get("snippet") or ""
                if snippet:
                    page.texts.append(snippet)
            return

        if tool_name == "fetch_page":
            try:
                data = json.loads(result_str)
            except json.JSONDecodeError:
                return
            if not isinstance(data, dict) or "text" not in data:
                return
            url = normalize_url(data.get("url") or "")
            if not url:
                return
            page = self._pages.setdefault(url, PageEvidence())
            page.title = data.get("title") or page.title
            page.texts.append(data.get("text") or "")
            page.fetched = True

    def has_url(self, url: str) -> bool:
        return normalize_url(url) in self._pages

    def corpus(self, url: str) -> str:
        page = self._pages.get(normalize_url(url))
        if not page:
            return ""
        return "\n".join(page.texts)

    def title_for(self, url: str) -> str:
        page = self._pages.get(normalize_url(url))
        if not page or not page.title:
            return url
        return page.title

    def fetched_urls(self) -> list[str]:
        return [u for u, p in self._pages.items() if p.fetched]

    def verify_snippet(self, url: str, snippet: str) -> tuple[bool, str]:
        key = normalize_url(url)
        if key not in self._pages:
            return False, "url not in retrieved evidence"
        snippet_n = _normalize_text(snippet)
        if len(snippet_n) < 12:
            return False, "snippet too short"
        corpus_n = _normalize_text(self.corpus(url))
        if snippet_n in corpus_n:
            return True, "exact substring match"
        # tolerate minor whitespace differences in long snippets
        words = snippet_n.split()
        if len(words) >= 4:
            probe = " ".join(words[: min(8, len(words))])
            if probe in corpus_n:
                return True, "partial substring match"
        return False, "snippet not found in text for this url"
