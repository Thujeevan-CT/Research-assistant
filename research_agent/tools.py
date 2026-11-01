from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

import httpx
from google.genai import types

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MAX_RESULTS = 5
FETCH_TIMEOUT = 15.0
FETCH_MAX_BYTES = 1_000_000
FETCH_MAX_TEXT = 8_000
FETCH_MAX_REDIRECTS = 5
ALLOWED_CONTENT_TYPES = (
    "text/html",
    "text/plain",
    "application/xhtml+xml",
)

WEB_SEARCH_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "A specific search query.",
        }
    },
    "required": ["query"],
}

FETCH_PAGE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "Absolute http or https URL to read.",
        }
    },
    "required": ["url"],
}

WEB_SEARCH_DECLARATION = types.FunctionDeclaration(
    name="web_search",
    description=(
        "Search the web. Returns up to 5 results with title, url, and snippet."
    ),
    parameters=WEB_SEARCH_PARAMETERS,
)

FETCH_PAGE_DECLARATION = types.FunctionDeclaration(
    name="fetch_page",
    description=(
        "Download a public http(s) page and return plain text, truncated. "
        "Use when a search snippet is too thin for fees, dates, or rules — "
        "prefer primary sources from search results."
    ),
    parameters=FETCH_PAGE_PARAMETERS,
)

RESEARCH_TOOLS = [
    types.Tool(
        function_declarations=[WEB_SEARCH_DECLARATION, FETCH_PAGE_DECLARATION]
    )
]


class _HTMLTextExtractor(HTMLParser):
    _SKIP = frozenset({"script", "style", "noscript"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in ("p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr", "section"):
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        raw = re.sub(r"[ \t]+\n", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _validate_http_url(url: str) -> str | None:
    url = url.strip()
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return None
    if not parsed.netloc:
        return None
    return url


def _extract_title(html: str) -> str:
    match = re.search(
        r"<title[^>]*>(.*?)</title>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ""
    title = re.sub(r"\s+", " ", match.group(1)).strip()
    return title


def _html_to_text(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


def _truncate_text(text: str) -> str:
    if len(text) <= FETCH_MAX_TEXT:
        return text
    return text[:FETCH_MAX_TEXT] + "\n[truncated]"


def _content_type_ok(content_type: str | None) -> bool:
    if not content_type:
        return True
    base = content_type.split(";", 1)[0].strip().lower()
    return base in ALLOWED_CONTENT_TYPES


def fetch_page(url: str) -> str:
    """Download a page and return JSON with url, title, and plain text."""
    valid = _validate_http_url(url)
    if not valid:
        return "fetch failed: invalid url"

    try:
        with httpx.Client(
            timeout=FETCH_TIMEOUT,
            follow_redirects=True,
            max_redirects=FETCH_MAX_REDIRECTS,
        ) as client:
            with client.stream("GET", valid) as response:
                if response.status_code >= 400:
                    return f"fetch failed: HTTP {response.status_code}"

                final = str(response.url)
                if urlparse(final).scheme not in ("http", "https"):
                    return "fetch failed: invalid url"

                content_type_header = response.headers.get("content-type")
                if not _content_type_ok(content_type_header):
                    return "fetch failed: unsupported content type"

                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > FETCH_MAX_BYTES:
                        remaining = FETCH_MAX_BYTES - (total - len(chunk))
                        if remaining > 0:
                            chunks.append(chunk[:remaining])
                        break
                    chunks.append(chunk)

                raw_bytes = b"".join(chunks)
                valid = final
    except httpx.TooManyRedirects:
        return "fetch failed: too many redirects"
    except httpx.HTTPError as exc:
        return f"fetch failed: {exc}"

    body = raw_bytes.decode("utf-8", errors="replace")
    content_type = (content_type_header or "").lower()
    if "text/plain" in content_type:
        text = body.strip()
        title = valid
    else:
        title = _extract_title(body) or valid
        text = _html_to_text(body)

    if not text:
        return "fetch failed: empty page"

    text = _truncate_text(text)
    return json.dumps(
        {"url": valid, "title": title, "text": text},
        ensure_ascii=False,
    )


def web_search(query: str, *, tavily_api_key: str) -> str:
    """Run Tavily search. Returns JSON string for the model (results or error)."""
    query = query.strip()
    if not query:
        return "search failed: empty query"

    payload = {
        "api_key": tavily_api_key,
        "query": query,
        "max_results": MAX_RESULTS,
        "include_answer": False,
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(TAVILY_SEARCH_URL, json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        return f"search failed: HTTP {exc.response.status_code}"
    except httpx.HTTPError as exc:
        return f"search failed: {exc}"

    results = []
    for item in data.get("results", [])[:MAX_RESULTS]:
        results.append(
            {
                "title": item.get("title") or "",
                "url": item.get("url") or "",
                "snippet": item.get("content") or item.get("snippet") or "",
            }
        )

    return json.dumps({"results": results}, ensure_ascii=False)


def run_tool(name: str, arguments: dict[str, Any] | str, *, tavily_api_key: str) -> str:
    """Dispatch a tool by name. Always returns a string for the tool result message."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            return f"tool error: invalid JSON arguments for {name}"

    if name == "web_search":
        query = arguments.get("query", "")
        if not isinstance(query, str):
            return "search failed: query must be a string"
        return web_search(query, tavily_api_key=tavily_api_key)

    if name == "fetch_page":
        url = arguments.get("url", "")
        if not isinstance(url, str):
            return "fetch failed: url must be a string"
        return fetch_page(url)

    return f"tool unavailable: {name}"
