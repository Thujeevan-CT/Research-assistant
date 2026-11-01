from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types

from research_agent.config import Settings
from research_agent.evidence import EvidenceStore, normalize_url

_OFFICIAL_PHRASES = re.compile(
    r"(?i)\b("
    r"official(?:ly)?|confirmed by|government portal|"
    r"federal ministry|according to the official"
    r")\b[^.]*"
)


@dataclass
class ClaimCheck:
    claim: str
    url: str
    snippet: str
    passed: bool
    reason: str

    def to_event(self) -> dict[str, Any]:
        return {
            "event": "claim_check",
            "claim": self.claim,
            "url": self.url,
            "snippet": self.snippet,
            "pass": self.passed,
            "reason": self.reason,
        }


CLAIMS_JSON_INSTRUCTION = """Return ONLY valid JSON (no markdown fences).

{
  "recent_news_note": "If the user asked for latest/recent news and tool results show no change since a date, state that here. Otherwise null or omit.",
  "claims": [
    {
      "claim": "One factual sentence for the user.",
      "url": "https://exact-url-from-tool-results",
      "snippet": "Exact contiguous quote copied from that URL's snippet or fetched page text"
    }
  ]
}

Rules:
- Every claim MUST include url + snippet copied verbatim from web_search or fetch_page results in this conversation.
- Do NOT attribute facts to an official body unless the snippet itself says so.
- Do NOT use blog URLs for minimum-wage or immigration fee figures unless the snippet is purely factual with no attribution fluff.
- Prefer primary sources that contain the numbers when they appear in search results.
- If you cannot find a supporting snippet for a fact, omit that claim entirely."""


def asks_for_news(question: str) -> bool:
    return bool(
        re.search(
            r"\b(latest|recent|news|what changed|what'?s new|update)\b",
            question,
            re.I,
        )
    )


def _text_from_response(response: types.GenerateContentResponse) -> str:
    if not response.candidates:
        return ""
    content = response.candidates[0].content
    if not content or not content.parts:
        return ""
    parts = [p.text for p in content.parts if p.text]
    return "\n".join(parts).strip()


def _parse_json_payload(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def extract_claims_payload(
    client: genai.Client,
    settings: Settings,
    system_prompt: str,
    contents: list[types.Content],
    question: str,
) -> dict[str, Any]:
    prompt = (
        f"User question:\n{question}\n\n"
        f"{CLAIMS_JSON_INSTRUCTION}"
    )
    verify_contents = list(contents)
    verify_contents.append(
        types.Content(role="user", parts=[types.Part(text=prompt)])
    )
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=verify_contents,
        config=config,
    )
    raw = _text_from_response(response)
    if not raw:
        return {"claims": []}
    try:
        return _parse_json_payload(raw)
    except json.JSONDecodeError:
        return {"claims": []}


def sanitize_claim_wording(claim: str, snippet: str) -> str:
    snippet_l = snippet.lower()
    if any(
        word in snippet_l
        for word in ("official", "confirmed", "ministry", "government portal")
    ):
        return claim.strip()
    cleaned = _OFFICIAL_PHRASES.sub("", claim)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,;.")
    return cleaned or claim.strip()


def verify_claims(
    store: EvidenceStore,
    claims: list[dict[str, Any]],
) -> list[ClaimCheck]:
    checks: list[ClaimCheck] = []
    for item in claims:
        claim = str(item.get("claim") or "").strip()
        url = normalize_url(str(item.get("url") or ""))
        snippet = str(item.get("snippet") or "").strip()
        if not claim or not url or not snippet:
            checks.append(
                ClaimCheck(
                    claim=claim or "(empty)",
                    url=url,
                    snippet=snippet,
                    passed=False,
                    reason="missing claim, url, or snippet",
                )
            )
            continue

        if not store.has_url(url):
            checks.append(
                ClaimCheck(claim, url, snippet, False, "url not in retrieved evidence")
            )
            continue

        ok, reason = store.verify_snippet(url, snippet)
        if not ok:
            checks.append(ClaimCheck(claim, url, snippet, False, reason))
            continue

        safe_claim = sanitize_claim_wording(claim, snippet)
        checks.append(ClaimCheck(safe_claim, url, snippet, True, reason))

    return checks


def build_answer_from_checks(
    checks: list[ClaimCheck],
    recent_news_note: str | None,
    store: EvidenceStore,
) -> str:
    passed = [c for c in checks if c.passed]
    lines: list[str] = ["## Summary", ""]
    if recent_news_note:
        lines.append(recent_news_note.strip())
        lines.append("")

    if not passed:
        lines.append(
            "No claims could be verified against retrieved sources for this question."
        )
    else:
        for check in passed:
            lines.append(f"- {check.claim}")

    lines.extend(["", "## Sources", ""])
    seen: set[str] = set()
    for check in passed:
        if check.url in seen:
            continue
        seen.add(check.url)
        title = store.title_for(check.url)
        lines.append(f"- {title} — {check.url}")

    return "\n".join(lines).strip()
