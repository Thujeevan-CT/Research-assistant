from __future__ import annotations

from google import genai
from google.genai import types

from research_agent.config import Settings

GROUNDING_INSTRUCTION = """You are verifying a draft research answer against tool results ONLY.

The conversation above contains web_search snippets and fetch_page text. You must NOT use outside knowledge.

Task:
1. For each factual bullet or sentence in the draft Summary, find an exact supporting phrase copied from those tool results (snippet or fetched text).
2. If there is no supporting phrase, REMOVE that claim from the Summary entirely. Do not replace it with memory.
3. Do not merge unrelated policies (e.g. enrollment cap exemptions vs dependent visa rules unless the source text links them).
4. Keep ## Sources; drop any source not used in the revised Summary.
5. Preserve as-of dates that appear in the sources. Do not state outdated fees if newer amounts appear in fetched text.

Output the full revised Markdown answer only (## Summary and ## Sources). No preamble."""


def collect_evidence_lines(contents: list[types.Content]) -> list[str]:
    """Flatten tool payloads into searchable lines for heuristics."""
    lines: list[str] = []
    for content in contents:
        for part in content.parts or []:
            fr = part.function_response
            if not fr or not fr.response:
                continue
            resp = fr.response
            if not isinstance(resp, dict):
                continue
            name = fr.name or ""
            if name == "web_search":
                for item in resp.get("results", []):
                    title = item.get("title") or ""
                    url = item.get("url") or ""
                    snippet = item.get("snippet") or ""
                    lines.append(f"SEARCH {url} | {title} | {snippet}")
            elif name == "fetch_page":
                url = resp.get("url") or ""
                title = resp.get("title") or ""
                text = resp.get("text") or ""
                lines.append(f"FETCH {url} | {title} | {text[:500]}")
            elif "error" in resp:
                lines.append(f"ERROR {name} | {resp['error']}")
    return lines


def _text_from_content(content: types.Content) -> str:
    chunks: list[str] = []
    for part in content.parts or []:
        if part.text:
            chunks.append(part.text)
    return "\n".join(chunks).strip()


def run_grounding_pass(
    client: genai.Client,
    settings: Settings,
    system_prompt: str,
    contents: list[types.Content],
    draft: str,
) -> str:
    """Second model pass: drop claims not supported by tool text."""
    verify_contents = list(contents)
    verify_contents.append(
        types.Content(role="model", parts=[types.Part(text=draft)])
    )
    verify_contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=GROUNDING_INSTRUCTION)],
        )
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
    if not response.candidates:
        return draft
    content = response.candidates[0].content
    if content is None:
        return draft
    revised = _text_from_content(content)
    return revised or draft
