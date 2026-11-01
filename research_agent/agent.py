from __future__ import annotations

import json
import sys
import uuid
from collections.abc import Iterator
from typing import Any

from google import genai
from google.genai import types

from research_agent.claims import (
    ClaimCheck,
    asks_for_news,
    build_answer_from_checks,
    extract_claims_payload,
    verify_claims,
)
from research_agent.config import Settings
from research_agent.evidence import EvidenceStore
from research_agent.prompts import get_system_prompt
from research_agent.tools import RESEARCH_TOOLS, run_tool
from research_agent.tools_schema import tool_declarations_snapshot
from research_agent.transcript import save_transcript

FORCED_ANSWER_HINT = (
    "You have used all allowed search rounds. Do not request more tools. "
    "Reply with a single line: Ready to finalize."
)


class ModelAPIError(Exception):
    """Raised when the Gemini API call fails."""


def iter_research(question: str, settings: Settings) -> Iterator[dict[str, Any]]:
    """Run the agent loop, yielding progress events for UI or logging."""
    run_id = uuid.uuid4().hex[:12]
    events_log: list[dict[str, Any]] = []
    final_answer: str | None = None
    status = "error"

    client = genai.Client(api_key=settings.gemini_api_key)
    news = asks_for_news(question)
    system_prompt = get_system_prompt(asks_news=news)
    tool_declarations = tool_declarations_snapshot()
    evidence_store = EvidenceStore()
    grounding_log: list[dict[str, Any]] = []
    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=question)]),
    ]

    afc_off = types.AutomaticFunctionCallingConfig(disable=True)
    with_tools = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=RESEARCH_TOOLS,
        automatic_function_calling=afc_off,
    )
    without_tools = types.GenerateContentConfig(
        system_instruction=system_prompt,
        automatic_function_calling=afc_off,
    )

    def emit(event: dict[str, Any]) -> dict[str, Any]:
        if event.get("event") not in ("transcript", "done"):
            events_log.append(event)
        return event

    yield emit(
        {"event": "started", "question": question, "max_rounds": settings.max_rounds}
    )

    try:
        for round_num in range(1, settings.max_rounds + 1):
            try:
                response = _generate(
                    client, settings.gemini_model, contents, with_tools
                )
            except ModelAPIError as exc:
                yield emit(
                    {"event": "error", "message": f"Model API error: {exc}"}
                )
                yield from _finish_run(
                    run_id,
                    question,
                    settings,
                    system_prompt,
                    tool_declarations,
                    contents,
                    events_log,
                    final_answer,
                    status,
                )
                return

            model_content = _candidate_content(response)
            if model_content is None:
                yield emit({"event": "error", "message": "Model returned nothing useful."})
                yield from _finish_run(
                    run_id,
                    question,
                    settings,
                    system_prompt,
                    tool_declarations,
                    contents,
                    events_log,
                    final_answer,
                    status,
                )
                return

            text = _text_from_content(model_content)
            calls = _function_calls(model_content)

            if calls:
                yield emit({"event": "round", "number": round_num})
                contents.append(model_content)
                response_parts: list[types.Part] = []
                for call in calls:
                    name = call.name or "unknown"
                    args: dict[str, Any] = dict(call.args or {})
                    if name == "web_search":
                        yield emit(
                            {"event": "search", "query": str(args.get("query", ""))}
                        )
                    elif name == "fetch_page":
                        yield emit(
                            {"event": "fetch", "url": str(args.get("url", ""))}
                        )
                    else:
                        yield emit(
                            {"event": "search", "query": f"[tool: {name}]"}
                        )

                    result_str = run_tool(
                        name, args, tavily_api_key=settings.tavily_api_key
                    )
                    evidence_store.ingest_tool_result(name, result_str)
                    for tool_event in _tool_result_events(name, args, result_str):
                        yield emit(tool_event)
                    response_parts.append(
                        _response_part_from_result(call, name, result_str)
                    )
                contents.append(types.Content(role="user", parts=response_parts))
                continue

            if text:
                final_text, retry_hint, checks = _finalize_with_claims(
                    client,
                    settings,
                    system_prompt,
                    contents,
                    question,
                    evidence_store,
                    round_num,
                    text,
                )
                for check in checks:
                    ev = check.to_event()
                    grounding_log.append(
                        {k: v for k, v in ev.items() if k != "event"}
                    )
                    yield emit(ev)

                if retry_hint is not None:
                    yield emit(
                        {
                            "event": "status",
                            "message": "More evidence needed — search or fetch_page, then retry.",
                        }
                    )
                    contents.append(
                        types.Content(
                            role="user",
                            parts=[types.Part(text=retry_hint)],
                        )
                    )
                    continue

                yield emit(
                    {
                        "event": "grounding",
                        "message": f"Verified {sum(1 for c in checks if c.passed)}/{len(checks)} claims",
                    }
                )
                final_answer = final_text
                status = "completed"
                yield emit({"event": "answer", "text": final_text})
                yield from _finish_run(
                    run_id,
                    question,
                    settings,
                    system_prompt,
                    tool_declarations,
                    contents,
                    events_log,
                    final_answer,
                    status,
                    grounding_log,
                )
                return

            yield emit({"event": "error", "message": "Model returned nothing useful."})
            yield from _finish_run(
                run_id,
                question,
                settings,
                system_prompt,
                tool_declarations,
                contents,
                events_log,
                final_answer,
                status,
            )
            return

        yield emit(
            {
                "event": "status",
                "message": "Search limit reached. Writing final answer…",
            }
        )
        contents.append(
            types.Content(role="user", parts=[types.Part(text=FORCED_ANSWER_HINT)])
        )
        try:
            response = _generate(
                client, settings.gemini_model, contents, without_tools
            )
        except ModelAPIError as exc:
            yield emit({"event": "error", "message": f"Model API error: {exc}"})
            yield from _finish_run(
                run_id,
                question,
                settings,
                system_prompt,
                tool_declarations,
                contents,
                events_log,
                final_answer,
                status,
            )
            return

        model_content = _candidate_content(response)
        if model_content is None:
            yield emit({"event": "error", "message": "Model returned nothing useful."})
            yield from _finish_run(
                run_id,
                question,
                settings,
                system_prompt,
                tool_declarations,
                contents,
                events_log,
                final_answer,
                status,
            )
            return

        text = _text_from_content(model_content)
        calls = _function_calls(model_content)
        if text and not calls:
            final_text, _retry, checks = _finalize_with_claims(
                client,
                settings,
                system_prompt,
                contents,
                question,
                evidence_store,
                settings.max_rounds,
                text,
            )
            for check in checks:
                ev = check.to_event()
                grounding_log.append({k: v for k, v in ev.items() if k != "event"})
                yield emit(ev)
            yield emit(
                {
                    "event": "grounding",
                    "message": f"Verified {sum(1 for c in checks if c.passed)}/{len(checks)} claims",
                }
            )
            final_answer = final_text
            status = "completed"
            yield emit({"event": "answer", "text": final_text})
            yield from _finish_run(
                run_id,
                question,
                settings,
                system_prompt,
                tool_declarations,
                contents,
                events_log,
                final_answer,
                status,
                grounding_log,
            )
            return

        yield emit({"event": "error", "message": "Model returned nothing useful."})
        yield from _finish_run(
            run_id,
            question,
            settings,
            system_prompt,
            tool_declarations,
            contents,
            events_log,
            final_answer,
            status,
        )
    except ModelAPIError as exc:
        yield emit({"event": "error", "message": f"Model API error: {exc}"})
        yield from _finish_run(
            run_id,
            question,
            settings,
            system_prompt,
            tool_declarations,
            contents,
            events_log,
            final_answer,
            status,
        )


def _finish_run(
    run_id: str,
    question: str,
    settings: Settings,
    system_prompt: str,
    tool_declarations: list[dict[str, Any]],
    contents: list[types.Content],
    events_log: list[dict[str, Any]],
    final_answer: str | None,
    status: str,
    grounding_log: list[dict[str, Any]] | None = None,
) -> Iterator[dict[str, Any]]:
    try:
        path = save_transcript(
            run_id=run_id,
            question=question,
            settings=settings,
            system_prompt=system_prompt,
            tool_declarations=tool_declarations,
            contents=contents,
            events=events_log,
            final_answer=final_answer,
            status=status,
            grounding_log=grounding_log or [],
        )
    except OSError as exc:
        print(f"Warning: could not save transcript: {exc}", file=sys.stderr)
        path = None

    if path is not None:
        yield {
            "event": "transcript",
            "run_id": run_id,
            "path": path.name,
        }
    yield {"event": "done"}


def research(question: str, settings: Settings) -> None:
    """Run the research agent loop and print the answer to stdout."""
    for event in iter_research(question, settings):
        _print_event(event)


def _print_event(event: dict[str, Any]) -> None:
    kind = event.get("event")
    if kind == "started":
        print(f"Question: {event['question']}\n")
    elif kind == "round":
        print(f"Round {event['number']}")
    elif kind == "search":
        print(f"  search: {event['query']}")
    elif kind == "result":
        print(f"    → {event['title']}")
    elif kind == "fetch":
        print(f"  fetch: {event['url']}")
    elif kind == "fetched":
        print(f"    → {event['title']}")
    elif kind == "fetch_error":
        print(f"    → {event['message']}")
    elif kind == "grounding":
        print(f"  [{event.get('message', 'grounding')}]")
    elif kind == "claim_check":
        mark = "pass" if event.get("pass") else "FAIL"
        print(f"  claim [{mark}]: {event.get('claim', '')[:70]}…")
        print(f"           {event.get('reason')}")
    elif kind == "status":
        print(event.get("message", ""))
    elif kind == "answer":
        print(f"Answer:\n{event['text']}")
    elif kind == "transcript":
        print(f"Transcript: {event['path']}")
    elif kind == "error":
        print(event.get("message", "Error"), file=sys.stderr)


def _finalize_with_claims(
    client: genai.Client,
    settings: Settings,
    system_prompt: str,
    contents: list[types.Content],
    question: str,
    evidence_store: EvidenceStore,
    round_num: int,
    _draft: str,
) -> tuple[str, str | None, list[ClaimCheck]]:
    """Build answer from verified claims; optionally request another tool round."""
    payload = extract_claims_payload(
        client, settings, system_prompt, contents, question
    )
    note = payload.get("recent_news_note")
    recent_note = note if isinstance(note, str) and note.strip() else None
    raw_claims = payload.get("claims") or []
    if not isinstance(raw_claims, list):
        raw_claims = []

    checks = verify_claims(evidence_store, raw_claims)
    passed = [c for c in checks if c.passed]

    if not passed and round_num < settings.max_rounds:
        return (
            "",
            "No claims passed verification. Search again, use fetch_page on primary "
            "sources from results, and ensure snippets are exact quotes from tool output.",
            checks,
        )

    answer = build_answer_from_checks(checks, recent_note, evidence_store)
    return answer, None, checks


def _generate(
    client: genai.Client,
    model: str,
    contents: list[types.Content],
    config: types.GenerateContentConfig,
) -> types.GenerateContentResponse:
    try:
        return client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
    except Exception as exc:
        raise ModelAPIError(str(exc)) from exc


def _candidate_content(
    response: types.GenerateContentResponse,
) -> types.Content | None:
    if not response.candidates:
        return None
    content = response.candidates[0].content
    if content is None or not content.parts:
        return None
    return content


def _text_from_content(content: types.Content) -> str:
    chunks: list[str] = []
    for part in content.parts or []:
        if part.text:
            chunks.append(part.text)
    return "\n".join(chunks).strip()


def _function_calls(content: types.Content) -> list[types.FunctionCall]:
    calls: list[types.FunctionCall] = []
    for part in content.parts or []:
        if part.function_call:
            calls.append(part.function_call)
    return calls


def _response_part_from_result(
    call: types.FunctionCall,
    name: str,
    result_str: str,
) -> types.Part:
    response_payload = _tool_result_payload(result_str)
    return types.Part(
        function_response=types.FunctionResponse(
            name=name,
            id=call.id,
            response=response_payload,
        )
    )


def _tool_result_payload(result_str: str) -> dict[str, Any]:
    try:
        parsed = json.loads(result_str)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {"error": result_str}


def _tool_result_events(
    name: str,
    args: dict[str, Any],
    result_str: str,
) -> Iterator[dict[str, Any]]:
    if name == "web_search":
        for title in _result_titles(result_str):
            yield {"event": "result", "title": title}
        return

    if name == "fetch_page":
        url = str(args.get("url", ""))
        payload = _tool_result_payload(result_str)
        if "error" in payload and "text" not in payload:
            yield {
                "event": "fetch_error",
                "url": url,
                "message": str(payload["error"]),
            }
            return
        yield {
            "event": "fetched",
            "url": str(payload.get("url") or url),
            "title": str(payload.get("title") or url),
        }


def _result_titles(result_str: str) -> list[str]:
    try:
        data = json.loads(result_str)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    titles: list[str] = []
    for item in data.get("results", []):
        title = item.get("title") or item.get("url") or "untitled"
        titles.append(title)
    return titles
