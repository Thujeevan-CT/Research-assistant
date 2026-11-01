from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from google.genai import types

from research_agent.config import Settings


def serialize_contents(contents: list[types.Content]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for content in contents:
        if hasattr(content, "model_dump"):
            serialized.append(content.model_dump(mode="json", exclude_none=True))
        else:
            serialized.append(_content_fallback(content))
    return serialized


def _content_fallback(content: types.Content) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    for part in content.parts or []:
        entry: dict[str, Any] = {}
        if part.text:
            entry["text"] = part.text
        if part.function_call:
            fc = part.function_call
            entry["function_call"] = {
                "name": fc.name,
                "args": dict(fc.args or {}),
                "id": fc.id,
            }
        if part.function_response:
            fr = part.function_response
            entry["function_response"] = {
                "name": fr.name,
                "response": fr.response,
                "id": fr.id,
            }
        if entry:
            parts.append(entry)
    return {"role": content.role, "parts": parts}


def save_transcript(
    *,
    run_id: str,
    question: str,
    settings: Settings,
    system_prompt: str,
    contents: list[types.Content],
    events: list[dict[str, Any]],
    final_answer: str | None,
    status: str,
    tool_declarations: list[dict[str, Any]] | None = None,
    grounding_log: list[dict[str, Any]] | None = None,
) -> Path | None:
    if not settings.save_transcripts:
        return None

    directory = settings.transcript_dir
    directory.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "question": question,
        "model": settings.gemini_model,
        "max_rounds": settings.max_rounds,
        "system_prompt": system_prompt,
        "tool_declarations": tool_declarations or [],
        "grounding_log": grounding_log or [],
        "events": events,
        "gemini_contents": serialize_contents(contents),
        "final_answer": final_answer,
        "status": status,
    }

    path = directory / f"{run_id}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def resolve_transcript_path(transcript_dir: Path, run_id: str) -> Path | None:
    if not run_id.isalnum() or len(run_id) != 12:
        return None
    if not all(c in "0123456789abcdef" for c in run_id.lower()):
        return None
    path = (transcript_dir / f"{run_id}.json").resolve()
    root = transcript_dir.resolve()
    if root not in path.parents and path.parent != root:
        return None
    if not path.is_file():
        return None
    return path
