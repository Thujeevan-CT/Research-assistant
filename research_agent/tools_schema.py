from __future__ import annotations

from typing import Any

from research_agent.tools import (
    FETCH_PAGE_DECLARATION,
    RESEARCH_TOOLS,
    WEB_SEARCH_DECLARATION,
)


def tool_declarations_snapshot() -> list[dict[str, Any]]:
    """JSON-safe tool schemas sent to the model (for transcripts and debugging)."""
    decls = [WEB_SEARCH_DECLARATION, FETCH_PAGE_DECLARATION]
    out: list[dict[str, Any]] = []
    for decl in decls:
        if hasattr(decl, "model_dump"):
            out.append(decl.model_dump(mode="json", exclude_none=True))
        else:
            out.append({"name": decl.name, "description": decl.description})
    _ = RESEARCH_TOOLS  # ensure Tool bundle stays in sync
    return out
