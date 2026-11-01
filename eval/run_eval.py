#!/usr/bin/env python3
"""Run eval cases against the research agent (live API keys required)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from research_agent.agent import iter_research
from research_agent.config import load_settings

CASES_PATH = Path(__file__).resolve().parent / "cases.json"


def _answer_text(events: list[dict]) -> str:
    for ev in reversed(events):
        if ev.get("event") == "answer":
            return ev.get("text") or ""
    return ""


def _had_search(events: list[dict]) -> bool:
    return any(ev.get("event") == "search" for ev in events)


def _had_fetch(events: list[dict]) -> bool:
    return any(ev.get("event") == "fetched" for ev in events)


def score_case(case: dict, answer: str, events: list[dict]) -> list[str]:
    failures: list[str] = []
    lower = answer.lower()

    for phrase in case.get("must_not_include", []):
        if phrase.lower() in lower:
            failures.append(f"must_not_include violated: {phrase!r}")

    for phrase in case.get("must_include_all", []):
        if phrase.lower() not in lower:
            failures.append(f"must_include_all missing: {phrase!r}")

    any_list = case.get("must_include_any", [])
    if any_list and not any(p.lower() in lower for p in any_list):
        failures.append(f"must_include_any missing (none of {any_list!r})")

    if case.get("requires_search") and not _had_search(events):
        failures.append("requires_search: no search event")

    if case.get("requires_fetch") and not _had_fetch(events):
        failures.append("requires_fetch: no successful fetch_page")

    return failures


def run_case(case: dict, settings) -> tuple[bool, list[str]]:
    events: list[dict] = []
    for event in iter_research(case["question"], settings):
        events.append(event)
    answer = _answer_text(events)
    failures = score_case(case, answer, events)
    return len(failures) == 0, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Run research agent eval cases")
    parser.add_argument("--id", help="Run a single case id")
    parser.add_argument("--list", action="store_true", help="List case ids")
    args = parser.parse_args()

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if args.list:
        for c in cases:
            print(c["id"])
        return

    if args.id:
        cases = [c for c in cases if c["id"] == args.id]
        if not cases:
            print(f"Unknown case id: {args.id}", file=sys.stderr)
            sys.exit(1)

    settings = load_settings()
    passed = 0
    for case in cases:
        ok, failures = run_case(case, settings)
        status = "PASS" if ok else "FAIL"
        print(f"{status}  {case['id']}")
        if failures:
            for f in failures:
                print(f"       - {f}")
        else:
            passed += 1

    print(f"\n{passed}/{len(cases)} passed")
    sys.exit(0 if passed == len(cases) else 1)


if __name__ == "__main__":
    main()
