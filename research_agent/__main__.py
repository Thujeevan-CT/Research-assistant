import sys

from research_agent.agent import research
from research_agent.config import load_settings, missing_required_keys

USAGE = 'Usage: python -m research_agent "your question"\n       python -m research_agent --web'


def _require_settings():
    try:
        settings = load_settings()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)

    missing = missing_required_keys(settings)
    if missing:
        print(
            "Missing required environment variable(s): "
            + ", ".join(missing),
            file=sys.stderr,
        )
        print("Copy app/.env.example to app/.env and set keys.", file=sys.stderr)
        sys.exit(1)
    return settings


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] in ("--web", "web"):
        from research_agent.web import run_server

        settings = _require_settings()
        run_server(settings)
        return

    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print(USAGE, file=sys.stderr)
        sys.exit(1)

    question = sys.argv[1].strip()
    settings = _require_settings()
    research(question, settings)


if __name__ == "__main__":
    main()
