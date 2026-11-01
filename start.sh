#!/usr/bin/env bash
set -euo pipefail

APP="$(cd "$(dirname "$0")" && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3.11+ is required. Install it, then run this script again."
  exit 1
fi

if [[ ! -d "$APP/.venv" ]]; then
  python3 -m venv "$APP/.venv"
fi

"$APP/.venv/bin/pip" install -q -r "$APP/requirements.txt"

if [[ ! -f "$APP/.env" ]]; then
  cp "$APP/.env.example" "$APP/.env"
  echo "Created .env from the example file."
  echo "Add GEMINI_API_KEY and TAVILY_API_KEY in .env, then run ./start.sh again."
  exit 1
fi

cd "$APP"
exec "$APP/.venv/bin/python" -m research_agent --web
