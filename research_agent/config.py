from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_MAX_ROUNDS = 4
DEFAULT_TRANSCRIPT_DIR = "runs"


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    tavily_api_key: str
    gemini_model: str
    max_rounds: int
    save_transcripts: bool
    transcript_dir: Path


def _app_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def load_settings() -> Settings:
    """Load settings from environment, with optional app/.env file."""
    load_dotenv(_app_dir() / ".env")

    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()
    gemini_model = (
        os.getenv("GEMINI_MODEL", "").strip()
        or os.getenv("GEMINI_CHAT_MODEL", "").strip()
        or DEFAULT_MODEL
    )

    raw_rounds = os.getenv("MAX_ROUNDS", str(DEFAULT_MAX_ROUNDS)).strip()
    try:
        max_rounds = int(raw_rounds)
    except ValueError as exc:
        raise ValueError(f"MAX_ROUNDS must be an integer, got {raw_rounds!r}") from exc
    if max_rounds < 1:
        raise ValueError(f"MAX_ROUNDS must be at least 1, got {max_rounds}")

    save_transcripts = os.getenv("SAVE_TRANSCRIPTS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    raw_dir = os.getenv("TRANSCRIPT_DIR", DEFAULT_TRANSCRIPT_DIR).strip()
    transcript_dir = Path(raw_dir)
    if not transcript_dir.is_absolute():
        transcript_dir = _app_dir() / transcript_dir

    return Settings(
        gemini_api_key=gemini_api_key,
        tavily_api_key=tavily_api_key,
        gemini_model=gemini_model,
        max_rounds=max_rounds,
        save_transcripts=save_transcripts,
        transcript_dir=transcript_dir,
    )


def missing_required_keys(settings: Settings) -> list[str]:
    missing: list[str] = []
    if not settings.gemini_api_key:
        missing.append("GEMINI_API_KEY")
    if not settings.tavily_api_key:
        missing.append("TAVILY_API_KEY")
    return missing
