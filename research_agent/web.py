from __future__ import annotations

import json
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from research_agent.agent import iter_research
from research_agent.config import Settings, load_settings, missing_required_keys
from research_agent.transcript import resolve_transcript_path

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_QUESTION_LENGTH = 2000


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="Research Assistant", version="0.1.0")
    app.state.settings = settings or load_settings()

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/research")
    def research_stream(payload: ResearchRequest) -> StreamingResponse:
        cfg: Settings = app.state.settings
        missing = missing_required_keys(cfg)
        if missing:
            raise HTTPException(
                status_code=500,
                detail=f"Missing configuration: {', '.join(missing)}",
            )

        question = payload.question.strip()

        def ndjson_stream():
            for event in iter_research(question, cfg):
                yield json.dumps(event, ensure_ascii=False) + "\n"

        return StreamingResponse(
            ndjson_stream(),
            media_type="application/x-ndjson",
        )

    @app.get("/api/transcripts/{run_id}")
    def download_transcript(run_id: str) -> FileResponse:
        cfg: Settings = app.state.settings
        path = resolve_transcript_path(cfg.transcript_dir, run_id)
        if path is None:
            if len(run_id) != 12 or not all(
                c in "0123456789abcdef" for c in run_id.lower()
            ):
                raise HTTPException(status_code=400, detail="Invalid run id")
            raise HTTPException(status_code=404, detail="Transcript not found")
        return FileResponse(
            path,
            media_type="application/json",
            filename=path.name,
        )

    return app


def run_server(settings: Settings | None = None) -> None:
    host = os.getenv("HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("PORT", "8000").strip() or "8000")
    cfg = settings or load_settings()
    app = create_app(cfg)
    print(f"Research Assistant UI: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
