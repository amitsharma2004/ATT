"""FastAPI application entrypoint for STT and Speaker Identification Service."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Setup rich console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

from backend.app.api.routes.pipeline import router as pipeline_router
from backend.app.core.config import get_settings
from backend.app.services.meeting_pipeline.pipeline.orchestrator import MeetingPipelineOrchestrator

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Modular pipeline for audio preprocessing, speaker diarization, Whisper STT, and speaker identification",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline_orchestrator = MeetingPipelineOrchestrator(settings=settings)

# Include API routes
app.include_router(pipeline_router)

# Mount static web UI
static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def serve_index():
        from fastapi.responses import FileResponse
        return FileResponse(static_dir / "index.html")


@app.get("/health")
async def health():
    """Service and pipeline health status."""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "environment": settings.app_env,
        "pipeline": pipeline_orchestrator.health_check(),
    }

