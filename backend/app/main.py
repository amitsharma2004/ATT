"""FastAPI application entrypoint for STT and Speaker Identification Service."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Production logging configuration: suppress third-party library spam
class ProductionConsoleFormatter(logging.Formatter):
    """Clean, high-signal logging format with minimal ANSI colors."""
    COLORS = {
        logging.DEBUG: "\033[36m",     # Cyan
        logging.INFO: "\033[32m",      # Green
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR: "\033[31m",     # Red
        logging.CRITICAL: "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        time_str = self.formatTime(record, "%H:%M:%S")
        level_str = f"{color}{record.levelname:<7}{self.RESET}"
        name_short = record.name.split(".")[-1]
        return f"{time_str} | {level_str} | [{name_short}] {record.getMessage()}"

handler = logging.StreamHandler()
handler.setFormatter(ProductionConsoleFormatter())
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers = [handler]

# Mute noisy internal/external library logs
NOISY_LOGGERS = [
    "transformers",
    "transformers_modules",
    "torch",
    "torchaudio",
    "pyannote",
    "pyannote.audio",
    "speechbrain",
    "numba",
    "urllib3",
    "huggingface_hub",
    "multipart",
    "multipart.multipart",
    "asyncio",
    "httpcore",
    "httpx",
]
for log_name in NOISY_LOGGERS:
    logging.getLogger(log_name).setLevel(logging.WARNING)


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

