"""API endpoints for Voice Enrollment, Audio Recording/Upload, and Meeting Transcription."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from backend.app.core.config import get_settings
from backend.app.services.meeting_pipeline.pipeline.orchestrator import MeetingPipelineOrchestrator
from backend.app.services.meeting_pipeline.sarvam import (
    SarvamBatchSTTService,
    SarvamBatchTranscriptResponse,
    SarvamError,
)

router = APIRouter(prefix="/api", tags=["pipeline"])
settings = get_settings()
orchestrator = MeetingPipelineOrchestrator(settings=settings)


class EnrolledMemberResponse(BaseModel):
    user_id: str
    name: str
    team_id: str
    samples_count: int
    embedding_dimension: int


@router.get("/profiles", response_model=List[EnrolledMemberResponse])
async def list_profiles(team_id: Optional[str] = None):
    """List all enrolled team voice profiles."""
    profiles = orchestrator.registry.list_profiles(team_id=team_id)
    return [
        EnrolledMemberResponse(
            user_id=p.user_id,
            name=p.name,
            team_id=p.team_id,
            samples_count=p.samples_count,
            embedding_dimension=p.embedding_dimension,
        )
        for p in profiles
    ]


@router.delete("/profiles/{user_id}")
async def delete_profile(user_id: str):
    """Delete a team member voice profile."""
    try:
        orchestrator.registry.delete_profile(user_id)
        return {"status": "deleted", "user_id": user_id}
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/enroll")
async def enroll_member(
    user_id: str = Form(...),
    name: str = Form(...),
    team_id: str = Form("default"),
    overwrite: bool = Form(True),
    files: List[UploadFile] = File(...),
):
    """Enroll a team member with one or multiple voice samples (recorded live or uploaded)."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one voice sample audio file is required")

    temp_paths: List[Path] = []
    try:
        for idx, file in enumerate(files):
            suffix = Path(file.filename or "sample.wav").suffix or ".wav"
            temp_file = settings.raw_storage_path / f"tmp_enroll_{user_id}_{idx}_{uuid.uuid4().hex[:6]}{suffix}"
            with open(temp_file, "wb") as f:
                shutil.copyfileobj(file.file, f)
            temp_paths.append(temp_file)

        profile = orchestrator.registry.enroll_team_member(
            user_id=user_id.strip(),
            name=name.strip(),
            sample_audio_paths=temp_paths,
            team_id=team_id.strip(),
            overwrite=overwrite,
        )

        return {
            "status": "enrolled",
            "user_id": profile.user_id,
            "name": profile.name,
            "samples_count": profile.samples_count,
            "embedding_dimension": profile.embedding_dimension,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {exc}")
    finally:
        for p in temp_paths:
            p.unlink(missing_ok=True)


@router.post("/transcribe")
async def transcribe_meeting(
    file: UploadFile = File(...),
    team_id: Optional[str] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    num_speakers: Optional[int] = Form(None),
    language: Optional[str] = Form(None),
):
    """Run full meeting pipeline on uploaded or live-recorded audio."""
    suffix = Path(file.filename or "recording.wav").suffix or ".wav"
    job_id = uuid.uuid4().hex
    raw_path = settings.raw_storage_path / f"meeting_{job_id}{suffix}"

    try:
        with open(raw_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Execute complete pipeline: Audio -> Diarization -> Whisper -> Alignment -> Matching
        result = orchestrator.process_meeting(
            audio_path=raw_path,
            team_id=team_id,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            num_speakers=num_speakers,
            language=language,
        )

        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Meeting processing failed: {exc}")


# ── Sarvam Batch STT Pipeline (Step 7A) ──
sarvam_service = SarvamBatchSTTService(settings=settings)


@router.post("/transcribe/sarvam", response_model=SarvamBatchTranscriptResponse)
async def transcribe_meeting_sarvam(
    file: UploadFile = File(...),
    team_id: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    with_diarization: Optional[bool] = Form(True),
):
    """Run Sarvam Batch STT with Diarization and Speaker Identification on uploaded audio file.
    
    Independent second pipeline preserving Whisper /api/transcribe untouched.
    """
    suffix = Path(file.filename or "recording.wav").suffix or ".wav"
    job_id = uuid.uuid4().hex
    raw_path = settings.raw_storage_path / f"sarvam_raw_{job_id}{suffix}"

    try:
        with open(raw_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Execute Sarvam pipeline: Validate -> Preprocess -> Submit -> Poll -> Download -> Parse -> Match
        result = await sarvam_service.transcribe_audio(
            audio_path=raw_path,
            team_id=team_id,
            model=model,
            language_code=language,
            with_diarization=with_diarization,
        )

        return result
    except SarvamError as exc:
        raise HTTPException(status_code=500, detail=f"Sarvam transcription error: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sarvam processing failed: {exc}")

