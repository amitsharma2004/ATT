"""API endpoints for Voice Enrollment, Audio Recording/Upload, and Meeting Transcription."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
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

        # Execute complete pipeline: Audio -> Diarization -> Whisper -> Alignment -> Matching.
        # process_meeting() is a long-running, blocking (sync) call — offload it to
        # FastAPI's threadpool so it doesn't freeze the event loop (and this
        # single-worker server's health checks/other requests) for the whole
        # duration of one meeting's processing.
        result = await run_in_threadpool(
            orchestrator.process_meeting,
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


class TranscriptSegmentUpdateRequest(BaseModel):
    speaker_name: Optional[str] = None
    speaker_cluster: Optional[str] = None
    user_id: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None
    text: str
    translated_text: Optional[str] = None
    english_line: Optional[str] = None


class UpdateTranscriptRequest(BaseModel):
    job_id: Optional[str] = None
    segments: List[TranscriptSegmentUpdateRequest]


@router.post("/transcript/update")
async def update_meeting_transcript(payload: UpdateTranscriptRequest):
    """Save edited meeting transcript to disk for persistence and auditability."""
    try:
        save_dir = settings.processed_storage_path / "edited_transcripts"
        save_dir.mkdir(parents=True, exist_ok=True)

        job_id = payload.job_id or uuid.uuid4().hex
        timestamp_str = uuid.uuid4().hex[:8]
        save_path = save_dir / f"transcript_{job_id}_{timestamp_str}.json"

        # Also keep a stable canonical file for this job_id
        canonical_path = save_dir / f"transcript_{job_id}_latest.json"

        import json
        data = {
            "job_id": job_id,
            "segments_count": len(payload.segments),
            "segments": [seg.model_dump() for seg in payload.segments],
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        with open(canonical_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return {
            "status": "success",
            "message": "Transcript updated successfully",
            "job_id": job_id,
            "segments_count": len(payload.segments),
            "saved_file": str(save_path.name),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to update transcript: {exc}")


class SpellCheckRequest(BaseModel):
    segments: List[TranscriptSegmentUpdateRequest]


@router.post("/transcript/spellcheck")
async def check_transcript_spelling(payload: SpellCheckRequest):
    """Check whole transcript for typos and company glossary fuzzy matches."""
    try:
        from backend.app.services.meeting_pipeline.spell_service import spell_check_service
        raw_segs = [s.model_dump() for s in payload.segments]
        result = spell_check_service.check_segments(raw_segs)
        return {
            "status": "success",
            "total_errors": result["total_errors"],
            "segments": result["segments"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Spell check failed: {exc}")


# DISABLED for this push: Whisper now runs with task="translate" and produces
# English directly, so the local-LLM chunked-translation job system (translate/
# package) isn't wired up for now. Left in place — uncomment together with
# orchestrator.py's stage 7 block to re-enable.
#
# class CreateTranslationJobRequest(BaseModel):
#     segments: List[TranscriptSegmentUpdateRequest]
#     target_language: str = "English"
#     max_input_tokens_per_chunk: int = 1500
#     translation_rules: Optional[List[str]] = None
#
#
# @router.post("/translation/jobs")
# async def create_translation_job(payload: CreateTranslationJobRequest):
#     """Create an asynchronous speaker-aware chunked translation job."""
#     try:
#         from backend.app.services.meeting_pipeline.translate import translation_job_manager
#         raw_segs = [s.model_dump() for s in payload.segments]
#         job = translation_job_manager.create_job(
#             segments=raw_segs,
#             target_language=payload.target_language,
#             max_input_tokens_per_chunk=payload.max_input_tokens_per_chunk,
#             translation_rules=payload.translation_rules,
#         )
#         return {
#             "status": "success",
#             "job_id": job.job_id,
#             "job_status": job.status.value,
#             "target_language": job.target_language,
#             "total_segments": job.total_segments,
#             "created_at": job.created_at,
#         }
#     except Exception as exc:
#         raise HTTPException(status_code=500, detail=f"Failed to create translation job: {exc}")
#
#
# @router.get("/translation/jobs/{job_id}")
# async def get_translation_job_status(job_id: str):
#     """Retrieve progress status or final translated output for a translation job."""
#     from backend.app.services.meeting_pipeline.translate import translation_job_manager
#     job = translation_job_manager.get_job(job_id)
#     if not job:
#         raise HTTPException(status_code=404, detail=f"Translation job '{job_id}' not found")
#
#     return {
#         "job_id": job.job_id,
#         "status": job.status.value,
#         "target_language": job.target_language,
#         "total_segments": job.total_segments,
#         "current_chunk": job.current_chunk,
#         "total_chunks": job.total_chunks,
#         "progress_percentage": job.progress_percentage,
#         "created_at": job.created_at,
#         "completed_at": job.completed_at,
#         "error_message": job.error_message,
#         "running_glossary": job.running_glossary,
#         "translated_segments": job.translated_segments,
#         "failed_chunks": job.failed_chunks,
#     }


