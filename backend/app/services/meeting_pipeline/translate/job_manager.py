"""
Asynchronous Translation Job Manager
====================================
backend/app/services/meeting_pipeline/translate/job_manager.py

Handles creation of translation jobs, pushing full transcripts, chunking them,
running local LLM translation with memory preservation, and tracking job progress.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from backend.app.services.meeting_pipeline.translate.chunked_translate import (
    SpeakerTranscriptSegment,
    TranslationMemory,
    translate_transcript,
)
from backend.app.services.meeting_pipeline.translation_service import indic_translation_service

logger = logging.getLogger(__name__)


class TranslationJobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TranslationJob:
    job_id: str
    target_language: str
    total_segments: int
    total_chunks: int = 0
    current_chunk: int = 0
    status: TranslationJobStatus = TranslationJobStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    translated_segments: List[Dict[str, Any]] = field(default_factory=list)
    running_glossary: Dict[str, str] = field(default_factory=dict)
    conversation_context: str = ""
    progress_percentage: float = 0.0
    failed_chunks: List[int] = field(default_factory=list)


class TranslationJobManager:
    """In-memory singleton job manager for chunked transcript translations."""

    def __init__(self):
        self._jobs: Dict[str, TranslationJob] = {}
        self._lock = threading.Lock()

    def create_job(
        self,
        segments: List[Dict[str, Any]],
        target_language: str = "English",
        max_input_tokens_per_chunk: int = 1500,
        max_output_tokens: int = 4096,
        translation_rules: Optional[List[str]] = None,
    ) -> TranslationJob:
        job_id = f"tr_job_{uuid.uuid4().hex[:12]}"
        job = TranslationJob(
            job_id=job_id,
            target_language=target_language,
            total_segments=len(segments),
        )
        with self._lock:
            self._jobs[job_id] = job

        # Run translation in a background worker thread
        thread = threading.Thread(
            target=self._run_job_worker,
            args=(job_id, segments, target_language, max_input_tokens_per_chunk, max_output_tokens, translation_rules),
            daemon=True,
        )
        thread.start()
        return job

    def get_job(self, job_id: str) -> Optional[TranslationJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> List[TranslationJob]:
        with self._lock:
            return list(self._jobs.values())

    def _run_job_worker(
        self,
        job_id: str,
        raw_segments: List[Dict[str, Any]],
        target_language: str,
        max_input_tokens_per_chunk: int,
        max_output_tokens: int,
        translation_rules: Optional[List[str]],
    ) -> None:
        logger.info("🚀 [TRANSLATION JOB START] Job ID: %s | Segments: %d | Target: %s", job_id, len(raw_segments), target_language)
        job = self.get_job(job_id)
        if not job:
            return

        with self._lock:
            job.status = TranslationJobStatus.PROCESSING

        try:
            # Convert raw segment dicts to SpeakerTranscriptSegment
            typed_segments = [
                SpeakerTranscriptSegment(
                    start=float(s.get("start", 0.0)),
                    end=float(s.get("end", 0.0)),
                    text=str(s.get("text", "")).strip(),
                    speaker=str(s.get("speaker", "SPEAKER")).strip(),
                )
                for s in raw_segments
            ]

            def on_chunk_completed(chunk_num: int, total_chunks: int, chunk_results: List[SpeakerTranscriptSegment]):
                with self._lock:
                    job.current_chunk = chunk_num
                    job.total_chunks = total_chunks
                    job.progress_percentage = round((chunk_num / total_chunks) * 100.0, 1)
                    logger.info("⚡ [TRANSLATION JOB PROGRESS] Job %s: chunk %d/%d (%.1f%%)", job_id, chunk_num, total_chunks, job.progress_percentage)

            # Local LLM translation call
            def translate_fn(prompt: str) -> str:
                return indic_translation_service.generate_raw_completion(
                    prompt=prompt,
                    max_new_tokens=max_output_tokens,
                    temperature=0.2,
                    timeout_s=120.0,
                )

            translated_segs, final_memory, failed_chunks = translate_transcript(
                segments=typed_segments,
                translate_fn=translate_fn,
                target_language=target_language,
                max_input_tokens_per_chunk=max_input_tokens_per_chunk,
                translation_rules=translation_rules,
                on_chunk_completed=on_chunk_completed,
            )

            with self._lock:
                job.translated_segments = [
                    {
                        "start": s.start,
                        "end": s.end,
                        "speaker": s.speaker,
                        "text": s.text,
                    }
                    for s in translated_segs
                ]
                job.running_glossary = dict(final_memory.glossary)
                job.conversation_context = final_memory.conversation_context
                job.failed_chunks = failed_chunks
                job.status = TranslationJobStatus.COMPLETED
                job.progress_percentage = 100.0
                job.completed_at = datetime.utcnow().isoformat()

            logger.info("✅ [TRANSLATION JOB COMPLETE] Job ID: %s successfully translated %d segments (failed chunks: %s)",
                         job_id, len(translated_segs), failed_chunks or "none")

        except Exception as exc:
            logger.error("❌ [TRANSLATION JOB FAILED] Job %s error: %s", job_id, exc, exc_info=True)
            with self._lock:
                job.status = TranslationJobStatus.FAILED
                job.error_message = str(exc)
                job.completed_at = datetime.utcnow().isoformat()


translation_job_manager = TranslationJobManager()
