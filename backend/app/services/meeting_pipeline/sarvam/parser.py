"""Parser transforming raw Sarvam Batch STT output into normalized domain models."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from backend.app.services.meeting_pipeline.sarvam.exceptions import SarvamParserError
from backend.app.services.meeting_pipeline.sarvam.models import (
    SarvamBatchTranscriptResponse,
    SarvamDiarizedSegment,
)

logger = logging.getLogger(__name__)


def format_speaker_label(raw_speaker: Any) -> str:
    """Normalize raw speaker identifier into standardized 'SPEAKER_XX' format.
    
    Examples:
        - "0" or 0 -> "SPEAKER_00"
        - "1" or 1 -> "SPEAKER_01"
        - "SPEAKER_0" -> "SPEAKER_00"
        - "spk_1" -> "SPEAKER_01"
        - "unknown" -> "SPEAKER_UNKNOWN"
    """
    if raw_speaker is None:
        return "SPEAKER_UNKNOWN"

    s_str = str(raw_speaker).strip()
    if not s_str:
        return "SPEAKER_UNKNOWN"

    # If it already looks like SPEAKER_00
    if s_str.upper().startswith("SPEAKER_"):
        suffix = s_str[8:]
        if suffix.isdigit():
            return f"SPEAKER_{int(suffix):02d}"
        return s_str.upper()

    # If numeric string or int (e.g. "0", "1", 2)
    if s_str.isdigit():
        return f"SPEAKER_{int(s_str):02d}"

    # If format like spk_0 or speaker0
    clean = s_str.lower().replace("speaker", "").replace("spk", "").replace("_", "").replace("-", "")
    if clean.isdigit():
        return f"SPEAKER_{int(clean):02d}"

    return f"SPEAKER_{s_str.upper()}"


class SarvamOutputParser:
    """Parses raw JSON outputs returned by Sarvam Batch Speech-to-Text API."""

    @classmethod
    def parse_job_output(
        cls,
        raw_output: Dict[str, Any],
        fallback_model: str = "saaras:v4",
        fallback_language: Optional[str] = "en-IN",
        job_id: Optional[str] = None,
        elapsed_seconds: Optional[float] = None,
    ) -> SarvamBatchTranscriptResponse:
        """Parse raw Sarvam output dictionary into SarvamBatchTranscriptResponse.
        
        Handles both:
        1. Top-level Sarvam output format:
           {
               "transcript": "...",
               "diarized_transcript": {"entries": [...]},
               "language_code": "en-IN",
               "language_probability": 1.0,
               ...
           }
        2. Nested wrapped format:
           {
               "job_id": "...",
               "sarvam_output": { ... }
           }
        """
        if not isinstance(raw_output, dict):
            raise SarvamParserError(f"Expected dict for Sarvam output, got {type(raw_output).__name__}")

        # Unwrap nested 'sarvam_output' if present
        data = raw_output.get("sarvam_output", raw_output)
        if not isinstance(data, dict):
            raise SarvamParserError("Nested 'sarvam_output' field must be a dictionary")

        # Extract metadata
        actual_job_id = job_id or raw_output.get("job_id") or data.get("job_id")
        request_id = data.get("request_id")
        model = raw_output.get("model") or fallback_model
        language = data.get("language_code") or fallback_language

        # Extract diarized entries
        diarized_transcript = data.get("diarized_transcript")
        entries: List[Dict[str, Any]] = []

        if isinstance(diarized_transcript, dict):
            raw_entries = diarized_transcript.get("entries")
            if isinstance(raw_entries, list):
                entries = raw_entries
        elif isinstance(diarized_transcript, list):
            entries = diarized_transcript

        segments: List[SarvamDiarizedSegment] = []
        distinct_speakers = set()

        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                logger.warning("Skipping malformed entry at index %d: %s", idx, entry)
                continue

            text = entry.get("transcript") or entry.get("text") or ""
            text = str(text).strip()
            if not text:
                continue

            # Timing parsing
            start_raw = entry.get("start_time_seconds")
            if start_raw is None:
                start_raw = entry.get("start")
            end_raw = entry.get("end_time_seconds")
            if end_raw is None:
                end_raw = entry.get("end")

            try:
                start_f = float(start_raw) if start_raw is not None else 0.0
                end_f = float(end_raw) if end_raw is not None else start_f + 0.1
                # Ensure end > start
                if end_f <= start_f:
                    end_f = start_f + 0.05
            except (ValueError, TypeError) as exc:
                logger.warning("Invalid timing in entry %d (%s), using 0.0 default: %s", idx, entry, exc)
                start_f = 0.0
                end_f = 0.1

            speaker_id_raw = entry.get("speaker_id")
            if speaker_id_raw is None:
                speaker_id_raw = entry.get("speaker")

            speaker_formatted = format_speaker_label(speaker_id_raw)
            distinct_speakers.add(speaker_formatted)

            try:
                seg = SarvamDiarizedSegment(
                    speaker=speaker_formatted,
                    start=round(start_f, 2),
                    end=round(end_f, 2),
                    text=text,
                )
                segments.append(seg)
            except Exception as exc:
                raise SarvamParserError(f"Failed to create segment from entry {idx}: {exc}") from exc

        # Fallback if no diarized entries were provided but full transcript exists
        if not segments:
            full_text = data.get("transcript") or ""
            if full_text and isinstance(full_text, str) and full_text.strip():
                speaker_label = "SPEAKER_00"
                distinct_speakers.add(speaker_label)
                segments.append(
                    SarvamDiarizedSegment(
                        speaker=speaker_label,
                        start=0.0,
                        end=1.0,
                        text=full_text.strip(),
                    )
                )

        # Sort segments chronologically
        segments.sort(key=lambda s: (s.start, s.end))

        # Determine audio duration estimate
        audio_duration = None
        if segments:
            audio_duration = max(s.end for s in segments)

        metadata: Dict[str, Any] = {
            "job_id": actual_job_id,
            "request_id": request_id,
            "language_probability": data.get("language_probability"),
            "audio_hash": data.get("audio_hash"),
            "elapsed_seconds": elapsed_seconds,
        }

        return SarvamBatchTranscriptResponse(
            engine="sarvam",
            model=model,
            language=language,
            audio_duration=round(audio_duration, 2) if audio_duration is not None else None,
            total_speakers=len(distinct_speakers),
            segments=segments,
            metadata=metadata,
        )
