"""Timestamp alignment module matching Whisper transcripts with Pyannote diarization."""
from backend.app.services.meeting_pipeline.alignment.aligner import (
    UNKNOWN_SPEAKER,
    TimestampAligner,
)
from backend.app.services.meeting_pipeline.alignment.exceptions import (
    AlignmentError,
    InvalidTimestampError,
)
from backend.app.services.meeting_pipeline.alignment.models import (
    AlignedTranscriptResult,
    AlignedTranscriptSegment,
    SpeakerCandidateEvidence,
)

__all__ = [
    "TimestampAligner",
    "UNKNOWN_SPEAKER",
    "AlignedTranscriptSegment",
    "AlignedTranscriptResult",
    "SpeakerCandidateEvidence",
    "AlignmentError",
    "InvalidTimestampError",
]
