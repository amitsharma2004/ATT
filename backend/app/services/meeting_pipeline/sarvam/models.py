"""Domain models for Sarvam Batch Speech-to-Text Pipeline."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


from backend.app.services.meeting_pipeline.speaker.models import SpeakerClusterMatch


class SarvamDiarizedSegment(BaseModel):
    """Diarized segment produced by Sarvam STT with speaker ID, timings, and optional resolved name."""
    model_config = ConfigDict(frozen=True)

    speaker: str = Field(..., description="Standardized anonymous speaker label, e.g. 'SPEAKER_00'")
    speaker_name: Optional[str] = Field(
        default=None,
        description="Resolved team member display name (e.g. 'Arjun') if matched",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Enrolled user ID if matched",
    )
    start: float = Field(..., ge=0.0, description="Segment start timestamp in seconds")
    end: float = Field(..., gt=0.0, description="Segment end timestamp in seconds")
    text: str = Field(..., description="Transcribed text for this segment")
    match_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of voice matching",
    )

    @field_validator("end")
    @classmethod
    def validate_time_order(cls, v: float, info) -> float:
        start = info.data.get("start")
        if start is not None and v < start:
            raise ValueError(f"End time ({v}) must be greater than or equal to start time ({start})")
        return v


class SarvamBatchTranscriptResponse(BaseModel):
    """Stable internal response returned by the Sarvam batch transcription pipeline."""
    model_config = ConfigDict(frozen=True)

    engine: str = Field(default="sarvam", description="STT engine identifier")
    model: str = Field(default="saaras:v4", description="Sarvam model used")
    language: Optional[str] = Field(default=None, description="Detected or configured language code")
    audio_duration: Optional[float] = Field(default=None, ge=0.0, description="Total audio duration in seconds")
    total_speakers: int = Field(default=0, ge=0, description="Total distinct speakers found in diarization")
    speaker_clusters: List[SpeakerClusterMatch] = Field(
        default_factory=list,
        description="Resolved speaker clusters with cosine similarity matches",
    )
    segments: List[SarvamDiarizedSegment] = Field(
        default_factory=list,
        description="Chronologically ordered diarized transcript segments with resolved identities",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Job and processing metadata (e.g. job_id, request_id, elapsed_seconds)",
    )

