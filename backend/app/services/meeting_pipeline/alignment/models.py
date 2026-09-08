"""Data models for Transcript and Diarization Alignment."""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SpeakerCandidateEvidence(BaseModel):
    """Evidence of temporal overlap for a candidate speaker on a transcript segment."""
    model_config = ConfigDict(frozen=True)

    speaker: str = Field(..., description="Candidate speaker identifier")
    overlap_duration: float = Field(..., ge=0.0, description="Total overlap duration in seconds")
    overlap_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ratio of overlap duration relative to transcript segment duration",
    )


class AlignedTranscriptSegment(BaseModel):
    """Represents a transcript segment bound to an anonymous speaker via temporal overlap."""
    model_config = ConfigDict(frozen=True)

    speaker: str = Field(
        ...,
        min_length=1,
        description="Assigned speaker label (e.g. 'SPEAKER_00', or 'UNKNOWN')",
    )
    start: float = Field(..., ge=0.0, description="Start timestamp in seconds")
    end: float = Field(..., gt=0.0, description="End timestamp in seconds")
    text: str = Field(..., min_length=1, description="Transcribed spoken text")
    alignment_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Deterministic confidence based on dominance and overlap ratio",
    )
    overlap_duration: float = Field(
        default=0.0,
        ge=0.0,
        description="Temporal overlap with the assigned speaker in seconds",
    )
    overlap_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Proportion of segment duration covered by the assigned speaker",
    )
    candidates: List[SpeakerCandidateEvidence] = Field(
        default_factory=list,
        description="Candidate speaker overlap evidence evaluated during alignment",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_bounds(cls, data: dict) -> dict:
        if not isinstance(data, dict):
            return data
        start = data.get("start")
        end = data.get("end")
        text = data.get("text")
        if start is not None and end is not None:
            if float(start) < 0.0:
                raise ValueError("start timestamp must be >= 0.0")
            if float(end) <= float(start):
                raise ValueError(
                    f"end timestamp ({end}) must be strictly greater than start ({start})"
                )
            data["start"] = round(float(start), 3)
            data["end"] = round(float(end), 3)
        if text is None or not str(text).strip():
            raise ValueError("text must not be empty or whitespace only")
        data["text"] = str(text).strip()
        return data


class AlignedTranscriptResult(BaseModel):
    """Complete aligned meeting transcript."""
    model_config = ConfigDict(frozen=True)

    segments: List[AlignedTranscriptSegment] = Field(default_factory=list)
    total_speakers: int = Field(default=0, ge=0)
    audio_duration: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_chronological_order(self) -> AlignedTranscriptResult:
        for i in range(len(self.segments) - 1):
            curr = self.segments[i]
            nxt = self.segments[i + 1]
            if nxt.start < curr.start:
                raise ValueError(
                    f"Aligned segments must be chronologically ordered. "
                    f"Segment {i+1} starts at {nxt.start} before segment {i} at {curr.start}"
                )
        return self
