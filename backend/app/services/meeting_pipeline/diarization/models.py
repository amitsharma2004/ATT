"""Data models for Speaker Diarization."""
from __future__ import annotations

from typing import List
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SpeakerSegment(BaseModel):
    """Represents a validated, timestamped speaker turn."""
    model_config = ConfigDict(frozen=True)

    speaker: str = Field(..., min_length=1, description="Anonymous speaker identifier, e.g. SPEAKER_00")
    start: float = Field(..., ge=0.0, description="Start time in seconds")
    end: float = Field(..., gt=0.0, description="End time in seconds")
    duration: float = Field(..., gt=0.0, description="Duration in seconds")

    @model_validator(mode="before")
    @classmethod
    def validate_and_compute_duration(cls, data: dict) -> dict:
        if not isinstance(data, dict):
            return data
        start = data.get("start")
        end = data.get("end")
        if start is not None and end is not None:
            if start < 0:
                raise ValueError("start time must be >= 0")
            if end <= start:
                raise ValueError(f"end time ({end}) must be strictly greater than start time ({start})")
            if "duration" not in data or data["duration"] is None:
                data["duration"] = round(end - start, 4)
            elif data["duration"] <= 0:
                raise ValueError("duration must be > 0")
        speaker = data.get("speaker")
        if not speaker or not str(speaker).strip():
            raise ValueError("speaker identifier must not be empty")
        return data


class DiarizationResult(BaseModel):
    """Complete output of a diarization run on an audio file."""
    model_config = ConfigDict(frozen=True)

    segments: List[SpeakerSegment] = Field(default_factory=list)
    total_speakers: int = Field(ge=0)
    audio_duration: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_chronological_order(self) -> DiarizationResult:
        for i in range(len(self.segments) - 1):
            curr = self.segments[i]
            nxt = self.segments[i + 1]
            if nxt.start < curr.start:
                raise ValueError(
                    f"Diarization segments must be chronologically ordered. "
                    f"Segment {i+1} starts at {nxt.start} before segment {i} at {curr.start}"
                )
        return self
