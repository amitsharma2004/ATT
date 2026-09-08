"""Data models for Speech-to-Text (Whisper)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TranscriptionSegment(BaseModel):
    """Represents a validated, timestamped transcript segment."""
    model_config = ConfigDict(frozen=True)

    start: float = Field(..., ge=0.0, description="Start timestamp in seconds")
    end: float = Field(..., gt=0.0, description="End timestamp in seconds")
    text: str = Field(..., min_length=1, description="Transcribed text content")

    @model_validator(mode="before")
    @classmethod
    def validate_segment_bounds_and_text(cls, data: dict) -> dict:
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
                    f"end timestamp ({end}) must be strictly greater than start timestamp ({start})"
                )

        if text is None or not str(text).strip():
            raise ValueError("text content must not be empty or whitespace only")

        data["text"] = str(text).strip()
        data["start"] = round(float(start), 3)
        data["end"] = round(float(end), 3)
        return data


class TranscriptionResult(BaseModel):
    """Complete structured output of a transcription run."""
    model_config = ConfigDict(frozen=True)

    segments: List[TranscriptionSegment] = Field(default_factory=list)
    language: Optional[str] = Field(
        default=None,
        description="Detected spoken language (e.g. 'en', 'hi', 'ta')",
    )
    duration_seconds: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Audio duration in seconds",
    )
    model_name: str = Field(..., description="Whisper model name used")
    device_used: str = Field(..., description="Device used for inference (cuda or cpu)")
    full_text: str = Field(
        default="",
        description="Concatenated transcript text across all segments",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional execution metadata (e.g. chunk length, compute type)",
    )

    @model_validator(mode="after")
    def validate_chronological_order_and_compute_full_text(self) -> TranscriptionResult:
        for i in range(len(self.segments) - 1):
            curr = self.segments[i]
            nxt = self.segments[i + 1]
            if nxt.start < curr.start:
                raise ValueError(
                    f"Transcription segments must be chronologically ordered. "
                    f"Segment {i+1} starts at {nxt.start} before segment {i} at {curr.start}"
                )

        if not self.full_text and self.segments:
            object.__setattr__(
                self,
                "full_text",
                " ".join(seg.text for seg in self.segments if seg.text),
            )

        return self
