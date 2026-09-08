"""Audio domain models for the STT & Speaker Identification Pipeline."""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class AudioFormat(str, Enum):
    """Supported audio format extensions."""
    WAV = "wav"
    MP3 = "mp3"
    M4A = "m4a"
    FLAC = "flac"
    OGG = "ogg"
    AAC = "aac"
    WMA = "wma"


class AudioMetadata(BaseModel):
    """Metadata extracted from or representing an audio file."""
    model_config = ConfigDict(frozen=True)

    file_path: Path
    file_name: str
    file_size_bytes: int = Field(ge=0)
    duration_seconds: float = Field(ge=0.0)
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    format: str


class AudioValidationResult(BaseModel):
    """Result of validating an uploaded/provided audio file."""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    metadata: Optional[AudioMetadata] = None


class QualityReport(BaseModel):
    """Audio quality metrics and issue flags."""
    model_config = ConfigDict(frozen=True)

    loudness_dbfs: float = 0.0
    clipping_ratio: float = 0.0
    noise_floor_dbfs: float = 0.0
    silence_ratio: float = 0.0
    is_clipped: bool = False
    is_too_quiet: bool = False
    is_noisy: bool = False
    is_mostly_silent: bool = False
    notes: List[str] = Field(default_factory=list)


class VADResult(BaseModel):
    """Voice Activity Detection result across the entire audio."""
    model_config = ConfigDict(frozen=True)

    speech_ratio: Optional[float] = None
    has_speech: Optional[bool] = None
    segments: List[tuple[float, float]] = Field(default_factory=list)


class ProcessedAudio(BaseModel):
    """Normalized & standardized audio artifact ready for downstream pipeline ingestion."""
    model_config = ConfigDict(frozen=True)

    job_id: str
    processed_path: Path
    original_path: Path
    duration_seconds: float = Field(ge=0.0)
    sample_rate: int = Field(default=16000)
    channels: int = Field(default=1)
    format: str = "wav"
    quality_report: Optional[QualityReport] = None
    vad_result: Optional[VADResult] = None
