"""Configuration options for Sarvam Batch Speech-to-Text Pipeline."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, SecretStr


class SarvamConfig(BaseModel):
    """Configuration specific to Sarvam Batch STT processing."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: Optional[SecretStr] = Field(
        default=None,
        description="Sarvam API subscription key",
    )
    model: str = Field(
        default="saaras:v4",
        description="Sarvam STT model identifier, e.g. saaras:v4",
    )
    language_code: str = Field(
        default="en-IN",
        description="Target language or dialect code, e.g. en-IN or unknown",
    )
    mode: str = Field(
        default="transcribe",
        description="Processing mode: transcribe or translate",
    )
    with_diarization: bool = Field(
        default=True,
        description="Whether to perform speaker diarization in Sarvam batch job",
    )
    poll_interval_seconds: int = Field(
        default=5,
        ge=1,
        description="Seconds between polling job status",
    )
    max_wait_seconds: int = Field(
        default=600,
        ge=1,
        description="Maximum seconds to wait for batch job completion before timeout",
    )

