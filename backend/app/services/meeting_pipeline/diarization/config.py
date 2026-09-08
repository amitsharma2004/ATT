"""Diarization configuration options."""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, SecretStr


class DiarizationConfig(BaseModel):
    """Configuration specific to the Pyannote Speaker Diarization stage."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "pyannote/speaker-diarization-3.1"
    huggingface_token: Optional[SecretStr] = None
    device: Literal["cuda", "cpu", "auto"] = "auto"
    merge_distance_sec: float = Field(
        default=0.25,
        ge=0.0,
        description="Maximum silence gap in seconds between adjacent segments of the same speaker to merge",
    )
