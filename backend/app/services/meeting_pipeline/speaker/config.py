"""Configuration settings for Speaker Embeddings and Matching."""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, SecretStr


class SpeakerConfig(BaseModel):
    """Configuration specific to voice embeddings, registry, and matching."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    embedding_model_name: str = "pyannote/wespeaker-voxceleb-resnet34-LM"
    huggingface_token: Optional[SecretStr] = None
    device: Literal["cuda", "cpu", "auto"] = "auto"

    # Multi-tier confidence thresholds
    high_confidence_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="High confidence threshold (definitive match)",
    )
    medium_confidence_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Medium confidence threshold (sufficient to assign)",
    )
    candidate_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Candidate threshold; below this score is considered UNKNOWN",
    )

    # Exclusive assignment
    enforce_exclusive_assignment: bool = Field(
        default=True,
        description="If True, one enrolled user cannot be assigned to multiple speaker clusters in the same meeting",
    )
