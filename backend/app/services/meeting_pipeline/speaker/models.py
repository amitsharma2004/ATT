"""Domain models for Voice Profiles, Speaker Embeddings, and Identification."""
from __future__ import annotations

import base64
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class VoiceProfile(BaseModel):
    """Team member voice profile with L2-normalized centroid embedding vector."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: str = Field(..., min_length=1, description="Unique identifier for the team member")
    name: str = Field(..., min_length=1, description="Human readable display name, e.g. 'Arjun'")
    team_id: str = Field(default="default", description="Team / workspace identifier")
    embedding: List[float] = Field(..., min_length=1, description="L2-normalized centroid embedding vector")
    embedding_model: str = Field(..., description="Embedding model identifier, e.g. 'pyannote/embedding'")
    embedding_dimension: int = Field(..., gt=0, description="Vector dimension, e.g. 512 or 192")
    samples_count: int = Field(default=1, ge=1, description="Number of audio samples averaged into centroid")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def validate_dimension_and_norm(self) -> VoiceProfile:
        if len(self.embedding) != self.embedding_dimension:
            raise ValueError(
                f"Embedding length ({len(self.embedding)}) does not match embedding_dimension ({self.embedding_dimension})"
            )
        arr = np.array(self.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if abs(norm - 1.0) > 0.05 and norm > 0.0:
            # Re-normalize to unit length
            unit = (arr / norm).tolist()
            object.__setattr__(self, "embedding", unit)
        return self

    def to_binary_bytes(self) -> bytes:
        """Serialize embedding vector to compact float32 binary bytes (~768 bytes for 192-d)."""
        return np.array(self.embedding, dtype=np.float32).tobytes()

    @classmethod
    def from_binary_bytes(cls, raw_bytes: bytes, **kwargs: Any) -> VoiceProfile:
        """Construct VoiceProfile from binary float32 bytes."""
        arr = np.frombuffer(raw_bytes, dtype=np.float32).tolist()
        return cls(embedding=arr, embedding_dimension=len(arr), **kwargs)


class SpeakerMatchCandidate(BaseModel):
    """Candidate match evaluation for a diarization speaker cluster."""
    model_config = ConfigDict(frozen=True)

    user_id: str
    name: str
    cosine_similarity: float = Field(..., ge=-1.0, le=1.0)


class SpeakerClusterMatch(BaseModel):
    """Final match decision for an anonymous diarization cluster (e.g. SPEAKER_00 -> Arjun)."""
    model_config = ConfigDict(frozen=True)

    cluster_speaker: str = Field(..., description="Anonymous label, e.g. 'SPEAKER_00'")
    matched_user_id: Optional[str] = Field(default=None, description="Enrolled user ID or None if UNKNOWN")
    matched_name: str = Field(..., description="Display name, e.g. 'Arjun', or 'UNKNOWN'")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    cosine_similarity: float = Field(default=0.0, ge=-1.0, le=1.0)
    match_status: str = Field(..., description="'HIGH_CONFIDENCE', 'MEDIUM_CONFIDENCE', or 'UNKNOWN'")
    candidates: List[SpeakerMatchCandidate] = Field(default_factory=list)


class NamedTranscriptSegment(BaseModel):
    """Transcript segment with resolved team member identity."""
    model_config = ConfigDict(frozen=True)

    speaker_name: str = Field(..., description="Resolved identity name (e.g. 'Arjun') or 'UNKNOWN'")
    speaker_cluster: str = Field(..., description="Original diarization label (e.g. 'SPEAKER_00')")
    user_id: Optional[str] = Field(default=None, description="Enrolled user_id if matched")
    start: float = Field(..., ge=0.0)
    end: float = Field(..., gt=0.0)
    text: str = Field(..., min_length=1)
    alignment_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    match_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class MeetingTranscriptionResult(BaseModel):
    """Final pipeline meeting output with resolved speaker identities."""
    model_config = ConfigDict(frozen=True)

    speaker_clusters: List[SpeakerClusterMatch] = Field(default_factory=list)
    segments: List[NamedTranscriptSegment] = Field(default_factory=list)
    audio_duration: float = Field(default=0.0, ge=0.0)
    total_speakers: int = Field(default=0, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
