"""Domain models for Voice Profiles, Speaker Embeddings, and Identification."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfidenceTier(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    UNKNOWN = "UNKNOWN"


class VoiceProfile(BaseModel):
    """Team member voice profile with centroid embedding and sample embeddings."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: str = Field(..., min_length=1, description="Unique identifier for the team member")
    name: str = Field(..., min_length=1, description="Human readable display name, e.g. 'Arjun'")
    team_id: str = Field(default="default", description="Team / workspace identifier")
    embedding: List[float] = Field(..., min_length=1, description="L2-normalized centroid embedding vector")
    embedding_model: str = Field(..., description="Embedding model identifier, e.g. 'pyannote/embedding'")
    embedding_dimension: int = Field(..., gt=0, description="Vector dimension, e.g. 256 or 192")
    samples_count: int = Field(default=1, ge=1, description="Number of audio samples averaged into centroid")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    sample_embeddings: List[List[float]] = Field(
        default_factory=list,
        description="Individual sample embeddings for best-sample matching"
    )

    @property
    def person_id(self) -> str:
        return self.user_id

    @property
    def display_name(self) -> str:
        return self.name

    @property
    def centroid(self) -> np.ndarray:
        return np.array(self.embedding, dtype=np.float32)

    @property
    def embeddings(self) -> List[np.ndarray]:
        if self.sample_embeddings:
            return [np.array(e, dtype=np.float32) for e in self.sample_embeddings]
        return [self.centroid]

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
        """Serialize embedding vector to compact float32 binary bytes."""
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
    normalized_score: float = Field(default=0.0)
    tier: str = Field(default="UNKNOWN")


class SpeakerClusterMatch(BaseModel):
    """Final match decision for an anonymous diarization cluster (e.g. SPEAKER_00 -> Arjun)."""
    model_config = ConfigDict(frozen=True)

    cluster_speaker: str = Field(..., description="Anonymous label, e.g. 'SPEAKER_00'")
    matched_user_id: Optional[str] = Field(default=None, description="Enrolled user ID or None if UNKNOWN")
    matched_name: str = Field(..., description="Display name, e.g. 'Arjun', or 'UNKNOWN'")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    cosine_similarity: float = Field(default=0.0, ge=-1.0, le=1.0)
    normalized_score: float = Field(default=0.0)
    match_status: str = Field(..., description="'HIGH', 'MEDIUM', or 'UNKNOWN'")
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
    translated_text: Optional[str] = Field(default=None, description="Line-by-line translated English text")
    english_line: Optional[str] = Field(default=None, description="Formatted speaker line: 'Speaker: English Text'")
    alignment_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    match_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class MeetingTranscriptionResult(BaseModel):
    """Final pipeline meeting output with resolved speaker identities."""
    model_config = ConfigDict(frozen=True)

    speaker_clusters: List[SpeakerClusterMatch] = Field(default_factory=list)
    segments: List[NamedTranscriptSegment] = Field(default_factory=list)
    audio_duration: float = Field(default=0.0, ge=0.0)
    total_speakers: int = Field(default=0, ge=0)
    summary: Optional[str] = Field(default=None, description="Executive summary generated by local Llama-3.1-8B")
    metadata: Dict[str, Any] = Field(default_factory=dict)
