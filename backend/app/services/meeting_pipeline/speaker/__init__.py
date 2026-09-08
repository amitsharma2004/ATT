"""Speaker embeddings, voice enrollment, voice registry, and cosine matching."""
from backend.app.services.meeting_pipeline.speaker.config import SpeakerConfig
from backend.app.services.meeting_pipeline.speaker.embedding import (
    SpeakerEmbeddingService,
    compute_centroid,
    l2_normalize,
)
from backend.app.services.meeting_pipeline.speaker.exceptions import (
    AuthenticationError,
    EmbeddingError,
    MatchingError,
    ModelUnavailableError,
    ProfileAlreadyExistsError,
    ProfileNotFoundError,
    RegistryError,
    SpeakerError,
)
from backend.app.services.meeting_pipeline.speaker.matcher import (
    SpeakerMatcher,
    cosine_similarity,
)
from backend.app.services.meeting_pipeline.speaker.models import (
    MeetingTranscriptionResult,
    NamedTranscriptSegment,
    SpeakerClusterMatch,
    SpeakerMatchCandidate,
    VoiceProfile,
)
from backend.app.services.meeting_pipeline.speaker.registry import VoiceRegistry

__all__ = [
    "SpeakerConfig",
    "SpeakerEmbeddingService",
    "l2_normalize",
    "compute_centroid",
    "VoiceProfile",
    "SpeakerMatchCandidate",
    "SpeakerClusterMatch",
    "NamedTranscriptSegment",
    "MeetingTranscriptionResult",
    "VoiceRegistry",
    "SpeakerMatcher",
    "cosine_similarity",
    "SpeakerError",
    "EmbeddingError",
    "ModelUnavailableError",
    "AuthenticationError",
    "RegistryError",
    "ProfileNotFoundError",
    "ProfileAlreadyExistsError",
    "MatchingError",
]
