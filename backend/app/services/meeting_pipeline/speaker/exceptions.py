"""Domain-level exceptions for Speaker Embeddings, Enrollment, and Matching."""


class SpeakerError(Exception):
    """Base exception for all speaker-related operations."""
    pass


class EmbeddingError(SpeakerError):
    """Raised when speaker embedding extraction fails."""
    pass


class ModelUnavailableError(SpeakerError):
    """Raised when the speaker embedding model cannot be loaded or found."""
    pass


class AuthenticationError(SpeakerError):
    """Raised when Hugging Face token is missing or unauthorized for the embedding model."""
    pass


class RegistryError(SpeakerError):
    """Base exception for voice registry persistence operations."""
    pass


class ProfileNotFoundError(RegistryError):
    """Raised when a requested voice profile does not exist in registry."""
    pass


class ProfileAlreadyExistsError(RegistryError):
    """Raised when enrolling a profile that already exists."""
    pass


class MatchingError(SpeakerError):
    """Raised when matching speaker clusters against enrolled voiceprints fails."""
    pass
