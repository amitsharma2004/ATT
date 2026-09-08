"""Domain-level exceptions for Speaker Diarization."""


class DiarizationError(Exception):
    """Base exception for all diarization-related errors."""
    pass


class PyannoteNotInstalledError(DiarizationError):
    """Raised when pyannote.audio or torch is not installed in the environment."""
    pass


class ModelUnavailableError(DiarizationError):
    """Raised when the specified diarization model cannot be found or downloaded."""
    pass


class DiarizationAuthenticationError(DiarizationError):
    """Raised when Hugging Face token is missing, invalid, or lacks gated model access."""
    pass


class InvalidAudioPathError(DiarizationError):
    """Raised when the audio file path is non-existent or invalid."""
    pass


class DiarizationInferenceError(DiarizationError):
    """Raised when model inference or pipeline execution fails on the audio."""
    pass
