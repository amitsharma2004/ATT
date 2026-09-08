"""Domain-level exceptions for Speech-to-Text (Whisper)."""


class STTError(Exception):
    """Base exception for STT transcription errors."""
    pass


class InvalidAudioPathError(STTError):
    """Raised when the input audio file does not exist or cannot be accessed."""
    pass


class ModelUnavailableError(STTError):
    """Raised when the specified Whisper model cannot be downloaded or found."""
    pass


class ModelLoadingError(STTError):
    """Raised when the Whisper model fails to initialize into memory."""
    pass


class CUDANotAvailableError(STTError):
    """Raised when CUDA device is explicitly requested but unavailable."""
    pass


class CUDAOutOfMemoryError(STTError):
    """Raised when CUDA OOM occurs during model loading or transcription."""
    pass


class TranscriptionError(STTError):
    """Raised when transcription fails during inference or decoding."""
    pass
