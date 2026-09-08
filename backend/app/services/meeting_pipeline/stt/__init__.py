"""Speech-to-Text package using OpenAI Whisper."""
from backend.app.services.meeting_pipeline.stt.config import STTConfig
from backend.app.services.meeting_pipeline.stt.exceptions import (
    CUDANotAvailableError,
    CUDAOutOfMemoryError,
    InvalidAudioPathError,
    ModelLoadingError,
    ModelUnavailableError,
    STTError,
    TranscriptionError,
)
from backend.app.services.meeting_pipeline.stt.models import (
    TranscriptionResult,
    TranscriptionSegment,
)
from backend.app.services.meeting_pipeline.stt.whisper_service import WhisperTranscriber

__all__ = [
    "STTConfig",
    "WhisperTranscriber",
    "TranscriptionSegment",
    "TranscriptionResult",
    "STTError",
    "InvalidAudioPathError",
    "ModelUnavailableError",
    "ModelLoadingError",
    "CUDANotAvailableError",
    "CUDAOutOfMemoryError",
    "TranscriptionError",
]
