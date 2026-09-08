"""Speaker diarization package using pyannote."""
from backend.app.services.meeting_pipeline.diarization.config import DiarizationConfig
from backend.app.services.meeting_pipeline.diarization.diarizer import SpeakerDiarizer
from backend.app.services.meeting_pipeline.diarization.exceptions import (
    DiarizationAuthenticationError,
    DiarizationError,
    DiarizationInferenceError,
    InvalidAudioPathError,
    ModelUnavailableError,
    PyannoteNotInstalledError,
)
from backend.app.services.meeting_pipeline.diarization.models import (
    DiarizationResult,
    SpeakerSegment,
)

__all__ = [
    "DiarizationConfig",
    "SpeakerDiarizer",
    "SpeakerSegment",
    "DiarizationResult",
    "DiarizationError",
    "PyannoteNotInstalledError",
    "ModelUnavailableError",
    "DiarizationAuthenticationError",
    "InvalidAudioPathError",
    "DiarizationInferenceError",
]
