"""Audio processing specific exceptions."""


class AudioError(Exception):
    """Base exception for audio processing failures."""
    pass


class AudioValidationError(AudioError):
    """Raised when audio fails file validation or constraints."""
    pass


class AudioConversionError(AudioError):
    """Raised when FFmpeg conversion or extraction fails."""
    pass
