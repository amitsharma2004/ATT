"""Domain-level exceptions for Sarvam Batch Speech-to-Text Pipeline."""
from __future__ import annotations


class SarvamError(Exception):
    """Base exception for all Sarvam pipeline operations."""
    pass


class SarvamAPIKeyMissingError(SarvamError):
    """Raised when SARVAM_API_KEY is not configured or empty."""
    pass


class SarvamClientError(SarvamError):
    """Raised when Sarvam SDK client fails to initialize or authentication fails."""
    pass


class SarvamJobCreationError(SarvamError):
    """Raised when creating a batch job on Sarvam fails."""
    pass


class SarvamFileUploadError(SarvamError):
    """Raised when uploading audio files to Sarvam fails."""
    pass


class SarvamJobStartError(SarvamError):
    """Raised when starting the created batch job fails."""
    pass


class SarvamJobTimeoutError(SarvamError):
    """Raised when batch job execution exceeds maximum wait timeout."""
    pass


class SarvamJobProcessingError(SarvamError):
    """Raised when Sarvam reports that the batch job has failed."""
    pass


class SarvamOutputDownloadError(SarvamError):
    """Raised when downloading output artifacts from Sarvam fails or produces no files."""
    pass


class SarvamParserError(SarvamError):
    """Raised when downloaded Sarvam output cannot be parsed into domain transcript models."""
    pass
