"""Domain-level exceptions for Timestamp Alignment."""


class AlignmentError(Exception):
    """Base exception for alignment failures."""
    pass


class InvalidTimestampError(AlignmentError):
    """Raised when timestamps are mathematically invalid or corrupted."""
    pass
