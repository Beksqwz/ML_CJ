"""Exceptions exposed by the backend-facing prediction library."""


class MLServiceError(Exception):
    """Base class for errors that callers can convert to API responses."""


class InvalidHorizonError(MLServiceError):
    """Raised when a horizon is not registered as a final model."""


class MissingModelError(MLServiceError):
    """Raised when the registry points to a missing or unreadable model."""


class RegistryNotFoundError(MLServiceError):
    """Raised when the final-model registry cannot be read or validated."""


class ConfigNotFoundError(MLServiceError):
    """Raised when a required feature or threshold configuration is unavailable."""


class ModelNotFoundError(MLServiceError):
    """Raised when a registered final model cannot be opened."""


class EmptySegmentListError(MLServiceError):
    """Raised when segment filtering receives an empty request list."""


class UnknownRoadSegmentError(MLServiceError):
    """Raised when a requested segment does not exist in the built city frame."""


class InvalidDatetimeError(MLServiceError):
    """Raised when a prediction hour cannot be parsed or lacks source coverage."""


class InvalidBBoxError(MLServiceError):
    """Raised when a map bounding box is malformed or outside coordinate order."""
