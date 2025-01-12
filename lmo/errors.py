"""Custom exceptions and warnings."""

__all__ = ("InvalidLMomentError",)


class InvalidLMomentError(ValueError):
    """L-moment(s) not within the theoretical bounds."""
