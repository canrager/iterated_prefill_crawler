"""Shared exception types — no internal imports to avoid circular dependencies."""


class APITimeoutError(Exception):
    """Raised when API calls exhaust the retry budget (request_max_total_s)."""
