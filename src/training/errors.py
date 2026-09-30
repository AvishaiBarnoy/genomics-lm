"""Shared training-engine exceptions."""


class NonFiniteStepError(RuntimeError):
    """Raised when a loss or accumulated gradient is not finite."""


class NonFiniteGroupLimitError(RuntimeError):
    """Raised when aborted accumulation groups exceed a configured limit."""
