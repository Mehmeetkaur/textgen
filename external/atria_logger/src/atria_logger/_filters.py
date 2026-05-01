"""Custom application-level logging filters.

This module defines a logging filter intended for distributed applications.
Only the process with rank ``DEFAULT_LOGGING_RANK`` will emit logs when this
filter is used.

Module attributes:
    DEFAULT_LOGGING_RANK (int): Rank value that is considered the "main"
        process for logging purposes.
"""

import logging
from typing import Final

_DEFAULT_LOGGING_RANK: Final[int] = 0


class DistributedFilter(logging.Filter):
    """Logging filter for distributed processes.

    This filter allows records only when the current process rank equals the
    configured `DEFAULT_LOGGING_RANK`.

    Attributes:
        _rank (int): Rank of the current process (private).
    """

    def __init__(self, rank: int) -> None:
        """Initialize the filter.

        Args:
            rank: The rank of the current process.
        """
        super().__init__()
        self._rank = rank

    @property
    def rank(self) -> int:
        """int: The rank of this filter's process (read-only)."""
        return self._rank

    def filter(self, record: logging.LogRecord) -> bool:
        """Determine whether a log record is allowed.

        Args:
            record: The logging.LogRecord to evaluate.

        Returns:
            True if this process's rank equals `DEFAULT_LOGGING_RANK`, False otherwise.
        """
        return self._rank == _DEFAULT_LOGGING_RANK

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"{self.__class__.__name__}(rank={self._rank})"
