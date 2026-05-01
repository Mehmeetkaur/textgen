"""Internal root logger setup for the Atria library.

This module defines the internal root logger and its adapter for the Atria library.
It ensures consistent logger configuration across the library, including:

- Distributed logging filtering based on process rank.
- Colored console logging.
- Optional file logging.
- Dynamic log level updates.

This module is intended for internal use only.
"""

import logging
import os
from pathlib import Path

from ._constants import _DEFAULT_COLOR_STYLES, _DEFAULT_LOG_FORMAT, _ROOT_LOGGER_NAME
from ._filters import DistributedFilter
from ._utilities import _attach_file_handler, _enable_colored_logging, _reset_logger

# Module-level root logger
_root_logger: logging.Logger = logging.getLogger(_ROOT_LOGGER_NAME)


class RootLoggerAdapter(logging.LoggerAdapter[logging.Logger]):
    """Adapter for the internal root logger of Atria.

    Ensures the root logger is configured once with colored console logging,
    distributed filter, and optional file logging. Provides methods to update
    log level and attach a log file dynamically.

    Attributes:
        _configured (bool): Whether the root logger has been configured.
        _file_handler (Optional[logging.FileHandler]): File handler for root logger.
    """

    def __init__(self, logger: logging.Logger) -> None:
        """Initialize the RootLoggerAdapter.

        Args:
            logger (logging.Logger): The internal root logger to wrap.
        """
        super().__init__(logger, extra={})
        self._configured: bool = False
        self._file_handler: logging.FileHandler | None = None
        self._init_env()
        self.configure()

    def _init_env(self) -> None:
        """Initialize the root logger's level from the ATRIA_LOG_LEVEL environment variable.

        If the environment variable is invalid or missing, defaults to INFO.
        """
        self._log_level = os.environ.get("ATRIA_LOG_LEVEL", "INFO").upper()
        self._rank: int = int(os.environ.get("RANK", 0))

    def configure(self) -> None:
        """Configure the root logger once.

        Adds a distributed logging filter and enables colored console logging.
        This method is idempotent.
        """
        if self._configured:
            return

        try:
            self.logger.setLevel(getattr(logging, self._log_level))
        except AttributeError:
            logging.warning(
                "Invalid ATRIA_LOG_LEVEL '%s' specified. Defaulting to INFO.",
                self._log_level,
            )
            self.logger.setLevel(logging.INFO)

        _enable_colored_logging(
            logger=self.logger,
            log_level=self.logger.level,
            styles=_DEFAULT_COLOR_STYLES,
            log_format=_DEFAULT_LOG_FORMAT,
        )

        # since parent filters are not applied to child propagated logs,
        # we need to add the distributed filter to each handler
        for handler in self.logger.handlers:
            if hasattr(handler, "setLevel"):
                handler.addFilter(DistributedFilter(rank=self._rank))

        self._configured = True

    def update_log_level(self, level: int) -> None:
        """Update the root logger level dynamically.

        Args:
            level (int): New logging level (e.g., logging.DEBUG, logging.INFO).
        """
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            if hasattr(handler, "setLevel"):
                handler.setLevel(level)

    def attach_file(
        self, path: str, log_format: str | None = None, level: int | None = None
    ) -> None:
        """Attach or replace a file handler for the root logger.

        Args:
            path (str): File path to write logs to.
            log_format (Optional[str]): Log message format. Defaults to `_DEFAULT_LOG_FORMAT`.
            level (Optional[int]): Logging level for the file handler. Defaults to the current root logger level.
        """
        log_format = log_format or _DEFAULT_LOG_FORMAT
        level = level or self.logger.level

        if self._file_handler:
            self.logger.removeHandler(self._file_handler)

        file_path = Path(path)
        self._file_handler = _attach_file_handler(
            self.logger, str(file_path), level, log_format
        )

        self._file_handler.addFilter(DistributedFilter(rank=self._rank))


# Module-level adapter instance
_root_adapter: RootLoggerAdapter = RootLoggerAdapter(_root_logger)


def get_root_adapter() -> RootLoggerAdapter:
    """Return the module-level root logger adapter.

    Returns:
        RootLoggerAdapter: The singleton root logger adapter instance.
    """
    return _root_adapter


def reload_adapter() -> None:
    """Recreate the root adapter to pick up environment variable changes.

    Useful in tests or when environment variables affecting logging change.
    """
    global _root_adapter
    _reset_logger(_root_logger)
    _root_adapter = RootLoggerAdapter(_root_logger)
