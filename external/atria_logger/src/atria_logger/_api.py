"""Public API for Atria library logging.

This module exposes the public logging API for Atria users. It provides functions to:

- Retrieve a module logger.
- Configure file logging for the library.
- Adjust the logging level dynamically.

All module loggers propagate to the root logger by default.
"""

import logging

from ._constants import _ROOT_LOGGER_NAME
from ._root import get_root_adapter

__all__ = ["get_logger", "enable_file_logging", "set_atria_log_level"]


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger for a module in the library.

    Module loggers propagate to the library's root logger by default.

    Args:
        name: Optional logger name. If None, returns the root logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # if the name does not start with the root logger name, prefix it
    # we do this to ensure all loggers are under the root logger hierarchy and propagate correctly
    if name is not None and not name.startswith(f"{_ROOT_LOGGER_NAME}."):
        name = f"{_ROOT_LOGGER_NAME}.{name}"

    return logging.getLogger(name or _ROOT_LOGGER_NAME)


def enable_file_logging(
    file_path: str, log_format: str | None = None, level: int | None = None
) -> None:
    """Enable file logging for the library root logger.

    This attaches or replaces a file handler for the root logger, so that
    all library logs are written to the specified file.

    Args:
        file_path: Path to the log file.
        log_format: Optional log message format. Defaults to root logger format.
        level: Optional logging level for the file. Defaults to current root logger level.
    """
    get_root_adapter().attach_file(file_path, log_format, level)


def set_atria_log_level(level: int) -> None:
    """Set the logging level for the library root logger.

    All module loggers that propagate to the root logger will respect this level.

    Args:
        level: New logging level (e.g., logging.DEBUG, logging.INFO).
    """
    get_root_adapter().update_log_level(level)
