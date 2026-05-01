"""Constants for logging configuration.

This module defines constants used to configure the application's logging
system. These values are intended for internal use by the logging
subsystem.

Attributes:
    _ROOT_LOGGER_NAME (str): The name of the application's root logger.
    _DEFAULT_LOG_FORMAT (str): The default format string for log messages.
    _DEFAULT_COLOR_STYLES (dict[str, dict[str, Any]]): Default color styles
        used to format log messages at various logging levels.
    _LOG_FILE_SUFFIXES (dict[int, str]): Suffixes for log files based on
        logging levels.
"""

import logging
from typing import Any, Final

_ROOT_LOGGER_NAME: Final[str] = "atria"
_DEFAULT_LOG_FORMAT: Final[str] = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
_DEFAULT_COLOR_STYLES: Final[dict[str, dict[str, Any]]] = {
    "critical": {"bold": True, "color": "red"},
    "debug": {"color": "green"},
    "error": {"color": "red"},
    "info": {"color": "cyan"},
    "notice": {"color": "magenta"},
    "spam": {"color": "green", "faint": True},
    "success": {"bold": True, "color": "green"},
    "verbose": {"color": "blue"},
    "warning": {"color": "yellow"},
}
_LOG_FILE_SUFFIXES: Final[dict[int, str]] = {
    logging.DEBUG: "_debug.log",
    logging.INFO: "_info.log",
    logging.WARNING: "_warning.log",
    logging.ERROR: "_error.log",
    logging.CRITICAL: "_critical.log",
}
