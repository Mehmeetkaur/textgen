"""
Unit Tests for Logger Handlers

This module contains unit tests for the logger handler configuration utilities
provided by the `attach_file_handler` and `attach_stream_handler` functions.
These tests ensure that handlers are correctly added to loggers and that log
messages are properly written to files or streams.

Test Functions:
    - test_configure_file_handler: Tests adding a file handler to a logger.
    - test_configure_stream_handler: Tests adding a stream handler to a logger.

Dependencies:
    - pytest: For running the test cases.
    - logging: For verifying logger behavior.
    - sys: For stream handling.
    - atria_logger.utilities: Provides the `attach_file_handler` and `attach_stream_handler` functions.

Author: Saifullah (saifullah3396@gmail.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import logging
import pathlib
import sys

import pytest

from atria_logger._utilities import _attach_file_handler, _attach_stream_handler


def test_configure_file_handler(tmp_path: pathlib.Path) -> None:
    """
    Test configuring a file handler for a logger.

    Ensures that a file handler is added to the logger and that log messages
    are correctly written to the specified file.

    Args:
        tmp_path (Path): A temporary directory provided by pytest for testing.

    Assertions:
        - The log file should be created at the specified path.
        - The log file should contain the expected log message.
    """
    log_file = tmp_path / "test.log"  # Use pytest's tmp_path for temporary files
    logger = logging.getLogger("test.file")

    # Set logger level to INFO so that INFO messages are not filtered out
    logger.setLevel(logging.INFO)

    # Add the file handler
    _attach_file_handler(logger, str(log_file), logging.INFO, "%(message)s")

    # Log a message
    logger.info("This is a test log message.")

    # Check if the log file has been created and contains the log message
    with open(log_file) as f:
        log_content = f.read()
        assert "This is a test log message." in log_content


def test_configure_stream_handler(caplog: pytest.LogCaptureFixture) -> None:  # noqa: F821
    """
    Test configuring a stream handler for a logger.

    Ensures that a stream handler logs messages to the correct stream.

    Args:
        caplog (pytest.LogCaptureFixture): A pytest fixture for capturing log messages.

    Assertions:
        - The captured log output should contain the expected log message.
    """
    logger = logging.getLogger("test.stream")
    _attach_stream_handler(logger, sys.stdout, logging.INFO, "%(message)s")

    with caplog.at_level(logging.INFO, logger="test.stream"):
        logger.info("Stream message test")

    assert "Stream message test" in caplog.text
