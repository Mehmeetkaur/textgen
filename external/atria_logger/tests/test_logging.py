import logging
from pathlib import Path

import pytest

from atria_logger import get_logger, set_atria_log_level


# ----------------------------
# Test: Logger inherits root level
# ----------------------------
def test_logger_inherits_root_level() -> None:
    """Test that a module logger inherits its level from the root logger."""
    set_atria_log_level(logging.DEBUG)
    logger = get_logger("atria.module")
    # Child loggers have level 0, meaning they inherit root level
    assert logger.level == 0
    assert logger.name == "atria.module"
    assert logger.getEffectiveLevel() == logging.DEBUG


# ----------------------------
# Test: Debug messages propagate
# ----------------------------
def test_debug_message_propagation(caplog: pytest.LogCaptureFixture) -> None:
    """Test that DEBUG messages are captured when root level is DEBUG."""
    from atria_logger import set_atria_log_level

    set_atria_log_level(logging.DEBUG)
    logger = get_logger("atria.debugtest")
    logger.debug("Debug message")
    messages = [rec.message for rec in caplog.records]
    assert "Debug message" in messages


# ----------------------------
# Test: Info messages propagate
# ----------------------------
def test_info_message_propagation(caplog: pytest.LogCaptureFixture) -> None:
    """Test that INFO messages propagate to the captured log."""
    from atria_logger import set_atria_log_level

    set_atria_log_level(logging.DEBUG)
    logger = get_logger("atria.infotest")
    logger.info("Info message")
    messages = [rec.message for rec in caplog.records]
    assert "Info message" in messages


def test_file_logging(tmp_path: Path) -> None:
    """Test that Atria logging writes messages to a temporary file."""
    from atria_logger import enable_file_logging, set_atria_log_level

    # Prepare temp log file
    log_file = tmp_path / "atria_test.log"

    # Attach file to root logger
    enable_file_logging(str(log_file), level=logging.INFO)

    # Get a module logger (propagates to root)
    logger = get_logger("atria.filetest")

    # Ensure log level allows info messages
    set_atria_log_level(logging.INFO)

    # Log a test message
    test_msg = "File log test"
    logger.info(test_msg)

    # Flush all handlers to make sure content is written
    for handler in logger.handlers:
        if hasattr(handler, "flush"):
            handler.flush()

    # Verify file exists
    assert log_file.exists()

    # Verify content
    content = log_file.read_text()
    assert test_msg in content
