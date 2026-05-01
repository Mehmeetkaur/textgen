import logging

from atria_logger import enable_file_logging, get_logger, set_atria_log_level

# Set global log level
set_atria_log_level(logging.DEBUG)

# Attach log file
enable_file_logging("/tmp/app.log")

# Get a module-level logger
logger = get_logger(__name__)

logger.debug("Debug message")
logger.info("Info message")
logger.error("Error message")
