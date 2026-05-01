# Atria Logger

Atria Logger is a lightweight and flexible Python logging library designed to simplify
logging and monitoring application behavior. It provides an easy-to-use interface
with support for colored logs, distributed environments, and file logging.

---

## Features

- Centralized root logger for your library/application.
- Colored logs using `coloredlogs` for better readability.
- Optional file logging for different log levels.
- Environment variable support for default log level and process rank.
- Propagation-friendly module-level loggers.

---

## Installation

```bash
pip install atria_logger
```

# Usage 
```bash
import logging

from atria_logger import enable_file_logging, get_logger, set_atria_log_level

# Set global log level
set_atria_log_level(logging.DEBUG)

# Attach log file
enable_file_logging("app.log")

# Get a module-level logger
logger = get_logger(__name__)

logger.debug("Debug message")
logger.info("Info message")
logger.error("Error message")

```