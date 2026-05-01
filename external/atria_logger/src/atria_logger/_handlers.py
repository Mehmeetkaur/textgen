"""A custom logging handler that forwards log records to another logger."""

import logging


class LoggerForwardHandler(logging.Handler):  # noqa: F821
    """
    A custom logging handler that forwards log records to another logger.

    Attributes:
        target (logging.Logger): The target logger to which log records will be forwarded.
    """

    def __init__(self, target: logging.Logger):
        super().__init__()
        self.target = target

    def emit(self, record):
        print("Forwarding log record:", record)
        self.target.handle(record)
