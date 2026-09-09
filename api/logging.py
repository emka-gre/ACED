"""Structured logging, one JSON object per line.

Analyses run in background threads, so a log line that does not say which analysis it belongs
to is close to useless once two run at once. `analysis_id` is threaded through as a logging
extra and lands as a field rather than being interpolated into the message, so the log can be
filtered on it.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

LOGGER_NAME = "aced.api"

#: Attributes `logging` puts on every record; anything else a caller passed as an `extra`
#: is ours and belongs in the JSON.
_STANDARD_FIELDS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None)))


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_FIELDS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Attach a JSON handler to the service logger, once."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    # The service's records are its own; letting them propagate would double them up under
    # uvicorn, which configures the root logger with its own handler.
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)
