"""
logger.py
─────────
Structured logging with different formats for dev and production.

Development: Human-readable colored output
Production:  JSON lines — easy to parse with tools like jq or ship to Datadog
"""

import logging
import logging.config
import json
import os
from pathlib import Path
from datetime import datetime

from src.utils.config_loader import cfg


class JsonFormatter(logging.Formatter):
    """Outputs each log line as a JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "time":    datetime.utcnow().isoformat() + "Z",
            "level":   record.levelname,
            "module":  record.module,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


class PrettyFormatter(logging.Formatter):
    """Colored, human-readable output for development."""

    COLORS = {
        "DEBUG":    "\033[36m",    # Cyan
        "INFO":     "\033[32m",    # Green
        "WARNING":  "\033[33m",    # Yellow
        "ERROR":    "\033[31m",    # Red
        "CRITICAL": "\033[35m",    # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        ts    = datetime.now().strftime("%H:%M:%S")
        return (
            f"{color}[{ts}] {record.levelname:<8}{self.RESET} "
            f"\033[90m{record.module}:\033[0m {record.getMessage()}"
        )


def _setup_logging() -> None:
    """Configure logging based on environment."""
    log_dir = Path(cfg.paths.logs)
    log_dir.mkdir(parents=True, exist_ok=True)

    is_dev   = cfg.app_env == "development"
    level    = getattr(logging, cfg.logging.level, logging.INFO)
    fmt_type = cfg.logging.format

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(
        PrettyFormatter() if is_dev else JsonFormatter()
    )

    # File handler (always JSON for parsing)
    log_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JsonFormatter())

    # Root logger configuration
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)   # Let handlers filter by their own level
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    # Silence noisy third-party loggers
    for noisy_lib in ["httpx", "urllib3", "chromadb", "sentence_transformers"]:
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)


# Run setup once on import
_setup_logging()


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger. Usage:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Hello!")
    """
    return logging.getLogger(name)
