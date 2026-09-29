"""Application logging configuration."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(log_file: Path, level: str = "INFO") -> None:
    """Keep detailed logs in a bounded UTF-8 file; the CLI owns terminal output."""
    resolved = Path(log_file).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(numeric_level)
    for handler in list(root.handlers):
        if getattr(handler, "_boss_react_handler", False):
            root.removeHandler(handler)
            handler.close()

    file_handler = RotatingFileHandler(
        resolved,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    file_handler._boss_react_handler = True  # type: ignore[attr-defined]

    root.addHandler(file_handler)
    logging.getLogger(__name__).info("日志已初始化: level=%s file=%s", level.upper(), resolved)
