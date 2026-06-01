"""Unified logging service — thin wrapper around stdlib logging.

Usage:
    from finn.logger import logger
    logger.info("→ node | model=%s", model)
    logger.debug("request payload: %s", data)
    logger.warning("retry %d/%d", attempt, max_retries)
    logger.error("booking failed: %s", err)
"""

from __future__ import annotations

import logging
import sys
from typing import Any


class Logger:
    """Singleton logger for the Finn agent.

    Wraps Python's ``logging`` with a simpler interface:
      - ``debug`` / ``info`` / ``warning`` / ``error``
      - one-time ``setup(level, stream)`` configures the underlying logger
    """

    _instance: Logger | None = None

    def __new__(cls) -> Logger:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._logger: logging.Logger | None = None
        return cls._instance

    # ── setup ───────────────────────────────────────────────────────

    def setup(
        self,
        level: int = logging.INFO,
        fmt: str = "%(asctime)s %(levelname)-5s %(message)s",
        datefmt: str = "%H:%M:%S",
        stream: Any = sys.stderr,
    ) -> None:
        """Configure the underlying logger (idempotent — call once at startup)."""
        if self._logger is not None:
            return

        self._logger = logging.getLogger("finn")
        self._logger.setLevel(level)

        # Avoid duplicate handlers if setup is called again
        if not self._logger.handlers:
            handler = logging.StreamHandler(stream)
            handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
            self._logger.addHandler(handler)

        # Quiet noisy neighbours
        for noisy in ("httpx", "openai", "httpcore"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    def _ensure(self) -> logging.Logger:
        if self._logger is None:
            self.setup()
        return self._logger  # type: ignore[return-value]

    # ── public API ──────────────────────────────────────────────────

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._ensure().debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._ensure().info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._ensure().warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._ensure().error(msg, *args, **kwargs)


# Module-level singleton
logger = Logger()
