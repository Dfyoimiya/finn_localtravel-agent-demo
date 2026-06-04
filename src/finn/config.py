"""Unified configuration management — reads from .env via python-dotenv.

Usage:
    from finn.config import config
    model = config.llm_model  # lazy-loads .env on first access
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


class Config:
    """Singleton configuration loaded from environment / .env file.

    Access fields directly — values are read once and cached.
    """

    _instance: Config | None = None

    def __new__(cls) -> Config:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def _load(self) -> None:
        if self._loaded:
            return
        load_dotenv(Path.cwd() / ".env")
        self._llm_api_key: str = os.getenv("LLM_API_KEY", "")
        self._llm_model: str = os.getenv("LLM_MODEL", "deepseek-v4-flash")
        self._llm_base_url: str = os.getenv(
            "LLM_BASE_URL", "https://api.deepseek.com"
        )
        self._amap_api_key: str = os.getenv("AMAP_API_KEY", "")
        self._loaded = True

    # ── LLM ──────────────────────────────────────────────────────────

    @property
    def llm_api_key(self) -> str:
        self._load()
        return self._llm_api_key

    @property
    def llm_model(self) -> str:
        self._load()
        return self._llm_model

    @property
    def llm_base_url(self) -> str:
        self._load()
        return self._llm_base_url

    # ── Amap ────────────────────────────────────────────────────────

    @property
    def amap_api_key(self) -> str:
        self._load()
        return self._amap_api_key


# Module-level singleton — import this everywhere
config = Config()
