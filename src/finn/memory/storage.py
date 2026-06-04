"""Low-level JSON file I/O for the memory system.

All data lives under ``~/.finn/``.  This module provides the raw
read/write helpers; higher-level logic lives in ``manager.py``.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from finn.logger import logger


def data_dir() -> Path:
    """Return ``~/.finn/``, creating it and the ``trips/`` sub-directory
    if they do not exist."""
    path = Path.home() / ".finn"
    trips = path / "trips"
    trips.mkdir(parents=True, exist_ok=True)
    return path


def read_json(filepath: Path) -> dict:
    """Read a JSON file. Returns an empty dict if the file does not exist."""
    if not filepath.exists():
        return {}
    try:
        return json.loads(filepath.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s", filepath, exc)
        return {}


def write_json(filepath: Path, data: dict) -> None:
    """Atomically write a JSON file (temp file + rename)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=filepath.parent,
            prefix=filepath.name + ".",
            delete=False,
        )
        json.dump(data, tmp, ensure_ascii=False, indent=2, default=str)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp.name, filepath)
    except OSError as exc:
        logger.error("Failed to write %s: %s", filepath, exc)
        # Clean up temp file if it exists
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


def profile_path(data_dir: Path) -> Path:
    """Path to ``~/.finn/profile.json``."""
    return data_dir / "profile.json"


def list_trip_files(data_dir: Path) -> list[Path]:
    """Return sorted list of trip JSON file paths, newest first."""
    trips_dir = data_dir / "trips"
    if not trips_dir.exists():
        return []
    files = list(trips_dir.glob("*.json"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files
