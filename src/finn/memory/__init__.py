"""Finn memory system — user profile, preference tracking, and trip history.

Public API
----------
- ``MemoryManager`` — singleton for all memory operations
- ``UserProfile`` — persistent user profile model
- ``TripMemory`` — episodic trip memory model
- ``PreferenceCategory`` — individual learned preference with confidence
- ``PreferenceProfile`` — aggregated preference view
- ``ProfileBuilder`` — cold-start guided profile setup
- ``extract_learnings_from_trip`` — heuristic preference extraction
"""

from finn.memory.builder import ProfileBuilder
from finn.memory.extractor import extract_learnings_from_trip
from finn.memory.manager import MemoryManager
from finn.memory.models import PreferenceCategory, PreferenceProfile, TripMemory, UserProfile
