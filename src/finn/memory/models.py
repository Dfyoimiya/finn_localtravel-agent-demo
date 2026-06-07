"""Memory and user profile Pydantic models.

These models define the persistent data structures for user profile,
preference tracking, and trip memory. They are serialised to JSON
under ``~/.finn/``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from finn.state import PartyMember


class PreferenceCategory(BaseModel):
    """A single learned preference item with confidence tracking.

    The same (category, value) pair that appears in multiple trips
    gets its confidence boosted and occurrences incremented.
    """

    category: Literal[
        "dining", "activity", "transport", "budget", "time", "area", "constraint", "general"
    ]
    value: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source: Literal["explicit", "inferred", "behavioral"] = "inferred"
    occurrences: int = 1
    first_seen: str = ""
    last_seen: str = ""


class PreferenceProfile(BaseModel):
    """Aggregated preferences and constraints learned from trip history.

    The top-level list fields are *denormalised convenience views*
    built from ``learned_items`` entries whose confidence >= 0.6.
    ``learned_items`` is the canonical source of truth.
    """

    dining: list[str] = Field(default_factory=list)
    activities: list[str] = Field(default_factory=list)
    transport: list[str] = Field(default_factory=list)
    preferred_area: list[str] = Field(default_factory=list)
    hard_constraints: list[str] = Field(default_factory=list)
    budget_range: tuple[float, float] | None = None
    preferred_time: str | None = None

    learned_items: list[PreferenceCategory] = Field(default_factory=list)


class UserProfile(BaseModel):
    """Persistent user profile stored in ``~/.finn/profile.json``.

    Reuses ``PartyMember`` from ``finn.state`` for saved companions.
    """

    version: int = 1
    created_at: str = ""
    updated_at: str = ""

    # Identifiers
    name: str = ""

    # Spatial defaults
    home_location: str | None = None
    work_location: str | None = None
    home_coords: str | None = None  # "116.46,39.90"

    # Party defaults
    default_party_size: int = 1
    saved_party_members: list[PartyMember] = Field(default_factory=list)

    # Preferences (aggregate)
    preferences: PreferenceProfile = Field(default_factory=PreferenceProfile)

    # Statistics
    total_trips: int = 0
    completed_trips: int = 0
    cancelled_trips: int = 0
    favorite_scenarios: dict[str, int] = Field(default_factory=dict)

    # Lifecycle
    setup_complete: bool = False


class TripMemory(BaseModel):
    """Episodic memory of a single completed trip.

    Stores a *reduced* copy of the extraction result — flat fields rather
    than the full nested Pydantic model so the on-disk format is stable
    even as the ExtractResult model evolves.
    """

    id: str = ""
    created_at: str = ""
    intent_summary: str = ""
    scenario: str = ""
    activity: str = ""
    date: str | None = None
    area: str | None = None
    party_size: int | None = None
    budget_total: float | None = None
    plan_notes: str = ""
    outcome: Literal["completed", "cancelled", "failed"] = "completed"
    user_feedback: str = ""
    extracted_learnings: list[PreferenceCategory] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
