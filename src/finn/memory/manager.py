"""MemoryManager singleton — central access point for all memory operations.

Follows the same singleton pattern as ``finn.config.Config``.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from finn.logger import logger
from finn.memory.models import PreferenceCategory, PreferenceProfile, TripMemory, UserProfile
from finn.memory.storage import data_dir, list_trip_files, profile_path, read_json, write_json

# Beijing timezone
CST = timezone(timedelta(hours=8))


def _now_iso() -> str:
    return datetime.now(CST).isoformat()


# ── Preference rebuild ──────────────────────────────────────────────────


def _rebuild_denormalised(preferences: PreferenceProfile) -> None:
    """Rebuild top-level list fields from ``learned_items`` with confidence >= 0.6."""
    dining: list[str] = []
    activities: list[str] = []
    transport: list[str] = []
    preferred_area: list[str] = []
    hard_constraints: list[str] = []

    for item in preferences.learned_items:
        if item.confidence < 0.6:
            continue
        if item.category == "dining" and item.value not in dining:
            dining.append(item.value)
        elif item.category == "activity" and item.value not in activities:
            activities.append(item.value)
        elif item.category == "transport" and item.value not in transport:
            transport.append(item.value)
        elif item.category == "area" and item.value not in preferred_area:
            preferred_area.append(item.value)
        elif item.category == "constraint" and item.value not in hard_constraints:
            hard_constraints.append(item.value)

    preferences.dining = dining
    preferences.activities = activities
    preferences.transport = transport
    preferences.preferred_area = preferred_area
    preferences.hard_constraints = hard_constraints


# ── Singleton ───────────────────────────────────────────────────────────


class MemoryManager:
    """Singleton for loading, saving, and updating user memory.

    Usage::

        mm = MemoryManager()
        profile = mm.load_profile()
        mm.update_preferences([...])
        ctx = mm.build_profile_context()
    """

    _instance: MemoryManager | None = None

    def __new__(cls) -> MemoryManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__initialised = False
        return cls._instance

    def __init__(self) -> None:
        if self.__initialised:
            return
        self._data_dir: Path = data_dir()
        self._profile: UserProfile | None = None
        self._last_decay_check: str = ""
        self.__initialised = True

    # ── Profile ────────────────────────────────────────────────────────

    def profile_exists(self) -> bool:
        """True if profile.json exists *and* setup is complete."""
        path = profile_path(self._data_dir)
        if not path.exists():
            return False
        data = read_json(path)
        return data.get("setup_complete", False)

    def load_profile(self) -> UserProfile:
        """Return cached profile or load from disk.

        Returns a default (empty) profile if no file exists.
        Runs decay check on load.
        """
        if self._profile is not None:
            return self._profile

        path = profile_path(self._data_dir)
        data = read_json(path)

        if not data:
            self._profile = self._create_default_profile()
        else:
            try:
                self._profile = UserProfile(**data)
            except Exception as exc:
                logger.warning("Corrupt profile.json, creating fresh: %s", exc)
                # Back up corrupt file
                backup = path.with_suffix(".json.bak")
                path.rename(backup)
                self._profile = self._create_default_profile()

        # Periodic decay check
        self.decay_preferences()
        return self._profile

    def save_profile(self, profile: UserProfile) -> None:
        """Write profile to disk and update in-memory cache."""
        profile.updated_at = _now_iso()
        self._profile = profile
        path = profile_path(self._data_dir)
        write_json(path, profile.model_dump())

    def create_default_profile(self) -> UserProfile:
        """Return a blank profile with setup_complete=False."""
        return self._create_default_profile()

    def _create_default_profile(self) -> UserProfile:
        p = UserProfile(
            created_at=_now_iso(),
            updated_at=_now_iso(),
            setup_complete=False,
        )
        self._profile = p
        return p

    # ── Trip Memory ──────────────────────────────────────────────────

    def save_trip(self, trip: TripMemory) -> None:
        """Persist a trip memory and increment profile statistics."""
        profile = self.load_profile()

        # Update stats
        profile.total_trips += 1
        if trip.outcome == "completed":
            profile.completed_trips += 1
        elif trip.outcome == "cancelled":
            profile.cancelled_trips += 1

        # Track favorite scenarios
        scenario = trip.scenario
        if scenario:
            profile.favorite_scenarios[scenario] = (
                profile.favorite_scenarios.get(scenario, 0) + 1
            )

        # Update saved party members if new ones appeared
        if not profile.saved_party_members and trip.party_size and trip.party_size > 1:
            # Party members are stored in the ExtractResult, not in TripMemory
            pass  # Handled at the call site in cli.py

        # Persist trip file
        trips_dir = self._data_dir / "trips"
        filename = f"{trip.created_at[:10]}_{trip.id}.json"
        filepath = trips_dir / filename
        write_json(filepath, trip.model_dump())

        self.save_profile(profile)
        logger.debug("Saved trip %s to %s", trip.id, filepath)

    def get_recent_trips(self, n: int = 5) -> list[TripMemory]:
        """Load the *n* most recent trips from disk."""
        trips: list[TripMemory] = []
        for path in list_trip_files(self._data_dir)[:n]:
            data = read_json(path)
            if data:
                try:
                    trips.append(TripMemory(**data))
                except Exception as exc:
                    logger.warning("Corrupt trip file %s: %s", path, exc)
        return trips

    def get_similar_trips(
        self, activity_keywords: str, limit: int = 3
    ) -> list[TripMemory]:
        """Find trips with overlapping activity or area tags.

        Simple keyword-overlap scoring — no vector DB needed.
        """
        scored: list[tuple[int, TripMemory]] = []
        keywords = activity_keywords.lower()
        for trip in self.get_recent_trips(20):
            score = 0
            text = (trip.activity + " " + (trip.area or "") + " " + trip.intent_summary).lower()
            for kw in keywords.split():
                if kw in text:
                    score += 1
            if score > 0:
                scored.append((score, trip))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:limit]]

    # ── Preferences ──────────────────────────────────────────────────

    def update_preferences(self, new_items: list[PreferenceCategory]) -> None:
        """Merge new preference items into the profile.

        Same (category, value) → boost confidence & occurrences.
        New items → append with their original confidence.
        """
        if not new_items:
            return

        profile = self.load_profile()
        learned = profile.preferences.learned_items
        now = _now_iso()

        for new_item in new_items:
            new_item.last_seen = now
            if not new_item.first_seen:
                new_item.first_seen = now

            # Find existing match by (category, value)
            match = _find_item(learned, new_item.category, new_item.value)
            if match is not None:
                match.occurrences += 1
                match.confidence = min(1.0, match.confidence + 0.10)
                match.last_seen = now
                # Upgrade source if new item is more authoritative
                if _source_rank(new_item.source) > _source_rank(match.source):
                    match.source = new_item.source
            else:
                learned.append(new_item)

        # Cap at 50 items, pruning lowest confidence first
        if len(learned) > 50:
            learned.sort(key=lambda x: x.confidence)
            removed = learned[: len(learned) - 50]
            learned = learned[len(learned) - 50 :]
            logger.debug("Pruned %d low-confidence preference items", len(removed))

        profile.preferences.learned_items = learned
        _rebuild_denormalised(profile.preferences)
        self.save_profile(profile)

    def decay_preferences(self) -> None:
        """Apply monthly decay to non-explicit preference items.

        Only runs if >30 days since last decay check. Explicit items
        (source="explicit") never decay.
        """
        now = datetime.now(CST)
        if self._last_decay_check:
            last = datetime.fromisoformat(self._last_decay_check)
            if (now - last).days < 30:
                return

        profile = self.load_profile()
        learned = profile.preferences.learned_items
        changed = False
        survivors: list[PreferenceCategory] = []

        for item in learned:
            if item.source == "explicit":
                survivors.append(item)
                continue
            if not item.last_seen:
                survivors.append(item)
                continue

            try:
                last_seen = datetime.fromisoformat(item.last_seen)
            except ValueError:
                survivors.append(item)
                continue

            months_since = (now - last_seen).days / 30.0
            decay = 0.03 * months_since
            item.confidence = max(0.1, item.confidence - decay)

            if item.confidence < 0.2:
                changed = True
                continue  # remove item
            survivors.append(item)

        if changed or len(survivors) != len(learned):
            profile.preferences.learned_items = survivors
            _rebuild_denormalised(profile.preferences)
            self.save_profile(profile)
            logger.debug(
                "Decay: %d items removed, %d remain",
                len(learned) - len(survivors),
                len(survivors),
            )

        self._last_decay_check = now.isoformat()

    def get_high_confidence_prefs(
        self, threshold: float = 0.6
    ) -> list[PreferenceCategory]:
        """Return learned items with confidence >= *threshold*."""
        profile = self.load_profile()
        return [item for item in profile.preferences.learned_items
                if item.confidence >= threshold]

    # ── Context Assembly ─────────────────────────────────────────────

    def build_profile_context(self) -> str:
        """Build a compact Chinese text block summarising the user profile.

        Returns an empty string if the profile is not set up. The block is
        capped at ~500 chars to stay within token budget for prompt injection.
        """
        profile = self.load_profile()
        if not profile.setup_complete:
            return ""

        lines: list[str] = []
        lines.append("[用户画像]")

        # Name
        if profile.name:
            lines.append(f"- 称呼: {profile.name}")

        # Home location
        if profile.home_location:
            lines.append(f"- 常住: {profile.home_location}")

        # Party members
        if profile.saved_party_members:
            parts: list[str] = []
            for i, m in enumerate(profile.saved_party_members[:4]):
                desc = f"{m.role}"
                if m.age:
                    desc += f"({m.age}岁"
                    if m.constraints:
                        desc += f", {'; '.join(m.constraints[:2])}"
                    desc += ")"
                parts.append(desc)
            if len(profile.saved_party_members) > 4:
                parts.append(f"...等{len(profile.saved_party_members)}人")
            lines.append(f"- 常用同行人: {', '.join(parts)}")

        # Party size default
        if not profile.saved_party_members and profile.default_party_size > 1:
            lines.append(f"- 通常出行人数: {profile.default_party_size}人")

        # Preferences by confidence tier
        prefs = profile.preferences
        high: list[str] = []
        med: list[str] = []
        low: list[str] = []
        for item in sorted(prefs.learned_items, key=lambda x: x.confidence, reverse=True):
            if item.confidence >= 0.8:
                high.append(item.value)
            elif item.confidence >= 0.5:
                med.append(item.value)
            elif item.confidence >= 0.4:
                low.append(item.value)

        # Group by category for dining preferences
        dining_high = [i for i in sorted(prefs.learned_items, key=lambda x: x.confidence, reverse=True)
                       if i.category == "dining" and i.confidence >= 0.4]
        if dining_high:
            dining_parts = [_confidence_label(d) for d in dining_high[:5]]
            lines.append(f"- 口味偏好: {', '.join(dining_parts)}")

        # Activity preferences
        act_items = [i for i in sorted(prefs.learned_items, key=lambda x: x.confidence, reverse=True)
                     if i.category == "activity" and i.confidence >= 0.4]
        if act_items:
            act_parts = [_confidence_label(a) for a in act_items[:4]]
            lines.append(f"- 偏好活动: {', '.join(act_parts)}")

        # Budget
        if prefs.budget_range:
            lo, hi = prefs.budget_range
            lines.append(f"- 人均预算: {_fmt_budget(lo, hi)}")

        # Constraints
        if prefs.hard_constraints:
            lines.append(f"- 已知约束: {', '.join(prefs.hard_constraints[:5])}")

        # Preferred areas
        if prefs.preferred_area:
            lines.append(f"- 常去区域: {', '.join(prefs.preferred_area[:4])}")

        lines.append("[/用户画像]")
        result = "\n".join(lines)

        # Cap at ~500 chars
        if len(result) > 500:
            result = result[:497] + "..."

        return result


# ── Helpers ─────────────────────────────────────────────────────────────


def _find_item(
    items: list[PreferenceCategory], category: str, value: str
) -> PreferenceCategory | None:
    """Find an existing preference item by (category, value)."""
    for item in items:
        if item.category == category and item.value == value:
            return item
    return None


def _source_rank(source: str) -> int:
    """Authority ranking for preference sources."""
    ranks = {"behavioral": 1, "inferred": 2, "explicit": 3}
    return ranks.get(source, 0)


def _confidence_label(item: PreferenceCategory) -> str:
    """Render a preference item with its confidence tier."""
    if item.confidence >= 0.8:
        tier = "高"
    elif item.confidence >= 0.5:
        tier = "中"
    else:
        tier = "低"
    return f"{item.value}({tier})"


def _fmt_budget(lo: float, hi: float) -> str:
    """Format a budget range nicely."""
    if lo <= 0:
        return f"{int(hi)}元以下"
    if hi >= 9999:
        return f"{int(lo)}元以上"
    return f"{int(lo)}-{int(hi)}元"
