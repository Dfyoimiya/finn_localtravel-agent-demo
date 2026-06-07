"""POI search — cache, strategy selection, dedup, and ranking.

"Weak agent" design: minimal LLM intervention. Strategy selection is
rule-based from ExtractResult tags; LLM fallback only when tags are ambiguous.
Direct MCP tool calls (no ReAct loop) for actual search execution.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

from finn.logger import logger
from finn.state import (
    ExtractResult,
    POICandidate,
    POISearchResult,
    SceneType,
    WeatherContext,
)


# ═══════════════════════════════════════════════════════════════════════
# POI Cache
# ═══════════════════════════════════════════════════════════════════════


class POICache:
    """[DEPRECATED] Use MCPHub from finn.hub for caching instead.

    In-memory TTL cache for POI search results.
    Kept for backward compatibility with memory system.
    """

    def __init__(self, ttl_seconds: int = 300):
        self._store: dict[str, tuple[float, POISearchResult]] = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> POISearchResult | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, result = entry
        if time.monotonic() - ts > self._ttl:
            del self._store[key]
            return None
        logger.debug("POI cache HIT for key=%s", key[:16])
        return result

    def set(self, key: str, result: POISearchResult) -> None:
        self._store[key] = (time.monotonic(), result)
        logger.debug("POI cache SET key=%s (%d candidates)", key[:16], len(result.candidates))

    def clear(self) -> None:
        self._store.clear()


# Module-level cache instance (shared across all invocations)
poi_cache = POICache()


# ═══════════════════════════════════════════════════════════════════════
# Tag → Amap POI type mapping
# ═══════════════════════════════════════════════════════════════════════

# Maps group tags to Amap POI type codes + Chinese search keywords
_TAG_TO_POI: dict[str, list[str]] = {
    # Age-related
    "child_": ["公园", "亲子", "游乐", "动物园", "博物馆"],
    "elderly_": ["公园", "茶馆", "美术馆", "博物馆"],
    # Diet-related
    "diet_low_calorie": ["轻食", "沙拉", "日料"],
    "diet_halal": ["清真", "西北菜", "新疆菜"],
    "diet_vegan": ["素食", "蔬食"],
    "diet_seafood_allergy": [],  # exclusion, not search
    "diet_spicy_avoid": ["清淡", "粤菜", "江浙菜", "日料"],
    # Activity preferences
    "photo_friendly": ["网红", "打卡", "拍照"],
    "quiet": ["茶馆", "书店", "美术馆"],
    "indoor": ["商场", "博物馆", "电影院"],
    "outdoor": ["公园", "景区", "徒步"],
    # Scene-specific
    "child_safe": ["亲子", "儿童乐园"],
    "wheelchair_accessible": [],  # filter post-search
    "non_smoking": [],  # filter post-search
    "pet_friendly": ["宠物友好"],
}

# Scene-level default tags
_SCENE_DEFAULT_TAGS: dict[SceneType, list[str]] = {
    SceneType.FAMILY: ["亲子", "公园", "博物馆", "动物园"],
    SceneType.FRIENDS: ["网红", "打卡", "火锅", "烧烤", "咖啡馆"],
}


# ═══════════════════════════════════════════════════════════════════════
# Search Strategy
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class SearchStrategy:
    """Parameters for a single POI search call."""

    keywords: list[str] = field(default_factory=list)
    radius_m: int = 5000
    max_results: int = 10
    poi_types: list[str] = field(default_factory=list)


# Keywords that indicate dining/food search intent
_DINING_KEYWORDS: set[str] = {
    "轻食", "沙拉", "日料", "素食", "清真", "西北菜", "新疆菜",
    "粤菜", "江浙菜", "火锅", "烧烤", "咖啡馆",
    # Common user-requested cuisines
    "西餐", "川菜", "湘菜", "东北菜", "韩国料理", "韩餐",
    "东南亚菜", "泰国菜", "越南菜", "意大利菜", "法国菜",
    "披萨", "牛排", "汉堡", "海鲜", "自助餐", "小吃",
    "面馆", "简餐", "快餐", "米线", "饺子", "包子", "食堂",
}


# Indoor activity keywords injected when weather is bad
_INDOOR_ACTIVITY_KEYWORDS: list[str] = [
    "商场", "博物馆", "美术馆", "电影院", "书店",
    "科技馆", "室内乐园", "密室", "桌游", "KTV",
    "展览馆", "图书馆", "剧院", "海洋馆",
]


def build_search_keywords(
    extract: ExtractResult,
    weather: WeatherContext | None = None,
) -> list[str]:
    """Derive search keywords from extract tags, preferences, and requirements.

    Returns a deduplicated list of Chinese search keywords with dining keywords
    interleaved among activity keywords so both categories get searched.

    When weather.indoor_recommended is True, indoor activity keywords are
    injected at high priority so the candidate pool has rain-safe options.
    """
    keywords: list[str] = []

    # 1. Scene defaults (highest priority — broadest signal)
    scene = extract.group.type
    if scene in _SCENE_DEFAULT_TAGS:
        keywords.extend(_SCENE_DEFAULT_TAGS[scene])

    # 2. Group tags → POI keywords
    for tag in extract.group.tags:
        for prefix, kws in _TAG_TO_POI.items():
            if tag.startswith(prefix) or tag == prefix:
                keywords.extend(kws)

    # 3. Requirements (must-haves)
    keywords.extend(extract.requirements.must_have_cuisine)
    keywords.extend(extract.requirements.must_have_activity_type)
    keywords.extend(extract.requirements.must_visit_pois)

    # 4. Soft preferences
    keywords.extend(extract.soft_constraints.preferred_cuisines)
    keywords.extend(extract.soft_constraints.preferred_poi_types)

    # 5. Chain template preferred activity types
    keywords.extend(extract.chain.preferred_activity_types)

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for kw in keywords:
        if kw and kw not in seen:
            seen.add(kw)
            unique.append(kw)

    # Separate dining from activity keywords, then interleave so both get
    # searched in the top-N that _execute_poi_search actually uses.
    activity_kws = [kw for kw in unique if kw not in _DINING_KEYWORDS]
    dining_kws = [kw for kw in unique if kw in _DINING_KEYWORDS]

    interleaved: list[str] = []
    max_len = max(len(activity_kws), len(dining_kws))
    for i in range(max_len):
        if i < len(activity_kws):
            interleaved.append(activity_kws[i])
        if i < len(dining_kws):
            interleaved.append(dining_kws[i])

    # ── Keyword diversity balance ──
    # When lunch is explicitly casual ("中午随便/午餐随便"), inject lunch-friendly
    # dining keywords so the candidate pool has non-hotpot lunch options.
    # Inject BEFORE interleaving so they appear in the top searched keywords.
    is_lunch_casual = False
    cp = extract.constraint_profile
    if cp and cp.meal_slots:
        for ms in cp.meal_slots:
            if ms.get("slot") == "lunch" and ms.get("casual") is True:
                is_lunch_casual = True
                break
    if not is_lunch_casual:
        notes = (extract.requirements.notes or "").lower()
        special = " ".join(extract.requirements.special_requests or []).lower()
        combined = f"{notes} {special}"
        is_lunch_casual = any(
            w in combined for w in ["中午随意", "午饭随便", "午餐随便", "中午随便"]
        )

    _CASUAL_LUNCH = {"简餐", "面馆", "小吃", "快餐", "米线", "饺子",
                      "饭", "粉", "粥", "面", "包子", "食堂"}

    if is_lunch_casual:
        # Prepend casual lunch keywords to dining keywords so they're searched first
        for kw in reversed(list(_CASUAL_LUNCH)):
            if kw not in seen:
                seen.add(kw)
                dining_kws.insert(0, kw)

    # When heavy/dinner-oriented keywords (火锅, 烧烤, etc.) dominate and
    # casual lunch keywords are still missing, inject fallback casual keywords.
    _HEAVY_DINING = {"火锅", "串串", "麻辣烫", "冒菜", "烧烤", "烤鸭", "烤鱼",
                     "烤肉", "海鲜", "铁板", "干锅", "酸菜鱼"}
    has_heavy = any(kw in _HEAVY_DINING for kw in interleaved)
    has_casual_dining = any(kw in _CASUAL_LUNCH for kw in dining_kws)
    if has_heavy and not has_casual_dining:
        dining_kws.append("简餐")
        dining_kws.append("面馆")
        logger.debug("Search keywords: injected casual lunch keywords for balance")

    # ── Weather-aware indoor pivot ──
    # When weather suggests indoor activities, prepend indoor activity keywords
    # so the candidate pool has rain/snow-safe play options. This runs BEFORE
    # interleaving so indoor keywords appear in the top-N that get searched.
    if weather and weather.indoor_recommended:
        indoor_injected = 0
        for kw in reversed(_INDOOR_ACTIVITY_KEYWORDS):
            if kw not in seen:
                seen.add(kw)
                activity_kws.insert(0, kw)
                indoor_injected += 1
        if indoor_injected:
            # Rebuild interleaved with the new activity keywords
            interleaved = []
            for i in range(max(len(activity_kws), len(dining_kws))):
                if i < len(activity_kws):
                    interleaved.append(activity_kws[i])
                if i < len(dining_kws):
                    interleaved.append(dining_kws[i])
            logger.debug("Search keywords: injected %d indoor activity keywords for weather",
                         indoor_injected)

    logger.debug("Search keywords: %s", interleaved[:10])
    return interleaved


def select_search_strategy(
    extract: ExtractResult,
    weather: WeatherContext | None = None,
) -> SearchStrategy:
    """Rule-based search strategy selection from ExtractResult.

    Maps group tags, preferences, and geo constraints to concrete
    Amap search parameters. No LLM call needed for standard cases.
    """
    keywords = build_search_keywords(extract, weather)

    # Radius from geo constraint
    radius = extract.geo.radius_m

    # Max results based on chain template
    max_results = min(extract.chain.max_nodes * 3, 15)

    # POI types from preferred_poi_types + chain
    poi_types = list(dict.fromkeys(
        extract.soft_constraints.preferred_poi_types
        + extract.chain.preferred_activity_types
    ))

    # ── Radius adjustment from transit time ──
    # Use both soft_constraints.max_transit_minutes and geo.max_transit_time_min
    max_transit = (
        extract.soft_constraints.max_transit_minutes
        or extract.geo.max_transit_time_min
    )
    transport = extract.soft_constraints.preferred_transport
    if transport and transport.value == "walk":
        speed_m_per_min = 83   # ~5 km/h
    elif transport and transport.value == "drive":
        speed_m_per_min = 500  # ~30 km/h city driving
    else:
        speed_m_per_min = 300  # mixed mode

    if max_transit and max_transit > 0:
        transit_radius = int(max_transit * speed_m_per_min)
        # Expand radius to match user's transit tolerance
        radius = max(radius, transit_radius)

    # ── Radius adjustment from constraint_desc ──
    desc = extract.geo.constraint_desc.lower()
    if "别太远" in desc or "不要太远" in desc or "路程不要太远" in desc:
        radius = min(radius, 5000)
    elif ("近" in desc or "附近" in desc) and "远" not in desc:
        radius = min(radius, 3000)
    elif "小时" in desc or "远一点" in desc or "远些" in desc:
        # Explicitly far — keep the transit-based estimate
        pass

    # Clamp to reasonable range
    radius = max(1000, min(100000, radius))

    strategy = SearchStrategy(
        keywords=keywords,
        radius_m=radius,
        max_results=max_results,
        poi_types=poi_types,
    )
    logger.info(
        "Search strategy: radius=%dm, keywords=%s, max=%d",
        radius, keywords[:5], max_results,
    )
    return strategy


def make_cache_key(strategy: SearchStrategy, city: str | None) -> str:
    """Build a deterministic cache key from search strategy + city."""
    payload = f"{city or ''}|{strategy.radius_m}|{','.join(sorted(strategy.keywords))}"
    return hashlib.md5(payload.encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════
# Dedup & Ranking
# ═══════════════════════════════════════════════════════════════════════


def deduplicate_pois(candidates: list[POICandidate]) -> list[POICandidate]:
    """Deduplicate POI candidates by Amap ID first, then name similarity.

    Returns a deduplicated list preserving original order.
    """
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    unique: list[POICandidate] = []

    for poi in candidates:
        # Primary: deduplicate by Amap ID
        if poi.id and poi.id in seen_ids:
            continue

        # Secondary: deduplicate by normalized name
        name_key = _normalize_name(poi.name)
        if name_key in seen_names:
            continue

        if poi.id:
            seen_ids.add(poi.id)
        seen_names.add(name_key)
        unique.append(poi)

    if len(candidates) != len(unique):
        logger.debug("Dedup: %d → %d candidates", len(candidates), len(unique))

    return unique


def _normalize_name(name: str) -> str:
    """Normalize a POI name for fuzzy dedup comparison."""
    # Remove branch suffixes, whitespace, common noise words
    import re
    name = name.strip().lower()
    # Remove parenthetical suffixes like "(三里屯店)"
    name = re.sub(r"[（(][^)）]*[)）]", "", name)
    # Remove whitespace
    name = re.sub(r"\s+", "", name)
    return name


def rank_pois(
    candidates: list[POICandidate],
    extract: ExtractResult,
) -> list[POICandidate]:
    """Rank POI candidates by preference match, distance, and rating.

    Scoring (weighted):
    - Preference tag match: 40%
    - Distance (closer = better): 25%
    - Rating: 20%
    - Avoid type penalty: -15%
    """
    avoid_types = set(extract.soft_constraints.avoid_poi_types)
    preferred_tags = set(
        extract.soft_constraints.preferred_cuisines
        + extract.soft_constraints.preferred_poi_types
        + extract.chain.preferred_activity_types
        + extract.requirements.must_have_cuisine
        + extract.requirements.must_have_activity_type
    )

    scored: list[tuple[float, POICandidate]] = []
    for poi in candidates:
        score = 0.0

        # Tag match (40%)
        if preferred_tags:
            poi_tag_set = set(poi.tags) | {poi.type, poi.name}
            tag_hits = sum(1 for t in preferred_tags if t.lower() in poi.name.lower()
                          or any(t.lower() in pt.lower() for pt in poi_tag_set))
            score += 0.4 * min(tag_hits / max(len(preferred_tags), 1), 1.0)

        # Distance (25%) — closer is better
        if poi.distance_m is not None and poi.distance_m > 0:
            dist_score = max(0, 1.0 - poi.distance_m / extract.geo.radius_m)
            score += 0.25 * dist_score
        else:
            score += 0.125  # unknown distance = mid

        # Rating (20%) — assume 5-point scale
        if poi.rating is not None:
            score += 0.2 * min(poi.rating / 5.0, 1.0)
        else:
            score += 0.1  # unknown rating = mid

        # Avoid penalty
        if avoid_types:
            poi_tag_set = set(poi.tags) | {poi.type}
            if avoid_types & poi_tag_set:
                score -= 0.15

        scored.append((score, poi))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored]
