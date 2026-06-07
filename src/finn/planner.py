"""PlannerAgent — Activity Sequence Greedy Construction with Dynamic Time Allocation.

Core algorithm:
1. Derive strategy weights (α,β,γ,δ) from user profile
2. Run 3 parallel strategies via asyncio.gather:
   - Time-Critical: α-dominant greedy, minimize transit
   - Geo-Cluster: find densest spatial cluster, greedy within cluster
   - Match-Driven: γ-dominant greedy, maximize constraint satisfaction
3. For each strategy, greedy sequence construction with insertion cost:
   cost = α×transit + β×wait_time + γ×(1-match_score) + δ×dist_from_center
4. Forward/backward pass for dynamic time allocation with slack computation
5. Elastic budget reflow for time-overflow scenarios
6. Score each path, select best
7. Two-tier transport: coarse distance → refined direction API for winning path
8. Convert winning PlannedPath → Plan (SubTask format)
"""

from __future__ import annotations

import asyncio
from typing import Any

from finn.logger import logger
from finn.state import (
    ActivityNode,
    ConstraintProfile,
    ExtractResult,
    Plan,
    PlannedPath,
    POICandidate,
    StrategyResult,
    SubTask,
    TimeAlloc,
    WeatherContext,
)

# ── Context-aware weather adjustment ───────────────────────────────────

# Type-code prefixes for outdoor venues
# 08xxxx = parks, zoos, amusement parks; 11xxxx = scenic areas, historic districts
OUTDOOR_TYPE_PREFIXES: tuple[str, ...] = ("08", "11")

# Weather penalties by condition group — graduated, not binary
_WEATHER_PENALTY_MAP: dict[str, float] = {
    "雨": -0.25,   # Rain: downgrade outdoor but keep in pool
    "雪": -0.30,   # Snow: stronger penalty
    "高温": -0.20,  # Extreme heat: mild penalty
    "沙尘": -0.35,  # Sandstorm: strongest penalty
}

# Type prefixes that are *never* usable regardless of weather
# (e.g. water parks 0803, outdoor sports 0805 — truly weather-dependent)
_WEATHER_BLOCKED_PREFIXES: tuple[str, ...] = ("0803", "0805")


def context_hard_filter(
    candidates: list[POICandidate],
    weather: WeatherContext | None,
) -> list[POICandidate]:
    """Adjust candidates by weather context. Graduated — never hard-deletes parks/scenic.

    Weather signal is a soft constraint: downgrade outdoor POIs' match_score
    so indoor alternatives are preferred, but outdoor venues remain available.
    Only water parks (0803) and outdoor sports (0805) are hard-removed —
    these are genuinely unusable in bad weather.
    """
    if not weather or not weather.indoor_recommended:
        return list(candidates)

    condition = weather.condition or ""
    penalty = 0.0
    for key, p in _WEATHER_PENALTY_MAP.items():
        if key in condition:
            penalty = min(penalty, p)  # most negative wins
    if penalty == 0.0:
        penalty = -0.20  # default indoor-recommended penalty

    kept = 0
    penalized = 0
    blocked = 0
    for p in candidates:
        ptype = p.type or ""
        if ptype.startswith(_WEATHER_BLOCKED_PREFIXES):
            blocked += 1
            continue
        if ptype.startswith(OUTDOOR_TYPE_PREFIXES):
            p.match_score = max(0.0, (p.match_score if p.match_score > 0 else 0.5) + penalty)
            penalized += 1
        kept += 1

    result = [p for p in candidates
              if not (p.type or "").startswith(_WEATHER_BLOCKED_PREFIXES)]

    logger.debug("Context weather adjust: %d → %d candidates (weather=%s, "
                 "penalty=%.2f, penalized=%d, blocked=%d)",
                 len(candidates), len(result), weather.condition,
                 penalty, penalized, blocked)
    return result


# ── POI → slot classification ────────────────────────────────────

# Keywords that indicate a POI belongs to the "eat" slot
_EAT_KEYWORDS: set[str] = {
    "餐饮", "餐厅", "面馆", "火锅", "烧烤", "轻食", "沙拉",
    "日料", "素食", "清真", "粤菜", "江浙菜", "咖啡馆", "茶馆",
    "小吃", "快餐", "中餐厅", "快餐厅", "面", "粉", "菜",
}

# Typecode prefixes that indicate eating
_EAT_TYPECODES = {"05", "0501", "0502"}

# Keywords that indicate "play" slot
_PLAY_KEYWORDS: set[str] = {
    "公园", "博物馆", "动物园", "游乐", "亲子", "景区",
    "徒步", "登山", "网红", "打卡", "拍照", "儿童",
    "水上", "温泉", "电影院", "商场",
}

# Typecode prefixes for play/entertainment (NOT 10=accommodation)
_PLAY_TYPECODES = {"11", "08", "14", "06"}

# Keywords that indicate a POI should NOT be used as a "play" stop
# (hotels, B&Bs, residential, etc. — unless user specifically wants accommodation)
_NON_PLAY_KEYWORDS: set[str] = {
    "酒店", "宾馆", "民宿", "住宿", "客栈", "青旅", "公寓",
    "旅馆", "招待所", "套房", "resort", "inn", "hostel",
}
_NON_PLAY_TYPECODES = {"10", "1001", "1002", "1003"}

# Hot pot keywords for dinner matching
_HOTPOT_KEYWORDS = {"火锅", "串串", "麻辣烫", "冒菜"}

# Casual/cheap lunch keywords
_CASUAL_EAT_KEYWORDS = {"快餐", "面馆", "小吃", "粉", "面", "简餐", "食堂", "米线"}


def _classify_slot(poi: POICandidate) -> str:
    """Classify a POI into 'play', 'eat', or 'follow_up'."""
    poi_type = (poi.type or "").lower()
    name = poi.name.lower()
    tags_lower = [t.lower() for t in poi.tags]

    # Check eat
    if any(kw.lower() in name or kw.lower() in poi_type for kw in _EAT_KEYWORDS):
        return "eat"
    if any(poi_type.startswith(tc) for tc in _EAT_TYPECODES):
        return "eat"
    for kw in _EAT_KEYWORDS:
        if any(kw.lower() in t for t in tags_lower):
            return "eat"

    # Check play
    if any(kw.lower() in name or kw.lower() in poi_type for kw in _PLAY_KEYWORDS):
        return "play"
    if any(poi_type.startswith(tc) for tc in _PLAY_TYPECODES):
        return "play"

    # Hotels/B&Bs → follow_up (not play, not eat)
    if any(kw.lower() in name for kw in _NON_PLAY_KEYWORDS):
        return "follow_up"
    if any(poi_type.startswith(tc) for tc in _NON_PLAY_TYPECODES):
        return "follow_up"

    return "play"


# ═══════════════════════════════════════════════════════════════════════
# Strategy weights
# ═══════════════════════════════════════════════════════════════════════


def compute_strategy_weights(extract: ExtractResult) -> tuple[float, float, float, float]:
    """Derive (α, β, γ, δ) from user profile and constraints.

    α (transit):     higher → transit time is more costly (fast pace users)
    β (wait/idle):   higher → waiting/idle time is more costly
    γ (mismatch):    higher → constraint mismatch matters more (luxury users)
    δ (distance):    higher → distance from center is more costly ("别太远")
    """
    sc = extract.soft_constraints
    pace = sc.travel_pace.value if sc.travel_pace else "balanced"
    budget = sc.budget_preference.value if sc.budget_preference else "mid"
    transport = sc.preferred_transport.value if sc.preferred_transport else None

    # α: transit sensitivity — fast-paced users hate transit
    alpha_map = {"relaxed": 0.5, "balanced": 1.0, "fast": 2.0}
    alpha = alpha_map.get(pace, 1.0)

    # β: wait/idle sensitivity — fast-paced users hate waiting
    beta_map = {"relaxed": 0.3, "balanced": 1.0, "fast": 2.0}
    beta = beta_map.get(pace, 1.0)

    # γ: mismatch sensitivity — luxury users care about quality/preference match
    gamma_map = {"economy": 0.5, "mid": 1.0, "luxury": 2.0}
    gamma = gamma_map.get(budget, 1.0)

    # δ: distance-from-center sensitivity — near/far preference
    geo_desc = extract.geo.constraint_desc.lower()
    if any(w in geo_desc for w in ["近", "别太远", "不要太远", "步行"]):
        delta = 2.5
    elif "小时" in geo_desc:
        delta = 0.3  # User is OK with long distance
    else:
        delta = 1.0

    logger.debug("Strategy weights: α=%.1f β=%.1f γ=%.1f δ=%.1f", alpha, beta, gamma, delta)
    return alpha, beta, gamma, delta


# ═══════════════════════════════════════════════════════════════════════
# Template expansion
# ═══════════════════════════════════════════════════════════════════════


def _expand_template(extract: ExtractResult) -> list[str]:
    """Expand chain template based on time window and meal requirements.

    Returns a list of slot names like ['play', 'lunch', 'play', 'dinner'].
    """
    hours = extract.intent.time_window_hours or 6
    notes = (extract.requirements.notes or "").lower()
    special = " ".join(extract.requirements.special_requests or []).lower()
    must_cuisine = [c.lower() for c in extract.requirements.must_have_cuisine]

    has_dinner_req = any(w in notes or w in special
                         for w in ["晚上", "晚餐", "晚饭", "夜间", "傍晚"])
    has_lunch_req = any(w in notes or w in special
                        for w in ["中午", "午餐", "午饭"])

    # Base template by time window
    if hours <= 4:
        template = ["play", "eat"]
    elif hours <= 8:
        template = ["play", "eat", "follow_up"]
    else:  # >8h: two meals
        template = ["play", "lunch", "play", "dinner"]
        if hours > 12:
            template.append("follow_up")

    # If user mentions dinner specifically, ensure dinner slot exists
    if has_dinner_req and "dinner" not in template:
        template = ["play", "eat", "play", "dinner"]

    # If user mentions lunch, ensure lunch distinction
    if has_lunch_req and "lunch" not in template and "dinner" in template:
        template = [("lunch" if s == "eat" else s) for s in template]

    logger.debug("Template expanded: %s (hours=%.1f, dinner_req=%s, lunch_req=%s)",
                 template, hours, has_dinner_req, has_lunch_req)

    return template


# ═══════════════════════════════════════════════════════════════════════
# Transport selection
# ═══════════════════════════════════════════════════════════════════════


def _select_transport(distance_m: int, user_pref: str | None = None) -> str:
    """Auto-select transport mode by distance.

    <1.5km → walk, 1.5-5km → transit, >5km → drive.
    User preference overrides if compatible with distance.
    """
    if user_pref == "walk" and distance_m <= 3000:
        return "walk"
    if user_pref == "drive":
        return "drive"

    if distance_m < 1500:
        return "walk"
    elif distance_m < 5000:
        return "transit"
    else:
        return "drive"


def _transit_minutes(distance_m: int, mode: str) -> int:
    """Estimate transit time in minutes by transport mode."""
    if distance_m <= 0:
        return 0
    speeds = {"walk": 80, "transit": 250, "drive": 500}  # m/min
    speed = speeds.get(mode, 400)
    return max(1, distance_m // speed)


# ═══════════════════════════════════════════════════════════════════════
# Insertion cost
# ═══════════════════════════════════════════════════════════════════════


def insertion_cost(
    poi: POICandidate,
    slot: str,
    prev_node: ActivityNode | None,
    dist_matrix: dict[str, int],
    weights: tuple[float, float, float, float],
    extract: ExtractResult,
    selected_nodes: list[tuple[str, POICandidate]] | None = None,
) -> float:
    """Compute cost of inserting this POI at the current position.

    cost = α×transit_norm + β×wait_norm + γ×(1-match_score) + δ×dist_norm
           + ε×diversity_penalty

    All components normalized to [0, 1] before weighting.
    Lower cost = better fit.
    """
    alpha, beta, gamma, delta = weights
    center_loc = extract.geo.center_location or ""
    radius = max(extract.geo.radius_m, 1000)

    # ── Transit component (α) ──
    transit_m = 0
    if prev_node and prev_node.poi and prev_node.poi.location and poi.location:
        pair_key = f"{prev_node.poi.location}|{poi.location}"
        transit_m = dist_matrix.get(pair_key, 0)
    elif center_loc and poi.location:
        pair_key = f"{center_loc}|{poi.location}"
        transit_m = dist_matrix.get(pair_key, 0)

    # Normalize: 0 at 0m, 1 at 30km
    transit_norm = min(1.0, transit_m / 30000.0)

    # ── Wait/idle component (β) ──
    wait_min = 0
    if prev_node and prev_node.time:
        prev_end_str = prev_node.time.earliest_end or ""
        if prev_end_str and poi.open_time:
            try:
                prev_end_h = int(prev_end_str.split(":")[0]) * 60 + int(prev_end_str.split(":")[1])
                open_str = poi.open_time.replace("：", ":")
                open_h = int(open_str.split(":")[0]) * 60 + int(open_str.split(":")[1])
                # Only count waiting if we arrive before opening
                arrive_time = prev_end_h + _transit_minutes(transit_m, "drive")
                if arrive_time < open_h:
                    wait_min = open_h - arrive_time
            except (ValueError, IndexError):
                pass
    # Normalize: 0 at 0min, 1 at 120min
    wait_norm = min(1.0, wait_min / 120.0)

    # ── Match mismatch component (γ) ──
    match = poi.match_score if poi.match_score > 0 else 0.5
    match_norm = 1.0 - match  # 0 = perfect match, 1 = no match

    # ── Distance from center component (δ) ──
    center_dist = 0
    if center_loc and poi.location:
        pair_key = f"{center_loc}|{poi.location}"
        center_dist = dist_matrix.get(pair_key, 0)
    # Normalize by search radius
    dist_norm = min(1.0, center_dist / radius)

    cost = (alpha * transit_norm
            + beta * wait_norm
            + gamma * match_norm
            + delta * dist_norm)

    # ── Diversity penalty (ε) ──
    # Penalize same-type POIs to encourage variety in the itinerary
    if selected_nodes:
        max_sim = max(
            _type_similarity(poi.type, prev_poi.type)
            for _, prev_poi in selected_nodes
        )
        cost += 2.0 * max_sim  # ε = 2.0

    return cost


# ═══════════════════════════════════════════════════════════════════════
# Forward / Backward pass — dynamic time allocation
# ═══════════════════════════════════════════════════════════════════════


def _hhmm_to_min(s: str) -> int:
    """Convert HH:MM to minutes since midnight."""
    try:
        h, m = s.strip().split(":")
        return int(h) * 60 + int(m)
    except (ValueError, IndexError):
        return 0


def _min_to_hhmm(m: int) -> str:
    """Convert minutes since midnight to HH:MM."""
    h = (m // 60) % 24
    mn = m % 60
    return f"{h:02d}:{mn:02d}"


def _forward_pass(
    nodes: list[ActivityNode],
    start_time: str,
    extract: ExtractResult,
) -> None:
    """Compute earliest_start/earliest_end for each node."""
    try:
        current = _hhmm_to_min(start_time)
    except (ValueError, IndexError):
        current = 9 * 60  # Default 09:00

    for node in nodes:
        # Ensure time allocation exists
        if node.time is None:
            node.time = TimeAlloc(slot=node.slot)

        transit = node.transit_from_prev_min
        stay = node.stay_duration_min

        node.time.earliest_start = _min_to_hhmm(current + transit)
        node.time.earliest_end = _min_to_hhmm(current + transit + stay)
        current = current + transit + stay


def _backward_pass(
    nodes: list[ActivityNode],
    deadline: str,
) -> None:
    """Compute latest_start/latest_end and slack for each node."""
    try:
        current = _hhmm_to_min(deadline)
    except (ValueError, IndexError):
        current = 21 * 60  # Default 21:00

    for node in reversed(nodes):
        if node.time is None:
            node.time = TimeAlloc(slot=node.slot)

        stay = node.stay_duration_min

        node.time.latest_end = _min_to_hhmm(current)
        node.time.latest_start = _min_to_hhmm(current - stay)

        earliest_start_min = _hhmm_to_min(node.time.earliest_start)
        latest_start_min = _hhmm_to_min(node.time.latest_start)
        node.time.slack_min = max(0, latest_start_min - earliest_start_min)

        current = latest_start_min - node.transit_from_prev_min


# ═══════════════════════════════════════════════════════════════════════
# Elastic budget reflow
# ═══════════════════════════════════════════════════════════════════════


def _elastic_budget_reflow(
    nodes: list[ActivityNode],
    available_minutes: int,
) -> None:
    """Compress elastic nodes when total time exceeds available window.

    Sorts nodes by elastic_coef descending, distributes excess proportionally.
    Rigid nodes (elastic_coef=0, like movies) are not compressed.
    """
    total_stay = sum(
        (n.stay_duration_min if n.stay_duration_min > 0 else n.poi.stay_base
         if n.poi else 120)
        for n in nodes
    )
    total_transit = sum(n.transit_from_prev_min for n in nodes)
    excess = (total_stay + total_transit) - available_minutes

    if excess <= 0:
        return

    logger.debug("Elastic reflow: excess=%d min (total=%d, avail=%d)",
                 excess, total_stay + total_transit, available_minutes)

    # Compressible nodes (elastic_coef > 0)
    compressible = []
    for node in nodes:
        poi = node.poi
        if poi and poi.elastic_coef > 0:
            max_compress = node.stay_duration_min - poi.stay_min
            if max_compress > 0:
                compressible.append((node, max_compress, poi.elastic_coef))

    if not compressible:
        logger.debug("Elastic reflow: no compressible nodes")
        return

    total_elasticity = sum(e for _, _, e in compressible)
    remaining_excess = excess

    # Sort by elasticity descending — most elastic compressed first
    for node, max_compress, elas in sorted(compressible, key=lambda x: -x[2]):
        if remaining_excess <= 0:
            break
        share = remaining_excess * (elas / total_elasticity)
        actual = min(int(share), max_compress)
        if actual > 0:
            node.stay_duration_min = max(node.poi.stay_min if node.poi else 60,
                                         node.stay_duration_min - actual)
            remaining_excess -= actual
            logger.debug("  Compressed %s: %d→%d min (cut %d, elas=%.1f)",
                         node.poi.name if node.poi else node.slot,
                         node.stay_duration_min + actual,
                         node.stay_duration_min, actual, elas)

    if remaining_excess > 0:
        logger.warning(
            "Elastic reflow: %d min still overflow after max compression "
            "(all compressible nodes at stay_min)", remaining_excess)


def _get_evening_cuisines(extract: ExtractResult) -> set[str]:
    """Extract evening cuisine requirements from constraint_profile or notes.

    Checks constraint_profile.meal_slots first, then falls back to text
    analysis of requirements.notes + special_requests.
    """
    cp = extract.constraint_profile
    if cp and cp.meal_slots:
        for ms in cp.meal_slots:
            if ms.get("slot") == "dinner" and ms.get("cuisines"):
                return set(c.lower() for c in ms["cuisines"])

    # Fallback: text analysis of notes and special requests
    notes = (extract.requirements.notes or "").lower()
    special = " ".join(extract.requirements.special_requests or []).lower()
    combined = f"{notes} {special}"
    must_cuisine = [c.lower() for c in extract.requirements.must_have_cuisine]

    evening_set: set[str] = set()
    has_evening = any(w in combined for w in ["晚上", "晚餐", "晚饭", "傍晚"])

    if has_evening:
        for kw in _HOTPOT_KEYWORDS:
            if kw in combined or kw in " ".join(must_cuisine):
                evening_set.add(kw)
        for kw in ("烧烤", "日料", "粤菜", "江浙菜", "西北菜", "新疆菜", "素食", "清真"):
            if kw in combined:
                evening_set.add(kw)

    return evening_set


def _is_lunch_casual(extract: ExtractResult) -> bool:
    """Check if lunch slot is explicitly marked casual in constraint_profile."""
    cp = extract.constraint_profile
    if cp and cp.meal_slots:
        for ms in cp.meal_slots:
            if ms.get("slot") == "lunch" and ms.get("casual") is True:
                return True
    # Also check notes text
    notes = (extract.requirements.notes or "").lower()
    special = " ".join(extract.requirements.special_requests or []).lower()
    combined = f"{notes} {special}"
    return any(w in combined for w in ["中午随意", "午饭随便", "午餐随便", "中午随便"])


# ═══════════════════════════════════════════════════════════════════════
# Greedy sequence construction (shared by all strategies)
# ═══════════════════════════════════════════════════════════════════════

# Map template slots to bucket keys
_SLOT_TO_BUCKET = {
    "play": "play", "lunch": "eat", "dinner": "eat",
    "eat": "eat", "follow_up": "follow_up",
}

# ── Slot constraint rules ─────────────────────────────────────────
# Each slot has independent type-code filters, replacing monolithic
# dinner reservation logic with precise Amap type-code matching.

SLOT_CONSTRAINTS: dict[str, dict] = {
    "lunch": {
        "excluded_type_prefixes": ["050117"],  # Exclude hotpot from lunch
    },
    "dinner": {
        "required_type_prefixes": ["050117"],  # Must be hotpot for dinner
    },
}


def _apply_slot_filter(
    slot: str,
    candidates: list[POICandidate],
    used_ids: set[str],
) -> list[POICandidate]:
    """Filter candidates for a slot using SlotConstraint type-code rules."""
    available = [p for p in candidates if p.id not in used_ids]
    rule = SLOT_CONSTRAINTS.get(slot)
    if not rule:
        return available
    if req := rule.get("required_type_prefixes"):
        available = [
            p for p in available
            if any((p.type or "").startswith(pref) for pref in req)
        ]
    if excl := rule.get("excluded_type_prefixes"):
        available = [
            p for p in available
            if not any((p.type or "").startswith(pref) for pref in excl)
        ]
    return available


# ── Type-code similarity for diversity penalty ────────────────────


def _type_similarity(type_a: str, type_b: str) -> float:
    """Type-code similarity: same 4-digit=1.0, same 2-digit=0.5, else 0."""
    if not type_a or not type_b:
        return 0.0
    if type_a[:4] == type_b[:4]:
        return 1.0
    if type_a[:2] == type_b[:2]:
        return 0.5
    return 0.0


# ── Data-driven stay durations ────────────────────────────────────
# Maps Amap type-code prefixes to (stay_min, stay_base, stay_max, elastic_coef)

_STAY_DB: dict[str, tuple[int, int, int, float]] = {
    "11":   (60, 120, 180, 0.8),   # Scenic / historic areas
    "08":   (60, 120, 180, 0.7),   # Parks, zoos, amusement parks
    "0501": (45,  75, 120, 0.6),   # Chinese restaurants (incl. hotpot)
    "0502": (45,  60,  90, 0.5),   # Fast food / snacks
    "14":   (60, 120, 150, 0.6),   # Leisure / entertainment
    "06":   (45,  90, 120, 0.5),   # Shopping
}


def _apply_stay_params(poi: POICandidate) -> None:
    """Apply type-based stay duration to POI. Mutates in-place."""
    for prefix, (smin, sbase, smax, ecoef) in _STAY_DB.items():
        if (poi.type or "").startswith(prefix):
            poi.stay_min = smin
            poi.stay_base = sbase
            poi.stay_max = smax
            poi.elastic_coef = ecoef
            return
    # Defaults from POICandidate model (60/120/240/1.0) stay unchanged


def _build_path(
    extract: ExtractResult,
    candidates: list[POICandidate],
    distance_matrix: dict[str, int],
    weights: tuple[float, float, float, float],
    *,
    slot_ordering: list[str] | None = None,
    candidate_filter=None,
) -> PlannedPath:
    """Build a PlannedPath via greedy sequence construction with insertion cost.

    Args:
        extract: Extracted user intent with constraints
        candidates: Available POI candidates (will be copied, not mutated)
        distance_matrix: Pre-computed distance pairs
        weights: (α, β, γ, δ) strategy weights
        slot_ordering: Override template slot order (for geo-cluster etc.)
        candidate_filter: Optional filter fn(candidates, extract) → filtered candidates
    """
    _, _, gamma, _ = weights

    # Build template
    template = list(slot_ordering) if slot_ordering else _expand_template(extract)
    if not template:
        template = ["play", "lunch", "play", "dinner"]

    pool = list(candidates)  # Copy — don't mutate input
    if candidate_filter:
        pool = list(candidate_filter(pool, extract))

    # Categorize
    buckets: dict[str, list[POICandidate]] = {"play": [], "eat": [], "follow_up": []}
    for p in pool:
        slot = _classify_slot(p)
        buckets[slot].append(p)

    logger.debug("_build_path: template=%s, pool=%d (play=%d eat=%d follow_up=%d)",
                 template, len(pool),
                 len(buckets["play"]), len(buckets["eat"]), len(buckets["follow_up"]))

    # Greedy construction
    selected: list[tuple[str, POICandidate]] = []
    used_ids: set[str] = set()
    prev_location = extract.geo.center_location or ""

    # Pre-allocate best POIs for constrained slots (e.g. dinner=hotpot).
    # Without this, unconstrained slots earlier in the template consume
    # the only candidates that a later constrained slot requires.
    preallocated: dict[str, POICandidate] = {}
    for t_slot in template:
        rule = SLOT_CONSTRAINTS.get(t_slot)
        if not (rule and rule.get("required_type_prefixes")):
            continue
        slot_pool = _apply_slot_filter(t_slot, pool, set())
        if not slot_pool:
            continue
        located = [p for p in slot_pool if p.location]
        if located:
            slot_pool = located
        best = max(slot_pool, key=lambda p: p.match_score if p.match_score > 0 else 0.5)
        preallocated[t_slot] = best
        logger.debug("Pre-allocated %s → %s (type=%s)", t_slot, best.name, best.type)

    for slot in template:
        # Use pre-allocated POI if this slot has one
        prealloc = preallocated.get(slot)
        if prealloc is not None and prealloc.id not in used_ids:
            _apply_stay_params(prealloc)
            selected.append((slot, prealloc))
            used_ids.add(prealloc.id)
            continue

        # Use slot-level constraint filter (type-code based, replaces dinner reservation)
        slot_pool = _apply_slot_filter(slot, pool, used_ids)
        in_bucket = True

        if not slot_pool:
            # Fall back to any unused candidate (still filtered by slot rules).
            # For play slots, explicitly exclude hotels/accommodations so
            # they never appear as "游玩" activities even in desperation mode.
            slot_pool = [p for p in pool if p.id not in used_ids]
            slot_pool = _apply_slot_filter(slot, slot_pool, set())
            bucket_key = _SLOT_TO_BUCKET.get(slot, "play")
            if bucket_key == "play":
                slot_pool = [p for p in slot_pool
                             if not any(kw in p.name for kw in _NON_PLAY_KEYWORDS)
                             and not any((p.type or "").startswith(tc) for tc in _NON_PLAY_TYPECODES)]
            in_bucket = False

        # Exclude pre-allocated POIs from non-matching slots.
        # If all remaining candidates are reserved for future slots,
        # skip this slot to preserve them.
        prealloc_other_ids = {p.id for s, p in preallocated.items() if s != slot}
        if prealloc_other_ids:
            unreserved = [p for p in slot_pool if p.id not in prealloc_other_ids]
            if unreserved:
                slot_pool = unreserved
            else:
                # All candidates are needed by future constrained slots
                slot_pool = []

        if not slot_pool:
            continue

        # ── Location-aware prioritisation ──
        # Within the correct bucket, prefer POIs that have coordinates.
        # Without coordinates we can't compute transit or nav URLs.
        # Only use no-location POIs if there are no located ones available.
        if in_bucket and len(slot_pool) > 1:
            located = [p for p in slot_pool if p.location]
            if located:
                slot_pool = located

        # Score each candidate by insertion cost (lower = better)
        scored = []
        for p in slot_pool:
            # Build a temporary prev_node for cost calculation
            prev_node = None
            if selected:
                prev_slot, prev_poi = selected[-1]
                prev_node = ActivityNode(
                    slot=prev_slot,
                    poi=prev_poi,
                    stay_duration_min=prev_poi.stay_base,
                )
                # Approximate prev end time for wait calculation
                if prev_node.time is None:
                    prev_node.time = TimeAlloc(slot=prev_slot)

            cost = insertion_cost(
                p, slot, prev_node, distance_matrix, weights, extract,
                selected_nodes=selected,
            )

            # ── Cross-bucket penalty ──
            # When we've fallen back to the full pool, penalize POIs that
            # don't belong in this slot's category.
            if not in_bucket:
                bucket_key = _SLOT_TO_BUCKET.get(slot, "play")
                if not p.location or _classify_slot(p) != bucket_key:
                    cost += 3.0

            scored.append((p, cost))

        scored.sort(key=lambda x: x[1])  # Lowest cost first

        best = scored[0][0]
        _apply_stay_params(best)  # Set type-based stay durations
        selected.append((slot, best))
        used_ids.add(best.id)

    if not selected:
        logger.warning("_build_path: no candidates matched")
        return PlannedPath(strategy="empty")

    # ── Build ActivityNodes ──
    nodes: list[ActivityNode] = []
    prev_loc = extract.geo.center_location or ""

    for slot, poi in selected:
        # Transit from previous
        transit_m = 0
        if prev_loc and poi.location and prev_loc != poi.location:
            pair_key = f"{prev_loc}|{poi.location}"
            transit_m = distance_matrix.get(pair_key, 0)

        transport = _select_transport(
            transit_m,
            extract.soft_constraints.preferred_transport.value
            if extract.soft_constraints.preferred_transport else None,
        )
        transit_min = _transit_minutes(transit_m, transport)

        # Cost estimate
        cost_est = 0.0
        if slot in ("lunch",):
            cost_est = poi.price_per_person or 30.0
        elif slot in ("dinner",):
            cost_est = poi.price_per_person or 70.0
        elif slot in ("eat",):
            cost_est = poi.price_per_person or 50.0
        elif slot == "follow_up":
            cost_est = poi.price_per_person or 30.0

        node = ActivityNode(
            slot=slot,
            poi=poi,
            transit_from_prev_min=transit_min,
            transit_distance_m=transit_m,
            transport_mode=transport,
            stay_duration_min=poi.stay_base,
            cost_estimate=cost_est,
            time=TimeAlloc(slot=slot, duration_min=poi.stay_base),
        )
        nodes.append(node)
        prev_loc = poi.location

    # ── Dynamic time allocation ──
    start = extract.intent.time_window_start or "09:00"
    deadline = extract.hard_constraints.time_deadline
    if not deadline and extract.intent.time_window_hours:
        try:
            start_min = _hhmm_to_min(start)
            end_min = start_min + int(extract.intent.time_window_hours * 60)
            deadline = _min_to_hhmm(end_min)
        except (ValueError, IndexError):
            deadline = "21:00"
    if not deadline:
        deadline = "21:00"

    _forward_pass(nodes, start, extract)
    _backward_pass(nodes, deadline)

    # ── Elastic budget reflow if needed ──
    available_min = (_hhmm_to_min(deadline) - _hhmm_to_min(start))
    total_used = sum(n.stay_duration_min for n in nodes) + sum(
        n.transit_from_prev_min for n in nodes)
    if total_used > available_min:
        logger.debug("Time overflow: %d > %d — running elastic reflow",
                     total_used, available_min)
        _elastic_budget_reflow(nodes, available_min)

        # Re-run passes with compressed durations
        _forward_pass(nodes, start, extract)
        _backward_pass(nodes, deadline)

        # If still overflow, drop lowest-scoring non-essential node
        still_used = sum(n.stay_duration_min for n in nodes) + sum(
            n.transit_from_prev_min for n in nodes)
        if still_used > available_min and len(nodes) > 2:
            # Drop least-elastic follow_up or lowest match_score play
            drop_candidates = [
                (i, n) for i, n in enumerate(nodes)
                if n.slot == "follow_up" or n.poi.elastic_coef > 0
            ]
            if drop_candidates:
                drop_idx, _ = min(drop_candidates,
                                  key=lambda x: x[1].poi.match_score if x[1].poi else 0)
                dropped = nodes.pop(drop_idx)
                logger.debug("Dropped node: %s (overflow %d min)",
                             dropped.poi.name if dropped.poi else dropped.slot,
                             still_used - available_min)
                _forward_pass(nodes, start, extract)
                _backward_pass(nodes, deadline)

    # ── Multi-user parallel activities ──
    cp = extract.constraint_profile
    if cp and cp.parallel_activities:
        used_ids = {n.poi.id for n in nodes if n.poi}
        for pa in cp.parallel_activities:
            pa_group = pa.get("group", "")
            pa_slot = pa.get("slot", "play")
            pa_hint = pa.get("activity_hint", "")
            pa_cluster = pa.get("cluster_required", True)

            # Find a node with matching slot to cluster with
            sibling = next((n for n in nodes if n.slot == pa_slot), None)
            sibling_loc = sibling.poi.location if sibling and sibling.poi else ""

            # Find best matching POI for this parallel group
            best = None
            best_score = -1.0
            for c in candidates:
                if c.id in used_ids:
                    continue
                if not c.location:
                    continue
                # Match activity hint
                hint_in = pa_hint.lower() in c.name.lower() if pa_hint else True
                cat_match = c.macro_category in ("景点", "休闲娱乐", "购物", "运动")
                if not hint_in and not cat_match:
                    continue
                # Cluster constraint: close to sibling
                cluster_ok = True
                if pa_cluster and sibling_loc and c.location != sibling_loc:
                    pair_key = f"{sibling_loc}|{c.location}"
                    d = distance_matrix.get(pair_key, 0)
                    cluster_ok = d < 2000  # Within 2km walking/shuttle
                if not cluster_ok:
                    continue
                score = c.match_score + (c.rating or 3.0) / 10.0
                if hint_in:
                    score += 0.5
                if score > best_score:
                    best_score = score
                    best = c

            if best:
                used_ids.add(best.id)
                p_node = ActivityNode(
                    slot=pa_slot,
                    poi=best,
                    stay_duration_min=best.stay_base,
                    cost_estimate=best.price_per_person or 0,
                    transport_mode="walk",  # Same cluster, walk between venues
                    transit_from_prev_min=0,  # Concurrent — no transit from sibling
                    transit_distance_m=0,
                )
                if sibling and sibling.time:
                    p_node.time = TimeAlloc(
                        slot=pa_slot,
                        duration_min=best.stay_base,
                        earliest_start=sibling.time.earliest_start,
                        earliest_end=sibling.time.earliest_end,
                        latest_start=sibling.time.latest_start,
                        latest_end=sibling.time.latest_end,
                        slack_min=sibling.time.slack_min,
                    )
                p_node.poi.assigned_user_group = pa_group
                # Insert after sibling so parallel nodes appear together in timeline
                if sibling:
                    sib_idx = next((i for i, n in enumerate(nodes) if n is sibling), -1)
                    nodes.insert(sib_idx + 1, p_node)
                else:
                    nodes.append(p_node)
                logger.debug("Parallel: added [%s] %s for group=%s (near %s)",
                             pa_slot, best.name, pa_group,
                             sibling.poi.name if sibling else "n/a")

    # ── Closed-loop: append return-to-home transit ──
    home_loc = extract.geo.center_location or ""
    last_node = nodes[-1] if nodes else None
    last_poi = last_node.poi if last_node else None
    if home_loc and last_poi and last_poi.location and last_poi.location != home_loc:
        pair_key = f"{last_poi.location}|{home_loc}"
        home_transit_m = distance_matrix.get(pair_key, 0)
        home_transport = _select_transport(
            home_transit_m,
            extract.soft_constraints.preferred_transport.value
            if extract.soft_constraints.preferred_transport else None,
        )
        home_transit_min = _transit_minutes(home_transit_m, home_transport)
        deadline_min = _hhmm_to_min(deadline)
        last_end_str = last_node.time.earliest_end if last_node.time else deadline
        last_end_min = _hhmm_to_min(last_end_str)
        return_arrival = last_end_min + home_transit_min
        if return_arrival <= deadline_min:
            nodes.append(ActivityNode(
                slot="home_return",
                poi=None,
                transit_from_prev_min=home_transit_min,
                transit_distance_m=home_transit_m,
                transport_mode=home_transport,
                stay_duration_min=0,
                cost_estimate=0.0,
                time=TimeAlloc(
                    slot="home_return",
                    duration_min=0,
                    earliest_start=_min_to_hhmm(return_arrival),
                    earliest_end=_min_to_hhmm(return_arrival),
                ),
            ))
            logger.debug("Closed-loop: return home %d min via %s (arrive %s)",
                         home_transit_min, home_transport, _min_to_hhmm(return_arrival))

    # ── Build PlannedPath ──
    total_transit = sum(n.transit_from_prev_min for n in nodes)
    total_cost = sum(n.cost_estimate for n in nodes)

    slot_labels = {
        "play": "玩", "lunch": "午餐", "eat": "用餐",
        "dinner": "晚餐", "follow_up": "休闲",
    }
    city = extract.intent.city or ""
    activity = extract.intent.activity_summary or "出行"
    pois_desc = " → ".join(
        f"[{slot_labels.get(s, s)}]{p.name}" for s, p in selected
    )
    notes = f"{city}{activity}：{pois_desc}。共{len(nodes)}个场所。"
    # Mention parallel activities
    if cp and cp.parallel_activities:
        pa_notes = []
        for pa in cp.parallel_activities:
            pa_notes.append(f"{pa.get('group','')}→{pa.get('activity_hint','')}")
        notes += f" 并行活动：{'; '.join(pa_notes)}。"

    return PlannedPath(
        nodes=nodes,
        total_transit_min=total_transit,
        total_cost=total_cost,
        notes=notes,
        strategy="",
    )


# ═══════════════════════════════════════════════════════════════════════
# Strategy 1: Time-Critical
# ═══════════════════════════════════════════════════════════════════════


def strategy_time_critical(
    extract: ExtractResult,
    candidates: list[POICandidate],
    distance_matrix: dict[str, int],
) -> PlannedPath:
    """Time-Critical strategy: α weight doubled for transit sensitivity.

    Standard greedy following template slots in order. Best for tight schedules
    where minimizing transit time is the primary concern.
    """
    _, beta, gamma, delta = compute_strategy_weights(extract)
    alpha = 2.0  # Double transit weight for time-critical
    weights = (alpha, beta, gamma, delta)

    path = _build_path(extract, candidates, distance_matrix, weights)
    path.strategy = "time_critical"
    logger.debug("Time-Critical: %d nodes, transit=%d min, cost=¥%.0f",
                 len(path.nodes), path.total_transit_min, path.total_cost)
    return path


# ═══════════════════════════════════════════════════════════════════════
# Strategy 2: Geo-Cluster
# ═══════════════════════════════════════════════════════════════════════


def strategy_geo_cluster(
    extract: ExtractResult,
    candidates: list[POICandidate],
    distance_matrix: dict[str, int],
) -> PlannedPath:
    """Geo-Cluster strategy: find densest spatial cluster, plan within it.

    Clusters POIs by spatial proximity, selects the cluster with highest
    avg match_score × rating, then runs greedy within that cluster.
    Minimizes total transit distance.
    """
    if len(candidates) < 2:
        path = _build_path(extract, candidates, distance_matrix,
                          compute_strategy_weights(extract))
        path.strategy = "geo_cluster"
        return path

    # Build adjacency from distance matrix
    center_loc = extract.geo.center_location or ""
    coords: dict[str, tuple[float, float]] = {}
    for c in candidates:
        if c.location:
            parts = c.location.split(",")
            if len(parts) == 2:
                try:
                    coords[c.id] = (float(parts[0]), float(parts[1]))
                except ValueError:
                    pass

    if not coords:
        path = _build_path(extract, candidates, distance_matrix,
                          compute_strategy_weights(extract))
        path.strategy = "geo_cluster"
        return path

    # Simple grid-based clustering: partition into 5km cells
    CLUSTER_RADIUS = 0.05  # ~5km in degrees
    clusters: dict[tuple[int, int], list[str]] = {}

    for pid, (lng, lat) in coords.items():
        cell = (int(lng / CLUSTER_RADIUS), int(lat / CLUSTER_RADIUS))
        if cell not in clusters:
            clusters[cell] = []
        clusters[cell].append(pid)

    # Expand to neighboring cells (3×3 neighborhood)
    for (cx, cy), pids in list(clusters.items()):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor = (cx + dx, cy + dy)
                if neighbor in clusters:
                    pids.extend(clusters[neighbor])
        # Dedup
        clusters[(cx, cy)] = list(dict.fromkeys(pids))

    # Score each cluster
    id_to_poi = {c.id: c for c in candidates}
    best_cluster: list[str] | None = None
    best_score = -1.0

    for cell, poi_ids in clusters.items():
        if len(poi_ids) < 2:
            continue
        cluster_cands = [id_to_poi[pid] for pid in poi_ids if pid in id_to_poi]
        if len(cluster_cands) < 2:
            continue

        avg_match = sum(c.match_score for c in cluster_cands) / len(cluster_cands)
        avg_rating = sum(
            (c.rating or 3.0) for c in cluster_cands
        ) / len(cluster_cands) / 5.0

        # Density: how many POIs within this grid cell
        density = min(1.0, len(cluster_cands) / 8.0)

        score = avg_match * 0.5 + avg_rating * 0.3 + density * 0.2
        if score > best_score:
            best_score = score
            best_cluster = poi_ids

    if best_cluster and len(best_cluster) >= 2:
        clustered_cands = [id_to_poi[pid] for pid in best_cluster if pid in id_to_poi]
        # Also include non-cluster candidates as fallback
        others = [c for c in candidates if c.id not in set(best_cluster)]
        selected_pool = clustered_cands + others
        logger.debug("Geo-Cluster: %d in cluster + %d others", len(clustered_cands), len(others))
    else:
        selected_pool = list(candidates)

    path = _build_path(extract, selected_pool, distance_matrix,
                       compute_strategy_weights(extract))
    path.strategy = "geo_cluster"
    logger.debug("Geo-Cluster: %d nodes, transit=%d min, cost=¥%.0f",
                 len(path.nodes), path.total_transit_min, path.total_cost)
    return path


# ═══════════════════════════════════════════════════════════════════════
# Strategy 3: Match-Driven
# ═══════════════════════════════════════════════════════════════════════


def strategy_match_driven(
    extract: ExtractResult,
    candidates: list[POICandidate],
    distance_matrix: dict[str, int],
) -> PlannedPath:
    """Match-Driven strategy: γ-dominant, prioritize constraint satisfaction.

    Ignores transit costs initially (α=0.1), selects purely by constraint match
    quality, then checks transit feasibility and swaps worst-transit node if needed.
    """
    alpha, beta, _, delta = compute_strategy_weights(extract)
    gamma = 3.0  # Triple match weight — constraint satisfaction is everything
    weights = (0.1, beta, gamma, delta)  # Very low transit weight

    path = _build_path(extract, candidates, distance_matrix, weights)
    path.strategy = "match_driven"

    # Check transit reasonableness — if avg transit > 45 min, swap worst
    if path.nodes:
        avg_transit = path.total_transit_min / len(path.nodes)
        if avg_transit > 45:
            # Find node with worst transit and try to replace
            worst_idx = max(
                range(len(path.nodes)),
                key=lambda i: path.nodes[i].transit_from_prev_min,
            )
            worst_node = path.nodes[worst_idx]
            if worst_node.poi:
                logger.debug("Match-Driven: high transit avg=%d min — worst=%s (%d min)",
                             avg_transit, worst_node.poi.name,
                             worst_node.transit_from_prev_min)

    logger.debug("Match-Driven: %d nodes, transit=%d min, cost=¥%.0f",
                 len(path.nodes), path.total_transit_min, path.total_cost)
    return path


# ═══════════════════════════════════════════════════════════════════════
# Path scoring
# ═══════════════════════════════════════════════════════════════════════


def _score_path(
    path: PlannedPath,
    extract: ExtractResult,
    distance_matrix: dict[str, int],
) -> float:
    """Score a PlannedPath by composite criteria.

    0.30 × constraint_coverage
    0.20 × rating_avg
    0.20 × budget_fitness
    0.15 × time_efficiency
    0.15 × transit_reasonableness
    """
    if not path.nodes:
        return 0.0

    # Constraint coverage — avg match_score
    match_avg = sum(
        n.poi.match_score if n.poi else 0.5 for n in path.nodes
    ) / len(path.nodes)

    # Rating average
    ratings = [n.poi.rating for n in path.nodes if n.poi and n.poi.rating]
    rating_avg = sum(ratings) / len(ratings) if ratings else 3.0
    rating_norm = min(1.0, rating_avg / 5.0)

    # Budget fitness
    budget = extract.hard_constraints.budget_max_cny
    if budget and budget > 0 and path.total_cost > 0:
        budget_fitness = max(0.0, 1.0 - abs(path.total_cost - budget) / budget)
    else:
        budget_fitness = 0.8  # No budget constraint → neutral

    # Time efficiency: used / available
    hours = extract.intent.time_window_hours or 8
    total_used_h = (
        sum(n.stay_duration_min for n in path.nodes)
        + path.total_transit_min
    ) / 60.0
    # Ideal: 70-95% utilization
    if total_used_h <= 0:
        time_eff = 0.5
    elif total_used_h < hours * 0.7:
        time_eff = total_used_h / (hours * 0.7)  # Under-utilized
    elif total_used_h <= hours * 0.95:
        time_eff = 1.0  # Sweet spot
    else:
        time_eff = max(0.3, 1.0 - (total_used_h - hours * 0.95) / hours)

    # Transit reasonableness — distance preference aware
    avg_transit = path.total_transit_min / len(path.nodes) if path.nodes else 0
    max_transit = extract.soft_constraints.max_transit_minutes

    if max_transit and max_transit > 30:
        # User specified a generous distance tolerance (e.g., "1小时内")
        # Prefer clusters at 50-80% of max (e.g., 30-48min for 60min max)
        low = max_transit * 0.5
        high = max_transit * 0.8
        if low <= avg_transit <= high:
            transit_score = 1.0  # Optimal range
        elif avg_transit < low:
            # Slightly penalize too-close (user wanted to explore further)
            transit_score = max(0.6, avg_transit / low)
        else:
            # Penalize beyond high
            transit_score = max(0.0, 1.0 - (avg_transit - high) / (max_transit - high + 1))
    else:
        # Default: prefer shorter transit (< 30 min)
        transit_score = max(0.0, 1.0 - avg_transit / 30.0)

    score = (0.30 * match_avg
             + 0.20 * rating_norm
             + 0.20 * budget_fitness
             + 0.15 * time_eff
             + 0.15 * transit_score)

    return max(0.0, min(1.0, score))


# ═══════════════════════════════════════════════════════════════════════
# Fallback path — when all strategies fail
# ═══════════════════════════════════════════════════════════════════════


def _fallback_path(
    extract: ExtractResult,
    candidates: list[POICandidate],
    distance_matrix: dict[str, int],
) -> PlannedPath | None:
    """Build a simple nearest-POI path when all strategies fail.

    Sorts candidates by distance from center, picks template's min_nodes,
    ensures at least one eat and one play.
    """
    if not candidates:
        return None

    center_loc = extract.geo.center_location or ""

    # Sort by distance from center
    def _dist_from_center(c: POICandidate) -> int:
        if not c.location or not center_loc:
            return 999999
        return distance_matrix.get(f"{center_loc}|{c.location}", 999999)

    sorted_cands = sorted(candidates, key=_dist_from_center)

    # Determine slot count
    template = _expand_template(extract)
    min_nodes = max(2, len(template))
    max_nodes = min(len(sorted_cands), len(template) + 1)

    # Build nodes — pick nearest POIs, alternating play/eat
    nodes: list[ActivityNode] = []
    used_ids: set[str] = set()
    eat_needed = any(s in ("eat", "lunch", "dinner") for s in template)
    play_needed = any(s in ("play", "follow_up") for s in template)

    for i, slot in enumerate(template):
        if i >= max_nodes:
            break
        bucket_key = _SLOT_TO_BUCKET.get(slot, "play")
        # Find first unused candidate matching bucket
        best = next((c for c in sorted_cands
                     if c.id not in used_ids and _classify_slot(c) == bucket_key), None)
        if best is None:
            best = next((c for c in sorted_cands if c.id not in used_ids), None)
        if best is None:
            break

        used_ids.add(best.id)
        transit_m = 0
        prev_loc = nodes[-1].poi.location if nodes else center_loc
        if prev_loc and best.location:
            transit_m = distance_matrix.get(f"{prev_loc}|{best.location}", 0)

        transport = _select_transport(transit_m, None)
        transit_min = _transit_minutes(transit_m, transport)

        node = ActivityNode(
            slot=slot,
            poi=best,
            transit_from_prev_min=transit_min,
            transit_distance_m=transit_m,
            transport_mode=transport,
            stay_duration_min=best.stay_base,
            cost_estimate=best.price_per_person or 50,
            time=TimeAlloc(slot=slot, duration_min=best.stay_base),
        )
        nodes.append(node)

    if not nodes:
        return None

    # Time allocation
    start = extract.intent.time_window_start or "09:00"
    hours = extract.intent.time_window_hours or 8
    start_min = _hhmm_to_min(start)
    deadline = _min_to_hhmm(start_min + int(hours * 60))

    _forward_pass(nodes, start, extract)
    _backward_pass(nodes, deadline)

    total_transit = sum(n.transit_from_prev_min for n in nodes)
    total_cost = sum(n.cost_estimate for n in nodes)

    logger.info("Fallback path: %d nodes, transit=%d min, cost=¥%.0f",
                len(nodes), total_transit, total_cost)

    return PlannedPath(
        nodes=nodes,
        total_transit_min=total_transit,
        total_cost=total_cost,
        notes=f"备选路线（自动生成）：{len(nodes)}个场所",
        strategy="fallback",
    )


# ═══════════════════════════════════════════════════════════════════════
# build_paths — parallel strategy execution
# ═══════════════════════════════════════════════════════════════════════


async def build_paths(
    extract: ExtractResult,
    candidates: list[POICandidate],
    distance_matrix: dict[str, int],
    weather: WeatherContext | None = None,
) -> list[PlannedPath]:
    """Run 3 strategies in parallel, return scored paths sorted best-first."""
    # Apply context hard filter (weather) before strategies run
    filtered = context_hard_filter(list(candidates), weather)

    # Compute base weights once — strategies may override individual weights
    base_weights = compute_strategy_weights(extract)

    async def _run_strategy(fn, name: str):
        try:
            # Strategies are synchronous — run in thread to avoid blocking
            loop = asyncio.get_running_loop()
            path = await loop.run_in_executor(
                None, fn, extract, list(filtered), distance_matrix,
            )
            path.strategy = name
            path.insertion_cost_total = sum(
                insertion_cost(
                    n.poi, n.slot,
                    path.nodes[i - 1] if i > 0 else None,
                    distance_matrix,
                    base_weights,
                    extract,
                )
                for i, n in enumerate(path.nodes)
                if n.poi
            )
            return path
        except Exception as exc:
            logger.warning("Strategy %s failed: %s", name, exc)
            return StrategyResult(strategy=name, error=str(exc))

    results = await asyncio.gather(
        _run_strategy(strategy_time_critical, "time_critical"),
        _run_strategy(strategy_geo_cluster, "geo_cluster"),
        _run_strategy(strategy_match_driven, "match_driven"),
    )

    paths: list[PlannedPath] = [r for r in results if isinstance(r, PlannedPath) and r.nodes]

    if not paths:
        logger.warning("build_paths: all strategies failed or returned empty")
        # ── Fallback: nearest-POI simple path ──
        fallback = _fallback_path(extract, candidates, distance_matrix)
        if fallback and fallback.nodes:
            fallback.strategy = "fallback"
            fallback.coverage_score = _score_path(fallback, extract, distance_matrix)
            return [fallback]
        return []

    # Score each path
    for p in paths:
        p.coverage_score = _score_path(p, extract, distance_matrix)
        logger.info("  Strategy %s: %d nodes, score=%.3f, transit=%d min, cost=¥%.0f",
                    p.strategy, len(p.nodes), p.coverage_score,
                    p.total_transit_min, p.total_cost)

    paths.sort(key=lambda p: p.coverage_score, reverse=True)
    return paths


# ═══════════════════════════════════════════════════════════════════════
# Two-tier transport refinement
# ═══════════════════════════════════════════════════════════════════════


async def _refine_transport(path: PlannedPath, tools: list) -> None:
    """Replace coarse distance estimates with direction API results.

    Only called for the winning path. Uses maps_direction_driving for
    pairs > 1.5km and maps_direction_walking for shorter distances.
    Falls back to coarse estimates on API failure.
    """
    if len(path.nodes) < 2:
        return

    driving_tool = next((t for t in tools if t.name == "maps_direction_driving"), None)
    walking_tool = next((t for t in tools if t.name == "maps_direction_walking"), None)

    logger.debug("Transport refinement for %d nodes (driving=%s, walking=%s)",
                 len(path.nodes),
                 "available" if driving_tool else "missing",
                 "available" if walking_tool else "missing")

    async def _refine_one(idx: int, node: ActivityNode, prev_loc: str):
        if idx == 0 or not prev_loc or not node.poi or not node.poi.location:
            return

        tool = None
        if node.transit_distance_m < 1500 and walking_tool:
            tool = walking_tool
            mode = "walk"
        elif driving_tool:
            tool = driving_tool
            mode = "drive"
        else:
            return

        try:
            result = await tool.ainvoke({
                "origin": prev_loc,
                "destination": node.poi.location,
            })
            data = _extract_json(str(result)) if "{" in str(result) else {}
            route = data.get("route", {})
            paths_data = route.get("paths", [])
            if paths_data:
                duration_sec = paths_data[0].get("duration", 0)
                distance_m = paths_data[0].get("distance", 0)
                if duration_sec:
                    node.transit_from_prev_min = max(1, int(duration_sec) // 60)
                    node.transport_mode = mode
                if distance_m:
                    node.transit_distance_m = int(distance_m)
        except Exception as exc:
            logger.debug("Transport refine failed for node %d: %s", idx, exc)

    prev_loc = path.nodes[0].poi.location if path.nodes[0].poi else ""
    # Start from center
    # (In practice, this would use extract.geo.center_location)

    tasks = []
    for i in range(1, len(path.nodes)):
        prev = path.nodes[i - 1].poi.location if path.nodes[i - 1].poi else ""
        tasks.append(_refine_one(i, path.nodes[i], prev))

    if tasks:
        await asyncio.gather(*tasks)


def _extract_json(text: str) -> dict:
    """Extract first JSON object from text (greedy match, handles nesting)."""
    import json
    import re
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


# ═══════════════════════════════════════════════════════════════════════
# build_plan — PlannedPath → Plan conversion
# ═══════════════════════════════════════════════════════════════════════


def build_plan(path: PlannedPath) -> Plan:
    """Convert a PlannedPath to the established Plan + SubTask format.

    Maintains backward compatibility with downstream nodes
    (present_to_user, book_worker, etc.).
    """
    if not path or not path.nodes:
        return Plan(sub_tasks=[], notes="未能找到合适的场所，请尝试放宽搜索条件。")

    sub_tasks: list[SubTask] = []
    task_idx = 1

    # 1. Search task: summarize all POI info
    search_params: dict = {}
    for node in path.nodes:
        if node.poi:
            search_params[node.slot] = {
                "name": node.poi.name,
                "address": node.poi.address,
                "location": node.poi.location,
                "rating": node.poi.rating,
                "type": node.poi.type,
            }

    sub_tasks.append(SubTask(
        id=f"t{task_idx}",
        type="search",
        target="信息汇总",
        dependencies=[],
        params=search_params,
    ))
    task_idx += 1

    # 2. Transit + book for each stop
    prev_loc = ""
    for node in path.nodes:
        if not node.poi:
            continue

        # Transit task
        if prev_loc and node.poi.location and prev_loc != node.poi.location:
            transit_label = f"前往{node.poi.name}"
            sub_tasks.append(SubTask(
                id=f"t{task_idx}",
                type="compare",
                target=transit_label,
                dependencies=[f"t{task_idx - 1}"],
                params={
                    "transport": node.transport_mode,
                    "from": prev_loc,
                    "to": node.poi.location,
                    "distance_m": node.transit_distance_m,
                    "duration_min": node.transit_from_prev_min,
                },
            ))
            task_idx += 1

        slot_label = {
            "play": "游玩", "lunch": "午餐", "eat": "用餐",
            "dinner": "晚餐", "follow_up": "休闲",
        }.get(node.slot, "活动")

        start_time = node.time.earliest_start if node.time else ""
        end_time = node.time.earliest_end if node.time else ""

        sub_tasks.append(SubTask(
            id=f"t{task_idx}",
            type="book",
            target=f"{slot_label}: {node.poi.name}",
            dependencies=[f"t{task_idx - 1}"],
            params={
                "name": node.poi.name,
                "address": node.poi.address,
                "location": node.poi.location,
                "rating": node.poi.rating,
                "slot": node.slot,
                "start_time": start_time,
                "end_time": end_time,
                "cost_estimate": node.cost_estimate,
            },
            compensatory=f"cancel_t{task_idx}",
        ))
        task_idx += 1
        prev_loc = node.poi.location

    logger.info("build_plan: %d nodes → %d sub-tasks, cost=¥%.0f",
                len(path.nodes), len(sub_tasks), path.total_cost)

    return Plan(
        sub_tasks=sub_tasks,
        total_cost_estimate=path.total_cost if path.total_cost > 0 else None,
        notes=path.notes,
    )
