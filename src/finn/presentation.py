"""PresentationAgent — PlanCard generation for user display.

Pre-assembles PlanCards with Amap nav URLs and transport estimates
before the HITL interrupt, so the CLI can render structured cards.

Uses PlannedPath timing data when available for accurate per-stop times;
falls back to Plan-based estimation for legacy compatibility.
"""

from __future__ import annotations

from finn.state import Plan, PlanCard, PlannedPath, POICandidate, WeatherContext


def generate_plan_cards(
    plan: Plan,
    poi_candidates: list[POICandidate],
    distance_matrix: dict[str, int] | None = None,
    weather: WeatherContext | None = None,
    selected_path: PlannedPath | None = None,
) -> list[PlanCard]:
    """Build display cards for each planned stop.

    If ``selected_path`` is provided, uses accurate timing from the
    PlannedPath (forward/backward pass results). Otherwise falls back
    to fixed 2h-per-stop estimation from the Plan.
    """
    if selected_path and selected_path.nodes:
        return _cards_from_path(selected_path)

    return _cards_from_plan(plan, poi_candidates, distance_matrix)


def _cards_from_path(path: PlannedPath) -> list[PlanCard]:
    """Build PlanCards from PlannedPath with accurate timing."""
    cards: list[PlanCard] = []

    for i, node in enumerate(path.nodes):
        if not node.poi:
            continue
        poi = node.poi
        time = node.time

        start_time = time.earliest_start if time else ""
        end_time = time.earliest_end if time else ""

        # Detail line
        detail_parts = []
        if poi.rating:
            detail_parts.append(f"评分{poi.rating}")
        if poi.price_per_person:
            detail_parts.append(f"人均¥{int(poi.price_per_person)}")
        detail_line = " | ".join(detail_parts) if detail_parts else ""

        # Transport from previous
        transport_line = ""
        taxi_line = ""
        if node.transit_from_prev_min > 0 or node.transit_distance_m > 0:
            mins = node.transit_from_prev_min
            label = {"walk": "步行", "transit": "公交", "drive": "驾车"}.get(
                node.transport_mode, "出行")
            if node.transit_distance_m > 0:
                km = node.transit_distance_m / 1000
                transport_line = f"{label}约{mins}分钟 ({km:.1f}km)"
            else:
                transport_line = f"{label}约{mins}分钟"

            if node.transport_mode == "drive" and node.transit_distance_m > 0:
                km = node.transit_distance_m / 1000
                taxi_line = f"打车约¥{max(8, int(km * 2.5))}"

        # Amap nav URL
        nav_url = ""
        if poi.location:
            lng, lat = _parse_lnglat(poi.location)
            if lng and lat:
                nav_url = (
                    f"https://uri.amap.com/navigation?"
                    f"to={lng},{lat},{poi.name}&mode=drive"
                )

        slot_labels = {
            "play": "游玩", "lunch": "午餐", "eat": "用餐",
            "dinner": "晚餐", "follow_up": "休闲",
        }
        activity_type = slot_labels.get(node.slot, node.slot)

        # Show user group for parallel activities
        if poi.assigned_user_group:
            group_labels = {"adults": "成人", "kids": "儿童", "elderly": "老人"}
            group_tag = group_labels.get(poi.assigned_user_group, poi.assigned_user_group)
            activity_type = f"{activity_type}({group_tag})"

        cards.append(PlanCard(
            step=i + 1,
            poi_id=poi.id,
            poi_name=poi.name,
            activity_type=activity_type,
            start_time=start_time,
            end_time=end_time,
            detail_line=detail_line,
            address=poi.address,
            transport_from_prev=transport_line,
            amap_nav_url=nav_url,
            taxi_estimate=taxi_line,
        ))

    return cards


def _cards_from_plan(
    plan: Plan,
    poi_candidates: list[POICandidate],
    distance_matrix: dict[str, int] | None = None,
) -> list[PlanCard]:
    """Legacy: build PlanCards from Plan with estimated 2h slots."""
    dist = distance_matrix or {}
    cards: list[PlanCard] = []

    book_tasks = [t for t in plan.sub_tasks if t.type == "book"]
    if not book_tasks:
        return cards

    prev_location = ""
    hour = 9

    for i, task in enumerate(book_tasks):
        params = task.params or {}
        name = str(params.get("name", task.target or ""))
        address = str(params.get("address", ""))
        location = str(params.get("location", ""))
        rating = params.get("rating")
        slot = str(params.get("slot", "play"))

        slot_labels = {
            "play": "游玩", "lunch": "午餐", "eat": "用餐",
            "dinner": "晚餐", "follow_up": "休闲",
        }
        activity_label = slot_labels.get(slot, slot)

        detail_parts = []
        if rating:
            detail_parts.append(f"评分{rating}")
        if slot in ("eat", "lunch", "dinner"):
            detail_parts.append("用餐")
        detail_line = " | ".join(detail_parts) if detail_parts else ""

        transport_line = ""
        taxi_line = ""
        if prev_location and location and dist:
            pair_key = f"{prev_location}|{location}"
            d = dist.get(pair_key, 0)
            if d > 0:
                mins = max(1, d // 400)
                km = d / 1000
                if d < 1000:
                    transport_line = f"步行约{mins}分钟 ({d}m)"
                else:
                    transport_line = f"驾车约{mins}分钟 ({km:.1f}km)"
                taxi_line = f"打车约¥{max(8, int(km * 2.5))}"

        nav_url = ""
        if location:
            lng, lat = _parse_lnglat(location)
            if lng and lat:
                nav_url = (
                    f"https://uri.amap.com/navigation?"
                    f"to={lng},{lat},{name}&mode=drive"
                )

        # Use actual times from plan if available, otherwise fixed 2h slots
        plan_start = str(params.get("start_time", ""))
        plan_end = str(params.get("end_time", ""))
        if plan_start and plan_end:
            start_time = plan_start
            end_time = plan_end
            # Update hour tracker for subsequent slots
            try:
                end_h = int(plan_end.split(":")[0])
                hour = end_h
            except (ValueError, IndexError):
                pass
        else:
            start_h = hour
            end_h = hour + 2
            hour = end_h
            start_time = f"{start_h:02d}:00"
            end_time = f"{end_h:02d}:00"

        cards.append(PlanCard(
            step=i + 1,
            poi_id=str(params.get("id", "")),
            poi_name=name,
            activity_type=activity_label,
            start_time=start_time,
            end_time=end_time,
            detail_line=detail_line,
            address=address,
            transport_from_prev=transport_line,
            amap_nav_url=nav_url,
            taxi_estimate=taxi_line,
        ))
        prev_location = location

    return cards


def _parse_lnglat(loc: str) -> tuple[str, str]:
    """Parse 'lng,lat' string into (lng, lat) tuple."""
    parts = loc.split(",")
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "", ""
