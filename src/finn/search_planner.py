"""Search Planner — LLM-driven search strategy formulation + execution.

Flow:
  1. formulate_search: LLM analyzes constraints, designs category-based search strategy
  2. execute_category_search: Parallel MCP search per category (max 2 rounds)
  3. route_after_search: Evaluate coverage → loop or proceed to multi-agent planning

Key design:
  - LLM formulates WHAT to search (categories, keywords, priorities)
  - Code executes HOW to search (MCP batch calls, dedup, ranking)
  - Max 2 search rounds with coverage evaluation in between
"""

from __future__ import annotations

import asyncio
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from finn.config import config
from finn.llm import extract_json, llm_invoke, make_model
from finn.logger import logger
from finn.mcp import mcp_session
from finn.state import (
    AgentState,
    CategorySearch,
    ExtractResult,
    POICandidate,
    POICategoryPool,
    SearchStrategy,
)

# ═══════════════════════════════════════════════════════════════════════
# Prompt: formulate search strategy from user constraints
# ═══════════════════════════════════════════════════════════════════════

_FORMULATE_SEARCH_PROMPT = """\
你是 Finn 的搜索策略制定模块。

你会收到用户的出行约束和需求。你的任务：制定高覆盖率、广泛搜索的 POI 搜索策略。

══════════════════════════════════════
搜索策略规则
══════════════════════════════════════

1. **一级分类覆盖**：必须覆盖以下类别中的相关类别：
   - 餐饮: 餐厅、火锅、面馆、咖啡馆、烧烤、日料等
   - 购物: 商场、购物中心、步行街、超市等
   - 文化: 博物馆、美术馆、展览、书店、图书馆等
   - 景点: 公园、动物园、植物园、景区、古镇等
   - 休闲娱乐: 电影院、KTV、桌游、密室、温泉等
   - 运动: 游泳、健身、滑雪、攀岩、卡丁车等
   - 户外: 徒步、登山、露营、滨江等（天气好时）
   - 室内: 商场、博物馆、展览馆等（天气差时优先）

2. **强制类别基线**（必须包含，不可跳过）：
   - 家庭场景：至少包含 景点 + 餐饮 + 文化 + 购物 四个类别
   - 朋友场景：至少包含 餐饮 + 休闲娱乐 + 购物 + 运动 四个类别
   - 天气差时：至少包含 室内 + 餐饮 + 购物 + 文化 四个类别
   一共选择 4-6 个类别。

3. **关键词设计**：每个类别 3-8 个具体中文搜索关键词。
   - 结合用户偏好菜系、活动类型、必去地点
   - 包含场景默认标签（如亲子→"亲子餐厅"、"儿童乐园"）
   - 包含高评分关键词变体（如"精品咖啡馆"、"热门火锅"）

4. **搜索方式**：
   - "both": 同时使用 around_search（周边搜索）和 text_search（全城搜索）
   - "around": 仅周边搜索（适合餐饮、购物等近距离需求）
   - "text": 仅全城搜索（适合景点、文化等距离容忍度高的类别）

5. **优先级**：priority 0-1，基于用户需求和约束重要性。

6. OUTDOOR WARNING: 如果天气不好（下雨/雪/高温），降低户外/景点类别的 priority，
   提高室内/购物/文化类别的 priority。

输出格式（只输出合法 JSON，不要 markdown）：
{
  "categories": [
    {
      "category": "<类别名>",
      "keywords": ["<关键词1>", "<关键词2>", ...],
      "search_type": "both" | "around" | "text",
      "target_count": <int 5-15>,
      "priority": <float 0-1>
    }
  ],
  "radius_m": <搜索半径 米>,
  "reasoning": "<中文，为什么选择这些类别和关键词>"
}
"""

_ROUND2_PROMPT = """\
你是 Finn 的搜索策略制定模块 — 第二轮搜索。

第一轮搜索结果已经返回，但某些类别覆盖不足。请根据以下信息制定补充搜索策略。

规则：
1. 只针对覆盖不足的类别进行补充搜索
2. 使用不同的关键词变体（如换同义词、调整价格段、改变区域）
3. 可以扩大搜索半径或改用不同搜索方式
4. 优先级集中在覆盖不足的类别

输出格式（同上）：
{
  "categories": [...],
  "radius_m": <int>,
  "reasoning": "<中文>"
}
"""


def _build_extract_context(extract: ExtractResult) -> str:
    """Build a human-readable context string from ExtractResult."""
    i = extract.intent
    parts: list[str] = []

    if i.activity_summary:
        parts.append(f"出行意图: {i.activity_summary}")
    if i.city:
        parts.append(f"城市: {i.city}")
    if i.plan_date:
        parts.append(f"日期: {i.plan_date}")
    if i.time_window_start:
        parts.append(f"开始时间: {i.time_window_start}")
    if i.time_window_hours:
        parts.append(f"可用时长: {i.time_window_hours}小时")
    if i.scene:
        parts.append(f"场景: {'家庭' if i.scene.value == 'family' else '朋友'}")
    if i.guest_count:
        parts.append(f"人数: {i.guest_count}")

    hc = extract.hard_constraints
    if hc.budget_max_cny:
        parts.append(f"预算上限: ¥{hc.budget_max_cny}")
    if hc.dietary_restrictions:
        parts.append(f"饮食限制: {', '.join(hc.dietary_restrictions)}")
    if hc.child_age is not None:
        parts.append(f"儿童年龄: {hc.child_age}岁")

    sc = extract.soft_constraints
    if sc.preferred_cuisines:
        parts.append(f"偏好菜系: {', '.join(sc.preferred_cuisines)}")
    if sc.preferred_poi_types:
        parts.append(f"偏好POI类型: {', '.join(sc.preferred_poi_types)}")
    if sc.avoid_poi_types:
        parts.append(f"避开类型: {', '.join(sc.avoid_poi_types)}")
    if sc.budget_preference:
        parts.append(f"预算偏好: {sc.budget_preference.value}")
    if sc.travel_pace:
        parts.append(f"节奏: {sc.travel_pace.value}")

    req = extract.requirements
    if req.must_visit_pois:
        parts.append(f"必去地点: {', '.join(req.must_visit_pois)}")
    if req.must_have_cuisine:
        parts.append(f"必吃菜系: {', '.join(req.must_have_cuisine)}")

    gp = extract.group
    if gp.tags:
        parts.append(f"人群标签: {', '.join(gp.tags)}")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# Node: formulate_search
# ═══════════════════════════════════════════════════════════════════════


async def formulate_search(state: AgentState) -> dict:
    """LLM-driven node: analyze constraints → design search strategy.

    Routes via Command(goto=...):
      Always → execute_category_search
    """
    extract = state.get("extract_result")
    if extract is None:
        logger.warning("formulate_search called without extract_result")
        return Command(goto="multi_agent_plan", update={"search_strategy": None})

    search_round = state.get("search_round", 0)
    weather = state.get("weather")

    # Build context
    extract_context = _build_extract_context(extract)

    weather_text = ""
    if weather and weather.condition:
        weather_text = (
            f"\n天气: {weather.date} {weather.condition} "
            f"{f'{weather.temp_low:.0f}~{weather.temp_high:.0f}°C' if weather.temp_low and weather.temp_high else ''}"
            f"{' | 建议室内活动' if weather.indoor_recommended else ''}"
        )

    center = extract.geo.center_location or state.get("user_coords", "")

    # Check existing pools for round 2
    existing_pools = state.get("category_pools", {})
    pool_summary = ""
    if existing_pools and search_round >= 1:
        pool_lines = ["\n第一轮搜索结果摘要:"]
        for cat, pool in existing_pools.items():
            pool_lines.append(
                f"  {cat}: {len(pool.candidates)}个候选 "
                f"(覆盖度 {pool.coverage_score:.2f})"
            )
        pool_summary = "\n".join(pool_lines)

    system_prompt = _FORMULATE_SEARCH_PROMPT if search_round == 0 else _ROUND2_PROMPT
    user_prompt = f"""用户约束:
{extract_context}
{weather_text}

当前位置: {center}
{pool_summary}

请制定搜索策略，输出 JSON。"""

    try:
        llm = make_model(temperature=0.3)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        text = await llm_invoke(llm, messages, "formulate_search", stream=True)
        data = extract_json(text)

        categories = []
        for c in data.get("categories", []):
            categories.append(CategorySearch(
                category=str(c.get("category", "")),
                keywords=[str(k) for k in c.get("keywords", [])],
                search_type=str(c.get("search_type", "both")),
                target_count=int(c.get("target_count", 10)),
                priority=float(c.get("priority", 1.0)),
            ))

        strategy = SearchStrategy(
            categories=categories,
            radius_m=int(data.get("radius_m", extract.geo.radius_m)),
            reasoning=str(data.get("reasoning", "")),
            round_number=search_round + 1,
        )

        logger.info("Search strategy R%d: %d categories — %s",
                     strategy.round_number, len(categories), strategy.reasoning[:80])
        for c in categories:
            logger.debug("  %s (p=%.1f): %s [%s]",
                         c.category, c.priority, c.keywords[:4], c.search_type)

        return Command(
            goto="execute_category_search",
            update={"search_strategy": strategy},
        )
    except Exception as exc:
        logger.warning("formulate_search failed: %s — using fallback strategy", exc)
        return _fallback_search_strategy(extract, search_round)


def _fallback_search_strategy(extract: ExtractResult, round_num: int) -> dict:
    """Rule-based fallback search strategy when LLM fails."""
    scene = extract.group.type.value if extract.group.type else "family"
    categories: list[CategorySearch] = []

    if scene == "family":
        categories = [
            CategorySearch(category="景点", keywords=["公园", "动物园", "博物馆", "植物园", "景区", "科技馆"],
                          search_type="both", target_count=10, priority=1.0),
            CategorySearch(category="餐饮", keywords=["亲子餐厅", "面馆", "简餐", "火锅", "粤菜", "西餐"],
                          search_type="both", target_count=10, priority=0.9),
            CategorySearch(category="文化", keywords=["博物馆", "美术馆", "科技馆", "图书馆", "展览"],
                          search_type="text", target_count=8, priority=0.8),
            CategorySearch(category="购物", keywords=["商场", "购物中心", "步行街"],
                          search_type="around", target_count=5, priority=0.6),
            CategorySearch(category="休闲娱乐", keywords=["电影院", "儿童乐园", "游乐场", "亲子"],
                          search_type="both", target_count=5, priority=0.5),
        ]
    else:
        categories = [
            CategorySearch(category="餐饮", keywords=["火锅", "烧烤", "网红餐厅", "日料", "咖啡馆", "西餐"],
                          search_type="both", target_count=12, priority=1.0),
            CategorySearch(category="休闲娱乐", keywords=["电影院", "KTV", "桌游", "密室", "酒吧"],
                          search_type="both", target_count=10, priority=0.9),
            CategorySearch(category="购物", keywords=["商场", "步行街", "购物中心", "集市"],
                          search_type="around", target_count=8, priority=0.7),
            CategorySearch(category="运动", keywords=["健身", "游泳", "攀岩", "卡丁车", "保龄球"],
                          search_type="text", target_count=6, priority=0.5),
            CategorySearch(category="景点", keywords=["公园", "景区", "网红打卡", "拍照"],
                          search_type="both", target_count=5, priority=0.4),
        ]

    # Inject user preferences into keywords
    preferred_cuisines = extract.soft_constraints.preferred_cuisines
    preferred_types = extract.soft_constraints.preferred_poi_types
    for c in categories:
        if c.category == "餐饮" and preferred_cuisines:
            c.keywords = preferred_cuisines[:3] + c.keywords
        if preferred_types:
            c.keywords = preferred_types[:2] + c.keywords

    strategy = SearchStrategy(
        categories=categories,
        radius_m=extract.geo.radius_m,
        reasoning="Fallback: rule-based strategy due to LLM parsing failure",
        round_number=round_num + 1,
    )

    return Command(
        goto="execute_category_search",
        update={"search_strategy": strategy},
    )


# ═══════════════════════════════════════════════════════════════════════
# Node: execute_category_search
# ═══════════════════════════════════════════════════════════════════════


async def execute_category_search(state: AgentState) -> dict:
    """Execute parallel POI search per category from the search strategy.

    Routes:
      - If coverage insufficient and round < 2 → formulate_search (round 2)
      - Else → multi_agent_plan
    """
    strategy = state.get("search_strategy")
    extract = state.get("extract_result")
    search_round = state.get("search_round", 0) + 1

    if not strategy or not extract:
        logger.warning("execute_category_search called without strategy/extract")
        return Command(goto="multi_agent_plan", update={"search_round": search_round})

    existing_pools = dict(state.get("category_pools", {}))

    t0 = time.monotonic()
    new_pools: dict[str, POICategoryPool] = {}

    try:
        async with mcp_session() as (tools, _server_id):
            # Run all category searches in parallel
            tasks = [
                _search_category(tools, cat, extract, strategy.radius_m)
                for cat in strategy.categories
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, (cat, result) in enumerate(zip(strategy.categories, results)):
                if isinstance(result, Exception):
                    logger.warning("Category '%s' search failed: %s", cat.category, result)
                    candidates, cov_score = [], 0.0
                else:
                    candidates, cov_score = result
                # Merge with existing pool if round 2
                if cat.category in existing_pools:
                    existing = existing_pools[cat.category]
                    merged = _merge_candidates(existing.candidates, candidates)
                    new_pools[cat.category] = POICategoryPool(
                        category=cat.category,
                        candidates=merged,
                        coverage_score=max(cov_score, existing.coverage_score),
                    )
                else:
                    new_pools[cat.category] = POICategoryPool(
                        category=cat.category,
                        candidates=candidates,
                        coverage_score=cov_score,
                    )
    except Exception as exc:
        logger.error("execute_category_search failed: %s", exc)
        return Command(
            goto="multi_agent_plan",
            update={"search_round": search_round, "category_pools": existing_pools},
        )

    elapsed = time.monotonic() - t0
    total_candidates = sum(len(p.candidates) for p in new_pools.values())

    # Log results
    for cat, pool in new_pools.items():
        logger.info("Category %s: %d candidates (coverage=%.2f)",
                     cat, len(pool.candidates), pool.coverage_score)

    logger.info("Search R%d: %d total candidates in %.1fs",
                search_round, total_candidates, elapsed)

    # Evaluate coverage
    coverage_sufficient = _evaluate_coverage(new_pools, strategy)

    if not coverage_sufficient and search_round < 2:
        logger.info("Coverage insufficient after R%d — scheduling round 2", search_round)
        return Command(
            goto="formulate_search",
            update={
                "search_round": search_round,
                "category_pools": new_pools,
            },
        )

    # If still insufficient after max rounds, inject missing mandatory categories
    if not coverage_sufficient and search_round >= 2:
        logger.warning("Coverage still insufficient after %d rounds — injecting fallback categories",
                       search_round)
        missing = _inject_missing_categories(new_pools, extract)
        if missing:
            for mc in missing:
                new_pools[mc.category] = mc
                logger.info("Injected fallback category: %s (%d candidates)",
                           mc.category, len(mc.candidates))

    # Coverage sufficient or max rounds reached → proceed
    return Command(
        goto="multi_agent_plan",
        update={
            "search_round": search_round,
            "category_pools": new_pools,
            # Also populate legacy poi_candidates for downstream compatibility
            "poi_candidates": _flatten_candidates(new_pools),
        },
    )


# ═══════════════════════════════════════════════════════════════════════
# Category search execution
# ═══════════════════════════════════════════════════════════════════════


async def _search_category(
    tools: list,
    cat: CategorySearch,
    extract: ExtractResult,
    default_radius: int,
) -> tuple[list[POICandidate], float]:
    """Search for one category — returns (candidates, coverage_score)."""
    from finn.hub import mcp_hub
    from finn.poi import deduplicate_pois, rank_pois
    from finn.nodes import _parse_amap_pois, _raw_to_candidate

    candidates: list[POICandidate] = []
    center_loc = extract.geo.center_location or ""
    city = extract.intent.city or ""

    around_tool = next((t for t in tools if t.name == "maps_around_search"), None)
    text_tool = next((t for t in tools if t.name == "maps_text_search"), None)
    detail_tool = next((t for t in tools if t.name == "maps_search_detail"), None)

    async def _around_search():
        if center_loc and around_tool and cat.search_type in ("around", "both"):
            batch_results = await mcp_hub.batch_around_search(
                tools, center_loc, cat.keywords, default_radius,
            )
            for kw, raw_pois in batch_results:
                for raw in raw_pois:
                    poi = _raw_to_candidate(raw, kw)
                    if poi:
                        candidates.append(poi)

    async def _text_searches():
        if text_tool and cat.search_type in ("text", "both"):
            async def _one_text(kw: str):
                try:
                    result = await text_tool.ainvoke({
                        "keywords": kw, "city": city,
                    })
                    return _parse_amap_pois(str(result), kw)
                except Exception:
                    return []

            tasks = [_one_text(kw) for kw in cat.keywords[:4]]
            results = await asyncio.gather(*tasks)
            for parsed in results:
                candidates.extend(parsed)

    await asyncio.gather(_around_search(), _text_searches())

    # Dedup
    candidates = deduplicate_pois(candidates)

    # Fetch details for top candidates (limited to target_count)
    detail_targets = candidates[:cat.target_count]
    if detail_targets and detail_tool:
        detail_ids = [c.id for c in detail_targets if c.id]
        details = await mcp_hub.batch_search_detail(tools, detail_ids)
        for poi in detail_targets:
            detail = details.get(poi.id, {})
            if detail:
                _enrich_poi_from_detail(poi, detail)

    # Rank
    candidates = rank_pois(candidates, extract)

    # Coverage score
    coverage = min(1.0, len(candidates) / max(cat.target_count, 1))

    return candidates, coverage


def _enrich_poi_from_detail(poi: POICandidate, detail: dict) -> None:
    """Enrich a POICandidate with detail data."""
    from finn.nodes import _parse_opentime, _parse_rating, _set_stay_params

    poi_data = detail
    if "pois" in detail and isinstance(detail["pois"], list):
        poi_data = detail["pois"][0] if detail["pois"] else detail

    if isinstance(poi_data, dict):
        loc = str(poi_data.get("location", ""))
        if loc and not poi.location:
            poi.location = loc

        biz = poi_data.get("biz_ext", {}) if isinstance(poi_data, dict) else {}
        open_time_raw = (
            poi_data.get("open_time", "") or poi_data.get("opentime2", "")
            or biz.get("opentime", "") or biz.get("open_time", "")
        )
        if open_time_raw:
            poi.open_time, poi.close_time = _parse_opentime(str(open_time_raw))

        cost = biz.get("cost", "") or poi_data.get("cost", "")
        if cost:
            try:
                poi.price_per_person = float(str(cost))
            except (ValueError, TypeError):
                pass

        rating = poi_data.get("rating", "") or biz.get("rating", "")
        if rating and poi.rating is None:
            poi.rating = _parse_rating(rating)

        _set_stay_params(poi)


def _merge_candidates(
    existing: list[POICandidate], new: list[POICandidate],
) -> list[POICandidate]:
    """Merge two candidate lists, deduplicating by ID and name."""
    from finn.poi import deduplicate_pois
    combined = existing + new
    return deduplicate_pois(combined)


def _evaluate_coverage(
    pools: dict[str, POICategoryPool], strategy: SearchStrategy,
) -> bool:
    """Check if overall coverage is sufficient to proceed to planning.

    Requires both:
    - At least 8 total candidates across all pools
    - At least 3 distinct categories with >= 3 candidates each
    - Each high-priority category (>= 0.7) must have >= 3 candidates
    """
    if not pools:
        return False

    total = sum(len(p.candidates) for p in pools.values())
    if total < 8:
        return False

    # Must have at least 3 categories with meaningful coverage
    categories_with_coverage = sum(
        1 for p in pools.values() if len(p.candidates) >= 3
    )
    if categories_with_coverage < 3:
        return False

    # Each high-priority category should have at least 3 candidates
    for cat in strategy.categories:
        if cat.priority >= 0.7:
            pool = pools.get(cat.category)
            if pool and len(pool.candidates) < 3:
                return False

    return True


def _flatten_candidates(pools: dict[str, POICategoryPool]) -> list[POICandidate]:
    """Flatten category pools into a single candidate list."""
    seen: set[str] = set()
    result: list[POICandidate] = []
    for pool in pools.values():
        for c in pool.candidates:
            if c.id and c.id not in seen:
                seen.add(c.id)
                result.append(c)
            elif not c.id and c.name not in seen:
                seen.add(c.name)
                result.append(c)
    return result


def _inject_missing_categories(
    pools: dict[str, POICategoryPool],
    extract: ExtractResult,
) -> list[POICategoryPool]:
    """If key categories are missing after max rounds, inject empty placeholders
    so downstream planners at least know which categories should have been searched.

    These are empty pools — they signal to the planner that these categories
    are expected but no results were found, rather than being silently absent.
    """
    scene = extract.group.type.value if extract.group.type else "family"

    if scene == "family":
        mandatory = ["景点", "餐饮", "文化", "购物"]
    else:
        mandatory = ["餐饮", "休闲娱乐", "购物", "运动"]

    missing: list[POICategoryPool] = []
    for cat in mandatory:
        if cat not in pools:
            missing.append(POICategoryPool(
                category=cat,
                candidates=[],
                coverage_score=0.0,
            ))

    return missing


# ═══════════════════════════════════════════════════════════════════════
# Edge routing
# ═══════════════════════════════════════════════════════════════════════


def route_after_search(state: AgentState) -> str:
    """Not used — routing is done via Command(goto=...) inside nodes."""
    return "multi_agent_plan"
