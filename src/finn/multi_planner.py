"""Multi-Agent Planner — 2 parallel LLM agents with different strategies.

Each agent receives the same POI category pools and user constraints, but is
prompted with a different optimization strategy. Both run concurrently via
asyncio.gather, then the results are passed to plan fusion.

Strategies (2):
  - constraint_satisfaction: maximize user constraint/preference matching
  - spatio_temporal: maximize geographic efficiency and time rationality
"""

from __future__ import annotations

import asyncio
import time

from langchain_core.messages import HumanMessage, SystemMessage

from finn.llm import extract_json, llm_invoke, make_model
from finn.logger import logger
from finn.state import (
    ActivityNode,
    AgentPlan,
    AgentState,
    ExtractResult,
    Plan,
    POICandidate,
    POICategoryPool,
    TimeAlloc,
    WeatherContext,
)
from langgraph.types import Command


# ═══════════════════════════════════════════════════════════════════════
# Strategy definitions
# ═══════════════════════════════════════════════════════════════════════

_STRATEGIES = [
    {
        "name": "constraint_satisfaction",
        "label": "约束满足",
        "prompt_extra": """\
策略：约束满足优先
- 核心目标：最大化满足用户的硬约束和软偏好
- 必去地点、必吃菜系必须包含，一个都不能少
- 预算严格控制在用户上限以内，不超支
- 饮食限制（清真/素食）必须严格遵守
- 偏好菜系和偏好场所类型优先选择
- 场景匹配：家庭→安全舒适有儿童设施，朋友→社交氛围好
- 质量和成本可以适度妥协，但约束不能破""",
    },
    {
        "name": "spatio_temporal",
        "label": "时空最优",
        "prompt_extra": """\
策略：时空合理性优先
- 核心目标：最小化无效移动，最大化有效游玩时间
- 优先选择地理位置集中的 POI 集群（参考地理聚类信息）
- 转场时间必须合理：相邻 POI 距离越近越好
- 交通方式优先步行>驾车>公交
- 根据每个 POI 类型合理分配停留时间：
  · 景点/博物馆: 90-180min
  · 餐饮: 60-90min
  · 购物: 60-120min
  · 休闲娱乐: 90-150min
- 午餐11:00-13:00，晚餐17:00-19:00 窗口内安排
- 天气感知：雨天优先室内，高温安排室内休息
- 总时长控制在用户可用时间窗口内，不留过大空档也不过度紧凑""",
    },
]


# ═══════════════════════════════════════════════════════════════════════
# Common system prompt
# ═══════════════════════════════════════════════════════════════════════

_BASE_PLANNER_PROMPT = """\
你是 Finn 的出行规划 Agent。

你的任务：基于用户约束和 POI 候选池，动态生成一份完整的出行行程计划。

══════════════════════════════════════
规划规则
══════════════════════════════════════

1. **使用候选池**：只能从提供的 POI 候选池中选择场所，不可编造。
2. **动态时间分配**：不预设固定的活动链。根据候选池中 POI 的类型、密度和用户偏好，动态决定行程结构。参考以下停留时间：
   - 景点/博物馆: 90-180min
   - 餐饮: 60-90min（快餐30-45min）
   - 购物: 60-120min
   - 休闲娱乐/密室/剧本杀: 90-150min
3. **转场估算**：相邻场所间转场时间基于距离估算。
   - <1km: 步行10-15min
   - 1-5km: 驾车5-15min
   - 5-15km: 驾车15-30min
   - >15km: 驾车30-60min
4. **用餐时序**：午餐11:00-13:00，晚餐17:00-19:00。
5. **约束遵守**：
   - 饮食限制必须满足（清真/素食）
   - 儿童安全：避免酒吧/夜店
   - 预算上限不能突破
6. **天气感知**：根据天气调整室内外活动比例。
7. **动态调整**：行程结构应灵活响应 POI 数据：
   - 某区域餐饮丰富→在该区域安排连续用餐+休闲
   - 某区域景点集中→延长该区域停留时间
   - 优先参考地理聚类信息，在同一个 Cluster 内安排相邻节点以减少转场
   - 根据 POI 类别自然分布决定 play/eat 序列，而非套用固定模板

══════════════════════════════════════
输出格式
══════════════════════════════════════

输出一个 JSON 对象：

{
  "nodes": [
    {
      "slot": "<play|lunch|dinner|eat|follow_up|rest>",
      "poi_id": "<候选池中的POI ID>",
      "poi_name": "<POI名称>",
      "start_time": "<HH:MM>",
      "end_time": "<HH:MM>",
      "stay_min": <int 停留分钟>,
      "transit_from_prev_min": <int 转场分钟>,
      "transport_mode": "walk" | "drive" | "transit",
      "cost_estimate": <float 预估花费>,
      "notes": "<选这个场所的理由>"
    }
  ],
  "total_transit_min": <int>,
  "total_cost": <float>,
  "reasoning": "<中文，描述整体规划思路和行程结构决策依据>"
}

- nodes 按时间顺序排列，行程结构由 POI 分布自然决定
- 每个 node 必须从候选池中引用真实 POI
- transit_from_prev_min: 第一个 node 为 0
- slot 标签只用于标识场所类型，不约束序列结构
- 输出 ONLY JSON，不要 markdown，不要 ``` 代码块
"""


# ═══════════════════════════════════════════════════════════════════════
# Node: multi_agent_plan
# ═══════════════════════════════════════════════════════════════════════


async def multi_agent_plan(state: AgentState) -> dict:
    """Run 2 planning agents in parallel (constraint + spatio-temporal).

    Each agent gets the same data but a different optimization strategy.
    Both run via asyncio.gather. Results passed to plan_fusion.
    """
    extract = state.get("extract_result")
    category_pools = state.get("category_pools", {})
    weather = state.get("weather")

    if not extract or not category_pools:
        logger.warning("multi_agent_plan called without extract/pools")
        return Command(goto="present_to_user", update={
            "agent_plans": [],
            "plan": Plan(sub_tasks=[], notes="Unable to generate plan — missing context. Please try again."),
        })

    # Build shared context (same for all agents)
    shared_context = _build_planning_context(extract, category_pools, weather)

    t0 = time.monotonic()

    # Run all strategy agents in parallel
    tasks = [
        _run_planner_agent(strategy, shared_context)
        for strategy in _STRATEGIES
    ]
    results = await asyncio.gather(*tasks)

    agent_plans: list[AgentPlan] = []
    for strategy, result in zip(_STRATEGIES, results):
        if result is not None:
            agent_plans.append(result)
            logger.info("Agent '%s': %d nodes, transit=%d min, cost=¥%.0f",
                        strategy["name"], len(result.nodes),
                        result.total_transit_min, result.total_cost)
        else:
            logger.warning("Agent '%s' returned no plan", strategy["name"])

    elapsed = time.monotonic() - t0
    logger.info("Multi-agent planning: %d/%d agents succeeded in %.1fs",
                len(agent_plans), len(_STRATEGIES), elapsed)

    if len(agent_plans) < 2:
        logger.warning("Too few agent plans (%d) — falling back to present", len(agent_plans))
        # Build best-effort plan from the single agent, or empty fallback
        fallback_plan = (
            _agent_plan_to_subtask_plan(agent_plans[0], category_pools)
            if agent_plans else
            Plan(sub_tasks=[], notes="Unable to generate plan — all agents failed. Please try again.")
        )
        return Command(goto="present_to_user", update={
            "agent_plans": agent_plans,
            "plan": fallback_plan,
        })

    return Command(
        goto="plan_fusion",
        update={"agent_plans": agent_plans},
    )


# ═══════════════════════════════════════════════════════════════════════
# Single agent runner
# ═══════════════════════════════════════════════════════════════════════


async def _run_planner_agent(strategy: dict, context: str) -> AgentPlan | None:
    """Run a single planner agent with the given strategy."""
    system_prompt = _BASE_PLANNER_PROMPT + "\n" + strategy["prompt_extra"]
    user_prompt = context

    try:
        llm = make_model(temperature=0.5)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        text = await llm_invoke(llm, messages, f"planner_{strategy['name']}")

        data = extract_json(text)
        return _parse_agent_plan(data, strategy["name"])
    except Exception as exc:
        logger.warning("Planner agent '%s' failed: %s", strategy["name"], exc)
        return None


def _parse_agent_plan(data: dict, agent_name: str) -> AgentPlan:
    """Parse LLM output into an AgentPlan."""
    nodes: list[ActivityNode] = []
    for n in data.get("nodes", []):
        time_alloc = TimeAlloc(
            slot=str(n.get("slot", "play")),
            duration_min=int(n.get("stay_min", 90)),
            earliest_start=str(n.get("start_time", "")),
            earliest_end=str(n.get("end_time", "")),
        )
        # Minimal POICandidate for the node (full data will be looked up)
        poi = POICandidate(
            id=str(n.get("poi_id", "")),
            name=str(n.get("poi_name", "")),
        )
        nodes.append(ActivityNode(
            slot=str(n.get("slot", "play")),
            poi=poi,
            time=time_alloc,
            transit_from_prev_min=int(n.get("transit_from_prev_min", 0)),
            transport_mode=str(n.get("transport_mode", "drive")),
            stay_duration_min=int(n.get("stay_min", 90)),
            cost_estimate=float(n.get("cost_estimate", 0)),
        ))

    return AgentPlan(
        agent_name=agent_name,
        strategy_desc=data.get("reasoning", ""),
        nodes=nodes,
        total_transit_min=int(data.get("total_transit_min", 0)),
        total_cost=float(data.get("total_cost", 0)),
        coverage_score=0.8,  # Will be computed by fusion
        reasoning=str(data.get("reasoning", "")),
    )


# ═══════════════════════════════════════════════════════════════════════
# Shared context builder
# ═══════════════════════════════════════════════════════════════════════


def _build_planning_context(
    extract: ExtractResult,
    category_pools: dict[str, POICategoryPool],
    weather: WeatherContext | None,
) -> str:
    """Build the shared planning context for all agents."""
    lines: list[str] = []

    # User constraints
    lines.append("══════════════════════════════════════")
    lines.append("用户约束")
    lines.append("══════════════════════════════════════")

    i = extract.intent
    lines.append(f"活动: {i.activity_summary or '未指定'}")
    lines.append(f"城市: {i.city or '未指定'}")
    lines.append(f"日期: {i.plan_date or '未指定'}")
    lines.append(f"开始时间: {i.time_window_start or '09:00'}")
    lines.append(f"可用时长: {i.time_window_hours or 6}小时")
    lines.append(f"人数: {i.guest_count or 1}")
    lines.append(f"场景: {'家庭' if i.scene and i.scene.value == 'family' else '朋友'}")

    hc = extract.hard_constraints
    if hc.budget_max_cny:
        lines.append(f"预算上限: ¥{hc.budget_max_cny}")
    if hc.dietary_restrictions:
        lines.append(f"饮食限制: {', '.join(hc.dietary_restrictions)}")
    if hc.child_age is not None:
        lines.append(f"儿童年龄: {hc.child_age}岁")
    if hc.time_deadline:
        lines.append(f"结束截止: {hc.time_deadline}")

    sc = extract.soft_constraints
    if sc.preferred_cuisines:
        lines.append(f"偏好菜系: {', '.join(sc.preferred_cuisines)}")
    if sc.preferred_poi_types:
        lines.append(f"偏好类型: {', '.join(sc.preferred_poi_types)}")
    if sc.avoid_poi_types:
        lines.append(f"避开类型: {', '.join(sc.avoid_poi_types)}")
    if sc.budget_preference:
        lines.append(f"预算偏好: {sc.budget_preference.value}")
    if sc.travel_pace:
        lines.append(f"节奏: {sc.travel_pace.value}")
    if sc.preferred_transport:
        lines.append(f"偏好交通: {sc.preferred_transport.value}")

    req = extract.requirements
    if req.must_visit_pois:
        lines.append(f"必去地点: {', '.join(req.must_visit_pois)}")
    if req.must_have_cuisine:
        lines.append(f"必吃菜系: {', '.join(req.must_have_cuisine)}")
    if req.special_requests:
        lines.append(f"特殊需求: {', '.join(req.special_requests)}")

    # Weather
    if weather and weather.condition:
        lines.append("")
        lines.append("══════════════════════════════════════")
        lines.append("天气")
        lines.append("══════════════════════════════════════")
        lines.append(f"{weather.date} {weather.condition} "
                     f"{f'{weather.temp_low:.0f}~{weather.temp_high:.0f}°C' if weather.temp_low and weather.temp_high else ''}"
                     f"{' | 建议室内' if weather.indoor_recommended else ''}")

    # POI Category Pools
    lines.append("")
    lines.append("══════════════════════════════════════")
    lines.append("POI 候选池（按类别，只能从这里选择）")
    lines.append("══════════════════════════════════════")

    for cat, pool in category_pools.items():
        lines.append(f"\n--- {cat} ({len(pool.candidates)}个) ---")
        for c in pool.candidates[:20]:  # Top 20 per category for planning visibility
            rating = f" ★{c.rating}" if c.rating else ""
            price = f" ¥{int(c.price_per_person)}/人" if c.price_per_person else ""
            loc = f" [{c.location}]" if c.location else ""
            address = f" | {c.address}" if c.address else ""
            lines.append(f"  [{c.id}] {c.name}{address}{rating}{price}{loc}")

    # ── Geographic clustering ──
    clusters = _compute_clusters(category_pools)
    if clusters:
        lines.append("")
        lines.append("══════════════════════════════════════")
        lines.append("POI 地理聚类（3km以内）— 方便就近安排")
        lines.append("══════════════════════════════════════")
        for cat, cluster_list in clusters.items():
            lines.append(f"\n{cat}:")
            for label, pois in cluster_list:
                poi_names = ", ".join(f"[{p.id}] {p.name}" for p in pois[:6])
                lines.append(f"  {label} ({len(pois)}个): {poi_names}")

    return "\n".join(lines)


def _compute_clusters(
    category_pools: dict[str, POICategoryPool],
    max_dist_m: int = 3000,
) -> dict[str, list[tuple[str, list[POICandidate]]]]:
    """Compute geographic clusters of POIs within max_dist_m.

    Uses greedy clustering: for each candidate, find an existing cluster
    it's close to; if none, start a new cluster.

    Returns {category: [(label, [POIs]), ...]} sorted by cluster size desc.
    """
    import math

    def haversine_m(loc1: str, loc2: str) -> float | None:
        """Distance in meters between two 'lng,lat' strings."""
        try:
            lng1, lat1 = map(float, loc1.split(","))
            lng2, lat2 = map(float, loc2.split(","))
        except (ValueError, AttributeError):
            return None
        R = 6371000
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlng / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def common_address_prefix(pois: list[POICandidate]) -> str:
        """Extract district/common area from addresses."""
        from collections import Counter
        districts = []
        for p in pois:
            addr = p.address or ""
            # Try to extract district name (e.g., "渝中区" from full address)
            for suffix in ["区", "县"]:
                idx = addr.find(suffix)
                if idx > 0:
                    # Find the start of the district name
                    start = max(0, idx - 3)
                    districts.append(addr[start:idx + 1])
                    break
        if districts:
            top = Counter(districts).most_common(1)[0][0]
            return f"~{top}"
        return ""

    result: dict[str, list[tuple[str, list[POICandidate]]]] = {}
    for cat, pool in category_pools.items():
        candidates = [c for c in pool.candidates if c.location]
        if len(candidates) < 2:
            continue

        clusters: list[list[POICandidate]] = []
        for c in candidates:
            placed = False
            for cl in clusters:
                for member in cl:
                    d = haversine_m(c.location, member.location)
                    if d is not None and d <= max_dist_m:
                        cl.append(c)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                clusters.append([c])

        # Label clusters, sorted by size
        labeled: list[tuple[str, list[POICandidate]]] = []
        for i, cl in enumerate(sorted(clusters, key=len, reverse=True)):
            if len(cl) >= 2:
                prefix = common_address_prefix(cl)
                labeled.append((f"Cluster {chr(65+i)}{prefix}", cl))

        if labeled:
            result[cat] = labeled

    return result
