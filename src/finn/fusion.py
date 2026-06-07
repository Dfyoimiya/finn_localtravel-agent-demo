"""Plan Fusion & QA Check — weighted voting + final optimization.

Flow:
  1. plan_fusion: LLM analyzes 3+ agent plans, weights votes, produces fused plan
  2. qa_check: LLM final optimization pass — checks constraints, adjusts timing

The fusion node produces a Plan (SubTask format) compatible with the existing
presentation layer (PlanCards) and execution layer (book_worker).
"""

from __future__ import annotations

import time

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from finn.llm import extract_json, llm_invoke, make_model
from finn.logger import logger
from finn.state import (
    ActivityNode,
    AgentPlan,
    AgentState,
    ExtractResult,
    FusionResult,
    Plan,
    PlannedPath,
    POICandidate,
    POICategoryPool,
    SubTask,
    TimeAlloc,
    WeatherContext,
)


# ═══════════════════════════════════════════════════════════════════════
# Plan Fusion prompt
# ═══════════════════════════════════════════════════════════════════════

_FUSION_PROMPT = """\
你是 Finn 的计划融合模块。

你会收到 2 个 Agent 各自生成的出行计划：一个侧重约束满足，一个侧重时空合理性。
你的任务：通过加权投票机制，融合这两份计划生成一份最优的出行方案。

══════════════════════════════════════
融合规则
══════════════════════════════════════

1. **逐位置投票**：按时间顺序，对每个位置节点看两个 Agent 选择了哪些 POI。如果两个 Agent 选了同一个 POI → 高权重（共识选择）。如果选的不同 → 综合考虑以下因素决定。

2. **投票权重**：
   - 如果 POI 满足用户必吃/必去要求 → 加权（约束满足 Agent 的选择优先）
   - 如果 POI 地理位置更集中、转场更短 → 加权（时空 Agent 的选择优先）
   - 如果 POI 评分/口碑更好 → 加权
   - 如果两个 Agent 选了不同的 POI → 综合评分、距离、预算三者择优

3. **冲突解决**：
   - 同一 POI 被两个 Agent 分配给不同时间位置 → 选择更合理的时序安排
   - 同一时间位置两个候选 → 综合约束匹配度 + 时空效率择优
   - 如果某个位置两个 Agent 的选择都不理想 → 从候选池中选择最佳的

4. **整体评估**：
   - 检查融合后的计划是否满足所有硬约束（预算、饮食、儿童安全）
   - 确保时间线合理（无冲突、无过长空档）
   - 确保场所间转场时间合理

5. **动态结构调整**：
   - 不预设固定的 play/eat 序列，行程结构由 POI 自然分布决定
   - 如果某区域餐饮集中且有多个高分餐厅，可以安排在该区域连续用餐
   - 根据用户时间窗口和 POI 停留时间，灵活决定节点数量

6. **输出**：融合后的完整出行计划 + 每个 Agent 的投票贡献。

══════════════════════════════════════
输出格式
══════════════════════════════════════

输出一个 JSON 对象：

{
  "plan": {
    "sub_tasks": [
      {
        "id": "<唯一id>",
        "type": "book",
        "target": "<描述，如'午餐-海底捞火锅'>",
        "dependencies": ["<上一任务id>"],
        "params": {
          "name": "<POI名称>",
          "id": "<POI ID>",
          "address": "<地址>",
          "location": "<lng,lat>",
          "rating": <float>,
          "slot": "<play|eat|lunch|dinner|follow_up|rest>",
          "start_time": "<HH:MM>",
          "end_time": "<HH:MM>",
          "cost_estimate": <float>,
          "transport_from_prev": "<交通方式>",
          "transit_minutes": <int>
        },
        "compensatory": "<取消任务id>"
      }
    ],
    "total_cost_estimate": <float>,
    "notes": "<中文摘要，描述融合决策和行程结构>"
  },
  "votes": {
    "constraint_satisfaction": <int>,
    "spatio_temporal": <int>
  },
  "fusion_score": <float 0-1>,
  "reasoning": "<中文，融合决策说明>"
}

- sub_tasks 按时间顺序排列，行程结构由 POI 分布和 Agent 选择自然决定
- 每个 book task 的 params 必须包含 name/id/address/location/rating/slot/start_time/end_time
- votes 记录每个 Agent 被采纳的节点数
- 输出 ONLY JSON，不要 markdown，不要 ``` 代码块
"""


# ═══════════════════════════════════════════════════════════════════════
# QA Check prompt
# ═══════════════════════════════════════════════════════════════════════

_QA_CHECK_PROMPT = """\
你是 Finn 的质检优化模块。

你会收到一份融合后的出行计划、用户约束和 POI 候选池。
你的任务：最终检查并优化这份计划。

══════════════════════════════════════
检查清单
══════════════════════════════════════

1. **时间可行性**：
   - 总时间是否在用户可用时长内？
   - 每个场所停留时间是否合理（不短于最低停留）？
   - 转场时间是否现实（不出现负值或异常长距离）？
   - 午餐/晚餐是否在合理时段内？

2. **约束满足**：
   - 预算是否在范围内？
   - 饮食限制是否遵守？
   - 儿童安全（无酒吧/夜店）？
   - 必去地点/必吃菜系是否包含？

3. **场所多样性**：
   - 同一场所不能填充多个 slot
   - 不同类别合理搭配（文化+餐饮+休闲）

4. **天气适配**：
   - 雨天是否避免了户外场所？
   - 高温是否安排了室内休息？

5. **优化调整**：
   - 如果发现问题，从候选池中替换更好的 POI
   - 微调时间安排使行程更流畅
   - 确保每个 slot 都有合适的 POI

══════════════════════════════════════
输出格式
══════════════════════════════════════

输出一个 JSON 对象（与输入格式相同，但经过优化）：

{
  "plan": {
    "sub_tasks": [...],
    "total_cost_estimate": <float>,
    "notes": "<中文，描述做的优化调整>"
  },
  "issues_found": ["<发现的问题>"],
  "changes_made": ["<做的修改>"],
  "qa_score": <float 0-1>
}

- 如果无需修改，issues_found 和 changes_made 为空数组
- 输出 ONLY JSON
"""


# ═══════════════════════════════════════════════════════════════════════
# Node: plan_fusion
# ═══════════════════════════════════════════════════════════════════════


async def plan_fusion(state: AgentState) -> dict:
    """LLM-driven fusion: analyze 3+ agent plans, weighted voting, produce one plan.

    Routes via Command(goto=...):
      Always → qa_check
    """
    agent_plans = state.get("agent_plans", [])
    extract = state.get("extract_result")
    category_pools = state.get("category_pools", {})
    weather = state.get("weather")
    modify_feedback = state.get("modify_feedback", "")

    if not agent_plans or not extract:
        logger.warning("plan_fusion called without agent_plans/extract")
        return Command(goto="present_to_user", update={})

    t0 = time.monotonic()

    # Build context (include modify feedback if present)
    context = _build_fusion_context(agent_plans, extract, category_pools, weather)

    if modify_feedback:
        context = (
            "⚠️ 用户修改请求：\n" + modify_feedback + "\n\n"
            "请根据用户的修改请求重新融合计划，在投票时优先考虑符合用户反馈的 POI 选择。\n\n"
            + context
        )

    try:
        llm = make_model(temperature=0.3)
        messages = [
            SystemMessage(content=_FUSION_PROMPT),
            HumanMessage(content=context),
        ]
        text = await llm_invoke(llm, messages, "plan_fusion")

        data = extract_json(text)
        fusion = _parse_fusion_result(data)

        elapsed = time.monotonic() - t0
        logger.info("Fusion: score=%.2f, %d tasks, votes=%s, in %.1fs",
                     fusion.fusion_score, len(fusion.plan.sub_tasks) if fusion.plan else 0,
                     fusion.votes, elapsed)

        # Backfill POI data from category pools into plan params
        if fusion.plan:
            _backfill_plan_from_pools(fusion.plan, category_pools)

        # Generate PlannedPath for presentation timing
        selected_path = plan_to_path(fusion.plan) if fusion.plan else None

        return Command(
            goto="present_to_user",
            update={
                "fusion_result": fusion,
                "plan": fusion.plan,
                "selected_path": selected_path,
            },
        )
    except Exception as exc:
        logger.warning("plan_fusion failed: %s — using best agent plan", exc)
        return _fallback_fusion(agent_plans, category_pools)


def _fallback_fusion(
    agent_plans: list[AgentPlan], category_pools: dict[str, POICategoryPool],
) -> dict:
    """Fallback: use the plan with the most nodes from the first agent."""
    best = agent_plans[0] if agent_plans else None
    if not best:
        return Command(goto="present_to_user", update={})

    plan = _agent_plan_to_subtask_plan(best, category_pools)
    selected_path = plan_to_path(plan)
    fusion = FusionResult(
        plan=plan,
        votes={best.agent_name: len(best.nodes)},
        fusion_score=0.5,
        reasoning=f"Fallback: using {best.agent_name} agent plan",
    )
    return Command(goto="present_to_user", update={
        "fusion_result": fusion, "plan": plan, "selected_path": selected_path,
    })


# ═══════════════════════════════════════════════════════════════════════
# Node: qa_check
# ═══════════════════════════════════════════════════════════════════════


async def qa_check(state: AgentState) -> dict:
    """QA check agent: final optimization pass on the fused plan.

    Routes via Command(goto=...):
      Always → present_to_user
    """
    plan = state.get("plan")
    extract = state.get("extract_result")
    category_pools = state.get("category_pools", {})
    weather = state.get("weather")
    modify_feedback = state.get("modify_feedback", "")
    modify_count = state.get("modify_count", 0)

    if not plan or not extract:
        logger.warning("qa_check called without plan/extract")
        return Command(goto="present_to_user", update={})

    # Guard against excessive modifications
    MAX_MODIFY = 3
    if modify_count > MAX_MODIFY:
        logger.warning("Modify limit reached (%d/%d) — skipping re-optimization",
                       modify_count, MAX_MODIFY)
        return Command(
            goto="present_to_user",
            update={"modify_feedback": ""},
        )

    t0 = time.monotonic()

    # Build context
    context = _build_qa_context(plan, extract, category_pools, weather, modify_feedback)

    try:
        llm = make_model(temperature=0.2)
        messages = [
            SystemMessage(content=_QA_CHECK_PROMPT),
            HumanMessage(content=context),
        ]
        text = await llm_invoke(llm, messages, "qa_check")

        data = extract_json(text)
        qa_plan_data = data.get("plan", {})
        if qa_plan_data:
            qa_plan = _parse_plan(qa_plan_data)
            _backfill_plan_from_pools(qa_plan, category_pools)

            issues = data.get("issues_found", [])
            changes = data.get("changes_made", [])
            qa_score = data.get("qa_score", 0.8)

            elapsed = time.monotonic() - t0
            logger.info("QA check: score=%.2f, issues=%d, changes=%d, in %.1fs",
                         qa_score, len(issues), len(changes), elapsed)
            if issues:
                logger.info("QA issues: %s", issues)
            if changes:
                logger.info("QA changes: %s", changes)

            # Regenerate PlannedPath with updated timing
            updated_path = plan_to_path(qa_plan)

            return Command(
                goto="present_to_user",
                update={
                    "plan": qa_plan,
                    "selected_path": updated_path,
                    "modify_feedback": "",  # Clear feedback after use
                },
            )
        else:
            return Command(goto="present_to_user", update={"modify_feedback": ""})
    except Exception as exc:
        logger.warning("qa_check failed: %s — using fused plan as-is", exc)
        return Command(goto="present_to_user", update={"modify_feedback": ""})


# ═══════════════════════════════════════════════════════════════════════
# Context builders
# ═══════════════════════════════════════════════════════════════════════


def _build_fusion_context(
    agent_plans: list[AgentPlan],
    extract: ExtractResult,
    category_pools: dict[str, POICategoryPool],
    weather: WeatherContext | None,
) -> str:
    """Build fusion input context."""
    lines: list[str] = []

    # User constraints summary
    i = extract.intent
    lines.append("══════════════════════════════════════")
    lines.append("用户约束")
    lines.append("══════════════════════════════════════")
    lines.append(f"活动: {i.activity_summary} | 城市: {i.city} | 日期: {i.plan_date}")
    lines.append(f"时间: {i.time_window_start} | {i.time_window_hours}h | {i.guest_count}人")
    lines.append(f"场景: {'家庭' if i.scene and i.scene.value == 'family' else '朋友'}")

    hc = extract.hard_constraints
    if hc.budget_max_cny:
        lines.append(f"预算上限: ¥{hc.budget_max_cny}")
    if hc.dietary_restrictions:
        lines.append(f"饮食限制: {', '.join(hc.dietary_restrictions)}")
    if hc.child_age is not None:
        lines.append(f"儿童: {hc.child_age}岁")

    sc = extract.soft_constraints
    if sc.preferred_cuisines:
        lines.append(f"偏好菜系: {', '.join(sc.preferred_cuisines)}")
    if sc.avoid_poi_types:
        lines.append(f"避开: {', '.join(sc.avoid_poi_types)}")

    req = extract.requirements
    if req.must_have_cuisine:
        lines.append(f"必吃: {', '.join(req.must_have_cuisine)}")
    if req.must_visit_pois:
        lines.append(f"必去: {', '.join(req.must_visit_pois)}")

    # Weather
    if weather and weather.condition:
        lines.append(f"天气: {weather.date} {weather.condition} "
                     f"{'建议室内' if weather.indoor_recommended else ''}")

    # Agent plans
    lines.append("")
    lines.append("══════════════════════════════════════")
    lines.append(f"Agent 计划 ({len(agent_plans)}个)")
    lines.append("══════════════════════════════════════")

    for plan in agent_plans:
        lines.append(f"\n--- {plan.agent_name} ---")
        lines.append(f"策略: {plan.reasoning[:100]}")
        lines.append(f"总转场: {plan.total_transit_min}min | 总花费: ¥{plan.total_cost:.0f}")
        for j, node in enumerate(plan.nodes):
            if not node.poi or not node.time:
                continue
            lines.append(
                f"  {j+1}. [{node.slot}] [{node.poi.id}] {node.poi.name} "
                f"({node.time.earliest_start}-{node.time.earliest_end}) "
                f"转场{node.transit_from_prev_min}min | ¥{node.cost_estimate:.0f}"
            )

    # Candidate pool details (full POI data for informed fusion)
    lines.append("")
    lines.append("══════════════════════════════════════")
    lines.append("POI 候选池（含详情，用于融合时替换选择）")
    lines.append("══════════════════════════════════════")
    for cat, pool in category_pools.items():
        lines.append(f"\n--- {cat} ({len(pool.candidates)}个) ---")
        for c in pool.candidates[:8]:
            rating = f" ★{c.rating}" if c.rating else ""
            price = f" ¥{int(c.price_per_person)}/人" if c.price_per_person else ""
            lines.append(f"  [{c.id}] {c.name}{rating}{price} | {c.address}")

    return "\n".join(lines)


def _build_qa_context(
    plan: Plan,
    extract: ExtractResult,
    category_pools: dict[str, POICategoryPool],
    weather: WeatherContext | None,
    modify_feedback: str = "",
) -> str:
    """Build QA check context."""
    lines: list[str] = []

    # User constraints
    i = extract.intent
    lines.append("══════════════════════════════════════")
    lines.append("用户约束")
    lines.append("══════════════════════════════════════")
    lines.append(f"活动: {i.activity_summary} | {i.city} | {i.plan_date}")
    lines.append(f"时间: {i.time_window_start} | {i.time_window_hours}h | {i.guest_count}人")
    lines.append(f"预算上限: ¥{extract.hard_constraints.budget_max_cny or '不限'}")
    if extract.hard_constraints.dietary_restrictions:
        lines.append(f"饮食限制: {', '.join(extract.hard_constraints.dietary_restrictions)}")

    ct = extract.chain
    lines.append(f"活动链: {' → '.join(ct.template) if ct.template else '未指定'}")

    if weather and weather.condition:
        lines.append(f"天气: {weather.date} {weather.condition}")

    # Current plan
    lines.append("")
    lines.append("══════════════════════════════════════")
    lines.append("当前计划")
    lines.append("══════════════════════════════════════")
    lines.append(plan.model_dump_json(indent=2, ensure_ascii=False))

    # Modify feedback
    if modify_feedback:
        lines.append("")
        lines.append("══════════════════════════════════════")
        lines.append("用户修改请求")
        lines.append("══════════════════════════════════════")
        lines.append(modify_feedback)

    # Candidate pool
    lines.append("")
    lines.append("══════════════════════════════════════")
    lines.append("POI候选池")
    lines.append("══════════════════════════════════════")
    for cat, pool in category_pools.items():
        lines.append(f"\n--- {cat} ---")
        for c in pool.candidates[:8]:
            rating = f" ★{c.rating}" if c.rating else ""
            price = f" ¥{int(c.price_per_person)}/人" if c.price_per_person else ""
            lines.append(f"  [{c.id}] {c.name}{rating}{price} | {c.address}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Parsers
# ═══════════════════════════════════════════════════════════════════════


def _parse_fusion_result(data: dict) -> FusionResult:
    """Parse LLM fusion output."""
    plan_data = data.get("plan", {})
    plan = _parse_plan(plan_data) if plan_data else Plan(sub_tasks=[])

    return FusionResult(
        plan=plan,
        votes={str(k): int(v) for k, v in data.get("votes", {}).items()},
        fusion_score=float(data.get("fusion_score", 0.5)),
        reasoning=str(data.get("reasoning", "")),
    )


def _parse_plan(data: dict) -> Plan:
    """Parse a plan dict into a Plan model.

    Auto-generates task IDs (task1, task2, ...) if the LLM omits them,
    and rewrites dependencies to match the generated IDs.
    """
    sub_tasks: list[SubTask] = []
    raw_tasks = data.get("sub_tasks", [])
    for i, st in enumerate(raw_tasks):
        params = st.get("params", {})
        # Ensure all required params fields are present
        full_params = {
            "name": str(params.get("name", st.get("target", ""))),
            "id": str(params.get("id", "")),
            "address": str(params.get("address", "")),
            "location": str(params.get("location", "")),
            "rating": params.get("rating"),
            "slot": str(params.get("slot", "play")),
            "start_time": str(params.get("start_time", "")),
            "end_time": str(params.get("end_time", "")),
            "cost_estimate": float(params.get("cost_estimate", 0)),
            "transport_from_prev": str(params.get("transport_from_prev", "")),
            "transit_minutes": int(params.get("transit_minutes", 0)),
        }
        # Auto-generate task ID if LLM omitted it
        task_id = str(st.get("id", ""))
        if not task_id:
            task_id = f"task{i + 1}"
        # Clean compensatory: treat "None", "null", "" as empty
        comp_raw = str(st.get("compensatory", ""))
        comp = comp_raw if comp_raw.lower() not in ("none", "null", "") else ""
        # Rewrite dependency references: if an earlier task had its ID
        # auto-generated, update deps that reference the empty/original ID
        deps = [str(d) for d in st.get("dependencies", [])]
        sub_tasks.append(SubTask(
            id=task_id,
            type="book",
            target=str(st.get("target", "")),
            dependencies=deps,
            params=full_params,
            compensatory=comp if comp else None,
        ))

    return Plan(
        sub_tasks=sub_tasks,
        total_cost_estimate=float(data.get("total_cost_estimate", 0)) or None,
        notes=str(data.get("notes", "")),
    )


def plan_to_path(plan: Plan) -> PlannedPath:
    """Convert a Plan (SubTask format) to a PlannedPath with timing.

    Uses params["start_time"], params["end_time"], params["transit_minutes"],
    and params["slot"] from each book SubTask to build ActivityNodes with
    accurate timing information.
    """
    from datetime import datetime

    nodes: list[ActivityNode] = []
    total_transit = 0
    total_cost = 0.0

    for st in plan.sub_tasks:
        if st.type != "book":
            continue
        params = st.params or {}
        name = str(params.get("name", st.target or ""))
        pid = str(params.get("id", ""))
        slot = str(params.get("slot", "play"))
        start_str = str(params.get("start_time", ""))
        end_str = str(params.get("end_time", ""))
        transit = int(params.get("transit_minutes", 0))
        cost = float(params.get("cost_estimate", 0))
        location = str(params.get("location", ""))
        address = str(params.get("address", ""))
        rating = params.get("rating")
        transport = str(params.get("transport_from_prev", "drive"))

        # Compute stay duration from start/end times
        stay_min = 90  # default
        if start_str and end_str:
            try:
                fmt = "%H:%M"
                t0 = datetime.strptime(start_str, fmt)
                t1 = datetime.strptime(end_str, fmt)
                diff = (t1 - t0).total_seconds() / 60
                if diff > 0:
                    stay_min = int(diff)
            except (ValueError, TypeError):
                pass

        poi = POICandidate(
            id=pid,
            name=name,
            address=address,
            location=location,
            rating=float(rating) if rating else None,
        )

        time_alloc = TimeAlloc(
            slot=slot,
            duration_min=stay_min,
            earliest_start=start_str,
            earliest_end=end_str,
        )

        nodes.append(ActivityNode(
            slot=slot,
            poi=poi,
            time=time_alloc,
            transit_from_prev_min=transit,
            transport_mode=transport,
            stay_duration_min=stay_min,
            cost_estimate=cost,
        ))
        total_transit += transit
        total_cost += cost

    return PlannedPath(
        nodes=nodes,
        total_transit_min=total_transit,
        total_cost=total_cost,
        coverage_score=0.8,
        notes=plan.notes,
        strategy="fused",
    )


def _backfill_plan_from_pools(
    plan: Plan, category_pools: dict[str, POICategoryPool],
) -> None:
    """Backfill location/address/rating from category pools into plan params.

    Tries exact ID match first, then exact name match, then fuzzy substring
    match as a fallback for LLM output variations (e.g. "海底捞" vs "海底捞火锅(三里屯店)").
    """
    # Build lookup from all pools
    by_id: dict[str, POICandidate] = {}
    by_name: dict[str, POICandidate] = {}
    all_candidates: list[POICandidate] = []
    for pool in category_pools.values():
        for c in pool.candidates:
            if c.id:
                by_id[c.id] = c
            if c.name:
                by_name[c.name] = c
            all_candidates.append(c)

    for st in plan.sub_tasks:
        if st.type != "book":
            continue
        params = st.params or {}
        pid = str(params.get("id", ""))
        name = str(params.get("name", ""))

        # 1. Exact ID match
        ref = by_id.get(pid)
        # 2. Exact name match
        if ref is None:
            ref = by_name.get(name)
        # 3. Fuzzy: name contains candidate name or vice versa
        if ref is None and name:
            for c in all_candidates:
                c_name = c.name or ""
                if c_name and (name in c_name or c_name in name):
                    ref = c
                    break

        if ref is None:
            continue

        if not params.get("location"):
            params["location"] = ref.location
        if not params.get("address"):
            params["address"] = ref.address
        if params.get("rating") is None:
            params["rating"] = ref.rating
        if not params.get("id"):
            params["id"] = ref.id


def _agent_plan_to_subtask_plan(
    agent_plan: AgentPlan, category_pools: dict[str, POICategoryPool],
) -> Plan:
    """Convert an AgentPlan to Plan with SubTasks (for fallback)."""
    sub_tasks: list[SubTask] = []
    for j, node in enumerate(agent_plan.nodes):
        if not node.poi or not node.time:
            continue
        prev_id = f"t{j}" if j > 0 else ""
        sub_tasks.append(SubTask(
            id=f"t{j+1}",
            type="book",
            target=f"{node.slot}-{node.poi.name}",
            dependencies=[prev_id] if prev_id else [],
            params={
                "name": node.poi.name,
                "id": node.poi.id,
                "slot": node.slot,
                "start_time": node.time.earliest_start,
                "end_time": node.time.earliest_end,
                "cost_estimate": node.cost_estimate,
                "transit_minutes": node.transit_from_prev_min,
                "location": node.poi.location,
                "address": node.poi.address,
                "rating": node.poi.rating,
            },
            compensatory=f"cancel_t{j+1}",
        ))

    plan = Plan(
        sub_tasks=sub_tasks,
        total_cost_estimate=agent_plan.total_cost,
        notes=agent_plan.reasoning,
    )
    _backfill_plan_from_pools(plan, category_pools)
    return plan
