"""LangGraph node implementations using LangChain.

Each node is a pure async function: (state, config) -> partial state update.
LLM nodes use LangChain's ChatOpenAI with manual ReAct loops for tool calling.

Text-based JSON extraction for structured output (DeepSeek V4 does not
support tool_choice in thinking mode).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command, interrupt, Send

from finn.config import config
from finn.llm import llm_invoke, make_model, react_loop
from finn.logger import logger
from finn.mcp import mcp_session
from finn.state import AgentState, BookingResult, Intent, Plan, SubTask, Verification

# ═══════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════


def _extract_json(text: str) -> dict:
    """Pull the first JSON object from LLM response text."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in response: {text[:200]}")
    return json.loads(match.group())


def _strip_nulls(data: dict, *keep: str) -> dict:
    """Remove keys with None values, except those listed in *keep."""
    return {k: v for k, v in data.items() if v is not None or k in keep}


def _normalize_plan(data: dict) -> dict:
    """Fix common LLM output issues in Plan JSON before validation."""
    valid_types = {"search", "compare", "book"}
    for st in data.get("sub_tasks", []):
        if st.get("type") not in valid_types:
            st["type"] = "search"
        if "dependencies" not in st:
            st["dependencies"] = []
        if "params" not in st:
            st["params"] = {}
    return data


# ═══════════════════════════════════════════════════════════════════════
# Node 1: clarify_intent
# ═══════════════════════════════════════════════════════════════════════

CLARIFY_SYSTEM_PROMPT = """\
你是 Finn 的意图提取模块——一个本地短途出行规划 Agent。

你的任务：分析对话，输出一个 JSON 对象，同时包含提取的意图数据和路由决策。

注意：对话中可能包含 [系统上下文] 标记，其中有当前时间、基于 IP 的用户位置，
以及 [用户画像]（偏好口味、常去区域、常用同行人、预算范围等）。
如果用户说"附近"、"今天"等模糊表述，使用系统上下文中的信息来填充具体值。
如果用户画像中有匹配的偏好或同行人信息，优先使用并填充到意图中。

只输出合法的 JSON 对象（不要 markdown，不要 ``` 代码块，不要额外文字）：

{
  "route": "plan" | "clarify" | "reject",
  "activity": "<用户想做什么，例如：喝咖啡然后看电影。不是出行需求则为空字符串>",
  "date": "<例如：周六、2026-06-07、今天。未知则为 null>",
  "start_location": "<出发地，例如：家、国贸。未知则为 null>",
  "scenario": "family | friends | couple | solo | unknown",
  "start_time": "<例如：14:00、下午2点。未知则为 null>",
  "duration_hours": <float 或 null>,
  "area": "<例如：朝阳区、三里屯。未知则为 null>",
  "radius_km": <float 或 null>,
  "party_size": <int 或 null>,
  "party_members": [
    {
      "role": "<self | spouse | child | friend | colleague>",
      "age": <int 或 null>,
      "constraints": ["<例如：减肥中、不吃辣、海鲜过敏>"],
      "preferences": ["<例如：喜欢户外、想喝奶茶>"]
    }
  ],
  "budget_total": <float 或 null>,
  "budget_per_person": <float 或 null>,
  "hard_constraints": ["<例如：需儿童座椅、18:00前结束、清真饮食、包间>"],
  "preferences": ["<例如：安静、户外、高评分、川菜、适合拍照>"],
  "follow_up_question": "<自然的中文追问，或 null>"
}

══════════════════════════════════════
路由规则 — 必须将 "route" 设为以下之一：
══════════════════════════════════════

"plan"
  activity、date、start_location 均非 null。
  可以进入任务分解与规划阶段。
  此时 follow_up_question 必须为 null。

"clarify"
  activity 非 null（用户有出行意图），但 date 或 start_location
  仍缺失。将 follow_up_question 设为一句简短的自然中文追问，
  每次只问一个缺失字段。

"reject"
  activity 为空——这不是出行规划请求。
  用户在闲聊、问事实性问题，或超出能力范围。

══════════════════════════════════════
提取规则
══════════════════════════════════════
1. 多轮合并：如果 prompt 中提供了 PREVIOUS INTENT，将最新消息合并进去——
   除非用户明确修改，否则保留已提取的字段。
2. 场景判断：老婆/孩子 → family；朋友/几个人/姐妹/兄弟 → friends；
   女朋友/男朋友/约会 → couple；我一个人 → solo。
3. 同行人：scenario 为 family/friends 时，提取 party_size 和 party_members。
   12 岁以下儿童 → 在对应成员的 constraints 中加入"需儿童座椅"等。
   饮食/健康备注 → 放入对应成员的 constraints。
4. 约束分类："必须"/"不能"/"一定要" → hard_constraints。
   "最好"/"喜欢"/"想" → preferences 或 member.preferences。
   同一项不要同时出现在两个列表中。
5. 诚实原则：只填用户实际提供的信息。null / 空列表好过瞎猜。
6. 追问限制：每次只问一个缺失字段。自然中文。如果 route 是 "plan" 或 "reject"，
   follow_up_question 必须为 null。"""


async def clarify_intent(state: AgentState) -> dict:
    """Node 1: Extract intent + route via Command."""
    previous_intent = state.get("intent")
    existing_json = ""
    if previous_intent and previous_intent.activity:
        existing_json = (
            "\n\nPrevious intent (carry forward unless user changes):\n"
            + previous_intent.model_dump_json(indent=2, exclude_none=True)
        )

    all_msgs = state["messages"]
    recent = all_msgs[-6:]
    history = "\n".join(
        f"{getattr(m, 'role', '')}: {getattr(m, 'content', str(m))}"
        for m in recent
    )
    user_prompt = f"Conversation:\n{history}{existing_json}"

    parse_error = False
    try:
        llm = make_model(temperature=0.3)
        messages = [
            SystemMessage(content=CLARIFY_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        text = await llm_invoke(llm, messages, "clarify_intent", stream=True)
        data = _extract_json(text)
        route = data.pop("route", "reject")
        clean = _strip_nulls(data, "activity")
        intent = Intent.model_validate(clean)
    except Exception:
        logger.warning("clarify_intent parse failure — falling back to clarify")
        parse_error = True
        route = "clarify"
        intent = Intent(activity="")

    # Fill missing_critical
    if not intent.activity or not intent.activity.strip():
        intent.missing_critical = ["activity"]
    else:
        intent.missing_critical = [
            f for f in ("activity", "date", "start_location")
            if getattr(intent, f) is None
        ]

    # Clarify iteration tracking
    clarify_iterations = state.get("clarify_iterations", 0)
    if route == "clarify":
        clarify_iterations += 1

    _log_intent(intent, route, clarify_iterations)

    # Fallback for parse errors
    if parse_error:
        return Command(
            goto=END,
            update={
                "intent": intent,
                "plan_iterations": 0,
                "clarify_iterations": clarify_iterations + 1,
                "messages": [{
                    "role": "assistant",
                    "content": "抱歉我没太理解，能换个方式说说你的需求吗？",
                }],
            },
        )

    # Enforce clarify limit
    if route == "clarify" and clarify_iterations >= MAX_CLARIFY_ITERATIONS:
        logger.info("Clarify limit reached (%d/%d) — forcing route",
                    clarify_iterations, MAX_CLARIFY_ITERATIONS)
        route = "plan" if intent.activity else "reject"

    # Route via Command
    if route == "plan":
        return Command(
            goto="decompose_and_plan",
            update={"intent": intent, "plan_iterations": 0, "clarify_iterations": 0},
        )
    elif route == "clarify":
        return Command(
            goto=END,
            update={"intent": intent, "plan_iterations": 0,
                    "clarify_iterations": clarify_iterations},
        )
    else:
        return Command(
            goto="reject",
            update={"intent": intent, "plan_iterations": 0, "clarify_iterations": 0},
        )


def _log_intent(intent: Intent, route: str, clarify_n: int = 0) -> None:
    info_parts = [
        f"activity={intent.activity!r}",
        f"date={intent.date!r}",
        f"loc={intent.start_location!r}",
    ]
    if clarify_n:
        info_parts.append(f"clarify={clarify_n}/{MAX_CLARIFY_ITERATIONS}")
    if intent.scenario != "unknown":
        info_parts.append(f"scenario={intent.scenario}")
    if intent.party_size:
        info_parts.append(f"party={intent.party_size}")
    if intent.hard_constraints:
        info_parts.append(f"hard={intent.hard_constraints}")
    if intent.preferences:
        info_parts.append(f"prefs={intent.preferences}")
    if intent.missing_critical:
        info_parts.append(f"missing={intent.missing_critical}")
    info_parts.append(f"→ {route}")
    logger.info("Intent | %s", " | ".join(info_parts))
    logger.debug("Intent JSON | %s",
                 intent.model_dump_json(indent=2, exclude_none=True))


# ═══════════════════════════════════════════════════════════════════════
# Node 2a: simple_answer
# ═══════════════════════════════════════════════════════════════════════

SIMPLE_ANSWER_PROMPT = """\
You are Finn, a helpful local assistant. Answer the user's question concisely.
If you don't know something, say so honestly. Keep answers under 200 words."""


async def simple_answer(state: AgentState) -> dict:
    """Node 2a: Answer a simple non-trip question."""
    llm = make_model(temperature=0.7)
    messages = [
        SystemMessage(content=SIMPLE_ANSWER_PROMPT),
        HumanMessage(content=state["messages"][-1].content),
    ]
    text = await llm_invoke(llm, messages, "simple_answer", stream=True)
    return {
        "messages": [{"role": "assistant", "content": text}],
        "next_action": "done",
    }


# ═══════════════════════════════════════════════════════════════════════
# Node 2c: reject
# ═══════════════════════════════════════════════════════════════════════

REJECT_MESSAGE = (
    "抱歉，我是出行规划助手，目前只能帮你规划周末外出吃喝玩乐、"
    "订票订座。你可以试试跟我说：\n"
    '• "周末想去南山爬山"\n'
    '• "帮我订周六晚上朝阳区的火锅"\n'
    '• "明天下午3点想看场电影"\n'
    "试试看？"
)


async def reject(state: AgentState) -> dict:
    """Node 2c: Inform user of capability boundaries."""
    return {
        "messages": [{"role": "assistant", "content": REJECT_MESSAGE}],
        "next_action": "done",
    }


# ═══════════════════════════════════════════════════════════════════════
# Node 2b: decompose_and_plan
# ═══════════════════════════════════════════════════════════════════════

DECOMPOSE_PROMPT = """\
你是 Finn 的任务分解与规划模块。

你会收到用户出行意图。你需要基于真实数据构造一份可执行的出行计划。

如果是修改已有计划：
- 以已有计划为基础，按用户修改请求调整
- 保留用户未要求修改的所有内容

══════════════════════════════════════
首要任务：使用工具搜索真实数据
══════════════════════════════════════

你拥有以下工具，必须主动调用它们来获取真实世界信息：

maps_geo — 将地名转为经纬度坐标
maps_around_search — 在指定坐标周边搜索 POI（名称、地址、评分）
maps_text_search — 关键词搜索特定类型商家
maps_search_detail — 查询 POI 详情（营业时间、电话、人均消费）
maps_regeocode — 将坐标转为地址
maps_distance — 计算两点直线距离
maps_direction_walking — 步行路线和耗时
maps_direction_driving — 驾车路线和耗时
maps_direction_bicycling — 骑行路线和耗时
maps_direction_transit_integrated — 公交/地铁路线和耗时
maps_weather — 查询天气

调用工具的基本流程：
1. 用 maps_weather 查天气
2. 用 maps_geo 获取出发地坐标
3. 用 maps_around_search 搜索周边目标场所
4. 对候选场所用 maps_search_detail 查详情
5. 用 maps_direction_* 计算场所间路线和耗时

══════════════════════════════════════
搜索完成后，输出出行计划 JSON
══════════════════════════════════════

{
  "sub_tasks": [
    {
      "id": "<唯一ID>",
      "type": "search | compare | book",
      "target": "<任务目标，例如 午餐、电影、交通>",
      "dependencies": ["<依赖的任务ID>" ...],
      "params": {<相关参数，包含真实搜索到的商家名称和地址>},
      "compensatory": "<对应的取消任务ID>" | null
    }
  ],
  "total_cost_estimate": <预估总花费 CNY，或 null>,
  "notes": "<中文摘要，介绍基于真实搜索结果的计划，包含具体商家名和路线>"
}

规则：
1. 最多 8 个子任务
2. "search" 任务无依赖，用于收集信息
3. "book" 任务至少依赖一个 "search" 任务
4. 每个 "book" 任务必须有 compensatory 取消任务
5. 不同地点间需加交通子任务（基于实际路线耗时）
6. 商家名称、地址必须来自工具返回的真实数据，禁止编造
7. 用户指定了预算则严格控制在预算内
8. 子任务按时间线有序排列"""


async def decompose_and_plan(state: AgentState) -> dict:
    """Node 2b: Decompose intent into executable sub-task DAG.

    Uses a ReAct loop with MCP tools to search for real venue data.
    """
    intent = state["intent"]
    modify_feedback = state.get("modify_feedback", "")
    existing_plan = state.get("plan")

    # Build intent description
    parts = [f"Activity: {intent.activity}"]
    if intent.date:
        parts.append(f"Date: {intent.date}")
    if intent.start_time:
        parts.append(f"Time: {intent.start_time}")
    if intent.duration_hours:
        parts.append(f"Duration: ~{intent.duration_hours}h")
    if intent.start_location:
        parts.append(f"Start: {intent.start_location}")
    if intent.area:
        parts.append(f"Area: {intent.area}")
    if intent.radius_km:
        parts.append(f"Max distance: {intent.radius_km}km")
    if intent.scenario and intent.scenario != "unknown":
        parts.append(f"Scenario: {intent.scenario}")
    if intent.party_size:
        parts.append(f"Party size: {intent.party_size}")
    for m in intent.party_members:
        member_str = f"  {m.role}"
        if m.age:
            member_str += f" (age {m.age})"
        if m.constraints:
            member_str += f" constraints: {m.constraints}"
        if m.preferences:
            member_str += f" prefs: {m.preferences}"
        parts.append(member_str)
    if intent.budget_total:
        parts.append(f"Budget total: ¥{intent.budget_total}")
    if intent.budget_per_person:
        parts.append(f"Budget per person: ¥{intent.budget_per_person}")
    if intent.hard_constraints:
        parts.append(f"HARD constraints: {intent.hard_constraints}")
    if intent.preferences:
        parts.append(f"Preferences: {intent.preferences}")

    lines = ["User intent:"]
    lines.extend(f"  {p}" for p in parts)

    if modify_feedback and existing_plan:
        plan_json = existing_plan.model_dump_json(indent=2, exclude_none=True)
        lines.append(f"\n*** REVISION REQUEST ***")
        lines.append(f'The user wants to modify the plan: "{modify_feedback}"')
        lines.append("Revise the following plan to incorporate this change:")
        lines.append(plan_json)
        lines.append("Keep everything the user didn't ask to change.")

    user_prompt = "\n".join(lines)

    # Run ReAct loop with MCP tools
    async with mcp_session() as (tools, server_id):
        llm = make_model(temperature=0.7)
        text = await react_loop(
            llm, tools, DECOMPOSE_PROMPT, user_prompt,
            "decompose_and_plan", stream=False,
        )

    data = _extract_json(text)
    data = _normalize_plan(data)
    clean = _strip_nulls(data)
    plan = Plan.model_validate(clean)

    return {"plan": plan, "next_action": "verify_plan"}


# ═══════════════════════════════════════════════════════════════════════
# Node 3: verify_plan
# ═══════════════════════════════════════════════════════════════════════

VERIFY_PROMPT = """\
You are a plan verifier for Finn, a local short-trip planning agent.
Evaluate the given plan against the original user intent.

Output ONLY a valid JSON object:
{
  "score": <0.0 to 1.0>,
  "issues": ["<issue description>" ...],
  "status": "pass" | "fix" | "reject"
}

Evaluation dimensions:
1. Temporal feasibility: Do times conflict? Are transit times realistic?
2. Budget: Does the estimated cost fit within the user's budget?
3. Completeness: Does the plan cover all user requirements?
4. Consistency: Do the sub-task dependencies make sense?
5. Practicality: Are the venues realistic for the location? Are they open?

Scoring:
- 0.8-1.0 → status="pass" (minor issues at most)
- 0.4-0.8 → status="fix" (fixable issues, adjust and re-verify)
- 0.0-0.4 → status="reject" (fundamentally broken, start over)

Be specific in issues. In Chinese.
Output ONLY the JSON, no other text."""


async def verify_plan(state: AgentState) -> dict:
    """Node 3: Validate plan against user intent."""
    intent = state["intent"]
    plan = state["plan"]
    plan_json = plan.model_dump_json(indent=2, exclude_none=True)

    i_lines = [f"Activity: {intent.activity}"]
    if intent.date:
        i_lines.append(f"Date: {intent.date}")
    if intent.start_time:
        i_lines.append(f"Time: {intent.start_time}")
    if intent.duration_hours:
        i_lines.append(f"Duration: ~{intent.duration_hours}h")
    if intent.start_location:
        i_lines.append(f"Start: {intent.start_location}")
    if intent.area:
        i_lines.append(f"Area: {intent.area}")
    if intent.scenario and intent.scenario != "unknown":
        i_lines.append(f"Scenario: {intent.scenario}")
    if intent.party_size:
        i_lines.append(f"Party: {intent.party_size}")
    for m in intent.party_members:
        mstr = f"  {m.role}"
        if m.age:
            mstr += f" (age {m.age})"
        if m.constraints:
            mstr += f" constraints={m.constraints}"
        i_lines.append(mstr)
    if intent.budget_total:
        i_lines.append(f"Budget total: ¥{intent.budget_total}")
    if intent.budget_per_person:
        i_lines.append(f"Budget/person: ¥{intent.budget_per_person}")
    if intent.hard_constraints:
        i_lines.append(f"HARD: {intent.hard_constraints}")
    if intent.preferences:
        i_lines.append(f"Prefs: {intent.preferences}")

    intent_text = "\n  ".join(i_lines)
    user_prompt = f"User intent:\n  {intent_text}\n\nPlan to verify:\n{plan_json}"

    llm = make_model(temperature=0.3)
    messages = [
        SystemMessage(content=VERIFY_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    text = await llm_invoke(llm, messages, "verify_plan", stream=True)
    data = _extract_json(text)
    verification = Verification.model_validate(data)

    iterations = state.get("plan_iterations", 0) + 1
    return {"verification": verification, "plan_iterations": iterations}


# ═══════════════════════════════════════════════════════════════════════
# Node 7: adjust_plan
# ═══════════════════════════════════════════════════════════════════════

ADJUST_PROMPT = """\
You are a plan editor for Finn, a local short-trip planning agent.
Revise the plan according to the provided feedback.

If user modification feedback is provided, prioritize the user's request:
add, remove, or change tasks to satisfy their requirements.
If verification issues are also present, address those too.

Output ONLY a valid JSON object with the same structure as the original plan:
{
  "sub_tasks": [...],
  "total_cost_estimate": <number or null>,
  "notes": "<revised summary in Chinese, mention what was changed>"
}

Rules (same as original planning):
1. Max 8 sub-tasks.
2. "search" tasks have no dependencies.
3. "book" tasks depend on at least one "search" task.
4. Every "book" task must have a compensatory cancel task.
5. Be specific about venues, times, prices.

Output ONLY the JSON, no other text."""


async def adjust_plan(state: AgentState) -> dict:
    """Node 7: Adjust plan based on verification issues and/or user feedback."""
    plan = state["plan"]
    verification = state.get("verification")
    modify_feedback = state.get("modify_feedback", "")

    plan_json = plan.model_dump_json(indent=2, exclude_none=True)
    parts = [f"Original plan:\n{plan_json}"]

    if verification and verification.issues:
        issues_text = "\n".join(f"  - {i}" for i in verification.issues)
        parts.append(f"\nVerification issues to fix:\n{issues_text}")

    if modify_feedback:
        parts.append(f"\nUser modification request:\n  {modify_feedback}")

    user_prompt = "\n".join(parts)

    llm = make_model(temperature=0.5)
    messages = [
        SystemMessage(content=ADJUST_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    text = await llm_invoke(llm, messages, "adjust_plan", stream=True)
    data = _extract_json(text)
    data = _normalize_plan(data)
    clean = _strip_nulls(data)
    revised = Plan.model_validate(clean)

    return {"plan": revised, "next_action": "verify_plan"}


# ═══════════════════════════════════════════════════════════════════════
# Node 4: present_to_user  ★ HITL interrupt
# ═══════════════════════════════════════════════════════════════════════


def _format_plan(plan: Plan) -> str:
    """Render a Plan as a human-readable string."""
    lines = ["📋  Your Weekend Plan\n"]
    lines.append(f"  {plan.notes}\n")

    if plan.total_cost_estimate:
        lines.append(f"  💰 Estimated total: ¥{plan.total_cost_estimate}\n")

    lines.append("  Tasks:")
    for st in plan.sub_tasks:
        icon = {"search": "🔍", "compare": "📊", "book": "🎫"}.get(st.type, "•")
        deps = f" (after: {', '.join(st.dependencies)})" if st.dependencies else ""
        comp = f" [rollback: {st.compensatory}]" if st.compensatory else ""
        lines.append(f"    {icon} {st.target} [{st.type}]{deps}{comp}")

    return "\n".join(lines)


async def present_to_user(state: AgentState) -> dict:
    """Node 4: Display plan and wait for user decision (HITL interrupt)."""
    plan = state["plan"]

    decision = interrupt({
        "type": "plan_review",
        "plan": _format_plan(plan),
        "options": ["confirm", "modify", "cancel"],
    })

    if isinstance(decision, dict):
        action = decision.get("action", "cancel")
        feedback = decision.get("feedback", "")
        return {
            "next_action": action,
            "modify_feedback": feedback,
            "plan_iterations": 0,
        }
    return {"next_action": decision}


# ═══════════════════════════════════════════════════════════════════════
# Node 5: book_worker  (fan-out via Send)
# ═══════════════════════════════════════════════════════════════════════

MAX_RETRIES = 2


def _simulate_booking(task: SubTask, retry_count: int = 0) -> BookingResult:
    """Simulate a booking API call with deterministic mixed results."""
    seed = int(hashlib.md5(task.id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed + retry_count * 1000)
    roll = rng.random()

    if "cancel" in task.id.lower():
        return BookingResult(
            task_id=task.id, status="success",
            order_id=f"CANCEL-{task.id[:8].upper()}",
        )

    if roll < 0.15:
        return BookingResult(
            task_id=task.id, status="failed",
            error="Network timeout after 30s",
            error_type="transient", retries=retry_count,
        )
    elif roll < 0.25:
        return BookingResult(
            task_id=task.id, status="failed",
            error=f"Sorry, '{task.target}' is fully booked / sold out",
            error_type="recoverable", retries=retry_count,
        )
    else:
        return BookingResult(
            task_id=task.id, status="success",
            order_id=f"ORD-{task.id[:8].upper()}-{abs(hash(task.id)) % 10000:04d}",
        )


async def book_worker(state: AgentState) -> dict:
    """Node 5: Execute a single booking task (invoked via Send fan-out)."""
    task_id = state.get("current_task_id", "")
    plan = state.get("plan")
    retry_count = state.get("current_retry_count", 0)

    task = next((st for st in plan.sub_tasks if st.id == task_id), None)
    if task is None:
        return {
            "bookings": {
                task_id: BookingResult(
                    task_id=task_id, status="failed",
                    error=f"Task {task_id} not found in plan",
                    error_type="fatal",
                )
            }
        }

    result = _simulate_booking(task, retry_count)
    return {"bookings": {task_id: result}}


# ═══════════════════════════════════════════════════════════════════════
# Node: verify_execution
# ═══════════════════════════════════════════════════════════════════════


async def verify_execution(state: AgentState) -> dict:
    """Check execution results: classify successes and failures."""
    bookings = state.get("bookings", {})
    plan = state["plan"]

    succeeded = []
    failed = []
    for task_id, result in bookings.items():
        if result.status == "success":
            succeeded.append(task_id)
        elif result.status == "failed":
            failed.append(task_id)

    lines = [f"Execution summary: {len(succeeded)} succeeded, {len(failed)} failed"]
    for tid in succeeded:
        r = bookings[tid]
        lines.append(f"  ✅ {tid}: {r.order_id}")
    for tid in failed:
        r = bookings[tid]
        lines.append(f"  ❌ {tid}: {r.error} [{r.error_type}]")

    if len(failed) == 0:
        es = "done"
    elif len(succeeded) == 0:
        es = "failed"
    else:
        es = "partial"

    return {
        "messages": [{"role": "assistant", "content": "\n".join(lines)}],
        "execution_status": es,
    }


# ═══════════════════════════════════════════════════════════════════════
# Node 6b: handle_failures
# ═══════════════════════════════════════════════════════════════════════


async def handle_failures(state: AgentState) -> dict:
    """Classify each failure and decide: retry, compensate, or escalate."""
    bookings = state.get("bookings", {})
    plan = state["plan"]

    has_retryable = False
    msgs: list[str] = []

    for task_id, result in bookings.items():
        if result.status != "failed":
            continue

        if result.error_type == "transient" and result.retries < MAX_RETRIES:
            msgs.append(f"🔄 {task_id}: transient error, will retry "
                        f"(attempt {result.retries + 1}/{MAX_RETRIES})")
            has_retryable = True

        elif result.error_type == "recoverable":
            task = next((st for st in plan.sub_tasks if st.id == task_id), None)
            if task and task.compensatory:
                comp_id = task.compensatory
                bookings[comp_id] = BookingResult(
                    task_id=comp_id, status="compensated",
                    error=f"Auto-compensated due to failure of {task_id}",
                )
            msgs.append(f"🔙 {task_id}: recoverable error, "
                        f"compensating up to {result.retries}")

        else:
            msgs.append(f"🚫 {task_id}: fatal error — {result.error}")

    next_action = "retry" if has_retryable else "notify"

    return {
        "bookings": bookings,
        "retry_count": state.get("retry_count", 0) + 1,
        "next_action": next_action,
        "messages": [{"role": "assistant", "content": "\n".join(msgs)}],
    }


# ═══════════════════════════════════════════════════════════════════════
# Node 6a: summarize_result
# ═══════════════════════════════════════════════════════════════════════


async def summarize_result(state: AgentState) -> dict:
    """Node 6a: Generate final summary of all execution results."""
    bookings = state.get("bookings", {})
    plan = state["plan"]

    succeeded = [r for r in bookings.values() if r.status == "success"]
    compensated = [r for r in bookings.values() if r.status == "compensated"]
    failed = [r for r in bookings.values()
              if r.status == "failed" and r.error_type == "fatal"]

    lines = ["Done! Here's your trip summary:\n"]

    if succeeded:
        lines.append("  Confirmed:")
        for r in succeeded:
            task = next((st for st in plan.sub_tasks if st.id == r.task_id), None)
            label = task.target if task else r.task_id
            lines.append(f"    ✅ {label} — {r.order_id}")

    if compensated:
        lines.append("\n  Compensated (auto-cancelled):")
        for r in compensated:
            lines.append(f"    🔙 {r.task_id} — {r.error}")

    if failed:
        lines.append("\n  Failed:")
        for r in failed:
            lines.append(f"    ❌ {r.task_id} — {r.error}")

    if not failed:
        lines.append("\n  Enjoy your trip! 🎉")

    return {
        "messages": [{"role": "assistant", "content": "\n".join(lines)}],
        "execution_status": "done",
    }


# ═══════════════════════════════════════════════════════════════════════
# Node: notify_user (non-retryable failure escalation)
# ═══════════════════════════════════════════════════════════════════════


async def notify_user(state: AgentState) -> dict:
    """Escalate non-retryable failures to the user for manual resolution."""
    bookings = state.get("bookings", {})
    failed = [r for r in bookings.values() if r.status == "failed"]
    compensated = [r for r in bookings.values() if r.status == "compensated"]

    lines = ["Found issues with your bookings:\n"]
    for r in failed:
        lines.append(f"  ❌ {r.task_id}: {r.error}")
    if compensated:
        lines.append("\n  The following were auto-cancelled:")
        for r in compensated:
            lines.append(f"  🔙 {r.task_id}: {r.error}")
    lines.append("\nPlease try again or modify your plan.")

    return {
        "messages": [{"role": "assistant", "content": "\n".join(lines)}],
        "next_action": "done",
    }


# ═══════════════════════════════════════════════════════════════════════
# Edge routing functions
# ═══════════════════════════════════════════════════════════════════════

MAX_CLARIFY_ITERATIONS = 6
MAX_PLAN_ITERATIONS = 4
MAX_RETRY_TOTAL = 3


def route_after_verify(state: AgentState) -> str:
    """Edge B: Route based on verification result."""
    verification = state.get("verification")
    iterations = state.get("plan_iterations", 0)

    if verification is None:
        return "reject"

    if verification.status == "pass":
        return "present_to_user"

    if verification.status == "fix" and iterations < MAX_PLAN_ITERATIONS:
        return "adjust_plan"

    return "reject"


def route_after_present(state: AgentState):
    """Edge C: Route based on user decision after plan review."""
    action = state.get("next_action", "cancel")
    if action == "confirm":
        plan = state["plan"]
        book_tasks = [
            st for st in (plan.sub_tasks if plan else [])
            if st.type == "book" and "cancel" not in st.id.lower()
        ]
        if not book_tasks:
            return "summarize"
        return [
            Send("book_worker", {
                "plan": plan,
                "current_task_id": st.id,
                "current_retry_count": 0,
            })
            for st in book_tasks
        ]
    elif action == "modify":
        return "decompose_and_plan"
    else:
        return "cancel"


def route_after_exec(state: AgentState) -> str:
    """Edge D: Route based on execution results."""
    es = state.get("execution_status", "failed")
    if es == "done":
        return "summarize"
    elif es == "partial":
        return "handle_failures"
    else:
        return "handle_failures"


def route_after_handle_failures(state: AgentState):
    """Edge E: Route after failure handling."""
    action = state.get("next_action", "notify")
    retry_count = state.get("retry_count", 0)

    if action == "retry" and retry_count < MAX_RETRY_TOTAL:
        plan = state["plan"]
        bookings = state.get("bookings", {})
        sends = [
            Send("book_worker", {
                "plan": plan,
                "current_task_id": tid,
                "current_retry_count": retry_count,
            })
            for tid, r in bookings.items()
            if r.status == "failed" and r.error_type == "transient"
        ]
        if sends:
            return sends
        return "notify"
    else:
        return "notify"
