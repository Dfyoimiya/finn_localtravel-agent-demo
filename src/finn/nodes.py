"""LangGraph node implementations using LangChain.

Each node is a pure async function: (state, config) -> partial state update.
LLM nodes use LangChain's ChatOpenAI with manual ReAct loops for tool calling.

Text-based JSON extraction for structured output (DeepSeek V4 does not
support tool_choice in thinking mode).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import END
from langgraph.types import Command, interrupt, Send

from finn.config import config
from finn.llm import llm_invoke, make_model, react_loop
from finn.logger import logger
from finn.mcp import mcp_session
from finn.state import (
    AgentState,
    BookingResult,
    ConstraintProfile,
    ExtractResult,
    Plan,
    POICandidate,
    POISearchResult,
    SceneType,
    SubTask,
    UpdateExtractResultInput,
    Verification,
)
from finn.poi import (
    poi_cache,
    build_search_keywords,
    deduplicate_pois,
    make_cache_key,
    rank_pois,
    select_search_strategy,
)

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


def _strip_empty_containers(obj):
    """Recursively remove empty lists, dicts, and None values from a dict."""
    if isinstance(obj, dict):
        return {
            k: _strip_empty_containers(v)
            for k, v in obj.items()
            if v is not None and v != [] and v != {} and v != ""
        }
    elif isinstance(obj, list):
        return [_strip_empty_containers(item) for item in obj]
    return obj


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


def _backfill_plan_locations(plan: Plan, candidates: list[POICandidate]) -> None:
    """Backfill location/address/rating from candidate pool into book task params.

    LLM-modified plans may drop param fields. This ensures transport info is
    available for PlanCard generation in _cards_from_plan.
    Mutates plan in-place.
    """
    from finn.state import POICandidate

    candidate_by_name: dict[str, POICandidate] = {}
    candidate_by_id: dict[str, POICandidate] = {}
    for c in candidates:
        if c.name:
            candidate_by_name[c.name] = c
        if c.id:
            candidate_by_id[c.id] = c

    for st in plan.sub_tasks:
        if st.type != "book":
            continue
        params = st.params or {}
        name = params.get("name", "")
        pid = params.get("id", "")
        # Try id first, then name
        ref = candidate_by_id.get(pid) or candidate_by_name.get(name)
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
        if not params.get("slot"):
            params["slot"] = "play"


# ═══════════════════════════════════════════════════════════════════════
# Dynamic System Prompt Builder
# ═══════════════════════════════════════════════════════════════════════

_BASE_PROMPT = """\
你是 Finn 的意图提取模块——一个本地短途出行规划 Agent。

你的任务：分析对话，输出一个 JSON 对象，同时包含提取的意图数据和路由决策。

注意：对话中可能包含 [系统上下文] 标记，其中有当前时间、基于 IP 的用户位置，
以及 [用户画像]（偏好口味、常去区域、常用同行人、预算范围等）。
如果用户说"附近"、"今天"等模糊表述，使用系统上下文中的信息来填充具体值。
如果用户画像中有匹配的偏好或同行人信息，优先使用并填充到意图中。

输出格式（只输出合法的 JSON 对象，不要 markdown，不要 ``` 代码块）：

{
  "route": "plan" | "clarify" | "reject",
  "intent": {
    "activity_summary": "<一句话总结，如'带孩子去公园玩然后吃饭'>",
    "city": "<城市名，如'北京'>",
    "plan_date": "<YYYY-MM-DD>",
    "time_window_start": "<HH:MM>",
    "time_window_hours": <float>,
    "guest_count": <int>,
    "scene": "family" | "friends",
    "raw_utterance": "<用户原始输入>"
  },
  "requirements": {
    "must_visit_pois": ["<必去地点>"],
    "must_have_cuisine": ["<必须菜系>"],
    "must_have_activity_type": ["<必须活动类型>"],
    "special_requests": ["<特殊需求>"],
    "notes": "<其他备注或 null>"
  },
  "hard_constraints": {
    "budget_max_cny": <float 或 null>,
    "dietary_restrictions": ["<饮食限制>"],
    "child_age": <int 或 null>,
    "accessibility_needed": <bool>,
    "time_deadline": "<HH:MM 或 null>",
    "must_include_poi_ids": []
  },
  "soft_constraints": {
    "budget_preference": "economy" | "mid" | "luxury" | null,
    "travel_pace": "relaxed" | "balanced" | "fast" | null,
    "preferred_poi_types": ["<偏好 POI 类型>"],
    "preferred_cuisines": ["<偏好菜系>"],
    "preferred_transport": "walk" | "transit" | "drive" | null,
    "max_transit_minutes": <float 或 null>,
    "avoid_poi_types": ["<不想去的类型>"]
  },
  "group": {
    "type": "family" | "friends",
    "tags": ["<结构化标签>"],
    "hard_constraints": ["<群组约束>"],
    "soft_preferences": ["<群组偏好>"]
  },
  "time": {
    "date": "<YYYY-MM-DD>",
    "start_time": "<HH:MM>",
    "end_time": "<HH:MM>",
    "flexibility": "strict" | "normal" | "high"
  },
  "geo": {
    "center_address": "<出发地址>",
    "radius_m": <int>,
    "max_transit_time_min": <int>,
    "constraint_desc": "<原始距离描述>"
  },
  "chain": {
    "template": ["play", "eat"],
    "min_nodes": <int>,
    "max_nodes": <int>,
    "preferred_activity_types": ["<活动类型关键词>"]
  },
  "constraint_profile": {
    "budget_max_cny": <float 或 null>,
    "dietary_restrictions": ["<饮食限制>"],
    "child_age": <int 或 null>,
    "accessibility_needed": <bool>,
    "time_deadline": "<HH:MM 或 null>",
    "must_visit_poi_names": ["<必去地点名>"],
    "must_have_cuisine": ["<必须菜系>"],
    "preferred_poi_types": ["<偏好POI类型>"],
    "preferred_cuisines": ["<偏好菜系>"],
    "preferred_transport": "walk" | "transit" | "drive" | null,
    "max_transit_minutes": <float 或 null>,
    "avoid_poi_types": ["<不想去的类型>"],
    "budget_preference": "economy" | "mid" | "luxury" | null,
    "travel_pace": "relaxed" | "balanced" | "fast" | null,
    "meal_slots": [{"slot": "<lunch|dinner>", "window": "<HH:MM-HH:MM>", "cuisines": ["<菜系>"], "casual": <bool>}],
    "parallel_activities": [{"group": "<adults|kids|elderly>", "slot": "<play|eat|follow_up>", "activity_hint": "<活动关键词>", "cluster_required": <bool>}],
    "distance_preference": "prefer_optimal_range" | "nearby_only" | "no_limit" | null,
    "preferred_transit_range_min": <int 或 null>,
    "preferred_transit_range_max": <int 或 null>
  },
  "confidence": <float 0-1>,
  "follow_up_question": "<自然的中文追问，或 null>"
}

══════════════════════════════════════
路由规则 — 必须将 "route" 设为以下之一：
══════════════════════════════════════

"plan"
  必要字段全部非空（activity_summary, city, plan_date,
  time_window_start, time_window_hours, guest_count）。
  budget_max_cny 按"预算推断"规则自动填充，不需要追问用户。
  可以进入 POI 搜索与规划阶段。
  此时 follow_up_question 必须为 null。

"clarify"
  activity_summary 非空（用户有出行意图），但仍有必要字段缺失。
  将 follow_up_question 设为一句简短的自然中文追问，每次只问一个缺失字段。

"reject"
  activity_summary 为空——这不是出行规划请求。用户闲聊、问事实性问题。

══════════════════════════════════════
通用提取规则
══════════════════════════════════════
1. 多轮合并：如果 prompt 中提供了 PREVIOUS EXTRACT，将最新消息合并进去——
   除非用户明确修改，否则保留已提取的字段。
2. 约束分类："必须"/"不能"/"一定要" → hard_constraints。
   "最好"/"喜欢"/"想" → soft_constraints 或 group.soft_preferences。
   同一项不要同时出现在两个列表中。
3. 诚实原则：只填用户实际提供的信息。null / 空列表好过瞎猜。
4. 追问限制：每次只问一个缺失字段。自然中文。如果 route 是 "plan" 或 "reject"，
   follow_up_question 必须为 null。
5. 预算推断（budget_max_cny 和 budget_preference）：
   - 如果用户画像中有明确的人均预算（如 "人均150元"），用它 × guest_count 作为总预算
   - 如果画像中的人均预算是无效天花板值（如 "99999元以下"、"不限"、"无上限"），视为未设置
   - 画像未设置或无效时，用常识根据城市和场景推断：家庭/朋友一日游一般人均 100-200 元
   - budget_preference 同步推断：人均<100→economy，100-300→mid，>300→luxury
   - 预算不需要追问用户，直接填充即可
6. group.tags 标签规范：
   - 人群: child_<age>, elderly_<age>, friends_<count>_mix
   - 饮食: diet_low_calorie, diet_halal, diet_vegan, diet_seafood_allergy, diet_spicy_avoid
   - 约束: child_safe, wheelchair_accessible, non_smoking
   - 偏好: photo_friendly, quiet, indoor, outdoor, pet_friendly
7. 用户画像偏差处理（CRITICAL — 以用户当前输入为最高优先级）：
   - 用户画像是历史偏好，可能已过时。当画像与本次用户原话冲突时，以本次原话为准。
   - 画像中标记为"（低）"的偏好是弱信号，直接忽略，不填充到任何字段。
   - 画像中标记为"（高）"的偏好仅在用户本次未明确表达时作为默认值使用。
   - 画像中的"常用同行人"若与本次出行人员不符，完全忽略画像中与该同行人绑定的偏好。
   - 画像中的"已知约束"（如"不吃辣"）仅在用户本次原话中未提及或未推翻时保留。
   - 画像中的预算范围若与用户本次表达冲突（如用户本次说"人均200"但画像说"人均50"），
     以本次为准。
   - 总原则：用户当前自然语言输入 > 用户本次明确约束 > 画像历史数据 > 系统默认推断
8. 活动链模板（chain.template）推断：
   - 可用时长 ≤4h → ["play", "eat"] 或 ["play"]
   - 可用时长 4-8h → ["play", "eat", "follow_up"]
   - 可用时长 8-12h → ["play", "eat", "play", "eat"] 或 ["play", "eat", "play", "follow_up"]
   - 可用时长 >12h → ["play", "eat", "play", "eat", "follow_up"]
   - 若用户明确提到午餐+晚餐两餐，template 必须包含两个 "eat" 槽位
   - min_nodes = template 长度，max_nodes = template 长度 + 1
   - **并行活动扩展**：若检测到并行活动（规则12），max_nodes += len(parallel_activities)
     并行活动不占用模板槽位，但在同一时段追加额外节点
9. 用餐时间（chain template 中 eat 槽位的隐含时序）：
   - 第一个 "eat" = 午餐（约11:00-13:00），第二个 "eat" = 晚餐（约17:00-19:00）
   - 若用户说"晚上吃X"或"晚饭X"，该要求对应最后一个 eat 槽位
   - 若用户说"中午随意"或"午饭随便"，对应第一个 eat 槽位不做严格约束
   - 将用餐时间要求写入 requirements.notes，如"晚餐必须火锅，午餐无要求"
10. 约束画像（constraint_profile）提取：
   - 从用户原话中提取硬约束和软约束，填充到输出 JSON 的 constraint_profile 字段
   - budget_max_cny：用户明确说的人均金额 × guest_count；若无明确金额则填 null
   - meal_slots 推断规则：
     · 可用时长 ≤4h → []（无分餐，所有 eat 都按通用处理）
     · 可用时长 4-8h → [{"slot":"lunch","window":"11:00-13:00","cuisines":[],"casual":true}]
     · 可用时长 8-12h → [{"slot":"lunch","window":"11:00-13:00","cuisines":[],"casual":true},
                        {"slot":"dinner","window":"17:00-19:00","cuisines":[],"casual":false}]
     · >12h → 同 8-12h
   - 若用户指定"晚上吃X"/"晚饭X"，将 cuisine 填入 dinner slot 的 cuisines 列表
   - 若用户说"中午随意"/"午饭随便"，lunch slot 标记 casual=true
   - preferred_transport: 从用户话中推断（"打车"→drive，"走走"→walk）
   - budget_preference / travel_pace: 从用户画像和原话推断
   - alpha/beta/gamma/delta 由代码 compute_strategy_weights() 计算，LLM 无需输出
   - 输出 constraint_profile 对象作为 JSON 的顶层字段
11. 转场时间/距离提取（CRITICAL — 理解用户距离偏好语义）：
    - 用户说"打车X小时内"、"车程X小时" → max_transit_minutes = X*60（单程上限）
    - 用户说"车程X分钟" → max_transit_minutes = X
    - **距离偏好语义**："X小时内"表示单程≤X小时的POI集群都可接受。
      用户不是要求越近越好，而是偏向**接近X小时的集群**（如X=1时，40-60分钟集群最优）。
      · 在 geo.constraint_desc 中记录原始描述（如"单程打车≤1h，偏好40-60min集群"）
      · max_transit_time_min 设为 X*60，作为硬上限
      · soft_constraints.max_transit_minutes 设为 X*60，作为软约束
      · 偏好区间为 [X*60*0.5, X*60*0.8]（即50%-80%上限），用户体验最优
    - 用户说"别太远"/"不要太远" → max_transit_minutes = 30（步行距离优先）
    - 用户说"远一点也行"/"不在乎距离" → max_transit_minutes = null（不限制）
    - 同时填充 geo.max_transit_time_min 和 soft_constraints.max_transit_minutes
    - constraint_profile 中也填充 max_transit_minutes 字段
12. 多用户并行活动检测（multi_user_parallel）：
    - 家庭/亲子场景隐式触发：用户带儿童去"儿童乐园"/"亲子餐厅"，大人在同一商圈
      可能有独立活动（购物/逛商场/咖啡）。检测到这种场景时，在 constraint_profile
      中填充 parallel_activities 数组。
    - 朋友场景显式触发：用户说"一部分人去X，另一部分人去Y"→ 直接提取
    - parallel_activities 格式：[{"group":"adults"|"kids"|"<用户名>", "slot":"play"|"eat"|"follow_up",
      "activity_hint":"<活动类型关键词>", "cluster_required":true}]
    - 关键约束：并行活动必须位于同一 cluster（同一商场/商圈/步行距离内）
    - 常见模式识别：
      · 儿童乐园/亲子活动 → 孩子去亲子项目，成人可去购物/餐饮/咖啡
      · 电影院 → 部分人看电影，其他人可逛街
      · 提到"老人" → 老人可能需要休闲场所，年轻人可去更活跃的活动
    - 如果判定存在并行活动，在 requirements.notes 中记录，如
      "检测到亲子并行：儿童需亲子活动，成人需同商圈购物/餐饮选择"
    - constraint_profile 中 parallel_activities 为空数组表示无并行
    """

_FAMILY_BLOCK = """
══════════════════════════════════════
家庭场景（family）专属规则
══════════════════════════════════════
1. 儿童信息提取优先级最高：
   - 必须提取 child_age（硬约束）
   - 根据年龄生成对应标签: child_<age>（如 child_5yo）
   - 12岁以下 → group.hard_constraints 加入 child_safe
   - 3岁以下 → group.hard_constraints 加入 stroller_accessible
2. 亲子友好筛选标签：
   - group.tags 加入: kid_friendly, family_restaurant
   - chain.preferred_activity_types 偏向: 公园, 动物园, 博物馆, 亲子餐厅
3. 安全与便利：
   - 需要午睡 → time.flexibility = "strict"
   - 儿童饮食限制 → hard_constraints.dietary_restrictions
   - 老人同行 → group.hard_constraints 加入 wheelchair_accessible
4. 预算: 家庭出行通常人均预算偏低，优先 economy/mid
"""

_FRIENDS_BLOCK = """
══════════════════════════════════════
朋友场景（friends）专属规则
════════════════════════════════════════
1. 群体共识提取：
   - 多个朋友可能有不同偏好 → group.tags 汇总所有人偏好
   - 根据人数生成标签: friends_<count>_mix
2. 社交属性标签：
   - group.tags 加入: social, photo_friendly
   - chain.preferred_activity_types 偏向: 网红店, 火锅, 烧烤, 咖啡馆, 酒吧
3. 预算与节奏：
   - 群体 AA → budget_preference 倾向于 mid
   - 多人需要更长的转场时间 → max_transit_minutes 适当增加
4. 多样性需求：
   - preferred_cuisines 可包含多种菜系
   - chain.template 可包含更多 follow_up 节点（如饭后咖啡/酒吧）
"""

_NO_SCENE_BLOCK = """
══════════════════════════════════════
场景判断指引
══════════════════════════════════════
当前尚未确定出行场景。请从对话中判断：
- 提到"孩子"/"带娃"/"亲子"/"老人"/"爸妈" → scene = "family"
- 提到"朋友"/"几个哥们"/"姐妹"/"同事"/"约" → scene = "friends"
- 如果无法判断 → 在 follow_up_question 中问"是和家人还是朋友一起？"
"""


def _build_clarify_prompt(scene: str | None) -> str:
    """Assemble system prompt dynamically based on detected scene type.

    When scene is unknown (first turn or ambiguous), include scene
    detection guidance. Once scene is known, include scene-specific
    extraction rules for more targeted field extraction.
    """
    prompt = _BASE_PROMPT

    if scene == "family":
        prompt += _FAMILY_BLOCK
    elif scene == "friends":
        prompt += _FRIENDS_BLOCK
    else:
        prompt += _NO_SCENE_BLOCK

    return prompt


# ═══════════════════════════════════════════════════════════════════════
# Node 1: clarify_intent
# ═══════════════════════════════════════════════════════════════════════

MAX_CLARIFY_ITERATIONS = 6


async def clarify_intent(state: AgentState) -> dict:
    """Node 1: Extract intent with dynamic prompt + incremental updates.

    Routes via Command(goto=...):
    - "plan" → context_agent (fetches weather for plan_date, then → poi_search)
    - "clarify" → END (return follow-up question to user)
    - "reject" → reject node
    """
    # Get previous extract for multi-turn continuity
    previous = state.get("extract_result")

    # Detect scene for dynamic prompt assembly
    current_scene = None
    if previous and previous.intent.scene:
        current_scene = previous.intent.scene.value

    system_prompt = _build_clarify_prompt(current_scene)

    # Build conversation history
    all_msgs = state["messages"]
    recent = all_msgs[-6:]
    history = "\n".join(
        f"{getattr(m, 'role', '')}: {getattr(m, 'content', str(m))}"
        for m in recent
    )

    # Include previous extract for multi-turn continuity
    # Only dump fields that have meaningful values (non-None, non-empty, non-default)
    existing_json = ""
    if previous and previous.intent.activity_summary:
        minimal = previous.model_dump(exclude_none=True, exclude_defaults=True)
        # Also strip empty lists/dicts to reduce noise
        minimal = _strip_empty_containers(minimal)
        existing_json = (
            "\n\nPREVIOUS EXTRACT (carry forward unless user explicitly changes):\n"
            + json.dumps(minimal, indent=2, ensure_ascii=False)
        )

    # Include weather context if available
    weather_ctx = ""
    w = state.get("weather")
    if w and w.condition:
        weather_ctx = (
            f"\n\n[天气]\n"
            f"日期: {w.date} | {w.condition} | "
            f"{f'{w.temp_low:.0f}~{w.temp_high:.0f}°C' if w.temp_low and w.temp_high else ''}"
            f"{' | 建议室内活动' if w.indoor_recommended else ''}"
        )

    user_prompt = f"Conversation:\n{history}{existing_json}{weather_ctx}"

    parse_error = False
    try:
        llm = make_model(temperature=0.3)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        text = await llm_invoke(llm, messages, "clarify_intent", stream=True)
        data = _extract_json(text)
        route = data.pop("route", "reject")

        # Build update from LLM output
        update = _parse_extract_update(data)
    except Exception as exc:
        logger.warning("clarify_intent failed: %s — falling back to clarify", exc)
        parse_error = True
        route = "clarify"
        update = UpdateExtractResultInput()

    # Apply incremental update
    if previous and previous.intent.activity_summary:
        # Multi-turn: merge into existing
        extract = previous
        updated_fields = extract.apply_update(update)
        if updated_fields:
            logger.debug("Extract updated: %s", updated_fields)
    else:
        # First turn: build from scratch
        extract = _build_extract_from_update(update)

    # Clarify iteration tracking
    clarify_iterations = state.get("clarify_iterations", 0)
    if route == "clarify":
        clarify_iterations += 1

    _log_extract(extract, route, clarify_iterations)

    # Fallback for parse errors
    if parse_error:
        return Command(
            goto=END,
            update={
                "extract_result": extract,
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
        route = "plan" if extract.intent.activity_summary else "reject"

    # Route via Command
    if route == "plan":
        return Command(
            goto="context_agent",
            update={
                "extract_result": extract,
                "plan_iterations": 0,
                "clarify_iterations": 0,
            },
        )
    elif route == "clarify":
        return Command(
            goto=END,
            update={
                "extract_result": extract,
                "plan_iterations": 0,
                "clarify_iterations": clarify_iterations,
            },
        )
    else:
        return Command(
            goto="reject",
            update={
                "extract_result": extract,
                "plan_iterations": 0,
                "clarify_iterations": 0,
            },
        )


def _parse_extract_update(data: dict) -> UpdateExtractResultInput:
    """Parse LLM JSON output into an UpdateExtractResultInput.

    Handles both full output (all sub-models) and partial output gracefully.
    """
    from finn.state import (
        UserIntent, UserRequirements, HardConstraints, SoftConstraints,
        GroupProfile, TimeWindow, GeoConstraint, ChainTemplate, ConstraintProfile,
    )

    def _safe_parse(model_cls, key: str):
        if key in data and data[key] is not None:
            try:
                return model_cls(**data[key])
            except Exception:
                logger.debug("Failed to parse %s from: %s", key, data.get(key))
        return None

    return UpdateExtractResultInput(
        intent=_safe_parse(UserIntent, "intent"),
        requirements=_safe_parse(UserRequirements, "requirements"),
        hard_constraints=_safe_parse(HardConstraints, "hard_constraints"),
        soft_constraints=_safe_parse(SoftConstraints, "soft_constraints"),
        group=_safe_parse(GroupProfile, "group"),
        time=_safe_parse(TimeWindow, "time"),
        geo=_safe_parse(GeoConstraint, "geo"),
        chain=_safe_parse(ChainTemplate, "chain"),
        constraint_profile=_safe_parse(ConstraintProfile, "constraint_profile"),
        confidence=data.get("confidence"),
        follow_up_question=data.get("follow_up_question"),
    )


def _build_extract_from_update(update: UpdateExtractResultInput) -> ExtractResult:
    """Build a fresh ExtractResult from an UpdateExtractResultInput."""
    extract = ExtractResult()
    extract.apply_update(update)
    return extract


def _log_extract(extract: ExtractResult, route: str, clarify_n: int = 0) -> None:
    """Log extract result for debugging."""
    i = extract.intent
    info_parts = [
        f"activity={i.activity_summary!r}",
        f"city={i.city!r}",
        f"date={i.plan_date!r}",
    ]
    if clarify_n:
        info_parts.append(f"clarify={clarify_n}/{MAX_CLARIFY_ITERATIONS}")
    if i.scene:
        info_parts.append(f"scene={i.scene.value}")
    if i.guest_count:
        info_parts.append(f"guests={i.guest_count}")
    if extract.group.tags:
        info_parts.append(f"tags={extract.group.tags}")
    if extract.hard_constraints.dietary_restrictions:
        info_parts.append(f"diet={extract.hard_constraints.dietary_restrictions}")
    if extract.soft_constraints.preferred_cuisines:
        info_parts.append(f"cuisine={extract.soft_constraints.preferred_cuisines}")
    if extract.soft_constraints.preferred_poi_types:
        info_parts.append(f"poi_types={extract.soft_constraints.preferred_poi_types}")
    info_parts.append(f"confidence={extract.confidence:.2f}")
    info_parts.append(f"→ {route}")
    logger.info("Extract | %s", " | ".join(info_parts))
    logger.debug("Extract JSON | %s",
                 extract.model_dump_json(indent=2, exclude_none=True))


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
    '• "周末想带家人去南山爬山"\n'
    '• "帮我订周六晚上朝阳区的火锅"\n'
    '• "明天下午3点和朋友想看电影"\n'
    "试试看？"
)


async def reject(state: AgentState) -> dict:
    """Node 2c: Inform user of capability boundaries."""
    return {
        "messages": [{"role": "assistant", "content": REJECT_MESSAGE}],
        "next_action": "done",
    }


# ═══════════════════════════════════════════════════════════════════════
# Node: poi_search  ★ Weak Agent
# ═══════════════════════════════════════════════════════════════════════


async def poi_search(state: AgentState) -> dict:
    """Weak-agent POI search: strategy selection + direct MCP tool calls.

    Flow:
    1. Read extract_result from state
    2. Re-fetch weather if date mismatches plan_date
    3. Select search strategy (rule-based from tags, weather-aware)
    4. Check cache → return cached if hit
    5. Geocode center address via MCP
    6. Call maps_around_search + maps_text_search via MCP
    7. Parse, deduplicate, rank results
    8. Cache → return POISearchResult
    """
    extract = state.get("extract_result")
    if extract is None:
        logger.warning("poi_search called without extract_result")
        return {"poi_candidates": []}

    # Populate center_location from IP-based user coords if LLM didn't set it.
    # This avoids ambiguous text geocoding (e.g. "西南大学" → 荣昌 vs 北碚).
    if not extract.geo.center_location and state.get("user_coords"):
        extract.geo.center_location = state["user_coords"]
        logger.debug("Using IP coords as center: %s", extract.geo.center_location)

    # Build strategy (weather-aware: injects indoor keywords when needed)
    weather = state.get("weather")
    strategy = select_search_strategy(extract, weather)

    # Check cache
    city = extract.intent.city
    cache_key = make_cache_key(strategy, city)
    cached = poi_cache.get(cache_key)
    if cached is not None:
        logger.info("POI search cache HIT — %d candidates", len(cached.candidates))
        return {
            "poi_candidates": cached.candidates,
            "messages": [{
                "role": "system",
                "content": f"[POI search: {len(cached.candidates)} candidates from cache]",
            }],
        }

    # Execute search via MCP
    t0 = time.monotonic()
    candidates: list[POICandidate] = []
    distance_matrix: dict[str, int] = {}

    try:
        async with mcp_session() as (tools, server_id):
            candidates = await _execute_poi_search(tools, extract, strategy)

            # Dedup and rank (inside session so distance matrix works)
            candidates = deduplicate_pois(candidates)
            candidates = rank_pois(candidates, extract)

            # ── Per-category detail fetch ──
            # Instead of top-15 overall (which can starve some categories),
            # fetch the best N from each category so every slot type has
            # POIs with location/price/hours data for the planner.
            from finn.hub import mcp_hub
            from finn.planner import _classify_slot

            # Categorize all candidates
            cat_buckets: dict[str, list[POICandidate]] = {
                "play": [], "eat": [], "follow_up": []
            }
            for c in candidates:
                slot = _classify_slot(c)
                if slot in cat_buckets:
                    cat_buckets[slot].append(c)

            # Per-category quotas (total ~15, adjusting for category availability)
            cat_quotas = {"play": 10, "eat": 8, "follow_up": 5}  # ~23 POIs, ~7s detail fetch
            detail_targets: list[POICandidate] = []
            for cat, quota in cat_quotas.items():
                detail_targets.extend(cat_buckets[cat][:quota])
            # Fill remaining to 23 with any unselected high-ranked candidates
            selected_ids = {c.id for c in detail_targets}
            for c in candidates:
                if len(detail_targets) >= 23:
                    break
                if c.id not in selected_ids:
                    detail_targets.append(c)
                    selected_ids.add(c.id)

            if detail_targets:
                detail_ids = [c.id for c in detail_targets if c.id]
                details = await mcp_hub.batch_search_detail(tools, detail_ids)
                loc_count = 0
                price_count = 0
                hours_count = 0
                for poi in detail_targets:
                    detail = details.get(poi.id, {})
                    if not detail:
                        continue
                    # Amap search_detail returns either:
                    #   {"pois": [{...}]} — array wrapper
                    #   {"name": ..., "location": ...} — flat object
                    poi_data = detail
                    if "pois" in detail and isinstance(detail["pois"], list):
                        poi_data = detail["pois"][0] if detail["pois"] else detail
                    biz = poi_data.get("biz_ext", {}) if isinstance(poi_data, dict) else {}
                    # ── Location (critical: around_search doesn't return it) ──
                    loc = poi_data.get("location", "") if isinstance(poi_data, dict) else ""
                    if loc and not poi.location:
                        poi.location = str(loc)
                        loc_count += 1
                    # ── Open time (check both top-level and biz_ext) ──
                    open_time_raw = (
                        poi_data.get("open_time", "") or poi_data.get("opentime2", "")
                        or biz.get("opentime", "") or biz.get("open_time", "")
                    ) if isinstance(poi_data, dict) else ""
                    if open_time_raw:
                        poi.open_time, poi.close_time = _parse_opentime(str(open_time_raw))
                        hours_count += 1
                    # ── Price per person (check both top-level and biz_ext) ──
                    cost = biz.get("cost", "") or poi_data.get("cost", "") if isinstance(poi_data, dict) else ""
                    if cost:
                        try:
                            poi.price_per_person = float(str(cost))
                            price_count += 1
                        except (ValueError, TypeError):
                            pass
                    # ── Rating ──
                    rating = (
                        poi_data.get("rating", "") or biz.get("rating", "")
                    ) if isinstance(poi_data, dict) else ""
                    if rating and poi.rating is None:
                        poi.rating = _parse_rating(rating)
                    # Set stay parameters by POI type
                    _set_stay_params(poi)

                logger.debug("Detail fetch: %d/%d resolved (loc=%d price=%d hours=%d) "
                             "categories play=%d eat=%d follow_up=%d",
                             len(details), len(detail_ids), loc_count, price_count, hours_count,
                             len(cat_buckets["play"]), len(cat_buckets["eat"]),
                             len(cat_buckets["follow_up"]))

                # Compute constraint match scores
                compute_match_scores(candidates, extract)

                # Apply type-based stay duration parameters (replaces fixed 120min default)
                from finn.planner import _apply_stay_params
                for poi in candidates:
                    _apply_stay_params(poi)

                # Boost indoor POIs if weather suggests indoor
                if state.get("weather") and state["weather"].indoor_recommended:
                    # Stronger boost: indoor-friendly POIs get significant preference
                    _INDOOR_KW = {"商场", "博物馆", "电影院", "美术馆", "餐厅",
                                  "咖啡馆", "茶馆", "书店", "购物", "火锅",
                                  "烧烤", "日料", "素食", "粤菜", "面馆",
                                  "小吃", "快餐", "中餐", "火锅", "串串"}
                    _OUTDOOR_KW = {"公园", "景区", "广场", "步行街", "动物园",
                                   "游乐", "水上", "温泉", "徒步", "登山",
                                   "植物园", "花卉", "滨江", "沙滩", "露营"}
                    for poi in candidates:
                        is_indoor = any(
                            kw in poi.name or kw in (poi.type or "")
                            or any(kw in t for t in poi.tags)
                            for kw in _INDOOR_KW
                        )
                        is_outdoor = any(
                            kw in poi.name or kw in (poi.type or "")
                            or any(kw in t for t in poi.tags)
                            for kw in _OUTDOOR_KW
                        )
                        if is_indoor and not is_outdoor:
                            poi.match_score = min(1.0, poi.match_score + 0.3)
                        elif is_outdoor:
                            poi.match_score = max(0.0, poi.match_score - 0.15)
                            poi.macro_category = "户外"  # tag for downstream

                logger.debug("Match scores: top-5=%s",
                             [(c.name, round(c.match_score, 2)) for c in candidates[:5]])

            # Build distance matrix — all POIs with location data
            # (per-category detail fetch means POIs beyond top-15 may have coords)

            center_loc = extract.geo.center_location or ""
            located = [c for c in candidates if c.location]
            if center_loc and located:
                # Center → each POI
                pairs = [(center_loc, c.location) for c in located]
                # All-pairs between located POIs for transit estimates
                for i in range(len(located)):
                    for j in range(i + 1, len(located)):
                        pairs.append((located[i].location, located[j].location))
                if pairs:
                    # Use user's preferred transport for distance estimation
                    transport_mode = "0"  # Default: driving
                    user_transport = extract.soft_constraints.preferred_transport
                    if user_transport:
                        transport_mode = {
                            "drive": "0", "walk": "1", "transit": "2",
                        }.get(user_transport.value if hasattr(user_transport, 'value') else str(user_transport), "0")
                    distance_matrix = await mcp_hub.batch_distance(tools, pairs, mode=transport_mode)
                    logger.debug("Distance matrix: %d pairs resolved (mode=%s)",
                                 len(distance_matrix), transport_mode)
    except Exception as exc:
        logger.error("POI search failed: %s", exc)
        return {
            "poi_candidates": [],
            "distance_matrix": {},
            "messages": [{
                "role": "system",
                "content": f"[POI search failed: {exc}. Will search during planning.]",
            }],
        }

    elapsed = time.monotonic() - t0
    logger.info("POI search: %d candidates in %.1fs (strategy=%s, radius=%dm, dist_matrix=%d)",
                len(candidates), elapsed, strategy.keywords[:3], strategy.radius_m,
                len(distance_matrix))

    # Cache result
    result = POISearchResult(
        candidates=candidates,
        cache_key=cache_key,
        strategy_used=str(strategy.keywords[:5]),
        from_cache=False,
    )
    poi_cache.set(cache_key, result)

    return {
        "poi_candidates": candidates,
        "distance_matrix": distance_matrix,
        "messages": [{
            "role": "system",
            "content": f"[POI search: {len(candidates)} candidates found]",
        }],
    }


async def _execute_poi_search(
    tools: list,
    extract: ExtractResult,
    strategy,
) -> list[POICandidate]:
    """Execute POI search via MCPHub batch methods (no LLM).

    1. Geocode center address → lng,lat (cached)
    2. Batch around_search with keywords → POI list (cached)
    3. Batch detail + distance matrix for top candidates
    """
    from finn.hub import mcp_hub

    candidates: list[POICandidate] = []

    # Find relevant tools
    geo_tool = next((t for t in tools if t.name == "maps_geo"), None)
    around_tool = next((t for t in tools if t.name == "maps_around_search"), None)
    text_tool = next((t for t in tools if t.name == "maps_text_search"), None)

    if not around_tool:
        logger.warning("maps_around_search tool not available")
        return candidates

    # Geocode center address to get coordinates.
    # Use geo.center_location (may already be set by poi_search from
    # state.user_coords — IP-based, avoids ambiguous text geocoding).
    center_loc = extract.geo.center_location or ""
    center_addr = extract.geo.center_address or extract.intent.city or ""

    # Prepend city to avoid ambiguous geocoding
    geo_city = extract.intent.city or ""
    if center_addr and geo_city and geo_city not in center_addr:
        center_addr = f"{geo_city}{center_addr}"

    if not center_loc and geo_tool and center_addr:
        try:
            result = await geo_tool.ainvoke({"address": center_addr})
            result_str = str(result)
            geo_data = _extract_json(result_str) if "{" in result_str else None
            if geo_data and "location" in geo_data:
                center_loc = geo_data["location"]
            elif geo_data and "geocodes" in geo_data:
                geocodes = geo_data["geocodes"]
                if geocodes:
                    center_loc = geocodes[0].get("location", "")
            else:
                match = re.search(r"(\d+\.\d+),\s*(\d+\.\d+)", result_str)
                if match:
                    center_loc = f"{match.group(1)},{match.group(2)}"
            if center_loc:
                extract.geo.center_location = center_loc
                logger.debug("Geocoded '%s' → %s", center_addr, center_loc)
        except Exception as exc:
            logger.warning("Geocode failed for '%s': %s", center_addr, exc)

    # ── Parallel: around search + text search ──
    async def _around_search():
        if center_loc and around_tool:
            search_kws = strategy.keywords[:5]
            batch_results = await mcp_hub.batch_around_search(
                tools, center_loc, search_kws, strategy.radius_m,
            )
            for kw, raw_pois in batch_results:
                for raw in raw_pois:
                    poi = _raw_to_candidate(raw, kw)
                    if poi:
                        candidates.append(poi)
                logger.debug("Parsed %d POIs from keyword '%s'", len(raw_pois), kw)

    async def _text_searches():
        if text_tool:
            async def _text_search(keyword: str):
                try:
                    result = await text_tool.ainvoke({
                        "keywords": keyword,
                        "city": geo_city,
                    })
                    return _parse_amap_pois(str(result), keyword)
                except Exception as exc:
                    logger.debug("Text search failed for '%s': %s", keyword, exc)
                    return []

            # Ensure must-have cuisines are always searched via text search
            text_kws = list(strategy.keywords[:3])
            must_cuisines = extract.requirements.must_have_cuisine
            for c in must_cuisines:
                if c and c not in text_kws:
                    text_kws.append(c)

            tasks = [_text_search(kw) for kw in text_kws]
            results = await asyncio.gather(*tasks)
            for parsed in results:
                candidates.extend(parsed)

    await asyncio.gather(_around_search(), _text_searches())

    return candidates


def _raw_to_candidate(raw: dict, source_keyword: str) -> POICandidate | None:
    """Convert a raw Amap POI dict to a POICandidate."""
    pid = raw.get("id", "")
    name = raw.get("name", "")
    if not pid or not name:
        return None
    loc = raw.get("location", "")
    # location may be a string or missing; handle both
    if not loc:
        # Parse from longitude/latitude if available
        lng = raw.get("longitude") or ""
        lat = raw.get("latitude") or ""
        if lng and lat:
            loc = f"{lng},{lat}"
    typecode = raw.get("type", "") or raw.get("typecode", "")
    tags = [source_keyword]
    return POICandidate(
        id=pid,
        name=name,
        address=raw.get("address", ""),
        location=str(loc) if loc else "",
        type=typecode,
        rating=None,
        distance_m=None,
        price_level=None,
        tags=tags,
        source=source_keyword,
        macro_category=_compute_macro_category(typecode, name, tags),
    )


def _parse_amap_pois(response_text: str, source_keyword: str) -> list[POICandidate]:
    """Parse Amap POI search response into POICandidate list.

    Handles both JSON and plain-text response formats.
    """
    candidates: list[POICandidate] = []

    # Try JSON parse first
    try:
        data = _extract_json(response_text)
    except ValueError:
        # Non-JSON response; skip parsing
        return candidates

    # Amap API response format: {"pois": [...]} or {"suggestion": ..., "pois": [...]}
    pois = data.get("pois", [])
    if not pois:
        return candidates

    for p in pois:
        if not isinstance(p, dict):
            continue
        loc = p.get("location", "")
        if not loc:
            # Some formats split lng/lat
            lng = p.get("longitude", "")
            lat = p.get("latitude", "")
            if lng and lat:
                loc = f"{lng},{lat}"

        pid = str(p.get("id", ""))
        pname = str(p.get("name", ""))
        ptype = str(p.get("typecode", p.get("type", "")))
        ptags = [source_keyword]
        candidates.append(POICandidate(
            id=pid,
            name=pname,
            address=str(p.get("address", "")),
            location=loc,
            type=ptype,
            rating=_parse_rating(p.get("biz_ext", {}).get("rating", p.get("rating"))),
            distance_m=_parse_int(p.get("distance")),
            price_level=None,
            tags=ptags,
            source=f"amap_{source_keyword}",
            macro_category=_compute_macro_category(ptype, pname, ptags),
        ))

    logger.debug("Parsed %d POIs from keyword '%s'", len(candidates), source_keyword)
    return candidates


def _parse_rating(value) -> float | None:
    """Safely parse a rating value to float."""
    if value is None:
        return None
    try:
        r = float(str(value))
        return r if r > 0 else None
    except (ValueError, TypeError):
        return None


def _parse_int(value) -> int | None:
    """Safely parse an int value."""
    if value is None:
        return None
    try:
        return int(float(str(value)))
    except (ValueError, TypeError):
        return None


# ── Macro category mapping ──

# Amap typecode prefixes → macro category
_MACRO_CATEGORY_MAP: dict[str, str] = {
    "05": "餐饮",
    "050": "餐饮", "0501": "餐饮", "0502": "餐饮", "0503": "餐饮", "0504": "餐饮",
    "06": "购物",
    "060": "购物", "0601": "购物", "0602": "购物", "0603": "购物", "0604": "购物", "0605": "购物",
    "07": "休闲娱乐",
    "070": "休闲娱乐", "0701": "休闲娱乐", "0702": "休闲娱乐", "0703": "休闲娱乐",
    "0704": "休闲娱乐", "0705": "休闲娱乐",
    "08": "景点",
    "080": "景点", "0801": "景点", "0802": "景点", "0803": "景点", "0804": "景点",
    "09": "景点",
    "090": "景点", "0901": "景点",
    "10": "购物",
    "11": "文化",
    "110": "文化", "1101": "文化", "1102": "文化",
    "12": "文化",
    "120": "文化", "1201": "文化", "1202": "文化",
    "13": "运动",
    "130": "运动", "1301": "运动", "1302": "运动",
    "14": "休闲娱乐",
    "140": "休闲娱乐", "1401": "休闲娱乐", "1402": "休闲娱乐", "1403": "休闲娱乐",
}

# Keyword → macro category overrides
_MACRO_KEYWORD_OVERRIDES: list[tuple[list[str], str]] = [
    (["火锅", "川菜", "粤菜", "面馆", "烧烤", "日料", "西餐", "自助", "海鲜",
      "咖啡", "奶茶", "甜点", "烘焙", "小吃", "快餐", "素食", "清真",
      "餐厅", "食堂", "大排档", "酒楼", "饭馆", "料理", "牛排",
      "茶餐厅", "早茶", "点心", "居酒屋", "串串", "麻辣烫"], "餐饮"),
    (["博物馆", "美术馆", "展览", "画廊", "图书馆", "书店", "科技馆",
      "天文馆", "纪念馆", "故居", "教堂", "寺庙", "道观", "清真寺"], "文化"),
    (["公园", "动物园", "植物园", "水族馆", "游乐园", "主题乐园",
      "风景", "山", "湖", "海滩", "古镇", "园林", "花", "长城"], "景点"),
    (["商场", "购物中心", "步行街", "市场", "超市", "便利店",
      "免税", "奥莱", "集市", "百货"], "购物"),
    (["电影院", "KTV", "网吧", "桌游", "密室", "剧本杀", "酒吧",
      "剧场", "音乐厅", "演出", "相声", "脱口秀",
      "温泉", "按摩", "足疗", "水疗"], "休闲娱乐"),
    (["游泳", "健身", "瑜伽", "篮球", "足球", "羽毛球", "乒乓球",
      "滑雪", "滑冰", "攀岩", "射箭", "卡丁车", "蹦床", "骑行"], "运动"),
]


def _compute_macro_category(typecode: str, name: str, tags: list[str]) -> str:
    """Determine macro category from Amap typecode + name + keyword tags.

    Returns one of: "餐饮" | "景点" | "文化" | "购物" | "休闲娱乐" | "运动" | ""
    """
    name_lower = name.lower() if name else ""
    tags_text = " ".join(tags).lower() if tags else ""

    # 1. Keyword overrides (highest precision)
    for keywords, category in _MACRO_KEYWORD_OVERRIDES:
        for kw in keywords:
            if kw in name_lower or kw in tags_text:
                return category

    # 2. Typecode prefix matching
    tc = str(typecode) if typecode else ""
    for prefix_len in (4, 3, 2):
        prefix = tc[:prefix_len]
        if prefix in _MACRO_CATEGORY_MAP:
            return _MACRO_CATEGORY_MAP[prefix]

    # 3. Fallback
    return ""


def _set_stay_params(poi: POICandidate) -> None:
    """Set stay_min/base/max and elastic_coef by POI type."""
    is_eat = any(
        kw in (poi.type or "").lower() or kw in poi.name.lower()
        for kw in ["餐饮", "餐厅", "火锅", "面馆", "烧烤", "轻食", "沙拉",
                    "咖啡馆", "茶馆", "小吃", "快餐", "中餐厅", "快餐厅"]
    )
    is_movie = "电影院" in poi.name or "影院" in poi.name
    is_shopping = "商场" in poi.name or "购物" in (poi.type or "")

    if is_movie:
        poi.stay_min = poi.stay_base = poi.stay_max = 120
        poi.elastic_coef = 0.0
    elif is_eat:
        poi.stay_min = 45
        poi.stay_base = 75
        poi.stay_max = 120
        poi.elastic_coef = 0.8
    elif is_shopping:
        poi.stay_min = 60
        poi.stay_base = 90
        poi.stay_max = 150
        poi.elastic_coef = 1.2
    else:  # play / follow_up
        poi.stay_min = 60
        poi.stay_base = 120
        poi.stay_max = 240
        poi.elastic_coef = 1.2


def _parse_opentime(opentime_str: str) -> tuple[str | None, str | None]:
    """Parse Amap opentime string like '09:00-22:00' or '24小时营业' into (open, close)."""
    if not opentime_str:
        return None, None
    s = opentime_str.replace("：", ":").strip()
    # Handle 24h cases
    if any(w in s for w in ("24小时", "全天", "24h")):
        return "00:00", "24:00"
    parts = s.split("-")
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return None, None


def _violates_hard_constraint(poi: POICandidate, extract: ExtractResult) -> bool:
    """Check if POI violates any hard constraint."""
    hc = extract.hard_constraints
    name_lower = poi.name.lower()
    tags_lower = [t.lower() for t in poi.tags]

    # Dietary restrictions
    if hc.dietary_restrictions:
        for diet in hc.dietary_restrictions:
            diet_lower = diet.lower()
            if "清真" in diet_lower or "halal" in diet_lower:
                halal_kw = {"清真", "西北", "新疆", "halal", "muslim"}
                if not any(kw.lower() in name_lower or
                          any(kw.lower() in t for t in tags_lower)
                          for kw in halal_kw):
                    return True
            if "素食" in diet_lower or "vegan" in diet_lower:
                veg_kw = {"素食", "蔬食", "vegan", "vegetarian"}
                if not any(kw.lower() in name_lower or
                          any(kw.lower() in t for t in tags_lower)
                          for kw in veg_kw):
                    return True

    # Child safety — bars/nightclubs with young children
    if hc.child_age is not None and hc.child_age < 12:
        unsafe = {"酒吧", "夜店", "ktv", "网吧", "bar", "club", "night"}
        if any(kw.lower() in name_lower or kw.lower() in (poi.type or "").lower()
               for kw in unsafe):
            return True

    return False


def compute_match_scores(
    candidates: list[POICandidate], extract: ExtractResult,
) -> None:
    """Mutate candidates in-place, setting match_score (0-1).

    Score components:
    - Hard constraint violation → 0
    - Cuisine match → up to 0.25
    - Activity type match → up to 0.20
    - Rating → up to 0.15
    - Price fit → up to 0.20
    - Group tag match → up to 0.10
    - Weather awareness → up to 0.10
    """
    req = extract.requirements
    sc = extract.soft_constraints
    hc = extract.hard_constraints
    gp = extract.group

    preferred_cuisines = [c.lower() for c in (
        req.must_have_cuisine + sc.preferred_cuisines
    )]
    preferred_types = [t.lower() for t in (
        req.must_have_activity_type + sc.preferred_poi_types
        + extract.chain.preferred_activity_types
    )]
    avoid_types = set(t.lower() for t in sc.avoid_poi_types)
    budget_pref = sc.budget_preference.value if sc.budget_preference else "mid"

    # Scene-based fallback: when user has no explicit preferences,
    # use scene defaults so match_score differentiates relevant POIs
    _SCENE_FALLBACK_TAGS: dict = {
        "family": {"公园", "博物馆", "亲子", "动物园", "游乐", "儿童", "景点"},
        "friends": {"网红", "火锅", "烧烤", "咖啡馆", "打卡", "酒吧"},
    }
    scene = extract.group.type.value if extract.group.type else "family"
    fallback_types = _SCENE_FALLBACK_TAGS.get(scene, set())
    use_fallback = not preferred_cuisines and not preferred_types
    if use_fallback:
        preferred_cuisines = list(fallback_types)  # treat as cuisine+type keywords
        preferred_types = list(fallback_types)

    for poi in candidates:
        # Hard constraint check first
        if _violates_hard_constraint(poi, extract):
            poi.match_score = 0.0
            continue

        score = 0.0
        name_lower = poi.name.lower()
        tags_lower = [t.lower() for t in poi.tags]
        type_lower = (poi.type or "").lower()

        # Cuisine match (0.25)
        if preferred_cuisines:
            hits = 0
            for c in preferred_cuisines:
                if c in name_lower or any(c in t for t in tags_lower):
                    hits += 1
            score += 0.25 * min(hits / max(len(preferred_cuisines), 1), 1.0)

        # Activity type match (0.20)
        if preferred_types:
            hits = 0
            for pt in preferred_types:
                if pt in name_lower or pt in type_lower or any(pt in t for t in tags_lower):
                    hits += 1
            score += 0.20 * min(hits / max(len(preferred_types), 1), 1.0)
        else:
            score += 0.10  # No type preference → neutral

        # Rating (0.15)
        if poi.rating is not None and poi.rating > 0:
            score += 0.15 * min(poi.rating / 5.0, 1.0)
        else:
            score += 0.075  # Unknown rating

        # Price fit (0.20)
        if poi.price_per_person is not None:
            price = poi.price_per_person
            ranges = {"economy": (0, 50), "mid": (30, 150), "luxury": (100, 9999)}
            lo, hi = ranges.get(budget_pref, (30, 150))
            if lo <= price <= hi:
                score += 0.20
            elif price < lo:
                score += 0.10  # Cheaper than expected
            else:
                score += 0.05  # More expensive
        else:
            score += 0.10  # Unknown price

        # Group tag match (0.10)
        group_tags = set(t.lower() for t in gp.tags)
        if group_tags:
            hits = sum(1 for gt in group_tags
                      if gt in name_lower or any(gt in t for t in tags_lower))
            score += 0.10 * min(hits / max(len(group_tags), 1), 1.0)

        # Weather awareness (0.10)
        weather = extract  # weather is in state, not extract — handled in poi_search caller
        # Indoor preference bonus handled by caller; set neutral here
        score += 0.05

        # Avoid type penalty
        if avoid_types:
            if any(a in name_lower or a in type_lower or any(a in t for t in tags_lower)
                   for a in avoid_types):
                score -= 0.15

        # Location availability penalty — POIs without coordinate data
        # cannot be properly routed (transit time, nav URLs, etc.)
        if not poi.location:
            score -= 0.30

        poi.match_score = max(0.0, min(1.0, score))


# ═══════════════════════════════════════════════════════════════════════
# Node 2b: decompose_and_plan
# ═══════════════════════════════════════════════════════════════════════

DECOMPOSE_PROMPT = """\
你是 Finn 的任务分解与规划模块。

你会收到用户出行意图和 POI 候选列表。你需要基于真实数据构造一份可执行的出行计划。

如果是修改已有计划：
- 以已有计划为基础，按用户修改请求调整
- 保留用户未要求修改的所有内容

══════════════════════════════════════
核心规则：POI 候选优先，工具调用最小化
══════════════════════════════════════

1. **首先使用预搜索的 POI candidates。** 它们的坐标、评分、地址已经就绪，
   直接用 maps_search_detail 查详情即可，不要再从头搜索。
2. 只有在 candidates 确实不满足需求时（类型不匹配/数量不足），才调用
   maps_around_search 或 maps_text_search 补充搜索，且最多 2 个关键词。
3. 同类搜索空结果立即换思路，不要穷举变体。
4. 找到 4-6 个合适候选就停止搜索，开始构建 plan。
5. 总共工具调用控制在 15 次以内。

══════════════════════════════════════
可用工具
══════════════════════════════════════

maps_weather — 查询天气
maps_search_detail — 查询 POI 详情（营业时间、电话、人均消费）
maps_around_search — 在指定坐标周边搜索 POI
maps_text_search — 关键词搜索特定类型商家
maps_geo — 将地名转为经纬度坐标
maps_distance — 计算两点直线距离
maps_direction_walking — 步行路线和耗时
maps_direction_driving — 驾车路线和耗时

工具流程：
1. maps_weather → 天气
2. maps_search_detail → 从 candidates 中挑选感兴趣的查详情（坐标已就绪）
3. maps_distance / maps_direction_* → 计算场所间路线
4. 仅在 candidates 不够用时 maps_around_search / maps_text_search

══════════════════════════════════════
输出出行计划 JSON
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


# ═══════════════════════════════════════════════════════════════════════
# Node: planner_agent ★ Zero-LLM algorithm-based (new)
# ═══════════════════════════════════════════════════════════════════════


def _pre_check_plan(plan: Plan, extract: ExtractResult) -> list[str]:
    """Deterministic pre-checks on the algorithm-built plan.

    Catches obvious violations before the LLM verification step,
    reducing the number of verify-and-adjust iterations.
    """
    issues: list[str] = []
    must_cuisine = [c.lower() for c in extract.requirements.must_have_cuisine]

    for st in plan.sub_tasks:
        if st.type != "book":
            continue
        params = st.params or {}
        slot = str(params.get("slot", ""))
        name = str(params.get("name", ""))
        name_lower = name.lower()

        # Dinner cuisine match
        if slot == "dinner" and must_cuisine:
            if not any(c in name_lower for c in must_cuisine):
                issues.append(f"晚餐'{name}'不满足必吃菜系: {must_cuisine}")

        # Dinner time window (17:00+)
        if slot == "dinner":
            start = str(params.get("start_time", ""))
            if start and start < "17:00":
                issues.append(f"晚餐'{name}'开始时间{start}不在晚餐时段(17:00后)")

        # POI must have coordinates
        if not params.get("location"):
            issues.append(f"场所'{name}'缺少坐标")

    return issues


async def planner_agent(state: AgentState) -> dict:
    """Algorithm-based plan builder — 3 parallel strategies, no LLM.

    Reads extract_result, poi_candidates, distance_matrix from state.
    Runs 3 strategies via asyncio.gather, scores, picks best.
    Converts winning PlannedPath → Plan.
    Two-tier transport: refined direction API for the selected path.
    """
    from finn.planner import build_paths, build_plan, _refine_transport

    extract = state.get("extract_result")
    candidates = state.get("poi_candidates", [])
    dist_matrix = state.get("distance_matrix", {})

    if not extract:
        logger.warning("planner_agent called without extract_result")
        return {"plan": None, "next_action": "reject"}

    # Run 3 parallel strategies (weather hard-filters outdoor POIs when applicable)
    weather = state.get("weather")
    paths = await build_paths(extract, list(candidates), dist_matrix, weather=weather)

    if not paths:
        logger.warning("Planner: all strategies returned empty paths")
        return {
            "plan": Plan(sub_tasks=[], notes="未能找到合适的路线，请尝试放宽搜索条件。"),
            "planned_paths": [],
            "selected_path": None,
            "messages": [{"role": "system", "content": "[Plan: no paths generated]"}],
        }

    # Select best path
    selected = paths[0]
    logger.info("Planner: selected %s (score=%.3f, %d nodes, transit=%d min, cost=¥%.0f)",
                selected.strategy, selected.coverage_score,
                len(selected.nodes), selected.total_transit_min, selected.total_cost)

    # Two-tier transport: refine transit for selected path
    try:
        from finn.mcp import mcp_session
        async with mcp_session() as (tools, _):
            await _refine_transport(selected, tools)
    except Exception as exc:
        logger.debug("Transport refinement skipped: %s", exc)

    # Convert to Plan
    plan = build_plan(selected)
    logger.info("Planner: %d sub-tasks, cost=¥%s, notes=%s",
                len(plan.sub_tasks), plan.total_cost_estimate, (plan.notes or "")[:80])

    # Deterministic pre-check: catch obvious violations before LLM verification
    pre_issues = _pre_check_plan(plan, extract)
    if pre_issues:
        plan.notes = (plan.notes or "") + " | 预检问题: " + "; ".join(pre_issues)
        logger.info("Planner pre-check: %d issues — %s", len(pre_issues), pre_issues)

    return {
        "plan": plan,
        "planned_paths": paths,
        "selected_path": selected,
        "messages": [{
            "role": "system",
            "content": f"[Plan built: {len(plan.sub_tasks)} tasks via {selected.strategy}, "
                       f"{len(paths)} strategies evaluated]",
        }],
    }


# ═══════════════════════════════════════════════════════════════════════
# -- LEGACY (pre-PlannerAgent) --
# decompose_and_plan, verify_plan, adjust_plan are kept for rollback
# via create_graph(use_new_planner=False). No longer used in new path.
# ═══════════════════════════════════════════════════════════════════════

async def decompose_and_plan(state: AgentState) -> dict:
    """[LEGACY] Node 2b: Decompose intent into executable sub-task DAG.

    Uses a ReAct loop with MCP tools to search for real venue data.
    """
    extract = state["extract_result"]
    modify_feedback = state.get("modify_feedback", "")
    existing_plan = state.get("plan")
    poi_candidates = state.get("poi_candidates", [])

    # Build intent description from extract_result
    parts = _build_intent_description(extract)

    # Include POI candidates as pre-researched options
    if poi_candidates:
        poi_lines = [
            "\nPre-researched POI candidates — use these FIRST, "
            "only search for more if none fit:"
        ]
        for c in poi_candidates[:30]:
            rating_str = f" ★{c.rating}" if c.rating else ""
            dist_str = f" {c.distance_m}m" if c.distance_m else ""
            loc_str = f" loc={c.location}" if c.location else ""
            poi_lines.append(
                f"  - [{c.id}] {c.name} | {c.address} | type={c.type}"
                f"{rating_str}{dist_str}{loc_str}"
            )
        parts.append("\n".join(poi_lines))

    lines = ["User intent:"]
    lines.extend(f"  {p}" for p in parts)

    if modify_feedback and existing_plan:
        plan_json = existing_plan.model_dump_json(indent=2, exclude_none=True)
        lines.append("\n*** REVISION REQUEST ***")
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


def _build_intent_description(extract: ExtractResult) -> list[str]:
    """Build user intent description lines from ExtractResult for downstream LLM prompts."""
    i = extract.intent
    parts: list[str] = []

    if i.activity_summary:
        parts.append(f"Activity: {i.activity_summary}")
    if i.city:
        parts.append(f"City: {i.city}")
    if i.plan_date:
        parts.append(f"Date: {i.plan_date}")
    if i.time_window_start:
        parts.append(f"Time: {i.time_window_start}")
    if i.time_window_hours:
        parts.append(f"Duration: ~{i.time_window_hours}h")
    if i.scene:
        scene_labels = {SceneType.FAMILY: "family", SceneType.FRIENDS: "friends"}
        parts.append(f"Scene: {scene_labels.get(i.scene, i.scene.value)}")
    if i.guest_count:
        parts.append(f"Party size: {i.guest_count}")

    # Time window
    t = extract.time
    if t.date:
        parts.append(f"Date: {t.date}")
    if t.start_time:
        parts.append(f"Start: {t.start_time}")
    if t.end_time:
        parts.append(f"End: {t.end_time}")

    # Geo
    g = extract.geo
    if g.center_address:
        parts.append(f"Start location: {g.center_address}")
    if g.radius_m:
        parts.append(f"Max distance: {g.radius_m}m")
    if g.constraint_desc:
        parts.append(f"Distance note: {g.constraint_desc}")

    # Requirements
    req = extract.requirements
    if req.must_visit_pois:
        parts.append(f"Must visit: {', '.join(req.must_visit_pois)}")
    if req.must_have_cuisine:
        parts.append(f"Must have cuisine: {', '.join(req.must_have_cuisine)}")
    if req.must_have_activity_type:
        parts.append(f"Must have activity: {', '.join(req.must_have_activity_type)}")
    if req.special_requests:
        parts.append(f"Special: {', '.join(req.special_requests)}")

    # Hard constraints
    hc = extract.hard_constraints
    if hc.budget_max_cny:
        parts.append(f"Budget max: ¥{hc.budget_max_cny}")
    if hc.dietary_restrictions:
        parts.append(f"Dietary: {', '.join(hc.dietary_restrictions)}")
    if hc.child_age is not None:
        parts.append(f"Child age: {hc.child_age}")
    if hc.accessibility_needed:
        parts.append("Accessibility needed")
    if hc.time_deadline:
        parts.append(f"Deadline: {hc.time_deadline}")

    # Soft constraints
    sc = extract.soft_constraints
    if sc.budget_preference:
        parts.append(f"Budget preference: {sc.budget_preference.value}")
    if sc.travel_pace:
        parts.append(f"Pace: {sc.travel_pace.value}")
    if sc.preferred_poi_types:
        parts.append(f"Preferred POI types: {', '.join(sc.preferred_poi_types)}")
    if sc.preferred_cuisines:
        parts.append(f"Preferred cuisines: {', '.join(sc.preferred_cuisines)}")
    if sc.preferred_transport:
        parts.append(f"Preferred transport: {sc.preferred_transport.value}")
    if sc.avoid_poi_types:
        parts.append(f"Avoid: {', '.join(sc.avoid_poi_types)}")

    # Group profile
    gp = extract.group
    if gp.tags:
        parts.append(f"Group tags: {', '.join(gp.tags)}")
    if gp.hard_constraints:
        parts.append(f"Group constraints: {', '.join(gp.hard_constraints)}")
    if gp.soft_preferences:
        parts.append(f"Group preferences: {', '.join(gp.soft_preferences)}")

    # Chain template
    ct = extract.chain
    if ct.preferred_activity_types:
        parts.append(f"Activity types: {', '.join(ct.preferred_activity_types)}")

    return parts


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

Evaluation dimensions (ONLY flag HARD issues — things that make the plan infeasible):
1. Temporal infeasibility: Do times conflict? Are transit times impossible?
2. Budget violation: Does cost clearly exceed user's stated budget?
3. Missing essentials: Is a must-visit POI or must-have cuisine completely absent?
4. Dependency broken: Do dependencies reference nonexistent tasks?
5. Venue realism: Are venues clearly nonexistent or closed during planned time?

DO NOT flag as issues:
- "未确认"/"未明确" — confirmation details are runtime concerns, not plan defects
- Missing optional preferences (e.g., "未安排徒步") when alternatives were planned
- "建议" improvements — only flag things that are BROKEN, not things that could be better
- Slot constraint violations (cuisine type, weather filtering) —
  the planner enforces these deterministically via type-code filters; trust them

Scoring:
- 0.85-1.0 → status="pass" (minor issues or none)
- 0.5-0.85 → status="fix" (real fixable issues like budget or time conflict)
- 0.0-0.5 → status="reject" (fundamentally broken)

In Chinese. Output ONLY the JSON, no other text."""


# ═══════════════════════════════════════════════════════════════════════
# Node: verify_and_adjust ★ LLM-driven plan verification + repair (new)
# Replaces checker_agent + fault_handler with a single LLM pass.
# Uses POI candidates + distance matrix from poi_search — no new POI search.
# ═══════════════════════════════════════════════════════════════════════

_VERIFY_ADJUST_PROMPT = """\
你是 Finn 的计划验证与修复模块。

你会收到：
1. 用户原始需求（对话历史）
2. 当前生成的出行计划
3. 可用的 POI 候选池（已搜索完成，不可搜索新的）
4. 场所间距离矩阵（用于估算转场时间）

你的任务：逐项检查当前计划是否满足用户需求，发现问题时从候选池中替换 POI 或调整时间安排。

══════════════════════════════════════
检查清单（按优先级）
══════════════════════════════════════

1. **用餐时序**：用户说的"晚上吃X"是否安排在了晚餐时段（17:00后）？
   "中午随意/午饭随便"对应的午餐时段（11:00-13:00）是否安排了合适的简餐？

2. **场所多样性**：同一场所/POI 不得填充多个 book slot。
   例如"解放碑步行街"不能同时作为上午游玩、午餐、下午游玩三个 slot。
   >8h 的行程至少需要 3 个不同的场所，>12h 至少需要 4 个。
   如果出现重复场所填充多个 slot，必须从候选池中选不同的场所替代。

3. **场所数量**：时间窗口是否充分利用？12h至少3-4个场所，8h至少2-3个。
   如果计划中场所太少（如12h只有2个），从候选池添加合适的场所。

4. **预算**：总花费是否在 budget_max_cny 范围内？超出则替换为更经济的选项。

5. **转场时间**：相邻场所间距离是否合理？>2h转场或>30km需要调整。

6. **饮食限制**：dietary_restrictions（清真/素食/过敏）是否被遵守？

7. **儿童安全**：有低龄儿童时是否避免了不安全场所（酒吧/夜店/网吧）？

══════════════════════════════════════
修复策略
══════════════════════════════════════

- 只能从 POI 候选池中选择替代场所，不得编造新的 POI 名称
- 候选池中每个POI有 name/address/location/type/rating
- 距离矩阵给出了场所间的距离（米），用于判断转场是否合理
- 调整 time slot 分配使计划符合用户的时间约束
- 保持 SubTask 结构：search → transit → book，含 dependencies 和 compensatory
- **CRITICAL — 修改计划时必须完整保留每个 book task 的 params 字段**：
  每个 book task 的 params 中必须包含 name, address, location, rating, slot,
  start_time, end_time, cost_estimate, id 等字段。从候选池的原始数据中复制，
  不得省略或截断。这是展示层生成交通信息和导航链接的必要数据。

══════════════════════════════════════
状态判断标准
══════════════════════════════════════

- status="pass"：计划无明显缺陷，场所多样且数量合理，时序正确
- status="fix"：存在可修复的问题（重复场所、时序错位、缺少场所等）
- **重要**：如果场所重复（同一名称出现 2 次以上）或 >8h 行程只有 <3 个不同场所，
  即使候选池不足，也必须 status="fix" 而非 "pass"，并尽最大努力从候选池中换入不同场所

══════════════════════════════════════
输出格式
══════════════════════════════════════

输出一个 JSON 对象：

{
  "score": <0.0-1.0 计划质量评分>,
  "issues_found": ["<发现的问题>", ...],
  "changes_made": ["<所做的修改>", ...],
  "status": "pass" | "fix",
  "plan": {
    "sub_tasks": [
      {
        "id": "t1",
        "type": "search" | "compare" | "book",
        "target": "<任务描述>",
        "dependencies": ["<上一任务id>"],
        "params": {
          "name": "<POI名称>",
          "address": "<详细地址>",
          "location": "<lng,lat>",
          "rating": <float>,
          "slot": "<play|lunch|dinner|eat|follow_up>",
          "start_time": "<HH:MM>",
          "end_time": "<HH:MM>",
          "cost_estimate": <float>,
          "id": "<POI ID>"
        }
      }
    ],
    "total_cost_estimate": <float or null>,
    "notes": "<修改后的中文摘要>"
  }
}

- status="pass": 计划满足用户需求，可以直接展示
- status="fix": 已修复发现的问题
- 如果无法修复（候选池确实没有合适的替代），在 issues_found 中说明，status="pass" 接受当前计划
- IMPORTANT: 修改计划时，每个 book task 的 params 必须完整（含 location/address/rating），
  从原计划或候选池中复制相应字段

输出 ONLY JSON，不要 markdown，不要 ``` 代码块。"""


async def verify_and_adjust(state: AgentState) -> dict:
    """LLM-driven plan verification + adjustment.

    Reviews the plan against user's original request, POI candidate pool,
    and distance matrix. Auto-fixes issues by swapping POIs from candidates.
    Does NOT search for new POIs (no maps_around_search/text_search).
    Can optionally call maps_search_detail for specifics on candidates.

    Replaces the old checker_agent → fault_handler loop with a single LLM pass.
    """
    from finn.llm import make_model, llm_invoke

    plan = state.get("plan")
    extract = state.get("extract_result")
    candidates = state.get("poi_candidates", [])
    dist_matrix = state.get("distance_matrix", {})
    modify_feedback = state.get("modify_feedback", "")
    plan_iterations = state.get("plan_iterations", 0)

    if not plan or not extract:
        return {"verification": Verification(score=0.0, issues=["No plan"], status="reject")}

    # ── Build user context ──
    # Original conversation
    all_msgs = state["messages"]
    user_msgs = [
        f"{getattr(m, 'role', '')}: {getattr(m, 'content', str(m))}"
        for m in all_msgs
        if getattr(m, 'role', '') in ("user", "human")
    ]
    conversation = "\n".join(user_msgs[-4:])  # Last 4 user messages for context

    # Extract summary
    intent_lines = _build_intent_description(extract)
    intent_text = "\n".join(f"  {line}" for line in intent_lines)

    # Plan JSON
    plan_json = plan.model_dump_json(indent=2, ensure_ascii=False)

    # POI candidate pool (top 20)
    candidate_lines = []
    for c in candidates[:20]:
        rating_str = f" ★{c.rating}" if c.rating else ""
        loc_str = f" loc={c.location}" if c.location else ""
        candidate_lines.append(
            f"  - [{c.id}] {c.name} | {c.address} | type={c.type}"
            f"{rating_str}{loc_str}"
        )
    candidate_text = "\n".join(candidate_lines) if candidate_lines else "(无候选)"

    # Distance matrix summary (top pairs)
    dist_lines = []
    for k, v in sorted(dist_matrix.items(), key=lambda x: x[1], reverse=True)[:15]:
        if v > 0:
            km = v / 1000
            dist_lines.append(f"  {k}: {v}m ({km:.1f}km)")
    dist_text = "\n".join(dist_lines) if dist_lines else "(无距离数据)"

    # Planned path timing data
    selected_path = state.get("selected_path")
    path_text = ""
    if selected_path and selected_path.nodes:
        path_lines = [f"Strategy: {selected_path.strategy}"]
        path_lines.append(f"Score: {selected_path.coverage_score:.3f}")
        path_lines.append(f"Total transit: {selected_path.total_transit_min} min")
        for i, node in enumerate(selected_path.nodes):
            if not node.poi or not node.time:
                continue
            name = node.poi.name
            est = node.time.earliest_start
            eet = node.time.earliest_end
            lst = node.time.latest_start
            let = node.time.latest_end
            slack = node.time.slack_min
            transit = node.transit_from_prev_min
            stay = node.stay_duration_min
            path_lines.append(
                f"  {i+1}. [{node.slot}] {name} | "
                f"E: {est}-{eet} | L: {lst}-{let} | "
                f"stay={stay}min transit={transit}min slack={slack}min"
            )
        path_text = "\n".join(path_lines)

    # Weather
    weather = state.get("weather")
    weather_text = ""
    if weather and weather.condition:
        weather_text = (
            f"{weather.date} {weather.condition} "
            f"{f'{weather.temp_low:.0f}~{weather.temp_high:.0f}°C' if weather.temp_low and weather.temp_high else ''}"
            f"{' | 建议室内' if weather.indoor_recommended else ''}"
        )

    # Modification request
    modify_text = f"\n\n⚠️ 用户修改请求: {modify_feedback}" if modify_feedback else ""

    system_prompt = _VERIFY_ADJUST_PROMPT
    user_prompt = f"""══════════════════════════════════════
用户原始需求
══════════════════════════════════════
{conversation}

══════════════════════════════════════
提取的意图
══════════════════════════════════════
{intent_text}

══════════════════════════════════════
当前计划
══════════════════════════════════════
{plan_json}

══════════════════════════════════════
POI 候选池（只能从这里选，不可搜索新POI）
══════════════════════════════════════
{candidate_text}

══════════════════════════════════════
距离矩阵（场所间距离）
══════════════════════════════════════
{dist_text}

{"══════════════════════════════════════" if path_text else ""}
{"时间分配详情（算法规划的时序和松弛度）" if path_text else ""}
{"══════════════════════════════════════" if path_text else ""}
{path_text}

{"══════════════════════════════════════" if weather_text else ""}
{"天气" if weather_text else ""}
{"══════════════════════════════════════" if weather_text else ""}
{weather_text}
{modify_text}

请逐项检查计划，输出验证结果 JSON。"""

    plan_modified = False
    try:
        llm = make_model(temperature=0.3)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        text = await llm_invoke(llm, messages, "verify_and_adjust", stream=False)
        data = _extract_json(text)

        # Parse verification
        score = float(data.get("score", 0.8))
        issues = data.get("issues_found", [])
        changes = data.get("changes_made", [])
        status = data.get("status", "pass")

        # Parse adjusted plan if present
        plan_data = data.get("plan")
        if plan_data and "sub_tasks" in plan_data:
            try:
                plan_data = _normalize_plan(plan_data)
                new_plan = Plan.model_validate(plan_data)
                _backfill_plan_locations(new_plan, candidates)
                plan = new_plan
                plan_modified = True
                logger.info("verify_and_adjust: plan modified (%d changes)", len(changes))
            except Exception as exc:
                logger.warning("verify_and_adjust: failed to parse adjusted plan: %s", exc)

        verification = Verification(
            score=max(0.0, min(1.0, score)),
            issues=issues,
            status=status if status in ("pass", "fix", "reject") else "pass",
        )

        # ── Programmatic safety net: venue repetition ──
        # The LLM sometimes misses same-venue-across-slots.
        # Check book-task names and force status→fix if repeated.
        venue_names: list[str] = []
        for st in plan.sub_tasks:
            if st.type == "book" and st.params:
                name = st.params.get("name", "")
                if name:
                    venue_names.append(name)
        dupes = {n for n in venue_names if venue_names.count(n) > 1}
        if dupes:
            dup_msg = f"场所重复：{'、'.join(dupes)} 出现在多个 slot——出行体验单一"
            if dup_msg not in issues:
                issues.append(dup_msg)
            verification.issues = issues
            verification.score = max(0.0, verification.score - 0.15)
            if verification.status == "pass":
                verification.status = "fix"
                logger.info("  verify: %s (status overridden pass→fix)", dup_msg)

        for issue in issues:
            logger.info("  verify: %s", issue)
        for change in changes:
            logger.info("  adjust: %s", change)

        logger.info("verify_and_adjust: score=%.2f status=%s issues=%d changes=%d",
                    score, verification.status, len(issues), len(changes))

    except Exception as exc:
        logger.warning("verify_and_adjust LLM failed: %s — accepting plan as-is", exc)
        verification = Verification(score=0.8, issues=[], status="pass")

    result = {
        "plan": plan,
        "verification": verification,
        "plan_iterations": plan_iterations + 1,
        "messages": [{
            "role": "system",
            "content": f"[Plan verified: score={verification.score:.0%}, {verification.status}]",
        }],
    }
    if plan_modified:
        result["selected_path"] = None  # Clear stale path; _cards_from_plan reads updated plan
    return result


def route_after_verify_adjust(state: AgentState) -> str:
    """Route after verify_and_adjust: pass → present, fix → retry (max 2).

    Includes a programmatic safety net: if plan has same-venue-in-multiple-slots
    or score < 0.70, overrides to "fix" for one more verification pass.
    """
    verification = state.get("verification")
    iterations = state.get("plan_iterations", 0)
    plan = state.get("plan")

    if verification is None:
        return "present_to_user"

    # ── Safety net: check venue diversity before passing ──
    if verification.status == "pass" and plan and iterations < 2:
        book_names = [
            st.params.get("name", "")
            for st in plan.sub_tasks
            if st.type == "book" and st.params
        ]
        distinct = len(set(book_names))
        dupes = len(book_names) - distinct
        low_score = verification.score < 0.70

        # 3+ book slots but only 1-2 distinct venues → broken plan
        if low_score or (dupes > 1 and distinct < 3 and len(book_names) >= 3):
            logger.info(
                "Safety net: overriding pass→fix "
                "(score=%.2f, books=%d, distinct=%d, dupes=%d)",
                verification.score, len(book_names), distinct, dupes,
            )
            return "verify_and_adjust"

    if verification.status == "pass":
        return "present_to_user"

    if verification.status == "fix" and iterations < 2:
        logger.info("Re-entering verify_and_adjust (iteration %d/2)", iterations + 1)
        return "verify_and_adjust"

    # After 2 iterations or reject, present (even if imperfect)
    return "present_to_user"


# ── Legacy nodes (kept for use_new_planner=False rollback) ──

async def checker_agent(state: AgentState) -> dict:
    """[LEGACY] Pure-rule plan verifier — no LLM. Replaced by verify_and_adjust."""
    from finn.checker import check_plan

    plan = state.get("plan")
    extract = state.get("extract_result")
    candidates = state.get("poi_candidates", [])
    dist_matrix = state.get("distance_matrix", {})

    if not plan or not extract:
        return {"verification": Verification(score=1.0, issues=[], status="pass")}

    _, verification = check_plan(plan, extract, list(candidates), dist_matrix)
    return {"verification": verification}


async def fault_handler(state: AgentState) -> dict:
    """[LEGACY] LLM-based L3 repair. Replaced by verify_and_adjust."""
    from finn.llm import make_model, react_loop

    plan = state.get("plan")
    extract = state.get("extract_result")
    verification = state.get("verification")
    modify_feedback = state.get("modify_feedback", "")
    candidates = state.get("poi_candidates", [])

    if not plan or not extract:
        return {"next_action": "reject"}

    if modify_feedback:
        issue_desc = f"用户修改请求: {modify_feedback}"
    elif verification and verification.issues:
        issue_desc = "验证问题:\n" + "\n".join(f"- {i}" for i in verification.issues)
    else:
        return {"next_action": "verify_plan"}

    candidate_summary = "\n".join(
        f"- [{c.id}] {c.name} | {c.address} | type={c.type}"
        for c in candidates[:10]
    )

    system_prompt = f"""\
你是行程修复助手。根据以下问题修改计划。

{issue_desc}

可用 POI 候选:
{candidate_summary}

在已有计划基础上修改，只调整有问题的部分。输出修改后的完整 Plan JSON。"""

    plan_json = plan.model_dump_json(indent=2, ensure_ascii=False)
    user_prompt = f"当前计划:\n{plan_json}\n\n请修复上述问题，输出完整计划 JSON。"

    try:
        llm = make_model(temperature=0.3)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        text = await llm_invoke(llm, messages, "fault_handler", stream=False)
        data = _extract_json(text)
        if data and "sub_tasks" in data:
            new_plan = Plan(**data)
            return {"plan": new_plan, "next_action": "verify_plan"}
    except Exception as exc:
        logger.warning("fault_handler LLM failed: %s", exc)

    return {"next_action": "verify_plan"}


def route_after_check(state: AgentState) -> str:
    """[LEGACY] Route based on checker result."""
    verification = state.get("verification")
    iterations = state.get("plan_iterations", 0)

    if verification is None:
        return "reject"

    if verification.status == "pass":
        return "present_to_user"

    if verification.status == "fix" and iterations < 2:
        return "fault_handler"

    return "reject"


async def verify_plan(state: AgentState) -> dict:
    """[LEGACY] Node 3: Validate plan against user intent."""
    extract = state["extract_result"]
    plan = state["plan"]
    plan_json = plan.model_dump_json(indent=2, exclude_none=True)

    intent_lines = _build_intent_description(extract)
    intent_text = "\n  ".join(intent_lines)
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
    """[LEGACY] Node 7: Adjust plan based on verification issues and/or user feedback."""
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


def _format_plan(plan: Plan | None) -> str:
    """Render a Plan as a human-readable string."""
    if plan is None:
        return "No plan generated. Please try again with more details."
    lines = ["Your Trip Plan\n"]
    if plan.notes:
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
    """Node 4: Display plan and wait for user decision (HITL interrupt).

    Returns a pure state update. Routing is handled by the conditional edge
    ``route_after_present``, which routes to ``fan_out_bookings`` (confirm),
    ``qa_check`` (modify), or ``summarize_result`` (cancel).
    """
    from finn.presentation import generate_plan_cards

    plan = state.get("plan")
    candidates = state.get("poi_candidates", [])
    dist_matrix = state.get("distance_matrix", {})
    weather = state.get("weather")
    selected_path = state.get("selected_path")

    # Generate PlanCards for rich display (uses PlannedPath timing if available)
    cards = generate_plan_cards(
        plan, list(candidates), dist_matrix, weather,
        selected_path=selected_path,
    )

    # Build display text (supports both PlanCard-rich and legacy format)
    display_text = _format_plan(plan)
    if cards:
        card_lines = []
        for c in cards:
            type_label = f"[{c.activity_type}] " if c.activity_type else ""
            transport = f"  ({c.transport_from_prev})" if c.transport_from_prev else ""
            taxi = f"  [{c.taxi_estimate}]" if c.taxi_estimate else ""
            card_lines.append(
                f"{c.start_time}-{c.end_time}  {type_label}{c.poi_name}"
                f"{transport}{taxi}"
            )
        display_text = "\n".join(card_lines) if card_lines else display_text

    # Weather warning for severe conditions
    if weather and weather.indoor_recommended:
        weather_line = f"\u26a0\ufe0f {weather.date} 天气：{weather.condition}，建议优先室内活动"
        display_text = weather_line + "\n\n" + display_text

    decision = interrupt({
        "type": "plan_review",
        "plan": display_text,
        "plan_cards": [c.model_dump() for c in cards],
        "options": ["confirm", "modify", "cancel"],
    })

    if isinstance(decision, dict):
        action = decision.get("action", "cancel")
        feedback = decision.get("feedback", "")
        modify_count = state.get("modify_count", 0)
        if action == "modify":
            modify_count += 1
    else:
        action = decision if isinstance(decision, str) else "cancel"
        feedback = ""

    return {
        "next_action": action,
        "modify_feedback": feedback,
        "modify_count": state.get("modify_count", 0) + 1 if action == "modify" else state.get("modify_count", 0),
        "plan_cards": cards,
    }


# ═══════════════════════════════════════════════════════════════════════
# Node 4b: fan_out_bookings — prepares state for conditional edge fan-out
# ═══════════════════════════════════════════════════════════════════════


async def fan_out_bookings(state: AgentState) -> dict:
    """No-op node: routing is handled by the conditional edge ``route_fanout``.

    Previously used ``Command(goto=[Send(...)])`` but the parallel
    ``add_edge("fan_out_bookings", "book_worker")`` caused a fallback
    invocation with empty ``current_task_id``.  Now the Send fan-out
    lives entirely in ``route_fanout``, which is a conditional-edge
    function (no parallel edge to conflict with).
    """
    plan = state.get("plan")
    book_tasks = [
        st for st in (plan.sub_tasks if plan else [])
        if st.type == "book" and "cancel" not in st.id.lower()
    ]
    if not book_tasks:
        return Command(goto="summarize_result")
    return {}


def route_fanout(state: AgentState):
    """Conditional edge after fan_out_bookings: Send fan-out or skip.

    Returns ``list[Send]`` for the fan-out case (LangGraph processes
    Send objects from conditional edges directly), or the string
    ``"summarize_result"`` when there are no book tasks to execute.
    """
    from langgraph.types import Send as _Send

    plan = state.get("plan")
    book_tasks = [
        st for st in (plan.sub_tasks if plan else [])
        if st.type == "book" and "cancel" not in st.id.lower()
    ]
    if not book_tasks:
        return "summarize_result"
    return [
        _Send("book_worker", {
            "plan": plan,
            "current_task_id": st.id,
            "current_retry_count": 0,
        })
        for st in book_tasks
    ]


# ═══════════════════════════════════════════════════════════════════════
# Node 5: book_worker  (fan-out via Send)
# ═══════════════════════════════════════════════════════════════════════

MAX_RETRIES = 2
BOOKING_TIMEOUT_SEC = 10.0
BOOKING_DELAY_RANGE = (0.3, 1.5)  # simulated network latency range


async def _simulate_booking(task: SubTask, retry_count: int = 0) -> BookingResult:
    """Simulate a booking API call using real POI data from task.params.

    Uses the actual Amap POI ID (params["id"]) for mock order generation
    and POI name (params["name"]) for realistic error messages.
    Produces deterministic outcomes per (task.id, retry_count) so results
    are reproducible across runs.
    """
    params = task.params or {}
    poi_id = str(params.get("id", task.id))
    poi_name = str(params.get("name", task.target))

    seed = int(hashlib.md5(task.id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed + retry_count * 1000)

    # Simulate network latency
    delay = rng.uniform(*BOOKING_DELAY_RANGE)
    await asyncio.sleep(delay)

    roll = rng.random()

    if "cancel" in task.id.lower():
        return BookingResult(
            task_id=task.id, status="success",
            order_id=f"CANCEL-{poi_id}",
            retries=retry_count,
        )

    if roll < 0.10:
        return BookingResult(
            task_id=task.id, status="failed",
            error=f"网络超时: {poi_name} 预约接口 30s 无响应",
            error_type="transient", retries=retry_count,
        )
    elif roll < 0.18:
        return BookingResult(
            task_id=task.id, status="failed",
            error=f"预约失败: {poi_name} 已满员/售罄，无法预订",
            error_type="recoverable", retries=retry_count,
        )
    else:
        return BookingResult(
            task_id=task.id, status="success",
            order_id=f"ORD-{poi_id}-{abs(hash(poi_id)) % 10000:04d}",
            retries=retry_count,
        )


async def book_worker(state: AgentState) -> dict:
    """Node 5: Execute a single booking task (invoked via Send fan-out).

    Wraps _simulate_booking with asyncio.wait_for to model a real booking
    timeout. On TimeoutError, returns a transient failure so the retry
    machinery can re-attempt.
    """
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

    try:
        result = await asyncio.wait_for(
            _simulate_booking(task, retry_count),
            timeout=BOOKING_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        result = BookingResult(
            task_id=task_id, status="failed",
            error=f"Booking timed out after {BOOKING_TIMEOUT_SEC:.0f}s",
            error_type="transient", retries=retry_count,
        )

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

    # Iterate over a snapshot — compensations mutate bookings during the loop
    for task_id, result in list(bookings.items()):
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
                # Simulate the compensatory cancel booking — produces a
                # proper order_id instead of a bare status record.
                comp_task = SubTask(
                    id=comp_id, type="book",
                    target=f"取消: {task.target}",
                    dependencies=[], params={},
                )
                comp_result = await _simulate_booking(comp_task, 0)
                comp_result.status = "compensated"
                comp_result.error = f"Auto-compensated due to failure of {task_id}"
                bookings[comp_id] = comp_result
                msgs.append(f"🔙 {task_id}: recoverable, compensated via {comp_id}")
            else:
                # No compensatory configured — escalate as fatal
                result.error_type = "fatal"
                result.error = f"预约失败且无备选: {result.error}"
                msgs.append(f"🚫 {task_id}: {result.error}")

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
    """Edge C: Route based on user decision after plan review.

    Returns string destinations only (no Send objects).
    ``fan_out_bookings`` is a fresh node that handles the fan-out to book_worker,
    avoiding Send-in-conditional-edge issues from resumed interrupt nodes.
    """
    action = state.get("next_action", "cancel")
    if action == "confirm":
        return "fan_out_bookings"
    elif action == "modify":
        return "qa_check"
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
    """Edge E: Route after failure handling.

    Fans out retries with per-task retry count (not global).
    Each booking result tracks its own retries field; the next attempt
    uses r.retries + 1 so the random seed shifts deterministically.
    Global retry_count caps total retry rounds via MAX_RETRY_TOTAL.
    """
    action = state.get("next_action", "notify")
    retry_count = state.get("retry_count", 0)

    if action == "retry" and retry_count < MAX_RETRY_TOTAL:
        plan = state["plan"]
        bookings = state.get("bookings", {})
        sends = [
            Send("book_worker", {
                "plan": plan,
                "current_task_id": tid,
                "current_retry_count": r.retries + 1,  # per-task, not global
            })
            for tid, r in bookings.items()
            if r.status == "failed"
            and r.error_type == "transient"
            and r.retries < MAX_RETRIES  # belt-and-suspenders with handle_failures
        ]
        if sends:
            return sends
        return "notify"
    else:
        return "notify"
