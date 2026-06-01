"""LangGraph node implementations.

Each node is a pure async function: (state, config) -> partial state update.
Nodes that require LLM reasoning use PydanticAI ReAct agents internally.

Text-based JSON extraction is used for structured output because DeepSeek V4
thinking mode does not support tool_choice (required by native output_type).
"""

import json
import re
import time

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from langgraph.types import interrupt, Send

from finn.config import config
from finn.logger import logger
from finn.cli import get_on_token, get_on_tool
from finn.state import AgentState, BookingResult, Intent, Plan, SubTask, Verification

# ═══════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════


def _make_model() -> OpenAIChatModel:
    provider = OpenAIProvider(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
    )
    return OpenAIChatModel(
        model_name=config.llm_model,
        provider=provider,
    )


async def _run_agent(
    agent: Agent, prompt: str, node: str, *, stream: bool = False
) -> str:
    """Run an agent and return its response text, with logging.

    Set ``stream=True`` to stream text deltas through the ``on_token``
    callback (for user-visible text responses).  Structured-output nodes
    should leave ``stream=False`` (default).
    """
    model = agent.model.model_name if agent.model else "?"
    on_token = get_on_token() if stream else None
    logger.info("→ %s | model=%s | prompt=%d chars", node, model, len(prompt))
    t0 = time.monotonic()

    try:
        if on_token is not None:
            # ── streaming path ──
            collected: list[str] = []
            async with agent.run_stream(prompt) as streamed:
                async for delta in streamed.stream_text(delta=True):
                    on_token(delta)
                    collected.append(delta)
            text = "".join(collected)
            if not text.strip():
                text = streamed.response.text
        else:
            # ── fast path (no streaming) ──
            result = await agent.run(prompt)
            text = result.response.text

        elapsed = time.monotonic() - t0
        logger.info("← %s | %d chars in %.1fs", node, len(text), elapsed)
        return text
    except Exception:
        elapsed = time.monotonic() - t0
        logger.error("✗ %s | failed after %.1fs", node, elapsed)
        raise


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
            st["type"] = "search"  # default to search for unknown types
        if "dependencies" not in st:
            st["dependencies"] = []
        if "params" not in st:
            st["params"] = {}
    return data


# ═══════════════════════════════════════════════════════════════════════
# Node 1: clarify_intent
# ═══════════════════════════════════════════════════════════════════════

CLARIFY_SYSTEM_PROMPT = """\
You are an intent extraction module for Finn, a local short-trip planning agent.
Analyze the conversation and extract/update structured trip planning intent.

Output ONLY a valid JSON object (no markdown, no extra text) with these fields:

{
  "activity": "<what the user wants to do, e.g. 喝咖啡然后看电影.  Empty string if not trip-related>",
  "date": "<e.g. 周六, 2026-06-07, 今天.  null if unknown>",
  "start_location": "<where they depart from, e.g. 家, 国贸.  null if unknown>",
  "scenario": "family | friends | couple | solo | unknown",
  "start_time": "<e.g. 14:00, 下午2点.  null if unknown>",
  "duration_hours": <float or null>,
  "area": "<e.g. 朝阳区, 三里屯.  null if unknown>",
  "radius_km": <float or null — how far they're willing to travel>,
  "party_size": <int or null>,
  "party_members": [
    {
      "role": "<self | spouse | child | friend | colleague>",
      "age": <int or null>,
      "constraints": ["<e.g. 减肥中, 不吃辣, 海鲜过敏>"],
      "preferences": ["<e.g. 喜欢户外, 想喝奶茶>"]
    }
  ],
  "budget_total": <float or null — total budget in CNY>,
  "budget_per_person": <float or null — per-person budget in CNY>,
  "hard_constraints": ["<e.g. 需儿童座椅, 18:00前结束, 清真饮食, 包间>"],
  "preferences": ["<e.g. 安静, 户外, 高评分, 川菜, 适合拍照>"],
  "follow_up_question": "<natural Chinese question asking for ONE missing critical field, or null>"
}

Rules:
1. CRITICAL FIELDS: activity, date, start_location.  These MUST be collected before planning.
2. If a PREVIOUS INTENT is provided in the prompt, MERGE the latest message into it —
   carry forward all already-extracted fields unless the user explicitly changes them.
3. SCENARIO detection: 老婆/孩子 → family; 朋友/几个人/姐妹/兄弟 → friends;
   女朋友/男朋友/约会 → couple; 我一个人 → solo.
4. PARTY: when scenario is family/friends, try to extract party_size and party_members.
   Children under 12 → note in constraints (e.g. "需要儿童座椅", "需亲子设施").
   Dietary/health constraints → put in member.constraints FOR THAT PERSON.
5. CONSTRAINTS: user says "必须"/"不能"/"一定要" → hard_constraints.
   User says "最好"/"喜欢"/"想" → preferences or member.preferences.
   DO NOT put the same item in both lists.
6. Only fill fields where the user has ACTUALLY provided information.
   Leave unknown fields as null / empty list.  NEVER guess or invent.
7. Ask only ONE missing critical field at a time in follow_up_question.
   Use natural conversational Chinese.  If all critical fields present, set to null.
8. Output ONLY the raw JSON object — no ``` fences, no extra text."""


# Critical fields for programmatic validation
_CRITICAL = ("activity", "date", "start_location")


async def clarify_intent(state: AgentState) -> dict:
    """Node 1: Extract structured intent from the conversation.

    Sends full conversation history + previous intent for multi-turn memory.
    Validates critical fields in code (not LLM) and logs the extracted intent.
    """
    # ── Build prompt ──
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
    prompt = f"Conversation:\n{history}{existing_json}"

    # ── LLM call ──
    try:
        agent = Agent(_make_model(), system_prompt=CLARIFY_SYSTEM_PROMPT)
        text = await _run_agent(agent, prompt, "clarify_intent")
        data = _extract_json(text)
        clean = _strip_nulls(data, "activity")
        intent = Intent.model_validate(clean)
    except Exception:
        logger.warning("clarify_intent parse failure")
        return {
            "intent": Intent(activity="", missing_critical=["activity"]),
            "next_action": "clarify",
            "messages": [{
                "role": "assistant",
                "content": "抱歉我没太理解，能换个方式说说你的需求吗？",
            }],
        }

    # ── Programmatic validation ──
    if not intent.activity or not intent.activity.strip():
        next_action = "reject"
        intent.missing_critical = ["activity"]
    else:
        missing = [f for f in _CRITICAL if getattr(intent, f) is None]
        intent.missing_critical = missing
        if missing:
            next_action = "clarify"
        else:
            next_action = "decompose_and_plan"

    # ── Log intent ──
    _log_intent(intent, next_action)

    return {
        "intent": intent,
        "next_action": next_action,
        "plan_iterations": 0,
    }


def _log_intent(intent: Intent, next_action: str) -> None:
    """Log extracted intent — summary at INFO, full JSON at DEBUG."""
    members = len(intent.party_members)
    info_parts = [
        f"activity={intent.activity!r}",
        f"date={intent.date!r}",
        f"loc={intent.start_location!r}",
    ]
    if intent.scenario != "unknown":
        info_parts.append(f"scenario={intent.scenario}")
    if intent.party_size:
        info_parts.append(f"party={intent.party_size}")
    if intent.hard_constraints:
        info_parts.append(f"hard={intent.hard_constraints}")
    if intent.preferences:
        info_parts.append(f"prefs={intent.preferences}")
    info_parts.append(f"missing={intent.missing_critical}")
    info_parts.append(f"→ {next_action}")

    logger.info("Intent | %s", " | ".join(info_parts))
    logger.debug(
        "Intent JSON | %s",
        intent.model_dump_json(indent=2, exclude_none=True),
    )


# ═══════════════════════════════════════════════════════════════════════
# Node 2a: simple_answer
# ═══════════════════════════════════════════════════════════════════════

SIMPLE_ANSWER_PROMPT = """\
You are Finn, a helpful local assistant. Answer the user's question concisely.
If you don't know something, say so honestly. Keep answers under 200 words."""


async def simple_answer(state: AgentState) -> dict:
    """Node 2a: Answer a simple non-trip question."""
    agent = Agent(_make_model(), system_prompt=SIMPLE_ANSWER_PROMPT)
    text = await _run_agent(
        agent, state["messages"][-1].content, "simple_answer", stream=True
    )
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
You are a task planning module for Finn, a local short-trip planning agent.
Given a structured user intent (and optionally a previous plan + revision request),
decompose it into a concrete, executable plan.

If this is a REVISION of an existing plan:
- Start from the existing plan as a base
- Apply the user's modification request
- Keep everything the user didn't ask to change
- Add, remove, or modify sub-tasks as requested

Output ONLY a valid JSON object:
{
  "sub_tasks": [
    {
      "id": "<unique_id>",
      "type": "search" | "compare" | "book",
      "target": "<what this accomplishes, e.g. lunch, movie, transport>",
      "dependencies": ["<task_id>" ...],
      "params": {<relevant parameters>},
      "compensatory": "<cancel_task_id>" | null
    }
  ],
  "total_cost_estimate": <number in CNY or null>,
  "notes": "<natural language summary of the plan in Chinese>"
}

Rules:
1. Max 8 sub-tasks.
2. "search" tasks have no dependencies — they gather information.
3. "book" tasks depend on at least one "search" task.
4. Every "book" task must have a compensatory: the id of a matching cancel/refund task.
5. Include transit between locations as tasks when needed.
6. Be specific: name real venues, real times, real prices for the given location.
7. If the user specified a budget, stay within it.
8. Order tasks to form a coherent timeline.

Output ONLY the JSON, no other text."""


async def decompose_and_plan(state: AgentState) -> dict:
    """Node 2b: Decompose intent into executable sub-task DAG.

    If `modify_feedback` is present, treats this as a revision
    of an existing plan incorporating user feedback.
    """
    intent = state["intent"]
    modify_feedback = state.get("modify_feedback", "")
    existing_plan = state.get("plan")

    # Build a rich intent description for the planner
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
        lines.append(f"Revise the following plan to incorporate this change:")
        lines.append(plan_json)
        lines.append("Keep everything the user didn't ask to change.")

    user_prompt = "\n".join(lines)

    agent = Agent(_make_model(), system_prompt=DECOMPOSE_PROMPT)
    text = await _run_agent(agent, user_prompt, "decompose_and_plan")
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

    # Build compact intent summary for verifier
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

    agent = Agent(_make_model(), system_prompt=VERIFY_PROMPT)
    text = await _run_agent(agent, user_prompt, "verify_plan")
    data = _extract_json(text)
    verification = Verification.model_validate(data)

    iterations = state.get("plan_iterations", 0) + 1

    return {
        "verification": verification,
        "plan_iterations": iterations,
    }


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

    agent = Agent(_make_model(), system_prompt=ADJUST_PROMPT)
    text = await _run_agent(agent, user_prompt, "adjust_plan")
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
    """Node 4: Display plan to user and wait for decision.

    Uses LangGraph interrupt() — the graph pauses here. The caller
    displays the plan, collects user choice, and resumes with Command.

    Resume value can be:
      "confirm" / "cancel"  → simple string
      {"action": "modify", "feedback": "..."}  → dict with user feedback
    """
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
            "plan_iterations": 0,  # user-driven change resets counter
        }
    return {"next_action": decision}


# ═══════════════════════════════════════════════════════════════════════
# Node 5: book_worker  (fan-out via Send)
# ═══════════════════════════════════════════════════════════════════════

import hashlib
import random

MAX_RETRIES = 2


def _simulate_booking(task: SubTask, retry_count: int = 0) -> BookingResult:
    """Simulate a booking API call with deterministic mixed results.

    In production this would be a real API call. For now, uses
    a hash of the task id to produce repeatable pass/fail outcomes.
    """
    seed = int(hashlib.md5(task.id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed + retry_count * 1000)  # retry shifts the outcome

    roll = rng.random()

    # Compensation/cancel tasks always succeed
    if "cancel" in task.id.lower():
        return BookingResult(
            task_id=task.id, status="success",
            order_id=f"CANCEL-{task.id[:8].upper()}",
        )

    if roll < 0.15:
        # 15% transient failure (timeout, network)
        return BookingResult(
            task_id=task.id, status="failed",
            error="Network timeout after 30s",
            error_type="transient", retries=retry_count,
        )
    elif roll < 0.25:
        # 10% recoverable failure (sold out, unavailable)
        return BookingResult(
            task_id=task.id, status="failed",
            error=f"Sorry, '{task.target}' is fully booked / sold out",
            error_type="recoverable", retries=retry_count,
        )
    else:
        # 75% success
        return BookingResult(
            task_id=task.id, status="success",
            order_id=f"ORD-{task.id[:8].upper()}-{abs(hash(task.id)) % 10000:04d}",
        )


async def book_worker(state: AgentState) -> dict:
    """Node 5: Execute a single booking task (invoked via Send fan-out).

    Receives `plan`, `current_task_id`, and `current_retry_count` set
    by Send.arg. Looks up the task in the plan, simulates the booking,
    and returns a partial booking result for merge-back.
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
    messages: list[str] = []

    for task_id, result in bookings.items():
        if result.status != "failed":
            continue

        if result.error_type == "transient" and result.retries < MAX_RETRIES:
            messages.append(f"🔄 {task_id}: transient error, will retry "
                            f"(attempt {result.retries + 1}/{MAX_RETRIES})")
            has_retryable = True

        elif result.error_type == "recoverable":
            # Compensate related successful bookings
            task = next((st for st in plan.sub_tasks if st.id == task_id), None)
            if task and task.compensatory:
                comp_id = task.compensatory
                bookings[comp_id] = BookingResult(
                    task_id=comp_id, status="compensated",
                    error=f"Auto-compensated due to failure of {task_id}",
                )
            messages.append(f"🔙 {task_id}: recoverable error, "
                            f"compensating up to {result.retries}")

        else:
            # Fatal or exhausted retries
            messages.append(f"🚫 {task_id}: fatal error — {result.error}")

    next_action = "retry" if has_retryable else "notify"

    return {
        "bookings": bookings,
        "retry_count": state.get("retry_count", 0) + 1,
        "next_action": next_action,
        "messages": [{"role": "assistant", "content": "\n".join(messages)}],
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
        lines.append(f"\n  Enjoy your trip! 🎉")

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


MAX_PLAN_ITERATIONS = 4  # initial + up to 3 adjust cycles
MAX_RETRY_TOTAL = 3      # max total retry rounds across all tasks


def route_after_clarify(state: AgentState) -> str:
    """Edge A: Route based on intent completeness."""
    action = state.get("next_action", "reject")
    if action == "decompose_and_plan":
        return "decompose_and_plan"
    elif action == "clarify":
        return "clarify"
    else:
        return "reject"


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

    Returns list[Send] for confirm (fan-out to book_worker),
    or a string destination for modify/cancel.
    """
    action = state.get("next_action", "cancel")
    if action == "confirm":
        plan = state["plan"]
        book_tasks = [
            st for st in (plan.sub_tasks if plan else [])
            if st.type == "book" and "cancel" not in st.id.lower()
        ]
        if not book_tasks:
            return "summarize"  # nothing to execute; skip to summary
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
    """Edge E: Route after failure handling.

    Returns list[Send] for retry (fan-out only failed transient tasks),
    or "notify" to escalate to user.
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
