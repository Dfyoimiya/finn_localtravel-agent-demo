"""Finn agent state definitions.

State flows through LangGraph nodes. Each node reads and writes these fields.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from langgraph.graph import MessagesState


def _merge_bookings(
    left: dict[str, "BookingResult"],
    right: dict[str, "BookingResult"],
) -> dict[str, "BookingResult"]:
    """Merge two booking dicts — right-side keys take precedence."""
    merged = dict(left)
    merged.update(right)
    return merged


# ── Phase 1: Clarify ─────────────────────────────────────────────────


class PartyMember(BaseModel):
    """A person or category of people in the party."""

    role: str = ""
    # "self", "spouse", "child", "friend", "colleague"
    age: int | None = None
    constraints: list[str] = Field(default_factory=list)
    # e.g. "减肥中", "不吃辣", "海鲜过敏", "需要午睡"
    preferences: list[str] = Field(default_factory=list)
    # e.g. "喜欢户外", "想喝奶茶", "想看IMAX"


class Intent(BaseModel):
    """Clarify phase — structured trip intent extracted from conversation.

    Design:
    - Every field the LLM can populate is above the fold.
    - ``missing_critical`` is set by *code* after extraction, not by the LLM.
    - Downstream nodes read ``activity`` / ``date`` / ``start_location``
      as canonical keys (no more ``goal`` / ``location`` ambiguity).
    """

    # ── Core (program-validated) ────────────────────────────────────
    activity: str = ""
    # "喝咖啡然后看电影", "亲子乐园+晚餐", "吃火锅"
    date: str | None = None
    # "周六", "2026-06-07", "今天"
    start_location: str | None = None
    # 出发地: "家", "国贸", "望京SOHO"

    # ── Scenario ────────────────────────────────────────────────────
    scenario: Literal["family", "friends", "couple", "solo", "unknown"] = "unknown"

    # ── Time ────────────────────────────────────────────────────────
    start_time: str | None = None
    # "14:00", "下午2点", "午饭后"
    duration_hours: float | None = None
    # 预计时长

    # ── Space ───────────────────────────────────────────────────────
    area: str | None = None
    # 活动区域/商圈
    radius_km: float | None = None
    # 可接受最远距离 (km), null=不限

    # ── Party ───────────────────────────────────────────────────────
    party_size: int | None = None
    party_members: list[PartyMember] = Field(default_factory=list)

    # ── Budget ──────────────────────────────────────────────────────
    budget_total: float | None = None
    budget_per_person: float | None = None

    # ── Constraints ─────────────────────────────────────────────────
    hard_constraints: list[str] = Field(default_factory=list)
    # 必须满足: "需儿童座椅", "18:00前结束", "清真饮食", "包间"
    preferences: list[str] = Field(default_factory=list)
    # 尽量满足: "安静", "户外", "高评分", "川菜", "适合拍照"

    # ── Validation (set by code, not LLM) ──────────────────────────
    missing_critical: list[str] = Field(default_factory=list)
    follow_up_question: str | None = None


# ── Phase 2: Plan ────────────────────────────────────────────────────


class SubTask(BaseModel):
    """A single unit of work within a plan."""

    id: str
    type: Literal["search", "compare", "book"]
    target: str
    dependencies: list[str] = Field(default_factory=list)
    params: dict = Field(default_factory=dict)
    compensatory: str | None = None


class Plan(BaseModel):
    """Decomposed plan: a DAG of sub-tasks with estimated cost."""

    sub_tasks: list[SubTask] = Field(default_factory=list)
    total_cost_estimate: float | None = None
    notes: str = ""


class Verification(BaseModel):
    """Result of plan verification."""

    score: float
    issues: list[str] = Field(default_factory=list)
    status: Literal["pass", "fix", "reject"] = "fix"


# ── Phase 3: Execution ───────────────────────────────────────────────


class BookingResult(BaseModel):
    """Outcome of a single booking sub-task."""

    task_id: str
    status: Literal["pending", "success", "failed", "cancelled", "compensated"]
    order_id: str | None = None
    error: str | None = None
    error_type: Literal["transient", "recoverable", "fatal"] | None = None
    retries: int = 0


# ── Agent State ──────────────────────────────────────────────────────


class AgentState(MessagesState):
    """State carried through every LangGraph node.

    Inherits ``messages`` with ``add_messages`` reducer from ``MessagesState``.
    """

    intent: Intent | None
    plan: Plan | None
    verification: Verification | None
    plan_iterations: int
    clarify_iterations: int
    next_action: str
    modify_feedback: str

    # Execution phase
    bookings: Annotated[dict[str, BookingResult], _merge_bookings]
    execution_status: Literal["idle", "running", "partial", "done", "failed", "compensated"]
    retry_count: int

    # Fan-out context (set via Send.arg for book_worker)
    current_task_id: str
    current_retry_count: int
