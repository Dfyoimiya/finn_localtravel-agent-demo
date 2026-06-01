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


class Intent(BaseModel):
    """Structured trip planning intent extracted from user input."""

    goal: str | None = None
    date: str | None = None
    location: str | None = None
    budget: float | None = None
    preferences: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    follow_up_question: str | None = None
    is_complete: bool = False


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

    Inherits `messages` with add_messages reducer from MessagesState.
    """

    intent: Intent | None
    plan: Plan | None
    verification: Verification | None
    plan_iterations: int
    next_action: str
    modify_feedback: str

    # Execution phase
    bookings: Annotated[dict[str, BookingResult], _merge_bookings]
    execution_status: Literal["idle", "running", "partial", "done", "failed", "compensated"]
    retry_count: int

    # Fan-out context (set via Send.arg for book_worker)
    current_task_id: str
    current_retry_count: int
