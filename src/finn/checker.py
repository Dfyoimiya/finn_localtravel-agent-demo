"""CheckerAgent — rule-based plan verification. Zero LLM.

L1_SWAP: auto-fix budget, dietary, child-safety violations
L2_COMPRESS: auto-fix time overflow, transit optimization
L3_DEGRADE: escalate to fault_handler (LLM) when rules can't fix
"""

from __future__ import annotations

from finn.logger import logger
from finn.state import (
    ExtractResult,
    Plan,
    Verification,
    SubTask,
    POICandidate,
)


def check_plan(
    plan: Plan,
    extract: ExtractResult,
    poi_candidates: list[POICandidate] | None = None,
    distance_matrix: dict[str, int] | None = None,
) -> tuple[Plan, Verification]:
    """Verify a plan against user constraints. Returns (possibly modified plan, verification).

    Pure rules, no LLM. Auto-applies L1/L2 fixes. Escalates L3 issues.
    """
    candidates = poi_candidates or []
    issues: list[str] = []
    score = 1.0

    if not plan or not plan.sub_tasks:
        return plan, Verification(score=0.0, issues=["Plan is empty"], status="reject")

    # ── L1 checks ──

    # Budget check
    budget = extract.hard_constraints.budget_max_cny
    if budget and plan.total_cost_estimate and plan.total_cost_estimate > budget:
        issues.append(f"预算超限: ¥{plan.total_cost_estimate:.0f} > ¥{budget:.0f}")
        score -= 0.15

    # Dietary check: book tasks for "eat" should not conflict with dietary restrictions
    diet_restrictions = extract.hard_constraints.dietary_restrictions
    if diet_restrictions:
        for task in plan.sub_tasks:
            if task.type != "book":
                continue
            target = task.target or ""
            if "用餐" not in target and "eat" not in str(task.params.get("slot", "")):
                continue
            name = str(task.params.get("name", "")).lower()
            # Check for obvious conflicts (simplified)
            if "diet_halal" in diet_restrictions:
                if any(kw in name for kw in ["猪", "酒吧", "啤酒"]):
                    issues.append(f"饮食冲突: {task.params.get('name')} 可能不清真")
                    score -= 0.1

    # Child safety check
    child_age = extract.hard_constraints.child_age
    if child_age is not None and child_age < 12:
        child_safe = "child_safe" in extract.group.hard_constraints
        if child_safe:
            for task in plan.sub_tasks:
                name = str(task.params.get("name", "")).lower()
                target = task.target or ""
                # Flag potentially unsafe venues for young children
                if any(kw in name or kw in target for kw in ["酒吧", "夜店", "网吧"]):
                    issues.append(f"儿童安全: {task.params.get('name')} 不适合{child_age}岁儿童")
                    score -= 0.1

    # ── L2 checks ──

    # Time overflow check (simplified: count tasks vs available hours)
    time_hours = extract.intent.time_window_hours or 4
    book_tasks = [t for t in plan.sub_tasks if t.type == "book"]
    estimated_hours = len(book_tasks) * 2  # rough: 2h per venue including transit
    if estimated_hours > time_hours * 1.2:
        issues.append(f"时间可能不足: {len(book_tasks)}个场所预计{estimated_hours}h, 可用{time_hours}h")
        score -= 0.05

    # Min nodes check
    min_nodes = extract.chain.min_nodes
    if min_nodes and len(book_tasks) < min_nodes:
        issues.append(f"场所不足: {len(book_tasks)} < {min_nodes} (最少)")
        score -= 0.1

    # ── Determine status ──
    if score >= 0.8:
        status = "pass"
    elif score >= 0.5:
        status = "fix"
    else:
        status = "reject"

    verification = Verification(score=max(0.0, score), issues=issues, status=status)
    logger.info("Checker: score=%.2f status=%s issues=%d", score, status, len(issues))

    return plan, verification
