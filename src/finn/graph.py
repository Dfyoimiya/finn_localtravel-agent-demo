"""LangGraph DAG — workflow orchestration layer.

Step 4 graph (with failure handling & compensation):

    START
      │
      ▼
    clarify_intent ──┬→ decompose_and_plan → verify_plan
                     │       ▲         ▲          │
                     │       │         │    pass  fix   reject
                     │       │         │      │    │      │
                     │       │  adjust_plan     │    │      │
                     │       │    │             │    │      │
                     │       │    └─────────────┘    │      │
                     │       │              ┌────────┘      │
                     │       │              ▼               │
                     │       │      ★ present_to_user       │
                     │       │         [interrupt]          │
                     │       │     ┌──────┼──────┐         │
                     │       │  confirm │ modify  cancel   │
                     │       │     │     │        │        │
                     │       │     ▼     ▼        │        │
                     │       │  execute  │        │        │
                     │       │  _bookings│        │        │
                     │       │     │     │        │        │
                     │       │     │     └──→ decompose    │
                     │       │     │          _and_plan    │
                     │       │     │              │        │
                     │       │     ▼              │        │
                     │       │ verify_execution    │        │
                     │       │     │               │        │
                     │       │  ┌──┴──┐            │        │
                     │       │ done  partial/fail  │        │
                     │       │  │      │           │        │
                     │       │  │   handle_failures│        │
                     │       │  │      │           │        │
                     │       │  │   ┌──┴──┐        │        │
                     │       │  │  retry notify    │        │
                     │       │  │   │     │        │        │
                     │       │  │   ▼     ▼        │        │
                     │       │  │ execute notify   │        │
                     │       │  │_bookings _user    │        │
                     │       │  │   │               │        │
                     │       │  │   └───────────────┤        │
                     │       │  ▼                   │        │
                     │       │ summarize_result     │        │
                     │       │     │                │        │
                     ├→ END (clarify)               │        │
                     └→ reject ◄────────────────────┘        │
                                                            │
                    All terminal → END ◄────────────────────┘
"""

from langgraph.graph import END, StateGraph

from finn.state import AgentState
from finn.nodes import (
    adjust_plan,
    clarify_intent,
    decompose_and_plan,
    execute_bookings,
    handle_failures,
    notify_user,
    present_to_user,
    reject,
    route_after_clarify,
    route_after_exec,
    route_after_handle_failures,
    route_after_present,
    route_after_verify,
    simple_answer,
    summarize_result,
    verify_execution,
    verify_plan,
)


def create_graph():
    """Build and compile the Finn agent DAG."""
    graph = StateGraph(AgentState)

    # ── Nodes ──
    graph.add_node("clarify_intent", clarify_intent)            # Node 1
    graph.add_node("simple_answer", simple_answer)              # Node 2a (reserved)
    graph.add_node("decompose_and_plan", decompose_and_plan)    # Node 2b
    graph.add_node("reject", reject)                            # Node 2c
    graph.add_node("verify_plan", verify_plan)                  # Node 3
    graph.add_node("present_to_user", present_to_user)          # Node 4 ★ HITL
    graph.add_node("execute_bookings", execute_bookings)        # Node 5
    graph.add_node("verify_execution", verify_execution)        # Node: execution check
    graph.add_node("handle_failures", handle_failures)          # Node 6b
    graph.add_node("summarize_result", summarize_result)        # Node 6a
    graph.add_node("notify_user", notify_user)                  # Node: escalation
    graph.add_node("adjust_plan", adjust_plan)                  # Node 7

    # ── Edges ──
    graph.set_entry_point("clarify_intent")

    # After clarify → plan / ask-more / reject
    graph.add_conditional_edges(
        "clarify_intent", route_after_clarify,
        {"decompose_and_plan": "decompose_and_plan", "clarify": END, "reject": "reject"},
    )

    # Plan → Verify
    graph.add_edge("decompose_and_plan", "verify_plan")

    # Adjust → Verify (re-loop)
    graph.add_edge("adjust_plan", "verify_plan")

    # Verify → Present / Adjust / Reject
    graph.add_conditional_edges(
        "verify_plan", route_after_verify,
        {"present_to_user": "present_to_user", "adjust_plan": "adjust_plan", "reject": "reject"},
    )

    # Present → Execute / Re-decompose (modify) / Cancel
    graph.add_conditional_edges(
        "present_to_user", route_after_present,
        {"execute_bookings": "execute_bookings", "decompose_and_plan": "decompose_and_plan", "cancel": END},
    )

    # Execute → Verify execution
    graph.add_edge("execute_bookings", "verify_execution")

    # Verify exec → Summarize / Handle failures
    graph.add_conditional_edges(
        "verify_execution", route_after_exec,
        {"summarize": "summarize_result", "handle_failures": "handle_failures"},
    )

    # Handle failures → Retry / Notify
    graph.add_conditional_edges(
        "handle_failures", route_after_handle_failures,
        {"execute_bookings": "execute_bookings", "notify": "notify_user"},
    )

    # Terminal nodes → END
    graph.add_edge("simple_answer", END)
    graph.add_edge("summarize_result", END)
    graph.add_edge("notify_user", END)
    graph.add_edge("reject", END)

    return graph.compile()
