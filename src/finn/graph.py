"""LangGraph DAG — workflow orchestration layer.

Step 5 graph (with Send() fan-out for parallel booking execution):

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
                     │       │  [Send()] │        │        │
                     │       │  ┌──┼──┐  │        │        │
                     │       │  ▼  ▼  ▼  │        │        │
                     │       │  B  B  B   │        │        │
                     │       │  W  W  W   │        │        │
                     │       │  │  │  │   │        │        │
                     │       │  └──┼──┘   │        │        │
                     │       │     ▼      │        │        │
                     │       │ verify_exec│       │        │
                     │       │     │      │        │        │
                     │       │  ┌──┴──┐   │        │        │
                     │       │ done partial│       │        │
                     │       │  │  fail    │        │        │
                     │       │  │   │     │        │        │
                     │       │  │ handle_failures  │        │
                     │       │  │   │     │        │        │
                     │       │  │ [Send()]│        │        │
                     │       │  │  │ │ │  │        │        │
                     │       │  │  ▼ ▼ ▼  │        │        │
                     │       │  │  B B B  │        │        │
                     │       │  │  W W W  │        │        │
                     │       │  │  │ │ │  │        │        │
                     │       │  │  └─┼─┘  │        │        │
                     │       │  │    ▼    │        │        │
                     │       │  │ verify_exec│     │        │
                     │       │  │    │     │        │        │
                     │       │  ▼    │     │        │        │
                     │       │ sum-  │     │        │        │
                     │       │ marize│     │        │        │
                     │       │       │     │        │        │
                     ├→ END (clarify)     │        │        │
                     └→ reject ◄──────────┘        │        │
                     BW = book_worker   notify_user │        │
                     [Send()] = route returns       │        │
                     list[Send] for parallel fan-out│        │
                                                    │        │
                    All terminal → END ◄────────────┘
"""

from langgraph.graph import END, StateGraph

from finn.state import AgentState
from finn.nodes import (
    adjust_plan,
    book_worker,
    clarify_intent,
    decompose_and_plan,
    handle_failures,
    notify_user,
    present_to_user,
    reject,
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
    graph.add_node("book_worker", book_worker)                  # Node 5 (fan-out)
    graph.add_node("verify_execution", verify_execution)        # Node: execution check
    graph.add_node("handle_failures", handle_failures)          # Node 6b
    graph.add_node("summarize_result", summarize_result)        # Node 6a
    graph.add_node("notify_user", notify_user)                  # Node: escalation
    graph.add_node("adjust_plan", adjust_plan)                  # Node 7

    # ── Edges ──
    graph.set_entry_point("clarify_intent")

    # clarify_intent uses Command(goto=...) for routing.
    # Fallback edge in case Command is not returned.
    graph.add_edge("clarify_intent", END)

    # Plan → Verify
    graph.add_edge("decompose_and_plan", "verify_plan")

    # Adjust → Verify (re-loop)
    graph.add_edge("adjust_plan", "verify_plan")

    # Verify → Present / Adjust / Reject
    graph.add_conditional_edges(
        "verify_plan", route_after_verify,
        {"present_to_user": "present_to_user", "adjust_plan": "adjust_plan", "reject": "reject"},
    )

    # Present → fan-out to book_worker (confirm) / Re-decompose (modify) / Cancel
    graph.add_conditional_edges(
        "present_to_user", route_after_present,
        {"decompose_and_plan": "decompose_and_plan", "cancel": END, "summarize": "summarize_result"},
    )

    # book_worker → Verify execution (after all fan-out tasks complete)
    graph.add_edge("book_worker", "verify_execution")

    # Verify exec → Summarize / Handle failures
    graph.add_conditional_edges(
        "verify_execution", route_after_exec,
        {"summarize": "summarize_result", "handle_failures": "handle_failures"},
    )

    # Handle failures → fan-out retry (list[Send]) / Notify
    graph.add_conditional_edges(
        "handle_failures", route_after_handle_failures,
        {"notify": "notify_user"},
    )

    # Terminal nodes → END
    graph.add_edge("simple_answer", END)
    graph.add_edge("summarize_result", END)
    graph.add_edge("notify_user", END)
    graph.add_edge("reject", END)

    return graph.compile()
