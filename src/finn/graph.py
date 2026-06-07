r"""LangGraph DAG — workflow orchestration layer.

LLM-driven planner DAG:

  START
    → clarify_intent ─┬→ plan: context_agent
                       │     → formulate_search (LLM: search strategy)
                       │       → execute_category_search (parallel, max 2 rounds)
                       │         → multi_agent_plan (3+ LLM agents, different strategies)
                       │           → plan_fusion (weighted voting)
                       │             → present_to_user [HITL interrupt]
                       │               ├→ confirm → fan_out_bookings
                       │               │    → book_worker (fan-out)
                       │               │    → verify_execution
                       │               │      ├→ summarize_result → END
                       │               │      └→ handle_failures → notify_user → END
                       │               ├→ modify → qa_check (LLM re-optimize)
                       │               │    → present_to_user
                       │               └→ cancel → END
                       ├→ clarify → END
                       └→ reject → reject → END

LLM is only used for: intent extraction, search strategy, planning, fusion,
and the modify/re-optimize loop. The execution path (fan_out → book → verify
→ handle_failures → summarize) is pure programmatic logic — zero LLM calls.
"""

from langgraph.graph import END, StateGraph

from finn.state import AgentState
from finn.nodes import (
    book_worker,
    clarify_intent,
    fan_out_bookings,
    handle_failures,
    notify_user,
    present_to_user,
    reject,
    route_after_exec,
    route_after_handle_failures,
    route_after_present,
    route_fanout,
    summarize_result,
    verify_execution,
)
from finn.context import context_agent
from finn.search_planner import (
    execute_category_search,
    formulate_search,
)
from finn.multi_planner import multi_agent_plan
from finn.fusion import plan_fusion, qa_check


def create_graph():
    """Build and compile the Finn agent DAG (new LLM-driven planner)."""
    graph = StateGraph(AgentState)

    # ── Nodes ──
    graph.add_node("clarify_intent", clarify_intent)              # Node 1: extract
    graph.add_node("context_agent", context_agent)                # Node 2: weather + geocode
    graph.add_node("formulate_search", formulate_search)          # Node 3: LLM search strategy
    graph.add_node("execute_category_search", execute_category_search)  # Node 4: parallel search
    graph.add_node("multi_agent_plan", multi_agent_plan)          # Node 5: 3+ parallel planners
    graph.add_node("plan_fusion", plan_fusion)                    # Node 6: weighted voting → directly to present
    graph.add_node("qa_check", qa_check)                          # Node 7: optimize (modify loop only)
    graph.add_node("present_to_user", present_to_user)            # Node 8: HITL interrupt
    graph.add_node("fan_out_bookings", fan_out_bookings)          # Node 8b: fan-out to book_worker
    graph.add_node("reject", reject)                              # Node: capability boundary

    # Execution phase (unchanged)
    graph.add_node("book_worker", book_worker)                    # Node: fan-out booking
    graph.add_node("verify_execution", verify_execution)          # Node: execution check
    graph.add_node("handle_failures", handle_failures)            # Node: retry/compensate
    graph.add_node("summarize_result", summarize_result)          # Node: final summary
    graph.add_node("notify_user", notify_user)                    # Node: escalation

    # ── Edges ──
    graph.set_entry_point("clarify_intent")

    # clarify_intent uses Command(goto=...) for routing:
    #   "plan" → context_agent
    #   "clarify" → END
    #   "reject" → reject
    # Fallback edge if Command is not returned
    graph.add_edge("clarify_intent", END)

    # context_agent → formulate_search (LLM designs search strategy)
    graph.add_edge("context_agent", "formulate_search")

    # formulate_search → execute_category_search (always, via Command goto)
    graph.add_edge("formulate_search", "execute_category_search")

    # execute_category_search → multi_agent_plan or loop back to formulate_search
    # (routing via Command goto inside node)
    graph.add_edge("execute_category_search", "multi_agent_plan")

    # multi_agent_plan → plan_fusion (always, via Command goto)
    graph.add_edge("multi_agent_plan", "plan_fusion")

    # plan_fusion → present_to_user (directly, skip redundant LLM qa_check)
    graph.add_edge("plan_fusion", "present_to_user")

    # qa_check → present_to_user (used only by modify loop via route_after_present)
    graph.add_edge("qa_check", "present_to_user")

    # ── HITL & Execution (same as before) ──

    # Present: confirm → fan_out_bookings → book_worker (fan-out), modify → qa_check, cancel → END
    graph.add_conditional_edges(
        "present_to_user", route_after_present,
        {"fan_out_bookings": "fan_out_bookings",
         "qa_check": "qa_check",
         "cancel": END,
         "summarize": "summarize_result"},
    )

    # fan_out_bookings → book_worker (fan-out via conditional edge returning list[Send])
    # or → summarize_result if no book tasks
    graph.add_conditional_edges(
        "fan_out_bookings", route_fanout,
        {"summarize_result": "summarize_result"},
    )

    # book_worker → verify_execution (after all fan-out tasks complete)
    graph.add_edge("book_worker", "verify_execution")

    # verify_execution → summarize / handle_failures
    graph.add_conditional_edges(
        "verify_execution", route_after_exec,
        {"summarize": "summarize_result", "handle_failures": "handle_failures"},
    )

    # handle_failures → fan-out retry (list[Send]) / notify
    graph.add_conditional_edges(
        "handle_failures", route_after_handle_failures,
        {"notify": "notify_user"},
    )

    # Terminal nodes → END
    graph.add_edge("summarize_result", END)
    graph.add_edge("notify_user", END)
    graph.add_edge("reject", END)

    return graph.compile()
