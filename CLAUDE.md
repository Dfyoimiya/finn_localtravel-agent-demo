# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync          # install dependencies
uv run finn      # run interactive CLI
uv add <pkg>     # add a dependency
```

## Architecture

Finn is a **local short-trip planning & execution agent** using a **DAG + ReAct hybrid**:

```
LangGraph DAG (orchestration) → each node internally runs a PydanticAI ReAct loop
```

Full architecture design: `docs/architecture.md`

### File map

| File | Role |
|---|---|
| `src/finn/state.py` | `AgentState` + `Intent` Pydantic models. Flows through every node. |
| `src/finn/nodes.py` | Node implementations (`clarify_intent`, `decompose_and_plan`, etc.) + edge routing functions. Each LLM node creates a PydanticAI `Agent` internally. |
| `src/finn/graph.py` | LangGraph `StateGraph` assembly — registers nodes, wires edges (including conditional routing). |
| `src/finn/main.py` | Entry point. Interactive CLI loop with checkpointing (`MemorySaver`). |
| `src/finn/agent.py` | Legacy factory (unused by current flow; kept for reference). |

### Current DAG (step 4 implemented)

```
START → clarify_intent ─┬→ decompose_and_plan → verify_plan
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
```

- `clarify_intent` — extracts structured `Intent` via JSON-formatted LLM output
- `simple_answer` — answers non-trip questions (reserved, currently unused edge)
- `decompose_and_plan` — decomposes intent into a `Plan` (sub-task DAG). Handles fresh plans and revision (modify flow passes existing plan + feedback)
- `verify_plan` — evaluates plan against 5 dimensions; outputs `Verification` (score 0-1, issues[], status pass/fix/reject)
- `adjust_plan` — fixes verification issues; re-enters verify (max 4 total iterations; user modify resets counter)
- `present_to_user` — **HITL**: `interrupt()` pauses graph; user chooses confirm/modify/cancel
- `execute_bookings` — simulated booking execution (deterministic hash-based: 75% success, 15% transient, 10% recoverable). Cancel tasks always succeed
- `verify_execution` — classifies results into succeeded/failed buckets; reports to state
- `handle_failures` — classifies each failure: transient→retry (max 2 per task), recoverable→compensate (run compensatory cancel task), fatal→escalate to user
- `summarize_result` — final report: confirmed bookings, compensated cancellations, failures
- `notify_user` — escalation for non-retryable failures: tells user what failed and asks them to try again
- `reject` — capability boundary message (non-trip queries)

### HITL interrupt flow

1. Graph hits `present_to_user` → `interrupt()` pauses execution
2. State returned with `__interrupt__` key; CLI displays plan + options
3. User picks: confirm → execute | modify → re-decompose | cancel → END
4. Resume via `Command(resume=<choice>)` — may trigger a **second interrupt** (revised plan)
5. CLI loops on interrupts until graph completes (cascading HITL)

### State models (`src/finn/state.py`)

| Model | Key fields |
|---|---|
| `Intent` | goal, date, location, budget, preferences, is_complete |
| `SubTask` | id, type (search/compare/book), target, dependencies, compensatory |
| `Plan` | sub_tasks[], total_cost_estimate, notes |
| `Verification` | score (0-1), issues[], status (pass/fix/reject) |
| `BookingResult` | task_id, status (pending/success/failed/cancelled/compensated), order_id, error, error_type (transient/recoverable/fatal), retries |
| `AgentState` | extends MessagesState: intent, plan, verification, plan_iterations, next_action, modify_feedback, bookings (dict[str, BookingResult]), execution_status (idle/running/partial/done/failed/compensated), retry_count |

### State flow

Each invocation creates `{"messages": [user_msg]}` → graph runs node-by-node → conditional edges route → terminal node appends assistant message → END.

`plan_iterations` limits verify→adjust to 4 total attempts. User modify resets the counter.

## LLM Provider

Uses DeepSeek via OpenAI-compatible API. Configure with `.env`:

```
LLM_API_KEY=sk-xxx
LLM_MODEL=deepseek-v4-flash
LLM_BASE_URL=https://api.deepseek.com
```

**Note:** DeepSeek V4 models in "thinking" mode do not support `tool_choice`. Structured output is achieved via text-based JSON extraction rather than PydanticAI's native `output_type`. Keep this approach for any node that needs structured output from the model.

To switch providers, change `LLM_BASE_URL` and `LLM_MODEL` to any OpenAI-compatible endpoint.
