# Finn Agent — Architecture Design

## Overview

Finn is a **local short-trip planning & execution agent** that takes natural language
input, researches POIs, plans an executable itinerary, presents it to the user for
approval, and completes bookings on the user's behalf.

The architecture is **LangGraph DAG (orchestration) + per-node LLM ReAct loop**.

---

## Design Principles

| Principle | Source | Meaning |
|---|---|---|
| **Plan-Execute-Verify-Replan** | VMAO (arXiv:2603.11445) | Cyclic verification with correction loops, not one-shot planning |
| **Route first, plan lazy** | "Two-Speed" Architecture | Classify intent cheaply; only run heavy planning for multi-step tasks |
| **User approves execution, not planning** | AutoGen / CrewAI consensus | Show the plan; user clicks approve; agent executes |
| **Saga compensation** | Conductor / Temporal pattern | Every book task has a registered compensatory cancel task |
| **Sync checkpointing** | LangGraph MemorySaver | Persist state every superstep; resume from crash |
| **Fan-out / Fan-in** | LLMCompiler (arXiv:2312.04511) | Independent sub-tasks run in parallel via `Send()`, results merged via reducer |

---

## DAG Topology

```
                              START
                                │
                                ▼
                       ┌─────────────────┐
                       │  clarify_intent │  ★ LLM-driven routing via Command(goto=...)
                       └────────┬────────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
                "clarify"    "plan"     "reject"
                    │           │           │
                    ▼           ▼           ▼
                  END    ┌───────────┐  ┌────────┐
                         │context_   │  │ reject │ → END
                         │agent      │  └────────┘
                         └─────┬─────┘
                               │
                               ▼
                       ┌───────────────┐
                       │formulate_     │  LLM: design search strategy
                       │search         │
                       └───────┬───────┘
                               │
                               ▼
                       ┌───────────────┐
                       │execute_       │  MCP batch search (≤2 rounds)
                       │category_search│ ←── loop back if coverage insufficient
                       └───────┬───────┘
                               │
                               ▼
                       ┌───────────────┐
                       │multi_agent_   │  2 LLM agents in parallel:
                       │plan           │  constraint_satisfaction + spatio_temporal
                       └───────┬───────┘
                               │
                               ▼
                       ┌───────────────┐
                       │plan_fusion    │  LLM weighted voting across agent plans
                       └───────┬───────┘
                               │
                               ▼
                       ┌───────────────┐
                       │present_to_user│  ★ HITL interrupt()
                       └───────┬───────┘
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
                "confirm"  "modify"   "cancel"
                    │          │          │
                    ▼          ▼          ▼
            ┌──────────┐ ┌──────────┐   END
            │fan_out_  │ │qa_check  │
            │bookings  │ │(≤3 loops)│──→ present_to_user
            └────┬─────┘ └──────────┘
                 │
                 ▼  Send() fan-out (parallel per sub-task)
            ┌──────────┐ ┌──────────┐ ┌──────────┐
            │book_     │ │book_     │ │book_     │  ...
            │worker    │ │worker    │ │worker    │
            └────┬─────┘ └────┬─────┘ └────┬─────┘
                 │            │            │
                 └────────────┼────────────┘
                              ▼  (reducer merges bookings dict)
                       ┌───────────────┐
                       │verify_execution│  classify: done / partial / failed
                       └───────┬───────┘
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
                  "done"   "partial"   "failed"
                    │     / "failed"      │
                    ▼          └────┐     │
            ┌──────────┐            ▼     │
            │summarize │    ┌──────────────┐
            │_result   │    │handle_failures│
            └────┬─────┘    └──────┬───────┘
                 │                 │
                 ▼         ┌───────┴───────┐
                END    "retry"         "notify"
                           │               │
                           ▼               ▼
                    Send() back to   ┌──────────┐
                    book_worker      │notify_user│ → END
                                     └──────────┘
```

**Key routing notes:**
- `clarify_intent` uses `Command(goto=...)` for LLM-driven routing (plan/clarify/reject). No separate edge function.
- `route_fanout` returns `list[Send("book_worker", {...})]` for parallel execution — implemented entirely in the conditional edge, not in a node.
- No LLM calls in the execution path (`fan_out_bookings` → `book_worker` → `verify_execution` → `handle_failures` → `summarize_result`); all programmatic logic.

---

## Phase 1: Extract — Intent Clarification

### Node: `clarify_intent`

Takes raw user input, extracts structured intent via LLM JSON-formatted output.

| Item | Detail |
|---|---|
| **Input** | User message + conversation history |
| **Output** | `ExtractResult`: `UserIntent` + `UserRequirements` + `HardConstraints` + `SoftConstraints` + `GroupProfile` + `TimeWindow` + `GeoConstraint` + `ChainTemplate` |
| **Routing** | LLM sets `route` field: `plan` → continue, `clarify` → END (follow-up), `reject` → capability boundary |
| **Multi-turn** | `apply_update()` merges incremental updates across turns. `MAX_CLARIFY_ITERATIONS` forces routing to `plan` or `reject` on exhaustion. |
| **Fallback** | JSON parse failure → `route="clarify"` with generic follow-up |

---

## Phase 2: Context — Weather & Geocoding

### Node: `context_agent`

Fetches environmental context before planning.

| Item | Detail |
|---|---|
| **Geocoding** | Resolves user location to lat/lng via Amap `geo` MCP tool |
| **Weather** | Fetches forecast via Amap `weather` MCP tool |
| **Fallback** | MCP failures → `weather=None`, downstream skips weather-dependent logic |

---

## Phase 3: Search — POI Discovery

### Node: `formulate_search`

LLM designs a search strategy: what categories to query, what keywords per category, what radius.

### Node: `execute_category_search`

Executes the strategy via parallel MCP calls (`batch_around_search` + `batch_search_detail`).

- Maximum **2 rounds**: if first-round coverage is insufficient, loops back to `formulate_search` via `Command(goto=...)`.
- Output: `category_pools` — dict of `POICategoryPool` (keyed by macro-category: dining, scenic, shopping, etc.)

---

## Phase 4: Plan — Dual-Agent + Fusion

### Node: `multi_agent_plan`

Two LLM agents run in parallel via `asyncio.gather`:

| Agent | Strategy | Optimization Target |
|---|---|---|
| `constraint_satisfaction` | Hard-constraint-first | Max constraint coverage + preference match |
| `spatio_temporal` | Geography-first | Min transit waste, max play time, cluster affinity |

Each outputs an `AgentPlan` (nodes with slot → poi_id → times → transit → cost → reasoning). If fewer than 2 succeed, fallback to single-agent result.

### Node: `plan_fusion`

LLM performs weighted voting across both agent plans per time-slot:

- **Consensus** (same POI) → high-weight adoption
- **Constraint conflict** → lean toward constraint agent
- **Spatial conflict** → lean toward spatio-temporal agent
- **Score conflict** → composite: α·rating + β·distance + γ·budget

Outputs `FusionResult` containing a unified `Plan` (SubTask DAG). LLM failure → `_fallback_fusion` uses Agent 1's plan with score penalty.

### Node: `qa_check`

LLM optimization pass verifying 4 dimensions:

1. **Time feasibility** — durations, transit realism, meal timing
2. **Constraint satisfaction** — budget, diet, child safety, must-visits
3. **Diversity** — no duplicate POI types
4. **Weather adaptation** — indoor preference when rainy

Fixes issues by substituting POIs from candidate pools. `modify_count` cap at 3.

---

## Phase 5: Present — HITL Gate

### Node: `present_to_user`

| Item | Detail |
|---|---|
| **Mechanism** | LangGraph `interrupt()` pauses execution |
| **Resume** | `Command(resume=<choice>)` with confirm/modify/cancel |
| **Display** | Formatted plan cards (POI name, time, cost, transit, notes) |
| **Cascading HITL** | Modify → QA → present again; CLI loops until graph completes |

---

## Phase 6: Execute — Fan-out Booking

### Node: `fan_out_bookings`

Transition node. The actual fan-out happens in the **conditional edge** `route_fanout`:

```python
def route_fanout(state):
    return [Send("book_worker", {
        "plan": plan,
        "current_task_id": st.id,
        "current_retry_count": 0,
    }) for st in plan.sub_tasks if st.type == "book" and "cancel" not in st.id.lower()]
```

### Node: `book_worker`

Single-task worker. Each instance receives `current_task_id` + `current_retry_count` from `Send.arg`. Runs a **simulated booking** (deterministic hash-based: 82% success, 10% transient timeout, 8% recoverable sold-out). Results merge back into `state.bookings` via custom `_merge_bookings` reducer.

### Node: `verify_execution`

Classifies all `BookingResult`s into buckets: `succeeded` / `failed`. Computes `execution_status`:
- All success → `"done"`
- Partial success → `"partial"`
- All failed → `"failed"`

### Node: `handle_failures`

Three-tier classification per failed booking:

| Error Type | Example | Action |
|---|---|---|
| **transient** | Network timeout, API rate limit | Retry (max 2), else escalate to fatal |
| **recoverable** | Venue full / sold out | Run `compensatory` cancel task → mark compensated |
| **fatal** | Permanent error, exhausted retries | Escalate to `notify_user` |

Retry routing: `route_after_handle_failures` returns `list[Send("book_worker", ...)]` for transient tasks.

### Node: `summarize_result`

Final report: confirmed bookings + compensated cancellations + unresolved failures.

### Node: `notify_user`

Escalation endpoint for non-retryable failures. Reports what failed and prompts user to retry manually.

---

## State Design

`AgentState` extends LangGraph `MessagesState` (inherits `messages` with `add_messages` reducer).

### Extract Phase

| Field | Type | Description |
|---|---|---|
| `extract_result` | `ExtractResult` | Aggregated intent (updated incrementally across clarify turns) |
| `clarify_iterations` | `int` | Guard against infinite clarify loop |
| `user_coords` | `str \| None` | Resolved "lng,lat" from geocoding |

### Context Phase

| Field | Type | Description |
|---|---|---|
| `weather` | `WeatherContext \| None` | Forecast data; `None` if MCP call failed |

### Search Phase

| Field | Type | Description |
|---|---|---|
| `search_strategy` | `SearchStrategy \| None` | LLM-designed search plan |
| `search_round` | `int` | Current round (max 2) |
| `category_pools` | `dict[str, POICategoryPool]` | POIs grouped by macro-category |

### Plan Phase

| Field | Type | Description |
|---|---|---|
| `plan` | `Plan \| None` | Sub-task DAG (final fused plan) |
| `agent_plans` | `list[AgentPlan]` | Raw plans from each parallel agent |
| `fusion_result` | `FusionResult \| None` | Weighted voting result |
| `plan_cards` | `list[PlanCard]` | UI-friendly display cards |
| `modify_count` | `int` | QA modify loop guard (max 3) |
| `plan_iterations` | `int` | Legacy verify→adjust guard |
| `modify_feedback` | `str` | User modification request text |

### Execution Phase

| Field | Type | Description |
|---|---|---|
| `next_action` | `str` | Routing signal |
| `bookings` | `dict[str, BookingResult]` | Merged via `_merge_bookings` reducer for fan-in |
| `execution_status` | `Literal["idle", "running", "partial", "done", "failed", "compensated"]` | Aggregate status |
| `current_task_id` | `str \| None` | Set by `Send.arg` per worker instance |
| `current_retry_count` | `int` | Set by `Send.arg` per worker instance |

### Core Data Models

```python
class SubTask:
    id: str
    type: Literal["search", "compare", "book"]
    target: str
    dependencies: list[str]
    params: dict
    compensatory: str | None       # cancel task ID for Saga compensation

class Plan:
    sub_tasks: list[SubTask]
    total_cost_estimate: float
    notes: str

class BookingResult:
    task_id: str
    status: Literal["pending", "success", "failed", "cancelled", "compensated"]
    order_id: str | None
    error: str | None
    error_type: Literal["transient", "recoverable", "fatal"] | None
    retries: int
```

---

## MCP Tool Calling Chain

```
LangGraph Node
  → mcp_session(server, url)         # streamable_http_client lifecycle
    → initialize + list_tools
    → whitelist filter (12 Amap APIs)
    → JSON Schema → Pydantic args_schema → LangChain BaseTool
  → react_loop(model, tools, ...)    # bind_tools → LLM → execute → ToolMessage → loop
    → max 5 tool call rounds
```

**Design note:** Sessions are NOT cached across nodes because LangGraph may run each node in a different asyncio task, and `anyio` cancel scopes don't cross task boundaries.

---

## Error Handling Summary

| Location | Strategy | User Perception |
|---|---|---|
| `clarify_intent` — JSON parse fail | Fallback to clarify route, generic follow-up | Perceived (re-asked) |
| `context_agent` — MCP fail | `weather=None`, downstream skips weather logic | Not perceived |
| `multi_agent_plan` — 1 agent fails | Use surviving agent's plan | Not perceived |
| `multi_agent_plan` — both fail | Empty plan, route to present (user sees degraded result) | Perceived |
| `plan_fusion` — LLM fail | `_fallback_fusion` uses Agent 1 plan | Not perceived |
| `qa_check` — LLM fail | Pass-through fused plan unchanged | Not perceived |
| `book_worker` — transient fail | Retry up to 2× via `Send()` fan-out | Not perceived (unless exhausted) |
| `book_worker` — recoverable fail | Run compensatory cancel, mark compensated | Perceived (informed in summary) |
| `book_worker` — fatal fail | Escalate to `notify_user` | Perceived (manual action needed) |
| User interrupt mid-flow | LangGraph checkpoint persists; resume from breakpoint | Perceived (resume prompt) |
| System crash | `MemorySaver` checkpoint; resume on restart | Not perceived |
