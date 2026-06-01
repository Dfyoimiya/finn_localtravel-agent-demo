# Finn Agent — DAG Workflow Design

## Overview

Finn is a **local short-trip planning & execution agent**. It takes natural language
input, proactively researches options, plans an executable itinerary, presents it to
the user for approval, and then **completes the bookings/orders on the user's behalf**.

The architecture is **LangGraph DAG (orchestration) + PydanticAI ReAct (per-node execution)**.

---

## Design Principles

| Principle | Source | Meaning |
|---|---|---|
| **Plan-Execute-Verify-Replan** | VMAO (arXiv:2603.11445) | Cyclic verification with correction loops, not one-shot planning |
| **Route first, plan lazy** | "Two-Speed" Architecture | Classify intent cheaply; only run heavy planning for multi-step tasks |
| **User approves execution, not planning** | AutoGen / CrewAI consensus | Show the plan; user clicks approve; agent executes |
| **Saga compensation** | Conductor / Temporal pattern | Every side-effect node has a registered compensating node |
| **Sync checkpointing** | LangGraph | Persist state before every external call; resume from crash |
| **Fan-out / Fan-in** | LLMCompiler (arXiv:2312.04511) | Independent sub-tasks run in parallel, results aggregated |

---

## DAG Topology

```
                                START
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  clarify_intent  │  Node 1
                         └────────┬────────┘
                                  │
                             ┌────┴────┐
                             │  route  │  Edge A
                             └────┬────┘
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
               ┌────────┐  ┌─────────────┐  ┌────────┐
               │ simple │  │ decompose   │  │ reject │  Node 2a/2b/2c
               │ answer │  │ _and_plan   │  │        │
               └───┬────┘  └──────┬──────┘  └───┬────┘
                   │              │              │
                   ▼              ▼              ▼
                RESPOND     ┌─────────┐      RESPOND
                            │ verify  │  Node 3
                            │ _plan   │
                            └────┬────┘
                                 │
                            ┌────┴────┐
                            │  pass?  │  Edge B
                            └────┬────┘
                           no    │ yes
                            │    ▼
                            │ ┌──────────────┐
                            │ │ present_to   │  Node 4  ★ HITL interrupt
                            │ │ _user        │
                            │ └──────┬───────┘
                            │        │
                            │   ┌────┴────┐
                            │   │ approve?│  Edge C
                            │   └────┬────┘
                            │   no   │ yes
                            │    │   ▼
                            │    │ ┌────────────────┐
                            │    │ │ execute        │  Node 5  ★ fan-out → fan-in
                            │    │ │ _bookings      │
                            │    │ │ ┌────────────┐ │
                            │    │ │ │ book_* × N │ │  (parallel ReAct agents)
                            │    │ │ └────────────┘ │
                            │    │ └───────┬────────┘
                            │    │         │
                            │    │    ┌────┴────┐
                            │    │    │ verify  │  Edge D
                            │    │    │ _exec   │
                            │    │    └────┬────┘
                            │    │         │
                            │    │    ┌────┴──────────┐
                            │    │    │                 │
                            │    │  all_ok         partial_fail
                            │    │    │                 │
                            │    │    ▼                 ▼
                            │    │ ┌──────────┐  ┌──────────────┐
                            │    │ │summarize │  │ handle       │  Node 6a/6b
                            │    │ │_result   │  │ _failures    │
                            │    │ └────┬─────┘  └──────┬───────┘
                            │    │      │               │
                            │    │      ▼          ┌────┴────┐
                            │    │   RESPOND       │retryable?│ Edge E
                            │    │                 └────┬────┘
                            │    │              yes     │ no
                            │    │               │      ▼
                            │    │               │   NOTIFY
                            │    │               │   _USER
                            │    │               │      │
                            │    │               │      ▼
                            │    │               │    RESPOND
                            │    │               │
                            │    │               ▼
                            │    │          execute_bookings
                            │    │          (retry failed only)
                            │    │               │
                            │    └───────────────┘
                            │
                            ▼
                      ┌──────────┐
                      │ adjust   │  Node 7
                      │ _plan    │
                      └────┬─────┘
                           │
                           ▼
                       verify_plan
                       (re-loop)
```

---

## Node Specifications

### Node 1: `clarify_intent` — Disambiguation & Information Gathering

| Item | Detail |
|---|---|
| **Type** | ReAct agent (can ask user follow-up questions) |
| **Input** | Raw user natural language |
| **Output** | Structured intent: `{goal, date, location, budget, preferences, missing_fields}` |
| **Behavior** | If information is missing (no date, no party size), actively ask the user. Loop until key fields are collected or user declines to provide more. |
| **Exit condition** | Required fields complete OR user says "whatever" OR 3 rounds of follow-up reached |

**Rationale:** Trip planning inherently requires structured parameters (date, location, party size, budget). Collect them upfront so downstream nodes never need to ask again.

---

### Edge A: `route` — Intent Classification

```
if intent is simple Q&A (weather, reviews, distance):
    → simple_answer
elif intent is trip planning (arrange, book, plan an outing):
    → decompose_and_plan
else:
    → reject (inform user of capability boundaries)
```

Uses rule-based matching on the `goal` field from `clarify_intent`. No LLM needed for routing.

---

### Node 2b: `decompose_and_plan` — Task Decomposition & Planning

The core planning node. Takes structured intent, outputs an executable sub-task DAG.

| Item | Detail |
|---|---|
| **Type** | ReAct agent + tool calls |
| **Tools** | Search venues, check hours, check pricing, check routes, check weather |
| **Input** | `{goal, date, location, budget, preferences}` |
| **Output** | `Plan { sub_tasks[], dependencies, estimated_cost, alternatives }` |

**Sub-task structure:**
```python
class SubTask:
    id: str
    type: Literal["search", "compare", "book"]
    target: str          # e.g. "dinner" | "movie_ticket" | "taxi"
    dependencies: list[str]  # IDs of sub-tasks this one depends on
    params: dict
    compensatory: str | None  # ID of compensating task
```

**Constraints:**
- Maximum 8 sub-tasks (prevents explosion)
- `search` tasks have no dependencies; can run in parallel
- `book` tasks depend on their corresponding `search` completing
- Every `book` task must have a `compensatory` (cancel/refund)

---

### Node 3: `verify_plan` — Plan Validation

| Item | Detail |
|---|---|
| **Type** | Pure LLM evaluation (no tool calls). Use a different model to avoid self-consistency bias. |
| **Input** | `Plan + UserIntent` |
| **Output** | `Verification { score: 0-1, issues: [...], status: pass | fix | reject }` |

**Validation dimensions:**

| Dimension | Check |
|---|---|
| Temporal feasibility | Do sub-task times conflict? Are transit times realistic? |
| Budget feasibility | Does the total fall within budget? |
| Logical consistency | Dinner → movie: does the movie start after dinner ends? |
| Completeness | Are all user requirements covered? Anything missing? |
| Business hours | Are booked venues actually open at the planned time? |

---

### Edge B: `verify_plan` Routing

```
if status == "pass" AND score >= 0.7:
    → present_to_user
else:
    → adjust_plan (carrying the issues list)
```

---

### Node 4: `present_to_user` — Plan Confirmation ★ HITL

| Item | Detail |
|---|---|
| **Type** | LangGraph `interrupt()` static breakpoint |
| **Behavior** | Format the plan for display, pause execution, wait for user action |
| **User actions** | Confirm / Modify / Cancel |

**Example display format:**
```
Your Weekend Plan

Saturday June 7
  12:00  Lunch @ Sushi Ichi (reservations available)  — ¥150/person
  14:30  Movie "XXX" @ MixC Cinema                     — ¥60/person
  17:00  Coffee @ %Arabica                             — ¥40/person

Total: ¥250/person × 2 people = ¥500

[Confirm] [Modify] [Cancel]
```

---

### Edge C: `present_to_user` Routing

```
if confirmed:
    → execute_bookings
elif modify:
    → adjust_plan (carrying modification requests)
else:
    → END (no side effects)
```

---

### Node 5: `execute_bookings` — Parallel Execution ★

Fan-out → fan-in pattern. Each booking task runs as an independent ReAct agent.

```
                    execute_bookings
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         book_table   book_ticket   book_activity
         (ReAct)      (ReAct)       (ReAct)
              │            │            │
              └────────────┼────────────┘
                           ▼
                      aggregate
```

| Item | Detail |
|---|---|
| **Implementation** | LangGraph `Send()` API, one `Send` per sub-task |
| **Each book_* internally** | PydanticAI ReAct loop: navigate → select → fill → submit → confirm |
| **Parallelism** | All independent book tasks execute concurrently |
| **Compensation** | Each `book_*` has a registered `cancel_*` node triggered on failure |
| **Safety** | Every external call carries an `idempotency_key` |

---

### Edge D: `verify_execution` Routing

```
if all succeeded:
    → summarize_result
elif partial success AND retryable:
    → handle_failures → execute_bookings (retry failed only)
elif all failed OR non-retryable:
    → notify_user (report failure, auto-cancel succeeded items)
```

---

### Node 6b: `handle_failures` — Failure Recovery

| Failure Type | Strategy |
|---|---|
| Network timeout / rate limit | Exponential backoff retry (max 3) |
| Venue full / sold out | Trigger `alternatives` query for substitutes |
| Payment failed | Notify user, pause for input |
| Permission / parameter error | Immediate abort, compensate all succeeded orders |

---

### Edge E: `handle_failures` Routing

```
if retryable:
    → execute_bookings (retry failed tasks only)
else:
    → notify_user → END
```

---

### Node 7: `adjust_plan` — Plan Correction

Receives `issues` from `verify_plan` or modification requests from user. Adjusts the plan and re-enters `verify_plan`.

| Adjustment Type | Strategy |
|---|---|
| `verify_plan` failed | Targeted fix based on issues (change time, venue, order) |
| User modification | Merge user changes → re-plan affected sub-tasks |
| 3rd attempt still fails | Abort auto-correction; present best partial plan to user |

**Loop limit:** `decompose → verify → adjust` loops at most 3 times to prevent infinite cycles.

---

## State Design

```python
class Intent(TypedDict):
    goal: str
    date: str | None
    location: str | None
    budget: float | None
    preferences: list[str]
    missing_fields: list[str]

class SubTask(TypedDict):
    id: str
    type: Literal["search", "compare", "book"]
    target: str
    dependencies: list[str]
    params: dict
    compensatory: str | None

class Plan(TypedDict):
    sub_tasks: list[SubTask]
    total_cost: float
    notes: str

class Verification(TypedDict):
    score: float           # 0-1
    issues: list[str]
    status: Literal["pass", "fix", "reject"]

class BookingResult(TypedDict):
    task_id: str
    status: Literal["pending", "success", "failed", "cancelled"]
    order_id: str | None
    error: str | None

class AgentState(TypedDict):
    # === User input ===
    user_input: str
    messages: list          # full conversation (LangGraph add_messages reducer)

    # === Clarify phase ===
    intent: Intent | None

    # === Plan phase ===
    plan: Plan | None
    verification: Verification | None
    plan_iterations: int    # correction count, max 3

    # === Execution phase ===
    bookings: dict[str, BookingResult]  # task_id → result
    execution_status: Literal["idle", "running", "partial", "done", "failed"]

    # === Control ===
    next_action: str        # set by edge functions for routing
```

---

## Error Handling Summary

| Error Location | Strategy | User Perception |
|---|---|---|
| `clarify` — insufficient info | Ask up to 3 rounds, then proceed with defaults | Perceived (follow-up questions) |
| `decompose` — cannot plan | Fallback to simple recommendations + notice | Perceived (informed) |
| `verify` — plan fails | Auto-correct up to 3 iterations | Not perceived |
| `execute` — partial failure | Compensate + retry + alternatives | Perceived (informed) |
| `execute` — total failure | Compensate all completed items + notify | Perceived (informed) |
| User interruption mid-flow | LangGraph checkpoint persists state; resume later | Perceived (resume prompt) |
| System crash | Sync checkpoint already persisted; resume on restart | Not perceived |

---

## Implementation Roadmap

1. **`clarify_intent` + `route`** — Structured intent extraction, simple vs complex routing
2. **`decompose_and_plan` + `verify_plan` + loop** — Core planning cycle
3. **`present_to_user` (HITL)** — User confirmation gate
4. **`execute_bookings` fan-out** — Parallel execution (mock tools first)
5. **`handle_failures` + compensation** — Error recovery loop

---

## References

- VMAO: Verified Multi-Agent Orchestration — arXiv:2603.11445
- LLMCompiler: Parallel Function Calling — arXiv:2312.04511
- TaskWeaver: Code-First Agent Framework — arXiv:2311.17541
- LangGraph: [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)
- CrewAI: [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)
- AutoGen: [microsoft/autogen](https://github.com/microsoft/autogen)
