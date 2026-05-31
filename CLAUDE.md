# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync          # install dependencies
uv run finn      # run the agent
uv add <pkg>     # add a dependency
```

## Architecture

Finn is a **local short-trip planning agent** using a **DAG + ReAct hybrid** architecture:

```
LangGraph DAG (orchestration)
  └── nodes are high-level workflow steps
      └── each node internally runs a PydanticAI ReAct (think → act → observe) loop
```

- **`src/finn/graph.py`** — LangGraph `StateGraph`. Defines the DAG: nodes (workflow steps) and edges (dependencies). Currently a minimal `START → react → END`. Grows to `intent → plan → execute* → verify → respond`.
- **`src/finn/agent.py`** — PydanticAI `Agent` factory. Each agent wraps an LLM and runs a ReAct loop (tool calls, observations, final answer). Provider is OpenAI-compatible, configured via env vars.
- **`src/finn/main.py`** — Entry point. Loads env, creates agent, compiles graph, invokes.

## LLM Provider

Uses DeepSeek via OpenAI-compatible API. Configure with `.env`:

```
LLM_API_KEY=sk-xxx
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com
```

To switch providers, change `LLM_BASE_URL` and `LLM_MODEL` to any OpenAI-compatible endpoint.
