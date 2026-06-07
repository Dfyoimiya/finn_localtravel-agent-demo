# Finn — 本地短途出行规划 Agent

**Finn** 是一个本地短途出行规划与执行 Agent，基于 **LangGraph DAG + ReAct** 混合架构，支持多轮对话、MCP 工具调用和人在回路（HITL）交互。

> Finn is a local short-trip planning & execution agent using a **LangGraph DAG + ReAct** hybrid architecture, with multi-turn dialogue, MCP tool calling, and human-in-the-loop interaction.

---

## 快速开始 | Quick Start

```bash
# 安装依赖 | Install dependencies
uv sync

# 配置 API Key | Configure API key
cp .env.example .env
# 编辑 .env 填入 LLM_API_KEY 和高德 API Key

# 运行交互式 CLI | Run interactive CLI
uv run finn        # 默认模式
uv run finn -v     # 详细日志模式 (verbose)
```

## 架构概览 | Architecture

```
START → clarify_intent → context_agent → formulate_search
       → execute_category_search → multi_agent_plan → plan_fusion
       → present_to_user [HITL interrupt]
          → confirm → fan_out_bookings → book_worker (并行)
             → verify_execution → summarize_result → END
          → modify → qa_check → present_to_user (循环)
          → cancel → END
```

- **Planning**: 双路并行 Agent（约束优先 + 时空优化）→ 加权融合 → QA 校验
- **Tool Calling**: MCP 协议连接高德地图 API（搜索/路径/天气），ReAct 循环调用
- **Exception Handling**: 预定失败三级分类（transient 重试 / recoverable 补偿 / fatal 上报）

> - **Planning**: Dual-agent parallel planning (constraint-first + spatio-temporal) → weighted fusion → QA verification
> - **Tool Calling**: MCP protocol connecting Amap APIs (search/routing/weather) via ReAct loop
> - **Exception Handling**: Three-tier booking failure classification (transient→retry, recoverable→compensate, fatal→escalate)

## 项目结构 | Project Structure

```
src/finn/
├── main.py          # 入口，CLI 循环 + checkpointing
├── graph.py         # LangGraph StateGraph 组装
├── state.py         # AgentState + 数据模型
├── nodes.py         # 节点实现 (clarify_intent, book_worker, etc.)
├── multi_planner.py # 双路 LLM Agent 并行规划
├── fusion.py        # 加权融合 + QA 校验
├── planner.py       # 传统贪心规划器 (legacy)
├── llm.py           # LLM 工厂 + ReAct 循环
├── mcp/             # MCP 客户端 + 工具转换
├── context.py       # 地理编码 + 天气上下文
├── hub.py           # MCP 批量搜索
├── cli.py           # prompt-toolkit 交互界面
└── checker.py       # 约束校验
```

## 文档 | Docs

- [架构设计](docs/architecture.md) — 完整架构说明
- [设计文档](docs/design.md) — Planning 策略、工具调用链路、异常处理机制

> - [Architecture](docs/architecture.md) — Full architecture description
> - [Design Doc](docs/design.md) — Planning strategy, tool calling chain, exception handling

## 环境变量 | Environment

| 变量 | 说明 | Description |
|---|---|---|
| `LLM_API_KEY` | LLM API 密钥 | LLM API key |
| `LLM_MODEL` | 模型名称 (默认 deepseek-v4-flash) | Model name |
| `LLM_BASE_URL` | API 地址 (默认 api.deepseek.com) | API base URL |
| `AMAP_MCP_URL` | 高德 MCP 服务地址 | Amap MCP server URL |
