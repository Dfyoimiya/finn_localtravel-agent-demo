# Finn 设计文档

## 1. Planning 策略

Finn 采用 **双路并行规划 + 加权融合** 的三阶段流水线：

### 阶段一：并行 Agent 规划

`multi_agent_plan` 节点同时启动两个 LLM Agent，接收相同的用户约束但不同的策略 Prompt，通过 `asyncio.gather` 并发执行：

| Agent | 策略 | 硬约束 | 优化目标 |
|---|---|---|---|
| **constraint_satisfaction** | 约束优先 | 必去 POI、必吃菜系、预算上限、饮食禁忌、儿童安全 | 最大化约束满足率 + 偏好匹配度 |
| **spatio_temporal** | 时空优化 | 同上 | 最小化无效移动、最大化有效游玩时长，偏好地理聚集和步行可达 |

每个 Agent 输出结构化 JSON 行程（slot → poi_id → 起止时间 → 交通方式 → 费用 → 推理）。若某 Agent 失败，系统降级使用存活 Agent 的结果。

### 阶段二：加权融合

`plan_fusion` 节点通过 LLM 对两个 Agent 的每个时间槽位进行加权投票：

- **共识**（两 Agent 选同一 POI）→ 高分采纳
- **约束冲突**→ 倾向 constraint_satisfaction 的选择
- **空间冲突**→ 倾向 spatio_temporal 的选择
- **评分冲突**→ 综合评分 = α·rating + β·距离 + γ·预算拟合

输出 `FusionResult`（含 Plan DAG + 每 Agent 投票计数）。

### 阶段三：QA 校验与修正

`qa_check` 节点校验融合结果的可行性：

1. **时间可行性** — 停留时长、交通耗时、用餐时段是否合理
2. **约束遵守** — 预算、饮食、儿童友好、必去项逐一核对
3. **多样性** — 同类型 POI 不重复出现
4. **天气适配** — 雨天倾向室内场所

发现问题时从候选池替换更优 POI 并调整时序。`modify_count` 上限 3 次，防止无限循环。

### HITL 修改循环

用户可在 `present_to_user` 中断点选择 modify → 触发 `qa_check` 重新优化 → 再次呈现。修改循环独立的 `modify_count` 计数器与规划迭代分离。

---

## 2. 工具调用链路

### MCP 连接生命周期

```
LangGraph Node 启动
  → mcp_session(server_name, server_url)
    → streamable_http_client 连接
    → initialize 握手
    → list_tools 获取工具清单
    → 白名单过滤 (MCP_TOOL_FILTERS)
    → JSON Schema → Pydantic args_schema 转换
    → 包装为 LangChain BaseTool
    → yield 工具列表给节点
  → 节点使用完毕后关闭连接
```

**设计要点**：
- 每个 LangGraph Node 独立打开/关闭 MCP 连接，不跨节点缓存。因为 LangGraph 可能在不同 asyncio task 中运行各节点，`anyio` cancel scope 绑定到特定 task，跨 task 复用会导致连接失效。
- 工具白名单 `MCP_TOOL_FILTERS` 仅暴露高德地图 12 个核心 API（搜索、地理编码、路径规划、天气）。
- `_MCPTool` 包装器将所有参数强制转为字符串，兼容 LLM 输出的 int/float 类型。

### ReAct 工具调用循环

```
react_loop(model, tools, system_msg, user_msg, max_tool_calls=5):
  1. bind_tools(tools) → LLM
  2. LLM 响应：
     a. 无 tool_calls → 返回文本，循环终止
     b. 有 tool_calls → 逐个执行匹配的工具
        → 追加 ToolMessage(results) 到对话
        → 回到步骤 2
  3. 达到 max_tool_calls 上限 → 强制 LLM 输出最终答案（不绑工具）
```

### 批量 POI 搜索

`mcp_hub.batch_around_search` 将多个关键词并行发送到高德周边搜索 API，`batch_search_detail` 并行获取 POI 详情。`execute_category_search` 节点支持最多 2 轮搜索：首轮覆盖不足时通过 `Command(goto=...)` 回到 `formulate_search` 重新设计搜索策略。

---

## 3. 异常处理机制

### 预定失败分级处理

```
verify_execution → 分类各 BookingResult
  │
  ├─ success → 直接进入 summarize_result
  │
  └─ failed → handle_failures 分类：
       ├─ transient（网络超时、临时 API 故障）
       │   → retry_count < MAX_RETRIES(2)? → Send() 重新推入 book_worker
       │   → retry_count >= 2?           → 升级为 fatal
       │
       ├─ recoverable（满员/售罄）
       │   → SubTask.compensatory 存在? → 执行补偿取消子任务，标记 compensated
       │   → 无 compensatory?          → 升级为 fatal
       │
       └─ fatal（永久性失败）
           → 收集到 notify_user → 告知用户手动处理
```

重试通过 LangGraph `Send()` fan-out 实现：`route_after_handle_failures` 返回 `list[Send("book_worker", {task_id, retry_count})]`，所有重试子任务并发执行。

### 规划降级链路

```
multi_agent_plan: 两 Agent 均失败 → 用单 Agent 结果或空 Plan
       ↓
plan_fusion: LLM 调用失败 → _fallback_fusion 直接用 Agent1 结果（扣分）
       ↓
qa_check: LLM 调用失败 → 透传融合结果不做修改
```

### 提取阶段保护

- `clarify_intent` JSON 解析失败 → 降级为 `route="clarify"`，返回通用追问
- `MAX_CLARIFY_ITERATIONS` 上限 → 强制路由 `plan`（有摘要时）或 `reject`
- 天气/地理编码 MCP 失败 → `weather=None`，下游节点自动跳过天气相关过滤逻辑
