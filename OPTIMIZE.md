# Deep Research — 优化方案

> 基于当前代码库的全新分析。按影响范围分类，每项包含：现状 → 问题 → 方案。

---

## 清理项 — 死代码（低风险，立即可做）

### 1. 三个文件完全未被引用

| 文件 | 现状 | 操作 |
|------|------|------|
| `services/text_processing.py` | `strip_tool_calls()` 随旧 `[TOOL_CALL]` 模式废弃 | **删除** |
| `services/tool_events.py` | `ToolCallTracker` 无导入方 | **删除** |
| `models.py:SummaryStateInput` | 定义在模块中，无任何代码使用 | **删除** |

### 2. `notes.py` 中的死函数

```python
# build_note_guidance() 仍生成 [TOOL_CALL:note:...] 文本
# 但已无任何文件导入它（ToolRunner 接管了工具调用）
```
→ 删除 `build_note_guidance()` 函数

---

## 架构问题 — 影响维护与性能

### 3. `run_stream()` 完全脱离 LangGraph 图

**现状**：
```python
# run() 使用 Self.graph.invoke() ← 走 StateGraph
# run_stream() 完全手工编排 ← 自己写 for 循环
```

**问题**：
- `run_stream()` 无法利用 Send API 的并行能力（任务串行执行）
- `run_stream()` 中完整拷贝了 `_execute_single_task()` 的逻辑但**不完全一致**（多了 SSE yield 事件）
- 如果修改了 `_execute_single_task()`，需要同步修改 `run_stream()` 中的对应逻辑

**方案**：将 `run_stream()` 的细粒度事件通过 LangGraph 的 `StreamWriter` 或自定义回调输出，使两种模式复用同一套图逻辑。

```python
def run_stream(self, topic):
    for event in self.graph.stream(initial_state, stream_mode="custom"):
        if isinstance(event, dict) and event.get("type"):
            yield event
```

改造后 `run_stream()` 不再需要 `for task in summary_state.todo_items:` 循环，由 Send API 自动并行。

---

### 4. `load_dotenv()` 每次请求重复执行

**现状**：`Configuration.from_env()` 内部每次调用 `load_dotenv()`。

**问题**：`load_dotenv()` 解析文件 → 设置 `os.environ` → 加锁。**每个 HTTP 请求都调一次**。

**方案**：移到模块级，只执行一次。

```python
# config.py 顶部
load_dotenv()
# from_env() 中去掉 load_dotenv() 调用
```

---

### 5. 两套 State 类型，手动转换

**现状**：`LangGraphState`（TypedDict）与 `SummaryState`（dataclass）字段完全重复，每节点用 `_graph_state_to_summary()` 转换。

**问题**：
- 每节点执行一次全字段拷贝（`plan_tasks`, `execute_one`, `generate_report` 各一次）
- 新增字段需要在两个类中都添加
- `LangGraphState` 的 reducer（`operator.add`）在转换过程中丢失

**方案**：统一为 `LangGraphState` TypedDict，下掉 `SummaryState`。Services 层直接接受 TypedDict 或转换一次即可。

---

### 6. `max_web_research_loops` 配置无效

**现状**：`config.py` 中定义了 `max_web_research_loops=3`，但代码中 `research_loop_count` 只递增，从不与上限比较。

**方案**：接入循环控制，或删除该配置项。

```python
if summary_state.research_loop_count >= self.config.max_web_research_loops:
    # 已达最大轮次，跳过更多搜索
```

---

### 7. `dispatch_search()` 的 `loop_count` 参数未使用

**现状**：函数签名接收 `loop_count: int = 0`，但内部从未使用。

**方案**：删除参数。

---

## 生产就绪 — 安全性与可靠性

### 8. LLM 调用没有超时

**现状**：`ChatOpenAI()` 使用默认 timeout（无限制）。

**问题**：DeepSeek 或本地 Ollama 响应慢时，HTTP 请求会挂起直到响应，没有 fallback。

**方案**：配置化 timeout。

```python
def _create_llm(config: Configuration) -> ChatOpenAI:
    kwargs = {"temperature": 0.0, "timeout": 60}  # 默认 60s
```

或从环境变量读取 `LLM_TIMEOUT`。

---

### 9. CORS 全开

**现状**：`main.py` 中 `allow_origins=["*"]` 硬编码。

**方案**：从 `.env` 读取。

```python
origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(CORSMiddleware, allow_origins=origins, ...)
```

`.env.example` 已有 `CORS_ORIGINS` 变量，只需接入代码。

---

### 10. 日志重复初始化

**现状**：
```python
logger.add(sys.stderr, level="INFO", ...)
logger.add(sink=sys.stderr, level="ERROR", ...)  # 重复
```

**问题**：两个 handler 都写 stderr，INFO 和 ERROR 都会输出两次。

**方案**：
```python
logger.remove()  # 清除默认 handler
logger.add(sys.stderr, level="INFO", format=...)
```

---

### 11. 缺少 LLM 连通性健康检查

**现状**：`/healthz` 只检查服务器是否运行。

**方案**：增加 LLM 探活（可选，不阻塞启动）。

```python
@app.get("/healthz")
def health_check():
    llm_ok = False
    try:
        _create_llm(Configuration.from_env()).invoke([HumanMessage(content="ping")])
        llm_ok = True
    except Exception:
        pass
    return {"status": "ok", "llm": llm_ok}
```

---

## 代码健康 — 命名与清理

### 12. 项目名称仍为 "hello-agents"

**现状**：`pyproject.toml` 名称、描述、FastAPI title 仍引用 hello_agents。

```toml
name = "helloagents-deep-researcher"
description = "...powered by HelloAgents."
```

**方案**：重命名为 `deep-researcher`，更新相关描述。

### 13. 配置项描述过时

**现状**：
```python
enable_notes: Field(description="...store task progress in NoteTool")
notes_workspace: Field(description="Directory for NoteTool")
```

**方案**：更新描述，"NoteTool" → "NoteStore"。

---

## 测试覆盖缺口

### 14. Agent 图的端到端测试缺失

**现有测试**：覆盖了 NoteStore、ToolRunner、工具函数、API 端点。

**缺失**：
- `plan_tasks → _route_tasks → execute_one → generate_report` 完整图执行
- planner JSON 解析失败 → 兜底 fallback 逻辑
- `run_stream()` 的 SSE 事件序列

**推荐新增**：

```
tests/test_agent.py
├── test_plan_tasks_node      # mock planner，验证 graph output
├── test_route_tasks_send     # 多个 pending → Send 列表
├── test_route_tasks_done     # 全部 completed → "generate_report"
├── test_execute_one_skipped  # 空任务
└── test_full_graph_invoke    # mock LLM + mock search → 验证完整输出
```

---

## 优化影响矩阵

| # | 条目 | 风险 | 工作量 | 收益 |
|---|------|------|--------|------|
| 1-2 | 删除死代码 | 低 | 10min | 减少认知负担 |
| 3 | run_stream 接入 LangGraph | 中 | 4h | 并行能力 + 消除重复逻辑 |
| 4 | load_dotenv 只执行一次 | 低 | 5min | 每次请求省一次文件 I/O |
| 5 | 统一 State | 中 | 2h | 减少拷贝、消除不一致 |
| 6 | max_web_research_loops 接入 | 低 | 15min | 配置真正生效 |
| 7 | 删除无用参数 | 低 | 5min | 代码整洁 |
| 8 | LLM timeout | 低 | 10min | 防止请求挂起 |
| 9 | CORS 可配置 | 低 | 10min | 生产安全 |
| 10 | 日志去重 | 低 | 5min | 日志清晰 |
| 11 | LLM 健康检查 | 低 | 15min | 运维友好 |
| 12-13 | 重命名 | 低 | 10min | 项目标识准确 |
| 14 | 补充测试 | 低 | 2h | 回归保障 |

> **建议优先级**：先做 1-2（死代码）和 4/6/7/8/9/10（低风险改进），再做 3（大重构）和 5（中风险）。
已全部完成