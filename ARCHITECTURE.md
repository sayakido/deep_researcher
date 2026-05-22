# Deep Research — 系统架构文档

> LangGraph 驱动的深度研究助手。自动拆解主题为调研子任务，多任务并行执行网络检索与 LLM 分析，生成结构化研究报告。

---

## 技术栈

| 类别 | 技术 |
|------|------|
| 框架 | FastAPI + LangGraph |
| LLM | `langchain-openai`（兼容 DeepSeek / Ollama / LMStudio / 任意 OpenAI 兼容 API） |
| 搜索引擎 | DuckDuckGo / Tavily / Perplexity / SearXNG |
| 工具调用 | LangChain `@tool` + `ToolRunner`（自动 tool call 循环） |
| 配置 | `python-dotenv` + Pydantic `BaseModel` |
| 测试 | pytest（34 tests） |
| 代码质量 | ruff + mypy |

---

## 项目结构

```
backend/
├── .env                     # 环境变量
├── .env.example             # 配置模板
├── pyproject.toml           # 依赖与工具配置
├── ARCHITECTURE.md          # 本文档
│
├── src/                     # Python 包（src 布局）
│   ├── __init__.py          # 包入口 + sys.path 修正
│   ├── main.py              # FastAPI 应用 & HTTP 端点
│   ├── agent.py             # LangGraph 编排器（核心）
│   ├── config.py            # 统一配置（env → Pydantic）
│   ├── models.py            # 数据模型 & LangGraph State
│   ├── prompts.py           # 全部中文 Prompt
│   ├── utils.py             # 工具函数
│   │
│   └── services/
│       ├── __init__.py
│       ├── planner.py       # 主题 → 调研任务拆解
│       ├── search.py        # 网络检索（多后端分发）
│       ├── summarizer.py    # 单任务总结（同步+流式）
│       ├── reporter.py      # 最终报告生成
│       ├── notes.py         # NoteStore（文件持久化）+ @tool 工具函数
│       └── tool_runner.py   # ToolRunner（tool call 循环引擎）
│
└── tests/
    ├── __init__.py
    ├── conftest.py          # 共享 fixtures（MockLLM / NoteStore）
    ├── test_agent.py        # LangGraph 图路由 + 节点测试
    ├── test_api.py          # FastAPI TestClient 端点测试
    ├── test_notes.py        # NoteStore CRUD
    ├── test_search.py       # 搜索引擎分发测试
    ├── test_tool_runner.py  # ToolRunner 工具循环 + 防无限循环
    └── test_utils.py        # strip_thinking / format_sources 等
```

---

## 架构图

```
┌──────────────────────────────────────────────────────────────────────────┐
│  FastAPI (main.py)                                                       │
│                                                                          │
│  POST /research ──▶ DeepResearchAgent.run()                             │
│  POST /research/stream ──▶ DeepResearchAgent.run_stream()               │
│  GET /healthz ──▶ {"status":"ok", "llm": true/false}                    │
└───────────────────────┬──────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  DeepResearchAgent (agent.py)                                           │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  LangGraph StateGraph                                               │ │
│  │                                                                     │ │
│  │  START ──▶ plan_tasks ──┐                                           │ │
│  │              │           │                                          │ │
│  │              ▼           │  [Send API 并行]                          │ │
│  │         _route_tasks ────┤                                          │ │
│  │              │           │                                          │ │
│  │              ├── Send(task_1) ──▶ execute_one ─┐                    │ │
│  │              ├── Send(task_2) ──▶ execute_one ─┤                    │ │
│  │              ├── Send(task_3) ──▶ execute_one ─┼──▶ generate_report │ │
│  │              └── Send(task_4) ──▶ execute_one ─┘         │          │ │
│  │                                                    │                  │ │
│  │                 全部完成 ──────────────────────────┘                  │ │
│  │                                                         │            │ │
│  │                                                    END               │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  依赖服务:                                                                │
│    planner.py      — LLM + ToolRunner → 任务拆解                         │
│    search.py       — dispatch_search()  → 网络检索                       │
│    summarizer.py   — LLM + ToolRunner → 任务总结（同步/流式）           │
│    reporter.py     — LLM + ToolRunner → 报告生成                        │
│    notes.py        — NoteStore 持久化 + @tool 工具函数                  │
│    tool_runner.py  — ToolRunner (tool call 循环引擎)                     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 核心工作流

### 1. 规划阶段 — `plan_tasks`

```
用户输入 topic
    │
    ▼
PlanningService.plan_todo_list(topic)
    │
    ├─ SystemMessage(todo_planner_system_prompt)
    ├─ HumanMessage(todo_planner_instructions)
    │
    ▼
ToolRunner.run([SystemMessage, HumanMessage])
    │  ┌─ LLM.bind_tools([create_note, ...])  ← 可选创建笔记
    │  └─ 自动处理 tool call 循环
    │
    ▼
响应文本 → JSON 解析 → List[TodoItem]
```

- 输出格式：`{"tasks": [{"title", "intent", "query"}, ...]}`
- 解析失败则生成一个兜底任务
- 所有 Prompt 为中文

### 2. 执行阶段 — `execute_one`（并行）

_route_tasks 为每个 pending 任务创建 Send，LangGraph 并行调度所有 execute_one 节点：_

```
foreach task in parallel:
    │
    ├── dispatch_search(task.query, config)
    │    ├─ duckduckgo → DDGS.text()
    │    ├─ tavily     → TavilyClient.search()
    │    ├─ perplexity → OpenAI(api_key, base_url).chat()
    │    └─ searxng    → requests.get(searxng_url/search)
    │
    ├── prepare_research_context(result)
    │    └─ 去重 + 格式化 + 可选全文抓取
    │
    └── SummarizationService.summarize_task(state, task, context)
         ├─ ToolRunner.run()  → 同步完整总结
         │     └─ LLM 自动调用 create_note / update_note 持久化任务笔记
         │
         └─ ToolRunner.stream() → 流式 chunk 输出（run_stream 模式）
```

**同步模式（`run`）**：LangGraph `invoke()` 自动编排全图
**流式模式（`run_stream`）**：手工编排，每步 yield SSE 事件

### 3. 报告阶段 — `generate_report`

```
collect all task.summary + task.sources_summary
    │
    ▼
ReportingService.generate_report(state)
    │
    ├─ 汇总所有任务结果为一组 messages
    ├─ ToolRunner.run() → LLM 生成结构化 Markdown
    │     └─ 完成后自动调用 create_note(type=conclusion) 持久化最终报告
    │
    ▼
返回报告文本 → 更新 state.running_summary
```

---

## LangGraph State

```python
class LangGraphState(TypedDict):
    research_topic: str
    todo_items: Annotated[List[TodoItem], operator.add]               # reducer: 合并
    web_research_results: Annotated[List[str], operator.add]          # reducer: 合并
    sources_gathered: Annotated[List[str], operator.add]              # reducer: 合并
    research_loop_count: Annotated[int, operator.add]                 # reducer: 累加
    running_summary: str
    structured_report: str | None
    report_note_id: str | None
    report_note_path: str | None
```

所有 `Annotated[T, operator.add]` 字段在 Send 并行分支返回时自动聚合：
- `todo_items`：各分支返回 `[task]`，合并为完整任务列表
- `research_loop_count`：各分支返回 `1`，累加为总轮数

`SummaryState`（dataclass）与 `LangGraphState` 结构对应，通过 `_graph_state_to_summary()` 转换，用于下游 Services 层。

---

## ToolRunner — 工具调用引擎

```
messages → LLM.bind_tools([create_note, update_note, read_note])
            │
            ▼
        LLM.invoke(messages)
            │
            ├── 无 tool_calls → 返回 response.content
            │
            └── 有 tool_calls → 遍历执行：
                    │
                    ├─ Tool.invoke(args)  → NoteStore.create/read/update
                    ├─ ToolMessage 追加到 messages
                    │
                    ▼
                重新调用 LLM（最多 6 轮）
                     │
                     └── 无 tool_calls → 返回/流式输出
```

### NoteStore 工具

| 工具 | 功能 | 参数 |
|------|------|------|
| `create_note` | 创建任务笔记 | title, note_type, tags, content |
| `read_note` | 读取已有笔记 | note_id |
| `update_note` | 更新笔记内容 | note_id, content |

---

## LLM 适配层

```python
def _create_llm(config: Configuration) -> ChatOpenAI:
    provider = config.llm_provider
    if provider == "ollama":
        # http://localhost:11434/v1  + api_key="ollama"
    elif provider == "lmstudio":
        # http://localhost:1234/v1  + 可选 api_key
    else:
        # 通用 OpenAI 兼容（DeepSeek 等）
        # base_url + api_key 均来自 .env
```

LLM 调用带有 `timeout=120` 秒，防止请求挂起。

| 环境变量 | 示例 | 说明 |
|---------|------|------|
| `LLM_PROVIDER` | `custom` | 使用通用 OpenAI 兼容模式 |
| `LLM_MODEL_ID` | `deepseek-chat` | 模型名称 |
| `LLM_BASE_URL` | `https://api.deepseek.com` | API 地址 |
| `LLM_API_KEY` | `sk-xxx` | API 密钥 |

---

## 搜索引擎适配

各后端实现在 `services/search.py`，通过 `SearchAPI` 枚举分发：

| 环境变量 | 值 | 说明 |
|---------|---|------|
| `SEARCH_API` | `tavily` | 搜索引擎后端 |
| `TAVILY_API_KEY` | `tvly-xxx` | Tavily 密钥 |
| `PERPLEXITY_API_KEY` | `pplx-xxx` | Perplexity 密钥 |
| `SEARXNG_BASE_URL` | `http://localhost:4000` | SearXNG 地址 |

DuckDuckGo 无需 API Key，适合开发调试。

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/healthz` | 健康检查 → `{"status":"ok", "llm": bool}`（含 LLM 连通性探测） |
| `POST` | `/research` | 同步研究 → `ResearchResponse` |
| `POST` | `/research/stream` | SSE 流式研究 |

### SSE 事件类型

| type | 说明 |
|------|------|
| `status` | 进度消息（初始化、搜索通知等） |
| `todo_list` | 规划后的任务列表 |
| `task_status` | 单任务状态（in_progress / skipped / completed） |
| `sources` | 搜索结果来源概览 |
| `task_summary_chunk` | 任务总结的流式文本块 |
| `final_report` | 最终完整报告 |
| `done` | 全部完成 |
| `error` | 错误信息 |

---

## 配置优先级

```
os.environ > .env 文件 > 代码默认值
```

`load_dotenv()` 在模块加载时执行一次（非每次请求）。`Configuration.from_env(overrides=...)` 支持 API 请求级覆盖（如切换搜索引擎）。

---

## Send API 并行机制

```
plan_tasks 结束
    │
    ▼
_route_tasks 返回 N 个 Send 对象
    │
    ├── 分支 A: execute_one(task_1)  ─┐
    ├── 分支 B: execute_one(task_2)  ─┤  同时运行
    ├── 分支 C: execute_one(task_3)  ─┤  互不等待
    └── 分支 D: execute_one(task_4)  ─┘
    │
    ▼
所有分支完成后 → 触发 generate_report
```

关键特性：
- LangGraph 引擎跟踪所有活跃分支，**全部完成后再推进到下一节点**
- reducer `operator.add` 自动聚合各分支的增量更新
- 每个 Send 接收独立子状态副本，无竞态问题

---

## 测试

```
tests/
├── conftest.py           # MockChatModel + NoteStore fixtures
├── test_agent.py         # LangGraph 图路由、节点、Send 路径 (7 tests)
├── test_api.py           # FastAPI TestClient 端点验证 (5 tests)
├── test_notes.py         # NoteStore CRUD (6 tests)
├── test_search.py        # 搜索引擎分发 (1 test)
├── test_tool_runner.py   # ToolRunner 工具循环 + max_turns 保护 (4 tests)
└── test_utils.py         # strip_thinking_tokens / format_sources (11 tests)
```

运行：`uv run pytest tests/`

---

## 启动方式

```bash
# 1. 配置
cp .env.example .env    # 编辑填入真实 API Key
uv sync                 # 安装依赖

# 2. 运行（二选一）
uv run uvicorn src.main:app --reload --port 8000
uv run python src/main.py
```
