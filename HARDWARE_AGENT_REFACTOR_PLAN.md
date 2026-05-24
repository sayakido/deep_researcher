# 硬件方案设计 Agent 改造方案

## 1. 当前工程判断

当前工程本质上是一个通用 Deep Research Agent，核心流程已经比较适合作为“硬件方案设计 Agent”的底座：

- `src/agent.py` 用 LangGraph 编排 `plan_tasks -> execute_one -> generate_report`，并支持并行执行子任务。
- `src/services/planner.py` 负责把用户主题拆成多个调研任务。
- `src/services/search.py` 负责联网检索和网页内容整理。
- `src/services/summarizer.py` 负责单任务总结。
- `src/services/reporter.py` 负责最终 Markdown 报告生成。
- `src/prompts.py` 集中放置规划、总结、报告提示词。
- `src/models.py` 定义 `TodoItem`、`SummaryState`、`LangGraphState` 等状态模型。
- `src/main.py` 提供 `/research` 和 `/research/stream` 两个接口。
- `src/services/notes.py` 提供文件型笔记工具，可继续用于保存方案过程和最终报告。

因此不建议推倒重写。更好的做法是保留现有 LangGraph、搜索、流式输出、笔记和 API 框架，把“通用研究”升级为“硬件方案设计”的领域化流程。

## 2. 目标产品形态

用户输入一段产品需求，例如：

> 做一个低功耗室外环境监测终端，需要采集温湿度、PM2.5、光照，支持 4G 或 LoRa 回传，电池供电，目标 BOM 控制在 150 元以内。

Agent 输出一份结构化硬件方案设计报告，至少包含：

1. 需求澄清与约束假设
2. 系统架构图或模块架构
3. 核心功能模块拆分
4. 主控、通信、传感器、电源、存储、接口等核心器件选型
5. 关键器件参数对比表
6. BOM 初估
7. 功耗、成本、尺寸、供应链、开发难度可行性分析
8. 主要风险与规避建议
9. 后续验证计划，包括原理图、PCB、样机、测试项建议
10. 信息来源与待确认事项

## 3. 推荐总体架构

保留当前三段式流程，但将语义从“研究”改成“硬件方案设计”：

```text
用户产品需求
  |
  v
需求解析与任务规划
  - 需求边界
  - 系统架构
  - 核心器件选型
  - 功耗/成本/供应链
  - 风险与验证
  |
  v
并行资料收集与模块分析
  - MCU/SoC
  - 通信模组
  - 传感器/执行器
  - 电源方案
  - 接口/结构/认证
  |
  v
方案综合
  - 推荐架构
  - 器件选型表
  - 备选方案
  - 可行性与风险
  - 下一步工程验证
```

当前 `plan_tasks -> execute_one -> generate_report` 可以直接映射到这个流程。

## 4. 关键改造点

### 4.1 Prompt 领域化

优先改 `src/prompts.py`。

建议把三个核心 prompt 从通用研究改为硬件方案设计：

- `todo_planner_system_prompt`
  - 角色改为“资深硬件系统架构师/硬件方案经理”。
  - 任务拆解固定覆盖：需求澄清、系统架构、核心器件、功耗电源、BOM 成本、风险认证、验证计划。
  - 输出仍保持 JSON，避免破坏 `PlanningService._extract_tasks()`。

- `task_summarizer_instructions`
  - 角色改为“硬件模块方案分析专家”。
  - 单任务总结不只摘录网页，而要输出：候选器件、关键参数、适用性、限制、替代料、工程风险。
  - 要求明确区分“已由资料支持”和“工程推断”。

- `report_writer_instructions`
  - 角色改为“硬件系统方案设计负责人”。
  - 最终报告模板改成硬件方案专用结构。
  - 强制输出表格：核心器件选型表、备选器件对比表、风险矩阵、验证计划表。

这是第一优先级，改动小、收益最大。

### 4.2 数据模型领域化

当前 `TodoItem` 只有 `title / intent / query / summary / sources_summary`，对于硬件方案不够表达结构化结果。

建议在 `src/models.py` 新增领域模型，不一定第一阶段就全部接入：

```python
@dataclass(kw_only=True)
class ComponentCandidate:
    category: str
    part_number: str
    manufacturer: str | None = None
    key_specs: dict[str, str] = field(default_factory=dict)
    estimated_price: str | None = None
    package: str | None = None
    availability: str | None = None
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    source_urls: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class HardwareDesignResult:
    requirements_summary: str
    architecture: str
    recommended_components: list[ComponentCandidate] = field(default_factory=list)
    alternative_components: list[ComponentCandidate] = field(default_factory=list)
    feasibility: str
    risks: list[str] = field(default_factory=list)
    validation_plan: list[str] = field(default_factory=list)
```

落地建议：

- 第一阶段先不强制 LLM 输出这些 Python 对象，只在最终报告中用 Markdown 表格表达。
- 第二阶段再让 reporter 输出 JSON + Markdown 双格式，便于前端展示和后续自动化。

### 4.3 API 命名与兼容

当前接口叫 `/research`，能用，但语义不准确。

建议保留旧接口兼容，同时新增：

- `POST /hardware/design`
- `POST /hardware/design/stream`

请求模型建议从 `ResearchRequest` 扩展为：

```python
class HardwareDesignRequest(BaseModel):
    requirement: str
    target_cost: str | None = None
    target_power: str | None = None
    production_volume: str | None = None
    preferred_regions: list[str] = []
    constraints: list[str] = []
    search_api: SearchAPI | None = None
```

第一阶段可以只接收 `requirement`，其他字段作为可选增强。

### 4.4 搜索策略增强

当前 `dispatch_search()` 是通用搜索，适合找资料，但硬件选型还需要更具体的查询策略。

建议新增 `src/services/hardware_search.py`，在不破坏原搜索服务的情况下增加硬件搜索 query 生成和结果整理：

- 器件选型查询：
  - `{需求关键词} MCU low power selection`
  - `{part_number} datasheet`
  - `{part_number} price availability`
  - `{part_number} reference design`

- 供应链/价格查询：
  - 优先检索 Digi-Key、Mouser、LCSC、立创商城、TI、ST、NXP、Nordic、Espressif、Quectel、SIMCom、Bosch、Sensirion 等官方或分销商页面。

- 风险查询：
  - `{part_number} errata`
  - `{part_number} lifecycle status`
  - `{module} certification CE FCC`

第一阶段可以仍使用当前 `dispatch_search()`，只通过 planner prompt 生成更精准的 query。
第二阶段再加入硬件站点优先级和数据源分类。

### 4.5 领域工具扩展

当前工具只有笔记 `create_note/read_note/update_note`。硬件 Agent 可以逐步增加工具：

1. `estimate_power_budget`
   - 输入各模块工作电流、睡眠电流、占空比。
   - 输出平均电流、电池续航粗估。

2. `estimate_bom_cost`
   - 输入器件列表和单价区间。
   - 输出 BOM 区间、成本风险。

3. `compare_components`
   - 输入多个候选器件。
   - 输出参数对比表。

4. `generate_architecture_mermaid`
   - 输入模块关系。
   - 输出 Mermaid 系统架构图。

这些工具可以放在 `src/services/hardware_tools.py`，再由 `DeepResearchAgent.__init__()` 合并到 `self.tools`。

### 4.6 输出模板固定化

最终报告建议固定为以下 Markdown 结构：

```markdown
# 硬件方案设计报告

## 1. 需求理解与假设

## 2. 推荐系统架构

## 3. 模块划分

## 4. 核心器件选型

| 模块 | 推荐器件 | 厂商 | 关键参数 | 选择理由 | 备选 |
|---|---|---|---|---|---|

## 5. 关键方案说明

## 6. 功耗与电源评估

## 7. BOM 成本初估

## 8. 可行性评估

## 9. 风险矩阵

| 风险 | 影响 | 概率 | 规避建议 | 验证方式 |
|---|---|---|---|---|

## 10. 后续验证计划

## 11. 资料来源

## 12. 待用户确认的问题
```

这样可以让 Agent 输出稳定，也方便前端渲染。

## 5. 建议实施阶段

### 阶段 1：最小可用版本

目标：低成本把当前 Deep Research 改成硬件方案设计 Agent。

改动：

- 改 `src/prompts.py` 三个 prompt。
- 把默认文案从“研究报告”调整为“硬件方案设计报告”。
- 在 `README.md` 或新增文档中补充硬件 Agent 用法。
- 新增几个面向硬件的测试用例，验证 planner 输出任务、reporter 生成固定章节。

预期效果：

- 用户输入产品需求后，系统能输出比较完整的硬件方案报告。
- 仍然复用当前 `/research` 和 `/research/stream`。
- 无需大改 LangGraph。

### 阶段 2：领域接口与结构化输出

目标：让产品语义和工程数据更清晰。

改动：

- 新增 `HardwareDesignRequest` 和 `HardwareDesignResponse`。
- 新增 `/hardware/design` 和 `/hardware/design/stream`。
- 在 `src/models.py` 加入 `ComponentCandidate`、`HardwareDesignResult` 等模型。
- reporter 同时输出：
  - `report_markdown`
  - `architecture_mermaid`
  - `component_table`
  - `risk_matrix`
  - `validation_plan`

预期效果：

- 前端可直接展示表格、架构图和风险矩阵。
- 后续可对器件库、BOM、功耗计算做自动化。

### 阶段 3：硬件工具与数据源增强

目标：从“会写方案”升级为“能辅助工程计算和选型”。

改动：

- 新增 `hardware_tools.py`：
  - 功耗预算估算
  - BOM 成本估算
  - 器件对比
  - 风险矩阵生成
- 新增 `hardware_search.py`：
  - datasheet 检索
  - 分销商价格检索
  - 官方资料优先级
  - 生命周期/停产风险检索
- 给 prompt 增加工具调用说明。

预期效果：

- 报告中的功耗、成本、器件对比更稳定。
- 能显式标注不确定数据来源和风险。

### 阶段 4：前端或工作流产品化

目标：从 API 工具变成可交互的硬件方案助手。

建议功能：

- 需求输入表单：产品类型、供电方式、通信方式、目标成本、工作环境、量产规模。
- 流式展示任务进度。
- 器件选型表支持导出 CSV。
- Mermaid 架构图渲染。
- 用户可对某个模块要求“换低成本方案”“换国产替代”“降低功耗”。
- 保存历史方案和迭代记录。

## 6. 推荐文件改动清单

第一阶段建议改：

- `src/prompts.py`
  - 替换规划、总结、报告 prompt。

- `src/services/planner.py`
  - fallback task 改成硬件方案默认任务，例如“需求澄清与系统架构初步设计”。

- `src/services/reporter.py`
  - prompt 中的任务汇总字段名改成硬件语义。

- `src/agent.py`
  - `_persist_final_report()` 中 note title 和 tags 从 `deep_research` 增加或替换为 `hardware_design`。

- `tests/test_agent.py`
  - 修正测试用例中的中文乱码文案。
  - 增加硬件任务 fallback 测试。

- `tests/test_api.py`
  - 第一阶段无需新增接口；第二阶段新增 `/hardware/design` 测试。

第二阶段建议新增：

- `src/services/hardware_tools.py`
- `src/services/hardware_search.py`
- `tests/test_hardware_tools.py`
- `tests/test_hardware_api.py`

## 7. Prompt 改造示例

规划 prompt 可以要求固定拆出 5 到 7 个任务：

```text
你是一名资深硬件系统架构师。请根据用户输入的产品需求，拆解出一组用于完成硬件方案设计的分析任务。

任务应覆盖：
1. 需求澄清与关键约束
2. 系统架构与模块划分
3. 主控/SoC/处理器选型
4. 通信、传感器、执行器等核心器件选型
5. 电源、功耗、热设计与可靠性
6. BOM 成本、供应链与生命周期风险
7. 样机验证、认证与量产风险

请输出 JSON：
{
  "tasks": [
    {
      "title": "...",
      "intent": "...",
      "query": "..."
    }
  ]
}
```

报告 prompt 可以要求：

```text
你是一名硬件系统方案设计负责人。请整合所有任务结果，输出一份面向产品经理、硬件工程师和采购评审的硬件方案设计报告。

必须包含：
- 需求理解与假设
- 推荐系统架构
- 模块划分
- 核心器件选型表
- 备选方案对比
- 功耗与电源评估
- BOM 成本初估
- 可行性判断
- 风险矩阵
- 验证计划
- 资料来源
- 待确认问题

对无法确认的数据必须标注“需进一步确认”，不要编造具体价格、库存或认证状态。
```

## 8. 风险与注意事项

- 硬件价格、库存、生命周期变化很快，报告必须标注检索日期和来源。
- Datasheet、参考设计、认证信息应尽量优先使用厂商官网和主流分销商。
- LLM 容易编造料号和参数，必须在 prompt 中要求“未找到可靠来源时不得给出确定结论”。
- BOM 价格只能做早期估算，不能替代正式采购报价。
- 功耗计算需要明确工作模式、占空比、温度范围和电池容量，否则只能输出区间。
- 对医疗、车规、安规、电池、射频认证等高风险领域，应输出更保守的验证建议。

## 9. 我的推荐路线

建议先做阶段 1 和阶段 2。

原因：

- 当前工程已有 LangGraph 并行调研和最终报告能力，最大短板是领域 prompt 和输出结构。
- 先把输出稳定成硬件方案报告，比一开始就做复杂器件库更快验证产品价值。
- 等输出格式稳定后，再接入功耗/BOM/器件库工具，风险更低。

最小可交付范围：

1. 替换三类 prompt 为硬件方案设计 prompt。
2. 固定最终报告模板。
3. 新增 `/hardware/design` 语义接口。
4. 增加 3 个测试样例：
   - 低功耗 IoT 终端
   - 工业控制板
   - 便携式消费电子设备
5. 输出中强制包含“待确认问题”和“风险矩阵”。

