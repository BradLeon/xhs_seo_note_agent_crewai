# Implement OptimizationStrategist Crew

**Status:** deployed
**Created:** 2025-11-24
**Updated:** 2025-11-24

## Summary

实现 OptimizationStrategist crew，将 GapReport 中识别的差距和弱特征转化为可执行的内容优化方案 (OptimizationPlan)，包含具体的标题、内容、话题修改建议和图片/视频优化 prompt。

## Motivation

这是内容理解和建议 agent 子系统的最后一个环节。前置任务已完成：
- CompetitorAnalyst: 产出 SuccessProfileReport（竞品成功模式分析）
- OwnedNoteAuditor: 产出 AuditReport（自有笔记诊断报告）
- GapFinder: 产出 GapReport（差距分析报告）

现在需要 OptimizationStrategist 将这些分析结果转化为**具体的、可执行的**优化建议，而不仅仅是"建议性"的泛泛之谈。

### 核心问题
回答用户的第三个问题："我该如何修改我的 meta_data（文案、封面）来提升具体指标？"

## Proposed Changes

### Files to Create

1. **`openspec/changes/0006-optimization-strategist/proposal.md`**
   - 本文件 - 完整提案文档

2. **`openspec/changes/0006-optimization-strategist/tasks.md`**
   - 实施任务清单

3. **`src/xhs_seo_optimizer/config/agents_optimization.yaml`**
   - OptimizationStrategist agent YAML 配置
   - 角色：优化策略师 (Optimization Strategist)
   - 目标：将差距分析转化为可执行的优化方案

4. **`src/xhs_seo_optimizer/config/tasks_optimization.yaml`**
   - 3个顺序任务配置:
     - `generate_text_optimizations`: 生成文本优化 (title, content, hashtags)
     - `generate_visual_prompts`: 生成视觉优化 prompt (cover, inner images)
     - `compile_optimization_plan`: 汇总生成最终 OptimizationPlan

5. **`src/xhs_seo_optimizer/crew_optimization.py`**
   - 主 Crew 类，遵循现有模式 (crew_gap_finder.py)
   - 使用 @CrewBase, @agent, @task, @crew 装饰器

6. **`tests/test_optimization_strategist.py`**
   - 测试文件，验证 crew 执行和输出格式

### Files to Modify

1. **`src/xhs_seo_optimizer/models/reports.py`**
   - 扩展 `OptimizationPlan` 模型（目前只是占位符）
   - 添加 `OptimizationItem`, `TitleOptimization`, `ContentOptimization`, `VisualOptimization` 等嵌套模型

2. **`src/xhs_seo_optimizer/models/__init__.py`**
   - 导出新模型

## Data Model Design

### OptimizationItem (单个优化项)
```python
class OptimizationItem(BaseModel):
    original: str                      # 原始内容
    optimized: str                     # 优化后内容
    rationale: str                     # 优化理由 (基于 GapReport)
    targeted_metrics: List[str]        # 针对的指标 (ctr, comment_rate, etc.)
    targeted_weak_features: List[str]  # 针对的弱特征
```

### TitleOptimization (标题优化)
```python
class TitleOptimization(BaseModel):
    alternatives: List[OptimizationItem]  # 3个备选标题
    recommended_index: int                 # 推荐选择 (0-2)
    selection_rationale: str               # 推荐理由
```

### ContentOptimization (内容优化)
```python
class ContentOptimization(BaseModel):
    opening_hook: OptimizationItem    # 开头钩子优化
    ending_cta: OptimizationItem      # 结尾互动召唤优化
    hashtags: OptimizationItem        # 话题标签优化
    body_improvements: List[str]      # 正文改进要点（非完整重写）
```

### VisualPrompt (视觉优化 Prompt)
```python
class VisualPrompt(BaseModel):
    image_type: str              # cover | inner_1 | inner_2 | ...
    prompt_text: str             # AIGC 生成 prompt (中文)
    style_reference: str         # 参考风格描述
    key_elements: List[str]      # 必须包含的元素
    color_scheme: str            # 推荐色彩方案
    targeted_metrics: List[str]  # 针对的指标
```

### VisualOptimization (视觉优化)
```python
class VisualOptimization(BaseModel):
    cover_prompt: VisualPrompt              # 封面图优化 prompt
    inner_image_prompts: List[VisualPrompt] # 内页图优化 prompts (可选)
    general_visual_guidelines: List[str]    # 通用视觉指南
```

### OptimizationPlan (完整优化方案)
```python
class OptimizationPlan(BaseModel):
    keyword: str
    owned_note_id: str

    # 优化内容
    title_optimization: TitleOptimization
    content_optimization: ContentOptimization
    visual_optimization: VisualOptimization

    # 摘要
    priority_summary: str                  # 优先执行的优化项
    expected_impact: Dict[str, str]        # 预期影响 {metric: improvement_description}

    # 元数据
    plan_timestamp: str
```

## Implementation Plan

### Step 1: 扩展数据模型 (reports.py)
- 添加 OptimizationItem, TitleOptimization, ContentOptimization, VisualPrompt, VisualOptimization
- 扩展 OptimizationPlan 模型
- 添加字段验证器

### Step 2: 创建 Agent 配置 (agents_optimization.yaml)
- 定义 optimization_strategist agent
- 角色、目标、背景故事

### Step 3: 创建 Task 配置 (tasks_optimization.yaml)
- generate_text_optimizations: 基于 GapReport 生成文本优化
- generate_visual_prompts: 基于 GapReport 生成视觉 prompt
- compile_optimization_plan: 汇总生成最终方案

### Step 4: 实现 Crew 类 (crew_optimization.py)
- 遵循 crew_gap_finder.py 模式
- 加载 gap_report.json, audit_report.json, success_profile_report.json
- 输出 optimization_plan.json

### Step 5: 测试和验证
- 创建测试文件
- 运行端到端测试
- 验证输出格式

## Testing Strategy

1. **单元测试**: 验证 Pydantic 模型的字段验证
2. **集成测试**: 使用真实的 gap_report.json 运行 crew
3. **输出验证**: 检查生成的 optimization_plan.json 格式正确性
4. **内容质量**: 人工审查优化建议的可执行性

## Input/Output

### Input
- `gap_report.json`: GapReport (差距分析)
- `audit_report.json`: AuditReport (原始笔记特征)
- `success_profile_report.json`: SuccessProfileReport (成功模式)
- `owned_note.json`: 原始笔记内容

### Output
- `optimization_plan.json`: OptimizationPlan (可执行优化方案)

## Risks and Considerations

1. **生成质量**: LLM 生成的内容质量可能参差不齐，需要好的 prompt 设计
2. **中文生成**: 确保生成的中文内容自然、符合小红书风格
3. **可执行性**: 建议必须是具体的、可直接使用的，而非泛泛的建议
4. **视觉 Prompt**: 由于没有 AIGC 基座，视觉建议以 prompt 形式输出，用户需自行使用 AI 生成

## Alternatives Considered

1. **直接生成完整重写的内容**: 风险太大，可能丢失原内容精华
2. **只提供建议列表**: 不够具体，用户难以执行
3. **生成图片而非 Prompt**: 需要集成 AIGC 服务，超出当前范围

选择当前方案的原因：
- 平衡了具体性和安全性
- 标题/结尾等关键部分直接生成，风险可控
- 视觉部分用 prompt 替代，保持灵活性
