# Design: CompetitorAnalyst Agent with CrewAI Infrastructure

## Architecture Overview

This change establishes the complete CrewAI multi-agent framework and implements the first working agent (CompetitorAnalyst). The design follows a modular architecture where agents orchestrate existing tools to generate structured insights.

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                      CrewAI Framework                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              config/agents.yaml                       │  │
│  │  - Orchestrator                                       │  │
│  │  - CompetitorAnalyst  ← IMPLEMENTED THIS CHANGE      │  │
│  │  - OwnedNoteAuditor   ← Placeholder                  │  │
│  │  - GapFinder          ← Placeholder                  │  │
│  │  - OptimizationStrategist ← Placeholder              │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              config/tasks.yaml                        │  │
│  │  - orchestrate_workflow                               │  │
│  │  - analyze_competitors ← IMPLEMENTED THIS CHANGE     │  │
│  │  - audit_owned_note    ← Placeholder                 │  │
│  │  - find_gaps           ← Placeholder                 │  │
│  │  - generate_strategy   ← Placeholder                 │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   crew.py                             │  │
│  │  @agent methods (return Agent instances)             │  │
│  │  @task methods (return Task instances)               │  │
│  │  @crew method (return Crew instance)                 │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      Tools Layer                            │
│  (Already implemented in changes 0001 & 0002)               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ - DataAggregatorTool                                 │   │
│  │ - MultiModalVisionTool                               │   │
│  │ - NLPAnalysisTool                                    │   │
│  │ - StatisticalDeltaTool                               │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     Data Models                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ models/note.py (existing)                            │   │
│  │  - Note, NoteMetaData, NotePrediction, NoteTag       │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ models/analysis_results.py (existing)                │   │
│  │  - VisionAnalysisResult, TextAnalysisResult          │   │
│  │  - AggregatedMetrics, GapAnalysis                    │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ models/reports.py (NEW)                              │   │
│  │  - FeaturePattern, SuccessProfileReport              │   │
│  │  - AuditReport, GapReport, OptimizationPlan          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## CompetitorAnalyst Agent Design

### Responsibility
Analyze **why** target_notes achieve high prediction scores by identifying content patterns that statistically correlate with high metrics (CTR, comment_rate, sort_score2, etc.).

### Input
```python
{
  "target_notes": List[Note],  # 20-100 high-performing notes
  "keyword": str               # e.g., "婴儿辅食推荐"
}
```

### Output
```python
SuccessProfileReport:
  keyword: str
  sample_size: int
  aggregated_stats: AggregatedMetrics  # Mean CTR, comment_rate, etc.

  # Feature patterns organized by type
  title_patterns: List[FeaturePattern]
  cover_patterns: List[FeaturePattern]
  content_patterns: List[FeaturePattern]
  tag_patterns: List[FeaturePattern]

  # LLM-synthesized insights
  key_success_factors: List[str]
  viral_formula_summary: str
```

### Processing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Aggregate Statistics                               │
├─────────────────────────────────────────────────────────────┤
│ Input: target_notes (List[Note])                            │
│ Tool: DataAggregatorTool                                    │
│ Output: AggregatedMetrics                                   │
│                                                             │
│ - Calculate mean, median, std for all prediction metrics   │
│ - Compute tag frequencies and modes                        │
│ - Identify sample size and outliers                        │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Extract Features from Each Note                    │
├─────────────────────────────────────────────────────────────┤
│ For each note in target_notes:                             │
│   - Run NLPAnalysisTool(note.meta_data)                    │
│     → TextAnalysisResult (30+ text features)               │
│   - Run MultiModalVisionTool(note.meta_data)               │
│     → VisionAnalysisResult (17+ visual features)           │
│                                                             │
│ Build feature matrix:                                       │
│   note_features[note_id] = {                               │
│     "text": TextAnalysisResult,                            │
│     "vision": VisionAnalysisResult,                        │
│     "prediction": note.prediction,                         │
│     "tag": note.tag                                        │
│   }                                                         │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Identify High-Scoring Patterns (Hybrid Attribution)│
├─────────────────────────────────────────────────────────────┤
│ For each metric (ctr, comment_rate, sort_score2, ...):     │
│                                                             │
│   Layer 1: Apply Attribution Rules                         │
│   ─────────────────────────────                            │
│   - Filter features relevant to this metric                │
│   - Example: For CTR, only consider title + cover          │
│   - Use METRIC_FEATURE_ATTRIBUTION mapping                 │
│                                                             │
│   Layer 2: Calculate Statistical Prevalence                │
│   ─────────────────────────────────────                    │
│   - Segment notes by metric quartiles                      │
│     Top 25% (high scorers) vs Bottom 75% (baseline)        │
│   - For each feature pattern:                              │
│     * Count occurrences in high vs baseline groups         │
│     * Calculate prevalence % in each group                 │
│     * Compute z-score and p-value                          │
│   - Filter patterns:                                        │
│     * Prevalence in high group ≥ 70%                       │
│     * p-value < 0.05 (statistically significant)           │
│                                                             │
│   Layer 3: LLM Validation & Formula Generation             │
│   ─────────────────────────────────────────                │
│   - For each significant pattern:                          │
│     * Provide statistical evidence to LLM                  │
│     * Include concrete examples from notes                 │
│     * Ask LLM to explain WHY pattern works                 │
│     * Generate actionable "creation formula" in Chinese    │
│                                                             │
│ Collect all FeaturePattern objects                         │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Synthesize Success Profile Report                  │
├─────────────────────────────────────────────────────────────┤
│ - Organize patterns by feature type                        │
│   * title_patterns (from TextAnalysisResult.title_*)       │
│   * cover_patterns (from VisionAnalysisResult.cover_*)     │
│   * content_patterns (from TextAnalysisResult.content_*)   │
│   * tag_patterns (from note.tag.*)                         │
│                                                             │
│ - Generate summary insights (LLM):                         │
│   * key_success_factors: Top 5 most impactful patterns     │
│   * viral_formula_summary: Holistic "creation template"    │
│                                                             │
│ - Return SuccessProfileReport (JSON)                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Hybrid Attribution Engine

### Layer 1: Domain Rules (METRIC_FEATURE_ATTRIBUTION)

Pre-defined mapping based on platform mechanics and user behavior:

```python
METRIC_FEATURE_ATTRIBUTION = {
    # CTR (Click-Through Rate) - 用户决定是否点击时只能看到标题和封面
    "ctr": {
        "features": [
            "title_pattern", "title_keywords", "title_emotion",
            "cover_quality", "cover_composition", "cover_color_scheme",
            "cover_thumbnail_appeal"
        ],
        "rationale": "用户点击前仅能看到标题和封面缩略图"
    },

    # Comment Rate - 评论由内容质量和互动引导决定
    "comment_rate": {
        "features": [
            "ending_technique", "ending_cta",
            "content_framework", "pain_points", "credibility_signals",
            "question_density", "controversial_points"
        ],
        "rationale": "评论行为由内容深度和结尾引导触发"
    },

    # Interaction Rate (综合互动) - 受多方面影响
    "interaction_rate": {
        "features": [
            "title_pattern", "cover_composition",
            "content_framework", "value_propositions",
            "visual_storytelling", "emotional_tone"
        ],
        "rationale": "综合互动受标题、视觉和内容共同影响"
    },

    # Sort Score (排序分数) - 平台算法综合评估
    "sort_score2": {
        "features": [
            "intention_lv2", "taxonomy2",
            "content_framework", "value_density",
            "credibility_signals", "authenticity"
        ],
        "rationale": "平台优先推荐有价值、真实的干货内容"
    },

    # Like Rate - 情感共鸣和视觉吸引
    "like_rate": {
        "features": [
            "emotional_triggers", "benefit_appeals",
            "visual_tone", "personal_style",
            "cover_aesthetic", "transformation_promise"
        ],
        "rationale": "点赞由情感共鸣和视觉美感驱动"
    },

    # Share Rate - 内容价值和社交货币
    "share_rate": {
        "features": [
            "value_propositions", "practical_tips",
            "social_proof", "exclusivity",
            "controversial_points", "transformation_promise"
        ],
        "rationale": "分享行为需要内容具备社交货币价值"
    },

    # Follow Rate - 长期价值和个人品牌
    "follow_rate": {
        "features": [
            "personal_style", "expertise_signals",
            "consistency", "value_density",
            "transformation_promise", "brand_consistency"
        ],
        "rationale": "关注决策基于对创作者长期价值的认可"
    }
}
```

### Layer 2: Direct Prevalence Analysis

**Key Design Decision:**
`target_notes` are already curated high-quality content (top search results), not a diverse sample. Therefore, we use **Direct Prevalence Analysis** instead of split+compare approach.

**Rationale:**
- Input is pre-filtered优质内容(排序靠前的笔记)
- Splitting top performers into "more优质" vs "less优质" groups yields minimal difference
- Statistical significance testing is inappropriate without a proper control group
- Instead: If a feature appears in ≥70% of target_notes, it's a success pattern

For each metric and its relevant features:

```python
def calculate_direct_prevalence(
    notes: List[Note],
    metric: str,
    relevant_features: List[str]
) -> Dict[str, PatternStats]:
    """
    Calculate direct prevalence of features in curated high-quality notes.

    Args:
        notes: Target notes (already pre-filtered top performers)
        metric: Metric name (e.g., 'ctr', 'comment_rate')
        relevant_features: Features affecting this metric (from Layer 1)

    Returns:
        {
            "feature_name": {
                "prevalence_pct": 85.0,  # % in all target notes
                "count": 17,  # notes with feature
                "total": 20,  # total notes
                "is_pattern": True  # prevalence >= 70%
            }
        }
    """
    # Get notes with this metric (no splitting)
    notes_with_metric = [n for n in notes if has_metric(n, metric)]

    # For each relevant feature, calculate direct prevalence
    patterns = {}
    for feature_name in relevant_features:
        feature_count = sum(1 for n in notes_with_metric if has_feature(n, feature_name))
        total = len(notes_with_metric)
        prevalence_pct = (feature_count / total * 100) if total > 0 else 0

        # Pattern identified if prevalence >= threshold
        is_pattern = prevalence_pct >= 70.0  # (or 50% for small samples)

        patterns[feature_name] = {
            "prevalence_pct": prevalence_pct,
            "count": feature_count,
            "total": total,
            "is_pattern": is_pattern
        }

    return patterns
```

**No Statistical Testing:**
- No z-score, no p-value (no control group to compare against)
- Simple threshold-based detection: prevalence ≥ 70% → pattern

### Layer 3: LLM Formula Synthesis

Prompt template for generating actionable formulas:

```python
FORMULA_SYNTHESIS_PROMPT = """
你是小红书内容策略专家。基于以下统计证据，生成一个简洁、可操作的"创作公式"。

**指标**: {metric_name}
**模式**: {pattern_description}
**统计证据**:
- 高分组（Top 25%）中使用该模式的比例: {prevalence_high}%
- 基线组（其他75%）中使用该模式的比例: {prevalence_baseline}%
- 统计显著性: z={z_score:.2f}, p={p_value:.4f}
- 样本量: 高分组n={sample_high}, 基线组n={sample_baseline}

**具体案例** (来自高分笔记):
{examples}

请回答以下问题:
1. **为什么这个模式有效?** (心理学/行为学原理)
2. **创作公式** (一句话总结如何应用)
3. **关键要素** (3-5个具体执行点)

输出格式:
{{
  "explanation": "...",
  "formula": "...",
  "key_elements": ["...", "...", "..."]
}}
"""
```

---

## Data Models

### FeaturePattern

Represents a single content pattern with statistical evidence.

```python
class FeaturePattern(BaseModel):
    """单个内容模式的统计分析 (Statistical analysis of a content pattern)."""

    feature_name: str = Field(
        description="特征名称 (Feature name, e.g., 'interrogative_title_pattern')"
    )

    feature_type: str = Field(
        description="特征类型 (title | cover | content | tag)"
    )

    description: str = Field(
        description="模式描述 (Pattern description in Chinese)"
    )

    prevalence_pct: float = Field(
        description="高分组中的流行度百分比 (% in high-scoring group)"
    )

    baseline_pct: float = Field(
        description="基线组中的流行度百分比 (% in baseline group)"
    )

    affected_metrics: Dict[str, float] = Field(
        description="影响的指标及其效应大小 (Metrics affected and effect sizes)",
        example={"ctr": 35.2, "comment_rate": 12.1}
    )

    statistical_evidence: str = Field(
        description="统计证据摘要 (e.g., 'z=3.2, p<0.001, n=85/100')"
    )

    z_score: float = Field(description="Z分数 (Z-score for prevalence difference)")
    p_value: float = Field(description="P值 (P-value for significance test)")

    sample_size_high: int = Field(description="高分组样本量")
    sample_size_baseline: int = Field(description="基线组样本量")

    examples: List[str] = Field(
        description="具体案例 (Concrete examples from high-scoring notes)",
        max_length=5
    )

    # LLM-generated insights
    why_it_works: str = Field(
        description="为什么有效 (Psychological/behavioral explanation)"
    )

    creation_formula: str = Field(
        description="创作公式 (Actionable one-sentence formula in Chinese)"
    )

    key_elements: List[str] = Field(
        description="关键执行要素 (3-5 specific execution points)",
        min_length=3,
        max_length=5
    )
```

### SuccessProfileReport

Complete analysis report from CompetitorAnalyst.

```python
class SuccessProfileReport(BaseModel):
    """竞品成功模式分析报告 (Success profile analysis report)."""

    keyword: str = Field(description="目标关键词")
    sample_size: int = Field(description="分析的竞品笔记数量")

    # Aggregated baseline statistics
    aggregated_stats: AggregatedMetrics = Field(
        description="竞品笔记的聚合统计 (从DataAggregatorTool获得)"
    )

    # Feature-centric patterns organized by type
    title_patterns: List[FeaturePattern] = Field(
        description="标题模式列表 (Patterns from title analysis)",
        default=[]
    )

    cover_patterns: List[FeaturePattern] = Field(
        description="封面图模式列表 (Patterns from cover image analysis)",
        default=[]
    )

    content_patterns: List[FeaturePattern] = Field(
        description="内容模式列表 (Patterns from content analysis)",
        default=[]
    )

    tag_patterns: List[FeaturePattern] = Field(
        description="标签模式列表 (Patterns from tag analysis)",
        default=[]
    )

    # Summary insights (LLM-synthesized)
    key_success_factors: List[str] = Field(
        description="关键成功要素 (Top 5 most impactful patterns)",
        min_length=3,
        max_length=5
    )

    viral_formula_summary: str = Field(
        description="爆款公式总结 (Holistic creation template summary)"
    )

    # Metadata
    analysis_timestamp: str = Field(
        description="分析时间戳 (ISO 8601 format)"
    )
```

---

## CrewAI Infrastructure

### Configuration Files

#### agents.yaml

```yaml
orchestrator:
  role: "项目总监 (Orchestrator)"
  goal: "协调多个分析师，生成完整的内容优化方案"
  backstory: |
    你是一位经验丰富的项目经理，擅长协调团队成员，
    确保各个分析环节有序进行，最终交付高质量的优化建议。
  tools: []
  llm: "deepseek/deepseek-chat"
  function_calling_llm: "deepseek/deepseek-chat"  # 工具调用用更轻量级模型

competitor_analyst:
  role: "竞品分析师 (Competitor Analyst)"
  goal: "分析为什么target_notes在关键词'{keyword}'下获得高分，总结出可复制的'创作公式'"
  backstory: |
    你是一位数据驱动的内容策略专家，擅长从高表现笔记中提取成功模式。
    你结合统计分析、内容理解和平台机制，能够准确归因：哪些特征真正影响哪些指标。
    你的分析报告总是包含充分的统计证据和可操作的创作公式。
  tools:
    - DataAggregatorTool
    - MultiModalVisionTool
    - NLPAnalysisTool
  llm: "deepseek/deepseek-chat"                      # 复杂推理
  function_calling_llm: "deepseek/deepseek-chat"    # 工具调用（可选用更便宜的模型）
  max_iter: 15
  verbose: true

owned_note_auditor:
  role: "客户笔记诊断师 (Owned Note Auditor)"
  goal: "深度诊断客户笔记owned_note的内容特征和预测表现"
  backstory: |
    你是一位细致入微的内容审计专家，能够全面分析笔记的优势和不足。
  tools:
    - MultiModalVisionTool
    - NLPAnalysisTool
  llm: "deepseek/deepseek-chat"
  function_calling_llm: "deepseek/deepseek-chat"

gap_finder:
  role: "差距定位员 (Gap Finder)"
  goal: "对比客户笔记与竞品成功模式，识别关键差距和优化机会"
  backstory: |
    你是一位锐利的对比分析专家，能够精准定位问题所在。
  tools:
    - StatisticalDeltaTool
  llm: "deepseek/deepseek-chat"
  function_calling_llm: "deepseek/deepseek-chat"

optimization_strategist:
  role: "优化策略师 (Optimization Strategist)"
  goal: "基于差距分析，生成具体的内容优化建议"
  backstory: |
    你是一位经验丰富的内容优化顾问，能够将分析结果转化为可执行的改进方案。
  tools: []
  llm: "deepseek/deepseek-chat"
  function_calling_llm: "deepseek/deepseek-chat"
```

#### tasks.yaml

```yaml
orchestrate_workflow:
  description: |
    协调整个分析流程:
    1. 并行执行竞品分析和客户笔记诊断
    2. 执行差距定位
    3. 生成优化策略
  expected_output: "完整的优化方案报告"
  agent: orchestrator

analyze_competitors:
  description: |
    分析target_notes在关键词'{keyword}'下的成功模式:
    1. 使用DataAggregatorTool计算统计摘要
    2. 使用NLPAnalysisTool和MultiModalVisionTool提取每个笔记的特征
    3. 应用混合归因逻辑识别高相关性模式
    4. 生成SuccessProfileReport，包含:
       - 按特征类型组织的模式列表 (title, cover, content, tag)
       - 每个模式的统计证据 (prevalence %, z-score, p-value)
       - LLM生成的创作公式和执行要点
  expected_output: "SuccessProfileReport (JSON格式)"
  agent: competitor_analyst
  # Note: output_file仅用于文档，实际在crew.py中配置output_pydantic
  output_file: "outputs/success_profile_report.json"

audit_owned_note:
  description: |
    深度分析客户笔记owned_note的内容特征和表现指标。
  expected_output: "AuditReport (JSON格式)"
  agent: owned_note_auditor

find_gaps:
  description: |
    对比owned_note与target_notes的成功模式，识别差距。
  expected_output: "GapReport (JSON格式)"
  agent: gap_finder
  # Note: context依赖在crew.py的@task方法中配置，不在YAML中

generate_strategy:
  description: |
    基于差距分析，生成具体的内容优化建议。
  expected_output: "OptimizationPlan (JSON格式)"
  agent: optimization_strategist
  # Note: context依赖在crew.py的@task方法中配置，不在YAML中
```

### crew.py Structure

```python
from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task, before_kickoff, after_kickoff
from typing import List
from xhs_seo_optimizer.models.note import Note
from xhs_seo_optimizer.models.reports import SuccessProfileReport
from xhs_seo_optimizer.tools import (
    DataAggregatorTool,
    MultiModalVisionTool,
    NLPAnalysisTool,
    StatisticalDeltaTool
)

@CrewBase
class XhsSeoOptimizerCrew:
    """小红书SEO优化多智能体系统 (Xiaohongshu SEO Optimizer Multi-Agent System)."""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # === Callback Hooks ===

    @before_kickoff
    def validate_inputs(self, inputs):
        """Validate inputs before crew execution."""
        if not inputs.get("target_notes"):
            raise ValueError("target_notes is required")
        if not inputs.get("keyword"):
            raise ValueError("keyword is required")
        return inputs

    @after_kickoff
    def process_output(self, output):
        """Process output after crew completion."""
        # Could save to database, send notifications, etc.
        return output

    # === Agent Definitions ===

    @agent
    def orchestrator(self) -> Agent:
        """项目总监 agent (manager for hierarchical process)."""
        return Agent(
            config=self.agents_config['orchestrator'],
            tools=[]
        )

    @agent
    def competitor_analyst(self) -> Agent:
        """竞品分析师 agent."""
        return Agent(
            config=self.agents_config['competitor_analyst'],
            tools=[
                DataAggregatorTool(),
                MultiModalVisionTool(),
                NLPAnalysisTool()
            ]
        )

    @agent
    def owned_note_auditor(self) -> Agent:
        """客户笔记诊断师 agent (placeholder)."""
        return Agent(
            config=self.agents_config['owned_note_auditor'],
            tools=[MultiModalVisionTool(), NLPAnalysisTool()]
        )

    @agent
    def gap_finder(self) -> Agent:
        """差距定位员 agent (placeholder)."""
        return Agent(
            config=self.agents_config['gap_finder'],
            tools=[StatisticalDeltaTool()]
        )

    @agent
    def optimization_strategist(self) -> Agent:
        """优化策略师 agent (placeholder)."""
        return Agent(
            config=self.agents_config['optimization_strategist'],
            tools=[]
        )

    # === Task Definitions ===

    @task
    def analyze_competitors_task(self) -> Task:
        """竞品分析任务."""
        return Task(
            config=self.tasks_config['analyze_competitors'],
            agent=self.competitor_analyst(),
            output_pydantic=SuccessProfileReport  # Type-safe output
        )

    @task
    def audit_owned_note_task(self) -> Task:
        """客户笔记诊断任务 (placeholder)."""
        return Task(
            config=self.tasks_config['audit_owned_note'],
            agent=self.owned_note_auditor()
        )

    @task
    def find_gaps_task(self) -> Task:
        """差距定位任务 (placeholder)."""
        # Context configured HERE in Python, not in YAML
        return Task(
            config=self.tasks_config['find_gaps'],
            agent=self.gap_finder(),
            context=[
                self.analyze_competitors_task(),
                self.audit_owned_note_task()
            ]
        )

    @task
    def generate_strategy_task(self) -> Task:
        """优化策略生成任务 (placeholder)."""
        return Task(
            config=self.tasks_config['generate_strategy'],
            agent=self.optimization_strategist(),
            context=[self.find_gaps_task()]
        )

    # === Crew Assembly ===

    @crew
    def crew(self) -> Crew:
        """组装完整的crew."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,  # Hierarchical with Orchestrator as manager
            manager_agent=self.orchestrator(),  # Orchestrator manages workflow
            verbose=True,
            output_log_file="outputs/crew.log"
        )
```

---

## Data Flow

### End-to-End Flow (Phase 1: CompetitorAnalyst)

```
Input:
  target_notes: List[Note]  # 20-100 high-performing notes
  keyword: str              # "婴儿辅食推荐"

                ↓

  ┌─────────────────────────────────┐
  │  CompetitorAnalyst.execute()    │
  └─────────────────────────────────┘
                ↓
  ┌─────────────────────────────────┐
  │  1. DataAggregatorTool          │
  │     Input: target_notes         │
  │     Output: AggregatedMetrics   │
  └─────────────────────────────────┘
                ↓
  ┌─────────────────────────────────┐
  │  2. For each note:              │
  │     - NLPAnalysisTool           │
  │     - MultiModalVisionTool      │
  │     Build feature matrix        │
  └─────────────────────────────────┘
                ↓
  ┌─────────────────────────────────┐
  │  3. Hybrid Attribution:         │
  │     Layer 1: Apply rules        │
  │     Layer 2: Calc prevalence    │
  │     Layer 3: LLM synthesis      │
  │     → List[FeaturePattern]      │
  └─────────────────────────────────┘
                ↓
  ┌─────────────────────────────────┐
  │  4. Organize by feature type    │
  │     Generate summary insights   │
  │     → SuccessProfileReport      │
  └─────────────────────────────────┘
                ↓

Output:
  SuccessProfileReport (JSON)
  - title_patterns: [...]
  - cover_patterns: [...]
  - content_patterns: [...]
  - tag_patterns: [...]
  - key_success_factors: [...]
  - viral_formula_summary: "..."
```

---

## Error Handling

### Validation
- YAML configuration validated on startup
- Pydantic models enforce schema compliance
- Tool outputs validated before passing to agents

### Graceful Degradation
- If NLP/Vision tools fail for a note, log warning and continue
- If LLM synthesis fails, use template-based formula
- If no significant patterns found, report "未发现显著模式"

### Logging
- Structured logging for all agent actions
- Tool execution times logged for performance monitoring
- Pattern discovery logged for debugging

---

## Performance Considerations

### Caching
- Cache NLPAnalysisTool and MultiModalVisionTool results per note_id
- Cache LLM synthesis results per pattern signature

### Parallelization
- Analyze notes in parallel (ThreadPoolExecutor)
- Process different feature types concurrently

### Cost Control
- Use DeepSeek for LLM synthesis (free tier)
- Batch vision API calls when possible
- Limit max notes to 100 per analysis

---

## Testing Strategy

### Unit Tests
- Test attribution rule filtering
- Test prevalence calculation logic
- Test FeaturePattern validation
- Test SuccessProfileReport schema

### Integration Tests
- End-to-end with real target_notes.json
- Verify statistical correctness
- Validate LLM output quality
- Test YAML configuration loading

### Manual Review
- Review generated formulas for actionability
- Verify pattern examples are relevant
- Check bilingual output quality

---

## Future Extensions (Out of Scope)

- OwnedNoteAuditor implementation
- GapFinder with cross-report analysis
- OptimizationStrategist with A/B testing
- Orchestrator with dynamic workflow planning
- Multi-keyword batch analysis
- Web UI for report visualization
