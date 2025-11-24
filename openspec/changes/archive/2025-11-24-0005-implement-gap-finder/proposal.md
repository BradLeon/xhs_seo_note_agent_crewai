# Implement GapFinder Agent for Performance Gap Analysis

**Status:** proposal
**Created:** 2025-11-21
**Updated:** 2025-11-21

## Summary

Implement the GapFinder agent as a standalone crew to identify and prioritize significant performance gaps between owned_note and target_notes by comparing SuccessProfileReport and AuditReport outputs.

## Motivation

As described in project.md, the GapFinder agent is the critical bridge between analysis and optimization:

1. **Answers "Why does owned_note score low?"** - The core question users want answered
2. **Enables data-driven optimization** - Provides objective evidence of what needs fixing
3. **Prioritizes effort** - Uses statistical significance to rank gaps by importance
4. **Maps metrics to features** - Connects performance deficits to actionable content changes

Currently, we have CompetitorAnalyst (analyzes what works) and OwnedNoteAuditor (analyzes current state), but no way to systematically compare them to identify critical gaps.

## Proposed Changes

### Architecture Decision: Standalone Crew

Following the established pattern from change 0004 (OwnedNoteAuditor), GapFinder will be implemented as a **standalone crew** (not integrated into main crew.py), because:

1. **Modular development** - Can be developed/tested independently
2. **Flexible execution** - Users can run gap analysis separately from full workflow
3. **Clear boundaries** - Takes two reports as input, produces one report as output
4. **Parallel to OwnedNoteAuditor pattern** - Maintains consistency in codebase structure

### Core Components

#### 1. Expand GapReport Model

**File:** `src/xhs_seo_optimizer/models/reports.py`

Expand the placeholder GapReport class to include:

```python
class MetricGap(BaseModel):
    """单个指标的差距分析 (Gap analysis for a single metric)."""

    metric_name: str  # e.g., "ctr", "comment_rate"
    owned_value: float
    target_mean: float
    delta_absolute: float
    delta_pct: float
    z_score: float
    p_value: float
    significance: str  # "critical", "very_significant", "significant", "marginal", "none"
    priority_rank: int  # 1-based ranking by importance

    # Feature attribution
    related_features: List[str]  # Features that affect this metric
    missing_features: List[str]  # Features absent in owned_note but present in success profile
    weak_features: List[str]  # Features present but poorly executed

    # Narrative
    gap_explanation: str  # Why this gap exists (2-3 sentences)
    recommendation_summary: str  # What to improve (1-2 sentences)

class GapReport(BaseModel):
    """差距分析报告 (Gap analysis report).

    Identifies statistically significant performance gaps between owned_note
    and target_notes, prioritizes them, and maps them to actionable features.
    """

    keyword: str
    owned_note_id: str

    # Statistical gap analysis
    significant_gaps: List[MetricGap]  # p < 0.05, ordered by priority
    marginal_gaps: List[MetricGap]  # 0.05 <= p < 0.10
    non_significant_gaps: List[MetricGap]  # p >= 0.10

    # Cross-metric insights
    top_priority_metrics: List[str]  # Top 3 metrics to focus on (by priority score)
    root_causes: List[str]  # 3-5 root causes across gaps (e.g., "weak title hook")
    impact_summary: str  # Overall narrative (100-200 chars)

    # Metadata
    sample_size: int  # From target_notes
    gap_timestamp: str  # ISO 8601
```

#### 2. Create GapFinder Crew

**File:** `src/xhs_seo_optimizer/crew_gap_finder.py`

```python
from crewai import Agent, Crew, Task
from crewai.project import CrewBase, agent, task, crew
from typing import Any, Dict
import json
from datetime import datetime, timezone

from .tools.statistical_delta import StatisticalDeltaTool
from .models.reports import SuccessProfileReport, AuditReport, GapReport

@CrewBase
class XhsSeoOptimizerCrewGapFinder:
    """差距定位员 Crew - Gap Finder for performance analysis."""

    agents_config = 'config/agents_gap_finder.yaml'
    tasks_config = 'config/tasks_gap_finder.yaml'

    def __init__(self):
        super().__init__()
        self.shared_context = {}

    @agent
    def gap_finder(self) -> Agent:
        """差距定位员 agent."""
        return Agent(
            config=self.agents_config['gap_finder'],
            tools=[StatisticalDeltaTool()],
            verbose=True,
            allow_delegation=False
        )

    @task
    def calculate_statistical_gaps(self) -> Task:
        """Task 1: Calculate statistical gaps using StatisticalDeltaTool."""
        return Task(
            config=self.tasks_config['calculate_statistical_gaps'],
            agent=self.gap_finder()
        )

    @task
    def map_gaps_to_features(self) -> Task:
        """Task 2: Map metric gaps to missing/weak features."""
        return Task(
            config=self.tasks_config['map_gaps_to_features'],
            agent=self.gap_finder(),
            context=[self.calculate_statistical_gaps()]
        )

    @task
    def prioritize_gaps(self) -> Task:
        """Task 3: Prioritize gaps and identify root causes."""
        return Task(
            config=self.tasks_config['prioritize_gaps'],
            agent=self.gap_finder(),
            context=[self.calculate_statistical_gaps(), self.map_gaps_to_features()]
        )

    @task
    def generate_gap_report(self) -> Task:
        """Task 4: Generate final GapReport JSON."""
        return Task(
            config=self.tasks_config['generate_gap_report'],
            agent=self.gap_finder(),
            context=[self.calculate_statistical_gaps(), self.map_gaps_to_features(), self.prioritize_gaps()],
            output_pydantic=GapReport
        )

    @crew
    def crew(self) -> Crew:
        """Create GapFinder crew."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True
        )

    @before_kickoff
    def validate_and_flatten_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and flatten inputs before crew execution.

        Following the pattern from OwnedNoteAuditor: flatten complex Pydantic models
        to simple dicts and extract key fields for YAML variable substitution.

        Args:
            inputs: Must contain:
                - success_profile_report: SuccessProfileReport dict or JSON string
                - audit_report: AuditReport dict or JSON string
                - keyword: str

        Returns:
            Flattened dict for YAML variable substitution

        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Validate required fields
        if 'success_profile_report' not in inputs:
            raise ValueError("inputs must contain 'success_profile_report'")
        if 'audit_report' not in inputs:
            raise ValueError("inputs must contain 'audit_report'")
        if 'keyword' not in inputs:
            raise ValueError("inputs must contain 'keyword'")

        # Parse success_profile_report (handle both dict and JSON string)
        success_profile = inputs['success_profile_report']
        if isinstance(success_profile, str):
            success_profile = json.loads(success_profile)

        # Parse audit_report (handle both dict and JSON string)
        audit_report = inputs['audit_report']
        if isinstance(audit_report, str):
            audit_report = json.loads(audit_report)

        # Validate audit_report has current_metrics
        if 'current_metrics' not in audit_report:
            raise ValueError("audit_report must contain 'current_metrics' field")

        # Extract flattened data for YAML variable substitution
        # Current metrics from owned_note (via audit_report)
        current_metrics = audit_report['current_metrics']

        # Target metrics from success_profile_report
        aggregated_stats = success_profile['aggregated_stats']
        prediction_stats = aggregated_stats['prediction_stats']

        # Flatten for YAML: target_mean_ctr, target_std_ctr, etc.
        target_means = {}
        target_stds = {}
        for metric_name, stats in prediction_stats.items():
            target_means[f"target_mean_{metric_name}"] = stats['mean']
            target_stds[f"target_std_{metric_name}"] = stats['std']

        # Store in shared context for tool/agent access (if needed)
        from xhs_seo_optimizer.shared_context import shared_context
        shared_context.set("success_profile_report", success_profile)
        shared_context.set("audit_report", audit_report)

        # Flatten inputs for YAML variable substitution
        inputs['note_id'] = audit_report['note_id']
        inputs['current_metrics'] = current_metrics
        inputs['target_means'] = target_means
        inputs['target_stds'] = target_stds
        inputs['sample_size'] = aggregated_stats['sample_size']

        # Keep original reports for agent context
        inputs['success_profile_report'] = success_profile
        inputs['audit_report'] = audit_report

        return inputs

    def kickoff(self, inputs: Dict[str, Any]) -> Any:
        """Execute gap analysis.

        Args:
            inputs: Must contain:
                - success_profile_report: SuccessProfileReport dict or JSON
                - audit_report: AuditReport dict or JSON (must have current_metrics field)
                - keyword: str
        """
        # @before_kickoff will validate and flatten inputs
        # Execute crew
        result = self.crew().kickoff(inputs=inputs)

        # Save output to file
        self._save_gap_report(result)

        return result

    def _save_gap_report(self, result: Any):
        """Save GapReport to outputs/gap_report.json."""
        import os
        os.makedirs("outputs", exist_ok=True)

        output_path = "outputs/gap_report.json"

        # Get JSON from result
        if hasattr(result, 'pydantic') and result.pydantic:
            report_json = result.pydantic.model_dump_json(indent=2)
        elif hasattr(result, 'json') and result.json:
            report_json = result.json
        elif hasattr(result, 'raw'):
            report_json = result.raw
        else:
            report_json = str(result)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_json)

        print(f"GapReport saved to {output_path}")
```

#### 3. Create YAML Configurations

**File:** `src/xhs_seo_optimizer/config/agents_gap_finder.yaml`

```yaml
gap_finder:
  role: >
    差距定位员 (Gap Finder) for keyword {keyword}
  goal: >
    识别客户笔记与竞品笔记之间的显著性能差距，并将差距映射到可执行的内容特征
    Identify statistically significant performance gaps between owned note and target notes,
    then map gaps to actionable content features
  backstory: >
    你是一位数据驱动的差距分析专家，擅长使用统计方法识别关键问题。
    你能够精准地将指标差距归因到具体的内容特征缺失或执行不足。
    你的分析为优化策略师提供了明确的改进方向。

    You are a data-driven gap analyst skilled at identifying critical issues using statistical methods.
    You can precisely attribute metric gaps to specific missing or poorly-executed content features.
    Your analysis provides clear direction for the optimization strategist.
```

**File:** `src/xhs_seo_optimizer/config/tasks_gap_finder.yaml`

```yaml
calculate_statistical_gaps:
  description: >
    使用StatisticalDeltaTool计算客户笔记与竞品笔记的统计差距。

    输入数据（已由@before_kickoff拍平）：
    - current_metrics: Dict[str, float] - 客户笔记的当前指标 (来自audit_report.current_metrics)
    - target_means: Dict[str, float] - 竞品笔记的平均值 (如target_mean_ctr, target_mean_comment_rate)
    - target_stds: Dict[str, float] - 竞品笔记的标准差 (如target_std_ctr, target_std_comment_rate)
    - sample_size: int - 竞品笔记样本量

    任务：
    1. 从inputs中获取已拍平的指标数据：
       - 客户笔记指标: current_metrics["ctr"], current_metrics["comment_rate"], etc.
       - 竞品均值: target_mean_ctr, target_mean_comment_rate, etc.
       - 竞品标准差: target_std_ctr, target_std_comment_rate, etc.

    2. 对每个指标计算统计差距：
       - z_score = (owned_value - target_mean) / target_std
       - p_value = 2 * (1 - norm.cdf(abs(z_score)))  # 双尾检验
       - significance: critical (p<0.001), very_significant (p<0.01), significant (p<0.05), marginal (p<0.10), none

    3. 按显著性分类: significant (p<0.05), marginal (0.05<=p<0.10), non_significant (p>=0.10)

    4. 计算优先级分数: priority_score = |z_score| * |delta_pct| / 100

    输出：GapAnalysis结果，包含significant_gaps, non_significant_gaps, priority_order
  expected_output: >
    StatisticalDeltaTool的完整JSON输出，包含：
    - significant_gaps: 统计显著的差距列表
    - non_significant_gaps: 非显著差距列表
    - priority_order: 按优先级排序的指标名称
    - sample_size: 竞品笔记样本量

    示例：
    {
      "significant_gaps": [
        {
          "metric": "ctr",
          "owned_value": 0.08,
          "target_mean": 0.14,
          "delta_absolute": -0.06,
          "delta_pct": -42.9,
          "z_score": -2.0,
          "p_value": 0.046,
          "significance": "significant",
          "interpretation": "ctr: owned_note is significantly lower..."
        }
      ],
      "priority_order": ["comment_rate", "ctr", "sort_score2"]
    }
  agent: gap_finder

map_gaps_to_features:
  description: >
    将指标差距映射到具体的内容特征缺失或执行不足。

    输入数据 (from context):
    - Task 1 输出: 统计差距分析 (significant_gaps list)
    - success_profile_report: 包含metric_profiles (每个指标的成功模式和相关特征)
    - audit_report: 包含text_features和visual_features (客户笔记的实际特征)

    任务：
    对于每个significant gap:
    1. 从success_profile_report.metric_profiles找到该指标的相关特征 (relevant_features)
    2. 比对audit_report的text_features/visual_features，识别：
       - missing_features: 成功模式中存在但客户笔记中缺失的特征
       - weak_features: 客户笔记中存在但执行不足的特征 (基于feature strength/score)
    3. 生成gap_explanation: 解释为什么这个指标差距存在 (2-3句话，基于特征对比)
    4. 生成recommendation_summary: 简要建议如何改进 (1-2句话)

    特征归因规则 (参考project.md):
    - ctr: 主要受title, cover影响
    - comment_rate: 主要受content (ending technique, engagement hooks)影响
    - sort_score2: 综合指标，受多个特征影响
    - interaction_rate: 受content质量和视觉吸引力影响

    输出：为每个significant gap添加feature attribution和narrative
  expected_output: >
    扩展的gap分析列表，每个gap包含：
    - 原始统计字段 (metric, owned_value, target_mean, z_score, p_value, etc.)
    - related_features: 影响该指标的特征列表
    - missing_features: 客户笔记中缺失的关键特征
    - weak_features: 客户笔记中执行不足的特征
    - gap_explanation: 差距原因解释 (2-3句话)
    - recommendation_summary: 改进建议 (1-2句话)

    示例：
    {
      "metric": "comment_rate",
      "owned_value": 0.0013,
      "target_mean": 0.012,
      "delta_pct": -89.2,
      "z_score": -3.67,
      "significance": "critical",
      "related_features": ["ending_technique", "engagement_hooks", "question_presence"],
      "missing_features": ["open_ended_question"],
      "weak_features": ["engagement_hooks"],
      "gap_explanation": "客户笔记的评论率显著低于竞品，主要因为结尾缺少开放式问题引导互动。竞品中85%使用结尾问题，而客户笔记仅使用陈述句。",
      "recommendation_summary": "在笔记结尾添加开放式问题（如"你家宝宝有这种情况吗？"），降低评论门槛。"
    }
  agent: gap_finder

prioritize_gaps:
  description: >
    对差距进行优先级排序，并识别跨指标的根本原因。

    输入数据 (from context):
    - Task 1: priority_order (按统计显著性和影响magnitude排序)
    - Task 2: feature-mapped gaps (包含missing_features和weak_features)

    任务：
    1. 使用Task 1的priority_order为每个gap分配priority_rank (1-based)
    2. 识别top_priority_metrics: 选择top 3最重要的指标
    3. 识别root_causes: 跨多个指标的共同原因
       - 方法: 统计missing_features和weak_features的频率
       - 选择出现频率最高的3-5个作为root causes
       - 用中文描述 (如: "标题缺乏情感钩子", "封面视觉冲击力不足")
    4. 生成impact_summary: 整体差距叙述 (100-200字符)
       - 说明主要问题和整体影响
       - 突出最严重的1-2个gap
       - 指出改进方向
  expected_output: >
    优先级排序结果和根本原因分析：
    - top_priority_metrics: 前3个需要重点改进的指标
    - root_causes: 3-5个跨指标的根本原因
    - impact_summary: 整体差距总结 (100-200字符)

    示例：
    {
      "top_priority_metrics": ["comment_rate", "ctr", "sort_score2"],
      "root_causes": [
        "标题缺乏情感钩子和疑问句",
        "结尾缺少开放式问题引导评论",
        "封面视觉吸引力不足",
        "内容缺少对比钩子和数据支撑"
      ],
      "impact_summary": "客户笔记在comment_rate和ctr上显著落后竞品（分别低89%和43%），导致sort_score2排名靠后。主要问题是标题缺乏吸引力、结尾无互动引导、封面视觉冲击力弱。"
    }
  agent: gap_finder

generate_gap_report:
  description: >
    生成最终的GapReport JSON输出。

    输入数据 (from all previous tasks):
    - Task 1: 统计差距分类 (significant_gaps, marginal_gaps, non_significant_gaps)
    - Task 2: 特征映射后的gap详细信息
    - Task 3: 优先级排序和根本原因
    - success_profile_report: 样本量信息
    - audit_report: owned_note_id

    任务：
    1. 整合所有分析结果到GapReport模型
    2. 确保所有字段符合Pydantic schema验证
    3. 添加metadata: keyword, owned_note_id, sample_size, gap_timestamp
    4. 输出为JSON格式

    输出格式必须严格遵循GapReport Pydantic模型，包含：
    - keyword, owned_note_id
    - significant_gaps: List[MetricGap]
    - marginal_gaps: List[MetricGap]
    - non_significant_gaps: List[MetricGap]
    - top_priority_metrics: List[str] (top 3)
    - root_causes: List[str] (3-5 items)
    - impact_summary: str (100-200 chars)
    - sample_size: int
    - gap_timestamp: str (ISO 8601)
  expected_output: >
    完整的GapReport JSON，符合Pydantic模型schema。
    该报告将保存到outputs/gap_report.json，并作为OptimizationStrategist的输入。
  agent: gap_finder
  output_pydantic: GapReport
```

#### 4. Create OpenSpec Specification

**File:** `openspec/changes/0005-implement-gap-finder/specs/gap-finder/spec.md`

Will contain detailed requirements and test scenarios (similar to owned-note-auditor/spec.md).

#### 5. Create Test Suite

**File:** `tests/test_gap_finder.py`

```python
"""Test suite for GapFinder crew."""

import json
import pytest
from xhs_seo_optimizer.crew_gap_finder import XhsSeoOptimizerCrewGapFinder
from xhs_seo_optimizer.models.reports import GapReport

class TestGapFinder:

    def test_basic_execution(self):
        """Test GapFinder crew executes successfully with real report data."""
        # Load SuccessProfileReport from outputs (from CompetitorAnalyst)
        with open("outputs/success_profile_report.json") as f:
            success_profile = json.load(f)

        # Load AuditReport from outputs (from OwnedNoteAuditor)
        with open("outputs/audit_report.json") as f:
            audit_report = json.load(f)

        # Execute GapFinder
        crew = XhsSeoOptimizerCrewGapFinder()
        result = crew.kickoff(inputs={
            "success_profile_report": success_profile,
            "audit_report": audit_report,
            "keyword": "DHA"
        })

        # Verify result
        assert result is not None
        assert result.pydantic is not None

        gap_report = result.pydantic
        assert isinstance(gap_report, GapReport)

        # Verify output file exists
        import os
        assert os.path.exists("outputs/gap_report.json")

    def test_gap_report_schema(self):
        """Test GapReport output conforms to schema."""
        # Load gap_report.json
        with open("outputs/gap_report.json") as f:
            gap_dict = json.load(f)

        # Should parse as GapReport
        gap_report = GapReport(**gap_dict)

        # Verify required fields
        assert gap_report.keyword is not None
        assert gap_report.owned_note_id is not None
        assert isinstance(gap_report.significant_gaps, list)
        assert isinstance(gap_report.top_priority_metrics, list)
        assert len(gap_report.top_priority_metrics) <= 3
        assert len(gap_report.root_causes) >= 3
        assert len(gap_report.root_causes) <= 5
        assert 100 <= len(gap_report.impact_summary) <= 200

    def test_feature_attribution(self):
        """Test gaps are mapped to specific features."""
        with open("outputs/gap_report.json") as f:
            gap_report = GapReport(**json.load(f))

        # Each significant gap MUST have feature attribution
        for gap in gap_report.significant_gaps:
            assert len(gap.related_features) > 0
            assert gap.gap_explanation is not None
            assert len(gap.gap_explanation) > 20
            assert gap.recommendation_summary is not None

    def test_priority_ordering(self):
        """Test gaps are properly prioritized."""
        with open("outputs/gap_report.json") as f:
            gap_report = GapReport(**json.load(f))

        # Priority ranks should be sequential
        ranks = [g.priority_rank for g in gap_report.significant_gaps]
        assert ranks == sorted(ranks)  # Should be in order

        # Top priority metrics should match highest-ranked gaps
        top_metrics = gap_report.top_priority_metrics
        assert len(top_metrics) <= 3

        # Verify top metrics are from significant gaps
        sig_metrics = [g.metric_name for g in gap_report.significant_gaps[:3]]
        for metric in top_metrics:
            assert metric in sig_metrics
```

### Files to Create

1. `src/xhs_seo_optimizer/crew_gap_finder.py` - Main crew implementation
2. `src/xhs_seo_optimizer/config/agents_gap_finder.yaml` - Agent definition
3. `src/xhs_seo_optimizer/config/tasks_gap_finder.yaml` - Task definitions
4. `openspec/changes/0005-implement-gap-finder/specs/gap-finder/spec.md` - Spec with requirements
5. `tests/test_gap_finder.py` - Test suite

### Files to Modify

1. `src/xhs_seo_optimizer/models/reports.py` - Expand GapReport model and add MetricGap model; **also add current_metrics field to AuditReport**
2. `src/xhs_seo_optimizer/models/__init__.py` - Export new models
3. `src/xhs_seo_optimizer/config/tasks_owned_note.yaml` - Update generate_audit_report task to include current_metrics
4. `openspec/specs/owned-note-auditor/spec.md` - Update requirements to reflect current_metrics field

### Files to Delete

None

## Implementation Plan

1. **Add current_metrics to AuditReport (15 min)**
   - Modify AuditReport in reports.py to add `current_metrics: Dict[str, float]` field
   - Update tasks_owned_note.yaml to instruct agent to populate this field
   - This is a prerequisite for GapFinder (needs owned_note metrics for comparison)

2. **Expand GapReport Model (30 min)**
   - Add MetricGap class to reports.py
   - Expand GapReport with full schema
   - Add field validators

3. **Create YAML Configurations (45 min)**
   - Write agents_gap_finder.yaml with agent definition
   - Write tasks_gap_finder.yaml with 4 task definitions
   - Ensure clear instructions for feature attribution

4. **Implement Crew Class (1 hour)**
   - Create crew_gap_finder.py with @CrewBase structure
   - Implement @agent and @task methods
   - Add shared_context handling
   - Add file saving logic

5. **Write Test Suite (45 min)**
   - Create test_gap_finder.py
   - Write basic execution test
   - Write schema validation test
   - Write feature attribution test

6. **Create Specification (30 min)**
   - Write specs/gap-finder/spec.md with requirements
   - Document test scenarios

7. **Integration Testing (30 min)**
   - Run CompetitorAnalyst → save success_profile_report.json
   - Run OwnedNoteAuditor → save audit_report.json
   - Run GapFinder with both reports as input
   - Verify gap_report.json output

## Testing Strategy

### Unit Tests
- Test GapReport Pydantic model validation
- Test MetricGap field constraints
- Test crew kickoff with valid inputs

### Integration Tests
- Test with real SuccessProfileReport and AuditReport JSON files
- Verify StatisticalDeltaTool is called correctly
- Verify feature mapping logic works across different gap scenarios
- Test output file creation

### End-to-End Tests
- Run full pipeline: CompetitorAnalyst → OwnedNoteAuditor → GapFinder
- Verify gap_report.json contains actionable insights
- Validate priority ordering makes sense

### Test Data
- Use docs/target_notes.json and docs/owned_note.json
- Generate success_profile_report.json from CompetitorAnalyst test
- Generate audit_report.json from OwnedNoteAuditor test
- Run GapFinder test with both

## Risks and Considerations

### Risk 1: LLM Hallucination in Feature Mapping
**Issue:** Agent may incorrectly map gaps to features without evidence

**Mitigation:**
- Task 2 instructions explicitly reference success_profile_report.metric_profiles for relevant_features
- Use audit_report's actual text_features/visual_features for comparison
- Require gap_explanation to cite specific feature values

### Risk 2: Complex Task Dependencies
**Issue:** Tasks 2-4 depend on Task 1 output, may fail if parsing is incorrect

**Mitigation:**
- Use Pydantic models to validate Task 1 output
- Add error handling in crew.kickoff()
- Test with various gap scenarios (zero gaps, many gaps, missing metrics)

### Risk 3: Priority Algorithm Ambiguity
**Issue:** Multiple ways to prioritize gaps (by p-value, by magnitude, by combined score)

**Mitigation:**
- Use StatisticalDeltaTool's existing priority_order (already combines z-score and magnitude)
- Document prioritization logic in tasks_gap_finder.yaml
- Test with edge cases (all gaps equal, one dominant gap)

### Risk 4: Root Cause Duplication
**Issue:** root_causes may overlap with individual gap explanations

**Mitigation:**
- Root causes should be higher-level patterns (e.g., "title weakness")
- Individual gap_explanations should be metric-specific (e.g., "CTR low due to title")
- Test that root_causes list is concise (3-5 items only)

## Alternatives Considered

### Alternative 1: Integrate GapFinder into Main Crew
**Rejected because:**
- Main crew (crew.py) is already complex with orchestration
- Standalone crew allows independent testing and reuse
- Follows established pattern from OwnedNoteAuditor (change 0004)

### Alternative 2: Single Task Instead of 4 Tasks
**Rejected because:**
- Breaking into 4 tasks provides clear checkpoints
- Easier to debug when agent fails at specific step
- Better context passing between stages

### Alternative 3: Use LLM for Statistical Analysis (No Tool)
**Rejected because:**
- LLMs hallucinate numeric calculations
- StatisticalDeltaTool already exists and is reliable
- Statistical significance requires precise math (z-score, p-value)

## Dependencies

- Requires change 0003 (CompetitorAnalyst) to be completed for SuccessProfileReport
- Requires change 0004 (OwnedNoteAuditor) to be completed for AuditReport
- Uses existing StatisticalDeltaTool from change 0002

## Success Criteria

1. ✅ GapFinder crew executes successfully with valid report inputs
2. ✅ GapReport conforms to Pydantic schema and validates
3. ✅ Gaps are correctly classified by statistical significance
4. ✅ Each significant gap includes feature attribution (missing_features, weak_features)
5. ✅ Top 3 priority metrics are correctly identified
6. ✅ Root causes are meaningful and actionable
7. ✅ Output saved to outputs/gap_report.json
8. ✅ All tests pass
9. ✅ Spec document is complete with test scenarios

## Next Steps After Implementation

After GapFinder is complete:
1. Implement OptimizationStrategist crew (uses GapReport as input)
2. Create end-to-end orchestrator that chains all crews
3. Add visualization of gap analysis results
