"""Report models - 报告数据模型.

Pydantic models for agent-generated reports:
- FeaturePattern: Single content pattern with statistical evidence
- FeatureAnalysis: Analysis of a single feature for a specific metric (metric-centric)
- MetricSuccessProfile: Success profile for a single prediction metric (metric-centric)
- SuccessProfileReport: CompetitorAnalyst output
- AuditReport: OwnedNoteAuditor output (placeholder)
- GapReport: GapFinder output (placeholder)
- OptimizationPlan: OptimizationStrategist output (placeholder)
"""

from typing import List, Dict
from pydantic import BaseModel, Field, field_validator

from .analysis_results import AggregatedMetrics


class FeaturePattern(BaseModel):
    """单个内容模式的统计分析 (Statistical analysis of a content pattern).

    Represents a discovered pattern in high-performing notes with:
    - Direct prevalence evidence (% of target notes with this feature)
    - Concrete examples from real notes
    - LLM-generated insights (why it works, creation formula)

    Uses Direct Prevalence Analysis - target notes are curated top performers,
    not a diverse sample, so we identify patterns by direct prevalence (≥70%).
    """

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
        description="在目标笔记中的流行度百分比 (% in target notes)",
        ge=0.0,
        le=100.0
    )

    affected_metrics: Dict[str, float] = Field(
        description="影响的指标及其效应大小 (Metrics affected and effect sizes)",
        example={"ctr": 35.2, "comment_rate": 12.1}
    )

    statistical_evidence: str = Field(
        description="统计证据摘要 (e.g., 'prevalence=85.0%, n=17/20')"
    )

    sample_size: int = Field(
        description="样本量 (Total sample size of target notes)",
        gt=0
    )

    examples: List[str] = Field(
        description="具体案例 (Concrete examples from high-scoring notes)"
    )

    # LLM-generated insights
    why_it_works: str = Field(
        description="为什么有效 (Psychological/behavioral explanation)"
    )

    creation_formula: str = Field(
        description="创作公式 (Actionable one-sentence formula in Chinese)"
    )

    key_elements: List[str] = Field(
        description="关键执行要素 (3-5 specific execution points)"
    )

    @field_validator('examples')
    @classmethod
    def validate_examples_length(cls, v: List[str]) -> List[str]:
        """Validate examples list has 1-5 items."""
        if not (1 <= len(v) <= 5):
            raise ValueError(f"examples must contain 1-5 items, got {len(v)}")
        return v

    @field_validator('key_elements')
    @classmethod
    def validate_key_elements_length(cls, v: List[str]) -> List[str]:
        """Validate key_elements list has 3-5 items."""
        if not (3 <= len(v) <= 5):
            raise ValueError(f"key_elements must contain 3-5 items, got {len(v)}")
        return v

    @field_validator('feature_type')
    @classmethod
    def validate_feature_type(cls, v: str) -> str:
        """Validate feature_type is one of: title, cover, content, tag."""
        allowed_types = {'title', 'cover', 'content', 'tag'}
        if v not in allowed_types:
            raise ValueError(f"feature_type must be one of {allowed_types}, got '{v}'")
        return v


class FeatureAnalysis(BaseModel):
    """单个特征对特定指标的分析结果 (Analysis of a single feature for a specific metric).

    Used in metric-centric analysis workflow to store analysis results
    for one feature affecting one metric.
    """

    feature_name: str = Field(
        description="特征名称 (e.g., 'emotional_triggers', 'title_pattern')"
    )

    prevalence_count: int = Field(
        description="具有该特征的笔记数 (Number of notes with this feature)",
        ge=0
    )

    prevalence_pct: float = Field(
        description="流行度百分比 (Prevalence percentage in filtered notes)",
        ge=0.0,
        le=100.0
    )

    examples: List[str] = Field(
        description="真实特征值案例 (Real feature values from high-performing notes)",
        max_length=5
    )

    # LLM-generated fields (context-aware analysis for this metric)
    why_it_works: str = Field(
        description="为什么这个特征对该指标有效 (Why this feature drives this specific metric)"
    )

    creation_formula: str = Field(
        description="可直接套用的创作模板或公式 (Actionable template for this feature)"
    )

    key_elements: List[str] = Field(
        description="3-5个具体、可验证的执行要点 (3-5 specific execution points)",
        min_length=3,
        max_length=5
    )

    @field_validator('examples')
    @classmethod
    def validate_examples_not_empty(cls, v: List[str]) -> List[str]:
        """Validate examples list is not empty and has at most 5 items."""
        if len(v) == 0:
            raise ValueError("examples cannot be empty")
        if len(v) > 5:
            raise ValueError(f"examples can have at most 5 items, got {len(v)}")
        return v

    @field_validator('key_elements')
    @classmethod
    def validate_key_elements_length(cls, v: List[str]) -> List[str]:
        """Validate key_elements has 3-5 items."""
        if not (3 <= len(v) <= 5):
            raise ValueError(f"key_elements must contain 3-5 items, got {len(v)}")
        return v


class MetricSuccessProfile(BaseModel):
    """单个预测指标的成功模式分析 (Success profile for a single prediction metric).

    Contains comprehensive analysis of ALL features affecting this metric,
    generated in a single LLM call for better context and coherence.

    This is the core data structure for metric-centric analysis workflow.
    """

    metric_name: str = Field(
        description="指标名称 (e.g., 'ctr', 'comment_rate', 'sort_score2')"
    )

    sample_size: int = Field(
        description="分析的笔记数量（方差过滤后） (Number of notes analyzed after variance filtering)",
        gt=0
    )

    variance_level: str = Field(
        description="方差水平：'low' (使用全部笔记) 或 'high' (使用top 50%笔记) | Variance level"
    )

    relevant_features: List[str] = Field(
        description="该指标的相关特征列表（来自attribution规则） (Relevant features from attribution rules)"
    )

    feature_analyses: Dict[str, FeatureAnalysis] = Field(
        description="每个特征的详细分析，key为feature_name (Detailed analysis for each feature)"
    )

    metric_success_narrative: str = Field(
        description="该指标成功的整体叙述（2-3句话，说明这些特征如何协同驱动该指标） (Holistic explanation)"
    )

    timestamp: str = Field(
        description="分析时间戳 (Analysis timestamp in ISO 8601 format)"
    )

    @field_validator('variance_level')
    @classmethod
    def validate_variance_level(cls, v: str) -> str:
        """Validate variance_level is one of: low, high, low_sample_fallback."""
        allowed_levels = {'low', 'high', 'low_sample_fallback'}
        if v not in allowed_levels:
            raise ValueError(f"variance_level must be one of {allowed_levels}, got '{v}'")
        return v

    @field_validator('metric_success_narrative')
    @classmethod
    def validate_narrative_length(cls, v: str) -> str:
        """Validate narrative is substantial (>30 characters)."""
        if len(v) <= 30:
            raise ValueError(f"metric_success_narrative must be >30 characters, got {len(v)}")
        return v


class SuccessProfileReport(BaseModel):
    """竞品成功模式分析报告 (Success profile analysis report).

    Complete report from CompetitorAnalyst agent analyzing why target_notes
    achieve high prediction scores. Uses metric-centric structure for gap analysis.

    Includes:
    - Aggregated baseline statistics
    - Metric-centric success profiles (one per prediction metric)
    - LLM-synthesized cross-metric summary insights
    """

    keyword: str = Field(
        description="目标关键词 (Target keyword)"
    )

    sample_size: int = Field(
        description="分析的竞品笔记数量 (Number of target notes analyzed)",
        gt=0
    )

    # Aggregated baseline statistics
    aggregated_stats: AggregatedMetrics = Field(
        description="竞品笔记的聚合统计 (Aggregated stats from DataAggregatorTool)"
    )

    # Metric-centric success profiles (core data structure)
    metric_profiles: List[MetricSuccessProfile] = Field(
        description="各指标的成功模式分析 (Success profiles for each prediction metric, metric-centric)"
    )

    # Cross-metric summary insights (LLM-synthesized)
    key_success_factors: List[str] = Field(
        description="关键成功要素 (Top 3-5 cross-metric success factors, concise and impactful)"
    )

    viral_formula_summary: str = Field(
        description="爆款公式总结 (Cross-metric holistic summary, concise and focused)"
    )

    # Metadata
    analysis_timestamp: str = Field(
        description="分析时间戳 (ISO 8601 format, e.g., '2025-11-14T10:30:00Z')"
    )

    @field_validator('key_success_factors')
    @classmethod
    def validate_key_success_factors_length(cls, v: List[str]) -> List[str]:
        """Validate key_success_factors list has 3-5 items."""
        if not (3 <= len(v) <= 5):
            raise ValueError(f"key_success_factors must contain 3-5 items, got {len(v)}")
        return v

    @field_validator('viral_formula_summary')
    @classmethod
    def validate_summary_length(cls, v: str) -> str:
        """Validate viral_formula_summary is substantial (>50 characters)."""
        if len(v) <= 50:
            raise ValueError(f"viral_formula_summary must be >50 characters, got {len(v)}")
        return v


# Placeholder models for future agents

class AuditReport(BaseModel):
    """自有笔记审计报告 (Owned note audit report).

    Placeholder for OwnedNoteAuditor agent output (to be implemented in future change).
    """

    note_id: str = Field(description="笔记ID")
    keyword: str = Field(description="关键词")
    audit_timestamp: str = Field(description="审计时间戳")

    # Additional fields will be added when OwnedNoteAuditor is implemented


class GapReport(BaseModel):
    """差距分析报告 (Gap analysis report).

    Placeholder for GapFinder agent output (to be implemented in future change).
    """

    keyword: str = Field(description="关键词")
    owned_note_id: str = Field(description="自有笔记ID")
    gap_timestamp: str = Field(description="分析时间戳")

    # Additional fields will be added when GapFinder is implemented


class OptimizationPlan(BaseModel):
    """优化策略方案 (Optimization strategy plan).

    Placeholder for OptimizationStrategist agent output (to be implemented in future change).
    """

    keyword: str = Field(description="关键词")
    owned_note_id: str = Field(description="自有笔记ID")
    plan_timestamp: str = Field(description="生成时间戳")

    # Additional fields will be added when OptimizationStrategist is implemented
