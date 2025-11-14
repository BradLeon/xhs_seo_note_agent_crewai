"""Report models - 报告数据模型.

Pydantic models for agent-generated reports:
- FeaturePattern: Single content pattern with statistical evidence
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
    - Statistical evidence (prevalence, z-score, p-value)
    - Concrete examples from real notes
    - LLM-generated insights (why it works, creation formula)
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
        description="高分组中的流行度百分比 (% in high-scoring group)",
        ge=0.0,
        le=100.0
    )

    baseline_pct: float = Field(
        description="基线组中的流行度百分比 (% in baseline group)",
        ge=0.0,
        le=100.0
    )

    affected_metrics: Dict[str, float] = Field(
        description="影响的指标及其效应大小 (Metrics affected and effect sizes)",
        example={"ctr": 35.2, "comment_rate": 12.1}
    )

    statistical_evidence: str = Field(
        description="统计证据摘要 (e.g., 'z=3.2, p<0.001, n=85/100')"
    )

    z_score: float = Field(
        description="Z分数 (Z-score for prevalence difference)"
    )

    p_value: float = Field(
        description="P值 (P-value for significance test)",
        ge=0.0,
        le=1.0
    )

    sample_size_high: int = Field(
        description="高分组样本量 (Sample size in high-scoring group)",
        gt=0
    )

    sample_size_baseline: int = Field(
        description="基线组样本量 (Sample size in baseline group)",
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


class SuccessProfileReport(BaseModel):
    """竞品成功模式分析报告 (Success profile analysis report).

    Complete report from CompetitorAnalyst agent analyzing why target_notes
    achieve high prediction scores. Includes:
    - Aggregated baseline statistics
    - Feature patterns organized by type (title/cover/content/tag)
    - LLM-synthesized summary insights
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
        description="关键成功要素 (Top 3-5 most impactful patterns)"
    )

    viral_formula_summary: str = Field(
        description="爆款公式总结 (Holistic creation template summary)"
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
