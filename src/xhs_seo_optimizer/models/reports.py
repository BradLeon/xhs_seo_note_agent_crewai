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

from typing import List, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator

from .analysis_results import AggregatedMetrics, TextAnalysisResult, VisionAnalysisResult


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

    @model_validator(mode='before')
    @classmethod
    def convert_null_lists_to_empty(cls, data: Any) -> Any:
        """Convert null/None values to empty lists for List[str] fields."""
        if isinstance(data, dict):
            list_fields = ['examples', 'key_elements']
            for field in list_fields:
                if field in data and data[field] is None:
                    data[field] = []
        return data


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

    @model_validator(mode='before')
    @classmethod
    def convert_null_lists_to_empty(cls, data: Any) -> Any:
        """Convert null/None values to empty lists for List[str] fields."""
        if isinstance(data, dict):
            list_fields = ['examples', 'key_elements']
            for field in list_fields:
                if field in data and data[field] is None:
                    data[field] = []
        return data


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

    @model_validator(mode='before')
    @classmethod
    def convert_null_lists_to_empty(cls, data: Any) -> Any:
        """Convert null/None values to empty lists for List[str] fields."""
        if isinstance(data, dict):
            if 'relevant_features' in data and data['relevant_features'] is None:
                data['relevant_features'] = []
        return data


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

    @model_validator(mode='before')
    @classmethod
    def convert_null_lists_to_empty(cls, data: Any) -> Any:
        """Convert null/None values to empty lists for List fields."""
        if isinstance(data, dict):
            list_fields = ['metric_profiles', 'key_success_factors']
            for field in list_fields:
                if field in data and data[field] is None:
                    data[field] = []
        return data


# Placeholder models for future agents

class AuditReport(BaseModel):
    """自营笔记审计报告 (Owned note audit report).

    Objective content understanding of owned (self-published) notes:
    - Feature extraction (text + visual analysis via tools)
    - Feature attribution (which features affect which metrics)
    - Objective feature summary (no strength/weakness judgment)

    Note: Strength/weakness judgment requires competitor comparison,
    which will be handled by GapFinder agent (future change).

    Generated by OwnedNoteAuditor agent.
    """

    # Basic info
    note_id: str = Field(
        description="笔记ID (Note ID)"
    )

    keyword: str = Field(
        description="目标关键词 (Target keyword)"
    )

    # Extracted features (raw tool outputs)
    text_features: TextAnalysisResult = Field(
        description="文本特征分析结果 (Text features from NLPAnalysisTool)"
    )

    visual_features: VisionAnalysisResult = Field(
        description="视觉特征分析结果 (Visual features from MultiModalVisionTool)"
    )

    # Current prediction metrics (for GapFinder)
    current_metrics: Dict[str, float] = Field(
        description="当前笔记的预测指标 (Prediction metrics from owned_note, excluding note_id)"
    )

    # Objective feature summary (no strength/weakness judgment)
    feature_summary: str = Field(
        description="客观特征摘要 (Objective description of feature combination, 50-300 chars, "
                    "no judgment words like 'good/bad/weak/strong'). "
                    "Example: '该笔记采用疑问句标题配合情感词汇,开头使用对比钩子,正文框架为问题解决型结构,封面视觉风格简约清新。'"
    )

    # Metadata
    audit_timestamp: str = Field(
        description="审计时间戳 (ISO 8601 format, e.g., '2025-11-20T10:30:00Z')"
    )

    @field_validator('feature_summary')
    @classmethod
    def validate_feature_summary_length(cls, v: str) -> str:
        """Validate feature_summary is substantial and objective (50-300 chars)."""
        if not (50 <= len(v) <= 300):
            raise ValueError(f"feature_summary must be 50-300 characters, got {len(v)}")
        # Check for judgment words (should be objective)
        judgment_words = ['好', '坏', '弱', '差', '优秀', '不足', 'weak', 'bad', 'poor', 'strong', 'excellent']
        lower_text = v.lower()
        found_judgments = [word for word in judgment_words if word in lower_text]
        if found_judgments:
            raise ValueError(f"feature_summary should be objective, found judgment words: {found_judgments}")
        return v


class MetricGap(BaseModel):
    """单个指标的差距分析 (Gap analysis for a single metric).

    Final schema for GapReport output. Includes all statistical, attribution,
    and narrative fields.
    """

    metric_name: str = Field(
        description="指标名称 (e.g., 'ctr', 'comment_rate', 'sort_score2')"
    )

    # Statistical fields
    owned_value: float = Field(
        description="客户笔记的当前值 (Owned note's current value)"
    )

    target_mean: float = Field(
        description="竞品笔记的平均值 (Target notes' mean value)"
    )

    target_std: float = Field(
        description="竞品笔记的标准差 (Target notes' std dev)"
    )

    delta_absolute: float = Field(
        description="绝对差距 (Absolute difference: owned - target)"
    )

    delta_pct: float = Field(
        description="百分比差距 (Percentage difference: (owned - target) / target * 100)"
    )

    z_score: float = Field(
        description="Z分数 (Z-score: (owned - target) / std)"
    )

    p_value: float = Field(
        description="P值 (P-value from two-tailed test)",
        ge=0.0,
        le=1.0
    )

    significance: str = Field(
        description="显著性水平: critical | very_significant | significant | marginal | none | undefined"
    )

    interpretation: str = Field(
        description="人类可读的解释 (Human-readable interpretation of the gap)"
    )

    priority_rank: int = Field(
        description="优先级排名 (1-based ranking, 1 is highest priority)",
        gt=0
    )

    # Feature attribution (from attribution.py)
    related_features: List[str] = Field(
        description="影响该指标的相关特征 (Features that affect this metric, from attribution.py)"
    )

    rationale: str = Field(
        description="特征归因理由 (Rationale explaining why these features affect the metric)"
    )

    # Feature mapping results
    missing_features: List[str] = Field(
        description="缺失的关键特征 (Features absent in owned_note but present in success profile)"
    )

    weak_features: List[str] = Field(
        description="执行不足的特征 (Features present but poorly executed)"
    )

    # Narrative
    gap_explanation: str = Field(
        description="差距原因解释 (Why this gap exists, 2-3 sentences, 50-200 chars)",
        min_length=10,
        max_length=200
    )

    recommendation_summary: str = Field(
        description="改进建议摘要 (What to improve, 1-2 sentences, 20-100 chars)",
        min_length=20,
        max_length=100
    )

    @field_validator('significance')
    @classmethod
    def validate_significance(cls, v: str) -> str:
        """Validate significance is one of allowed values."""
        allowed = {'critical', 'very_significant', 'significant', 'marginal', 'none', 'undefined'}
        if v not in allowed:
            raise ValueError(f"significance must be one of {allowed}, got '{v}'")
        return v

    @model_validator(mode='before')
    @classmethod
    def convert_null_lists_to_empty(cls, data: Any) -> Any:
        """Convert null/None values to empty lists for List[str] fields."""
        if isinstance(data, dict):
            list_fields = ['related_features', 'missing_features', 'weak_features']
            for field in list_fields:
                if field in data and data[field] is None:
                    data[field] = []
        return data


class GapReport(BaseModel):
    """差距分析报告 (Gap analysis report).

    Identifies statistically significant performance gaps between owned_note
    and target_notes, prioritizes them, and maps them to actionable features.

    Generated by GapFinder agent.
    """

    keyword: str = Field(
        description="目标关键词 (Target keyword)"
    )

    owned_note_id: str = Field(
        description="自有笔记ID (Owned note ID)"
    )

    # Statistical gap analysis
    significant_gaps: List[MetricGap] = Field(
        description="显著性差距列表 (p < 0.05), ordered by priority_rank"
    )

    marginal_gaps: List[MetricGap] = Field(
        description="边缘显著差距列表 (0.05 <= p < 0.10)"
    )

    non_significant_gaps: List[MetricGap] = Field(
        description="非显著差距列表 (p >= 0.10)"
    )

    # Cross-metric insights
    top_priority_metrics: List[str] = Field(
        description="Top 3 metrics to focus on (by priority score)",
        max_length=3
    )

    root_causes: List[str] = Field(
        description="3-5 root causes across gaps (e.g., '标题缺乏情感钩子')",
        min_length=3,
        max_length=5
    )

    impact_summary: str = Field(
        description="Overall narrative (50-500 chars)",
        min_length=50,
        max_length=500
    )

    # Metadata
    sample_size: int = Field(
        description="竞品笔记样本量 (Sample size from target_notes)",
        gt=0
    )

    gap_timestamp: str = Field(
        description="分析时间戳 (ISO 8601 format, e.g., '2025-11-21T10:30:00Z')"
    )

    @field_validator('top_priority_metrics')
    @classmethod
    def validate_top_priority_metrics_length(cls, v: List[str]) -> List[str]:
        """Validate top_priority_metrics has at most 3 items."""
        if len(v) > 3:
            raise ValueError(f"top_priority_metrics can have at most 3 items, got {len(v)}")
        return v

    @field_validator('root_causes')
    @classmethod
    def validate_root_causes_length(cls, v: List[str]) -> List[str]:
        """Validate root_causes has 3-5 items."""
        if not (3 <= len(v) <= 5):
            raise ValueError(f"root_causes must contain 3-5 items, got {len(v)}")
        return v

    @field_validator('impact_summary')
    @classmethod
    def validate_impact_summary_length(cls, v: str) -> str:
        """Validate impact_summary is 50-500 characters."""
        if not (50 <= len(v) <= 500):
            raise ValueError(f"impact_summary must be 50-500 characters, got {len(v)}")
        return v

    @model_validator(mode='before')
    @classmethod
    def convert_null_lists_to_empty(cls, data: Any) -> Any:
        """Convert null/None values to empty lists for List fields."""
        if isinstance(data, dict):
            list_fields = [
                'significant_gaps', 'marginal_gaps', 'non_significant_gaps',
                'top_priority_metrics', 'root_causes'
            ]
            for field in list_fields:
                if field in data and data[field] is None:
                    data[field] = []
        return data


class OptimizationPlan(BaseModel):
    """优化策略方案 (Optimization strategy plan).

    Placeholder for OptimizationStrategist agent output (to be implemented in future change).
    """

    keyword: str = Field(description="关键词")
    owned_note_id: str = Field(description="自有笔记ID")
    plan_timestamp: str = Field(description="生成时间戳")

    # Additional fields will be added when OptimizationStrategist is implemented
