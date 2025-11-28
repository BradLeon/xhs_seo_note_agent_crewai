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

from typing import List, Dict, Any, Optional
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from .analysis_results import AggregatedMetrics, TextAnalysisResult, VisionAnalysisResult, UnifiedGap


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

    feature_analyses: Optional[Dict[str, FeatureAnalysis]] = Field(
        default=None,
        description="每个特征的详细分析，key为feature_name (Detailed analysis for each feature, Optional - 低方差时可为空)"
    )

    metric_success_narrative: str = Field(
        description="该指标成功的整体叙述（2-3句话，说明这些特征如何协同驱动该指标） (Holistic explanation)"
    )

    timestamp: Optional[str] = Field(
        default=None,
        description="分析时间戳 (Analysis timestamp in ISO 8601 format, Optional)"
    )

    @field_validator('variance_level')
    @classmethod
    def validate_variance_level(cls, v: str) -> str:
        """Validate variance_level is one of: low, medium, high."""
        allowed_levels = {'low', 'medium', 'high'}
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
    def convert_null_to_defaults(cls, data: Any) -> Any:
        """Convert null/None values to appropriate defaults."""
        if isinstance(data, dict):
            if 'relevant_features' in data and data['relevant_features'] is None:
                data['relevant_features'] = []
            # feature_analyses: None -> {} (empty dict, Optional field)
            if 'feature_analyses' in data and data['feature_analyses'] is None:
                data['feature_analyses'] = {}
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

    # ========== Phase 0001 新增字段 ==========

    # 内容创作意图 (Forward reference - 类定义在文件末尾)
    content_intent: "ContentIntent" = Field(
        description="内容创作意图 (core_theme, target_persona, key_message) - 必填"
    )

    # 视觉主体信息 (Forward reference - 类定义在文件末尾)
    visual_subjects: Optional["VisualSubjects"] = Field(
        default=None,
        description="视觉主体信息 (subject_type, must_preserve, original_urls)"
    )

    # 营销感相关
    marketing_level: Optional[str] = Field(
        default=None,
        description="当前营销感级别 (来自 note.tag.note_marketing_integrated_level)"
    )

    is_soft_ad: bool = Field(
        default=False,
        description="是否被标记为软广 (marketing_level == '软广')"
    )

    marketing_sensitivity: str = Field(
        default="low",
        description="营销敏感度: high(已是软广需降低) | medium(接近边界) | low(安全)"
    )

    # Metadata
    audit_timestamp: str = Field(
        description="审计时间戳 (ISO 8601 format, e.g., '2025-11-20T10:30:00Z')"
    )

    @field_validator('marketing_sensitivity')
    @classmethod
    def validate_marketing_sensitivity(cls, v: str) -> str:
        """Validate marketing_sensitivity is one of allowed values."""
        allowed = {'high', 'medium', 'low'}
        if v not in allowed:
            raise ValueError(f"marketing_sensitivity must be one of {allowed}, got '{v}'")
        return v

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

    # Statistical gap analysis (uses UnifiedGap from analysis_results)
    significant_gaps: List[UnifiedGap] = Field(
        description="显著性差距列表 (p < 0.05), ordered by priority_rank"
    )

    marginal_gaps: List[UnifiedGap] = Field(
        default_factory=list,
        description="边缘显著差距列表 (0.05 <= p < 0.10)"
    )

    non_significant_gaps: List[UnifiedGap] = Field(
        default_factory=list,
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


class OptimizationItem(BaseModel):
    """单个优化项 (Single optimization item).

    Represents one specific content modification with rationale and targeting info.
    """

    original: str = Field(
        description="原始内容 (Original content before optimization)"
    )

    optimized: str = Field(
        description="优化后内容 (Optimized content, ready to use)"
    )

    rationale: str = Field(
        description="优化理由 (Why this change improves performance, 50-200 chars)",
        min_length=20,
        max_length=300
    )

    targeted_metrics: List[str] = Field(
        description="针对的指标 (Metrics this optimization targets, e.g., ['ctr', 'comment_rate'])"
    )

    targeted_weak_features: List[str] = Field(
        description="针对的弱特征 (Weak features this optimization addresses)"
    )

    @field_validator('targeted_metrics')
    @classmethod
    def validate_targeted_metrics_not_empty(cls, v: List[str]) -> List[str]:
        """Validate targeted_metrics - allow empty for cases like hashtag optimization."""
        # Allow empty list - some optimizations (like hashtags) may not target specific metrics
        return v if v is not None else []


class TitleOptimization(BaseModel):
    """标题优化方案 (Title optimization with alternatives).

    Provides 1-3 alternative titles with one recommended choice.
    If title doesn't need optimization, can include just 1 item with original title.
    """

    alternatives: List[OptimizationItem] = Field(
        description="1-3个备选标题 (1-3 alternative titles; if no optimization needed, include 1 with original)"
    )

    recommended_index: int = Field(
        description="推荐选择的索引 (Index of recommended title, 0-2)",
        ge=0
    )

    selection_rationale: str = Field(
        description="推荐理由或无需优化说明 (Why recommended, or why no optimization needed)",
        min_length=10,
        max_length=200
    )

    @field_validator('alternatives')
    @classmethod
    def validate_alternatives_count(cls, v: List[OptimizationItem]) -> List[OptimizationItem]:
        """Validate 1-3 alternatives."""
        if not (1 <= len(v) <= 3):
            raise ValueError(f"alternatives must contain 1-3 items, got {len(v)}")
        return v

    @model_validator(mode='after')
    def validate_recommended_index_in_range(self):
        """Validate recommended_index is within alternatives range."""
        if self.recommended_index >= len(self.alternatives):
            raise ValueError(f"recommended_index ({self.recommended_index}) must be < len(alternatives) ({len(self.alternatives)})")
        return self


class ContentOptimization(BaseModel):
    """内容优化方案 (Content optimization for text elements).

    Includes opening hook, ending CTA, hashtags, and body improvement suggestions.
    Fields are Optional - only include those that need optimization.
    """

    opening_hook: Optional[OptimizationItem] = Field(
        default=None,
        description="开头钩子优化 (Opening hook optimization, null if no optimization needed)"
    )

    ending_cta: Optional[OptimizationItem] = Field(
        default=None,
        description="结尾互动召唤优化 (Ending CTA optimization, null if no optimization needed)"
    )

    hashtags: Optional[OptimizationItem] = Field(
        default=None,
        description="话题标签优化 (Hashtag optimization, null if no optimization needed)"
    )

    body_improvements: List[str] = Field(
        default_factory=list,
        description="正文改进要点 (Key improvement points for body content, empty if none needed)"
    )

    skip_reason: Optional[str] = Field(
        default=None,
        description="跳过优化的原因 (Reason why certain optimizations were skipped)"
    )


class VisualPrompt(BaseModel):
    """视觉优化 Prompt (Visual optimization prompt for AIGC).

    Since no AIGC model is integrated, provides structured prompts for image generation.
    """

    image_type: str = Field(
        description="图片类型 (cover | inner_1 | inner_2 | ...)"
    )

    prompt_text: str = Field(
        description="AIGC 生成 prompt (Chinese, detailed description for image generation)",
        min_length=50,
        max_length=500
    )

    style_reference: str = Field(
        description="参考风格描述 (Style reference, e.g., '小红书爆款育儿笔记风格')",
        min_length=10,
        max_length=100
    )

    key_elements: List[str] = Field(
        description="必须包含的元素 (Elements that must be in the image)",
        min_length=2,
        max_length=6
    )

    color_scheme: str = Field(
        description="推荐色彩方案 (Recommended color scheme, e.g., '暖色调，柔和黄色和白色为主')",
        min_length=10,
        max_length=100
    )

    targeted_metrics: List[str] = Field(
        description="针对的指标 (Metrics this visual targets)"
    )

    @field_validator('image_type')
    @classmethod
    def validate_image_type(cls, v: str) -> str:
        """Validate image_type format."""
        if not v.startswith('inner_') and v != 'cover':
            raise ValueError(f"image_type must be 'cover' or 'inner_N', got '{v}'")
        return v

    @field_validator('key_elements')
    @classmethod
    def validate_key_elements_length(cls, v: List[str]) -> List[str]:
        """Validate key_elements has 2-6 items."""
        if not (2 <= len(v) <= 6):
            raise ValueError(f"key_elements must contain 2-6 items, got {len(v)}")
        return v


class VisualOptimization(BaseModel):
    """视觉优化方案 (Visual optimization with prompts).

    Provides AIGC prompts for cover and inner images.
    Fields are Optional - only include those that need optimization.
    """

    cover_prompt: Optional[VisualPrompt] = Field(
        default=None,
        description="封面图优化 prompt (Cover image generation prompt, null if no optimization needed)"
    )

    inner_image_prompts: List[VisualPrompt] = Field(
        default_factory=list,
        description="内页图优化 prompts (Inner image generation prompts, empty if none needed)"
    )

    general_visual_guidelines: List[str] = Field(
        default_factory=list,
        description="通用视觉指南 (General visual improvement guidelines, can be empty)"
    )

    skip_reason: Optional[str] = Field(
        default=None,
        description="跳过视觉优化的原因 (Reason why visual optimization was skipped)"
    )


class OptimizationPlan(BaseModel):
    """优化策略方案 (Optimization strategy plan).

    Complete optimization plan generated by OptimizationStrategist agent.
    Transforms GapReport insights into actionable content modifications.

    Includes:
    - Title optimization (3 alternatives)
    - Content optimization (opening, ending, hashtags)
    - Visual optimization (AIGC prompts for images)
    - Priority summary and expected impact
    """

    keyword: str = Field(description="关键词 (Target keyword)")

    owned_note_id: str = Field(description="自有笔记ID (Owned note ID)")

    # Core optimization content
    title_optimization: TitleOptimization = Field(
        description="标题优化方案 (Title optimization with 3 alternatives)"
    )

    content_optimization: ContentOptimization = Field(
        description="内容优化方案 (Content optimization for text elements)"
    )

    visual_optimization: VisualOptimization = Field(
        description="视觉优化方案 (Visual optimization with AIGC prompts)"
    )

    # Summary and impact
    priority_summary: str = Field(
        description="优先执行建议 (Which optimizations to prioritize, 50-200 chars)",
        min_length=50,
        max_length=300
    )

    expected_impact: Dict[str, str] = Field(
        description="预期影响 (Expected impact per metric, e.g., {'ctr': '预计提升10-15%', 'comment_rate': '预计提升20-30%'})"
    )

    # Metadata
    plan_timestamp: str = Field(
        description="生成时间戳 (ISO 8601 format, e.g., '2025-11-24T10:30:00Z')"
    )

    @field_validator('expected_impact')
    @classmethod
    def validate_expected_impact_not_empty(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate expected_impact has at least one entry."""
        if len(v) == 0:
            raise ValueError("expected_impact cannot be empty")
        return v

    @model_validator(mode='before')
    @classmethod
    def convert_null_to_defaults(cls, data: Any) -> Any:
        """Convert null/None values to appropriate defaults."""
        if isinstance(data, dict):
            if 'expected_impact' in data and data['expected_impact'] is None:
                data['expected_impact'] = {}
        return data


# ============================================================================
# Phase 0001: 图像生成与内容一致性优化 - 新增模型
# ============================================================================

class ContentIntent(BaseModel):
    """内容创作意图 - 确保优化一致性的锚点.

    用于在分模块优化时保持内容的中心思想一致性。
    所有优化（标题、开头、结尾、图片）都必须服务于这些核心要素。
    """

    # 必须字段
    core_theme: str = Field(
        description="核心主题 (必须, e.g., 'DHA选购攻略', '新手妈妈育儿经验')",
        min_length=2,
        max_length=50
    )

    target_persona: str = Field(
        description="目标人群 (必须, 结合owned_note内容和keyword确定, e.g., '新手妈妈', '健身爱好者')",
        min_length=2,
        max_length=50
    )

    key_message: str = Field(
        description="关键信息/核心卖点 (必须, e.g., '科学配比是关键', 'DHA含量是选择标准')",
        min_length=5,
        max_length=100
    )

    # 可选字段
    unique_angle: Optional[str] = Field(
        default=None,
        description="独特角度 (可选, e.g., '老爸测评专业视角', '医生妈妈的建议')"
    )

    emotional_tone: Optional[str] = Field(
        default=None,
        description="情感基调 (可选, e.g., '专业但亲切', '轻松幽默', '真诚分享')"
    )

    @field_validator('core_theme', 'target_persona', 'key_message')
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """Validate required fields are not empty or whitespace."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace")
        return v.strip()


class VisualSubjects(BaseModel):
    """视觉主体信息 - 确保生图主体一致性.

    从原始图片中提取核心主体信息，确保优化后的图片
    保留品牌、产品、人物等关键视觉元素。
    """

    subject_type: str = Field(
        description="主体类型: product | person | brand | scene | none"
    )

    subject_description: str = Field(
        description="主体描述 (e.g., 'DHA鱼油瓶装产品，红色瓶盖', '博主本人出镜')",
        min_length=5,
        max_length=200
    )

    brand_elements: List[str] = Field(
        default_factory=list,
        description="品牌元素 (e.g., ['老爸测评logo', '品牌特定配色', '产品包装'])"
    )

    must_preserve: List[str] = Field(
        description="必须保留的元素 (生图时必须包含这些元素以保持一致性)"
    )

    # 保留原始图片URL作为参考
    original_cover_url: str = Field(
        description="原始封面图URL (可作为生图模型的参考输入)"
    )

    original_inner_urls: List[str] = Field(
        default_factory=list,
        description="原始内页图URLs (可作为生图参考)"
    )

    @field_validator('subject_type')
    @classmethod
    def validate_subject_type(cls, v: str) -> str:
        """Validate subject_type is one of allowed values."""
        allowed = {'product', 'person', 'brand', 'scene', 'none'}
        if v not in allowed:
            raise ValueError(f"subject_type must be one of {allowed}, got '{v}'")
        return v

    @field_validator('must_preserve')
    @classmethod
    def validate_must_preserve_not_empty(cls, v: List[str]) -> List[str]:
        """Validate must_preserve has at least one item (unless subject_type is 'none')."""
        # Note: This validation is relaxed; actual enforcement depends on subject_type
        return v

    @model_validator(mode='before')
    @classmethod
    def convert_null_lists_to_empty(cls, data: Any) -> Any:
        """Convert null/None values to empty lists for List fields."""
        if isinstance(data, dict):
            list_fields = ['brand_elements', 'must_preserve', 'original_inner_urls']
            for field in list_fields:
                if field in data and data[field] is None:
                    data[field] = []
        return data


class GeneratedImage(BaseModel):
    """单张生成图片的结果."""

    image_type: str = Field(
        description="图片类型: cover | inner_1 | inner_2 | ..."
    )

    success: bool = Field(
        description="生成是否成功"
    )

    image_url: Optional[str] = Field(
        default=None,
        description="图片URL (Base64 data URL 或 CDN URL)"
    )

    local_path: Optional[str] = Field(
        default=None,
        description="本地保存路径 (e.g., 'outputs/images/cover_xxx.png')"
    )

    error: Optional[str] = Field(
        default=None,
        description="错误信息 (如果生成失败)"
    )

    prompt_used: str = Field(
        description="使用的生成prompt"
    )

    reference_image_used: Optional[str] = Field(
        default=None,
        description="使用的参考图URL (如果有)"
    )


class GeneratedImages(BaseModel):
    """生成的图片结果集合."""

    cover_image: Optional[GeneratedImage] = Field(
        default=None,
        description="封面图生成结果"
    )

    inner_images: List[GeneratedImage] = Field(
        default_factory=list,
        description="内页图生成结果列表"
    )

    generation_timestamp: str = Field(
        description="生成时间戳 (ISO 8601 format)"
    )

    total_generated: int = Field(
        default=0,
        description="成功生成的图片总数"
    )

    total_failed: int = Field(
        default=0,
        description="生成失败的图片总数"
    )

    @model_validator(mode='before')
    @classmethod
    def convert_null_lists_to_empty(cls, data: Any) -> Any:
        """Convert null/None values to empty lists."""
        if isinstance(data, dict):
            if 'inner_images' in data and data['inner_images'] is None:
                data['inner_images'] = []
        return data


class MarketingCheck(BaseModel):
    """营销感检查结果."""

    original_score: float = Field(
        description="原始内容的营销感评分 (0-1, 越高营销感越重)",
        ge=0.0,
        le=1.0
    )

    optimized_score: float = Field(
        description="优化后内容的营销感评分",
        ge=0.0,
        le=1.0
    )

    level: str = Field(
        description="营销感级别: low | medium | high | critical"
    )

    passed: bool = Field(
        description="是否通过检查 (优化后评分不高于原始评分)"
    )

    issues: List[str] = Field(
        default_factory=list,
        description="发现的营销感问题"
    )

    suggestions: List[str] = Field(
        default_factory=list,
        description="降低营销感的建议"
    )

    @field_validator('level')
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate level is one of allowed values."""
        allowed = {'low', 'medium', 'high', 'critical'}
        if v not in allowed:
            raise ValueError(f"level must be one of {allowed}, got '{v}'")
        return v

    @model_validator(mode='before')
    @classmethod
    def convert_null_lists_to_empty(cls, data: Any) -> Any:
        """Convert null/None values to empty lists."""
        if isinstance(data, dict):
            list_fields = ['issues', 'suggestions']
            for field in list_fields:
                if field in data and data[field] is None:
                    data[field] = []
        return data


class OptimizedNote(BaseModel):
    """优化后的完整笔记 - 可直接用于发布.

    整合所有优化结果（标题、内容、图片）为标准Note格式，
    包含溯源信息和验证结果。
    """

    # 基础信息
    note_id: str = Field(
        description="新笔记ID (原ID + '_optimized')"
    )

    original_note_id: str = Field(
        description="原始笔记ID"
    )

    keyword: str = Field(
        description="目标关键词"
    )

    # 优化后的内容
    title: str = Field(
        description="优化后的标题"
    )

    content: str = Field(
        description="优化后的正文内容"
    )

    cover_image_url: str = Field(
        description="封面图URL (生成的新图或原始图)"
    )

    inner_image_urls: List[str] = Field(
        default_factory=list,
        description="内页图URLs"
    )

    # 留空字段（发布后才能获得）
    prediction: Optional[Dict] = Field(
        default=None,
        description="预测指标 (留空，发布后获得)"
    )

    tag: Optional[Dict] = Field(
        default=None,
        description="平台标签 (留空，发布后获得)"
    )

    # 溯源与验证
    content_intent: ContentIntent = Field(
        description="内容创作意图 (用于验证一致性)"
    )

    marketing_check: Optional[MarketingCheck] = Field(
        default=None,
        description="营销感检查结果 (如果原笔记是软广)"
    )

    optimization_summary: str = Field(
        description="优化摘要 (简述做了哪些优化)",
        min_length=20,
        max_length=500
    )

    # 图片来源追踪
    cover_image_source: str = Field(
        description="封面图来源: generated | original",
        default="original"
    )

    inner_images_source: str = Field(
        description="内页图来源: generated | original | mixed",
        default="original"
    )

    # 元数据
    optimized_timestamp: str = Field(
        description="优化时间戳 (ISO 8601 format)"
    )

    @field_validator('cover_image_source')
    @classmethod
    def validate_cover_source(cls, v: str) -> str:
        """Validate cover_image_source is one of allowed values."""
        allowed = {'generated', 'original'}
        if v not in allowed:
            raise ValueError(f"cover_image_source must be one of {allowed}, got '{v}'")
        return v

    @field_validator('inner_images_source')
    @classmethod
    def validate_inner_source(cls, v: str) -> str:
        """Validate inner_images_source is one of allowed values."""
        allowed = {'generated', 'original', 'mixed'}
        if v not in allowed:
            raise ValueError(f"inner_images_source must be one of {allowed}, got '{v}'")
        return v

    @model_validator(mode='before')
    @classmethod
    def convert_null_lists_to_empty(cls, data: Any) -> Any:
        """Convert null/None values to empty lists."""
        if isinstance(data, dict):
            if 'inner_image_urls' in data and data['inner_image_urls'] is None:
                data['inner_image_urls'] = []
        return data


# ============================================================================
# Forward Reference Resolution
# ============================================================================
# AuditReport uses forward references to ContentIntent and VisualSubjects
# which are defined later in this file. This rebuilds the model to resolve them.
AuditReport.model_rebuild()
