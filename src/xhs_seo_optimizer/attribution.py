"""Feature attribution rules for metrics.

This module defines which content features causally affect which prediction metrics,
based on platform mechanics and user behavior psychology.

**重要**: 所有features必须在TextAnalysisResult/VisionAnalysisResult中真实存在
小红书平台机制：双列瀑布流，用户点击前只能看到【标题、封面缩略图、作者、点赞数】
"""

from typing import Dict, List

# Type alias for attribution rule
AttributionRule = Dict[str, any]

METRIC_FEATURE_ATTRIBUTION: Dict[str, AttributionRule] = {
    # ========================================
    # CTR (Click-Through Rate)
    # 用户决定是否点击时的可见元素
    # ========================================
    "ctr": {
        "features": [
            # Title features (从TextAnalysisResult)
            "title_pattern",  # 标题套路模板（疑问句、数字列表等）
            "title_keywords",  # 标题关键词
            "title_emotion",  # 标题情感倾向

            # Cover thumbnail features (从VisionAnalysisResult)
            "thumbnail_appeal",  # 封面图吸引力（最关键）
            "visual_tone",  # 视觉调性
            "color_scheme",  # 色彩方案
            "text_ocr_content_highlight",  # 封面图内突出文字
        ],
        "rationale": "用户在信息流中仅能看到：标题、封面缩略图、作者昵称、点赞数。CTR完全由标题和封面缩略图的吸引力决定"
    },

    # ========================================
    # Comment Rate - 评论由内容深度和引导决定
    # ========================================
    "comment_rate": {
        "features": [
            # Opening hooks (从TextAnalysisResult)
            "opening_strategy",  # 开头策略
            "opening_hook",  # 开头钩子类型
            "opening_impact",  # 开头冲击力

            # Content depth (从TextAnalysisResult)
            "content_framework",  # 正文框架
            "content_logic",  # 内容逻辑层次
            "pain_points",  # 痛点挖掘
            "pain_intensity",  # 痛点强度
            "value_propositions",  # 价值主张

            # Ending CTA (从TextAnalysisResult)
            "ending_technique",  # 结尾技巧
            "ending_cta",  # 行动召唤
            "ending_resonance",  # 结尾共鸣度

            # Credibility (从TextAnalysisResult)
            "credibility_signals",  # 可信度信号
            "social_proof",  # 社会证明
        ],
        "rationale": "评论行为由内容深度、互动引导（ending_cta）和可信度触发。开头钩子吸引阅读，痛点共鸣和结尾召唤促进评论"
    },

    # ========================================
    # Interaction Rate (综合互动率)
    # = like_rate + comment_rate + collect_rate
    # ========================================
    "interaction_rate": {
        "features": [
            # Title (从TextAnalysisResult)
            "title_pattern",
            "title_emotion",

            # Cover & Visual (从VisionAnalysisResult)
            "thumbnail_appeal",
            "visual_tone",
            "color_scheme",

            # Content quality (从TextAnalysisResult)
            "content_framework",
            "value_propositions",
            "emotional_triggers",
            "emotional_intensity",

            # Visual storytelling (从VisionAnalysisResult)
            "visual_storytelling",  # 内页图叙事连贯性
            "image_style",  # 图片风格
        ],
        "rationale": "综合互动（点赞+评论+收藏）受标题、封面、内容情感和视觉叙事共同影响"
    },

    # ========================================
    # Sort Score (平台排序分数)
    # 平台算法综合评估
    # ========================================
    "sort_score2": {
        "features": [
            # Tag classification (从Note.tag)
            "intention_lv2",  # 意图分类
            "taxonomy2",  # 品类分类

            # Content quality (从TextAnalysisResult)
            "content_framework",  # 内容框架
            "structure_completeness",  # 结构完整性
            "readability_score",  # 可读性评分
            "word_count",  # 字数统计

            # Value & credibility (从TextAnalysisResult)
            "value_propositions",  # 价值主张
            "value_hierarchy",  # 价值层次
            "credibility_signals",  # 可信度信号
            "authority_indicators",  # 权威性指标

            # Visual quality (从VisionAnalysisResult)
            "image_quality",  # 图片质量
            "image_content_relation",  # 图文关联度
        ],
        "rationale": "平台优先推荐分类准确、内容完整、有价值的优质内容。意图分类、内容质量、可信度和图文匹配是关键"
    },

    # ========================================
    # Like Rate - 情感共鸣和视觉美感
    # ========================================
    "like_rate": {
        "features": [
            # Emotional appeal (从TextAnalysisResult)
            "emotional_triggers",  # 情感触发器
            "emotional_intensity",  # 情感强度
            "benefit_appeals",  # 利益点吸引
            "transformation_promise",  # 转变承诺

            # Visual aesthetics (从VisionAnalysisResult)
            "visual_tone",  # 视觉调性
            "color_scheme",  # 色彩方案
            "image_style",  # 图片风格
            "visual_storytelling",  # 视觉叙事

            # Personal touch (从VisionAnalysisResult)
            "personal_style",  # 个人风格
            "realistic_and_emotional_tone",  # 真实感和情绪基调
        ],
        "rationale": "点赞是最轻量级的互动，由情感共鸣和视觉美感驱动。真实感和个人风格能增强情感连接"
    },

    # ========================================
    # Share Rate - 社交货币价值
    # ========================================
    "share_rate": {
        "features": [
            # High-value content (从TextAnalysisResult)
            "value_propositions",  # 价值主张
            "value_hierarchy",  # 价值层次
            "benefit_appeals",  # 利益点吸引
            "transformation_promise",  # 转变承诺

            # Social proof (从TextAnalysisResult)
            "social_proof",  # 社会证明
            "credibility_signals",  # 可信度信号
            "authority_indicators",  # 权威性指标

            # Psychological triggers (从TextAnalysisResult)
            "urgency_indicators",  # 紧迫感指标
            "scarcity_elements",  # 稀缺性元素

            # Title appeal (从TextAnalysisResult)
            "title_pattern",  # 标题套路
            "title_emotion",  # 标题情感
        ],
        "rationale": "分享需要内容具备社交货币价值，即转发后能让分享者显得有眼光、有价值。权威性、稀缺性和利益点是关键"
    },

    # ========================================
    # Follow Rate - 创作者长期价值
    # ========================================
    "follow_rate": {
        "features": [
            # Personal branding (从VisionAnalysisResult)
            "personal_style",  # 个人风格
            "brand_consistency",  # 品牌一致性
            "visual_tone",  # 视觉调性

            # Expertise & value (从TextAnalysisResult)
            "authority_indicators",  # 权威性指标
            "credibility_signals",  # 可信度信号
            "value_propositions",  # 价值主张
            "value_hierarchy",  # 价值层次

            # Content quality (从TextAnalysisResult)
            "structure_completeness",  # 结构完整性
            "readability_score",  # 可读性评分
            "transformation_promise",  # 转变承诺

            # Visual quality (从VisionAnalysisResult)
            "image_quality",  # 图片质量
            "visual_hierarchy",  # 视觉层次
        ],
        "rationale": "关注决策基于对创作者长期价值的认可。个人风格、品牌一致性、专业性和持续提供价值的能力是关键"
    },

    # ========================================
    # Collect Rate (收藏率) - 实用参考价值
    # ========================================
    "collect_rate": {
        "features": [
            # Practical value (从TextAnalysisResult)
            "value_propositions",  # 价值主张
            "benefit_appeals",  # 利益点吸引
            "content_framework",  # 内容框架
            "content_logic",  # 内容逻辑层次

            # Credibility (从TextAnalysisResult)
            "credibility_signals",  # 可信度信号
            "authority_indicators",  # 权威性指标
            "social_proof",  # 社会证明

            # Structure (从TextAnalysisResult)
            "structure_completeness",  # 结构完整性
            "paragraph_structure",  # 段落结构特点
            "word_count",  # 字数统计（干货通常较长）

            # Title (从TextAnalysisResult)
            "title_pattern",  # 标题套路
        ],
        "rationale": "收藏表明内容有实用价值，用户想留存以便未来参考。结构完整、逻辑清晰、可信度高的干货内容收藏率高"
    },

    # ========================================
    # CES Rate (Complete Engagement Score)
    # 完整互动评分（平台定义的综合指标）
    # ========================================
    "ces_rate": {
        "features": [
            # Engagement drivers (从TextAnalysisResult)
            "opening_strategy",  # 开头策略（吸引阅读）
            "opening_hook",  # 开头钩子
            "content_framework",  # 内容框架
            "ending_technique",  # 结尾技巧
            "ending_cta",  # 行动召唤

            # Visual engagement (从VisionAnalysisResult)
            "visual_storytelling",  # 视觉叙事
            "visual_hierarchy",  # 视觉层次
            "image_content_relation",  # 图文关联度

            # Emotional & value (从TextAnalysisResult)
            "emotional_triggers",  # 情感触发
            "value_propositions",  # 价值主张
        ],
        "rationale": "完整互动需要从开头到结尾全程吸引用户。开头钩子、内容框架、视觉叙事和结尾召唤共同决定用户是否完整阅读并互动"
    },

    # ========================================
    # Fav Rate (收藏率 - 另一个可能的指标名)
    # 与collect_rate相同
    # ========================================
    "fav_rate": {
        "features": [
            # Same as collect_rate
            "value_propositions",
            "benefit_appeals",
            "content_framework",
            "content_logic",
            "credibility_signals",
            "authority_indicators",
            "social_proof",
            "structure_completeness",
            "paragraph_structure",
            "word_count",
            "title_pattern",
        ],
        "rationale": "收藏（fav_rate）与收藏率（collect_rate）相同：实用干货价值+结构清晰+可信度高"
    },
}


# ========================================
# Feature -> Content Area Mapping
# 显式定义每个feature属于哪个内容区域，避免在prompt中进行模式匹配
# ========================================

FEATURE_CONTENT_AREA: Dict[str, str] = {
    # Title features → "title"
    "title_pattern": "title",
    "title_keywords": "title",
    "title_emotion": "title",

    # Opening features → "opening"
    "opening_strategy": "opening",
    "opening_hook": "opening",
    "opening_impact": "opening",

    # Body/content features → "body"
    "content_framework": "body",
    "content_logic": "body",
    "pain_points": "body",
    "pain_intensity": "body",
    "value_propositions": "body",
    "value_hierarchy": "body",
    "emotional_triggers": "body",
    "emotional_intensity": "body",
    "benefit_appeals": "body",
    "transformation_promise": "body",
    "credibility_signals": "body",
    "authority_indicators": "body",
    "social_proof": "body",
    "urgency_indicators": "body",
    "scarcity_elements": "body",
    "structure_completeness": "body",
    "paragraph_structure": "body",
    "readability_score": "body",
    "word_count": "body",

    # Ending features → "ending"
    "ending_technique": "ending",
    "ending_cta": "ending",
    "ending_resonance": "ending",

    # Hashtag/taxonomy features → "hashtags"
    "intention_lv2": "hashtags",
    "taxonomy2": "hashtags",

    # Visual features → "visual"
    "thumbnail_appeal": "visual",
    "visual_tone": "visual",
    "color_scheme": "visual",
    "text_ocr_content_highlight": "visual",
    "visual_storytelling": "visual",
    "image_style": "visual",
    "image_quality": "visual",
    "image_content_relation": "visual",
    "personal_style": "visual",
    "realistic_and_emotional_tone": "visual",
    "brand_consistency": "visual",
    "visual_hierarchy": "visual",
}


def get_feature_content_area(feature: str) -> str:
    """Get the content area for a specific feature.

    Args:
        feature: Feature name (e.g., 'title_emotion', 'opening_hook')

    Returns:
        Content area string: 'title', 'opening', 'body', 'ending', 'hashtags', or 'visual'
        Returns 'unknown' if feature not found.
    """
    return FEATURE_CONTENT_AREA.get(feature, "unknown")


def get_relevant_features(metric: str) -> List[str]:
    """Get relevant features for a specific metric.

    Args:
        metric: Metric name (e.g., 'ctr', 'comment_rate')

    Returns:
        List of feature names relevant to the metric

    Raises:
        KeyError: If metric is not found in METRIC_FEATURE_ATTRIBUTION
    """
    if metric not in METRIC_FEATURE_ATTRIBUTION:
        raise KeyError(f"Metric '{metric}' not found in METRIC_FEATURE_ATTRIBUTION. "
                      f"Available metrics: {list(METRIC_FEATURE_ATTRIBUTION.keys())}")

    return METRIC_FEATURE_ATTRIBUTION[metric]["features"]


def get_attribution_rationale(metric: str) -> str:
    """Get rationale for why certain features affect a metric.

    Args:
        metric: Metric name (e.g., 'ctr', 'comment_rate')

    Returns:
        Rationale string explaining the attribution logic

    Raises:
        KeyError: If metric is not found in METRIC_FEATURE_ATTRIBUTION
    """
    if metric not in METRIC_FEATURE_ATTRIBUTION:
        raise KeyError(f"Metric '{metric}' not found in METRIC_FEATURE_ATTRIBUTION")

    return METRIC_FEATURE_ATTRIBUTION[metric]["rationale"]


def get_all_metrics() -> List[str]:
    """Get list of all metrics with attribution rules defined.

    Returns:
        List of metric names
    """
    return list(METRIC_FEATURE_ATTRIBUTION.keys())


# ========================================
# Inverted Index: Feature -> Metrics
# ========================================

def build_feature_metric_index() -> Dict[str, List[str]]:
    """Build inverted index mapping each feature to the metrics it affects.

    Returns:
        Dict[feature_name, List[metric_names]]
        Example: {"title_emotion": ["ctr", "interaction_rate", "share_rate"]}
    """
    inverted_index: Dict[str, List[str]] = {}

    for metric, rule in METRIC_FEATURE_ATTRIBUTION.items():
        for feature in rule["features"]:
            if feature not in inverted_index:
                inverted_index[feature] = []
            if metric not in inverted_index[feature]:
                inverted_index[feature].append(metric)

    return inverted_index


# Cached inverted index (built once at module load)
_FEATURE_METRIC_INDEX: Dict[str, List[str]] = None


def get_feature_metric_index() -> Dict[str, List[str]]:
    """Get the cached feature-to-metric inverted index.

    Returns:
        Dict mapping feature names to list of metrics they affect
    """
    global _FEATURE_METRIC_INDEX
    if _FEATURE_METRIC_INDEX is None:
        _FEATURE_METRIC_INDEX = build_feature_metric_index()
    return _FEATURE_METRIC_INDEX


def get_metrics_affected_by_feature(feature: str) -> List[str]:
    """Get all metrics that a specific feature affects.

    Args:
        feature: Feature name (e.g., 'title_emotion', 'thumbnail_appeal')

    Returns:
        List of metric names affected by this feature.
        Returns empty list if feature not found.
    """
    index = get_feature_metric_index()
    return index.get(feature, [])


def get_all_features() -> List[str]:
    """Get list of all unique features across all attribution rules.

    Returns:
        List of unique feature names
    """
    index = get_feature_metric_index()
    return list(index.keys())


def build_optimization_context(
    priority_metrics: List[str],
    weak_features: List[str],
    missing_features: List[str]
) -> Dict[str, any]:
    """Build optimization context for LLM based on metrics and features.

    This function creates a structured context that tells the LLM:
    - Which features need optimization
    - Why (which metrics they affect)
    - The attribution rationale

    Args:
        priority_metrics: Metrics that need improvement (from GapReport)
        weak_features: Features present but poorly executed
        missing_features: Features completely absent

    Returns:
        Structured optimization context dict
    """
    context = {
        "priority_metrics": [],
        "features_to_optimize": {},
        "features_by_content_area": {}
    }

    # Add priority metrics with their feature mappings
    for metric in priority_metrics:
        if metric in METRIC_FEATURE_ATTRIBUTION:
            context["priority_metrics"].append({
                "metric": metric,
                "relevant_features": METRIC_FEATURE_ATTRIBUTION[metric]["features"],
                "rationale": METRIC_FEATURE_ATTRIBUTION[metric]["rationale"]
            })

    # Collect all features that need optimization
    all_problem_features = set(weak_features) | set(missing_features)

    # Build features_to_optimize with their impact info
    for feature in all_problem_features:
        affected_metrics = get_metrics_affected_by_feature(feature)
        # Filter to only priority metrics for relevance
        priority_affected = [m for m in affected_metrics if m in priority_metrics]

        context["features_to_optimize"][feature] = {
            "status": "weak" if feature in weak_features else "missing",
            "content_area": get_feature_content_area(feature),  # 显式指定内容区域
            "affects_metrics": affected_metrics,
            "priority_metrics_affected": priority_affected,
            "optimization_action": "改进现有内容" if feature in weak_features else "新增内容元素"
        }

    # Also group features by content_area for easier prompt consumption
    features_by_area: Dict[str, List[str]] = {}
    for feature, info in context["features_to_optimize"].items():
        area = info["content_area"]
        if area not in features_by_area:
            features_by_area[area] = []
        features_by_area[area].append(feature)
    context["features_by_content_area"] = features_by_area

    return context
