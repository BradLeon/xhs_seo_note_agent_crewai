"""Feature attribution rules for metrics.

This module defines which content features causally affect which prediction metrics,
based on platform mechanics and user behavior psychology.
"""

from typing import Dict, List

# Type alias for attribution rule
AttributionRule = Dict[str, any]

# TODO, 这个规则性的东西需要认真人工review遍。

METRIC_FEATURE_ATTRIBUTION: Dict[str, AttributionRule] = {
    # CTR (Click-Through Rate) - 用户决定是否点击时只能看到标题和封面
    "ctr": {
        "features": [
            # Title features (从NLPAnalysisTool.TextAnalysisResult)
            "title_pattern",
            "title_keywords",
            "title_emotion",
            "title_length",
            "interrogative_title",
            "benefit_focused_title",
            "curiosity_gap",

            # Cover features (从MultiModalVisionTool.VisionAnalysisResult)
            "cover_quality",
            "cover_composition",
            "cover_color_scheme",
            "thumbnail_appeal",
            "visual_tone",
            "cover_text_overlay",
        ],
        "rationale": "用户点击前仅能看到标题和封面缩略图，因此CTR仅受这两个元素影响"
    },

    # Comment Rate - 评论由内容质量和互动引导决定
    "comment_rate": {
        "features": [
            # Content features (从NLPAnalysisTool.TextAnalysisResult)
            "ending_technique",
            "ending_cta",
            "content_framework",
            "pain_points",
            "credibility_signals",
            "question_density",
            "controversial_points",
            "value_propositions",
            "personal_story",

            # Title features (may influence)
            "title_pattern",
            "interrogative_title",
        ],
        "rationale": "评论行为由内容深度和结尾引导触发，标题可能影响初始期待"
    },

    # Interaction Rate (综合互动) - 受多方面影响
    "interaction_rate": {
        "features": [
            # Title features
            "title_pattern",
            "title_emotion",
            "benefit_focused_title",

            # Cover features
            "cover_composition",
            "visual_tone",
            "thumbnail_appeal",

            # Content features
            "content_framework",
            "value_propositions",
            "emotional_triggers",
            "visual_storytelling",
        ],
        "rationale": "综合互动（点赞+评论+收藏）受标题、视觉和内容共同影响"
    },

    # Sort Score (排序分数) - 平台算法综合评估
    "sort_score2": {
        "features": [
            # Tag features (从Note.tag)
            "intention_lv2",
            "taxonomy2",

            # Content features
            "content_framework",
            "value_density",
            "credibility_signals",
            "authenticity",
            "practical_tips",
            "expertise_signals",

            # Title features
            "title_relevance",
            "keyword_coverage",
        ],
        "rationale": "平台优先推荐有价值、真实的干货内容，意图分类和内容质量是关键"
    },

    # Like Rate - 情感共鸣和视觉吸引
    "like_rate": {
        "features": [
            # Content features
            "emotional_triggers",
            "benefit_appeals",
            "transformation_promise",
            "personal_style",

            # Visual features
            "visual_tone",
            "cover_aesthetic",
            "visual_storytelling",
            "color_scheme",

            # Title features
            "title_emotion",
            "benefit_focused_title",
        ],
        "rationale": "点赞由情感共鸣和视觉美感驱动，是最轻量级的互动形式"
    },

    # Share Rate - 内容价值和社交货币
    "share_rate": {
        "features": [
            # Content features
            "value_propositions",
            "practical_tips",
            "social_proof",
            "exclusivity",
            "controversial_points",
            "transformation_promise",
            "expertise_signals",

            # Title features
            "title_pattern",
            "benefit_focused_title",
        ],
        "rationale": "分享行为需要内容具备社交货币价值，即转发后能让分享者获得正面形象"
    },

    # Follow Rate - 长期价值和个人品牌
    "follow_rate": {
        "features": [
            # Content features
            "personal_style",
            "expertise_signals",
            "consistency",
            "value_density",
            "transformation_promise",
            "brand_consistency",
            "authenticity",

            # Visual features
            "visual_tone",
            "brand_consistency",
        ],
        "rationale": "关注决策基于对创作者长期价值的认可，而非单篇内容"
    },

    # Collect Rate (收藏率) - 实用价值和未来参考
    "collect_rate": {
        "features": [
            # Content features
            "practical_tips",
            "value_density",
            "content_framework",
            "credibility_signals",
            "step_by_step_guide",

            # Title features
            "title_pattern",
            "benefit_focused_title",
        ],
        "rationale": "收藏行为表明内容有实用价值，用户想留存以便未来参考"
    },

    # Click Valid Rate (有效点击率) - 内容与预期的匹配度
    "click_valid_rate": {
        "features": [
            # Title features
            "title_relevance",
            "title_accuracy",
            "clickbait_level",

            # Content features
            "content_framework",
            "value_propositions",
            "expectation_match",
        ],
        "rationale": "有效点击率反映标题与内容的匹配度，低clickbait和高相关性提升此指标"
    },

    # Note Valid Rate (笔记有效率) - 平台质量评估
    "note_valid_rate": {
        "features": [
            # Content features
            "authenticity",
            "credibility_signals",
            "content_completeness",
            "value_density",

            # Tag features
            "intention_lv2",
            "taxonomy2",
        ],
        "rationale": "平台判定笔记是否有效的指标，低质量、违规、广告嫌疑会降低此值"
    },
}


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
