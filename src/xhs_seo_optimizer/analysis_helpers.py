"""Analysis helper functions for CompetitorAnalyst agent.

This module provides helper functions for the CompetitorAnalyst agent to:
1. Aggregate statistics from target notes
2. Extract features using NLP and Vision tools
3. Identify statistically significant patterns
4. Synthesize LLM-generated formulas
5. Generate summary insights
"""

import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import json

import scipy.stats as stats

from xhs_seo_optimizer.models.note import Note
from xhs_seo_optimizer.models.analysis_results import (
    AggregatedMetrics,
    TextAnalysisResult,
    VisionAnalysisResult,
)
from xhs_seo_optimizer.models.reports import FeaturePattern
from xhs_seo_optimizer.attribution import get_relevant_features, get_all_metrics

logger = logging.getLogger(__name__)


# === Task 3.2: Aggregate Statistics ===

def aggregate_statistics(notes: List[Note]) -> AggregatedMetrics:
    """Calculate baseline statistics across all target notes.

    Args:
        notes: List of target notes to aggregate

    Returns:
        AggregatedMetrics object with statistical summary

    Raises:
        ValueError: If notes list is empty
    """
    if not notes or len(notes) == 0:
        raise ValueError("At least one note is required for aggregation")

    logger.info(f"Aggregating statistics for {len(notes)} notes...")

    # Lazy import to avoid circular dependency
    from xhs_seo_optimizer.tools import DataAggregatorTool

    # Use DataAggregatorTool
    aggregator = DataAggregatorTool()
    result = aggregator._run(notes=notes)

    logger.info(f"Aggregation complete: {result.sample_size} notes, "
                f"{result.outliers_removed} outliers removed")

    return result


# === Task 3.3: Extract Features Matrix ===

def extract_features_matrix(notes: List[Note]) -> Dict[str, Dict]:
    """Extract NLP and Vision features for all notes.

    Args:
        notes: List of notes to analyze

    Returns:
        Dictionary mapping note_id to feature dict:
        {
            note_id: {
                "text": TextAnalysisResult or None,
                "vision": VisionAnalysisResult or None,
                "prediction": NotePrediction,
                "tag": NoteTag
            }
        }
    """
    logger.info(f"Extracting features from {len(notes)} notes...")

    # Lazy import to avoid circular dependency
    from xhs_seo_optimizer.tools import NLPAnalysisTool, MultiModalVisionTool

    nlp_tool = NLPAnalysisTool()
    vision_tool = MultiModalVisionTool()

    features_matrix = {}

    for i, note in enumerate(notes):
        logger.info(f"Processing note {i+1}/{len(notes)}: {note.note_id}")

        note_features = {
            "prediction": note.prediction,
            "tag": note.tag,
            "text": None,
            "vision": None,
        }

        # Extract text features
        try:
            text_result = nlp_tool._run(note_metadata=note.meta_data)
            note_features["text"] = text_result
        except Exception as e:
            logger.warning(f"NLP analysis failed for note {note.note_id}: {e}")

        # Extract vision features (may fail if no images)
        try:
            vision_result = vision_tool._run(note_metadata=note.meta_data)
            note_features["vision"] = vision_result
        except Exception as e:
            logger.warning(f"Vision analysis failed for note {note.note_id}: {e}")

        features_matrix[note.note_id] = note_features

    successful_text = sum(1 for f in features_matrix.values() if f["text"] is not None)
    successful_vision = sum(1 for f in features_matrix.values() if f["vision"] is not None)

    logger.info(f"Feature extraction complete: {successful_text}/{len(notes)} text, "
                f"{successful_vision}/{len(notes)} vision")

    return features_matrix


# === Task 3.4: Identify Patterns (Layer 1 & 2) ===

def identify_patterns(
    notes: List[Note],
    features_matrix: Dict[str, Dict],
    aggregated_stats: AggregatedMetrics,
    min_prevalence_pct: float = 70.0,
    max_p_value: float = 0.05
) -> List[FeaturePattern]:
    """Identify statistically significant patterns using attribution rules.

    This implements Layer 1 (Domain Rules) and Layer 2 (Statistical Correlation)
    of the hybrid attribution engine.

    Args:
        notes: List of notes
        features_matrix: Feature data for each note
        aggregated_stats: Baseline statistics
        min_prevalence_pct: Minimum prevalence threshold (default 70%)
        max_p_value: Maximum p-value threshold (default 0.05)

    Returns:
        List of FeaturePattern objects (without LLM fields filled)
    """
    logger.info("Identifying patterns with statistical significance...")

    patterns = []

    # Get all metrics that have attribution rules
    metrics = get_all_metrics()

    for metric in metrics:
        # Check if this metric has data
        if metric not in aggregated_stats.prediction_stats:
            logger.debug(f"Skipping {metric}: no data in aggregated_stats")
            continue

        logger.info(f"Analyzing patterns for metric: {metric}")

        # Layer 1: Get relevant features for this metric
        relevant_features = get_relevant_features(metric)

        # Sort notes by metric value
        metric_values = []
        for note in notes:
            try:
                value = getattr(note.prediction, metric)
                if value is not None:
                    metric_values.append((note, value))
            except AttributeError:
                continue

        if len(metric_values) < 10:
            logger.warning(f"Insufficient data for {metric}: only {len(metric_values)} notes")
            continue

        # Sort by metric value descending
        metric_values.sort(key=lambda x: x[1], reverse=True)

        # Split into top 25% (high) vs rest (baseline)
        cutoff = max(1, len(metric_values) // 4)
        high_group = [note for note, _ in metric_values[:cutoff]]
        baseline_group = [note for note, _ in metric_values[cutoff:]]

        logger.debug(f"{metric}: {len(high_group)} high, {len(baseline_group)} baseline")

        # Layer 2: Calculate pattern prevalence for relevant features
        # This is a simplified implementation - in production you'd extract
        # actual feature values from features_matrix and check prevalence
        # For now, we'll create a placeholder pattern structure

        # NOTE: Full implementation would:
        # 1. For each relevant_feature, check how many notes in high_group have it
        # 2. Calculate prevalence_pct = (count_high / len(high_group)) * 100
        # 3. Calculate baseline_pct = (count_baseline / len(baseline_group)) * 100
        # 4. Compute z-score and p-value for the difference
        # 5. Filter patterns with prevalence_pct >= min_prevalence_pct and p_value < max_p_value

        # Placeholder: Return empty list for now - full implementation needed
        # This allows the system to compile and run without actual pattern detection

    logger.info(f"Pattern identification complete: {len(patterns)} patterns found")

    return patterns


def _calculate_pattern_stats(
    high_count: int,
    high_total: int,
    baseline_count: int,
    baseline_total: int
) -> Tuple[float, float, float]:
    """Calculate statistical metrics for a pattern.

    Args:
        high_count: Number of high-group notes with pattern
        high_total: Total notes in high group
        baseline_count: Number of baseline notes with pattern
        baseline_total: Total notes in baseline group

    Returns:
        Tuple of (prevalence_pct, z_score, p_value)
    """
    prevalence_pct = (high_count / high_total * 100) if high_total > 0 else 0
    baseline_pct = (baseline_count / baseline_total * 100) if baseline_total > 0 else 0

    # Z-test for proportions
    if high_total > 0 and baseline_total > 0:
        p_high = high_count / high_total
        p_baseline = baseline_count / baseline_total
        p_combined = (high_count + baseline_count) / (high_total + baseline_total)

        if p_combined > 0 and p_combined < 1:
            se = (p_combined * (1 - p_combined) * (1/high_total + 1/baseline_total)) ** 0.5
            z_score = (p_high - p_baseline) / se if se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
        else:
            z_score = 0
            p_value = 1.0
    else:
        z_score = 0
        p_value = 1.0

    return prevalence_pct, z_score, p_value


# === Task 3.5: Synthesize Formulas (Layer 3) ===

def synthesize_formulas(
    patterns: List[FeaturePattern],
    notes: List[Note]
) -> List[FeaturePattern]:
    """Use LLM to generate explanations and formulas for patterns.

    This implements Layer 3 (LLM Validation) of the hybrid attribution engine.

    Args:
        patterns: List of patterns with statistical evidence
        notes: Original notes for examples

    Returns:
        Updated list of patterns with LLM-generated fields filled
    """
    logger.info(f"Synthesizing formulas for {len(patterns)} patterns...")

    # NOTE: Full implementation would:
    # 1. For each pattern, build LLM prompt with:
    #    - Pattern description and statistical evidence
    #    - Concrete examples from high-scoring notes
    # 2. Call LLM API (DeepSeek via OpenRouter)
    # 3. Parse response to extract:
    #    - why_it_works: Psychological explanation
    #    - creation_formula: One-sentence actionable formula in Chinese
    #    - key_elements: 3-5 specific execution points
    # 4. Update FeaturePattern objects with LLM fields

    # Placeholder: Return patterns unchanged for now
    # This allows system to compile without actual LLM integration

    for pattern in patterns:
        # Placeholder values - full implementation would call LLM
        pattern.why_it_works = "待LLM分析 - 需要实现LLM调用逻辑"
        pattern.creation_formula = "待生成 - 需要实现LLM调用逻辑"
        pattern.key_elements = ["要素1", "要素2", "要素3"]

    logger.info("Formula synthesis complete")

    return patterns


# === Task 3.6: Generate Summary Insights ===

def generate_summary_insights(
    patterns: List[FeaturePattern]
) -> Tuple[List[str], str]:
    """Generate key success factors and viral formula summary.

    Args:
        patterns: All discovered patterns

    Returns:
        Tuple of (key_success_factors, viral_formula_summary)
    """
    logger.info(f"Generating summary insights from {len(patterns)} patterns...")

    # Select top 3-5 patterns by combined impact (z_score * prevalence_pct)
    if len(patterns) == 0:
        # Fallback for no patterns
        key_success_factors = [
            "样本量不足，无法识别显著模式",
            "建议增加分析笔记数量",
            "或降低统计显著性阈值"
        ]
        viral_formula_summary = (
            "由于样本量不足或统计显著性要求较高，本次分析未能识别出明确的成功模式。"
            "建议增加目标笔记数量或调整分析参数后重新分析。"
        )
        return key_success_factors, viral_formula_summary

    # Sort patterns by impact score
    scored_patterns = []
    for pattern in patterns:
        impact_score = pattern.z_score * pattern.prevalence_pct
        scored_patterns.append((pattern, impact_score))

    scored_patterns.sort(key=lambda x: x[1], reverse=True)

    # Select top 3-5 patterns
    top_patterns = [p for p, _ in scored_patterns[:5]]

    # NOTE: Full implementation would:
    # 1. Build LLM prompt with top patterns and their evidence
    # 2. Ask LLM to synthesize:
    #    - key_success_factors: 3-5 concise insights
    #    - viral_formula_summary: Holistic creation template (>100 chars)
    # 3. Parse and validate response

    # Placeholder implementation
    key_success_factors = []
    for i, pattern in enumerate(top_patterns[:5]):
        factor = f"{pattern.feature_name}: {pattern.description}"
        key_success_factors.append(factor)

    # Ensure we have 3-5 factors
    while len(key_success_factors) < 3:
        key_success_factors.append(f"待分析的成功因素 {len(key_success_factors) + 1}")

    viral_formula_summary = (
        f"基于 {len(patterns)} 个统计显著的内容模式分析，"
        f"发现高表现笔记主要具备以下特征：{', '.join([p.feature_name for p in top_patterns[:3]])}。"
        f"这些模式在高分组中的流行度显著高于基线组，具有可复制性。"
    )

    logger.info("Summary insights generated")

    return key_success_factors, viral_formula_summary


# === Utility Functions ===

def get_current_timestamp() -> str:
    """Get current timestamp in ISO 8601 format.

    Returns:
        ISO 8601 timestamp string
    """
    return datetime.utcnow().isoformat() + 'Z'
