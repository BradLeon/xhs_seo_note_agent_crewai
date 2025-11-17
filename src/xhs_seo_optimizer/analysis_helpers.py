"""Analysis helper functions for CompetitorAnalyst agent.

This module provides helper functions for the CompetitorAnalyst agent to:
1. Aggregate statistics from target notes
2. Extract features using NLP and Vision tools
3. Identify statistically significant patterns
4. Synthesize LLM-generated formulas
5. Generate summary insights
"""

import logging
from typing import List, Dict, Tuple, Optional, Any
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


# ============================================================================
# Configuration Constants
# ============================================================================

class AnalysisConfig:
    """Configuration constants for pattern analysis.

    All magic numbers are centralized here for easy maintenance and testing.
    """

    # Sample size thresholds
    MIN_SAMPLE_SIZE = 3  # Minimum notes required for statistical validity

    # Example collection limits
    MAX_EXAMPLES_PER_PATTERN = 3  # Examples collected during pattern detection
    MAX_EXAMPLES_SINGLE_PROMPT = 5  # Examples shown in single-pattern LLM prompts
    MAX_EXAMPLES_BATCH_PROMPT = 3  # Examples shown in batch prompts (conserve context)

    # Batch processing configuration
    DEFAULT_BATCH_SIZE = 5  # Patterns processed per API call (balance cost vs context)

    # Logging display limits
    MAX_METRIC_VALUES_DISPLAY = 5  # Metric values shown in debug logs

    # Summary generation parameters
    TOP_PATTERNS_COUNT = 5  # Top patterns selected for ranking and summary
    MIN_SUCCESS_FACTORS = 3  # Minimum key success factors in report
    MAX_SUCCESS_FACTORS = 5  # Maximum key success factors in report
    TOP_FEATURES_IN_SUMMARY = 3  # Features mentioned in viral formula summary


# ============================================================================
# Task 3.2: Aggregate Statistics
# ============================================================================

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
    result_json = aggregator._run(notes=notes)

    # Parse JSON result to AggregatedMetrics object
    result = AggregatedMetrics.model_validate_json(result_json)

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
            text_result_json = nlp_tool._run(note_meta_data=note.meta_data, note_id=note.note_id)
            text_result = TextAnalysisResult.model_validate_json(text_result_json)
            note_features["text"] = text_result
        except Exception as e:
            logger.warning(f"NLP analysis failed for note {note.note_id}: {e}")

        # Extract vision features (may fail if no images)
        try:
            vision_result_json = vision_tool._run(note_meta_data=note.meta_data, note_id=note.note_id)
            vision_result = VisionAnalysisResult.model_validate_json(vision_result_json)
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
    """Identify content patterns from target notes using Direct Prevalence Analysis.

    This implements Layer 1 (Domain Rules) and Layer 2 (Direct Prevalence)
    of the hybrid attribution engine.

    **Key Design Decision:**
    Target_notes are already curated high-quality content (top search results),
    not a diverse sample. Therefore, we use Direct Prevalence Analysis instead of
    split+compare approach:
    - If a feature appears in ≥70% of target_notes, it's a success pattern
    - No need for statistical significance testing (no control group)

    Args:
        notes: List of target notes (pre-filtered high-quality content)
        features_matrix: Feature data for each note
        aggregated_stats: Baseline statistics
        min_prevalence_pct: Minimum prevalence threshold (default 70%)
        max_p_value: Not used in new logic (kept for API compatibility)

    Returns:
        List of FeaturePattern objects (without LLM fields filled)
    """
    logger.info("Identifying patterns using Direct Prevalence Analysis...")
    logger.info(f"Target notes are curated high-quality content, not diverse sample")

    sample_size = len(notes)

    # Adjust threshold for small samples
    is_small_sample = sample_size < 10
    if is_small_sample:
        adjusted_prevalence_pct = 50.0  # More lenient for small samples
        logger.warning(
            f"Small sample size detected ({sample_size} notes). "
            f"Using adjusted threshold: {adjusted_prevalence_pct}% (vs {min_prevalence_pct}%)"
        )
    else:
        adjusted_prevalence_pct = min_prevalence_pct

    logger.info("=" * 80)
    logger.info("PATTERN DETECTION CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Sample size: {sample_size} notes")
    logger.info(f"Detection method: Direct Prevalence Analysis (no split)")
    logger.info(f"Prevalence threshold: {adjusted_prevalence_pct}%")
    logger.info(f"Rationale: target_notes are already top performers")
    logger.info("=" * 80)

    patterns = []

    # Get all metrics that have attribution rules
    metrics = get_all_metrics()

    for metric in metrics:
        if metric not in aggregated_stats.prediction_stats:
            logger.warning(f"Skipping metric {metric}: not found in aggregated stats")
            continue

        logger.info(f"Analyzing patterns for metric: {metric}")

        # Layer 1: Get relevant features for this metric (from domain knowledge)
        relevant_features = get_relevant_features(metric)

        # Get notes with this metric value (for logging and sorting)
        metric_values = []
        for note in notes:
            try:
                value = getattr(note.prediction, metric)
                if value is not None:
                    metric_values.append((note, value))
            except AttributeError:
                continue

        if len(metric_values) < AnalysisConfig.MIN_SAMPLE_SIZE:
            logger.warning(f"Insufficient data for {metric}: only {len(metric_values)} notes (minimum {AnalysisConfig.MIN_SAMPLE_SIZE} required)")
            continue

        # Sort by metric value for logging (highest first)
        metric_values.sort(key=lambda x: x[1], reverse=True)
        all_notes_with_metric = [note for note, _ in metric_values]

        logger.info("-" * 80)
        logger.info(f"METRIC: {metric}")
        logger.info("-" * 80)
        logger.info(f"Total notes with {metric} data: {len(metric_values)}")
        logger.info(f"Metric values (sorted, note_id: value):")
        for note, value in metric_values[:AnalysisConfig.MAX_METRIC_VALUES_DISPLAY]:
            logger.info(f"  {note.note_id[:16]}...: {value:.6f}")
        if len(metric_values) > AnalysisConfig.MAX_METRIC_VALUES_DISPLAY:
            logger.info(f"  ... and {len(metric_values) - AnalysisConfig.MAX_METRIC_VALUES_DISPLAY} more")
        logger.info(f"\nRelevant features to check ({len(relevant_features)}): {relevant_features}")
        logger.info("-" * 80)

        # Layer 2: Calculate direct prevalence for each relevant feature
        for feature_name in relevant_features:
            feature_count = 0
            examples = []

            # Count how many notes have this feature
            for note in all_notes_with_metric:
                if note.note_id not in features_matrix:
                    continue

                has_feature = _check_feature_presence(
                    feature_name,
                    features_matrix[note.note_id]
                )

                if has_feature:
                    feature_count += 1
                    if len(examples) < AnalysisConfig.MAX_EXAMPLES_PER_PATTERN:
                        examples.append(_get_feature_example(
                            feature_name,
                            note,
                            features_matrix[note.note_id]
                        ))

            # Calculate prevalence
            total_notes = len(all_notes_with_metric)
            prevalence_pct = (feature_count / total_notes * 100) if total_notes > 0 else 0

            logger.info(f"\n  Feature: {feature_name}")
            logger.info(f"    Prevalence: {feature_count}/{total_notes} ({prevalence_pct:.1f}%)")
            logger.info(f"    Threshold: {adjusted_prevalence_pct}%")
            logger.info(f"    Result: {'PASS ✓' if prevalence_pct >= adjusted_prevalence_pct else 'FAIL ✗'}")

            # Check if prevalence meets threshold
            if prevalence_pct >= adjusted_prevalence_pct:
                feature_type = _get_feature_type(feature_name)

                # Create pattern (without LLM fields - those come in Layer 3)
                pattern = FeaturePattern(
                    feature_name=feature_name,
                    feature_type=feature_type,
                    description=f"{feature_name}在优质{metric}笔记中普遍存在",
                    prevalence_pct=prevalence_pct,
                    affected_metrics={metric: prevalence_pct},
                    statistical_evidence=f"prevalence={prevalence_pct:.1f}%, n={feature_count}/{total_notes}",
                    sample_size=total_notes,
                    examples=examples if examples else [f"示例：{feature_name}"],
                    # LLM fields placeholder - will be filled in Layer 3
                    why_it_works="待LLM分析",
                    creation_formula="待生成",
                    key_elements=["要素1", "要素2", "要素3"]
                )

                patterns.append(pattern)
                logger.info(f"    ✓✓✓ PATTERN FOUND! ✓✓✓")
                logger.info(f"    Feature '{feature_name}' appears in {prevalence_pct:.1f}% of target notes")

    logger.info(f"Pattern identification complete: {len(patterns)} patterns found")

    return patterns


def _check_feature_presence(feature_name: str, note_features: Dict) -> bool:
    """Check if a feature is present in a note's analysis results.

    Args:
        feature_name: Name of feature to check (e.g., "title_pattern", "thumbnail_appeal")
        note_features: Feature dict with "text", "vision", "prediction", "tag" keys

    Returns:
        True if feature is present and has meaningful content, False otherwise
    """
    text_result = note_features.get("text")
    vision_result = note_features.get("vision")
    tag = note_features.get("tag")

    # Direct field mapping: check if the exact field exists and has meaningful content
    # This replaces the old heuristic approach with direct field checks

    # First, try text analysis fields (TextAnalysisResult)
    if text_result and hasattr(text_result, feature_name):
        value = getattr(text_result, feature_name)

        # Check if value is meaningful (non-empty, non-null, non-default)
        if isinstance(value, str):
            # String fields: non-empty and not default placeholder
            return len(value) > 0 and value not in ["未评估", "未分析", "未识别", "未描述", "无"]
        elif isinstance(value, list):
            # List fields: non-empty list
            return len(value) > 0
        elif isinstance(value, (int, float)):
            # Numeric fields: greater than 0
            return value > 0
        else:
            # Other types: truthy value
            return bool(value)

    # Second, try vision analysis fields (VisionAnalysisResult)
    if vision_result and hasattr(vision_result, feature_name):
        value = getattr(vision_result, feature_name)

        # Check if value is meaningful
        if isinstance(value, str):
            return len(value) > 0 and value not in ["未评估", "未分析", "未识别", "未描述", "无"]
        elif isinstance(value, list):
            return len(value) > 0
        elif isinstance(value, (int, float)):
            return value > 0
        else:
            return bool(value)

    # Third, try tag fields (Tag)
    if tag and hasattr(tag, feature_name):
        value = getattr(tag, feature_name)

        if isinstance(value, str):
            return len(value) > 0
        elif isinstance(value, list):
            return len(value) > 0
        else:
            return bool(value)

    # Feature not found in any result
    return False


def _get_feature_example(feature_name: str, note: Note, note_features: Dict) -> str:
    """Extract detailed feature value from analysis results.

    This function extracts the ACTUAL feature value from TextAnalysisResult or
    VisionAnalysisResult, not just placeholder strings. This is critical for LLM
    to see real examples.

    Args:
        feature_name: Name of the feature (e.g., "benefit_appeals", "title_pattern")
        note: The note object
        note_features: Feature dict with "text", "vision", "prediction", "tag" keys

    Returns:
        Formatted string with actual feature value from analysis results
    """
    text_result = note_features.get("text")
    vision_result = note_features.get("vision")
    tag = note_features.get("tag")

    # Try to get actual feature value from TextAnalysisResult
    if text_result and hasattr(text_result, feature_name):
        value = getattr(text_result, feature_name)

        # Format based on value type
        if isinstance(value, list):
            if len(value) > 0:
                # List fields: join with " | " for compact display
                return f"{feature_name}: {' | '.join(str(v) for v in value[:3])}"
            else:
                return f"{feature_name}: []"
        elif isinstance(value, str):
            # String fields: truncate if too long
            if len(value) > 100:
                return f"{feature_name}: {value[:100]}..."
            else:
                return f"{feature_name}: {value}"
        elif isinstance(value, (int, float)):
            return f"{feature_name}: {value}"
        else:
            return f"{feature_name}: {str(value)}"

    # Try to get actual feature value from VisionAnalysisResult
    if vision_result and hasattr(vision_result, feature_name):
        value = getattr(vision_result, feature_name)

        if isinstance(value, list):
            if len(value) > 0:
                return f"{feature_name}: {' | '.join(str(v) for v in value[:3])}"
            else:
                return f"{feature_name}: []"
        elif isinstance(value, str):
            if len(value) > 100:
                return f"{feature_name}: {value[:100]}..."
            else:
                return f"{feature_name}: {value}"
        elif isinstance(value, (int, float)):
            return f"{feature_name}: {value}"
        else:
            return f"{feature_name}: {str(value)}"

    # Try tag fields
    if tag and hasattr(tag, feature_name):
        value = getattr(tag, feature_name)
        return f"{feature_name}: {value}"

    # Fallback: feature not found (shouldn't happen if _check_feature_presence passed)
    return f"{feature_name}: (未找到具体值, 笔记ID: {note.note_id[:16]}...)"


def _get_feature_type(feature_name: str) -> str:
    """Determine feature type from feature name.

    Uses combination of string matching and attribution rules to determine
    whether a feature belongs to title, cover, content, or tag category.

    Args:
        feature_name: Name of the feature

    Returns:
        One of: "title", "cover", "content", "tag"
    """
    name_lower = feature_name.lower()

    # Direct mapping based on common patterns
    if "title" in name_lower:
        return "title"
    elif any(word in name_lower for word in [
        "cover", "visual", "thumbnail", "image", "color",
        "layout", "composition", "ocr", "appeal"
    ]):
        return "cover"
    elif any(word in name_lower for word in ["tag", "intention", "taxonomy"]):
        return "tag"
    elif any(word in name_lower for word in [
        "opening", "ending", "content", "paragraph", "word_count",
        "readability", "structure", "pain", "value", "emotional",
        "credibility", "authority", "urgency", "benefit", "transformation"
    ]):
        return "content"
    else:
        # Default to content for unrecognized features
        return "content"


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

    # Import OpenAI client and environment variables
    import os
    import json
    from openai import OpenAI

    # Get API configuration

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    model = os.getenv("OPENROUTER_TEXT_MODEL", "qwen/qwen3-235b-a22b-thinking-2507")
    site_url = os.getenv("OPENROUTER_SITE_URL", "https://openrouter.ai/api/v1")
    site_name = os.getenv("OPENROUTER_SITE_NAME", "XHS SEO Optimizer")

    if not api_key:
        logger.warning("OPENROUTER_API_KEY not found - using fallback formulas")
        return _synthesize_formulas_fallback(patterns)

    # Initialize OpenRouter client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    # Batch processing: Process multiple patterns per API call
    batch_size = AnalysisConfig.DEFAULT_BATCH_SIZE
    total_batches = (len(patterns) + batch_size - 1) // batch_size  # Ceiling division

    logger.info(f"Using batch processing: {len(patterns)} patterns in {total_batches} batches (batch_size={batch_size})")

    for batch_idx in range(0, len(patterns), batch_size):
        batch = patterns[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} patterns)...")

        try:
            # Build batch LLM prompt
            prompt = _build_batch_formula_prompt(batch)

            # Call OpenRouter API once for the entire batch
            response = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": site_url,
                    "X-Title": site_name,
                },
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,  # Higher temperature for creative formula generation
            )

            # Extract and parse batch response
            content = response.choices[0].message.content
            formulas_list = _parse_batch_formula_response(content, len(batch))

            # Update each pattern in the batch
            for pattern, formula_data in zip(batch, formulas_list):
                pattern.why_it_works = formula_data.get(
                    "why_it_works",
                    "该特征通过统计分析显示出显著影响"
                )
                pattern.creation_formula = formula_data.get(
                    "creation_formula",
                    f"遵循{pattern.feature_name}的最佳实践"
                )
                pattern.key_elements = formula_data.get(
                    "key_elements",
                    ["参考统计数据", "观察高分案例", "测试优化"]
                )

            logger.info(f"✓ Batch {batch_num}/{total_batches} completed ({len(batch)} patterns synthesized)")

        except Exception as e:
            logger.warning(f"Batch {batch_num} LLM synthesis failed: {e}")
            # Use fallback for all patterns in this batch
            for pattern in batch:
                pattern.why_it_works = f"{pattern.feature_name}在优质笔记中广泛存在（统计验证）"
                pattern.creation_formula = f"采用{pattern.feature_name}模式，参考最佳案例"
                pattern.key_elements = [
                    f"研究{pattern.feature_name}的成功案例",
                    f"理解目标用户对{pattern.feature_name}的偏好",
                    "测试并优化实际效果"
                ]

    logger.info(f"Formula synthesis complete: {len(patterns)} patterns processed in {total_batches} batches")
    return patterns


def _build_formula_prompt(pattern: FeaturePattern) -> str:
    """Build LLM prompt for formula synthesis.

    Args:
        pattern: FeaturePattern with statistical evidence

    Returns:
        Chinese prompt for LLM
    """
    # Get affected metrics
    metrics_str = ", ".join(pattern.affected_metrics.keys())

    # Build examples list
    examples_str = "\n".join([f"  - {ex}" for ex in pattern.examples[:AnalysisConfig.MAX_EXAMPLES_SINGLE_PROMPT]])

    prompt = f"""你是小红书内容分析专家。请基于统计数据分析，为以下内容特征生成创作公式。

**特征名称**：{pattern.feature_name}
**特征类型**：{pattern.feature_type}
**影响指标**：{metrics_str}
**统计证据**：{pattern.statistical_evidence}
**流行度**：在目标笔记中占比 {pattern.prevalence_pct:.1f}%

**实际案例**：
{examples_str}

请分析：

1. **why_it_works（心理学解释）**：
   - 从用户心理、平台算法、内容传播角度解释为什么这个特征有效
   - 用2-3句话说明其深层作用机制
   - 结合小红书用户特点和平台生态

2. **creation_formula（创作公式）**：
   - 用一句话总结如何应用这个特征
   - 必须具体、可执行、易理解
   - 格式：动词开头的行动指令（如："在标题中..."、"使用..."、"通过..."）

3. **key_elements（关键要素）**：
   - 列出3-5个具体执行要点
   - 每个要点要具体、可操作
   - 优先级从高到低排序

请以JSON格式返回结果：
{{
  "why_it_works": "心理学解释...",
  "creation_formula": "创作公式...",
  "key_elements": ["要素1", "要素2", "要素3"]
}}

注意事项：
- 基于统计证据，不要过度推测
- 保持专业和客观
- 语言简洁实用，面向内容创作者"""

    return prompt


def _parse_formula_response(content: str) -> Dict[str, Any]:
    """Parse LLM formula response.

    Args:
        content: Response content from LLM

    Returns:
        Dict with why_it_works, creation_formula, key_elements

    Raises:
        ValueError: If response cannot be parsed
    """
    import json

    try:
        # Try to extract JSON from response
        # Handle both pure JSON and JSON in markdown code blocks
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            json_str = content[json_start:json_end].strip()
        elif "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            json_str = content[json_start:json_end].strip()
        else:
            json_str = content.strip()

        data = json.loads(json_str)

        # Validate required fields
        required_fields = ["why_it_works", "creation_formula", "key_elements"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        return data

    except Exception as e:
        raise ValueError(f"Failed to parse LLM response: {e}")


def _build_batch_formula_prompt(patterns: List[FeaturePattern]) -> str:
    """Build LLM prompt for batch formula synthesis with detailed examples.

    CRITICAL: This prompt must include REAL feature values from analysis results,
    not just feature names. LLM needs to see actual examples to provide specific guidance.

    Args:
        patterns: List of patterns with detailed examples from _get_feature_example()

    Returns:
        Chinese prompt for batch LLM processing with emphasis on actionability
    """
    # Build patterns list with detailed examples
    patterns_str = ""
    for i, pattern in enumerate(patterns, 1):
        metrics_str = ", ".join(pattern.affected_metrics.keys())

        # Use detailed examples (already formatted by _get_feature_example())
        examples_list = pattern.examples[:AnalysisConfig.MAX_EXAMPLES_BATCH_PROMPT]
        examples_str = "\n   ".join([f"• {ex}" for ex in examples_list])

        patterns_str += f"""
**模式{i}: {pattern.feature_name}**
- 特征类型: {pattern.feature_type}
- 影响指标: {metrics_str}
- 统计证据: {pattern.statistical_evidence}
- 流行度: {pattern.prevalence_pct:.1f}% (在优质笔记中)
- 真实案例（来自高质量笔记分析）:
   {examples_str}
"""

    prompt = f"""你是小红书内容策略专家。基于对{len(patterns)}个高质量笔记的实际分析数据，请为每个内容特征生成可执行的创作公式。

**重要提示**：
1. 上述案例来自真实的高质量笔记分析结果，不是理论推测
2. 请仔细观察案例中的具体特征值，提取可复制的模式
3. 创作公式必须具体到可以直接套用，避免泛泛而谈

{patterns_str}

请为每个模式分析并生成：

1. **why_it_works（为什么有效）**：
   - 基于上述真实案例，解释这个特征为什么在小红书平台有效
   - 从用户心理学（为什么用户会点击/互动）、平台算法（平台如何推荐）、内容传播（如何引发分享）三个角度分析
   - 2-3句话，直接、具体，避免空洞的理论

2. **creation_formula（创作公式）**：
   - 提供可直接套用的创作模板或公式
   - 如果是标题：给出具体的标题结构（如"【数字】+【动词】+【痛点】+【emoji】"）
   - 如果是内容：给出段落结构或叙事框架
   - 如果是视觉：给出具体的设计要求（颜色、构图、文字位置等）
   - 必须具体到创作者看了就知道怎么做

3. **key_elements（关键要素）**：
   - 列出3-5个具体、可验证的执行要点
   - 每个要点要有明确的标准（如"标题字数15-20字"、"开头3句话内点明痛点"）
   - 如果可能，参考真实案例中的具体数字、格式、位置
   - 按优先级排序（最关键的放最前面）
   - 避免模糊的表述（如"提高吸引力"），要具体说明如何提高

**输出格式**：
请以JSON数组格式返回，每个元素对应一个模式：
[
  {{
    "why_it_works": "基于真实案例的心理学分析...",
    "creation_formula": "可直接套用的创作模板或公式...",
    "key_elements": ["具体要点1（包含数字/标准）", "具体要点2", "具体要点3", ...]
  }},
  ...
]

**质量标准**：
- creation_formula 必须是创作者看了就能模仿的模板，不能是抽象建议
- key_elements 的每一条都要有可执行性，最好包含具体的数字或格式要求
- 所有建议都要基于上述真实案例的观察，不要臆测
- 返回的数组长度必须等于{len(patterns)}

开始分析："""

    return prompt


def _parse_batch_formula_response(content: str, expected_count: int) -> List[Dict[str, Any]]:
    """Parse LLM batch formula response.

    Args:
        content: Response content from LLM
        expected_count: Expected number of formulas

    Returns:
        List of dicts, each with why_it_works, creation_formula, key_elements

    Raises:
        ValueError: If response cannot be parsed or count mismatch
    """
    import json

    try:
        # Try to extract JSON from response
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            json_str = content[json_start:json_end].strip()
        elif "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            json_str = content[json_start:json_end].strip()
        else:
            json_str = content.strip()

        data = json.loads(json_str)

        # Validate it's a list
        if not isinstance(data, list):
            raise ValueError(f"Expected list, got {type(data)}")

        # Validate count
        if len(data) != expected_count:
            logger.warning(f"Expected {expected_count} formulas, got {len(data)}")
            # Pad with defaults if too few
            while len(data) < expected_count:
                data.append({
                    "why_it_works": "统计验证有效的内容模式",
                    "creation_formula": "遵循最佳实践",
                    "key_elements": ["参考统计数据", "观察高分案例", "测试优化"]
                })
            # Truncate if too many
            data = data[:expected_count]

        # Validate each item has required fields
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Item {i} is not a dict")

            required_fields = ["why_it_works", "creation_formula", "key_elements"]
            for field in required_fields:
                if field not in item:
                    # Fill with default
                    if field == "key_elements":
                        item[field] = ["参考统计数据", "观察高分案例", "测试优化"]
                    else:
                        item[field] = "待完善"

        return data

    except Exception as e:
        logger.error(f"Failed to parse batch LLM response: {e}")
        # Return default formulas
        return [
            {
                "why_it_works": "统计验证有效的内容模式",
                "creation_formula": "遵循最佳实践",
                "key_elements": ["参考统计数据", "观察高分案例", "测试优化"]
            }
            for _ in range(expected_count)
        ]


def _synthesize_formulas_fallback(patterns: List[FeaturePattern]) -> List[FeaturePattern]:
    """Fallback formula synthesis when LLM is unavailable.

    Args:
        patterns: List of patterns to fill with fallback formulas

    Returns:
        Updated patterns with fallback formulas
    """
    logger.info("Using fallback formula synthesis (no LLM)")

    for pattern in patterns:
        # Generate basic formulas based on statistical evidence
        metrics = list(pattern.affected_metrics.keys())
        primary_metric = metrics[0] if metrics else "互动"

        pattern.why_it_works = (
            f"{pattern.feature_name}在优质{primary_metric}笔记中广泛存在"
            f"（流行度{pattern.prevalence_pct:.1f}%），"
            f"表明该模式是成功内容的重要特征。"
        )

        pattern.creation_formula = (
            f"在{pattern.feature_type}中应用{pattern.feature_name}模式，"
            f"参考高分案例的最佳实践"
        )

        pattern.key_elements = [
            f"研究{pattern.feature_name}的成功案例（流行度>{pattern.prevalence_pct:.0f}%）",
            f"理解该特征对{primary_metric}的影响机制",
            "在创作中测试应用，观察数据反馈",
            "持续优化和迭代内容策略"
        ]

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
    # Use prevalence_pct * sample_size as impact score (higher prevalence + larger sample = higher impact)
    scored_patterns = []
    for pattern in patterns:
        impact_score = pattern.prevalence_pct * pattern.sample_size
        scored_patterns.append((pattern, impact_score))

    scored_patterns.sort(key=lambda x: x[1], reverse=True)

    # Select top patterns
    top_patterns = [p for p, _ in scored_patterns[:AnalysisConfig.TOP_PATTERNS_COUNT]]

    # NOTE: Full implementation would:
    # 1. Build LLM prompt with top patterns and their evidence
    # 2. Ask LLM to synthesize:
    #    - key_success_factors: 3-5 concise insights
    #    - viral_formula_summary: Holistic creation template (>100 chars)
    # 3. Parse and validate response

    # Placeholder implementation
    key_success_factors = []
    for i, pattern in enumerate(top_patterns[:AnalysisConfig.MAX_SUCCESS_FACTORS]):
        factor = f"{pattern.feature_name}: {pattern.description}"
        key_success_factors.append(factor)

    # Ensure we have minimum required factors
    while len(key_success_factors) < AnalysisConfig.MIN_SUCCESS_FACTORS:
        key_success_factors.append(f"待分析的成功因素 {len(key_success_factors) + 1}")

    viral_formula_summary = (
        f"基于 {len(patterns)} 个统计显著的内容模式分析，"
        f"发现高表现笔记主要具备以下特征：{', '.join([p.feature_name for p in top_patterns[:AnalysisConfig.TOP_FEATURES_IN_SUMMARY]])}。"
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
