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

        # Layer 2: Calculate pattern prevalence for each relevant feature
        for feature_name in relevant_features:
            # Extract feature presence for all notes
            high_count = 0
            baseline_count = 0
            examples = []

            # Count prevalence in high group
            for note in high_group:
                if note.note_id not in features_matrix:
                    continue

                has_feature = _check_feature_presence(
                    feature_name,
                    features_matrix[note.note_id]
                )

                if has_feature:
                    high_count += 1
                    if len(examples) < 3:  # Collect up to 3 examples
                        examples.append(_get_feature_example(
                            feature_name,
                            note,
                            features_matrix[note.note_id]
                        ))

            # Count prevalence in baseline group
            for note in baseline_group:
                if note.note_id not in features_matrix:
                    continue

                has_feature = _check_feature_presence(
                    feature_name,
                    features_matrix[note.note_id]
                )

                if has_feature:
                    baseline_count += 1

            # Calculate statistics
            if len(high_group) == 0 or len(baseline_group) == 0:
                continue

            prevalence_pct, z_score, p_value = _calculate_pattern_stats(
                high_count, len(high_group),
                baseline_count, len(baseline_group)
            )

            baseline_pct = (baseline_count / len(baseline_group) * 100) if len(baseline_group) > 0 else 0

            # Filter by significance thresholds
            if prevalence_pct >= min_prevalence_pct and p_value < max_p_value:
                # Calculate effect size (delta percentage)
                delta_pct = prevalence_pct - baseline_pct

                # Determine feature type
                feature_type = _get_feature_type(feature_name)

                # Create pattern (without LLM fields - those come in Layer 3)
                pattern = FeaturePattern(
                    feature_name=feature_name,
                    feature_type=feature_type,
                    description=f"{feature_name}模式在高{metric}笔记中显著流行",
                    prevalence_pct=prevalence_pct,
                    baseline_pct=baseline_pct,
                    affected_metrics={metric: delta_pct},
                    statistical_evidence=f"z={z_score:.2f}, p={p_value:.4f}, n={high_count}/{len(high_group)}",
                    z_score=z_score,
                    p_value=p_value,
                    sample_size_high=len(high_group),
                    sample_size_baseline=len(baseline_group),
                    examples=examples if examples else [f"示例：{feature_name}"],
                    # LLM fields placeholder - will be filled in Layer 3
                    why_it_works="待LLM分析",
                    creation_formula="待生成",
                    key_elements=["要素1", "要素2", "要素3"]
                )

                patterns.append(pattern)
                logger.info(f"Found significant pattern: {feature_name} for {metric} "
                          f"(prevalence={prevalence_pct:.1f}%, p={p_value:.4f})")

    logger.info(f"Pattern identification complete: {len(patterns)} patterns found")

    return patterns


def _check_feature_presence(feature_name: str, note_features: Dict) -> bool:
    """Check if a feature is present in a note's analysis results.

    Args:
        feature_name: Name of feature to check (e.g., "interrogative_title")
        note_features: Feature dict with "text", "vision", "prediction", "tag" keys

    Returns:
        True if feature is present, False otherwise
    """
    text_result = note_features.get("text")
    vision_result = note_features.get("vision")
    tag = note_features.get("tag")

    # Map feature names to actual checks in analysis results
    # This is a heuristic mapping - can be improved with more sophisticated logic

    # Title features
    if "title" in feature_name.lower():
        if text_result and hasattr(text_result, 'title_framework'):
            title_framework = text_result.title_framework
            if "interrogative" in feature_name.lower() or "疑问" in feature_name.lower():
                return "疑问句" in title_framework or "？" in getattr(note_features.get("prediction"), "title", "")
            if "benefit" in feature_name.lower():
                return any(word in title_framework for word in ["利益", "福利", "优惠", "实用"])
            if "curiosity" in feature_name.lower():
                return any(word in title_framework for word in ["好奇", "悬念", "揭秘"])
            return len(title_framework) > 0  # Has some title pattern
        return False

    # Cover/Visual features
    if "cover" in feature_name.lower() or "visual" in feature_name.lower() or "thumbnail" in feature_name.lower():
        if vision_result:
            if "quality" in feature_name.lower():
                return hasattr(vision_result, 'image_quality') and "高" in vision_result.image_quality
            if "composition" in feature_name.lower():
                return hasattr(vision_result, 'image_composition') and len(vision_result.image_composition) > 0
            if "color" in feature_name.lower():
                return hasattr(vision_result, 'color_scheme') and len(vision_result.color_scheme) > 0
            if "appeal" in feature_name.lower() or "thumbnail" in feature_name.lower():
                return hasattr(vision_result, 'thumbnail_appeal') and "强" in vision_result.thumbnail_appeal
            return True  # Has vision analysis
        return False

    # Content features
    if "content" in feature_name.lower() or "ending" in feature_name.lower():
        if text_result:
            if "ending" in feature_name.lower() or "cta" in feature_name.lower():
                return hasattr(text_result, 'ending_technique') and len(text_result.ending_technique) > 0
            if "framework" in feature_name.lower():
                return hasattr(text_result, 'content_framework') and len(text_result.content_framework) > 0
            if "pain" in feature_name.lower():
                return hasattr(text_result, 'pain_points') and len(text_result.pain_points) > 0
            if "value" in feature_name.lower():
                return hasattr(text_result, 'value_propositions') and len(text_result.value_propositions) > 0
            return True  # Has text analysis
        return False

    # Tag features
    if "intention" in feature_name.lower() or "taxonomy" in feature_name.lower():
        if tag:
            if hasattr(tag, 'intention_lv2'):
                return tag.intention_lv2 is not None and len(tag.intention_lv2) > 0
            if hasattr(tag, 'taxonomy2'):
                return tag.taxonomy2 is not None and len(tag.taxonomy2) > 0
        return False

    # Default: feature name exists somewhere in the text/vision results
    return False


def _get_feature_example(feature_name: str, note: Note, note_features: Dict) -> str:
    """Get a concrete example of a feature from a note.

    Args:
        feature_name: Name of the feature
        note: The note object
        note_features: Feature dict for the note

    Returns:
        String example of the feature
    """
    # Try to extract relevant example text
    if "title" in feature_name.lower() and note.meta_data:
        return f"标题: {note.meta_data.title[:50]}..."

    if "cover" in feature_name.lower() and note.meta_data:
        return f"封面: {note.meta_data.cover_image_url[:50]}..."

    if "content" in feature_name.lower() and note.meta_data:
        content = note.meta_data.content
        return f"内容: {content[:50]}..." if content else "内容示例"

    if "tag" in feature_name.lower() and note.tag:
        return f"标签: {getattr(note.tag, 'intention_lv2', 'N/A')}"

    return f"示例: {feature_name} (笔记ID: {note.note_id})"


def _get_feature_type(feature_name: str) -> str:
    """Determine feature type from feature name.

    Args:
        feature_name: Name of the feature

    Returns:
        One of: "title", "cover", "content", "tag"
    """
    name_lower = feature_name.lower()

    if "title" in name_lower:
        return "title"
    elif any(word in name_lower for word in ["cover", "visual", "thumbnail", "image", "color"]):
        return "cover"
    elif any(word in name_lower for word in ["tag", "intention", "taxonomy"]):
        return "tag"
    else:
        # Default to content for most features
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
    model = os.getenv("OPENROUTER_TEXT_MODEL", "deepseek/deepseek-chat-v3.1")
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

    # Process each pattern
    for i, pattern in enumerate(patterns, 1):
        logger.info(f"Synthesizing formula {i}/{len(patterns)}: {pattern.feature_name}")

        try:
            # Build LLM prompt with statistical evidence and examples
            prompt = _build_formula_prompt(pattern)

            # Call OpenRouter API
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

            # Extract and parse response
            content = response.choices[0].message.content
            formula_data = _parse_formula_response(content)

            # Update pattern with LLM-generated insights
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

            logger.info(f"✓ Formula synthesized for {pattern.feature_name}")

        except Exception as e:
            logger.warning(f"LLM synthesis failed for {pattern.feature_name}: {e}")
            # Use fallback for this pattern
            pattern.why_it_works = f"{pattern.feature_name}在高分笔记中显著流行（统计验证）"
            pattern.creation_formula = f"采用{pattern.feature_name}模式，参考最佳案例"
            pattern.key_elements = [
                f"研究{pattern.feature_name}的成功案例",
                f"理解目标用户对{pattern.feature_name}的偏好",
                "测试并优化实际效果"
            ]

    logger.info("Formula synthesis complete")
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
    examples_str = "\n".join([f"  - {ex}" for ex in pattern.examples[:5]])

    prompt = f"""你是小红书内容分析专家。请基于统计数据分析，为以下内容特征生成创作公式。

**特征名称**：{pattern.feature_name}
**特征类型**：{pattern.feature_type}
**影响指标**：{metrics_str}
**统计证据**：{pattern.statistical_evidence}
**流行度**：在高分笔记中占比 {pattern.prevalence_pct:.1f}%，基线占比 {pattern.baseline_pct:.1f}%

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
            f"{pattern.feature_name}在高{primary_metric}笔记中的流行度显著高于基线"
            f"（{pattern.prevalence_pct:.1f}% vs {pattern.baseline_pct:.1f}%），"
            f"统计检验显示该模式与成功内容存在强相关性。"
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
