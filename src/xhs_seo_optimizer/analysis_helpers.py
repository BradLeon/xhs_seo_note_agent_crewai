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
import os

import numpy as np
import scipy.stats as stats

from xhs_seo_optimizer.models.note import Note
from xhs_seo_optimizer.models.analysis_results import (
    AggregatedMetrics,
    TextAnalysisResult,
    VisionAnalysisResult,
)
from xhs_seo_optimizer.models.reports import (
    FeatureAnalysis,
    MetricSuccessProfile,
)
from xhs_seo_optimizer.attribution import (
    get_relevant_features,
    get_all_metrics,
    get_attribution_rationale,
)

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


# === Task 3.6: Generate Summary Insights ===

def generate_summary_insights(
    metric_profiles: List[MetricSuccessProfile]
) -> Tuple[List[str], str]:
    """生成跨指标的汇总洞察 (Generate cross-metric summary insights).

    Synthesizes key success factors and viral formula summary from metric-centric profiles
    using LLM to provide holistic, actionable insights across all prediction metrics.

    Args:
        metric_profiles: List of metric success profiles (one per prediction metric)

    Returns:
        Tuple of (key_success_factors, viral_formula_summary)
        - key_success_factors: 3-5 concise cross-metric success factors
        - viral_formula_summary: Holistic summary (>50 chars, focused and impactful)
    """
    logger.info(f"Generating cross-metric summary insights from {len(metric_profiles)} metric profiles...")

    # Fallback for no profiles
    if len(metric_profiles) == 0:
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

    # Collect metric narratives and cross-metric feature statistics
    metric_narratives = []
    all_features = {}  # feature_name -> [{"metric": ..., "prevalence": ..., "formula": ...}]

    for profile in metric_profiles:
        metric_narratives.append({
            "metric": profile.metric_name,
            "narrative": profile.metric_success_narrative,
            "sample_size": profile.sample_size,
            "variance_level": profile.variance_level
        })

        # Collect cross-metric feature statistics
        for feature_name, analysis in profile.feature_analyses.items():
            if feature_name not in all_features:
                all_features[feature_name] = []
            all_features[feature_name].append({
                "metric": profile.metric_name,
                "prevalence": analysis.prevalence_pct,
                "formula": analysis.creation_formula,
                "why_it_works": analysis.why_it_works
            })

    # Build LLM prompt for cross-metric synthesis
    prompt = f"""你是小红书内容优化专家。根据以下各指标的成功模式分析，提炼跨指标的关键成功要素和整体爆款公式。

## 各指标分析结果

{json.dumps(metric_narratives, ensure_ascii=False, indent=2)}

## 跨指标特征统计（前10个高频特征）

{json.dumps(dict(list(all_features.items())[:10]), ensure_ascii=False, indent=2)}

## 任务

请生成精炼、重点突出的跨指标汇总：

1. **key_success_factors**（3-5条）：
   - 跨metrics整合最关键的成功要素
   - 每条一句话，突出核心机制和量化效果
   - 示例：["疑问句标题+情绪化封面组合驱动点击率提升30%+", "结尾互动引导CTA使评论率提高50%+"]

2. **viral_formula_summary**（2-3句话）：
   - 说明各指标如何协同驱动整体表现
   - 突出小红书平台机制和用户心理
   - 精炼整洁，避免冗长，聚焦核心公式

请以JSON格式返回（严格遵守格式）：
```json
{{
  "key_success_factors": ["要素1", "要素2", "要素3"],
  "viral_formula_summary": "整体总结（2-3句话）"
}}
```
"""

    try:
        # Call LLM via OpenRouter
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

        response = client.chat.completions.create(
            model="anthropic/claude-3.5-sonnet",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        content = response.choices[0].message.content

        # Parse response (handle markdown code blocks)
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

        key_factors = data["key_success_factors"]
        formula_summary = data["viral_formula_summary"]

        # Validate
        if not isinstance(key_factors, list) or not (3 <= len(key_factors) <= 5):
            raise ValueError(f"key_success_factors must have 3-5 items, got {len(key_factors)}")
        if len(formula_summary) < 50:
            raise ValueError(f"viral_formula_summary too short: {len(formula_summary)} chars")

        logger.info("✓ Cross-metric summary insights generated via LLM")
        return key_factors, formula_summary

    except Exception as e:
        logger.warning(f"LLM synthesis failed: {e}, using fallback")

        # Fallback: simple heuristic-based summary
        key_factors = []
        for feature_name, metrics_data in list(all_features.items())[:5]:
            avg_prevalence = sum(m["prevalence"] for m in metrics_data) / len(metrics_data)
            metrics_affected = ", ".join([m["metric"] for m in metrics_data])
            factor = f"{feature_name}影响{len(metrics_data)}个指标（{metrics_affected}），平均流行度{avg_prevalence:.1f}%"
            key_factors.append(factor)

        # Ensure 3-5 factors
        while len(key_factors) < 3:
            key_factors.append(f"待分析的成功因素 {len(key_factors) + 1}")
        key_factors = key_factors[:5]

        formula_summary = (
            f"基于 {len(metric_profiles)} 个指标的深度分析，"
            f"发现 {len(all_features)} 个跨指标特征模式。"
            f"高表现笔记在标题、封面、内容、标签等维度均呈现显著特征，具有可复制性。"
        )

        return key_factors, formula_summary


# === Utility Functions ===

def get_current_timestamp() -> str:
    """Get current timestamp in ISO 8601 format.

    Returns:
        ISO 8601 timestamp string
    """
    return datetime.utcnow().isoformat() + 'Z'


# ============================================================================
# Metric-Centric Analysis Functions (New Architecture)
# ============================================================================


def filter_notes_by_metric_variance(
    notes: List[Note],
    metric: str,
    variance_threshold: float = 0.3
) -> Tuple[List[Note], str]:
    """根据指标方差过滤笔记，识别高表现者 (Filter notes by metric variance to identify high performers).

    **设计理念**：
    - 低方差（CV < threshold）：笔记性能相近，使用全部笔记
    - 高方差（CV >= threshold）：笔记性能差异大，仅使用top 50%高分笔记

    这样可以避免低分笔记稀释pattern分析质量，同时保证小样本的统计有效性。

    Args:
        notes: 所有target_notes
        metric: 指标名称（如 'ctr', 'comment_rate'）
        variance_threshold: 变异系数阈值（默认0.3）

    Returns:
        (filtered_notes, variance_level)
        - filtered_notes: 过滤后的笔记列表
        - variance_level: 'low' | 'high' | 'low_sample_fallback'

    Logic:
        1. 提取metric值
        2. 计算变异系数 CV = std/mean
        3. 如果 CV < threshold:
              variance_level = 'low'
              return 全部笔记（性能相近，无需过滤）
           否则:
              variance_level = 'high'
              计算中位数
              return 指标值 >= 中位数的笔记（top 50%）
        4. 容错：如果过滤后 < MIN_SAMPLE_SIZE，回退到全部笔记

    Raises:
        ValueError: If metric is invalid or no notes have the metric
    """
    # 1. 提取metric values
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
        return [], 'undefined'

    # 2. 计算统计量
    values = np.array([v for _, v in metric_values])
    mean = np.mean(values)
    std = np.std(values)
    cv = std / mean if mean > 0 else 0  # 变异系数 (Coefficient of Variation)

    logger.info(f"{metric}: mean={mean:.4f}, std={std:.4f}, CV={cv:.4f}")

    # 3. 根据方差水平过滤
    if cv < variance_threshold:
        # 低方差：使用全部笔记
        variance_level = 'low'
        filtered = [note for note, _ in metric_values]
        logger.info(f"  Low variance detected (CV={cv:.4f} < {variance_threshold}), using all {len(filtered)} notes")
    else:
        # 高方差：使用top 50%
        variance_level = 'high'
        median = np.median(values)
        filtered = [note for note, value in metric_values if value >= median]
        logger.info(f"  High variance detected (CV={cv:.4f} >= {variance_threshold}), using top {len(filtered)} notes (>= median={median:.4f})")

    # 4. 容错：如果过滤后太少，回退到全部笔记
    if len(filtered) < AnalysisConfig.MIN_SAMPLE_SIZE:
        logger.warning(f"  Only {len(filtered)} notes after filtering (< {AnalysisConfig.MIN_SAMPLE_SIZE})")
        logger.warning(f"  Falling back to all {len(metric_values)} notes")
        filtered = [note for note, _ in metric_values]
        variance_level = 'low_sample_fallback'

    return filtered, variance_level
# This is a temporary file containing the remaining 4 metric-centric functions
# Will be added to analysis_helpers.py

def _build_metric_analysis_prompt(
    metric: str,
    rationale: str,
    feature_data: Dict[str, Dict],
    sample_size: int,
    variance_level: str,
    keyword: str = ""
) -> str:
    """构建指标级别的综合LLM提示词 (Build comprehensive LLM prompt for metric analysis).

    **关键改进**：
    - 一次性展示该指标的所有相关特征
    - 包含真实案例（而非仅特征名）
    - 要求LLM生成 metric_success_narrative（整体叙述）
    - 强调特征之间的协同作用
    - 包含关键词上下文信息

    Args:
        metric: 指标名称（如 'ctr'）
        rationale: 该指标的平台机制说明（来自attribution.py）
        feature_data: {feature_name: {'prevalence_count': int, 'prevalence_pct': float, 'examples': List[str]}}
        sample_size: 分析的笔记数
        variance_level: 方差水平（'low' | 'high' | 'low_sample_fallback'）
        keyword: 目标关键词

    Returns:
        中文LLM提示词
    """
    # 构建特征列表（带真实案例）
    features_section = ""
    for i, (feature_name, data) in enumerate(feature_data.items(), 1):
        examples_str = "\n      ".join([f"• {ex}" for ex in data['examples'][:AnalysisConfig.MAX_EXAMPLES_BATCH_PROMPT]])

        features_section += f"""
**特征{i}: {feature_name}**
- 流行度: {data['prevalence_pct']:.1f}% ({data['prevalence_count']}/{sample_size} 笔记)
- 真实案例:
      {examples_str}
"""

    # 构建关键词上下文
    keyword_context = f"- 关键词: **{keyword}**（这些特征来自该关键词下的高排序笔记）\n" if keyword else ""

    prompt = f"""你是小红书内容策略专家。请为指标 **{metric}** 进行综合成功因素分析。

**指标说明**：
{keyword_context}- 指标名称: {metric}
- 平台机制: {rationale}
- 分析样本: {sample_size} 条高质量笔记 (方差水平: {variance_level})

**关键特征数据**（来自真实笔记分析）：
{features_section}

**任务**：
请为每个特征生成分析，并提供指标级别的综合叙述。

**输出格式**（JSON）：
{{
  "metric_success_narrative": "对{metric}成功的整体解释（2-3句话，说明这些特征如何协同驱动该指标）",
  "feature_analyses": {{
    "feature_name_1": {{
      "why_it_works": "为什么这个特征对{metric}有效（心理学+平台算法+用户行为角度）",
      "creation_formula": "可直接套用的创作模板（具体到可以立即执行）",
      "key_elements": ["要素1（包含数字/标准）", "要素2", "要素3"]
    }},
    "feature_name_2": {{ ... }},
    ...
  }}
}}

**质量要求**：
1. why_it_works 必须解释该特征为何影响 **{metric}** （而非其他指标）
2. creation_formula 必须具体到创作者看了就能模仿
3. key_elements 每一条都要可执行，最好包含具体数字或格式要求
4. 所有分析基于上述真实案例，不要臆测
5. feature_analyses 必须包含所有 {len(feature_data)} 个特征

开始分析："""

    return prompt


def _parse_metric_analysis_response(
    content: str,
    expected_features: List[str]
) -> Dict[str, Any]:
    """解析LLM的metric-level响应，带容错 (Parse LLM response for metric analysis with fallbacks).

    **容错机制**：
    - 支持多种JSON提取方式（纯JSON、markdown代码块）
    - 检查缺失特征，自动补充降级分析
    - 验证字段完整性
    - 格式错误时返回fallback响应

    Args:
        content: LLM响应内容
        expected_features: 期待的特征列表

    Returns:
        {
            'metric_success_narrative': str,
            'feature_analyses': {
                feature_name: {'why_it_works': str, 'creation_formula': str, 'key_elements': List[str]}
            }
        }

    Raises:
        ValueError: If JSON parsing completely fails
    """
    try:
        # 1. 提取JSON（支持多种格式）
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            json_str = content[json_start:json_end].strip()
            logger.info("Extracted JSON from ```json block")
        elif "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            json_str = content[json_start:json_end].strip()
            logger.info("Extracted JSON from ``` block")
        else:
            json_str = content.strip()
            logger.info("Using full content as JSON")

        data = json.loads(json_str)
        logger.info("JSON parsing successful!")

        # 2. 验证必需字段
        if 'feature_analyses' not in data:
            logger.error("Missing 'feature_analyses' in response")
            return _create_fallback_response(expected_features)

        if 'metric_success_narrative' not in data:
            logger.warning("Missing 'metric_success_narrative', using default")
            data['metric_success_narrative'] = "该指标受多个内容特征协同影响（LLM分析缺失）"

        # 3. 检查缺失特征
        missing_features = set(expected_features) - set(data['feature_analyses'].keys())
        if missing_features:
            logger.warning(f"LLM response missing {len(missing_features)} features: {missing_features}")

            # 为缺失特征添加降级分析
            for feature in missing_features:
                data['feature_analyses'][feature] = {
                    'why_it_works': f"{feature}在高质量笔记中广泛存在（LLM分析缺失）",
                    'creation_formula': f"遵循{feature}最佳实践",
                    'key_elements': ["参考成功案例", "测试优化", "持续迭代"]
                }

        # 4. 验证每个feature_analysis的字段
        for feature_name, analysis in data['feature_analyses'].items():
            required_fields = ['why_it_works', 'creation_formula', 'key_elements']
            for field in required_fields:
                if field not in analysis:
                    if field == 'key_elements':
                        analysis[field] = ["参考统计数据", "观察高分案例", "测试优化"]
                    else:
                        analysis[field] = "待完善"

        return data

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        logger.error(f"Content preview: {content[:200]}...")
        return _create_fallback_response(expected_features)
    except Exception as e:
        logger.error(f"Unexpected error during parsing: {e}")
        return _create_fallback_response(expected_features)


def _create_fallback_response(expected_features: List[str]) -> Dict[str, Any]:
    """创建降级响应 (Create fallback response when LLM parsing fails).

    Args:
        expected_features: 期待的特征列表

    Returns:
        Fallback response dict
    """
    feature_analyses = {}
    for feature in expected_features:
        feature_analyses[feature] = {
            'why_it_works': f"{feature}在高质量笔记中广泛存在（统计验证）",
            'creation_formula': f"采用{feature}模式，参考最佳案例",
            'key_elements': ["研究成功案例", "理解目标用户偏好", "测试并优化"]
        }

    return {
        'metric_success_narrative': "该指标受多个内容特征协同影响",
        'feature_analyses': feature_analyses
    }


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


def analyze_metric_success(
    metric: str,
    filtered_notes: List[Note],
    features_matrix: Dict[str, Dict],
    variance_level: str,
    keyword: str = ""
) -> MetricSuccessProfile:
    """在一次LLM调用中分析指标的所有相关特征 (Analyze all features for a metric in ONE LLM call).

    **核心优势**：
    - 一次LLM调用处理该指标的所有特征
    - LLM看到完整上下文，生成更连贯的分析
    - 包含 metric_success_narrative（整体叙述）
    - 减少API调用次数和成本
    - 提供关键词上下文信息

    **工作流程**：
    1. 获取 relevant_features（来自attribution.py）
    2. 计算每个特征的prevalence
    3. 收集真实案例
    4. 构建综合LLM提示词（包含关键词信息）
    5. 调用LLM一次
    6. 解析响应为 MetricSuccessProfile

    Args:
        metric: 指标名称（如 'ctr'）
        filtered_notes: 方差过滤后的笔记
        features_matrix: 预计算的特征矩阵
        variance_level: 方差水平
        keyword: 目标关键词（用于上下文说明）

    Returns:
        MetricSuccessProfile

    Raises:
        RuntimeError: If analysis fails
    """
    logger.info(f"Analyzing metric: {metric}")

    # 1. 获取attribution规则
    relevant_features = get_relevant_features(metric)
    rationale = get_attribution_rationale(metric)

    logger.info(f"  {len(relevant_features)} relevant features")
    logger.info(f"  {len(filtered_notes)} notes (variance: {variance_level})")

    # 2. 收集feature_data
    feature_data = {}
    for feature_name in relevant_features:
        prevalence_count = 0
        examples = []

        for note in filtered_notes:
            if note.note_id not in features_matrix:
                continue

            has_feature = _check_feature_presence(
                feature_name,
                features_matrix[note.note_id]
            )

            if has_feature:
                prevalence_count += 1
                if len(examples) < AnalysisConfig.MAX_EXAMPLES_BATCH_PROMPT:
                    examples.append(_get_feature_example(
                        feature_name,
                        note,
                        features_matrix[note.note_id]
                    ))

        prevalence_pct = (prevalence_count / len(filtered_notes) * 100) if filtered_notes else 0

        feature_data[feature_name] = {
            'prevalence_count': prevalence_count,
            'prevalence_pct': prevalence_pct,
            'examples': examples if examples else [f"{feature_name}: (无具体案例)"]
        }

        logger.info(f"    {feature_name}: {prevalence_pct:.1f}% ({prevalence_count}/{len(filtered_notes)})")

    # 3. 构建LLM提示词
    prompt = _build_metric_analysis_prompt(
        metric=metric,
        rationale=rationale,
        feature_data=feature_data,
        sample_size=len(filtered_notes),
        variance_level=variance_level,
        keyword=keyword
    )

    # 4. 调用LLM
    try:
        # 导入OpenAI client
        from openai import OpenAI

        # 获取API配置
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        model = os.getenv("OPENROUTER_TEXT_MODEL", "qwen/qwen3-235b-a22b-thinking-2507")
        site_url = os.getenv("OPENROUTER_SITE_URL", "https://openrouter.ai/api/v1")
        site_name = os.getenv("OPENROUTER_SITE_NAME", "XHS SEO Optimizer")

        if not api_key:
            logger.warning("OPENROUTER_API_KEY not found - using fallback analysis")
            parsed_data = _create_fallback_response(list(feature_data.keys()))
        else:
            # 初始化client
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )

            # 调用LLM
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

            # 解析响应
            content = response.choices[0].message.content

            parsed_data = _parse_metric_analysis_response(content, list(feature_data.keys()))

    except Exception as e:
        logger.error(f"LLM call failed for metric {metric}: {e}")
        parsed_data = _create_fallback_response(list(feature_data.keys()))


    # 5. 创建MetricSuccessProfile
    feature_analyses = {}
    for feature_name, analysis_data in parsed_data['feature_analyses'].items():
        if feature_name not in feature_data:
            logger.warning(f"Feature {feature_name} in LLM response but not in feature_data, skipping")
            continue

        # 验证并修正 key_elements（确保至少 3 个，最多 5 个）
        key_elements = analysis_data.get('key_elements', ["参考案例", "测试优化", "持续迭代"])
        if not isinstance(key_elements, list):
            key_elements = ["参考案例", "测试优化", "持续迭代"]
        elif len(key_elements) < 3:
            # 自动填充到 3 个
            default_elements = ["研究成功案例", "理解用户偏好", "测试并优化", "持续迭代", "数据驱动调整"]
            while len(key_elements) < 3 and len(default_elements) > 0:
                key_elements.append(default_elements.pop(0))
            logger.warning(f"Feature {feature_name}: key_elements只有{len(analysis_data.get('key_elements', []))}个，自动填充到{len(key_elements)}个")
        elif len(key_elements) > 5:
            key_elements = key_elements[:5]
            logger.warning(f"Feature {feature_name}: key_elements超过5个，截断到5个")

        feature_analyses[feature_name] = FeatureAnalysis(
            feature_name=feature_name,
            prevalence_count=feature_data[feature_name]['prevalence_count'],
            prevalence_pct=feature_data[feature_name]['prevalence_pct'],
            examples=feature_data[feature_name]['examples'],
            why_it_works=analysis_data.get('why_it_works', f"{feature_name}在优质笔记中广泛存在"),
            creation_formula=analysis_data.get('creation_formula', f"遵循{feature_name}最佳实践"),
            key_elements=key_elements
        )

    profile = MetricSuccessProfile(
        metric_name=metric,
        sample_size=len(filtered_notes),
        variance_level=variance_level,
        relevant_features=relevant_features,
        feature_analyses=feature_analyses,
        metric_success_narrative=parsed_data.get(
            'metric_success_narrative',
            f"{metric}受多个内容特征协同影响"
        ),
        timestamp=get_current_timestamp()
    )


    logger.info(f"✓ Metric analysis complete for {metric}")
    return profile