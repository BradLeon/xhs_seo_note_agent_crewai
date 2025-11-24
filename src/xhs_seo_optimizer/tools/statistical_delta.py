"""Statistical Delta Tool - 统计差距分析工具.

Determines statistical significance of gaps between owned_note and target_notes.
Calculates z-scores, p-values, and prioritizes gaps for optimization.
"""

import json
import logging
from typing import List, Optional, Dict, Any
import math

from scipy import stats
from crewai.tools import BaseTool
from pydantic import Field, BaseModel

from ..models.analysis_results import Gap, GapAnalysis
from ..attribution import get_relevant_features, get_attribution_rationale

logger = logging.getLogger(__name__)


class StatisticalDeltaInput(BaseModel):
    """Input schema for StatisticalDeltaTool.

    Smart mode: Tool automatically reads from shared_context.
    No parameters needed - just call statistical_delta_analysis().
    """
    pass  # No parameters required - data comes from shared_context


class StatisticalDeltaTool(BaseTool):
    """统计差距分析工具 (Statistical Delta Tool).

    Determines if gaps between owned_note and target_notes are statistically significant.
    Used by GapFinder to prioritize which deficiencies truly matter.

    Smart mode: Automatically reads audit_report and success_profile_report from shared_context.
    Just call: statistical_delta_analysis()
    """

    name: str = "statistical_delta_analysis"
    description: str = (
        "Analyzes statistical significance of performance gaps between owned note and target notes. "
        "Smart mode: No parameters needed - call statistical_delta_analysis(). "
        "Tool automatically reads audit_report and success_profile_report from shared_context. "
        "Output: GapAnalysis JSON with significant/marginal/non_significant gaps. "
        "使用场景：判断客户笔记与竞品笔记间的差距是否具有统计显著性，帮助优先级排序。"
    )
    args_schema: type[BaseModel] = StatisticalDeltaInput

    alpha: float = Field(
        default=0.05,
        description="显著性阈值 (Significance threshold, default 0.05 for 95% confidence)"
    )

    def _run(self) -> str:
        """Analyze statistical significance of gaps.

        Smart mode: Reads data from shared_context automatically.

        Returns:
            JSON string of GapAnalysis model

        Raises:
            ValueError: If required data not found in shared_context
        """
        # Smart mode: Read from shared_context
        from xhs_seo_optimizer.shared_context import shared_context

        audit_report = shared_context.get("audit_report")
        success_profile_report = shared_context.get("success_profile_report")

        if not audit_report:
            raise ValueError(
                "audit_report not found in shared_context. "
                "Ensure @before_kickoff has stored it correctly."
            )
        if not success_profile_report:
            raise ValueError(
                "success_profile_report not found in shared_context. "
                "Ensure @before_kickoff has stored it correctly."
            )

        # Extract data from reports
        current_metrics = audit_report.get('current_metrics', {})
        if not current_metrics:
            raise ValueError("audit_report must contain 'current_metrics'")

        aggregated_stats = success_profile_report.get('aggregated_stats', {})
        prediction_stats = aggregated_stats.get('prediction_stats', {})
        if not prediction_stats:
            raise ValueError("success_profile_report must contain aggregated_stats.prediction_stats")

        sample_size = aggregated_stats.get('sample_size', 0)

        logger.info(f"Smart mode: Analyzing gaps for {len(current_metrics)} metrics")

        try:
            # Calculate gaps for all metrics
            all_gaps = self._calculate_all_gaps(current_metrics, prediction_stats)

            # Separate gaps by significance level
            significant_gaps = []  # p < 0.05
            marginal_gaps = []     # 0.05 <= p < 0.10
            non_significant_gaps = []  # p >= 0.10 or undefined

            for g in all_gaps:
                if g.significance == "undefined":
                    non_significant_gaps.append(g)
                elif g.p_value < 0.05:  # significant, very_significant, critical
                    significant_gaps.append(g)
                elif g.p_value < 0.10:  # marginal
                    marginal_gaps.append(g)
                else:  # none
                    non_significant_gaps.append(g)

            # Calculate priority order
            priority_order = self._calculate_priority_order(all_gaps)

            # Create result
            result = GapAnalysis(
                significant_gaps=significant_gaps,
                marginal_gaps=marginal_gaps,
                non_significant_gaps=non_significant_gaps,
                priority_order=priority_order,
                sample_size=sample_size
            )

            return result.model_dump_json(indent=2)

        except Exception as e:
            logger.error(f"Statistical delta analysis failed: {str(e)}")
            raise RuntimeError(f"Statistical delta analysis failed: {str(e)}") from e

    def _calculate_all_gaps(
        self,
        current_metrics: Dict[str, float],
        prediction_stats: Dict[str, Dict[str, Any]]
    ) -> List[Gap]:
        """Calculate gaps for all available metrics.

        Args:
            current_metrics: Dict of metric_name -> owned_value
            prediction_stats: Dict of metric_name -> {mean, median, std, min, max, count}

        Returns:
            List of Gap objects
        """
        gaps = []

        for metric_name, stats_dict in prediction_stats.items():
            # Get owned value from current_metrics
            owned_value = current_metrics.get(metric_name)

            # Skip if metric not in current_metrics
            if owned_value is None:
                logger.warning(f"Metric {metric_name} not found in current_metrics")
                continue

            owned_value = float(owned_value)
            target_mean = stats_dict.get('mean', 0.0)
            target_std = stats_dict.get('std', 0.0)

            # Calculate deltas
            delta_absolute = owned_value - target_mean

            # Calculate percentage delta (handle division by zero)
            if target_mean != 0:
                delta_pct = (delta_absolute / target_mean) * 100
            elif owned_value == 0:
                delta_pct = 0.0  # Both zero
            else:
                delta_pct = float('inf') if owned_value > 0 else float('-inf')

            # Calculate z-score and p-value
            if target_std > 0:
                z_score = delta_absolute / target_std
                # Two-tailed test
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            else:
                # Cannot calculate z-score with zero std
                z_score = float('inf') if delta_absolute > 0 else (float('-inf') if delta_absolute < 0 else 0.0)
                p_value = 0.0 if delta_absolute != 0 else 1.0

            # Classify significance
            significance = self._classify_significance(p_value, target_std)

            # Generate interpretation
            interpretation = self._generate_interpretation(
                metric_name, owned_value, target_mean, delta_pct, z_score, significance, target_std
            )

            # Get attribution info from attribution.py
            # Skip metrics without attribution rules (e.g., 'impression')
            try:
                related_features = get_relevant_features(metric_name)
                rationale = get_attribution_rationale(metric_name)
            except KeyError:
                logger.info(f"Skipping metric '{metric_name}': no attribution rules available")
                continue

            # Create Gap object
            gap = Gap(
                metric=metric_name,
                owned_value=owned_value,
                target_mean=target_mean,
                target_std=target_std,
                delta_absolute=delta_absolute,
                delta_pct=delta_pct,
                z_score=z_score,
                p_value=p_value,
                significance=significance,
                interpretation=interpretation,
                related_features=related_features,
                rationale=rationale
            )

            gaps.append(gap)

        return gaps

    def _classify_significance(self, p_value: float, target_std: float) -> str:
        """Classify significance level based on p-value.

        Args:
            p_value: Statistical p-value
            target_std: Target standard deviation (to detect undefined cases)

        Returns:
            Significance level string
        """
        # Handle undefined case (zero variance)
        if target_std == 0:
            return "undefined"

        # Standard significance levels
        if p_value < 0.001:
            return "critical"  # >99.9% confidence
        elif p_value < 0.01:
            return "very_significant"  # >99% confidence
        elif p_value < 0.05:
            return "significant"  # >95% confidence
        elif p_value < 0.10:
            return "marginal"  # >90% confidence
        else:
            return "none"  # Not significant

    def _generate_interpretation(
        self,
        metric_name: str,
        owned_value: float,
        target_mean: float,
        delta_pct: float,
        z_score: float,
        significance: str,
        target_std: float
    ) -> str:
        """Generate human-readable interpretation of gap.

        Args:
            metric_name: Name of the metric
            owned_value: Owned note value
            target_mean: Target notes mean
            delta_pct: Percentage difference
            z_score: Z-score
            significance: Significance classification
            target_std: Target standard deviation

        Returns:
            Human-readable interpretation string
        """
        # Handle undefined case
        if target_std == 0:
            if owned_value == target_mean:
                return f"{metric_name}: owned value equals all target values ({owned_value:.4f}), zero variance"
            else:
                return f"{metric_name}: owned value ({owned_value:.4f}) differs from targets ({target_mean:.4f}), but targets have zero variance"

        # Determine direction
        if delta_pct > 0:
            direction = "higher"
        elif delta_pct < 0:
            direction = "lower"
        else:
            return f"{metric_name}: owned value matches target mean ({owned_value:.4f})"

        # Format based on significance level
        if significance == "critical":
            return (
                f"{metric_name}: owned_note is critically {direction} than target_notes "
                f"({abs(delta_pct):.1f}% difference, z={z_score:.2f}σ, p={0.001:.3f})"
            )
        elif significance == "very_significant":
            return (
                f"{metric_name}: owned_note is very significantly {direction} than target_notes "
                f"({abs(delta_pct):.1f}% difference, z={z_score:.2f}σ, p<0.01)"
            )
        elif significance == "significant":
            return (
                f"{metric_name}: owned_note is significantly {direction} than target_notes "
                f"({abs(delta_pct):.1f}% difference, z={z_score:.2f}σ, p<0.05)"
            )
        elif significance == "marginal":
            return (
                f"{metric_name}: owned_note is marginally {direction} than target_notes "
                f"({abs(delta_pct):.1f}% difference, z={z_score:.2f}σ, p<0.10)"
            )
        else:
            return (
                f"{metric_name}: owned_note is within normal range "
                f"({abs(delta_pct):.1f}% difference, z={z_score:.2f}σ, not significant)"
            )

    def _calculate_priority_order(self, gaps: List[Gap]) -> List[str]:
        """Calculate priority order combining statistical significance and magnitude.

        Args:
            gaps: List of Gap objects

        Returns:
            List of metric names sorted by priority (highest first)
        """
        # Calculate priority score for each gap
        gap_priorities = []

        for gap in gaps:
            # Skip undefined gaps
            if gap.significance == "undefined":
                continue

            # Priority = |z_score| * |delta_pct| / 100
            # This combines statistical significance with practical importance
            if not math.isinf(gap.delta_pct):
                priority_score = abs(gap.z_score) * abs(gap.delta_pct) / 100
            else:
                # Handle infinite delta_pct (target_mean = 0)
                priority_score = abs(gap.z_score) * 100  # Use large value

            gap_priorities.append((gap.metric, priority_score))

        # Sort by priority score (descending)
        gap_priorities.sort(key=lambda x: x[1], reverse=True)

        # Return metric names in priority order
        return [metric for metric, _ in gap_priorities]
