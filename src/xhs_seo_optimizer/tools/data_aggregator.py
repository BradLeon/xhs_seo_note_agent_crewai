"""Data Aggregator Tool - 数据聚合工具.

Calculates statistical aggregates across multiple notes for competitor analysis.
Computes mean, median, std, min, max for prediction metrics and tag frequencies.
"""

import json
import logging
from typing import List, Dict, Any
from collections import Counter

import numpy as np
from crewai.tools import BaseTool
from pydantic import Field

from ..models.note import Note
from ..models.analysis_results import MetricStats, AggregatedMetrics

logger = logging.getLogger(__name__)


class DataAggregatorTool(BaseTool):
    """数据聚合工具 (Data Aggregator Tool).

    Calculates statistical summaries across multiple notes.
    Used by CompetitorAnalyst to quantify "winning patterns" in target_notes.
    """

    name: str = "Data Aggregator"
    description: str = (
        "Aggregates statistical metrics across multiple Xiaohongshu notes. "
        "Input: List of Note objects. "
        "Output: AggregatedMetrics with mean, median, std, min, max for predictions and tag frequencies. "
        "使用场景：计算多个笔记的统计摘要，识别高表现笔记的共同模式。"
    )

    remove_outliers: bool = Field(
        default=False,
        description="是否移除异常值 (Whether to remove outliers using z-score threshold)"
    )
    outlier_threshold: float = Field(
        default=2.0,
        description="异常值阈值 (Z-score threshold for outlier removal, default ±2σ)"
    )

    def _run(self, notes: List[Note]) -> str:
        """Aggregate statistics across multiple notes.

        Args:
            notes: List of Note objects to analyze (typically 3-10 target_notes)

        Returns:
            JSON string of AggregatedMetrics model

        Raises:
            ValueError: If notes list is empty or invalid
        """
        # Validate input
        if not notes or len(notes) == 0:
            raise ValueError("notes list must contain at least one note")

        try:
            # Aggregate prediction metrics
            prediction_stats = self._aggregate_prediction_metrics(notes)

            # Aggregate tag frequencies and modes
            tag_frequencies, tag_modes = self._aggregate_tags(notes)

            # Create result model
            result = AggregatedMetrics(
                prediction_stats=prediction_stats,
                tag_frequencies=tag_frequencies,
                tag_modes=tag_modes,
                sample_size=len(notes),
                outliers_removed=0  # Will be updated if outlier removal is implemented
            )

            return result.model_dump_json(indent=2)

        except Exception as e:
            logger.error(f"Data aggregation failed: {str(e)}")
            raise RuntimeError(f"Data aggregation failed: {str(e)}") from e

    def _aggregate_prediction_metrics(self, notes: List[Note]) -> Dict[str, MetricStats]:
        """Aggregate prediction metrics across notes.

        Args:
            notes: List of Note objects

        Returns:
            Dict mapping metric name to MetricStats
        """
        # All prediction metrics to aggregate
        metric_names = [
            "ctr", "ces_rate", "interaction_rate", "like_rate", "fav_rate",
            "comment_rate", "share_rate", "follow_rate", "sort_score2", "impression"
        ]

        prediction_stats = {}

        for metric_name in metric_names:
            # Extract values for this metric
            values = []
            for note in notes:
                try:
                    # Handle both attribute name and alias
                    if metric_name == "sort_score2":
                        value = getattr(note.prediction, "sort_score2", None)
                    else:
                        value = getattr(note.prediction, metric_name, None)

                    if value is not None:
                        values.append(float(value))
                except (AttributeError, TypeError) as e:
                    logger.warning(f"Missing or invalid {metric_name} in note {note.note_id}: {e}")
                    continue

            # Skip if no valid values
            if not values:
                logger.warning(f"No valid values found for metric: {metric_name}")
                continue

            # Convert to numpy array for calculations
            values_array = np.array(values)

            # Remove outliers if enabled
            if self.remove_outliers and len(values) > 2:
                values_array, removed_count = self._remove_outliers(values_array)
                if removed_count > 0:
                    logger.info(f"Removed {removed_count} outliers from {metric_name}")

            # Calculate statistics
            if len(values_array) > 0:
                prediction_stats[metric_name] = MetricStats(
                    mean=float(np.mean(values_array)),
                    median=float(np.median(values_array)),
                    std=float(np.std(values_array, ddof=1)) if len(values_array) > 1 else 0.0,
                    min=float(np.min(values_array)),
                    max=float(np.max(values_array)),
                    count=len(values_array)
                )

        return prediction_stats

    def _aggregate_tags(self, notes: List[Note]) -> tuple[Dict[str, Dict[str, int]], Dict[str, str]]:
        """Aggregate tag frequencies and calculate modes.

        Args:
            notes: List of Note objects

        Returns:
            Tuple of (tag_frequencies, tag_modes)
        """
        # Tag fields to aggregate
        tag_fields = [
            "intention_lv1", "intention_lv2",
            "taxonomy1", "taxonomy2", "taxonomy3",
            "note_marketing_integrated_level"
        ]

        tag_frequencies = {}
        tag_modes = {}

        for field_name in tag_fields:
            # Collect all values for this tag field
            values = []
            for note in notes:
                try:
                    value = getattr(note.tag, field_name, None)
                    if value is not None and value != "":
                        values.append(str(value))
                except AttributeError:
                    continue

            if not values:
                continue

            # Count frequencies
            counter = Counter(values)
            tag_frequencies[field_name] = dict(counter)

            # Calculate mode (most common value)
            # If tie, use alphabetically first for determinism
            most_common = counter.most_common()
            if most_common:
                max_count = most_common[0][1]
                # Get all values with max count
                modes = [item[0] for item in most_common if item[1] == max_count]
                # Sort alphabetically and take first
                tag_modes[field_name] = sorted(modes)[0]

        return tag_frequencies, tag_modes

    def _remove_outliers(self, values: np.ndarray) -> tuple[np.ndarray, int]:
        """Remove outliers using z-score method.

        Args:
            values: Numpy array of values

        Returns:
            Tuple of (filtered values, number of outliers removed)
        """
        if len(values) < 3:
            return values, 0

        # Calculate z-scores
        mean = np.mean(values)
        std = np.std(values, ddof=1)

        if std == 0:
            # All values identical, no outliers
            return values, 0

        z_scores = np.abs((values - mean) / std)

        # Filter values within threshold
        mask = z_scores <= self.outlier_threshold
        filtered_values = values[mask]

        outliers_removed = len(values) - len(filtered_values)

        # Ensure we don't remove all data
        if len(filtered_values) == 0:
            logger.warning("All values would be removed as outliers, keeping original data")
            return values, 0

        return filtered_values, outliers_removed
