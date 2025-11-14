"""Tests for DataAggregatorTool - 数据聚合工具测试.

Uses REAL DATA from docs/target_notes.json to test the tool.
"""

import json
import pytest
from pathlib import Path

from xhs_seo_optimizer.tools.data_aggregator import DataAggregatorTool
from xhs_seo_optimizer.models.note import Note
from xhs_seo_optimizer.models.analysis_results import AggregatedMetrics


class TestDataAggregatorTool:
    """Test suite for DataAggregatorTool using real data."""

    @pytest.fixture
    def tool(self):
        """Create DataAggregatorTool instance."""
        return DataAggregatorTool()

    @pytest.fixture
    def tool_with_outlier_removal(self):
        """Create DataAggregatorTool with outlier removal enabled."""
        return DataAggregatorTool(remove_outliers=True, outlier_threshold=2.0)

    @pytest.fixture
    def real_target_notes(self):
        """Load real target notes from docs/target_notes.json."""
        docs_dir = Path(__file__).parent.parent.parent / "docs"
        target_notes_path = docs_dir / "target_notes.json"

        if not target_notes_path.exists():
            pytest.skip("target_notes.json not found in docs/")

        with open(target_notes_path, "r", encoding="utf-8") as f:
            notes_data = json.load(f)

        return [Note.from_json(note_data) for note_data in notes_data]

    @pytest.fixture
    def subset_target_notes(self, real_target_notes):
        """Get first 3 target notes for specific tests."""
        return real_target_notes[:3]

    # === Test with Real Data ===

    def test_aggregate_real_target_notes(self, tool, real_target_notes):
        """Test: Aggregate all real target notes from JSON file."""
        result_json = tool._run(notes=real_target_notes)
        result = json.loads(result_json)

        print(f"\n=== Aggregated {result['sample_size']} Real Target Notes ===")

        # Should have prediction stats
        assert len(result["prediction_stats"]) > 0
        print(f"Metrics analyzed: {list(result['prediction_stats'].keys())}")

        # Check specific metrics exist
        assert "ctr" in result["prediction_stats"]
        assert "comment_rate" in result["prediction_stats"]
        assert "sort_score2" in result["prediction_stats"]

        # Print key statistics
        ctr_stats = result["prediction_stats"]["ctr"]
        print(f"\nCTR Statistics:")
        print(f"  Mean: {ctr_stats['mean']:.4f}")
        print(f"  Median: {ctr_stats['median']:.4f}")
        print(f"  Std: {ctr_stats['std']:.4f}")
        print(f"  Range: [{ctr_stats['min']:.4f}, {ctr_stats['max']:.4f}]")

        comment_stats = result["prediction_stats"]["comment_rate"]
        print(f"\nComment Rate Statistics:")
        print(f"  Mean: {comment_stats['mean']:.6f}")
        print(f"  Median: {comment_stats['median']:.6f}")
        print(f"  Std: {comment_stats['std']:.6f}")

        # Should have tag frequencies
        assert len(result["tag_frequencies"]) > 0
        print(f"\nTag dimensions analyzed: {list(result['tag_frequencies'].keys())}")

        # Print tag distribution
        if "intention_lv2" in result["tag_frequencies"]:
            print(f"\nIntention LV2 distribution:")
            for tag, count in result["tag_frequencies"]["intention_lv2"].items():
                print(f"  {tag}: {count}")

        # Should have tag modes
        assert len(result["tag_modes"]) > 0
        print(f"\nMost common tags:")
        for field, mode in result["tag_modes"].items():
            print(f"  {field}: {mode}")

        # Should have correct sample size
        assert result["sample_size"] == len(real_target_notes)

    def test_aggregate_subset_of_notes(self, tool, subset_target_notes):
        """Test: Aggregate first 3 target notes."""
        result_json = tool._run(notes=subset_target_notes)
        result = json.loads(result_json)

        assert result["sample_size"] == 3

        # Verify CTR statistics
        ctr_stats = result["prediction_stats"]["ctr"]
        assert ctr_stats["count"] == 3
        assert ctr_stats["min"] <= ctr_stats["mean"] <= ctr_stats["max"]
        assert ctr_stats["std"] >= 0

        print(f"\n=== Subset Analysis (3 notes) ===")
        print(f"CTR mean: {ctr_stats['mean']:.4f}")
        print(f"CTR std: {ctr_stats['std']:.4f}")

    def test_all_prediction_metrics_present(self, tool, subset_target_notes):
        """Test: All prediction metrics should be aggregated."""
        result = json.loads(tool._run(notes=subset_target_notes))

        # Expected metrics based on NotePrediction model
        expected_metrics = [
            "ctr", "ces_rate", "interaction_rate", "like_rate", "fav_rate",
            "comment_rate", "share_rate", "follow_rate", "sort_score2"
        ]

        for metric in expected_metrics:
            assert metric in result["prediction_stats"], f"Missing metric: {metric}"
            stats = result["prediction_stats"][metric]
            assert "mean" in stats
            assert "median" in stats
            assert "std" in stats
            assert "min" in stats
            assert "max" in stats
            assert "count" in stats

    def test_tag_frequencies_and_modes(self, tool, real_target_notes):
        """Test: Tag frequencies and modes are calculated correctly."""
        result = json.loads(tool._run(notes=real_target_notes))

        # Should have tag frequencies
        assert "intention_lv1" in result["tag_frequencies"]
        assert "intention_lv2" in result["tag_frequencies"]
        assert "taxonomy1" in result["tag_frequencies"]
        assert "taxonomy2" in result["tag_frequencies"]

        # Should have tag modes
        assert "intention_lv1" in result["tag_modes"]
        assert "intention_lv2" in result["tag_modes"]

        # Mode should be one of the frequent values
        for field, mode in result["tag_modes"].items():
            assert mode in result["tag_frequencies"][field]

        print(f"\n=== Tag Analysis ===")
        for field in ["intention_lv1", "intention_lv2", "taxonomy1", "taxonomy2"]:
            if field in result["tag_modes"]:
                print(f"{field}: {result['tag_modes'][field]}")

    def test_outlier_removal_on_real_data(self, tool_with_outlier_removal, real_target_notes):
        """Test: Outlier removal works on real data."""
        result = json.loads(tool_with_outlier_removal._run(notes=real_target_notes))

        # Check if any outliers were removed
        print(f"\n=== Outlier Removal Test ===")
        print(f"Outliers removed: {result.get('outliers_removed', 0)}")

        # Stats should still be valid
        for metric_name, stats in result["prediction_stats"].items():
            assert stats["count"] > 0
            assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_output_schema_validation(self, tool, subset_target_notes):
        """Test: Output conforms to AggregatedMetrics schema."""
        result_json = tool._run(notes=subset_target_notes)
        result_dict = json.loads(result_json)

        # MUST validate against Pydantic model
        aggregated = AggregatedMetrics(**result_dict)

        # Verify required fields exist
        assert isinstance(aggregated.prediction_stats, dict)
        assert isinstance(aggregated.tag_frequencies, dict)
        assert isinstance(aggregated.tag_modes, dict)
        assert isinstance(aggregated.sample_size, int)
        assert isinstance(aggregated.outliers_removed, int)

        # Each MetricStats should be valid
        for metric_name, stats in aggregated.prediction_stats.items():
            assert hasattr(stats, 'mean')
            assert hasattr(stats, 'median')
            assert hasattr(stats, 'std')
            assert hasattr(stats, 'min')
            assert hasattr(stats, 'max')
            assert hasattr(stats, 'count')

    def test_empty_notes_list_raises_error(self, tool):
        """Test: Empty notes list raises ValueError."""
        with pytest.raises(ValueError, match="at least one note"):
            tool._run(notes=[])

    def test_statistical_correctness(self, tool, subset_target_notes):
        """Test: Statistical calculations are mathematically correct."""
        result = json.loads(tool._run(notes=subset_target_notes))

        # Manually extract CTR values to verify
        ctr_values = [note.prediction.ctr for note in subset_target_notes]

        import numpy as np
        expected_mean = np.mean(ctr_values)
        expected_median = np.median(ctr_values)
        expected_std = np.std(ctr_values, ddof=1) if len(ctr_values) > 1 else 0.0
        expected_min = np.min(ctr_values)
        expected_max = np.max(ctr_values)

        ctr_stats = result["prediction_stats"]["ctr"]

        assert abs(ctr_stats["mean"] - expected_mean) < 0.0001
        assert abs(ctr_stats["median"] - expected_median) < 0.0001
        assert abs(ctr_stats["std"] - expected_std) < 0.0001
        assert ctr_stats["min"] == expected_min
        assert ctr_stats["max"] == expected_max

        print(f"\n=== Statistical Correctness Verification ===")
        print(f"Expected mean: {expected_mean:.6f}, Actual: {ctr_stats['mean']:.6f}")
        print(f"Expected std: {expected_std:.6f}, Actual: {ctr_stats['std']:.6f}")
        print("✅ Statistical calculations are correct!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
