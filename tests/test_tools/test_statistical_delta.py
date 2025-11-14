"""Tests for StatisticalDeltaTool - 统计差距分析工具测试.

Uses REAL DATA from docs/owned_note.json and docs/target_notes.json.
"""

import json
import pytest
from pathlib import Path

from xhs_seo_optimizer.tools.data_aggregator import DataAggregatorTool
from xhs_seo_optimizer.tools.statistical_delta import StatisticalDeltaTool
from xhs_seo_optimizer.models.note import Note
from xhs_seo_optimizer.models.analysis_results import AggregatedMetrics, GapAnalysis


class TestStatisticalDeltaTool:
    """Test suite for StatisticalDeltaTool using real data."""

    @pytest.fixture
    def tool(self):
        """Create StatisticalDeltaTool instance with default alpha=0.05."""
        return StatisticalDeltaTool()

    @pytest.fixture
    def tool_with_custom_alpha(self):
        """Create StatisticalDeltaTool with custom alpha=0.10."""
        return StatisticalDeltaTool(alpha=0.10)

    @pytest.fixture
    def real_owned_note(self):
        """Load real owned note from docs/owned_note.json."""
        docs_dir = Path(__file__).parent.parent.parent / "docs"
        owned_note_path = docs_dir / "owned_note.json"

        if not owned_note_path.exists():
            pytest.skip("owned_note.json not found in docs/")

        with open(owned_note_path, "r", encoding="utf-8") as f:
            note_data = json.load(f)

        return Note.from_json(note_data)

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
    def real_aggregated_stats(self, real_target_notes):
        """Aggregate real target notes using DataAggregatorTool."""
        aggregator = DataAggregatorTool()
        result_json = aggregator._run(notes=real_target_notes)
        return AggregatedMetrics(**json.loads(result_json))

    # === Test with Real Data ===

    def test_analyze_real_owned_note_gaps(self, tool, real_owned_note, real_aggregated_stats):
        """Test: Analyze gaps between real owned note and target notes."""
        result_json = tool._run(owned_note=real_owned_note, target_stats=real_aggregated_stats)
        result = json.loads(result_json)

        print(f"\n=== Real Data Gap Analysis ===")
        print(f"Sample size: {result['sample_size']} target notes")
        print(f"Significant gaps found: {len(result['significant_gaps'])}")
        print(f"Non-significant gaps: {len(result['non_significant_gaps'])}")

        # Should have some gaps identified
        total_gaps = len(result["significant_gaps"]) + len(result["non_significant_gaps"])
        assert total_gaps > 0

        # Print top significant gaps
        if result["significant_gaps"]:
            print(f"\n=== Top 5 Significant Gaps ===")
            for i, gap in enumerate(result["significant_gaps"][:5], 1):
                print(f"\n{i}. {gap['metric']}:")
                print(f"   Owned: {gap['owned_value']:.6f}")
                print(f"   Target mean: {gap['target_mean']:.6f}")
                print(f"   Delta: {gap['delta_pct']:.1f}%")
                print(f"   Z-score: {gap['z_score']:.2f}")
                print(f"   P-value: {gap['p_value']:.4f}")
                print(f"   Significance: {gap['significance']}")
                print(f"   {gap['interpretation']}")

        # Print priority order
        if result["priority_order"]:
            print(f"\n=== Priority Order (Top 5) ===")
            for i, metric in enumerate(result["priority_order"][:5], 1):
                print(f"{i}. {metric}")

    def test_significance_classification(self, tool, real_owned_note, real_aggregated_stats):
        """Test: Significance levels are classified correctly."""
        result = json.loads(tool._run(owned_note=real_owned_note, target_stats=real_aggregated_stats))

        # Check that gaps have valid significance levels
        valid_levels = {"critical", "very_significant", "significant", "marginal", "none", "undefined"}

        for gap in result["significant_gaps"] + result["non_significant_gaps"]:
            assert gap["significance"] in valid_levels, f"Invalid significance: {gap['significance']}"

        # Significant gaps should have p < alpha
        for gap in result["significant_gaps"]:
            if gap["significance"] != "undefined":
                assert gap["p_value"] < tool.alpha, f"{gap['metric']} has p={gap['p_value']} but is in significant_gaps"

        print(f"\n=== Significance Distribution ===")
        sig_counts = {}
        for gap in result["significant_gaps"] + result["non_significant_gaps"]:
            sig_level = gap["significance"]
            sig_counts[sig_level] = sig_counts.get(sig_level, 0) + 1

        for level, count in sorted(sig_counts.items()):
            print(f"{level}: {count} gaps")

    def test_priority_ordering(self, tool, real_owned_note, real_aggregated_stats):
        """Test: Priority order reflects combined significance and magnitude."""
        result = json.loads(tool._run(owned_note=real_owned_note, target_stats=real_aggregated_stats))

        # Should have priority order
        assert len(result["priority_order"]) > 0

        # Priority order should contain metric names
        all_gaps = result["significant_gaps"] + result["non_significant_gaps"]
        all_metrics = {gap["metric"] for gap in all_gaps if gap["significance"] != "undefined"}

        for metric in result["priority_order"]:
            assert metric in all_metrics, f"{metric} in priority_order but not in gaps"

        print(f"\n=== Top Priority Metrics ===")
        for i, metric in enumerate(result["priority_order"][:10], 1):
            # Find the gap for this metric
            gap = next((g for g in all_gaps if g["metric"] == metric), None)
            if gap:
                priority_score = abs(gap["z_score"]) * abs(gap["delta_pct"]) / 100
                print(f"{i}. {metric}: z={gap['z_score']:.2f}, delta={gap['delta_pct']:.1f}%, priority={priority_score:.4f}")

    def test_interpretation_quality(self, tool, real_owned_note, real_aggregated_stats):
        """Test: Interpretations are clear and informative."""
        result = json.loads(tool._run(owned_note=real_owned_note, target_stats=real_aggregated_stats))

        all_gaps = result["significant_gaps"] + result["non_significant_gaps"]

        for gap in all_gaps:
            interpretation = gap["interpretation"]

            # Should mention the metric name
            assert gap["metric"] in interpretation

            # Should indicate direction (lower/higher) or status
            direction_words = ["lower", "higher", "within normal range", "equals", "differs", "zero variance"]
            assert any(word in interpretation for word in direction_words), \
                f"Interpretation missing direction: {interpretation}"

            # Should be non-empty
            assert len(interpretation) > 20, "Interpretation too short"

        print(f"\n=== Sample Interpretations ===")
        for gap in all_gaps[:3]:
            print(f"\n{gap['metric']}:")
            print(f"  {gap['interpretation']}")

    def test_output_schema_validation(self, tool, real_owned_note, real_aggregated_stats):
        """Test: Output conforms to GapAnalysis schema."""
        result_json = tool._run(owned_note=real_owned_note, target_stats=real_aggregated_stats)
        result_dict = json.loads(result_json)

        # MUST validate against Pydantic model
        gap_analysis = GapAnalysis(**result_dict)

        # Verify required fields
        assert isinstance(gap_analysis.significant_gaps, list)
        assert isinstance(gap_analysis.non_significant_gaps, list)
        assert isinstance(gap_analysis.priority_order, list)
        assert isinstance(gap_analysis.sample_size, int)

        # Each gap MUST have required fields
        for gap in gap_analysis.significant_gaps + gap_analysis.non_significant_gaps:
            assert hasattr(gap, "metric")
            assert hasattr(gap, "owned_value")
            assert hasattr(gap, "target_mean")
            assert hasattr(gap, "target_std")
            assert hasattr(gap, "delta_absolute")
            assert hasattr(gap, "delta_pct")
            assert hasattr(gap, "z_score")
            assert hasattr(gap, "p_value")
            assert hasattr(gap, "significance")
            assert hasattr(gap, "interpretation")

        print("\n✅ Output schema validation passed!")

    def test_custom_alpha_threshold(self, tool_with_custom_alpha, real_owned_note, real_aggregated_stats):
        """Test: Custom alpha threshold is applied correctly."""
        # Run with default alpha=0.05
        tool_default = StatisticalDeltaTool(alpha=0.05)
        result_default = json.loads(tool_default._run(owned_note=real_owned_note, target_stats=real_aggregated_stats))

        # Run with custom alpha=0.10
        result_custom = json.loads(tool_with_custom_alpha._run(owned_note=real_owned_note, target_stats=real_aggregated_stats))

        print(f"\n=== Alpha Threshold Comparison ===")
        print(f"Alpha=0.05: {len(result_default['significant_gaps'])} significant gaps")
        print(f"Alpha=0.10: {len(result_custom['significant_gaps'])} significant gaps")

        # With alpha=0.10, we should have same or more significant gaps
        assert len(result_custom["significant_gaps"]) >= len(result_default["significant_gaps"])

        # Verify alpha is used correctly
        for gap in result_custom["significant_gaps"]:
            if gap["significance"] != "undefined":
                assert gap["p_value"] < 0.10

    def test_statistical_correctness(self, tool, real_owned_note, real_aggregated_stats):
        """Test: Z-scores and p-values are mathematically correct."""
        result = json.loads(tool._run(owned_note=real_owned_note, target_stats=real_aggregated_stats))

        import numpy as np
        from scipy import stats as scipy_stats

        # Verify a few gaps manually
        for gap in (result["significant_gaps"] + result["non_significant_gaps"])[:3]:
            metric = gap["metric"]
            owned_value = gap["owned_value"]
            target_mean = gap["target_mean"]
            target_std = gap["target_std"]

            if target_std > 0:
                # Calculate expected z-score
                expected_z = (owned_value - target_mean) / target_std

                # Calculate expected p-value (two-tailed)
                expected_p = 2 * (1 - scipy_stats.norm.cdf(abs(expected_z)))

                # Verify
                assert abs(gap["z_score"] - expected_z) < 0.01, \
                    f"{metric}: z-score mismatch (expected {expected_z:.4f}, got {gap['z_score']:.4f})"
                assert abs(gap["p_value"] - expected_p) < 0.01, \
                    f"{metric}: p-value mismatch (expected {expected_p:.4f}, got {gap['p_value']:.4f})"

        print("\n✅ Statistical calculations verified!")

    def test_identifies_critical_gaps(self, tool, real_owned_note, real_aggregated_stats):
        """Test: Tool can identify and classify gaps appropriately."""
        result = json.loads(tool._run(owned_note=real_owned_note, target_stats=real_aggregated_stats))

        # Check if we found any critical gaps
        critical_gaps = [g for g in result["significant_gaps"] if g["significance"] == "critical"]
        very_sig_gaps = [g for g in result["significant_gaps"] if g["significance"] == "very_significant"]
        sig_gaps = [g for g in result["significant_gaps"] if g["significance"] == "significant"]

        print(f"\n=== Gap Classification Results ===")
        print(f"Critical gaps (p<0.001): {len(critical_gaps)}")
        print(f"Very significant gaps (p<0.01): {len(very_sig_gaps)}")
        print(f"Significant gaps (p<0.05): {len(sig_gaps)}")
        print(f"Total significant gaps: {len(result['significant_gaps'])}")
        print(f"Non-significant gaps: {len(result['non_significant_gaps'])}")

        if critical_gaps:
            print(f"\n=== Critical Gaps Details ===")
            for gap in critical_gaps:
                print(f"{gap['metric']}: {gap['delta_pct']:.1f}% (z={gap['z_score']:.2f}, p={gap['p_value']:.6f})")
        else:
            print(f"\n✅ No critical gaps found - owned_note performs close to target level!")

        # Tool should analyze all metrics (significant or not)
        total_gaps = len(result["significant_gaps"]) + len(result["non_significant_gaps"])
        assert total_gaps > 0, "No gaps analyzed"

        # If no significant gaps, that means owned_note is performing well
        if len(result["significant_gaps"]) == 0:
            print("✅ Good news: No significant performance gaps detected!")
            print("   Owned note is statistically comparable to high-performing targets.")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
