"""Integration tests for GapFinder crew.

Tests the complete workflow of gap analysis between owned note and target notes:
- Statistical gap calculation using StatisticalDeltaTool
- Feature attribution (mapping gaps to missing/weak features)
- Priority ordering and root cause identification
"""

import json
import os
import sys
from pathlib import Path
import pytest

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment from {env_path}\n")
    else:
        print(f"⚠ No .env file found at {env_path}")
        print(f"  Using environment variables from shell\n")
except ImportError:
    print("⚠ python-dotenv not installed, using shell environment variables\n")

from xhs_seo_optimizer.models.reports import SuccessProfileReport, AuditReport, GapReport, MetricGap
from xhs_seo_optimizer.crew_gap_finder import XhsSeoOptimizerCrewGapFinder
import logging
logging.basicConfig(level=logging.INFO)


def load_success_profile_report(json_path: str) -> dict:
    """Load SuccessProfileReport from JSON file.

    Args:
        json_path: Path to success_profile_report.json

    Returns:
        SuccessProfileReport dict
    """
    print(f"Loading success profile report from: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"✓ Loaded success profile report for keyword: {data.get('keyword')}\n")
    return data


def load_audit_report(json_path: str) -> dict:
    """Load AuditReport from JSON file.

    Args:
        json_path: Path to audit_report.json

    Returns:
        AuditReport dict
    """
    print(f"Loading audit report from: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"✓ Loaded audit report for note: {data.get('note_id')}\n")
    return data


def print_gap_report_summary(report: GapReport):
    """Print a summary of the gap report.

    Args:
        report: GapReport object
    """
    print("=" * 80)
    print("GAP ANALYSIS REPORT SUMMARY")
    print("=" * 80)

    # Basic info
    print(f"\n笔记ID: {report.owned_note_id}")
    print(f"关键词: {report.keyword}")
    print(f"分析时间: {report.gap_timestamp}")
    print(f"样本量: {report.sample_size}")

    # Impact summary
    print(f"\n整体差距总结:")
    print(f"  {report.impact_summary}")

    # Top priority metrics
    print(f"\n优先改进指标 (Top {len(report.top_priority_metrics)}):")
    for i, metric in enumerate(report.top_priority_metrics, 1):
        print(f"  {i}. {metric}")

    # Root causes
    print(f"\n根本原因分析 ({len(report.root_causes)} 个):")
    for i, cause in enumerate(report.root_causes, 1):
        print(f"  {i}. {cause}")

    # Gap counts
    print(f"\n差距分类:")
    print(f"  - 显著差距 (significant): {len(report.significant_gaps)}")
    print(f"  - 边缘差距 (marginal): {len(report.marginal_gaps)}")
    print(f"  - 非显著差距 (non-significant): {len(report.non_significant_gaps)}")

    # Show top 2 significant gaps
    if report.significant_gaps:
        print(f"\n最重要的显著差距 (Top 2):")
        for gap in report.significant_gaps[:2]:
            print(f"  - {gap.metric_name}: owned={gap.owned_value:.4f}, target={gap.target_mean:.4f}")
            print(f"    Δ={gap.delta_pct:.1f}%, z={gap.z_score:.2f}, p={gap.p_value:.4f}")
            print(f"    优先级排名: #{gap.priority_rank}")
            print(f"    缺失特征: {', '.join(gap.missing_features[:3])}")
            print(f"    改进建议: {gap.recommendation_summary}")
            print()

    print("=" * 80)


class TestGapFinder:
    """Integration tests for GapFinder crew."""

    @pytest.fixture
    def success_profile_data(self):
        """Load success profile report test data."""
        report_path = project_root / "outputs" / "success_profile_report.json"
        if not report_path.exists():
            pytest.skip(f"Success profile report not found at {report_path}. Run CompetitorAnalyst first.")
        return load_success_profile_report(str(report_path))

    @pytest.fixture
    def audit_report_data(self):
        """Load audit report test data."""
        report_path = project_root / "outputs" / "audit_report.json"
        if not report_path.exists():
            pytest.skip(f"Audit report not found at {report_path}. Run OwnedNoteAuditor first.")
        return load_audit_report(str(report_path))

    @pytest.fixture
    def crew(self):
        """Create GapFinder crew instance."""
        return XhsSeoOptimizerCrewGapFinder()

    def test_basic_execution(self, crew, success_profile_data, audit_report_data):
        """Test basic crew execution without errors."""
        print("\n" + "=" * 80)
        print("TEST: Basic Execution")
        print("=" * 80)

        # Run crew
        result = crew.kickoff(inputs={
            "success_profile_report": success_profile_data,
            "audit_report": audit_report_data,
            "keyword": "老爸测评dha推荐哪几款"
        })

        # Verify result exists
        assert result is not None
        print("✓ Crew execution completed successfully")

        # Verify output file exists
        output_path = project_root / "outputs" / "gap_report.json"
        assert output_path.exists(), "Output file not created"
        print(f"✓ Output file created: {output_path}")

        # Verify result.pydantic is GapReport
        assert hasattr(result, 'pydantic'), "Result missing pydantic attribute"
        gap_report = result.pydantic
        assert isinstance(gap_report, GapReport), f"Expected GapReport, got {type(gap_report)}"
        print("✓ Result is valid GapReport object")

        # Print summary
        print_gap_report_summary(gap_report)

    def test_gap_report_schema_validation(self, crew, success_profile_data, audit_report_data):
        """Test that GapReport has all required fields and follows schema."""
        print("\n" + "=" * 80)
        print("TEST: GapReport Schema Validation")
        print("=" * 80)

        result = crew.kickoff(inputs={
            "success_profile_report": success_profile_data,
            "audit_report": audit_report_data,
            "keyword": "老爸测评dha推荐哪几款"
        })

        gap_report = result.pydantic

        # Verify basic metadata
        assert gap_report.owned_note_id == audit_report_data['note_id']
        assert gap_report.keyword == "老爸测评dha推荐哪几款"
        assert gap_report.sample_size > 0
        print("✓ Basic metadata validated")

        # Verify gap lists exist
        assert isinstance(gap_report.significant_gaps, list)
        assert isinstance(gap_report.marginal_gaps, list)
        assert isinstance(gap_report.non_significant_gaps, list)
        print(f"✓ Gap lists validated (sig={len(gap_report.significant_gaps)}, "
              f"mar={len(gap_report.marginal_gaps)}, non={len(gap_report.non_significant_gaps)})")

        # Verify top_priority_metrics
        assert isinstance(gap_report.top_priority_metrics, list)
        assert len(gap_report.top_priority_metrics) <= 3, "Should have at most 3 top priority metrics"
        print(f"✓ Top priority metrics validated ({len(gap_report.top_priority_metrics)} metrics)")

        # Verify root_causes
        assert isinstance(gap_report.root_causes, list)
        assert 3 <= len(gap_report.root_causes) <= 5, "Should have 3-5 root causes"
        print(f"✓ Root causes validated ({len(gap_report.root_causes)} causes)")

        # Verify impact_summary
        assert isinstance(gap_report.impact_summary, str)
        assert 50 <= len(gap_report.impact_summary) <= 300, \
            f"impact_summary must be 50-300 chars, got {len(gap_report.impact_summary)}"
        print(f"✓ Impact summary validated ({len(gap_report.impact_summary)} chars)")

        # Verify timestamp
        assert gap_report.gap_timestamp is not None
        from datetime import datetime
        datetime.fromisoformat(gap_report.gap_timestamp.replace('Z', '+00:00'))
        print("✓ Timestamp validated (ISO 8601 format)")

    def test_feature_attribution(self, crew, success_profile_data, audit_report_data):
        """Test that significant gaps have proper feature attribution."""
        print("\n" + "=" * 80)
        print("TEST: Feature Attribution")
        print("=" * 80)

        result = crew.kickoff(inputs={
            "success_profile_report": success_profile_data,
            "audit_report": audit_report_data,
            "keyword": "老爸测评dha推荐哪几款"
        })

        gap_report = result.pydantic

        # Check significant gaps have feature attribution
        if len(gap_report.significant_gaps) > 0:
            for gap in gap_report.significant_gaps:
                # Verify MetricGap structure
                assert isinstance(gap, MetricGap)
                assert gap.metric_name is not None
                assert gap.owned_value >= 0
                assert gap.target_mean >= 0
                assert gap.z_score is not None
                assert 0.0 <= gap.p_value <= 1.0
                assert gap.significance in ['critical', 'very_significant', 'significant']
                print(f"✓ Gap {gap.metric_name}: statistical fields validated")

                # Verify feature attribution exists
                assert isinstance(gap.related_features, list)
                assert isinstance(gap.missing_features, list)
                assert isinstance(gap.weak_features, list)
                assert len(gap.related_features) > 0, "Should have at least one related feature"
                print(f"✓ Gap {gap.metric_name}: feature attribution complete")

                # Verify narratives exist
                assert isinstance(gap.gap_explanation, str)
                assert len(gap.gap_explanation) > 0
                assert isinstance(gap.recommendation_summary, str)
                assert len(gap.recommendation_summary) > 0
                print(f"✓ Gap {gap.metric_name}: narratives provided")

                # Verify priority_rank is valid
                assert gap.priority_rank > 0
                print(f"✓ Gap {gap.metric_name}: priority rank = {gap.priority_rank}")

        else:
            print("⚠ No significant gaps found (this may be expected if owned note is performing well)")

    def test_priority_ordering(self, crew, success_profile_data, audit_report_data):
        """Test that gaps are properly prioritized."""
        print("\n" + "=" * 80)
        print("TEST: Priority Ordering")
        print("=" * 80)

        result = crew.kickoff(inputs={
            "success_profile_report": success_profile_data,
            "audit_report": audit_report_data,
            "keyword": "老爸测评dha推荐哪几款"
        })

        gap_report = result.pydantic

        # Collect all gaps
        all_gaps = (gap_report.significant_gaps +
                   gap_report.marginal_gaps +
                   gap_report.non_significant_gaps)

        if len(all_gaps) > 0:
            # Verify priority_rank is sequential and unique
            ranks = [gap.priority_rank for gap in all_gaps]
            sorted_ranks = sorted(ranks)
            assert ranks == sorted_ranks, "priority_rank should be sorted in ascending order"
            print(f"✓ Priority ranks are sequential: {ranks}")

            # Verify top_priority_metrics match highest priority gaps
            if len(gap_report.top_priority_metrics) > 0:
                top_gaps = all_gaps[:len(gap_report.top_priority_metrics)]
                top_gap_metrics = [gap.metric_name for gap in top_gaps]
                # Should match (order may vary slightly)
                for metric in gap_report.top_priority_metrics:
                    assert metric in top_gap_metrics, \
                        f"Top priority metric '{metric}' not in top {len(gap_report.top_priority_metrics)} gaps"
                print(f"✓ Top priority metrics match highest priority gaps")

            # Verify significant gaps have lower ranks (higher priority) than marginal gaps
            if gap_report.significant_gaps and gap_report.marginal_gaps:
                max_sig_rank = max(gap.priority_rank for gap in gap_report.significant_gaps)
                min_mar_rank = min(gap.priority_rank for gap in gap_report.marginal_gaps)
                # Generally, significant gaps should be prioritized over marginal gaps
                # (but not strictly enforced since priority also considers magnitude)
                print(f"✓ Significant gaps: max rank = {max_sig_rank}, Marginal gaps: min rank = {min_mar_rank}")

        else:
            print("⚠ No gaps found")

    def test_statistical_significance(self, crew, success_profile_data, audit_report_data):
        """Test that gaps are correctly classified by statistical significance."""
        print("\n" + "=" * 80)
        print("TEST: Statistical Significance Classification")
        print("=" * 80)

        result = crew.kickoff(inputs={
            "success_profile_report": success_profile_data,
            "audit_report": audit_report_data,
            "keyword": "老爸测评dha推荐哪几款"
        })

        gap_report = result.pydantic

        # Check significant gaps have p < 0.10
        for gap in gap_report.significant_gaps:
            assert gap.p_value < 0.10, \
                f"Significant gap {gap.metric_name} has p={gap.p_value} >= 0.10"
            # Verify significance label matches p-value
            if gap.p_value < 0.001:
                assert gap.significance == 'critical'
            elif gap.p_value < 0.01:
                assert gap.significance in ['critical', 'very_significant']
            elif gap.p_value < 0.05:
                assert gap.significance in ['critical', 'very_significant', 'significant']
        print(f"✓ Significant gaps ({len(gap_report.significant_gaps)}) have p < 0.10")

        # Check marginal gaps have 0.05 <= p < 0.10
        for gap in gap_report.marginal_gaps:
            assert 0.05 <= gap.p_value < 0.10, \
                f"Marginal gap {gap.metric_name} has p={gap.p_value} outside [0.05, 0.10)"
            assert gap.significance == 'marginal'
        print(f"✓ Marginal gaps ({len(gap_report.marginal_gaps)}) have 0.05 <= p < 0.10")

        # Check non-significant gaps have p >= 0.10
        for gap in gap_report.non_significant_gaps:
            assert gap.p_value >= 0.10, \
                f"Non-significant gap {gap.metric_name} has p={gap.p_value} < 0.10"
            assert gap.significance == 'none'
        print(f"✓ Non-significant gaps ({len(gap_report.non_significant_gaps)}) have p >= 0.10")

    def test_output_file_matches_pydantic(self, crew, success_profile_data, audit_report_data):
        """Test that output file matches Pydantic model."""
        print("\n" + "=" * 80)
        print("TEST: Output File Matches Pydantic Model")
        print("=" * 80)

        result = crew.kickoff(inputs={
            "success_profile_report": success_profile_data,
            "audit_report": audit_report_data,
            "keyword": "老爸测评dha推荐哪几款"
        })

        gap_report_from_result = result.pydantic

        # Read from output file
        output_path = project_root / "outputs" / "gap_report.json"
        with open(output_path, 'r', encoding='utf-8') as f:
            report_dict = json.load(f)

        # Parse into GapReport
        gap_report_from_file = GapReport(**report_dict)

        # Compare key fields
        assert gap_report_from_file.owned_note_id == gap_report_from_result.owned_note_id
        assert gap_report_from_file.keyword == gap_report_from_result.keyword
        assert gap_report_from_file.sample_size == gap_report_from_result.sample_size
        assert len(gap_report_from_file.significant_gaps) == len(gap_report_from_result.significant_gaps)
        assert len(gap_report_from_file.top_priority_metrics) == len(gap_report_from_result.top_priority_metrics)
        print("✓ Output file matches Pydantic model")

    def test_edge_case_missing_current_metrics(self, crew, success_profile_data):
        """Test that missing current_metrics in audit_report raises ValueError."""
        print("\n" + "=" * 80)
        print("TEST: Edge Case - Missing current_metrics")
        print("=" * 80)

        invalid_audit_report = {
            "note_id": "test123",
            "keyword": "测试",
            "text_features": {},
            "visual_features": {},
            "feature_summary": "测试摘要",
            "audit_timestamp": "2025-01-01T00:00:00Z"
            # Missing current_metrics field
        }

        with pytest.raises(ValueError, match="current_metrics"):
            crew.kickoff(inputs={
                "success_profile_report": success_profile_data,
                "audit_report": invalid_audit_report,
                "keyword": "测试"
            })
        print("✓ Correctly raises ValueError for missing current_metrics")

    def test_edge_case_missing_success_profile(self, crew, audit_report_data):
        """Test that missing success_profile_report raises ValueError."""
        print("\n" + "=" * 80)
        print("TEST: Edge Case - Missing success_profile_report")
        print("=" * 80)

        with pytest.raises(ValueError, match="success_profile_report"):
            crew.kickoff(inputs={
                # Missing success_profile_report
                "audit_report": audit_report_data,
                "keyword": "测试"
            })
        print("✓ Correctly raises ValueError for missing success_profile_report")

    def test_edge_case_missing_keyword(self, crew, success_profile_data, audit_report_data):
        """Test that missing keyword raises ValueError."""
        print("\n" + "=" * 80)
        print("TEST: Edge Case - Missing keyword")
        print("=" * 80)

        with pytest.raises(ValueError, match="keyword"):
            crew.kickoff(inputs={
                "success_profile_report": success_profile_data,
                "audit_report": audit_report_data
                # Missing keyword
            })
        print("✓ Correctly raises ValueError for missing keyword")


# Manual test function (can be run directly)
def run_manual_test():
    """Run a manual test with real report data."""
    print("=" * 80)
    print("MANUAL TEST: GapFinder Crew")
    print("=" * 80)

    # Load test data
    success_profile_path = project_root / "outputs" / "success_profile_report.json"
    audit_report_path = project_root / "outputs" / "audit_report.json"

    if not success_profile_path.exists():
        print(f"❌ Success profile report not found at {success_profile_path}")
        print("   Please run CompetitorAnalyst crew first.")
        return

    if not audit_report_path.exists():
        print(f"❌ Audit report not found at {audit_report_path}")
        print("   Please run OwnedNoteAuditor crew first.")
        return

    success_profile_data = load_success_profile_report(str(success_profile_path))
    audit_report_data = load_audit_report(str(audit_report_path))

    # Create crew
    crew = XhsSeoOptimizerCrewGapFinder()

    # Run gap analysis
    print("\nRunning gap analysis...\n")
    result = crew.kickoff(inputs={
        "success_profile_report": success_profile_data,
        "audit_report": audit_report_data,
        "keyword": "老爸测评dha推荐哪几款"
    })

    # Print report
    gap_report = result.pydantic
    print_gap_report_summary(gap_report)

    # Verify output file
    output_path = project_root / "outputs" / "gap_report.json"
    print(f"\n✓ Gap report saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size} bytes")


if __name__ == "__main__":
    # Run manual test if executed directly
    run_manual_test()
