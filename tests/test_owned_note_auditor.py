"""Integration tests for OwnedNoteAuditor crew.

Tests the complete workflow of objective content understanding for owned notes:
- Feature extraction (text + visual)
- Objective feature summary (no strength/weakness judgment)
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

from xhs_seo_optimizer.models.note import Note
from xhs_seo_optimizer.models.reports import AuditReport
from xhs_seo_optimizer.crew_owned_note import XhsSeoOptimizerCrewOwnedNote
import logging
logging.basicConfig(level=logging.INFO)


def load_owned_note(json_path: str) -> Note:
    """Load owned note from JSON file.

    Args:
        json_path: Path to owned_note.json

    Returns:
        Note object
    """
    print(f"Loading owned note from: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert flat JSON to Note object using Note.from_json()
    note = Note.from_json(data)
    print(f"✓ Loaded owned note: {note.note_id}\n")

    return note


def print_audit_report_summary(report: AuditReport):
    """Print a summary of the audit report.

    Args:
        report: AuditReport object
    """
    print("=" * 80)
    print("OWNED NOTE AUDIT REPORT SUMMARY")
    print("=" * 80)

    # Basic info
    print(f"\n笔记ID: {report.note_id}")
    print(f"关键词: {report.keyword}")
    print(f"审计时间: {report.audit_timestamp}")

    # Feature summary
    print(f"\n客观特征摘要:")
    print(f"  {report.feature_summary}")

    # Feature details (summary)
    print(f"\n提取的特征:")
    print(f"  - 文本特征: {len(report.text_features.model_dump())} 个字段")
    print(f"    - 标题套路: {report.text_features.title_pattern}")
    print(f"    - 开头策略: {report.text_features.opening_strategy}")
    print(f"    - 正文框架: {report.text_features.content_framework}")
    print(f"    - 结尾技巧: {report.text_features.ending_technique}")
    print(f"  - 视觉特征: {len(report.visual_features.model_dump())} 个字段")
    print(f"    - 封面吸引力: {report.visual_features.thumbnail_appeal}")
    print(f"    - 色彩方案: {report.visual_features.color_scheme}")
    print(f"    - 视觉调性: {report.visual_features.visual_tone}")

    print("\n" + "=" * 80)


class TestOwnedNoteAuditor:
    """Integration tests for OwnedNoteAuditor crew."""

    @pytest.fixture
    def owned_note_data(self):
        """Load owned note test data."""
        docs_path = project_root / "docs" / "owned_note.json"
        return load_owned_note(str(docs_path))

    @pytest.fixture
    def crew(self):
        """Create OwnedNoteAuditor crew instance."""
        return XhsSeoOptimizerCrewOwnedNote().crew()

    def test_basic_execution(self, crew, owned_note_data):
        """Test basic crew execution without errors."""
        print("\n" + "=" * 80)
        print("TEST: Basic Execution")
        print("=" * 80)

        # Run crew
        result = crew.kickoff(inputs={
            "owned_note": owned_note_data,
            "keyword": "老爸测评dha推荐哪几款"
        })

        # Verify result exists
        assert result is not None
        print("✓ Crew execution completed successfully")

        # Verify output file exists
        output_path = project_root / "outputs" / "audit_report.json"
        assert output_path.exists(), "Output file not created"
        print(f"✓ Output file created: {output_path}")

        # Verify result.pydantic is AuditReport
        assert hasattr(result, 'pydantic'), "Result missing pydantic attribute"
        audit_report = result.pydantic
        assert isinstance(audit_report, AuditReport), f"Expected AuditReport, got {type(audit_report)}"
        print("✓ Result is valid AuditReport object")

        # Print summary
        print_audit_report_summary(audit_report)

    def test_report_schema_validation(self, crew, owned_note_data):
        """Test that AuditReport has all required fields."""
        print("\n" + "=" * 80)
        print("TEST: Report Schema Validation")
        print("=" * 80)

        result = crew.kickoff(inputs={
            "owned_note": owned_note_data,
            "keyword": "老爸测评dha推荐哪几款"
        })

        audit_report = result.pydantic

        # Verify basic fields
        assert audit_report.note_id == owned_note_data.note_id
        assert audit_report.keyword == "老爸测评dha推荐哪几款"
        print("✓ Basic fields (note_id, keyword) validated")

        # Verify features
        assert audit_report.text_features is not None
        assert audit_report.visual_features is not None
        assert audit_report.text_features.note_id == owned_note_data.note_id
        assert audit_report.visual_features.note_id == owned_note_data.note_id
        print("✓ Extracted features validated")

        # Verify feature summary
        assert isinstance(audit_report.feature_summary, str)
        assert 50 <= len(audit_report.feature_summary) <= 300, \
            f"feature_summary must be 50-300 chars, got {len(audit_report.feature_summary)}"
        # Should not contain judgment words
        judgment_words = ['好', '坏', '弱', '差', '优秀', '不足', 'weak', 'bad', 'poor', 'strong', 'excellent']
        lower_summary = audit_report.feature_summary.lower()
        found_judgments = [word for word in judgment_words if word in lower_summary]
        assert len(found_judgments) == 0, f"feature_summary should be objective, found: {found_judgments}"
        print(f"✓ Feature summary validated ({len(audit_report.feature_summary)} chars, objective)")

        # Verify timestamp
        assert audit_report.audit_timestamp is not None
        # Should be ISO 8601 format
        from datetime import datetime
        datetime.fromisoformat(audit_report.audit_timestamp.replace('Z', '+00:00'))
        print("✓ Timestamp validated (ISO 8601 format)")

    def test_feature_extraction_quality(self, crew, owned_note_data):
        """Test that extracted features are meaningful."""
        print("\n" + "=" * 80)
        print("TEST: Feature Extraction Quality")
        print("=" * 80)

        result = crew.kickoff(inputs={
            "owned_note": owned_note_data,
            "keyword": "老爸测评dha推荐哪几款"
        })

        audit_report = result.pydantic

        # Text features quality checks
        text_features = audit_report.text_features
        assert len(text_features.title_pattern) > 0, "title_pattern should not be empty"
        assert len(text_features.opening_strategy) > 0, "opening_strategy should not be empty"
        assert len(text_features.content_framework) > 0, "content_framework should not be empty"
        assert isinstance(text_features.title_keywords, list)
        assert isinstance(text_features.content_logic, list)
        assert isinstance(text_features.pain_points, list)
        assert isinstance(text_features.value_propositions, list)
        print("✓ Text features are complete and meaningful")

        # Visual features quality checks
        visual_features = audit_report.visual_features
        assert visual_features.image_count >= 1, "Should have at least cover image"
        assert len(visual_features.image_quality) > 0
        assert len(visual_features.thumbnail_appeal) > 0
        assert len(visual_features.color_scheme) > 0
        print("✓ Visual features are complete and meaningful")

    def test_feature_summary_objectivity(self, crew, owned_note_data):
        """Test that feature summary is objective (no judgment)."""
        print("\n" + "=" * 80)
        print("TEST: Feature Summary Objectivity")
        print("=" * 80)

        result = crew.kickoff(inputs={
            "owned_note": owned_note_data,
            "keyword": "老爸测评dha推荐哪几款"
        })

        audit_report = result.pydantic

        # Should be objective description
        summary = audit_report.feature_summary

        # Check length
        assert 50 <= len(summary) <= 300, f"Summary length {len(summary)} not in range [50, 300]"

        # Should not contain judgment words
        judgment_words = ['好', '坏', '弱', '差', '优秀', '不足', 'weak', 'bad', 'poor', 'strong', 'excellent']
        lower_summary = summary.lower()
        found_judgments = [word for word in judgment_words if word in lower_summary]

        assert len(found_judgments) == 0, \
            f"Feature summary should be objective, but found judgment words: {found_judgments}\nSummary: {summary}"

        print(f"✓ Feature summary is objective ({len(summary)} chars)")
        print(f"  Summary: {summary}")

    def test_output_file_matches_pydantic(self, crew, owned_note_data):
        """Test that output file matches Pydantic model."""
        print("\n" + "=" * 80)
        print("TEST: Output File Matches Pydantic Model")
        print("=" * 80)

        result = crew.kickoff(inputs={
            "owned_note": owned_note_data,
            "keyword": "老爸测评dha推荐哪几款"
        })

        audit_report_from_result = result.pydantic

        # Read from output file
        output_path = project_root / "outputs" / "audit_report.json"
        with open(output_path, 'r', encoding='utf-8') as f:
            report_dict = json.load(f)

        # Parse into AuditReport
        audit_report_from_file = AuditReport(**report_dict)

        # Compare key fields
        assert audit_report_from_file.note_id == audit_report_from_result.note_id
        assert audit_report_from_file.keyword == audit_report_from_result.keyword
        assert audit_report_from_file.audit_timestamp == audit_report_from_result.audit_timestamp
        assert audit_report_from_file.feature_summary == audit_report_from_result.feature_summary
        print("✓ Output file matches Pydantic model")

    def test_edge_case_missing_note_id(self, crew):
        """Test that missing note_id raises ValueError."""
        print("\n" + "=" * 80)
        print("TEST: Edge Case - Missing note_id")
        print("=" * 80)

        invalid_input = {
            "owned_note": {
                # Missing note_id
                "meta_data": {"title": "测试", "content": "内容"},
                "prediction": {"ctr": 0.10},
                "tag": {}
            },
            "keyword": "测试"
        }

        with pytest.raises(ValueError, match="note_id"):
            crew.kickoff(inputs=invalid_input)
        print("✓ Correctly raises ValueError for missing note_id")

    def test_edge_case_missing_prediction(self, crew):
        """Test that missing prediction raises ValueError."""
        print("\n" + "=" * 80)
        print("TEST: Edge Case - Missing prediction")
        print("=" * 80)

        invalid_input = {
            "owned_note": {
                "note_id": "test123",
                "meta_data": {"title": "测试", "content": "内容"},
                # Missing prediction field
                "tag": {}
            },
            "keyword": "测试"
        }

        with pytest.raises(ValueError, match="prediction"):
            crew.kickoff(inputs=invalid_input)
        print("✓ Correctly raises ValueError for missing prediction")


# Manual test function (can be run directly)
def run_manual_test():
    """Run a manual test with docs/owned_note.json."""
    print("=" * 80)
    print("MANUAL TEST: OwnedNoteAuditor Crew")
    print("=" * 80)

    # Load test data
    docs_path = project_root / "docs" / "owned_note.json"
    owned_note_data = load_owned_note(str(docs_path))

    # Create crew
    crew = XhsSeoOptimizerCrewOwnedNote().crew()

    # Run audit
    print("\nRunning audit...\n")
    result = crew.kickoff(inputs={
        "owned_note": owned_note_data,
        "keyword": "老爸测评dha推荐哪几款"
    })

    # Print report
    audit_report = result.pydantic
    print_audit_report_summary(audit_report)

    # Verify output file
    output_path = project_root / "outputs" / "audit_report.json"
    print(f"\n✓ Audit report saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size} bytes")


if __name__ == "__main__":
    # Run manual test if executed directly
    run_manual_test()
