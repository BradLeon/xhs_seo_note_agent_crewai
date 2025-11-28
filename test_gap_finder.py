#!/usr/bin/env python
"""Test GapFinder crew directly with existing audit_report and success_profile_report.

Usage:
    PYTHONPATH=src:$PYTHONPATH python test_gap_finder.py
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from xhs_seo_optimizer.crew_gap_finder import XhsSeoOptimizerCrewGapFinder


def main():
    # Load existing reports
    outputs_dir = Path(__file__).parent / "outputs"

    audit_report_path = outputs_dir / "audit_report.json"
    success_profile_path = outputs_dir / "success_profile_report.json"

    if not audit_report_path.exists():
        print(f"❌ audit_report.json not found at {audit_report_path}")
        return 1

    if not success_profile_path.exists():
        print(f"❌ success_profile_report.json not found at {success_profile_path}")
        return 1

    print("📂 Loading existing reports...")
    with open(audit_report_path, 'r', encoding='utf-8') as f:
        audit_report = json.load(f)

    with open(success_profile_path, 'r', encoding='utf-8') as f:
        success_profile_report = json.load(f)

    # Extract keyword from audit_report
    keyword = audit_report.get('keyword', '老爸测评dha推荐哪几款')

    print(f"✅ Loaded audit_report for note: {audit_report.get('note_id')}")
    print(f"✅ Loaded success_profile_report with {success_profile_report.get('sample_size')} samples")
    print(f"✅ Keyword: {keyword}")

    # Prepare inputs for GapFinder
    inputs = {
        'audit_report': audit_report,
        'success_profile_report': success_profile_report,
        'keyword': keyword,
    }

    # Run GapFinder crew
    print("\n🚀 Starting GapFinder crew...")
    print("=" * 60)

    gap_finder = XhsSeoOptimizerCrewGapFinder()
    result = gap_finder.kickoff(inputs=inputs, save_to_file=True)

    print("=" * 60)
    print("\n✅ GapFinder completed!")

    # Show summary
    if hasattr(result, 'pydantic') and result.pydantic:
        gap_report = result.pydantic
        print(f"\n📊 Gap Report Summary:")
        print(f"  - Significant gaps: {len(gap_report.significant_gaps)}")
        print(f"  - Marginal gaps: {len(gap_report.marginal_gaps)}")
        print(f"  - Non-significant gaps: {len(gap_report.non_significant_gaps)}")
        print(f"  - Top priority metrics: {gap_report.top_priority_metrics}")
        print(f"  - Root causes: {gap_report.root_causes[:3]}...")
    else:
        print(f"\n📄 Raw output (first 500 chars):")
        print(str(result)[:500])

    print(f"\n💾 Report saved to outputs/gap_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
