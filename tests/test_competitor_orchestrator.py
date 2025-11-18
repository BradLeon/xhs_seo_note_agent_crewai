"""Test script for CompetitorAnalysisOrchestrator tool.

This script tests the complete analysis workflow using real target_notes data.
"""

import json
import os
import sys
from pathlib import Path

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
from xhs_seo_optimizer.models.reports import SuccessProfileReport
from xhs_seo_optimizer.tools import CompetitorAnalysisOrchestrator
import logging
logging.basicConfig(level=logging.INFO)


def load_target_notes(json_path: str) -> list[Note]:
    """Load target notes from JSON file.

    Args:
        json_path: Path to target_notes.json

    Returns:
        List of Note objects
    """
    print(f"Loading target notes from: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    notes = [Note.from_json(note_dict) for note_dict in data]
    print(f"✓ Loaded {len(notes)} target notes\n")

    return notes


def print_report_summary(report_json: str):
    """Print a summary of the analysis report.

    Args:
        report_json: JSON string of SuccessProfileReport
    """
    # Parse JSON into SuccessProfileReport object
    report = SuccessProfileReport.model_validate_json(report_json)

    print("=" * 80)
    print("COMPETITOR ANALYSIS REPORT SUMMARY")
    print("=" * 80)

    # Basic info
    print(f"\n关键词: {report.keyword}")
    print(f"样本数量: {report.sample_size}")
    print(f"分析时间: {report.analysis_timestamp}")

    # Aggregated stats
    print(f"\n聚合统计:")
    stats = report.aggregated_stats
    print(f"  - 移除异常值: {stats.outliers_removed}")
    print(f"  - 分析指标数: {len(stats.prediction_stats)}")
    print(f"  - 标签维度数: {len(stats.tag_frequencies)}")

    # Metric profiles summary
    print(f"\n指标分析结果: {len(report.metric_profiles)} 个指标")
    for profile in report.metric_profiles:
        print(f"  - {profile.metric_name}: {len(profile.feature_analyses)} 个特征 "
              f"(样本:{profile.sample_size}, {profile.variance_level} variance)")

    # Key success factors
    print(f"\n关键成功因素:")
    for i, factor in enumerate(report.key_success_factors, 1):
        print(f"  {i}. {factor}")

    # Viral formula summary
    print(f"\n爆款公式总结:")
    print(f"  {report.viral_formula_summary}")

    # Metric-centric detailed analysis
    print("\n" + "=" * 80)
    print("详细指标分析（Metric-Centric）")
    print("=" * 80)

    for profile in report.metric_profiles:
        print(f"\n【{profile.metric_name.upper()}】")
        print(f"样本量: {profile.sample_size} ({profile.variance_level} variance)")
        print(f"整体叙述: {profile.metric_success_narrative}")
        print(f"\n相关特征分析:")

        for feature_name, analysis in profile.feature_analyses.items():
            print(f"\n  • {feature_name}")
            print(f"    流行度: {analysis.prevalence_pct:.1f}% ({analysis.prevalence_count}/{profile.sample_size})")
            print(f"    为什么有效: {analysis.why_it_works}")
            print(f"    创作公式: {analysis.creation_formula}")
            print(f"    关键要素: {', '.join(analysis.key_elements)}")
            if analysis.examples:
                print(f"    示例: {analysis.examples[0]}")


def main():
    """Main test function."""
    print("\n" + "=" * 80)
    print("测试 CompetitorAnalysisOrchestrator")
    print("=" * 80 + "\n")

    # Load target notes
    notes_path = project_root / "docs" / "target_notes.json"
    target_notes = load_target_notes(str(notes_path))

    # Create orchestrator tool
    print("创建 CompetitorAnalysisOrchestrator 工具...")
    orchestrator = CompetitorAnalysisOrchestrator()
    print("✓ 工具创建成功\n")

    # Run analysis
    keyword = "老爸测评dha推荐哪几款"
    print(f"开始分析竞品笔记 (关键词: {keyword})...")
    print("-" * 80 + "\n")

    try:
        report_json = orchestrator._run(
            target_notes=target_notes,
            keyword=keyword
        )

        print("\n" + "-" * 80)
        print("✓ 分析完成！\n")

        # Print summary
        print_report_summary(report_json)

        # Save report to file
        output_dir = project_root / "tests" / "output"
        output_dir.mkdir(exist_ok=True)

        output_path = output_dir / "competitor_analysis_report.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            report_dict = json.loads(report_json)
            json.dump(report_dict, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 80)
        print(f"完整报告已保存到: {output_path}")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
