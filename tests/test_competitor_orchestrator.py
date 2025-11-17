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
from xhs_seo_optimizer.models.reports import SuccessProfileReport, FeaturePattern
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

    # Patterns summary
    total_patterns = (
        len(report.title_patterns) +
        len(report.cover_patterns) +
        len(report.content_patterns) +
        len(report.tag_patterns)
    )

    print(f"\n发现的成功模式: {total_patterns} 个")
    print(f"  - 标题模式: {len(report.title_patterns)}")
    print(f"  - 封面模式: {len(report.cover_patterns)}")
    print(f"  - 内容模式: {len(report.content_patterns)}")
    print(f"  - 标签模式: {len(report.tag_patterns)}")

    # Key success factors
    print(f"\n关键成功因素:")
    for i, factor in enumerate(report.key_success_factors, 1):
        print(f"  {i}. {factor}")

    # Viral formula summary
    print(f"\n爆款公式总结:")
    print(f"  {report.viral_formula_summary}")

    # Pattern details
    print("\n" + "=" * 80)
    print("详细模式分析")
    print("=" * 80)

    for pattern_type, patterns in [
        ("标题模式", report.title_patterns),
        ("封面模式", report.cover_patterns),
        ("内容模式", report.content_patterns),
        ("标签模式", report.tag_patterns)
    ]:
        if patterns:
            print(f"\n【{pattern_type}】")
            for i, pattern in enumerate(patterns, 1):
                print(f"\n{i}. {pattern.feature_name}")
                print(f"   流行度: {pattern.prevalence_pct:.1f}% (在{pattern.sample_size}个目标笔记中)")
                print(f"   统计证据: {pattern.statistical_evidence}")
                print(f"   影响指标: {', '.join(pattern.affected_metrics.keys())}")
                print(f"   为什么有效: {pattern.why_it_works}")
                print(f"   创作公式: {pattern.creation_formula}")
                print(f"   关键要素:")
                for j, element in enumerate(pattern.key_elements, 1):
                    print(f"      {j}) {element}")
                if pattern.examples:
                    print(f"   示例: {pattern.examples[0]}")


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
