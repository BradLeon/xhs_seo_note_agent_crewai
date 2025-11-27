"""Test script for refactored CompetitorAnalyst agent (without orchestrator tool).

This script tests the agent-based workflow where the agent coordinates atomic tools
directly, following the CrewAI principle of agents as coordinators.
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

from xhs_seo_optimizer.models.note import Note, ComplexInput
from xhs_seo_optimizer.models.reports import SuccessProfileReport
from xhs_seo_optimizer.crew_competitor_analyst import XhsSeoOptimizerCrewCompetitorAnalyst
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


def print_report_summary(report: SuccessProfileReport):
    """Print a summary of the analysis report.

    Args:
        report: SuccessProfileReport object
    """
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

    # Metric-centric detailed analysis (first metric only for brevity)
    if report.metric_profiles:
        print("\n" + "=" * 80)
        print("详细指标分析示例（首个指标）")
        print("=" * 80)

        profile = report.metric_profiles[0]
        print(f"\n【{profile.metric_name.upper()}】")
        print(f"样本量: {profile.sample_size} ({profile.variance_level} variance)")
        print(f"整体叙述: {profile.metric_success_narrative}")
        print(f"\n相关特征分析 (前3个):")

        for i, (feature_name, analysis) in enumerate(list(profile.feature_analyses.items())[:3], 1):
            print(f"\n  {i}. {feature_name}")
            print(f"     流行度: {analysis.prevalence_pct:.1f}%")
            print(f"     为什么有效: {analysis.why_it_works}")
            print(f"     创作公式: {analysis.creation_formula}")


def test_with_crew():
    """Test using the full CrewAI crew (agent-based approach)."""
    print("\n" + "=" * 80)
    print("测试重构后的 CompetitorAnalyst Agent (无 Orchestrator Tool)")
    print("=" * 80 + "\n")

    # Load target notes
    notes_path = project_root / "docs" / "target_notes.json"
    target_notes = load_target_notes(str(notes_path))

    # Create simplified crew
    print("创建 XhsSeoOptimizerCrewCompetitorAnalyst (简化版，仅含 competitor_analyst)...")
    crew = XhsSeoOptimizerCrewCompetitorAnalyst().crew()
    print("✓ Crew 创建成功\n")

    # Prepare inputs using ComplexInput wrapper (following CrewAI official pattern)
    keyword = "老爸测评dha推荐哪几款"
    complex_input = ComplexInput(
        target_notes=target_notes,
        keyword=keyword,
        owned_note=None  # Not needed for competitor analysis only
    )

    print(f"开始使用 CrewAI 分析竞品笔记 (关键词: {keyword})...")
    print("注意: 这是 agent 协调 atomic tools 的方式，不是单一 orchestrator tool")
    print(f"✓ 使用 ComplexInput 包装器序列化输入 ({len(target_notes)} 条笔记)")
    print("-" * 80 + "\n")

    try:
        # Run the crew with serialized inputs (model_dump() converts Pydantic to dict)
        result = crew.kickoff(inputs=complex_input.model_dump())

        print("\n" + "-" * 80)
        print("✓ 分析完成！\n")

        # The result should be a SuccessProfileReport
        if hasattr(result, 'pydantic'):
            # Extract the pydantic model from CrewOutput
            report = result.pydantic
            if isinstance(report, SuccessProfileReport):
                print_report_summary(report)

                # Save report to file
                output_dir = project_root / "tests" / "output"
                output_dir.mkdir(exist_ok=True)

                output_path = output_dir / "competitor_agent_refactored_report.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report.model_dump(), f, ensure_ascii=False, indent=2, default=str)

                print("\n" + "=" * 80)
                print(f"完整报告已保存到: {output_path}")
                print("=" * 80 + "\n")
            else:
                print(f"警告: 结果不是 SuccessProfileReport，而是 {type(report)}")
        else:
            print(f"警告: 结果格式意外: {type(result)}")
            print(f"结果内容: {result}")

    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_helper_functions():
    """Test the helper functions directly with shared context (for debugging)."""
    print("\n" + "=" * 80)
    print("测试辅助函数 (使用共享上下文)")
    print("=" * 80 + "\n")

    # Load target notes
    notes_path = project_root / "docs" / "target_notes.json"
    target_notes = load_target_notes(str(notes_path))

    # Import context and tools
    from xhs_seo_optimizer.context import CrewContext
    from xhs_seo_optimizer.tools import (
        DataAggregatorTool,
        NLPAnalysisTool,
        MultiModalVisionTool
    )
    from xhs_seo_optimizer.analysis_helpers import (
        extract_features_matrix,
        filter_notes_by_metric_variance,
        analyze_metric_success,
        generate_summary_insights
    )
    from xhs_seo_optimizer.attribution import get_all_metrics

    keyword = "老爸测评dha推荐哪几款"

    try:
        # Create shared context
        print("创建共享上下文...")
        context = CrewContext()

        # Store notes in context
        notes_key = context.store_notes("target_notes", target_notes)
        for note in target_notes:
            meta_key = f"note_{note.note_id}_metadata"
            context.store_data(meta_key, note.meta_data, "note_metadata")
        print(f"✓ 存储了 {len(target_notes)} 条笔记到上下文\n")

        # Step 1: Aggregate statistics using DataAggregatorTool with context
        print("步骤1: 统计聚合 (使用 DataAggregatorTool with context)...")
        aggregator = DataAggregatorTool(context=context)
        aggregated_json = aggregator._run(notes_key="target_notes")
        aggregated_stats = json.loads(aggregated_json)
        print(f"✓ 完成: {aggregated_stats['sample_size']} 笔记, "
              f"{aggregated_stats['outliers_removed']} 异常值移除\n")

        # Step 2: Extract features (demonstrate tool usage with context)
        print("步骤2: 特征提取 (演示工具使用上下文)...")

        # Test NLP tool with context
        nlp_tool = NLPAnalysisTool(context=context)
        first_note = target_notes[0]
        nlp_result = nlp_tool._run(note_meta_data_key=f"note_{first_note.note_id}_metadata")
        print(f"  - NLP分析工具测试: 成功分析了note_{first_note.note_id}")

        # Test Vision tool with context
        vision_tool = MultiModalVisionTool(context=context)
        vision_result = vision_tool._run(note_meta_data_key=f"note_{first_note.note_id}_metadata")
        print(f"  - 视觉分析工具测试: 成功分析了note_{first_note.note_id}")

        # Continue with full feature extraction
        features_matrix = extract_features_matrix(target_notes)
        print(f"✓ 完成: 提取了 {len(features_matrix)} 条笔记的特征\n")

        # Step 3-4: Analyze each metric
        print("步骤3-4: 指标分析...")
        metric_profiles = []
        all_metrics = get_all_metrics()

        for metric in all_metrics[:2]:  # Test first 2 metrics only for speed
            print(f"  分析指标: {metric}")

            # Filter notes by variance
            filtered_notes, variance_level = filter_notes_by_metric_variance(
                target_notes, metric
            )

            if filtered_notes:
                # Analyze metric
                profile = analyze_metric_success(
                    metric=metric,
                    filtered_notes=filtered_notes,
                    features_matrix=features_matrix,
                    variance_level=variance_level,
                    keyword=keyword
                )
                metric_profiles.append(profile)
                print(f"    ✓ 完成: {profile.sample_size} 笔记, "
                      f"{len(profile.feature_analyses)} 特征\n")
            else:
                print(f"    ⚠ 跳过: 数据不足\n")

        # Step 5: Generate summary
        print("步骤5: 生成汇总...")
        key_factors, viral_formula = generate_summary_insights(metric_profiles)
        print(f"✓ 完成: {len(key_factors)} 个关键因素\n")

        print("关键成功因素:")
        for i, factor in enumerate(key_factors, 1):
            print(f"  {i}. {factor}")

        print(f"\n爆款公式: {viral_formula}")

        print("\n" + "=" * 80)
        print("✓ 辅助函数测试成功！")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main test function."""
    import argparse
    parser = argparse.ArgumentParser(description="Test refactored CompetitorAnalyst agent")
    parser.add_argument(
        "--mode",
        choices=["crew", "helpers", "both"],
        default="crew",
        help="Test mode: crew (full crew), helpers (helper functions), or both"
    )
    args = parser.parse_args()

    if args.mode in ["helpers", "both"]:
        test_helper_functions()

    if args.mode in ["crew", "both"]:
        test_with_crew()


if __name__ == "__main__":
    main()