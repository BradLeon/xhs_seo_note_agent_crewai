"""Integration test for DataAggregator and StatisticalDelta tools.

Uses REAL DATA from docs/owned_note.json and docs/target_notes.json.
Tests the complete workflow: aggregate target_notes -> compare with owned_note.
"""

import json
from pathlib import Path

from xhs_seo_optimizer.tools.data_aggregator import DataAggregatorTool
from xhs_seo_optimizer.tools.statistical_delta import StatisticalDeltaTool
from xhs_seo_optimizer.models.note import Note
from xhs_seo_optimizer.models.analysis_results import AggregatedMetrics


def load_real_data():
    """Load real data from JSON files."""
    docs_dir = Path(__file__).parent.parent / "docs"

    # Load owned note
    with open(docs_dir / "owned_note.json", "r", encoding="utf-8") as f:
        owned_note_data = json.load(f)
    owned_note = Note.from_json(owned_note_data)

    # Load target notes
    with open(docs_dir / "target_notes.json", "r", encoding="utf-8") as f:
        target_notes_data = json.load(f)
    target_notes = [Note.from_json(note_data) for note_data in target_notes_data]

    return owned_note, target_notes


def test_integration_with_real_data():
    """Test complete workflow with REAL Xiaohongshu note data."""

    print("\n" + "="*80)
    print("INTEGRATION TEST: Data Aggregation + Statistical Delta Analysis")
    print("Using REAL data from owned_note.json and target_notes.json")
    print("="*80)

    # Load real data
    owned_note, target_notes = load_real_data()

    print(f"\n📊 Data Loaded:")
    print(f"  - Owned note: {owned_note.note_id}")
    print(f"    Title: {owned_note.meta_data.title}")
    print(f"  - Target notes: {len(target_notes)} high-performing competitor notes")

    # Step 1: Aggregate target notes
    print("\n" + "-"*80)
    print("STEP 1: Aggregating Target Notes (CompetitorAnalyst)")
    print("-"*80)

    aggregator = DataAggregatorTool()
    aggregated_json = aggregator._run(notes=target_notes)
    aggregated_stats = AggregatedMetrics(**json.loads(aggregated_json))

    print(f"\n✅ Aggregated {aggregated_stats.sample_size} target notes")
    print(f"\n📈 Key Metrics (Target Notes Average):")
    print(f"  - CTR: {aggregated_stats.prediction_stats['ctr'].mean:.4f} (±{aggregated_stats.prediction_stats['ctr'].std:.4f})")
    print(f"  - Comment Rate: {aggregated_stats.prediction_stats['comment_rate'].mean:.6f} (±{aggregated_stats.prediction_stats['comment_rate'].std:.6f})")
    print(f"  - Sort Score: {aggregated_stats.prediction_stats['sort_score2'].mean:.4f} (±{aggregated_stats.prediction_stats['sort_score2'].std:.4f})")
    print(f"  - Interaction Rate: {aggregated_stats.prediction_stats['interaction_rate'].mean:.6f} (±{aggregated_stats.prediction_stats['interaction_rate'].std:.6f})")

    print(f"\n🏷️  Most Common Tags:")
    for field, mode in list(aggregated_stats.tag_modes.items())[:4]:
        print(f"  - {field}: {mode}")

    # Step 2: Analyze statistical deltas
    print("\n" + "-"*80)
    print("STEP 2: Analyzing Performance Gaps (GapFinder)")
    print("-"*80)

    delta_analyzer = StatisticalDeltaTool()
    gap_analysis_json = delta_analyzer._run(owned_note=owned_note, target_stats=aggregated_stats)
    gap_analysis = json.loads(gap_analysis_json)

    print(f"\n✅ Gap analysis completed")
    print(f"  - Significant gaps: {len(gap_analysis['significant_gaps'])}")
    print(f"  - Non-significant gaps: {len(gap_analysis['non_significant_gaps'])}")

    # Step 3: Display results
    print("\n" + "="*80)
    print("ANALYSIS RESULTS: What needs optimization?")
    print("="*80)

    if gap_analysis['significant_gaps']:
        print(f"\n🔴 SIGNIFICANT GAPS (p < 0.05) - Require Attention:")
        print("-"*80)

        for i, gap in enumerate(gap_analysis['significant_gaps'][:10], 1):
            print(f"\n{i}. {gap['metric'].upper()}")
            print(f"   Owned: {gap['owned_value']:.6f}  |  Target: {gap['target_mean']:.6f}")
            print(f"   Gap: {gap['delta_pct']:+.1f}%  |  Z-score: {gap['z_score']:.2f}σ  |  P-value: {gap['p_value']:.4f}")
            print(f"   Significance: {gap['significance'].upper()}")
            print(f"   📝 {gap['interpretation']}")

    if gap_analysis['priority_order']:
        print(f"\n" + "-"*80)
        print("🎯 OPTIMIZATION PRIORITY ORDER (Top 10)")
        print("-"*80)
        for i, metric in enumerate(gap_analysis['priority_order'][:10], 1):
            # Find the gap details
            all_gaps = gap_analysis['significant_gaps'] + gap_analysis['non_significant_gaps']
            gap = next((g for g in all_gaps if g['metric'] == metric), None)
            if gap:
                priority_score = abs(gap['z_score']) * abs(gap['delta_pct']) / 100
                print(f"{i:2d}. {metric:20s} (priority: {priority_score:.2f}, delta: {gap['delta_pct']:+6.1f}%, z: {gap['z_score']:+5.2f}σ)")

    # Non-significant gaps (for completeness)
    if gap_analysis['non_significant_gaps']:
        print(f"\n" + "-"*80)
        print("✅ NON-SIGNIFICANT GAPS (p >= 0.05) - Within Normal Range:")
        print("-"*80)
        for gap in gap_analysis['non_significant_gaps'][:5]:
            print(f"  • {gap['metric']}: {gap['delta_pct']:+.1f}% (z={gap['z_score']:.2f}, p={gap['p_value']:.3f})")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    # Calculate summary statistics
    total_gaps = len(gap_analysis['significant_gaps']) + len(gap_analysis['non_significant_gaps'])
    sig_gaps = len(gap_analysis['significant_gaps'])
    critical_gaps = len([g for g in gap_analysis['significant_gaps'] if g['significance'] == 'critical'])
    very_sig_gaps = len([g for g in gap_analysis['significant_gaps'] if g['significance'] == 'very_significant'])

    print(f"\n📊 Gap Statistics:")
    print(f"  - Total metrics analyzed: {total_gaps}")
    print(f"  - Significant gaps: {sig_gaps} ({sig_gaps/total_gaps*100:.1f}%)")
    print(f"  - Critical gaps (p<0.001): {critical_gaps}")
    print(f"  - Very significant gaps (p<0.01): {very_sig_gaps}")
    print(f"  - Sample size: {gap_analysis['sample_size']} target notes")

    # Key takeaways
    print(f"\n💡 Key Takeaways:")
    if gap_analysis['priority_order']:
        top_3_metrics = gap_analysis['priority_order'][:3]
        print(f"  - Focus optimization on: {', '.join(top_3_metrics)}")

    avg_gap_pct = sum(abs(g['delta_pct']) for g in gap_analysis['significant_gaps']) / len(gap_analysis['significant_gaps']) if gap_analysis['significant_gaps'] else 0
    print(f"  - Average significant gap: {avg_gap_pct:.1f}%")

    # Verify workflow integrity
    print("\n" + "="*80)
    print("INTEGRATION TEST VERIFICATION")
    print("="*80)

    assert len(gap_analysis['significant_gaps']) + len(gap_analysis['non_significant_gaps']) > 0, \
        "❌ No gaps identified"
    assert len(gap_analysis['priority_order']) > 0, \
        "❌ Priority order is empty"
    assert aggregated_stats.sample_size == len(target_notes), \
        "❌ Sample size mismatch"

    print("\n✅ Integration test PASSED!")
    print("✅ DataAggregatorTool → StatisticalDeltaTool workflow verified")
    print("✅ Real-world data processing successful")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    test_integration_with_real_data()
