# Implement Data Aggregation Tools

## Why

The XHS SEO optimizer system requires two critical statistical tools to enable data-driven gap analysis between high-performing `target_notes` and underperforming `owned_note`:

1. **DataAggregator Tool**: CompetitorAnalyst needs to quantify "what makes winners win" by calculating statistical aggregates (mean, median, variance, distributions) across multiple `target_notes`. Without this tool, the agent cannot identify statistically significant patterns in successful notes.

2. **StatisticalDelta Tool**: GapFinder needs to determine which gaps between `owned_note` and `target_notes` are statistically significant vs. normal variation. Without this tool, the system cannot prioritize which deficiencies truly matter and deserve optimization effort.

**Current Problem**:
- MultiModalVisionTool and NLPAnalysisTool can analyze individual notes ✅
- But we cannot aggregate insights across multiple notes ❌
- We cannot measure gap significance statistically ❌
- Agents must manually eyeball differences without quantitative rigor ❌

**Impact**:
- CompetitorAnalyst cannot generate `SuccessProfileReport` with statistical confidence
- GapFinder cannot produce `GapReport` with prioritized, significant gaps
- OptimizationStrategist receives weak signals → generates suboptimal recommendations
- End users get vague advice instead of data-driven, prioritized action items

## What Changes

### High-Level Changes

**New Tools**:
- `src/xhs_seo_optimizer/tools/data_aggregator.py` - Statistical aggregation across notes
- `src/xhs_seo_optimizer/tools/statistical_delta.py` - Gap significance testing

**New Models**:
- `AggregatedMetrics` model for structured aggregation results
- `GapAnalysis` model for statistical delta outputs

**Updated Tests**:
- `tests/test_tools/test_data_aggregator.py` - Aggregation correctness tests
- `tests/test_tools/test_statistical_delta.py` - Statistical significance tests

### DataAggregator Tool

**Purpose**: Calculate statistical summaries across multiple notes to identify "winning patterns"

**Key Capabilities**:
- Aggregate `prediction` metrics (mean, median, variance, min, max)
- Aggregate `tag` frequencies and modes (most common values)
- Filter outliers (configurable threshold, default: ±2 standard deviations)
- Group analysis by tag dimensions (e.g., avg CTR by `intention_lv2`)

**Input**: List of Note objects (typically 3-10 `target_notes`)
**Output**: `AggregatedMetrics` model with statistical summaries

**Example Output**:
```python
{
  "prediction_stats": {
    "ctr": {"mean": 0.14, "median": 0.13, "std": 0.02, "min": 0.10, "max": 0.18},
    "comment_rate": {"mean": 0.012, "median": 0.011, "std": 0.003, ...},
    "sortScore": {"mean": 0.45, "median": 0.44, ...}
  },
  "tag_frequencies": {
    "intention_lv2": {"经验分享": 7, "产品推荐": 2, "干货教程": 1},
    "note_marketing_integrated_level": {"软广": 6, "商品推荐": 3, "无": 1}
  },
  "tag_modes": {
    "intention_lv2": "经验分享",
    "taxonomy2": "婴童用品"
  },
  "sample_size": 10,
  "outliers_removed": 0
}
```

### StatisticalDelta Tool

**Purpose**: Determine if gaps between `owned_note` and `target_notes` are statistically significant

**Key Capabilities**:
- Compare single value (owned_note metric) against distribution (target_notes stats)
- Calculate z-score for each metric
- Determine significance level (p-value < 0.05 = significant gap)
- Rank gaps by both magnitude and significance
- Provide interpretation ("this gap is critical" vs "this is normal variation")

**Input**:
- `owned_note.prediction` metrics
- `AggregatedMetrics` from DataAggregator (target_notes stats)
- Optional: significance threshold (default α=0.05)

**Output**: `GapAnalysis` model with prioritized gaps

**Example Output**:
```python
{
  "significant_gaps": [
    {
      "metric": "comment_rate",
      "owned_value": 0.0013,
      "target_mean": 0.012,
      "delta_pct": -89.2,  # 89.2% lower
      "z_score": -3.67,     # 3.67 standard deviations below mean
      "p_value": 0.0001,    # Highly significant
      "significance": "critical",  # critical | significant | minor | none
      "interpretation": "owned_note comment_rate is critically lower than target_notes (3.67σ below mean)"
    },
    {
      "metric": "ctr",
      "owned_value": 0.10,
      "target_mean": 0.14,
      "delta_pct": -28.6,
      "z_score": -2.00,
      "p_value": 0.046,
      "significance": "significant",
      "interpretation": "owned_note CTR is significantly lower than target_notes (2.0σ below mean)"
    }
  ],
  "non_significant_gaps": [
    {
      "metric": "like_rate",
      "owned_value": 0.011,
      "target_mean": 0.0113,
      "delta_pct": -2.7,
      "z_score": -0.90,
      "p_value": 0.368,
      "significance": "none",
      "interpretation": "owned_note like_rate is within normal range (p=0.368)"
    }
  ],
  "priority_order": ["comment_rate", "ctr", "sortScore", ...]  # Sorted by significance
}
```

### Spec Deltas

**New Specs**:
- `data-aggregator` - Statistical aggregation requirements
- `statistical-delta` - Gap significance testing requirements

Both specs will include:
- SHALL/MUST requirements for statistical correctness
- Scenarios covering edge cases (empty lists, single note, outliers, etc.)
- Test coverage requirements (>90% for statistical logic)
