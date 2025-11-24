# Design Document: GapFinder Agent

## Overview

The GapFinder agent bridges the analysis gap between CompetitorAnalyst (what works) and OwnedNoteAuditor (current state) by identifying statistically significant performance deficits and mapping them to actionable content features.

## Architecture

### System Position

```
┌──────────────────────┐
│ CompetitorAnalyst    │──┐
│ (target_notes)       │  │
└──────────────────────┘  │
                          ├──► SuccessProfileReport ──┐
┌──────────────────────┐  │                            │
│ OwnedNoteAuditor     │──┘                            │
│ (owned_note)         │                               │
└──────────────────────┘                               │
           AuditReport ──────────────────────────────┐ │
                                                      │ │
                                            ┌─────────▼─▼──────────┐
                                            │    GapFinder         │
                                            │  (差距定位员)          │
                                            └─────────┬────────────┘
                                                      │
                                                 GapReport
                                                      │
                                            ┌─────────▼────────────┐
                                            │ OptimizationStrategist│
                                            │  (未实现)             │
                                            └──────────────────────┘
```

### Design Principles

1. **Data-Driven Analysis**: Use statistical significance (z-score, p-value) rather than subjective judgment
2. **Feature Attribution**: Map every metric gap to specific content features
3. **Prioritization**: Combine statistical significance with practical magnitude
4. **Actionable Output**: Provide clear direction for optimization

### Input/Output Contract

**Inputs:**
- `success_profile_report`: SuccessProfileReport (from CompetitorAnalyst)
  - Contains: aggregated_stats (mean/std for all metrics)
  - Contains: metric_profiles (success patterns per metric)
- `audit_report`: AuditReport (from OwnedNoteAuditor)
  - Contains: **current_metrics** (owned_note actual prediction values) - Dict[str, float]
  - Contains: text_features, visual_features (extracted content features)
  - Contains: note_id, keyword
- `keyword`: str (for context)

**Note:** AuditReport.current_metrics is extracted from owned_note.prediction by OwnedNoteAuditor during report generation.

**Output:**
- `gap_report`: GapReport
  - Contains: significant_gaps (p < 0.05), prioritized by importance
  - Contains: feature attribution (missing_features, weak_features per gap)
  - Contains: root_causes (3-5 cross-metric patterns)
  - Contains: actionable recommendations

## Task Breakdown

The GapFinder crew executes 4 sequential tasks:

### Task 1: Calculate Statistical Gaps

**Purpose:** Use rigorous statistical methods to identify significant performance deficits

**Process:**
1. Extract owned_note metrics from `current_metrics` (already flattened by @before_kickoff from audit_report.current_metrics)
2. Extract target_notes statistics from `target_means` and `target_stds` (already flattened by @before_kickoff from success_profile_report.aggregated_stats)
3. For each metric, calculate gaps:
   - Calculate z-score: `(owned_value - target_mean) / target_std`
   - Calculate p-value: two-tailed normal distribution
   - Classify significance: critical (p<0.001), very_significant (p<0.01), significant (p<0.05), marginal (p<0.10), none
4. Calculate priority score: `|z_score| * |delta_pct| / 100`
5. Sort gaps by priority score

**Output:** GapAnalysis JSON
- significant_gaps: List of gaps with p < 0.05
- marginal_gaps: List of gaps with 0.05 <= p < 0.10
- non_significant_gaps: List of gaps with p >= 0.10
- priority_order: Metrics sorted by priority score

**Why This Matters:**
- Prevents wasting effort on random variation
- Identifies gaps that are statistically unlikely to be chance
- Combines significance (confidence) with magnitude (impact)

### Task 2: Map Gaps to Features

**Purpose:** Connect abstract metric deficits to concrete content weaknesses

**Process:**
For each significant gap:

1. **Identify relevant features**
   - Look up `success_profile_report.metric_profiles[metric_name]`
   - Extract `relevant_features` list (e.g., for `comment_rate`: ["ending_technique", "engagement_hooks", "question_presence"])

2. **Compare with owned_note features**
   - Get owned_note's actual feature values from `audit_report.text_features` and `audit_report.visual_features`
   - Classify features as:
     - **missing_features**: Present in success profile but absent in owned_note
       - Example: Success profile has 85% prevalence of "open_ended_question" in ending_technique, but owned_note has None
     - **weak_features**: Present in both but poorly executed in owned_note
       - Example: Owned_note has "engagement_hooks" but score is 2/10 vs. target average 8/10

3. **Generate gap narrative**
   - `gap_explanation` (2-3 sentences): Explain WHY this gap exists
     - Cite specific feature differences
     - Reference success profile patterns
     - Example: "客户笔记的评论率显著低于竞品（低89%，p<0.001），主要因为结尾缺少开放式问题。竞品中85%使用结尾提问，而客户笔记仅用陈述句。"
   - `recommendation_summary` (1-2 sentences): WHAT to improve
     - Specific, actionable suggestion
     - Example: "在笔记结尾添加开放式问题（如'你家宝宝有这种情况吗？'），降低评论门槛。"

**Output:** Enriched gap objects with feature attribution

**Why This Matters:**
- Moves from "CTR is low" (abstract) to "title lacks emotional hook" (concrete)
- Provides clear direction for OptimizationStrategist
- Evidence-based recommendations

### Task 3: Prioritize Gaps

**Purpose:** Focus effort on highest-impact improvements

**Process:**

1. **Assign priority ranks**
   - Use `priority_order` from Task 1
   - Assign priority_rank: 1 (highest), 2, 3, ...
   - Add to each MetricGap object

2. **Identify top priority metrics**
   - Select top 3 metrics from priority_order
   - These become `top_priority_metrics` list

3. **Extract root causes**
   - Aggregate `missing_features` and `weak_features` across all significant gaps
   - Count frequency of each feature issue
   - Select top 3-5 most frequent issues
   - Describe in Chinese, user-friendly language
   - Example root causes:
     - "标题缺乏情感钩子和疑问句" (affects CTR)
     - "结尾缺少开放式问题引导评论" (affects comment_rate)
     - "封面视觉冲击力不足" (affects CTR, interaction_rate)
     - "内容缺少对比钩子和数据支撑" (affects credibility)

4. **Generate impact summary**
   - 100-200 character narrative
   - Highlight 1-2 most critical gaps
   - Mention overall impact on performance
   - Example: "客户笔记在comment_rate和ctr上显著落后竞品（分别低89%和43%），导致sort_score2排名靠后。主要问题是标题缺乏吸引力、结尾无互动引导、封面视觉冲击力弱。"

**Output:**
- top_priority_metrics: List[str] (max 3)
- root_causes: List[str] (3-5 items)
- impact_summary: str (100-200 chars)

**Why This Matters:**
- Users can't fix everything at once - need focus
- Root causes identify systemic issues vs. one-off problems
- Impact summary provides executive overview

### Task 4: Generate Gap Report

**Purpose:** Package all analysis into structured, machine-readable format

**Process:**

1. Collect all data from previous tasks
2. Construct MetricGap objects for each gap (with all fields)
3. Construct GapReport object:
   - Add metadata: keyword, owned_note_id, sample_size, gap_timestamp (ISO 8601)
   - Add gap lists: significant_gaps, marginal_gaps, non_significant_gaps
   - Add synthesis: top_priority_metrics, root_causes, impact_summary
4. Validate against Pydantic schema
5. Serialize to JSON
6. Return as output_pydantic

**Output:** GapReport (JSON)

**Why This Matters:**
- Structured output enables programmatic consumption by OptimizationStrategist
- Pydantic validation ensures data quality
- JSON format enables storage and future analysis

## Data Flow

```
@before_kickoff: Flatten Reports
  │
  ├─► Extract from audit_report.current_metrics
  │   → current_metrics["ctr"], current_metrics["comment_rate"], etc.
  │
  ├─► Extract from success_profile_report.aggregated_stats
  │   → target_mean_ctr, target_std_ctr, etc.
  │
  └─► Store in shared_context + inputs for YAML variable substitution

Task 1: Calculate Statistical Gaps
  │
  ├──► Use flattened data from inputs
  │      Input: current_metrics (Dict[str, float])
  │             target_means (flattened: target_mean_ctr, etc.)
  │             target_stds (flattened: target_std_ctr, etc.)
  │      Process: For each metric, calculate z_score, p_value
  │      Output: z_score, p_value, significance, priority_order
  │
  └──► Task 2: Feature Attribution
         Input: success_profile_report.metric_profiles[metric].relevant_features
                audit_report.text_features + visual_features
         Process: Compare and classify (missing vs. weak)
         Output: related_features, missing_features, weak_features
                 gap_explanation, recommendation_summary
         │
         └──► Task 3: Prioritization
                Input: priority_order from Task 1
                       feature lists from Task 2
                Process: Rank gaps, aggregate root causes
                Output: priority_rank, top_priority_metrics, root_causes
                │
                └──► Task 4: Report Generation
                       Input: All above + metadata
                       Process: Construct GapReport Pydantic model
                       Output: outputs/gap_report.json
```

## Feature Attribution Logic

### Metric → Feature Mapping

Based on project.md and success-profile-report structure:

| Metric | Primary Features | Secondary Features |
|--------|------------------|-------------------|
| **ctr** | title_pattern, thumbnail_appeal, image_style | cover_color_scheme, text_on_image |
| **comment_rate** | ending_technique, engagement_hooks, question_presence | cta_presence, relatability |
| **interaction_rate** | content_framework, visual_storytelling, credibility_signals | emotional_appeal |
| **like_rate** | visual_appeal, content_quality | authenticity |
| **fav_rate** | practical_value, information_density | visual_organization |
| **share_rate** | viral_potential, emotional_resonance | content_uniqueness |
| **sort_score2** | Composite (all above) | - |

### Comparison Logic

For each relevant feature:

1. **Check presence**
   - Does owned_note have this feature?
   - Source: `audit_report.text_features` or `visual_features`

2. **Check execution quality**
   - If present, how does it compare to success profile?
   - Use feature scores/values from audit_report
   - Compare to success profile patterns

3. **Classify**
   - **Missing**: Feature not present in owned_note but high prevalence in success profile (>70%)
   - **Weak**: Feature present but score < threshold or execution differs from success pattern
   - **Adequate**: Feature present and execution similar to success profile (not mentioned in gap)

### Example: comment_rate Gap

```
Metric: comment_rate
Owned value: 0.0013
Target mean: 0.012
Delta: -89.2%, z=-3.67, p<0.001 (critical)

Relevant features (from success_profile_report.metric_profiles["comment_rate"]):
- ending_technique
- engagement_hooks
- question_presence

Owned note features (from audit_report.text_features):
- ending_technique: "statement" (陈述句)
- engagement_hooks: score 3/10
- question_presence: False

Success profile patterns (from metric_profiles["comment_rate"].feature_analyses):
- ending_technique: 85% use "open_ended_question", prevalence_pct=85%
- engagement_hooks: Average score 8/10
- question_presence: 80% have at least one question in content

Classification:
- missing_features: ["open_ended_question"]  # 85% prevalence, but owned_note has None
- weak_features: ["engagement_hooks"]  # Present but score 3/10 vs. 8/10

gap_explanation:
"客户笔记的评论率显著低于竞品（低89%，z=-3.67σ，p<0.001），主要因为结尾缺少开放式问题引导互动。
竞品中85%使用结尾提问，而客户笔记仅使用陈述句。此外，内容中的互动钩子较弱（3/10 vs. 竞品8/10）。"

recommendation_summary:
"在笔记结尾添加开放式问题（如'你家宝宝有这种情况吗？'），并在正文中增加2-3个互动引导点。"
```

## Error Handling

### Input Validation
- Missing required fields → ValueError with clear message
- Invalid JSON → JSONDecodeError with guidance
- Mismatched data types → Pydantic ValidationError

### Statistical Edge Cases
- Zero standard deviation → StatisticalDeltaTool handles (significance="undefined")
- Missing metrics in owned_note → Skip with warning log
- All gaps non-significant → Still generate report, note in impact_summary

### LLM Failures
- Feature mapping task fails → Retry with simplified prompt
- Gap explanation too short (<20 chars) → Regenerate
- Root causes <3 or >5 → Adjust extraction logic

## Performance Considerations

### Task Execution Time
- Task 1 (Statistical): ~5-10 seconds (tool call)
- Task 2 (Feature Mapping): ~30-60 seconds (LLM reasoning for each gap)
- Task 3 (Prioritization): ~15-30 seconds (LLM synthesis)
- Task 4 (Report Generation): ~5-10 seconds (JSON serialization)
- **Total: ~1-2 minutes**

### Optimization Opportunities
- Task 2 could batch multiple gaps into single LLM call
- Caching success_profile_report for repeated analysis
- Pre-compute feature attribution rules

## Testing Strategy

### Unit Tests
- GapReport Pydantic model validation
- MetricGap field constraints
- Crew initialization

### Integration Tests
- End-to-end with real reports
- StatisticalDeltaTool integration
- File output verification

### Validation Tests
- Feature attribution correctness (manual review)
- Priority ordering makes sense
- Root causes are non-redundant
- Impact summary is coherent

### Edge Case Tests
- No significant gaps (all p >= 0.05)
- Many significant gaps (>10)
- Missing metrics in owned_note
- Empty success profile (edge case)

## Future Enhancements

1. **Visualization**: Generate charts showing gaps visually
2. **Historical Tracking**: Compare gap reports over time
3. **Batch Analysis**: Analyze multiple owned_notes at once
4. **Confidence Intervals**: Add confidence intervals to gap estimates
5. **Sensitivity Analysis**: Test how robust priorities are to outliers

## References

- project.md: Agent architecture and workflow
- StatisticalDeltaTool: Tool implementation for gap calculation
- SuccessProfileReport spec: Input contract
- AuditReport spec: Input contract
- Change 0003: CompetitorAnalyst implementation
- Change 0004: OwnedNoteAuditor implementation
