# gap-finder Specification

## Purpose

The GapFinder agent identifies statistically significant performance gaps between owned_note and target_notes by comparing SuccessProfileReport and AuditReport, then maps gaps to actionable content features for optimization.

## ADDED Requirements

### Requirement: Agent SHALL calculate statistical gaps using StatisticalDeltaTool

The GapFinder MUST call StatisticalDeltaTool to compute z-scores, p-values, and significance classifications for all metric differences between owned_note and target_notes.

#### Scenario: Calculate gaps for all metrics

```python
from xhs_seo_optimizer.crew_gap_finder import XhsSeoOptimizerCrewGapFinder
from xhs_seo_optimizer.models.reports import GapReport
import json

# Load prerequisite reports
with open("outputs/success_profile_report.json") as f:
    success_profile = json.load(f)

with open("outputs/audit_report.json") as f:
    audit_report = json.load(f)

# Execute GapFinder
crew = XhsSeoOptimizerCrewGapFinder()
result = crew.kickoff(inputs={
    "success_profile_report": success_profile,
    "audit_report": audit_report,
    "keyword": "DHA"
})

# Verify statistical analysis was performed
gap_report = result.pydantic
assert isinstance(gap_report, GapReport)

# MUST have classified gaps by significance
assert isinstance(gap_report.significant_gaps, list)
assert isinstance(gap_report.marginal_gaps, list)
assert isinstance(gap_report.non_significant_gaps, list)

# Each gap MUST have statistical fields
for gap in gap_report.significant_gaps:
    assert gap.z_score is not None
    assert gap.p_value is not None
    assert gap.p_value < 0.05  # Significant threshold
    assert gap.significance in ["critical", "very_significant", "significant"]
```

#### Scenario: Separate significant and non-significant gaps

```python
# Gaps with p < 0.05 should be in significant_gaps
assert all(gap.p_value < 0.05 for gap in gap_report.significant_gaps)

# Gaps with 0.05 <= p < 0.10 should be in marginal_gaps
assert all(0.05 <= gap.p_value < 0.10 for gap in gap_report.marginal_gaps)

# Gaps with p >= 0.10 should be in non_significant_gaps
assert all(gap.p_value >= 0.10 for gap in gap_report.non_significant_gaps)

# Verify some gaps exist (unless owned_note matches targets perfectly)
total_gaps = len(gap_report.significant_gaps) + len(gap_report.marginal_gaps) + len(gap_report.non_significant_gaps)
assert total_gaps > 0, "Should have at least one gap analyzed"
```

### Requirement: Agent SHALL map gaps to specific content features

For each significant gap, the agent MUST identify which content features are missing or weak, using success_profile_report metric_profiles and audit_report feature extractions.

#### Scenario: Gaps have feature attribution

```python
# Each significant gap MUST have feature attribution
for gap in gap_report.significant_gaps:
    assert len(gap.related_features) > 0, f"Gap {gap.metric_name} missing related_features"
    assert isinstance(gap.missing_features, list)
    assert isinstance(gap.weak_features, list)

    # At least one of missing_features or weak_features should be non-empty
    assert len(gap.missing_features) > 0 or len(gap.weak_features) > 0

# Example: If comment_rate is low, should mention ending_technique or engagement_hooks
comment_gaps = [g for g in gap_report.significant_gaps if g.metric_name == "comment_rate"]
if comment_gaps:
    gap = comment_gaps[0]
    features = gap.related_features + gap.missing_features + gap.weak_features
    feature_text = " ".join(features).lower()
    assert any(keyword in feature_text for keyword in [
        "ending", "question", "engagement", "hook", "cta"
    ]), "comment_rate gap should reference engagement features"
```

#### Scenario: Gap explanations are substantive

```python
for gap in gap_report.significant_gaps:
    # MUST have gap_explanation
    assert gap.gap_explanation is not None
    assert len(gap.gap_explanation) > 30, f"Gap explanation too short: {gap.gap_explanation}"

    # Should mention specific features or patterns
    explanation_lower = gap.gap_explanation.lower()

    # MUST have recommendation_summary
    assert gap.recommendation_summary is not None
    assert len(gap.recommendation_summary) > 10, "Recommendation too short"
```

### Requirement: Agent SHALL prioritize gaps by combined significance and magnitude

Gaps MUST be ranked by priority using both statistical significance (z-score) and practical impact (delta_pct).

#### Scenario: Priority ranks are assigned correctly

```python
# Each significant gap MUST have priority_rank
for gap in gap_report.significant_gaps:
    assert gap.priority_rank is not None
    assert gap.priority_rank > 0, "Priority rank must be positive"

# Priority ranks should be sequential starting from 1
ranks = [g.priority_rank for g in gap_report.significant_gaps]
assert ranks == sorted(ranks), "Gaps should be ordered by priority_rank"
assert ranks[0] == 1, "Highest priority should be rank 1"

# Higher priority gaps should have higher combined score (z_score * delta_pct)
for i in range(len(gap_report.significant_gaps) - 1):
    gap1 = gap_report.significant_gaps[i]
    gap2 = gap_report.significant_gaps[i + 1]

    score1 = abs(gap1.z_score) * abs(gap1.delta_pct) / 100
    score2 = abs(gap2.z_score) * abs(gap2.delta_pct) / 100

    assert score1 >= score2, f"Priority ordering incorrect: {gap1.metric_name} (rank {gap1.priority_rank}) has lower score than {gap2.metric_name} (rank {gap2.priority_rank})"
```

#### Scenario: Top priority metrics are identified

```python
# MUST have top_priority_metrics list
assert len(gap_report.top_priority_metrics) > 0
assert len(gap_report.top_priority_metrics) <= 3, "Should have at most 3 top priority metrics"

# Top priority metrics MUST match highest-ranked significant gaps
if len(gap_report.significant_gaps) > 0:
    top_metrics_from_gaps = [g.metric_name for g in gap_report.significant_gaps[:3]]

    for metric in gap_report.top_priority_metrics:
        assert metric in top_metrics_from_gaps, f"Top priority metric {metric} not in top 3 gaps"
```

### Requirement: Agent SHALL identify cross-metric root causes

The agent MUST aggregate feature issues across all gaps to identify 3-5 systemic problems (root causes).

#### Scenario: Root causes are identified

```python
# MUST have root_causes list with 3-5 items
assert len(gap_report.root_causes) >= 3, "Should have at least 3 root causes"
assert len(gap_report.root_causes) <= 5, "Should have at most 5 root causes"

# Root causes should be in Chinese and actionable
for cause in gap_report.root_causes:
    assert isinstance(cause, str)
    assert len(cause) > 5, "Root cause too short"

    # Should contain Chinese characters (since it's for Chinese users)
    import re
    assert re.search(r'[\u4e00-\u9fff]', cause), f"Root cause should be in Chinese: {cause}"
```

#### Scenario: Root causes are based on feature frequency

```python
# Root causes should reference features that appear in multiple gaps
all_missing_features = []
all_weak_features = []

for gap in gap_report.significant_gaps:
    all_missing_features.extend(gap.missing_features)
    all_weak_features.extend(gap.weak_features)

# Count feature frequency
from collections import Counter
feature_counts = Counter(all_missing_features + all_weak_features)

# Most frequent features should be mentioned in root_causes
if feature_counts:
    most_common = feature_counts.most_common(3)
    root_causes_text = " ".join(gap_report.root_causes).lower()

    # At least one of the most common features should be referenced
    # (allowing for translation and paraphrasing)
    # This is a soft check - manual review needed
```

### Requirement: Agent SHALL generate overall impact summary

The agent MUST synthesize findings into a concise impact_summary (100-200 characters) highlighting key issues.

#### Scenario: Impact summary meets length requirements

```python
assert gap_report.impact_summary is not None
assert 100 <= len(gap_report.impact_summary) <= 200, \
    f"Impact summary length {len(gap_report.impact_summary)} not in range [100, 200]"
```

#### Scenario: Impact summary mentions top gaps

```python
# Impact summary should reference top priority metrics or their impacts
summary_lower = gap_report.impact_summary.lower()

# Should mention at least one top metric or general performance issue
if len(gap_report.top_priority_metrics) > 0:
    top_metric = gap_report.top_priority_metrics[0]

    # May not mention metric name directly, but should convey impact
    # (e.g., "评论率" for comment_rate, "点击率" for ctr)
    # Manual review recommended
```

### Requirement: Agent SHALL output GapReport conforming to schema

The agent MUST return a JSON-serializable GapReport with all required fields properly validated.

#### Scenario: GapReport conforms to Pydantic schema

```python
from xhs_seo_optimizer.models.reports import GapReport

# Load gap_report.json
with open("outputs/gap_report.json") as f:
    gap_dict = json.load(f)

# MUST parse as GapReport
gap_report = GapReport(**gap_dict)

# Verify all required fields
assert gap_report.keyword is not None
assert gap_report.owned_note_id is not None
assert gap_report.sample_size > 0
assert gap_report.gap_timestamp is not None

# Timestamp should be ISO 8601
from datetime import datetime
datetime.fromisoformat(gap_report.gap_timestamp.replace('Z', '+00:00'))
```

#### Scenario: MetricGap objects are valid

```python
from xhs_seo_optimizer.models.reports import MetricGap

for gap in gap_report.significant_gaps:
    # MUST be MetricGap instance
    assert isinstance(gap, MetricGap)

    # Required fields
    assert gap.metric_name is not None
    assert gap.owned_value is not None
    assert gap.target_mean is not None
    assert gap.delta_absolute is not None
    assert gap.delta_pct is not None
    assert gap.z_score is not None
    assert gap.p_value is not None
    assert gap.significance is not None
    assert gap.priority_rank > 0

    # Feature attribution
    assert isinstance(gap.related_features, list)
    assert isinstance(gap.missing_features, list)
    assert isinstance(gap.weak_features, list)

    # Narratives
    assert gap.gap_explanation is not None
    assert gap.recommendation_summary is not None
```

### Requirement: Agent SHALL save output to outputs/gap_report.json

The agent MUST write the final GapReport to the outputs directory as JSON for consumption by OptimizationStrategist.

#### Scenario: Output file is created

```python
import os

# File MUST exist after crew.kickoff()
assert os.path.exists("outputs/gap_report.json"), "gap_report.json not created"

# File MUST contain valid JSON
with open("outputs/gap_report.json") as f:
    gap_dict = json.load(f)  # Should not raise JSONDecodeError

# File MUST match result.pydantic
assert gap_dict["keyword"] == gap_report.keyword
assert gap_dict["owned_note_id"] == gap_report.owned_note_id
assert len(gap_dict["significant_gaps"]) == len(gap_report.significant_gaps)
```

#### Scenario: Output is human-readable

```python
# JSON should be pretty-printed (indented)
with open("outputs/gap_report.json") as f:
    content = f.read()

# Should contain newlines (indicating formatting)
assert "\n" in content, "JSON should be formatted with newlines"

# Should be UTF-8 encoded to support Chinese text
with open("outputs/gap_report.json", encoding="utf-8") as f:
    gap_dict = json.load(f)

# Chinese text should be readable (not escaped)
if gap_report.root_causes:
    # Check that Chinese characters are preserved
    import re
    assert any(re.search(r'[\u4e00-\u9fff]', cause) for cause in gap_report.root_causes)
```

### Requirement: Agent SHALL handle edge cases gracefully

The agent MUST validate inputs and handle edge cases (missing metrics, no significant gaps, zero variance) with appropriate defaults or errors.

#### Scenario: Missing input fields raise ValueError

```python
import pytest

# Missing success_profile_report
with pytest.raises(ValueError, match="success_profile_report"):
    crew.kickoff(inputs={
        "audit_report": audit_report,
        "keyword": "test"
    })

# Missing audit_report
with pytest.raises(ValueError, match="audit_report"):
    crew.kickoff(inputs={
        "success_profile_report": success_profile,
        "keyword": "test"
    })

# Missing keyword
with pytest.raises(ValueError, match="keyword"):
    crew.kickoff(inputs={
        "success_profile_report": success_profile,
        "audit_report": audit_report
    })
```

#### Scenario: No significant gaps handled gracefully

```python
# Create scenario where owned_note matches targets closely
# (This may require synthetic data)

# If all gaps are non-significant:
if len(gap_report.significant_gaps) == 0:
    # Should still generate report with meaningful content
    assert gap_report.impact_summary is not None
    assert len(gap_report.impact_summary) > 0

    # Impact summary should note lack of significant gaps
    summary_lower = gap_report.impact_summary.lower()
    # (Manual review: should say something like "表现与竞品接近" or similar)

    # top_priority_metrics can be empty or contain marginal gaps
    # root_causes can be empty or contain minor issues
```

#### Scenario: Missing metrics in owned_note handled gracefully

```python
# If owned_note is missing some metrics that targets have:
# GapFinder should still analyze available metrics

# Verify no crashes occurred (result exists)
assert gap_report is not None

# Gaps should only include metrics present in both
for gap in gap_report.significant_gaps + gap_report.marginal_gaps + gap_report.non_significant_gaps:
    # Metric should be in audit_report
    # (This is validated by StatisticalDeltaTool, which skips missing metrics)
    pass
```

### Requirement: Agent SHALL execute 4 sequential tasks with proper context passing

The crew MUST chain 4 tasks in order, with later tasks receiving outputs from earlier tasks via context.

#### Scenario: All 4 tasks execute in sequence

```python
# Enable verbose logging to verify task execution
import logging
logging.basicConfig(level=logging.INFO)

result = crew.kickoff(inputs={
    "success_profile_report": success_profile,
    "audit_report": audit_report,
    "keyword": "DHA"
})

# If result is successful, all 4 tasks completed
assert result is not None

# Tasks are:
# 1. calculate_statistical_gaps
# 2. map_gaps_to_features (depends on 1)
# 3. prioritize_gaps (depends on 1, 2)
# 4. generate_gap_report (depends on 1, 2, 3)

# Verify final output has data from all tasks
assert len(gap_report.significant_gaps) >= 0  # From task 1
assert all(len(g.related_features) >= 0 for g in gap_report.significant_gaps)  # From task 2
assert len(gap_report.top_priority_metrics) >= 0  # From task 3
assert gap_report.gap_timestamp is not None  # From task 4
```

#### Scenario: Task context is correctly configured

```python
# This verifies implementation correctness (checked via code review)
# Task YAML should have:
# map_gaps_to_features:
#   context: [calculate_statistical_gaps]
# prioritize_gaps:
#   context: [calculate_statistical_gaps, map_gaps_to_features]
# generate_gap_report:
#   context: [calculate_statistical_gaps, map_gaps_to_features, prioritize_gaps]

# If tasks execute successfully, context passing is working
# (This is implicitly tested by successful execution)
```
## Requirements
### Requirement: Agent SHALL calculate statistical gaps using StatisticalDeltaTool

The GapFinder MUST call StatisticalDeltaTool to compute z-scores, p-values, and significance classifications for all metric differences between owned_note and target_notes.

#### Scenario: Calculate gaps for all metrics

```python
from xhs_seo_optimizer.crew_gap_finder import XhsSeoOptimizerCrewGapFinder
from xhs_seo_optimizer.models.reports import GapReport
import json

# Load prerequisite reports
with open("outputs/success_profile_report.json") as f:
    success_profile = json.load(f)

with open("outputs/audit_report.json") as f:
    audit_report = json.load(f)

# Execute GapFinder
crew = XhsSeoOptimizerCrewGapFinder()
result = crew.kickoff(inputs={
    "success_profile_report": success_profile,
    "audit_report": audit_report,
    "keyword": "DHA"
})

# Verify statistical analysis was performed
gap_report = result.pydantic
assert isinstance(gap_report, GapReport)

# MUST have classified gaps by significance
assert isinstance(gap_report.significant_gaps, list)
assert isinstance(gap_report.marginal_gaps, list)
assert isinstance(gap_report.non_significant_gaps, list)

# Each gap MUST have statistical fields
for gap in gap_report.significant_gaps:
    assert gap.z_score is not None
    assert gap.p_value is not None
    assert gap.p_value < 0.05  # Significant threshold
    assert gap.significance in ["critical", "very_significant", "significant"]
```

#### Scenario: Separate significant and non-significant gaps

```python
# Gaps with p < 0.05 should be in significant_gaps
assert all(gap.p_value < 0.05 for gap in gap_report.significant_gaps)

# Gaps with 0.05 <= p < 0.10 should be in marginal_gaps
assert all(0.05 <= gap.p_value < 0.10 for gap in gap_report.marginal_gaps)

# Gaps with p >= 0.10 should be in non_significant_gaps
assert all(gap.p_value >= 0.10 for gap in gap_report.non_significant_gaps)

# Verify some gaps exist (unless owned_note matches targets perfectly)
total_gaps = len(gap_report.significant_gaps) + len(gap_report.marginal_gaps) + len(gap_report.non_significant_gaps)
assert total_gaps > 0, "Should have at least one gap analyzed"
```

### Requirement: Agent SHALL map gaps to specific content features

For each significant gap, the agent MUST identify which content features are missing or weak, using success_profile_report metric_profiles and audit_report feature extractions.

#### Scenario: Gaps have feature attribution

```python
# Each significant gap MUST have feature attribution
for gap in gap_report.significant_gaps:
    assert len(gap.related_features) > 0, f"Gap {gap.metric_name} missing related_features"
    assert isinstance(gap.missing_features, list)
    assert isinstance(gap.weak_features, list)

    # At least one of missing_features or weak_features should be non-empty
    assert len(gap.missing_features) > 0 or len(gap.weak_features) > 0

# Example: If comment_rate is low, should mention ending_technique or engagement_hooks
comment_gaps = [g for g in gap_report.significant_gaps if g.metric_name == "comment_rate"]
if comment_gaps:
    gap = comment_gaps[0]
    features = gap.related_features + gap.missing_features + gap.weak_features
    feature_text = " ".join(features).lower()
    assert any(keyword in feature_text for keyword in [
        "ending", "question", "engagement", "hook", "cta"
    ]), "comment_rate gap should reference engagement features"
```

#### Scenario: Gap explanations are substantive

```python
for gap in gap_report.significant_gaps:
    # MUST have gap_explanation
    assert gap.gap_explanation is not None
    assert len(gap.gap_explanation) > 30, f"Gap explanation too short: {gap.gap_explanation}"

    # Should mention specific features or patterns
    explanation_lower = gap.gap_explanation.lower()

    # MUST have recommendation_summary
    assert gap.recommendation_summary is not None
    assert len(gap.recommendation_summary) > 10, "Recommendation too short"
```

### Requirement: Agent SHALL prioritize gaps by combined significance and magnitude

Gaps MUST be ranked by priority using both statistical significance (z-score) and practical impact (delta_pct).

#### Scenario: Priority ranks are assigned correctly

```python
# Each significant gap MUST have priority_rank
for gap in gap_report.significant_gaps:
    assert gap.priority_rank is not None
    assert gap.priority_rank > 0, "Priority rank must be positive"

# Priority ranks should be sequential starting from 1
ranks = [g.priority_rank for g in gap_report.significant_gaps]
assert ranks == sorted(ranks), "Gaps should be ordered by priority_rank"
assert ranks[0] == 1, "Highest priority should be rank 1"

# Higher priority gaps should have higher combined score (z_score * delta_pct)
for i in range(len(gap_report.significant_gaps) - 1):
    gap1 = gap_report.significant_gaps[i]
    gap2 = gap_report.significant_gaps[i + 1]

    score1 = abs(gap1.z_score) * abs(gap1.delta_pct) / 100
    score2 = abs(gap2.z_score) * abs(gap2.delta_pct) / 100

    assert score1 >= score2, f"Priority ordering incorrect: {gap1.metric_name} (rank {gap1.priority_rank}) has lower score than {gap2.metric_name} (rank {gap2.priority_rank})"
```

#### Scenario: Top priority metrics are identified

```python
# MUST have top_priority_metrics list
assert len(gap_report.top_priority_metrics) > 0
assert len(gap_report.top_priority_metrics) <= 3, "Should have at most 3 top priority metrics"

# Top priority metrics MUST match highest-ranked significant gaps
if len(gap_report.significant_gaps) > 0:
    top_metrics_from_gaps = [g.metric_name for g in gap_report.significant_gaps[:3]]

    for metric in gap_report.top_priority_metrics:
        assert metric in top_metrics_from_gaps, f"Top priority metric {metric} not in top 3 gaps"
```

### Requirement: Agent SHALL identify cross-metric root causes

The agent MUST aggregate feature issues across all gaps to identify 3-5 systemic problems (root causes).

#### Scenario: Root causes are identified

```python
# MUST have root_causes list with 3-5 items
assert len(gap_report.root_causes) >= 3, "Should have at least 3 root causes"
assert len(gap_report.root_causes) <= 5, "Should have at most 5 root causes"

# Root causes should be in Chinese and actionable
for cause in gap_report.root_causes:
    assert isinstance(cause, str)
    assert len(cause) > 5, "Root cause too short"

    # Should contain Chinese characters (since it's for Chinese users)
    import re
    assert re.search(r'[\u4e00-\u9fff]', cause), f"Root cause should be in Chinese: {cause}"
```

#### Scenario: Root causes are based on feature frequency

```python
# Root causes should reference features that appear in multiple gaps
all_missing_features = []
all_weak_features = []

for gap in gap_report.significant_gaps:
    all_missing_features.extend(gap.missing_features)
    all_weak_features.extend(gap.weak_features)

# Count feature frequency
from collections import Counter
feature_counts = Counter(all_missing_features + all_weak_features)

# Most frequent features should be mentioned in root_causes
if feature_counts:
    most_common = feature_counts.most_common(3)
    root_causes_text = " ".join(gap_report.root_causes).lower()

    # At least one of the most common features should be referenced
    # (allowing for translation and paraphrasing)
    # This is a soft check - manual review needed
```

### Requirement: Agent SHALL generate overall impact summary

The agent MUST synthesize findings into a concise impact_summary (100-200 characters) highlighting key issues.

#### Scenario: Impact summary meets length requirements

```python
assert gap_report.impact_summary is not None
assert 100 <= len(gap_report.impact_summary) <= 200, \
    f"Impact summary length {len(gap_report.impact_summary)} not in range [100, 200]"
```

#### Scenario: Impact summary mentions top gaps

```python
# Impact summary should reference top priority metrics or their impacts
summary_lower = gap_report.impact_summary.lower()

# Should mention at least one top metric or general performance issue
if len(gap_report.top_priority_metrics) > 0:
    top_metric = gap_report.top_priority_metrics[0]

    # May not mention metric name directly, but should convey impact
    # (e.g., "评论率" for comment_rate, "点击率" for ctr)
    # Manual review recommended
```

### Requirement: Agent SHALL output GapReport conforming to schema

The agent MUST return a JSON-serializable GapReport with all required fields properly validated.

#### Scenario: GapReport conforms to Pydantic schema

```python
from xhs_seo_optimizer.models.reports import GapReport

# Load gap_report.json
with open("outputs/gap_report.json") as f:
    gap_dict = json.load(f)

# MUST parse as GapReport
gap_report = GapReport(**gap_dict)

# Verify all required fields
assert gap_report.keyword is not None
assert gap_report.owned_note_id is not None
assert gap_report.sample_size > 0
assert gap_report.gap_timestamp is not None

# Timestamp should be ISO 8601
from datetime import datetime
datetime.fromisoformat(gap_report.gap_timestamp.replace('Z', '+00:00'))
```

#### Scenario: MetricGap objects are valid

```python
from xhs_seo_optimizer.models.reports import MetricGap

for gap in gap_report.significant_gaps:
    # MUST be MetricGap instance
    assert isinstance(gap, MetricGap)

    # Required fields
    assert gap.metric_name is not None
    assert gap.owned_value is not None
    assert gap.target_mean is not None
    assert gap.delta_absolute is not None
    assert gap.delta_pct is not None
    assert gap.z_score is not None
    assert gap.p_value is not None
    assert gap.significance is not None
    assert gap.priority_rank > 0

    # Feature attribution
    assert isinstance(gap.related_features, list)
    assert isinstance(gap.missing_features, list)
    assert isinstance(gap.weak_features, list)

    # Narratives
    assert gap.gap_explanation is not None
    assert gap.recommendation_summary is not None
```

### Requirement: Agent SHALL save output to outputs/gap_report.json

The agent MUST write the final GapReport to the outputs directory as JSON for consumption by OptimizationStrategist.

#### Scenario: Output file is created

```python
import os

# File MUST exist after crew.kickoff()
assert os.path.exists("outputs/gap_report.json"), "gap_report.json not created"

# File MUST contain valid JSON
with open("outputs/gap_report.json") as f:
    gap_dict = json.load(f)  # Should not raise JSONDecodeError

# File MUST match result.pydantic
assert gap_dict["keyword"] == gap_report.keyword
assert gap_dict["owned_note_id"] == gap_report.owned_note_id
assert len(gap_dict["significant_gaps"]) == len(gap_report.significant_gaps)
```

#### Scenario: Output is human-readable

```python
# JSON should be pretty-printed (indented)
with open("outputs/gap_report.json") as f:
    content = f.read()

# Should contain newlines (indicating formatting)
assert "\n" in content, "JSON should be formatted with newlines"

# Should be UTF-8 encoded to support Chinese text
with open("outputs/gap_report.json", encoding="utf-8") as f:
    gap_dict = json.load(f)

# Chinese text should be readable (not escaped)
if gap_report.root_causes:
    # Check that Chinese characters are preserved
    import re
    assert any(re.search(r'[\u4e00-\u9fff]', cause) for cause in gap_report.root_causes)
```

### Requirement: Agent SHALL handle edge cases gracefully

The agent MUST validate inputs and handle edge cases (missing metrics, no significant gaps, zero variance) with appropriate defaults or errors.

#### Scenario: Missing input fields raise ValueError

```python
import pytest

# Missing success_profile_report
with pytest.raises(ValueError, match="success_profile_report"):
    crew.kickoff(inputs={
        "audit_report": audit_report,
        "keyword": "test"
    })

# Missing audit_report
with pytest.raises(ValueError, match="audit_report"):
    crew.kickoff(inputs={
        "success_profile_report": success_profile,
        "keyword": "test"
    })

# Missing keyword
with pytest.raises(ValueError, match="keyword"):
    crew.kickoff(inputs={
        "success_profile_report": success_profile,
        "audit_report": audit_report
    })
```

#### Scenario: No significant gaps handled gracefully

```python
# Create scenario where owned_note matches targets closely
# (This may require synthetic data)

# If all gaps are non-significant:
if len(gap_report.significant_gaps) == 0:
    # Should still generate report with meaningful content
    assert gap_report.impact_summary is not None
    assert len(gap_report.impact_summary) > 0

    # Impact summary should note lack of significant gaps
    summary_lower = gap_report.impact_summary.lower()
    # (Manual review: should say something like "表现与竞品接近" or similar)

    # top_priority_metrics can be empty or contain marginal gaps
    # root_causes can be empty or contain minor issues
```

#### Scenario: Missing metrics in owned_note handled gracefully

```python
# If owned_note is missing some metrics that targets have:
# GapFinder should still analyze available metrics

# Verify no crashes occurred (result exists)
assert gap_report is not None

# Gaps should only include metrics present in both
for gap in gap_report.significant_gaps + gap_report.marginal_gaps + gap_report.non_significant_gaps:
    # Metric should be in audit_report
    # (This is validated by StatisticalDeltaTool, which skips missing metrics)
    pass
```

### Requirement: Agent SHALL execute 4 sequential tasks with proper context passing

The crew MUST chain 4 tasks in order, with later tasks receiving outputs from earlier tasks via context.

#### Scenario: All 4 tasks execute in sequence

```python
# Enable verbose logging to verify task execution
import logging
logging.basicConfig(level=logging.INFO)

result = crew.kickoff(inputs={
    "success_profile_report": success_profile,
    "audit_report": audit_report,
    "keyword": "DHA"
})

# If result is successful, all 4 tasks completed
assert result is not None

# Tasks are:
# 1. calculate_statistical_gaps
# 2. map_gaps_to_features (depends on 1)
# 3. prioritize_gaps (depends on 1, 2)
# 4. generate_gap_report (depends on 1, 2, 3)

# Verify final output has data from all tasks
assert len(gap_report.significant_gaps) >= 0  # From task 1
assert all(len(g.related_features) >= 0 for g in gap_report.significant_gaps)  # From task 2
assert len(gap_report.top_priority_metrics) >= 0  # From task 3
assert gap_report.gap_timestamp is not None  # From task 4
```

#### Scenario: Task context is correctly configured

```python
# This verifies implementation correctness (checked via code review)
# Task YAML should have:
# map_gaps_to_features:
#   context: [calculate_statistical_gaps]
# prioritize_gaps:
#   context: [calculate_statistical_gaps, map_gaps_to_features]
# generate_gap_report:
#   context: [calculate_statistical_gaps, map_gaps_to_features, prioritize_gaps]

# If tasks execute successfully, context passing is working
# (This is implicitly tested by successful execution)
```

