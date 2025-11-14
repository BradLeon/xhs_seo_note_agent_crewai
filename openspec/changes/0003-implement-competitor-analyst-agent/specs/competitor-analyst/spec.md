# competitor-analyst Specification

## Purpose
The CompetitorAnalyst agent analyzes why target_notes achieve high prediction scores by identifying content patterns that statistically correlate with high metrics (CTR, comment_rate, sort_score2, etc.). It generates a SuccessProfileReport containing feature-centric patterns with statistical evidence and actionable "creation formulas."

## ADDED Requirements

### Requirement: Agent SHALL aggregate target_notes statistics using DataAggregatorTool

The CompetitorAnalyst MUST use DataAggregatorTool to calculate baseline statistics across all target_notes before analyzing individual patterns.

#### Scenario: Aggregate 50 target notes for keyword "婴儿辅食推荐"

```python
from xhs_seo_optimizer.crew import XhsSeoOptimizerCrew
from xhs_seo_optimizer.models.note import Note
from xhs_seo_optimizer.models.reports import SuccessProfileReport
import json

# Load target notes
with open("docs/target_notes.json") as f:
    target_notes = [Note.from_json(note_data) for note_data in json.load(f)]

# Create crew and execute
crew_instance = XhsSeoOptimizerCrew().crew()

# Crew MUST call DataAggregatorTool internally via CompetitorAnalyst
result = crew_instance.kickoff(inputs={
    "target_notes": target_notes,
    "keyword": "婴儿辅食推荐"
})

# Access output (Pydantic model if configured, or parse JSON)
report = result.pydantic  # Or: SuccessProfileReport(**json.loads(result.raw))

# Result MUST include aggregated_stats
assert report.aggregated_stats is not None
assert report.aggregated_stats.sample_size == len(target_notes)
assert "ctr" in report.aggregated_stats.prediction_stats
assert "comment_rate" in report.aggregated_stats.prediction_stats
```

### Requirement: Agent SHALL extract content features for all target notes

The agent MUST run NLPAnalysisTool and MultiModalVisionTool for each note to build a feature matrix.

#### Scenario: Extract features from 3 target notes

```python
# Crew MUST analyze all notes
result = crew_instance.kickoff(inputs={
    "target_notes": target_notes[:3],
    "keyword": "婴儿辅食推荐"
})

report = result.pydantic  # Or: SuccessProfileReport(**json.loads(result.raw))

# Verify that patterns reference features from both text and vision analysis
all_patterns = (report.title_patterns + report.cover_patterns +
                report.content_patterns + report.tag_patterns)

assert len(all_patterns) > 0, "No patterns discovered"

# At least one pattern MUST come from NLP analysis (title/content)
text_patterns = [p for p in all_patterns if p.feature_type in ["title", "content"]]
assert len(text_patterns) > 0

# At least one pattern SHOULD come from Vision analysis (cover)
vision_patterns = [p for p in all_patterns if p.feature_type == "cover"]
# Note: May be zero if no significant visual patterns found
```

#### Scenario: Handle notes with missing images gracefully

```python
# Create note without cover image
note_no_cover = Note.from_json({
    "note_id": "test123",
    "meta_data": {
        "title": "测试标题",
        "content": "测试内容",
        "cover_image_url": "",  # Missing
        "inner_image_urls": []
    },
    "prediction": {"ctr": 0.10, "sort_score2": 0.40},
    "tag": {"intention_lv2": "经验分享"}
})

# Crew SHOULD continue analysis even if vision analysis fails
result = crew_instance.kickoff(inputs={
    "target_notes": [note_no_cover],
    "keyword": "测试"
})

# MUST NOT raise error, may have fewer cover_patterns
report = result.pydantic
assert report.sample_size == 1
```

### Requirement: Agent MUST apply attribution rules to filter relevant features per metric

The agent SHALL use METRIC_FEATURE_ATTRIBUTION mapping to ensure only causally relevant features are analyzed for each metric.

#### Scenario: CTR analysis only considers title and cover features

```python
# When analyzing CTR patterns
result = crew_instance.kickoff(inputs={
    "target_notes": target_notes,
    "keyword": "婴儿辅食推荐"
})

report = result.pydantic

# Find patterns that affect CTR
ctr_patterns = [
    p for p in (report.title_patterns + report.cover_patterns +
                report.content_patterns + report.tag_patterns)
    if "ctr" in p.affected_metrics
]

# CTR-affecting patterns MUST only be title or cover types
for pattern in ctr_patterns:
    assert pattern.feature_type in ["title", "cover"], \
        f"CTR should not be affected by {pattern.feature_type} (attribution rule violation)"
```

#### Scenario: comment_rate analysis considers content and ending features

```python
# Find patterns that affect comment_rate
comment_patterns = [
    p for p in (report.title_patterns + report.cover_patterns +
                report.content_patterns + report.tag_patterns)
    if "comment_rate" in p.affected_metrics
]

# comment_rate patterns MUST be from content analysis
for pattern in comment_patterns:
    assert pattern.feature_type in ["content", "title"], \
        f"comment_rate affected by {pattern.feature_type} may violate attribution rules"
```

### Requirement: Agent SHALL identify statistically significant patterns with prevalence >70% and p<0.05

The agent MUST calculate pattern prevalence in high-scoring vs baseline groups and compute statistical significance.

#### Scenario: Identify high-CTR title pattern with 85% prevalence

```python
# Assume 85% of high-CTR notes use interrogative titles
result = crew_instance.kickoff(inputs={
    "target_notes": target_notes,
    "keyword": "婴儿辅食推荐"
})

report = result.pydantic

# Find interrogative title pattern (if significant)
title_patterns = report.title_patterns
significant_patterns = [p for p in title_patterns if p.p_value < 0.05]

# Each significant pattern MUST have:
for pattern in significant_patterns:
    assert pattern.prevalence_pct >= 70.0, \
        f"Pattern {pattern.feature_name} has prevalence {pattern.prevalence_pct}% < 70%"
    assert pattern.p_value < 0.05, \
        f"Pattern {pattern.feature_name} not statistically significant (p={pattern.p_value})"
    assert pattern.z_score > 0, "Z-score must be positive for high prevalence"
    assert pattern.sample_size_high > 0
    assert len(pattern.examples) >= 1, "Must provide at least one example"
```

#### Scenario: Filter out non-significant patterns with p>=0.05

```python
# Patterns with prevalence <70% or p>=0.05 SHOULD NOT be included
all_patterns = (report.title_patterns + report.cover_patterns +
                report.content_patterns + report.tag_patterns)

for pattern in all_patterns:
    # Either high prevalence OR statistically significant
    # (some patterns may have lower prevalence but very strong significance)
    if pattern.prevalence_pct < 70.0:
        # If prevalence is low, MUST have very strong significance
        assert pattern.p_value < 0.01, \
            f"Low-prevalence pattern {pattern.feature_name} lacks strong significance"
```

### Requirement: Agent MUST synthesize actionable formulas using LLM

For each identified pattern, the agent SHALL use an LLM to generate human-readable explanations and creation formulas in Chinese.

#### Scenario: Generate formula for interrogative title pattern

```python
result = crew_instance.kickoff(inputs={
    "target_notes": target_notes,
    "keyword": "婴儿辅食推荐"
})

report = result.pydantic

# Find a title pattern
title_patterns = report.title_patterns
assert len(title_patterns) > 0, "Should have at least one title pattern"

pattern = title_patterns[0]

# MUST have LLM-generated fields
assert pattern.why_it_works is not None
assert len(pattern.why_it_works) > 20, "Explanation too short"

assert pattern.creation_formula is not None
assert len(pattern.creation_formula) > 10, "Formula too short"

assert pattern.key_elements is not None
assert 3 <= len(pattern.key_elements) <= 5, "Should have 3-5 key elements"

# Formula MUST be in Chinese (contains Chinese characters)
import re
assert re.search(r'[\u4e00-\u9fff]', pattern.creation_formula), \
    "Formula must be in Chinese"
```

#### Scenario: LLM synthesis includes statistical evidence in explanation

```python
# why_it_works SHOULD reference the statistical evidence
pattern = title_patterns[0]

# Check that statistical terms or numbers appear in explanation
explanation = pattern.why_it_works.lower()
has_stats = any(keyword in explanation for keyword in [
    "85%", "70%", "显著", "统计", "高分", "样本"
])

# Optional: Explanation may or may not reference stats explicitly
# But it MUST be substantive
assert len(pattern.why_it_works) > 50, "Explanation should be detailed"
```

### Requirement: Agent SHALL output SuccessProfileReport conforming to schema

The agent MUST return a JSON-serializable SuccessProfileReport with all required fields populated.

#### Scenario: Report conforms to SuccessProfileReport schema

```python
from xhs_seo_optimizer.models.reports import SuccessProfileReport

result = crew_instance.kickoff(inputs={
    "target_notes": target_notes,
    "keyword": "婴儿辅食推荐"
})

# Access as Pydantic model (if output_pydantic configured)
report = result.pydantic

# Or parse from JSON if needed
# result_dict = json.loads(result.raw)
# report = SuccessProfileReport(**result_dict)

# Verify required fields exist
assert report.keyword == "婴儿辅食推荐"
assert report.sample_size == len(target_notes)
assert report.aggregated_stats is not None

assert isinstance(report.title_patterns, list)
assert isinstance(report.cover_patterns, list)
assert isinstance(report.content_patterns, list)
assert isinstance(report.tag_patterns, list)

assert isinstance(report.key_success_factors, list)
assert 3 <= len(report.key_success_factors) <= 5

assert isinstance(report.viral_formula_summary, str)
assert len(report.viral_formula_summary) > 50

assert report.analysis_timestamp is not None
```

#### Scenario: Each FeaturePattern in report has complete fields

```python
all_patterns = (report.title_patterns + report.cover_patterns +
                report.content_patterns + report.tag_patterns)

for pattern in all_patterns:
    # Required fields
    assert pattern.feature_name is not None
    assert pattern.feature_type in ["title", "cover", "content", "tag"]
    assert pattern.description is not None

    # Statistical fields
    assert 0 <= pattern.prevalence_pct <= 100
    assert 0 <= pattern.baseline_pct <= 100
    assert pattern.prevalence_pct > pattern.baseline_pct, \
        "High-group prevalence should exceed baseline"

    assert isinstance(pattern.affected_metrics, dict)
    assert len(pattern.affected_metrics) > 0

    assert pattern.z_score is not None
    assert pattern.p_value is not None
    assert 0 <= pattern.p_value <= 1

    assert pattern.sample_size_high > 0
    assert pattern.sample_size_baseline > 0

    # Examples
    assert isinstance(pattern.examples, list)
    assert 1 <= len(pattern.examples) <= 5

    # LLM fields
    assert pattern.why_it_works is not None
    assert pattern.creation_formula is not None
    assert isinstance(pattern.key_elements, list)
    assert 3 <= len(pattern.key_elements) <= 5
```

### Requirement: Agent SHALL organize patterns by feature type to avoid duplication

The agent MUST group patterns into title_patterns, cover_patterns, content_patterns, and tag_patterns lists.

#### Scenario: Patterns are correctly categorized by feature type

```python
result = crew_instance.kickoff(inputs={
    "target_notes": target_notes,
    "keyword": "婴儿辅食推荐"
})

report = result.pydantic

# Verify categorization
for pattern in report.title_patterns:
    assert pattern.feature_type == "title"

for pattern in report.cover_patterns:
    assert pattern.feature_type == "cover"

for pattern in report.content_patterns:
    assert pattern.feature_type == "content"

for pattern in report.tag_patterns:
    assert pattern.feature_type == "tag"
```

#### Scenario: No duplicate patterns across categories

```python
all_pattern_names = []
for pattern_list in [report.title_patterns, report.cover_patterns,
                     report.content_patterns, report.tag_patterns]:
    for pattern in pattern_list:
        all_pattern_names.append(pattern.feature_name)

# No duplicate feature names
assert len(all_pattern_names) == len(set(all_pattern_names)), \
    "Duplicate patterns found across categories"
```

### Requirement: Agent SHALL generate key success factors and viral formula summary

The agent MUST synthesize a holistic summary of the top success patterns using an LLM.

#### Scenario: key_success_factors includes 3-5 most impactful patterns

```python
result = crew_instance.kickoff(inputs={
    "target_notes": target_notes,
    "keyword": "婴儿辅食推荐"
})

report = result.pydantic

assert 3 <= len(report.key_success_factors) <= 5

# Each success factor MUST be a substantive insight
for factor in report.key_success_factors:
    assert len(factor) > 20, f"Success factor too short: {factor}"

    # Should contain Chinese
    import re
    assert re.search(r'[\u4e00-\u9fff]', factor)
```

#### Scenario: viral_formula_summary provides holistic creation template

```python
summary = report.viral_formula_summary

# MUST be substantial
assert len(summary) > 100, "Summary too short"

# MUST be in Chinese
import re
assert re.search(r'[\u4e00-\u9fff]', summary)

# SHOULD reference multiple feature types
# (Optional: may focus on most impactful patterns)
```

### Requirement: Agent SHALL handle edge cases gracefully

The agent MUST validate inputs and handle edge cases with clear behavior.

#### Scenario: Empty target_notes list raises ValueError

```python
import pytest

with pytest.raises(ValueError, match="at least one.*note"):
    crew_instance.kickoff(inputs={
        "target_notes": [],
        "keyword": "测试"
    })
```

#### Scenario: Single note returns report with limited patterns

```python
single_note = target_notes[0]

result = crew_instance.kickoff(inputs={
    "target_notes": [single_note],
    "keyword": "测试"
})

report = result.pydantic

assert report.sample_size == 1

# May have zero patterns (cannot compute prevalence with n=1)
all_patterns = (report.title_patterns + report.cover_patterns +
                report.content_patterns + report.tag_patterns)

# Agent SHOULD still provide analysis, even if no statistical patterns
# (e.g., describe the single note's features)
assert report.aggregated_stats is not None
```

#### Scenario: Notes with identical predictions do not cause division by zero

```python
# Create notes with same prediction values (zero variance)
identical_notes = []
for i in range(5):
    note = Note.from_json({
        "note_id": f"id{i}",
        "meta_data": {"title": f"标题{i}", "content": "内容"},
        "prediction": {"ctr": 0.10, "sort_score2": 0.40},  # Identical
        "tag": {"intention_lv2": "经验分享"}
    })
    identical_notes.append(note)

# Crew MUST NOT crash
result = crew_instance.kickoff(inputs={
    "target_notes": identical_notes,
    "keyword": "测试"
})

report = result.pydantic

# With zero variance, cannot separate high vs low groups
# Agent MAY return zero patterns or fall back to descriptive analysis
assert report.sample_size == 5
```
