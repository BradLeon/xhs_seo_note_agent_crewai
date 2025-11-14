# statistical-delta Specification

## ADDED Requirements

### Requirement: Tool SHALL calculate statistical significance of gaps

The StatisticalDeltaTool MUST compute z-scores and p-values to determine if differences between owned_note metrics and target_notes aggregates are statistically significant.

#### Scenario: Identify significant CTR gap

```python
from tools.statistical_delta import StatisticalDeltaTool
from models.note import Note
from models.analysis_results import AggregatedMetrics

# Target notes have mean CTR = 0.14, std = 0.03
target_stats = AggregatedMetrics(
    prediction_stats={
        "ctr": {"mean": 0.14, "std": 0.03, "median": 0.14, "min": 0.10, "max": 0.18, "count": 10}
    },
    tag_frequencies={},
    tag_modes={},
    sample_size=10,
    outliers_removed=0
)

# Owned note has CTR = 0.08 (much lower)
owned_note = Note.from_json({"prediction": {"ctr": 0.08, "sortScore": 0.42}, ...})

tool = StatisticalDeltaTool()
result_json = tool._run(owned_note=owned_note, target_stats=target_stats)
result = json.loads(result_json)

# Find CTR gap
ctr_gap = next(g for g in result["significant_gaps"] if g["metric"] == "ctr")

assert ctr_gap["owned_value"] == 0.08
assert ctr_gap["target_mean"] == 0.14
assert ctr_gap["delta_absolute"] == -0.06
assert ctr_gap["delta_pct"] == pytest.approx(-42.9, abs=0.1)  # (0.08-0.14)/0.14 = -42.9%
assert ctr_gap["z_score"] == pytest.approx(-2.0, abs=0.1)    # (0.08-0.14)/0.03 = -2.0
assert ctr_gap["p_value"] < 0.05  # Statistically significant
assert ctr_gap["significance"] in ["significant", "very_significant", "critical"]
```

#### Scenario: Classify significance levels correctly

```python
# Critical gap: z=-3.5, p<0.001
gap1 = {"z_score": -3.5, "p_value": 0.0005}
assert tool._classify_significance(gap1["p_value"]) == "critical"

# Very significant: z=-2.8, p<0.01
gap2 = {"z_score": -2.8, "p_value": 0.005}
assert tool._classify_significance(gap2["p_value"]) == "very_significant"

# Significant: z=-2.0, p<0.05
gap3 = {"z_score": -2.0, "p_value": 0.046}
assert tool._classify_significance(gap3["p_value"]) == "significant"

# Marginal: z=-1.7, p<0.10
gap4 = {"z_score": -1.7, "p_value": 0.089}
assert tool._classify_significance(gap4["p_value"]) == "marginal"

# Not significant: z=-0.5, p>0.10
gap5 = {"z_score": -0.5, "p_value": 0.617}
assert tool._classify_significance(gap5["p_value"]) == "none"
```

### Requirement: Tool MUST separate significant and non-significant gaps

The tool SHALL classify gaps into two lists based on significance threshold (default α=0.05).

#### Scenario: Separate gaps by significance

```python
target_stats = AggregatedMetrics(
    prediction_stats={
        "ctr": {"mean": 0.14, "std": 0.03, ...},        # Owned=0.08: significant
        "comment_rate": {"mean": 0.012, "std": 0.003, ...},  # Owned=0.011: not significant
        "sortScore": {"mean": 0.45, "std": 0.05, ...}   # Owned=0.30: very significant
    },
    ...
)

owned_note = Note.from_json({
    "prediction": {"ctr": 0.08, "comment_rate": 0.011, "sortScore": 0.30},
    ...
})

result = json.loads(tool._run(owned_note=owned_note, target_stats=target_stats))

# CTR and sortScore should be in significant_gaps (p < 0.05)
significant_metrics = [g["metric"] for g in result["significant_gaps"]]
assert "ctr" in significant_metrics
assert "sortScore" in significant_metrics

# comment_rate should be in non_significant_gaps (p >= 0.05)
non_significant_metrics = [g["metric"] for g in result["non_significant_gaps"]]
assert "comment_rate" in non_significant_metrics
```

### Requirement: Tool SHALL prioritize gaps by combined significance and magnitude

The tool MUST sort gaps by a priority score combining statistical significance (z-score) and practical importance (delta percentage).

#### Scenario: Priority order correct

```python
# Gap A: z=-3.0, delta=-30%  → priority = 3.0 * 0.30 = 0.90
# Gap B: z=-2.0, delta=-80%  → priority = 2.0 * 0.80 = 1.60 (higher!)
# Gap C: z=-4.0, delta=-10%  → priority = 4.0 * 0.10 = 0.40

target_stats = AggregatedMetrics(
    prediction_stats={
        "metric_a": {"mean": 0.10, "std": 0.01, ...},  # Owned=0.07: z=-3.0, delta=-30%
        "metric_b": {"mean": 0.10, "std": 0.025, ...}, # Owned=0.02: z=-3.2, delta=-80%
        "metric_c": {"mean": 0.10, "std": 0.01, ...},  # Owned=0.09: z=-1.0, delta=-10%
    },
    ...
)

owned_note = Note.from_json({
    "prediction": {"metric_a": 0.07, "metric_b": 0.02, "metric_c": 0.09},
    ...
})

result = json.loads(tool._run(owned_note=owned_note, target_stats=target_stats))

# Priority order MUST reflect combined score
assert result["priority_order"][0] == "metric_b"  # Highest priority (z * delta)
assert result["priority_order"][1] == "metric_a"
assert "metric_c" in result["priority_order"][2:]  # Lower priority or non-significant
```

### Requirement: Tool MUST handle edge cases gracefully

The tool SHALL validate inputs and handle special cases with appropriate defaults or errors.

#### Scenario: Handle zero standard deviation (all targets identical)

```python
# All target notes have identical CTR = 0.14
target_stats = AggregatedMetrics(
    prediction_stats={
        "ctr": {"mean": 0.14, "std": 0.0, ...}  # Zero variance!
    },
    ...
)

owned_note = Note.from_json({"prediction": {"ctr": 0.10}, ...})

result = json.loads(tool._run(owned_note=owned_note, target_stats=target_stats))

# Cannot compute z-score with std=0
ctr_gap = result["significant_gaps"][0] if result["significant_gaps"] else result["non_significant_gaps"][0]
assert ctr_gap["z_score"] == float('inf') or ctr_gap["z_score"] == float('-inf')
assert ctr_gap["significance"] == "undefined"
assert "zero variance" in ctr_gap["interpretation"].lower()
```

#### Scenario: Handle both values at zero

```python
# Both owned and target are zero
target_stats = AggregatedMetrics(
    prediction_stats={"share_rate": {"mean": 0.0, "std": 0.0, ...}},
    ...
)

owned_note = Note.from_json({"prediction": {"share_rate": 0.0}, ...})

result = json.loads(tool._run(owned_note=owned_note, target_stats=target_stats))

gap = result["non_significant_gaps"][0]
assert gap["delta_absolute"] == 0.0
assert gap["delta_pct"] == 0.0  # Or undefined, not NaN
assert gap["significance"] == "none"
```

#### Scenario: Handle missing metrics in owned_note

```python
# Owned note missing sortScore metric
owned_note = Note.from_json({"prediction": {"ctr": 0.10}, ...})  # No sortScore

target_stats = AggregatedMetrics(
    prediction_stats={
        "ctr": {"mean": 0.14, ...},
        "sortScore": {"mean": 0.45, ...}  # Present in targets
    },
    ...
)

result = json.loads(tool._run(owned_note=owned_note, target_stats=target_stats))

# Tool MUST still analyze CTR
assert any(g["metric"] == "ctr" for g in result["significant_gaps"] + result["non_significant_gaps"])

# sortScore gap SHOULD appear with "N/A" or be skipped
sortScore_gaps = [g for g in result["significant_gaps"] + result["non_significant_gaps"] if g["metric"] == "sortScore"]
if sortScore_gaps:
    assert "missing" in sortScore_gaps[0]["interpretation"].lower() or sortScore_gaps[0]["owned_value"] is None
```

### Requirement: Tool SHALL generate human-readable interpretations

The tool MUST provide textual interpretations of each gap for agent consumption.

#### Scenario: Generate clear interpretation text

```python
# Significant gap
gap = {
    "metric": "comment_rate",
    "owned_value": 0.0013,
    "target_mean": 0.012,
    "delta_pct": -89.2,
    "z_score": -3.67,
    "p_value": 0.0001,
    "significance": "critical"
}

interpretation = tool._generate_interpretation(gap)

assert "comment_rate" in interpretation
assert "critically lower" in interpretation or "significantly lower" in interpretation
assert "3.67" in interpretation or "3.7" in interpretation  # z-score mentioned
assert "89" in interpretation or "89.2" in interpretation   # Percentage mentioned
```

### Requirement: Tool MUST return structured GapAnalysis model

The tool SHALL return JSON string containing GapAnalysis Pydantic model.

#### Scenario: Output conforms to GapAnalysis schema

```python
from models.analysis_results import GapAnalysis

result_json = tool._run(owned_note=owned_note, target_stats=target_stats)
result_dict = json.loads(result_json)

# MUST validate against Pydantic model
gap_analysis = GapAnalysis(**result_dict)

# Verify required fields
assert isinstance(gap_analysis.significant_gaps, list)
assert isinstance(gap_analysis.non_significant_gaps, list)
assert isinstance(gap_analysis.priority_order, list)
assert isinstance(gap_analysis.sample_size, int)

# Each gap MUST have required fields
for gap in gap_analysis.significant_gaps:
    assert hasattr(gap, "metric")
    assert hasattr(gap, "owned_value")
    assert hasattr(gap, "target_mean")
    assert hasattr(gap, "z_score")
    assert hasattr(gap, "p_value")
    assert hasattr(gap, "significance")
    assert hasattr(gap, "interpretation")
```

### Requirement: Tool SHALL be usable as CrewAI tool

The tool MUST inherit from BaseTool and be assignable to GapFinder agent.

#### Scenario: Tool can be added to GapFinder agent

```python
from crewai import Agent
from tools.statistical_delta import StatisticalDeltaTool

agent = Agent(
    role="Gap Finder",
    goal="Identify significant performance gaps",
    backstory="Expert at statistical analysis",
    tools=[StatisticalDeltaTool()]
)

assert len(agent.tools) == 1
assert isinstance(agent.tools[0], StatisticalDeltaTool)
assert agent.tools[0].name == "Statistical Delta Analyzer"
assert "statistical significance" in agent.tools[0].description.lower()
```

### Requirement: Tool SHALL support configurable significance threshold

The tool MUST allow users to specify custom significance threshold (default α=0.05).

#### Scenario: Use custom significance threshold

```python
# Use more lenient threshold (α=0.10 instead of 0.05)
tool = StatisticalDeltaTool(alpha=0.10)

# Gap with p=0.08 should be significant with α=0.10
target_stats = AggregatedMetrics(
    prediction_stats={"ctr": {"mean": 0.14, "std": 0.03, ...}},
    ...
)
owned_note = Note.from_json({"prediction": {"ctr": 0.10}, ...})  # z ≈ -1.33, p ≈ 0.08

result = json.loads(tool._run(owned_note=owned_note, target_stats=target_stats))

# With α=0.10, this gap SHOULD be in significant_gaps
assert any(g["metric"] == "ctr" for g in result["significant_gaps"])
```
