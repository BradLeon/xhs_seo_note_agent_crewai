# data-aggregator Specification

## ADDED Requirements

### Requirement: Tool SHALL calculate statistical aggregates across multiple notes

The DataAggregatorTool MUST compute mean, median, standard deviation, min, and max for all numeric `prediction` metrics across a list of Note objects.

#### Scenario: Aggregate CTR across 3 target notes

```python
from tools.data_aggregator import DataAggregatorTool
from models.note import Note

notes = [
    Note.from_json({"note_id": "1", "prediction": {"ctr": 0.10, "sortScore": 0.40}, ...}),
    Note.from_json({"note_id": "2", "prediction": {"ctr": 0.14, "sortScore": 0.45}, ...}),
    Note.from_json({"note_id": "3", "prediction": {"ctr": 0.16, "sortScore": 0.50}, ...})
]

tool = DataAggregatorTool()
result_json = tool._run(notes=notes)
result = json.loads(result_json)

# Verify CTR statistics
assert result["prediction_stats"]["ctr"]["mean"] == 0.1333  # (0.10+0.14+0.16)/3
assert result["prediction_stats"]["ctr"]["median"] == 0.14
assert 0.029 < result["prediction_stats"]["ctr"]["std"] < 0.031  # ~0.03
assert result["prediction_stats"]["ctr"]["min"] == 0.10
assert result["prediction_stats"]["ctr"]["max"] == 0.16
assert result["prediction_stats"]["ctr"]["count"] == 3
```

#### Scenario: Aggregate all prediction metrics

```python
# Tool MUST handle all 10 prediction metrics
metrics = ["ctr", "ces_rate", "interaction_rate", "like_rate", "fav_rate",
           "comment_rate", "share_rate", "follow_rate", "sortScore", "impression"]

result = json.loads(tool._run(notes=notes))

for metric in metrics:
    assert metric in result["prediction_stats"]
    assert "mean" in result["prediction_stats"][metric]
    assert "median" in result["prediction_stats"][metric]
    assert "std" in result["prediction_stats"][metric]
    assert "min" in result["prediction_stats"][metric]
    assert "max" in result["prediction_stats"][metric]
    assert "count" in result["prediction_stats"][metric]
```

### Requirement: Tool MUST calculate tag frequencies and modes

The tool SHALL count occurrences of each tag value and identify the most common value (mode) for categorical tag fields.

#### Scenario: Calculate tag frequencies

```python
notes = [
    Note.from_json({"tag": {"intention_lv2": "经验分享", "taxonomy2": "婴童用品"}, ...}),
    Note.from_json({"tag": {"intention_lv2": "经验分享", "taxonomy2": "婴童用品"}, ...}),
    Note.from_json({"tag": {"intention_lv2": "产品推荐", "taxonomy2": "婴童食品"}, ...}),
]

result = json.loads(tool._run(notes=notes))

# Verify frequency counts
assert result["tag_frequencies"]["intention_lv2"] == {
    "经验分享": 2,
    "产品推荐": 1
}
assert result["tag_frequencies"]["taxonomy2"] == {
    "婴童用品": 2,
    "婴童食品": 1
}

# Verify modes (most common values)
assert result["tag_modes"]["intention_lv2"] == "经验分享"
assert result["tag_modes"]["taxonomy2"] == "婴童用品"
```

#### Scenario: Handle tied modes deterministically

```python
# When multiple values have same frequency, tool MUST return first alphabetically
notes = [
    Note.from_json({"tag": {"intention_lv2": "产品推荐"}, ...}),
    Note.from_json({"tag": {"intention_lv2": "经验分享"}, ...}),
]

result = json.loads(tool._run(notes=notes))

# Both have frequency 1, tool MUST choose alphabetically first
assert result["tag_frequencies"]["intention_lv2"] == {"产品推荐": 1, "经验分享": 1}
assert result["tag_modes"]["intention_lv2"] == "产品推荐"  # "产" < "经" in unicode
```

### Requirement: Tool SHALL handle edge cases gracefully

The tool MUST validate inputs and handle edge cases with clear error messages.

#### Scenario: Empty notes list raises error

```python
tool = DataAggregatorTool()

with pytest.raises(ValueError, match="at least one note"):
    tool._run(notes=[])
```

#### Scenario: Single note returns stats with zero variance

```python
notes = [Note.from_json({"prediction": {"ctr": 0.10}, ...})]

result = json.loads(tool._run(notes=notes))

assert result["prediction_stats"]["ctr"]["mean"] == 0.10
assert result["prediction_stats"]["ctr"]["median"] == 0.10
assert result["prediction_stats"]["ctr"]["std"] == 0.0  # No variance
assert result["prediction_stats"]["ctr"]["min"] == 0.10
assert result["prediction_stats"]["ctr"]["max"] == 0.10
assert result["prediction_stats"]["ctr"]["count"] == 1
assert result["sample_size"] == 1
```

#### Scenario: Missing prediction fields skipped with warning

```python
notes = [
    Note.from_json({"note_id": "1", "prediction": {"ctr": 0.10}, ...}),  # Missing sortScore
    Note.from_json({"note_id": "2", "prediction": {"ctr": 0.14, "sortScore": 0.45}, ...}),
]

# Tool SHOULD log warning but continue
result = json.loads(tool._run(notes=notes))

assert result["prediction_stats"]["ctr"]["count"] == 2  # Both have CTR
assert result["prediction_stats"]["sortScore"]["count"] == 1  # Only one has sortScore
```

### Requirement: Tool MUST support optional outlier removal

The tool SHALL provide configurable outlier detection and removal using z-score or IQR method.

#### Scenario: Remove outliers using 2-sigma threshold

```python
# Normal values: 0.10, 0.11, 0.12, 0.13
# Outlier: 0.50 (way outside 2σ)
notes = [
    Note.from_json({"prediction": {"ctr": 0.10}, ...}),
    Note.from_json({"prediction": {"ctr": 0.11}, ...}),
    Note.from_json({"prediction": {"ctr": 0.12}, ...}),
    Note.from_json({"prediction": {"ctr": 0.13}, ...}),
    Note.from_json({"prediction": {"ctr": 0.50}, ...}),  # Outlier
]

tool = DataAggregatorTool(remove_outliers=True, outlier_threshold=2.0)
result = json.loads(tool._run(notes=notes))

# Outlier SHOULD be removed
assert result["prediction_stats"]["ctr"]["count"] == 4  # Not 5
assert result["prediction_stats"]["ctr"]["mean"] < 0.15  # Mean not inflated by 0.50
assert result["outliers_removed"] == 1
```

#### Scenario: Outlier removal disabled by default

```python
tool = DataAggregatorTool()  # Default: remove_outliers=False
result = json.loads(tool._run(notes=notes))

assert result["prediction_stats"]["ctr"]["count"] == 5  # All included
assert result["outliers_removed"] == 0
```

### Requirement: Tool MUST return structured AggregatedMetrics model

The tool SHALL return JSON string containing AggregatedMetrics Pydantic model with all required fields.

#### Scenario: Output conforms to AggregatedMetrics schema

```python
from models.analysis_results import AggregatedMetrics

result_json = tool._run(notes=notes)

# MUST be valid JSON
result_dict = json.loads(result_json)

# MUST validate against Pydantic model
aggregated = AggregatedMetrics(**result_dict)

# Verify required fields exist
assert isinstance(aggregated.prediction_stats, dict)
assert isinstance(aggregated.tag_frequencies, dict)
assert isinstance(aggregated.tag_modes, dict)
assert isinstance(aggregated.sample_size, int)
assert isinstance(aggregated.outliers_removed, int)
```

### Requirement: Tool SHALL be usable as CrewAI tool

The tool MUST inherit from BaseTool and be assignable to CrewAI agents.

#### Scenario: Tool can be added to CompetitorAnalyst

```python
from crewai import Agent
from tools.data_aggregator import DataAggregatorTool

agent = Agent(
    role="Competitor Analyst",
    goal="Analyze target notes",
    backstory="Expert at finding patterns",
    tools=[DataAggregatorTool()]
)

assert len(agent.tools) == 1
assert isinstance(agent.tools[0], DataAggregatorTool)
assert agent.tools[0].name == "Data Aggregator"
assert "statistical aggregates" in agent.tools[0].description.lower()
```
