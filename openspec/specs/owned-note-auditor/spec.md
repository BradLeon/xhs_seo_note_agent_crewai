# owned-note-auditor Specification

## Purpose
TBD - created by archiving change 0004-implement-owned-note-auditor. Update Purpose after archive.
## Requirements
### Requirement: Agent SHALL extract content features using NLP and Vision tools

The OwnedNoteAuditor MUST call NLPAnalysisTool and MultiModalVisionTool to extract comprehensive text and visual features from the owned note.

#### Scenario: Extract features from owned note

```python
from xhs_seo_optimizer.crew_owned_note import XhsSeoOptimizerCrewOwnedNote
from xhs_seo_optimizer.models.note import Note
from xhs_seo_optimizer.models.reports import AuditReport
import json

# Load owned note
with open("docs/owned_note.json") as f:
    owned_note_data = json.load(f)

# Create crew and execute
crew = XhsSeoOptimizerCrewOwnedNote()
result = crew.kickoff(inputs={
    "owned_note": owned_note_data,
    "keyword": "DHA"
})

# Access output
audit_report = result.pydantic  # Or: AuditReport(**json.loads(result.raw))

# MUST have extracted features
assert audit_report.text_features is not None
assert audit_report.visual_features is not None

# Text features from NLPAnalysisTool (30+ fields)
assert audit_report.text_features.title_pattern is not None
assert audit_report.text_features.opening_strategy is not None
assert audit_report.text_features.content_framework is not None
assert audit_report.text_features.ending_technique is not None

# Visual features from MultiModalVisionTool (17+ fields)
assert audit_report.visual_features.image_count > 0
assert audit_report.visual_features.image_style is not None
assert audit_report.visual_features.thumbnail_appeal is not None
```

#### Scenario: Handle missing images gracefully

```python
# Note with no inner images
note_no_inner = {
    "note_id": "test123",
    "meta_data": {
        "title": "测试标题",
        "content": "测试内容",
        "cover_image_url": "http://example.com/cover.jpg",
        "inner_image_urls": []  # No inner images
    },
    "prediction": {"ctr": 0.10},
    "tag": {"intention_lv2": "经验分享"}
}

# SHOULD NOT crash, vision analysis proceeds with cover only
result = crew.kickoff(inputs={
    "owned_note": note_no_inner,
    "keyword": "测试"
})

audit_report = result.pydantic

# Visual features MUST still be populated (using cover image)
assert audit_report.visual_features is not None
assert audit_report.visual_features.image_count >= 1  # At least cover
```

### Requirement: Agent SHALL analyze metric performance against thresholds

The agent MUST evaluate all 10 prediction metrics and identify which are underperforming based on defined thresholds or statistical baselines.

#### Scenario: Identify weak metrics below threshold

```python
# Owned note with low CTR and sort_score2
owned_note_data = {
    "note_id": "weak123",
    "meta_data": {...},
    "prediction": {
        "ctr": 0.03,  # Below threshold (e.g., 0.05)
        "comment_rate": 0.0020,  # Normal
        "sort_score2": 0.25,  # Below threshold (e.g., 0.35)
        # ... other metrics
    },
    "tag": {...}
}

result = crew.kickoff(inputs={
    "owned_note": owned_note_data,
    "keyword": "测试"
})

audit_report = result.pydantic

# MUST identify weak metrics
assert "ctr" in audit_report.weak_metrics
assert "sort_score2" in audit_report.weak_metrics
assert "comment_rate" not in audit_report.weak_metrics  # Normal

# MUST include current metric values
assert audit_report.current_metrics["ctr"] == 0.03
assert audit_report.current_metrics["sort_score2"] == 0.25
```

#### Scenario: Identify strong metrics above threshold

```python
owned_note_data = {
    "note_id": "strong123",
    "meta_data": {...},
    "prediction": {
        "ctr": 0.15,  # High
        "comment_rate": 0.0050,  # High
        "like_rate": 0.080,  # Normal
        # ...
    },
    "tag": {...}
}

result = crew.kickoff(inputs={
    "owned_note": owned_note_data,
    "keyword": "测试"
})

audit_report = result.pydantic

# MUST identify strong metrics
assert "ctr" in audit_report.strong_metrics
assert "comment_rate" in audit_report.strong_metrics
assert "like_rate" not in audit_report.strong_metrics  # Normal range
```

### Requirement: Agent SHALL identify content weaknesses

The agent MUST compare extracted features against best practices or common success patterns to identify missing or weak content elements.

#### Scenario: Identify missing emotional hook in title

```python
# Note with weak title pattern
owned_note_data = {
    "note_id": "test123",
    "meta_data": {
        "title": "DHA产品介绍",  # Generic, no emotional appeal
        "content": "...",
        ...
    },
    "prediction": {"ctr": 0.03},  # Low CTR
    "tag": {...}
}

result = crew.kickoff(inputs={
    "owned_note": owned_note_data,
    "keyword": "DHA"
})

audit_report = result.pydantic

# MUST identify content weaknesses
assert len(audit_report.content_weaknesses) > 0

# Should mention title issues (based on NLP analysis)
weaknesses_text = " ".join(audit_report.content_weaknesses).lower()
assert any(keyword in weaknesses_text for keyword in [
    "标题", "title", "情感", "emotional", "hook", "钩子"
])
```

#### Scenario: Identify low thumbnail appeal

```python
# Note with low visual appeal score
owned_note_data = {
    "note_id": "test123",
    "meta_data": {...},
    "prediction": {"ctr": 0.02},  # Very low CTR
    "tag": {...}
}

# Assume vision analysis returns low thumbnail_appeal
result = crew.kickoff(inputs={
    "owned_note": owned_note_data,
    "keyword": "测试"
})

audit_report = result.pydantic

# If vision features show weak thumbnail appeal, SHOULD be in weaknesses
# (This depends on threshold logic in identify_weaknesses task)
assert len(audit_report.content_weaknesses) > 0

# Check if visual weaknesses are mentioned
weaknesses_text = " ".join(audit_report.content_weaknesses)
# May include: "封面吸引力不足", "thumbnail appeal", "视觉冲击力弱" etc.
```

### Requirement: Agent SHALL identify content strengths

The agent MUST also identify strong features that are already working well in the owned note.

#### Scenario: Identify strong credibility signals

```python
owned_note_data = {
    "note_id": "test123",
    "meta_data": {
        "title": "...",
        "content": "作为资深营养师,我推荐的DHA品牌经过严格测试...",  # Strong authority
        ...
    },
    "prediction": {"comment_rate": 0.0045},  # Good engagement
    "tag": {...}
}

result = crew.kickoff(inputs={
    "owned_note": owned_note_data,
    "keyword": "DHA"
})

audit_report = result.pydantic

# MUST identify content strengths
assert len(audit_report.content_strengths) > 0

# Should mention credibility/authority (based on NLP analysis)
strengths_text = " ".join(audit_report.content_strengths).lower()
assert any(keyword in strengths_text for keyword in [
    "可信", "credibility", "权威", "authority", "专业"
])
```

#### Scenario: Identify strong visual storytelling

```python
owned_note_data = {
    "note_id": "test123",
    "meta_data": {...},
    "prediction": {"ctr": 0.12},  # Good CTR
    "tag": {...}
}

# Assume vision analysis returns high visual_storytelling score
result = crew.kickoff(inputs={
    "owned_note": owned_note_data,
    "keyword": "测试"
})

audit_report = result.pydantic

# SHOULD identify visual strengths
assert len(audit_report.content_strengths) > 0

# May include: "视觉叙事性强", "图文配合好", "visual storytelling" etc.
```

### Requirement: Agent SHALL generate overall diagnosis summary

The agent MUST synthesize findings into a concise overall diagnosis that highlights the key issues.

#### Scenario: Overall diagnosis summarizes main weaknesses

```python
result = crew.kickoff(inputs={
    "owned_note": owned_note_data,
    "keyword": "DHA"
})

audit_report = result.pydantic

# MUST have overall diagnosis
assert audit_report.overall_diagnosis is not None
assert len(audit_report.overall_diagnosis) >= 50, "Diagnosis too short"
assert len(audit_report.overall_diagnosis) <= 200, "Diagnosis too verbose"

# MUST be in Chinese
import re
assert re.search(r'[\u4e00-\u9fff]', audit_report.overall_diagnosis)

# SHOULD reference key issues (weak metrics or content gaps)
diagnosis_lower = audit_report.overall_diagnosis.lower()
# Example: "CTR偏低,标题缺乏情感钩子,封面视觉吸引力不足"
```

### Requirement: Agent SHALL output AuditReport conforming to schema

The agent MUST return a JSON-serializable AuditReport with all required fields populated.

#### Scenario: Report conforms to AuditReport schema

```python
from xhs_seo_optimizer.models.reports import AuditReport

result = crew.kickoff(inputs={
    "owned_note": owned_note_data,
    "keyword": "DHA"
})

# Access as Pydantic model
audit_report = result.pydantic

# Verify required fields exist
assert audit_report.note_id == owned_note_data["note_id"]
assert audit_report.keyword == "DHA"

assert isinstance(audit_report.current_metrics, dict)
assert len(audit_report.current_metrics) == 10  # All 10 metrics

assert audit_report.text_features is not None
assert audit_report.visual_features is not None

assert isinstance(audit_report.weak_metrics, list)
assert isinstance(audit_report.strong_metrics, list)

assert isinstance(audit_report.content_weaknesses, list)
assert isinstance(audit_report.content_strengths, list)

assert audit_report.overall_diagnosis is not None

assert audit_report.audit_timestamp is not None
# Timestamp format: ISO 8601
import datetime
datetime.datetime.fromisoformat(audit_report.audit_timestamp)  # Should not raise
```

#### Scenario: Report can be serialized to JSON

```python
# MUST be JSON-serializable
import json

audit_dict = audit_report.model_dump()
json_str = json.dumps(audit_dict, ensure_ascii=False, indent=2)

# Should round-trip
parsed = json.loads(json_str)
assert parsed["note_id"] == audit_report.note_id
```

### Requirement: Agent SHALL use shared_context to prevent LLM hallucination

The agent MUST store owned note data in shared_context and use smart mode (note_id) for tool calls.

#### Scenario: Tools are called with note_id only

```python
# This scenario verifies implementation pattern (tested via code review or logging)
# In @before_kickoff:
# shared_context.set("owned_note_data", {...})

# In tasks, agent MUST call tools as:
# multimodal_vision_analysis(note_id="67adc607000000002901c2c0")
# nlp_text_analysis(note_id="67adc607000000002901c2c0")

# NOT as (legacy mode):
# multimodal_vision_analysis(note_metadata={...})  # BAD: causes hallucination

# This is enforced via task instructions in YAML and verified in tests by:
# 1. Checking shared_context is populated in @before_kickoff
# 2. Reviewing agent logs for tool call parameters
```

### Requirement: Agent SHALL execute 4 sequential tasks

The agent MUST chain 4 tasks in order: extract features → analyze metrics → identify weaknesses → generate report.

#### Scenario: All 4 tasks execute in sequence

```python
# Enable verbose logging to verify task execution
import logging
logging.basicConfig(level=logging.INFO)

result = crew.kickoff(inputs={
    "owned_note": owned_note_data,
    "keyword": "DHA"
})

# Verify output from all tasks (via logs or result)
# Task 1: extract_content_features → text_features + visual_features
# Task 2: analyze_metric_performance → weak_metrics + strong_metrics
# Task 3: identify_weaknesses → content_weaknesses + content_strengths
# Task 4: generate_audit_report → final AuditReport JSON

audit_report = result.pydantic
assert audit_report is not None  # All tasks completed successfully
```

#### Scenario: Later tasks receive context from earlier tasks

```python
# Task 2 (analyze_metric_performance) MUST have access to Task 1 output
# Task 3 (identify_weaknesses) MUST have access to Task 1 and Task 2 outputs
# Task 4 (generate_audit_report) MUST have access to all previous outputs

# This is verified via task context configuration in YAML:
# analyze_metric_performance:
#   context: [extract_content_features]
# identify_weaknesses:
#   context: [extract_content_features, analyze_metric_performance]
# generate_audit_report:
#   context: [extract_content_features, analyze_metric_performance, identify_weaknesses]
```

### Requirement: Agent SHALL handle edge cases gracefully

The agent MUST validate inputs and handle edge cases with clear error messages.

#### Scenario: Missing note_id raises ValueError

```python
import pytest

invalid_input = {
    "owned_note": {
        # Missing note_id
        "meta_data": {"title": "测试", "content": "内容"},
        "prediction": {"ctr": 0.10},
        "tag": {}
    },
    "keyword": "测试"
}

with pytest.raises(ValueError, match="note_id.*required"):
    crew.kickoff(inputs=invalid_input)
```

#### Scenario: Missing prediction metrics raises ValueError

```python
invalid_input = {
    "owned_note": {
        "note_id": "test123",
        "meta_data": {"title": "测试", "content": "内容"},
        # Missing prediction field
        "tag": {}
    },
    "keyword": "测试"
}

with pytest.raises(ValueError, match="prediction.*required"):
    crew.kickoff(inputs=invalid_input)
```

#### Scenario: Empty title/content handled gracefully

```python
# Note with empty content fields
minimal_note = {
    "note_id": "test123",
    "meta_data": {
        "title": "",  # Empty
        "content": "",  # Empty
        "cover_image_url": "http://example.com/img.jpg",
        "inner_image_urls": []
    },
    "prediction": {"ctr": 0.01},
    "tag": {"intention_lv2": "经验分享"}
}

# SHOULD NOT crash, tools handle empty text gracefully
result = crew.kickoff(inputs={
    "owned_note": minimal_note,
    "keyword": "测试"
})

audit_report = result.pydantic

# May have limited text features, but MUST have visual features
assert audit_report.visual_features is not None

# SHOULD identify weak text content
assert len(audit_report.content_weaknesses) > 0
```

### Requirement: Agent SHALL write output to outputs/audit_report.json

The agent MUST save the final AuditReport to the outputs directory as JSON.

#### Scenario: Output file is created with valid JSON

```python
import os
import json

# Remove old output if exists
output_path = "outputs/audit_report.json"
if os.path.exists(output_path):
    os.remove(output_path)

result = crew.kickoff(inputs={
    "owned_note": owned_note_data,
    "keyword": "DHA"
})

# File MUST be created
assert os.path.exists(output_path)

# File MUST contain valid JSON
with open(output_path) as f:
    report_dict = json.load(f)

# MUST match result
assert report_dict["note_id"] == owned_note_data["note_id"]
assert report_dict["keyword"] == "DHA"
```

#### Scenario: Output file matches Pydantic model

```python
from xhs_seo_optimizer.models.reports import AuditReport

# Read from file and parse
with open("outputs/audit_report.json") as f:
    report_from_file = AuditReport(**json.load(f))

# Should match result.pydantic
assert report_from_file.note_id == result.pydantic.note_id
assert report_from_file.audit_timestamp == result.pydantic.audit_timestamp
```

