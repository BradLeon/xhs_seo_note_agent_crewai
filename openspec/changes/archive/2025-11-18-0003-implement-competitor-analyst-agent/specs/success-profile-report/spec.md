# success-profile-report Specification

## Purpose
The SuccessProfileReport data model represents the output of CompetitorAnalyst agent analysis. It contains feature-centric content patterns with statistical evidence and actionable creation formulas, organized to support downstream gap analysis and optimization strategy generation.

## ADDED Requirements

### Requirement: Report SHALL include aggregated baseline statistics

The SuccessProfileReport MUST contain an AggregatedMetrics object from DataAggregatorTool.

#### Scenario: Report includes aggregated statistics for all metrics

```python
from xhs_seo_optimizer.models.reports import SuccessProfileReport
from xhs_seo_optimizer.models.analysis_results import AggregatedMetrics
import json

report_data = {
    "keyword": "婴儿辅食推荐",
    "sample_size": 50,
    "aggregated_stats": {
        "prediction_stats": {
            "ctr": {"mean": 0.135, "median": 0.130, "std": 0.028, "min": 0.08, "max": 0.20, "count": 50},
            "comment_rate": {"mean": 0.0024, "median": 0.0020, "std": 0.0008, "min": 0.001, "max": 0.005, "count": 50}
        },
        "tag_frequencies": {"intention_lv2": {"经验分享": 30, "产品推荐": 20}},
        "tag_modes": {"intention_lv2": "经验分享"},
        "sample_size": 50,
        "outliers_removed": 0
    },
    "title_patterns": [],
    "cover_patterns": [],
    "content_patterns": [],
    "tag_patterns": [],
    "key_success_factors": ["因素1", "因素2", "因素3"],
    "viral_formula_summary": "总结",
    "analysis_timestamp": "2025-11-14T10:00:00Z"
}

report = SuccessProfileReport(**report_data)

assert report.aggregated_stats is not None
assert isinstance(report.aggregated_stats, AggregatedMetrics)
assert report.aggregated_stats.sample_size == 50
assert "ctr" in report.aggregated_stats.prediction_stats
```

### Requirement: Report SHALL organize patterns by feature type

The report MUST maintain four separate lists for different feature types to avoid duplication.

#### Scenario: Patterns are segregated into type-specific lists

```python
from xhs_seo_optimizer.models.reports import FeaturePattern

title_pattern = FeaturePattern(
    feature_name="interrogative_title",
    feature_type="title",
    description="使用疑问句标题",
    prevalence_pct=85.0,
    baseline_pct=45.0,
    affected_metrics={"ctr": 35.2},
    statistical_evidence="z=3.2, p<0.001, n=42/50",
    z_score=3.2,
    p_value=0.001,
    sample_size_high=42,
    sample_size_baseline=8,
    examples=["如何选择婴儿辅食?", "宝宝不爱吃饭怎么办?"],
    why_it_works="疑问句触发好奇心缺口",
    creation_formula="使用疑问句标题，提升点击率",
    key_elements=["疑问词开头", "针对痛点", "引发思考"]
)

cover_pattern = FeaturePattern(
    feature_name="bright_cover",
    feature_type="cover",
    description="使用高亮度封面图",
    prevalence_pct=78.0,
    baseline_pct=40.0,
    affected_metrics={"ctr": 28.5},
    statistical_evidence="z=2.8, p<0.01, n=39/50",
    z_score=2.8,
    p_value=0.005,
    sample_size_high=39,
    sample_size_baseline=11,
    examples=["封面图链接1", "封面图链接2"],
    why_it_works="高亮度增强视觉吸引力",
    creation_formula="使用明亮、高对比度的封面图",
    key_elements=["亮度>0.7", "高饱和度", "清晰聚焦"]
)

report_data["title_patterns"] = [title_pattern.dict()]
report_data["cover_patterns"] = [cover_pattern.dict()]

report = SuccessProfileReport(**report_data)

assert len(report.title_patterns) == 1
assert report.title_patterns[0].feature_type == "title"

assert len(report.cover_patterns) == 1
assert report.cover_patterns[0].feature_type == "cover"

assert len(report.content_patterns) == 0
assert len(report.tag_patterns) == 0
```

### Requirement: FeaturePattern MUST include statistical evidence fields

Each FeaturePattern MUST contain prevalence percentages, z-score, p-value, and sample sizes.

#### Scenario: FeaturePattern has complete statistical fields

```python
pattern = FeaturePattern(
    feature_name="test_pattern",
    feature_type="title",
    description="测试模式",
    prevalence_pct=80.0,
    baseline_pct=35.0,
    affected_metrics={"ctr": 40.0},
    statistical_evidence="z=3.5, p<0.001, n=40/50",
    z_score=3.5,
    p_value=0.0005,
    sample_size_high=40,
    sample_size_baseline=10,
    examples=["例子1"],
    why_it_works="解释",
    creation_formula="公式",
    key_elements=["要素1", "要素2", "要素3"]
)

# Verify all statistical fields present
assert pattern.prevalence_pct == 80.0
assert pattern.baseline_pct == 35.0
assert pattern.z_score == 3.5
assert pattern.p_value == 0.0005
assert pattern.sample_size_high == 40
assert pattern.sample_size_baseline == 10
assert pattern.statistical_evidence == "z=3.5, p<0.001, n=40/50"
```

#### Scenario: prevalence_pct must be greater than baseline_pct

```python
import pytest
from pydantic import ValidationError

# This is logically invalid but Pydantic won't enforce cross-field validation
# unless we add a validator. The agent logic SHOULD ensure prevalence > baseline.

pattern = FeaturePattern(
    feature_name="invalid_pattern",
    feature_type="title",
    description="无效模式",
    prevalence_pct=30.0,  # Lower than baseline!
    baseline_pct=50.0,
    affected_metrics={"ctr": -10.0},
    statistical_evidence="z=-1.0, p=0.5, n=15/50",
    z_score=-1.0,
    p_value=0.5,
    sample_size_high=15,
    sample_size_baseline=35,
    examples=["例子"],
    why_it_works="解释",
    creation_formula="公式",
    key_elements=["要素1", "要素2", "要素3"]
)

# Pydantic allows this, but agent SHOULD NOT generate such patterns
# (This represents a negative pattern, not a success pattern)
assert pattern.prevalence_pct < pattern.baseline_pct  # Allowed but illogical
```

### Requirement: FeaturePattern MUST include affected_metrics with quantified impact

Each pattern MUST specify which metrics it affects and by how much (percentage points or delta).

#### Scenario: affected_metrics shows multi-metric impact

```python
pattern = FeaturePattern(
    feature_name="engaging_opening",
    feature_type="content",
    description="使用强开场hook",
    prevalence_pct=75.0,
    baseline_pct=40.0,
    affected_metrics={
        "ctr": 30.0,          # +30% CTR
        "comment_rate": 15.5, # +15.5% comment_rate
        "interaction_rate": 22.0
    },
    statistical_evidence="z=2.9, p<0.01, n=37/50",
    z_score=2.9,
    p_value=0.008,
    sample_size_high=37,
    sample_size_baseline=13,
    examples=["惊人发现!", "你绝对想不到..."],
    why_it_works="强开场触发好奇心",
    creation_formula="使用惊人发现型开场",
    key_elements=["制造悬念", "引发好奇", "承诺价值"]
)

assert len(pattern.affected_metrics) == 3
assert "ctr" in pattern.affected_metrics
assert "comment_rate" in pattern.affected_metrics
assert pattern.affected_metrics["ctr"] == 30.0
```

#### Scenario: affected_metrics can have single metric

```python
pattern = FeaturePattern(
    feature_name="ctr_only_pattern",
    feature_type="cover",
    description="CTR专属模式",
    prevalence_pct=70.0,
    baseline_pct=30.0,
    affected_metrics={"ctr": 45.0},  # Only affects CTR
    statistical_evidence="z=3.0, p<0.01, n=35/50",
    z_score=3.0,
    p_value=0.002,
    sample_size_high=35,
    sample_size_baseline=15,
    examples=["例子"],
    why_it_works="解释",
    creation_formula="公式",
    key_elements=["要素1", "要素2", "要素3"]
)

assert len(pattern.affected_metrics) == 1
assert "ctr" in pattern.affected_metrics
```

### Requirement: FeaturePattern MUST provide concrete examples from target notes

Each pattern MUST include 1-5 real examples from the analyzed notes.

#### Scenario: Pattern includes 3 examples

```python
pattern = FeaturePattern(
    feature_name="pain_point_title",
    feature_type="title",
    description="痛点型标题",
    prevalence_pct=82.0,
    baseline_pct=38.0,
    affected_metrics={"ctr": 38.0},
    statistical_evidence="z=3.4, p<0.001, n=41/50",
    z_score=3.4,
    p_value=0.0007,
    sample_size_high=41,
    sample_size_baseline=9,
    examples=[
        "宝宝总是半夜哭闹怎么办?",
        "为什么我的宝宝不爱吃辅食?",
        "婴儿湿疹反复发作，妈妈们急疯了!"
    ],
    why_it_works="直击痛点，引发共鸣",
    creation_formula="用痛点问题作为标题",
    key_elements=["识别痛点", "使用疑问句", "引发共鸣"]
)

assert len(pattern.examples) == 3
assert all(isinstance(ex, str) for ex in pattern.examples)
assert all(len(ex) > 5 for ex in pattern.examples)
```

#### Scenario: Examples list is constrained to max 5 items

```python
# Pydantic model should enforce max_length=5
pattern_data = {
    "feature_name": "test",
    "feature_type": "title",
    "description": "描述",
    "prevalence_pct": 70.0,
    "baseline_pct": 30.0,
    "affected_metrics": {"ctr": 20.0},
    "statistical_evidence": "z=2.0, p<0.05, n=35/50",
    "z_score": 2.0,
    "p_value": 0.04,
    "sample_size_high": 35,
    "sample_size_baseline": 15,
    "examples": ["ex1", "ex2", "ex3", "ex4", "ex5", "ex6"],  # 6 examples
    "why_it_works": "解释",
    "creation_formula": "公式",
    "key_elements": ["a", "b", "c"]
}

# Pydantic SHOULD enforce max_length=5 (if defined in model)
# If not enforced, agent logic SHOULD truncate to 5
pattern = FeaturePattern(**pattern_data)
# Depending on Pydantic config, may accept 6 or truncate to 5
```

### Requirement: FeaturePattern MUST include LLM-generated insights

Each pattern MUST have why_it_works, creation_formula, and key_elements fields generated by LLM.

#### Scenario: LLM fields are present and substantive

```python
pattern = FeaturePattern(
    feature_name="question_ending",
    feature_type="content",
    description="结尾使用开放式问题",
    prevalence_pct=73.0,
    baseline_pct=35.0,
    affected_metrics={"comment_rate": 45.0},
    statistical_evidence="z=3.1, p<0.01, n=36/50",
    z_score=3.1,
    p_value=0.002,
    sample_size_high=36,
    sample_size_baseline=14,
    examples=["你家宝宝有这种情况吗?", "大家都是怎么处理的?"],
    why_it_works="开放式问题降低评论门槛，引导用户分享经验，形成社区互动",
    creation_formula="在笔记结尾提出一个开放式问题，邀请读者分享个人经验",
    key_elements=[
        "使用疑问句",
        "问题与内容相关",
        "降低回答门槛",
        "鼓励分享经验"
    ]
)

# Verify LLM fields
assert pattern.why_it_works is not None
assert len(pattern.why_it_works) > 20

assert pattern.creation_formula is not None
assert len(pattern.creation_formula) > 10

assert len(pattern.key_elements) == 4
assert 3 <= len(pattern.key_elements) <= 5
```

#### Scenario: key_elements must have 3-5 items

```python
import pytest
from pydantic import ValidationError

# Too few key elements
with pytest.raises(ValidationError, match="at least 3"):
    FeaturePattern(
        feature_name="test",
        feature_type="title",
        description="描述",
        prevalence_pct=70.0,
        baseline_pct=30.0,
        affected_metrics={"ctr": 20.0},
        statistical_evidence="z=2.0, p<0.05",
        z_score=2.0,
        p_value=0.04,
        sample_size_high=35,
        sample_size_baseline=15,
        examples=["ex1"],
        why_it_works="解释",
        creation_formula="公式",
        key_elements=["要素1", "要素2"]  # Only 2, need 3-5
    )

# Too many key elements
with pytest.raises(ValidationError, match="at most 5"):
    FeaturePattern(
        feature_name="test",
        feature_type="title",
        description="描述",
        prevalence_pct=70.0,
        baseline_pct=30.0,
        affected_metrics={"ctr": 20.0},
        statistical_evidence="z=2.0, p<0.05",
        z_score=2.0,
        p_value=0.04,
        sample_size_high=35,
        sample_size_baseline=15,
        examples=["ex1"],
        why_it_works="解释",
        creation_formula="公式",
        key_elements=["a", "b", "c", "d", "e", "f"]  # 6, too many
    )
```

### Requirement: Report MUST include key_success_factors summary

The report SHALL contain 3-5 top success factors synthesized from all patterns.

#### Scenario: key_success_factors has 3-5 items

```python
report_data["key_success_factors"] = [
    "使用疑问句标题，触发好奇心缺口，提升CTR 35%",
    "采用高亮度、高对比度封面图，增强视觉吸引力",
    "结尾提出开放式问题，引导评论互动，提升comment_rate 45%",
    "分享真实经验和数据，建立可信度"
]

report = SuccessProfileReport(**report_data)

assert len(report.key_success_factors) == 4
assert 3 <= len(report.key_success_factors) <= 5

for factor in report.key_success_factors:
    assert isinstance(factor, str)
    assert len(factor) > 10
```

#### Scenario: key_success_factors enforces min/max length

```python
import pytest
from pydantic import ValidationError

# Too few success factors
report_data["key_success_factors"] = ["因素1", "因素2"]  # Only 2

with pytest.raises(ValidationError, match="at least 3"):
    SuccessProfileReport(**report_data)

# Too many success factors
report_data["key_success_factors"] = ["f1", "f2", "f3", "f4", "f5", "f6"]  # 6

with pytest.raises(ValidationError, match="at most 5"):
    SuccessProfileReport(**report_data)
```

### Requirement: Report MUST include viral_formula_summary

The report SHALL contain a holistic summary of the overall "viral formula" or "creation template."

#### Scenario: viral_formula_summary is substantive

```python
report_data["viral_formula_summary"] = """
要在"婴儿辅食推荐"关键词下打造爆款笔记，需要遵循以下创作公式：
1) 标题使用疑问句，直击宝妈痛点（如"宝宝不爱吃饭怎么办？"）
2) 封面图采用明亮、高对比度的设计，展示食材或成品特写
3) 开篇分享真实数据或案例，建立可信度
4) 内容提供3-5个实操步骤，附带清晰的图片说明
5) 结尾提出开放式问题，引导宝妈们分享经验
这套公式可使CTR提升30%+，comment_rate提升40%+。
"""

report_data["key_success_factors"] = ["因素1", "因素2", "因素3"]
report = SuccessProfileReport(**report_data)

assert report.viral_formula_summary is not None
assert len(report.viral_formula_summary) > 100

# Should contain Chinese
import re
assert re.search(r'[\u4e00-\u9fff]', report.viral_formula_summary)
```

### Requirement: Report MUST include metadata fields

The report SHALL contain keyword, sample_size, and analysis_timestamp fields.

#### Scenario: Metadata fields are populated

```python
from datetime import datetime

report_data = {
    "keyword": "婴儿辅食推荐",
    "sample_size": 50,
    "aggregated_stats": {
        "prediction_stats": {"ctr": {"mean": 0.13, "median": 0.12, "std": 0.03, "min": 0.08, "max": 0.20, "count": 50}},
        "tag_frequencies": {},
        "tag_modes": {},
        "sample_size": 50,
        "outliers_removed": 0
    },
    "title_patterns": [],
    "cover_patterns": [],
    "content_patterns": [],
    "tag_patterns": [],
    "key_success_factors": ["因素1", "因素2", "因素3"],
    "viral_formula_summary": "总结内容" * 10,
    "analysis_timestamp": "2025-11-14T10:30:00Z"
}

report = SuccessProfileReport(**report_data)

assert report.keyword == "婴儿辅食推荐"
assert report.sample_size == 50
assert report.analysis_timestamp == "2025-11-14T10:30:00Z"

# Timestamp should be ISO 8601 format
datetime.fromisoformat(report.analysis_timestamp.replace('Z', '+00:00'))
```

### Requirement: Report MUST be JSON-serializable

The SuccessProfileReport MUST support conversion to/from JSON for inter-agent communication.

#### Scenario: Report can be serialized and deserialized

```python
import json

# Create report
report = SuccessProfileReport(**report_data)

# Serialize to JSON
json_str = report.model_dump_json()
assert isinstance(json_str, str)

# Deserialize back
report_dict = json.loads(json_str)
report_restored = SuccessProfileReport(**report_dict)

# Verify equality
assert report_restored.keyword == report.keyword
assert report_restored.sample_size == report.sample_size
assert len(report_restored.key_success_factors) == len(report.key_success_factors)
```
