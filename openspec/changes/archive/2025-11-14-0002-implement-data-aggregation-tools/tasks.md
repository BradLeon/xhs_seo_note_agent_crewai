# Tasks for 0002-implement-data-aggregation-tools

## Phase 1: Setup and Models (Tasks 1-2)

### Task 1: Create AggregatedMetrics and GapAnalysis models
**Status**: pending
**Priority**: high
**Dependencies**: None

Create Pydantic models in `src/xhs_seo_optimizer/models/analysis_results.py`:
- `MetricStats` model (mean, median, std, min, max, count)
- `AggregatedMetrics` model (prediction_stats, tag_frequencies, tag_modes, sample_size, outliers_removed)
- `Gap` model (metric, owned_value, target_mean, delta_pct, z_score, p_value, significance, interpretation)
- `GapAnalysis` model (significant_gaps, non_significant_gaps, priority_order, sample_size)

**Validation**:
- [ ] Models validate with Pydantic
- [ ] All fields have proper types and descriptions
- [ ] Can serialize/deserialize to/from JSON

### Task 2: Add numpy and scipy dependencies
**Status**: pending
**Priority**: high
**Dependencies**: None

Update `pyproject.toml` to include:
```toml
dependencies = [
    ...
    "numpy>=1.24.0",
    "scipy>=1.10.0",
]
```

**Validation**:
- [ ] `pip install -e .` succeeds
- [ ] Can import numpy and scipy.stats

## Phase 2: DataAggregator Tool (Tasks 3-6)

### Task 3: Implement DataAggregatorTool skeleton
**Status**: pending
**Priority**: high
**Dependencies**: Task 1

Create `src/xhs_seo_optimizer/tools/data_aggregator.py`:
- Tool class inheriting from `BaseTool`
- Input schema with Pydantic (notes list, optional remove_outliers flag)
- `_run()` method signature
- Name and description fields

**Validation**:
- [ ] Tool can be instantiated
- [ ] Tool can be added to CrewAI agent
- [ ] Calling `_run()` doesn't crash (even if returns placeholder)

### Task 4: Implement prediction metrics aggregation
**Status**: pending
**Priority**: high
**Dependencies**: Task 3

Add statistical calculation logic:
- Extract prediction values from list of notes
- Calculate mean, median, std, min, max using numpy
- Handle missing values gracefully
- Return MetricStats for each prediction field

**Validation**:
- [ ] Correct stats for CTR across 3 notes
- [ ] All 10 prediction metrics computed
- [ ] Handles missing fields without crashing

### Task 5: Implement tag frequency and mode calculation
**Status**: pending
**Priority**: high
**Dependencies**: Task 3

Add tag aggregation logic:
- Count frequencies using Python Counter
- Calculate mode (most common value)
- Handle ties deterministically (alphabetical order)
- Process all tag fields (intention_lv1/2, taxonomy1/2/3, note_marketing_integrated_level)

**Validation**:
- [ ] Frequency counts correct
- [ ] Mode selection correct for single mode
- [ ] Deterministic mode selection for ties

### Task 6: Implement outlier removal (optional)
**Status**: pending
**Priority**: medium
**Dependencies**: Task 4

Add outlier detection and removal:
- Calculate z-scores for each value
- Remove values beyond threshold (default: ±2σ)
- Track number of outliers removed
- Recompute stats after removal

**Validation**:
- [ ] Outliers correctly identified and removed
- [ ] Stats recomputed after removal
- [ ] outliers_removed count accurate
- [ ] Works when remove_outliers=False (default)

## Phase 3: StatisticalDelta Tool (Tasks 7-10)

### Task 7: Implement StatisticalDeltaTool skeleton
**Status**: pending
**Priority**: high
**Dependencies**: Task 1

Create `src/xhs_seo_optimizer/tools/statistical_delta.py`:
- Tool class inheriting from `BaseTool`
- Input schema (owned_note, target_stats AggregatedMetrics)
- `_run()` method signature
- Name and description fields

**Validation**:
- [ ] Tool can be instantiated
- [ ] Tool can be added to CrewAI agent
- [ ] Calling `_run()` doesn't crash

### Task 8: Implement z-score and p-value calculation
**Status**: pending
**Priority**: high
**Dependencies**: Task 7

Add statistical significance testing:
- Calculate z-score: `(owned_value - target_mean) / target_std`
- Calculate p-value using `scipy.stats.norm.sf()` (two-tailed)
- Handle edge cases (std=0, both values zero)
- Compute for all prediction metrics

**Validation**:
- [ ] Z-scores match manual calculation
- [ ] P-values match scipy.stats results
- [ ] Edge cases handled gracefully

### Task 9: Implement significance classification and prioritization
**Status**: pending
**Priority**: high
**Dependencies**: Task 8

Add gap classification and sorting:
- Classify significance (critical/very_significant/significant/marginal/none)
- Separate gaps into significant and non-significant lists
- Calculate priority score: `abs(z_score) * abs(delta_pct)`
- Sort priority_order by priority score

**Validation**:
- [ ] Significance levels correctly classified
- [ ] Gaps properly separated by p-value threshold
- [ ] Priority order sorted correctly

### Task 10: Implement interpretation generation
**Status**: pending
**Priority**: medium
**Dependencies**: Task 9

Add human-readable interpretations:
- Template-based interpretation strings
- Include metric name, significance level, z-score, percentage delta
- Handle special cases (undefined, missing data)

**Validation**:
- [ ] Interpretations are clear and accurate
- [ ] All key information included (metric, significance, magnitude)
- [ ] Special cases have appropriate messages

## Phase 4: Testing (Tasks 11-14)

### Task 11: Write DataAggregator unit tests
**Status**: pending
**Priority**: high
**Dependencies**: Task 6

Create `tests/test_tools/test_data_aggregator.py`:
- Test basic aggregation (mean, median, std)
- Test tag frequency and mode calculation
- Test edge cases (empty list, single note, all identical)
- Test outlier removal
- Test with real data from docs/target_notes.json

**Validation**:
- [ ] All scenarios from spec pass
- [ ] Test coverage >90%
- [ ] Tests pass with pytest

### Task 12: Write StatisticalDelta unit tests
**Status**: pending
**Priority**: high
**Dependencies**: Task 10

Create `tests/test_tools/test_statistical_delta.py`:
- Test z-score and p-value calculation
- Test significance classification
- Test priority ordering
- Test edge cases (std=0, missing metrics)
- Test with real data

**Validation**:
- [ ] All scenarios from spec pass
- [ ] Test coverage >90%
- [ ] Tests pass with pytest

### Task 13: Integration test with both tools
**Status**: pending
**Priority**: medium
**Dependencies**: Tasks 11, 12

Create integration test demonstrating workflow:
- DataAggregator aggregates target_notes
- StatisticalDelta compares owned_note against aggregates
- Verify end-to-end data flow

**Validation**:
- [ ] Workflow completes successfully
- [ ] Output from DataAggregator feeds into StatisticalDelta
- [ ] Results are sensible for real data

### Task 14: Update documentation
**Status**: pending
**Priority**: low
**Dependencies**: Task 13

Update project documentation:
- Add tool descriptions to TOOLS_README.md
- Add usage examples to docstrings
- Update project.md with tool details (if needed)

**Validation**:
- [ ] Documentation is clear and accurate
- [ ] Code examples work
- [ ] Docstrings follow Google style

## Execution Strategy

**Parallel Work Opportunities**:
- Tasks 3-6 (DataAggregator) and Tasks 7-10 (StatisticalDelta) can be developed in parallel after Task 1-2
- Task 11 and Task 12 (testing) can be written in parallel

**Critical Path**:
1. Task 1 (Models) → Task 2 (Dependencies)
2. Task 3-6 (DataAggregator implementation)
3. Task 7-10 (StatisticalDelta implementation)
4. Task 11-12 (Testing)
5. Task 13-14 (Integration and docs)

**Estimated Effort**:
- Phase 1: 1-2 hours
- Phase 2: 3-4 hours
- Phase 3: 3-4 hours
- Phase 4: 2-3 hours
- **Total: 9-13 hours**

**Risk Mitigation**:
- Start with Task 1 (models) to define clear contracts
- Write tests early (Tasks 11-12) to catch issues
- Test with real data from docs/ to ensure correctness
- Handle edge cases explicitly (std=0, missing data, etc.)
