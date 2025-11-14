# Implementation Tasks

## Overview
This document breaks down the implementation of CompetitorAnalyst agent and CrewAI infrastructure into granular, verifiable tasks. Tasks are ordered to deliver incremental user-visible progress.

---

## Phase 1: Data Models (Foundation)

### Task 1.1: Create FeaturePattern model
**File**: `src/xhs_seo_optimizer/models/reports.py`

**Description**: Define the FeaturePattern Pydantic model for representing content patterns with statistical evidence.

**Acceptance Criteria**:
- [ ] Model has all required fields: `feature_name`, `feature_type`, `description`, `prevalence_pct`, `baseline_pct`, `affected_metrics`, `statistical_evidence`, `z_score`, `p_value`, `sample_size_high`, `sample_size_baseline`, `examples`, `why_it_works`, `creation_formula`, `key_elements`
- [ ] Field validation enforces: 0 ≤ prevalence_pct ≤ 100, 0 ≤ p_value ≤ 1, 1 ≤ len(examples) ≤ 5, 3 ≤ len(key_elements) ≤ 5
- [ ] Model is JSON-serializable (test with `.model_dump_json()`)
- [ ] Docstrings in Chinese + English

**Estimated Effort**: 1 hour

---

### Task 1.2: Create SuccessProfileReport model
**File**: `src/xhs_seo_optimizer/models/reports.py`

**Description**: Define the SuccessProfileReport Pydantic model for CompetitorAnalyst output.

**Acceptance Criteria**:
- [ ] Model has all required fields: `keyword`, `sample_size`, `aggregated_stats`, `title_patterns`, `cover_patterns`, `content_patterns`, `tag_patterns`, `key_success_factors`, `viral_formula_summary`, `analysis_timestamp`
- [ ] Validation enforces: 3 ≤ len(key_success_factors) ≤ 5, len(viral_formula_summary) > 50
- [ ] `aggregated_stats` field type is `AggregatedMetrics` (from `models/analysis_results.py`)
- [ ] Model can be instantiated from dict and serialized to JSON
- [ ] Unit test verifies schema compliance

**Estimated Effort**: 1 hour

---

### Task 1.3: Create placeholder report models
**File**: `src/xhs_seo_optimizer/models/reports.py`

**Description**: Define placeholder models for future agents (AuditReport, GapReport, OptimizationPlan).

**Acceptance Criteria**:
- [ ] `AuditReport` model with minimal fields (to be expanded in future change)
- [ ] `GapReport` model with minimal fields
- [ ] `OptimizationPlan` model with minimal fields
- [ ] All models inherit from `BaseModel` and are importable
- [ ] Docstrings explain "placeholder for future implementation"

**Estimated Effort**: 30 minutes

---

### Task 1.4: Update models __init__.py
**File**: `src/xhs_seo_optimizer/models/__init__.py`

**Description**: Export new report models for easy import.

**Acceptance Criteria**:
- [ ] `from .reports import FeaturePattern, SuccessProfileReport, AuditReport, GapReport, OptimizationPlan` added
- [ ] `__all__` list updated
- [ ] Import test passes: `from xhs_seo_optimizer.models import SuccessProfileReport`

**Estimated Effort**: 5 minutes

---

## Phase 2: CrewAI Configuration (Infrastructure)

### Task 2.1: Create agents.yaml
**File**: `src/xhs_seo_optimizer/config/agents.yaml`

**Description**: Define YAML configuration for all 5 agents.

**Acceptance Criteria**:
- [ ] Orchestrator, CompetitorAnalyst, OwnedNoteAuditor, GapFinder, OptimizationStrategist defined
- [ ] Each agent has: `role`, `goal`, `backstory`, `tools`, `llm`
- [ ] CompetitorAnalyst tools list: `[DataAggregatorTool, MultiModalVisionTool, NLPAnalysisTool]`
- [ ] All roles and backstories are bilingual (Chinese + English)
- [ ] YAML is valid (test with `yaml.safe_load()`)

**Estimated Effort**: 1.5 hours

---

### Task 2.2: Create tasks.yaml
**File**: `src/xhs_seo_optimizer/config/tasks.yaml`

**Description**: Define YAML configuration for all 5 tasks.

**Acceptance Criteria**:
- [ ] orchestrate_workflow, analyze_competitors, audit_owned_note, find_gaps, generate_strategy defined
- [ ] Each task has: `description`, `expected_output`, `agent`
- [ ] analyze_competitors has: `output_file: "outputs/success_profile_report.json"`, `context: []`
- [ ] find_gaps has: `context: [analyze_competitors, audit_owned_note]` (task dependencies)
- [ ] YAML is valid

**Estimated Effort**: 1 hour

---

### Task 2.3: Create XhsSeoOptimizerCrew class scaffold
**File**: `src/xhs_seo_optimizer/crew.py`

**Description**: Implement the base Crew class with decorators but minimal logic.

**Acceptance Criteria**:
- [ ] Class inherits from `CrewBase`
- [ ] `agents_config = 'config/agents.yaml'` and `tasks_config = 'config/tasks.yaml'` attributes set
- [ ] `@agent` methods for all 5 agents (return `Agent` instances)
- [ ] `@task` methods for all 5 tasks (return `Task` instances)
- [ ] `@crew` method returns `Crew` instance with `process=Process.sequential`
- [ ] Placeholder implementations for non-CompetitorAnalyst agents (no logic yet)
- [ ] Class can be instantiated without errors

**Estimated Effort**: 2 hours

---

### Task 2.4: Implement competitor_analyst() agent method
**File**: `src/xhs_seo_optimizer/crew.py`

**Description**: Fully implement the competitor_analyst agent with tools assigned.

**Acceptance Criteria**:
- [ ] Method loads config from `self.agents_config['competitor_analyst']`
- [ ] Instantiates and assigns tools: `DataAggregatorTool()`, `NLPAnalysisTool()`, `MultiModalVisionTool()`
- [ ] Returns `Agent` instance with tools attached
- [ ] Agent can be executed (test with dummy task)

**Estimated Effort**: 30 minutes

---

### Task 2.5: Implement analyze_competitors_task() method
**File**: `src/xhs_seo_optimizer/crew.py`

**Description**: Implement the analyze_competitors task method.

**Acceptance Criteria**:
- [ ] Method loads config from `self.tasks_config['analyze_competitors']`
- [ ] Returns `Task` instance with agent set to `self.competitor_analyst()`
- [ ] Task has expected_output defined
- [ ] Can be added to crew.tasks list

**Estimated Effort**: 20 minutes

---

## Phase 3: CompetitorAnalyst Logic (Core Implementation)

### Task 3.1: Implement METRIC_FEATURE_ATTRIBUTION mapping
**File**: `src/xhs_seo_optimizer/crew.py` (or new `src/xhs_seo_optimizer/attribution.py`)

**Description**: Define the attribution rules mapping metrics to relevant features.

**Acceptance Criteria**:
- [ ] Dictionary constant with keys: `ctr`, `comment_rate`, `interaction_rate`, `like_rate`, `share_rate`, `follow_rate`, `sort_score2`
- [ ] Each value is a dict with `features` (list) and `rationale` (string)
- [ ] Example: `"ctr": {"features": ["title_pattern", "cover_quality", ...], "rationale": "用户点击前仅能看到标题和封面"}`
- [ ] Comprehensive coverage of all 10 prediction metrics

**Estimated Effort**: 1 hour

---

### Task 3.2: Implement aggregate_statistics() helper
**File**: `src/xhs_seo_optimizer/crew.py`

**Description**: Helper method to run DataAggregatorTool and get baseline stats.

**Acceptance Criteria**:
- [ ] Function signature: `def aggregate_statistics(notes: List[Note]) -> AggregatedMetrics`
- [ ] Instantiates DataAggregatorTool
- [ ] Calls `tool._run(notes=notes)`
- [ ] Returns AggregatedMetrics object
- [ ] Unit test with 3 sample notes

**Estimated Effort**: 30 minutes

---

### Task 3.3: Implement extract_features_matrix() helper
**File**: `src/xhs_seo_optimizer/crew.py`

**Description**: Extract NLP and Vision features for all notes in parallel.

**Acceptance Criteria**:
- [ ] Function signature: `def extract_features_matrix(notes: List[Note]) -> Dict[str, Dict]`
- [ ] For each note, run NLPAnalysisTool and MultiModalVisionTool
- [ ] Returns: `{note_id: {"text": TextAnalysisResult, "vision": VisionAnalysisResult, "prediction": NotePrediction, "tag": NoteTag}}`
- [ ] Gracefully handles missing images (log warning, set vision=None)
- [ ] Optional: Use ThreadPoolExecutor for parallelization
- [ ] Unit test with 2 notes

**Estimated Effort**: 2 hours

---

### Task 3.4: Implement identify_patterns() helper (Layer 1 & 2)
**File**: `src/xhs_seo_optimizer/crew.py`

**Description**: Identify statistically significant patterns using attribution rules and prevalence analysis.

**Acceptance Criteria**:
- [ ] Function signature: `def identify_patterns(notes: List[Note], features_matrix: Dict, aggregated_stats: AggregatedMetrics) -> List[FeaturePattern]`
- [ ] For each metric:
  - [ ] Apply attribution rules to filter relevant features (Layer 1)
  - [ ] Segment notes into top 25% (high) vs rest (baseline) by metric value
  - [ ] Calculate pattern prevalence in each group
  - [ ] Compute z-score and p-value for prevalence difference
  - [ ] Filter patterns: prevalence_pct ≥ 70% AND p_value < 0.05
- [ ] Returns list of FeaturePattern objects (without LLM fields yet)
- [ ] Unit test with synthetic data

**Estimated Effort**: 4 hours

---

### Task 3.5: Implement synthesize_formulas() helper (Layer 3)
**File**: `src/xhs_seo_optimizer/crew.py`

**Description**: Use LLM to generate explanations and creation formulas for each pattern.

**Acceptance Criteria**:
- [ ] Function signature: `def synthesize_formulas(patterns: List[FeaturePattern], notes: List[Note]) -> List[FeaturePattern]`
- [ ] For each pattern:
  - [ ] Build LLM prompt with statistical evidence and examples
  - [ ] Call DeepSeek LLM (via OpenRouter)
  - [ ] Parse response to get `why_it_works`, `creation_formula`, `key_elements`
  - [ ] Update FeaturePattern object
- [ ] Handle LLM failures gracefully (fallback to template)
- [ ] Test with 1 sample pattern

**Estimated Effort**: 3 hours

---

### Task 3.6: Implement generate_summary_insights() helper
**File**: `src/xhs_seo_optimizer/crew.py`

**Description**: Generate key_success_factors and viral_formula_summary using LLM.

**Acceptance Criteria**:
- [ ] Function signature: `def generate_summary_insights(patterns: List[FeaturePattern]) -> Tuple[List[str], str]`
- [ ] Select top 3-5 patterns by combined impact (z_score * delta_pct)
- [ ] Build LLM prompt requesting holistic summary
- [ ] Parse response to get `key_success_factors` (list) and `viral_formula_summary` (string)
- [ ] Validate: 3 ≤ len(key_success_factors) ≤ 5, len(summary) > 100
- [ ] Test with 5 sample patterns

**Estimated Effort**: 2 hours

---

### Task 3.7: Integrate all helpers into CompetitorAnalyst.execute()
**File**: `src/xhs_seo_optimizer/crew.py`

**Description**: Orchestrate all helper methods into the main agent execution flow.

**Acceptance Criteria**:
- [ ] Function signature: `def execute_competitor_analyst(target_notes: List[Note], keyword: str) -> str`
- [ ] Step 1: `aggregated_stats = aggregate_statistics(target_notes)`
- [ ] Step 2: `features_matrix = extract_features_matrix(target_notes)`
- [ ] Step 3: `patterns = identify_patterns(notes, features_matrix, aggregated_stats)`
- [ ] Step 4: `patterns = synthesize_formulas(patterns, notes)`
- [ ] Step 5: `key_factors, summary = generate_summary_insights(patterns)`
- [ ] Step 6: Organize patterns by feature_type into title/cover/content/tag lists
- [ ] Step 7: Create SuccessProfileReport object
- [ ] Step 8: Return `report.model_dump_json()`
- [ ] Error handling: ValueError for empty notes, log warnings for missing data
- [ ] Integration test with real target_notes.json

**Estimated Effort**: 2 hours

---

## Phase 4: Testing (Validation)

### Task 4.1: Write unit tests for data models
**File**: `tests/test_models/test_reports.py`

**Description**: Test FeaturePattern and SuccessProfileReport schema validation.

**Acceptance Criteria**:
- [ ] Test FeaturePattern field validation (prevalence range, key_elements length, etc.)
- [ ] Test SuccessProfileReport schema compliance
- [ ] Test JSON serialization/deserialization
- [ ] Test edge cases (min/max values)
- [ ] All tests pass

**Estimated Effort**: 1.5 hours

---

### Task 4.2: Write unit tests for attribution logic
**File**: `tests/test_agents/test_attribution.py`

**Description**: Test METRIC_FEATURE_ATTRIBUTION filtering.

**Acceptance Criteria**:
- [ ] Test that CTR analysis only considers title/cover features
- [ ] Test that comment_rate analysis only considers content/ending features
- [ ] Test all 7+ metrics have valid attribution rules
- [ ] All tests pass

**Estimated Effort**: 1 hour

---

### Task 4.3: Write unit tests for helper functions
**File**: `tests/test_agents/test_competitor_analyst_helpers.py`

**Description**: Test individual helper methods in isolation.

**Acceptance Criteria**:
- [ ] Test `aggregate_statistics()` with sample notes
- [ ] Test `extract_features_matrix()` with 2 notes (1 with image, 1 without)
- [ ] Test `identify_patterns()` with synthetic data (known prevalence)
- [ ] Test `synthesize_formulas()` with mock LLM response
- [ ] All tests pass

**Estimated Effort**: 3 hours

---

### Task 4.4: Write integration test with real data
**File**: `tests/test_agents/test_competitor_analyst.py`

**Description**: End-to-end test with real target_notes.json.

**Acceptance Criteria**:
- [ ] Load real target_notes from `docs/target_notes.json`
- [ ] Execute CompetitorAnalyst agent
- [ ] Verify SuccessProfileReport output:
  - [ ] All required fields present
  - [ ] At least 1 pattern discovered (if data allows)
  - [ ] Statistical evidence included
  - [ ] LLM formulas in Chinese
  - [ ] key_success_factors has 3-5 items
- [ ] Test passes with 100% schema compliance
- [ ] Print detailed report for manual review

**Estimated Effort**: 2 hours

---

### Task 4.5: Write Crew infrastructure tests
**File**: `tests/test_crew/test_crew_infrastructure.py`

**Description**: Test YAML configuration loading and Crew class instantiation.

**Acceptance Criteria**:
- [ ] Test agents.yaml loads successfully and has all 5 agents
- [ ] Test tasks.yaml loads successfully and has all 5 tasks
- [ ] Test XhsSeoOptimizerCrew() instantiation
- [ ] Test competitor_analyst() returns Agent with tools
- [ ] Test analyze_competitors_task() returns Task
- [ ] All tests pass

**Estimated Effort**: 1.5 hours

---

## Phase 5: Documentation & Polish

### Task 5.1: Create CLI entry point (main.py)
**File**: `src/xhs_seo_optimizer/main.py`

**Description**: Implement command-line interface for running analysis.

**Acceptance Criteria**:
- [ ] Uses `argparse` for argument parsing
- [ ] Arguments: `--keyword`, `--target-notes`, `--owned-note`, `--output`
- [ ] Loads notes from JSON files
- [ ] Instantiates XhsSeoOptimizerCrew and runs analyze_competitors
- [ ] Saves SuccessProfileReport to output file
- [ ] Provides `--help` documentation
- [ ] Error handling for missing files
- [ ] Test: `python main.py --keyword "测试" --target-notes docs/target_notes.json --output outputs/report.json`

**Estimated Effort**: 2 hours

---

### Task 5.2: Create example usage documentation
**File**: `docs/competitor_analyst_usage.md`

**Description**: Document how to use CompetitorAnalyst agent.

**Acceptance Criteria**:
- [ ] Installation instructions (dependencies)
- [ ] Example CLI usage with explanations
- [ ] Example Python API usage
- [ ] Explanation of SuccessProfileReport fields
- [ ] Troubleshooting section (common errors)
- [ ] Screenshots or example outputs

**Estimated Effort**: 1.5 hours

---

### Task 5.3: Add logging and progress indicators
**File**: `src/xhs_seo_optimizer/crew.py`

**Description**: Improve user experience with structured logging.

**Acceptance Criteria**:
- [ ] Log: "Aggregating statistics for {n} notes..."
- [ ] Log: "Extracting features from note {i}/{n}..."
- [ ] Log: "Identified {n} significant patterns"
- [ ] Log: "Synthesizing formulas with LLM..."
- [ ] Use `logging` module (INFO level)
- [ ] Optional: Progress bar with `tqdm` for note processing

**Estimated Effort**: 1 hour

---

## Phase 6: Validation & Cleanup

### Task 6.1: Run openspec validate --strict
**Command**: `openspec validate 0003 --strict`

**Description**: Ensure proposal passes all OpenSpec validation rules.

**Acceptance Criteria**:
- [ ] No MUST/SHALL requirement violations
- [ ] All scenarios are testable
- [ ] Proposal/design/tasks files are complete
- [ ] No validation errors or warnings

**Estimated Effort**: 30 minutes (+ fixes if needed)

---

### Task 6.2: Run all tests and verify 100% pass rate
**Command**: `pytest tests/test_agents/ tests/test_models/ -v`

**Description**: Execute full test suite.

**Acceptance Criteria**:
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Test coverage ≥ 80% for new code (optional but recommended)
- [ ] No flaky tests

**Estimated Effort**: 30 minutes (+ fixes if needed)

---

### Task 6.3: Manual review of generated formulas
**Description**: Human review of LLM-generated creation formulas for quality.

**Acceptance Criteria**:
- [ ] Run CompetitorAnalyst on real data
- [ ] Review top 5 patterns:
  - [ ] Formulas are in Chinese
  - [ ] Formulas are actionable (not generic)
  - [ ] Statistical evidence is accurate
  - [ ] Examples are relevant
- [ ] Document any quality issues for future improvement

**Estimated Effort**: 1 hour

---

### Task 6.4: Performance profiling (optional)
**Description**: Measure execution time and identify bottlenecks.

**Acceptance Criteria**:
- [ ] Run analysis on 50 notes and measure total time
- [ ] Identify slowest steps (likely: NLP/Vision tool calls)
- [ ] Document performance baseline
- [ ] Optional: Implement caching or parallelization improvements

**Estimated Effort**: 1 hour (optional)

---

## Dependency Graph

```
Phase 1 (Data Models) ─┐
                       ├─> Phase 2 (Configuration) ─┐
                       └─────────────────────────────├─> Phase 3 (Logic) ─> Phase 4 (Testing) ─> Phase 5 (Docs) ─> Phase 6 (Validation)
```

**Critical Path**: Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 6

**Parallelizable**:
- Task 1.1-1.3 can be done in parallel
- Task 2.1-2.2 can be done in parallel
- Task 4.1-4.3 can be done in parallel
- Phase 5 can start once Phase 4 is complete

---

## Estimated Total Effort

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| Phase 1: Data Models | 4 | 2.5 hours |
| Phase 2: Configuration | 5 | 5 hours |
| Phase 3: Core Logic | 7 | 16 hours |
| Phase 4: Testing | 5 | 9 hours |
| Phase 5: Documentation | 3 | 4.5 hours |
| Phase 6: Validation | 4 | 2.5 hours |
| **Total** | **28 tasks** | **~40 hours** |

**Note**: Estimates are for an experienced developer. Actual time may vary based on debugging, iteration, and LLM prompt engineering.

---

## Success Criteria

✅ All 28 tasks completed
✅ All tests pass (100% pass rate)
✅ `openspec validate 0003 --strict` passes
✅ CompetitorAnalyst generates SuccessProfileReport from real data
✅ Generated formulas are in Chinese and actionable
✅ Full CrewAI infrastructure supports future agent additions
✅ Documentation is complete and clear
