# Tasks for Change 0005: Implement GapFinder Agent

## Task 0: Add current_metrics to AuditReport ✅
**Status:** pending
**Estimated time:** 15 min

### Subtasks:
1. Modify AuditReport in `src/xhs_seo_optimizer/models/reports.py`
   - Add field: `current_metrics: Dict[str, float] = Field(description="当前笔记的预测指标")`
   - This field will be populated by OwnedNoteAuditor from owned_note.prediction

2. Update `src/xhs_seo_optimizer/config/tasks_owned_note.yaml`
   - In `generate_audit_report` task description, add step to collect current_metrics
   - Instruct agent to extract from {current_metrics} variable (flattened by @before_kickoff)

3. Update `openspec/specs/owned-note-auditor/spec.md`
   - Verify requirement scenarios include current_metrics checks (already exist)

### Validation:
- AuditReport can be instantiated with current_metrics field
- Field validates as Dict[str, float]
- OwnedNoteAuditor test passes with current_metrics in output

---

## Task 1: Expand GapReport Model ✅
**Status:** pending
**Estimated time:** 30 min

### Subtasks:
1. Add MetricGap class to `src/xhs_seo_optimizer/models/reports.py`
   - Fields: metric_name, owned_value, target_mean, deltas, z_score, p_value, significance, priority_rank
   - Feature attribution: related_features, missing_features, weak_features
   - Narratives: gap_explanation, recommendation_summary
   - Add field validators (e.g., priority_rank > 0, significance in allowed values)

2. Expand GapReport class in same file
   - Replace placeholder fields
   - Add: significant_gaps, marginal_gaps, non_significant_gaps (List[MetricGap])
   - Add: top_priority_metrics (List[str], max 3 items)
   - Add: root_causes (List[str], 3-5 items)
   - Add: impact_summary (str, 100-200 chars)
   - Add: sample_size, gap_timestamp
   - Add field validators

3. Update `src/xhs_seo_optimizer/models/__init__.py`
   - Export MetricGap and updated GapReport

### Validation:
- MetricGap and GapReport can be instantiated with valid data
- Field validators reject invalid inputs
- Models are JSON-serializable

---

## Task 2: Create YAML Configurations ✅
**Status:** pending
**Estimated time:** 45 min

### Subtasks:
1. Create `src/xhs_seo_optimizer/config/agents_gap_finder.yaml`
   - Define gap_finder agent with role, goal, backstory
   - Use bilingual descriptions (Chinese + English)
   - Reference {keyword} placeholder

2. Create `src/xhs_seo_optimizer/config/tasks_gap_finder.yaml`
   - Task 1: calculate_statistical_gaps
     - Instructions to call StatisticalDeltaTool
     - Input: success_profile_report.aggregated_stats, audit_report.current_metrics
     - Output: GapAnalysis JSON with significant_gaps, priority_order
   - Task 2: map_gaps_to_features
     - Instructions to map each gap to features
     - Use success_profile_report.metric_profiles for relevant_features
     - Compare with audit_report text_features/visual_features
     - Output: Gaps with feature attribution
   - Task 3: prioritize_gaps
     - Use priority_order from Task 1
     - Identify top 3 metrics
     - Extract root causes from feature patterns
     - Generate impact_summary
   - Task 4: generate_gap_report
     - Integrate all results into GapReport
     - Set output_pydantic: GapReport

### Validation:
- YAML files are valid and parseable
- Task descriptions are clear and actionable
- Context dependencies are correct

---

## Task 3: Implement Crew Class ✅
**Status:** pending
**Estimated time:** 1 hour

### Subtasks:
1. Create `src/xhs_seo_optimizer/crew_gap_finder.py`
   - Implement XhsSeoOptimizerCrewGapFinder with @CrewBase
   - Load agents_gap_finder.yaml and tasks_gap_finder.yaml

2. Define @agent method
   - gap_finder() returns Agent with StatisticalDeltaTool

3. Define @task methods
   - calculate_statistical_gaps()
   - map_gaps_to_features() with context=[calculate_statistical_gaps]
   - prioritize_gaps() with context=[calculate_statistical_gaps, map_gaps_to_features]
   - generate_gap_report() with context=all, output_pydantic=GapReport

4. Define @crew method
   - Return Crew with all agents and tasks

5. Implement kickoff() method
   - Validate inputs (success_profile_report, audit_report, keyword required)
   - Parse JSON strings to dicts if needed
   - Store reports in shared_context
   - Execute crew().kickoff(inputs)
   - Call _save_gap_report(result)
   - Return result

6. Implement _save_gap_report() helper
   - Create outputs/ directory
   - Save result to outputs/gap_report.json
   - Handle Pydantic/JSON/raw formats

### Validation:
- Crew class can be instantiated
- kickoff() validates inputs correctly
- File is saved to outputs/gap_report.json

---

## Task 4: Write Test Suite ✅
**Status:** pending
**Estimated time:** 45 min

### Subtasks:
1. Create `tests/test_gap_finder.py`
   - Import XhsSeoOptimizerCrewGapFinder, GapReport

2. Write test_basic_execution()
   - Load success_profile_report.json and audit_report.json
   - Execute crew.kickoff()
   - Assert result.pydantic is GapReport
   - Verify outputs/gap_report.json exists

3. Write test_gap_report_schema()
   - Load gap_report.json
   - Parse as GapReport
   - Verify all required fields present
   - Check field constraints (top_priority_metrics <= 3, root_causes 3-5, etc.)

4. Write test_feature_attribution()
   - Load gap_report.json
   - For each significant_gap, verify:
     - related_features is not empty
     - gap_explanation is substantial (>20 chars)
     - recommendation_summary exists

5. Write test_priority_ordering()
   - Load gap_report.json
   - Verify priority_rank values are sequential
   - Verify top_priority_metrics match highest-ranked gaps

### Validation:
- All tests pass
- Tests cover key functionality
- Edge cases are handled

---

## Task 5: Create Specification ✅
**Status:** pending
**Estimated time:** 30 min

### Subtasks:
1. Create `openspec/changes/0005-implement-gap-finder/specs/gap-finder/spec.md`
   - Write Purpose section
   - Define 6-8 requirements with test scenarios
   - Follow format from owned-note-auditor/spec.md

2. Key requirements to document:
   - Agent SHALL calculate statistical gaps using StatisticalDeltaTool
   - Agent SHALL map gaps to specific content features
   - Agent SHALL prioritize gaps by combined significance and magnitude
   - Agent SHALL identify cross-metric root causes
   - Agent SHALL output GapReport conforming to schema
   - Agent SHALL save output to outputs/gap_report.json

3. For each requirement, write 2-3 test scenarios
   - Use executable Python code blocks
   - Include both happy path and edge cases

### Validation:
- Spec document is complete
- Test scenarios are clear and executable
- Requirements are testable

---

## Task 6: Integration Testing ✅
**Status:** pending
**Estimated time:** 30 min

### Subtasks:
1. Generate prerequisite data files
   - Run CompetitorAnalyst test → outputs/success_profile_report.json
   - Run OwnedNoteAuditor test → outputs/audit_report.json
   - Verify both files exist and are valid JSON

2. Run GapFinder test
   - Execute pytest tests/test_gap_finder.py -v
   - Verify outputs/gap_report.json is created
   - Inspect gap_report.json for quality
     - Are significant gaps correctly identified?
     - Do feature attributions make sense?
     - Are priority rankings reasonable?

3. Manual review
   - Read impact_summary - is it coherent?
   - Check root_causes - are they actionable?
   - Verify top_priority_metrics align with data

4. Edge case testing
   - Test with owned_note that has no significant gaps
   - Test with owned_note missing some metrics
   - Test with very small target_notes sample

### Validation:
- All integration tests pass
- Output quality is acceptable
- Edge cases are handled gracefully

---

## Task 7: Documentation and Cleanup ✅
**Status:** pending
**Estimated time:** 15 min

### Subtasks:
1. Update README or docs with GapFinder usage example
2. Add docstrings to crew_gap_finder.py methods
3. Add inline comments for complex logic
4. Verify all files follow project conventions (project.md)

### Validation:
- Documentation is clear
- Code is well-commented
- Follows project style guide

---

## Total Estimated Time: ~4.25 hours

## Dependencies
- Task 0 must be completed first (AuditReport needs current_metrics)
- Task 2 depends on Tasks 0 and 1 (needs GapReport model and current_metrics)
- Task 3 depends on Task 2 (needs YAML configs)
- Task 4 depends on Tasks 0-3 (needs models and configs)
- Task 6 depends on Tasks 0, 3-5 (needs implementation + tests + updated AuditReport)

## Parallel Work Opportunities
- Task 0 should be done first (prerequisite)
- Tasks 1, 2, 5 can be done in parallel after Task 0
- Tasks 4 can start once Task 3 is partially complete
- Task 7 can be done incrementally throughout
