# Implementation Tasks

## Task List

### 1. Create new branch
- Create `feature/owned-note-auditor` branch from current `refactor/agent-orchestrator-separation`
- **Validation**: `git branch --show-current` shows new branch

### 2. Extend AuditReport model in reports.py
- Read current placeholder AuditReport class
- Add complete field definitions with Field descriptors
- Include nested models: TextAnalysisResult, VisionAnalysisResult
- Add validators for required fields (note_id, timestamp, etc.)
- Add model_dump() serialization support
- **Validation**: Model instantiates without errors, passes Pydantic validation

### 3. Create crew_owned_note.py
- Copy structure from crew_simple.py
- Rename class to XhsSeoOptimizerCrewOwnedNote
- Update agents_config/tasks_config paths to owned_note variants
- Implement @before_kickoff hook:
  - Validate owned_note input (note_id, prediction, meta_data required)
  - Serialize Note object to dict if needed
  - Store in shared_context["owned_note_data"]
  - Return flattened dict for YAML variables
- Implement @agent owned_note_auditor:
  - Tools: [MultiModalVisionTool(), NLPAnalysisTool()] (no DataAggregatorTool)
  - Same LLM config as CompetitorAnalyst
- Implement 4 @task methods:
  - extract_content_features_task()
  - analyze_metric_performance_task()
  - identify_weaknesses_task()
  - generate_audit_report_task()
- Chain tasks with context parameter
- **Validation**: File imports without errors, class structure matches @CrewBase pattern

### 4. Create config/agents_owned_note.yaml
- Define owned_note_auditor agent
- Role: "自营笔记审计员 (Owned Note Auditor)"
- Goal: Use {keyword} and {note_id} variables
- Backstory: Bilingual (Chinese primary, English supplementary)
- Follow project conventions from openspec/project.md
- **Validation**: YAML parses correctly, follows bilingual pattern

### 5. Create config/tasks_owned_note.yaml
- Define 4 tasks matching task methods:
  1. extract_content_features:
     - Description: Call NLP + Vision tools with note_id
     - Expected output: "Text and visual features extracted"
     - No output_file (intermediate task)
  2. analyze_metric_performance:
     - Description: Analyze 10 prediction metrics, identify weak/strong
     - Expected output: "Metric performance analysis with weak_metrics list"
     - Context: [extract_content_features]
  3. identify_weaknesses:
     - Description: Compare features against best practices
     - Expected output: "Content weaknesses and strengths lists"
     - Context: [extract_content_features, analyze_metric_performance]
  4. generate_audit_report:
     - Description: Compile final AuditReport
     - Expected output: "Complete AuditReport JSON"
     - Output file: "outputs/audit_report.json"
     - Context: [all previous tasks]
- Include 禁止事项 sections
- **Validation**: YAML parses correctly, task names match @task methods

### 6. Write integration test test_owned_note_auditor.py
- Import XhsSeoOptimizerCrewOwnedNote
- Load docs/owned_note.json as test data
- Test scenario 1: Basic execution
  - kickoff() completes without errors
  - AuditReport has all required fields
- Test scenario 2: Feature extraction validation
  - text_features and visual_features are populated
  - Features match expected structure
- Test scenario 3: Weakness identification
  - weak_metrics list is non-empty (if metrics are actually weak)
  - content_weaknesses list contains actionable items
- Test scenario 4: Edge case handling
  - Missing note_id raises ValueError
  - Missing prediction raises ValueError
- Test scenario 5: Output file creation
  - outputs/audit_report.json exists
  - JSON is valid and matches schema
- **Validation**: pytest runs successfully, coverage ≥70%

### 7. Manual testing with real data
- Run crew with docs/owned_note.json
- Inspect outputs/audit_report.json manually
- Verify:
  - Weak metrics match visual inspection of prediction values
  - Content weaknesses are actionable and make sense
  - Overall diagnosis is coherent
- Check agent logs for:
  - Correct tool calls (note_id mode, not legacy mode)
  - No hallucinated data
  - All 4 tasks executed in sequence
- **Validation**: Manual review confirms high-quality output

### 8. Run openspec validate
- Run `openspec validate 0004 --strict`
- Fix any validation errors
- Ensure all spec scenarios are covered by tests
- **Validation**: Validation passes with no errors

### 9. Update todo list and mark completed
- Mark all tasks as completed in TodoWrite
- **Validation**: Todo list reflects completion

## Dependencies

- Task 2 (AuditReport model) must complete before Task 3 (crew implementation)
- Task 3 (crew) must complete before Task 4-5 (YAML configs)
- Tasks 4-5 (YAML configs) must complete before Task 6 (tests)
- Task 6 (tests) should complete before Task 7 (manual testing)

## Parallelizable Work

- Tasks 4 and 5 (both YAML configs) can be created in parallel after Task 3
- Task 6 (test writing) can start in parallel with Task 7 (manual testing)

## Estimated Effort

- Total: ~3-4 hours
- Model extension: 30 min
- Crew implementation: 1 hour
- YAML configs: 30 min
- Tests: 1 hour
- Manual testing & validation: 1 hour
