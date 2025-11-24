# Proposal: Implement OwnedNoteAuditor Crew

## Summary
Create a standalone crew for auditing owned (self-published) notes to diagnose metric performance and identify content weaknesses through comprehensive NLP and vision analysis.

## Motivation

### Problem
Currently, the system can analyze competitor notes to extract success patterns (via CompetitorAnalyst), but lacks the ability to audit owned notes to identify:
- Which of the 10 prediction metrics are underperforming
- What content features are missing or weak
- What strengths the note already possesses

### Current State
- CompetitorAnalyst crew analyzes multiple competitor notes → SuccessProfileReport
- Tools (NLPAnalysisTool, MultiModalVisionTool) exist but are only used for competitor analysis
- AuditReport model exists as placeholder in reports.py with no implementation
- No systematic way to diagnose owned note weaknesses

### Goals
1. Enable diagnostic analysis of owned notes (single-note focus)
2. Reuse existing NLP and Vision tools for consistency
3. Provide actionable insights on metric weaknesses and content gaps
4. Lay foundation for future GapFinder agent (owned vs competitor comparison)

## Proposed Changes

### New Capabilities
1. **owned-note-auditor** (spec): Crew that audits owned notes for weaknesses

### New Components
1. **crew_owned_note.py**: Standalone crew following CompetitorAnalyst pattern
2. **config/agents_owned_note.yaml**: Agent configuration for OwnedNoteAuditor
3. **config/tasks_owned_note.yaml**: 4 sequential task definitions
4. **Extended AuditReport model**: Complete diagnostic output structure
5. **tests/test_owned_note_auditor.py**: Integration tests

### Modified Components
1. **models/reports.py**: Extend AuditReport from placeholder to full model

## Design Approach

### Architecture Pattern
Follow proven CompetitorAnalyst design:
- Use `@CrewBase` decorator
- Sequential process (4 tasks, single agent)
- `@before_kickoff` for input validation and shared_context storage
- Smart mode tool calls (note_id parameter)

### Key Design Decisions

#### 1. Standalone vs Integrated
**Decision**: Create separate `crew_owned_note.py` file
**Rationale**:
- Different use case (single-note audit vs multi-note analysis)
- Different output structure (AuditReport vs SuccessProfileReport)
- Clearer separation of concerns
- Easier to test and maintain independently

#### 2. Task Structure
**Decision**: 4 sequential tasks (similar to CompetitorAnalyst's 4-task chain)
**Tasks**:
1. `extract_content_features_task`: Call NLPAnalysisTool + MultiModalVisionTool
2. `analyze_metric_performance_task`: Evaluate 10 prediction metrics against thresholds
3. `identify_weaknesses_task`: Compare extracted features against best practices
4. `generate_audit_report_task`: Compile final AuditReport with all findings

**Rationale**:
- Clear separation of concerns (feature extraction → metric analysis → diagnosis → reporting)
- Easier to debug individual stages
- Consistent with existing crew architecture
- Previous experience shows multi-step tasks in single task lead to LLM confusion

#### 3. Report Scope
**Decision**: Basic diagnostic focus (no competitor comparison in this change)
**Scope**:
- Current metrics analysis (identify low performers)
- Content features extraction (text + visual)
- Weaknesses identification (missing/weak features)
- Strengths identification (what's already good)
- **OUT OF SCOPE**: Competitor comparison, optimization recommendations (future GapFinder)

**Rationale**:
- Keep change focused and testable
- Establish foundation before adding complexity
- GapFinder agent will handle comparison logic later

#### 4. Input Handling
**Decision**: Support Note object or dict via @before_kickoff
**Pattern**:
```python
@before_kickoff
def validate_and_prepare_inputs(self, inputs: Union[Note, Dict]):
    # Validate owned note data
    # Serialize to dict if Note object
    # Store in shared_context['owned_note_data']
    # Return flattened dict for YAML variable substitution
```

**Rationale**:
- Consistent with CompetitorAnalyst pattern
- Prevents LLM hallucination via shared_context
- Flexible for different input sources

#### 5. Tool Reuse
**Decision**: Reuse existing MultiModalVisionTool + NLPAnalysisTool (no new tools)
**Rationale**:
- Tools already support single-note analysis via smart mode (note_id)
- Ensures consistency in feature extraction across owned and competitor notes
- Reduces implementation complexity and testing burden

### Data Flow

```
Input: owned_note (Note object or dict)
  ↓
@before_kickoff: Validate & store in shared_context
  ↓
Task 1: extract_content_features
  - NLPAnalysisTool(note_id) → TextAnalysisResult (30+ features)
  - MultiModalVisionTool(note_id) → VisionAnalysisResult (17+ features)
  ↓
Task 2: analyze_metric_performance
  - Read prediction metrics from owned_note
  - Compare against thresholds (e.g., CTR < 0.05 = weak)
  - Identify low-performing metrics
  ↓
Task 3: identify_weaknesses
  - Analyze extracted features for gaps
  - Compare against best practice patterns
  - Compile weaknesses list (metric-based + content-based)
  ↓
Task 4: generate_audit_report
  - Compile AuditReport with all findings
  - Output to outputs/audit_report.json
  ↓
Output: AuditReport (JSON)
```

### AuditReport Model Structure

```python
class AuditReport(BaseModel):
    """自营笔记审计报告 (Owned Note Audit Report)"""

    # Basic info
    note_id: str
    keyword: str

    # Current metrics
    current_metrics: Dict[str, float]  # All 10 prediction metrics

    # Extracted features (raw tool outputs)
    text_features: TextAnalysisResult
    visual_features: VisionAnalysisResult

    # Metric analysis
    weak_metrics: List[str]  # e.g., ["ctr", "sort_score2"]
    strong_metrics: List[str]  # e.g., ["comment_rate"]

    # Content analysis
    content_weaknesses: List[str]  # e.g., ["Missing emotional hook in opening", "Low thumbnail appeal"]
    content_strengths: List[str]  # e.g., ["Strong credibility signals", "Clear value proposition"]

    # Summary
    overall_diagnosis: str  # 50-100 char summary of key issues

    # Metadata
    audit_timestamp: str  # ISO 8601
```

## Dependencies

### Requires
- Existing MultiModalVisionTool (from change 0001)
- Existing NLPAnalysisTool (from change 0001)
- Shared context infrastructure
- Note data models

### Blocks
- Future GapFinder agent implementation (will compare AuditReport against SuccessProfileReport)

### Related
- CompetitorAnalyst (change 0003): Same architectural pattern
- MultiModalVision (change 0001): Tool reuse
- NLP Analysis (change 0001): Tool reuse

## Implementation Plan

### High-Level Steps

1. **Proposal Phase** (Current)
   - Create OpenSpec proposal structure ✓
   - Write proposal.md ✓
   - Write specs/owned-note-auditor/spec.md
   - Write tasks.md
   - Validate with `openspec validate 0004 --strict`

2. **Implementation Phase** (After approval)
   - Create feature branch
   - Extend AuditReport model
   - Implement crew_owned_note.py
   - Create YAML configs (agents + tasks)
   - Write integration tests
   - Manual testing with docs/owned_note.json

3. **Validation Phase**
   - Run all tests (pytest)
   - Verify outputs with real data
   - Check openspec validation passes
   - Code review

### Validation Criteria

**Functional**:
- [ ] Crew successfully processes docs/owned_note.json
- [ ] All 4 tasks execute in sequence
- [ ] AuditReport contains all required fields
- [ ] Extracted features match tool output structures
- [ ] Weaknesses list is non-empty and actionable

**Technical**:
- [ ] Test coverage ≥ 70%
- [ ] `openspec validate 0004 --strict` passes
- [ ] No LLM hallucination in tool calls (verified via logs)
- [ ] Output JSON validates against AuditReport schema

**Quality**:
- [ ] Code follows project conventions (line length, docstrings, etc.)
- [ ] YAML configs follow bilingual pattern (Chinese + English)
- [ ] Error handling for missing images, malformed input

## Risks and Mitigations

### Risk 1: Tool hallucination on single note
**Likelihood**: Medium
**Impact**: High (incorrect analysis)
**Mitigation**:
- Use shared_context pattern proven in CompetitorAnalyst
- Smart mode tool calls (note_id only, no metadata passing)
- Extensive logging for debugging

### Risk 2: Threshold selection for "weak" metrics
**Likelihood**: Medium
**Impact**: Medium (false positives/negatives)
**Mitigation**:
- Use statistical approach if possible (e.g., bottom 25th percentile)
- Document threshold rationale in task YAML
- Make thresholds configurable in future iterations

### Risk 3: Task complexity leading to LLM confusion
**Likelihood**: Low (lesson learned from CompetitorAnalyst)
**Impact**: High (task failures)
**Mitigation**:
- Keep tasks simple and focused (4 tasks, not 2 complex ones)
- Clear task descriptions with numbered steps
- Explicit "禁止事项" (forbidden actions) in YAML

### Risk 4: Missing image URLs in owned_note.json
**Likelihood**: Low
**Impact**: Medium (vision analysis fails)
**Mitigation**:
- Validate URLs in @before_kickoff
- MultiModalVisionTool already handles missing images gracefully
- Fail fast with clear error message

## Success Metrics

### Immediate (Change 0004)
- [ ] OwnedNoteAuditor crew executes successfully on test data
- [ ] AuditReport output quality verified manually
- [ ] All tests pass, openspec validation passes
- [ ] Documentation complete

### Future (Post-Implementation)
- Integration with GapFinder agent
- Batch auditing of multiple owned notes
- A/B testing framework using audit insights

## Future Work

**Not included in this change**:
1. **Competitor comparison**: Will be handled by future GapFinder agent
2. **Optimization recommendations**: Specific suggestions for improvement
3. **Batch auditing**: Process multiple owned notes in one run
4. **Threshold tuning**: Statistical calibration of weakness thresholds
5. **Advanced features**:
   - Trend analysis (compare against historical audits)
   - Score-based diagnosis (quantify overall health)
   - Priority ranking of weaknesses

**Why deferred**:
- Keep initial implementation focused and testable
- Gather real-world usage data before adding complexity
- Establish clear interfaces for future extensions

## Appendix

### Example Usage

```python
from xhs_seo_optimizer.crew_owned_note import XhsSeoOptimizerCrewOwnedNote
from xhs_seo_optimizer.models.note import Note
import json

# Load owned note
with open('docs/owned_note.json') as f:
    note_data = json.load(f)

# Create crew
crew = XhsSeoOptimizerCrewOwnedNote()

# Run audit
result = crew.kickoff(inputs={'owned_note': note_data, 'keyword': 'DHA'})

# Check output
audit_report = json.loads(open('outputs/audit_report.json').read())
print(f"Weak metrics: {audit_report['weak_metrics']}")
print(f"Content weaknesses: {audit_report['content_weaknesses']}")
```

### References
- CompetitorAnalyst implementation: `src/xhs_seo_optimizer/crew_simple.py`
- Tool documentation: `openspec/specs/multimodal-vision/spec.md`, `openspec/specs/nlp-analysis/spec.md`
- Project conventions: `openspec/project.md`
