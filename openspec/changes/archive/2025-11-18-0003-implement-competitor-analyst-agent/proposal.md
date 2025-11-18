# Proposal: Implement CompetitorAnalyst Agent with Full Crew Infrastructure

**Change ID**: 0003-implement-competitor-analyst-agent
**Status**: Proposed
**Author**: AI Agent
**Date**: 2025-11-14

## Summary

Implement the **first agent (CompetitorAnalyst)** in the multi-agent SEO optimization system, along with complete CrewAI infrastructure to support all 5 agents defined in the PRD. The CompetitorAnalyst analyzes why target_notes achieve high prediction scores and extracts reusable "creation formulas/patterns" to guide content optimization.

**Scope**:
- Complete CrewAI framework setup (agents.yaml, tasks.yaml, crew.py)
- Hierarchical process with Orchestrator as manager agent
- CompetitorAnalyst agent with hybrid attribution logic
- SuccessProfileReport data model (feature-centric, statistically quantified)
- Integration with existing tools (DataAggregator, NLP, Vision, StatisticalDelta)
- Comprehensive testing with real Xiaohongshu data

## Motivation

### Problem
The project has implemented all foundational tools (NLPAnalysisTool, MultiModalVisionTool, DataAggregatorTool, StatisticalDeltaTool), but **no agents exist yet** to orchestrate these tools into an intelligent workflow. Users cannot currently:

1. Understand **why** competitor notes achieve high CTR, comment_rate, or sort_score
2. Extract actionable "creation formulas" from high-performing content patterns
3. Run end-to-end gap analysis workflows automatically

### Current State
- ✅ 4 analysis tools fully implemented and tested
- ✅ Pydantic models for Note, predictions, tags, and analysis results
- ❌ No agents or CrewAI configuration
- ❌ No workflow orchestration
- ❌ No report generation for insights

### Goals
1. **Establish CrewAI infrastructure** for all 5 agents (Orchestrator, CompetitorAnalyst, OwnedNoteAuditor, GapFinder, OptimizationStrategist)
2. **Fully implement CompetitorAnalyst** as the first working agent
3. **Generate SuccessProfileReport** with:
   - Feature-centric patterns (title, cover, content, tags)
   - Statistical quantification (prevalence %, z-score, p-value)
   - LLM-synthesized "creation formulas" in Chinese
   - Hybrid attribution (statistical correlation + domain rules + LLM validation)

## Proposed Changes

### New Capabilities
1. **competitor-analyst**: Agent that analyzes target_notes to identify success patterns
2. **success-profile-report**: Structured report model for discovered patterns
3. **crew-infrastructure**: Complete CrewAI framework supporting multi-agent workflows

### Components to Create

#### 1. CrewAI Configuration Files
- `src/xhs_seo_optimizer/config/agents.yaml`: YAML definitions for all 5 agents
- `src/xhs_seo_optimizer/config/tasks.yaml`: YAML definitions for all 5 tasks
- `src/xhs_seo_optimizer/crew.py`: Crew class with @agent, @task, @crew decorators

#### 2. Report Data Models
- `src/xhs_seo_optimizer/models/reports.py`:
  - `FeaturePattern`: Single pattern with statistical evidence
  - `SuccessProfileReport`: CompetitorAnalyst output (feature-centric)
  - `AuditReport`: Placeholder for OwnedNoteAuditor
  - `GapReport`: Placeholder for GapFinder
  - `OptimizationPlan`: Placeholder for OptimizationStrategist

#### 3. Agent Logic
- CompetitorAnalyst implementation in crew.py:
  - Hybrid attribution logic (3-layer approach)
  - Feature extraction pipeline (NLP + Vision for all notes)
  - Pattern identification (prevalence analysis with statistical tests)
  - LLM synthesis for "creation formulas"

#### 4. Testing
- `tests/test_agents/test_competitor_analyst.py`: Unit tests with real data
- End-to-end workflow validation

### Design Approach

#### Hybrid Attribution (3-Layer)
```
Layer 1: Domain Rules
  └─ Pre-defined METRIC_FEATURE_ATTRIBUTION map
     (e.g., CTR ← title+cover, comment_rate ← ending+questions)

Layer 2: Statistical Correlation
  └─ Calculate pattern prevalence in top 25% vs rest
     (e.g., 85% of high-CTR notes use interrogative titles)

Layer 3: LLM Validation
  └─ Explain WHY pattern works + generate actionable formula
     (e.g., "疑问句标题触发好奇心缺口效应，提升点击欲望")
```

#### Feature-Centric Report Structure
Instead of organizing by metrics (CTR formula, comment_rate formula), organize by **features** to avoid duplication and provide richer insights:

```python
SuccessProfileReport:
  title_patterns: [
    {
      feature: "interrogative_title_pattern",
      prevalence: 85.0%,
      affected_metrics: {"ctr": +35.2%, "comment_rate": +12.1%},
      evidence: "z=3.2, p<0.001, n=85/100",
      formula: "使用疑问句标题，触发好奇心缺口，提升点击率"
    }
  ]
  cover_patterns: [...]
  content_patterns: [...]
  tag_patterns: [...]
```

### Dependencies
- Requires: All tools from 0001 and 0002 (already implemented ✅)
- Blocks: 0004-implement-gap-finder-agent (needs SuccessProfileReport as input)

## Implementation Plan

See [design.md](design.md) for detailed architecture and [tasks.md](tasks.md) for granular task breakdown.

### High-Level Steps
1. Create CrewAI configuration (agents.yaml, tasks.yaml)
2. Implement SuccessProfileReport and FeaturePattern models
3. Build attribution rule engine
4. Implement CompetitorAnalyst agent logic
5. Create crew.py with decorators
6. Write comprehensive tests
7. Validate with real Xiaohongshu data

### Validation Criteria
- ✅ `openspec validate 0003 --strict` passes
- ✅ All tests pass with real data (target_notes.json)
- ✅ SuccessProfileReport conforms to schema
- ✅ LLM generates Chinese formulas
- ✅ Statistical evidence included in all patterns
- ✅ Crew framework supports future agent additions

## Risks and Mitigations

### Risk 1: LLM Synthesis Quality
**Risk**: LLM-generated formulas may be generic or inconsistent
**Mitigation**:
- Provide structured prompts with statistical evidence
- Include concrete examples from target_notes
- Validate output format with Pydantic schemas
- Use temperature=0.3 for consistency

### Risk 2: Attribution Accuracy
**Risk**: Incorrect feature-metric attribution leads to wrong conclusions
**Mitigation**:
- Start with conservative domain rules (Layer 1)
- Require high prevalence threshold (>70%) for pattern detection
- Demand statistical significance (p<0.05)
- Manual review of top patterns in initial iterations

### Risk 3: Performance with Large Note Sets
**Risk**: Analyzing 100+ notes with NLP+Vision tools may be slow
**Mitigation**:
- Implement caching for analysis results
- Support parallel processing for note analysis
- Provide progress logging
- Consider sampling strategies for >200 notes

### Risk 4: Crew Configuration Complexity
**Risk**: YAML-based multi-agent setup may be error-prone
**Mitigation**:
- Validate YAML schemas on startup
- Provide clear error messages
- Include example configurations in docs
- Comprehensive unit tests for configuration loading

## Success Metrics

1. **Functional**: CompetitorAnalyst generates SuccessProfileReport from target_notes
2. **Quality**: Reports include ≥3 statistically significant patterns per metric
3. **Usability**: Formulas are in Chinese and actionable (validated by manual review)
4. **Extensibility**: Adding OwnedNoteAuditor agent requires <100 LOC
5. **Reliability**: All tests pass with 100% schema compliance

## Future Work (Out of Scope)

- OwnedNoteAuditor implementation (change 0004)
- GapFinder implementation (change 0005)
- OptimizationStrategist implementation (change 0006)
- Orchestrator orchestration logic (change 0007)
- Web UI for report visualization
- Batch processing for multiple keywords
