# Implementation Tasks: OptimizationStrategist Crew

## Task Overview

| Task | Description | Dependencies | Estimate |
|------|-------------|--------------|----------|
| 1 | Expand OptimizationPlan model in reports.py | None | 30 min |
| 2 | Create agents_optimization.yaml | None | 15 min |
| 3 | Create tasks_optimization.yaml | Task 1 | 30 min |
| 4 | Implement crew_optimization.py | Tasks 1-3 | 45 min |
| 5 | Create test file | Task 4 | 20 min |
| 6 | Run end-to-end test | Task 5 | 15 min |
| 7 | Validate output format | Task 6 | 10 min |

## Detailed Tasks

### Task 1: Expand OptimizationPlan Model [PRIORITY: HIGH] ✅

**File:** `src/xhs_seo_optimizer/models/reports.py`

**Actions:**
- [x] Add `OptimizationItem` model with fields: original, optimized, rationale, targeted_metrics, targeted_weak_features
- [x] Add `TitleOptimization` model with fields: alternatives (3 items), recommended_index, selection_rationale
- [x] Add `ContentOptimization` model with fields: opening_hook, ending_cta, hashtags, body_improvements
- [x] Add `VisualPrompt` model with fields: image_type, prompt_text, style_reference, key_elements, color_scheme, targeted_metrics
- [x] Add `VisualOptimization` model with fields: cover_prompt, inner_image_prompts, general_visual_guidelines
- [x] Expand `OptimizationPlan` model with all new nested models
- [x] Add field validators for constraints (e.g., alternatives must have exactly 3 items)

**Validation:**
```python
# Test model creation
from xhs_seo_optimizer.models.reports import OptimizationPlan
plan = OptimizationPlan(keyword="test", owned_note_id="123", ...)
```

---

### Task 2: Create Agent Configuration [PRIORITY: HIGH] ✅

**File:** `src/xhs_seo_optimizer/config/agents_optimization.yaml`

**Content:**
```yaml
optimization_strategist:
  role: >
    优化策略师 (Optimization Strategist) for {keyword}
  goal: >
    基于差距分析报告(GapReport)，为客户笔记生成具体、可执行的优化方案，
    包含标题备选、内容改进和视觉优化建议，确保每个建议都能直接应用。
  backstory: >
    你是一位资深的小红书内容创作专家和营销策略师。
    你擅长将数据分析转化为可执行的创意方案，能够写出符合平台调性的爆款文案。
    你的建议从不泛泛而谈，总是具体到可以直接复制粘贴使用。
```

**Validation:**
- [x] YAML 语法正确
- [x] 角色描述清晰

---

### Task 3: Create Task Configuration [PRIORITY: HIGH] ✅

**File:** `src/xhs_seo_optimizer/config/tasks_optimization.yaml`

**Tasks to define:**

1. **generate_text_optimizations**
   - Input: gap_report, audit_report, owned_note
   - Output: TextOptimizationResult (intermediate)
   - Focus: title alternatives, opening_hook, ending_cta, hashtags

2. **generate_visual_prompts**
   - Input: gap_report, audit_report, success_profile_report
   - Output: VisualOptimizationResult (intermediate)
   - Focus: cover_prompt, inner_image_prompts, visual guidelines

3. **compile_optimization_plan**
   - Input: TextOptimizationResult, VisualOptimizationResult, gap_report
   - Output: OptimizationPlan (final)
   - Focus: priority_summary, expected_impact, assembly

**Validation:**
- [x] YAML 语法正确
- [x] Task descriptions reference correct inputs

---

### Task 4: Implement Crew Class [PRIORITY: HIGH] ✅

**File:** `src/xhs_seo_optimizer/crew_optimization.py`

**Implementation pattern (following crew_gap_finder.py):**
```python
@CrewBase
class XhsSeoOptimizerCrewOptimization:
    agents_config = 'config/agents_optimization.yaml'
    tasks_config = 'config/tasks_optimization.yaml'

    def __init__(self):
        # LLM configuration
        pass

    @before_kickoff
    def load_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Load gap_report.json, audit_report.json, etc.
        pass

    @agent
    def optimization_strategist(self) -> Agent:
        pass

    @task
    def generate_text_optimizations(self) -> Task:
        pass

    @task
    def generate_visual_prompts(self) -> Task:
        pass

    @task
    def compile_optimization_plan(self) -> Task:
        pass

    @crew
    def crew(self) -> Crew:
        pass
```

**Validation:**
- [x] Crew can be instantiated
- [x] All decorators work correctly

---

### Task 5: Create Test File [PRIORITY: MEDIUM] ✅

**File:** `tests/test_optimization_strategist.py`

**Test cases:**
- [x] Test model validation (OptimizationPlan, OptimizationItem, etc.)
- [x] Test crew instantiation
- [x] Test basic execution with mock data
- [x] Test output format validation

**Validation:**
```bash
pytest tests/test_optimization_strategist.py -v
```

---

### Task 6: Run End-to-End Test [PRIORITY: HIGH] ✅

**Command:**
```bash
PYTHONPATH=src:$PYTHONPATH python3 -c "
from xhs_seo_optimizer.crew_optimization import XhsSeoOptimizerCrewOptimization

crew_instance = XhsSeoOptimizerCrewOptimization()
crew = crew_instance.crew()
result = crew.kickoff(inputs={
    'keyword': '老爸测评dha推荐哪几款'
})
print(result)
"
```

**Expected output:**
- [x] `outputs/optimization_plan.json` created
- [x] JSON structure matches OptimizationPlan schema

---

### Task 7: Validate Output Format [PRIORITY: MEDIUM] ✅

**Validation script:**
```python
import json
from xhs_seo_optimizer.models.reports import OptimizationPlan

with open('outputs/optimization_plan.json', 'r') as f:
    data = json.load(f)

plan = OptimizationPlan(**data)
print(f"Title alternatives: {len(plan.title_optimization.alternatives)}")
print(f"Priority summary: {plan.priority_summary}")
```

**Success criteria:**
- [x] JSON parses without errors
- [x] Pydantic validation passes
- [x] Title has exactly 3 alternatives
- [x] All required fields present
- [x] Chinese content is natural and actionable

---

## Dependencies

```
Task 1 (models) ─┬─> Task 3 (tasks.yaml) ─┬─> Task 4 (crew.py) ─> Task 5 (tests)
                 │                        │
Task 2 (agents) ─┴────────────────────────┘
                                                       │
                                                       v
                                              Task 6 (e2e test)
                                                       │
                                                       v
                                              Task 7 (validation)
```

## Notes

- Tasks 1 and 2 can be done in parallel
- Task 4 depends on both Tasks 1-3
- End-to-end test requires all previous tasks complete
