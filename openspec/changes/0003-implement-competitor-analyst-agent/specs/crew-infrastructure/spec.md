# crew-infrastructure Specification

## Purpose
The crew-infrastructure capability establishes the complete CrewAI framework for the multi-agent SEO optimization system. It provides YAML-based configuration for all 5 agents and tasks, a Crew class with decorator-based methods, and structured report passing between agents.

## ADDED Requirements

### Requirement: System SHALL define all 5 agents in agents.yaml

The configuration file MUST contain definitions for Orchestrator, CompetitorAnalyst, OwnedNoteAuditor, GapFinder, and OptimizationStrategist.

#### Scenario: Load agents configuration and verify all 5 agents exist

```python
import yaml
from pathlib import Path

config_path = Path("src/xhs_seo_optimizer/config/agents.yaml")
assert config_path.exists(), "agents.yaml not found"

with open(config_path) as f:
    agents_config = yaml.safe_load(f)

# MUST have all 5 agents
required_agents = [
    "orchestrator",
    "competitor_analyst",
    "owned_note_auditor",
    "gap_finder",
    "optimization_strategist"
]

for agent_name in required_agents:
    assert agent_name in agents_config, f"Missing agent: {agent_name}"

    agent = agents_config[agent_name]
    assert "role" in agent
    assert "goal" in agent
    assert "backstory" in agent
    assert "llm" in agent
```

#### Scenario: CompetitorAnalyst has required tools configured

```python
agents_config = yaml.safe_load(open("src/xhs_seo_optimizer/config/agents.yaml"))

competitor_analyst = agents_config["competitor_analyst"]

assert "tools" in competitor_analyst
assert isinstance(competitor_analyst["tools"], list)

# MUST include DataAggregatorTool, NLPAnalysisTool, MultiModalVisionTool
required_tools = [
    "DataAggregatorTool",
    "NLPAnalysisTool",
    "MultiModalVisionTool"
]

for tool in required_tools:
    assert tool in competitor_analyst["tools"], f"Missing tool: {tool}"
```

### Requirement: System SHALL define all 5 tasks in tasks.yaml

The configuration file MUST contain task definitions with descriptions, expected outputs, and agent assignments.

#### Scenario: Load tasks configuration and verify all 5 tasks exist

```python
import yaml

config_path = Path("src/xhs_seo_optimizer/config/tasks.yaml")
assert config_path.exists(), "tasks.yaml not found"

with open(config_path) as f:
    tasks_config = yaml.safe_load(f)

# MUST have all 5 tasks
required_tasks = [
    "orchestrate_workflow",
    "analyze_competitors",
    "audit_owned_note",
    "find_gaps",
    "generate_strategy"
]

for task_name in required_tasks:
    assert task_name in tasks_config, f"Missing task: {task_name}"

    task = tasks_config[task_name]
    assert "description" in task
    assert "expected_output" in task
    assert "agent" in task
```

#### Scenario: analyze_competitors task has correct configuration

```python
tasks_config = yaml.safe_load(open("src/xhs_seo_optimizer/config/tasks.yaml"))

analyze_task = tasks_config["analyze_competitors"]

assert analyze_task["agent"] == "competitor_analyst"
assert analyze_task["expected_output"] == "SuccessProfileReport (JSON格式)"
assert "context" in analyze_task
assert isinstance(analyze_task["context"], list)

# Output file is optional but recommended
if "output_file" in analyze_task:
    assert "success_profile_report" in analyze_task["output_file"]
```

### Requirement: System MUST implement XhsSeoOptimizerCrew class with decorators

The crew.py file SHALL define a CrewBase-derived class with @agent, @task, and @crew decorators.

#### Scenario: Crew class is correctly structured

```python
from xhs_seo_optimizer.crew import XhsSeoOptimizerCrew
from crewai import Agent, Task, Crew
from crewai.project import CrewBase

# MUST be a CrewBase subclass
assert issubclass(XhsSeoOptimizerCrew, CrewBase)

crew = XhsSeoOptimizerCrew()

# MUST have agents_config and tasks_config attributes
assert hasattr(crew, "agents_config")
assert hasattr(crew, "tasks_config")

assert crew.agents_config == "config/agents.yaml"
assert crew.tasks_config == "config/tasks.yaml"
```

#### Scenario: competitor_analyst() method returns Agent instance

```python
from crewai import Agent

crew = XhsSeoOptimizerCrew()

agent = crew.competitor_analyst()

# MUST return an Agent instance
assert isinstance(agent, Agent)

# MUST have tools assigned
assert hasattr(agent, "tools")
assert len(agent.tools) >= 3  # DataAggregator, NLP, Vision

# Verify tool types
from xhs_seo_optimizer.tools import DataAggregatorTool, NLPAnalysisTool, MultiModalVisionTool

tool_types = [type(tool) for tool in agent.tools]
assert DataAggregatorTool in tool_types
assert NLPAnalysisTool in tool_types
assert MultiModalVisionTool in tool_types
```

### Requirement: System SHALL support both sequential and parallel task execution

The Crew instance MUST be configurable for different execution processes.

#### Scenario: Crew can be created with sequential process

```python
from crewai import Process

crew_instance = XhsSeoOptimizerCrew().crew()

# Default SHOULD be sequential for this project
assert crew_instance.process in [Process.sequential, Process.hierarchical]
```

#### Scenario: Future support for parallel execution (Phase 1)

```python
# Phase 1: CompetitorAnalyst and OwnedNoteAuditor run in parallel
# Phase 2-4: Sequential

# This is a design note, not enforced in this change
# The crew structure MUST allow for future parallel configuration
```

### Requirement: System MUST validate YAML configuration on startup

The Crew class SHALL validate that YAML files are correctly formatted and contain required fields.

#### Scenario: Invalid agents.yaml raises clear error

```python
import pytest
import tempfile
import yaml

# Create invalid YAML (missing required field)
invalid_config = {
    "competitor_analyst": {
        "role": "竞品分析师",
        # Missing "goal" field
        "backstory": "...",
        "llm": "deepseek/deepseek-chat"
    }
}

with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    yaml.dump(invalid_config, f)
    invalid_path = f.name

# Mock the config path
import os
os.environ['AGENTS_CONFIG_OVERRIDE'] = invalid_path

# SHOULD raise validation error
with pytest.raises(ValueError, match="goal.*required"):
    crew = XhsSeoOptimizerCrew()
    crew.competitor_analyst()
```

### Requirement: System SHALL configure task dependencies in Python code

Task context dependencies MUST be specified in crew.py @task methods, NOT in YAML configuration.

#### Scenario: find_gaps task depends on analyze_competitors and audit_owned_note

```python
from xhs_seo_optimizer.crew import XhsSeoOptimizerCrew

crew = XhsSeoOptimizerCrew()

# Task dependencies configured in Python
find_gaps_task = crew.find_gaps_task()

# Verify task has context configured
assert hasattr(find_gaps_task, 'context')
assert len(find_gaps_task.context) == 2
# Context contains Task instances, not just names
```

#### Scenario: tasks.yaml does NOT contain context field

```python
import yaml

with open("src/xhs_seo_optimizer/config/tasks.yaml") as f:
    tasks_config = yaml.safe_load(f)

find_gaps = tasks_config["find_gaps"]

# Context MAY appear as comment/documentation but is not used
# Configuration happens in crew.py code
```

### Requirement: System SHALL pass structured reports between agents via Pydantic models

Task outputs MUST be Pydantic models configured via output_pydantic parameter.

#### Scenario: analyze_competitors returns SuccessProfileReport

```python
from xhs_seo_optimizer.models.reports import SuccessProfileReport
from xhs_seo_optimizer.models.note import Note
import json

crew = XhsSeoOptimizerCrew()

# Verify task has output_pydantic configured
analyze_task = crew.analyze_competitors_task()
# Note: output_pydantic attribute may be internal to CrewAI

# Execute crew
target_notes = [Note.from_json(data) for data in json.load(open("docs/target_notes.json"))]
result = crew.crew().kickoff(inputs={
    "target_notes": target_notes,
    "keyword": "婴儿辅食推荐"
})

# Access as Pydantic model (type-safe)
report = result.pydantic

assert isinstance(report, SuccessProfileReport)
assert report.keyword == "婴儿辅食推荐"
assert report.sample_size > 0
```

#### Scenario: output_pydantic ensures type safety

```python
# When output_pydantic is configured, result.pydantic returns typed model
crew_instance = crew.crew()
result = crew_instance.kickoff(inputs={
    "target_notes": target_notes,
    "keyword": "测试"
})

# Type-safe access
report: SuccessProfileReport = result.pydantic

# IDE autocomplete and type checking work
assert hasattr(report, 'title_patterns')
assert hasattr(report, 'key_success_factors')
```

#### Scenario: find_gaps task can consume SuccessProfileReport as context

```python
# This scenario demonstrates inter-task data flow
# (full implementation in future changes)

# Pseudocode:
# analyze_task outputs SuccessProfileReport
# audit_task outputs AuditReport
# find_gaps_task consumes both as context

crew = XhsSeoOptimizerCrew()

# Task dependencies defined in tasks.yaml
find_gaps_task = crew.find_gaps_task()

# MUST have context from previous tasks
assert "context" in crew.tasks_config["find_gaps"]
assert "analyze_competitors" in crew.tasks_config["find_gaps"]["context"]
assert "audit_owned_note" in crew.tasks_config["find_gaps"]["context"]
```

### Requirement: System MUST log agent interactions for debugging

The Crew SHALL provide structured logging for task execution, tool calls, and inter-agent communication.

#### Scenario: Crew logs task start and completion

```python
import logging
from io import StringIO

# Capture logs
log_stream = StringIO()
handler = logging.StreamHandler(log_stream)
logger = logging.getLogger("crewai")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

crew_instance = XhsSeoOptimizerCrew().crew()

# Execute a task
crew_instance.kickoff(inputs={"target_notes": target_notes, "keyword": "测试"})

# Logs SHOULD include task names and status
log_output = log_stream.getvalue()

assert "analyze_competitors" in log_output or "CompetitorAnalyst" in log_output
# (Exact log format depends on CrewAI library)
```

### Requirement: System SHALL provide CLI entry point for running analysis

The main.py file MUST provide a command-line interface for executing the workflow.

#### Scenario: Run analysis from command line

```python
import subprocess
import json

# Run CLI (assuming main.py implements argparse)
result = subprocess.run([
    "python", "src/xhs_seo_optimizer/main.py",
    "--keyword", "婴儿辅食推荐",
    "--target-notes", "docs/target_notes.json",
    "--owned-note", "docs/owned_note.json",
    "--output", "outputs/result.json"
], capture_output=True, text=True)

assert result.returncode == 0, f"CLI failed: {result.stderr}"

# Output file SHOULD be created
import os
assert os.path.exists("outputs/result.json")

# Output SHOULD be valid JSON
with open("outputs/result.json") as f:
    output_data = json.load(f)

assert "success_profile_report" in output_data or "optimization_plan" in output_data
```

#### Scenario: CLI provides help documentation

```python
result = subprocess.run([
    "python", "src/xhs_seo_optimizer/main.py",
    "--help"
], capture_output=True, text=True)

assert result.returncode == 0
assert "--keyword" in result.stdout
assert "--target-notes" in result.stdout
assert "--owned-note" in result.stdout
```

### Requirement: System SHALL support extensibility for future agents

The infrastructure MUST allow adding new agents with minimal code changes.

#### Scenario: Adding a new agent requires only YAML + decorator method

```python
# To add a new agent (e.g., "ContentGenerator"):
# 1. Add to agents.yaml:
#    content_generator:
#      role: "内容生成器"
#      goal: "..."
#      backstory: "..."
#      tools: [...]
#      llm: "deepseek/deepseek-chat"

# 2. Add @agent method in crew.py:
#    @agent
#    def content_generator(self) -> Agent:
#        return Agent(
#            config=self.agents_config['content_generator'],
#            tools=[...]
#        )

# 3. Add task to tasks.yaml and corresponding @task method

# NO changes to core infrastructure required
# This demonstrates extensibility
```

### Requirement: Report models MUST be defined in models/reports.py

The system SHALL maintain all report Pydantic models in a dedicated module.

#### Scenario: reports.py defines all report models

```python
from xhs_seo_optimizer.models import reports

# MUST have SuccessProfileReport
assert hasattr(reports, "SuccessProfileReport")
assert hasattr(reports, "FeaturePattern")

# MAY have placeholder models for future agents
# (Optional in this change, but good practice)
if hasattr(reports, "AuditReport"):
    assert issubclass(reports.AuditReport, BaseModel)

if hasattr(reports, "GapReport"):
    assert issubclass(reports.GapReport, BaseModel)

if hasattr(reports, "OptimizationPlan"):
    assert issubclass(reports.OptimizationPlan, BaseModel)
```

#### Scenario: All report models are Pydantic BaseModel subclasses

```python
from pydantic import BaseModel

assert issubclass(reports.SuccessProfileReport, BaseModel)
assert issubclass(reports.FeaturePattern, BaseModel)

# Verify JSON serialization works
report = reports.SuccessProfileReport(
    keyword="test",
    sample_size=10,
    aggregated_stats=...,
    title_patterns=[],
    cover_patterns=[],
    content_patterns=[],
    tag_patterns=[],
    key_success_factors=["f1", "f2", "f3"],
    viral_formula_summary="summary" * 20,
    analysis_timestamp="2025-11-14T10:00:00Z"
)

json_str = report.model_dump_json()
assert isinstance(json_str, str)
```

### Requirement: System SHALL handle errors gracefully with clear messages

The Crew and agents MUST provide informative error messages for common failure scenarios.

#### Scenario: Missing input data raises ValueError

```python
import pytest

crew = XhsSeoOptimizerCrew()

# Missing required input
with pytest.raises(ValueError, match="target_notes.*required"):
    crew.competitor_analyst().execute_task({
        "keyword": "测试"
        # Missing "target_notes"
    })
```

#### Scenario: Tool execution failure is logged and handled

```python
# If NLPAnalysisTool fails for a note, agent SHOULD:
# 1. Log the error
# 2. Skip that note
# 3. Continue with remaining notes
# 4. Include warning in report

# This is a behavior requirement, tested via integration tests
```

### Requirement: System SHALL provide callback hooks for workflow customization

The Crew MUST support @before_kickoff and @after_kickoff decorators for input validation and output processing.

#### Scenario: @before_kickoff validates inputs

```python
@CrewBase
class XhsSeoOptimizerCrew:
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @before_kickoff
    def validate_inputs(self, inputs):
        """Validate inputs before crew execution."""
        if not inputs.get("target_notes"):
            raise ValueError("target_notes is required")
        if not inputs.get("keyword"):
            raise ValueError("keyword is required")

        # Can modify inputs if needed
        inputs["validated"] = True
        return inputs
```

#### Scenario: @after_kickoff processes output

```python
@CrewBase
class XhsSeoOptimizerCrew:
    # ...

    @after_kickoff
    def process_output(self, output):
        """Process output after crew completion."""
        # Save to database
        # Send notifications
        # Log metrics
        print(f"Crew completed successfully: {output.raw[:100]}...")
        return output

# Hook is executed automatically
crew_instance = XhsSeoOptimizerCrew().crew()
result = crew_instance.kickoff(inputs={...})
# process_output is called automatically with result
```

#### Scenario: Hooks can raise exceptions to abort workflow

```python
@before_kickoff
def validate_inputs(self, inputs):
    target_notes = inputs.get("target_notes", [])
    if len(target_notes) == 0:
        raise ValueError("At least one target note is required")
    if len(target_notes) > 200:
        raise ValueError("Too many notes, max 200 allowed")
    return inputs

# Will raise ValueError and not execute crew
crew_instance.kickoff(inputs={"target_notes": [], "keyword": "test"})
```

### Requirement: System SHALL support configuration via environment variables

The system MUST allow overriding LLM models and API keys via environment variables.

#### Scenario: Override LLM model via environment variable

```python
import os

os.environ["COMPETITOR_ANALYST_LLM"] = "openai/gpt-4o"

crew = XhsSeoOptimizerCrew()
agent = crew.competitor_analyst()

# Agent SHOULD use overridden LLM
# (Exact implementation depends on CrewAI configuration)
```

#### Scenario: API keys loaded from .env file

```python
from dotenv import load_dotenv
import os

load_dotenv()

# MUST have required API keys
assert "OPENROUTER_API_KEY" in os.environ or "OPENAI_API_KEY" in os.environ

# Tools should use these keys automatically
from xhs_seo_optimizer.tools import MultiModalVisionTool

tool = MultiModalVisionTool()
# Should not raise authentication error
```
