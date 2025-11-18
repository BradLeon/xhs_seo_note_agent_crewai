"""Simplified crew for testing CompetitorAnalyst agent only.

This is a minimal crew configuration that only includes the CompetitorAnalyst
agent and analyze_competitors task, for testing the refactored agent-based approach.
"""

from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task, before_kickoff
from typing import Dict, Any

from xhs_seo_optimizer.models.reports import SuccessProfileReport
from xhs_seo_optimizer.tools import (
    DataAggregatorTool,
    MultiModalVisionTool,
    NLPAnalysisTool,
)


@CrewBase
class XhsSeoOptimizerCrewSimple:
    """Simplified crew for testing CompetitorAnalyst only.

    This crew only includes:
    - competitor_analyst agent
    - analyze_competitors_task

    It uses sequential process (not hierarchical) since we don't need a manager
    for a single agent.
    """

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks_simple.yaml'

    @before_kickoff
    def validate_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate inputs before crew execution."""
        if not inputs.get("target_notes"):
            raise ValueError("target_notes is required")
        if not isinstance(inputs.get("target_notes"), list):
            raise ValueError("target_notes must be a list")
        if len(inputs.get("target_notes", [])) == 0:
            raise ValueError("At least one target note is required")
        if not inputs.get("keyword"):
            raise ValueError("keyword is required")

        inputs["validated"] = True
        return inputs

    @agent
    def competitor_analyst(self) -> Agent:
        """竞品分析师 agent.

        The agent that coordinates atomic tools to analyze target_notes.
        """
        return Agent(
            config=self.agents_config['competitor_analyst'],
            tools=[
                DataAggregatorTool(),
                MultiModalVisionTool(),
                NLPAnalysisTool()
            ],
            verbose=True
        )

    @task
    def analyze_competitors_task(self) -> Task:
        """竞品分析任务.

        The only task in this simplified crew.
        """
        return Task(
            config=self.tasks_config['analyze_competitors'],
            agent=self.competitor_analyst(),
            output_pydantic=SuccessProfileReport
        )

    @crew
    def crew(self) -> Crew:
        """Assemble the simplified crew.

        Uses sequential process since we only have one agent/task.
        """
        return Crew(
            agents=self.agents,  # Will only contain competitor_analyst
            tasks=self.tasks,    # Will only contain analyze_competitors_task
            process=Process.sequential,  # Simple sequential process
            verbose=True
        )