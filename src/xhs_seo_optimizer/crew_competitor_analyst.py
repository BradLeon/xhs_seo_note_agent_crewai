"""CompetitorAnalyst crew for analyzing target notes success patterns.

This crew analyzes target_notes to extract success patterns for a given keyword.
It uses the CompetitorAnalyst agent with atomic tools (data aggregation, NLP, vision).
"""

from crewai import Agent, Crew, Task, Process, LLM
from crewai.project import CrewBase, agent, crew, task, before_kickoff
from typing import Dict, Any
import os

from xhs_seo_optimizer.models.reports import SuccessProfileReport
from xhs_seo_optimizer.tools import (
    DataAggregatorTool,
    MultiModalVisionTool,
    NLPAnalysisTool,
)


@CrewBase
class XhsSeoOptimizerCrewCompetitorAnalyst:
    """Crew for analyzing competitor notes success patterns.

    This crew includes:
    - competitor_analyst agent
    - 4 sequential tasks: aggregate → extract features → analyze metrics → generate report

    It uses sequential process (not hierarchical) since we don't need a manager
    for a single agent.

    Following official CrewAI pattern: complex objects are serialized to dicts
    using model_dump() in @before_kickoff, then tools receive these dicts.
    """

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks_competitor_analyst.yaml'

    def __init__(self):
        # Check if proxy is configured via environment variables
        # The underlying HTTP client will automatically use HTTP_PROXY/HTTPS_PROXY
        proxy = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
        if proxy:
            print(f"使用代理 (通过环境变量): {proxy}")

        # LLM configuration for OpenRouter
        llm_config = {
            'base_url': 'https://openrouter.ai/api/v1',
            'api_key': os.getenv("OPENROUTER_API_KEY", ""),
            'temperature': 0.1
        }

        self.custom_llm = LLM(
            model='openrouter/google/gemini-2.5-flash-lite',
            **llm_config
        )

        self.fuction_llm = LLM(
            model='openrouter/deepseek/deepseek-r1-0528',
            **llm_config


        )

    @before_kickoff
    def validate_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and flatten inputs before crew execution.

        Following official CrewAI pattern: serialize complex Pydantic models
        to dicts using model_dump() instead of storing in shared context.

        Supports both:
        1. ComplexInput Pydantic object (will be converted via model_dump())
        2. Dict with target_notes, keyword, owned_note fields
        """
        # Handle ComplexInput Pydantic object
        from xhs_seo_optimizer.models.note import ComplexInput
        if isinstance(inputs, ComplexInput):
            inputs = inputs.model_dump()

        # Validate required fields
        if not inputs.get("target_notes"):
            raise ValueError("target_notes is required")
        if not isinstance(inputs.get("target_notes"), list):
            raise ValueError("target_notes must be a list")
        if len(inputs.get("target_notes", [])) == 0:
            raise ValueError("At least one target note is required")
        if not inputs.get("keyword"):
            raise ValueError("keyword is required")

        # Remove complex Note objects from inputs (CrewAI only supports basic types)
        target_notes = inputs.pop("target_notes")

        # Serialize List[Note] to List[dict] using model_dump() (official pattern)
        # Handle both Note objects and already-serialized dicts (from ComplexInput.model_dump())
        target_notes_data = []
        for note in target_notes:
            if isinstance(note, dict):
                # Already serialized (e.g., from ComplexInput.model_dump())
                target_notes_data.append(note)
            else:
                # Note object - serialize it
                target_notes_data.append(note.model_dump())

        # Handle optional owned_note
        owned_note = inputs.pop("owned_note", None)
        if owned_note is not None:
            if isinstance(owned_note, dict):
                # Already serialized
                inputs["owned_note_data"] = owned_note
            else:
                # Note object - serialize it
                inputs["owned_note_data"] = owned_note.model_dump()

        # Store serialized data in shared context (for tools to access directly)
        # This works around CrewAI limitation where LLM agents hallucinate data
        # instead of reliably passing large lists as tool parameters
        from xhs_seo_optimizer.shared_context import shared_context
        shared_context.set("target_notes_data", target_notes_data)
        shared_context.set("target_notes_count", len(target_notes))

        # Also store in inputs for metadata and variable substitution in YAML
        inputs["target_notes_data"] = target_notes_data
        inputs["target_notes_count"] = len(target_notes)
        inputs["validated"] = True

        return inputs

    @agent
    def competitor_analyst(self) -> Agent:
        """竞品分析师 agent.

        The agent that coordinates atomic tools to analyze target_notes.
        Tools now receive serialized dict data directly (no context needed).
        """
        return Agent(
            config=self.agents_config['competitor_analyst'],
            tools=[
                DataAggregatorTool(),
                MultiModalVisionTool(),
                NLPAnalysisTool()
            ],
            llm=self.custom_llm,
            function_calling_llm=self.fuction_llm,
            verbose=True
        )

    @task
    def aggregate_statistics_task(self) -> Task:
        """Task 1: 聚合统计数据.

        Calls data_aggregator() to compute statistical summaries.
        """
        return Task(
            config=self.tasks_config['aggregate_statistics'],
            agent=self.competitor_analyst()
        )

    @task
    def extract_features_task(self) -> Task:
        """Task 2: 提取特征矩阵.

        Extracts text and visual features from all notes.
        Depends on Task 1 for reference data.
        """
        return Task(
            config=self.tasks_config['extract_features'],
            agent=self.competitor_analyst(),
            context=[self.aggregate_statistics_task()]  # Receives aggregated_stats
        )

    @task
    def analyze_metrics_task(self) -> Task:
        """Task 3: 指标分析.

        Performs metric-centric analysis for all 10 metrics.
        Depends on Tasks 1 & 2.
        """
        return Task(
            config=self.tasks_config['analyze_metrics'],
            agent=self.competitor_analyst(),
            context=[
                self.aggregate_statistics_task(),  # Needs aggregated_stats
                self.extract_features_task()        # Needs features_matrix
            ]
        )

    @task
    def generate_report_task(self) -> Task:
        """Task 4: 生成最终报告.

        Generates final SuccessProfileReport with cross-metric summary.
        Depends on Tasks 1 & 3.
        """
        return Task(
            config=self.tasks_config['generate_report'],
            agent=self.competitor_analyst(),
            context=[
                self.aggregate_statistics_task(),  # Needs aggregated_stats
                self.analyze_metrics_task()        # Needs metric_profiles
            ],
            output_pydantic=SuccessProfileReport  # Final output validation
        )

    @crew
    def crew(self) -> Crew:
        """Assemble the competitor analyst crew.

        Uses sequential process with 4 chained tasks:
        1. aggregate_statistics_task
        2. extract_features_task (depends on 1)
        3. analyze_metrics_task (depends on 1, 2)
        4. generate_report_task (depends on 1, 3)
        """
        return Crew(
            agents=self.agents,  # Contains competitor_analyst
            tasks=self.tasks,    # Contains 4 sequential tasks
            process=Process.sequential,  # Execute tasks in order
            verbose=True
        )
