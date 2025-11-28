"""Gap Finder Crew - 差距定位员.

Identifies statistically significant performance gaps between owned_note and target_notes,
then maps gaps to actionable content features.

Architecture:
- General tasks (free-form text): Use LLM with OpenRouter
- Structured output tasks (response_model): Use OpenAICompletion with Gemini OpenAI-compatible endpoint
"""

import re
from crewai import Agent, Crew, Task, LLM
from crewai.llms.providers.openai.completion import OpenAICompletion
from crewai.project import CrewBase, agent, task, crew, before_kickoff
from typing import Any, Dict
import json
import os

from .tools.statistical_delta import StatisticalDeltaTool
from .models.reports import GapReport
from .models.analysis_results import GapAnalysis


@CrewBase
class XhsSeoOptimizerCrewGapFinder:
    """差距定位员 Crew - Gap Finder for performance analysis.

    Compares SuccessProfileReport (what works) with AuditReport (current state)
    to identify critical performance gaps and their root causes.

    Architecture:
    - Tasks 1: Use gap_finder with OpenRouter LLM (free-form text)
    - Task 2: Use report_generator with OpenAICompletion + Gemini (structured output via response_model)
    """

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks_gap_finder.yaml'

    def __init__(self):
        """Initialize Gap Finder crew."""
        self.shared_context = {}

        # General LLM (OpenRouter) - for free-form analysis tasks
        self.general_llm = OpenAICompletion(
            model='google/gemini-2.5-flash',
            base_url='https://openrouter.ai/api/v1',
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            temperature=0.1,
        )

        # Structured output LLM (Gemini OpenAI-compatible endpoint)
        self.structured_llm = OpenAICompletion(
            model='gemini-2.5-flash',
            base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
            api_key=os.getenv("GOOGLE_API_KEY", ""),
            temperature=0.1,
        )

    @agent
    def gap_finder(self) -> Agent:
        """差距定位员 agent (for free-form analysis tasks).

        Uses OpenRouter LLM for free-form text output.
        Data-driven analyst that identifies statistical gaps and maps them
        to specific content features.
        """
        return Agent(
            config=self.agents_config['gap_finder'],
            tools=[StatisticalDeltaTool()],
            llm=self.general_llm,
            verbose=True,
            allow_delegation=False
        )

    @agent
    def report_generator(self) -> Agent:
        """报告生成 agent (for structured output tasks).

        Uses OpenAICompletion + Gemini for native structured output.
        Specialized for tasks with response_model that require strict JSON conformance.
        """
        return Agent(
            config=self.agents_config['report_generator'],
            llm=self.general_llm,
            verbose=False  # Less verbose for structured output
        )

    @task
    def calculate_statistical_gaps(self) -> Task:
        """Task 1: Calculate statistical gaps using StatisticalDeltaTool.

        Output: GapAnalysis with significant_gaps, marginal_gaps, non_significant_gaps.
        Each Gap includes related_features and rationale from attribution.py.

        Uses gap_finder agent with OpenRouter LLM for free-form analysis.
        """
        return Task(
            config=self.tasks_config['calculate_statistical_gaps'],
            agent=self.gap_finder(),
            response_model=GapAnalysis
        )

    @task
    def generate_gap_report(self) -> Task:
        """Task 2: Generate final GapReport JSON (structured output).

        Merges feature mapping, prioritization, and report generation into one task.
        Output: GapReport conforming to the final report schema with all metadata.

        Uses report_generator agent with OpenAICompletion + response_model
        for native structured output via Gemini OpenAI-compatible API.
        """
        return Task(
            config=self.tasks_config['generate_gap_report'],
            agent=self.report_generator(),  # Uses structured_llm
            response_model=GapReport,  # Native structured output!
            context=[self.calculate_statistical_gaps()]
        )

    @crew
    def crew(self) -> Crew:
        """Create GapFinder crew with sequential task execution."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True
        )

    @before_kickoff
    def validate_and_flatten_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and flatten inputs before crew execution.

        Following the pattern from OwnedNoteAuditor: flatten complex Pydantic models
        to simple dicts and extract key fields for YAML variable substitution.

        Args:
            inputs: Must contain:
                - success_profile_report: SuccessProfileReport dict or JSON string
                - audit_report: AuditReport dict or JSON string
                - keyword: str

        Returns:
            Flattened dict for YAML variable substitution

        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Validate required fields
        if 'success_profile_report' not in inputs:
            raise ValueError("inputs must contain 'success_profile_report'")
        if 'audit_report' not in inputs:
            raise ValueError("inputs must contain 'audit_report'")
        if 'keyword' not in inputs:
            raise ValueError("inputs must contain 'keyword'")

        # Parse success_profile_report (handle both dict and JSON string)
        success_profile = inputs['success_profile_report']
        if isinstance(success_profile, str):
            success_profile = json.loads(success_profile)

        # Parse audit_report (handle both dict and JSON string)
        audit_report = inputs['audit_report']
        if isinstance(audit_report, str):
            audit_report = json.loads(audit_report)

        # Validate audit_report has current_metrics
        if 'current_metrics' not in audit_report:
            raise ValueError(
                "audit_report must contain 'current_metrics' field. "
                "Please ensure OwnedNoteAuditor has been updated to include this field."
            )

        # Extract flattened data for YAML variable substitution
        # Current metrics from owned_note (via audit_report)
        current_metrics = audit_report['current_metrics']

        # Target metrics from success_profile_report
        aggregated_stats = success_profile['aggregated_stats']
        prediction_stats = aggregated_stats['prediction_stats']

        # Flatten for YAML: target_mean_ctr, target_std_ctr, etc.
        target_means = {}
        target_stds = {}
        for metric_name, stats in prediction_stats.items():
            target_means[f"target_mean_{metric_name}"] = stats['mean']
            target_stds[f"target_std_{metric_name}"] = stats['std']

        # Store in shared context for tool/agent access (if needed)
        from xhs_seo_optimizer.shared_context import shared_context
        shared_context.set("success_profile_report", success_profile)
        shared_context.set("audit_report", audit_report)

        # Flatten inputs for YAML variable substitution
        inputs['note_id'] = audit_report['note_id']
        inputs['current_metrics'] = current_metrics
        inputs['target_means'] = target_means
        inputs['target_stds'] = target_stds
        inputs['sample_size'] = aggregated_stats['sample_size']

        # Keep original reports for agent context
        inputs['success_profile_report'] = success_profile
        inputs['audit_report'] = audit_report

        return inputs

    def kickoff(self, inputs: Dict[str, Any], save_to_file: bool = True) -> GapReport:
        """Execute gap analysis and return structured output.

        The generate_gap_report task uses response_model=GapReport with
        OpenAICompletion + Gemini for native structured output. No post-processing needed.

        Args:
            inputs: Must contain:
                - success_profile_report: SuccessProfileReport dict or JSON
                - audit_report: AuditReport dict or JSON (must have current_metrics field)
                - keyword: str
            save_to_file: Whether to save output to outputs/gap_report.json.
                Set to False when running in Flow mode (state handles data passing).

        Returns:
            GapReport: Validated Pydantic model instance
        """
        print("\n" + "="*60)
        print("🚀 Starting GapFinder crew execution...")
        print("="*60 + "\n")

        # Execute CrewAI task flow
        # The last task (generate_gap_report) uses response_model for native structured output
        crew_result = self.crew().kickoff(inputs=inputs)

        # Parse the structured output from crew result
        # When response_model is used, crew_result.raw contains JSON string
        if crew_result.pydantic:
            # If CrewAI already parsed it as Pydantic
            structured_report = crew_result.pydantic
        else:
            # Parse from raw JSON string
            structured_report = GapReport.model_validate_json(crew_result.raw)

        # Save report to file (optional)
        if save_to_file:
            self._save_gap_report(structured_report)

        print("\n" + "="*60)
        print("✅ GapReport generated successfully!")
        print("="*60 + "\n")

        return structured_report

    def _save_gap_report(self, report: GapReport) -> None:
        """Save GapReport to outputs/gap_report.json.

        Args:
            report: GapReport Pydantic model instance
        """
        os.makedirs("outputs", exist_ok=True)
        output_path = "outputs/gap_report.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report.model_dump_json(indent=2, ensure_ascii=False))

        print(f"✓ GapReport saved to {output_path}")
