"""Gap Finder Crew - 差距定位员.

Identifies statistically significant performance gaps between owned_note and target_notes,
then maps gaps to actionable content features.
"""

from crewai import Agent, Crew, Task, LLM
from crewai.project import CrewBase, agent, task, crew, before_kickoff
from typing import Any, Dict
import json
import os

from .tools.statistical_delta import StatisticalDeltaTool
from .models.reports import GapReport
from .models.analysis_results import GapAnalysis
from .utils import format_json_output


@CrewBase
class XhsSeoOptimizerCrewGapFinder:
    """差距定位员 Crew - Gap Finder for performance analysis.

    Compares SuccessProfileReport (what works) with AuditReport (current state)
    to identify critical performance gaps and their root causes.
    """

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks_gap_finder.yaml'

    def __init__(self):
        """Initialize Gap Finder crew."""
        self.shared_context = {}

        # LLM configuration for OpenRouter with retry
        # Note: response_format not supported for OpenRouter provider
        # Rely on guardrails + output_pydantic for JSON validation
        llm_config = {
            'base_url': 'https://openrouter.ai/api/v1',
            'api_key': os.getenv("OPENROUTER_API_KEY", ""),
            'temperature': 0.1,
            'num_retries': 3,  # LiteLLM auto-retry on API errors
        }

        self.custom_llm = LLM(
            model='openrouter/google/gemini-2.5-flash-lite',
            **llm_config
        )

    @agent
    def gap_finder(self) -> Agent:
        """差距定位员 agent.

        Data-driven analyst that identifies statistical gaps and maps them
        to specific content features.
        """
        return Agent(
            config=self.agents_config['gap_finder'],
            tools=[StatisticalDeltaTool()],
            llm=self.custom_llm,
            verbose=True,
            allow_delegation=False
        )

    @task
    def calculate_statistical_gaps(self) -> Task:
        """Task 1: Calculate statistical gaps using StatisticalDeltaTool.

        Output: GapAnalysis with significant_gaps, marginal_gaps, non_significant_gaps.
        Each Gap includes related_features and rationale from attribution.py.
        """
        return Task(
            config=self.tasks_config['calculate_statistical_gaps'],
            agent=self.gap_finder(),
            output_pydantic=GapAnalysis,  # Enforce output schema
            guardrails=[format_json_output],  # Extract JSON from trailing text
            guardrail_max_retries=2
        )

    @task
    def generate_gap_report(self) -> Task:
        """Task 2: Generate final GapReport JSON.

        Merges feature mapping, prioritization, and report generation into one task.
        Output: GapReport conforming to the final report schema with all metadata.
        """
        return Task(
            config=self.tasks_config['generate_gap_report'],
            agent=self.gap_finder(),
            context=[self.calculate_statistical_gaps()],
            output_pydantic=GapReport,  # Final output validation
            guardrails=[format_json_output],  # Extract JSON from markdown/duplicate outputs
            guardrail_max_retries=2
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

    def kickoff(self, inputs: Dict[str, Any], save_to_file: bool = True) -> Any:
        """Execute gap analysis.

        Args:
            inputs: Must contain:
                - success_profile_report: SuccessProfileReport dict or JSON
                - audit_report: AuditReport dict or JSON (must have current_metrics field)
                - keyword: str
            save_to_file: Whether to save output to outputs/gap_report.json.
                Set to False when running in Flow mode (state handles data passing).

        Returns:
            CrewOutput with GapReport as pydantic attribute
        """
        # @before_kickoff will validate and flatten inputs automatically
        # Execute crew
        result = self.crew().kickoff(inputs=inputs)

        # Save output to file (optional, skip in Flow mode)
        if save_to_file:
            self._save_gap_report(result)

        return result

    def _save_gap_report(self, result: Any):
        """Save GapReport to outputs/gap_report.json.

        Args:
            result: CrewOutput from crew execution
        """
        os.makedirs("outputs", exist_ok=True)
        output_path = "outputs/gap_report.json"

        # Get JSON from result (try multiple formats for robustness)
        if hasattr(result, 'pydantic') and result.pydantic:
            # Preferred: Pydantic model
            report_json = result.pydantic.model_dump_json(indent=2, ensure_ascii=False)
        elif hasattr(result, 'json') and result.json:
            # Alternative: JSON string
            report_json = result.json
        elif hasattr(result, 'raw'):
            # Fallback: Raw output
            report_json = result.raw
        else:
            # Last resort: Convert to string
            report_json = str(result)

        # Write to file with UTF-8 encoding (for Chinese text)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_json)

        print(f"✓ GapReport saved to {output_path}")
