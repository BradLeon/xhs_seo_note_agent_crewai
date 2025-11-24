"""Owned Note Auditor crew for objective content understanding.

This crew provides objective analysis of owned (self-published) notes:
- Feature extraction (text + visual analysis via NLP and Vision tools)
- Objective feature summary (no strength/weakness judgment)

Note: Strength/weakness judgment requires competitor comparison,
which will be handled by GapFinder agent (future implementation).
Feature attribution can be retrieved via attribution.py rules if needed.
"""

from crewai import Agent, Crew, Task, Process, LLM
from crewai.project import CrewBase, agent, crew, task, before_kickoff
from typing import Dict, Any, Union
import os
import json

from xhs_seo_optimizer.models.reports import AuditReport
from xhs_seo_optimizer.models.note import Note
from xhs_seo_optimizer.tools import (
    MultiModalVisionTool,
    NLPAnalysisTool,
)


@CrewBase
class XhsSeoOptimizerCrewOwnedNote:
    """Crew for objective content understanding of owned notes.

    This crew includes:
    - owned_note_auditor agent
    - 2 sequential tasks: extract features → generate report

    Uses sequential process with a single agent executing all tasks.
    Provides objective analysis without strength/weakness judgment (requires GapFinder).

    Following official CrewAI pattern: complex objects are serialized to dicts
    using model_dump() in @before_kickoff, then stored in shared_context for tools.
    """

    agents_config = 'config/agents_owned_note.yaml'
    tasks_config = 'config/tasks_owned_note.yaml'

    def __init__(self):
        # Check if proxy is configured via environment variables
        proxy = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
        if proxy:
            print(f"使用代理 (通过环境变量): {proxy}")

        # LLM configuration for OpenRouter (same as CompetitorAnalyst)
        # Use temperature=0.0 for report generation to ensure valid JSON output
        llm_config = {
            'base_url': 'https://openrouter.ai/api/v1',
            'api_key': os.getenv("OPENROUTER_API_KEY", ""),
            'temperature': 0.0
        }

        self.custom_llm = LLM(
            model='openrouter/google/gemini-2.5-flash-lite',
            **llm_config
        )

        self.function_llm = LLM(
            model='openrouter/deepseek/deepseek-r1-0528',
            **llm_config
        )

    @before_kickoff
    def validate_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and flatten inputs before crew execution.

        Following official CrewAI pattern: serialize complex Pydantic models
        to dicts using model_dump() and store in shared_context.

        Supports both:
        1. Note Pydantic object (will be converted via model_dump())
        2. Dict with owned_note, keyword fields

        Args:
            inputs: Dict with keys:
                - owned_note: Note object or dict
                - keyword: str

        Returns:
            Flattened dict for YAML variable substitution

        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Validate required fields
        if "owned_note" not in inputs:
            raise ValueError("owned_note is required")
        if not inputs.get("keyword"):
            raise ValueError("keyword is required")

        owned_note = inputs.pop("owned_note")

        # Serialize Note object to dict
        if isinstance(owned_note, dict):
            owned_note_data = owned_note
        elif isinstance(owned_note, Note):
            owned_note_data = owned_note.model_dump()
        else:
            raise ValueError(f"owned_note must be Note object or dict, got {type(owned_note)}")

        # Validate required fields in owned_note_data
        if "note_id" not in owned_note_data:
            raise ValueError("owned_note must have note_id field")
        if "prediction" not in owned_note_data:
            raise ValueError("owned_note must have prediction field")
        if "meta_data" not in owned_note_data:
            raise ValueError("owned_note must have meta_data field")

        # Store serialized data in shared context (for tools to access)
        # Tools will use smart mode: multimodal_vision_analysis(note_id="xxx")
        from xhs_seo_optimizer.shared_context import shared_context
        shared_context.set("owned_note_data", owned_note_data)

        # Also store in inputs for metadata and variable substitution in YAML
        inputs["note_id"] = owned_note_data["note_id"]
        inputs["owned_note_data"] = owned_note_data
        # Flatten prediction dict for YAML variable substitution
        # Filter out note_id and non-numeric fields to match Dict[str, float] schema
        prediction_dict = owned_note_data["prediction"]
        current_metrics = {
            k: v for k, v in prediction_dict.items()
            if k != "note_id" and isinstance(v, (int, float))
        }
        # Store as JSON string for YAML substitution to avoid Python dict repr with single quotes
        # This ensures LLM sees proper JSON format (double quotes) and can copy it correctly
        inputs["current_metrics"] = json.dumps(current_metrics, ensure_ascii=False)
        inputs["validated"] = True

        return inputs

    @agent
    def owned_note_auditor(self) -> Agent:
        """自营笔记审计员 agent.

        The agent that provides objective content understanding of owned notes.
        Uses NLP and Vision tools to extract features and analyze feature attribution.
        Does not make strength/weakness judgments (requires GapFinder).
        """
        return Agent(
            config=self.agents_config['owned_note_auditor'],
            tools=[
                MultiModalVisionTool(),
                NLPAnalysisTool()
                # Note: No DataAggregatorTool - we analyze a single note, not statistics
            ],
            llm=self.custom_llm,
            function_calling_llm=self.function_llm,
            verbose=True
        )

    @task
    def extract_content_features_task(self) -> Task:
        """Task 1: 提取内容特征.

        Calls NLPAnalysisTool and MultiModalVisionTool to extract comprehensive
        text and visual features from the owned note.
        """
        return Task(
            config=self.tasks_config['extract_content_features'],
            agent=self.owned_note_auditor()
        )

    @task
    def generate_audit_report_task(self) -> Task:
        """Task 2: 生成审计报告.

        Generates final AuditReport with objective analysis:
        - text_features, visual_features
        - feature_summary (objective description, no judgment)

        Depends on extract_content_features_task.
        """
        return Task(
            config=self.tasks_config['generate_audit_report'],
            agent=self.owned_note_auditor(),
            context=[
                self.extract_content_features_task()
            ],
            output_pydantic=AuditReport,  # Final output validation
            output_file="outputs/audit_report.json"
        )

    @crew
    def crew(self) -> Crew:
        """Assemble the owned note auditor crew.

        Uses sequential process with 2 chained tasks:
        1. extract_content_features_task
        2. generate_audit_report_task (depends on 1)
        """
        return Crew(
            agents=self.agents,  # Contains owned_note_auditor
            tasks=self.tasks,    # Contains 2 sequential tasks
            process=Process.sequential,  # Execute tasks in order
            verbose=True
        )
