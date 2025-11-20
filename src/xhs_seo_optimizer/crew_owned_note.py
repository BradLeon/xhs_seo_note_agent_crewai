"""Owned Note Auditor crew for diagnosing owned note weaknesses.

This crew analyzes owned (self-published) notes to identify:
- Metric weaknesses (which prediction metrics are underperforming)
- Content gaps (missing or weak features)
- Content strengths (features that are already working well)
"""

from crewai import Agent, Crew, Task, Process, LLM
from crewai.project import CrewBase, agent, crew, task, before_kickoff
from typing import Dict, Any, Union
import os

from xhs_seo_optimizer.models.reports import AuditReport
from xhs_seo_optimizer.models.note import Note
from xhs_seo_optimizer.tools import (
    MultiModalVisionTool,
    NLPAnalysisTool,
)


@CrewBase
class XhsSeoOptimizerCrewOwnedNote:
    """Crew for auditing owned notes.

    This crew includes:
    - owned_note_auditor agent
    - 4 sequential tasks: extract features → analyze metrics → identify weaknesses → generate report

    Uses sequential process with a single agent executing all tasks.

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
        llm_config = {
            'base_url': 'https://openrouter.ai/api/v1',
            'api_key': os.getenv("OPENROUTER_API_KEY", ""),
            'temperature': 0.1
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
        inputs["validated"] = True

        return inputs

    @agent
    def owned_note_auditor(self) -> Agent:
        """自营笔记审计员 agent.

        The agent that analyzes owned notes to identify weaknesses and strengths.
        Uses NLP and Vision tools to extract features, then diagnoses issues.
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
    def analyze_metric_performance_task(self) -> Task:
        """Task 2: 分析指标表现.

        Analyzes the 10 prediction metrics to identify which are underperforming
        (weak_metrics) and which are strong (strong_metrics).

        Depends on Task 1 for context.
        """
        return Task(
            config=self.tasks_config['analyze_metric_performance'],
            agent=self.owned_note_auditor(),
            context=[self.extract_content_features_task()]
        )

    @task
    def identify_weaknesses_task(self) -> Task:
        """Task 3: 识别弱点和优势.

        Compares extracted features against best practices to identify:
        - content_weaknesses: Missing or weak features
        - content_strengths: Features that are already working well

        Depends on Tasks 1 & 2 for comprehensive analysis.
        """
        return Task(
            config=self.tasks_config['identify_weaknesses'],
            agent=self.owned_note_auditor(),
            context=[
                self.extract_content_features_task(),
                self.analyze_metric_performance_task()
            ]
        )

    @task
    def generate_audit_report_task(self) -> Task:
        """Task 4: 生成审计报告.

        Generates final AuditReport with all findings:
        - current_metrics
        - text_features, visual_features
        - weak_metrics, strong_metrics
        - content_weaknesses, content_strengths
        - overall_diagnosis

        Depends on all previous tasks.
        """
        return Task(
            config=self.tasks_config['generate_audit_report'],
            agent=self.owned_note_auditor(),
            context=[
                self.extract_content_features_task(),
                self.analyze_metric_performance_task(),
                self.identify_weaknesses_task()
            ],
            output_pydantic=AuditReport,  # Final output validation
            output_file="outputs/audit_report.json"
        )

    @crew
    def crew(self) -> Crew:
        """Assemble the owned note auditor crew.

        Uses sequential process with 4 chained tasks:
        1. extract_content_features_task
        2. analyze_metric_performance_task (depends on 1)
        3. identify_weaknesses_task (depends on 1, 2)
        4. generate_audit_report_task (depends on 1, 2, 3)
        """
        return Crew(
            agents=self.agents,  # Contains owned_note_auditor
            tasks=self.tasks,    # Contains 4 sequential tasks
            process=Process.sequential,  # Execute tasks in order
            verbose=True
        )
