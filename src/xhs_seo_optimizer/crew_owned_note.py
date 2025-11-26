"""Owned Note Auditor crew for objective content understanding.

This crew provides objective analysis of owned (self-published) notes:
- Feature extraction (text + visual analysis via NLP and Vision tools)
- Content intent extraction (core_theme, target_persona, key_message)
- Visual subjects extraction (subject_type, must_preserve, original_urls)
- Marketing sensitivity detection
- Objective feature summary (no strength/weakness judgment)

Note: Strength/weakness judgment requires competitor comparison,
which will be handled by GapFinder agent (future implementation).
Feature attribution can be retrieved via attribution.py rules if needed.
"""

from crewai import Agent, Crew, Task, Process, LLM
from crewai.project import CrewBase, agent, crew, task, before_kickoff
from typing import Dict, Any, Union, Optional
import os
import json

from xhs_seo_optimizer.models.reports import AuditReport, ContentIntent, VisualSubjects
from xhs_seo_optimizer.models.note import Note
from xhs_seo_optimizer.tools import (
    MultiModalVisionTool,
    NLPAnalysisTool,
    determine_marketing_sensitivity,
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

        # ========== Phase 0001: Marketing Sensitivity Detection ==========
        # Extract marketing level from note tags
        tag_data = owned_note_data.get("tag", {})
        marketing_level = tag_data.get("note_marketing_integrated_level", "")
        is_soft_ad = marketing_level == "软广"
        marketing_sensitivity = determine_marketing_sensitivity(marketing_level)

        inputs["marketing_level"] = marketing_level
        inputs["is_soft_ad"] = is_soft_ad
        inputs["marketing_sensitivity"] = marketing_sensitivity

        # Store in shared context for downstream tasks
        shared_context.set("marketing_level", marketing_level)
        shared_context.set("is_soft_ad", is_soft_ad)
        shared_context.set("marketing_sensitivity", marketing_sensitivity)

        print(f"\n{'='*60}")
        print(f"📊 Marketing Sensitivity Analysis:")
        print(f"   marketing_level: {marketing_level}")
        print(f"   is_soft_ad: {is_soft_ad}")
        print(f"   marketing_sensitivity: {marketing_sensitivity}")
        print(f"{'='*60}\n")

        # ========== Phase 0001: Extract content for intent analysis ==========
        meta_data = owned_note_data.get("meta_data", {})
        inputs["original_title"] = meta_data.get("title", "")
        inputs["original_content"] = meta_data.get("content", "")
        inputs["original_cover_url"] = meta_data.get("cover_image_url", "")
        inputs["original_inner_urls"] = json.dumps(
            meta_data.get("inner_image_urls", []),
            ensure_ascii=False
        )

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
    def extract_content_intent_task(self) -> Task:
        """Task 2: 提取内容创作意图 (Phase 0001).

        Extracts ContentIntent from owned note:
        - core_theme: 核心主题
        - target_persona: 目标人群 (结合keyword确定)
        - key_message: 关键信息/核心卖点
        - unique_angle: 独特角度 (可选)
        - emotional_tone: 情感基调 (可选)

        Uses LLM reasoning based on title and content.
        """
        return Task(
            config=self.tasks_config['extract_content_intent'],
            agent=self.owned_note_auditor()
        )

    @task
    def extract_visual_subjects_task(self) -> Task:
        """Task 3: 提取视觉主体信息 (Phase 0001).

        Extracts VisualSubjects from owned note images:
        - subject_type: 主体类型 (product/person/brand/scene/none)
        - subject_description: 主体描述
        - brand_elements: 品牌元素
        - must_preserve: 必须保留的元素
        - original_cover_url: 原始封面图URL
        - original_inner_urls: 原始内页图URLs

        Depends on extract_content_features_task (uses vision analysis results).
        """
        return Task(
            config=self.tasks_config['extract_visual_subjects'],
            agent=self.owned_note_auditor(),
            context=[
                self.extract_content_features_task()
            ]
        )

    @task
    def generate_audit_report_task(self) -> Task:
        """Task 4: 生成审计报告.

        Generates final AuditReport with objective analysis:
        - text_features, visual_features
        - content_intent (Phase 0001)
        - visual_subjects (Phase 0001)
        - marketing_level, is_soft_ad, marketing_sensitivity (Phase 0001)
        - feature_summary (objective description, no judgment)

        Depends on all previous tasks.
        """
        return Task(
            config=self.tasks_config['generate_audit_report'],
            agent=self.owned_note_auditor(),
            context=[
                self.extract_content_features_task(),
                self.extract_content_intent_task(),
                self.extract_visual_subjects_task()
            ],
            output_pydantic=AuditReport,  # Final output validation
            output_file="outputs/audit_report.json"
        )

    @crew
    def crew(self) -> Crew:
        """Assemble the owned note auditor crew.

        Uses sequential process with 4 chained tasks (Phase 0001 updated):
        1. extract_content_features_task - 提取文本和视觉特征
        2. extract_content_intent_task - 提取内容创作意图
        3. extract_visual_subjects_task - 提取视觉主体信息
        4. generate_audit_report_task - 生成完整审计报告
        """
        return Crew(
            agents=self.agents,  # Contains owned_note_auditor
            tasks=self.tasks,    # Contains 4 sequential tasks
            process=Process.sequential,  # Execute tasks in order
            verbose=True
        )
