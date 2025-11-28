"""Optimization Strategist Crew - 优化策略师.

Transforms GapReport insights into actionable content optimization plans,
including title alternatives, content rewrites, visual prompts, and image generation.

Phase 0001 updates:
- Content intent constraint injection for consistency
- Visual subjects constraint for image generation
- Marketing sensitivity control for soft-ad notes
- Actual image generation via ImageGeneratorTool
- Final OptimizedNote output in Note format
"""

from crewai import Agent, Crew, Task, LLM
from crewai.project import CrewBase, agent, task, crew, before_kickoff
from typing import Any, Dict, List, Optional
import json
import os
from datetime import datetime

from .utils import format_json_output
from .models.reports import (
    OptimizationPlan,
    OptimizedNote,
    GeneratedImages,
    GeneratedImage,
    ContentIntent,
    VisualSubjects,
    MarketingCheck
)
from .tools import ImageGeneratorTool, MarketingSentimentTool


@CrewBase
class XhsSeoOptimizerCrewOptimization:
    """优化策略师 Crew - Optimization Strategist for content optimization.

    Transforms GapReport (performance gaps) into actionable OptimizationPlan
    with specific content modifications and visual prompts.
    """

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks_optimization.yaml'

    def __init__(self):
        """Initialize Optimization Strategist crew."""
        self.shared_context = {}

        # LLM configuration for OpenRouter with retry
        # Note: response_format not supported for OpenRouter provider
        # Rely on guardrails + output_pydantic for JSON validation
        llm_config = {
            'base_url': 'https://openrouter.ai/api/v1',
            'api_key': os.getenv("OPENROUTER_API_KEY", ""),
            'temperature': 0.0,
            'num_retries': 3,  # LiteLLM auto-retry on API errors
        }

        self.custom_llm = LLM(
            model='openrouter/google/gemini-2.5-flash',
            **llm_config
        )


    @agent
    def optimization_strategist(self) -> Agent:
        """优化策略师 agent.

        Creative strategist that transforms gap analysis into actionable
        content optimizations with specific, executable recommendations.
        """
        return Agent(
            config=self.agents_config['optimization_strategist'],
            tools=[MarketingSentimentTool()],  # Phase 0001: Add marketing check tool
            verbose=True,
            llm=self.custom_llm,
            allow_delegation=False
        )

    @agent
    def image_generator(self) -> Agent:
        """图像生成 agent (Phase 0001).

        Specialized agent for image generation using AIGC tools.
        Uses ImageGeneratorTool to generate images based on visual prompts.
        """
        return Agent(
            config=self.agents_config['image_generator'],
            tools=[ImageGeneratorTool()],
            verbose=True,
            llm=self.custom_llm,
            allow_delegation=False
        )

    @task
    def generate_text_optimizations(self) -> Task:
        """Task 1: Generate text content optimizations.

        Output: Text optimization items including title alternatives,
        opening hook, ending CTA, and hashtags.
        """
        return Task(
            config=self.tasks_config['generate_text_optimizations'],
            agent=self.optimization_strategist()
        )

    @task
    def generate_visual_prompts(self) -> Task:
        """Task 2: Generate visual optimization prompts for AIGC.

        Output: VisualPrompt objects for cover and inner images.
        """
        return Task(
            config=self.tasks_config['generate_visual_prompts'],
            agent=self.optimization_strategist(),
            context=[self.generate_text_optimizations()]
        )

    @task
    def compile_optimization_plan(self) -> Task:
        """Task 3: Compile final OptimizationPlan.

        Output: Complete OptimizationPlan with all optimizations,
        priority summary, and expected impact.
        """
        return Task(
            config=self.tasks_config['compile_optimization_plan'],
            agent=self.optimization_strategist(),
            context=[
                self.generate_text_optimizations(),
                self.generate_visual_prompts()
            ],
            output_pydantic=OptimizationPlan,  # Final output validation
            guardrails=[format_json_output],  # Extract JSON from markdown/prefix
            guardrail_max_retries=2
        )

    @task
    def generate_images(self) -> Task:
        """Task 4: Generate actual images using AIGC (Phase 0001).

        Calls ImageGeneratorTool to generate images based on visual prompts.
        Uses original images as reference for style/subject consistency.

        Output: GeneratedImages with cover and inner image results.
        """
        return Task(
            config=self.tasks_config['generate_images'],
            agent=self.image_generator(),
            context=[
                self.generate_visual_prompts(),
                self.compile_optimization_plan()
            ]
        )

    @task
    def compile_optimized_note(self) -> Task:
        """Task 5: Compile final OptimizedNote in Note format (Phase 0001).

        Integrates all optimization results into a publishable Note format.
        Includes marketing check if original note was soft-ad.

        Output: OptimizedNote (can be directly published).
        """
        return Task(
            config=self.tasks_config['compile_optimized_note'],
            agent=self.optimization_strategist(),
            context=[
                self.compile_optimization_plan(),
                self.generate_images()
            ],
            output_pydantic=OptimizedNote,
            output_file="outputs/optimized_note.json",
            guardrails=[format_json_output],  # Extract JSON from markdown/prefix
            guardrail_max_retries=2
        )

    def _flatten_text_features(self, text_features: Dict) -> Dict[str, str]:
        """Extract and flatten text features for YAML variable substitution."""
        return {
            'current_opening_hook': text_features.get('opening_hook', ''),
            'current_ending_cta': text_features.get('ending_cta', ''),
            'current_title_pattern': text_features.get('title_pattern', ''),
            'current_title_emotion': text_features.get('title_emotion', ''),
            'current_content_framework': text_features.get('content_framework', ''),
        }

    def _flatten_visual_features(self, visual_features: Dict) -> Dict[str, str]:
        """Extract and flatten visual features for YAML variable substitution."""
        return {
            'current_thumbnail_appeal': visual_features.get('thumbnail_appeal', ''),
            'current_color_scheme': visual_features.get('color_scheme', ''),
            'current_visual_tone': visual_features.get('visual_tone', ''),
        }

    @crew
    def crew(self) -> Crew:
        """Create OptimizationStrategist crew with sequential task execution.

        Phase 0001 updated: 5 sequential tasks:
        1. generate_text_optimizations - 文本优化 (with ContentIntent constraint)
        2. generate_visual_prompts - 视觉prompt生成 (with VisualSubjects constraint)
        3. compile_optimization_plan - 编译优化方案
        4. generate_images - 实际生成图片 (ImageGeneratorTool)
        5. compile_optimized_note - 格式化为Note输出
        """
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True
        )

    @before_kickoff
    def validate_and_flatten_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and flatten inputs before crew execution.

        Supports two modes:
        1. FLOW MODE: Reports passed directly in inputs dict (from Flow state)
        2. STANDALONE MODE: Reports loaded from outputs/ directory (backward compatible)

        Args:
            inputs: Must contain:
                - keyword: str (target keyword)
            Optional (for FLOW mode):
                - gap_report: dict
                - audit_report: dict
                - success_profile_report: dict
                - owned_note: dict

        Returns:
            Flattened dict with all report data for YAML variable substitution

        Raises:
            ValueError: If required data is missing or invalid
        """
        # Validate keyword
        if 'keyword' not in inputs:
            raise ValueError("inputs must contain 'keyword'")

        keyword = inputs['keyword']

        # ========================================
        # Mode Detection: FLOW vs STANDALONE
        # ========================================
        # If all reports are provided in inputs, use FLOW mode
        # Otherwise, load from files (STANDALONE mode)
        is_flow_mode = all(
            key in inputs
            for key in ['gap_report', 'audit_report', 'success_profile_report', 'owned_note']
        )

        print(f"\n{'='*80}")
        print(f"🔍 DEBUG: validate_and_flatten_inputs called with keyword: {keyword}")
        print(f"🔍 DEBUG: Mode: {'FLOW' if is_flow_mode else 'STANDALONE'}")
        print(f"{'='*80}\n")

        if is_flow_mode:
            # ========================================
            # FLOW MODE: Use reports from inputs
            # ========================================
            gap_report = inputs['gap_report']
            audit_report = inputs['audit_report']
            success_profile_report = inputs['success_profile_report']
            owned_note = inputs['owned_note']

            # Handle Pydantic model inputs (convert to dict if needed)
            if hasattr(gap_report, 'model_dump'):
                gap_report = gap_report.model_dump()
            if hasattr(audit_report, 'model_dump'):
                audit_report = audit_report.model_dump()
            if hasattr(success_profile_report, 'model_dump'):
                success_profile_report = success_profile_report.model_dump()
            if hasattr(owned_note, 'model_dump'):
                owned_note = owned_note.model_dump()

            print("[OptimizationCrew] Running in FLOW mode - using reports from inputs")
        else:
            # ========================================
            # STANDALONE MODE: Load from files
            # ========================================
            print("[OptimizationCrew] Running in STANDALONE mode - loading reports from files")

            # Load gap_report.json
            gap_report_path = "outputs/gap_report.json"
            if not os.path.exists(gap_report_path):
                raise ValueError(f"Gap report not found: {gap_report_path}. Run GapFinder first.")
            with open(gap_report_path, 'r', encoding='utf-8') as f:
                gap_report = json.load(f)

            # Load audit_report.json
            audit_report_path = "outputs/audit_report.json"
            if not os.path.exists(audit_report_path):
                raise ValueError(f"Audit report not found: {audit_report_path}. Run OwnedNoteAuditor first.")
            with open(audit_report_path, 'r', encoding='utf-8') as f:
                audit_report = json.load(f)

            # Load success_profile_report.json
            success_profile_path = "outputs/success_profile_report.json"
            if not os.path.exists(success_profile_path):
                raise ValueError(f"Success profile not found: {success_profile_path}. Run CompetitorAnalyst first.")
            with open(success_profile_path, 'r', encoding='utf-8') as f:
                success_profile_report = json.load(f)

            # Load owned_note.json for original content
            owned_note_path = "docs/owned_note.json"
            if not os.path.exists(owned_note_path):
                raise ValueError(f"Owned note not found: {owned_note_path}")
            with open(owned_note_path, 'r', encoding='utf-8') as f:
                owned_note = json.load(f)

        # Store in shared context
        from xhs_seo_optimizer.shared_context import shared_context
        shared_context.set("gap_report", gap_report)
        shared_context.set("audit_report", audit_report)
        shared_context.set("success_profile_report", success_profile_report)
        shared_context.set("owned_note", owned_note)
        shared_context.set("top_priority_metrics", gap_report.get('top_priority_metrics', []))

        # Extract key fields for YAML variable substitution
        inputs['note_id'] = gap_report.get('owned_note_id', owned_note.get('note_id'))
        inputs['original_note_id'] = inputs['note_id']  # Alias for task YAML template
        inputs['original_title'] = owned_note.get('title', '')
        inputs['original_content'] = owned_note.get('content', '')

        # Extract gap report insights
        inputs['top_priority_metrics'] = gap_report.get('top_priority_metrics', [])
        inputs['root_causes'] = gap_report.get('root_causes', [])
        inputs['impact_summary'] = gap_report.get('impact_summary', '')

        # Extract weak features and missing features from all gaps
        all_weak_features = set()
        all_missing_features = set()
        for gap_list in [gap_report.get('significant_gaps', []),
                         gap_report.get('marginal_gaps', []),
                         gap_report.get('non_significant_gaps', [])]:
            for gap in gap_list:
                all_weak_features.update(gap.get('weak_features', []))
                all_missing_features.update(gap.get('missing_features', []))
        inputs['all_weak_features'] = list(all_weak_features)
        inputs['all_missing_features'] = list(all_missing_features)

        # Build dynamic optimization context using attribution rules
        from xhs_seo_optimizer.attribution import build_optimization_context
        optimization_context = build_optimization_context(
            priority_metrics=inputs['top_priority_metrics'],
            weak_features=list(all_weak_features),
            missing_features=list(all_missing_features)
        )
        inputs['optimization_context'] = optimization_context
        # Store optimization_context in shared context for validation use
        shared_context.set("optimization_context", optimization_context)

        # DEBUG: Print optimization context
        print(f"\n{'='*80}")
        print(f"🔍 DEBUG: optimization_context generated:")
        print(f"  Priority metrics: {[m['metric'] for m in optimization_context['priority_metrics']]}")
        print(f"  Features to optimize: {list(optimization_context['features_to_optimize'].keys())}")
        print(f"  Features by content area: {list(optimization_context['features_by_content_area'].keys())}")
        for area, features in optimization_context['features_by_content_area'].items():
            print(f"    {area}: {features}")
        print(f"{'='*80}\n")

        # DEBUG: Print original note content that LLM should use
        print(f"\n{'='*80}")
        print(f"🔍 DEBUG: Original owned_note content (这是LLM应该使用的原始内容):")
        print(f"  keyword: {inputs['keyword']}")
        print(f"  original_title: {inputs['original_title']}")
        print(f"  original_content (前200字符): {inputs['original_content'][:200]}...")
        print(f"  owned_note.title: {owned_note.get('title', 'N/A')}")
        print(f"  owned_note.content (前100字符): {owned_note.get('content', 'N/A')[:100]}...")
        print(f"{'='*80}\n")

        # Keep full reports for agent context
        inputs['gap_report'] = gap_report
        inputs['audit_report'] = audit_report
        inputs['success_profile_report'] = success_profile_report
        inputs['owned_note'] = owned_note

        # Extract text features for optimization context
        inputs['text_features'] = audit_report.get('text_features', {})
        inputs['visual_features'] = audit_report.get('visual_features', {})

        # Flatten text features for YAML variable substitution
        text_features = audit_report.get('text_features', {})
        inputs.update(self._flatten_text_features(text_features))

        # Flatten visual features for YAML variable substitution
        visual_features = audit_report.get('visual_features', {})
        inputs.update(self._flatten_visual_features(visual_features))

        # ========================================
        # 条件逻辑判断：在Python中判断，而非让LLM判断
        # ========================================

        # 1. 判断是否需要优化visual区域
        need_visual_optimization = 'visual' in optimization_context.get('features_by_content_area', {})
        inputs['need_visual_optimization'] = need_visual_optimization

        if need_visual_optimization:
            visual_features_list = optimization_context['features_by_content_area']['visual']
            inputs['visual_features_to_optimize'] = ', '.join(visual_features_list)
        else:
            inputs['visual_features_to_optimize'] = ''

        # 2. 拍平priority_metrics为简单变量
        inputs['priority_metrics_list'] = ', '.join(inputs['top_priority_metrics'])
        inputs['priority_metrics_count'] = len(inputs['top_priority_metrics'])

        # 3. 拍平priority_metrics的详细信息（包含rationale）
        priority_metrics_info = []
        for m in optimization_context['priority_metrics']:
            priority_metrics_info.append(f"{m['metric']}: {m['rationale']}")
        inputs['priority_metrics_with_rationale'] = '\n'.join(priority_metrics_info)

        # 4. 按内容区域分组特征（拍平为独立变量）
        for area in ['title', 'opening', 'body', 'ending', 'hashtags']:
            features = optimization_context['features_by_content_area'].get(area, [])
            if features:
                inputs[f'{area}_features_to_optimize'] = ', '.join(features)
            else:
                inputs[f'{area}_features_to_optimize'] = ''

        # DEBUG: Print flattened variables for YAML
        print(f"\n{'='*80}")
        print(f"🔍 DEBUG: 拍平后的变量（YAML 可直接使用）:")
        print(f"  keyword: {inputs['keyword']}")
        print(f"  original_title: {inputs['original_title']}")
        print(f"  original_content (前100字符): {inputs['original_content'][:100]}...")
        print(f"  current_opening_hook: {inputs.get('current_opening_hook', 'N/A')}")
        print(f"  current_ending_cta: {inputs.get('current_ending_cta', 'N/A')}")
        print(f"  current_title_pattern: {inputs.get('current_title_pattern', 'N/A')}")
        print(f"  current_title_emotion: {inputs.get('current_title_emotion', 'N/A')}")
        print(f"  current_thumbnail_appeal: {inputs.get('current_thumbnail_appeal', 'N/A')}")
        print(f"  current_color_scheme: {inputs.get('current_color_scheme', 'N/A')}")
        print(f"{'='*80}\n")

        # DEBUG: Print optimization context variables
        print(f"\n{'='*80}")
        print(f"🔍 DEBUG: Optimization Context Variables (for LLM):")
        print(f"  priority_metrics_list: {inputs['priority_metrics_list']}")
        print(f"  priority_metrics_count: {inputs['priority_metrics_count']}")
        print(f"  need_visual_optimization: {inputs['need_visual_optimization']}")
        print(f"  visual_features_to_optimize: {inputs['visual_features_to_optimize']}")
        print(f"  title_features_to_optimize: {inputs['title_features_to_optimize']}")
        print(f"  opening_features_to_optimize: {inputs['opening_features_to_optimize']}")
        print(f"  body_features_to_optimize: {inputs['body_features_to_optimize']}")
        print(f"  ending_features_to_optimize: {inputs['ending_features_to_optimize']}")
        print(f"  hashtags_features_to_optimize: {inputs['hashtags_features_to_optimize']}")
        print(f"{'='*80}\n")

        # ========================================
        # Phase 0001: Load ContentIntent and VisualSubjects from audit_report
        # ========================================

        # Extract ContentIntent (for content consistency)
        # Variable names prefixed with content_intent_ to match YAML template
        content_intent = audit_report.get('content_intent', {})
        if content_intent:
            inputs['content_intent_core_theme'] = content_intent.get('core_theme', '')
            inputs['content_intent_target_persona'] = content_intent.get('target_persona', '')
            inputs['content_intent_key_message'] = content_intent.get('key_message', '')
            inputs['content_intent_unique_angle'] = content_intent.get('unique_angle', '')
            inputs['content_intent_emotional_tone'] = content_intent.get('emotional_tone', '')
            inputs['content_intent'] = content_intent
            shared_context.set("content_intent", content_intent)
        else:
            # Fallback: extract from owned_note if not in audit_report
            inputs['content_intent_core_theme'] = ''
            inputs['content_intent_target_persona'] = ''
            inputs['content_intent_key_message'] = ''
            inputs['content_intent_unique_angle'] = ''
            inputs['content_intent_emotional_tone'] = ''
            inputs['content_intent'] = {}

        # Extract VisualSubjects (for image generation consistency)
        # Variable names prefixed with visual_ to match YAML template
        visual_subjects = audit_report.get('visual_subjects', {})
        if visual_subjects:
            inputs['visual_subject_type'] = visual_subjects.get('subject_type', 'none')
            inputs['visual_subject_description'] = visual_subjects.get('subject_description', '')
            inputs['visual_brand_elements'] = json.dumps(
                visual_subjects.get('brand_elements', []),
                ensure_ascii=False
            )
            inputs['visual_must_preserve'] = json.dumps(
                visual_subjects.get('must_preserve', []),
                ensure_ascii=False
            )
            inputs['original_cover_url'] = visual_subjects.get('original_cover_url', '')
            inputs['original_inner_urls'] = json.dumps(
                visual_subjects.get('original_inner_urls', []),
                ensure_ascii=False
            )
            inputs['visual_subjects'] = visual_subjects
            shared_context.set("visual_subjects", visual_subjects)
        else:
            # Fallback: use owned_note data
            meta_data = owned_note.get('meta_data', {})
            inputs['visual_subject_type'] = 'none'
            inputs['visual_subject_description'] = ''
            inputs['visual_brand_elements'] = '[]'
            inputs['visual_must_preserve'] = '[]'
            inputs['original_cover_url'] = meta_data.get('cover_image_url', '')
            inputs['original_inner_urls'] = json.dumps(
                meta_data.get('inner_image_urls', []),
                ensure_ascii=False
            )
            inputs['visual_subjects'] = {}

        # Extract Marketing sensitivity (for soft-ad control)
        inputs['marketing_level'] = audit_report.get('marketing_level', '')
        inputs['is_soft_ad'] = audit_report.get('is_soft_ad', False)
        inputs['marketing_sensitivity'] = audit_report.get('marketing_sensitivity', 'low')
        shared_context.set("marketing_sensitivity", inputs['marketing_sensitivity'])
        shared_context.set("is_soft_ad", inputs['is_soft_ad'])

        # DEBUG: Print Phase 0001 variables
        print(f"\n{'='*80}")
        print(f"🔍 DEBUG: Phase 0001 Variables:")
        print(f"  ContentIntent:")
        print(f"    content_intent_core_theme: {inputs.get('content_intent_core_theme', 'N/A')}")
        print(f"    content_intent_target_persona: {inputs.get('content_intent_target_persona', 'N/A')}")
        print(f"    content_intent_key_message: {inputs.get('content_intent_key_message', 'N/A')}")
        print(f"    content_intent_unique_angle: {inputs.get('content_intent_unique_angle', 'N/A')}")
        print(f"    content_intent_emotional_tone: {inputs.get('content_intent_emotional_tone', 'N/A')}")
        print(f"  VisualSubjects:")
        print(f"    visual_subject_type: {inputs.get('visual_subject_type', 'N/A')}")
        print(f"    visual_subject_description: {inputs.get('visual_subject_description', 'N/A')[:50]}...")
        print(f"    visual_must_preserve: {inputs.get('visual_must_preserve', 'N/A')}")
        print(f"    original_cover_url: {inputs.get('original_cover_url', 'N/A')[:50]}...")
        print(f"  Marketing:")
        print(f"    marketing_sensitivity: {inputs.get('marketing_sensitivity', 'N/A')}")
        print(f"    is_soft_ad: {inputs.get('is_soft_ad', 'N/A')}")
        print(f"    marketing_level: {inputs.get('marketing_level', 'N/A')}")
        print(f"{'='*80}\n")

        return inputs

    def kickoff(self, inputs: Dict[str, Any]) -> Any:
        """Execute optimization strategy generation.

        Args:
            inputs: Must contain:
                - keyword: str (target keyword)

        Returns:
            CrewOutput with OptimizationPlan as pydantic attribute
        """
        # @before_kickoff will validate and load all inputs automatically
        # Execute crew
        result = self.crew().kickoff(inputs=inputs)

        # Save output to file
        self._save_optimization_plan(result)

        return result

    def _validate_and_fix_output(self, result: Any, priority_metrics: List[str]) -> Any:
        """验证并修正LLM输出，防止幻觉.

        Args:
            result: CrewOutput from crew execution
            priority_metrics: List of priority metrics that should be predicted

        Returns:
            Modified result with validated expected_impact (if applicable)
        """
        if not hasattr(result, 'pydantic') or not result.pydantic:
            return result

        pydantic_obj = result.pydantic

        # Phase 0001: 最终输出是 OptimizedNote，没有 expected_impact
        # 只有中间输出 OptimizationPlan 才有 expected_impact
        if not hasattr(pydantic_obj, 'expected_impact'):
            print(f"\n🔍 DEBUG: 输出类型为 {type(pydantic_obj).__name__}，跳过 expected_impact 验证")
            return result

        plan = pydantic_obj

        # 1. 验证expected_impact只包含priority_metrics
        if plan.expected_impact:
            invalid_metrics = set(plan.expected_impact.keys()) - set(priority_metrics)
            missing_metrics = set(priority_metrics) - set(plan.expected_impact.keys())

            if invalid_metrics:
                print(f"\n⚠️  WARNING: LLM预测了不在priority_metrics中的指标: {invalid_metrics}")
                print(f"   Priority metrics应该是: {priority_metrics}")
                # 移除无效指标
                for metric in invalid_metrics:
                    del plan.expected_impact[metric]
                    print(f"   已移除无效指标: {metric}")

            if missing_metrics:
                print(f"\n⚠️  WARNING: LLM缺少以下priority_metrics的预测: {missing_metrics}")
                # 添加占位符
                for metric in missing_metrics:
                    plan.expected_impact[metric] = f"优化方案针对{metric}的预期效果待评估"
                    print(f"   已添加占位符: {metric}")

        # 2. 验证visual optimization逻辑一致性 (只对 OptimizationPlan)
        if hasattr(plan, 'visual_optimization'):
            has_cover_prompt = plan.visual_optimization and plan.visual_optimization.cover_prompt is not None

            # 从shared_context获取optimization_context
            from xhs_seo_optimizer.shared_context import shared_context
            optimization_context = shared_context.get('optimization_context', {})
            has_visual_features = 'visual' in optimization_context.get('features_by_content_area', {})

            if has_visual_features and not has_cover_prompt:
                print(f"\n⚠️  WARNING: 需要优化visual特征但LLM未生成cover_prompt")
                visual_features = optimization_context['features_by_content_area']['visual']
                print(f"   Visual features: {visual_features}")
            elif not has_visual_features and has_cover_prompt:
                print(f"\n⚠️  WARNING: 不需要优化visual但LLM生成了cover_prompt（可能是误判）")

        return result

    def _save_optimization_plan(self, result: Any):
        """Save crew output to outputs directory.

        Phase 0001: Output is OptimizedNote (saved to optimized_note.json)
        Legacy: Output was OptimizationPlan (saved to optimization_plan.json)

        Args:
            result: CrewOutput from crew execution
        """
        os.makedirs("outputs", exist_ok=True)

        # 验证并修正输出
        from xhs_seo_optimizer.shared_context import shared_context
        priority_metrics = shared_context.get('top_priority_metrics', [])
        if not priority_metrics:
            # 从gap_report中提取
            gap_report = shared_context.get('gap_report', {})
            priority_metrics = gap_report.get('top_priority_metrics', [])

        if priority_metrics:
            print(f"\n🔍 DEBUG: 找到priority_metrics进行验证: {priority_metrics}")
            result = self._validate_and_fix_output(result, priority_metrics)
        else:
            print("\n⚠️  WARNING: 无法获取priority_metrics，跳过验证")

        # Determine output type and filename
        output_type = "unknown"
        if hasattr(result, 'pydantic') and result.pydantic:
            output_type = type(result.pydantic).__name__

        # Phase 0001: OptimizedNote -> optimized_note.json
        # Legacy: OptimizationPlan -> optimization_plan.json
        if output_type == "OptimizedNote":
            output_path = "outputs/optimized_note.json"
        else:
            output_path = "outputs/optimization_plan.json"

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

        print(f"✓ {output_type} saved to {output_path}")
