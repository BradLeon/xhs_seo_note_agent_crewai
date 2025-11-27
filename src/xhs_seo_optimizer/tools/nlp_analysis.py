"""NLP Analysis Tool - 文本分析工具.

Analyzes Xiaohongshu note text using LLM for deep structural analysis.
Deep content structure analysis including title, opening, content, ending strategies.
"""

import os
import json
from typing import Any, Dict, Optional
from openai import OpenAI
from crewai.tools import BaseTool
from pydantic import Field, ConfigDict, BaseModel

from ..models.analysis_results import TextAnalysisResult
from ..models.note import NoteMetaData


class NLPAnalysisInput(BaseModel):
    """Input schema for NLPAnalysisTool.

    Supports two calling modes:
    1. Smart mode (recommended): Pass note_id, tool auto-fetches metadata from shared_context
    2. Legacy mode: Pass note_metadata directly (backward compatible)

    Agent usage examples:
        智能模式: nlp_text_analysis(note_id="5e96b4f700000000010040e6")
        传统模式: nlp_text_analysis(note_metadata={...}, note_id="xxx")
    """
    note_id: Optional[str] = Field(
        default=None,
        description=(
            "笔记ID - 工具会自动从系统中获取该笔记的 metadata 并分析。"
            "推荐使用此模式，只需传入 note_id。"
        )
    )
    note_metadata: Optional[dict] = Field(
        default=None,
        description=(
            "可选：直接传入序列化的 NoteMetaData dict。"
            "如果不传，工具会根据 note_id 自动从系统获取。"
            "Required keys: note_id, title, content, cover_image_url."
        )
    )


class NLPAnalysisTool(BaseTool):
    """文本分析工具 (NLP Analysis Tool).

    Deep structural analysis of XHS note content using LLM.
    Analyzes title patterns, opening hooks, content framework, ending techniques, etc.

    Following official CrewAI pattern: receives serialized dict data,
    reconstructs NoteMetaData internally for processing.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "nlp_text_analysis"
    description: str = (
        "Analyzes Xiaohongshu note text structure deeply. "
        "Smart mode: Just pass note_id, tool auto-fetches metadata - nlp_text_analysis(note_id='xxx'). "
        "Legacy mode: Pass note_metadata directly. "
        "Output: TextAnalysisResult JSON with 30+ text features. "
        "使用场景：深度分析笔记内容结构，包括标题、开头、正文、结尾的套路和技巧。"
    )
    args_schema: type[BaseModel] = NLPAnalysisInput

    # OpenRouter configuration for LLM analysis
    api_key: str = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    model: str = Field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_TEXT_MODEL",
            "deepseek/deepseek-chat-v3.1"
            #openrouter/google/gemini-2.5-flash-lite
        )
    )
    site_url: str = Field(
        default_factory=lambda: os.getenv("OPENROUTER_SITE_URL", "XHS SEO Optimizer")
    )
    site_name: str = Field(
        default_factory=lambda: os.getenv("OPENROUTER_SITE_NAME", "XHS SEO Optimizer")
    )

    def _run(
        self,
        note_id: Optional[str] = None,
        note_metadata: Optional[dict] = None
    ) -> str:
        """Analyze note content structure and return analysis result.

        Supports two modes:
        1. Smart mode: Pass note_id, auto-fetch metadata from shared_context
        2. Legacy mode: Pass note_metadata directly

        Args:
            note_id: Note ID to analyze (tool will fetch metadata automatically)
            note_metadata: Optional - directly provide serialized NoteMetaData dict

        Returns:
            JSON string of TextAnalysisResult

        Raises:
            ValueError: If API key is missing or both note_id and note_metadata are missing
            RuntimeError: If LLM analysis fails
        """
        # Validate API key
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found. "
                "Please set it in .env file or environment variables."
            )

        # Smart mode: Fetch from shared_context if only note_id provided
        if not note_metadata and note_id:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Smart mode: Fetching metadata for note_id={note_id} from shared_context")

            from xhs_seo_optimizer.shared_context import shared_context
            # Support both target_notes_data (CompetitorAnalyst) and owned_note_data (OwnedNoteAuditor)
            notes = shared_context.get("target_notes_data", [])
            if not notes:
                # Try owned_note_data (single note)
                owned_note = shared_context.get("owned_note_data")
                if owned_note:
                    notes = [owned_note]

            # Find the note by note_id
            for note in notes:
                if note.get('note_id') == note_id:
                    # Support both nested (meta_data) and flat structure
                    if 'meta_data' in note:
                        note_metadata = note.get('meta_data')
                    else:
                        # Flat structure - create meta_data from top-level fields
                        note_metadata = {
                            'note_id': note.get('note_id'),
                            'title': note.get('title', ''),
                            'content': note.get('content', ''),
                            'cover_image_url': note.get('cover_image_url', ''),
                            'inner_image_urls': note.get('inner_image_urls', []),
                        }
                    logger.info(f"✓ Found note metadata for {note_id}")
                    break

            if not note_metadata:
                raise ValueError(
                    f"Note with ID '{note_id}' not found in shared_context. "
                    f"Available notes: {[n.get('note_id') for n in notes]}"
                )

        # Validate we have metadata now
        if not note_metadata or not note_metadata.get('content'):
            raise ValueError(
                "Either note_id (for smart mode) or note_metadata (for legacy mode) must be provided. "
                "note_metadata must contain 'content' field."
            )

        try:
            # Reconstruct NoteMetaData from serialized dict
            actual_meta_data = NoteMetaData(**note_metadata)

            # Analyze with LLM
            analysis_dict = self._analyze_with_llm(actual_meta_data)

            # Create TextAnalysisResult
            result = TextAnalysisResult(
                note_id=note_id or note_metadata.get('note_id', 'unknown'),
                **analysis_dict
            )

            # Return as JSON string for CrewAI tool compatibility
            return result.model_dump_json(indent=2)

        except Exception as e:
            # Return error immediately (no fallback per requirements)
            raise RuntimeError(f"NLP analysis failed: {str(e)}") from e

    def _analyze_with_llm(self, note_meta_data: NoteMetaData) -> Dict[str, Any]:
        """Analyze note content structure using LLM via OpenRouter.

        Args:
            note_meta_data: NoteMetaData object

        Returns:
            Dict with structural analysis features

        Raises:
            RuntimeError: If API call fails
        """
        # Initialize OpenRouter client
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )

        # Construct analysis prompt
        prompt = self._build_semantic_prompt(note_meta_data)

        try:
            # Call OpenRouter API with free Gemini model
            response = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                },
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more consistent analysis
            )

            # Extract and parse response
            content = response.choices[0].message.content
            return self._parse_semantic_response(content)

        except Exception as e:
            raise RuntimeError(f"LLM analysis failed: {str(e)}") from e

    def _build_semantic_prompt(self, note_meta_data: NoteMetaData) -> str:
        """Build prompt for LLM semantic analysis.

        Args:
            note_meta_data: NoteMetaData object

        Returns:
            Prompt string in Chinese for better Gemini performance
        """
        prompt = f"""请分析这段小红书笔记文本，
        深度分析小红书笔记的内容结构，包括标题、开头、正文、结尾的套路和技巧，
    挖掘出笔记内容的结构化规律和可复制的写作框架，
    理解内容如何在情感和价值层面与用户产生共鸣，驱动互动和传播

**分析内容**：
笔记标题:{note_meta_data.title}
笔记正文：{note_meta_data.content}

请分析：
    1. **标题分析**：
       - 标题套路和模板识别（如：数字型、疑问型、对比型、情感型）
       - 关键词布局和SEO优化策略
       - 标题情感倾向和心理暗示

    2. **开头策略分析**：
       - 开头钩子类型（悬念、痛点、数据、故事、问题等）
       - 开头冲击力和吸引力评估
       - 用户注意力捕获机制

    3. **正文框架分析**：
       - 内容逻辑结构（总分总、递进式、对比式等）
       - 段落组织和信息层次
       - 论证方式和说服力构建

    4. **结尾技巧分析**：
       - 结尾类型（总结型、互动型、悬念型、行动召唤型）
       - CTA设计和用户引导策略
       - 情感共鸣和记忆点营造

    5. **痛点挖掘分析**：
       - 识别内容触及的用户痛点和焦虑点
       - 评估痛点描述的具体性和共鸣度
       - 分析痛点与目标用户需求的匹配度

    6. **价值主张分析**：
       - 提取核心价值承诺和利益点
       - 分析价值层次和优先级排序
       - 评估价值主张的差异化和吸引力

    7. **情感触发分析**：
       - 识别情感触发器类型（恐惧、欲望、骄傲、归属感等）
       - 分析情感强度和感染力
       - 评估情感与理性的平衡

    8. **可信度建设分析**：
       - 识别权威性信号和专业性体现
       - 分析社会证明和成功案例使用
       - 评估可信度建立的有效性

    9. **心理驱动分析**：
       - 紧迫感和稀缺性营造
       - 损失厌恶和获得期望的利用
       - 从众心理和社会认同的激发

expected_output:

    - title_pattern: 标题套路分析
    - title_keywords: 标题关键词列表
    - title_emotion: 标题情感倾向
    - opening_strategy: 开头策略描述
    - opening_hook: 开头钩子类型
    - opening_impact: 开头冲击力评估
    - content_framework: 正文框架类型
    - content_logic: 内容逻辑层次列表
    - paragraph_structure: 段落结构特点
    - ending_technique: 结尾技巧类型
    - ending_cta: 行动召唤内容
    - ending_resonance: 结尾共鸣度评估
    - word_count: 字数统计
    - readability_score: 可读性评分
    - structure_completeness: 结构完整性评估
    - pain_points: 痛点挖掘列表
    - pain_intensity: 痛点强度评估
    - value_propositions: 价值主张列表
    - value_hierarchy: 价值层次排序
    - emotional_triggers: 情感触发器列表
    - emotional_intensity: 情感强度评估
    - credibility_signals: 可信度信号列表
    - authority_indicators: 权威性指标列表
    - urgency_indicators: 紧迫感指标列表
    - social_proof: 社会证明列表
    - scarcity_elements: 稀缺性元素列表
    - benefit_appeals: 利益点吸引列表
    - transformation_promise: 转变承诺描述

请以JSON格式返回结果，包含以上所有字段。如果某些字段不确定或无结果，使用null。

特别注意：
- 小红书用户更喜欢真诚、有帮助价值的内容
- 过强的营销感会降低用户信任
- 好的开头钩子能显著提升点击率和完读率
"""

        return prompt

    def _parse_semantic_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response into structured dict.

        Args:
            content: Response content from LLM

        Returns:
            Dict with analysis features matching TextAnalysisResult schema

        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            # Try to extract JSON from response
            # Handle both pure JSON and JSON in markdown code blocks
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            else:
                json_str = content.strip()

            data = json.loads(json_str)

            # Ensure all required fields exist with defaults
            required_fields = {
                "title_pattern": "未识别",
                "title_keywords": [],
                "title_emotion": "中性",
                "opening_strategy": "未识别",
                "opening_hook": "未识别",
                "opening_impact": "中等",
                "content_framework": "未识别",
                "content_logic": [],
                "paragraph_structure": "未识别",
                "ending_technique": "未识别",
                "ending_cta": "无",
                "ending_resonance": "中等",
                "word_count": 0,
                "readability_score": "中等",
                "structure_completeness": "中等",
                "pain_points": [],
                "pain_intensity": "中等",
                "value_propositions": [],
                "value_hierarchy": [],
                "emotional_triggers": [],
                "emotional_intensity": "中等",
                "credibility_signals": [],
                "authority_indicators": [],
                "urgency_indicators": [],
                "social_proof": [],
                "scarcity_elements": [],
                "benefit_appeals": [],
                "transformation_promise": "无",
            }

            # Merge data with defaults and handle type conversions
            result = {}
            for field, default in required_fields.items():
                value = data.get(field, default)

                # Handle None values
                if value is None:
                    result[field] = default
                    continue

                # Type conversions and cleaning
                if isinstance(default, str):
                    # Convert to string if needed
                    if isinstance(value, (int, float, bool)):
                        result[field] = str(value)
                    elif value == "null" or value == "None":
                        result[field] = default
                    else:
                        result[field] = value
                elif isinstance(default, list):
                    # Convert to list if needed
                    if isinstance(value, str):
                        if value in ["null", "None", "[]", ""]:
                            result[field] = default
                        else:
                            result[field] = [value]  # Wrap string in list
                    elif isinstance(value, list):
                        result[field] = value
                    else:
                        result[field] = default
                elif isinstance(default, int):
                    # Convert to int if needed
                    try:
                        result[field] = int(value)
                    except (ValueError, TypeError):
                        result[field] = default
                else:
                    result[field] = value

            # Add detailed analysis
            result["detailed_analysis"] = content

            return result

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {str(e)}") from e
        except Exception as e:
            raise ValueError(f"Failed to parse response: {str(e)}") from e
