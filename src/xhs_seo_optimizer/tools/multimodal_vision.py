"""MultiModal Vision Tool - 多模态视觉分析工具.

Analyzes Xiaohongshu note images using Google Gemini 2.5 Flash Lite via OpenRouter.
Supports multi-image analysis (cover image + inner images).
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional
from openai import OpenAI
from crewai.tools import BaseTool
from pydantic import Field, ConfigDict, BaseModel

from ..models.analysis_results import VisionAnalysisResult
from ..models.note import NoteMetaData

logger = logging.getLogger(__name__)


class MultiModalVisionInput(BaseModel):
    """Input schema for MultiModalVisionTool.

    Supports two calling modes:
    1. Smart mode (recommended): Pass note_id, tool auto-fetches metadata from shared_context
    2. Legacy mode: Pass note_metadata directly (backward compatible)

    Agent usage examples:
        智能模式: multimodal_vision_analysis(note_id="5e96b4f700000000010040e6")
        传统模式: multimodal_vision_analysis(note_metadata={...}, note_id="xxx")
    """
    note_id: Optional[str] = Field(
        default=None,
        description=(
            "笔记ID - 工具会自动从系统中获取该笔记的 metadata（包括封面图和内页图）并分析。"
            "推荐使用此模式，只需传入 note_id。"
        )
    )
    note_metadata: Optional[dict] = Field(
        default=None,
        description=(
            "可选：直接传入序列化的 NoteMetaData dict。"
            "如果不传，工具会根据 note_id 自动从系统获取。"
            "Required keys: cover_image_url. Optional: inner_image_urls."
        )
    )


class MultiModalVisionTool(BaseTool):
    """多模态视觉分析工具 (Multimodal Vision Analysis Tool).

    Uses Google Gemini 2.5 Flash Lite via OpenRouter to analyze note images.
    Supports multi-image analysis: 1 cover image + up to 4 inner images.
    Returns structured visual features as VisionAnalysisResult.

    Cost: ~$0.002 per image, ~$0.01 per note (5 images).

    Following official CrewAI pattern: receives serialized dict data,
    reconstructs NoteMetaData internally for processing.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "multimodal_vision_analysis"
    description: str = (
        "Analyzes Xiaohongshu note images (cover + inner images) to extract visual features. "
        "Smart mode: Just pass note_id, tool auto-fetches images - multimodal_vision_analysis(note_id='xxx'). "
        "Legacy mode: Pass note_metadata directly. "
        "Output: VisionAnalysisResult JSON with 17+ visual features. "
        "使用场景：分析笔记封面图和内页图的视觉设计元素，包括风格、色彩、排版、OCR等。"
    )
    args_schema: type[BaseModel] = MultiModalVisionInput

    # OpenRouter configuration
    api_key: str = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    model: str = Field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_VISION_MODEL",
            "google/gemini-2.5-flash-lite"
        )
    )
    site_url: str = Field(
        default_factory=lambda: os.getenv("OPENROUTER_SITE_URL", "XHS SEO Optimizer")
    )
    site_name: str = Field(
        default_factory=lambda: os.getenv("OPENROUTER_SITE_NAME", "XHS SEO Optimizer")
    )

    # Image limits
    max_inner_images: int = Field(default=4, description="最多分析的内页图数量")

    def _run(
        self,
        note_id: Optional[str] = None,
        note_metadata: Optional[dict] = None
    ) -> str:
        """Analyze note images (cover + inner images) and return vision analysis result.

        Supports two modes:
        1. Smart mode: Pass note_id, auto-fetch metadata from shared_context
        2. Legacy mode: Pass note_metadata directly

        Args:
            note_id: Note ID to analyze (tool will fetch metadata automatically)
            note_metadata: Optional - directly provide serialized NoteMetaData dict

        Returns:
            JSON string of VisionAnalysisResult

        Raises:
            ValueError: If API key is missing or both note_id and note_metadata are missing
            RuntimeError: If API call fails
        """
        # Validate API key
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found. "
                "Please set it in .env file or environment variables."
            )

        # Smart mode: Fetch from shared_context if only note_id provided
        if not note_metadata and note_id:
            logger.info(f"Smart mode: Fetching metadata for note_id={note_id} from shared_context")

            from xhs_seo_optimizer.shared_context import shared_context
            notes = shared_context.get("target_notes_data", [])

            # Find the note by note_id
            for note in notes:
                if note.get('note_id') == note_id:
                    note_metadata = note.get('meta_data')
                    logger.info(f"✓ Found note metadata for {note_id}")
                    break

            if not note_metadata:
                raise ValueError(
                    f"Note with ID '{note_id}' not found in shared_context. "
                    f"Available notes: {[n.get('note_id') for n in notes]}"
                )

        # Validate we have metadata now
        if not note_metadata or not note_metadata.get('cover_image_url'):
            raise ValueError(
                "Either note_id (for smart mode) or note_metadata (for legacy mode) must be provided. "
                "note_metadata must contain 'cover_image_url' field."
            )

        try:
            # Reconstruct NoteMetaData from serialized dict
            actual_meta_data = NoteMetaData(**note_metadata)

            # Collect image URLs (1 cover + max 4 inner images)
            image_urls = [actual_meta_data.cover_image_url]

            if actual_meta_data.inner_image_urls:
                # Limit to max_inner_images
                inner_urls = actual_meta_data.inner_image_urls[:self.max_inner_images]
                image_urls.extend(inner_urls)

            # Analyze images with vision model
            analysis_dict = self._analyze_with_vision_model(
                note_meta_data=actual_meta_data,
                image_urls=image_urls
            )

            # Create VisionAnalysisResult
            result = VisionAnalysisResult(
                note_id=note_id or note_metadata.get('note_id', 'unknown'),
                **analysis_dict
            )

            # Return as JSON string for CrewAI tool compatibility
            return result.model_dump_json(indent=2)

        except Exception as e:
            # Return error immediately (no fallback per requirements)
            raise RuntimeError(f"Vision analysis failed: {str(e)}") from e

    def _analyze_with_vision_model(
        self,
        note_meta_data: NoteMetaData,
        image_urls: List[str]
    ) -> Dict[str, Any]:
        """Analyze multiple images using Gemini 2.5 Flash Lite via OpenRouter.

        Args:
            note_meta_data: NoteMetaData object
            image_urls: List of image URLs (cover + inner images, max 5)

        Returns:
            Dict with visual features matching VisionAnalysisResult schema

        Raises:
            RuntimeError: If API call fails
        """
        # Initialize OpenRouter client
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )

        # Construct analysis prompt
        prompt = self._build_vision_prompt(note_meta_data)

        try:
            # Build multi-image message content
            message_content = [
                {"type": "text", "text": prompt}
            ]

            # Add all images (cover image first, then inner images)
            for idx, image_url in enumerate(image_urls):
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })

            # Call OpenRouter API with Gemini 2.5 Flash Lite (multi-image support)
            response = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                },
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": message_content
                    }
                ],
                temperature=0.3,  # Lower temperature for more consistent analysis
            )

            # Extract and parse response
            content = response.choices[0].message.content

            return self._parse_vision_response(content)

        except Exception as e:
            raise RuntimeError(f"OpenRouter API call failed: {str(e)}") from e

    def _build_vision_prompt(self, note_meta_data: NoteMetaData) -> str:
        """Build prompt for vision model analysis.

        Args:
            note_meta_data: NoteMetaData object

        Returns:
            Prompt string in Chinese for better Gemini performance
        """
        base_prompt = f"""请分析这张小红书笔记图片，提取以下视觉特征：
        全面分析小红书笔记的视觉设计元素，包括图片风格、色彩搭配、排版设计等，
        结合笔记的标题和文案，理解作者的表达意图和目标受众，总结视觉呈现对内容传播效果和用户体验的影响规律。
        总结成内容创作者可以落地操作的视觉设计语言 。


        具体可以分析以下维度：

    1、**图片基础分析**：
       - 图片数量、质量和清晰度评估
       - 图片内容和主题的关联度
       - 图片构图和拍摄技巧分析

    2. **视觉风格分析**：
       - 图片风格定位
       - 色彩方案和色彩心理学应用
       - 整体视觉调性和品牌一致性

    3. **排版设计分析**：
       - 布局设计和视觉层次构建
       - 图片与内容文字的搭配关系
       - 构图方式和主体焦点
       - 留白和平衡感的运用

    4. **图内文字分析**：
       - 通过OCR能力识别图中文字内容
       - 特别是文字大小和颜色的差异突出的视觉重点文字

    5. **用户体验分析**：
       - 封面图对于用户停留点击的吸引力
       - 内页图与文案内容叙事的一致性、连贯性
       - 视觉效果呈现的真实感、情绪基调

    6. **品牌识别分析**：
       - 是否符合平台调性以及呈现调性是什么
       - 是否符合个人风格以及呈现风格是什么
       - 是否透露品牌元素以及呈现品牌元素是什么

    输入内容:
    笔记标题:{note_meta_data.title}
    笔记正文：{note_meta_data.content}
    封面图：{note_meta_data.cover_image_url}
    内页图：{note_meta_data.inner_image_urls if note_meta_data.inner_image_urls else "无"}

    expected_output:
        - image_count: 图片数量
        - image_quality: 图片质量评估
        - image_content_relation: 图片内容和标题/正文的关联度
        - image_composition: 图片构图和拍摄技巧分析
        - image_style: 图片风格定位
        - color_scheme: 色彩方案分析
        - visual_tone: 视觉调性评估
        - layout_style: 排版风格描述
        - visual_hierarchy: 视觉层次分析
        - text_ocr_content: 文字OCR识别内容
        - text_ocr_content_highlight: 文字OCR识别内容中突出的视觉重点文字
        - user_experience_analysis: 用户体验分析
        - thumbnail_appeal: 首图吸引力分析
        - visual_storytelling: 视觉叙事评估
        - realistic_and_emotional_tone: 用户体验的真实感和情绪基调评估
        - brand_consistency: 品牌一致性评估
        - personal_style: 个人风格特点


    请以JSON格式返回结果，包含以上所有字段。如果某些字段不适用，请使用null。
"""

        return base_prompt

    def _parse_vision_response(self, content: str) -> Dict[str, Any]:
        """Parse vision model response into structured dict.

        Args:
            content: Response content from vision model

        Returns:
            Dict matching VisionAnalysisResult schema

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

            # Define all required fields with defaults
            required_fields = {
                "image_count": 0,
                "image_quality": "未评估",
                "image_content_relation": "未评估",
                "image_composition": "未分析",
                "image_style": "未识别",
                "color_scheme": "未分析",
                "visual_tone": "未评估",
                "layout_style": "未描述",
                "visual_hierarchy": "未分析",
                "text_ocr_content": "无",
                "text_ocr_content_highlight": "无",
                "user_experience_analysis": "未分析",
                "thumbnail_appeal": "未评估",
                "visual_storytelling": "未评估",
                "realistic_and_emotional_tone": "未评估",
                "brand_consistency": "未评估",
                "personal_style": "未识别",
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
                    if isinstance(value, list):
                        # Handle lists: join elements with newline or comma
                        if field in ["text_ocr_content", "text_ocr_content_highlight"]:
                            # For OCR content, join with newline for readability
                            result[field] = "\n".join(str(v) for v in value if v)
                        else:
                            # For other fields, join with comma
                            result[field] = ", ".join(str(v) for v in value if v)
                    elif isinstance(value, (int, float, bool)):
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
            logger.error("=" * 80)
            logger.error("JSON PARSING ERROR")
            logger.error("=" * 80)
            logger.error(f"Error: {str(e)}")
            logger.error(f"Error position: line {e.lineno} column {e.colno} (char {e.pos})")
            logger.error(f"Error message: {e.msg}")
            logger.error("=" * 80)
            raise ValueError(f"Failed to parse JSON response: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error during parsing: {str(e)}")
            raise ValueError(f"Failed to parse vision response: {str(e)}") from e
