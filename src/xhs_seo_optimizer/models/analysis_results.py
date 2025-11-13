"""Analysis result models - 分析结果数据模型.

Pydantic models for content analysis outputs from tools:
- VisionAnalysisResult: Visual feature analysis from MultiModalVisionTool
- TextAnalysisResult: Text feature analysis from NLPAnalysisTool
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class VisionAnalysisResult(BaseModel):
    """视觉分析结果 (Vision analysis result).

    Output from MultiModalVisionTool analyzing note images.
    Comprehensive visual design analysis including cover and inner images.
    """

    note_id: str = Field(description="笔记ID")

    # 图片基础分析 (Image Basic Analysis)
    image_count: int = Field(description="图片数量（封面图+内页图）")
    image_quality: str = Field(description="图片质量评估")
    image_content_relation: str = Field(description="图片与标题/正文的关联度")
    image_composition: str = Field(description="图片构图和拍摄技巧分析")

    # 视觉风格分析 (Visual Style Analysis)
    image_style: str = Field(description="图片风格定位")
    color_scheme: str = Field(description="色彩方案分析")
    visual_tone: str = Field(description="视觉调性评估")

    # 排版设计分析 (Layout Design Analysis)
    layout_style: str = Field(description="排版风格描述")
    visual_hierarchy: str = Field(description="视觉层次分析")

    # 图内文字分析 (OCR Text Analysis)
    text_ocr_content: str = Field(description="OCR识别的图中文字内容")
    text_ocr_content_highlight: str = Field(description="OCR识别中突出的视觉重点文字")

    # 用户体验分析 (User Experience Analysis)
    user_experience_analysis: str = Field(description="整体用户体验分析")
    thumbnail_appeal: str = Field(description="封面图（首图）吸引力分析")
    visual_storytelling: str = Field(description="内页图视觉叙事连贯性评估")
    realistic_and_emotional_tone: str = Field(description="真实感和情绪基调评估")

    # 品牌识别分析 (Brand Recognition Analysis)
    brand_consistency: str = Field(description="品牌一致性评估（平台调性+个人风格+品牌元素）")
    personal_style: str = Field(description="个人风格特点")

    # 详细分析 (Optional)
    detailed_analysis: Optional[str] = Field(
        default=None,
        description="详细分析说明 (from LLM)"
    )


class TextAnalysisResult(BaseModel):
    """文本分析结果 (Text analysis result).

    Output from NLPAnalysisTool analyzing note text content.
    Deep structural analysis of XHS note content.
    """

    note_id: str = Field(description="笔记ID")

    # 标题分析 (Title Analysis)
    title_pattern: str = Field(description="标题套路和模板识别")
    title_keywords: List[str] = Field(description="标题关键词列表")
    title_emotion: str = Field(description="标题情感倾向")

    # 开头策略分析 (Opening Strategy Analysis)
    opening_strategy: str = Field(description="开头策略描述")
    opening_hook: str = Field(description="开头钩子类型")
    opening_impact: str = Field(description="开头冲击力评估")

    # 正文框架分析 (Content Framework Analysis)
    content_framework: str = Field(description="正文框架类型")
    content_logic: List[str] = Field(description="内容逻辑层次列表")
    paragraph_structure: str = Field(description="段落结构特点")

    # 结尾技巧分析 (Ending Technique Analysis)
    ending_technique: str = Field(description="结尾技巧类型")
    ending_cta: str = Field(description="行动召唤内容")
    ending_resonance: str = Field(description="结尾共鸣度评估")

    # 基础指标 (Basic Metrics)
    word_count: int = Field(description="字数统计")
    readability_score: str = Field(description="可读性评分")
    structure_completeness: str = Field(description="结构完整性评估")

    # 痛点挖掘分析 (Pain Point Analysis)
    pain_points: List[str] = Field(description="痛点挖掘列表")
    pain_intensity: str = Field(description="痛点强度评估")

    # 价值主张分析 (Value Proposition Analysis)
    value_propositions: List[str] = Field(description="价值主张列表")
    value_hierarchy: List[str] = Field(description="价值层次排序")

    # 情感触发分析 (Emotional Trigger Analysis)
    emotional_triggers: List[str] = Field(description="情感触发器列表")
    emotional_intensity: str = Field(description="情感强度评估")

    # 可信度建设分析 (Credibility Analysis)
    credibility_signals: List[str] = Field(description="可信度信号列表")
    authority_indicators: List[str] = Field(description="权威性指标列表")

    # 心理驱动分析 (Psychological Drivers)
    urgency_indicators: List[str] = Field(description="紧迫感指标列表")
    social_proof: List[str] = Field(description="社会证明列表")
    scarcity_elements: List[str] = Field(description="稀缺性元素列表")

    # 利益吸引分析 (Benefit Appeal Analysis)
    benefit_appeals: List[str] = Field(description="利益点吸引列表")
    transformation_promise: str = Field(description="转变承诺描述")

    # Optional detailed analysis
    detailed_analysis: Optional[str] = Field(
        default=None,
        description="详细分析说明 (from LLM)"
    )
