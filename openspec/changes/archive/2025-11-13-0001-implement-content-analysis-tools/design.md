# Design: Content Analysis Tools

## Overview

This document describes the architectural design for two foundational tools that provide content understanding capabilities to the XHS SEO Optimizer crew.

## Design Principles

1. **Tool Pattern Over Agent Pattern**: Wrap complex analysis in Tools for reusability
2. **LLM-Powered Internals**: Use vision/language models for semantic understanding
3. **Structured Outputs**: Return Pydantic models for type safety
4. **Fail-Safe**: Graceful degradation when APIs fail
5. **Bilingual Support**: Handle both Chinese and English content

## Tool Architecture

### MultiModalVisionTool

**Purpose**: Analyze visual content in Xiaohongshu notes (cover images, inner images, videos)

**Input Schema**:
```python
class VisionToolInput(BaseModel):
    image_url: str = Field(description="URL of image to analyze")
    analysis_type: Optional[str] = Field(
        default="comprehensive",
        description="Type of analysis: comprehensive, style, text_only"
    )
```

**Output Schema**:
```python
class VisionAnalysisResult(BaseModel):
    style: str  # e.g., "lifestyle", "product_shot", "infographic"
    visual_type: str  # e.g., "authentic_photo", "professional_photo", "graphic_design"
    color_palette: List[str]  # e.g., ["warm", "bright", "pastel"]
    text_overlay: Optional[str]  # OCR extracted text
    text_overlay_style: Optional[str]  # e.g., "bold_title", "handwritten", "emoji_rich"
    composition: str  # e.g., "centered", "rule_of_thirds", "flat_lay"
    subject_focus: str  # e.g., "person", "product", "scene"
    authenticity_level: str  # e.g., "high", "medium", "low" (advertising feel)
    emotion: str  # e.g., "happy", "calm", "excited"
    xiaohongshu_style_match: str  # "high", "medium", "low"
```

**Implementation Approach**:

1. **Image Fetching**:
   ```python
   def _fetch_image(self, url: str) -> bytes:
       headers = {
           "User-Agent": "Mozilla/5.0...",
           "Referer": "https://www.xiaohongshu.com/"
       }
       response = requests.get(url, headers=headers, timeout=30)
       return response.content
   ```

2. **Vision API Integration**:
   ```python
   def _analyze_with_vision_model(self, image_url: str) -> Dict:
       # Use OpenRouter with Gemini 2.5 Flash Lite
       from openai import OpenAI

       client = OpenAI(
           base_url="https://openrouter.ai/api/v1",
           api_key=os.getenv("OPENROUTER_API_KEY")
       )

       prompt = """分析这张小红书笔记图片，提取以下特征：
       1. 视觉风格 (style): lifestyle/product_shot/infographic/graphic_design
       2. 视觉类型 (visual_type): authentic_photo/professional_photo/graphic_design
       3. 色彩调性 (color_palette): 列表形式，如 ["warm", "bright", "pastel"]
       4. 图片文字 (text_overlay): OCR提取的文字内容
       5. 文字样式 (text_overlay_style): bold_title/handwritten/emoji_rich/minimalist
       6. 构图方式 (composition): centered/rule_of_thirds/flat_lay/dynamic
       7. 主体焦点 (subject_focus): person/product/scene/mixed
       8. 真实感程度 (authenticity_level): high/medium/low（越真实越好，避免硬广感）
       9. 情绪氛围 (emotion): happy/calm/excited/curious/warm/energetic
       10. 小红书风格匹配度 (xiaohongshu_style_match): high/medium/low

       请用JSON格式返回，确保所有字段都有值。"""

       response = client.chat.completions.create(
           extra_headers={
               "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", ""),
               "X-Title": os.getenv("OPENROUTER_SITE_NAME", ""),
           },
           model=os.getenv("OPENROUTER_VISION_MODEL", "google/gemini-2.5-flash-lite"),
           messages=[{
               "role": "user",
               "content": [
                   {"type": "text", "text": prompt},
                   {"type": "image_url", "image_url": {"url": image_url}}
               ]
           }],
           temperature=0.7,
           max_tokens=500
       )

       return json.loads(response.choices[0].message.content)
   ```

   **Key Advantages:**
   - Direct URL support (no base64 encoding needed)
   - ~100x cheaper than GPT-4V ($0.002 vs $0.20 per image)
   - Fast inference with Gemini Flash Lite
   - OpenRouter provides access to 200+ models for flexibility

3. **Retry and Error Handling**:
   ```python
   @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
   def _run(self, image_url: str, analysis_type: str = "comprehensive") -> VisionAnalysisResult:
       try:
           # Fetch image with retry logic
           image_data = self._fetch_image(image_url)

           # Analyze with Gemini vision model
           analysis = self._analyze_with_vision_model(image_url)

           # Validate and return structured result
           return VisionAnalysisResult(**analysis)

       except Exception as e:
           # Log error details
           logger.error(f"Vision analysis failed for {image_url}: {e}")
           # Return error immediately (no fallback)
           raise ToolExecutionError(f"Failed to analyze image: {str(e)}")
   ```

   **Error Strategy:**
   - Retry up to 3 times with exponential backoff for transient failures
   - Log all errors for debugging
   - Raise exception immediately instead of returning fallback
   - Agents can decide how to handle tool failures

**Caching Strategy**:
- Cache results by image URL hash
- Use local disk cache or Redis for persistence
- Cache TTL: 7 days (images don't change)

### NLPAnalysisTool

**Purpose**: Analyze text content in Xiaohongshu notes (titles, content)

**Input Schema**:
```python
class NLPToolInput(BaseModel):
    text: str = Field(description="Text to analyze (title or content)")
    text_type: str = Field(description="Type: 'title' or 'content'")
```

**Output Schema**:
```python
class TextAnalysisResult(BaseModel):
    # Basic features
    length: int  # Character count
    word_count: int
    sentence_count: int

    # Structural features
    has_questions: bool
    question_types: List[str]  # e.g., ["open_ended", "rhetorical"]
    has_emojis: bool
    emoji_count: int
    has_hashtags: bool
    hashtag_count: int

    # Semantic features
    sentiment: str  # "positive", "neutral", "negative"
    sentiment_score: float  # -1 to 1
    emotion_tags: List[str]  # e.g., ["excited", "curious", "caring"]

    # Content hooks
    hook_type: Optional[str]  # e.g., "question", "curiosity", "emotion", "benefit"
    call_to_action: Optional[str]  # Detected CTA text
    engagement_triggers: List[str]  # e.g., ["question_at_end", "relatable_story"]

    # Marketing analysis
    marketing_feel: str  # "soft", "moderate", "hard"
    authenticity_level: str  # "high", "medium", "low"

    # Keywords
    keywords: List[str]
    xiaohongshu_buzzwords: List[str]  # Platform-specific terms
```

**Implementation Approach**:

1. **Traditional NLP Features**:
   ```python
   def _extract_basic_features(self, text: str) -> Dict:
       # Use spaCy for Chinese text processing
       doc = self.nlp_zh(text)

       return {
           "length": len(text),
           "word_count": len([token for token in doc if not token.is_punct]),
           "sentence_count": len(list(doc.sents)),
           "has_questions": "？" in text or "?" in text,
           "has_emojis": bool(emoji.emoji_count(text)),
           "emoji_count": emoji.emoji_count(text)
       }
   ```

2. **LLM Semantic Analysis** (OpenRouter + Free Gemini):
   ```python
   def _semantic_analysis(self, text: str, text_type: str) -> Dict:
       # Use OpenRouter with free Gemini model
       from openai import OpenAI

       client = OpenAI(
           base_url="https://openrouter.ai/api/v1",
           api_key=os.getenv("OPENROUTER_API_KEY")
       )

       prompt = f"""分析以下小红书{text_type}的语义特征：

"{text}"

请提取以下特征并用JSON格式返回：

1. sentiment: 情感倾向 (positive/neutral/negative)
2. sentiment_score: 情感分数 (-1.0 到 1.0)
3. emotion_tags: 情绪标签列表，如 ["excited", "curious", "caring"]
4. hook_type: 吸引力手法 (question/curiosity/emotion/benefit/contrast)
5. engagement_triggers: 互动触发点列表，如 ["question_at_end", "relatable_story"]
6. marketing_feel: 营销感强度 (soft/moderate/hard)
7. authenticity_level: 真实感程度 (high/medium/low)

注意：
- soft营销：自然分享，真实感强
- hard营销：明显推销，广告感强
- 小红书用户更喜欢soft营销和高真实感的内容

请确保返回有效的JSON格式。"""

       response = client.chat.completions.create(
           extra_headers={
               "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", ""),
               "X-Title": os.getenv("OPENROUTER_SITE_NAME", ""),
           },
           model=os.getenv("OPENROUTER_TEXT_MODEL", "google/gemini-2.0-flash-thinking-exp-1219:free"),
           messages=[{"role": "user", "content": prompt}],
           temperature=0.3,
           max_tokens=500
       )

       return json.loads(response.choices[0].message.content)
   ```

   **Hybrid Approach Benefits:**
   - **Traditional NLP (spaCy)**: Fast, deterministic features (length, emojis, questions)
   - **LLM Analysis (Gemini)**: Deep semantic understanding (sentiment, hooks, marketing feel)
   - **Free tier model**: Nearly zero cost for text analysis
   - **Best of both worlds**: Speed + accuracy at minimal cost

3. **Xiaohongshu-Specific Pattern Detection**:
   ```python
   XIAOHONGSHU_PATTERNS = {
       "buzzwords": ["种草", "拔草", "好物", "测评", "干货", "攻略", "宝妈", "姐妹"],
       "question_endings": ["吗？", "呢？", "吧？", "？评论区见", "你觉得呢"],
       "cta_patterns": ["评论区", "点赞", "收藏", "关注", "私信"],
   }

   def _detect_platform_patterns(self, text: str) -> Dict:
       return {
           "xiaohongshu_buzzwords": [
               word for word in self.XIAOHONGSHU_PATTERNS["buzzwords"]
               if word in text
           ],
           "call_to_action": next(
               (cta for cta in self.XIAOHONGSHU_PATTERNS["cta_patterns"] if cta in text),
               None
           )
       }
   ```

## Integration with Agents

**Usage Example in Agent**:
```python
# In CompetitorAnalyst agent
from tools.multimodal_vision import MultiModalVisionTool
from tools.nlp_analysis import NLPAnalysisTool

class CompetitorAnalyst(Agent):
    def __init__(self):
        self.vision_tool = MultiModalVisionTool()
        self.nlp_tool = NLPAnalysisTool()

    def analyze_note(self, note: Note) -> Dict:
        # Analyze cover image
        vision_result = self.vision_tool.run(note.cover_image_url)

        # Analyze title
        title_analysis = self.nlp_tool.run(
            text=note.title,
            text_type="title"
        )

        # Analyze content
        content_analysis = self.nlp_tool.run(
            text=note.content,
            text_type="content"
        )

        return {
            "visual_features": vision_result,
            "title_features": title_analysis,
            "content_features": content_analysis
        }
```

## Data Flow

```
User Input (Note URLs)
    |
    v
Agent calls Tool.run()
    |
    v
Tool validates input (Pydantic)
    |
    v
Tool fetches/preprocesses data
    |
    v
Tool calls LLM API (with retry)
    |
    v
Tool validates output (Pydantic)
    |
    v
Tool returns structured result
    |
    v
Agent uses result for analysis
```

## Error Handling Strategy

1. **Input Validation**: Pydantic models catch malformed inputs
2. **Network Errors**: Retry with exponential backoff (max 3 attempts)
3. **API Errors**: Log error, return fallback result
4. **Parsing Errors**: Use default values, log warning
5. **Timeout Handling**: 30s timeout for image fetch, 60s for API calls

## Performance Considerations

**Latency**:
- Vision analysis: ~5-10s per image
- Text analysis: ~2-3s per text

**Optimization**:
- Parallel processing for multiple images
- Batch text analysis when possible
- Cache results aggressively

**Cost** (with OpenRouter + Gemini):
- Vision API: ~$0.001-0.002 per image (Gemini 2.5 Flash Lite)
- Text API: ~$0.0001 per analysis or FREE (Gemini free tier)
- **Budget for 100 notes**: ~$0.20-0.30 (100x cheaper than GPT-4V!)
  - 100 images × $0.002 = $0.20
  - 200 text analyses × ~$0 = $0.00
  - **Total**: ~$0.20-0.30

## Testing Strategy

**Unit Tests**:
```python
def test_vision_tool_with_mock():
    with patch('openai.ChatCompletion.create') as mock_api:
        mock_api.return_value = mock_vision_response()
        tool = MultiModalVisionTool()
        result = tool.run("https://example.com/image.jpg")
        assert result.style in ["lifestyle", "product_shot", "infographic"]
        assert result.authenticity_level in ["high", "medium", "low"]
```

**Integration Tests**:
```python
def test_vision_tool_with_real_data():
    tool = MultiModalVisionTool()
    # Use actual image from docs/owned_note.json
    owned_note = load_json("docs/owned_note.json")
    result = tool.run(owned_note["cover_image_url"])
    assert result is not None
    assert len(result.color_palette) > 0
```

## Security Considerations

1. **API Key Management**: Store in .env, never commit
2. **Input Sanitization**: Validate URLs before fetching
3. **Rate Limiting**: Respect API rate limits
4. **Data Privacy**: Don't log sensitive content

## Future Enhancements

1. **Video Analysis**: Extract keyframes and analyze
2. **Advanced OCR**: Use Tesseract or cloud OCR for better text extraction
3. **Fine-tuned Models**: Train custom models on Xiaohongshu data
4. **Batch Processing**: Optimize for analyzing multiple notes at once
5. **Real-time Analysis**: WebSocket integration for live feedback
