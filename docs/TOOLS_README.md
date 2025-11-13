# Content Analysis Tools - 内容分析工具

Standalone tools for analyzing Xiaohongshu (小红书) note content using AI models via OpenRouter.

## Overview

This package provides two main content analysis tools:

1. **MultiModalVisionTool** - Analyzes images using Google Gemini 2.5 Flash Lite
2. **NLPAnalysisTool** - Analyzes text using hybrid approach (traditional NLP + free Gemini model)

### Key Features

- 🚀 **Cost-Effective**: ~$0.20-0.30 per 100 notes (100x cheaper than OpenAI GPT-4V)
- 🎯 **Xiaohongshu-Specific**: Tailored for platform-specific features (buzzwords, style matching, etc.)
- 🔧 **Standalone Tools**: Can be used independently or integrated with CrewAI agents
- 📊 **Structured Output**: Pydantic models for type-safe results
- 🔁 **Retry Logic**: Built-in retry for handling CDN issues
- 🌐 **Bilingual**: Chinese and English support

## Installation

### 1. Install Dependencies

```bash
# Using pip
pip install -e .

# Or using uv
uv pip install -e .
```

### 2. Set Up Environment Variables

Copy the example environment file and add your API key:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenRouter API key:

```bash
# OpenRouter API Configuration
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Model Configuration (optional, these are defaults)
OPENROUTER_VISION_MODEL=google/gemini-2.5-flash-lite
OPENROUTER_TEXT_MODEL=google/gemini-2.0-flash-thinking-exp-1219:free

# Site Configuration (optional)
OPENROUTER_SITE_URL=https://your-site.com
OPENROUTER_SITE_NAME=XHS SEO Optimizer
```

**Get Your API Key**: Sign up at [OpenRouter](https://openrouter.ai/) to get your API key.

## Quick Start

### Single Note Analysis

```python
import json
from src.xhs_seo_optimizer.models import Note
from src.xhs_seo_optimizer.tools import MultiModalVisionTool, NLPAnalysisTool

# Load note data
with open("docs/owned_note.json", "r") as f:
    note_data = json.load(f)

note = Note.from_json(note_data)

# Analyze text
nlp_tool = NLPAnalysisTool()
text_result = nlp_tool._run(
    text=note.meta_data.content,
    title=note.meta_data.title,
    note_id=note.note_id
)
print(json.loads(text_result))

# Analyze image
vision_tool = MultiModalVisionTool()
vision_result = vision_tool._run(
    image_url=note.meta_data.cover_image_url,
    note_id=note.note_id
)
print(json.loads(vision_result))
```

### Run Examples

```bash
# Single note analysis
python examples/analyze_note.py

# Batch analysis
python examples/batch_analyze.py
```

## Tools Documentation

### MultiModalVisionTool

Analyzes Xiaohongshu note images to extract visual features.

#### Features Extracted

| Feature | Description | Type |
|---------|-------------|------|
| `style` | Visual style (e.g., "清新自然", "高级感") | str |
| `visual_type` | Image type (e.g., "实物拍摄", "场景展示") | str |
| `color_palette` | Main colors | List[str] |
| `text_overlay` | Has text overlay | bool |
| `text_overlay_style` | Overlay style (if present) | Optional[str] |
| `composition` | Composition technique | str |
| `subject_focus` | Main subject focus | str |
| `authenticity_level` | Authenticity rating | str (high/medium/low) |
| `emotion` | Emotional tone | str |
| `xiaohongshu_style_match` | Platform style match score | float (0-1) |

#### Usage

```python
from src.xhs_seo_optimizer.tools import MultiModalVisionTool

tool = MultiModalVisionTool()

# Analyze single image
result_json = tool._run(
    image_url="https://example.com/image.jpg",
    note_id="note-123",
    analysis_focus="composition"  # Optional focus area
)

# Parse result
import json
result = json.loads(result_json)
print(f"Style: {result['style']}")
print(f"XHS Match: {result['xiaohongshu_style_match']:.2%}")
```

#### Cost

- **Model**: Google Gemini 2.5 Flash Lite via OpenRouter
- **Price**: ~$0.002 per image (~$0.20 per 100 notes)

### NLPAnalysisTool

Analyzes Xiaohongshu note text using hybrid approach (traditional NLP + LLM semantic analysis).

#### Features Extracted

**Basic Features (Traditional NLP)**:
- `length`: Text length in characters
- `word_count`: Number of words
- `has_questions`: Contains questions
- `has_emojis`: Contains emojis
- `emoji_count`: Number of emojis
- `keywords`: Extracted keywords
- `xiaohongshu_buzzwords`: Platform buzzwords detected

**Semantic Features (LLM Analysis)**:
- `sentiment`: Sentiment (positive/neutral/negative)
- `sentiment_score`: Sentiment score (-1 to 1)
- `emotion_tags`: Emotion tags
- `hook_type`: Opening hook type
- `engagement_triggers`: Engagement triggers
- `marketing_feel`: Marketing intensity (low/medium/high)
- `authenticity_level`: Authenticity rating (high/medium/low)

#### Usage

```python
from src.xhs_seo_optimizer.tools import NLPAnalysisTool

tool = NLPAnalysisTool()

# Analyze text
result_json = tool._run(
    text="姐妹们！今天必须给你们分享这个神仙好物...",
    title="好物分享",  # Optional
    note_id="note-123"
)

# Parse result
import json
result = json.loads(result_json)
print(f"Sentiment: {result['sentiment']}")
print(f"Buzzwords: {result['xiaohongshu_buzzwords']}")
print(f"Marketing Feel: {result['marketing_feel']}")
```

#### Cost

- **Traditional NLP**: Free (spaCy, regex)
- **LLM Analysis**: Free (google/gemini-2.0-flash-thinking-exp-1219:free)
- **Total**: ~$0 per 100 notes

#### Graceful Degradation

If `OPENROUTER_API_KEY` is not set, the tool will:
- Still extract all basic features (traditional NLP)
- Return neutral defaults for semantic features
- No errors thrown - gracefully degrades

## Data Models

### Note

Complete note structure combining metadata, predictions, and tags.

```python
from src.xhs_seo_optimizer.models import Note

note = Note.from_json(note_data)

# Access metadata
print(note.meta_data.title)
print(note.meta_data.content)
print(note.meta_data.cover_image_url)

# Access predictions
print(note.prediction.ctr)
print(note.prediction.sort_score2)

# Access tags
print(note.tag.intention_lv1)
print(note.tag.taxonomy1)
```

### Analysis Results

```python
from src.xhs_seo_optimizer.models import VisionAnalysisResult, TextAnalysisResult

# Vision result
vision_result = VisionAnalysisResult(
    note_id="note-123",
    style="清新自然",
    xiaohongshu_style_match=0.85,
    # ... other fields
)

# Text result
text_result = TextAnalysisResult(
    note_id="note-123",
    sentiment="positive",
    sentiment_score=0.8,
    # ... other fields
)
```

## Integration with CrewAI

These tools are designed as standalone tools but can be easily integrated with CrewAI agents.

```python
from crewai import Agent
from src.xhs_seo_optimizer.tools import MultiModalVisionTool, NLPAnalysisTool

# Create agent with tools
analyst = Agent(
    role="Content Analyst",
    goal="Analyze Xiaohongshu note content",
    tools=[
        MultiModalVisionTool(),
        NLPAnalysisTool()
    ],
    verbose=True
)
```

## Testing

### Run Unit Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_tools/test_multimodal_vision.py
pytest tests/test_tools/test_nlp_analysis.py

# Run integration tests
pytest tests/test_integration.py

# Run with coverage
pytest --cov=src tests/
```

### Test with Real API (Optional)

Set `OPENROUTER_API_KEY` in your environment to run real API tests:

```bash
export OPENROUTER_API_KEY=your-key-here
pytest tests/test_integration.py -k "test_real_api"
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key | *Required* |
| `OPENROUTER_VISION_MODEL` | Vision model ID | `google/gemini-2.5-flash-lite` |
| `OPENROUTER_TEXT_MODEL` | Text model ID | `google/gemini-2.0-flash-thinking-exp-1219:free` |
| `OPENROUTER_SITE_URL` | Your site URL | `` |
| `OPENROUTER_SITE_NAME` | Your site name | `XHS SEO Optimizer` |

### Xiaohongshu Buzzwords

The NLP tool includes a built-in list of Xiaohongshu platform buzzwords:

```python
XHS_BUZZWORDS = [
    "集美", "集美们", "姐妹们", "宝子", "宝子们",
    "绝绝子", "yyds", "awsl", "dddd", "u1s1",
    "真香", "冲鸭", "安排", "盘它", "种草",
    "拔草", "好物", "必买", "回购", "无限回购",
    "神仙", "宝藏", "超爱", "爱了爱了", "太绝了"
]
```

You can customize this list by modifying the `NLPAnalysisTool.XHS_BUZZWORDS` attribute.

## Error Handling

Both tools follow a **fail-fast** approach:

- Invalid inputs raise `ValueError` immediately
- API failures raise `RuntimeError` immediately
- No fallback mechanisms (as per requirements)

```python
try:
    result = vision_tool._run(image_url="invalid-url")
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"API failed: {e}")
```

## Performance

### Vision Tool
- **Speed**: ~2-3 seconds per image
- **Retry**: 3 attempts with exponential backoff
- **Timeout**: 10 seconds per request

### NLP Tool
- **Basic Features**: <0.1 seconds (no API call)
- **LLM Analysis**: ~1-2 seconds (with API call)
- **Total**: ~1-2 seconds per note

### Batch Processing
- **100 notes**: ~3-5 minutes
- **Cost**: ~$0.20-0.30 for 100 notes

## Troubleshooting

### Issue: "OPENROUTER_API_KEY not found"

**Solution**: Make sure you have:
1. Created a `.env` file in the project root
2. Added `OPENROUTER_API_KEY=your-key-here`
3. The `.env` file is loaded (use `python-dotenv`)

### Issue: "Vision analysis failed: OpenRouter API call failed"

**Possible causes**:
1. Invalid API key
2. Network issues
3. Image URL not accessible
4. Rate limit exceeded

**Solution**: Check API key, network connection, and OpenRouter dashboard for errors.

### Issue: "Failed to parse JSON response"

**Possible causes**:
1. Model returned unexpected format
2. Model API changed

**Solution**: Check the `detailed_analysis` field in the result for the raw response.

## Cost Optimization

### Current Cost (with OpenRouter + Gemini)
- Vision: $0.002/image
- Text: Free
- **Total**: ~$0.20-0.30 per 100 notes

### Tips for Cost Reduction
1. Batch process notes to reduce overhead
2. Cache results to avoid re-analysis
3. Use vision analysis only on cover images (skip inner images)
4. Adjust retry logic to fail faster

## Development

### Project Structure

```
src/xhs_seo_optimizer/
├── models/
│   ├── note.py              # Note data models
│   └── analysis_results.py  # Analysis result models
└── tools/
    ├── multimodal_vision.py # Vision analysis tool
    └── nlp_analysis.py      # NLP analysis tool

tests/
├── test_tools/
│   ├── test_multimodal_vision.py
│   └── test_nlp_analysis.py
└── test_integration.py

examples/
├── analyze_note.py          # Single note example
└── batch_analyze.py         # Batch analysis example
```

### Contributing

When adding new features:
1. Update the tool classes in `src/xhs_seo_optimizer/tools/`
2. Add corresponding Pydantic models if needed
3. Write unit tests in `tests/test_tools/`
4. Update this documentation
5. Add usage examples in `examples/`

## License

See LICENSE file in project root.

## Support

For issues and questions:
- Create an issue in the GitHub repository
- Check the OpenRouter documentation: https://openrouter.ai/docs
- Review CrewAI documentation: https://docs.crewai.com/
