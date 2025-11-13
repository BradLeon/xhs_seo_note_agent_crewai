# Tasks: Implement Content Analysis Tools

## Overview

Ordered list of work items to implement MultiModalVisionTool and NLPAnalysisTool. Tasks are designed to deliver incremental, verifiable progress.

**Parallelization**: Tasks 6-9 (Vision Tool) and Tasks 10-13 (NLP Tool) can be worked on in parallel after completing Tasks 1-5.

---

## Phase 1: Project Setup (Tasks 1-3)

### Task 1: Create project directory structure
**Dependencies**: None
**Parallelizable**: No

**Actions**:
- Create `src/xhs_seo_optimizer/` package directory
- Create `src/xhs_seo_optimizer/tools/` directory
- Create `src/xhs_seo_optimizer/models/` directory
- Create `tests/test_tools/` directory
- Add `__init__.py` files to make packages importable

**Validation**:
```bash
ls -la src/xhs_seo_optimizer/
ls -la src/xhs_seo_optimizer/tools/
ls -la src/xhs_seo_optimizer/models/
ls -la tests/test_tools/
# All directories should exist with __init__.py files
```

**Deliverable**: Project structure following CrewAI conventions

---

### Task 2: Set up pyproject.toml with dependencies
**Dependencies**: Task 1
**Parallelizable**: No

**Actions**:
- Create `pyproject.toml` at project root
- Define project metadata (name, version, description)
- List dependencies:
  - `crewai>=0.28.0`
  - `crewai-tools>=0.2.0`
  - `openai>=1.10.0`
  - `requests>=2.31.0`
  - `pillow>=10.0.0`
  - `spacy>=3.7.0`
  - `emoji>=2.8.0`
  - `python-dotenv>=1.0.0`
  - `pydantic>=2.5.0`
  - `tenacity>=8.2.0`
  - `pytest>=7.4.0`
  - `pytest-mock>=3.12.0`
- Set Python version requirement: `>=3.10`

**Validation**:
```bash
cat pyproject.toml
# Should contain all dependencies
pip install -e .
# Should install without errors
```

**Deliverable**: Working `pyproject.toml` with all dependencies

---

### Task 3: Create .env.example and configure environment
**Dependencies**: Task 2
**Parallelizable**: No

**Actions**:
- Create `.env.example` at project root
- Add environment variables:
  ```
  # OpenRouter API Configuration
  OPENROUTER_API_KEY=sk-or-v1-your-key-here
  OPENROUTER_VISION_MODEL=google/gemini-2.5-flash-lite
  OPENROUTER_TEXT_MODEL=google/gemini-2.0-flash-thinking-exp-1219:free
  OPENROUTER_SITE_URL=https://your-site.com
  OPENROUTER_SITE_NAME=XHS SEO Optimizer

  # API Configuration
  TEMPERATURE=0.7
  MAX_RETRIES=3
  TIMEOUT=60
  ```
- Add `.env` to `.gitignore`
- Create README section on environment setup

**Validation**:
```bash
cat .env.example
# Should contain all required OpenRouter environment variables
```

**Deliverable**: Environment configuration template for OpenRouter

---

## Phase 2: Data Models (Tasks 4-5)

### Task 4: Implement Note data models
**Dependencies**: Task 2
**Parallelizable**: No

**Actions**:
- Create `src/xhs_seo_optimizer/models/note.py`
- Define Pydantic models:
  - `NoteMetaData`: title, content, cover_image_url, inner_image_urls
  - `NotePrediction`: sortScore, ctr, ces_rate, interaction_rate, etc.
  - `NoteTag`: intention_lv1, intention_lv2, taxonomy1-3, marketing_level
  - `Note`: Combines meta_data, prediction, tag
- Add docstrings (bilingual: Chinese + English)
- Add validation rules (URL format, score ranges)

**Validation**:
```python
from models.note import Note, NoteMetaData, NotePrediction, NoteTag

# Should parse example data
import json
with open('docs/owned_note.json') as f:
    data = json.load(f)

note = Note(
    note_id=data['note_id'],
    meta_data=NoteMetaData(**data),
    prediction=NotePrediction(**data['prediction']),
    tag=NoteTag(**data['tag'])
)
print(note.note_id)  # Should print note ID
```

**Deliverable**: Pydantic models for Note data structures

---

### Task 5: Implement analysis result models
**Dependencies**: Task 4
**Parallelizable**: No

**Actions**:
- Create `src/xhs_seo_optimizer/models/analysis_results.py`
- Define `VisionAnalysisResult` model:
  - style, visual_type, color_palette
  - text_overlay, text_overlay_style
  - composition, subject_focus
  - authenticity_level, emotion
  - xiaohongshu_style_match
- Define `TextAnalysisResult` model:
  - length, word_count, sentence_count
  - has_questions, question_types
  - has_emojis, emoji_count, has_hashtags
  - sentiment, sentiment_score, emotion_tags
  - hook_type, call_to_action, engagement_triggers
  - marketing_feel, authenticity_level
  - keywords, xiaohongshu_buzzwords
- Add factory methods like `create_fallback()` for error handling

**Validation**:
```python
from models.analysis_results import VisionAnalysisResult, TextAnalysisResult

# Should create with valid data
vision_result = VisionAnalysisResult(
    style="lifestyle",
    color_palette=["warm", "bright"],
    authenticity_level="high",
    # ... other fields
)
print(vision_result.style)

# Should create fallback
fallback = VisionAnalysisResult.create_fallback()
assert fallback.style == "unknown"
```

**Deliverable**: Pydantic models for analysis results

---

## Phase 3: MultiModalVision Tool (Tasks 6-9)

### Task 6: Implement MultiModalVisionTool base class
**Dependencies**: Task 5
**Parallelizable**: Can start after Task 5

**Actions**:
- Create `src/xhs_seo_optimizer/tools/multimodal_vision.py`
- Define `MultiModalVisionTool` class inheriting from `BaseTool`
- Set tool name and description (bilingual)
- Define input schema `VisionToolInput` with Pydantic
- Implement `_run()` method skeleton
- Add logging setup

**Validation**:
```python
from tools.multimodal_vision import MultiModalVisionTool

tool = MultiModalVisionTool()
assert tool.name == "MultiModal Vision Analyzer"
assert hasattr(tool, '_run')
print(tool.description)  # Should print bilingual description
```

**Deliverable**: Base MultiModalVisionTool class structure

---

### Task 7: Implement image fetching with retry logic
**Dependencies**: Task 6
**Parallelizable**: No

**Actions**:
- Add `_fetch_image()` method to MultiModalVisionTool
- Set proper headers for Xiaohongshu CDN:
  ```python
  headers = {
      "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) ...",
      "Referer": "https://www.xiaohongshu.com/"
  }
  ```
- Implement retry logic with `tenacity`:
  - Max 3 attempts
  - Exponential backoff (2s, 4s, 8s)
  - Timeout: 30 seconds
- Add error handling for 403, 404, timeout
- Log all fetch attempts

**Validation**:
```python
tool = MultiModalVisionTool()

# Test with real XHS image URL from docs/owned_note.json
import json
with open('docs/owned_note.json') as f:
    note = json.load(f)

image_data = tool._fetch_image(note['cover_image_url'])
assert image_data is not None
assert len(image_data) > 0
print(f"Fetched {len(image_data)} bytes")
```

**Deliverable**: Working image fetching with retry logic

---

### Task 8: Integrate Gemini 2.5 Flash Lite for visual analysis
**Dependencies**: Task 7
**Parallelizable**: No

**Actions**:
- Add `_analyze_with_vision_model()` method
- Initialize OpenRouter client in `__init__`:
  ```python
  from openai import OpenAI
  self.client = OpenAI(
      base_url="https://openrouter.ai/api/v1",
      api_key=os.getenv("OPENROUTER_API_KEY")
  )
  self.model = os.getenv("OPENROUTER_VISION_MODEL", "google/gemini-2.5-flash-lite")
  ```
- Use direct image URL (no base64 encoding needed with Gemini)
- Create bilingual prompt for vision analysis:
  ```python
  prompt = """分析这张小红书笔记图片，提取以下特征：

  1. 视觉风格 (style): lifestyle/product_shot/infographic/graphic_design
  2. 视觉类型 (visual_type): authentic_photo/professional_photo/graphic_design
  3. 色彩调性 (color_palette): 列出主要色调，如 ["warm", "bright"]
  4. 图片文字 (text_overlay): OCR提取的文字内容
  5. 文字样式 (text_overlay_style): bold_title/handwritten/emoji_rich/minimalist
  6. 构图方式 (composition): centered/rule_of_thirds/flat_lay/dynamic
  7. 主体焦点 (subject_focus): person/product/scene/mixed
  8. 真实感程度 (authenticity_level): high/medium/low（越真实越好，避免硬广感）
  9. 情绪氛围 (emotion): happy/calm/excited/curious/warm/energetic
  10. 小红书风格匹配度 (xiaohongshu_style_match): high/medium/low

  请用JSON格式返回，确保所有字段都有值。"""
  ```
- Call OpenRouter API with Gemini:
  ```python
  response = self.client.chat.completions.create(
      extra_headers={
          "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", ""),
          "X-Title": os.getenv("OPENROUTER_SITE_NAME", ""),
      },
      model=self.model,
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
  ```
- Parse JSON response
- Map to `VisionAnalysisResult` model
- Add error handling (raise exception immediately, no fallback)

**Validation**:
```python
tool = MultiModalVisionTool()

# Test with owned note cover image
import json
with open('docs/owned_note.json') as f:
    note = json.load(f)

result = tool.run(note['cover_image_url'])

assert isinstance(result, VisionAnalysisResult)
assert result.style in ["lifestyle", "product_shot", "infographic", "graphic_design"]
assert len(result.color_palette) > 0
assert result.authenticity_level in ["high", "medium", "low"]
print(f"Style: {result.style}, Authenticity: {result.authenticity_level}")
```

**Deliverable**: Working vision analysis with Gemini 2.5 Flash Lite via OpenRouter

---

### Task 9: Add unit tests for MultiModalVisionTool
**Dependencies**: Task 8
**Parallelizable**: No

**Actions**:
- Create `tests/test_tools/test_multimodal_vision.py`
- Write unit tests with mocked APIs:
  - `test_tool_initialization()`
  - `test_fetch_image_with_mock()`
  - `test_analyze_with_mock_api()`
  - `test_retry_on_403()`
  - `test_timeout_handling()`
  - `test_fallback_on_error()`
- Write integration test with real data:
  - `test_analyze_owned_note_cover()`
- Use `pytest-mock` for mocking
- Aim for >80% coverage

**Validation**:
```bash
pytest tests/test_tools/test_multimodal_vision.py -v
# All tests should pass

pytest tests/test_tools/test_multimodal_vision.py --cov=src/xhs_seo_optimizer/tools/multimodal_vision
# Coverage should be >80%
```

**Deliverable**: Comprehensive tests for vision tool

---

## Phase 4: NLPAnalysis Tool (Tasks 10-13)

### Task 10: Implement NLPAnalysisTool base class
**Dependencies**: Task 5
**Parallelizable**: Can start after Task 5 (parallel with Task 6)

**Actions**:
- Create `src/xhs_seo_optimizer/tools/nlp_analysis.py`
- Define `NLPAnalysisTool` class inheriting from `BaseTool`
- Set tool name and description (bilingual)
- Define input schema `NLPToolInput` with Pydantic
- Load spaCy models in `__init__`:
  ```python
  import spacy
  self.nlp_zh = spacy.load("zh_core_web_sm")
  ```
- Implement `_run()` method skeleton
- Add logging setup

**Validation**:
```python
from tools.nlp_analysis import NLPAnalysisTool

tool = NLPAnalysisTool()
assert tool.name == "NLP Text Analyzer"
assert tool.nlp_zh is not None
assert hasattr(tool, '_run')
print(tool.description)
```

**Deliverable**: Base NLPAnalysisTool class structure

---

### Task 11: Implement basic text feature extraction
**Dependencies**: Task 10
**Parallelizable**: No

**Actions**:
- Add `_extract_basic_features()` method
- Count length, words, sentences using spaCy
- Detect questions (check for "？" or "?")
- Detect emojis using `emoji` library
- Detect hashtags (check for "#...#" or "[话题]")
- Extract keyword patterns

**Validation**:
```python
tool = NLPAnalysisTool()

# Test with owned note title
import json
with open('docs/owned_note.json') as f:
    note = json.load(f)

result = tool.run(text=note['title'], text_type="title")

assert result.length == len(note['title'])
assert result.word_count > 0
assert isinstance(result.has_emojis, bool)
print(f"Length: {result.length}, Words: {result.word_count}, Emojis: {result.emoji_count}")
```

**Deliverable**: Basic text feature extraction working

---

### Task 12: Integrate LLM for semantic analysis
**Dependencies**: Task 11
**Parallelizable**: No

**Actions**:
- Add `_semantic_analysis()` method
- Create bilingual prompt for semantic analysis:
  ```python
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
  ```
- Initialize OpenRouter client in `__init__`:
  ```python
  from openai import OpenAI
  self.client = OpenAI(
      base_url="https://openrouter.ai/api/v1",
      api_key=os.getenv("OPENROUTER_API_KEY")
  )
  self.text_model = os.getenv("OPENROUTER_TEXT_MODEL",
                              "google/gemini-2.0-flash-thinking-exp-1219:free")
  ```
- Call OpenRouter API with free Gemini model:
  ```python
  response = self.client.chat.completions.create(
      extra_headers={
          "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", ""),
          "X-Title": os.getenv("OPENROUTER_SITE_NAME", ""),
      },
      model=self.text_model,
      messages=[{"role": "user", "content": prompt}],
      temperature=0.3,
      max_tokens=500
  )
  ```
- Parse JSON response
- Add Xiaohongshu-specific pattern detection (traditional NLP):
  - Buzzwords: "种草", "拔草", "好物", "测评", etc.
  - CTA patterns: "评论区", "点赞", "收藏"
- Combine basic features (from Task 11) + semantic features (LLM) into `TextAnalysisResult`
- Add graceful error handling (use safe defaults for LLM features if API fails)

**Validation**:
```python
tool = NLPAnalysisTool()

# Test with owned note content
import json
with open('docs/owned_note.json') as f:
    note = json.load(f)

result = tool.run(text=note['content'], text_type="content")

assert result.sentiment in ["positive", "neutral", "negative"]
assert -1.0 <= result.sentiment_score <= 1.0
assert result.marketing_feel in ["soft", "moderate", "hard"]
assert len(result.emotion_tags) > 0
print(f"Sentiment: {result.sentiment}, Marketing: {result.marketing_feel}")
print(f"XHS Buzzwords: {result.xiaohongshu_buzzwords}")
```

**Deliverable**: Full hybrid NLP analysis (Traditional NLP + Free Gemini LLM via OpenRouter)

---

### Task 13: Add unit tests for NLPAnalysisTool
**Dependencies**: Task 12
**Parallelizable**: No

**Actions**:
- Create `tests/test_tools/test_nlp_analysis.py`
- Write unit tests with mocked APIs:
  - `test_tool_initialization()`
  - `test_basic_features()`
  - `test_question_detection()`
  - `test_emoji_detection()`
  - `test_sentiment_analysis_with_mock()`
  - `test_buzzword_detection()`
  - `test_fallback_on_error()`
- Write integration tests with real data:
  - `test_analyze_owned_note_title()`
  - `test_analyze_owned_note_content()`
  - `test_analyze_target_notes()`
- Aim for >80% coverage

**Validation**:
```bash
pytest tests/test_tools/test_nlp_analysis.py -v
# All tests should pass

pytest tests/test_tools/test_nlp_analysis.py --cov=src/xhs_seo_optimizer/tools/nlp_analysis
# Coverage should be >80%
```

**Deliverable**: Comprehensive tests for NLP tool

---

## Phase 5: Integration & Documentation (Tasks 14-15)

### Task 14: End-to-end integration testing
**Dependencies**: Tasks 9, 13
**Parallelizable**: No

**Actions**:
- Create `tests/test_integration.py`
- Write integration tests:
  - `test_analyze_all_owned_note_features()`: Analyze complete owned note
  - `test_analyze_multiple_target_notes()`: Process all target notes
  - `test_tools_work_together()`: Both tools on same note
- Test with all example data from `docs/`
- Verify output formats match expected schemas
- Test error scenarios (bad URLs, empty text)

**Validation**:
```bash
pytest tests/test_integration.py -v
# All integration tests should pass

# Should successfully analyze:
# - docs/owned_note.json (1 note)
# - docs/target_notes.json (multiple notes)
```

**Deliverable**: Working end-to-end integration

---

### Task 15: Add usage examples and documentation
**Dependencies**: Task 14
**Parallelizable**: No

**Actions**:
- Add comprehensive docstrings to both tools
- Create usage examples in docstrings:
  ```python
  """
  Examples:
      >>> tool = MultiModalVisionTool()
      >>> result = tool.run("https://example.com/image.jpg")
      >>> print(result.style)  # "lifestyle"
  """
  ```
- Update README.md with:
  - Tool descriptions
  - Installation instructions
  - Usage examples
  - API reference
- Add inline comments explaining key logic
- Create example scripts:
  - `examples/analyze_note.py`: Analyze a single note
  - `examples/batch_analyze.py`: Analyze multiple notes

**Validation**:
```bash
# Check docstrings
python -c "from tools.multimodal_vision import MultiModalVisionTool; help(MultiModalVisionTool)"

# Run example scripts
python examples/analyze_note.py docs/owned_note.json
# Should print analysis results
```

**Deliverable**: Complete documentation and examples

---

## Success Criteria

All tasks completed when:
- ✅ Project structure follows CrewAI conventions
- ✅ Dependencies installed and working
- ✅ Both tools implement BaseTool correctly
- ✅ Vision tool analyzes images successfully
- ✅ NLP tool analyzes text successfully
- ✅ All unit tests pass with >80% coverage
- ✅ Integration tests pass with real data
- ✅ Documentation complete with examples
- ✅ Code follows project conventions (bilingual comments, type hints, etc.)

## Estimated Timeline

- **Phase 1** (Setup): 2 hours
- **Phase 2** (Models): 2 hours
- **Phase 3** (Vision Tool): 6 hours
- **Phase 4** (NLP Tool): 6 hours
- **Phase 5** (Integration & Docs): 3 hours

**Total**: ~19 hours (2-3 days with focused work)

**Parallelization**: Phases 3 and 4 can overlap, reducing total time to ~15 hours.
