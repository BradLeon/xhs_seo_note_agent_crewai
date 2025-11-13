# Spec: MultiModal Vision Analysis Tool

**Capability**: multimodal-vision
**Related**: nlp-analysis

## ADDED Requirements

### Requirement: Vision tool class structure

The MultiModalVisionTool must inherit from CrewAI's BaseTool and implement the standard tool interface for seamless integration with agents.

#### Scenario: Tool initialization

```python
from tools.multimodal_vision import MultiModalVisionTool

tool = MultiModalVisionTool()
assert tool.name == "MultiModal Vision Analyzer"
assert tool.description contains "分析图片和视频的视觉特征"
assert hasattr(tool, '_run')
```

#### Scenario: Tool can be added to agent

```python
from crewai import Agent
from tools.multimodal_vision import MultiModalVisionTool

agent = Agent(
    role="Analyst",
    goal="Analyze content",
    backstory="Expert analyst",
    tools=[MultiModalVisionTool()]
)

assert len(agent.tools) == 1
assert isinstance(agent.tools[0], MultiModalVisionTool)
```

### Requirement: Image fetching from Xiaohongshu CDN

The tool must successfully fetch images from Xiaohongshu CDN URLs with proper headers to avoid 403 errors.

#### Scenario: Fetch image with proper headers

```python
tool = MultiModalVisionTool()
image_url = "http://sns-img-hw.xhscdn.com/notes_pre_post/..."

image_data = tool._fetch_image(image_url)

assert image_data is not None
assert len(image_data) > 0
assert isinstance(image_data, bytes)
```

#### Scenario: Handle 403 errors with retry

```python
tool = MultiModalVisionTool()
bad_url = "http://example.com/blocked.jpg"

# Should retry with exponential backoff
with pytest.raises(ImageFetchError):
    tool._fetch_image(bad_url)

# Verify retry attempts were made
assert tool.retry_count == 3
```

#### Scenario: Handle timeout errors

```python
tool = MultiModalVisionTool()
slow_url = "http://slow-server.com/image.jpg"

with pytest.raises(TimeoutError):
    tool._fetch_image(slow_url)  # Should timeout after 30s
```

### Requirement: Vision API integration

The tool must integrate with OpenRouter API using Google Gemini 2.5 Flash Lite vision model to analyze images and extract visual features at minimal cost.

#### Scenario: Analyze image with vision model

```python
tool = MultiModalVisionTool()
image_url = "http://sns-img-hw.xhscdn.com/test-image.jpg"

result = tool.run(image_url)

assert isinstance(result, VisionAnalysisResult)
assert result.style in ["lifestyle", "product_shot", "infographic", "graphic_design"]
assert len(result.color_palette) > 0
assert result.authenticity_level in ["high", "medium", "low"]
```

#### Scenario: Extract OCR text from image

```python
tool = MultiModalVisionTool()
image_with_text = "http://sns-img-hw.xhscdn.com/image-with-overlay.jpg"

result = tool.run(image_with_text)

assert result.text_overlay is not None
assert len(result.text_overlay) > 0
assert result.text_overlay_style in ["bold_title", "handwritten", "emoji_rich", "minimalist"]
```

#### Scenario: Analyze image composition

```python
tool = MultiModalVisionTool()
image_url = "http://sns-img-hw.xhscdn.com/cover-image.jpg"

result = tool.run(image_url)

assert result.composition in ["centered", "rule_of_thirds", "flat_lay", "dynamic"]
assert result.subject_focus in ["person", "product", "scene", "mixed"]
```

### Requirement: Structured output validation

All vision analysis results must conform to the VisionAnalysisResult Pydantic model for type safety.

#### Scenario: Output matches schema

```python
tool = MultiModalVisionTool()
result = tool.run("http://example.com/image.jpg")

# Pydantic validation ensures these fields exist
assert hasattr(result, 'style')
assert hasattr(result, 'color_palette')
assert hasattr(result, 'authenticity_level')
assert hasattr(result, 'xiaohongshu_style_match')

# Type checking
assert isinstance(result.color_palette, list)
assert all(isinstance(color, str) for color in result.color_palette)
```

#### Scenario: Handle malformed API responses

```python
tool = MultiModalVisionTool()

# Mock API returning invalid JSON
with patch.object(tool.client.chat.completions, 'create') as mock_api:
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content='{"invalid": "response"}'))]
    mock_api.return_value = mock_response

    # Should raise error immediately (no fallback)
    with pytest.raises(ToolExecutionError):
        tool.run("http://example.com/image.jpg")
```

### Requirement: Xiaohongshu-specific analysis

The tool must evaluate how well the image matches Xiaohongshu platform style and conventions.

#### Scenario: Evaluate platform style match

```python
tool = MultiModalVisionTool()

# Typical Xiaohongshu lifestyle image
xhs_style_image = "http://sns-img-hw.xhscdn.com/typical-xhs-style.jpg"
result1 = tool.run(xhs_style_image)
assert result1.xiaohongshu_style_match == "high"

# Hard advertising image
ad_style_image = "http://example.com/product-catalog.jpg"
result2 = tool.run(ad_style_image)
assert result2.xiaohongshu_style_match == "low"
assert result2.authenticity_level == "low"
```

#### Scenario: Detect emotion in image

```python
tool = MultiModalVisionTool()
image_url = "http://sns-img-hw.xhscdn.com/emotional-scene.jpg"

result = tool.run(image_url)

assert result.emotion in ["happy", "calm", "excited", "curious", "warm", "energetic"]
```

### Requirement: Error handling

The tool must handle errors gracefully by logging details and raising clear exceptions (no fallback - errors propagate to agents).

#### Scenario: Raise error on API failure

```python
tool = MultiModalVisionTool()

with patch.object(tool.client.chat.completions, 'create', side_effect=Exception("API Error")):
    # Should raise error immediately (no fallback)
    with pytest.raises(ToolExecutionError):
        tool.run("http://example.com/image.jpg")
```

#### Scenario: Log errors for debugging

```python
tool = MultiModalVisionTool()

with patch.object(tool.client.chat.completions, 'create', side_effect=Exception("API Error")):
    with pytest.raises(ToolExecutionError):
        tool.run("http://example.com/image.jpg")

    # Verify error was logged
    assert "Vision analysis failed" in caplog.text
    assert "API Error" in caplog.text
```

### Requirement: Performance and caching

The tool should cache results to avoid redundant API calls and reduce costs.

#### Scenario: Cache results by URL

```python
tool = MultiModalVisionTool()
image_url = "http://example.com/image.jpg"

# First call
result1 = tool.run(image_url)

# Second call should use cache
with patch.object(tool.client.chat.completions, 'create') as mock_api:
    result2 = tool.run(image_url)

    # API should not be called again (cache hit)
    mock_api.assert_not_called()
    assert result1.model_dump() == result2.model_dump()
```

#### Scenario: Process within acceptable time

```python
tool = MultiModalVisionTool()
image_url = "http://sns-img-hw.xhscdn.com/test-image.jpg"

import time
start = time.time()
result = tool.run(image_url)
duration = time.time() - start

# Should complete within 15 seconds (including API call)
assert duration < 15.0
```

### Requirement: Test coverage

The MultiModalVisionTool must have comprehensive unit and integration tests.

#### Scenario: Unit tests with mocked APIs

```python
# tests/test_tools/test_multimodal_vision.py

def test_vision_tool_basic():
    tool = MultiModalVisionTool()
    assert tool.name == "MultiModal Vision Analyzer"

def test_fetch_image_with_mock(requests_mock):
    requests_mock.get('http://example.com/image.jpg', content=b'fake_image_data')
    tool = MultiModalVisionTool()
    data = tool._fetch_image('http://example.com/image.jpg')
    assert data == b'fake_image_data'

def test_analyze_with_mock_api():
    tool = MultiModalVisionTool()
    with patch.object(tool.client.chat.completions, 'create') as mock_api:
        mock_api.return_value = create_mock_vision_response()
        result = tool.run('http://example.com/image.jpg')
        assert isinstance(result, VisionAnalysisResult)
```

#### Scenario: Integration tests with real data

```python
def test_analyze_owned_note_cover():
    """Test with actual data from docs/owned_note.json"""
    tool = MultiModalVisionTool()
    owned_note = load_json('docs/owned_note.json')

    result = tool.run(owned_note['cover_image_url'])

    assert result is not None
    assert result.style != "unknown"
    assert len(result.color_palette) > 0
```

## Dependencies

- **crewai-tools**: BaseTool class
- **openai**: OpenAI-compatible client for OpenRouter API
- **requests**: HTTP client for image fetching from Xiaohongshu CDN
- **pillow**: Image preprocessing (optional, for future enhancements)
- **pydantic**: Output validation and structured data models
- **tenacity**: Retry logic with exponential backoff
- **pytest**: Testing framework
- **pytest-mock**: Mocking for unit tests

## Environment Variables

```bash
OPENROUTER_API_KEY=sk-or-v1-xxxxx          # OpenRouter API key
OPENROUTER_VISION_MODEL=google/gemini-2.5-flash-lite  # Vision model
OPENROUTER_SITE_URL=https://your-site.com  # Optional, for rankings
OPENROUTER_SITE_NAME=XHS SEO Optimizer     # Optional, for rankings
MAX_RETRIES=3                               # Retry attempts
TIMEOUT=60                                  # Request timeout in seconds
```
