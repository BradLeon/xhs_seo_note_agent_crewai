# Spec: NLP Analysis Tool

**Capability**: nlp-analysis
**Related**: multimodal-vision

## ADDED Requirements

### Requirement: NLP tool class structure

The NLPAnalysisTool must inherit from CrewAI's BaseTool and handle both Chinese and English text analysis using a **hybrid approach**:
- **Traditional NLP** (spaCy, regex): Fast, deterministic features (length, emojis, questions)
- **LLM-powered analysis** (Gemini via OpenRouter): Semantic understanding (sentiment, hooks, marketing feel)

#### Scenario: Tool initialization

```python
from tools.nlp_analysis import NLPAnalysisTool

tool = NLPAnalysisTool()
assert tool.name == "NLP Text Analyzer"
assert "分析文本内容" in tool.description
assert hasattr(tool, '_run')
assert tool.nlp_zh is not None  # Chinese model loaded
```

#### Scenario: Tool can be added to agent

```python
from crewai import Agent
from tools.nlp_analysis import NLPAnalysisTool

agent = Agent(
    role="Content Analyst",
    goal="Analyze text patterns",
    backstory="Expert in content analysis",
    tools=[NLPAnalysisTool()]
)

assert len(agent.tools) == 1
assert isinstance(agent.tools[0], NLPAnalysisTool)
```

### Requirement: Basic text feature extraction

The tool must extract basic structural features from text (length, word count, sentence count).

#### Scenario: Extract basic features from Chinese text

```python
tool = NLPAnalysisTool()
text = "跟着老爸选DHA｜双A同补真的赢麻了。养娃以后才知道，宝宝成长的每一步都需要精心呵护。"

result = tool.run(text=text, text_type="title")

assert result.length == len(text)
assert result.word_count > 0
assert result.sentence_count == 2
```

#### Scenario: Extract basic features from English text

```python
tool = NLPAnalysisTool()
text = "Best DHA supplements for babies! Check out these amazing options."

result = tool.run(text=text, text_type="title")

assert result.length == len(text)
assert result.word_count > 0
assert result.sentence_count >= 1
```

### Requirement: Question detection

The tool must detect questions in text and classify question types.

#### Scenario: Detect Chinese questions

```python
tool = NLPAnalysisTool()
text = "你家宝宝吃DHA了吗？选哪个牌子比较好呢？"

result = tool.run(text=text, text_type="content")

assert result.has_questions == True
assert "open_ended" in result.question_types
assert result.engagement_triggers contains "question"
```

#### Scenario: Detect rhetorical questions

```python
tool = NLPAnalysisTool()
text = "难道不是每个宝宝都需要补DHA吗？这还用说吗？"

result = tool.run(text=text, text_type="content")

assert result.has_questions == True
assert "rhetorical" in result.question_types
```

#### Scenario: Detect no questions

```python
tool = NLPAnalysisTool()
text = "这款DHA真的很好用。推荐给大家。"

result = tool.run(text=text, text_type="content")

assert result.has_questions == False
assert len(result.question_types) == 0
```

### Requirement: Emoji and hashtag detection

The tool must detect and count emojis and hashtags in text.

#### Scenario: Detect emojis

```python
tool = NLPAnalysisTool()
text = "养娃神器 🧠发育的关键期！让宝宝悄悄赢在起跑线上🏃"

result = tool.run(text=text, text_type="content")

assert result.has_emojis == True
assert result.emoji_count == 2
```

#### Scenario: Detect hashtags

```python
tool = NLPAnalysisTool()
text = "#宝宝营养品[话题]# #儿童DHA[话题]# #育儿好物[话题]#"

result = tool.run(text=text, text_type="content")

assert result.has_hashtags == True
assert result.hashtag_count == 3
```

### Requirement: Sentiment analysis

The tool must analyze sentiment and emotion in text using LLM-powered semantic analysis (via OpenRouter + free Gemini model).

#### Scenario: Detect positive sentiment

```python
tool = NLPAnalysisTool()
text = "真的太好用了！宝宝爱吃，我也很放心，强烈推荐！"

result = tool.run(text=text, text_type="content")

assert result.sentiment == "positive"
assert result.sentiment_score > 0.5
assert "excited" in result.emotion_tags or "happy" in result.emotion_tags
```

#### Scenario: Detect neutral sentiment

```python
tool = NLPAnalysisTool()
text = "DHA是一种营养补充剂，适合婴幼儿食用。"

result = tool.run(text=text, text_type="content")

assert result.sentiment == "neutral"
assert -0.2 < result.sentiment_score < 0.2
```

#### Scenario: Detect negative sentiment

```python
tool = NLPAnalysisTool()
text = "买了很失望，宝宝根本不吃，浪费钱。"

result = tool.run(text=text, text_type="content")

assert result.sentiment == "negative"
assert result.sentiment_score < -0.3
```

### Requirement: Content hook detection

The tool must identify content hooks and engagement triggers used in titles and content.

#### Scenario: Detect question hook

```python
tool = NLPAnalysisTool()
text = "为什么我没有早点发现这个DHA？"

result = tool.run(text=text, text_type="title")

assert result.hook_type == "question"
assert "curiosity" in result.engagement_triggers
```

#### Scenario: Detect curiosity hook

```python
tool = NLPAnalysisTool()
text = "别再瞎买了！这3款DHA才是真的好"

result = tool.run(text=text, text_type="title")

assert result.hook_type in ["curiosity", "contrast"]
assert len(result.engagement_triggers) > 0
```

#### Scenario: Detect emotion hook

```python
tool = NLPAnalysisTool()
text = "当妈妈后才懂的心酸｜宝宝营养真的不能马虎"

result = tool.run(text=text, text_type="title")

assert result.hook_type == "emotion"
assert "relatable_story" in result.engagement_triggers
```

#### Scenario: Detect benefit hook

```python
tool = NLPAnalysisTool()
text = "照着这个方法选DHA，宝宝大脑发育快人一步"

result = tool.run(text=text, text_type="title")

assert result.hook_type == "benefit"
```

### Requirement: Call-to-action detection

The tool must detect CTAs that encourage engagement (comments, likes, follows).

#### Scenario: Detect comment CTA

```python
tool = NLPAnalysisTool()
text = "你家宝宝吃什么牌子的DHA？评论区聊聊吧！"

result = tool.run(text=text, text_type="content")

assert result.call_to_action is not None
assert "评论区" in result.call_to_action
assert "question_at_end" in result.engagement_triggers
```

#### Scenario: Detect like/collect CTA

```python
tool = NLPAnalysisTool()
text = "喜欢的姐妹记得点赞收藏哦～"

result = tool.run(text=text, text_type="content")

assert result.call_to_action is not None
assert "点赞" in result.call_to_action or "收藏" in result.call_to_action
```

### Requirement: Marketing feel analysis

The tool must evaluate how "advertising-like" the text feels (soft/moderate/hard).

#### Scenario: Detect soft marketing

```python
tool = NLPAnalysisTool()
text = "养娃以后才知道，宝宝成长的每一步都需要精心呵护。我选的是诺特兰德DHA～"

result = tool.run(text=text, text_type="content")

assert result.marketing_feel == "soft"
assert result.authenticity_level == "high"
```

#### Scenario: Detect hard marketing

```python
tool = NLPAnalysisTool()
text = "诺特兰德DHA藻油，买二送一，限时特惠，立即下单！"

result = tool.run(text=text, text_type="content")

assert result.marketing_feel == "hard"
assert result.authenticity_level == "low"
```

### Requirement: Xiaohongshu-specific pattern detection

The tool must recognize platform-specific buzzwords and patterns common on Xiaohongshu.

#### Scenario: Detect Xiaohongshu buzzwords

```python
tool = NLPAnalysisTool()
text = "姐妹们！这个好物必须种草！真的是宝妈必备的干货攻略！"

result = tool.run(text=text, text_type="content")

assert len(result.xiaohongshu_buzzwords) > 0
assert any(word in result.xiaohongshu_buzzwords for word in ["种草", "好物", "宝妈", "干货", "攻略"])
```

#### Scenario: Detect platform-specific patterns

```python
tool = NLPAnalysisTool()
text = "跟着老爸选DHA｜双A同补真的赢麻了"

result = tool.run(text=text, text_type="title")

# Detect "｜" separator pattern common in XHS titles
assert result.xiaohongshu_buzzwords or result.authenticity_level == "high"
```

### Requirement: Keyword extraction

The tool must extract important keywords from text, especially product names and topics.

#### Scenario: Extract keywords from content

```python
tool = NLPAnalysisTool()
text = "诺特兰德DHA藻油ARA真的让我安心。双A同补，黄金1:1配比，专利裂壶藻。"

result = tool.run(text=text, text_type="content")

assert len(result.keywords) > 0
assert "DHA" in result.keywords
assert "藻油" in result.keywords or "诺特兰德" in result.keywords
```

### Requirement: Structured output validation

All NLP analysis results must conform to the TextAnalysisResult Pydantic model.

#### Scenario: Output matches schema

```python
tool = NLPAnalysisTool()
result = tool.run(text="测试文本", text_type="title")

# Pydantic ensures all required fields exist
assert hasattr(result, 'length')
assert hasattr(result, 'sentiment')
assert hasattr(result, 'marketing_feel')
assert hasattr(result, 'engagement_triggers')

# Type validation
assert isinstance(result.length, int)
assert isinstance(result.emotion_tags, list)
assert isinstance(result.has_questions, bool)
```

#### Scenario: Handle malformed LLM responses

```python
tool = NLPAnalysisTool()

with patch.object(tool.client.chat.completions, 'create') as mock_api:
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content='{"invalid": "response"}'))]
    mock_api.return_value = mock_response

    result = tool.run(text="测试", text_type="title")

    # Should return fallback with safe defaults for LLM features
    # Basic features (traditional NLP) should still work
    assert result.length == 2
    assert result.sentiment == "neutral"  # Safe default
    assert result.marketing_feel == "moderate"  # Safe default
```

### Requirement: Performance optimization

The tool should process text efficiently and cache results when appropriate.

#### Scenario: Fast processing for short texts

```python
tool = NLPAnalysisTool()
title = "跟着老爸选DHA｜双A同补真的赢麻了"

import time
start = time.time()
result = tool.run(text=title, text_type="title")
duration = time.time() - start

# Should complete within 5 seconds
assert duration < 5.0
```

#### Scenario: Handle long content efficiently

```python
tool = NLPAnalysisTool()
long_content = "很长的内容..." * 1000  # ~1000 characters

result = tool.run(text=long_content, text_type="content")

# Should not crash and complete reasonably fast
assert result is not None
assert result.length == len(long_content)
```

### Requirement: Error handling

The tool must handle errors gracefully and provide fallback results.

#### Scenario: Handle API failures gracefully

```python
tool = NLPAnalysisTool()

with patch.object(tool.client.chat.completions, 'create', side_effect=Exception("API Error")):
    result = tool.run(text="测试文本", text_type="title")

    # Should return valid result with traditional NLP features
    # LLM features use safe defaults
    assert isinstance(result, TextAnalysisResult)
    assert result.length == 4  # Traditional NLP still works
    assert result.sentiment == "neutral"  # Safe LLM default
    assert result.marketing_feel == "moderate"  # Safe LLM default
```

#### Scenario: Handle empty text

```python
tool = NLPAnalysisTool()
result = tool.run(text="", text_type="title")

assert result.length == 0
assert result.word_count == 0
assert result.has_questions == False
```

### Requirement: Test coverage

The NLPAnalysisTool must have comprehensive unit and integration tests.

#### Scenario: Unit tests with mocked LLM

```python
# tests/test_tools/test_nlp_analysis.py

def test_nlp_tool_basic():
    tool = NLPAnalysisTool()
    assert tool.name == "NLP Text Analyzer"

def test_basic_features():
    tool = NLPAnalysisTool()
    result = tool.run(text="测试文本。", text_type="title")
    assert result.length == 5
    assert result.sentence_count == 1

def test_question_detection():
    tool = NLPAnalysisTool()
    result = tool.run(text="你觉得呢？", text_type="content")
    assert result.has_questions == True
```

#### Scenario: Integration tests with real data

```python
def test_analyze_owned_note_title():
    """Test with actual data from docs/owned_note.json"""
    tool = NLPAnalysisTool()
    owned_note = load_json('docs/owned_note.json')

    result = tool.run(
        text=owned_note['title'],
        text_type="title"
    )

    assert result is not None
    assert result.length > 0
    assert result.sentiment in ["positive", "neutral", "negative"]
```

## Dependencies

- **crewai-tools**: BaseTool class
- **openai**: OpenAI-compatible client for OpenRouter API
- **spacy**: Chinese/English text processing (`zh_core_web_sm`, `en_core_web_sm`)
- **emoji**: Emoji detection and counting
- **pydantic**: Output validation and structured data models
- **tenacity**: Retry logic for API calls (optional)
- **pytest**: Testing framework
- **pytest-mock**: Mocking for unit tests

## Environment Variables

```bash
OPENROUTER_API_KEY=sk-or-v1-xxxxx          # OpenRouter API key
OPENROUTER_TEXT_MODEL=google/gemini-2.0-flash-thinking-exp-1219:free  # Free text model
OPENROUTER_SITE_URL=https://your-site.com  # Optional, for rankings
OPENROUTER_SITE_NAME=XHS SEO Optimizer     # Optional, for rankings
```

## Hybrid Approach Summary

**Traditional NLP (Fast & Deterministic)**:
- Word/character count, sentence count
- Emoji detection and counting
- Hashtag detection and counting
- Question mark detection
- Platform-specific buzzword matching

**LLM-Powered (Semantic Understanding)**:
- Sentiment analysis (positive/neutral/negative)
- Emotion tagging (excited, curious, caring, etc.)
- Hook type identification (question, curiosity, emotion, benefit)
- Engagement trigger detection (question_at_end, relatable_story)
- Marketing feel assessment (soft/moderate/hard)
- Authenticity level evaluation (high/medium/low)

**Benefits**:
- Speed: Traditional NLP is instant
- Accuracy: LLM provides nuanced understanding
- Cost: Free tier model keeps costs near zero
- Robustness: Falls back to traditional features if LLM fails
