"""Tests for NLPAnalysisTool - 文本分析工具测试.

Uses real data from docs/target_notes.json to test the tool.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.xhs_seo_optimizer.tools.nlp_analysis import NLPAnalysisTool
from src.xhs_seo_optimizer.models.analysis_results import TextAnalysisResult
from src.xhs_seo_optimizer.models.note import Note, NoteMetaData


class TestNLPAnalysisTool:
    """Test suite for NLPAnalysisTool."""

    @pytest.fixture
    def tool(self):
        """Create NLPAnalysisTool instance with test API key."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-api-key',
            'OPENROUTER_TEXT_MODEL': 'google/gemini-2.0-flash-thinking-exp-1219:free',
            'OPENROUTER_SITE_URL': 'https://test.com',
            'OPENROUTER_SITE_NAME': 'Test App'
        }):
            return NLPAnalysisTool()

    @pytest.fixture
    def sample_notes(self):
        """Load real notes from target_notes.json."""
        docs_dir = Path(__file__).parent.parent.parent / "docs"
        with open(docs_dir / "target_notes.json", "r", encoding="utf-8") as f:
            notes_data = json.load(f)
        # Return first 2 notes
        print("get note_datas size:", len(notes_data))
        return [Note.from_json(note_data) for note_data in notes_data[:3]]

    @pytest.fixture
    def mock_text_analysis_response(self):
        """Mock complete text analysis response with all new fields."""
        return {
            # 标题分析
            "title_pattern": "推荐型标题（包含权威背书）",
            "title_keywords": ["老爸评测", "母婴用品", "推荐", "整理"],
            "title_emotion": "积极推荐",

            # 开头策略
            "opening_strategy": "信任背书开头",
            "opening_hook": "权威引用",
            "opening_impact": "高",

            # 正文框架
            "content_framework": "列表式",
            "content_logic": ["引言说明", "分类列举", "具体推荐"],
            "paragraph_structure": "清单式分点",

            # 结尾技巧
            "ending_technique": "无明显结尾",
            "ending_cta": "无",
            "ending_resonance": "中等",

            # 基础指标
            "word_count": 450,
            "readability_score": "高",
            "structure_completeness": "完整",

            # 痛点挖掘
            "pain_points": ["育儿产品选择困难", "不知道哪些产品可信"],
            "pain_intensity": "中等",

            # 价值主张
            "value_propositions": ["权威推荐", "整理归纳", "省时省力"],
            "value_hierarchy": ["权威性", "便利性", "可信度"],

            # 情感触发
            "emotional_triggers": ["信任", "从众", "便利"],
            "emotional_intensity": "中等",

            # 可信度建设
            "credibility_signals": ["老爸评测背书", "具体品牌列举"],
            "authority_indicators": ["老爸评测", "专业评测机构"],

            # 心理驱动
            "urgency_indicators": [],
            "social_proof": ["关注魏老爸"],
            "scarcity_elements": [],

            # 利益吸引
            "benefit_appeals": ["方便查找", "权威推荐", "避免踩坑"],
            "transformation_promise": "帮助家长选择安全可靠的母婴用品"
        }

    def test_tool_initialization(self, tool):
        """Test tool initializes with correct configuration."""
        assert tool.name == "nlp_text_analysis"
        assert tool.api_key == "test-api-key"
        assert "gemini" in tool.model.lower()
        assert "深度分析" in tool.description

    def test_run_missing_api_key(self):
        """Test _run raises error when API key is missing."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': ''}):
            tool = NLPAnalysisTool()
            note_meta_data = NoteMetaData(
                note_id="test-note-id",
                title="测试标题",
                content="测试内容",
                cover_image_url="https://example.com/image.jpg",
                inner_image_urls=[]
            )

            with pytest.raises(ValueError, match="OPENROUTER_API_KEY not found"):
                tool._run(note_meta_data=note_meta_data, note_id="test")

    def test_run_empty_content(self, tool):
        """Test _run raises error for empty content."""
        note_meta_data = NoteMetaData(
            note_id="test-note-id",
            title="测试标题",
            content="",
            cover_image_url="https://example.com/image.jpg",
            inner_image_urls=[]
        )

        with pytest.raises(ValueError, match="cannot be empty"):
            tool._run(note_meta_data=note_meta_data, note_id="test")

    @patch('src.xhs_seo_optimizer.tools.nlp_analysis.OpenAI')
    def test_run_success_with_real_note(
        self,
        mock_openai_class,
        tool,
        sample_notes,
        mock_text_analysis_response
    ):
        """Test successful analysis with real note data."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(mock_text_analysis_response)
        mock_client.chat.completions.create.return_value = mock_response

        # Get first real note
        note = sample_notes[0]

        # Run analysis
        result_json = tool._run(
            note_meta_data=note.meta_data,
            note_id=note.note_id
        )

        # Verify result is valid JSON
        result_dict = json.loads(result_json)
        assert result_dict["note_id"] == note.note_id

        # Verify all new fields exist
        assert "title_pattern" in result_dict
        assert "opening_hook" in result_dict
        assert "content_framework" in result_dict
        assert "ending_technique" in result_dict
        assert "word_count" in result_dict
        assert "pain_points" in result_dict
        assert "value_propositions" in result_dict
        assert "emotional_triggers" in result_dict
        assert "credibility_signals" in result_dict
        assert "transformation_promise" in result_dict

        # Verify can be deserialized to TextAnalysisResult
        result_obj = TextAnalysisResult(**result_dict)
        assert result_obj.note_id == note.note_id
        assert result_obj.title_pattern == "推荐型标题（包含权威背书）"
        assert len(result_obj.title_keywords) > 0
        assert result_obj.word_count == 450

    @patch('src.xhs_seo_optimizer.tools.nlp_analysis.OpenAI')
    def test_analyze_with_llm_api_call(
        self,
        mock_openai_class,
        tool,
        sample_notes,
        mock_text_analysis_response
    ):
        """Test LLM API call with correct parameters."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(mock_text_analysis_response)
        mock_client.chat.completions.create.return_value = mock_response

        note = sample_notes[0]

        # Call method
        result = tool._analyze_with_llm(note.meta_data)

        # Verify OpenAI client was initialized correctly
        mock_openai_class.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1",
            api_key="test-api-key"
        )

        # Verify API call
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "gemini" in call_kwargs["model"].lower()
        assert call_kwargs["temperature"] == 0.3
        assert "HTTP-Referer" in call_kwargs["extra_headers"]

        # Verify result
        assert result["title_pattern"] == "推荐型标题（包含权威背书）"
        assert result["word_count"] == 450
        assert len(result["pain_points"]) == 2

    def test_build_semantic_prompt(self, tool, sample_notes):
        """Test semantic analysis prompt construction."""
        note = sample_notes[0]
        prompt = tool._build_semantic_prompt(note.meta_data)

        # Verify prompt contains key instructions
        assert "小红书笔记文本" in prompt
        assert "标题分析" in prompt
        assert "开头策略分析" in prompt
        assert "正文框架分析" in prompt
        assert "结尾技巧分析" in prompt
        assert "痛点挖掘分析" in prompt
        assert "价值主张分析" in prompt
        assert "expected_output" in prompt

        # Verify note content is included
        assert note.meta_data.title in prompt
        assert note.meta_data.content[:50] in prompt

    def test_parse_semantic_response_json(self, tool, mock_text_analysis_response):
        """Test parsing plain JSON response."""
        content = json.dumps(mock_text_analysis_response)
        result = tool._parse_semantic_response(content)

        assert result["title_pattern"] == "推荐型标题（包含权威背书）"
        assert result["opening_hook"] == "权威引用"
        assert result["word_count"] == 450
        assert len(result["pain_points"]) == 2
        assert len(result["value_propositions"]) == 3

    def test_parse_semantic_response_markdown(self, tool):
        """Test parsing JSON in markdown code block."""
        content = """分析结果如下：

```json
{
    "title_pattern": "疑问型标题",
    "title_keywords": ["DHA", "推荐"],
    "title_emotion": "中性",
    "opening_strategy": "产品介绍",
    "opening_hook": "产品到货",
    "opening_impact": "中等",
    "content_framework": "叙事式",
    "content_logic": ["背景", "产品介绍", "购买理由"],
    "paragraph_structure": "流水账式",
    "ending_technique": "无",
    "ending_cta": "无",
    "ending_resonance": "低",
    "word_count": 80,
    "readability_score": "中等",
    "structure_completeness": "基本完整",
    "pain_points": ["不知道买哪个DHA"],
    "pain_intensity": "低",
    "value_propositions": ["价格优惠"],
    "value_hierarchy": ["价格"],
    "emotional_triggers": ["便宜"],
    "emotional_intensity": "低",
    "credibility_signals": ["老爸测评推荐"],
    "authority_indicators": ["老爸测评"],
    "urgency_indicators": [],
    "social_proof": ["群里买的大家都说好"],
    "scarcity_elements": [],
    "benefit_appeals": ["省钱"],
    "transformation_promise": "帮助宝宝大脑发育"
}
```

分析完成。
"""

        result = tool._parse_semantic_response(content)
        assert result["title_pattern"] == "疑问型标题"
        assert result["word_count"] == 80
        assert result["opening_hook"] == "产品到货"

    def test_parse_semantic_response_partial_data(self, tool):
        """Test parsing handles partial data with defaults."""
        content = json.dumps({
            "title_pattern": "数字型标题",
            "title_keywords": ["整理"],
            # Missing other fields
        })
        result = tool._parse_semantic_response(content)

        # Should fill in defaults
        assert result["title_pattern"] == "数字型标题"
        assert result["title_keywords"] == ["整理"]
        assert result["opening_hook"] == "未识别"  # Default
        assert result["word_count"] == 0  # Default
        assert result["pain_points"] == []  # Default

    def test_parse_semantic_response_invalid_json(self, tool):
        """Test parsing raises error for invalid JSON."""
        content = "This is not valid JSON"

        with pytest.raises(ValueError, match="Failed to parse"):
            tool._parse_semantic_response(content)

    @patch('src.xhs_seo_optimizer.tools.nlp_analysis.OpenAI')
    def test_run_multiple_notes(
        self,
        mock_openai_class,
        tool,
        sample_notes,
        mock_text_analysis_response
    ):
        """Test analyzing multiple notes in sequence."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(mock_text_analysis_response)
        mock_client.chat.completions.create.return_value = mock_response

        results = []
        for note in sample_notes:
            result_json = tool._run(
                note_meta_data=note.meta_data,
                note_id=note.note_id
            )
            results.append(json.loads(result_json))

        # Verify all notes were analyzed
        assert len(results) == len(sample_notes)
        for i, result in enumerate(results):
            assert result["note_id"] == sample_notes[i].note_id
            assert "title_pattern" in result
            assert "word_count" in result

    @patch('src.xhs_seo_optimizer.tools.nlp_analysis.OpenAI')
    def test_run_error_handling(self, mock_openai_class, tool, sample_notes):
        """Test _run error handling and immediate return."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("API error")

        note = sample_notes[0]

        with pytest.raises(RuntimeError, match="NLP analysis failed"):
            tool._run(note_meta_data=note.meta_data, note_id=note.note_id)

        # Verify no fallback attempted (error raised immediately)
        mock_client.chat.completions.create.assert_called_once()

    def test_all_required_fields_in_response(self, tool, mock_text_analysis_response):
        """Test that all 26+ required fields are present in mock response."""
        expected_fields = [
            # 标题分析 (3)
            "title_pattern", "title_keywords", "title_emotion",
            # 开头策略 (3)
            "opening_strategy", "opening_hook", "opening_impact",
            # 正文框架 (3)
            "content_framework", "content_logic", "paragraph_structure",
            # 结尾技巧 (3)
            "ending_technique", "ending_cta", "ending_resonance",
            # 基础指标 (3)
            "word_count", "readability_score", "structure_completeness",
            # 痛点挖掘 (2)
            "pain_points", "pain_intensity",
            # 价值主张 (2)
            "value_propositions", "value_hierarchy",
            # 情感触发 (2)
            "emotional_triggers", "emotional_intensity",
            # 可信度建设 (2)
            "credibility_signals", "authority_indicators",
            # 心理驱动 (3)
            "urgency_indicators", "social_proof", "scarcity_elements",
            # 利益吸引 (2)
            "benefit_appeals", "transformation_promise"
        ]

        for field in expected_fields:
            assert field in mock_text_analysis_response, f"Missing field: {field}"

        # Verify total count
        assert len(expected_fields) == 28

    @patch('src.xhs_seo_optimizer.tools.nlp_analysis.OpenAI')
    def test_real_note_content_in_prompt(
        self,
        mock_openai_class,
        tool,
        sample_notes,
        mock_text_analysis_response
    ):
        """Test that real note content is included in API prompt."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(mock_text_analysis_response)
        mock_client.chat.completions.create.return_value = mock_response

        note = sample_notes[0]

        # Run analysis
        tool._run(note_meta_data=note.meta_data, note_id=note.note_id)

        # Get the prompt that was sent
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        prompt_content = messages[0]["content"]

        # Verify note title and content are in prompt
        assert note.meta_data.title in prompt_content
        assert note.meta_data.content[:100] in prompt_content
