"""Tests for MultiModalVisionTool - 多模态视觉分析工具测试.

Uses real data from docs/target_notes.json to test the tool.
Tests multi-image analysis (cover + inner images).
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.xhs_seo_optimizer.tools.multimodal_vision import MultiModalVisionTool
from src.xhs_seo_optimizer.models.analysis_results import VisionAnalysisResult
from src.xhs_seo_optimizer.models.note import Note, NoteMetaData


class TestMultiModalVisionTool:
    """Test suite for MultiModalVisionTool."""

    @pytest.fixture
    def tool(self):
        """Create MultiModalVisionTool instance with test API key."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-api-key',
            'OPENROUTER_VISION_MODEL': 'google/gemini-2.5-flash-lite',
            'OPENROUTER_SITE_URL': 'https://test.com',
            'OPENROUTER_SITE_NAME': 'Test App'
        }):
            return MultiModalVisionTool()

    @pytest.fixture
    def sample_notes(self):
        """Load real notes from target_notes.json."""
        docs_dir = Path(__file__).parent.parent.parent / "docs"
        with open(docs_dir / "target_notes.json", "r", encoding="utf-8") as f:
            notes_data = json.load(f)
        # Return first 2 notes
        return [Note.from_json(note_data) for note_data in notes_data[:2]]

    @pytest.fixture
    def mock_vision_response(self):
        """Mock complete vision analysis response with all new fields."""
        return {
            # 图片基础分析
            "image_count": 2,
            "image_quality": "高清，清晰度好",
            "image_content_relation": "与标题和正文高度相关，展示产品列表",
            "image_composition": "平铺式构图，整齐排列",

            # 视觉风格分析
            "image_style": "清单式、实用型",
            "color_scheme": "温和柔和的色调，白色背景为主",
            "visual_tone": "专业、可信、实用",

            # 排版设计分析
            "layout_style": "列表式排版，信息密集",
            "visual_hierarchy": "标题-分类-产品名，层次清晰",

            # OCR 文字分析
            "text_ocr_content": "VC、儿童补铁、钙片、进口VD、锌、益生菌、婴儿米粉等产品名称和品牌",
            "text_ocr_content_highlight": "老爸评测推荐",

            # 用户体验分析
            "user_experience_analysis": "信息量大，适合收藏查阅，真实感强",
            "thumbnail_appeal": "高 - 标题醒目，包含权威背书",
            "visual_storytelling": "清单式叙事，逻辑连贯",
            "realistic_and_emotional_tone": "真实、可信、专业",

            # 品牌识别分析
            "brand_consistency": "符合小红书实用分享调性，个人笔记风格",
            "personal_style": "整理控、实用主义、注重权威背书",

            # 关键指标
            "scroll_stopping_power": "高 - 醒目的表情符号和'老爸评测'关键词"
        }

    def test_tool_initialization(self, tool):
        """Test tool initializes with correct configuration."""
        assert tool.name == "multimodal_vision_analysis"
        assert tool.api_key == "test-api-key"
        assert tool.model == "google/gemini-2.5-flash-lite"
        assert tool.max_inner_images == 4
        assert "封面图和内页图" in tool.description

    def test_run_missing_api_key(self):
        """Test _run raises error when API key is missing."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': ''}):
            tool = MultiModalVisionTool()
            note_meta_data = NoteMetaData(
                title="测试标题",
                content="测试内容",
                cover_image_url="https://example.com/cover.jpg",
                inner_image_urls=["https://example.com/inner1.jpg"]
            )

            with pytest.raises(ValueError, match="OPENROUTER_API_KEY not found"):
                tool._run(note_meta_data=note_meta_data, note_id="test")

    def test_run_empty_cover_image(self, tool):
        """Test _run raises error for empty cover_image_url."""
        note_meta_data = NoteMetaData(
            title="测试标题",
            content="测试内容",
            cover_image_url="",
            inner_image_urls=[]
        )

        with pytest.raises(ValueError, match="cannot be empty"):
            tool._run(note_meta_data=note_meta_data, note_id="test")

    @patch('src.xhs_seo_optimizer.tools.multimodal_vision.OpenAI')
    def test_run_success_with_real_note(
        self,
        mock_openai_class,
        tool,
        sample_notes,
        mock_vision_response
    ):
        """Test successful vision analysis with real note data."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(mock_vision_response)
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
        assert "image_count" in result_dict
        assert "image_quality" in result_dict
        assert "image_style" in result_dict
        assert "color_scheme" in result_dict
        assert "text_ocr_content" in result_dict
        assert "thumbnail_appeal" in result_dict
        assert "visual_storytelling" in result_dict
        assert "scroll_stopping_power" in result_dict

        # Verify can be deserialized to VisionAnalysisResult
        result_obj = VisionAnalysisResult(**result_dict)
        assert result_obj.note_id == note.note_id
        assert result_obj.image_count == 2
        assert result_obj.thumbnail_appeal == "高 - 标题醒目，包含权威背书"

    @patch('src.xhs_seo_optimizer.tools.multimodal_vision.OpenAI')
    def test_multi_image_api_call(
        self,
        mock_openai_class,
        tool,
        sample_notes,
        mock_vision_response
    ):
        """Test multi-image API call with correct parameters."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(mock_vision_response)
        mock_client.chat.completions.create.return_value = mock_response

        # Get note with multiple images
        note = sample_notes[0]
        expected_image_count = 1 + min(4, len(note.meta_data.inner_image_urls))

        # Run analysis
        tool._run(note_meta_data=note.meta_data, note_id=note.note_id)

        # Verify OpenAI client was initialized correctly
        mock_openai_class.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1",
            api_key="test-api-key"
        )

        # Verify API call
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "google/gemini-2.5-flash-lite"
        assert call_kwargs["temperature"] == 0.3

        # Verify multi-image content structure
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        content = messages[0]["content"]
        assert isinstance(content, list)

        # Count image_url items (should be 1 cover + N inner, max 5 total)
        image_items = [item for item in content if item["type"] == "image_url"]
        assert len(image_items) == expected_image_count
        assert len(image_items) <= 5  # Max 5 images total

        # Verify text prompt is first item
        assert content[0]["type"] == "text"
        assert "小红书笔记图片" in content[0]["text"]

    @patch('src.xhs_seo_optimizer.tools.multimodal_vision.OpenAI')
    def test_inner_image_limit(
        self,
        mock_openai_class,
        tool,
        mock_vision_response
    ):
        """Test that inner images are limited to max 4."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(mock_vision_response)
        mock_client.chat.completions.create.return_value = mock_response

        # Create note with 10 inner images
        note_meta_data = NoteMetaData(
            title="测试标题",
            content="测试内容",
            cover_image_url="https://example.com/cover.jpg",
            inner_image_urls=[
                f"https://example.com/inner{i}.jpg" for i in range(10)
            ]
        )

        # Run analysis
        tool._run(note_meta_data=note_meta_data, note_id="test")

        # Verify only 5 images sent (1 cover + 4 inner)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        content = call_kwargs["messages"][0]["content"]
        image_items = [item for item in content if item["type"] == "image_url"]
        assert len(image_items) == 5  # 1 cover + 4 inner (max)

    def test_build_vision_prompt(self, tool, sample_notes):
        """Test vision prompt construction with NoteMetaData."""
        note = sample_notes[0]
        prompt = tool._build_vision_prompt(note.meta_data)

        # Verify prompt contains key instructions
        assert "小红书笔记图片" in prompt
        assert "图片基础分析" in prompt
        assert "视觉风格分析" in prompt
        assert "排版设计分析" in prompt
        assert "OCR" in prompt
        assert "用户体验分析" in prompt
        assert "品牌识别分析" in prompt
        assert "expected_output" in prompt

        # Verify note metadata is included
        assert note.meta_data.title in prompt
        assert note.meta_data.content[:100] in prompt
        assert note.meta_data.cover_image_url in prompt

    def test_parse_vision_response_json(self, tool, mock_vision_response):
        """Test parsing plain JSON response."""
        content = json.dumps(mock_vision_response)
        result = tool._parse_vision_response(content)

        assert result["image_count"] == 2
        assert result["image_style"] == "清单式、实用型"
        assert result["thumbnail_appeal"] == "高 - 标题醒目，包含权威背书"
        assert result["scroll_stopping_power"] == "高 - 醒目的表情符号和'老爸评测'关键词"

    def test_parse_vision_response_markdown(self, tool):
        """Test parsing JSON in markdown code block."""
        content = """视觉分析结果：

```json
{
    "image_count": 3,
    "image_quality": "中等",
    "image_content_relation": "相关",
    "image_composition": "简单排列",
    "image_style": "生活化",
    "color_scheme": "自然色调",
    "visual_tone": "轻松",
    "layout_style": "随意",
    "visual_hierarchy": "平铺",
    "text_ocr_content": "产品名称",
    "text_ocr_content_highlight": "DHA",
    "user_experience_analysis": "一般",
    "thumbnail_appeal": "中等",
    "visual_storytelling": "简单",
    "realistic_and_emotional_tone": "真实",
    "brand_consistency": "符合",
    "personal_style": "生活记录",
    "scroll_stopping_power": "中等"
}
```

分析完成。
"""

        result = tool._parse_vision_response(content)
        assert result["image_count"] == 3
        assert result["image_style"] == "生活化"
        assert result["scroll_stopping_power"] == "中等"

    def test_parse_vision_response_partial_data(self, tool):
        """Test parsing handles partial data with defaults."""
        content = json.dumps({
            "image_count": 1,
            "image_style": "简约",
            # Missing other fields
        })
        result = tool._parse_vision_response(content)

        # Should fill in defaults
        assert result["image_count"] == 1
        assert result["image_style"] == "简约"
        assert result["image_quality"] == "未评估"  # Default
        assert result["text_ocr_content"] == "无"  # Default
        assert result["scroll_stopping_power"] == "未评估"  # Default

    def test_parse_vision_response_invalid_json(self, tool):
        """Test parsing raises error for invalid JSON."""
        content = "This is not valid JSON"

        with pytest.raises(ValueError, match="Failed to parse"):
            tool._parse_vision_response(content)

    @patch('src.xhs_seo_optimizer.tools.multimodal_vision.OpenAI')
    def test_run_multiple_notes(
        self,
        mock_openai_class,
        tool,
        sample_notes,
        mock_vision_response
    ):
        """Test analyzing multiple notes in sequence."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(mock_vision_response)
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
            assert "image_count" in result
            assert "scroll_stopping_power" in result

    @patch('src.xhs_seo_optimizer.tools.multimodal_vision.OpenAI')
    def test_run_error_handling(self, mock_openai_class, tool, sample_notes):
        """Test _run error handling and immediate return."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("API error")

        note = sample_notes[0]

        with pytest.raises(RuntimeError, match="Vision analysis failed"):
            tool._run(note_meta_data=note.meta_data, note_id=note.note_id)

        # Verify no fallback attempted (error raised immediately)
        mock_client.chat.completions.create.assert_called_once()

    def test_all_required_fields_in_response(self, mock_vision_response):
        """Test that all 18 required fields are present in mock response."""
        expected_fields = [
            # 图片基础分析 (4)
            "image_count", "image_quality", "image_content_relation", "image_composition",
            # 视觉风格分析 (3)
            "image_style", "color_scheme", "visual_tone",
            # 排版设计分析 (2)
            "layout_style", "visual_hierarchy",
            # OCR 文字分析 (2)
            "text_ocr_content", "text_ocr_content_highlight",
            # 用户体验分析 (4)
            "user_experience_analysis", "thumbnail_appeal", "visual_storytelling",
            "realistic_and_emotional_tone",
            # 品牌识别分析 (2)
            "brand_consistency", "personal_style",
            # 关键指标 (1)
            "scroll_stopping_power"
        ]

        for field in expected_fields:
            assert field in mock_vision_response, f"Missing field: {field}"

        # Verify total count
        assert len(expected_fields) == 18

    @patch('src.xhs_seo_optimizer.tools.multimodal_vision.OpenAI')
    def test_cover_and_inner_images_order(
        self,
        mock_openai_class,
        tool,
        mock_vision_response
    ):
        """Test that cover image is sent first, then inner images."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(mock_vision_response)
        mock_client.chat.completions.create.return_value = mock_response

        # Create note with cover and inner images
        note_meta_data = NoteMetaData(
            title="测试",
            content="内容",
            cover_image_url="https://example.com/cover.jpg",
            inner_image_urls=[
                "https://example.com/inner1.jpg",
                "https://example.com/inner2.jpg"
            ]
        )

        # Run analysis
        tool._run(note_meta_data=note_meta_data, note_id="test")

        # Get sent images
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        content = call_kwargs["messages"][0]["content"]
        image_items = [item for item in content if item["type"] == "image_url"]

        # Verify order: cover first, then inner images
        assert image_items[0]["image_url"]["url"] == "https://example.com/cover.jpg"
        assert image_items[1]["image_url"]["url"] == "https://example.com/inner1.jpg"
        assert image_items[2]["image_url"]["url"] == "https://example.com/inner2.jpg"

    @patch('src.xhs_seo_optimizer.tools.multimodal_vision.OpenAI')
    def test_only_cover_image(
        self,
        mock_openai_class,
        tool,
        mock_vision_response
    ):
        """Test analysis with only cover image (no inner images)."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(mock_vision_response)
        mock_client.chat.completions.create.return_value = mock_response

        # Create note with only cover image
        note_meta_data = NoteMetaData(
            title="测试",
            content="内容",
            cover_image_url="https://example.com/cover.jpg",
            inner_image_urls=[]
        )

        # Run analysis
        result_json = tool._run(note_meta_data=note_meta_data, note_id="test")

        # Should succeed
        result = json.loads(result_json)
        assert result["note_id"] == "test"

        # Verify only 1 image sent
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        content = call_kwargs["messages"][0]["content"]
        image_items = [item for item in content if item["type"] == "image_url"]
        assert len(image_items) == 1

    @patch('src.xhs_seo_optimizer.tools.multimodal_vision.OpenAI')
    def test_real_note_urls_in_api_call(
        self,
        mock_openai_class,
        tool,
        sample_notes,
        mock_vision_response
    ):
        """Test that real note image URLs are included in API call."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(mock_vision_response)
        mock_client.chat.completions.create.return_value = mock_response

        note = sample_notes[0]

        # Run analysis
        tool._run(note_meta_data=note.meta_data, note_id=note.note_id)

        # Get sent images
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        content = call_kwargs["messages"][0]["content"]
        image_items = [item for item in content if item["type"] == "image_url"]

        # Verify cover image URL
        assert image_items[0]["image_url"]["url"] == note.meta_data.cover_image_url

        # Verify inner image URLs (up to 4)
        for i, inner_url in enumerate(note.meta_data.inner_image_urls[:4], start=1):
            assert image_items[i]["image_url"]["url"] == inner_url
