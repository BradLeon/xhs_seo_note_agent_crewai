"""Integration tests for content analysis tools - 集成测试.

Tests the complete workflow with real note data from docs/ directory.
"""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.xhs_seo_optimizer.models.note import Note
from src.xhs_seo_optimizer.tools.multimodal_vision import MultiModalVisionTool
from src.xhs_seo_optimizer.tools.nlp_analysis import NLPAnalysisTool
from src.xhs_seo_optimizer.models.analysis_results import (
    VisionAnalysisResult,
    TextAnalysisResult
)


class TestIntegration:
    """Integration test suite for content analysis tools."""

    @pytest.fixture
    def docs_dir(self):
        """Path to docs directory with sample data."""
        return Path(__file__).parent.parent / "docs"

    @pytest.fixture
    def owned_note_data(self, docs_dir):
        """Load owned_note.json sample data."""
        with open(docs_dir / "owned_note.json", "r", encoding="utf-8") as f:
            return json.load(f)

    @pytest.fixture
    def target_notes_data(self, docs_dir):
        """Load target_notes.json sample data."""
        with open(docs_dir / "target_notes.json", "r", encoding="utf-8") as f:
            return json.load(f)

    @pytest.fixture
    def vision_tool(self):
        """Create vision tool with test config."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-api-key',
            'OPENROUTER_VISION_MODEL': 'google/gemini-2.5-flash-lite',
        }):
            return MultiModalVisionTool()

    @pytest.fixture
    def nlp_tool(self):
        """Create NLP tool with test config."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-api-key',
            'OPENROUTER_TEXT_MODEL': 'google/gemini-2.0-flash-thinking-exp-1219:free',
        }):
            return NLPAnalysisTool()

    def test_note_model_from_json(self, owned_note_data):
        """Test Note model can be created from sample data."""
        note = Note.from_json(owned_note_data)

        # Verify note structure
        assert note.note_id == owned_note_data["note_id"]
        assert note.meta_data.title == owned_note_data["title"]
        assert note.meta_data.content == owned_note_data["content"]
        assert note.meta_data.cover_image_url == owned_note_data["cover_image_url"]

        # Verify prediction
        assert note.prediction.ctr == owned_note_data["prediction"]["ctr"]
        assert note.prediction.sort_score2 == owned_note_data["prediction"]["sort_score2"]

        # Verify tag
        assert note.tag.intention_lv1 == owned_note_data["tag"]["intention_lv1"]
        assert note.tag.taxonomy1 == owned_note_data["tag"]["taxonomy1"]

    def test_note_model_from_target_notes(self, target_notes_data):
        """Test Note model works with target notes (array of notes)."""
        # Assuming target_notes.json is an array
        if isinstance(target_notes_data, list) and len(target_notes_data) > 0:
            first_note = target_notes_data[0]
            note = Note.from_json(first_note)

            assert note.note_id is not None
            assert note.meta_data.title is not None
            assert note.prediction.ctr is not None

    def test_nlp_tool_with_real_note(self, nlp_tool, owned_note_data):
        """Test NLP tool with real note text (without LLM call)."""
        # Test basic features without LLM
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': ''}):
            tool_no_llm = NLPAnalysisTool()

            result_json = tool_no_llm._run(
                text=owned_note_data["content"],
                title=owned_note_data["title"],
                note_id=owned_note_data["note_id"]
            )

            result = json.loads(result_json)

            # Verify basic features extracted
            assert result["note_id"] == owned_note_data["note_id"]
            assert result["length"] > 0
            assert result["word_count"] > 0
            assert isinstance(result["has_emojis"], bool)
            assert isinstance(result["keywords"], list)
            assert isinstance(result["xiaohongshu_buzzwords"], list)

            # Should have neutral sentiment without LLM
            assert result["sentiment"] == "neutral"

    @patch('src.xhs_seo_optimizer.tools.nlp_analysis.OpenAI')
    def test_nlp_tool_with_llm_mock(
        self,
        mock_openai_class,
        nlp_tool,
        owned_note_data
    ):
        """Test NLP tool with mocked LLM response."""
        # Setup mock LLM response
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "sentiment": "positive",
            "sentiment_score": 0.8,
            "emotion_tags": ["推荐", "真诚"],
            "hook_type": "产品推荐",
            "engagement_triggers": ["经验分享", "产品推荐"],
            "marketing_feel": "medium",
            "authenticity_level": "high"
        })
        mock_client.chat.completions.create.return_value = mock_response

        # Run analysis
        result_json = nlp_tool._run(
            text=owned_note_data["content"],
            title=owned_note_data["title"],
            note_id=owned_note_data["note_id"]
        )

        result = json.loads(result_json)

        # Verify combined features (basic + semantic)
        assert result["note_id"] == owned_note_data["note_id"]
        assert result["length"] > 0  # Basic feature
        assert result["sentiment"] == "positive"  # Semantic feature
        assert result["sentiment_score"] == 0.8
        assert len(result["emotion_tags"]) > 0

        # Verify result can be deserialized
        text_result = TextAnalysisResult(**result)
        assert text_result.note_id == owned_note_data["note_id"]

    @patch('src.xhs_seo_optimizer.tools.multimodal_vision.OpenAI')
    def test_vision_tool_with_real_image_url(
        self,
        mock_openai_class,
        vision_tool,
        owned_note_data
    ):
        """Test vision tool with real image URL (mocked API)."""
        # Setup mock vision response
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "style": "清新自然",
            "visual_type": "产品展示",
            "color_palette": ["暖色调", "柔和"],
            "text_overlay": True,
            "text_overlay_style": "产品信息标注",
            "composition": "中心构图",
            "subject_focus": "产品特写",
            "authenticity_level": "high",
            "emotion": "温馨",
            "xiaohongshu_style_match": 0.85
        })
        mock_client.chat.completions.create.return_value = mock_response

        # Run analysis with cover image
        result_json = vision_tool._run(
            image_url=owned_note_data["cover_image_url"],
            note_id=owned_note_data["note_id"]
        )

        result = json.loads(result_json)

        # Verify vision analysis result
        assert result["note_id"] == owned_note_data["note_id"]
        assert result["style"] is not None
        assert result["visual_type"] is not None
        assert isinstance(result["color_palette"], list)
        assert isinstance(result["text_overlay"], bool)
        assert result["composition"] is not None
        assert 0 <= result["xiaohongshu_style_match"] <= 1

        # Verify result can be deserialized
        vision_result = VisionAnalysisResult(**result)
        assert vision_result.note_id == owned_note_data["note_id"]

    @patch('src.xhs_seo_optimizer.tools.multimodal_vision.OpenAI')
    @patch('src.xhs_seo_optimizer.tools.nlp_analysis.OpenAI')
    def test_complete_note_analysis_workflow(
        self,
        mock_nlp_openai,
        mock_vision_openai,
        vision_tool,
        nlp_tool,
        owned_note_data
    ):
        """Test complete workflow: load note -> analyze text -> analyze images."""
        # Setup mocks
        # Vision mock
        mock_vision_client = MagicMock()
        mock_vision_openai.return_value = mock_vision_client
        mock_vision_response = MagicMock()
        mock_vision_response.choices = [MagicMock()]
        mock_vision_response.choices[0].message.content = json.dumps({
            "style": "生活化",
            "visual_type": "场景展示",
            "color_palette": ["暖色调"],
            "text_overlay": True,
            "text_overlay_style": "重点标注",
            "composition": "中心构图",
            "subject_focus": "产品使用场景",
            "authenticity_level": "high",
            "emotion": "温馨",
            "xiaohongshu_style_match": 0.9
        })
        mock_vision_client.chat.completions.create.return_value = mock_vision_response

        # NLP mock
        mock_nlp_client = MagicMock()
        mock_nlp_openai.return_value = mock_nlp_client
        mock_nlp_response = MagicMock()
        mock_nlp_response.choices = [MagicMock()]
        mock_nlp_response.choices[0].message.content = json.dumps({
            "sentiment": "positive",
            "sentiment_score": 0.85,
            "emotion_tags": ["推荐", "真诚", "分享"],
            "hook_type": "经验分享",
            "engagement_triggers": ["产品推荐", "使用体验", "话题引导"],
            "marketing_feel": "medium",
            "authenticity_level": "high"
        })
        mock_nlp_client.chat.completions.create.return_value = mock_nlp_response

        # Step 1: Load note
        note = Note.from_json(owned_note_data)
        assert note.note_id is not None

        # Step 2: Analyze text
        text_result_json = nlp_tool._run(
            text=note.meta_data.content,
            title=note.meta_data.title,
            note_id=note.note_id
        )
        text_result = json.loads(text_result_json)
        assert text_result["sentiment"] == "positive"

        # Step 3: Analyze cover image
        vision_result_json = vision_tool._run(
            image_url=note.meta_data.cover_image_url,
            note_id=note.note_id
        )
        vision_result = json.loads(vision_result_json)
        assert vision_result["xiaohongshu_style_match"] > 0

        # Verify both analyses completed successfully
        assert text_result["note_id"] == vision_result["note_id"] == note.note_id

    def test_error_handling_empty_text(self, nlp_tool):
        """Test error handling for empty text."""
        with pytest.raises(ValueError, match="cannot be empty"):
            nlp_tool._run(text="", note_id="test")

    def test_error_handling_invalid_url(self, vision_tool):
        """Test error handling for invalid image URL."""
        with pytest.raises(ValueError, match="Invalid image_url"):
            vision_tool._run(image_url="not-a-url", note_id="test")

    def test_data_files_exist(self, docs_dir):
        """Test that required data files exist."""
        assert (docs_dir / "owned_note.json").exists()
        assert (docs_dir / "target_notes.json").exists()
        assert (docs_dir / "keyword.json").exists()

    def test_data_files_valid_json(self, docs_dir):
        """Test that data files contain valid JSON."""
        for json_file in ["owned_note.json", "target_notes.json", "keyword.json"]:
            file_path = docs_dir / json_file
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                assert data is not None

    def test_owned_note_structure(self, owned_note_data):
        """Test owned_note.json has expected structure."""
        # Required top-level fields
        assert "note_id" in owned_note_data
        assert "title" in owned_note_data
        assert "content" in owned_note_data
        assert "cover_image_url" in owned_note_data
        assert "prediction" in owned_note_data
        assert "tag" in owned_note_data

        # Prediction fields
        prediction = owned_note_data["prediction"]
        assert "ctr" in prediction
        assert "sort_score2" in prediction

        # Tag fields
        tag = owned_note_data["tag"]
        assert "intention_lv1" in tag
        assert "taxonomy1" in tag

    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="Requires OPENROUTER_API_KEY for real API testing"
    )
    def test_real_api_nlp_analysis(self, owned_note_data):
        """Test with real API (optional, requires API key)."""
        tool = NLPAnalysisTool()

        result_json = tool._run(
            text=owned_note_data["content"],
            title=owned_note_data["title"],
            note_id=owned_note_data["note_id"]
        )

        result = json.loads(result_json)

        # Verify we got real semantic analysis
        assert result["sentiment"] in ["positive", "neutral", "negative"]
        assert -1 <= result["sentiment_score"] <= 1
        # With real API, should have non-empty emotion_tags
        assert len(result["emotion_tags"]) > 0

    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="Requires OPENROUTER_API_KEY for real API testing"
    )
    def test_real_api_vision_analysis(self, owned_note_data):
        """Test with real API (optional, requires API key)."""
        tool = MultiModalVisionTool()

        result_json = tool._run(
            image_url=owned_note_data["cover_image_url"],
            note_id=owned_note_data["note_id"]
        )

        result = json.loads(result_json)

        # Verify we got real vision analysis
        assert result["style"] is not None
        assert len(result["color_palette"]) > 0
        assert 0 <= result["xiaohongshu_style_match"] <= 1
