"""Tests for ImageGeneratorTool - 图像生成工具测试.

Phase 0001: Tests for image generation via OpenRouter Gemini API.
"""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import base64

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from xhs_seo_optimizer.tools.image_generator import ImageGeneratorTool


class TestImageGeneratorToolInstantiation:
    """Test ImageGeneratorTool instantiation."""

    def test_tool_instantiation_with_env_vars(self):
        """Test tool can be instantiated with environment variables."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-api-key'
        }):
            tool = ImageGeneratorTool()
            assert tool is not None
            assert tool.name == "image_generator"

    def test_tool_has_required_attributes(self):
        """Test tool has required attributes."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-api-key'
        }):
            tool = ImageGeneratorTool()
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
            assert hasattr(tool, '_run')


class TestImageGeneratorToolPromptBuilding:
    """Test prompt building logic."""

    @pytest.fixture
    def tool(self):
        """Create ImageGeneratorTool instance."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-api-key'
        }):
            return ImageGeneratorTool()

    def test_build_full_prompt_basic(self, tool):
        """Test basic prompt building."""
        prompt = "温馨的母婴场景，年轻妈妈和宝宝"
        full_prompt = tool._build_full_prompt(prompt, None, None)

        assert prompt in full_prompt
        # Should include XHS style guidance
        assert "小红书" in full_prompt or "真实" in full_prompt

    def test_build_full_prompt_with_must_preserve(self, tool):
        """Test prompt building with must_preserve elements."""
        prompt = "产品特写图"
        must_preserve = ["诺特兰德Logo", "黄色包装", "DHA标识"]

        full_prompt = tool._build_full_prompt(prompt, must_preserve, None)

        assert prompt in full_prompt
        # All must_preserve elements should be included
        for element in must_preserve:
            assert element in full_prompt

    def test_build_full_prompt_with_style_reference(self, tool):
        """Test prompt building with style reference."""
        prompt = "产品特写图"
        style_ref = "小红书爆款育儿笔记风格，真实生活感"

        full_prompt = tool._build_full_prompt(prompt, None, style_ref)

        assert prompt in full_prompt
        assert style_ref in full_prompt


class TestImageGeneratorToolAPICall:
    """Test API call logic with mocks."""

    @pytest.fixture
    def tool(self):
        """Create ImageGeneratorTool instance."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-api-key'
        }):
            return ImageGeneratorTool()

    def test_generate_image_success(self, tool):
        """Test successful image generation by mocking OpenAI client."""
        # Create mock response with image data
        test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        mock_message = MagicMock()
        mock_message.content = [
            {"type": "image", "data": test_image_base64}
        ]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]

        with patch('xhs_seo_optimizer.tools.image_generator.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            result = tool._generate_image("测试prompt")

            assert result['success'] is True
            assert 'image_data' in result

    def test_generate_image_failure(self, tool):
        """Test image generation failure handling."""
        with patch('xhs_seo_optimizer.tools.image_generator.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client

            result = tool._generate_image("测试prompt")

            assert result['success'] is False
            assert result['error'] is not None
            assert "API Error" in result['error']


class TestImageGeneratorToolImageSaving:
    """Test image saving logic."""

    @pytest.fixture
    def tool(self):
        """Create ImageGeneratorTool instance."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-api-key'
        }):
            return ImageGeneratorTool()

    def test_save_image_creates_directory(self, tool, tmp_path):
        """Test that save_image creates output directory if not exists."""
        # Create a simple test image base64
        test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        output_dir = tmp_path / "outputs" / "images"

        with patch.object(tool, 'output_dir', str(output_dir)):
            path = tool._save_image(test_image_base64, "cover")

            assert output_dir.exists()
            assert Path(path).exists()

    def test_save_image_correct_filename(self, tool, tmp_path):
        """Test that saved image has correct filename pattern."""
        test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        output_dir = tmp_path / "outputs" / "images"

        with patch.object(tool, 'output_dir', str(output_dir)):
            path = tool._save_image(test_image_base64, "cover")

            filename = Path(path).name
            assert "cover" in filename
            assert filename.endswith(".png")


class TestImageGeneratorToolRun:
    """Test the main _run method."""

    @pytest.fixture
    def tool(self):
        """Create ImageGeneratorTool instance."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-api-key'
        }):
            return ImageGeneratorTool()

    def test_run_returns_json_string(self, tool):
        """Test that _run returns valid JSON string."""
        mock_result = {
            'success': True,
            'image_data': 'base64data',
            'error': None
        }

        with patch.object(tool, '_generate_image', return_value=mock_result):
            with patch.object(tool, '_save_image', return_value='/path/to/image.png'):
                result = tool._run(
                    prompt="测试prompt",
                    image_type="cover"
                )

                # Result should be valid JSON
                parsed = json.loads(result)
                assert 'success' in parsed

    def test_run_with_all_parameters(self, tool):
        """Test _run with all optional parameters."""
        mock_result = {
            'success': True,
            'image_data': 'base64data',
            'error': None
        }

        with patch.object(tool, '_generate_image', return_value=mock_result):
            with patch.object(tool, '_save_image', return_value='/path/to/image.png'):
                result = tool._run(
                    prompt="温馨母婴场景",
                    reference_image_url="https://example.com/ref.jpg",
                    must_preserve_elements=["产品Logo", "黄色包装"],
                    style_reference="小红书风格",
                    image_type="cover"
                )

                parsed = json.loads(result)
                assert parsed['success'] is True
                assert 'local_path' in parsed

    def test_run_handles_generation_failure(self, tool):
        """Test _run properly handles generation failure."""
        mock_result = {
            'success': False,
            'image_data': None,
            'error': "API rate limit exceeded"
        }

        with patch.object(tool, '_generate_image', return_value=mock_result):
            result = tool._run(prompt="测试prompt")

            parsed = json.loads(result)
            assert parsed['success'] is False
            assert 'error' in parsed
            assert parsed['error'] is not None


class TestImageGeneratorToolIntegration:
    """Integration tests (require API key, marked as slow)."""

    @pytest.mark.slow
    @pytest.mark.skipif(
        not os.environ.get('OPENROUTER_API_KEY'),
        reason="OPENROUTER_API_KEY not set"
    )
    def test_real_api_call(self):
        """Test real API call (requires valid API key)."""
        tool = ImageGeneratorTool()

        result = tool._run(
            prompt="一只可爱的小猫咪，简单的卡通风格",
            image_type="test"
        )

        parsed = json.loads(result)
        # Note: Success depends on API availability and quota
        assert 'success' in parsed

        if parsed['success']:
            assert 'local_path' in parsed
            assert parsed['local_path'] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
