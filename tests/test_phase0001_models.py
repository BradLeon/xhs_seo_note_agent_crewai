"""Tests for Phase 0001 data models.

Tests for ContentIntent, VisualSubjects, GeneratedImages, MarketingCheck, OptimizedNote.
"""

import pytest
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from xhs_seo_optimizer.models.reports import (
    ContentIntent,
    VisualSubjects,
    GeneratedImage,
    GeneratedImages,
    MarketingCheck,
    OptimizedNote
)


class TestContentIntent:
    """Test ContentIntent model."""

    def test_content_intent_valid_required_only(self):
        """Test ContentIntent with only required fields."""
        intent = ContentIntent(
            core_theme="DHA选购攻略",
            target_persona="新手妈妈",
            key_message="科学配比是关键"
        )

        assert intent.core_theme == "DHA选购攻略"
        assert intent.target_persona == "新手妈妈"
        assert intent.key_message == "科学配比是关键"
        assert intent.unique_angle is None
        assert intent.emotional_tone is None

    def test_content_intent_valid_all_fields(self):
        """Test ContentIntent with all fields."""
        intent = ContentIntent(
            core_theme="DHA选购攻略",
            target_persona="新手妈妈",
            key_message="科学配比是关键",
            unique_angle="老爸测评专业视角",
            emotional_tone="专业但亲切"
        )

        assert intent.unique_angle == "老爸测评专业视角"
        assert intent.emotional_tone == "专业但亲切"

    def test_content_intent_core_theme_min_length(self):
        """Test ContentIntent core_theme minimum length validation."""
        with pytest.raises(ValueError):
            ContentIntent(
                core_theme="D",  # Too short, min 2
                target_persona="新手妈妈",
                key_message="科学配比是关键"
            )

    def test_content_intent_core_theme_max_length(self):
        """Test ContentIntent core_theme maximum length validation."""
        with pytest.raises(ValueError):
            ContentIntent(
                core_theme="A" * 51,  # Too long, max 50
                target_persona="新手妈妈",
                key_message="科学配比是关键"
            )

    def test_content_intent_key_message_min_length(self):
        """Test ContentIntent key_message minimum length validation."""
        with pytest.raises(ValueError):
            ContentIntent(
                core_theme="DHA选购",
                target_persona="妈妈",
                key_message="短"  # Too short, min 5
            )

    def test_content_intent_serialization(self):
        """Test ContentIntent JSON serialization."""
        intent = ContentIntent(
            core_theme="DHA选购攻略",
            target_persona="新手妈妈",
            key_message="科学配比是关键",
            unique_angle="专业测评",
            emotional_tone="亲切"
        )

        json_str = intent.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed['core_theme'] == "DHA选购攻略"
        assert parsed['unique_angle'] == "专业测评"


class TestVisualSubjects:
    """Test VisualSubjects model."""

    def test_visual_subjects_valid(self):
        """Test VisualSubjects with valid data."""
        subjects = VisualSubjects(
            subject_type="product",
            subject_description="诺特兰德DHA藻油产品，黄色包装瓶",
            brand_elements=["诺特兰德Logo", "黄色包装"],
            must_preserve=["产品包装", "Logo"],
            original_cover_url="https://example.com/cover.jpg",
            original_inner_urls=["https://example.com/inner1.jpg"]
        )

        assert subjects.subject_type == "product"
        assert len(subjects.brand_elements) == 2
        assert len(subjects.must_preserve) == 2

    def test_visual_subjects_valid_types(self):
        """Test VisualSubjects with different subject types."""
        valid_types = ["product", "person", "brand", "scene", "none"]

        for subject_type in valid_types:
            subjects = VisualSubjects(
                subject_type=subject_type,
                subject_description="这是一个测试的主体描述，用于验证不同类型",  # min_length=5
                brand_elements=[],
                must_preserve=[],
                original_cover_url="https://example.com/cover.jpg",
                original_inner_urls=[]
            )
            assert subjects.subject_type == subject_type

    def test_visual_subjects_empty_lists(self):
        """Test VisualSubjects with empty lists."""
        subjects = VisualSubjects(
            subject_type="scene",
            subject_description="温馨家居场景",
            brand_elements=[],
            must_preserve=[],
            original_cover_url="https://example.com/cover.jpg",
            original_inner_urls=[]
        )

        assert subjects.brand_elements == []
        assert subjects.must_preserve == []
        assert subjects.original_inner_urls == []


class TestGeneratedImage:
    """Test GeneratedImage model."""

    def test_generated_image_success(self):
        """Test GeneratedImage with successful generation."""
        image = GeneratedImage(
            image_type="cover",
            success=True,
            image_url="https://example.com/generated.png",
            local_path="/outputs/images/cover_123.png",
            error=None,
            prompt_used="温馨母婴场景",
            reference_image_used="https://example.com/ref.jpg"
        )

        assert image.success is True
        assert image.image_url is not None
        assert image.error is None

    def test_generated_image_failure(self):
        """Test GeneratedImage with failed generation."""
        image = GeneratedImage(
            image_type="cover",
            success=False,
            image_url=None,
            local_path=None,
            error="API rate limit exceeded",
            prompt_used="温馨母婴场景",
            reference_image_used=None
        )

        assert image.success is False
        assert image.image_url is None
        assert "rate limit" in image.error

    def test_generated_image_types(self):
        """Test GeneratedImage with different image types."""
        valid_types = ["cover", "inner_1", "inner_2", "inner_3"]

        for image_type in valid_types:
            image = GeneratedImage(
                image_type=image_type,
                success=True,
                prompt_used="测试prompt"
            )
            assert image.image_type == image_type


class TestGeneratedImages:
    """Test GeneratedImages model."""

    def test_generated_images_with_cover_only(self):
        """Test GeneratedImages with only cover image."""
        images = GeneratedImages(
            cover_image=GeneratedImage(
                image_type="cover",
                success=True,
                image_url="https://example.com/cover.png",
                prompt_used="封面图prompt"
            ),
            inner_images=[],
            generation_timestamp="2025-01-15T10:30:00Z"
        )

        assert images.cover_image is not None
        assert len(images.inner_images) == 0

    def test_generated_images_with_inner(self):
        """Test GeneratedImages with cover and inner images."""
        images = GeneratedImages(
            cover_image=GeneratedImage(
                image_type="cover",
                success=True,
                prompt_used="封面图"
            ),
            inner_images=[
                GeneratedImage(
                    image_type="inner_1",
                    success=True,
                    prompt_used="内页图1"
                ),
                GeneratedImage(
                    image_type="inner_2",
                    success=False,
                    error="Generation failed",
                    prompt_used="内页图2"
                )
            ],
            generation_timestamp="2025-01-15T10:30:00Z"
        )

        assert len(images.inner_images) == 2
        assert images.inner_images[0].success is True
        assert images.inner_images[1].success is False


class TestMarketingCheck:
    """Test MarketingCheck model."""

    def test_marketing_check_passed(self):
        """Test MarketingCheck with passed result."""
        check = MarketingCheck(
            original_score=0.6,
            optimized_score=0.3,
            level="low",
            passed=True,
            issues=[],
            suggestions=[]
        )

        assert check.passed is True
        assert check.optimized_score < check.original_score

    def test_marketing_check_failed(self):
        """Test MarketingCheck with failed result."""
        check = MarketingCheck(
            original_score=0.4,
            optimized_score=0.7,  # Increased - bad!
            level="high",
            passed=False,
            issues=["营销感增加", "使用了硬广词汇"],
            suggestions=["减少推销语气", "使用客观描述"]
        )

        assert check.passed is False
        assert len(check.issues) > 0
        assert len(check.suggestions) > 0

    def test_marketing_check_levels(self):
        """Test MarketingCheck with different levels."""
        valid_levels = ["low", "medium", "high", "critical"]

        for level in valid_levels:
            check = MarketingCheck(
                original_score=0.5,
                optimized_score=0.4,
                level=level,
                passed=True,
                issues=[],
                suggestions=[]
            )
            assert check.level == level


class TestOptimizedNote:
    """Test OptimizedNote model."""

    @pytest.fixture
    def valid_content_intent(self):
        """Create valid ContentIntent for testing."""
        return ContentIntent(
            core_theme="DHA选购攻略",
            target_persona="新手妈妈",
            key_message="科学配比是关键"
        )

    def test_optimized_note_valid(self, valid_content_intent):
        """Test OptimizedNote with valid data."""
        note = OptimizedNote(
            note_id="optimized_123",
            original_note_id="123",
            keyword="老爸测评dha推荐",
            title="救命！宝妈必看的DHA选购指南",
            content="养娃以后才知道...",
            cover_image_url="https://example.com/cover.png",
            inner_image_urls=["https://example.com/inner1.png"],
            prediction=None,
            tag=None,
            content_intent=valid_content_intent,
            marketing_check=None,
            optimization_summary="标题情感化改写，开头增加痛点钩子，封面图重新生成以提升吸引力",  # min_length=20
            cover_image_source="generated",
            inner_images_source="original",
            optimized_timestamp="2025-01-15T10:30:00Z"
        )

        assert note.note_id == "optimized_123"
        assert note.content_intent.core_theme == "DHA选购攻略"

    def test_optimized_note_with_marketing_check(self, valid_content_intent):
        """Test OptimizedNote with marketing check."""
        marketing_check = MarketingCheck(
            original_score=0.5,
            optimized_score=0.3,
            level="low",
            passed=True,
            issues=[],
            suggestions=[]
        )

        note = OptimizedNote(
            note_id="optimized_456",
            original_note_id="456",
            keyword="测试关键词",
            title="测试标题",
            content="测试内容",
            cover_image_url="https://example.com/cover.png",
            inner_image_urls=[],
            content_intent=valid_content_intent,
            marketing_check=marketing_check,
            optimization_summary="优化摘要：标题改写、开头钩子优化、营销感降低处理",  # min_length=20
            cover_image_source="original",
            inner_images_source="original",
            optimized_timestamp="2025-01-15T10:30:00Z"
        )

        assert note.marketing_check is not None
        assert note.marketing_check.passed is True

    def test_optimized_note_image_sources(self, valid_content_intent):
        """Test OptimizedNote with different image source values."""
        # cover_image_source: only 'generated' or 'original'
        # inner_images_source: 'generated', 'original', or 'mixed'
        test_cases = [
            ("generated", "generated"),
            ("original", "original"),
            ("generated", "mixed"),
            ("original", "mixed"),
        ]

        for cover_source, inner_source in test_cases:
            note = OptimizedNote(
                note_id="test",
                original_note_id="orig",
                keyword="keyword",
                title="title",
                content="content",
                cover_image_url="https://example.com/cover.png",
                inner_image_urls=[],
                content_intent=valid_content_intent,
                optimization_summary=f"图片来源测试 - 封面{cover_source}、内页{inner_source}，验证模型接受有效值",  # min_length=20
                cover_image_source=cover_source,
                inner_images_source=inner_source,
                optimized_timestamp="2025-01-15T10:30:00Z"
            )
            assert note.cover_image_source == cover_source
            assert note.inner_images_source == inner_source

    def test_optimized_note_serialization(self, valid_content_intent):
        """Test OptimizedNote JSON serialization."""
        note = OptimizedNote(
            note_id="optimized_789",
            original_note_id="789",
            keyword="DHA推荐",
            title="优化后的标题",
            content="优化后的内容",
            cover_image_url="https://example.com/cover.png",
            inner_image_urls=["https://example.com/inner1.png"],
            content_intent=valid_content_intent,
            optimization_summary="标题和开头优化，封面图使用AI生成，内页图保留原有图片",  # min_length=20
            cover_image_source="generated",
            inner_images_source="mixed",
            optimized_timestamp="2025-01-15T10:30:00Z"
        )

        json_str = note.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed['note_id'] == "optimized_789"
        assert parsed['content_intent']['core_theme'] == "DHA选购攻略"
        assert parsed['cover_image_source'] == "generated"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
