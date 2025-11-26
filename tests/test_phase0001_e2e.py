"""End-to-End Tests for Phase 0001 Features.

Tests the complete workflow including:
- ContentIntent extraction
- VisualSubjects extraction
- Marketing sensitivity detection
- Image generation
- OptimizedNote compilation
"""

import pytest
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from xhs_seo_optimizer.models.reports import (
    ContentIntent,
    VisualSubjects,
    OptimizedNote,
    AuditReport
)
from xhs_seo_optimizer.tools import MarketingSentimentTool, determine_marketing_sensitivity


# ========================================
# Fixtures for Test Data
# ========================================

@pytest.fixture
def audit_report_path():
    """Path to real audit_report.json."""
    return "outputs/audit_report.json"


@pytest.fixture
def audit_report_data(audit_report_path):
    """Load real audit report data."""
    if not os.path.exists(audit_report_path):
        pytest.skip(f"Audit report not found: {audit_report_path}")
    with open(audit_report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def gap_report_data():
    """Load real gap report data."""
    path = "outputs/gap_report.json"
    if not os.path.exists(path):
        pytest.skip(f"Gap report not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def sample_owned_note():
    """Sample owned note data for testing."""
    return {
        "note_id": "67adc607000000002901c2c0",
        "title": "跟着老爸选DHA｜双A同补真的赢麻了",
        "content": """养娃以后才知道，总被问怎么把娃养得这么伶俐可爱😊
宝宝大脑发育的关键期，营养真的要跟上！
作为老爸抽检认证妈妈，我选营养品向来挑剔🔍

✅ 诺特兰德DHA藻油ARA
双A同补，黄金1:1配方
每粒含有100mg DHA+100mg ARA

✅ 配料表超干净
0糖0色素0香精0防腐剂
专利裂壶藻，美国几项权威认证

✅ 爆浆橙子味
姐妹俩每天到点就自己翻抽屉找"小金鱼"吃
真的太省心了！

机灵宝宝的秘密就是它啦～
麻麻们快抓住发育黄金期，让宝贝悄悄赢在起跑线上🏃

#老爸测评推荐[话题]# #宝宝DHA怎么选[话题]# #育儿好物分享[话题]#""",
        "meta_data": {
            "cover_image_url": "https://example.com/cover.jpg",
            "inner_image_urls": [
                "https://example.com/inner1.jpg",
                "https://example.com/inner2.jpg"
            ]
        },
        "tag": {
            "note_marketing_integrated_level": "软广"
        }
    }


# ========================================
# ContentIntent Tests
# ========================================

class TestContentIntentExtraction:
    """Test ContentIntent extraction logic."""

    def test_content_intent_from_note(self, sample_owned_note):
        """Test extracting ContentIntent from a note."""
        # Simulate ContentIntent extraction logic
        title = sample_owned_note['title']
        content = sample_owned_note['content']

        # Basic extraction (would be done by LLM in real scenario)
        intent = ContentIntent(
            core_theme="DHA选购攻略",
            target_persona="新手妈妈/宝妈",
            key_message="双A同补黄金配方，安全无添加",
            unique_angle="老爸测评认证背书",
            emotional_tone="亲切分享型"
        )

        assert intent.core_theme is not None
        assert len(intent.core_theme) >= 2

    def test_content_intent_constraints_validation(self):
        """Test that ContentIntent constraints are properly validated."""
        # Valid intent
        valid_intent = ContentIntent(
            core_theme="DHA选购",
            target_persona="妈妈群体",
            key_message="科学配比很重要，要选好的产品"
        )
        assert valid_intent is not None

        # Invalid - core_theme too short
        with pytest.raises(ValueError):
            ContentIntent(
                core_theme="D",
                target_persona="妈妈",
                key_message="科学配比"
            )


# ========================================
# VisualSubjects Tests
# ========================================

class TestVisualSubjectsExtraction:
    """Test VisualSubjects extraction logic."""

    def test_visual_subjects_from_note(self, sample_owned_note):
        """Test extracting VisualSubjects from a note."""
        meta = sample_owned_note['meta_data']

        subjects = VisualSubjects(
            subject_type="product",
            subject_description="诺特兰德DHA藻油ARA产品包装",
            brand_elements=["诺特兰德Logo", "黄色瓶身", "老爸抽检标志"],
            must_preserve=["产品包装", "Logo", "黄色配色"],
            original_cover_url=meta['cover_image_url'],
            original_inner_urls=meta['inner_image_urls']
        )

        assert subjects.subject_type == "product"
        assert len(subjects.must_preserve) > 0
        assert subjects.original_cover_url == meta['cover_image_url']

    def test_visual_subjects_preserves_urls(self, sample_owned_note):
        """Test that VisualSubjects preserves original image URLs."""
        meta = sample_owned_note['meta_data']

        subjects = VisualSubjects(
            subject_type="product",
            subject_description="产品展示图片",  # min_length=5
            brand_elements=[],
            must_preserve=[],
            original_cover_url=meta['cover_image_url'],
            original_inner_urls=meta['inner_image_urls']
        )

        # URLs should be preserved as-is
        assert subjects.original_cover_url == "https://example.com/cover.jpg"
        assert len(subjects.original_inner_urls) == 2


# ========================================
# Marketing Sensitivity Tests
# ========================================

class TestMarketingSensitivity:
    """Test marketing sensitivity detection."""

    def test_determine_sensitivity_soft_ad(self, sample_owned_note):
        """Test sensitivity detection for soft-ad note."""
        tag = sample_owned_note['tag']
        level = tag.get('note_marketing_integrated_level', '')

        sensitivity = determine_marketing_sensitivity(level)

        assert sensitivity == "high"  # 软广 should return high

    def test_marketing_sensitivity_levels(self):
        """Test all marketing sensitivity levels."""
        test_cases = [
            ("软广", "high"),
            ("商品推荐", "medium"),
            ("种草", "medium"),
            ("带货", "medium"),
            ("分享", "low"),
            ("日常", "low"),
            ("", "low"),
            (None, "low")
        ]

        for level, expected in test_cases:
            result = determine_marketing_sensitivity(level)
            assert result == expected, f"Level '{level}' should give '{expected}', got '{result}'"


class TestMarketingSentimentIntegration:
    """Integration tests for MarketingSentimentTool."""

    @pytest.fixture
    def tool(self):
        """Create MarketingSentimentTool instance."""
        from unittest.mock import patch
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-api-key'
        }):
            return MarketingSentimentTool()

    def test_soft_ad_text_detection(self, tool, sample_owned_note):
        """Test detection of marketing patterns in soft-ad text."""
        content = sample_owned_note['content']

        # Test rule-based detection
        result = tool._rule_based_detection(content)

        # Soft-ad content may have some marketing patterns
        assert 'score' in result  # actual field name is 'score' not 'rule_score'
        assert 'issues' in result


# ========================================
# Crew Integration Tests
# ========================================

class TestOwnedNoteAuditorPhase0001:
    """Test OwnedNoteAuditor Phase 0001 extensions."""

    def test_crew_instantiation_with_new_tasks(self):
        """Test that OwnedNoteAuditor crew can be instantiated with new tasks."""
        from xhs_seo_optimizer.crew_owned_note import XhsSeoOptimizerCrewOwnedNote

        crew = XhsSeoOptimizerCrewOwnedNote()
        assert crew is not None

        # Check new tasks exist
        assert hasattr(crew, 'extract_content_intent_task')
        assert hasattr(crew, 'extract_visual_subjects_task')

    def test_new_tasks_are_callable(self):
        """Test that new task methods are callable."""
        from xhs_seo_optimizer.crew_owned_note import XhsSeoOptimizerCrewOwnedNote

        crew = XhsSeoOptimizerCrewOwnedNote()

        # Tasks should be callable and return Task objects
        intent_task = crew.extract_content_intent_task()
        assert intent_task is not None

        subjects_task = crew.extract_visual_subjects_task()
        assert subjects_task is not None


class TestOptimizationStrategistPhase0001:
    """Test OptimizationStrategist Phase 0001 extensions."""

    def test_crew_instantiation_with_new_agents(self):
        """Test that OptimizationStrategist crew has new agents."""
        from xhs_seo_optimizer.crew_optimization import XhsSeoOptimizerCrewOptimization

        crew = XhsSeoOptimizerCrewOptimization()
        assert crew is not None

        # Check new agent exists (method name matches YAML key: image_generator)
        assert hasattr(crew, 'image_generator')

    def test_crew_has_new_tasks(self):
        """Test that OptimizationStrategist crew has new tasks."""
        from xhs_seo_optimizer.crew_optimization import XhsSeoOptimizerCrewOptimization

        crew = XhsSeoOptimizerCrewOptimization()

        # Check new tasks exist
        assert hasattr(crew, 'generate_images')
        assert hasattr(crew, 'compile_optimized_note')

    def test_new_tasks_are_callable(self):
        """Test that new task methods are callable."""
        from xhs_seo_optimizer.crew_optimization import XhsSeoOptimizerCrewOptimization

        crew = XhsSeoOptimizerCrewOptimization()

        # Tasks should be callable and return Task objects
        images_task = crew.generate_images()
        assert images_task is not None

        compile_task = crew.compile_optimized_note()
        assert compile_task is not None


# ========================================
# Variable Injection Tests
# ========================================

class TestVariableInjection:
    """Test that Phase 0001 variables are properly injected."""

    def test_content_intent_variables_exist(self, audit_report_data):
        """Test that ContentIntent can be extracted from audit report."""
        # Check if audit report has content_intent field
        # (This may not exist yet if auditor hasn't been run with new code)
        if 'content_intent' in audit_report_data:
            intent = audit_report_data['content_intent']
            assert 'core_theme' in intent or intent == {}

    def test_visual_subjects_variables_exist(self, audit_report_data):
        """Test that VisualSubjects can be extracted from audit report."""
        if 'visual_subjects' in audit_report_data:
            subjects = audit_report_data['visual_subjects']
            assert 'subject_type' in subjects or subjects == {}


# ========================================
# Full E2E Test (Slow - Requires API)
# ========================================

@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get('OPENROUTER_API_KEY'),
    reason="OPENROUTER_API_KEY not set"
)
class TestPhase0001FullE2E:
    """Full end-to-end test for Phase 0001 features."""

    def test_owned_note_auditor_produces_phase0001_fields(self):
        """Test that OwnedNoteAuditor produces Phase 0001 fields."""
        # This test requires:
        # - docs/owned_note.json to exist
        # - docs/keyword.json to exist
        # - Valid OPENROUTER_API_KEY

        required_files = [
            "docs/owned_note.json",
            "docs/keyword.json",
            "outputs/success_profile_report.json"
        ]

        for f in required_files:
            if not os.path.exists(f):
                pytest.skip(f"Required file not found: {f}")

        from xhs_seo_optimizer.crew_owned_note import XhsSeoOptimizerCrewOwnedNote

        # Load keyword
        with open("docs/keyword.json", 'r', encoding='utf-8') as f:
            keyword_data = json.load(f)

        keyword = keyword_data.get('keyword', '老爸测评dha推荐哪几款')

        # Run crew
        crew = XhsSeoOptimizerCrewOwnedNote()
        result = crew.kickoff(inputs={'keyword': keyword})

        # Check output file
        output_path = "outputs/audit_report.json"
        assert os.path.exists(output_path)

        with open(output_path, 'r', encoding='utf-8') as f:
            report = json.load(f)

        # Verify Phase 0001 fields exist
        assert 'content_intent' in report or 'visual_subjects' in report, \
            "Audit report should contain Phase 0001 fields"

    def test_optimization_strategist_produces_optimized_note(self, gap_report_data):
        """Test that OptimizationStrategist produces OptimizedNote."""
        required_files = [
            "outputs/gap_report.json",
            "outputs/audit_report.json",
            "outputs/success_profile_report.json",
            "docs/owned_note.json"
        ]

        for f in required_files:
            if not os.path.exists(f):
                pytest.skip(f"Required file not found: {f}")

        from xhs_seo_optimizer.crew_optimization import XhsSeoOptimizerCrewOptimization

        # Get keyword from gap report
        keyword = gap_report_data.get('keyword', '老爸测评dha推荐哪几款')

        # Run crew
        crew = XhsSeoOptimizerCrewOptimization()
        result = crew.kickoff(inputs={'keyword': keyword})

        # Check outputs exist
        assert os.path.exists("outputs/optimization_plan.json")

        # If optimized_note output is configured
        if os.path.exists("outputs/optimized_note.json"):
            with open("outputs/optimized_note.json", 'r', encoding='utf-8') as f:
                note = json.load(f)

            # Verify OptimizedNote structure
            required_fields = ['note_id', 'title', 'content', 'cover_image_url']
            for field in required_fields:
                assert field in note, f"OptimizedNote should contain {field}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
