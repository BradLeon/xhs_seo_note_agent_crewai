"""Tests for OptimizationStrategist crew and models.

Tests model validation, crew instantiation, and execution flow.
"""

import pytest
import json
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from xhs_seo_optimizer.models.reports import (
    OptimizationItem,
    TitleOptimization,
    ContentOptimization,
    VisualPrompt,
    VisualOptimization,
    OptimizationPlan
)


class TestOptimizationModels:
    """Test Pydantic models for OptimizationPlan."""

    def test_optimization_item_valid(self):
        """Test OptimizationItem with valid data."""
        item = OptimizationItem(
            original="跟着老爸选DHA｜双A同补真的赢麻了",
            optimized="救命！为什么我没早点发现这个DHA神器！双A同补宝妈必看",
            rationale="原标题缺乏情感钩子和悬念感。新标题使用疑问句式和感叹词增强情绪吸引力，针对CTR提升。",
            targeted_metrics=["ctr", "sort_score2"],
            targeted_weak_features=["title_emotion", "title_pattern"]
        )
        assert item.original == "跟着老爸选DHA｜双A同补真的赢麻了"
        assert "救命" in item.optimized
        assert len(item.targeted_metrics) == 2

    def test_optimization_item_empty_metrics_allowed(self):
        """Test OptimizationItem allows empty targeted_metrics (e.g., for hashtag optimization)."""
        # Empty targeted_metrics is now allowed for general optimizations like hashtags
        item = OptimizationItem(
            original="原标题",
            optimized="新标题",
            rationale="这是一个足够长的优化理由，说明为什么要做这个改变。",
            targeted_metrics=[],  # Empty - now allowed
            targeted_weak_features=["feature1"]
        )
        assert item.targeted_metrics == []

    def test_title_optimization_valid(self):
        """Test TitleOptimization with valid 3 alternatives."""
        alternatives = [
            OptimizationItem(
                original="原标题",
                optimized=f"优化标题{i}",
                rationale="这是一个足够长的优化理由，说明为什么要做这个改变。",
                targeted_metrics=["ctr"],
                targeted_weak_features=["title_emotion"]
            )
            for i in range(3)
        ]
        title_opt = TitleOptimization(
            alternatives=alternatives,
            recommended_index=0,
            selection_rationale="推荐第一个标题因为它具有更强的情感吸引力和悬念感。"
        )
        assert len(title_opt.alternatives) == 3
        assert title_opt.recommended_index == 0

    def test_title_optimization_allows_1_to_3_alternatives(self):
        """Test TitleOptimization allows 1-3 alternatives (flexible for no-optimization case)."""
        # Test with 2 alternatives (should pass now)
        alternatives = [
            OptimizationItem(
                original="原标题",
                optimized=f"优化标题{i}",
                rationale="这是一个足够长的优化理由，说明为什么要做这个改变。",
                targeted_metrics=["ctr"],
                targeted_weak_features=["title_emotion"]
            )
            for i in range(2)
        ]
        title_opt = TitleOptimization(
            alternatives=alternatives,
            recommended_index=0,
            selection_rationale="推荐第一个标题，因为它情感表达更强。"
        )
        assert len(title_opt.alternatives) == 2

        # Test with 0 alternatives (should fail)
        with pytest.raises(ValueError, match="alternatives must contain 1-3 items"):
            TitleOptimization(
                alternatives=[],
                recommended_index=0,
                selection_rationale="无效的空列表，需要至少一个备选项。"
            )

    def test_visual_prompt_valid(self):
        """Test VisualPrompt with valid data."""
        prompt = VisualPrompt(
            image_type="cover",
            prompt_text="一位年轻妈妈温柔地看着宝宝，手中拿着DHA营养品。背景是温馨的家居环境，光线柔和。画面传递出母爱和健康成长的氛围。",
            style_reference="小红书爆款育儿笔记风格，真实生活感",
            key_elements=["年轻妈妈", "宝宝", "DHA产品", "温馨家居"],
            color_scheme="暖色调，柔和的黄色和白色为主，绿色点缀代表健康",
            targeted_metrics=["ctr", "sort_score2"]
        )
        assert prompt.image_type == "cover"
        assert len(prompt.key_elements) == 4

    def test_visual_prompt_invalid_type_fails(self):
        """Test VisualPrompt fails with invalid image_type."""
        with pytest.raises(ValueError, match="image_type must be 'cover' or 'inner_N'"):
            VisualPrompt(
                image_type="invalid_type",  # Should fail
                prompt_text="一位年轻妈妈温柔地看着宝宝，手中拿着DHA营养品。背景是温馨的家居环境，光线柔和。",
                style_reference="小红书爆款育儿笔记风格",
                key_elements=["妈妈", "宝宝"],
                color_scheme="暖色调，柔和的黄色和白色",
                targeted_metrics=["ctr"]
            )

    def test_visual_prompt_inner_type_valid(self):
        """Test VisualPrompt with inner_N image type."""
        prompt = VisualPrompt(
            image_type="inner_1",
            prompt_text="产品成分表特写图，清晰展示DHA和ARA含量。采用简洁的信息图表设计，让用户一目了然。白色背景搭配蓝色数据图表，专业且清晰。",
            style_reference="小红书信息图风格，专业简洁大方",
            key_elements=["成分表", "DHA含量", "ARA含量"],
            color_scheme="白底蓝字，专业清晰，数据可视化风格",
            targeted_metrics=["sort_score2"]
        )
        assert prompt.image_type == "inner_1"

    def test_content_optimization_valid(self):
        """Test ContentOptimization with valid data."""
        content_opt = ContentOptimization(
            opening_hook=OptimizationItem(
                original="养娃以后才知道",
                optimized="姐妹们！养娃之后才发现，有些东西真的不能省！",
                rationale="原开头过于平淡，新开头使用感叹句和直接称呼增强亲切感和紧迫感。",
                targeted_metrics=["ctr", "interaction_rate"],
                targeted_weak_features=["opening_hook", "opening_impact"]
            ),
            ending_cta=OptimizationItem(
                original="麻麻们快抓住发育黄金期",
                optimized="你家宝宝在吃什么DHA？评论区聊聊呀！收藏这篇，选购不踩坑～",
                rationale="原结尾缺乏明确互动引导。新结尾包含开放式问题和收藏提醒，促进评论和收藏。",
                targeted_metrics=["comment_rate", "fav_rate"],
                targeted_weak_features=["ending_cta", "social_proof"]
            ),
            hashtags=OptimizationItem(
                original="#宝宝营养品[话题]# #儿童DHA[话题]#",
                optimized="#老爸测评推荐[话题]# #宝宝DHA怎么选[话题]# #宝妈必看[话题]# #育儿好物分享[话题]# #DHA藻油推荐[话题]#",
                rationale="优化后的话题标签更具针对性，包含用户搜索意图关键词和热门话题。",
                targeted_metrics=["sort_score2", "ctr"],
                targeted_weak_features=["taxonomy2"]
            ),
            body_improvements=[
                "在产品介绍部分增加对比数据，如'含量比XX品牌高30%'",
                "添加更多真实使用场景描述，如宝宝吃的具体反应",
                "增加一段社交证明，如'身边好多宝妈都在用'"
            ]
        )
        assert content_opt.opening_hook.optimized.startswith("姐妹们")
        assert len(content_opt.body_improvements) == 3


class TestOptimizationPlan:
    """Test complete OptimizationPlan model."""

    def test_optimization_plan_valid(self):
        """Test OptimizationPlan with valid complete data."""
        # Create title optimization
        alternatives = [
            OptimizationItem(
                original="跟着老爸选DHA",
                optimized=f"标题备选{i+1}",
                rationale="这是一个足够长的优化理由，针对CTR提升。",
                targeted_metrics=["ctr"],
                targeted_weak_features=["title_emotion"]
            )
            for i in range(3)
        ]
        title_opt = TitleOptimization(
            alternatives=alternatives,
            recommended_index=0,
            selection_rationale="第一个标题情感表达最强，最适合目标用户。"
        )

        # Create content optimization
        content_opt = ContentOptimization(
            opening_hook=OptimizationItem(
                original="养娃以后",
                optimized="姐妹们！养娃之后才发现...",
                rationale="原开头过于平淡，新开头使用感叹句式和直接称呼增强亲切感和吸引力。",
                targeted_metrics=["ctr"],
                targeted_weak_features=["opening_hook"]
            ),
            ending_cta=OptimizationItem(
                original="麻麻们快",
                optimized="评论区聊聊你家宝宝...",
                rationale="原结尾缺乏互动引导，新结尾包含开放式问题促进评论互动。",
                targeted_metrics=["comment_rate"],
                targeted_weak_features=["ending_cta"]
            ),
            hashtags=OptimizationItem(
                original="#话题1#",
                optimized="#老爸测评# #DHA推荐#",
                rationale="优化后的话题标签更具针对性，包含热门搜索关键词提升曝光和精准分类。",
                targeted_metrics=["sort_score2"],
                targeted_weak_features=["taxonomy2"]
            ),
            body_improvements=["改进点1", "改进点2", "改进点3"]
        )

        # Create visual optimization
        visual_opt = VisualOptimization(
            cover_prompt=VisualPrompt(
                image_type="cover",
                prompt_text="温馨母婴场景，年轻妈妈和宝宝在明亮的客厅互动，妈妈手持DHA产品微笑着看着宝宝。光线柔和自然，画面温馨感人，传递健康成长的幸福氛围。背景是简约的现代家居风格。",
                style_reference="小红书爆款育儿笔记风格，真实生活感",
                key_elements=["妈妈", "宝宝", "产品", "温馨"],
                color_scheme="暖色调，柔和的黄色和白色为主，绿色点缀",
                targeted_metrics=["ctr"]
            ),
            inner_image_prompts=[],
            general_visual_guidelines=[
                "保持真实生活感，避免过度修图",
                "使用明亮温馨的色调",
                "产品展示要自然融入场景"
            ]
        )

        # Create complete plan
        plan = OptimizationPlan(
            keyword="老爸测评dha推荐哪几款",
            owned_note_id="67adc607000000002901c2c0",
            title_optimization=title_opt,
            content_optimization=content_opt,
            visual_optimization=visual_opt,
            priority_summary="建议优先执行标题优化和结尾互动召唤优化，这两项对CTR和评论率影响最大。其次优化封面图，提升视觉吸引力。",
            expected_impact={
                "ctr": "预计提升10-15%",
                "comment_rate": "预计提升20-30%",
                "sort_score2": "预计提升5-10%"
            },
            plan_timestamp="2025-11-24T10:30:00Z"
        )

        assert plan.keyword == "老爸测评dha推荐哪几款"
        assert len(plan.title_optimization.alternatives) == 3
        assert len(plan.expected_impact) == 3


class TestCrewInstantiation:
    """Test crew can be instantiated."""

    def test_crew_instantiation(self):
        """Test OptimizationStrategist crew can be created."""
        from xhs_seo_optimizer.crew_optimization import XhsSeoOptimizerCrewOptimization

        crew_instance = XhsSeoOptimizerCrewOptimization()
        assert crew_instance is not None

    def test_crew_agent_exists(self):
        """Test optimization_strategist agent can be created."""
        from xhs_seo_optimizer.crew_optimization import XhsSeoOptimizerCrewOptimization

        crew_instance = XhsSeoOptimizerCrewOptimization()
        agent = crew_instance.optimization_strategist()
        assert agent is not None
        assert agent.role is not None


class TestCrewInputValidation:
    """Test crew input validation."""

    def test_missing_keyword_raises_error(self):
        """Test that missing keyword raises ValueError."""
        from xhs_seo_optimizer.crew_optimization import XhsSeoOptimizerCrewOptimization

        crew_instance = XhsSeoOptimizerCrewOptimization()
        with pytest.raises(ValueError, match="inputs must contain 'keyword'"):
            crew_instance.validate_and_flatten_inputs({})


# ========================================
# E2E Tests with Real Data
# ========================================

@pytest.fixture
def gap_report_path():
    """Path to real gap_report.json."""
    return "outputs/gap_report.json"


@pytest.fixture
def gap_report_data(gap_report_path):
    """Load real gap report data."""
    with open(gap_report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def audit_report_path():
    """Path to real audit_report.json."""
    return "outputs/audit_report.json"


@pytest.fixture
def audit_report_data(audit_report_path):
    """Load real audit report data."""
    with open(audit_report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def keyword_data():
    """Load keyword from docs."""
    with open("docs/keyword.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['keyword']


class TestOptimizationStrategistE2E:
    """End-to-end tests with real data from outputs/ and docs/."""

    def test_validate_input_files_exist(self, gap_report_path, audit_report_path):
        """Test that required input files exist."""
        assert os.path.exists(gap_report_path), f"Gap report not found: {gap_report_path}"
        assert os.path.exists(audit_report_path), f"Audit report not found: {audit_report_path}"
        assert os.path.exists("outputs/success_profile_report.json"), "Success profile not found"
        assert os.path.exists("docs/owned_note.json"), "Owned note not found"
        assert os.path.exists("docs/keyword.json"), "Keyword file not found"

    def test_optimization_context_generation(self, gap_report_data):
        """Test optimization_context generation with real gap report data."""
        from xhs_seo_optimizer.attribution import build_optimization_context

        # Extract data from real gap report
        top_priority_metrics = gap_report_data.get('top_priority_metrics', [])

        # Collect weak and missing features from gaps
        all_weak_features = set()
        all_missing_features = set()
        for gap_list in [gap_report_data.get('significant_gaps', []),
                         gap_report_data.get('marginal_gaps', []),
                         gap_report_data.get('non_significant_gaps', [])]:
            for gap in gap_list:
                all_weak_features.update(gap.get('weak_features', []))
                all_missing_features.update(gap.get('missing_features', []))

        # Build optimization context
        context = build_optimization_context(
            priority_metrics=top_priority_metrics,
            weak_features=list(all_weak_features),
            missing_features=list(all_missing_features)
        )

        # Validate context structure
        assert 'priority_metrics' in context
        assert 'features_to_optimize' in context
        assert 'features_by_content_area' in context

        # Validate priority_metrics
        assert len(context['priority_metrics']) > 0
        for pm in context['priority_metrics']:
            assert 'metric' in pm
            assert 'relevant_features' in pm
            assert 'rationale' in pm

        # Validate features_to_optimize
        for feature, info in context['features_to_optimize'].items():
            assert 'status' in info  # "weak" or "missing"
            assert 'content_area' in info  # NEW: explicit content_area
            assert 'affects_metrics' in info
            assert 'priority_metrics_affected' in info
            assert 'optimization_action' in info

            # Validate content_area is one of expected values
            assert info['content_area'] in ['title', 'opening', 'body', 'ending', 'hashtags', 'visual', 'unknown']

        # Validate features_by_content_area
        assert isinstance(context['features_by_content_area'], dict)
        for area, features in context['features_by_content_area'].items():
            assert isinstance(features, list)
            assert len(features) > 0

    def test_features_by_content_area_structure(self, gap_report_data):
        """Test that features are correctly grouped by content_area."""
        from xhs_seo_optimizer.attribution import build_optimization_context

        # Extract test data
        weak_features = ['title_emotion', 'opening_hook', 'ending_cta']
        missing_features = ['thumbnail_appeal', 'visual_tone']

        context = build_optimization_context(
            priority_metrics=['ctr', 'comment_rate'],
            weak_features=weak_features,
            missing_features=missing_features
        )

        # Check features_by_content_area grouping
        features_by_area = context['features_by_content_area']

        # title_emotion should be in "title" area
        if 'title' in features_by_area:
            assert 'title_emotion' in features_by_area['title']

        # opening_hook should be in "opening" area
        if 'opening' in features_by_area:
            assert 'opening_hook' in features_by_area['opening']

        # ending_cta should be in "ending" area
        if 'ending' in features_by_area:
            assert 'ending_cta' in features_by_area['ending']

        # thumbnail_appeal and visual_tone should be in "visual" area
        if 'visual' in features_by_area:
            assert 'thumbnail_appeal' in features_by_area['visual']
            assert 'visual_tone' in features_by_area['visual']

    @pytest.mark.slow
    @pytest.mark.skipif(
        os.environ.get('SKIP_E2E') == '1',
        reason="Skipping E2E test (set SKIP_E2E=0 to run)"
    )
    def test_end_to_end_with_real_data(self, keyword_data):
        """End-to-end test with real data. SLOW: Calls LLM APIs."""
        from xhs_seo_optimizer.crew_optimization import XhsSeoOptimizerCrewOptimization

        # Create crew instance
        crew_instance = XhsSeoOptimizerCrewOptimization()

        # Execute with real keyword
        result = crew_instance.kickoff(inputs={'keyword': keyword_data})

        # Validate result exists
        assert result is not None

        # Check if result has pydantic attribute (OptimizedNote - Phase 0001)
        if hasattr(result, 'pydantic') and result.pydantic:
            optimized_note = result.pydantic

            # Validate OptimizedNote structure (Phase 0001 final output)
            assert optimized_note.keyword == keyword_data
            assert optimized_note.note_id is not None
            assert optimized_note.original_note_id is not None
            assert optimized_note.title is not None
            assert optimized_note.content is not None
            assert optimized_note.cover_image_url is not None
            assert optimized_note.content_intent is not None
            assert optimized_note.optimization_summary is not None

            # Validate cover_image_source and inner_images_source
            assert optimized_note.cover_image_source in ["generated", "original"]
            assert optimized_note.inner_images_source in ["generated", "original", "mixed"]

        # Check output file was created (Phase 0001: optimized_note.json)
        output_path = "outputs/optimized_note.json"
        assert os.path.exists(output_path), f"Output file not created: {output_path}"

        # Validate output file is valid JSON
        with open(output_path, 'r', encoding='utf-8') as f:
            output_data = json.load(f)

        assert 'keyword' in output_data
        assert output_data['keyword'] == keyword_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
