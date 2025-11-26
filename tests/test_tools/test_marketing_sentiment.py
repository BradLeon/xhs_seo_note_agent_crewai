"""Tests for MarketingSentimentTool - 营销感检测工具测试.

Phase 0001: Tests for marketing sentiment detection.
"""

import json
import os
import pytest
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from xhs_seo_optimizer.tools.marketing_sentiment import (
    MarketingSentimentTool,
    determine_marketing_sensitivity
)


class TestMarketingSentimentToolInstantiation:
    """Test MarketingSentimentTool instantiation."""

    def test_tool_instantiation(self):
        """Test tool can be instantiated."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-api-key'
        }):
            tool = MarketingSentimentTool()
            assert tool is not None
            assert tool.name == "marketing_sentiment_detector"

    def test_tool_has_pattern_lists(self):
        """Test tool has pattern detection lists."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-api-key'
        }):
            tool = MarketingSentimentTool()
            assert hasattr(tool, 'HARD_AD_PATTERNS')
            assert hasattr(tool, 'EXAGGERATION_PATTERNS')
            assert hasattr(tool, 'SOFT_AD_PATTERNS')
            assert hasattr(tool, 'CTA_PATTERNS')


class TestRuleBasedDetection:
    """Test rule-based marketing detection."""

    @pytest.fixture
    def tool(self):
        """Create MarketingSentimentTool instance."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-api-key'
        }):
            return MarketingSentimentTool()

    def test_detect_hard_ad_patterns(self, tool):
        """Test detection of hard advertising patterns."""
        text_with_hard_ad = "这个产品超好用！快点击链接购买吧，现在下单还有优惠券！"

        result = tool._rule_based_detection(text_with_hard_ad)

        assert result['hard_ad_count'] > 0
        assert any('购买' in issue or '下单' in issue or '链接' in issue
                   for issue in result['issues'])

    def test_detect_exaggeration_patterns(self, tool):
        """Test detection of exaggeration patterns."""
        text_with_exaggeration = "这是全网最好的产品，必买！绝绝子！yyds无敌！"

        result = tool._rule_based_detection(text_with_exaggeration)

        assert result['exaggeration_count'] > 0
        assert any('最' in issue or '必买' in issue or 'yyds' in issue
                   for issue in result['issues'])

    def test_detect_soft_ad_patterns(self, tool):
        """Test detection of soft advertising patterns."""
        text_with_soft_ad = "安利给你们这个我用了三年的好物，真心推荐每个人都试试！"

        result = tool._rule_based_detection(text_with_soft_ad)

        assert result['soft_ad_count'] > 0

    def test_detect_cta_patterns(self, tool):
        """Test detection of call-to-action patterns."""
        text_with_cta = "姐妹们快冲！赶紧入手！现在就买！"

        result = tool._rule_based_detection(text_with_cta)

        assert result['cta_count'] > 0

    def test_clean_text_low_score(self, tool):
        """Test that clean text has low marketing score."""
        clean_text = """
        养娃以后才知道，宝宝的营养补充真的很重要。
        我给我家宝宝选的是这款DHA，主要看中了它的成分比较干净，
        没有添加剂，宝宝也爱吃。
        分享给大家参考一下～
        """

        result = tool._rule_based_detection(clean_text)

        # Clean text should have relatively low counts
        total_issues = (result['hard_ad_count'] +
                        result['exaggeration_count'] +
                        result['cta_count'])
        assert total_issues <= 2  # Allow for some false positives

    def test_calculate_rule_score(self, tool):
        """Test rule-based score calculation."""
        # High marketing text
        high_marketing = "必买！最好的产品，点击链接购买，优惠券限时！快冲！"
        result_high = tool._rule_based_detection(high_marketing)

        # Low marketing text
        low_marketing = "分享一下我的使用体验，希望对大家有帮助。"
        result_low = tool._rule_based_detection(low_marketing)

        assert result_high['rule_score'] > result_low['rule_score']


class TestDetermineMarketingSensitivity:
    """Test marketing sensitivity determination function."""

    def test_soft_ad_returns_high(self):
        """Test that 软广 returns high sensitivity."""
        result = determine_marketing_sensitivity("软广")
        assert result == "high"

    def test_product_recommendation_returns_medium(self):
        """Test that product recommendation types return medium."""
        for level in ["商品推荐", "种草", "带货"]:
            result = determine_marketing_sensitivity(level)
            assert result == "medium"

    def test_other_returns_low(self):
        """Test that other types return low sensitivity."""
        for level in ["分享", "日常", "", None, "其他"]:
            result = determine_marketing_sensitivity(level)
            assert result == "low"


class TestMarketingSentimentToolRun:
    """Test the main _run method."""

    @pytest.fixture
    def tool(self):
        """Create MarketingSentimentTool instance."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-api-key'
        }):
            return MarketingSentimentTool()

    def test_run_returns_json_string(self, tool):
        """Test that _run returns valid JSON string."""
        # Mock LLM detection to avoid API calls
        with patch.object(tool, '_llm_based_detection', return_value={
            'llm_score': 0.3,
            'analysis': '营销感适中',
            'issues': [],
            'suggestions': []
        }):
            result = tool._run("这是一段测试文本")

            # Result should be valid JSON
            parsed = json.loads(result)
            assert 'score' in parsed
            assert 'level' in parsed
            assert 'issues' in parsed
            assert 'suggestions' in parsed

    def test_run_with_context(self, tool):
        """Test _run with context parameter."""
        with patch.object(tool, '_llm_based_detection', return_value={
            'llm_score': 0.2,
            'analysis': '分享性质为主',
            'issues': [],
            'suggestions': []
        }):
            result = tool._run(
                text="分享一下我的使用体验",
                context="这是一篇育儿分享笔记"
            )

            parsed = json.loads(result)
            assert parsed['level'] in ['low', 'medium', 'high', 'critical']

    def test_level_thresholds(self, tool):
        """Test that level is correctly determined by score thresholds."""
        test_cases = [
            (0.1, 'low'),
            (0.35, 'medium'),
            (0.55, 'high'),
            (0.85, 'critical')
        ]

        for score, expected_level in test_cases:
            with patch.object(tool, '_rule_based_detection', return_value={
                'rule_score': score,
                'hard_ad_count': 0,
                'exaggeration_count': 0,
                'soft_ad_count': 0,
                'cta_count': 0,
                'issues': []
            }):
                with patch.object(tool, '_llm_based_detection', return_value={
                    'llm_score': score,
                    'analysis': '',
                    'issues': [],
                    'suggestions': []
                }):
                    result = tool._run("测试文本")
                    parsed = json.loads(result)
                    assert parsed['level'] == expected_level, \
                        f"Score {score} should give level {expected_level}, got {parsed['level']}"


class TestMarketingSentimentToolSuggestions:
    """Test suggestion generation."""

    @pytest.fixture
    def tool(self):
        """Create MarketingSentimentTool instance."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-api-key'
        }):
            return MarketingSentimentTool()

    def test_high_score_generates_suggestions(self, tool):
        """Test that high marketing score generates suggestions."""
        high_marketing_text = "必买！这是最好的产品！快点击链接购买！优惠券限时！"

        with patch.object(tool, '_llm_based_detection', return_value={
            'llm_score': 0.8,
            'analysis': '营销感很强',
            'issues': ['过多硬广词汇', '夸张用语'],
            'suggestions': ['减少推销语气', '使用更客观的描述']
        }):
            result = tool._run(high_marketing_text)
            parsed = json.loads(result)

            assert len(parsed['suggestions']) > 0

    def test_low_score_minimal_suggestions(self, tool):
        """Test that low marketing score has minimal suggestions."""
        clean_text = "分享一下我的使用体验，这款产品我用了三个月，感觉还不错。"

        with patch.object(tool, '_llm_based_detection', return_value={
            'llm_score': 0.1,
            'analysis': '内容真实自然',
            'issues': [],
            'suggestions': []
        }):
            result = tool._run(clean_text)
            parsed = json.loads(result)

            # Low score should have few or no suggestions
            assert len(parsed['suggestions']) <= 1


class TestMarketingSentimentToolIntegration:
    """Integration tests (require API key, marked as slow)."""

    @pytest.mark.slow
    @pytest.mark.skipif(
        not os.environ.get('OPENROUTER_API_KEY'),
        reason="OPENROUTER_API_KEY not set"
    )
    def test_real_api_call_high_marketing(self):
        """Test real API call with high marketing text."""
        tool = MarketingSentimentTool()

        high_marketing_text = """
        姐妹们！这个产品简直绝绝子！必买！！！
        全网最低价，点击链接立即购买，限时优惠券！
        不买后悔一辈子！快冲！！！
        """

        result = tool._run(high_marketing_text)
        parsed = json.loads(result)

        assert parsed['score'] > 0.5
        assert parsed['level'] in ['high', 'critical']

    @pytest.mark.slow
    @pytest.mark.skipif(
        not os.environ.get('OPENROUTER_API_KEY'),
        reason="OPENROUTER_API_KEY not set"
    )
    def test_real_api_call_low_marketing(self):
        """Test real API call with low marketing text."""
        tool = MarketingSentimentTool()

        clean_text = """
        养娃以后才知道，宝宝的营养补充确实需要花心思。
        我给宝宝选了这款DHA，主要是因为成分比较干净。
        用了两个月，宝宝挺爱吃的，分享给大家参考。
        """

        result = tool._run(clean_text)
        parsed = json.loads(result)

        assert parsed['score'] < 0.5
        assert parsed['level'] in ['low', 'medium']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
