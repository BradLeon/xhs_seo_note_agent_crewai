"""Marketing Sentiment Tool - 营销感检测工具.

Detects marketing/advertising sentiment intensity in text content.
Used to ensure optimized content doesn't increase marketing feel,
especially for notes already marked as "软广" (soft advertising).
"""

import os
import json
import re
from typing import Any, Dict, List, Optional
from openai import OpenAI
from crewai.tools import BaseTool
from pydantic import Field, ConfigDict, BaseModel


class MarketingSentimentInput(BaseModel):
    """Input schema for MarketingSentimentTool."""

    text: str = Field(
        description="待检测的文本内容 (标题 + 正文)"
    )
    context: Optional[str] = Field(
        default=None,
        description="可选的上下文信息，帮助更准确判断营销意图"
    )


class MarketingSentimentTool(BaseTool):
    """营销感检测工具 (Marketing Sentiment Detector).

    Detects marketing/advertising sentiment intensity in XHS note content.
    Combines rule-based detection with LLM analysis for accurate assessment.

    Output includes:
    - score: 0-1 float (higher = more marketing-heavy)
    - level: low | medium | high | critical
    - issues: specific marketing issues found
    - suggestions: recommendations to reduce marketing feel
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "marketing_sentiment_detector"
    description: str = (
        "检测文本内容的营销感强度。"
        "返回营销感评分(0-1)、级别(low/medium/high/critical)、"
        "具体问题点和降低营销感的建议。"
        "使用场景：检查优化后的内容是否会被平台标记为软广。"
    )
    args_schema: type[BaseModel] = MarketingSentimentInput

    # OpenRouter configuration for LLM analysis
    api_key: str = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    model: str = Field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_TEXT_MODEL",
            "deepseek/deepseek-chat"
        )
    )

    # Rule-based detection patterns
    HARD_AD_PATTERNS: List[str] = [
        r'购买|下单|立即买|马上买|点击购买',
        r'链接在|戳链接|点链接|商品链接',
        r'优惠券|折扣码|满减|限时优惠|秒杀',
        r'官方旗舰|官方店铺|官方授权',
        r'原价\d+|现价\d+|只要\d+元|仅需\d+',
    ]

    EXAGGERATION_PATTERNS: List[str] = [
        r'最好的|第一名|NO\.?1|冠军|销量第一',
        r'必买|必入|必囤|人手一个|不买后悔',
        r'绝绝子|yyds|天花板|断货王',
        r'100%|完美|无敌|最强|最佳|最优',
    ]

    SOFT_AD_PATTERNS: List[str] = [
        r'推荐给|安利给|种草|拔草',
        r'回购|复购|无限回购|已经买了\d+',
        r'品牌方|合作|赠送|寄来的',
        r'#ad|#广告|#合作',
    ]

    CTA_PATTERNS: List[str] = [
        r'快去|赶紧|抓紧|冲冲冲|买它',
        r'姐妹们快|宝子们快|家人们快',
        r'不要犹豫|别犹豫|直接冲',
    ]

    def _run(self, text: str, context: Optional[str] = None) -> str:
        """Detect marketing sentiment in text content.

        Args:
            text: Text content to analyze (title + body)
            context: Optional context for better assessment

        Returns:
            JSON string with detection results
        """
        # Step 1: Rule-based detection
        rule_results = self._rule_based_detection(text)

        # Step 2: LLM-based analysis for nuanced detection
        llm_results = self._llm_based_detection(text, context, rule_results)

        # Step 3: Combine results and calculate final score
        final_results = self._combine_results(rule_results, llm_results)

        return json.dumps(final_results, ensure_ascii=False, indent=2)

    def _rule_based_detection(self, text: str) -> Dict[str, Any]:
        """Rule-based marketing pattern detection.

        Returns:
            Dict with pattern matches and rule-based score
        """
        issues = []
        pattern_counts = {
            'hard_ad': 0,
            'exaggeration': 0,
            'soft_ad': 0,
            'cta': 0
        }

        # Check hard advertising patterns
        for pattern in self.HARD_AD_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                pattern_counts['hard_ad'] += len(matches)
                issues.append(f"硬广词汇: {', '.join(set(matches))}")

        # Check exaggeration patterns
        for pattern in self.EXAGGERATION_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                pattern_counts['exaggeration'] += len(matches)
                issues.append(f"夸张用语: {', '.join(set(matches))}")

        # Check soft advertising patterns
        for pattern in self.SOFT_AD_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                pattern_counts['soft_ad'] += len(matches)
                issues.append(f"软广信号: {', '.join(set(matches))}")

        # Check CTA patterns
        for pattern in self.CTA_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                pattern_counts['cta'] += len(matches)
                issues.append(f"强CTA: {', '.join(set(matches))}")

        # Calculate rule-based score (0-1)
        # Weights: hard_ad=0.4, exaggeration=0.25, soft_ad=0.2, cta=0.15
        max_counts = {'hard_ad': 3, 'exaggeration': 4, 'soft_ad': 3, 'cta': 3}
        weights = {'hard_ad': 0.4, 'exaggeration': 0.25, 'soft_ad': 0.2, 'cta': 0.15}

        score = 0.0
        for key, weight in weights.items():
            normalized = min(pattern_counts[key] / max_counts[key], 1.0)
            score += normalized * weight

        return {
            'score': min(score, 1.0),
            'pattern_counts': pattern_counts,
            'issues': issues
        }

    def _llm_based_detection(
        self,
        text: str,
        context: Optional[str],
        rule_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """LLM-based nuanced marketing detection.

        Uses LLM to detect subtle marketing signals that rules might miss.
        """
        if not self.api_key:
            # Fallback if no API key
            return {'score': 0.0, 'analysis': '无API密钥，跳过LLM分析', 'suggestions': []}

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )

        prompt = f"""请分析以下小红书笔记内容的营销感强度。

## 内容
{text[:2000]}  # Limit text length

## 规则检测已发现的问题
{json.dumps(rule_results['issues'], ensure_ascii=False) if rule_results['issues'] else '无'}

## 分析要求
1. 评估整体营销感（0-1分，0=纯分享，1=硬广告）
2. 识别隐性营销信号（规则可能漏掉的）
3. 提供降低营销感的具体建议

## 输出格式（严格JSON）
{{
  "score": 0.X,
  "hidden_signals": ["信号1", "信号2"],
  "tone_analysis": "整体语气分析",
  "suggestions": ["建议1", "建议2", "建议3"]
}}
"""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是小红书内容审核专家，擅长识别软广和营销内容。请严格按JSON格式输出。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            result_text = response.choices[0].message.content.strip()

            # Parse JSON from response
            # Handle potential markdown code blocks
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0].strip()
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0].strip()

            result = json.loads(result_text)
            return {
                'score': float(result.get('score', 0.0)),
                'hidden_signals': result.get('hidden_signals', []),
                'tone_analysis': result.get('tone_analysis', ''),
                'suggestions': result.get('suggestions', [])
            }

        except Exception as e:
            print(f"⚠️ LLM marketing analysis failed: {e}")
            return {
                'score': 0.0,
                'hidden_signals': [],
                'tone_analysis': f'LLM分析失败: {str(e)}',
                'suggestions': []
            }

    def _combine_results(
        self,
        rule_results: Dict[str, Any],
        llm_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine rule-based and LLM results into final assessment.

        Returns:
            Final marketing sentiment assessment
        """
        # Weighted combination of scores
        # Rule-based is more reliable for hard patterns
        # LLM catches nuanced signals
        rule_weight = 0.6
        llm_weight = 0.4

        combined_score = (
            rule_results['score'] * rule_weight +
            llm_results['score'] * llm_weight
        )

        # Determine level based on score
        if combined_score >= 0.7:
            level = 'critical'
        elif combined_score >= 0.5:
            level = 'high'
        elif combined_score >= 0.3:
            level = 'medium'
        else:
            level = 'low'

        # Combine issues
        all_issues = rule_results['issues'].copy()
        if llm_results.get('hidden_signals'):
            all_issues.extend([f"隐性信号: {s}" for s in llm_results['hidden_signals']])

        # Combine suggestions
        suggestions = llm_results.get('suggestions', [])
        if not suggestions:
            # Default suggestions based on level
            if level in ['critical', 'high']:
                suggestions = [
                    '移除直接的购买引导词汇',
                    '减少夸张性描述，使用客观表达',
                    '降低CTA强度，改用软性互动',
                    '增加真实使用体验描述',
                    '避免提及价格、优惠信息'
                ]
            elif level == 'medium':
                suggestions = [
                    '适当减少推荐性词汇',
                    '增加个人真实感受描述',
                    '使用更自然的表达方式'
                ]

        return {
            'score': round(combined_score, 3),
            'level': level,
            'issues': all_issues,
            'suggestions': suggestions,
            'rule_score': round(rule_results['score'], 3),
            'llm_score': round(llm_results['score'], 3),
            'tone_analysis': llm_results.get('tone_analysis', '')
        }


def determine_marketing_sensitivity(marketing_level: str) -> str:
    """Determine marketing sensitivity based on platform tag.

    Args:
        marketing_level: Value from note.tag.note_marketing_integrated_level

    Returns:
        "high": Already marked as soft ad, must reduce marketing feel
        "medium": Close to soft ad boundary, need attention
        "low": Safe, normal optimization
    """
    if marketing_level == "软广":
        return "high"
    elif marketing_level in ["商品推荐", "种草", "带货"]:
        return "medium"
    else:
        return "low"
