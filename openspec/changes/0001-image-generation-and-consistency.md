# 图像生成工具与内容一致性优化

**Status:** applied
**Created:** 2025-11-26
**Updated:** 2025-11-26

## Summary

为 OptimizationStrategist 增加实际图像生成能力，并通过引入 ContentIntent（内容创作意图）和 VisualSubjects（视觉主体）机制解决分模块优化后的内容一致性问题。同时增加营销感控制机制，避免优化后内容被平台标记为"软广"。

## Motivation

当前系统存在以下问题：

1. **生图能力缺失**: `generate_visual_prompts` task 只生成 prompt，未实际生成图片
2. **内容一致性差**: 分模块优化（标题、开头、结尾、图片）缺乏统一的"中心思想锚点"，组合后含义不够连贯
3. **视觉主体丢失**: 优化后的图片可能丢失原有的品牌/产品/人物主体
4. **输出格式不完整**: 最终输出是 OptimizationPlan（中间结果），而非可直接使用的 Note 格式
5. **营销感失控**: 若原笔记已被标记为"软广"，优化后可能加重营销感，导致被平台打压

## Proposed Changes

### Phase 0: 意图与主体提取（新增）

在 OwnedNoteAuditor 中增加两个提取任务：

#### 0a. ContentIntent 提取

```python
class ContentIntent(BaseModel):
    """内容创作意图 - 确保优化一致性的锚点"""
    core_theme: str = Field(description="核心主题 (必须, e.g., 'DHA选购攻略')")
    target_persona: str = Field(description="目标人群 (必须, 结合owned_note和keyword确定, e.g., '新手妈妈')")
    key_message: str = Field(description="关键信息/核心卖点 (必须, e.g., '科学配比是关键')")
    unique_angle: Optional[str] = Field(default=None, description="独特角度 (可选, e.g., '老爸测评专业视角')")
    emotional_tone: Optional[str] = Field(default=None, description="情感基调 (可选, e.g., '专业但亲切')")
```

提取逻辑：
- 分析 owned_note.title + owned_note.content 提取主题和关键信息
- 结合 keyword 确定 target_persona（谁会搜索这个关键词）
- 可由用户在 inputs 中覆盖提供

#### 0b. VisualSubjects 提取

```python
class VisualSubjects(BaseModel):
    """视觉主体信息 - 确保生图主体一致性"""
    subject_type: str = Field(description="主体类型: product | person | brand | scene | none")
    subject_description: str = Field(description="主体描述 (e.g., 'DHA鱼油瓶装产品，红色瓶盖')")
    brand_elements: List[str] = Field(default_factory=list, description="品牌元素 (e.g., ['老爸测评logo', '特定配色'])")
    must_preserve: List[str] = Field(description="必须保留的元素 (生图时必须包含)")

    # 新增: 保留原始图片URL作为参考
    original_cover_url: str = Field(description="原始封面图URL (可作为生图参考)")
    original_inner_urls: List[str] = Field(default_factory=list, description="原始内页图URLs")
```

提取逻辑：
- 复用 MultiModalVisionTool 的分析结果
- 识别图片中的核心主体（产品、人物、品牌元素）
- 保留原始图片URL，供生图模型作为参考输入

### Phase 1: 营销感控制机制（新增）

#### 1a. MarketingSentimentTool

```python
class MarketingSentimentTool(BaseTool):
    """营销感检测工具 - 检测文本内容的营销/广告感强度"""
    name: str = "Marketing Sentiment Detector"
    description: str = "检测文本的营销感强度，返回评分和问题点"

    def _run(self, text: str) -> dict:
        """
        Returns:
            {
                "score": float,  # 0-1, 越高营销感越重
                "level": str,  # "low" | "medium" | "high" | "critical"
                "issues": List[str],  # 具体问题点
                "suggestions": List[str]  # 降低营销感的建议
            }
        """
```

检测维度：
- 硬广词汇（"购买"、"下单"、"链接在..."）
- 过度夸张（"最好"、"第一"、"必买"）
- 价格/促销信息密度
- CTA（行动召唤）强度
- 品牌/产品提及频率

#### 1b. 营销感约束注入

在 AuditReport 中增加：
```python
class AuditReport(BaseModel):
    # ... 现有字段 ...

    # 新增
    marketing_level: str = Field(description="当前营销感级别 (来自 note.tag.note_marketing_integrated_level)")
    is_soft_ad: bool = Field(description="是否被标记为软广")
    marketing_sensitivity: str = Field(description="营销敏感度: high(已是软广需降低) | medium | low")
```

当 `is_soft_ad=True` 时：
- 所有文本优化必须通过 MarketingSentimentTool 验证
- 优化后的营销感评分不得高于原始内容
- 在 YAML prompt 中注入强约束

### Phase 2: 现有 Agent 修改

#### 2a. OwnedNoteAuditor 修改

新增 Task：
- `extract_content_intent`: 提取 ContentIntent
- `extract_visual_subjects`: 提取 VisualSubjects

修改 AuditReport 输出：
```python
class AuditReport(BaseModel):
    # ... 现有字段 ...

    # 新增
    content_intent: ContentIntent
    visual_subjects: VisualSubjects
    marketing_level: str
    is_soft_ad: bool
    marketing_sensitivity: str
```

#### 2b. OptimizationStrategist 修改

**修改 Task 1: generate_text_optimizations**
- 注入 ContentIntent 约束（所有优化必须服务于 core_theme, target_persona, key_message）
- 若 is_soft_ad=True，注入营销感控制约束
- 调用 MarketingSentimentTool 验证优化结果

**修改 Task 2: generate_visual_prompts**
- 注入 VisualSubjects.must_preserve 约束
- prompt 中包含 original_cover_url 作为风格参考
- key_elements 必须包含 must_preserve 中的项目

**新增 Task 4: generate_images**
- 调用 ImageGeneratorTool 生成实际图片
- 输入：visual_prompts + original_image_urls (作为参考)
- 输出：GeneratedImages

**新增 Task 5: compile_optimized_note**
- 整合所有优化结果为 Note 格式
- 最终营销感检查（若 is_soft_ad=True）
- 输出：OptimizedNote

### Phase 3: 新增工具

#### 3a. ImageGeneratorTool

```python
class ImageGeneratorTool(BaseTool):
    """图像生成工具 - 使用 Gemini 生成图片"""
    name: str = "Image Generator"
    description: str = "根据prompt生成图片，支持参考图输入"

    def _run(
        self,
        prompt: str,
        reference_image_url: Optional[str] = None,
        must_preserve_elements: Optional[List[str]] = None
    ) -> dict:
        """
        Args:
            prompt: 图片生成描述
            reference_image_url: 参考图URL（保持风格/主体一致）
            must_preserve_elements: 必须包含的元素

        Returns:
            {
                "success": bool,
                "image_url": str,  # Base64 data URL
                "local_path": str,  # 保存的本地路径
                "error": Optional[str]
            }
        """
```

API 调用示例：
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# 构建 prompt（包含必须保留的元素）
full_prompt = f"{prompt}\n\n必须包含以下元素: {', '.join(must_preserve_elements)}"

response = client.chat.completions.create(
    model="google/gemini-3-pro-image-preview",
    messages=[{"role": "user", "content": full_prompt}],
    extra_body={"modalities": ["image", "text"]}
)
```

### Phase 4: 新增数据模型

```python
# models/reports.py 新增

class GeneratedImages(BaseModel):
    """生成的图片结果"""
    cover_image: Optional[Dict] = Field(
        default=None,
        description="封面图 {url, local_path, success, error}"
    )
    inner_images: List[Dict] = Field(
        default_factory=list,
        description="内页图列表"
    )
    generation_timestamp: str = Field(description="生成时间戳")


class OptimizedNote(BaseModel):
    """优化后的完整笔记 - 可直接用于发布"""
    # 基础信息
    note_id: str = Field(description="新笔记ID (原ID + '_optimized')")
    original_note_id: str = Field(description="原始笔记ID")

    # 优化后的内容
    title: str = Field(description="优化后的标题")
    content: str = Field(description="优化后的正文内容")
    cover_image_url: str = Field(description="封面图URL (生成的或原始的)")
    inner_image_urls: List[str] = Field(default_factory=list, description="内页图URLs")

    # 留空字段（发布后获得）
    prediction: Optional[Dict] = Field(default=None, description="留空")
    tag: Optional[Dict] = Field(default=None, description="留空")

    # 溯源与验证
    content_intent: ContentIntent = Field(description="内容创作意图")
    marketing_check: Dict = Field(description="营销感检查结果")
    optimization_summary: str = Field(description="优化摘要")

    # 元数据
    optimized_timestamp: str = Field(description="优化时间戳")
```

## Files to Create

- `src/xhs_seo_optimizer/tools/image_generator.py` - ImageGeneratorTool 实现
- `src/xhs_seo_optimizer/tools/marketing_sentiment.py` - MarketingSentimentTool 实现
- `src/xhs_seo_optimizer/config/agents_owned_note.yaml` - 新增 intent 提取 agent 配置（如需要）
- `src/xhs_seo_optimizer/config/tasks_owned_note.yaml` - 新增 intent 提取 task 配置
- `tests/test_tools/test_image_generator.py` - ImageGeneratorTool 单元测试
- `tests/test_tools/test_marketing_sentiment.py` - MarketingSentimentTool 单元测试

## Files to Modify

- `src/xhs_seo_optimizer/models/reports.py` - 新增 ContentIntent, VisualSubjects, GeneratedImages, OptimizedNote 模型
- `src/xhs_seo_optimizer/models/analysis_results.py` - 若需要扩展 VisionAnalysisResult
- `src/xhs_seo_optimizer/crew_owned_note.py` - 新增 extract_content_intent, extract_visual_subjects tasks
- `src/xhs_seo_optimizer/config/tasks_owned_note.yaml` - 新增 task 配置
- `src/xhs_seo_optimizer/crew_optimization.py` - 新增 generate_images, compile_optimized_note tasks；修改现有 tasks 注入约束
- `src/xhs_seo_optimizer/config/tasks_optimization.yaml` - 修改现有 task prompts，新增 task 配置
- `src/xhs_seo_optimizer/tools/__init__.py` - 导出新工具

## Implementation Plan

### Step 1: 数据模型扩展
1. 在 `models/reports.py` 中新增 ContentIntent, VisualSubjects, GeneratedImages, OptimizedNote 模型
2. 修改 AuditReport 增加 content_intent, visual_subjects, marketing 相关字段

### Step 2: MarketingSentimentTool 实现
1. 创建 `tools/marketing_sentiment.py`
2. 实现营销感检测逻辑（可基于规则 + LLM）
3. 编写单元测试

### Step 3: OwnedNoteAuditor 扩展
1. 新增 `extract_content_intent` task - 提取 ContentIntent
2. 新增 `extract_visual_subjects` task - 提取 VisualSubjects（复用 Vision 分析）
3. 修改 `generate_audit_report` task - 输出完整 AuditReport
4. 更新 YAML 配置

### Step 4: ImageGeneratorTool 实现
1. 创建 `tools/image_generator.py`
2. 实现 OpenRouter Gemini API 调用
3. 实现图片保存逻辑（Base64 → 本地文件）
4. 支持参考图输入
5. 编写单元测试

### Step 5: OptimizationStrategist 扩展
1. 修改 `@before_kickoff` - 加载 ContentIntent, VisualSubjects
2. 修改 Task 1 YAML - 注入 ContentIntent 约束 + 营销感约束
3. 修改 Task 2 YAML - 注入 VisualSubjects.must_preserve 约束
4. 新增 Task 4: `generate_images` - 调用 ImageGeneratorTool
5. 新增 Task 5: `compile_optimized_note` - 格式化输出
6. 若 is_soft_ad=True，在 compile 阶段进行最终营销感验证

### Step 6: 集成测试
1. 端到端测试完整流程
2. 测试营销感控制（软广场景）
3. 测试视觉主体保留
4. 测试内容一致性

## Testing Strategy

### 单元测试
- `test_marketing_sentiment.py`: 测试各种营销感场景检测
- `test_image_generator.py`: Mock API 测试生图逻辑
- `test_models.py`: 测试新增数据模型验证

### 集成测试
- `test_owned_note_auditor_integration.py`: 测试 ContentIntent + VisualSubjects 提取
- `test_optimization_integration.py`: 测试完整优化流程

### 场景测试
- 普通笔记优化（无营销感问题）
- 软广笔记优化（需降低营销感）
- 有明确品牌主体的笔记（需保留主体）
- 无明显主体的场景类笔记

## Risks and Considerations

### 1. API 成本与配额
- Gemini 图像生成有调用成本
- 建议：增加生图开关，允许跳过实际生图只输出 prompt

### 2. 图片质量不可控
- AI 生成的图片可能不符合预期
- 建议：保留原图作为 fallback，生图失败时使用原图

### 3. 营销感检测准确性
- 规则检测可能有误判
- 建议：结合 LLM 判断，提供人工复核选项

### 4. 中心思想提取准确性
- LLM 提取的 ContentIntent 可能不准确
- 建议：允许用户在 inputs 中覆盖提供

### 5. 视觉主体识别
- 复杂图片的主体识别可能不准
- 建议：保留原图 URL 作为强参考

## Alternatives Considered

### Alternative 1: 纯 Prompt 输出（不实际生图）
- 优点：无 API 成本，实现简单
- 缺点：用户仍需手动生图，体验不完整
- 决定：不采用，但提供开关允许跳过生图

### Alternative 2: ContentIntent 由用户必须提供
- 优点：准确性高
- 缺点：增加用户负担
- 决定：不采用，改为自动提取 + 用户可选覆盖

### Alternative 3: 营销感控制作为独立 Agent
- 优点：职责分离清晰
- 缺点：增加系统复杂度和执行时间
- 决定：不采用，改为 Tool 形式，按需调用

## Workflow Diagram (Updated)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Phase 0: CompetitorAnalyst (现有)                                       │
│  Input: target_notes, keyword                                           │
│  Output: success_profile_report.json                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Phase 1: OwnedNoteAuditor (扩展)                                        │
├─────────────────────────────────────────────────────────────────────────┤
│  Task 1: extract_content_features (现有)                                 │
│  Task 2: extract_content_intent (新增)                                   │
│    → ContentIntent {core_theme, target_persona, key_message, ...}       │
│  Task 3: extract_visual_subjects (新增)                                  │
│    → VisualSubjects {subject_type, must_preserve, original_urls, ...}   │
│  Task 4: generate_audit_report (修改)                                    │
│    → AuditReport + ContentIntent + VisualSubjects + marketing_level     │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Phase 2: GapFinder (现有)                                               │
│  Input: audit_report.json, success_profile_report.json                  │
│  Output: gap_report.json                                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Phase 3: OptimizationStrategist (扩展)                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  @before_kickoff:                                                       │
│    - 加载 ContentIntent, VisualSubjects (从 audit_report)               │
│    - 检查 is_soft_ad，设置营销感约束                                      │
│                                                                         │
│  Task 1: generate_text_optimizations (修改)                              │
│    约束: ContentIntent + 营销感控制 (if is_soft_ad)                       │
│    工具: MarketingSentimentTool (验证)                                   │
│                                                                         │
│  Task 2: generate_visual_prompts (修改)                                  │
│    约束: VisualSubjects.must_preserve + original_urls                   │
│                                                                         │
│  Task 3: compile_optimization_plan (现有)                                │
│                                                                         │
│  Task 4: generate_images (新增)                                          │
│    工具: ImageGeneratorTool                                             │
│    输入: visual_prompts + original_urls (参考图)                         │
│    输出: GeneratedImages                                                │
│                                                                         │
│  Task 5: compile_optimized_note (新增)                                   │
│    输入: OptimizationPlan + GeneratedImages + ContentIntent             │
│    验证: 最终营销感检查 (if is_soft_ad)                                   │
│    输出: OptimizedNote (Note格式，可直接发布)                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Final Outputs:                                                         │
│  - outputs/optimization_plan.json (详细优化方案)                         │
│  - outputs/optimized_note.json (可发布的Note格式)                        │
│  - outputs/images/ (生成的图片文件)                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

## Marketing Sensitivity Logic

```python
# 营销敏感度判断逻辑
def determine_marketing_sensitivity(note_tag: NoteTag) -> str:
    """
    判断营销敏感度等级

    Returns:
        "high": 已被标记为软广，必须降低营销感
        "medium": 接近软广边界，需要注意
        "low": 安全，正常优化即可
    """
    level = note_tag.note_marketing_integrated_level

    if level == "软广":
        return "high"
    elif level in ["商品推荐", "种草"]:
        return "medium"
    else:
        return "low"
```

当 `marketing_sensitivity == "high"` 时的约束：
1. 文本优化后必须通过 MarketingSentimentTool 检测
2. 优化后的 marketing_score 必须 ≤ 原始 marketing_score
3. 禁止添加硬广词汇、价格信息、强 CTA
4. 在 compile_optimized_note 阶段进行最终验证
