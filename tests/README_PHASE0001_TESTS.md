# Phase 0001 测试说明

本文档说明如何运行 Phase 0001（图像生成与内容一致性）的测试。

## 测试文件结构

```
tests/
├── test_tools/
│   ├── test_image_generator.py      # ImageGeneratorTool 单元测试
│   └── test_marketing_sentiment.py  # MarketingSentimentTool 单元测试
├── test_phase0001_models.py         # Phase 0001 数据模型测试
├── test_phase0001_e2e.py            # 端到端集成测试
└── README_PHASE0001_TESTS.md        # 本文档
```

## 测试数据

测试需要以下数据文件：

- `docs/keyword.json` - 关键词配置
- `docs/owned_note.json` - 待优化的笔记数据
- `outputs/audit_report.json` - OwnedNoteAuditor 输出（集成测试需要）
- `outputs/gap_report.json` - GapFinder 输出（集成测试需要）
- `outputs/success_profile_report.json` - CompetitorAnalyst 输出（集成测试需要）

## 运行测试

### 1. 运行所有 Phase 0001 单元测试（不需要 API）

```bash
# 运行工具测试
pytest tests/test_tools/test_image_generator.py -v
pytest tests/test_tools/test_marketing_sentiment.py -v

# 运行模型测试
pytest tests/test_phase0001_models.py -v

# 运行所有 Phase 0001 测试（排除需要 API 的慢测试）
pytest tests/test_phase0001_*.py tests/test_tools/test_image_generator.py tests/test_tools/test_marketing_sentiment.py -v -m "not slow"
```

### 2. 运行集成测试（需要 API Key）

```bash
# 设置环境变量
export OPENROUTER_API_KEY="your-api-key"

# 运行 E2E 测试
pytest tests/test_phase0001_e2e.py -v

# 运行所有测试（包括慢测试）
pytest tests/ -v
```

### 3. 运行特定测试类

```bash
# 只测试 ContentIntent
pytest tests/test_phase0001_models.py::TestContentIntent -v

# 只测试营销感检测
pytest tests/test_tools/test_marketing_sentiment.py::TestRuleBasedDetection -v

# 只测试图像生成
pytest tests/test_tools/test_image_generator.py::TestImageGeneratorToolAPICall -v
```

## 测试标记

- `@pytest.mark.slow` - 需要 API 调用的慢测试
- `@pytest.mark.skipif(not os.environ.get('OPENROUTER_API_KEY'))` - 需要 API Key 的测试

## 测试覆盖范围

### 1. ImageGeneratorTool 测试

| 测试 | 描述 |
|-----|------|
| test_tool_instantiation | 工具实例化 |
| test_build_full_prompt_* | Prompt 构建逻辑 |
| test_generate_image_* | 图像生成 API 调用 |
| test_save_image_* | 图像保存逻辑 |
| test_run_* | 主方法测试 |

### 2. MarketingSentimentTool 测试

| 测试 | 描述 |
|-----|------|
| test_detect_hard_ad_patterns | 硬广检测 |
| test_detect_exaggeration_patterns | 夸张用语检测 |
| test_detect_soft_ad_patterns | 软广检测 |
| test_detect_cta_patterns | CTA 检测 |
| test_level_thresholds | 评分阈值 |

### 3. 数据模型测试

| 模型 | 测试内容 |
|-----|---------|
| ContentIntent | 字段验证、长度限制、序列化 |
| VisualSubjects | 主体类型、URL 保留 |
| GeneratedImage | 成功/失败状态 |
| GeneratedImages | 封面图+内页图组合 |
| MarketingCheck | 通过/失败检查 |
| OptimizedNote | 完整结构、序列化 |

### 4. 集成测试

| 测试 | 描述 |
|-----|------|
| TestContentIntentExtraction | ContentIntent 提取逻辑 |
| TestVisualSubjectsExtraction | VisualSubjects 提取逻辑 |
| TestMarketingSensitivity | 营销敏感度判断 |
| TestOwnedNoteAuditorPhase0001 | Auditor 新任务测试 |
| TestOptimizationStrategistPhase0001 | Strategist 新任务测试 |

## 预期测试结果

### 单元测试（无 API）
- 所有测试应通过
- 无需网络连接

### 集成测试（需要 API）
- 需要有效的 `OPENROUTER_API_KEY`
- 需要已生成的 `outputs/*.json` 文件
- 图像生成测试可能因 API 配额限制而跳过

## 常见问题

### Q: 测试提示缺少文件怎么办？

确保已运行完整的 pipeline：
```bash
# 1. 运行 CompetitorAnalyst
python -c "from xhs_seo_optimizer.crew_competitor import XhsSeoOptimizerCrewCompetitor; crew = XhsSeoOptimizerCrewCompetitor(); crew.kickoff(inputs={'keyword': '老爸测评dha推荐哪几款'})"

# 2. 运行 OwnedNoteAuditor
python -c "from xhs_seo_optimizer.crew_owned_note import XhsSeoOptimizerCrewOwnedNote; crew = XhsSeoOptimizerCrewOwnedNote(); crew.kickoff(inputs={'keyword': '老爸测评dha推荐哪几款'})"

# 3. 运行 GapFinder
python -c "from xhs_seo_optimizer.crew_gap_finder import XhsSeoOptimizerCrewGapFinder; crew = XhsSeoOptimizerCrewGapFinder(); crew.kickoff(inputs={'keyword': '老爸测评dha推荐哪几款'})"
```

### Q: 图像生成测试失败怎么办？

检查：
1. `OPENROUTER_API_KEY` 是否有效
2. API 配额是否充足
3. 网络连接是否正常

### Q: 营销感检测结果不准确怎么办？

营销感检测结合规则和 LLM，准确率受：
1. 规则模式覆盖范围
2. LLM 模型能力
3. 上下文信息质量

可调整 `MarketingSentimentTool` 中的模式和权重。
