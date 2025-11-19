# CrewAI Complex Input & LLM Hallucination 修复经验总结

## 问题背景

在构建小红书SEO优化Multi-Agent系统时，遇到了三个核心问题：

1. **复杂Pydantic对象无法传递给Crew** - ComplexInput包含嵌套的List[Note]对象，YAML和工具无法解析
2. **LLM Agent幻觉问题** - Agent尝试传递大型数据结构时，会创造假数据而非使用真实数据
3. **任务复杂度超限** - 单个任务过于复杂（70+行描述，40+工具调用），导致Agent中途停止

## 核心解决方案

### 1. ComplexInput包装器 + model_dump()模式

**问题**: CrewAI的`crew.kickoff(inputs=...)`只接受基本类型的dict，无法处理嵌套Pydantic对象

**解决方案**:

创建`ComplexInput`包装类 (`src/xhs_seo_optimizer/models/note.py`):

```python
class ComplexInput(BaseModel):
    """ComplexInput wrapper for crew kickoff inputs."""
    target_notes: List[Note] = Field(description="竞品笔记列表")
    keyword: str = Field(description="目标关键词")
    owned_note: Optional[Note] = Field(default=None)
```

使用模式:
```python
# 创建ComplexInput对象
complex_input = ComplexInput(
    target_notes=[note1, note2, note3],
    keyword="AI技术"
)

# 使用model_dump()序列化后传递
crew.kickoff(inputs=complex_input.model_dump())
```

在`@before_kickoff`中处理 (`src/xhs_seo_optimizer/crew_simple.py`):
```python
@before_kickoff
def validate_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
    # 支持ComplexInput对象和dict两种模式
    if isinstance(inputs, ComplexInput):
        inputs = inputs.model_dump()

    # 序列化List[Note]为List[dict]
    target_notes = inputs.pop("target_notes")
    target_notes_data = [
        note.model_dump() if not isinstance(note, dict) else note
        for note in target_notes
    ]

    # 存储到shared_context
    from xhs_seo_optimizer.shared_context import shared_context
    shared_context.set("target_notes_data", target_notes_data)

    return inputs
```

**关键经验**:
- ✅ 使用Pydantic的`model_dump()`进行序列化（官方推荐）
- ✅ 在`@before_kickoff`中处理复杂对象，转换为基本类型
- ✅ 支持两种模式（ComplexInput对象 + 普通dict）以保持向后兼容

### 2. SharedContext单例 + Smart Mode工具模式

**问题**: LLM Agent在尝试传递大型数据结构（如List[dict]包含4条笔记）作为工具参数时，会幻觉出假数据

**真实案例**:
```
# Agent想象的假数据
Tool Input: {
  "note_id": "6572104408411101190",  # ❌ 假ID
  "note_metadata": {
    "title": "老爸测评推荐的DHA产品",  # ❌ 想象的内容
    ...
  }
}

# 真实数据应该是
Real Note ID: "5e96b4f700000000010040e6"  # ✅ 来自target_notes.json
```

**解决方案**:

#### 2.1 创建SharedContext单例 (`src/xhs_seo_optimizer/shared_context.py`)

```python
class SharedContext:
    """Thread-safe singleton for sharing data between crew hooks and tools."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._data = {}
        return cls._instance

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

# 全局单例
shared_context = SharedContext()
```

#### 2.2 实现Smart Mode工具模式

**核心思想**: 工具只需要接收简单的ID（如note_id），然后自动从shared_context获取完整数据

修改工具的Input Schema (`src/xhs_seo_optimizer/tools/nlp_analysis.py`):

```python
class NLPAnalysisInput(BaseModel):
    """Input schema with Smart Mode support."""

    note_id: Optional[str] = Field(
        default=None,
        description=(
            "笔记ID - 工具会自动从系统中获取该笔记的metadata并分析。"
            "推荐使用此模式，只需传入note_id。"
        )
    )
    note_metadata: Optional[dict] = Field(
        default=None,
        description=(
            "可选：直接传入序列化的NoteMetaData dict。"
            "如果不传，工具会根据note_id自动从系统获取。"
        )
    )
```

在工具的`_run`方法中实现Smart Mode:

```python
def _run(self, note_id: Optional[str] = None, note_metadata: Optional[dict] = None) -> str:
    # Smart mode: 从shared_context自动获取
    if not note_metadata and note_id:
        logger.info(f"Smart mode: Fetching metadata for note_id={note_id}")

        from xhs_seo_optimizer.shared_context import shared_context
        notes = shared_context.get("target_notes_data", [])

        # 查找对应note_id的笔记
        for note in notes:
            if note.get('note_id') == note_id:
                note_metadata = note.get('meta_data')
                logger.info(f"✓ Found note metadata for {note_id}")
                break

        if not note_metadata:
            raise ValueError(f"Note with ID '{note_id}' not found")

    # 继续分析...
```

在任务描述中引导Agent使用Smart Mode (`src/xhs_seo_optimizer/config/tasks_simple.yaml`):

```yaml
extract_features:
  description: |
    **执行步骤** (对每条笔记):
    1. 调用 nlp_text_analysis(note_id="笔记ID")
       - ⚠️ 重要: 只传 note_id 参数，工具会自动获取该笔记的文本内容并分析

    2. 调用 multimodal_vision_analysis(note_id="笔记ID")
       - ⚠️ 重要: 只传 note_id 参数，工具会自动获取该笔记的图片并分析

    **禁止事项**:
    - ❌ 不要创造或想象笔记ID
    - ❌ 不要创造或想象笔记内容
    - ❌ 不要传递 note_metadata 参数（工具会自动获取）
    - ✅ 只使用真实的笔记ID（来自上游任务）
```

**关键经验**:
- ✅ LLM擅长传递简单类型（string ID），不擅长传递复杂嵌套结构
- ✅ 工具内部从shared_context获取数据，比让Agent传递数据更可靠
- ✅ 明确的任务描述（包含禁止事项）可以有效引导Agent行为
- ✅ 支持两种模式（Smart + Legacy）保持向后兼容

### 3. 任务拆分策略

**问题**: 单一任务描述过长（70+行）且需要40+次工具调用，超出Agent的iteration限制，导致中途停止

**原始单任务结构**:
```yaml
analyze_competitors:
  description: |
    步骤1: 调用data_aggregator()聚合统计
    步骤2: 对每条笔记调用nlp_text_analysis + multimodal_vision_analysis
    步骤3: 对10个指标逐一分析
    步骤4: 生成最终报告
    步骤5: 执行跨指标总结
    # 总共需要约40+次工具调用
```

**解决方案**: 拆分为4个顺序任务，使用`context`参数链接 (`src/xhs_seo_optimizer/config/tasks_simple.yaml`)

```yaml
# Task 1: 聚合统计（1次工具调用）
aggregate_statistics:
  description: |
    调用 data_aggregator() 工具（无需参数），自动从共享上下文获取笔记数据
  expected_output: "AggregatedMetrics JSON"
  agent: competitor_analyst

# Task 2: 提取特征（8次工具调用：4条笔记 × 2个工具）
extract_features:
  description: |
    对每条笔记:
    1. nlp_text_analysis(note_id="xxx")
    2. multimodal_vision_analysis(note_id="xxx")
  expected_output: "Features matrix JSON"
  agent: competitor_analyst

# Task 3: 指标分析（10次LLM调用：10个指标）
analyze_metrics:
  description: |
    对10个指标逐一分析，识别显著特征
  expected_output: "List of 10 MetricSuccessProfile"
  agent: competitor_analyst

# Task 4: 生成报告（1次总结）
generate_report:
  description: |
    生成最终报告，包含关键成功因素和爆款公式
  expected_output: "Complete SuccessProfileReport"
  agent: competitor_analyst
  output_file: "outputs/success_profile_report.json"
```

在Crew中使用`context`参数链接任务 (`src/xhs_seo_optimizer/crew_simple.py`):

```python
@task
def extract_features_task(self) -> Task:
    return Task(
        config=self.tasks_config['extract_features'],
        agent=self.competitor_analyst(),
        context=[self.aggregate_statistics_task()]  # 依赖Task 1
    )

@task
def analyze_metrics_task(self) -> Task:
    return Task(
        config=self.tasks_config['analyze_metrics'],
        agent=self.competitor_analyst(),
        context=[
            self.aggregate_statistics_task(),  # 依赖Task 1
            self.extract_features_task()       # 依赖Task 2
        ]
    )

@task
def generate_report_task(self) -> Task:
    return Task(
        config=self.tasks_config['generate_report'],
        agent=self.competitor_analyst(),
        context=[
            self.aggregate_statistics_task(),  # 依赖Task 1
            self.analyze_metrics_task()        # 依赖Task 3
        ],
        output_pydantic=SuccessProfileReport
    )
```

同时增加Agent的iteration限制 (`src/xhs_seo_optimizer/config/agents.yaml`):

```yaml
competitor_analyst:
  role: "竞品分析师 (Competitor Analyst)"
  goal: "分析target_notes并总结创作公式"
  backstory: |
    你是一位数据驱动的内容策略专家...
  verbose: true
  max_iter: 50  # 增加迭代限制（默认15）
  allow_delegation: false
```

**关键经验**:
- ✅ 单个任务的工具调用次数应控制在10次以内
- ✅ 使用`context`参数传递上游任务输出，Agent可以直接访问
- ✅ 任务拆分应基于逻辑边界（聚合→提取→分析→总结）
- ✅ 增加`max_iter`作为安全边界，但不应依赖它来处理复杂任务

### 4. 调试增强 - 详细日志记录

**问题**: 工具执行失败时（如JSON解析错误），难以定位LLM返回的原始内容

**解决方案**: 在关键工具中添加详细的request/response日志 (`src/xhs_seo_optimizer/tools/multimodal_vision.py`)

```python
def _analyze_with_vision_model(self, note_meta_data, image_urls):
    prompt = self._build_vision_prompt(note_meta_data)

    # ========== REQUEST LOGGING ==========
    logger.info("=" * 80)
    logger.info("LLM REQUEST - MultiModal Vision Analysis")
    logger.info("=" * 80)
    logger.info(f"Model: {self.model}")
    logger.info(f"Note ID: {note_meta_data.note_id}")
    logger.info(f"Image URLs: {image_urls}")
    logger.info(f"Prompt (first 500 chars):\n{prompt[:500]}...")
    logger.info("=" * 80)

    # API调用
    response = client.chat.completions.create(...)
    content = response.choices[0].message.content

    # ========== RESPONSE LOGGING ==========
    logger.info("=" * 80)
    logger.info("LLM RESPONSE - MultiModal Vision Analysis")
    logger.info("=" * 80)
    logger.info(f"Response length: {len(content)} characters")
    logger.info(f"Full response content:\n{content}")
    logger.info("=" * 80)

    return self._parse_vision_response(content)

def _parse_vision_response(self, content: str):
    # ========== JSON EXTRACTION LOGGING ==========
    logger.info("=" * 80)
    logger.info("JSON EXTRACTION")
    logger.info("=" * 80)

    if "```json" in content:
        json_start = content.find("```json") + 7
        json_end = content.find("```", json_start)
        json_str = content[json_start:json_end].strip()
        logger.info("Extraction method: ```json block")

    logger.info(f"Extracted JSON length: {len(json_str)} characters")
    logger.info(f"Extracted JSON content:\n{json_str}")
    logger.info("=" * 80)

    # 解析JSON
    data = json.loads(json_str)
    logger.info("✓ JSON parsing successful!")
```

**关键经验**:
- ✅ 记录完整的LLM request prompt（帮助调试prompt工程问题）
- ✅ 记录完整的LLM response content（帮助定位JSON格式问题）
- ✅ 记录JSON提取过程（提取方法、长度、内容）
- ✅ 使用分隔符（`"=" * 80`）使日志易于阅读

## 技术栈参考

- **CrewAI**: 0.80+ (官方文档强调使用`model_dump()`而非`dict()`)
- **Pydantic**: v2.x (使用`model_dump()`进行序列化)
- **OpenRouter**: 使用免费的Gemini 2.5 Flash Lite模型
- **Python**: 3.10+

## 最佳实践总结

### ✅ DO (推荐做法)

1. **数据序列化**: 使用Pydantic的`model_dump()`而非`dict()`
2. **复杂输入**: 创建包装类（ComplexInput）在`@before_kickoff`中处理
3. **大型数据**: 存储在shared_context，工具自动获取（Smart Mode）
4. **任务设计**: 单任务工具调用数≤10次，使用`context`链接多任务
5. **工具设计**: 优先接收简单类型（ID），支持两种模式（Smart + Legacy）
6. **调试**: 在关键工具中添加详细的request/response日志
7. **任务描述**: 明确列出禁止事项，引导Agent正确行为

### ❌ DON'T (避免做法)

1. **不要**: 直接传递复杂嵌套Pydantic对象给`crew.kickoff()`
2. **不要**: 让Agent传递大型数据结构（List[dict]）作为工具参数
3. **不要**: 创建需要40+次工具调用的单一任务
4. **不要**: 假设Agent能够记住或推理出复杂数据结构
5. **不要**: 忽略LLM的幻觉倾向（总是验证工具输入的真实性）
6. **不要**: 依赖增加`max_iter`来解决任务复杂度问题（治标不治本）

## 性能优化建议

1. **并行化**: 未来可以考虑并行调用NLP和Vision工具（当前是串行）
2. **缓存**: 对相同note_id的重复调用可以加缓存层
3. **批处理**: Vision API支持多图片，已优化（1次调用分析5张图）
4. **成本控制**:
   - Gemini 2.5 Flash Lite: 免费（文本分析）
   - Gemini 2.5 Flash Lite: ~$0.002/图片（视觉分析）
   - 每条笔记成本: ~$0.01 (5张图)

## 相关文件清单

### 新增文件
- `src/xhs_seo_optimizer/shared_context.py` - SharedContext单例实现

### 修改文件
- `src/xhs_seo_optimizer/models/note.py` - 添加ComplexInput类
- `src/xhs_seo_optimizer/crew_simple.py` - 修改@before_kickoff，拆分4个任务
- `src/xhs_seo_optimizer/config/tasks_simple.yaml` - 从1个任务拆分为4个
- `src/xhs_seo_optimizer/config/agents.yaml` - 增加max_iter=50
- `src/xhs_seo_optimizer/tools/data_aggregator.py` - 实现Smart Mode（自动从shared_context获取）
- `src/xhs_seo_optimizer/tools/nlp_analysis.py` - 实现Smart Mode（note_id自动获取）
- `src/xhs_seo_optimizer/tools/multimodal_vision.py` - 实现Smart Mode + 详细日志
- `tests/test_competitor_agent_refactored.py` - 使用ComplexInput包装器

## 测试验证

运行以下命令验证修复：

```bash
python tests/test_competitor_agent_refactored.py
```

**预期输出**:
- ✅ Task 1: aggregate_statistics 成功（工具自动从shared_context获取数据）
- ✅ Task 2: extract_features 成功（Smart Mode使用真实note_id）
- ✅ Task 3: analyze_metrics 成功（生成10个MetricSuccessProfile）
- ✅ Task 4: generate_report 成功（生成完整SuccessProfileReport）
- ✅ 最终输出: `outputs/success_profile_report.json`

**验证点**:
- 日志中应显示 "Smart mode: Fetching metadata for note_id=5e96b4f700000000010040e6"
- 工具输入中应该是真实的note_id（非幻觉ID）
- 所有4个任务都完整执行
- 最终报告包含key_success_factors和viral_formula_summary

## 参考资料

- [CrewAI Official Docs - Complex Inputs](https://docs.crewai.com/concepts/crews#crew-inputs)
- [CrewAI Official Docs - Before Kickoff Hook](https://docs.crewai.com/concepts/crews#before-kickoff-hook)
- [Pydantic v2 Serialization](https://docs.pydantic.dev/latest/concepts/serialization/)
- [OpenRouter API Docs](https://openrouter.ai/docs)

## 作者
AI Agent修复实践 - 2025年1月

## License
MIT
