# Implement Content Analysis Tools

**Status:** proposal
**Created:** 2025-11-12
**Updated:** 2025-11-12

## Summary

Implement two foundational tools for analyzing Xiaohongshu note content: `MultiModalVisionTool` for visual analysis (images/videos) and `NLPAnalysisTool` for text analysis (titles/content). These tools will be used by all analysis agents in the system to understand note content characteristics.

## Motivation

The XHS SEO Optimizer crew requires deep understanding of note content to:
1. Identify visual patterns in high-performing notes (cover images, composition, style)
2. Extract text patterns from titles and content (structure, hooks, sentiment)
3. Correlate content features with performance metrics (CTR, comment_rate, etc.)

Without these tools, agents cannot perform meaningful gap analysis or generate actionable recommendations.

## Proposed Changes

### Files to Create

**Core Tools:**
- `src/xhs_seo_optimizer/tools/__init__.py` - Tools package initialization
- `src/xhs_seo_optimizer/tools/multimodal_vision.py` - Vision analysis tool
- `src/xhs_seo_optimizer/tools/nlp_analysis.py` - NLP analysis tool

**Models:**
- `src/xhs_seo_optimizer/models/__init__.py` - Models package initialization
- `src/xhs_seo_optimizer/models/note.py` - Note data models (Pydantic)
- `src/xhs_seo_optimizer/models/analysis_results.py` - Analysis result models

**Tests:**
- `tests/test_tools/test_multimodal_vision.py` - Vision tool tests
- `tests/test_tools/test_nlp_analysis.py` - NLP tool tests

**Configuration:**
- `.env.example` - Environment variable template
- `pyproject.toml` - Project dependencies and metadata

### Architecture Decision

**Tool vs Agent Approach:**
While these components could be implemented as agents themselves, we wrap them as Tools for:
- **Reusability**: Multiple agents can call the same tool
- **Simplicity**: Direct function calls vs. agent orchestration overhead
- **Determinism**: Tools provide consistent, fast results
- **Cost**: Avoid nested LLM calls for deterministic analysis

**API Provider Choice:**
We use **OpenRouter + Gemini** instead of OpenAI for significant cost savings:
- **Vision**: `google/gemini-2.5-flash-lite` (~$0.002/image vs. GPT-4V's ~$0.20/image)
- **Text**: `google/gemini-2.0-flash-thinking-exp-1219:free` (free tier!)
- **Cost reduction**: ~100x cheaper while maintaining good quality
- **Flexibility**: OpenRouter provides access to 200+ models if we need to switch

**LLM-Powered Analysis:**
We use LLM-powered analysis within tools for:
- Image understanding (via Gemini 2.5 Flash Lite vision capabilities)
- Semantic text analysis (via free Gemini model for sentiment, hooks, marketing feel)
- Traditional NLP (spaCy, regex) for fast, deterministic features (word count, emojis, etc.)

## Implementation Plan

### Phase 1: Project Setup (Tasks 1-3)
1. Create directory structure following CrewAI conventions
2. Set up `pyproject.toml` with dependencies
3. Create `.env.example` and configure environment

### Phase 2: Data Models (Tasks 4-5)
4. Implement Note data models (meta_data, prediction, tag schemas)
5. Implement analysis result models (VisionAnalysis, TextAnalysis)

### Phase 3: MultiModalVision Tool (Tasks 6-9)
6. Implement base MultiModalVisionTool class
7. Add image fetching and preprocessing
8. Integrate GPT-4V for visual analysis
9. Add unit tests for vision tool

### Phase 4: NLPAnalysis Tool (Tasks 10-13)
10. Implement base NLPAnalysisTool class
11. Add text preprocessing and feature extraction
12. Integrate LLM for semantic analysis
13. Add unit tests for NLP tool

### Phase 5: Integration Testing (Tasks 14-15)
14. Test tools with example data from docs/
15. Create usage examples in docstrings

## Testing Strategy

**Unit Tests:**
- Mock external API calls (OpenAI, image fetching)
- Test input validation and error handling
- Verify output schema compliance

**Integration Tests:**
- Test with real example data from `docs/owned_note.json` and `docs/target_notes.json`
- Validate end-to-end tool functionality
- Test with various image formats and text lengths

**Mocking Strategy:**
- Use `pytest-mock` for API calls
- Create fixture data for consistent testing
- Test both success and failure scenarios

## Risks and Considerations

**1. API Rate Limits**
- Risk: OpenRouter API has rate limits
- Mitigation: Implement exponential backoff, respect rate limits, return errors immediately

**2. Image URL Access**
- Risk: Xiaohongshu CDN may block requests or require authentication
- Mitigation: Add proper headers (User-Agent, Referer), retry logic up to 3 times

**3. Chinese Text Processing**
- Risk: NLP tools may not handle Chinese text well
- Mitigation: Use spaCy Chinese models (`zh_core_web_sm`), test with real data early

**4. Cost Management**
- Risk: Even with cheap models, costs can add up at scale
- Mitigation:
  - Use `google/gemini-2.5-flash-lite` (~$0.002/image, 100x cheaper than GPT-4V)
  - Use free tier models for text analysis when possible
  - Cache results aggressively (7-day TTL)
  - Estimated cost: ~$0.22 per 100 notes (very affordable)

**5. Tool Output Format**
- Risk: Inconsistent outputs make agent integration difficult
- Mitigation: Use Pydantic models for type safety, comprehensive validation

**6. API Provider Dependency**
- Risk: OpenRouter or Gemini service disruption
- Mitigation: Return error immediately (no fallback), log errors clearly for debugging

## Alternatives Considered

**Alternative 1: Implement as Agents Instead of Tools**
- Pros: More flexible, can have memory and context
- Cons: Slower, more expensive, harder to reuse, overkill for deterministic tasks
- Decision: Use Tools with LLM-powered internals

**Alternative 2: Use Only Traditional NLP (No LLM)**
- Pros: Faster, cheaper, more predictable
- Cons: Less semantic understanding, misses nuanced patterns
- Decision: Hybrid approach - traditional features + LLM insights

**Alternative 3: Single Combined Analysis Tool**
- Pros: Simpler architecture, one tool to maintain
- Cons: Violates separation of concerns, harder to test, less reusable
- Decision: Keep separate for clarity and modularity

**Alternative 4: Use OpenAI GPT-4V Instead of Gemini**
- Pros: More mature API, potentially better quality
- Cons: 100x more expensive (~$0.20/image vs. $0.002/image), slower
- Decision: Use OpenRouter + Gemini 2.5 Flash Lite for cost efficiency

**Alternative 5: Use Paid Gemini Models Instead of Free Tier**
- Pros: Higher rate limits, potentially better quality
- Cons: Added cost
- Decision: Start with free/cheap models, upgrade if quality issues arise

## Success Criteria

**MVP Definition:**
1. ✅ MultiModalVisionTool can analyze images from URLs
2. ✅ Vision tool returns structured output (style, colors, text_overlay, composition)
3. ✅ NLPAnalysisTool can analyze Chinese and English text
4. ✅ NLP tool returns structured output (sentiment, questions, hooks, keywords)
5. ✅ Both tools have >80% test coverage
6. ✅ Tools work with example data from docs/

**Future Enhancements:**
- Video analysis support
- Advanced OCR for text in images
- Multi-language NLP support
- Fine-tuned models for Xiaohongshu content
