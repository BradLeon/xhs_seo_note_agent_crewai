# Design: Data Aggregation Tools

## Architecture Overview

Both tools follow the same pattern as existing content analysis tools (MultiModalVision, NLPAnalysis):
- Inherit from CrewAI's `BaseTool`
- Define clear `Input` schemas with Pydantic
- Return structured outputs (Pydantic models, converted to JSON strings)
- Pure computation - no external API calls (unlike vision/NLP tools)
- Deterministic behavior for reliability

## DataAggregator Tool Design

### Statistical Methods

**Choice of Statistics**:
- **Mean**: Primary measure for central tendency (simple, interpretable)
- **Median**: Robustness against outliers (critical for small samples of 3-10 notes)
- **Std Dev**: Measure of variation (needed for StatisticalDelta z-scores)
- **Min/Max**: Range boundaries (useful for context)

**Why These Metrics?**:
- Small sample sizes (typically 3-10 target_notes) → median more reliable than mean for skewed data
- Need std dev for downstream statistical significance testing
- Min/max help identify outliers and data quality issues

**Outlier Handling**:
```python
# Configurable outlier removal (default: ±2σ)
# Using IQR method as alternative for small samples:
Q1 = np.percentile(values, 25)
Q3 = np.percentile(values, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
```

**Decision**: Start with simple ±2σ removal, add IQR method if needed based on testing.

### Tag Aggregation Strategy

**Frequency Counting**:
```python
# Simple frequency dict
tag_frequencies = {
  "intention_lv2": Counter(["经验分享", "经验分享", "产品推荐", ...])
}
```

**Mode Calculation**:
```python
# Most common value
tag_modes = {
  "intention_lv2": tag_frequencies["intention_lv2"].most_common(1)[0][0]
}
```

**Handling Ties**: If multiple modes exist with same frequency, return first alphabetically for determinism.

### Grouped Aggregation (Future Enhancement)

CompetitorAnalyst may want to know: "Among notes with `intention_lv2='经验分享'`, what's the average CTR?"

**API Design**:
```python
data_aggregator._run(
    notes=target_notes,
    group_by="intention_lv2"  # Optional parameter
)
```

**Decision**: Implement basic aggregation first, add grouping in future change if needed.

## StatisticalDelta Tool Design

### Significance Testing Approach

**Problem**: Compare single value (owned_note) against distribution (target_notes)

**Options Considered**:

1. **Z-Test (One-Sample)**:
   - Assumes target_notes distribution is normal
   - Formula: `z = (x - μ) / σ`
   - Pros: Simple, fast, interpretable
   - Cons: Assumes normality (may not hold for small N)

2. **T-Test (One-Sample)**:
   - More appropriate for small samples (N<30)
   - Formula: `t = (x - μ) / (s / √n)`
   - Pros: Better for small N, robust to non-normality
   - Cons: Slightly more complex

3. **Bootstrap/Permutation Test**:
   - Non-parametric, no distribution assumptions
   - Pros: Most robust
   - Cons: Computationally expensive for our simple use case

**Decision**: Use **z-test for simplicity**, but calculate with sample std dev (not population std dev) to account for small N. This is a pragmatic middle ground:

```python
z_score = (owned_value - target_mean) / target_std
p_value = 2 * (1 - norm.cdf(abs(z_score)))  # Two-tailed test
```

**Rationale**:
- Sample sizes are small (3-10) but not tiny (>1)
- We care more about magnitude of difference than precise p-values
- Z-score intuition is easier for agents to understand ("3σ below mean" is clear)
- Can upgrade to t-test later if needed without changing API

### Significance Levels

**Classification Thresholds**:
```python
if p_value < 0.001:
    significance = "critical"   # >99.9% confidence
elif p_value < 0.01:
    significance = "very_significant"  # >99% confidence
elif p_value < 0.05:
    significance = "significant"  # >95% confidence (standard threshold)
elif p_value < 0.10:
    significance = "marginal"   # >90% confidence
else:
    significance = "none"       # Not significant
```

**Decision**: Use standard α=0.05 as default threshold, but provide 4-tier classification for nuance.

### Gap Prioritization

**Sorting Strategy**:
Gaps should be sorted by *both* significance AND magnitude:

```python
priority_score = abs(z_score) * abs(delta_pct / 100)
# Example: z=-3.0, delta=-50% → score = 3.0 * 0.5 = 1.5
#          z=-2.0, delta=-90% → score = 2.0 * 0.9 = 1.8 (higher priority)
```

**Rationale**:
- A gap that's statistically significant but tiny (e.g., -2% CTR) is less actionable
- A gap that's large but not significant (e.g., -30% CTR with z=-1.5) may still matter
- Combining both factors gives better prioritization for OptimizationStrategist

**Decision**: Sort by `abs(z_score)` primarily, use `delta_pct` as tiebreaker. This keeps statistical rigor while acknowledging practical importance.

## Data Models

### AggregatedMetrics Model

```python
class MetricStats(BaseModel):
    """Statistics for a single metric"""
    mean: float
    median: float
    std: float
    min: float
    max: float
    count: int  # Number of data points after outlier removal

class AggregatedMetrics(BaseModel):
    """Aggregated statistics across multiple notes"""
    prediction_stats: Dict[str, MetricStats] = Field(
        description="Statistics for each prediction metric"
    )
    tag_frequencies: Dict[str, Dict[str, int]] = Field(
        description="Frequency counts for each tag dimension"
    )
    tag_modes: Dict[str, str] = Field(
        description="Most common value for each tag dimension"
    )
    sample_size: int = Field(description="Number of notes analyzed")
    outliers_removed: int = Field(description="Number of outliers excluded")
```

### GapAnalysis Model

```python
class Gap(BaseModel):
    """Analysis of a single metric gap"""
    metric: str
    owned_value: float
    target_mean: float
    target_std: float
    delta_absolute: float
    delta_pct: float  # Percentage change
    z_score: float
    p_value: float
    significance: str  # critical | very_significant | significant | marginal | none
    interpretation: str  # Human-readable summary

class GapAnalysis(BaseModel):
    """Statistical analysis of gaps between owned and target notes"""
    significant_gaps: List[Gap] = Field(
        description="Gaps with p < 0.05, sorted by priority"
    )
    non_significant_gaps: List[Gap] = Field(
        description="Gaps with p >= 0.05, for completeness"
    )
    priority_order: List[str] = Field(
        description="Metric names sorted by priority (z_score * delta_pct)"
    )
    sample_size: int = Field(description="Number of target notes used")
```

## Error Handling

### DataAggregator

**Edge Cases**:
1. Empty notes list → Raise `ValueError` with clear message
2. Single note → Return stats but warn that variance is undefined
3. All values identical → std=0, handle division by zero in delta calculation
4. Missing fields in notes → Skip those notes with warning log
5. All outliers removed → Raise error (can't compute meaningful stats)

### StatisticalDelta

**Edge Cases**:
1. target_std = 0 (all target values identical) → Can't compute z-score, return "undefined" significance
2. owned_value = 0, target_mean = 0 → delta_pct undefined, use absolute delta only
3. No target notes → Raise `ValueError`
4. Metric doesn't exist in owned_note → Return "N/A" gap with explanation

## Performance Considerations

**Data Volume**:
- Typical: 3-10 target_notes per analysis
- Max: ~50 notes (batch analysis)
- Each note has ~10 prediction metrics + 5 tag fields

**Optimization Strategy**:
- Use numpy for vectorized operations (much faster than Python loops)
- Compute all stats in single pass where possible
- Cache aggregated results (agents may call multiple times)

**Memory Footprint**:
- Small - all computation in-memory
- No external API calls = no network overhead
- Estimated: <10MB for 50 notes

## Testing Strategy

**Unit Tests** (>90% coverage required):
- Correctness: mean/median/std calculations match numpy
- Edge cases: empty lists, single value, all identical
- Outlier removal: verify correct filtering
- Significance: z-scores and p-values match scipy.stats
- Sorting: priority order correct

**Integration Tests**:
- Test with real note data from `docs/target_notes.json`
- Verify output schema compliance
- Test workflow: DataAggregator output → StatisticalDelta input

**Property-Based Tests** (optional, using Hypothesis):
- Invariants: mean always between min and max
- Statistical properties: p-values sum to ~50% over random data
- Symmetry: swapping owned and target reverses delta sign

## Future Enhancements

1. **Grouped Aggregation**: Compute stats grouped by tag dimensions
2. **Time-Series Tracking**: Track how gaps evolve over multiple iterations
3. **Multi-Variable Analysis**: Correlation between meta_data features and prediction scores
4. **Confidence Intervals**: Report 95% CI for target means
5. **Effect Size**: Cohen's d for practical significance alongside statistical significance
6. **Visualization**: Generate simple charts (box plots, distributions) for reports

## Alternatives Considered

### Alternative 1: Use pandas for aggregation
- **Pros**: Mature library, well-tested, familiar to data scientists
- **Cons**: Heavy dependency (30MB+), overkill for simple stats
- **Decision**: Use numpy/scipy - lighter, sufficient for our needs

### Alternative 2: Statistical Delta as LLM-powered tool
- **Pros**: LLM can provide richer interpretations
- **Cons**: Non-deterministic, expensive, slower, harder to test
- **Decision**: Pure computation with deterministic interpretation templates

### Alternative 3: Combine both tools into one
- **Pros**: Simpler API, fewer tools
- **Cons**: Violates single responsibility, harder to test/reuse
- **Decision**: Keep separate - CompetitorAnalyst uses DataAggregator alone, GapFinder uses both

### Alternative 4: Use R or Julia for statistical rigor
- **Pros**: More sophisticated statistical methods available
- **Cons**: Additional language dependency, harder to integrate with Python CrewAI
- **Decision**: Python scipy/numpy is sufficient for our use case

## References

- [scipy.stats documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [One-sample z-test](https://en.wikipedia.org/wiki/One-sample_z-test)
- [Statistical significance](https://en.wikipedia.org/wiki/Statistical_significance)
- [Effect size and practical significance](https://en.wikipedia.org/wiki/Effect_size)
