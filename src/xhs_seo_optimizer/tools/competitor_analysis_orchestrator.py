"""CompetitorAnalysisOrchestrator tool - orchestrates full analysis workflow.

This tool wraps the entire CompetitorAnalyst workflow, allowing the agent to
perform complex multi-step analysis with a single tool call.
"""

import logging
from pathlib import Path
from typing import List, Type
from pydantic import BaseModel, Field

from crewai.tools import BaseTool

from xhs_seo_optimizer.models.note import Note
from xhs_seo_optimizer.models.reports import SuccessProfileReport
from xhs_seo_optimizer.analysis_helpers import (
    aggregate_statistics,
    extract_features_matrix,
    filter_notes_by_metric_variance,
    analyze_metric_success,
    generate_summary_insights,
    get_current_timestamp,
    AnalysisConfig,
)
from xhs_seo_optimizer.attribution import get_all_metrics

logger = logging.getLogger(__name__)


class CompetitorAnalysisInput(BaseModel):
    """Input schema for CompetitorAnalysisOrchestrator."""

    target_notes: List[Note] = Field(
        description="List of high-performing target notes to analyze"
    )
    keyword: str = Field(
        description="Target keyword these notes are competing for"
    )


class CompetitorAnalysisOrchestrator(BaseTool):
    """Orchestrates the full competitor analysis workflow.

    This tool performs the complete analysis pipeline using METRIC-CENTRIC approach:
    1. Aggregate statistics
    2. Extract features from all notes
    3. For each metric:
         - Filter notes by variance
         - Analyze all relevant features in ONE LLM call
         - Generate MetricSuccessProfile
    4. Generate cross-metric summary insights (key success factors + viral formula)
    5. Assemble SuccessProfileReport (metric-centric structure)

    **Key Improvement**: Reduces LLM calls from ~19 to ~10 by analyzing
    all features for a metric together instead of separately.

    **Data Structure**: Returns metric-centric profiles for gap analysis workflow,
    where downstream agents compare owned notes' metrics against competitor baselines.
    """

    name: str = "CompetitorAnalysisOrchestrator"
    description: str = (
        "Analyzes target notes to identify success patterns. "
        "Performs statistical analysis, feature extraction, and pattern identification. "
        "Returns a complete SuccessProfileReport with patterns and formulas."
    )
    args_schema: Type[BaseModel] = CompetitorAnalysisInput

    def _run(
        self,
        target_notes: List[Note],
        keyword: str
    ) -> str:
        """Execute the full competitor analysis workflow.

        Args:
            target_notes: List of high-performing notes to analyze
            keyword: Target keyword

        Returns:
            JSON string of SuccessProfileReport
        """
        logger.info(f"Starting competitor analysis for keyword: {keyword}")
        logger.info(f"Analyzing {len(target_notes)} target notes")

        try:
            # Step 1: Aggregate statistics
            logger.info("Step 1: Aggregating statistics...")
            aggregated_stats = aggregate_statistics(target_notes)

            # Step 2: Extract features
            logger.info("Step 2: Extracting features...")
            features_matrix = extract_features_matrix(target_notes)

            # Step 3: Metric-centric analysis (NEW APPROACH)
            logger.info("Step 3: Analyzing patterns (metric-centric approach)...")
            metric_profiles = []

            for metric in get_all_metrics():
                # 3.1: Filter notes by metric variance
                filtered_notes, variance_level = filter_notes_by_metric_variance(
                    target_notes, metric
                )

                # 3.2: Analyze all features for this metric in ONE LLM call
                if len(filtered_notes) >= AnalysisConfig.MIN_SAMPLE_SIZE:
                    profile = analyze_metric_success(
                        metric=metric,
                        filtered_notes=filtered_notes,
                        features_matrix=features_matrix,
                        variance_level=variance_level,
                        keyword=keyword
                    )
                    metric_profiles.append(profile)
                else:
                    logger.warning(f"Skipping {metric}: insufficient notes after filtering")

            logger.info(f"✓ Created {len(metric_profiles)} metric success profiles")

            # Step 4: Generate cross-metric summary insights
            logger.info("Step 4: Generating cross-metric summary insights...")
            key_factors, formula_summary = generate_summary_insights(metric_profiles)

            # Step 5: Create SuccessProfileReport (metric-centric structure)
            logger.info("Step 5: Creating SuccessProfileReport...")
            report = SuccessProfileReport(
                keyword=keyword,
                sample_size=len(target_notes),
                aggregated_stats=aggregated_stats,
                metric_profiles=metric_profiles,
                key_success_factors=key_factors,
                viral_formula_summary=formula_summary,
                analysis_timestamp=get_current_timestamp()
            )

            logger.info("✓ Competitor analysis complete!")
            logger.info(f"Analyzed {len(metric_profiles)} metrics with metric-centric approach")

            # Save report to local file
            output_dir = Path.cwd() / "output"
            output_dir.mkdir(exist_ok=True)

            output_path = output_dir / f"competitor_analysis_{keyword[:20]}.json"
            report_json = report.model_dump_json(indent=2)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_json)

            logger.info(f"Report saved to: {output_path}")

            # Return JSON
            return report_json

        except Exception as e:
            logger.error(f"Competitor analysis failed: {e}", exc_info=True)
            raise
