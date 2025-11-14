"""CompetitorAnalysisOrchestrator tool - orchestrates full analysis workflow.

This tool wraps the entire CompetitorAnalyst workflow, allowing the agent to
perform complex multi-step analysis with a single tool call.
"""

import logging
from typing import List, Type
from pydantic import BaseModel, Field

from crewai.tools import BaseTool

from xhs_seo_optimizer.models.note import Note
from xhs_seo_optimizer.models.reports import SuccessProfileReport, FeaturePattern
from xhs_seo_optimizer.analysis_helpers import (
    aggregate_statistics,
    extract_features_matrix,
    identify_patterns,
    synthesize_formulas,
    generate_summary_insights,
    get_current_timestamp,
)

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

    This tool performs the complete analysis pipeline:
    1. Aggregate statistics
    2. Extract features from all notes
    3. Identify statistically significant patterns
    4. Synthesize LLM-generated formulas
    5. Generate summary insights
    6. Assemble SuccessProfileReport
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

            # Step 3: Identify patterns (Layer 1 & 2)
            logger.info("Step 3: Identifying patterns...")
            patterns = identify_patterns(
                target_notes,
                features_matrix,
                aggregated_stats
            )

            # Step 4: Synthesize formulas (Layer 3)
            logger.info("Step 4: Synthesizing formulas...")
            patterns = synthesize_formulas(patterns, target_notes)

            # Step 5: Generate summary insights
            logger.info("Step 5: Generating summary insights...")
            key_factors, formula_summary = generate_summary_insights(patterns)

            # Step 6: Organize patterns by feature type
            logger.info("Step 6: Organizing patterns...")
            title_patterns = [p for p in patterns if p.feature_type == "title"]
            cover_patterns = [p for p in patterns if p.feature_type == "cover"]
            content_patterns = [p for p in patterns if p.feature_type == "content"]
            tag_patterns = [p for p in patterns if p.feature_type == "tag"]

            # Step 7: Create SuccessProfileReport
            logger.info("Step 7: Creating report...")
            report = SuccessProfileReport(
                keyword=keyword,
                sample_size=len(target_notes),
                aggregated_stats=aggregated_stats,
                title_patterns=title_patterns,
                cover_patterns=cover_patterns,
                content_patterns=content_patterns,
                tag_patterns=tag_patterns,
                key_success_factors=key_factors,
                viral_formula_summary=formula_summary,
                analysis_timestamp=get_current_timestamp()
            )

            logger.info("Competitor analysis complete!")
            logger.info(f"Found {len(patterns)} total patterns: "
                       f"{len(title_patterns)} title, {len(cover_patterns)} cover, "
                       f"{len(content_patterns)} content, {len(tag_patterns)} tag")

            # Return JSON
            return report.model_dump_json(indent=2)

        except Exception as e:
            logger.error(f"Competitor analysis failed: {e}", exc_info=True)
            raise
