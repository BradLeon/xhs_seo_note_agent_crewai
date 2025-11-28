"""Flow State - XHS SEO Optimizer Flow 状态模型.

Pydantic model for managing state across the CrewAI Flow.
Holds user inputs, intermediate outputs from each crew, and final results.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from xhs_seo_optimizer.models.reports import (
    SuccessProfileReport,
    AuditReport,
    GapReport,
    OptimizedNote,
)


class XhsSeoFlowState(BaseModel):
    """Flow state for XHS SEO optimization pipeline.

    Holds all data throughout the Flow execution lifecycle:
    - User inputs (immutable after initialization)
    - Intermediate outputs from each crew
    - Final optimized note output
    - Execution metadata
    """

    # === User Inputs (set at flow start) ===
    keyword: str = Field(
        default="",
        description="Target SEO keyword for optimization"
    )

    target_notes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Competitor notes (serialized from Note objects)"
    )

    owned_note: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Client's note to optimize (serialized from Note object)"
    )

    # === Intermediate Outputs ===
    success_profile_report: Optional[SuccessProfileReport] = Field(
        default=None,
        description="Success patterns from competitor analysis (CompetitorAnalyst output)"
    )

    audit_report: Optional[AuditReport] = Field(
        default=None,
        description="Audit results for owned note (OwnedNoteAuditor output)"
    )

    gap_report: Optional[GapReport] = Field(
        default=None,
        description="Performance gaps analysis (GapFinder output)"
    )

    # === Final Output ===
    optimized_note: Optional[OptimizedNote] = Field(
        default=None,
        description="Final optimized note ready for publishing (OptimizationStrategist output)"
    )

    # === Execution Metadata ===
    flow_started_at: Optional[str] = Field(
        default=None,
        description="ISO timestamp when flow started"
    )

    flow_completed_at: Optional[str] = Field(
        default=None,
        description="ISO timestamp when flow completed"
    )

    errors: List[str] = Field(
        default_factory=list,
        description="List of errors encountered during flow execution"
    )

    class Config:
        """Pydantic config for flow state."""
        arbitrary_types_allowed = True

    def is_parallel_phase_complete(self) -> bool:
        """Check if both parallel crews have completed."""
        return (
            self.success_profile_report is not None and
            self.audit_report is not None
        )

    def has_errors(self) -> bool:
        """Check if any errors occurred during flow execution."""
        return len(self.errors) > 0

    def add_error(self, error: str) -> None:
        """Add an error message to the errors list."""
        self.errors.append(error)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the flow state for logging/debugging."""
        return {
            "keyword": self.keyword,
            "target_notes_count": len(self.target_notes),
            "owned_note_id": self.owned_note.get("note_id") if self.owned_note else None,
            "success_profile_ready": self.success_profile_report is not None,
            "audit_report_ready": self.audit_report is not None,
            "gap_report_ready": self.gap_report is not None,
            "optimized_note_ready": self.optimized_note is not None,
            "errors_count": len(self.errors),
            "flow_started_at": self.flow_started_at,
            "flow_completed_at": self.flow_completed_at,
        }
