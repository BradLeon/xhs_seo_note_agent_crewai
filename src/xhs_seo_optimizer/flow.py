"""XHS SEO Optimizer Flow - CrewAI Flow 编排器.

Orchestrates the 4 crews (CompetitorAnalyst, OwnedNoteAuditor, GapFinder,
OptimizationStrategist) using CrewAI Flow for sequential execution and
state management.

Architecture (Sequential to avoid API rate limiting):
    @start() receive_inputs
              |
              v
    analyze_competitors
              |
              v
      audit_owned_note
              |
              v
         find_gaps
              |
              v
    generate_optimization
              |
              v
       compile_results

Note: API retry is handled by LiteLLM (num_retries=3 in crew LLM configs).
"""

from datetime import datetime, timezone
from typing import Any

from crewai.flow.flow import Flow, listen, start

from xhs_seo_optimizer.flow_state import XhsSeoFlowState
from xhs_seo_optimizer.crew_competitor_analyst import XhsSeoOptimizerCrewCompetitorAnalyst
from xhs_seo_optimizer.crew_owned_note import XhsSeoOptimizerCrewOwnedNote
from xhs_seo_optimizer.crew_gap_finder import XhsSeoOptimizerCrewGapFinder
from xhs_seo_optimizer.crew_optimization import XhsSeoOptimizerCrewOptimization


class XhsSeoOptimizerFlow(Flow[XhsSeoFlowState]):
    """CrewAI Flow for XHS SEO optimization pipeline.

    Orchestrates 4 crews in sequential execution to avoid API rate limiting:
    1. CompetitorAnalyst
    2. OwnedNoteAuditor
    3. GapFinder
    4. OptimizationStrategist

    Usage:
        flow = XhsSeoOptimizerFlow()
        result = flow.kickoff(inputs={
            "keyword": "老爸测评dha推荐哪几款",
            "target_notes": [...],
            "owned_note": {...}
        })
    """

    # =========================================================================
    # Entry Point
    # =========================================================================

    @start()
    def receive_inputs(self) -> XhsSeoFlowState:
        """Entry point: Validate inputs and record flow start time.

        Returns:
            Updated flow state with validated inputs
        """
        self.state.flow_started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        print("\n" + "=" * 60)
        print("[Flow] XHS SEO Optimization Flow Started")
        print("=" * 60)

        # Validate required inputs
        if not self.state.keyword:
            raise ValueError("keyword is required")
        if not self.state.target_notes or len(self.state.target_notes) == 0:
            raise ValueError("At least one target_note is required")
        if not self.state.owned_note:
            raise ValueError("owned_note is required")

        print(f"[Flow] Keyword: {self.state.keyword}")
        print(f"[Flow] Target Notes: {len(self.state.target_notes)}")
        print(f"[Flow] Owned Note ID: {self.state.owned_note.get('note_id', 'unknown')}")
        print("-" * 60 + "\n")

        return self.state

    # =========================================================================
    # Sequential Phase 1 - CompetitorAnalyst
    # =========================================================================

    @listen(receive_inputs)
    def analyze_competitors(self) -> Any:
        """Phase 1: Analyze competitor notes.

        Runs CompetitorAnalyst crew to extract success patterns from target_notes.

        Returns:
            SuccessProfileReport from CompetitorAnalyst crew
        """
        print("[Flow] Starting CompetitorAnalyst crew...")

        try:
            crew = XhsSeoOptimizerCrewCompetitorAnalyst()

            # Call our wrapped kickoff() which returns Pydantic directly
            self.state.success_profile_report = crew.kickoff(inputs={
                "target_notes": self.state.target_notes,
                "keyword": self.state.keyword,
            })
            print(f"[Flow] CompetitorAnalyst completed successfully")

        except Exception as e:
            error_msg = f"CompetitorAnalyst failed: {str(e)}"
            print(f"[Flow] Error: {error_msg}")
            self.state.add_error(error_msg)

        return self.state.success_profile_report

    # =========================================================================
    # Sequential Phase 2 - OwnedNoteAuditor
    # =========================================================================

    @listen(analyze_competitors)
    def audit_owned_note(self) -> Any:
        """Phase 2: Audit owned note (after CompetitorAnalyst).

        Runs OwnedNoteAuditor crew to analyze the client's note.

        Returns:
            AuditReport from OwnedNoteAuditor crew
        """
        print("\n" + "-" * 60)
        print("[Flow] Starting OwnedNoteAuditor crew...")
        print("-" * 60)

        try:
            crew = XhsSeoOptimizerCrewOwnedNote()

            # Call our wrapped kickoff() which returns Pydantic directly
            self.state.audit_report = crew.kickoff(inputs={
                "owned_note": self.state.owned_note,
                "keyword": self.state.keyword,
            })
            print(f"[Flow] OwnedNoteAuditor completed successfully")

        except Exception as e:
            error_msg = f"OwnedNoteAuditor failed: {str(e)}"
            print(f"[Flow] Error: {error_msg}")
            self.state.add_error(error_msg)

        return self.state.audit_report

    # =========================================================================
    # Sequential Phase 3 - GapFinder
    # =========================================================================

    @listen(audit_owned_note)
    def find_gaps(self) -> Any:
        """Phase 3: Find gaps (after OwnedNoteAuditor).

        Runs GapFinder crew to identify performance gaps.

        Returns:
            GapReport from GapFinder crew
        """
        print("\n" + "-" * 60)
        print("[Flow] Starting GapFinder crew...")
        print("-" * 60)

        # Check if previous phases completed successfully
        if not self.state.is_parallel_phase_complete():
            error_msg = "Cannot run GapFinder: missing SuccessProfileReport or AuditReport"
            print(f"[Flow] Error: {error_msg}")
            self.state.add_error(error_msg)
            return None

        try:
            crew = XhsSeoOptimizerCrewGapFinder()

            # Call our wrapped kickoff() which returns Pydantic directly
            # Note: save_to_file=False when running in Flow mode
            self.state.gap_report = crew.kickoff(inputs={
                "success_profile_report": self.state.success_profile_report.model_dump(),
                "audit_report": self.state.audit_report.model_dump(),
                "keyword": self.state.keyword,
            }, save_to_file=False)
            print(f"[Flow] GapFinder completed successfully")

        except Exception as e:
            error_msg = f"GapFinder failed: {str(e)}"
            print(f"[Flow] Error: {error_msg}")
            self.state.add_error(error_msg)

        return self.state.gap_report

    # =========================================================================
    # Sequential Phase 4 - OptimizationStrategist
    # =========================================================================

    @listen(find_gaps)
    def generate_optimization(self) -> Any:
        """Phase 4: Generate optimization plan and images.

        Runs OptimizationStrategist crew to create the optimized note.

        Returns:
            OptimizedNote from OptimizationStrategist crew
        """
        print("\n" + "-" * 60)
        print("[Flow] Starting OptimizationStrategist crew...")
        print("-" * 60)

        # Check if GapFinder completed successfully
        if not self.state.gap_report:
            error_msg = "Cannot run OptimizationStrategist: missing GapReport"
            print(f"[Flow] Error: {error_msg}")
            self.state.add_error(error_msg)
            return None

        try:
            crew = XhsSeoOptimizerCrewOptimization()

            # Call our wrapped kickoff() which returns Pydantic directly
            self.state.optimized_note = crew.kickoff(inputs={
                "keyword": self.state.keyword,
                "gap_report": self.state.gap_report.model_dump(),
                "audit_report": self.state.audit_report.model_dump(),
                "success_profile_report": self.state.success_profile_report.model_dump(),
                "owned_note": self.state.owned_note,
            })
            print(f"[Flow] OptimizationStrategist completed successfully")

        except Exception as e:
            error_msg = f"OptimizationStrategist failed: {str(e)}"
            print(f"[Flow] Error: {error_msg}")
            self.state.add_error(error_msg)

        return self.state.optimized_note

    # =========================================================================
    # Final Phase 5 - Compile Results
    # =========================================================================

    @listen(generate_optimization)
    def compile_results(self) -> XhsSeoFlowState:
        """Phase 5: Finalize and return results.

        Records completion time and logs summary.

        Returns:
            Final flow state with all results
        """
        self.state.flow_completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        print("\n" + "=" * 60)
        print("[Flow] XHS SEO Optimization Flow Complete!")
        print("=" * 60)
        print(f"[Flow] Keyword: {self.state.keyword}")

        if self.state.optimized_note:
            print(f"[Flow] Optimized Note ID: {self.state.optimized_note.note_id}")
            print(f"[Flow] Title: {self.state.optimized_note.title[:50]}...")
        else:
            print("[Flow] Warning: No optimized note generated")

        if self.state.has_errors():
            print(f"[Flow] Errors encountered: {len(self.state.errors)}")
            for i, error in enumerate(self.state.errors, 1):
                print(f"  {i}. {error}")

        print(f"[Flow] Started: {self.state.flow_started_at}")
        print(f"[Flow] Completed: {self.state.flow_completed_at}")
        print("=" * 60 + "\n")

        return self.state
