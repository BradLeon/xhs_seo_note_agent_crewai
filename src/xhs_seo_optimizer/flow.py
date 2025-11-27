"""XHS SEO Optimizer Flow - CrewAI Flow 编排器.

Orchestrates the 4 crews (CompetitorAnalyst, OwnedNoteAuditor, GapFinder,
OptimizationStrategist) using CrewAI Flow for parallel execution and
state management.

Architecture:
    @start() receive_inputs
              |
    +---------+---------+
    |                   |
    v                   v
analyze_competitors  audit_owned_note  (parallel)
    |                   |
    +---------+---------+
              |
              v (and_)
         find_gaps
              |
              v
    generate_optimization
              |
              v
       compile_results
"""

from datetime import datetime
from typing import Any

from crewai.flow.flow import Flow, listen, start, and_

from xhs_seo_optimizer.flow_state import XhsSeoFlowState
from xhs_seo_optimizer.crew_competitor_analyst import XhsSeoOptimizerCrewCompetitorAnalyst
from xhs_seo_optimizer.crew_owned_note import XhsSeoOptimizerCrewOwnedNote
from xhs_seo_optimizer.crew_gap_finder import XhsSeoOptimizerCrewGapFinder
from xhs_seo_optimizer.crew_optimization import XhsSeoOptimizerCrewOptimization


class XhsSeoOptimizerFlow(Flow[XhsSeoFlowState]):
    """CrewAI Flow for XHS SEO optimization pipeline.

    Orchestrates 4 crews with parallel execution where possible:
    1. CompetitorAnalyst + OwnedNoteAuditor (parallel)
    2. GapFinder (joins parallel results)
    3. OptimizationStrategist (sequential)

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
        self.state.flow_started_at = datetime.utcnow().isoformat() + "Z"

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
    # Parallel Phase - CompetitorAnalyst & OwnedNoteAuditor
    # =========================================================================

    @listen(receive_inputs)
    def analyze_competitors(self) -> Any:
        """Phase 1A: Analyze competitor notes (parallel with audit_owned_note).

        Runs CompetitorAnalyst crew to extract success patterns from target_notes.

        Returns:
            SuccessProfileReport from CompetitorAnalyst crew
        """
        print("[Flow] Starting CompetitorAnalyst crew...")

        try:
            crew = XhsSeoOptimizerCrewCompetitorAnalyst()
            result = crew.crew().kickoff(inputs={
                "target_notes": self.state.target_notes,
                "keyword": self.state.keyword,
            })

            # Extract Pydantic output from CrewOutput
            if hasattr(result, 'pydantic') and result.pydantic:
                self.state.success_profile_report = result.pydantic
                print(f"[Flow] CompetitorAnalyst completed successfully")
            else:
                # Fallback: try to parse from raw output
                print(f"[Flow] Warning: CompetitorAnalyst returned non-pydantic output")
                self.state.add_error("CompetitorAnalyst did not return Pydantic output")

        except Exception as e:
            error_msg = f"CompetitorAnalyst failed: {str(e)}"
            print(f"[Flow] Error: {error_msg}")
            self.state.add_error(error_msg)

        return self.state.success_profile_report

    @listen(receive_inputs)
    def audit_owned_note(self) -> Any:
        """Phase 1B: Audit owned note (parallel with analyze_competitors).

        Runs OwnedNoteAuditor crew to analyze the client's note.

        Returns:
            AuditReport from OwnedNoteAuditor crew
        """
        print("[Flow] Starting OwnedNoteAuditor crew...")

        try:
            crew = XhsSeoOptimizerCrewOwnedNote()
            result = crew.crew().kickoff(inputs={
                "owned_note": self.state.owned_note,
                "keyword": self.state.keyword,
            })

            # Extract Pydantic output from CrewOutput
            if hasattr(result, 'pydantic') and result.pydantic:
                self.state.audit_report = result.pydantic
                print(f"[Flow] OwnedNoteAuditor completed successfully")
            else:
                print(f"[Flow] Warning: OwnedNoteAuditor returned non-pydantic output")
                self.state.add_error("OwnedNoteAuditor did not return Pydantic output")

        except Exception as e:
            error_msg = f"OwnedNoteAuditor failed: {str(e)}"
            print(f"[Flow] Error: {error_msg}")
            self.state.add_error(error_msg)

        return self.state.audit_report

    # =========================================================================
    # Join Phase - GapFinder
    # =========================================================================

    @listen(and_(analyze_competitors, audit_owned_note))
    def find_gaps(self) -> Any:
        """Phase 2: Find gaps (after BOTH parallel crews complete).

        Uses and_() to wait for both analyze_competitors AND audit_owned_note.
        Runs GapFinder crew to identify performance gaps.

        Returns:
            GapReport from GapFinder crew
        """
        print("\n" + "-" * 60)
        print("[Flow] Both Phase 1 crews completed. Starting GapFinder...")
        print("-" * 60)

        # Check if parallel phase completed successfully
        if not self.state.is_parallel_phase_complete():
            error_msg = "Cannot run GapFinder: missing parallel phase outputs"
            print(f"[Flow] Error: {error_msg}")
            self.state.add_error(error_msg)
            return None

        try:
            crew = XhsSeoOptimizerCrewGapFinder()
            result = crew.crew().kickoff(inputs={
                "success_profile_report": self.state.success_profile_report.model_dump(),
                "audit_report": self.state.audit_report.model_dump(),
                "keyword": self.state.keyword,
            })

            if hasattr(result, 'pydantic') and result.pydantic:
                self.state.gap_report = result.pydantic
                print(f"[Flow] GapFinder completed successfully")
            else:
                print(f"[Flow] Warning: GapFinder returned non-pydantic output")
                self.state.add_error("GapFinder did not return Pydantic output")

        except Exception as e:
            error_msg = f"GapFinder failed: {str(e)}"
            print(f"[Flow] Error: {error_msg}")
            self.state.add_error(error_msg)

        return self.state.gap_report

    # =========================================================================
    # Sequential Phase - OptimizationStrategist
    # =========================================================================

    @listen(find_gaps)
    def generate_optimization(self) -> Any:
        """Phase 3: Generate optimization plan and images.

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
            result = crew.crew().kickoff(inputs={
                "keyword": self.state.keyword,
                "gap_report": self.state.gap_report.model_dump(),
                "audit_report": self.state.audit_report.model_dump(),
                "success_profile_report": self.state.success_profile_report.model_dump(),
                "owned_note": self.state.owned_note,
            })

            if hasattr(result, 'pydantic') and result.pydantic:
                self.state.optimized_note = result.pydantic
                print(f"[Flow] OptimizationStrategist completed successfully")
            else:
                print(f"[Flow] Warning: OptimizationStrategist returned non-pydantic output")
                self.state.add_error("OptimizationStrategist did not return Pydantic output")

        except Exception as e:
            error_msg = f"OptimizationStrategist failed: {str(e)}"
            print(f"[Flow] Error: {error_msg}")
            self.state.add_error(error_msg)

        return self.state.optimized_note

    # =========================================================================
    # Final Phase - Compile Results
    # =========================================================================

    @listen(generate_optimization)
    def compile_results(self) -> XhsSeoFlowState:
        """Phase 4: Finalize and return results.

        Records completion time and logs summary.

        Returns:
            Final flow state with all results
        """
        self.state.flow_completed_at = datetime.utcnow().isoformat() + "Z"

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
