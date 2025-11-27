"""Integration tests for CrewAI Flow - Flow 集成测试.

Tests the complete XhsSeoOptimizerFlow with mocked crew outputs.

Note: CrewAI Flow uses internal state management. We test using the
initial_state parameter and by directly manipulating the flow's internal state.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from xhs_seo_optimizer.flow import XhsSeoOptimizerFlow
from xhs_seo_optimizer.flow_state import XhsSeoFlowState
from xhs_seo_optimizer.models.reports import (
    SuccessProfileReport,
    AuditReport,
    GapReport,
    OptimizedNote,
)


class TestFlowState:
    """Test XhsSeoFlowState model."""

    def test_default_state(self):
        """Test state initializes with default values."""
        state = XhsSeoFlowState()

        assert state.keyword == ""
        assert state.target_notes == []
        assert state.owned_note is None
        assert state.success_profile_report is None
        assert state.audit_report is None
        assert state.gap_report is None
        assert state.optimized_note is None
        assert state.errors == []
        assert state.flow_started_at is None
        assert state.flow_completed_at is None

    def test_state_with_inputs(self):
        """Test state with user inputs."""
        state = XhsSeoFlowState(
            keyword="测试关键词",
            target_notes=[{"note_id": "note1"}, {"note_id": "note2"}],
            owned_note={"note_id": "owned1", "title": "测试标题"},
        )

        assert state.keyword == "测试关键词"
        assert len(state.target_notes) == 2
        assert state.owned_note["note_id"] == "owned1"

    def test_add_error(self):
        """Test adding errors to state."""
        state = XhsSeoFlowState()

        state.add_error("Error 1")
        state.add_error("Error 2")

        assert len(state.errors) == 2
        assert "Error 1" in state.errors
        assert "Error 2" in state.errors

    def test_has_errors(self):
        """Test has_errors method."""
        state = XhsSeoFlowState()

        assert not state.has_errors()

        state.add_error("Some error")

        assert state.has_errors()

    def test_is_parallel_phase_complete(self):
        """Test parallel phase completion check."""
        state = XhsSeoFlowState()

        # Initially incomplete
        assert not state.is_parallel_phase_complete()

        # Mock reports
        state.success_profile_report = MagicMock()
        assert not state.is_parallel_phase_complete()

        state.audit_report = MagicMock()
        assert state.is_parallel_phase_complete()


class TestFlowValidation:
    """Test Flow input validation."""

    def test_missing_keyword_raises(self):
        """Test that missing keyword raises ValueError."""
        initial_state = XhsSeoFlowState(
            keyword="",
            target_notes=[{"note_id": "note1"}],
            owned_note={"note_id": "owned1"},
        )
        flow = XhsSeoOptimizerFlow()
        # Set internal state directly for testing
        flow._state = initial_state

        with pytest.raises(ValueError, match="keyword is required"):
            flow.receive_inputs()

    def test_empty_target_notes_raises(self):
        """Test that empty target_notes raises ValueError."""
        initial_state = XhsSeoFlowState(
            keyword="测试",
            target_notes=[],
            owned_note={"note_id": "owned1"},
        )
        flow = XhsSeoOptimizerFlow()
        flow._state = initial_state

        with pytest.raises(ValueError, match="At least one target_note is required"):
            flow.receive_inputs()

    def test_missing_owned_note_raises(self):
        """Test that missing owned_note raises ValueError."""
        initial_state = XhsSeoFlowState(
            keyword="测试",
            target_notes=[{"note_id": "note1"}],
            owned_note=None,
        )
        flow = XhsSeoOptimizerFlow()
        flow._state = initial_state

        with pytest.raises(ValueError, match="owned_note is required"):
            flow.receive_inputs()

    def test_valid_inputs_pass(self):
        """Test that valid inputs pass validation."""
        initial_state = XhsSeoFlowState(
            keyword="测试关键词",
            target_notes=[{"note_id": "note1"}],
            owned_note={"note_id": "owned1"},
        )
        flow = XhsSeoOptimizerFlow()
        flow._state = initial_state

        result = flow.receive_inputs()

        assert result.flow_started_at is not None
        assert "Z" in result.flow_started_at


class TestFlowExecution:
    """Test Flow step execution with mocked crews."""

    @pytest.fixture
    def sample_target_notes(self):
        """Sample target notes for testing."""
        return [
            {
                "note_id": "target1",
                "title": "爆款标题1",
                "content": "内容1",
                "cover_image_url": "https://example.com/img1.jpg",
                "prediction": {
                    "ctr": 0.15,
                    "sort_score2": 0.8,
                    "like_rate": 0.12,
                    "collect_rate": 0.08,
                    "comment_rate": 0.05,
                    "share_rate": 0.02,
                    "follow_rate": 0.01,
                    "thumbnail_appeal": 0.9,
                    "category_relevance": 0.85,
                    "quality_score": 0.88,
                },
            },
            {
                "note_id": "target2",
                "title": "爆款标题2",
                "content": "内容2",
                "cover_image_url": "https://example.com/img2.jpg",
                "prediction": {
                    "ctr": 0.18,
                    "sort_score2": 0.85,
                    "like_rate": 0.14,
                    "collect_rate": 0.09,
                    "comment_rate": 0.06,
                    "share_rate": 0.025,
                    "follow_rate": 0.012,
                    "thumbnail_appeal": 0.92,
                    "category_relevance": 0.87,
                    "quality_score": 0.9,
                },
            },
        ]

    @pytest.fixture
    def sample_owned_note(self):
        """Sample owned note for testing."""
        return {
            "note_id": "owned1",
            "title": "待优化标题",
            "content": "待优化内容",
            "cover_image_url": "https://example.com/owned.jpg",
            "inner_image_urls": ["https://example.com/inner1.jpg"],
            "prediction": {
                "ctr": 0.05,
                "sort_score2": 0.4,
                "like_rate": 0.03,
                "collect_rate": 0.02,
                "comment_rate": 0.01,
                "share_rate": 0.005,
                "follow_rate": 0.003,
                "thumbnail_appeal": 0.5,
                "category_relevance": 0.6,
                "quality_score": 0.55,
            },
        }

    @pytest.fixture
    def mock_success_profile_report(self):
        """Mock SuccessProfileReport for testing."""
        return MagicMock(
            spec=SuccessProfileReport,
            keyword="测试关键词",
            aggregated_stats={
                "sample_size": 2,
                "prediction_stats": {
                    "ctr": {"mean": 0.165, "std": 0.015},
                    "sort_score2": {"mean": 0.825, "std": 0.025},
                },
            },
        )

    @pytest.fixture
    def mock_audit_report(self):
        """Mock AuditReport for testing."""
        return MagicMock(
            spec=AuditReport,
            note_id="owned1",
            current_metrics={
                "ctr": 0.05,
                "sort_score2": 0.4,
            },
        )

    @pytest.fixture
    def mock_gap_report(self):
        """Mock GapReport for testing."""
        return MagicMock(
            spec=GapReport,
            note_id="owned1",
            significant_gaps=[
                {"metric": "ctr", "delta": -0.115, "p_value": 0.001},
                {"metric": "sort_score2", "delta": -0.425, "p_value": 0.0001},
            ],
        )

    @pytest.fixture
    def mock_optimized_note(self):
        """Mock OptimizedNote for testing."""
        return MagicMock(
            spec=OptimizedNote,
            note_id="owned1_optimized",
            original_note_id="owned1",
            title="优化后的爆款标题",
        )

    def test_receive_inputs_sets_timestamp(
        self, sample_target_notes, sample_owned_note
    ):
        """Test receive_inputs sets flow_started_at."""
        initial_state = XhsSeoFlowState(
            keyword="测试",
            target_notes=sample_target_notes,
            owned_note=sample_owned_note,
        )
        flow = XhsSeoOptimizerFlow()
        flow._state = initial_state

        before = datetime.utcnow().isoformat()
        result = flow.receive_inputs()
        after = datetime.utcnow().isoformat()

        assert result.flow_started_at is not None
        # Timestamp should be between before and after
        assert before <= result.flow_started_at.replace("Z", "") <= after + "Z"

    @patch("xhs_seo_optimizer.flow.XhsSeoOptimizerCrewCompetitorAnalyst")
    def test_analyze_competitors_success(
        self,
        mock_crew_class,
        sample_target_notes,
        sample_owned_note,
        mock_success_profile_report,
    ):
        """Test analyze_competitors with successful crew execution."""
        # Setup mock
        mock_crew_instance = MagicMock()
        mock_crew_class.return_value = mock_crew_instance
        mock_result = MagicMock()
        mock_result.pydantic = mock_success_profile_report
        mock_crew_instance.crew.return_value.kickoff.return_value = mock_result

        # Setup flow
        initial_state = XhsSeoFlowState(
            keyword="测试",
            target_notes=sample_target_notes,
            owned_note=sample_owned_note,
        )
        flow = XhsSeoOptimizerFlow()
        flow._state = initial_state

        # Execute
        result = flow.analyze_competitors()

        # Verify
        assert flow.state.success_profile_report is not None
        assert result == mock_success_profile_report
        mock_crew_instance.crew.return_value.kickoff.assert_called_once()

    @patch("xhs_seo_optimizer.flow.XhsSeoOptimizerCrewOwnedNote")
    def test_audit_owned_note_success(
        self,
        mock_crew_class,
        sample_target_notes,
        sample_owned_note,
        mock_audit_report,
    ):
        """Test audit_owned_note with successful crew execution."""
        # Setup mock
        mock_crew_instance = MagicMock()
        mock_crew_class.return_value = mock_crew_instance
        mock_result = MagicMock()
        mock_result.pydantic = mock_audit_report
        mock_crew_instance.crew.return_value.kickoff.return_value = mock_result

        # Setup flow
        initial_state = XhsSeoFlowState(
            keyword="测试",
            target_notes=sample_target_notes,
            owned_note=sample_owned_note,
        )
        flow = XhsSeoOptimizerFlow()
        flow._state = initial_state

        # Execute
        result = flow.audit_owned_note()

        # Verify
        assert flow.state.audit_report is not None
        assert result == mock_audit_report
        mock_crew_instance.crew.return_value.kickoff.assert_called_once()

    @patch("xhs_seo_optimizer.flow.XhsSeoOptimizerCrewGapFinder")
    def test_find_gaps_success(
        self,
        mock_crew_class,
        sample_target_notes,
        sample_owned_note,
        mock_success_profile_report,
        mock_audit_report,
        mock_gap_report,
    ):
        """Test find_gaps with successful crew execution."""
        # Setup mock
        mock_crew_instance = MagicMock()
        mock_crew_class.return_value = mock_crew_instance
        mock_result = MagicMock()
        mock_result.pydantic = mock_gap_report
        mock_crew_instance.crew.return_value.kickoff.return_value = mock_result

        # Setup flow with parallel phase completed
        initial_state = XhsSeoFlowState(
            keyword="测试",
            target_notes=sample_target_notes,
            owned_note=sample_owned_note,
        )
        flow = XhsSeoOptimizerFlow()
        flow._state = initial_state

        # Mock model_dump for Pydantic models
        mock_success_profile_report.model_dump = MagicMock(return_value={})
        mock_audit_report.model_dump = MagicMock(return_value={})

        flow.state.success_profile_report = mock_success_profile_report
        flow.state.audit_report = mock_audit_report

        # Execute
        result = flow.find_gaps()

        # Verify
        assert flow.state.gap_report is not None
        assert result == mock_gap_report

    def test_find_gaps_missing_parallel_outputs(
        self, sample_target_notes, sample_owned_note
    ):
        """Test find_gaps fails gracefully when parallel outputs missing."""
        initial_state = XhsSeoFlowState(
            keyword="测试",
            target_notes=sample_target_notes,
            owned_note=sample_owned_note,
        )
        flow = XhsSeoOptimizerFlow()
        flow._state = initial_state
        # Don't set success_profile_report or audit_report

        result = flow.find_gaps()

        assert result is None
        assert flow.state.has_errors()

    @patch("xhs_seo_optimizer.flow.XhsSeoOptimizerCrewOptimization")
    def test_generate_optimization_success(
        self,
        mock_crew_class,
        sample_target_notes,
        sample_owned_note,
        mock_success_profile_report,
        mock_audit_report,
        mock_gap_report,
        mock_optimized_note,
    ):
        """Test generate_optimization with successful crew execution."""
        # Setup mock
        mock_crew_instance = MagicMock()
        mock_crew_class.return_value = mock_crew_instance
        mock_result = MagicMock()
        mock_result.pydantic = mock_optimized_note
        mock_crew_instance.crew.return_value.kickoff.return_value = mock_result

        # Setup flow with gap_report
        initial_state = XhsSeoFlowState(
            keyword="测试",
            target_notes=sample_target_notes,
            owned_note=sample_owned_note,
        )
        flow = XhsSeoOptimizerFlow()
        flow._state = initial_state

        # Mock model_dump for Pydantic models
        mock_success_profile_report.model_dump = MagicMock(return_value={})
        mock_audit_report.model_dump = MagicMock(return_value={})
        mock_gap_report.model_dump = MagicMock(return_value={})

        flow.state.success_profile_report = mock_success_profile_report
        flow.state.audit_report = mock_audit_report
        flow.state.gap_report = mock_gap_report

        # Execute
        result = flow.generate_optimization()

        # Verify
        assert flow.state.optimized_note is not None
        assert result == mock_optimized_note

    def test_generate_optimization_missing_gap_report(
        self, sample_target_notes, sample_owned_note
    ):
        """Test generate_optimization fails gracefully when gap_report missing."""
        initial_state = XhsSeoFlowState(
            keyword="测试",
            target_notes=sample_target_notes,
            owned_note=sample_owned_note,
        )
        flow = XhsSeoOptimizerFlow()
        flow._state = initial_state
        # Don't set gap_report

        result = flow.generate_optimization()

        assert result is None
        assert flow.state.has_errors()

    def test_compile_results_sets_timestamp(
        self,
        sample_target_notes,
        sample_owned_note,
        mock_optimized_note,
    ):
        """Test compile_results sets flow_completed_at."""
        initial_state = XhsSeoFlowState(
            keyword="测试",
            target_notes=sample_target_notes,
            owned_note=sample_owned_note,
        )
        flow = XhsSeoOptimizerFlow()
        flow._state = initial_state

        flow.state.flow_started_at = datetime.utcnow().isoformat() + "Z"
        flow.state.optimized_note = mock_optimized_note
        mock_optimized_note.note_id = "test_optimized"
        mock_optimized_note.title = "优化后的标题测试"

        before = datetime.utcnow().isoformat()
        result = flow.compile_results()
        after = datetime.utcnow().isoformat()

        assert result.flow_completed_at is not None
        assert before <= result.flow_completed_at.replace("Z", "") <= after + "Z"


class TestFlowErrorHandling:
    """Test Flow error handling scenarios."""

    @pytest.fixture
    def basic_flow(self):
        """Create flow with basic valid inputs."""
        initial_state = XhsSeoFlowState(
            keyword="测试",
            target_notes=[{"note_id": "note1"}],
            owned_note={"note_id": "owned1"},
        )
        flow = XhsSeoOptimizerFlow()
        flow._state = initial_state
        return flow

    @patch("xhs_seo_optimizer.flow.XhsSeoOptimizerCrewCompetitorAnalyst")
    def test_crew_exception_captured(self, mock_crew_class, basic_flow):
        """Test that crew exceptions are captured as errors."""
        mock_crew_class.return_value.crew.return_value.kickoff.side_effect = \
            Exception("API error")

        result = basic_flow.analyze_competitors()

        assert result is None
        assert basic_flow.state.has_errors()
        assert "CompetitorAnalyst failed" in basic_flow.state.errors[0]

    @patch("xhs_seo_optimizer.flow.XhsSeoOptimizerCrewCompetitorAnalyst")
    def test_non_pydantic_output_warning(self, mock_crew_class, basic_flow):
        """Test warning when crew returns non-pydantic output."""
        mock_result = MagicMock()
        mock_result.pydantic = None  # No pydantic output
        mock_crew_class.return_value.crew.return_value.kickoff.return_value = \
            mock_result

        result = basic_flow.analyze_competitors()

        assert result is None
        assert basic_flow.state.has_errors()
        assert "did not return Pydantic output" in basic_flow.state.errors[0]
