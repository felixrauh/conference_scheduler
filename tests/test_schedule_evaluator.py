"""
Tests for the Schedule Evaluator module.

Tests all four metrics:
1. Missed attendance
2. Session hops
3. Incoherent sessions
4. Presenter violations
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schedule_evaluator import (
    ScheduleEvaluator,
    EvaluationMetrics,
    evaluate_schedule,
    load_schedule_csv,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def simple_schedule():
    """Create a simple schedule for testing."""
    # 2 timeslots (TS1, TS2), 2 rooms (A, B), 2 slots per room
    data = [
        # Timeslot 1
        {"Session_ID": "TS1", "Block_ID": "B1", "Slot": 1, "Room": "Room A", "Talk_ID": "T01", "Presenter_ID": "P1"},
        {"Session_ID": "TS1", "Block_ID": "B1", "Slot": 1, "Room": "Room B", "Talk_ID": "T02", "Presenter_ID": "P2"},
        {"Session_ID": "TS1", "Block_ID": "B1", "Slot": 2, "Room": "Room A", "Talk_ID": "T03", "Presenter_ID": "P3"},
        {"Session_ID": "TS1", "Block_ID": "B1", "Slot": 2, "Room": "Room B", "Talk_ID": "T04", "Presenter_ID": "P4"},
        # Timeslot 2
        {"Session_ID": "TS2", "Block_ID": "B2", "Slot": 1, "Room": "Room A", "Talk_ID": "T05", "Presenter_ID": "P5"},
        {"Session_ID": "TS2", "Block_ID": "B2", "Slot": 1, "Room": "Room B", "Talk_ID": "T06", "Presenter_ID": "P6"},
        {"Session_ID": "TS2", "Block_ID": "B2", "Slot": 2, "Room": "Room A", "Talk_ID": "T07", "Presenter_ID": "P7"},
        {"Session_ID": "TS2", "Block_ID": "B2", "Slot": 2, "Room": "Room B", "Talk_ID": "T08", "Presenter_ID": "P8"},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def preferences_no_conflict():
    """Preferences with no conflicts - each person wants non-parallel talks."""
    return {
        "User1": {"T01", "T05"},  # Different timeslots
        "User2": {"T03", "T07"},  # Different timeslots
    }


@pytest.fixture
def preferences_with_conflicts():
    """Preferences with conflicts - parallel talks wanted."""
    return {
        "User1": {"T01", "T02"},  # Both at TS1, slot 1 - 1 missed
        "User2": {"T01", "T02", "T03", "T04"},  # All at TS1 - 3 missed (slot1: 1, slot2: 1, wait that's wrong)
        # Actually: slot 1 has T01, T02 parallel -> 1 missed
        #           slot 2 has T03, T04 parallel -> 1 missed
        # But the "timeslot" is TS1 which contains both slots!
        # So at TS1 we have 4 talks parallel, and if all 4 preferred, 3 missed
    }


@pytest.fixture
def preferences_needing_hops():
    """Preferences that require room switches within a block."""
    return {
        # In Block B1: prefers T01 (Room A, slot 1) and T04 (Room B, slot 2)
        # Must hop from A to B -> 1 hop
        "User1": {"T01", "T04"},
        # In Block B1: prefers T02 (Room B, slot 1) and T03 (Room A, slot 2)
        # Must hop from B to A -> 1 hop
        "User2": {"T02", "T03"},
        # No hop needed - both in Room A
        "User3": {"T01", "T03"},
    }


@pytest.fixture
def talk_keywords_coherent():
    """Keywords where sessions are coherent."""
    return {
        "T01": {"Optimization", "ML"},
        "T02": {"Optimization", "VRP"},
        "T03": {"Optimization", "Scheduling"},
        "T04": {"Optimization", "Graphs"},
        "T05": {"Analytics", "ML"},
        "T06": {"Analytics", "Forecasting"},
        "T07": {"Analytics", "Process Mining"},
        "T08": {"Analytics", "Time Series"},
    }


@pytest.fixture
def talk_keywords_incoherent():
    """Keywords where some sessions are incoherent."""
    return {
        "T01": {"Optimization"},
        "T02": {"Analytics"},  # No shared keyword with T01 in Room sessions
        "T03": {"Scheduling"},
        "T04": {"ML"},
        "T05": {"VRP"},
        "T06": {"Graphs"},
        "T07": {"Forecasting"},
        "T08": {"Process Mining"},
    }


@pytest.fixture
def presenter_unavailability():
    """Some presenters unavailable at certain timeslots."""
    return {
        "P1": {"TS1"},  # P1 unavailable at TS1 but presenting T01 at TS1
        "P5": {"TS2"},  # P5 unavailable at TS2 but presenting T05 at TS2
    }


# =============================================================================
# TESTS: MISSED ATTENDANCE
# =============================================================================

class TestMissedAttendance:
    """Tests for missed attendance computation."""
    
    def test_no_conflicts(self, simple_schedule, preferences_no_conflict):
        """No missed attendance when no parallel conflicts."""
        evaluator = ScheduleEvaluator(
            schedule_df=simple_schedule,
            preferences=preferences_no_conflict
        )
        total, per_participant = evaluator.compute_missed_attendance()
        
        assert total == 0
        assert len(per_participant) == 0
    
    def test_simple_conflict(self, simple_schedule):
        """One participant wants two parallel talks -> 1 missed."""
        preferences = {"User1": {"T01", "T02"}}  # Both at same slot
        
        evaluator = ScheduleEvaluator(
            schedule_df=simple_schedule,
            preferences=preferences
        )
        total, per_participant = evaluator.compute_missed_attendance()
        
        # T01 and T02 are at same timeslot TS1 (different slots within, but same Session_ID)
        # Wait, looking at the schedule: both are at TS1, slot 1
        # So 1 missed
        assert total == 1
        assert per_participant.get("User1") == 1
    
    def test_multiple_conflicts(self, simple_schedule):
        """Multiple parallel conflicts across timeslots."""
        preferences = {
            "User1": {"T01", "T02", "T05", "T06"}  # 2 at TS1 slot1, 2 at TS2 slot1
        }
        
        evaluator = ScheduleEvaluator(
            schedule_df=simple_schedule,
            preferences=preferences
        )
        total, per_participant = evaluator.compute_missed_attendance()
        
        # At TS1: T01, T02 parallel -> 1 missed
        # At TS2: T05, T06 parallel -> 1 missed
        # Total: 2 missed
        assert total == 2
        assert per_participant.get("User1") == 2
    
    def test_three_parallel(self):
        """Three parallel talks -> 2 missed."""
        # Create schedule with 3 parallel rooms
        data = [
            {"Session_ID": "TS1", "Block_ID": "B1", "Slot": 1, "Room": "A", "Talk_ID": "T01"},
            {"Session_ID": "TS1", "Block_ID": "B1", "Slot": 1, "Room": "B", "Talk_ID": "T02"},
            {"Session_ID": "TS1", "Block_ID": "B1", "Slot": 1, "Room": "C", "Talk_ID": "T03"},
        ]
        schedule = pd.DataFrame(data)
        preferences = {"User1": {"T01", "T02", "T03"}}
        
        evaluator = ScheduleEvaluator(schedule_df=schedule, preferences=preferences)
        total, _ = evaluator.compute_missed_attendance()
        
        assert total == 2  # Can attend 1, miss 2


# =============================================================================
# TESTS: SESSION HOPS
# =============================================================================

class TestSessionHops:
    """Tests for session hop computation."""
    
    def test_no_hops_same_room(self, simple_schedule):
        """No hops needed when all preferences in same room."""
        preferences = {"User1": {"T01", "T03"}}  # Both in Room A, Block B1
        
        evaluator = ScheduleEvaluator(
            schedule_df=simple_schedule,
            preferences=preferences
        )
        total, per_participant = evaluator.compute_session_hops()
        
        assert total == 0
    
    def test_one_hop_different_rooms(self, simple_schedule):
        """One hop needed when consecutive slots in different rooms."""
        preferences = {"User1": {"T01", "T04"}}  # Room A slot1 -> Room B slot2
        
        evaluator = ScheduleEvaluator(
            schedule_df=simple_schedule,
            preferences=preferences
        )
        total, per_participant = evaluator.compute_session_hops()
        
        assert total == 1
        assert per_participant.get("User1") == 1
    
    def test_multiple_hops(self, simple_schedule, preferences_needing_hops):
        """Multiple users with different hop requirements."""
        evaluator = ScheduleEvaluator(
            schedule_df=simple_schedule,
            preferences=preferences_needing_hops
        )
        total, per_participant = evaluator.compute_session_hops()
        
        # User1: T01 (A) -> T04 (B) = 1 hop
        # User2: T02 (B) -> T03 (A) = 1 hop
        # User3: T01 (A) -> T03 (A) = 0 hops
        assert per_participant.get("User1") == 1
        assert per_participant.get("User2") == 1
        assert "User3" not in per_participant  # 0 hops
        assert total == 2
    
    def test_single_preference_no_hop(self, simple_schedule):
        """Single preference means no hops possible."""
        preferences = {"User1": {"T01"}}
        
        evaluator = ScheduleEvaluator(
            schedule_df=simple_schedule,
            preferences=preferences
        )
        total, _ = evaluator.compute_session_hops()
        
        assert total == 0
    
    def test_optimal_path_choice(self):
        """DP should find optimal path when multiple options exist."""
        # 3 rooms, 3 slots
        data = [
            {"Session_ID": "TS1", "Block_ID": "B1", "Slot": 1, "Room": "A", "Talk_ID": "T01"},
            {"Session_ID": "TS1", "Block_ID": "B1", "Slot": 1, "Room": "B", "Talk_ID": "T02"},
            {"Session_ID": "TS1", "Block_ID": "B1", "Slot": 1, "Room": "C", "Talk_ID": "T03"},
            {"Session_ID": "TS1", "Block_ID": "B1", "Slot": 2, "Room": "A", "Talk_ID": "T04"},
            {"Session_ID": "TS1", "Block_ID": "B1", "Slot": 2, "Room": "B", "Talk_ID": "T05"},
            {"Session_ID": "TS1", "Block_ID": "B1", "Slot": 2, "Room": "C", "Talk_ID": "T06"},
            {"Session_ID": "TS1", "Block_ID": "B1", "Slot": 3, "Room": "A", "Talk_ID": "T07"},
            {"Session_ID": "TS1", "Block_ID": "B1", "Slot": 3, "Room": "B", "Talk_ID": "T08"},
            {"Session_ID": "TS1", "Block_ID": "B1", "Slot": 3, "Room": "C", "Talk_ID": "T09"},
        ]
        schedule = pd.DataFrame(data)
        
        # Prefer: T01 (A), T05 (B), T07 (A)
        # Options: A->B->A = 2 hops
        preferences = {"User1": {"T01", "T05", "T07"}}
        
        evaluator = ScheduleEvaluator(schedule_df=schedule, preferences=preferences)
        total, per_participant = evaluator.compute_session_hops()
        
        assert per_participant.get("User1") == 2


# =============================================================================
# TESTS: INCOHERENT SESSIONS
# =============================================================================

class TestIncoherentSessions:
    """Tests for keyword coherence checking."""
    
    def test_coherent_sessions(self, simple_schedule, talk_keywords_coherent):
        """All sessions share keywords -> no incoherence."""
        evaluator = ScheduleEvaluator(
            schedule_df=simple_schedule,
            preferences={},
            talk_keywords=talk_keywords_coherent
        )
        count, details = evaluator.compute_incoherent_sessions()
        
        # Check each room-session
        # Room A at TS1: T01 (Opt, ML), T03 (Opt, Sched) -> share "Optimization"
        # Room B at TS1: T02 (Opt, VRP), T04 (Opt, Graphs) -> share "Optimization"
        # Room A at TS2: T05 (Ana, ML), T07 (Ana, PM) -> share "Analytics"
        # Room B at TS2: T06 (Ana, Fore), T08 (Ana, TS) -> share "Analytics"
        assert count == 0
    
    def test_incoherent_sessions(self, simple_schedule, talk_keywords_incoherent):
        """Sessions without shared keywords."""
        evaluator = ScheduleEvaluator(
            schedule_df=simple_schedule,
            preferences={},
            talk_keywords=talk_keywords_incoherent
        )
        count, details = evaluator.compute_incoherent_sessions()
        
        # All room-sessions have no shared keywords
        # 4 room-sessions total
        assert count == 4
        assert len(details) == 4
    
    def test_missing_keywords(self, simple_schedule):
        """Handle talks with missing keywords gracefully."""
        keywords = {
            "T01": {"Optimization"},
            # T02 missing - should be treated as empty set
            "T03": {"Optimization"},
        }
        
        evaluator = ScheduleEvaluator(
            schedule_df=simple_schedule,
            preferences={},
            talk_keywords=keywords
        )
        count, details = evaluator.compute_incoherent_sessions()
        
        # Room A TS1: T01 and T03 share Optimization -> coherent
        # Room B TS1: T02 (no keywords), T04 (no keywords) -> incoherent
        # Other rooms also incoherent
        assert count >= 1
    
    def test_single_talk_session_always_coherent(self):
        """Single talk in a session is always coherent."""
        data = [
            {"Session_ID": "TS1", "Block_ID": "B1", "Slot": 1, "Room": "A", "Talk_ID": "T01"},
        ]
        schedule = pd.DataFrame(data)
        keywords = {"T01": {"Optimization"}}
        
        evaluator = ScheduleEvaluator(
            schedule_df=schedule,
            preferences={},
            talk_keywords=keywords
        )
        count, _ = evaluator.compute_incoherent_sessions()
        
        assert count == 0


# =============================================================================
# TESTS: PRESENTER VIOLATIONS
# =============================================================================

class TestPresenterViolations:
    """Tests for presenter availability violations."""
    
    def test_no_violations(self, simple_schedule):
        """No violations when no unavailability."""
        evaluator = ScheduleEvaluator(
            schedule_df=simple_schedule,
            preferences={},
            presenter_unavailability={}
        )
        count, _ = evaluator.compute_presenter_violations()
        
        assert count == 0
    
    def test_single_violation(self, simple_schedule, presenter_unavailability):
        """Detect presenter in unavailable timeslot."""
        evaluator = ScheduleEvaluator(
            schedule_df=simple_schedule,
            preferences={},
            presenter_unavailability=presenter_unavailability,
            talk_presenter={f"T0{i}": f"P{i}" for i in range(1, 9)}
        )
        count, details = evaluator.compute_presenter_violations()
        
        # P1 unavailable at TS1, presenting T01 at TS1 -> violation
        # P5 unavailable at TS2, presenting T05 at TS2 -> violation
        assert count == 2
        
        # Check details contain the violations
        violation_presenters = {d["presenter_id"] for d in details}
        assert "P1" in violation_presenters
        assert "P5" in violation_presenters
    
    def test_presenter_from_schedule(self, simple_schedule, presenter_unavailability):
        """Get presenter from schedule when talk_presenter not provided."""
        evaluator = ScheduleEvaluator(
            schedule_df=simple_schedule,
            preferences={},
            presenter_unavailability=presenter_unavailability,
            talk_presenter={}  # Empty, should fall back to schedule
        )
        count, _ = evaluator.compute_presenter_violations()
        
        # Should still find violations via Presenter_ID column
        assert count == 2


# =============================================================================
# TESTS: FULL EVALUATION
# =============================================================================

class TestFullEvaluation:
    """Tests for complete evaluation pipeline."""
    
    def test_evaluate_returns_metrics(self, simple_schedule, preferences_with_conflicts):
        """evaluate() returns complete EvaluationMetrics."""
        evaluator = ScheduleEvaluator(
            schedule_df=simple_schedule,
            preferences=preferences_with_conflicts
        )
        metrics = evaluator.evaluate()
        
        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.total_missed_attendance > 0
        assert metrics.total_talks == 8
        assert metrics.total_timeslots == 2
    
    def test_metrics_to_dict(self, simple_schedule, preferences_no_conflict):
        """Metrics can be converted to dict for JSON export."""
        evaluator = ScheduleEvaluator(
            schedule_df=simple_schedule,
            preferences=preferences_no_conflict
        )
        metrics = evaluator.evaluate()
        
        d = metrics.to_dict()
        assert "total_missed_attendance" in d
        assert "total_session_hops" in d
        assert "incoherent_sessions" in d
        assert "presenter_violations" in d
        assert "statistics" in d
    
    def test_metrics_str(self, simple_schedule, preferences_no_conflict):
        """Metrics has readable string representation."""
        evaluator = ScheduleEvaluator(
            schedule_df=simple_schedule,
            preferences=preferences_no_conflict
        )
        metrics = evaluator.evaluate()
        
        s = str(metrics)
        assert "SCHEDULE EVALUATION METRICS" in s
        assert "Total Missed Attendance" in s


# =============================================================================
# INTEGRATION TEST WITH REAL DATA (if available)
# =============================================================================

class TestWithRealData:
    """Integration tests using actual project data."""
    
    def test_evaluate_real_schedule(self):
        """Test with real schedule if available."""
        schedule_path = Path("output/schedule.csv")
        if not schedule_path.exists():
            pytest.skip("Real schedule not found")
        
        # Minimal evaluation - just check it runs
        schedule_df = load_schedule_csv(str(schedule_path))
        
        evaluator = ScheduleEvaluator(
            schedule_df=schedule_df,
            preferences={}  # Empty preferences = no missed attendance
        )
        metrics = evaluator.evaluate()
        
        assert metrics.total_talks > 0
        assert metrics.total_timeslots > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
