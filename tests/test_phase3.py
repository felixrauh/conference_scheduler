"""
Tests for Phase 3: Block Scheduling and Room Assignment

Verifies that MILP (Gurobi) and Hungarian (scipy) methods produce
equivalent optimal solutions for the block scheduling problem.
"""

import pytest

from src.phase2 import Block
from src.phase3 import (
    Phase3Input,
    Phase3Result,
    solve_phase3,
    schedule_blocks_milp,
    schedule_blocks_hungarian,
    compute_violation_cost,
    assign_rooms_greedy
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_blocks():
    """Create simple test blocks."""
    return [
        Block(
            block_id="B1",
            block_type="2R2T",
            tuples=[("T01", "T02"), ("T03", "T04")],
            hopping_cost=0
        ),
        Block(
            block_id="B2",
            block_type="2R2T",
            tuples=[("T05", "T06"), ("T07", "T08")],
            hopping_cost=0
        ),
    ]


@pytest.fixture
def simple_timeslots():
    """Create simple timeslots matching the 2R2T block type."""
    return [
        {"id": "TS1", "type_id": "2R2T", "name": "Morning", "start": "09:00"},
        {"id": "TS2", "type_id": "2R2T", "name": "Afternoon", "start": "14:00"},
    ]


@pytest.fixture
def simple_talk_presenter():
    """Map talks to presenters."""
    return {
        "T01": "P1", "T02": "P2", "T03": "P3", "T04": "P4",
        "T05": "P5", "T06": "P6", "T07": "P7", "T08": "P8",
    }


@pytest.fixture
def unavailability_with_conflicts():
    """Presenter unavailability that creates clear preferences."""
    return {
        # P1 and P2 unavailable in morning -> B1 should go to afternoon
        "P1": {"TS1"},
        "P2": {"TS1"},
        # P5 unavailable in afternoon -> B2 should go to morning
        "P5": {"TS2"},
    }


@pytest.fixture
def no_unavailability():
    """No presenter conflicts - any assignment is optimal."""
    return {}


@pytest.fixture
def simple_preferences():
    """Simple participant preferences."""
    return {
        "A1": {"T01", "T05"},
        "A2": {"T02", "T06"},
        "A3": {"T03", "T07"},
    }


@pytest.fixture
def simple_room_capacities():
    """Simple room capacities."""
    return {"R1": 100, "R2": 50}


# =============================================================================
# Tests for violation cost computation
# =============================================================================

def test_compute_violation_cost_no_conflicts(simple_blocks, simple_talk_presenter, no_unavailability):
    """No unavailability should mean zero violations."""
    block = simple_blocks[0]
    cost = compute_violation_cost(
        block, "TS1", simple_talk_presenter, no_unavailability)
    assert cost == 0


def test_compute_violation_cost_with_conflicts(simple_blocks, simple_talk_presenter, unavailability_with_conflicts):
    """Should count violations correctly."""
    block = simple_blocks[0]  # Contains T01-T04, presented by P1-P4

    # P1 and P2 unavailable in TS1 -> 2 violations
    cost_ts1 = compute_violation_cost(
        block, "TS1", simple_talk_presenter, unavailability_with_conflicts
    )
    assert cost_ts1 == 2

    # No one unavailable in TS2 for this block
    cost_ts2 = compute_violation_cost(
        block, "TS2", simple_talk_presenter, unavailability_with_conflicts
    )
    assert cost_ts2 == 0


# =============================================================================
# Tests for Hungarian method
# =============================================================================

def test_hungarian_basic(simple_blocks, simple_timeslots, simple_talk_presenter, no_unavailability):
    """Hungarian should work with no conflicts."""
    result = schedule_blocks_hungarian(
        simple_blocks, simple_timeslots,
        simple_talk_presenter, no_unavailability
    )

    assert len(result) == 2
    assigned_blocks = {r[0].block_id for r in result}
    assigned_slots = {r[1]["id"] for r in result}

    assert assigned_blocks == {"B1", "B2"}
    assert assigned_slots == {"TS1", "TS2"}


def test_hungarian_with_conflicts(
    simple_blocks, simple_timeslots, simple_talk_presenter, unavailability_with_conflicts
):
    """Hungarian should minimize violations."""
    result = schedule_blocks_hungarian(
        simple_blocks, simple_timeslots,
        simple_talk_presenter, unavailability_with_conflicts
    )

    total_violations = sum(v for _, _, v in result)
    assignment = {r[0].block_id: r[1]["id"] for r in result}

    # Optimal: B1 -> TS2, B2 -> TS1 (0 total violations)
    # Suboptimal: B1 -> TS1, B2 -> TS2 (2 + 1 = 3 violations)
    assert total_violations == 0
    assert assignment["B1"] == "TS2"
    assert assignment["B2"] == "TS1"


# =============================================================================
# Tests comparing MILP and Hungarian
# =============================================================================

def test_milp_hungarian_equivalent_with_conflicts(
    simple_blocks, simple_timeslots, simple_talk_presenter, unavailability_with_conflicts
):
    """Both methods should find the same optimal violation count and assignment."""
    try:
        milp_result = schedule_blocks_milp(
            simple_blocks, simple_timeslots,
            simple_talk_presenter, unavailability_with_conflicts,
            verbose=False
        )
    except ImportError:
        pytest.skip("Gurobi not available")

    hungarian_result = schedule_blocks_hungarian(
        simple_blocks, simple_timeslots,
        simple_talk_presenter, unavailability_with_conflicts
    )

    milp_violations = sum(v for _, _, v in milp_result)
    hungarian_violations = sum(v for _, _, v in hungarian_result)

    assert milp_violations == hungarian_violations

    milp_assignment = {r[0].block_id: r[1]["id"] for r in milp_result}
    hungarian_assignment = {r[0].block_id: r[1]["id"] for r in hungarian_result}

    assert milp_assignment == hungarian_assignment


def test_solve_phase3_method_parameter(
    simple_blocks, simple_timeslots, simple_talk_presenter,
    no_unavailability, simple_preferences, simple_room_capacities
):
    """Test solve_phase3 with both method options."""
    phase3_input = Phase3Input(
        blocks=simple_blocks,
        timeslots=simple_timeslots,
        room_capacities=simple_room_capacities,
        talk_presenter=simple_talk_presenter,
        presenter_unavailability=no_unavailability,
        preferences=simple_preferences
    )

    result_hungarian = solve_phase3(
        phase3_input, method="hungarian", verbose=False)
    assert isinstance(result_hungarian, Phase3Result)
    assert len(result_hungarian.assignments) == 2

    try:
        result_milp = solve_phase3(phase3_input, method="milp", verbose=False)
        assert isinstance(result_milp, Phase3Result)
        assert len(result_milp.assignments) == 2
        assert result_milp.total_violations == result_hungarian.total_violations
    except ImportError:
        pytest.skip("Gurobi not available")


def test_solve_phase3_invalid_method():
    """Test that invalid method raises error."""
    phase3_input = Phase3Input(
        blocks=[],
        timeslots=[],
        room_capacities={},
        talk_presenter={},
        presenter_unavailability={},
        preferences={}
    )

    with pytest.raises(ValueError, match="Unknown method"):
        solve_phase3(phase3_input, method="invalid")


# =============================================================================
# Tests for room assignment
# =============================================================================

def test_room_assignment_greedy(simple_blocks, simple_preferences, simple_room_capacities):
    """Test greedy room assignment."""
    block = simple_blocks[0]

    room_mapping, gap = assign_rooms_greedy(
        block, simple_room_capacities, simple_preferences
    )

    # Should have 2 rooms assigned (block has 2R2T)
    assert len(room_mapping) == 2
    assert set(room_mapping.values()) == {"R1", "R2"}


# =============================================================================
# Larger random test
# =============================================================================

def test_milp_hungarian_equivalent_larger():
    """Test with a larger instance to ensure MILP–Hungarian equivalence holds."""
    import random
    random.seed(42)

    n_blocks = 5

    blocks = []
    talk_id = 1
    talk_presenter = {}
    for i in range(n_blocks):
        tuples = []
        for j in range(3):  # 3 tuples per block
            t = tuple(f"T{talk_id + k:03d}" for k in range(4))
            for tid in t:
                talk_presenter[tid] = f"P{talk_id}"
                talk_id += 1
            tuples.append(t)
        blocks.append(
            Block(block_id=f"B{i+1}", block_type="4R3T", tuples=tuples, hopping_cost=0))

    timeslots = [{"id": f"TS{i+1}", "type_id": "4R3T", "name": f"Slot {i+1}"}
                 for i in range(n_blocks)]

    presenters = list(set(talk_presenter.values()))
    unavailability = {}
    for p in random.sample(presenters, min(10, len(presenters))):
        unavailability[p] = {random.choice([ts["id"] for ts in timeslots])}

    hungarian_result = schedule_blocks_hungarian(
        blocks, timeslots, talk_presenter, unavailability
    )

    try:
        milp_result = schedule_blocks_milp(
            blocks, timeslots, talk_presenter, unavailability, verbose=False
        )
    except ImportError:
        pytest.skip("Gurobi not available")

    milp_violations = sum(v for _, _, v in milp_result)
    hungarian_violations = sum(v for _, _, v in hungarian_result)

    assert milp_violations == hungarian_violations, \
        f"MILP found {milp_violations} violations, Hungarian found {hungarian_violations}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
