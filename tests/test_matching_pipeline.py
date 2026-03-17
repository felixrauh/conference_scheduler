#!/usr/bin/env python
"""
Test the matching pipeline with synthetic data.

Creates a small problem instance and runs the matching pipeline
to verify all phases work correctly.
"""

from src.matching_pipeline import (
    run_matching_pipeline,
    solve_phase_a,
    solve_phase_b,
    solve_phase_c,
    compute_tuple_cost_talk_level,
    build_co_preference_matrix
)
from src.instance import build_instance
from src.data_loader import ConferenceData
import pandas as pd
import sys
sys.path.insert(0, '.')


def create_test_instance(n_talks=24, n_rooms=4, k=3, n_blocks=2):
    """
    Create a synthetic test instance.

    Args:
        n_talks: Total number of talks
        n_rooms: Number of parallel rooms
        k: Talks per session (3 or 4)
        n_blocks: Number of time blocks
    """
    # Verify n_talks = n_rooms * k * n_blocks
    expected = n_rooms * k * n_blocks
    assert n_talks == expected, f"n_talks={n_talks} != n_rooms*k*n_blocks={expected}"

    # Create talks
    talks_df = pd.DataFrame({
        "talk_id": [f"T{i:02d}" for i in range(1, n_talks + 1)],
        "title": [f"Talk {i}" for i in range(1, n_talks + 1)],
        "presenter_id": [f"P{i:02d}" for i in range(1, n_talks + 1)]
    })

    # Create participants with preferences
    # Create overlapping preference patterns
    n_participants = 20
    participants_df = pd.DataFrame({
        "participant_id": [f"A{i:02d}" for i in range(1, n_participants + 1)]
    })

    # Create preferences: each participant likes some talks
    preference_rows = []
    for p_idx in range(1, n_participants + 1):
        # Each participant likes talks in a pattern
        # e.g., participant 1 likes T01, T02, T05, T06, ...
        liked = []
        for t_idx in range(1, n_talks + 1):
            # Create clusters of interest
            cluster = (t_idx - 1) // 4
            if (p_idx + cluster) % 3 == 0:
                liked.append(f"T{t_idx:02d}")

        for talk_id in liked:
            preference_rows.append({
                "participant_id": f"A{p_idx:02d}",
                "talk_id": talk_id
            })

    preferences_df = pd.DataFrame(preference_rows)

    # Create block types
    block_types = {
        f"{n_rooms}R{k}T": {
            "n": n_rooms,
            "k": k,
            "count": n_blocks
        }
    }

    # Create timeslots
    timeslots = []
    for i in range(n_blocks):
        timeslots.append({
            "id": f"TS{i+1}",
            "type_id": f"{n_rooms}R{k}T",
            "rooms": [f"Room{r}" for r in range(n_rooms)]
        })

    # Create availability (empty - all available)
    availability_df = pd.DataFrame(
        columns=["presenter_id", "unavailable_timeslot"])

    # Create ConferenceData
    data = ConferenceData(
        conference_name="Test Conference",
        rooms=[f"Room{r}" for r in range(n_rooms)],
        block_types=block_types,
        timeslots=timeslots,
        talks=talks_df,
        preferences=preferences_df,
        availability=availability_df,
        room_capacities={f"Room{r}": 50 for r in range(n_rooms)}
    )

    # Validate and build instance
    errors = data.validate()
    if errors:
        print("Validation errors:", errors)

    return build_instance(data)


def test_phase_a():
    """Test Phase A (pair matching)."""
    print("\n" + "=" * 50)
    print("TEST: Phase A - Pair Matching")
    print("=" * 50)

    # 24 talks, 4 rooms, k=3, 2 blocks → 8 room-sessions of 3 talks
    # n_3 = 8 (all 3-talk sessions)
    # M = n_3 + 2*n_4 = 8 + 0 = 8 pairs
    # Singles = 24 - 16 = 8

    instance = create_test_instance(n_talks=24, n_rooms=4, k=3, n_blocks=2)

    n_3 = 8  # 4 rooms * 2 blocks
    n_4 = 0

    pairs, singles, weight, time_spent = solve_phase_a(
        instance, n_3, n_4,
        time_limit=30.0,
        verbose=True
    )

    assert len(pairs) == 8, f"Expected 8 pairs, got {len(pairs)}"
    assert len(singles) == 8, f"Expected 8 singles, got {len(singles)}"

    # All talks should be accounted for
    matched_talks = set()
    for p in pairs:
        matched_talks.add(p[0])
        matched_talks.add(p[1])
    for s in singles:
        matched_talks.add(s)

    assert len(
        matched_talks) == 24, f"Not all talks covered: {len(matched_talks)}"

    print(
        f"✓ Phase A passed: {len(pairs)} pairs, {len(singles)} singles, weight={weight}")
    return True


def test_phase_b():
    """Test Phase B (block formation)."""
    print("\n" + "=" * 50)
    print("TEST: Phase B - Block Formation")
    print("=" * 50)

    instance = create_test_instance(n_talks=24, n_rooms=4, k=3, n_blocks=2)

    n_3 = 8
    n_4 = 0

    # First run Phase A
    pairs, singles, _, _ = solve_phase_a(
        instance, n_3, n_4, time_limit=30.0, verbose=False)

    # Now test Phase B
    blocks_3, blocks_4, weight, time_spent = solve_phase_b(
        pairs, singles, n_3, n_4,
        instance.preferences,
        time_limit=30.0,
        verbose=True
    )

    assert len(blocks_3) == 8, f"Expected 8 3-blocks, got {len(blocks_3)}"
    assert len(blocks_4) == 0, f"Expected 0 4-blocks, got {len(blocks_4)}"

    # Each block should have 3 talks
    for block in blocks_3:
        assert len(block) == 3, f"Block has wrong size: {len(block)}"

    print(
        f"✓ Phase B passed: {len(blocks_3)} 3-blocks, {len(blocks_4)} 4-blocks, marginal={weight}")
    return True


def test_full_pipeline():
    """Test the full matching pipeline."""
    print("\n" + "=" * 50)
    print("TEST: Full Matching Pipeline")
    print("=" * 50)

    instance = create_test_instance(n_talks=24, n_rooms=4, k=3, n_blocks=2)

    result, phase3_result = run_matching_pipeline(
        instance,
        time_limit=120.0,
        verbose=True,
        run_phase3=False  # Skip Phase 3 for now (needs timeslots)
    )

    # Verify results
    assert len(result.pairs) == 8
    assert len(result.singles) == 8
    assert len(result.blocks_3) == 8
    assert len(result.blocks_4) == 0

    # 8 3-blocks with 4 rooms = 2 tuples (each with 4 blocks)
    assert len(
        result.tuples_3) == 2, f"Expected 2 tuples, got {len(result.tuples_3)}"

    print(f"\n✓ Full pipeline passed!")
    print(f"  - Pairs: {len(result.pairs)}, weight={result.phase_a_weight}")
    print(
        f"  - 3-blocks: {len(result.blocks_3)}, marginal={result.phase_b_weight}")
    print(f"  - Tuples: {len(result.tuples_3)}, missed={result.phase_c_cost}")
    print(
        f"  - Blocks: {len(result.ordered_blocks)}, benefit={result.phase_d_benefit}")
    print(f"  - Total time: {result.total_time:.2f}s")

    return True


def test_tuple_cost():
    """Test tuple cost computation."""
    print("\n" + "=" * 50)
    print("TEST: Tuple Cost Computation")
    print("=" * 50)

    # Simple test case
    blocks = (
        ("T1", "T2", "T3"),  # Block 1
        ("T4", "T5", "T6"),  # Block 2
    )

    preferences = {
        "A1": {"T1", "T2"},       # 2 in block 1, 0 in block 2 → missed = 0
        # 1 in each → missed = 1 (picks block 1, misses T4)
        "A2": {"T1", "T4"},
        # 1 in block 1, 2 in block 2 → missed = 1 (picks block 2)
        "A3": {"T1", "T4", "T5"},
        "A4": {"T7"},             # 0 in both → missed = 0
    }

    cost = compute_tuple_cost_talk_level(blocks, preferences)
    expected = 0 + 1 + 1 + 0  # = 2

    assert cost == expected, f"Expected cost {expected}, got {cost}"
    print(f"✓ Tuple cost test passed: cost={cost}")
    return True


def create_test_instance_variable_n(n_talks, block_types_config):
    """
    Create a synthetic test instance with variable n (room counts).

    Args:
        n_talks: Total number of talks
        block_types_config: Dict of {type_id: {"n": rooms, "k": talks, "count": blocks}}
    """
    # Verify n_talks matches config
    expected = sum(bt["n"] * bt["k"] * bt["count"]
                   for bt in block_types_config.values())
    assert n_talks == expected, f"n_talks={n_talks} != expected={expected}"

    # Determine max rooms
    max_rooms = max(bt["n"] for bt in block_types_config.values())

    # Create talks
    talks_df = pd.DataFrame({
        "talk_id": [f"T{i:02d}" for i in range(1, n_talks + 1)],
        "title": [f"Talk {i}" for i in range(1, n_talks + 1)],
        "presenter_id": [f"P{i:02d}" for i in range(1, n_talks + 1)]
    })

    # Create participants with preferences
    n_participants = 20
    participants_df = pd.DataFrame({
        "participant_id": [f"A{i:02d}" for i in range(1, n_participants + 1)]
    })

    # Create preferences
    preference_rows = []
    for p_idx in range(1, n_participants + 1):
        for t_idx in range(1, n_talks + 1):
            cluster = (t_idx - 1) // 4
            if (p_idx + cluster) % 3 == 0:
                preference_rows.append({
                    "participant_id": f"A{p_idx:02d}",
                    "talk_id": f"T{t_idx:02d}"
                })

    preferences_df = pd.DataFrame(preference_rows)

    # Create timeslots
    timeslots = []
    slot_idx = 0
    for type_id, bt in block_types_config.items():
        for _ in range(bt["count"]):
            timeslots.append({
                "id": f"TS{slot_idx+1}",
                "type_id": type_id,
                "rooms": [f"Room{r}" for r in range(bt["n"])]
            })
            slot_idx += 1

    # Create availability (empty)
    availability_df = pd.DataFrame(
        columns=["presenter_id", "unavailable_timeslot"])

    # Create ConferenceData
    data = ConferenceData(
        conference_name="Test Conference Variable N",
        rooms=[f"Room{r}" for r in range(max_rooms)],
        block_types=block_types_config,
        timeslots=timeslots,
        talks=talks_df,
        preferences=preferences_df,
        availability=availability_df,
        room_capacities={f"Room{r}": 50 for r in range(max_rooms)}
    )

    errors = data.validate()
    if errors:
        print("Validation errors:", errors)

    return build_instance(data)


def test_variable_n_pipeline():
    """Test pipeline with variable n (different room counts per block)."""
    print("\n" + "=" * 50)
    print("TEST: Variable N Pipeline")
    print("=" * 50)

    # Create a problem with:
    # - 2 blocks with 4 rooms, k=3 (2 * 4 * 3 = 24 talks)
    # - 1 block with 3 rooms, k=3 (1 * 3 * 3 = 9 talks)
    # Total: 33 talks

    block_types_config = {
        "4R3T": {"n": 4, "k": 3, "count": 2},
        "3R3T": {"n": 3, "k": 3, "count": 1}
    }
    n_talks = 2*4*3 + 1*3*3  # 33 talks

    instance = create_test_instance_variable_n(n_talks, block_types_config)

    print(f"Created instance: {n_talks} talks")
    print(f"  Block types: {block_types_config}")

    # Run the full pipeline
    result, phase3_result = run_matching_pipeline(
        instance,
        time_limit=60.0,
        verbose=True,
        run_phase3=False
    )

    # Verify output
    assert result is not None, "Pipeline should return a result"

    # Check blocks created
    n_blocks_expected = 2 + 1  # 2 4R3T blocks + 1 3R3T block
    assert len(result.ordered_blocks) == n_blocks_expected, \
        f"Expected {n_blocks_expected} blocks, got {len(result.ordered_blocks)}"

    # Verify all talks are scheduled
    scheduled_talks = set()
    for block in result.ordered_blocks:
        for tuple_ in block.tuples:
            for talk in tuple_:
                scheduled_talks.add(talk)

    assert len(scheduled_talks) == n_talks, \
        f"Expected {n_talks} scheduled talks, got {len(scheduled_talks)}"

    # Verify block sizes (tuple lengths)
    tuple_sizes = {}
    for block in result.ordered_blocks:
        n = len(block.tuples[0])  # Size of first tuple = number of rooms
        tuple_sizes[n] = tuple_sizes.get(n, 0) + 1

    print(f"  Tuple sizes: {tuple_sizes}")

    # We should have blocks with 4-tuples and 3-tuples
    assert 4 in tuple_sizes, "Expected some blocks with 4 rooms"
    assert 3 in tuple_sizes, "Expected some blocks with 3 rooms"
    assert tuple_sizes[
        4] == 2, f"Expected 2 blocks with 4 rooms, got {tuple_sizes.get(4, 0)}"
    assert tuple_sizes[
        3] == 1, f"Expected 1 block with 3 rooms, got {tuple_sizes.get(3, 0)}"

    print(f"✓ Variable N pipeline test passed!")
    print(f"  Total cost: {result.phase_c_cost}")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("MATCHING PIPELINE TESTS")
    print("=" * 70)

    all_passed = True

    try:
        all_passed &= test_tuple_cost()
        all_passed &= test_phase_a()
        all_passed &= test_phase_b()
        all_passed &= test_full_pipeline()
        all_passed &= test_variable_n_pipeline()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 70)

    sys.exit(0 if all_passed else 1)
