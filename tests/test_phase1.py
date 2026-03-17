#!/usr/bin/env python3
"""
Test script for Phase 1 optimizer.

Creates a small test instance and runs the Phase 1 optimization.
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phase1 import Phase1Optimizer, solve_phase1
from src.instance import ProblemInstance, build_instance
from src.data_loader import ConferenceData


def create_test_instance() -> ProblemInstance:
    """
    Create a small test instance for Phase 1.

    Configuration:
    - 12 talks (T001-T012)
    - 8 participants (P001-P008) with varied preferences
    - Block types:
        - 'morning': n=4, k=2, count=1 -> 8 slots (2 tuples of size 4)
        - 'lunch': n=1, k=1, count=1 -> 1 slot (1 tuple of size 1)  
        - 'afternoon': n=3, k=1, count=1 -> 3 slots (1 tuple of size 3)
    - Total: 8 + 1 + 3 = 12 slots = 12 talks ✓
    """
    # Talks
    talks = [f'T{str(i).zfill(3)}' for i in range(1, 13)]
    talks_df = pd.DataFrame({
        'talk_id': talks,
        'title': [f'Talk {i}' for i in range(1, 13)],
        # All different presenters
        'presenter_id': [f'S{str(i).zfill(2)}' for i in range(1, 13)],
        'track': ['AI', 'ML', 'AI', 'ML', 'Systems', 'AI', 'ML', 'Systems', 'AI', 'ML', 'Systems', 'AI']
    })

    # Participants
    participants = [f'P{str(i).zfill(3)}' for i in range(1, 9)]
    participants_df = pd.DataFrame({
        'participant_id': participants,
        'name': [f'Participant {i}' for i in range(1, 9)],
        'email': [f'p{i}@example.com' for i in range(1, 9)]
    })

    # Preferences - varied to create interesting conflicts
    pref_data = [
        ('P001', ['T001', 'T002', 'T005', 'T009']),  # Wants 4 talks
        ('P002', ['T003', 'T004', 'T006', 'T010']),  # Wants 4 talks
        ('P003', ['T001', 'T003', 'T007', 'T011']),  # Wants 4 talks
        ('P004', ['T002', 'T004', 'T008', 'T012']),  # Wants 4 talks
        ('P005', ['T005', 'T006', 'T009', 'T010']),  # Wants 4 talks
        ('P006', ['T001', 'T005', 'T009']),          # Wants 3 talks
        ('P007', ['T003', 'T007', 'T010', 'T011', 'T012']),  # Wants 5 talks
        ('P008', ['T002', 'T004', 'T006', 'T008']),  # Wants 4 talks
    ]

    pref_rows = []
    for pid, tids in pref_data:
        for tid in tids:
            pref_rows.append({'participant_id': pid, 'talk_id': tid})
    preferences_df = pd.DataFrame(pref_rows)

    # Block types
    block_types = {
        # 4 parallel × 2 talks = 8 slots -> 2 tuples of 4
        'morning': {'n': 4, 'k': 2, 'count': 1},
        # 1 parallel × 1 talk = 1 slot -> 1 tuple of 1
        'lunch': {'n': 1, 'k': 1, 'count': 1},
        # 3 parallel × 1 talk = 3 slots -> 1 tuple of 3
        'afternoon': {'n': 3, 'k': 1, 'count': 1}
    }
    # Tuple types derived: {4: 2, 1: 1, 3: 1} -> total slots = 4*2 + 1*1 + 3*1 = 12 ✓

    # Timeslots (minimal - not used in Phase 1)
    timeslots = [
        {'id': 'TS1', 'start_time': 'Mon 09:00',
            'type_id': 'morning', 'rooms': ['R1', 'R2', 'R3', 'R4']},
        {'id': 'TS2', 'start_time': 'Mon 12:00',
            'type_id': 'lunch', 'rooms': ['R1']},
        {'id': 'TS3', 'start_time': 'Mon 14:00',
            'type_id': 'afternoon', 'rooms': ['R1', 'R2', 'R3']},
    ]

    # Availability (empty - no unavailability constraints)
    availability_df = pd.DataFrame(
        columns=['presenter_id', 'unavailable_timeslot'])

    # Build preference matrix
    preference_matrix = {}
    for _, row in preferences_df.iterrows():
        pid = row['participant_id']
        tid = row['talk_id']
        if pid not in preference_matrix:
            preference_matrix[pid] = set()
        preference_matrix[pid].add(tid)

    # Create ConferenceData
    conference_data = ConferenceData(
        conference_name="Test Conference",
        rooms=['R1', 'R2', 'R3', 'R4'],
        block_types=block_types,
        timeslots=timeslots,
        talks=talks_df,
        preferences=preferences_df,
        availability=availability_df,
        preference_matrix=preference_matrix,
        presenter_unavailability={}
    )

    # Build problem instance
    instance = build_instance(conference_data)

    return instance


def test_phase1():
    """Run Phase 1 test."""
    print("=" * 70)
    print("PHASE 1 TEST")
    print("=" * 70)

    # Create test instance
    instance = create_test_instance()

    print(f"\nProblem Configuration:")
    print(f"  Talks: {len(instance.talks)}")
    print(f"  Participants: {len(instance.participants)}")
    print(
        f"  Total Preferences: {sum(len(p) for p in instance.preferences.values())}")
    print(f"\nBlock Types:")
    for type_id, spec in instance.block_types.items():
        total_slots = spec['n'] * spec['k'] * spec['count']
        print(
            f"  {type_id}: n={spec['n']}, k={spec['k']}, count={spec['count']} -> {total_slots} slots")

    # Solve Phase 1
    print("\n" + "-" * 70)
    print("Running Phase 1 Optimization...")
    print("-" * 70)

    result = solve_phase1(instance, time_limit=60.0, verbose=True)

    print("\n" + "-" * 70)
    print("SOLUTION SUMMARY")
    print("-" * 70)

    print(f"\nSelected {len(result)} tuples:")
    for i, ntuple in enumerate(result, 1):
        cost = instance.compute_tuple_cost(ntuple)
        print(f"  {i}. Size {len(ntuple)}: {ntuple} (missed: {cost})")

    # Verify all talks are covered
    covered_talks = set()
    for ntuple in result:
        covered_talks.update(ntuple)

    missing = set(instance.talks) - covered_talks
    extra = covered_talks - set(instance.talks)

    print(f"\nVerification:")
    print(f"  All talks covered: {len(missing) == 0}")
    if missing:
        print(f"  Missing talks: {missing}")
    if extra:
        print(f"  Extra talks: {extra}")

    return result


def test_phase1_optimizer_class():
    """Test Phase1Optimizer class directly."""
    print("\n" + "=" * 70)
    print("PHASE 1 OPTIMIZER CLASS TEST")
    print("=" * 70)

    instance = create_test_instance()

    # Filter: no same-presenter constraint (already different in test data)
    def is_feasible(ntuple):
        return not instance.talks_have_same_presenter(ntuple)

    with Phase1Optimizer() as optimizer:
        optimizer.set_problem_instance(instance)

        print(f"\nDerived tuple types:")
        for n_tau, p_tau in optimizer.tuple_types:
            print(f"  Size {n_tau}: need {p_tau} tuples")

        optimizer.build_model(
            filter_fn=is_feasible,
            time_limit=60.0,
            verbose=True
        )

        status = optimizer.solve()

        optimizer.display_results(detailed=True)

        # Get results in different formats
        result = optimizer.get_result()
        result_by_size = optimizer.get_result_by_size()
        summary = optimizer.get_summary()

        print("\n" + "-" * 70)
        print("Result by size:")
        for size, tuples in sorted(result_by_size.items()):
            print(f"  Size {size}: {len(tuples)} tuples")
            for t in tuples:
                print(f"    {t}")

        print("\nSummary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    # Run tests
    test_phase1()
    test_phase1_optimizer_class()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)
