"""
Heuristic Pipeline (No Gurobi Required)

Phase 1: Greedy tuple construction
Phase 2: Matching-based partition + ordering
Phase 3: Hungarian algorithm for room assignment

Uses scipy for optimization instead of Gurobi.
"""

import time
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass

from src.instance import compute_infeasible_talk_pairs


@dataclass
class FixedSequenceConfig:
    """Configuration for a pre-specified fixed sequence (e.g., an award session).

    A fixed sequence is a list of talks that run SEQUENTIALLY in ONE room.
    This adds an extra 'column' to an existing block.
    """
    name: str                           # Descriptive name (e.g., "SpecialSession_1")
    talks: List[str]                    # Talk IDs in sequence order (k talks)
    # Block type to attach to (e.g., "4R4T")
    target_block_type: str
    result_block_type: str              # Resulting block type (e.g., "5R4T")


@dataclass
class PipelineConfig:
    """Configuration for heuristic pipeline."""
    phase1_time_limit: float = 120.0
    phase2_partition_strategy: str = "matching"
    phase2_ordering_strategy: str = "enumerate"
    phase2_local_search_iterations: int = 100
    verbose: bool = False
    fixed_sequences: Optional[List[FixedSequenceConfig]] = None
    # Feasibility retry: max retries when presenter unavailabilities cause infeasibility
    max_feasibility_retries: int = 10


@dataclass
class RoomAssignment:
    """Assignment of room positions to physical rooms for a block."""
    block: Any
    timeslot: Dict
    room_mapping: Dict[int, str]
    violations: int = 0


@dataclass
class Phase3Result:
    """Complete Phase 3 output."""
    assignments: List[RoomAssignment]
    total_violations: int
    total_capacity_gap: int


def run_heuristic_pipeline(config: PipelineConfig, instance, data) -> Dict[str, Any]:
    """
    Run the heuristic pipeline (no Gurobi required).

    Args:
        config: Pipeline configuration
        instance: ProblemInstance with talks, participants, preferences
        data: ConferenceData with raw data

    Returns:
        Dictionary with results including timings and phase3_result
    """
    from src.phase2 import solve_phase2, Phase2Input, FixedSequence

    results = {}

    # -------------------------------------------------------------------------
    # Handle fixed sequences: exclude their talks from Phase 1
    # and adjust block_types to account for the "missing" room
    # -------------------------------------------------------------------------
    fixed_sequence_talks = set()
    fixed_sequences_for_phase2 = []
    adjusted_block_types = dict(instance.block_types)  # Copy for modification

    if config.fixed_sequences:
        print(
            f"\n[Fixed Sequences] Processing {len(config.fixed_sequences)} pre-specified sequences...")

        for fs_config in config.fixed_sequences:
            # Collect all talk IDs in this fixed sequence
            for talk_id in fs_config.talks:
                fixed_sequence_talks.add(talk_id)

            # Convert to FixedSequence for Phase 2
            fixed_sequences_for_phase2.append(FixedSequence(
                name=fs_config.name,
                talks=tuple(fs_config.talks),
                target_block_type=fs_config.target_block_type,
                result_block_type=fs_config.result_block_type
            ))

            print(f"  ✓ {fs_config.name}: {len(fs_config.talks)} sequential talks "
                  f"({fs_config.target_block_type} → {fs_config.result_block_type})")

            # Adjust block_types for Phase 1
            target_type = fs_config.target_block_type
            result_type = fs_config.result_block_type

            if result_type in adjusted_block_types:
                adjusted_block_types[result_type]["count"] -= 1
                if adjusted_block_types[result_type]["count"] <= 0:
                    del adjusted_block_types[result_type]

                if target_type not in adjusted_block_types:
                    match = re.match(r'(\d+)R(\d+)T', target_type)
                    if match:
                        n, k = int(match.group(1)), int(match.group(2))
                        adjusted_block_types[target_type] = {
                            "n": n, "k": k, "count": 1}
                else:
                    adjusted_block_types[target_type]["count"] += 1

        print(
            f"  ✓ Excluding {len(fixed_sequence_talks)} talks from Phase 1 optimization")
        print(f"  ✓ Adjusted block types for Phase 1: {adjusted_block_types}")

    # -------------------------------------------------------------------------
    # PHASE 1: Greedy tuple construction
    # -------------------------------------------------------------------------
    print("\n[Phase 1] Greedy tuple construction...")

    # Derive tuple types from adjusted block types
    tuple_types = _derive_tuple_types(adjusted_block_types)

    if config.verbose:
        print(f"  Tuple requirements: {tuple_types}")

    # Filter out fixed sequence talks
    talks_for_phase1 = [
        t for t in instance.talks if t not in fixed_sequence_talks]

    # Compute infeasible pairs (presenter availability conflicts)
    infeasible_pairs = compute_infeasible_talk_pairs(
        instance, verbose=config.verbose)
    if infeasible_pairs:
        print(
            f"  📅 {len(infeasible_pairs)} talk pairs excluded (presenter conflicts)")

    # Compute forbidden tuple sizes (talks that can't be in certain size tuples)
    from src.instance import compute_forbidden_tuple_sizes
    forbidden_sizes = compute_forbidden_tuple_sizes(
        instance, verbose=config.verbose)
    if forbidden_sizes:
        print(f"  📅 {len(forbidden_sizes)} talks have tuple size restrictions")

    phase1_start = time.time()

    tuples_by_size = _greedy_phase1(
        talks_for_phase1,
        instance.preferences,
        tuple_types,
        talk_presenter=instance.talk_presenter,
        infeasible_pairs=infeasible_pairs,
        forbidden_sizes=forbidden_sizes,
        max_time=config.phase1_time_limit
    )

    # Compute total cost
    total_cost = sum(
        _compute_tuple_cost(t, instance.preferences)
        for tuples in tuples_by_size.values()
        for t in tuples
    )

    results['phase1_time'] = time.time() - phase1_start
    results['phase1_objective'] = total_cost

    print(f"  ✓ Missed attendances: {total_cost}")
    print(f"  ✓ Time: {results['phase1_time']:.1f}s")

    # -------------------------------------------------------------------------
    # PHASE 2: Room hopping
    # -------------------------------------------------------------------------
    print("\n[Phase 2] Minimizing room hopping...")

    # Use adjusted block types for Phase 2 partitioning
    block_specs = [(bt["n"], bt["k"], bt["count"], tid)
                   for tid, bt in adjusted_block_types.items()]

    phase2_input = Phase2Input(
        tuples_by_n=tuples_by_size,
        block_specs=block_specs,
        preferences=instance.preferences,
        fixed_sequences=fixed_sequences_for_phase2 if fixed_sequences_for_phase2 else None
    )

    phase2_start = time.time()

    # Get all timeslot IDs for feasibility checking
    all_timeslots = {ts['id'] for ts in data.timeslots}
    talk_presenter = {row['talk_id']: row['presenter_id']
                      for _, row in data.talks.iterrows()}

    # Build timeslots_by_type for type-aware feasibility checking
    timeslots_by_type: Dict[str, List[str]] = {}
    for ts in data.timeslots:
        type_id = ts.get('type_id')
        if type_id:
            if type_id not in timeslots_by_type:
                timeslots_by_type[type_id] = []
            timeslots_by_type[type_id].append(ts['id'])

    # Use feasibility-checking Phase 2 if we have presenter unavailabilities
    if instance.presenter_unavailability and config.max_feasibility_retries > 0:
        # Always show this summary (regardless of verbose)
        print(
            f"  📅 Checking {len(instance.presenter_unavailability)} presenter availability constraints")
        from src.phase2 import solve_phase2_with_feasibility_check
        phase2_result, is_feasible = solve_phase2_with_feasibility_check(
            phase2_input,
            talk_presenter=talk_presenter,
            presenter_unavailability=instance.presenter_unavailability,
            all_timeslots=all_timeslots,
            partition_strategy=config.phase2_partition_strategy,
            ordering_strategy=config.phase2_ordering_strategy,
            use_local_search=True,
            local_search_iterations=config.phase2_local_search_iterations,
            max_retries=config.max_feasibility_retries,
            verbose=config.verbose,
            timeslots_by_type=timeslots_by_type
        )
        results['phase2_feasible'] = is_feasible
        if not is_feasible:
            print(
                f"  ⚠ Warning: Could not find fully feasible partition after {config.max_feasibility_retries} retries")
    else:
        phase2_result = solve_phase2(
            phase2_input,
            partition_strategy=config.phase2_partition_strategy,
            ordering_strategy=config.phase2_ordering_strategy,
            use_local_search=True,
            local_search_iterations=config.phase2_local_search_iterations,
            verbose=config.verbose
        )
        results['phase2_feasible'] = True

    results['phase2_time'] = time.time() - phase2_start
    results['phase2_hopping'] = phase2_result.total_hopping

    print(f"  ✓ Created {len(phase2_result.blocks)} blocks")
    print(f"  ✓ Total hopping: {phase2_result.total_hopping}")
    print(f"  ✓ Time: {results['phase2_time']:.1f}s")

    # -------------------------------------------------------------------------
    # PHASE 3: Hungarian assignment
    # -------------------------------------------------------------------------
    print("\n[Phase 3] Assigning to timeslots (Hungarian)...")

    if not data.room_capacities:
        data.room_capacities = {room: 100 for room in data.rooms}

    phase3_start = time.time()
    phase3_result = solve_phase3_hungarian(
        phase2_result.blocks, data, instance)
    results['phase3_time'] = time.time() - phase3_start
    results['phase3_result'] = phase3_result
    results['blocks'] = phase2_result.blocks

    print(f"  ✓ Assigned to {len(data.timeslots)} timeslots")
    print(f"  ✓ Time: {results['phase3_time']:.1f}s")

    return results


def solve_phase3_hungarian(blocks, data, instance) -> Phase3Result:
    """
    Solve Phase 3 using Hungarian algorithm (no Gurobi).

    Assigns blocks to timeslots while:
    1. Respecting block type ↔ timeslot type_id matching (must be exact)
    2. Minimizing presenter constraint violations

    Args:
        blocks: List of blocks from Phase 2
        data: ConferenceData
        instance: ProblemInstance

    Returns:
        Phase3Result with assignments
    """
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    n_blocks = len(blocks)
    timeslots = data.timeslots[:n_blocks]

    talk_presenter = {
        row['talk_id']: row['presenter_id']
        for _, row in data.talks.iterrows()
    }

    # Create cost matrix (violations + type mismatch penalty)
    INCOMPATIBLE_PENALTY = 1000000  # Large penalty for type mismatch
    cost_matrix = []

    # Print type info for debugging
    block_type_counts = {}
    for block in blocks:
        bt = block.block_type
        block_type_counts[bt] = block_type_counts.get(bt, 0) + 1

    timeslot_type_counts = {}
    for ts in timeslots:
        tt = ts.get('type_id', 'unknown')
        timeslot_type_counts[tt] = timeslot_type_counts.get(tt, 0) + 1

    print(f"  Block types: {block_type_counts}")
    print(f"  Timeslot types: {timeslot_type_counts}")

    for block in blocks:
        row = []
        block_type = block.block_type

        for ts in timeslots:
            ts_type = ts.get('type_id', block_type)  # Fallback for legacy

            # Check exact type compatibility: block_type must match type_id
            if block_type != ts_type:
                row.append(INCOMPATIBLE_PENALTY)
                continue

            # Calculate presenter violations
            violations = 0
            for ntuple in block.tuples:
                for talk_id in ntuple:
                    presenter = talk_presenter.get(talk_id)
                    if presenter:
                        unavail = instance.presenter_unavailability.get(
                            presenter, set())
                        ts_id = ts.get('id', ts) if isinstance(
                            ts, dict) else ts
                        if ts_id in unavail:
                            violations += 1
            row.append(violations)
        cost_matrix.append(row)

    # Solve assignment
    cost_array = np.array(cost_matrix)
    row_ind, col_ind = linear_sum_assignment(cost_array)

    # Build assignments
    assignments = []
    rooms = list(data.rooms)[:5]  # Max 5 rooms

    for block_idx, ts_idx in zip(row_ind, col_ind):
        block = blocks[block_idx]
        timeslot = timeslots[ts_idx]
        n_rooms = len(block.tuples[0]) if block.tuples else 0

        # Simple room mapping
        room_mapping = {
            i: rooms[i] if i < len(rooms) else f"Room_{i}"
            for i in range(n_rooms)
        }

        assignments.append(RoomAssignment(
            block=block,
            timeslot=timeslot if isinstance(timeslot, dict) else {
                'id': timeslot},
            room_mapping=room_mapping,
            violations=int(cost_array[block_idx, ts_idx])
        ))

    return Phase3Result(
        assignments=assignments,
        total_violations=int(cost_array[row_ind, col_ind].sum()),
        total_capacity_gap=0
    )


def _derive_tuple_types(block_types: Dict) -> List[Tuple[int, int]]:
    """Derive tuple requirements from block types."""
    tuple_types = []
    for type_id, spec in block_types.items():
        n, k, count = spec['n'], spec['k'], spec['count']
        tuples_needed = k * count
        found = False
        for i, (nt, pt) in enumerate(tuple_types):
            if nt == n:
                tuple_types[i] = (nt, pt + tuples_needed)
                found = True
                break
        if not found:
            tuple_types.append((n, tuples_needed))
    return tuple_types


def _compute_tuple_cost(ntuple: Tuple[str, ...], preferences: Dict[str, Set[str]]) -> int:
    """Compute missed attendance cost for an n-tuple."""
    cost = 0
    for p_id, prefs in preferences.items():
        count = sum(1 for t in ntuple if t in prefs)
        if count > 1:
            cost += count - 1
    return cost


def _build_reverse_index(preferences: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """Build reverse index: talk_id -> participants."""
    talk_to_participants = defaultdict(set)
    for p_id, prefs in preferences.items():
        for talk_id in prefs:
            talk_to_participants[talk_id].add(p_id)
    return dict(talk_to_participants)


def _greedy_phase1(
    talks: List[str],
    preferences: Dict[str, Set[str]],
    tuple_types: List[Tuple[int, int]],
    talk_presenter: Optional[Dict[str, str]] = None,
    infeasible_pairs: Optional[Set[Tuple[str, str]]] = None,
    forbidden_sizes: Optional[Dict[str, Set[int]]] = None,
    max_time: float = 120.0
) -> Dict[int, List[Tuple[str, ...]]]:
    """
    Greedy construction of n-tuples.

    Strategy:
    1. Build pair costs based on co-preference
    2. Add placeholders if needed
    3. Greedily construct tuples minimizing conflict
    4. Respect same-presenter, infeasible pair, and tuple size constraints

    Args:
        talks: List of talk IDs
        preferences: Participant preferences
        tuple_types: List of (tuple_size, count)
        talk_presenter: Mapping talk_id -> presenter_id (for same-presenter check)
        infeasible_pairs: Set of (talk_i, talk_j) pairs to exclude
        forbidden_sizes: Dict mapping talk_id -> set of forbidden tuple sizes
        max_time: Maximum time in seconds

    Returns:
        Dictionary mapping tuple size to list of tuples
    """
    start = time.time()
    infeasible_pairs = infeasible_pairs or set()
    talk_presenter = talk_presenter or {}
    forbidden_sizes = forbidden_sizes or {}

    # Build reverse index and pair costs
    talk_to_participants = _build_reverse_index(preferences)

    pair_costs = {}
    talks_list = list(talks)
    for i, t1 in enumerate(talks_list):
        p1 = talk_to_participants.get(t1, set())
        for t2 in talks_list[i+1:]:
            p2 = talk_to_participants.get(t2, set())
            pair_costs[tuple(sorted([t1, t2]))] = len(p1 & p2)

    # Add placeholders if needed
    total_slots = sum(n * count for n, count in tuple_types)
    all_talks = list(talks)
    if len(all_talks) < total_slots:
        n_placeholders = total_slots - len(all_talks)
        print(f"  Adding {n_placeholders} placeholder slots")
        for i in range(n_placeholders):
            all_talks.append(f"PLACEHOLDER_{i+1}")

    # Greedy construction
    tuples_by_size: Dict[int, List[Tuple[str, ...]]] = {}
    remaining = set(all_talks)

    # Pre-compute how many talks are needed and eligible for each size
    size_needs: Dict[int, int] = {n: n * count for n, count in tuple_types}
    all_sizes = set(size_needs.keys())

    # Compute how many sizes each talk is eligible for
    def eligible_sizes(talk: str) -> Set[int]:
        forbidden = forbidden_sizes.get(talk, set())
        return all_sizes - forbidden

    # Identify talks that are restricted (can't go in the smallest size)
    # These MUST be placed in larger sizes
    smallest_size = min(all_sizes)
    restricted_talks = {
        t for t in remaining if smallest_size in forbidden_sizes.get(t, set())}

    if restricted_talks:
        print(
            f"  📅 {len(restricted_talks)} talks restricted from size-{smallest_size} (must go in larger sizes)")

    # Reserve talks that can ONLY go in one size
    # These must be prioritized for that size
    reserved: Dict[int, Set[str]] = {n: set() for n in all_sizes}
    for talk in remaining:
        sizes = eligible_sizes(talk)
        if len(sizes) == 1:
            only_size = next(iter(sizes))
            reserved[only_size].add(talk)

    if any(reserved[n] for n in all_sizes):
        print(f"  📅 Reserved talks (can only go in one size):")
        for n in sorted(reserved.keys(), reverse=True):
            if reserved[n]:
                print(
                    f"      Size-{n}: {len(reserved[n])} talks ({', '.join(sorted(reserved[n])[:5])}{'...' if len(reserved[n]) > 5 else ''})")

    # Process larger tuples first
    for n, count in sorted(tuple_types, key=lambda x: -x[0]):
        tuples_by_size[n] = []

        for _ in range(count):
            if len(remaining) < n:
                break

            if time.time() - start > max_time:
                print(f"  ⚠ Phase 1 time limit reached")
                break

            # Filter candidates: only talks that CAN be in size-n tuples
            eligible = [
                t for t in remaining if n not in forbidden_sizes.get(t, set())]

            # Also exclude reserved talks for OTHER sizes (they should be saved for their required size)
            other_reserved = set()
            for other_n, talks_set in reserved.items():
                if other_n != n:
                    other_reserved.update(talks_set & remaining)

            # If we have enough eligible talks without touching reserved ones, exclude them
            non_reserved_eligible = [
                t for t in eligible if t not in other_reserved]
            if len(non_reserved_eligible) >= n:
                eligible = non_reserved_eligible
            # Otherwise we have to use some reserved talks

            if not eligible:
                # Fallback: all remaining talks (shouldn't happen with proper constraints)
                eligible = list(remaining)

            # PRIORITY: If there are restricted talks still remaining, start with one of them
            # This ensures restricted talks get placed in larger sizes before we run out of slots
            restricted_remaining = [
                t for t in eligible if t in restricted_talks]

            if restricted_remaining:
                # Start with a restricted talk (least popular among restricted)
                candidates = sorted(
                    restricted_remaining,
                    key=lambda t: len(talk_to_participants.get(t, set()))
                )
            else:
                # No restricted talks left, use normal selection (least popular)
                candidates = sorted(
                    eligible,
                    key=lambda t: len(talk_to_participants.get(t, set()))
                )

            current = [candidates[0]]
            # Pool: remaining talks that are eligible for this size
            pool = [t for t in eligible if t != current[0]]

            # Helper to check if a talk can be added to current tuple
            def can_add(t: str) -> bool:
                # Check tuple size restriction
                if n in forbidden_sizes.get(t, set()):
                    return False
                # Check same presenter
                if talk_presenter:
                    t_presenter = talk_presenter.get(t)
                    for c in current:
                        if talk_presenter.get(c) == t_presenter:
                            return False
                # Check infeasible pairs
                for c in current:
                    pair = tuple(sorted([t, c]))
                    if pair in infeasible_pairs:
                        return False
                return True

            # Greedily add talks with minimum conflict
            while len(current) < n and pool:
                # Filter pool to only feasible candidates
                feasible = [t for t in pool if can_add(t)]
                if not feasible:
                    # No feasible candidates - try adding from all remaining (not just eligible)
                    fallback = [
                        t for t in remaining if t not in current and can_add(t)]
                    if fallback:
                        feasible = fallback
                    else:
                        # Last resort: take any to complete the tuple
                        feasible = [t for t in pool if t not in current]
                        if not feasible:
                            break  # Can't complete this tuple
                best = min(feasible, key=lambda t: sum(
                    pair_costs.get(tuple(sorted([t, c])), 0) for c in current
                ))
                current.append(best)
                pool.remove(best)

            if len(current) == n:
                tuples_by_size[n].append(tuple(current))
                for t in current:
                    remaining.discard(t)

    return tuples_by_size
