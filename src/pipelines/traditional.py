"""
Traditional 3-Phase Pipeline

Phase 1: Maximize attendance (tuple selection via MILP or column generation)
Phase 2: Minimize room hopping (partition + ordering)
Phase 3: Room assignment (MILP or Hungarian)

Requires Gurobi for Phase 1 and Phase 3 (unless using hungarian for Phase 3).

Supports fixed blocks: pre-specified talk groupings that skip Phase 1 optimization
and are inserted directly in Stage 2 of Phase 2. Useful for special sessions (e.g.,
award or sponsored sessions) where talk groupings are already decided.
"""

import sys
import time
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class FixedSequenceConfig:
    """Configuration for a pre-specified fixed sequence (e.g., an award session).

    A fixed sequence is a list of talks that run SEQUENTIALLY in ONE room
    (one after another across k timeslots). This adds an extra 'column' to
    an existing block, converting e.g. a 4R4T block into a 5R4T block.

    The sequence talks are excluded from Phase 1, then inserted into a block
    after Phase 2 Stage 1 partitioning, before Stage 2 ordering optimization.
    """
    name: str                           # Descriptive name (e.g., "SpecialSession_1")
    talks: List[str]                    # Talk IDs in sequence order (k talks)
    # Block type to attach to (e.g., "4R4T")
    target_block_type: str
    result_block_type: str              # Resulting block type (e.g., "5R4T")


@dataclass
class PipelineConfig:
    """Configuration for traditional pipeline."""
    phase1_method: str = "column_generation"  # explicit, column_generation
    phase1_time_limit: float = 300.0
    phase1_pricing_strategy: str = "auto"
    # Explicit enumeration max_cost threshold (None = auto, lower = more aggressive filtering)
    # For 100+ talks, use 2-4 to keep memory manageable
    phase1_explicit_max_cost: Optional[int] = None
    phase2_partition_strategy: str = "greedy"
    phase2_ordering_strategy: str = "enumerate"
    phase2_local_search_iterations: int = 2000
    phase3_method: str = "milp"  # milp, hungarian
    phase3_time_limit: float = 60.0
    verbose: bool = False
    # Fixed sequences: pre-specified sequential talks for special sessions
    # Each sequence adds one 'column' (room) to an existing block
    fixed_sequences: Optional[List[FixedSequenceConfig]] = None
    # Feasibility retry: max retries when presenter unavailabilities cause infeasibility
    max_feasibility_retries: int = 10


def run_traditional_pipeline(config: PipelineConfig, instance, data) -> Dict[str, Any]:
    """
    Run the traditional 3-phase pipeline.

    Args:
        config: Pipeline configuration
        instance: ProblemInstance with talks, participants, preferences
        data: ConferenceData with raw data

    Returns:
        Dictionary with results including timings and phase3_result
    """
    import gurobipy as gp
    from src.phase2 import Phase2Input, FixedSequence
    from src.phase3 import solve_phase3, Phase3Input

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

            # Adjust block_types for Phase 1:
            # The target_block_type is what we optimize (with 1 less room)
            # The result_block_type is the original block in sessions.xlsx
            # E.g., target=4R4T, result=5R4T means: reduce one 5R4T to 4R4T for Phase 1
            target_type = fs_config.target_block_type  # e.g., "4R4T"
            result_type = fs_config.result_block_type  # e.g., "5R4T"

            if result_type in adjusted_block_types:
                # Decrease count of the result type (the original block)
                adjusted_block_types[result_type]["count"] -= 1
                if adjusted_block_types[result_type]["count"] <= 0:
                    del adjusted_block_types[result_type]

                # Add/increase count of the target type (the reduced block)
                if target_type not in adjusted_block_types:
                    # Parse nRkT format to get n and k
                    import re
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
    # PHASE 1: Maximize attendance (for non-fixed talks only)
    # -------------------------------------------------------------------------
    print("\n[Phase 1] Maximizing attendance...")

    # Use adjusted block types (with reduced rooms for fixed sequence blocks)
    tuple_types = _derive_tuple_types(adjusted_block_types)

    if config.verbose:
        print(f"  Tuple types: {tuple_types}")

    # Filter out fixed sequence talks
    talks_for_phase1 = [
        t for t in instance.talks if t not in fixed_sequence_talks]

    if config.verbose and fixed_sequence_talks:
        print(
            f"  Talks for Phase 1: {len(talks_for_phase1)} (excluded {len(fixed_sequence_talks)} fixed)")

    # Add placeholders if needed
    total_slots = sum(
        n * count for n, count in tuple_types) if tuple_types else 0
    if len(talks_for_phase1) < total_slots:
        n_placeholders = total_slots - len(talks_for_phase1)
        print(f"  Adding {n_placeholders} placeholder slots")
        for i in range(n_placeholders):
            talks_for_phase1.append(f"PLACEHOLDER_{i+1}")

    phase1_start = time.time()

    if total_slots == 0:
        # All blocks are fixed - no Phase 1 needed
        tuples_by_size = {}
        phase1_obj = 0.0
        print("  ✓ Skipped (all blocks pre-specified)")
    elif config.phase1_method == "column_generation":
        # Need to create adjusted instance for column generation
        tuples_by_size, phase1_obj, phase1_cols = _run_phase1_column_generation(
            talks_for_phase1, instance, tuple_types, config
        )
        results['phase1_columns'] = phase1_cols
    else:
        tuples_by_size, phase1_obj = _run_phase1_explicit(
            instance, talks_for_phase1, adjusted_block_types, config
        )

    results['phase1_time'] = time.time() - phase1_start
    results['phase1_objective'] = phase1_obj

    if total_slots > 0:
        print(f"  ✓ Objective: {phase1_obj:.0f} missed attendances")
        print(f"  ✓ Time: {results['phase1_time']:.1f}s")

    # -------------------------------------------------------------------------
    # PHASE 2: Minimize room hopping
    # -------------------------------------------------------------------------
    print("\n[Phase 2] Minimizing room hopping...")

    # Build block specs from ADJUSTED block types (with reduced rooms for fixed sequences)
    block_specs = []
    for type_id, bt in adjusted_block_types.items():
        if bt["count"] > 0:
            block_specs.append((bt["n"], bt["k"], bt["count"], type_id))

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
        from src.phase2 import solve_phase2
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
    # PHASE 3: Room assignment
    # -------------------------------------------------------------------------
    print("\n[Phase 3] Assigning to timeslots...")

    # Ensure room capacities exist
    if not data.room_capacities:
        data.room_capacities = {room: 100 for room in data.rooms}

    phase3_input = Phase3Input(
        blocks=phase2_result.blocks,
        timeslots=data.timeslots,
        room_capacities=data.room_capacities,
        talk_presenter={row['talk_id']: row['presenter_id']
                        for _, row in data.talks.iterrows()},
        presenter_unavailability=instance.presenter_unavailability,
        preferences=instance.preferences
    )

    phase3_start = time.time()

    if config.phase3_method == "milp":
        phase3_result = solve_phase3(phase3_input, verbose=config.verbose)
    else:
        from src.pipelines.heuristic import solve_phase3_hungarian
        phase3_result = solve_phase3_hungarian(
            phase2_result.blocks, data, instance)

    results['phase3_time'] = time.time() - phase3_start
    results['phase3_result'] = phase3_result
    results['blocks'] = phase2_result.blocks

    print(f"  ✓ Assigned to {len(data.timeslots)} timeslots")
    print(f"  ✓ Time: {results['phase3_time']:.1f}s")

    return results


def _derive_tuple_types(block_types: Dict) -> List[tuple]:
    """Derive tuple requirements from block types."""
    tuple_types = []
    for type_id, spec in block_types.items():
        n = spec['n']
        k = spec['k']
        count = spec['count']
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


def _run_phase1_column_generation(talks, instance, tuple_types, config):
    """Run Phase 1 using column generation."""
    import gurobipy as gp
    from src.columngeneration_phase1.phase1_column_generation_enhanced import Phase1ColumnGenerationEnhanced

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()

        solver = Phase1ColumnGenerationEnhanced(
            env,
            talks,
            instance.participants,
            instance.preferences,
            tuple_types,
            pricing_strategy=config.phase1_pricing_strategy,
            verbose=config.verbose
        )

        result = solver.solve(
            max_iterations=50,
            optimality_gap=1e-4,
            time_limit=config.phase1_time_limit
        )

        if result['status'] != gp.GRB.OPTIMAL:
            raise RuntimeError(
                f"Phase 1 failed with status {result['status']}")

        tuples_by_size = solver.get_result_by_type()
        return tuples_by_size, result['objective'], result['stats']['final_columns']


def _run_phase1_explicit(instance, talks_for_phase1, adjusted_block_types, config):
    """Run Phase 1 using explicit enumeration."""
    from src.phase1 import solve_phase1
    from copy import deepcopy

    # Create a modified instance with placeholder talks and adjusted block types
    modified_instance = deepcopy(instance)
    modified_instance.talks = list(talks_for_phase1)
    modified_instance.block_types = dict(adjusted_block_types)

    # Register placeholder talks so lookups don't fail
    for t in talks_for_phase1:
        if t.startswith("PLACEHOLDER_"):
            modified_instance.talk_presenter[t] = f"PRESENTER_{t}"

    phase1_result = solve_phase1(
        modified_instance,
        time_limit=config.phase1_time_limit,
        verbose=config.verbose,
        method="explicit",
        max_cost=config.phase1_explicit_max_cost
    )

    # Convert to tuples_by_size format
    tuples_by_size = {}
    for t in phase1_result:
        n = len(t)
        if n not in tuples_by_size:
            tuples_by_size[n] = []
        tuples_by_size[n].append(t)

    phase1_obj = sum(
        modified_instance.compute_tuple_cost(t) for t in phase1_result
    )
    return tuples_by_size, phase1_obj
