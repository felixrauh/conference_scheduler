"""
Phase 3: Block Scheduling and Room Assignment

Two sub-problems:
1. Assign blocks to timeslots (minimize presenter availability violations)
2. Assign room positions to physical rooms (maximize capacity gap)

Methods:
- Block scheduling: "milp" (Gurobi) or "hungarian" (scipy)
- Room assignment: greedy (provably optimal, no solver needed)
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field

from .phase2 import Block


# Lazy import for Gurobi (may not be available)
def _import_gurobi():
    import gurobipy as gp
    from gurobipy import GRB
    return gp, GRB


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Phase3Input:
    """Input data for Phase 3."""
    blocks: List[Block]
    timeslots: List[Dict]  # Each has 'id', 'start_time', etc.
    room_capacities: Dict[str, int]  # room_id -> capacity
    talk_presenter: Dict[str, str]  # talk_id -> presenter_id
    # presenter_id -> set of timeslot_ids
    presenter_unavailability: Dict[str, Set[str]]
    preferences: Dict[str, Set[str]]  # participant_id -> set of talk_ids


@dataclass
class RoomAssignment:
    """Assignment of room positions to physical rooms for a block."""
    block: Block
    timeslot: Dict
    room_mapping: Dict[int, str]  # position (0-indexed) -> room_id
    violations: int  # number of presenter availability violations


@dataclass
class Phase3Result:
    """Complete Phase 3 output."""
    assignments: List[RoomAssignment]
    total_violations: int
    total_capacity_gap: int


# =============================================================================
# ROOM SESSION INTEREST
# =============================================================================

def compute_room_session_interest(
    block: Block,
    room_position: int,
    preferences: Dict[str, Set[str]]
) -> int:
    """
    Compute the number of participants interested in a room session.

    A room session is all talks in the same room position across all tuples
    in a block. Interest = participants who like at least one of these talks.

    Args:
        block: The block
        room_position: 0-indexed position within each tuple
        preferences: participant_id -> set of preferred talk_ids

    Returns:
        Number of interested participants
    """
    # Get all talks in this room position
    talks_in_room = set()
    for ntuple in block.tuples:
        if room_position < len(ntuple):
            talks_in_room.add(ntuple[room_position])

    # Count participants who like at least one
    interested = 0
    for participant, prefs in preferences.items():
        if prefs & talks_in_room:  # intersection is non-empty
            interested += 1

    return interested


# =============================================================================
# ROOM ASSIGNMENT (GREEDY OPTIMAL)
# =============================================================================

def assign_rooms_greedy(
    block: Block,
    room_capacities: Dict[str, int],
    preferences: Dict[str, Set[str]]
) -> Tuple[Dict[int, str], int]:
    """
    Assign room positions to physical rooms using greedy algorithm.

    The greedy approach is optimal for this problem:
    - Sort room positions by interest (descending)
    - Sort rooms by capacity (descending)
    - Match in order

    Args:
        block: The block with tuples
        room_capacities: room_id -> capacity
        preferences: participant_id -> set of talk_ids

    Returns:
        Tuple of (position -> room_id mapping, total capacity gap)
    """
    n_rooms = len(block.tuples[0]) if block.tuples else 0

    # Compute interest for each position
    position_interest = []
    for pos in range(n_rooms):
        interest = compute_room_session_interest(block, pos, preferences)
        position_interest.append((pos, interest))

    # Sort positions by interest (highest first)
    position_interest.sort(key=lambda x: -x[1])

    # Sort rooms by capacity (highest first)
    # Take only as many rooms as we need
    sorted_rooms = sorted(room_capacities.items(),
                          key=lambda x: -x[1])[:n_rooms]

    # Greedy matching
    room_mapping = {}
    total_gap = 0

    for i, (pos, interest) in enumerate(position_interest):
        room_id, capacity = sorted_rooms[i]
        room_mapping[pos] = room_id
        gap = capacity - interest
        total_gap += gap

    return room_mapping, total_gap


# =============================================================================
# BLOCK SCHEDULING
# =============================================================================

def compute_violation_cost(
    block: Block,
    timeslot_id: str,
    talk_presenter: Dict[str, str],
    presenter_unavailability: Dict[str, Set[str]]
) -> int:
    """
    Compute presenter availability violations if block assigned to timeslot.
    """
    violations = 0

    # Get all talks in the block
    all_talks = [talk for ntuple in block.tuples for talk in ntuple]

    for talk_id in all_talks:
        presenter_id = talk_presenter.get(talk_id)
        if presenter_id:
            unavailable = presenter_unavailability.get(presenter_id, set())
            if timeslot_id in unavailable:
                violations += 1

    return violations


def schedule_blocks_milp(
    blocks: List[Block],
    timeslots: List[Dict],
    talk_presenter: Dict[str, str],
    presenter_unavailability: Dict[str, Set[str]],
    time_limit: float = 60.0,
    verbose: bool = True
) -> List[Tuple[Block, Dict, int]]:
    """
    Assign blocks to timeslots minimizing availability violations.

    This is a bipartite matching problem solved with MILP.
    Requires Gurobi.

    IMPORTANT: Blocks are only assigned to timeslots with matching type_id.
    E.g., a block with block_type="4R3T" can only go to a timeslot with
    type_id="4R3T".

    Args:
        blocks: List of blocks to schedule
        timeslots: List of available timeslots
        talk_presenter: talk_id -> presenter_id
        presenter_unavailability: presenter_id -> set of unavailable timeslots
        time_limit: Solver time limit
        verbose: Print solver output

    Returns:
        List of (block, assigned_timeslot, violations) tuples
    """
    gp, GRB = _import_gurobi()

    if len(blocks) != len(timeslots):
        raise ValueError(
            f"Mismatch: {len(blocks)} blocks but {len(timeslots)} timeslots"
        )

    if len(blocks) == 0:
        return []

    # Verify block-timeslot type compatibility exists
    block_type_counts = {}
    for block in blocks:
        bt = block.block_type
        block_type_counts[bt] = block_type_counts.get(bt, 0) + 1

    timeslot_type_counts = {}
    for ts in timeslots:
        tt = ts.get("type_id", "unknown")
        timeslot_type_counts[tt] = timeslot_type_counts.get(tt, 0) + 1

    if verbose:
        print(f"  Block types: {block_type_counts}")
        print(f"  Timeslot types: {timeslot_type_counts}")

    # Check feasibility
    for bt, count in block_type_counts.items():
        ts_count = timeslot_type_counts.get(bt, 0)
        if ts_count < count:
            raise ValueError(
                f"Type mismatch: {count} blocks of type {bt} but only "
                f"{ts_count} timeslots of that type. "
                f"Block types: {block_type_counts}, Timeslot types: {timeslot_type_counts}"
            )

    # Compute violation costs
    costs = {}
    for b_idx, block in enumerate(blocks):
        for ts in timeslots:
            cost = compute_violation_cost(
                block, ts["id"],
                talk_presenter, presenter_unavailability
            )
            costs[(b_idx, ts["id"])] = cost

    # Build model
    model = gp.Model("block_scheduling")
    model.Params.TimeLimit = time_limit
    if not verbose:
        model.Params.OutputFlag = 0

    # Variables: z[b,t] = 1 if block b assigned to timeslot t
    # Only create variables for compatible (block, timeslot) pairs
    compatible_pairs = []
    for b_idx, block in enumerate(blocks):
        for ts in timeslots:
            # Only allow assignment if types match
            # Fallback for legacy
            ts_type = ts.get("type_id", block.block_type)
            if block.block_type == ts_type:
                compatible_pairs.append((b_idx, ts["id"]))

    z = model.addVars(compatible_pairs, vtype=GRB.BINARY, name="z")

    # Objective: minimize total violations
    model.setObjective(
        gp.quicksum(
            costs[(b_idx, ts_id)] * z[b_idx, ts_id]
            for b_idx, ts_id in compatible_pairs
        ),
        GRB.MINIMIZE
    )

    # Constraints: each block assigned to exactly one timeslot
    for b_idx in range(len(blocks)):
        compatible_ts = [ts_id for bi,
                         ts_id in compatible_pairs if bi == b_idx]
        model.addConstr(
            gp.quicksum(z[b_idx, ts_id] for ts_id in compatible_ts) == 1,
            name=f"block_{b_idx}"
        )

    # Constraints: each timeslot gets exactly one block
    for ts in timeslots:
        compatible_blocks = [bi for bi,
                             ts_id in compatible_pairs if ts_id == ts["id"]]
        model.addConstr(
            gp.quicksum(z[b_idx, ts["id"]]
                        for b_idx in compatible_blocks) == 1,
            name=f"timeslot_{ts['id']}"
        )

    # Solve
    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or model.SolCount == 0:
        raise RuntimeError(
            f"Block scheduling failed with status {model.Status}")

    # Extract assignment
    result = []
    for b_idx, block in enumerate(blocks):
        for ts in timeslots:
            if (b_idx, ts["id"]) in z and z[b_idx, ts["id"]].X > 0.5:
                violations = costs[(b_idx, ts["id"])]
                result.append((block, ts, violations))
                break

    return result


def schedule_blocks_hungarian(
    blocks: List[Block],
    timeslots: List[Dict],
    talk_presenter: Dict[str, str],
    presenter_unavailability: Dict[str, Set[str]],
    verbose: bool = True
) -> List[Tuple[Block, Dict, int]]:
    """
    Assign blocks to timeslots minimizing availability violations.

    Uses scipy's Hungarian algorithm (linear_sum_assignment).
    No Gurobi required.

    IMPORTANT: Blocks are only assigned to timeslots with matching type_id.
    E.g., a block with block_type="4R3T" can only go to a timeslot with
    type_id="4R3T". This is enforced by setting infinite cost for
    incompatible pairs.

    Args:
        blocks: List of blocks to schedule
        timeslots: List of available timeslots
        talk_presenter: talk_id -> presenter_id
        presenter_unavailability: presenter_id -> set of unavailable timeslots
        verbose: Print progress

    Returns:
        List of (block, assigned_timeslot, violations) tuples
    """
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    n = len(blocks)
    if n == 0:
        return []

    if n != len(timeslots):
        raise ValueError(
            f"Mismatch: {n} blocks but {len(timeslots)} timeslots"
        )

    # Verify block-timeslot type compatibility exists
    block_type_counts = {}
    for block in blocks:
        bt = block.block_type
        block_type_counts[bt] = block_type_counts.get(bt, 0) + 1

    timeslot_type_counts = {}
    for ts in timeslots:
        tt = ts.get("type_id", "unknown")
        timeslot_type_counts[tt] = timeslot_type_counts.get(tt, 0) + 1

    if verbose:
        print(f"  Block types: {block_type_counts}")
        print(f"  Timeslot types: {timeslot_type_counts}")

    # Check feasibility
    for bt, count in block_type_counts.items():
        ts_count = timeslot_type_counts.get(bt, 0)
        if ts_count < count:
            raise ValueError(
                f"Type mismatch: {count} blocks of type {bt} but only "
                f"{ts_count} timeslots of that type. "
                f"Block types: {block_type_counts}, Timeslot types: {timeslot_type_counts}"
            )

    # Build cost matrix with infinite cost for type mismatches
    INFINITY = 10**9
    cost_matrix = np.zeros((n, n), dtype=int)
    for b_idx, block in enumerate(blocks):
        for t_idx, ts in enumerate(timeslots):
            # Check type compatibility
            # Fallback for legacy
            ts_type = ts.get("type_id", block.block_type)
            if block.block_type != ts_type:
                cost_matrix[b_idx, t_idx] = INFINITY
            else:
                cost = compute_violation_cost(
                    block, ts["id"],
                    talk_presenter, presenter_unavailability
                )
                cost_matrix[b_idx, t_idx] = cost

    # Solve assignment problem (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Extract result
    result = []
    for b_idx, t_idx in zip(row_ind, col_ind):
        block = blocks[b_idx]
        ts = timeslots[t_idx]
        violations = int(cost_matrix[b_idx, t_idx])
        if violations >= INFINITY:
            raise RuntimeError(
                f"Block {block.block_id} (type={block.block_type}) assigned to "
                f"incompatible timeslot {ts['id']} (type={ts.get('type_id', 'unknown')})"
            )
        result.append((block, ts, violations))

    return result


# =============================================================================
# MAIN PHASE 3 SOLVER
# =============================================================================

def solve_phase3(
    phase3_input: Phase3Input,
    method: str = "milp",
    time_limit: float = 60.0,
    verbose: bool = True
) -> Phase3Result:
    """
    Solve Phase 3: block scheduling and room assignment.

    Args:
        phase3_input: Phase3Input with all required data
        method: "milp" (Gurobi) or "hungarian" (scipy)
        time_limit: Solver time limit for block scheduling (milp only)
        verbose: Print progress

    Returns:
        Phase3Result with complete assignments
    """
    if method not in ("milp", "hungarian"):
        raise ValueError(
            f"Unknown method: {method}. Use 'milp' or 'hungarian'.")

    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 3: BLOCK SCHEDULING & ROOM ASSIGNMENT")
        print("=" * 70)
        print(f"  Method: {method}")

    # Step 1: Schedule blocks to timeslots
    if verbose:
        print("\n--- Step 1: Block Scheduling ---")

    if method == "milp":
        scheduled = schedule_blocks_milp(
            phase3_input.blocks,
            phase3_input.timeslots,
            phase3_input.talk_presenter,
            phase3_input.presenter_unavailability,
            time_limit=time_limit,
            verbose=verbose
        )
    else:  # hungarian
        scheduled = schedule_blocks_hungarian(
            phase3_input.blocks,
            phase3_input.timeslots,
            phase3_input.talk_presenter,
            phase3_input.presenter_unavailability,
            verbose=verbose
        )

    total_violations = sum(v for _, _, v in scheduled)
    if verbose:
        print(f"Total presenter availability violations: {total_violations}")

    # Step 2: Assign rooms within each block
    if verbose:
        print("\n--- Step 2: Room Assignment ---")

    assignments = []
    total_gap = 0

    for block, timeslot, violations in scheduled:
        room_mapping, gap = assign_rooms_greedy(
            block,
            phase3_input.room_capacities,
            phase3_input.preferences
        )

        assignment = RoomAssignment(
            block=block,
            timeslot=timeslot,
            room_mapping=room_mapping,
            violations=violations
        )
        assignments.append(assignment)
        total_gap += gap

        if verbose:
            n_rooms = len(room_mapping)
            print(f"  {block.block_id} @ {timeslot['id']}: "
                  f"violations={violations}, rooms={n_rooms}")

    if verbose:
        print(f"\nTotal capacity gap: {total_gap}")

    return Phase3Result(
        assignments=assignments,
        total_violations=total_violations,
        total_capacity_gap=total_gap
    )
