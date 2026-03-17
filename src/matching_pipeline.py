"""
Matching-Based Pipeline: Bottom-Up Conference Scheduling

An alternative to the top-down Phase 1 → Phase 2 → Phase 3 approach.
This pipeline builds sessions bottom-up:

Phase A: Pair Matching
    - Match talks into pairs maximizing co-preference weight
    
Phase B: Block Formation  
    - Combine pairs into 3-blocks (pair + single) and 4-blocks (pair + pair)
    - Joint MILP optimization
    
Phase C: Tuple Selection
    - Select which blocks run in parallel, minimizing missed attendance
    - Cost = sum of missed talks (not blocks) per participant
    
Phase D: Talk Ordering
    - Order talks within blocks to maximize room-hopping opportunities

After Phase D, the standard Phase 3 (room assignment) is applied.

TERMINOLOGY NOTE:
- In this pipeline, a "block" from Phases A-B is a group of talks for ONE room session
  (e.g., 3 or 4 talks that will be given consecutively in one room)
- A "tuple" in Phase C groups these blocks to run in parallel (one per room)
- This maps to the original pipeline where:
  - Original "n-tuple" = talks in parallel at one timeslot
  - Original "Block" = multiple n-tuples forming a session (k timeslots)
  
In the matching pipeline:
- Phase C "tuple" = original "n-tuple" (parallel talks at one timeslot)
- Multiple Phase C tuples form the original "Block" (k tuples = k timeslots)

See docs/matching_pipeline_specification.md for full mathematical formulations.
"""

from typing import Dict, List, Set, Tuple, Optional, FrozenSet
from dataclasses import dataclass, field
from itertools import combinations, permutations, product
from collections import Counter
import time

import gurobipy as gp
from gurobipy import GRB

from .instance import ProblemInstance, compute_infeasible_talk_pairs
from .phase2 import Block
from .phase3 import Phase3Input, Phase3Result, solve_phase3


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MatchingPipelineResult:
    """Complete result from matching pipeline (Phases A-D)."""

    # Phase A results
    pairs: List[Tuple[str, str]]          # Matched pairs of talk_ids
    singles: List[str]                     # Unmatched single talk_ids
    phase_a_weight: int                    # Total co-preference weight of pairs

    # Phase B results
    blocks_3: List[Tuple[str, str, str]]   # 3-blocks (pair + single)
    blocks_4: List[Tuple[str, str, str, str]]  # 4-blocks (pair + pair)
    phase_b_weight: int                    # Total marginal weight added

    # Phase C results
    tuples_3: List[Tuple[Tuple[str, str, str], ...]]  # Tuples of 3-blocks
    tuples_4: List[Tuple[Tuple[str, str, str, str], ...]]  # Tuples of 4-blocks
    phase_c_cost: int                      # Total missed attendance

    # Phase D results
    ordered_blocks: List[Block]            # Final ordered blocks
    phase_d_benefit: int                   # Room-hopping benefit

    # Timing
    phase_a_time: float = 0.0
    phase_b_time: float = 0.0
    phase_c_time: float = 0.0
    phase_d_time: float = 0.0

    @property
    def total_time(self) -> float:
        return self.phase_a_time + self.phase_b_time + self.phase_c_time + self.phase_d_time


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_co_preference_weight(
    talk_i: str,
    talk_j: str,
    preferences: Dict[str, Set[str]]
) -> int:
    """
    Compute co-preference weight between two talks.

    Weight = number of participants who want BOTH talks.
    """
    count = 0
    for p_id, prefs in preferences.items():
        if talk_i in prefs and talk_j in prefs:
            count += 1
    return count


def build_co_preference_matrix(
    talks: List[str],
    preferences: Dict[str, Set[str]]
) -> Dict[Tuple[str, str], int]:
    """
    Build complete co-preference matrix for all talk pairs.

    Returns dict mapping (talk_i, talk_j) -> weight where i < j lexicographically.
    """
    weights = {}
    for i, talk_i in enumerate(talks):
        for talk_j in talks[i+1:]:
            # Ensure consistent ordering
            pair = (talk_i, talk_j) if talk_i < talk_j else (talk_j, talk_i)
            weights[pair] = compute_co_preference_weight(
                talk_i, talk_j, preferences)
    return weights


def get_weight(
    talk_i: str,
    talk_j: str,
    weights: Dict[Tuple[str, str], int]
) -> int:
    """Get weight from matrix with consistent ordering."""
    pair = (talk_i, talk_j) if talk_i < talk_j else (talk_j, talk_i)
    return weights.get(pair, 0)


def is_block_feasible(
    talk_ids: Tuple[str, ...],
    talk_presenter: Dict[str, str],
    presenter_unavailability: Dict[str, Set[str]],
    all_timeslots: Set[str]
) -> bool:
    """
    Check if a block of talks can be scheduled in at least one timeslot.

    A block is infeasible if the union of all presenters' unavailable 
    timeslots covers all available timeslots.

    Args:
        talk_ids: Tuple of talk IDs in the block
        talk_presenter: Mapping from talk_id to presenter_id
        presenter_unavailability: Mapping from presenter_id to unavailable timeslots
        all_timeslots: Set of all timeslot IDs

    Returns:
        True if feasible (can be scheduled), False otherwise
    """
    if not all_timeslots:
        return True  # No timeslots defined, assume feasible

    combined_unavailable: Set[str] = set()
    for talk_id in talk_ids:
        presenter = talk_presenter.get(talk_id)
        if presenter and presenter in presenter_unavailability:
            combined_unavailable |= presenter_unavailability[presenter]

    # Feasible if at least one timeslot is available
    return combined_unavailable < all_timeslots


# =============================================================================
# PHASE A: PAIR MATCHING
# =============================================================================

def solve_phase_a(
    instance: ProblemInstance,
    n_3: int,
    n_4: int,
    time_limit: float = 60.0,
    verbose: bool = True
) -> Tuple[List[Tuple[str, str]], List[str], int, float]:
    """
    Phase A: Maximum weight matching with cardinality constraint.

    Match exactly M = n_3 + 2*n_4 pairs of talks to maximize co-preference.

    Args:
        instance: Problem instance with talks and preferences
        n_3: Number of 3-talk sessions (each needs 1 pair + 1 single)
        n_4: Number of 4-talk sessions (each needs 2 pairs)
        time_limit: Gurobi time limit
        verbose: Print progress

    Returns:
        Tuple of (pairs, singles, total_weight, solve_time)
    """
    start_time = time.time()

    talks = instance.talks
    preferences = instance.preferences
    M = n_3 + 2 * n_4  # Number of pairs to match

    if verbose:
        print(f"Phase A: Matching {M} pairs from {len(talks)} talks")
        print(f"  (n_3={n_3} 3-blocks, n_4={n_4} 4-blocks)")

    # Build co-preference weights
    weights = build_co_preference_matrix(talks, preferences)

    # Create model
    model = gp.Model("PhaseA_PairMatching")
    model.setParam('OutputFlag', 1 if verbose else 0)
    model.setParam('TimeLimit', time_limit)

    # Variables: y[i,j] = 1 if talks i,j are matched
    y = {}
    for talk_i in talks:
        for talk_j in talks:
            if talk_i < talk_j:
                y[talk_i, talk_j] = model.addVar(
                    vtype=GRB.BINARY, name=f"y_{talk_i}_{talk_j}")

    # Objective: maximize total co-preference weight
    model.setObjective(
        gp.quicksum(
            get_weight(talk_i, talk_j, weights) * y[talk_i, talk_j]
            for talk_i, talk_j in y.keys()
        ),
        GRB.MAXIMIZE
    )

    # Constraint 1: Each talk in at most one pair
    for talk in talks:
        model.addConstr(
            gp.quicksum(
                y[min(talk, other), max(talk, other)]
                for other in talks if other != talk
            ) <= 1,
            name=f"matching_{talk}"
        )

    # Constraint 2: Exactly M pairs
    model.addConstr(
        gp.quicksum(y[i, j] for i, j in y.keys()) == M,
        name="cardinality"
    )

    # Constraint 3: Exclude infeasible pairs (presenter unavailability conflicts)
    infeasible_pairs = compute_infeasible_talk_pairs(instance, verbose=verbose)
    if infeasible_pairs:
        for t1, t2 in infeasible_pairs:
            pair_key = (t1, t2) if t1 < t2 else (t2, t1)
            if pair_key in y:
                model.addConstr(y[pair_key] == 0, name=f"infeasible_{t1}_{t2}")
        if verbose:
            print(
                f"  Added {len(infeasible_pairs)} infeasible pair constraints")

    # Solve
    model.optimize()

    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        raise ValueError(f"Phase A failed with status {model.Status}")

    # Extract solution
    pairs = []
    matched_talks = set()
    total_weight = 0

    for (talk_i, talk_j), var in y.items():
        if var.X > 0.5:
            pairs.append((talk_i, talk_j))
            matched_talks.add(talk_i)
            matched_talks.add(talk_j)
            total_weight += get_weight(talk_i, talk_j, weights)

    singles = [t for t in talks if t not in matched_talks]

    solve_time = time.time() - start_time

    if verbose:
        print(f"  Found {len(pairs)} pairs with total weight {total_weight}")
        print(f"  Remaining singles: {len(singles)}")
        print(f"  Time: {solve_time:.2f}s")

    return pairs, singles, total_weight, solve_time


# =============================================================================
# PHASE B: BLOCK FORMATION
# =============================================================================

def solve_phase_b(
    pairs: List[Tuple[str, str]],
    singles: List[str],
    n_3: int,
    n_4: int,
    preferences: Dict[str, Set[str]],
    time_limit: float = 60.0,
    verbose: bool = True,
    talk_presenter: Optional[Dict[str, str]] = None,
    presenter_unavailability: Optional[Dict[str, Set[str]]] = None,
    all_timeslots: Optional[Set[str]] = None
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str, str]], int, float]:
    """
    Phase B: Form 3-blocks and 4-blocks from pairs and singles.

    Joint MILP:
    - 4-blocks: pair + pair (n_4 blocks)
    - 3-blocks: pair + single (n_3 blocks)

    Maximize marginal weight contribution.
    Optionally excludes blocks with presenter unavailability conflicts.

    Args:
        pairs: List of matched pairs from Phase A
        singles: List of unmatched singles from Phase A
        n_3: Number of 3-blocks to form
        n_4: Number of 4-blocks to form
        preferences: Participant preferences
        time_limit: Gurobi time limit
        verbose: Print progress
        talk_presenter: Optional mapping from talk_id to presenter_id
        presenter_unavailability: Optional mapping from presenter_id to unavailable timeslots
        all_timeslots: Optional set of all timeslot IDs

    Returns:
        Tuple of (blocks_3, blocks_4, marginal_weight, solve_time)
    """
    start_time = time.time()

    if verbose:
        print(f"Phase B: Forming {n_3} 3-blocks and {n_4} 4-blocks")

    # Build co-preference matrix for all talks
    all_talks = []
    for p in pairs:
        all_talks.extend(p)
    all_talks.extend(singles)
    weights = build_co_preference_matrix(all_talks, preferences)

    # Index pairs
    pair_idx = {p: i for i, p in enumerate(pairs)}
    single_idx = {s: i for i, s in enumerate(singles)}

    # Check if we should filter infeasible blocks
    check_feasibility = (
        talk_presenter is not None and
        presenter_unavailability is not None and
        all_timeslots is not None and
        len(all_timeslots) > 0
    )

    # Create model
    model = gp.Model("PhaseB_BlockFormation")
    model.setParam('OutputFlag', 1 if verbose else 0)
    model.setParam('TimeLimit', time_limit)

    # Variables for 4-blocks: z[p1, p2] = 1 if pairs p1, p2 form a 4-block
    z = {}
    for i, p1 in enumerate(pairs):
        for p2 in pairs[i+1:]:
            z[p1, p2] = model.addVar(
                vtype=GRB.BINARY, name=f"z_{pair_idx[p1]}_{pair_idx[p2]}")

    # Variables for 3-blocks: u[p, s] = 1 if pair p and single s form a 3-block
    u = {}
    for p in pairs:
        for s in singles:
            u[p, s] = model.addVar(
                vtype=GRB.BINARY, name=f"u_{pair_idx[p]}_{single_idx[s]}")

    # Marginal weight for 4-block: cross-pair edges
    def marginal_weight_4(p1, p2):
        """Weight of edges between pairs p1 and p2."""
        i, j = p1
        k, l = p2
        return (get_weight(i, k, weights) + get_weight(i, l, weights) +
                get_weight(j, k, weights) + get_weight(j, l, weights))

    # Marginal weight for 3-block: edges from pair to single
    def marginal_weight_3(p, s):
        """Weight of edges from pair p to single s."""
        i, j = p
        return get_weight(i, s, weights) + get_weight(j, s, weights)

    # Objective: maximize total marginal weight
    model.setObjective(
        gp.quicksum(marginal_weight_4(p1, p2) * z[p1, p2] for p1, p2 in z.keys()) +
        gp.quicksum(marginal_weight_3(p, s) * u[p, s] for p, s in u.keys()),
        GRB.MAXIMIZE
    )

    # Constraint 1: Each pair used exactly once (in 4-block OR 3-block)
    for p in pairs:
        # Sum over all 4-blocks containing p
        z_terms = gp.quicksum(
            z[min(p, p2, key=lambda x: pair_idx[x]),
              max(p, p2, key=lambda x: pair_idx[x])]
            for p2 in pairs if p2 != p and (
                (p, p2) in z or (p2, p) in z
            )
        )
        # Need to handle both orderings in z
        z_terms_fixed = gp.quicksum(
            z[p1, p2] for p1, p2 in z.keys() if p in (p1, p2)
        )
        # Sum over all 3-blocks containing p
        u_terms = gp.quicksum(u[p, s] for s in singles)

        model.addConstr(z_terms_fixed + u_terms == 1,
                        name=f"pair_once_{pair_idx[p]}")

    # Constraint 2: Each single used exactly once
    for s in singles:
        model.addConstr(
            gp.quicksum(u[p, s] for p in pairs) == 1,
            name=f"single_once_{single_idx[s]}"
        )

    # Constraint 3: Exactly n_4 4-blocks
    model.addConstr(
        gp.quicksum(z[p1, p2] for p1, p2 in z.keys()) == n_4,
        name="count_4blocks"
    )

    # (n_3 3-blocks is implied by constraints 1-3)

    # Constraint 4: Exclude infeasible blocks (presenter unavailability conflicts)
    if check_feasibility:
        n_infeasible_4 = 0
        n_infeasible_3 = 0

        # Check 4-blocks
        for (p1, p2) in z.keys():
            block_talks = p1 + p2
            if not is_block_feasible(block_talks, talk_presenter,
                                     presenter_unavailability, all_timeslots):
                model.addConstr(z[p1, p2] == 0,
                                name=f"infeasible_4block_{pair_idx[p1]}_{pair_idx[p2]}")
                n_infeasible_4 += 1

        # Check 3-blocks
        for (p, s) in u.keys():
            block_talks = p + (s,)
            if not is_block_feasible(block_talks, talk_presenter,
                                     presenter_unavailability, all_timeslots):
                model.addConstr(u[p, s] == 0,
                                name=f"infeasible_3block_{pair_idx[p]}_{single_idx[s]}")
                n_infeasible_3 += 1

        if verbose and (n_infeasible_4 > 0 or n_infeasible_3 > 0):
            print(f"  Excluded {n_infeasible_4} infeasible 4-blocks, "
                  f"{n_infeasible_3} infeasible 3-blocks")

    # Solve
    model.optimize()

    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        raise ValueError(f"Phase B failed with status {model.Status}")

    # Extract solution
    blocks_4 = []
    for (p1, p2), var in z.items():
        if var.X > 0.5:
            # Combine into 4-tuple
            blocks_4.append(p1 + p2)

    blocks_3 = []
    for (p, s), var in u.items():
        if var.X > 0.5:
            blocks_3.append(p + (s,))

    marginal_weight = int(model.ObjVal)
    solve_time = time.time() - start_time

    if verbose:
        print(
            f"  Formed {len(blocks_3)} 3-blocks and {len(blocks_4)} 4-blocks")
        print(f"  Marginal weight: {marginal_weight}")
        print(f"  Time: {solve_time:.2f}s")

    return blocks_3, blocks_4, marginal_weight, solve_time


# =============================================================================
# PHASE C: TUPLE SELECTION
# =============================================================================

def compute_tuple_cost_talk_level(
    block_tuple: Tuple[Tuple[str, ...], ...],
    preferences: Dict[str, Set[str]]
) -> int:
    """
    Compute missed attendance cost for a tuple of blocks (talk-level).

    For each participant:
    1. Count interested talks per block
    2. Choose the block with most interested talks
    3. Cost = sum of interested talks in OTHER blocks

    Args:
        block_tuple: Tuple of blocks (each block is a tuple of talk_ids)
        preferences: Participant preferences

    Returns:
        Total missed talks across all participants
    """
    cost = 0

    for p_id, prefs in preferences.items():
        # Count interested talks per block
        block_counts = []
        for block in block_tuple:
            count = sum(1 for talk in block if talk in prefs)
            block_counts.append(count)

        if not any(block_counts):
            continue

        # Participant chooses block with max interested talks
        max_count = max(block_counts)
        total_interested = sum(block_counts)

        # Missed = total - max (talks in other blocks)
        missed = total_interested - max_count
        cost += missed

    return cost


def solve_phase_c(
    blocks_3: List[Tuple[str, str, str]],
    blocks_4: List[Tuple[str, str, str, str]],
    tuple_requirements_3: Dict[int, int],
    tuple_requirements_4: Dict[int, int],
    preferences: Dict[str, Set[str]],
    time_limit: float = 120.0,
    verbose: bool = True,
    forbidden_solutions_3: Optional[List[List[Tuple]]] = None,
    forbidden_solutions_4: Optional[List[List[Tuple]]] = None
) -> Tuple[List[Tuple], List[Tuple], int, float]:
    """
    Phase C: Select which blocks run in parallel (tuple selection).

    Supports variable tuple sizes (different number of parallel rooms).

    Args:
        blocks_3: List of 3-blocks
        blocks_4: List of 4-blocks
        tuple_requirements_3: {tuple_size: count} for 3-blocks
            e.g., {4: 2, 5: 1} means 2 4-tuples and 1 5-tuple of 3-blocks
        tuple_requirements_4: {tuple_size: count} for 4-blocks
        preferences: Participant preferences
        time_limit: Gurobi time limit
        verbose: Print progress
        forbidden_solutions_3: List of previous 3-block solutions to exclude
        forbidden_solutions_4: List of previous 4-block solutions to exclude

    Returns:
        Tuple of (tuples_3, tuples_4, total_cost, solve_time)
    """
    start_time = time.time()

    if verbose:
        print(f"Phase C: Selecting tuples (variable sizes)")
        print(f"  {len(blocks_3)} 3-blocks, requirements: {tuple_requirements_3}")
        print(f"  {len(blocks_4)} 4-blocks, requirements: {tuple_requirements_4}")

    tuples_3, cost_3 = [], 0
    tuples_4, cost_4 = [], 0

    # Solve for 3-blocks
    if blocks_3 and tuple_requirements_3:
        tuples_3, cost_3, _ = _solve_tuple_selection_variable(
            blocks_3, tuple_requirements_3, preferences, time_limit/2, verbose, "3-block",
            forbidden_solutions=forbidden_solutions_3
        )

    # Solve for 4-blocks
    if blocks_4 and tuple_requirements_4:
        tuples_4, cost_4, _ = _solve_tuple_selection_variable(
            blocks_4, tuple_requirements_4, preferences, time_limit/2, verbose, "4-block",
            forbidden_solutions=forbidden_solutions_4
        )

    total_cost = cost_3 + cost_4
    solve_time = time.time() - start_time

    if verbose:
        print(f"  Total missed attendance: {total_cost}")
        print(f"  Time: {solve_time:.2f}s")

    return tuples_3, tuples_4, total_cost, solve_time


def _solve_tuple_selection_variable(
    blocks: List[Tuple[str, ...]],
    tuple_requirements: Dict[int, int],
    preferences: Dict[str, Set[str]],
    time_limit: float,
    verbose: bool,
    label: str,
    forbidden_solutions: Optional[List[List[Tuple]]] = None
) -> Tuple[List[Tuple], int, float]:
    """
    Solve tuple selection with variable tuple sizes.

    Args:
        blocks: List of blocks (all same k)
        tuple_requirements: {tuple_size: count_needed}
        preferences: Participant preferences
        time_limit: Solver time limit
        verbose: Print progress
        label: Label for logging
        forbidden_solutions: List of previous solutions to exclude via no-good cuts

    Returns:
        Tuple of (selected_tuples, total_cost, solve_time)
    """
    if not blocks or not tuple_requirements:
        return [], 0, 0.0

    start_time = time.time()

    # Validate: sum of (size * count) should equal len(blocks)
    total_slots = sum(size * count for size,
                      count in tuple_requirements.items())
    if total_slots != len(blocks):
        raise ValueError(
            f"Tuple requirements mismatch: {total_slots} slots needed, "
            f"but {len(blocks)} {label}s available"
        )

    # Generate all possible tuples for each size
    all_tuples = []  # List of (tuple, size)
    tuple_costs = {}

    for tuple_size in tuple_requirements.keys():
        size_tuples = list(combinations(blocks, tuple_size))
        if verbose:
            print(
                f"  Generating {label} {tuple_size}-tuples: {len(size_tuples)} candidates")

        for t in size_tuples:
            all_tuples.append((t, tuple_size))
            tuple_costs[t] = compute_tuple_cost_talk_level(t, preferences)

    # Create model
    model = gp.Model(f"PhaseC_{label}_variable")
    model.setParam('OutputFlag', 1 if verbose else 0)
    model.setParam('TimeLimit', time_limit)

    # Variables: x[t] = 1 if tuple t is selected
    x = {}
    for i, (t, size) in enumerate(all_tuples):
        x[t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")

    # Objective: minimize total cost
    model.setObjective(
        gp.quicksum(tuple_costs[t] * x[t] for t, _ in all_tuples),
        GRB.MINIMIZE
    )

    # Constraint 1: Each block in exactly one tuple
    for block in blocks:
        model.addConstr(
            gp.quicksum(x[t] for t, _ in all_tuples if block in t) == 1,
            name=f"cover_{blocks.index(block)}"
        )

    # Constraint 2: Select exactly the required number of tuples per size
    for tuple_size, count_needed in tuple_requirements.items():
        model.addConstr(
            gp.quicksum(x[t] for t, size in all_tuples if size ==
                        tuple_size) == count_needed,
            name=f"count_size_{tuple_size}"
        )

    # Constraint 3: No-good cuts to exclude previous solutions
    if forbidden_solutions:
        for sol_idx, prev_solution in enumerate(forbidden_solutions):
            # No-good cut: sum of selected tuples from prev solution <= |prev| - 1
            prev_tuples_in_x = [t for t in prev_solution if t in x]
            if prev_tuples_in_x:
                model.addConstr(
                    gp.quicksum(x[t] for t in prev_tuples_in_x) <= len(
                        prev_tuples_in_x) - 1,
                    name=f"nogood_{sol_idx}"
                )
        if verbose:
            print(f"  Added {len(forbidden_solutions)} no-good cuts")

    # Solve
    model.optimize()

    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        raise ValueError(
            f"Phase C ({label}) failed with status {model.Status}")

    if model.SolCount == 0:
        raise ValueError(f"Phase C ({label}) found no feasible solution")

    # Extract solution
    selected_tuples = []
    for t, _ in all_tuples:
        if x[t].X > 0.5:
            selected_tuples.append(t)

    total_cost = int(model.ObjVal)
    solve_time = time.time() - start_time

    if verbose:
        sizes = {}
        for t in selected_tuples:
            s = len(t)
            sizes[s] = sizes.get(s, 0) + 1
        print(f"  Selected {label} tuples by size: {sizes}")

    return selected_tuples, total_cost, solve_time


# =============================================================================
# PHASE D: TALK ORDERING
# =============================================================================

def compute_achievable_attendance(
    ordered_blocks: List[Tuple[str, ...]],
    participant_prefs: Set[str]
) -> int:
    """
    Compute maximum talks a participant can attend via room-hopping.

    At each timeslot, the participant can choose any one talk from any block.
    They maximize total attended talks.

    Args:
        ordered_blocks: List of blocks, each with talks in order
        participant_prefs: Set of preferred talk_ids

    Returns:
        Maximum number of talks the participant can attend
    """
    if not ordered_blocks:
        return 0

    k = len(ordered_blocks[0])  # talks per block
    attended = 0

    for slot in range(k):
        # At this timeslot, check each block for a preferred talk
        for block in ordered_blocks:
            if slot < len(block) and block[slot] in participant_prefs:
                attended += 1
                break  # Can only attend one talk per slot

    return attended


def compute_hopping_benefit(
    ordered_blocks: List[Tuple[str, ...]],
    preferences: Dict[str, Set[str]]
) -> int:
    """
    Compute total room-hopping benefit across all participants.

    Benefit = (talks attended with hopping) - (talks attended without hopping)

    Without hopping: participant attends all talks in their best block.
    With hopping: participant can switch rooms each timeslot.
    """
    benefit = 0

    for p_id, prefs in preferences.items():
        # With hopping
        with_hopping = compute_achievable_attendance(ordered_blocks, prefs)

        # Without hopping: best block
        best_block_count = 0
        for block in ordered_blocks:
            count = sum(1 for talk in block if talk in prefs)
            best_block_count = max(best_block_count, count)

        benefit += with_hopping - best_block_count

    return benefit


def solve_phase_d(
    tuples_3: List[Tuple[Tuple[str, ...], ...]],
    tuples_4: List[Tuple[Tuple[str, ...], ...]],
    preferences: Dict[str, Set[str]],
    block_types: Dict[str, Dict],
    verbose: bool = True
) -> Tuple[List[Block], int, float]:
    """
    Phase D: Order talks within blocks to maximize room-hopping benefit.

    For each tuple of blocks (parallel room sessions), enumerate all possible
    orderings of talks within each block and select the one with maximum benefit.

    Creates Block structures compatible with Phase 3:
    - A Block contains k n-tuples (one per timeslot in the session)
    - Each n-tuple has n talks (one per room, where n varies per tuple)

    Args:
        tuples_3: Selected tuples of 3-blocks from Phase C (variable sizes)
        tuples_4: Selected tuples of 4-blocks from Phase C (variable sizes)
        preferences: Participant preferences
        block_types: Block type configuration from instance
        verbose: Print progress

    Returns:
        Tuple of (blocks_for_phase3, total_benefit, solve_time)
    """
    start_time = time.time()

    if verbose:
        print(f"Phase D: Ordering talks for room-hopping")
        print(
            f"  {len(tuples_3)} 3-block session groups, {len(tuples_4)} 4-block session groups")

    blocks_for_phase3 = []
    total_benefit = 0

    # Build a mapping from (k, n) to a list of available block_type IDs.
    # Multiple sessions can share the same (k, n) dimensions (e.g., TA, TB, FA, FB
    # are all 5-room, 4-talk). Each created block consumes one ID from the list.
    kn_to_block_types: Dict[Tuple[int, int], List[str]] = {}
    for type_id, bt in block_types.items():
        k, n = bt["k"], bt["n"]
        for _ in range(bt["count"]):
            kn_to_block_types.setdefault((k, n), []).append(type_id)

    # Process 3-block tuples
    # Each tuple from Phase C represents one timeslot with n parallel room-sessions
    blocks_for_phase3.extend(_create_phase3_blocks(
        tuples_3, 3, preferences, kn_to_block_types, verbose
    ))

    # Process 4-block tuples
    blocks_for_phase3.extend(_create_phase3_blocks(
        tuples_4, 4, preferences, kn_to_block_types, verbose
    ))

    # Compute total benefit
    for block in blocks_for_phase3:
        total_benefit += _compute_block_hopping_benefit(block, preferences)

    solve_time = time.time() - start_time

    if verbose:
        print(f"  Created {len(blocks_for_phase3)} blocks for Phase 3")
        print(f"  Total room-hopping benefit: {total_benefit}")
        print(f"  Time: {solve_time:.2f}s")

    return blocks_for_phase3, total_benefit, solve_time


def _create_phase3_blocks(
    session_tuples: List[Tuple[Tuple[str, ...], ...]],
    k: int,
    preferences: Dict[str, Set[str]],
    kn_to_block_types: Dict[Tuple[int, int], List[str]],
    verbose: bool
) -> List[Block]:
    """
    Create Phase 3-compatible Block structures from session tuples.

    In the matching pipeline:
    - Phase A/B create "blocks" = groups of k talks for one room-session
    - Phase C creates "tuples" = groups of n "blocks" to run in parallel

    A Phase C tuple = (block_room0, block_room1, ..., block_room_{n-1})
    where each block has k talks, and n can vary per tuple.

    For Phase 3, we need Block objects where:
    - Block.tuples = list of k n-tuples
    - Each n-tuple = (talk_room0_slot_i, talk_room1_slot_i, ...)

    So we need to transpose: from (rooms × slots) to (slots × rooms)

    The kn_to_block_types dict maps (k, n) to a list of available type IDs.
    Each block creation pops one ID from the list, ensuring unique assignment
    when multiple sessions share the same dimensions.
    """
    if not session_tuples:
        return []

    blocks = []

    for tuple_idx, room_blocks in enumerate(session_tuples):
        # room_blocks is a tuple of n blocks, each block has k talks
        # room_blocks[room_idx][slot_idx] = talk_id
        # n is derived from the tuple size
        n_rooms = len(room_blocks)

        # First, find best ordering for room-hopping
        best_ordering, benefit = _find_best_ordering(room_blocks, preferences)

        # Now transpose: create k n-tuples (one per slot)
        n_tuples = []
        for slot_idx in range(k):
            # Get the talk from each room at this slot
            slot_tuple = tuple(best_ordering[room_idx][slot_idx]
                               for room_idx in range(n_rooms))
            n_tuples.append(slot_tuple)

        # Consume a block type ID from the available list for this (k, n)
        available = kn_to_block_types.get((k, n_rooms), [])
        if available:
            block_type = available.pop(0)
        else:
            block_type = f"{n_rooms}R{k}T"

        # Create Block for Phase 3
        block = Block(
            block_id=f"M{k}n{n_rooms}_{tuple_idx}",
            block_type=block_type,
            tuples=n_tuples,
            hopping_cost=0  # Will be computed by Phase 3 if needed
        )
        blocks.append(block)

    return blocks


def _compute_block_hopping_benefit(
    block: Block,
    preferences: Dict[str, Set[str]]
) -> int:
    """Compute room-hopping benefit for a single Phase 3 Block."""
    if not block.tuples:
        return 0

    n_rooms = len(block.tuples[0])
    k = len(block.tuples)

    total_benefit = 0

    for p_id, prefs in preferences.items():
        # With hopping: at each slot, take best available
        with_hopping = 0
        for slot_tuple in block.tuples:
            for talk in slot_tuple:
                if talk in prefs:
                    with_hopping += 1
                    break

        # Without hopping: pick best room, stay there
        best_room_count = 0
        for room_idx in range(n_rooms):
            room_count = sum(1 for slot_tuple in block.tuples
                             if slot_tuple[room_idx] in prefs)
            best_room_count = max(best_room_count, room_count)

        total_benefit += with_hopping - best_room_count

    return total_benefit


def _find_best_ordering(
    block_tuple: Tuple[Tuple[str, ...], ...],
    preferences: Dict[str, Set[str]]
) -> Tuple[List[Tuple[str, ...]], int]:
    """
    Find the best ordering of talks within each block.

    Enumerates all possible orderings and selects the one with maximum
    room-hopping benefit.

    For k talks per block and r rooms:
    - 3-blocks: (3!)^r = 6^r orderings
    - 4-blocks: (4!)^r = 24^r orderings

    For large search spaces, uses random sampling instead of exhaustive search.
    """
    import random
    import math

    k = len(block_tuple[0])
    n_rooms = len(block_tuple)

    # Calculate total orderings
    factorial_k = math.factorial(k)
    total_orderings = factorial_k ** n_rooms

    # Threshold for exhaustive search (about 100K orderings)
    MAX_EXHAUSTIVE = 100_000

    # For each block, generate all permutations
    block_perms = [list(permutations(block)) for block in block_tuple]

    best_ordering = None
    best_benefit = -float('inf')

    if total_orderings <= MAX_EXHAUSTIVE:
        # Exhaustive search
        for ordering_combo in product(*block_perms):
            benefit = compute_hopping_benefit(
                list(ordering_combo), preferences)
            if benefit > best_benefit:
                best_benefit = benefit
                best_ordering = list(ordering_combo)
    else:
        # Random sampling for large search spaces
        # Sample enough to have high confidence in finding a good solution
        n_samples = min(50_000, total_orderings)

        for _ in range(n_samples):
            # Randomly pick one permutation for each block
            ordering_combo = tuple(random.choice(perms)
                                   for perms in block_perms)
            benefit = compute_hopping_benefit(
                list(ordering_combo), preferences)
            if benefit > best_benefit:
                best_benefit = benefit
                best_ordering = list(ordering_combo)

        # Also try the identity ordering (original order)
        identity = [block for block in block_tuple]
        identity_benefit = compute_hopping_benefit(identity, preferences)
        if identity_benefit > best_benefit:
            best_benefit = identity_benefit
            best_ordering = identity

    return best_ordering, best_benefit


# =============================================================================
# MAIN PIPELINE FUNCTION
# =============================================================================

def run_matching_pipeline(
    instance: ProblemInstance,
    time_limit: float = 300.0,
    verbose: bool = True,
    run_phase3: bool = True,
    phase3_method: str = "milp",
    max_feasibility_retries: int = 10
) -> Tuple[MatchingPipelineResult, Optional[Phase3Result]]:
    """
    Run the complete matching-based pipeline (Phases A-D, optionally Phase 3).

    Includes feasibility checks for presenter unavailabilities:
    - Phase A: Excludes pairs with conflicting unavailabilities
    - Phase B: Excludes blocks with conflicting unavailabilities
    - Phase C: Retries with no-good cuts if Phase 3 finds violations

    Args:
        instance: Problem instance with talks and preferences
        time_limit: Total time limit (distributed across phases)
        verbose: Print progress
        run_phase3: Whether to run Phase 3 (room assignment)
        phase3_method: "milp" or "hungarian" for Phase 3
        max_feasibility_retries: Max retries for Phase C when violations found

    Returns:
        Tuple of (MatchingPipelineResult, Phase3Result or None)
    """
    if verbose:
        print("=" * 60)
        print("MATCHING PIPELINE: Bottom-Up Conference Scheduling")
        print("=" * 60)

    # =========================================================================
    # Extract block configuration
    # =========================================================================
    # Count total room-sessions by k (talks per session)
    # Also track tuple requirements: how many n-tuples needed for each k

    n_3 = 0  # Total 3-talk room-sessions
    n_4 = 0  # Total 4-talk room-sessions

    # Tuple requirements: for each k, how many tuples of each size n
    # tuple_requirements_3[n] = count of n-tuples needed for 3-blocks
    # tuple_requirements_4[n] = count of n-tuples needed for 4-blocks
    tuple_requirements_3: Dict[int, int] = {}
    tuple_requirements_4: Dict[int, int] = {}

    for type_id, bt in instance.block_types.items():
        n = bt["n"]  # rooms (tuple size)
        k = bt["k"]  # talks per session
        count = bt["count"]  # number of such blocks

        # Each block has n rooms, and needs 1 tuple of size n
        # count blocks → count tuples of size n

        # Total room-sessions: count * n
        total_room_sessions = count * n

        if k == 3:
            n_3 += total_room_sessions
            tuple_requirements_3[n] = tuple_requirements_3.get(n, 0) + count
        elif k == 4:
            n_4 += total_room_sessions
            tuple_requirements_4[n] = tuple_requirements_4.get(n, 0) + count
        else:
            raise ValueError(
                f"Matching pipeline only supports k=3 or k=4, got k={k}")

    if verbose:
        print(f"\nConfiguration:")
        print(f"  {n_3} room-sessions with 3 talks (3-blocks)")
        if tuple_requirements_3:
            print(f"    Tuple requirements: {tuple_requirements_3}")
        print(f"  {n_4} room-sessions with 4 talks (4-blocks)")
        if tuple_requirements_4:
            print(f"    Tuple requirements: {tuple_requirements_4}")
        print(f"  Total talks: {len(instance.talks)}")

    # Validate talk count
    expected_talks = 3 * n_3 + 4 * n_4
    if len(instance.talks) != expected_talks:
        raise ValueError(
            f"Talk count ({len(instance.talks)}) does not match slot count "
            f"({expected_talks}) from block_types. Add placeholder talks before "
            f"calling the matching pipeline so these counts match."
        )

    # Validate tuple requirements
    # Sum of (size * count) should equal total room-sessions
    req_3_total = sum(size * count for size,
                      count in tuple_requirements_3.items())
    req_4_total = sum(size * count for size,
                      count in tuple_requirements_4.items())
    if req_3_total != n_3:
        raise ValueError(
            f"3-block tuple requirements sum to {req_3_total} room-sessions "
            f"but need {n_3}. Check block_types configuration."
        )
    if req_4_total != n_4:
        raise ValueError(
            f"4-block tuple requirements sum to {req_4_total} room-sessions "
            f"but need {n_4}. Check block_types configuration."
        )

    # Phase A: Pair Matching
    if verbose:
        print("\n" + "-" * 40)
        print("PHASE A: Pair Matching")
        print("-" * 40)
    pairs, singles, phase_a_weight, phase_a_time = solve_phase_a(
        instance, n_3, n_4,
        time_limit=time_limit * 0.15,
        verbose=verbose
    )

    # Phase B: Block Formation
    if verbose:
        print("\n" + "-" * 40)
        print("PHASE B: Block Formation")
        print("-" * 40)

    # Get all timeslots for feasibility checking
    all_timeslots = instance.get_all_timeslots()

    blocks_3, blocks_4, phase_b_weight, phase_b_time = solve_phase_b(
        pairs, singles, n_3, n_4,
        instance.preferences,
        time_limit=time_limit * 0.15,
        verbose=verbose,
        talk_presenter=instance.talk_presenter,
        presenter_unavailability=instance.presenter_unavailability,
        all_timeslots=all_timeslots
    )

    # Phase C: Tuple Selection (with variable tuple sizes)
    # Phase C, D, 3 with feasibility retry loop
    forbidden_solutions_3: List[List[Tuple]] = []
    forbidden_solutions_4: List[List[Tuple]] = []

    phase_c_time_total = 0.0
    phase_d_time_total = 0.0
    feasibility_achieved = False

    for retry in range(max_feasibility_retries):
        if verbose:
            print("\n" + "-" * 40)
            if retry == 0:
                print("PHASE C: Tuple Selection")
            else:
                print(f"PHASE C: Tuple Selection (retry {retry})")
            print("-" * 40)

        tuples_3, tuples_4, phase_c_cost, phase_c_time = solve_phase_c(
            blocks_3, blocks_4,
            tuple_requirements_3, tuple_requirements_4,
            instance.preferences,
            time_limit=time_limit * 0.5 /
            (retry + 1),  # Reduce time on retries
            verbose=verbose,
            forbidden_solutions_3=forbidden_solutions_3 if forbidden_solutions_3 else None,
            forbidden_solutions_4=forbidden_solutions_4 if forbidden_solutions_4 else None
        )
        phase_c_time_total += phase_c_time

        # Phase D: Talk Ordering
        if verbose:
            print("\n" + "-" * 40)
            print("PHASE D: Talk Ordering")
            print("-" * 40)
        ordered_blocks, phase_d_benefit, phase_d_time = solve_phase_d(
            tuples_3, tuples_4,
            instance.preferences,
            instance.block_types,
            verbose=verbose
        )
        phase_d_time_total += phase_d_time

        # Check Phase 3 for violations
        if run_phase3:
            if verbose:
                print("\n" + "-" * 40)
                print("PHASE 3: Room Assignment (checking feasibility)")
                print("-" * 40)

            phase3_result = _run_phase3(
                ordered_blocks,
                instance,
                method=phase3_method,
                time_limit=time_limit * 0.2,
                verbose=verbose
            )

            if phase3_result.total_violations == 0:
                feasibility_achieved = True
                if verbose and retry > 0:
                    print(
                        f"  ✓ Feasible solution found after {retry + 1} attempts")
                break
            else:
                if verbose:
                    print(
                        f"  ⚠ Found {phase3_result.total_violations} presenter violations")
                # Add current solution to forbidden list
                if tuples_3:
                    forbidden_solutions_3.append(list(tuples_3))
                if tuples_4:
                    forbidden_solutions_4.append(list(tuples_4))
        else:
            # Not running Phase 3, just accept first solution
            phase3_result = None
            feasibility_achieved = True
            break

    if not feasibility_achieved and verbose:
        print(
            f"  ⚠ Warning: Could not find feasible solution after {max_feasibility_retries} retries")

    # Summary of Phases A-D
    if verbose:
        print("\n" + "=" * 60)
        print("PHASES A-D COMPLETE")
        print("=" * 60)
        print(
            f"Phase A: {len(pairs)} pairs, weight={phase_a_weight}, time={phase_a_time:.2f}s")
        print(
            f"Phase B: {len(blocks_3)}+{len(blocks_4)} room-sessions, marginal={phase_b_weight}, time={phase_b_time:.2f}s")
        print(
            f"Phase C: {len(tuples_3)}+{len(tuples_4)} parallel groups, missed={phase_c_cost}, time={phase_c_time_total:.2f}s")
        print(
            f"Phase D: {len(ordered_blocks)} blocks for Phase 3, benefit={phase_d_benefit}, time={phase_d_time_total:.2f}s")
        total_time = phase_a_time + phase_b_time + \
            phase_c_time_total + phase_d_time_total
        print(f"Phases A-D time: {total_time:.2f}s")
        if len(forbidden_solutions_3) > 0 or len(forbidden_solutions_4) > 0:
            print(
                f"  (required {len(forbidden_solutions_3) + len(forbidden_solutions_4)} retries)")

    pipeline_result = MatchingPipelineResult(
        pairs=pairs,
        singles=singles,
        phase_a_weight=phase_a_weight,
        blocks_3=blocks_3,
        blocks_4=blocks_4,
        phase_b_weight=phase_b_weight,
        tuples_3=tuples_3,
        tuples_4=tuples_4,
        phase_c_cost=phase_c_cost,
        ordered_blocks=ordered_blocks,
        phase_d_benefit=phase_d_benefit,
        phase_a_time=phase_a_time,
        phase_b_time=phase_b_time,
        phase_c_time=phase_c_time_total,
        phase_d_time=phase_d_time_total
    )

    return pipeline_result, phase3_result


def _run_phase3(
    blocks: List[Block],
    instance: ProblemInstance,
    method: str = "milp",
    time_limit: float = 60.0,
    verbose: bool = True
) -> Phase3Result:
    """
    Run Phase 3 using the standard room assignment logic.

    Builds timeslots from instance.block_types so their type_ids match
    the block_type assigned by Phase D (both use structural names like '5R4T'
    or session names like 'TA' — whichever is in block_types).

    If predefined timeslots exist (from sessions.csv), we map their type_ids
    to match the block_types used by the matching pipeline.
    """
    # Build timeslots that match block type IDs
    # The matching pipeline assigns block_type from instance.block_types keys,
    # so timeslot type_ids must use the same keys.
    timeslots = []

    if instance.timeslots_by_type:
        # Predefined timeslots exist (from sessions.csv) with session-name type_ids.
        # Map them to the block_types keys used by the matching pipeline.
        # Build mapping: session_name -> block_type key (by matching n and k)
        session_to_block_type = {}
        for ts_type_id, ts_list in instance.timeslots_by_type.items():
            for ts in ts_list:
                # Find the block_type in instance.block_types that matches this timeslot's dimensions
                n_rooms = len(ts.get("rooms", []))
                matched = False
                for bt_id, bt_info in instance.block_types.items():
                    if bt_info["n"] == n_rooms:
                        # Check if this timeslot's session name IS the block_type key
                        if ts_type_id == bt_id:
                            matched = True
                            break
                if matched:
                    # Session names are block_type keys — use as-is
                    timeslots.append(ts)
                else:
                    # Session names differ from block_type keys — remap
                    # Find block_type by matching (n, k) dimensions
                    k_talks = instance.block_types.get(ts_type_id, {}).get("k")
                    if k_talks is None:
                        # ts_type_id not in block_types; find by structural match
                        for bt_id, bt_info in instance.block_types.items():
                            if bt_info["n"] == n_rooms:
                                remapped_ts = dict(ts)
                                remapped_ts["type_id"] = bt_id
                                timeslots.append(remapped_ts)
                                matched = True
                                break
                    else:
                        timeslots.append(ts)

        # If remapping didn't produce enough timeslots, fall back to block_types
        if len(timeslots) != len(blocks):
            timeslots = []

    if not timeslots:
        # Build from block_types directly (guaranteed to match block type IDs)
        ts_idx = 0
        for type_id, bt in instance.block_types.items():
            for i in range(bt["count"]):
                ts_idx += 1
                timeslots.append({
                    "id": f"TS_{type_id}_{i}",
                    "type_id": type_id,
                    "rooms": [f"R{r}" for r in range(bt["n"])]
                })

    # Get room capacities from conference data
    room_capacities = instance.conference_data.room_capacities
    if not room_capacities:
        # Default capacities if not specified
        room_capacities = {f"R{r}": 100 for r in range(10)}

    phase3_input = Phase3Input(
        blocks=blocks,
        timeslots=timeslots,
        room_capacities=room_capacities,
        talk_presenter=instance.talk_presenter,
        presenter_unavailability=instance.presenter_unavailability,
        preferences=instance.preferences
    )

    return solve_phase3(
        phase3_input,
        method=method,
        time_limit=time_limit,
        verbose=verbose
    )
