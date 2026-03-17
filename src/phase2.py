"""
Phase 2: Minimize Room Hopping

Given the n-tuples from Phase 1, assemble them into blocks (k consecutive tuples)
to minimize total room hopping across all participants.

A participant "hops" (room switch) when they move between different rooms
to attend their preferred talks in consecutive tuples.

When a participant prefers multiple talks in a tuple, they optimally choose
the one that minimizes total room switches (dynamic programming approach).

This module implements:
1. Room hopping computation for a participant in a block
2. Greedy partition of tuples into blocks (with different sizes)
3. Enumeration-based ordering optimization within each block
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from itertools import permutations
import random


@dataclass
class Block:
    """A block of k ordered tuples."""

    block_id: str              # Unique identifier
    block_type: str            # e.g., '4R3T' (for Phase 3 matching)
    tuples: List[Tuple[str, ...]]  # Ordered list of tuples
    hopping_cost: int          # Total hopping for this block


@dataclass
class FixedSequence:
    """A pre-specified sequence of talks that run sequentially in one room.

    Used for special sessions (e.g., an award or sponsored session) where
    talk groupings are already decided. The sequence is attached to a block
    after Stage 1 partitioning, adding one 'column' (room) to the block.

    Example: A 4R4T block (4 rooms × 4 timeslots) with a SpecialSession
    sequence of 4 talks becomes a 5R4T block (5 rooms × 4 timeslots).
    """
    name: str                           # Descriptive name (e.g., "SpecialSession_1")
    talks: Tuple[str, ...]              # Talk IDs in sequence order (k talks)
    # Block type to attach to (e.g., "4R4T")
    target_block_type: str
    result_block_type: str              # Resulting block type (e.g., "5R4T")


@dataclass
class Phase2Input:
    """Input to Phase 2."""

    # Selected tuples from Phase 1, grouped by tuple size (n)
    # {n: [tuples of size n]}
    tuples_by_n: Dict[int, List[Tuple[str, ...]]]

    # Block specifications: [(n, k, count, block_type), ...]
    # n = number of rooms (parallel talks), k = timeslots, count = how many blocks
    # e.g., [(4, 3, 2, '4R3T'), (5, 4, 1, '5R4T')]
    block_specs: List[Tuple[int, int, int, str]]

    # Participant preferences: {participant_id: set of talk_ids}
    preferences: Dict[str, Set[str]]

    # Pre-specified fixed sequences (optional)
    # These are inserted after Stage 1 partitioning, before Stage 2 ordering
    # Each sequence adds a 'column' to a matching block type
    fixed_sequences: Optional[List[FixedSequence]] = None


@dataclass
class Phase2Result:
    """Output from Phase 2."""

    blocks: List[Block]
    total_hopping: int


# =============================================================================
# Room Hopping Computation (Minimum Room Switches)
# =============================================================================

def compute_participant_hopping(
    block_tuples: List[Tuple[str, ...]],
    participant_prefs: Set[str]
) -> int:
    """
    Compute minimum room switches for one participant in a block.

    Uses dynamic programming to find the optimal choice of rooms when
    the participant prefers multiple talks in a tuple.

    Args:
        block_tuples: Ordered list of tuples in the block (each tuple has n talks in rooms 0..n-1)
        participant_prefs: Set of talk_ids this participant wants to attend

    Returns:
        Minimum number of room switches needed to attend preferred talks
    """
    if not block_tuples:
        return 0

    n_rooms = len(block_tuples[0])
    k = len(block_tuples)

    # Get preferred rooms per tuple
    preferred_rooms = []
    for ntuple in block_tuples:
        rooms = set()
        for r, t in enumerate(ntuple):
            if t in participant_prefs:
                rooms.add(r)
        preferred_rooms.append(rooms)

    # If 0 or 1 tuples have preferences, no room switches
    attended_count = sum(1 for rooms in preferred_rooms if rooms)
    if attended_count <= 1:
        return 0

    # DP: dp[r] = min switches to be in room r at current tuple
    INF = float('inf')
    dp = [INF] * n_rooms

    # Initialize with first tuple that has preferences
    first_idx = None
    for i, rooms in enumerate(preferred_rooms):
        if rooms:
            first_idx = i
            for r in rooms:
                dp[r] = 0
            break

    if first_idx is None:
        return 0

    # Process remaining tuples
    for i in range(first_idx + 1, k):
        if preferred_rooms[i]:
            # Must attend one of these rooms - compute min cost to reach each
            new_dp = [INF] * n_rooms
            for r in preferred_rooms[i]:
                for prev_r in range(n_rooms):
                    if dp[prev_r] < INF:
                        cost = dp[prev_r] + (0 if prev_r == r else 1)
                        new_dp[r] = min(new_dp[r], cost)
            dp = new_dp
        # If no preference at this tuple, dp stays the same (no room switch needed)

    # Return minimum cost at the last attended tuple
    return min(dp[r] for r in range(n_rooms) if dp[r] < INF)


def compute_block_hopping(
    block_tuples: List[Tuple[str, ...]],
    preferences: Dict[str, Set[str]]
) -> int:
    """Compute total room hopping for a block across all participants."""
    total = 0
    for p_id, prefs in preferences.items():
        total += compute_participant_hopping(block_tuples, prefs)
    return total


# =============================================================================
# Ordering Optimization (Stage 2)
# =============================================================================

def optimize_block_ordering(
    tuples: List[Tuple[str, ...]],
    preferences: Dict[str, Set[str]],
    strategy: str = "enumerate"
) -> Tuple[List[Tuple[str, ...]], int]:
    """
    Find optimal ordering of tuples within a block.

    Args:
        tuples: Unordered tuples for one block
        preferences: Participant preferences
        strategy: "enumerate" | "greedy"

    Returns:
        (ordered_tuples, hopping_cost)
    """
    if strategy == "enumerate":
        return _order_by_enumeration(tuples, preferences)
    elif strategy == "greedy":
        return _order_greedy(tuples, preferences)
    else:
        raise ValueError(f"Unknown ordering strategy: {strategy}")


def _order_by_enumeration(
    tuples: List[Tuple[str, ...]],
    preferences: Dict[str, Set[str]]
) -> Tuple[List[Tuple[str, ...]], int]:
    """
    Find optimal ordering by trying all permutations.

    Complexity: O(k! * |P| * k) where k = len(tuples)
    Feasible for k <= 4 (24 permutations max).
    """
    if len(tuples) <= 1:
        return list(tuples), 0

    best_ordering = None
    best_cost = float('inf')

    for perm in permutations(tuples):
        ordered = list(perm)
        cost = compute_block_hopping(ordered, preferences)
        if cost < best_cost:
            best_cost = cost
            best_ordering = ordered

    return best_ordering, best_cost


def _order_greedy(
    tuples: List[Tuple[str, ...]],
    preferences: Dict[str, Set[str]]
) -> Tuple[List[Tuple[str, ...]], int]:
    """
    Greedy ordering: at each position, pick tuple that minimizes incremental hopping.

    This is a fallback for larger k values where enumeration is too slow.
    """
    if len(tuples) <= 1:
        return list(tuples), compute_block_hopping(list(tuples), preferences)

    remaining = list(tuples)
    ordered = []

    while remaining:
        best_next = None
        best_incremental = float('inf')

        for candidate in remaining:
            test_order = ordered + [candidate]
            # Compute partial hopping (approximation)
            cost = compute_block_hopping(test_order, preferences)
            if cost < best_incremental:
                best_incremental = cost
                best_next = candidate

        ordered.append(best_next)
        remaining.remove(best_next)

    final_cost = compute_block_hopping(ordered, preferences)
    return ordered, final_cost


# =============================================================================
# Partition Strategy (Stage 1)
# =============================================================================

def compute_pairwise_compatibility(
    tuple1: Tuple[str, ...],
    tuple2: Tuple[str, ...],
    preferences: Dict[str, Set[str]]
) -> int:
    """
    Compute compatibility score between two tuples.

    Higher score = more compatible (fewer participants attend both).
    If a participant attends both tuples, putting them in the same block
    with a gap would cause hopping.

    Returns negative of "conflict count" - pairs with fewer conflicts are better.
    """
    conflict = 0
    for prefs in preferences.values():
        attends_1 = any(t in prefs for t in tuple1)
        attends_2 = any(t in prefs for t in tuple2)
        if attends_1 and attends_2:
            conflict += 1
    return -conflict  # Higher is better (less conflict)


def partition_tuples_into_blocks(
    tuples_by_n: Dict[int, List[Tuple[str, ...]]],
    block_specs: List[Tuple[int, int, int, str]],
    preferences: Dict[str, Set[str]],
    strategy: str = "greedy",
    verbose: bool = False,
    perturbation_positions: Optional[Set[Tuple[int, int]]] = None,
    forbidden_matchings: Optional[List[List[Tuple[int, int]]]] = None,
    talk_presenter: Optional[Dict[str, str]] = None,
    presenter_unavailability: Optional[Dict[str, Set[str]]] = None,
    all_timeslots: Optional[Set[str]] = None,
    timeslots_by_type: Optional[Dict[str, List[str]]] = None
) -> List[Tuple[List[Tuple[str, ...]], str]]:
    """
    Partition tuples into groups for each block, respecting tuple size constraints.

    Args:
        tuples_by_n: {n: [tuples of size n]} - tuples grouped by size
        block_specs: [(n, k, count, block_type), ...] - block specifications
        preferences: Participant preferences
        strategy: "greedy" | "random" | "matching"
            - greedy: Simple greedy based on conflict count
            - random: Random assignment
            - matching: Paper's approach using assignment problem + perfect matching
        verbose: Print progress (used by matching strategy)
        perturbation_positions: For greedy strategy - set of (block_idx, position)
            where the second-best option should be chosen (for retry logic)
        forbidden_matchings: For matching strategy - list of previous matchings
            to exclude via no-good cuts (for retry logic)
        talk_presenter: Optional mapping talk_id -> presenter_id
        presenter_unavailability: Optional mapping presenter_id -> unavailable timeslots
        all_timeslots: Optional set of all timeslot IDs
        timeslots_by_type: Optional mapping block_type -> list of timeslot IDs

    Returns:
        List of (unordered tuple group, block_type) for each block

    Note: Tuples of size n can only go into blocks with n rooms.
    """
    if strategy == "greedy":
        return _partition_greedy_variable_n(
            tuples_by_n, block_specs, preferences,
            perturbation_positions=perturbation_positions
        )
    elif strategy == "random":
        return _partition_random_variable_n(tuples_by_n, block_specs)
    elif strategy == "matching":
        return partition_tuples_matching_variable_k(
            tuples_by_n, block_specs, preferences, verbose=verbose,
            forbidden_matchings=forbidden_matchings,
            talk_presenter=talk_presenter,
            presenter_unavailability=presenter_unavailability,
            all_timeslots=all_timeslots,
            timeslots_by_type=timeslots_by_type
        )
    else:
        raise ValueError(f"Unknown partition strategy: {strategy}")


def _partition_random_variable_n(
    tuples_by_n: Dict[int, List[Tuple[str, ...]]],
    block_specs: List[Tuple[int, int, int, str]]
) -> List[Tuple[List[Tuple[str, ...]], str]]:
    """
    Random partition respecting tuple size constraints.
    """
    # Make mutable copies
    tuples_by_n = {n: list(tuples) for n, tuples in tuples_by_n.items()}
    for n in tuples_by_n:
        random.shuffle(tuples_by_n[n])

    result = []
    for n, k, count, block_type in block_specs:
        available = tuples_by_n.get(n, [])
        for _ in range(count):
            group = available[:k]
            available = available[k:]
            result.append((group, block_type))
        tuples_by_n[n] = available

    return result


def _partition_greedy_variable_n(
    tuples_by_n: Dict[int, List[Tuple[str, ...]]],
    block_specs: List[Tuple[int, int, int, str]],
    preferences: Dict[str, Set[str]],
    perturbation_positions: Optional[Set[Tuple[int, int]]] = None
) -> List[Tuple[List[Tuple[str, ...]], str]]:
    """
    Greedy partition respecting tuple size constraints.

    For each n, partition the n-tuples into blocks of that size.

    Args:
        tuples_by_n: {n: [tuples of size n]}
        block_specs: [(n, k, count, block_type), ...]
        preferences: Participant preferences
        perturbation_positions: Optional set of (block_idx, position) pairs where
            the second-best option should be chosen instead of the best.
            Used for retry logic when previous partition was infeasible.

    Returns:
        List of (unordered tuple group, block_type) for each block
    """
    if perturbation_positions is None:
        perturbation_positions = set()

    # Make mutable copies
    remaining_by_n = {n: list(tuples) for n, tuples in tuples_by_n.items()}

    result = []
    block_idx = 0

    # Process each block spec
    for n, k, count, block_type in block_specs:
        remaining = remaining_by_n.get(n, [])

        for _ in range(count):
            block_tuples = []
            position = 0

            while len(block_tuples) < k and remaining:
                if not block_tuples:
                    # Pick first tuple
                    chosen = remaining[0]
                else:
                    # Rank all candidates by compatibility score
                    candidates_with_scores = []
                    for candidate in remaining:
                        score = sum(
                            compute_pairwise_compatibility(
                                candidate, member, preferences)
                            for member in block_tuples
                        )
                        candidates_with_scores.append((candidate, score))

                    # Sort by score descending (higher = better)
                    candidates_with_scores.sort(
                        key=lambda x: x[1], reverse=True)

                    # Check if we should perturb (pick second-best)
                    if (block_idx, position) in perturbation_positions and len(candidates_with_scores) > 1:
                        chosen = candidates_with_scores[1][0]  # Second-best
                    else:
                        chosen = candidates_with_scores[0][0]  # Best

                block_tuples.append(chosen)
                remaining.remove(chosen)
                position += 1

            result.append((block_tuples, block_type))
            block_idx += 1

        remaining_by_n[n] = remaining

    return result


# =============================================================================
# Matching-Based Partition (Paper's Approach)
# =============================================================================

def compute_hop_cost_between_talks(
    talk_i: str,
    talk_j: str,
    tuple1: Tuple[str, ...],
    tuple2: Tuple[str, ...],
    preferences: Dict[str, Set[str]]
) -> int:
    """
    Compute d_{i,j}: the number of participants who will hop if talk i (from tuple1)
    and talk j (from tuple2) are scheduled in the same room position consecutively.

    A participant contributes to d_{i,j} if:
    - They want talk i AND want some other talk in tuple2 (not j), OR
    - They want talk j AND want some other talk in tuple1 (not i)

    This counts participants who will be "pulled away" from room position.
    """
    count = 0
    tuple1_set = set(tuple1)
    tuple2_set = set(tuple2)

    for prefs in preferences.values():
        # Case 1: wants talk i, and wants something else in tuple2 (not j)
        wants_i = talk_i in prefs
        wants_other_in_tuple2 = any(
            t in prefs and t != talk_j for t in tuple2_set)

        # Case 2: wants talk j, and wants something else in tuple1 (not i)
        wants_j = talk_j in prefs
        wants_other_in_tuple1 = any(
            t in prefs and t != talk_i for t in tuple1_set)

        if (wants_i and wants_other_in_tuple2) or (wants_j and wants_other_in_tuple1):
            count += 1

    return count


def compute_edge_cost_assignment(
    tuple1: Tuple[str, ...],
    tuple2: Tuple[str, ...],
    preferences: Dict[str, Set[str]]
) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Compute the edge cost between two n-tuples using an assignment problem.

    Build an n×n cost matrix where cost[i][j] = d_{tuple1[i], tuple2[j]}.
    Solve the assignment problem to find the minimum cost way to "align" the tuples.

    Returns:
        (cost, assignment) where assignment is list of (pos1, pos2) pairs
    """
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    n = len(tuple1)
    assert len(tuple2) == n, "Tuples must have same size"

    # Build cost matrix
    cost_matrix = np.zeros((n, n), dtype=int)
    for i, talk_i in enumerate(tuple1):
        for j, talk_j in enumerate(tuple2):
            cost_matrix[i, j] = compute_hop_cost_between_talks(
                talk_i, talk_j, tuple1, tuple2, preferences
            )

    # Solve assignment problem (minimize cost)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    total_cost = cost_matrix[row_ind, col_ind].sum()
    assignment = list(zip(row_ind.tolist(), col_ind.tolist()))

    return total_cost, assignment


def minimum_cost_perfect_matching(
    nodes: List[int],
    edge_costs: Dict[Tuple[int, int], int]
) -> List[Tuple[int, int]]:
    """
    Find minimum cost perfect matching on a complete graph.

    Uses NetworkX's implementation if available, otherwise brute force for small graphs.

    Args:
        nodes: List of node indices (must be even number)
        edge_costs: Dict mapping (i, j) -> cost for each edge

    Returns:
        List of (i, j) pairs forming the matching
    """
    n = len(nodes)
    assert n % 2 == 0, "Need even number of nodes for perfect matching"

    if n == 0:
        return []

    if n == 2:
        return [(nodes[0], nodes[1])]

    try:
        import networkx as nx

        # Build graph - only add edges that exist in edge_costs and are finite
        G = nx.Graph()
        G.add_nodes_from(nodes)

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                ni, nj = nodes[i], nodes[j]
                # Only add edge if it exists in edge_costs
                if (ni, nj) in edge_costs:
                    cost = edge_costs[(ni, nj)]
                elif (nj, ni) in edge_costs:
                    cost = edge_costs[(nj, ni)]
                else:
                    # Skip this edge - it was excluded (e.g., due to conflicts)
                    continue

                # Skip edges with infinite cost (infeasible pairings)
                if cost == float('inf') or cost == float('-inf'):
                    continue

                # NetworkX max_weight_matching maximizes, so negate costs
                # Use a large constant minus cost to avoid negative weights
                # that could cause numerical issues
                G.add_edge(ni, nj, weight=1000000 - cost)

        # Find maximum weight matching (which is minimum cost due to the transform)
        matching = nx.max_weight_matching(G, maxcardinality=True)

        # Check if we got a perfect matching
        if len(matching) != len(nodes) // 2:
            raise ValueError(
                f"Could not find perfect matching: got {len(matching)} pairs for {len(nodes)} nodes. "
                f"Graph has {G.number_of_edges()} edges."
            )

        return list(matching)

    except ImportError:
        # Fallback: brute force for small instances
        if n > 8:
            raise ImportError("NetworkX required for large matching problems")

        from itertools import combinations

        def all_perfect_matchings(nodes):
            """Generate all perfect matchings via recursion."""
            if len(nodes) == 0:
                yield []
                return
            if len(nodes) == 2:
                yield [(nodes[0], nodes[1])]
                return

            first = nodes[0]
            rest = nodes[1:]
            for i, partner in enumerate(rest):
                remaining = rest[:i] + rest[i+1:]
                for sub_matching in all_perfect_matchings(remaining):
                    yield [(first, partner)] + sub_matching

        best_matching = None
        best_cost = float('inf')

        for matching in all_perfect_matchings(nodes):
            cost = sum(
                edge_costs.get((i, j), edge_costs.get((j, i), 0))
                for i, j in matching
            )
            if cost < best_cost:
                best_cost = cost
                best_matching = matching

        return best_matching


def minimum_cost_perfect_matching_milp(
    nodes: List[int],
    edge_costs: Dict[Tuple[int, int], int],
    forbidden_matchings: Optional[List[List[Tuple[int, int]]]] = None,
    verbose: bool = False
) -> List[Tuple[int, int]]:
    """
    Find minimum cost perfect matching via MILP with no-good cuts.

    This version uses Gurobi and supports excluding previous solutions
    via no-good cuts, enabling retry logic when a matching leads to
    infeasible schedules.

    Args:
        nodes: List of node indices (must be even number)
        edge_costs: Dict mapping (i, j) -> cost for each edge
        forbidden_matchings: List of previous matchings to exclude.
            Each matching is a list of (i, j) edge tuples.
        verbose: Print solver output

    Returns:
        List of (i, j) pairs forming the matching
    """
    import gurobipy as gp
    from gurobipy import GRB

    n = len(nodes)
    assert n % 2 == 0, "Need even number of nodes for perfect matching"

    if n == 0:
        return []

    if n == 2:
        # Only one possible matching
        if forbidden_matchings and any(
            set(frozenset(e) for e in m) == {frozenset((nodes[0], nodes[1]))}
            for m in forbidden_matchings
        ):
            raise ValueError("No feasible matching: only option is forbidden")
        return [(nodes[0], nodes[1])]

    if forbidden_matchings is None:
        forbidden_matchings = []

    # Build model
    model = gp.Model("min_cost_perfect_matching")
    model.Params.OutputFlag = 1 if verbose else 0

    # Variables: x[i,j] = 1 if edge (i,j) is in matching
    x = {}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            ni, nj = nodes[i], nodes[j]
            cost = edge_costs.get((ni, nj), edge_costs.get((nj, ni), 0))
            x[ni, nj] = model.addVar(
                vtype=GRB.BINARY, obj=cost, name=f"x_{ni}_{nj}")

    model.ModelSense = GRB.MINIMIZE

    # Constraint: each node matched exactly once
    for node in nodes:
        incident_edges = []
        for other in nodes:
            if other != node:
                key = (min(node, other), max(node, other))
                if key in x:
                    incident_edges.append(x[key])
        model.addConstr(gp.quicksum(incident_edges) == 1, name=f"match_{node}")

    # No-good cuts: exclude previous matchings
    for idx, prev_matching in enumerate(forbidden_matchings):
        # Normalize edges to (min, max) format
        edges_in_prev = []
        for i, j in prev_matching:
            key = (min(i, j), max(i, j))
            if key in x:
                edges_in_prev.append(x[key])

        if edges_in_prev:
            # At least one edge must be different from previous solution
            model.addConstr(
                gp.quicksum(edges_in_prev) <= len(edges_in_prev) - 1,
                name=f"nogood_{idx}"
            )

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        if model.Status == GRB.INFEASIBLE:
            raise ValueError(
                f"No feasible matching found after excluding {len(forbidden_matchings)} previous solutions"
            )
        raise RuntimeError(
            f"Matching solver failed with status {model.Status}")

    # Extract solution
    matching = []
    for (i, j), var in x.items():
        if var.X > 0.5:
            matching.append((i, j))

    return matching


def partition_tuples_matching_based(
    tuples: List[Tuple[str, ...]],
    preferences: Dict[str, Set[str]],
    target_k: int = 4,
    verbose: bool = False,
    forbidden_matchings: Optional[List[List[Tuple[int, int]]]] = None,
    talk_presenter: Optional[Dict[str, str]] = None,
    presenter_unavailability: Optional[Dict[str, Set[str]]] = None,
    all_timeslots: Optional[Set[str]] = None,
    timeslots_by_type: Optional[Dict[str, List[str]]] = None,
    target_block_types: Optional[List[str]] = None
) -> List[List[Tuple[str, ...]]]:
    """
    Partition tuples into k-blocks using the matching-based approach from the paper.

    Algorithm:
    1. Build complete graph G where nodes = tuples
    2. Edge cost = assignment problem solution (optimal room alignment)
    3. Solve min-cost perfect matching to get 2-blocks
    4. If k=4: recursively pair 2-blocks to get 4-blocks

    Presenter Availability:
    If talk_presenter and presenter_unavailability are provided, edges between
    tuples that would create an infeasible block (no common available timeslot)
    are assigned infinite cost (excluded from matching).

    Args:
        tuples: List of n-tuples from Phase 1
        preferences: Participant preferences
        target_k: Target block size (2 or 4)
        verbose: Print progress
        forbidden_matchings: List of previous matchings to exclude via no-good cuts.
            Each matching is a list of (i, j) edge tuples representing which
            tuples were paired together in a previous (infeasible) solution.
        talk_presenter: Optional mapping talk_id -> presenter_id
        presenter_unavailability: Optional mapping presenter_id -> unavailable timeslots
        all_timeslots: Optional set of all timeslot IDs
        timeslots_by_type: Optional mapping block_type -> list of timeslot IDs
        target_block_types: Optional list of block types these tuples will become.
            Used to filter compatibility check to only matching timeslots.

    Returns:
        List of blocks, where each block is a list of k tuples (ordered)
    """
    if verbose:
        print(f"\n--- Matching-based partition (target k={target_k}) ---")
        print(
            f"  Input: {len(tuples)} tuples of size {len(tuples[0]) if tuples else 0}")
        if forbidden_matchings:
            print(
                f"  Excluding {len(forbidden_matchings)} previous matching(s)")
        if talk_presenter and presenter_unavailability:
            print(f"  Using presenter availability to filter incompatible pairings")
        if target_block_types:
            print(f"  Target block types: {target_block_types}")

    n_tuples = len(tuples)

    if n_tuples == 0:
        return []

    if n_tuples == 1:
        return [[tuples[0]]]

    # Need even number for perfect matching
    if n_tuples % 2 != 0:
        raise ValueError(
            f"Need even number of tuples for matching, got {n_tuples}")

    # Step 1: Build edge costs using assignment problem
    if verbose:
        print(
            f"  Computing edge costs ({n_tuples * (n_tuples - 1) // 2} pairs)...")

    edge_costs = {}
    edge_assignments = {}  # Store optimal assignments for later use
    incompatible_count = 0

    # Compute target timeslots based on block types
    # If we know the target block types, only check compatibility against those timeslots
    target_timeslots = all_timeslots
    if target_block_types and timeslots_by_type:
        target_timeslots = set()
        for bt in target_block_types:
            ts_list = timeslots_by_type.get(bt, [])
            for ts in ts_list:
                if isinstance(ts, dict):
                    target_timeslots.add(ts['id'])
                else:
                    target_timeslots.add(ts)
        if verbose and target_timeslots:
            print(f"  Filtering to timeslots: {sorted(target_timeslots)}")

    for i in range(n_tuples):
        for j in range(i + 1, n_tuples):
            # Check if combining these tuples would be feasible
            if talk_presenter and presenter_unavailability and target_timeslots:
                is_compatible, _ = check_tuples_compatible(
                    [tuples[i], tuples[j]],
                    talk_presenter,
                    presenter_unavailability,
                    target_timeslots,  # Use filtered timeslots
                    timeslots_by_type=timeslots_by_type
                )
                if not is_compatible:
                    # Assign very high cost to prevent this pairing
                    edge_costs[(i, j)] = float('inf')
                    edge_assignments[(i, j)] = []
                    incompatible_count += 1
                    continue

            cost, assignment = compute_edge_cost_assignment(
                tuples[i], tuples[j], preferences
            )
            edge_costs[(i, j)] = cost
            edge_assignments[(i, j)] = assignment

    if incompatible_count > 0 and verbose:
        print(
            f"  Excluded {incompatible_count} incompatible pairings (presenter conflicts)")

    # Step 2: Find minimum cost perfect matching to get 2-blocks
    if verbose:
        print(f"  Solving minimum cost perfect matching...")

    nodes = list(range(n_tuples))

    # Use MILP version if we have forbidden matchings (for no-good cuts)
    if forbidden_matchings:
        matching = minimum_cost_perfect_matching_milp(
            nodes, edge_costs,
            forbidden_matchings=forbidden_matchings,
            verbose=verbose
        )
    else:
        matching = minimum_cost_perfect_matching(nodes, edge_costs)

    if verbose:
        matching_cost = sum(edge_costs.get(
            (i, j), edge_costs.get((j, i), 0)) for i, j in matching)
        print(f"  Matching found with cost {matching_cost}")

    # Create 2-blocks with optimal ordering from assignment
    two_blocks = []
    for i, j in matching:
        # Ensure consistent ordering (i < j)
        if i > j:
            i, j = j, i

        # Get the optimal assignment for ordering
        assignment = edge_assignments.get(
            (i, j), edge_assignments.get((j, i), []))

        # The assignment tells us how to align room positions
        # For 2-blocks, ordering is (tuple_i, tuple_j)
        two_blocks.append([tuples[i], tuples[j]])

    if target_k == 2:
        return two_blocks

    if target_k == 4:
        # Step 3: Pair 2-blocks to create 4-blocks
        if len(two_blocks) % 2 != 0:
            raise ValueError(
                f"Need even number of 2-blocks for 4-blocks, got {len(two_blocks)}")

        if verbose:
            print(f"  Building 4-blocks from {len(two_blocks)} 2-blocks...")

        # Compute costs between 2-blocks
        # For 2-blocks (e1, e2) and (e3, e4), there are 4 ways to concatenate:
        # (e1, e2, e3, e4), (e1, e2, e4, e3), (e2, e1, e3, e4), (e2, e1, e4, e3)
        # We compute cost based on the transition between 2nd and 3rd tuple

        block_edge_costs = {}
        block_best_ordering = {}
        block_incompatible_count = 0

        for i in range(len(two_blocks)):
            for j in range(i + 1, len(two_blocks)):
                block_i = two_blocks[i]  # [e1, e2]
                block_j = two_blocks[j]  # [e3, e4]

                # Check if combining these 2-blocks would be feasible
                # Use target_timeslots (already filtered to target block types)
                if talk_presenter and presenter_unavailability and target_timeslots:
                    all_tuples = block_i + block_j  # Combine all 4 tuples
                    is_compatible, _ = check_tuples_compatible(
                        all_tuples,
                        talk_presenter,
                        presenter_unavailability,
                        target_timeslots,  # Use filtered timeslots
                        timeslots_by_type=timeslots_by_type
                    )
                    if not is_compatible:
                        # Skip this pairing entirely - don't add to edge costs
                        # This prevents inf values that cause networkx issues
                        block_incompatible_count += 1
                        continue

                best_cost = float('inf')
                best_order = None

                # Try all 4 concatenation orderings
                orderings = [
                    (block_i[0], block_i[1], block_j[0],
                     block_j[1]),  # e1, e2, e3, e4
                    (block_i[0], block_i[1], block_j[1],
                     block_j[0]),  # e1, e2, e4, e3
                    (block_i[1], block_i[0], block_j[0],
                     block_j[1]),  # e2, e1, e3, e4
                    (block_i[1], block_i[0], block_j[1],
                     block_j[0]),  # e2, e1, e4, e3
                ]

                for ordering in orderings:
                    # Cost is based on transition between position 1→2 and 2→3
                    # (we already accounted for 0→1 and 2→3 within 2-blocks)
                    # Key transition is between ordering[1] and ordering[2]
                    cost, _ = compute_edge_cost_assignment(
                        ordering[1], ordering[2], preferences
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_order = ordering

                block_edge_costs[(i, j)] = best_cost
                block_best_ordering[(i, j)] = best_order

        if block_incompatible_count > 0 and verbose:
            print(
                f"  Excluded {block_incompatible_count} incompatible 2-block pairings (presenter conflicts)")

        # Check if we have enough valid edges for a perfect matching
        if len(block_edge_costs) < len(two_blocks) // 2:
            raise ValueError(
                f"Cannot form 4-blocks: only {len(block_edge_costs)} valid 2-block pairings "
                f"for {len(two_blocks)} 2-blocks. Too many presenter conflicts."
            )

        # Find minimum cost perfect matching on 2-blocks
        block_nodes = list(range(len(two_blocks)))
        block_matching = minimum_cost_perfect_matching(
            block_nodes, block_edge_costs)

        # Create 4-blocks
        four_blocks = []
        for i, j in block_matching:
            if i > j:
                i, j = j, i
            ordering = block_best_ordering.get(
                (i, j), block_best_ordering.get((j, i)))
            if ordering is None:
                # This shouldn't happen after the fix, but guard against it
                raise ValueError(
                    f"No valid ordering found for 2-block pair ({i}, {j}). "
                    f"This indicates a matching was found with an incompatible pair."
                )
            four_blocks.append(list(ordering))

        if verbose:
            print(f"  Created {len(four_blocks)} 4-blocks")

        return four_blocks

    if target_k == 3:
        # For 3-blocks: use dummy tuple approach from paper
        # This is more complex - implement if needed
        raise NotImplementedError("3-blocks via matching not yet implemented")

    raise ValueError(f"Unsupported target_k={target_k}, must be 2 or 4")


def partition_tuples_matching_variable_k(
    tuples_by_n: Dict[int, List[Tuple[str, ...]]],
    block_specs: List[Tuple[int, int, int, str]],
    preferences: Dict[str, Set[str]],
    verbose: bool = False,
    forbidden_matchings: Optional[List[List[Tuple[int, int]]]] = None,
    talk_presenter: Optional[Dict[str, str]] = None,
    presenter_unavailability: Optional[Dict[str, Set[str]]] = None,
    all_timeslots: Optional[Set[str]] = None,
    timeslots_by_type: Optional[Dict[str, List[str]]] = None
) -> List[Tuple[List[Tuple[str, ...]], str]]:
    """
    Partition tuples using the matching-based approach, supporting variable k.

    This is the main entry point for the paper's approach.

    Args:
        tuples_by_n: {n: [tuples of size n]}
        block_specs: [(n, k, count, block_type), ...]
        preferences: Participant preferences
        verbose: Print progress
        forbidden_matchings: List of previous matchings to exclude via no-good cuts
        talk_presenter: Optional mapping talk_id -> presenter_id
        presenter_unavailability: Optional mapping presenter_id -> unavailable timeslots
        all_timeslots: Optional set of all timeslot IDs
        timeslots_by_type: Optional mapping block_type -> list of timeslot IDs

    Returns:
        List of (tuple_group, block_type) for each block
    """
    result = []
    remaining_by_n = {n: list(tuples) for n, tuples in tuples_by_n.items()}

    # Group block specs by (n, k)
    from collections import defaultdict
    specs_by_nk = defaultdict(list)
    for n, k, count, block_type in block_specs:
        specs_by_nk[(n, k)].append((count, block_type))

    for (n, k), specs in specs_by_nk.items():
        total_count = sum(count for count, _ in specs)
        tuples_needed = total_count * k

        available = remaining_by_n.get(n, [])
        if len(available) < tuples_needed:
            raise ValueError(
                f"Not enough {n}-tuples: need {tuples_needed}, have {len(available)}"
            )

        # Take the tuples we need
        tuples_to_partition = available[:tuples_needed]
        remaining_by_n[n] = available[tuples_needed:]

        # Get the block types for this partition
        target_block_types = [bt for _, bt in specs]

        if verbose:
            print(
                f"\nPartitioning {len(tuples_to_partition)} {n}-tuples into {total_count} {k}-blocks")
            print(f"  Target block types: {target_block_types}")

        if k in (2, 4):
            # Use matching-based approach, with fallback to greedy if matching fails
            try:
                blocks = partition_tuples_matching_based(
                    tuples_to_partition, preferences, target_k=k, verbose=verbose,
                    forbidden_matchings=forbidden_matchings,
                    talk_presenter=talk_presenter,
                    presenter_unavailability=presenter_unavailability,
                    all_timeslots=all_timeslots,
                    timeslots_by_type=timeslots_by_type,
                    target_block_types=target_block_types
                )
            except ValueError as e:
                # Perfect matching failed (e.g., too many presenter conflicts)
                # Fall back to greedy approach which handles infeasibility gracefully
                import warnings
                warnings.warn(
                    f"Matching-based partitioning failed for k={k}: {e}. "
                    f"Falling back to greedy approach (solution quality may be reduced)."
                )
                if verbose:
                    print(f"  ⚠ Matching failed: {e}")
                    print(f"  ⚠ Falling back to greedy partitioning...")
                blocks = _partition_tuples_greedy_compatible(
                    tuples_to_partition, preferences, k,
                    talk_presenter, presenter_unavailability,
                    all_timeslots, timeslots_by_type, verbose,
                    target_block_types=target_block_types
                )
        else:
            # Greedy with compatibility check for other k values (e.g., k=3)
            if verbose:
                print(f"  (Using greedy with compatibility for k={k})")
            blocks = _partition_tuples_greedy_compatible(
                tuples_to_partition, preferences, k,
                talk_presenter, presenter_unavailability,
                all_timeslots, timeslots_by_type, verbose,
                target_block_types=target_block_types
            )

        # Smart block type assignment based on forbidden block types
        # We need to assign block types such that blocks containing restricted talks
        # don't get assigned to block types they're forbidden from
        result.extend(_assign_block_types_smart(
            blocks, specs, talk_presenter, presenter_unavailability,
            all_timeslots, timeslots_by_type, verbose
        ))

    return result


def _assign_block_types_smart(
    blocks: List[List[Tuple[str, ...]]],
    specs: List[Tuple[int, str]],  # [(count, block_type), ...]
    talk_presenter: Optional[Dict[str, str]],
    presenter_unavailability: Optional[Dict[str, Set[str]]],
    all_timeslots: Optional[Set[str]],
    timeslots_by_type: Optional[Dict[str, List[str]]],
    verbose: bool = False
) -> List[Tuple[List[Tuple[str, ...]], str]]:
    """
    Assign block types to blocks, respecting presenter availability.

    Some blocks contain talks whose presenters cannot attend certain block types
    (because all timeslots of that block type are unavailable to the presenter).

    This function assigns block types to minimize violations by:
    1. Computing which block types each block is forbidden from
    2. Assigning block types so restricted blocks avoid their forbidden types

    Args:
        blocks: List of blocks (each block is a list of tuples)
        specs: [(count, block_type), ...] - how many of each block type we need
        talk_presenter: mapping talk_id -> presenter_id
        presenter_unavailability: mapping presenter_id -> unavailable timeslots
        all_timeslots: set of all timeslot IDs
        timeslots_by_type: mapping block_type -> list of timeslot dicts with 'id' key

    Returns:
        List of (block, block_type) pairs
    """
    total_blocks = len(blocks)
    total_slots = sum(count for count, _ in specs)

    if total_blocks > total_slots:
        raise ValueError(
            f"More blocks ({total_blocks}) than type slots ({total_slots})")

    # If no presenter info, fall back to simple sequential assignment
    if not talk_presenter or not presenter_unavailability or not timeslots_by_type:
        result = []
        block_idx = 0
        for count, block_type in specs:
            for _ in range(count):
                if block_idx < len(blocks):
                    result.append((blocks[block_idx], block_type))
                    block_idx += 1
        return result

    # Build mapping: block_type -> set of timeslot IDs
    type_to_timeslots = {}
    for type_id, ts_list in timeslots_by_type.items():
        type_to_timeslots[type_id] = set()
        for ts in ts_list:
            if isinstance(ts, dict):
                type_to_timeslots[type_id].add(ts['id'])
            else:
                type_to_timeslots[type_id].add(ts)

    # For each block, compute which block types it is forbidden from
    block_forbidden_types: List[Set[str]] = []
    block_types_in_specs = set(bt for _, bt in specs)

    for block in blocks:
        forbidden = set()

        # Get all talks in this block
        all_talks = set()
        for tuple_ in block:
            all_talks.update(tuple_)

        # For each block type in specs, check if any talk's presenter can't attend
        for block_type in block_types_in_specs:
            timeslots = type_to_timeslots.get(block_type, set())
            if not timeslots:
                continue

            # Check each talk's presenter
            for talk_id in all_talks:
                presenter = talk_presenter.get(talk_id)
                if not presenter or presenter not in presenter_unavailability:
                    continue

                unavail = presenter_unavailability[presenter]
                avail_in_type = timeslots - unavail

                if not avail_in_type:
                    # Presenter can't attend any timeslot of this block type
                    forbidden.add(block_type)
                    break  # No need to check more talks

        block_forbidden_types.append(forbidden)

    if verbose:
        restricted_blocks = [(i, f)
                             for i, f in enumerate(block_forbidden_types) if f]
        if restricted_blocks:
            print(f"  Block type restrictions:")
            for i, forbidden in restricted_blocks:
                talks = set()
                for tuple_ in blocks[i]:
                    talks.update(tuple_)
                print(
                    f"    Block {i} (talks: {sorted(talks)[:3]}...): forbidden from {forbidden}")

    # Now assign block types using a greedy approach:
    # - First assign blocks with restrictions to types they CAN use
    # - Then fill remaining slots

    # Build list of (block_idx, forbidden_types) sorted by number of restrictions (most restricted first)
    block_indices = list(range(len(blocks)))
    block_indices.sort(key=lambda i: (-len(block_forbidden_types[i]), i))

    # Track available slots per block type
    available_slots = {}
    for count, block_type in specs:
        available_slots[block_type] = available_slots.get(
            block_type, 0) + count

    # Assign block types
    result: List[Optional[Tuple[List[Tuple[str, ...]], str]]] = [
        None] * len(blocks)

    for block_idx in block_indices:
        forbidden = block_forbidden_types[block_idx]

        # Find a block type this block can use (has slots and not forbidden)
        assigned = False
        for block_type in available_slots:
            if available_slots[block_type] > 0 and block_type not in forbidden:
                result[block_idx] = (blocks[block_idx], block_type)
                available_slots[block_type] -= 1
                assigned = True
                break

        if not assigned:
            # No compatible type found - this is a constraint violation
            # Fall back to any available type
            for block_type in available_slots:
                if available_slots[block_type] > 0:
                    if verbose:
                        print(
                            f"    WARNING: Block {block_idx} assigned to forbidden type {block_type}")
                    result[block_idx] = (blocks[block_idx], block_type)
                    available_slots[block_type] -= 1
                    break

    # Filter out None entries (shouldn't happen, but be safe)
    return [r for r in result if r is not None]


# =============================================================================
# Local Search Improvement
# =============================================================================

def local_search_swap(
    partition: List[Tuple[List[Tuple[str, ...]], str]],
    preferences: Dict[str, Set[str]],
    max_iterations: int = 1000
) -> List[Tuple[List[Tuple[str, ...]], str]]:
    """
    Improve partition by swapping tuples between blocks with same tuple size (n).

    Tries random swaps and accepts improvements.
    Note: Only swaps between blocks with same n (tuple size) are valid.
    """
    # Group blocks by tuple size (n = len of tuples in the group)
    n_to_indices = {}
    for i, (group, block_type) in enumerate(partition):
        if group:
            n = len(group[0])  # Tuple size
            if n not in n_to_indices:
                n_to_indices[n] = []
            n_to_indices[n].append(i)

    current = [list(g) for g, bt in partition]
    block_types = [bt for _, bt in partition]

    # Cache per-block costs to avoid recomputing all blocks on every iteration
    block_costs = []
    for group in current:
        _, cost = optimize_block_ordering(group, preferences)
        block_costs.append(cost)
    current_cost = sum(block_costs)

    for iteration in range(max_iterations):
        # Pick a random tuple size (n) that has at least 2 blocks
        swappable_ns = [n for n, indices in n_to_indices.items()
                        if len(indices) >= 2]
        if not swappable_ns:
            break

        n = random.choice(swappable_ns)
        indices = n_to_indices[n]

        # Pick two random blocks with this tuple size
        i1, i2 = random.sample(indices, 2)

        # Pick random tuples to swap
        pos1 = random.randrange(len(current[i1]))
        pos2 = random.randrange(len(current[i2]))

        # Swap
        current[i1][pos1], current[i2][pos2] = current[i2][pos2], current[i1][pos1]

        # Recompute only the two affected blocks
        new_cost_i1 = optimize_block_ordering(current[i1], preferences)[1]
        new_cost_i2 = optimize_block_ordering(current[i2], preferences)[1]
        new_cost = current_cost - block_costs[i1] - block_costs[i2] + new_cost_i1 + new_cost_i2

        if new_cost < current_cost:
            block_costs[i1] = new_cost_i1
            block_costs[i2] = new_cost_i2
            current_cost = new_cost
        else:
            # Revert swap
            current[i1][pos1], current[i2][pos2] = current[i2][pos2], current[i1][pos1]

    return [(tuple(g), bt) for g, bt in zip(current, block_types)]


def _partition_tuples_greedy_compatible(
    tuples: List[Tuple[str, ...]],
    preferences: Dict[str, Set[str]],
    k: int,
    talk_presenter: Optional[Dict[str, str]],
    presenter_unavailability: Optional[Dict[str, Set[str]]],
    all_timeslots: Optional[Set[str]],
    timeslots_by_type: Optional[Dict[str, List[str]]],
    verbose: bool = False,
    target_block_types: Optional[List[str]] = None
) -> List[List[Tuple[str, ...]]]:
    """
    Greedy partitioning of tuples into k-blocks with compatibility checking.

    For k values not handled by matching (like k=3), we use a greedy approach
    that respects presenter availability constraints.

    Algorithm:
    1. Start with the first unassigned tuple
    2. Find compatible tuples to add to the block (those that don't create 
       infeasible presenter combinations)
    3. Among compatible tuples, pick the one with minimum room hopping cost
    4. Repeat until block has k tuples
    5. Start a new block with remaining tuples

    Args:
        tuples: List of tuples to partition
        preferences: Participant preferences for hopping cost
        k: Target block size (number of tuples per block)
        talk_presenter: Mapping talk_id -> presenter_id
        presenter_unavailability: Mapping presenter_id -> unavailable timeslots
        all_timeslots: Set of all timeslot IDs
        timeslots_by_type: Mapping block_type -> list of timeslot IDs
        target_block_types: List of block types these tuples will become

    Returns:
        List of blocks, where each block is a list of k tuples
    """
    if len(tuples) % k != 0:
        raise ValueError(
            f"Cannot evenly partition {len(tuples)} tuples into blocks of {k}")

    # Compute target timeslots based on block types
    target_timeslots = all_timeslots
    if target_block_types and timeslots_by_type:
        target_timeslots = set()
        for bt in target_block_types:
            ts_list = timeslots_by_type.get(bt, [])
            for ts in ts_list:
                if isinstance(ts, dict):
                    target_timeslots.add(ts['id'])
                else:
                    target_timeslots.add(ts)

    blocks = []
    remaining = list(tuples)
    incompatible_skips = 0

    while remaining:
        # Start a new block with the first available tuple
        block = [remaining.pop(0)]

        # Add k-1 more tuples to complete the block
        while len(block) < k and remaining:
            best_tuple = None
            best_cost = float('inf')
            best_idx = -1

            for idx, candidate in enumerate(remaining):
                # Check if adding this tuple maintains compatibility
                if talk_presenter and presenter_unavailability and target_timeslots:
                    test_block = block + [candidate]
                    is_compatible, _ = check_tuples_compatible(
                        test_block,
                        talk_presenter,
                        presenter_unavailability,
                        target_timeslots,  # Use filtered timeslots
                        timeslots_by_type=timeslots_by_type
                    )
                    if not is_compatible:
                        incompatible_skips += 1
                        continue

                # Compute cost based on transition from last tuple in block
                cost, _ = compute_edge_cost_assignment(
                    block[-1], candidate, preferences)

                if cost < best_cost:
                    best_cost = cost
                    best_tuple = candidate
                    best_idx = idx

            if best_tuple is not None:
                block.append(best_tuple)
                remaining.pop(best_idx)
            else:
                # No compatible tuple found - fall back to first available
                if remaining:
                    if verbose:
                        print(f"  ⚠ No compatible tuple found, using fallback")
                    block.append(remaining.pop(0))

        blocks.append(block)

    if incompatible_skips > 0 and verbose:
        print(
            f"  Skipped {incompatible_skips} incompatible pairings during greedy partitioning")

    return blocks


# =============================================================================
# Tuple Compatibility Check (for Phase 2 partitioning)
# =============================================================================

def check_tuples_compatible(
    tuples: List[Tuple[str, ...]],
    talk_presenter: Dict[str, str],
    presenter_unavailability: Dict[str, Set[str]],
    all_timeslots: Set[str],
    block_type: Optional[str] = None,
    timeslots_by_type: Optional[Dict[str, List[str]]] = None
) -> Tuple[bool, Set[str]]:
    """
    Check if combining multiple tuples would create a feasible block.

    This is used during Phase 2 partitioning to prevent grouping tuples
    whose combined presenter unavailabilities leave no valid timeslot.

    Args:
        tuples: List of tuples to potentially combine
        talk_presenter: Mapping talk_id -> presenter_id
        presenter_unavailability: Mapping presenter_id -> unavailable timeslots
        all_timeslots: Set of all timeslot IDs
        block_type: Optional block type for type-aware checking
        timeslots_by_type: Optional mapping block_type -> list of timeslot IDs

    Returns:
        Tuple of (is_compatible, available_timeslots)
    """
    # Collect all unavailable timeslots for all presenters across all tuples
    blocked_timeslots = set()

    for ntuple in tuples:
        for talk_id in ntuple:
            presenter = talk_presenter.get(talk_id)
            if presenter:
                unavailable = presenter_unavailability.get(presenter, set())
                blocked_timeslots |= unavailable

    available = all_timeslots - blocked_timeslots

    # If we have block type info, further filter by matching timeslots
    if timeslots_by_type and block_type:
        matching_timeslots = set(timeslots_by_type.get(block_type, []))
        available = available & matching_timeslots

    is_compatible = len(available) > 0

    return is_compatible, available


# =============================================================================
# Block Feasibility Check
# =============================================================================

def check_block_feasibility(
    block: Block,
    talk_presenter: Dict[str, str],
    presenter_unavailability: Dict[str, Set[str]],
    all_timeslots: Set[str],
    timeslots_by_type: Optional[Dict[str, List[str]]] = None
) -> Tuple[bool, Set[str]]:
    """
    Check if a block can be scheduled in at least one timeslot of matching type.

    A block is infeasible if:
    - The union of all presenters' unavailable timeslots covers ALL available timeslots, OR
    - None of the timeslots matching the block's type are available to all presenters

    Args:
        block: The block to check
        talk_presenter: Mapping talk_id -> presenter_id
        presenter_unavailability: Mapping presenter_id -> set of unavailable timeslots
        all_timeslots: Set of all timeslot IDs (e.g., {"TA", "TB", "TC", "TD", "FA", "FB", "FC"})
        timeslots_by_type: Optional mapping block_type -> list of timeslot IDs of that type

    Returns:
        Tuple of (is_feasible, available_timeslots)
    """
    # Collect all unavailable timeslots for all presenters in this block
    blocked_timeslots = set()

    for ntuple in block.tuples:
        for talk_id in ntuple:
            presenter = talk_presenter.get(talk_id)
            if presenter:
                unavailable = presenter_unavailability.get(presenter, set())
                blocked_timeslots |= unavailable

    available = all_timeslots - blocked_timeslots

    # If we have block type info, further filter by matching timeslots
    if timeslots_by_type and block.block_type:
        matching_timeslots = set(timeslots_by_type.get(block.block_type, []))
        available = available & matching_timeslots

    is_feasible = len(available) > 0

    return is_feasible, available


def check_all_blocks_feasibility(
    blocks: List[Block],
    talk_presenter: Dict[str, str],
    presenter_unavailability: Dict[str, Set[str]],
    all_timeslots: Set[str],
    verbose: bool = False,
    timeslots_by_type: Optional[Dict[str, List[str]]] = None
) -> Tuple[bool, List[int]]:
    """
    Check feasibility of all blocks.

    Args:
        blocks: List of blocks to check
        talk_presenter: Mapping talk_id -> presenter_id
        presenter_unavailability: Mapping presenter_id -> unavailable timeslots
        all_timeslots: Set of all timeslot IDs
        verbose: Whether to print details
        timeslots_by_type: Optional mapping block_type -> list of timeslot IDs

    Returns:
        Tuple of (all_feasible, list of infeasible block indices)
    """
    infeasible_indices = []

    for idx, block in enumerate(blocks):
        is_feasible, available = check_block_feasibility(
            block, talk_presenter, presenter_unavailability, all_timeslots,
            timeslots_by_type=timeslots_by_type
        )
        if not is_feasible:
            infeasible_indices.append(idx)
            if verbose:
                print(
                    f"  ⚠ Block {block.block_id} (type={block.block_type}) is INFEASIBLE")
                print(
                    f"      No available timeslots of matching type")

    all_feasible = len(infeasible_indices) == 0
    return all_feasible, infeasible_indices


# =============================================================================
# Main Phase 2 Solver
# =============================================================================

def solve_phase2(
    phase2_input: Phase2Input,
    partition_strategy: str = "greedy",
    ordering_strategy: str = "enumerate",
    use_local_search: bool = True,
    local_search_iterations: int = 1000,
    verbose: bool = True,
    perturbation_positions: Optional[Set[Tuple[int, int]]] = None,
    forbidden_matchings: Optional[List[List[Tuple[int, int]]]] = None,
    talk_presenter: Optional[Dict[str, str]] = None,
    presenter_unavailability: Optional[Dict[str, Set[str]]] = None,
    all_timeslots: Optional[Set[str]] = None,
    timeslots_by_type: Optional[Dict[str, List[str]]] = None
) -> Phase2Result:
    """
    Solve Phase 2: Minimize total hopping.

    Args:
        phase2_input: Input containing tuples_by_n, block_specs, and preferences
        partition_strategy: "greedy" | "random" | "matching"
        ordering_strategy: "enumerate" | "greedy"
        use_local_search: Whether to apply local search after initial partition
        local_search_iterations: Max iterations for local search
        verbose: Print progress
        perturbation_positions: For greedy strategy - positions to perturb (retry logic)
        forbidden_matchings: For matching strategy - previous matchings to exclude (retry logic)
        talk_presenter: Optional mapping talk_id -> presenter_id (for presenter availability)
        presenter_unavailability: Optional mapping presenter_id -> unavailable timeslots
        all_timeslots: Optional set of all timeslot IDs
        timeslots_by_type: Optional mapping block_type -> list of timeslot IDs

    Returns:
        Phase2Result with assembled blocks
    """
    # Compute totals for display
    total_tuples = sum(len(tuples)
                       for tuples in phase2_input.tuples_by_n.values())

    if verbose:
        print("=" * 70)
        print("PHASE 2: MINIMIZE SESSION HOPPING")
        print("=" * 70)
        print(f"Input: {total_tuples} tuples")
        for n, tuples in sorted(phase2_input.tuples_by_n.items()):
            print(f"  - {len(tuples)} tuples of size {n}")
        print(f"Block specs: {phase2_input.block_specs}")
        print(f"Participants: {len(phase2_input.preferences)}")
        print(f"Partition strategy: {partition_strategy}")
        print(f"Ordering strategy: {ordering_strategy}")
        print(f"Local search: {use_local_search}")
        if perturbation_positions:
            print(f"Perturbation positions: {perturbation_positions}")
        if forbidden_matchings:
            print(
                f"Forbidden matchings: {len(forbidden_matchings)} previous solution(s)")

    # Stage 1: Partition tuples into blocks
    if verbose:
        print("\n--- Stage 1: Partitioning tuples into blocks ---")

    partition = partition_tuples_into_blocks(
        phase2_input.tuples_by_n,
        phase2_input.block_specs,
        phase2_input.preferences,
        strategy=partition_strategy,
        verbose=verbose,
        perturbation_positions=perturbation_positions,
        forbidden_matchings=forbidden_matchings,
        talk_presenter=talk_presenter,
        presenter_unavailability=presenter_unavailability,
        all_timeslots=all_timeslots,
        timeslots_by_type=timeslots_by_type
    )

    if verbose:
        print(f"Created {len(partition)} blocks")
        for i, (group, block_type) in enumerate(partition):
            n = len(group[0]) if group else 0
            print(
                f"  Block {i+1} ({block_type}): {len(group)} tuples of size {n}")

    # Optional: Local search to improve partition
    if use_local_search:
        if verbose:
            print(
                f"\n--- Local Search ({local_search_iterations} iterations) ---")

        partition = local_search_swap(
            partition,
            phase2_input.preferences,
            max_iterations=local_search_iterations
        )

    # Stage 2: Optimize ordering within each block
    # Note: For matching strategy, tuples are already ordered; re-optimize to be safe
    if verbose:
        print("\n--- Stage 2: Optimizing tuple ordering within blocks ---")

    # -------------------------------------------------------------------------
    # Insert fixed sequences into matching blocks (between Stage 1 and ordering)
    # -------------------------------------------------------------------------
    # Each fixed sequence adds a "column" (one room) to a block of matching type
    # E.g., a SpecialSession sequence of 4 talks attached to a 4R4T block → 5R4T block

    if phase2_input.fixed_sequences:
        if verbose:
            print(
                f"\n  Inserting {len(phase2_input.fixed_sequences)} fixed sequences...")

        # Build a mutable list of partitions to modify
        partition_list = list(partition)

        for fixed_seq in phase2_input.fixed_sequences:
            # Find a block of the target type to attach to
            attached = False
            for idx, (tuple_group, block_type) in enumerate(partition_list):
                if block_type == fixed_seq.target_block_type:
                    # Verify the sequence length matches block's k (number of tuples)
                    k = len(tuple_group)
                    if len(fixed_seq.talks) != k:
                        print(f"  ⚠ Warning: {fixed_seq.name} has {len(fixed_seq.talks)} talks "
                              f"but block has {k} timeslots. Skipping.")
                        continue

                    # Expand each tuple by adding one talk from the sequence
                    # tuple_group[i] + (fixed_seq.talks[i],) for each timeslot i
                    expanded_tuples = []
                    for i, tup in enumerate(tuple_group):
                        expanded_tuple = tuple(
                            list(tup) + [fixed_seq.talks[i]])
                        expanded_tuples.append(expanded_tuple)

                    # Update partition with expanded tuples and new block type
                    partition_list[idx] = (
                        expanded_tuples, fixed_seq.result_block_type)
                    attached = True

                    if verbose:
                        print(f"  ✓ {fixed_seq.name}: attached to block {idx+1} "
                              f"({fixed_seq.target_block_type} → {fixed_seq.result_block_type})")
                    break

            if not attached:
                print(
                    f"  ⚠ Warning: No {fixed_seq.target_block_type} block found for {fixed_seq.name}")

        # Update partition
        partition = partition_list

    # -------------------------------------------------------------------------
    # Ordering optimization for all blocks
    # -------------------------------------------------------------------------
    blocks = []
    total_hopping = 0
    block_counter = 1  # For generating block IDs

    # Process all partitioned blocks (including those with attached sequences)
    for i, (tuple_group, block_type) in enumerate(partition):
        ordered_tuples, cost = optimize_block_ordering(
            list(tuple_group),
            phase2_input.preferences,
            strategy=ordering_strategy
        )

        block = Block(
            block_id=f"B{block_counter:02d}",
            block_type=block_type,
            tuples=ordered_tuples,
            hopping_cost=cost
        )
        blocks.append(block)
        total_hopping += cost

        if verbose:
            print(f"  {block.block_id} ({block_type}): hopping = {cost}")

        block_counter += 1

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"TOTAL HOPPING: {total_hopping}")
        if phase2_input.fixed_sequences:
            print(
                f"  ({len(phase2_input.fixed_sequences)} fixed sequences attached)")
        print(f"{'=' * 70}")

    return Phase2Result(blocks=blocks, total_hopping=total_hopping)


# =============================================================================
# Legacy Functions (for backwards compatibility)
# =============================================================================

def compute_hopping_number(
    block_tuples: List[Tuple[str, ...]],
    participant_prefs: Set[str]
) -> int:
    """Alias for compute_participant_hopping."""
    return compute_participant_hopping(block_tuples, participant_prefs)


def compute_hop_coefficient(
    block_tuples: List[Tuple[str, ...]],
    preferences: Dict[str, Set[str]]
) -> int:
    """Alias for compute_block_hopping."""
    return compute_block_hopping(block_tuples, preferences)


# =============================================================================
# Phase 2 with Feasibility Retry Loop
# =============================================================================

@dataclass
class Phase2ResultWithMatching(Phase2Result):
    """Phase2Result extended with matching information for retry logic."""
    matching_used: Optional[List[Tuple[int, int]]] = None


def solve_phase2_with_feasibility_check(
    phase2_input: Phase2Input,
    talk_presenter: Dict[str, str],
    presenter_unavailability: Dict[str, Set[str]],
    all_timeslots: Set[str],
    partition_strategy: str = "greedy",
    ordering_strategy: str = "enumerate",
    use_local_search: bool = True,
    local_search_iterations: int = 1000,
    max_retries: int = 10,
    verbose: bool = True,
    timeslots_by_type: Optional[Dict[str, List[str]]] = None
) -> Tuple[Phase2Result, bool]:
    """
    Solve Phase 2 with automatic retry on presenter infeasibility.

    If the resulting blocks are infeasible (no valid timeslot for some block
    due to presenter unavailabilities), the algorithm retries with:
    - Greedy strategy: perturb positions in infeasible blocks
    - Matching strategy: exclude the previous matching via no-good cut

    Args:
        phase2_input: Phase 2 input data
        talk_presenter: Mapping talk_id -> presenter_id
        presenter_unavailability: Mapping presenter_id -> unavailable timeslots
        all_timeslots: Set of all timeslot IDs
        partition_strategy: "greedy" | "matching"
        ordering_strategy: "enumerate" | "greedy"
        use_local_search: Whether to apply local search
        local_search_iterations: Max iterations for local search
        max_retries: Maximum number of retry attempts
        verbose: Print progress
        timeslots_by_type: Mapping block_type -> list of timeslot IDs (for type-aware feasibility)

    Returns:
        Tuple of (Phase2Result, is_feasible)
    """
    # Track state for retries
    perturbation_positions: Set[Tuple[int, int]] = set()
    forbidden_matchings: List[List[Tuple[int, int]]] = []

    # Show presenter availability info
    if verbose and presenter_unavailability:
        constrained_presenters = len(presenter_unavailability)
        print(
            f"\n  📅 Presenter availability constraints: {constrained_presenters} presenters have restrictions")
        print(
            f"     (max {max_retries} retry attempts if partition is infeasible)")

    for attempt in range(max_retries + 1):
        if verbose and attempt > 0:
            print(f"\n{'='*70}")
            print(f"PHASE 2 RETRY ATTEMPT {attempt}/{max_retries}")
            print(f"{'='*70}")

        try:
            # Run Phase 2 (passing presenter availability for matching strategy)
            result = solve_phase2(
                phase2_input,
                partition_strategy=partition_strategy,
                ordering_strategy=ordering_strategy,
                use_local_search=use_local_search,
                local_search_iterations=local_search_iterations,
                verbose=verbose,
                perturbation_positions=perturbation_positions if partition_strategy == "greedy" else None,
                forbidden_matchings=forbidden_matchings if partition_strategy == "matching" else None,
                talk_presenter=talk_presenter,
                presenter_unavailability=presenter_unavailability,
                all_timeslots=all_timeslots,
                timeslots_by_type=timeslots_by_type
            )

            # Check feasibility (with block type awareness if available)
            all_feasible, infeasible_indices = check_all_blocks_feasibility(
                result.blocks,
                talk_presenter,
                presenter_unavailability,
                all_timeslots,
                verbose=verbose,
                timeslots_by_type=timeslots_by_type
            )

            if all_feasible:
                # Always show success (important status info)
                if attempt > 0:
                    print(
                        f"  ✓ Found feasible partition after {attempt} retries")
                else:
                    print(f"  ✓ Partition feasible for presenter constraints")
                return result, True

            if attempt >= max_retries:
                # Always show failure (important status info)
                print(
                    f"  ⚠ Could not find feasible partition after {max_retries} retries")
                if verbose:
                    print(
                        f"    Infeasible blocks: {[result.blocks[i].block_id for i in infeasible_indices]}")
                return result, False

            # Prepare for retry
            if verbose:
                print(
                    f"\n  Found {len(infeasible_indices)} infeasible block(s), preparing retry...")

            if partition_strategy == "greedy":
                # Add perturbation at position 1 of each infeasible block
                for block_idx in infeasible_indices:
                    # Try different positions on each retry
                    position_to_perturb = attempt % 4  # Cycle through positions 0-3
                    perturbation_positions.add(
                        (block_idx, position_to_perturb))
                if verbose:
                    print(
                        f"  Greedy: perturbing positions {perturbation_positions}")

            elif partition_strategy == "matching":
                # Extract the matching used (we need to track this)
                # For now, we'll add a dummy forbidden matching based on block structure
                # A more sophisticated approach would track the actual MILP matching
                matching_edges = []
                for block in result.blocks:
                    # Each block came from pairing tuples - reconstruct approximately
                    # This is a simplification; proper tracking would be better
                    if len(block.tuples) >= 2:
                        # Use tuple indices as proxy (this is approximate)
                        for i in range(0, len(block.tuples) - 1, 2):
                            matching_edges.append((i, i + 1))
                if matching_edges:
                    forbidden_matchings.append(matching_edges)
                if verbose:
                    print(
                        f"  Matching: now excluding {len(forbidden_matchings)} previous solution(s)")

        except ValueError as e:
            if "No feasible matching" in str(e):
                if verbose:
                    print(f"\n✗ No more feasible matchings available: {e}")
                # Return last result if available
                return result if 'result' in locals() else Phase2Result(blocks=[], total_hopping=0), False
            raise

    # Should not reach here
    return result, False
