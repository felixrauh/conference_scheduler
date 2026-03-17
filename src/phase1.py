"""
Phase 1: Maximize Total Attendance

Solves CSP-n to determine which talks run in parallel,
minimizing missed attendance based on participant preferences.

Supports both:
- Fixed n: Original paper formulation (all blocks have same n)
- Variable n: Extended formulation with different tuple sizes

Supports two solution methods:
- Explicit enumeration: Heuristic-filtered full enumeration (good for small instances)
- Column generation: Only generates promising tuples (scales to large instances)

Based on Vangerven et al. (2018), Section 5.1.
"""

from typing import Dict, List, Set, Tuple, Optional, Callable, Union, Literal
from itertools import combinations
from dataclasses import dataclass
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
import gurobipy as gp
from gurobipy import GRB

from .instance import ProblemInstance, compute_infeasible_talk_pairs


# Type definitions
NTuple = Tuple[str, ...]  # Tuple of talk_ids that run in parallel
Phase1Result = List[NTuple]  # Simple list of tuples
Phase1ResultByType = Dict[int, List[NTuple]]  # Grouped by tuple size


def generate_ntuples(
    talks: List[str],
    n: int,
    filter_fn: Optional[Callable[[Tuple[str, ...]], bool]] = None
) -> List[Tuple[str, ...]]:
    """
    Generate all n-tuples of talks, optionally filtering infeasible ones.

    Args:
        talks: List of talk_ids
        n: Size of tuples (number of parallel sessions)
        filter_fn: Optional function that returns True if tuple is feasible

    Returns:
        List of feasible n-tuples
    """
    all_tuples = list(combinations(talks, n))

    if filter_fn is not None:
        all_tuples = [t for t in all_tuples if filter_fn(t)]

    return all_tuples


def compute_tuple_cost(ntuple: NTuple, preferences: Dict[str, Set[str]]) -> int:
    """
    Compute missed attendance cost for an n-tuple.

    Cost = sum over participants of max(0, preferred_in_tuple - 1)

    Args:
        ntuple: Tuple of talk_ids scheduled in parallel
        preferences: Dict mapping participant_id to set of preferred talk_ids

    Returns:
        Integer cost representing total missed attendance
    """
    cost = 0
    for p_id, prefs in preferences.items():
        preferred_count = sum(1 for talk_id in ntuple if talk_id in prefs)
        if preferred_count > 1:
            cost += preferred_count - 1
    return cost


def compute_tuple_cost_fast(ntuple: NTuple, talk_to_participants: Dict[str, Set[str]]) -> int:
    """
    Compute missed attendance cost using pre-computed reverse index.

    This is much faster than compute_tuple_cost when processing many tuples,
    as it only considers participants who actually prefer talks in the tuple.

    Args:
        ntuple: Tuple of talk_ids scheduled in parallel
        talk_to_participants: Reverse index mapping talk_id to set of participant_ids

    Returns:
        Integer cost representing total missed attendance
    """
    # Count how many talks each participant wants in this tuple
    participant_counts = Counter()
    for talk_id in ntuple:
        for p_id in talk_to_participants.get(talk_id, set()):
            participant_counts[p_id] += 1

    # Sum up missed attendance (count - 1 for each participant with count > 1)
    return sum(count - 1 for count in participant_counts.values() if count > 1)


def build_reverse_preference_index(preferences: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """
    Build reverse index: talk_id -> set of participants who prefer it.

    Args:
        preferences: Dict mapping participant_id to set of preferred talk_ids

    Returns:
        Dict mapping talk_id to set of participant_ids
    """
    talk_to_participants: Dict[str, Set[str]] = {}
    for p_id, prefs in preferences.items():
        for talk_id in prefs:
            if talk_id not in talk_to_participants:
                talk_to_participants[talk_id] = set()
            talk_to_participants[talk_id].add(p_id)
    return talk_to_participants


def compute_popular_pairs(
    talks: List[str],
    preferences: Dict[str, Set[str]],
    max_parallel: int,
    min_profile_fraction: float = 0.1,
    verbose: bool = True
) -> Set[Tuple[str, str]]:
    """
    Compute popular talk pairs that should NOT be scheduled together.

    A pair is popular if many participants want to see both talks, indicating
    a scheduling conflict. We filter tuples containing popular pairs.

    Strategy:
    1. Count co-occurrences: for each preference profile, increment counter
       for all pairs (talk_i, talk_j) where both are preferred
    2. Identify popular pairs: those appearing in >= min_profile_fraction of profiles
    3. Limit per talk: keep at most (num_talks / max_parallel) pairs per talk

    Args:
        talks: List of all talk IDs
        preferences: Dict mapping participant_id to set of preferred talk_ids
        max_parallel: Maximum number of parallel sessions (largest tuple size)
        min_profile_fraction: Minimum fraction of profiles for a pair to be popular
        verbose: Whether to print statistics

    Returns:
        Set of (talk_i, talk_j) tuples representing popular pairs (unordered)
    """
    from collections import defaultdict

    # Build co-occurrence matrix
    pair_counts = defaultdict(int)
    num_profiles = len(preferences)

    for p_id, prefs in preferences.items():
        prefs_list = sorted(prefs)  # Sort for consistent ordering
        # Add all pairs in this profile
        for i, talk_i in enumerate(prefs_list):
            for talk_j in prefs_list[i+1:]:
                # Store as sorted tuple for consistency
                pair = tuple(sorted([talk_i, talk_j]))
                pair_counts[pair] += 1

    if num_profiles == 0:
        return set()

    # Threshold: pair must appear in at least min_profile_fraction of profiles
    min_count = max(1, int(num_profiles * min_profile_fraction))

    # Find pairs above threshold
    candidate_pairs = {
        pair: count for pair, count in pair_counts.items()
        if count >= min_count
    }

    # Limit per talk: keep at most (num_talks / max_parallel) pairs per talk
    max_pairs_per_talk = max(1, len(talks) // max_parallel)

    # Count pairs per talk
    talk_pair_counts = defaultdict(list)
    for pair, count in candidate_pairs.items():
        talk_i, talk_j = pair
        talk_pair_counts[talk_i].append((pair, count))
        talk_pair_counts[talk_j].append((pair, count))

    # For each talk, keep only top max_pairs_per_talk pairs by count
    popular_pairs = set()
    for talk_id, pairs_with_counts in talk_pair_counts.items():
        # Sort by count descending, take top max_pairs_per_talk
        pairs_with_counts.sort(key=lambda x: x[1], reverse=True)
        top_pairs = pairs_with_counts[:max_pairs_per_talk]
        for pair, count in top_pairs:
            popular_pairs.add(pair)

    if verbose:
        print(f"\nPopular pairs analysis:")
        print(f"  Profiles: {num_profiles}")
        print(
            f"  Min count threshold: {min_count} ({min_profile_fraction*100:.0f}%)")
        print(f"  Candidate pairs (>= threshold): {len(candidate_pairs)}")
        print(f"  Max pairs per talk: {max_pairs_per_talk}")
        print(f"  Final popular pairs: {len(popular_pairs)}")

        if popular_pairs:
            # Show top 10 most popular pairs
            sorted_pairs = sorted(
                [(p, candidate_pairs.get(p, 0)) for p in popular_pairs],
                key=lambda x: x[1],
                reverse=True
            )[:10]
            print(f"  Top popular pairs:")
            for pair, count in sorted_pairs:
                pct = 100 * count / num_profiles
                print(
                    f"    {pair[0]} + {pair[1]}: {count} profiles ({pct:.1f}%)")

    return popular_pairs


def count_preferred_in_tuple(ntuple: NTuple, preferred_talks: Set[str]) -> int:
    """Count how many talks in the tuple are preferred by at least one participant."""
    return sum(1 for t in ntuple if t in preferred_talks)


def heuristic_filter_tuples(
    tuples: List[NTuple],
    talk_to_participants: Dict[str, Set[str]],
    all_talks: Set[str],
    max_preferred: int = 3,
    max_cost: int = 2,
    ensure_coverage: bool = True
) -> List[NTuple]:
    """
    Heuristically filter tuples to reduce problem size while maintaining quality.

    Key insight: A tuple only has cost > 0 if it contains 2+ talks that the 
    SAME participant wants. So we can aggressively filter tuples with many
    "preferred" talks (talks someone wants to see).

    Args:
        tuples: List of candidate tuples
        talk_to_participants: Reverse preference index
        all_talks: Set of all talk IDs (for coverage check)
        max_preferred: Maximum number of preferred talks allowed in a tuple
        max_cost: Maximum cost allowed for a tuple
        ensure_coverage: If True, keep additional tuples to ensure all talks covered

    Returns:
        Filtered list of tuples
    """
    preferred_talks = set(talk_to_participants.keys())

    kept_tuples = []
    covered_talks = set()
    backup_tuples = []  # Tuples that exceed limits but might be needed for coverage

    for ntuple in tuples:
        n_preferred = count_preferred_in_tuple(ntuple, preferred_talks)

        # Tuples with 0-1 preferred talks always have cost 0 → keep them
        if n_preferred <= 1:
            kept_tuples.append(ntuple)
            covered_talks.update(ntuple)
            continue

        # Tuples with 2+ preferred talks → check cost
        cost = compute_tuple_cost_fast(ntuple, talk_to_participants)

        if n_preferred <= max_preferred and cost <= max_cost:
            kept_tuples.append(ntuple)
            covered_talks.update(ntuple)
        elif ensure_coverage:
            # Might need this tuple for coverage - keep as backup
            backup_tuples.append((ntuple, cost))

    # Ensure all talks are covered
    if ensure_coverage:
        uncovered = all_talks - covered_talks
        if uncovered:
            # Sort backup tuples by cost and add until covered
            backup_tuples.sort(key=lambda x: x[1])
            for ntuple, cost in backup_tuples:
                if any(t in uncovered for t in ntuple):
                    kept_tuples.append(ntuple)
                    covered_talks.update(ntuple)
                    uncovered = all_talks - covered_talks
                    if not uncovered:
                        break

    return kept_tuples


# Module-level variables for multiprocessing (workers can't access instance methods)
_talk_to_participants_global: Dict[str, Set[str]] = {}
_preferred_talks_global: Set[str] = set()
_max_preferred_global: int = 3
_max_cost_global: int = 2


def _compute_cost_batch(tuples_batch: List[NTuple]) -> List[Tuple[NTuple, int]]:
    """Compute costs for a batch of tuples (for parallel processing)."""
    return [(nt, compute_tuple_cost_fast(nt, _talk_to_participants_global)) for nt in tuples_batch]


class Phase1Optimizer:
    """
    Phase 1 optimizer for conference scheduling.

    Minimizes missed attendance by selecting optimal n-tuples of talks
    to run in parallel according to block type specifications.

    Supports variable n: different block types can have different numbers
    of parallel sessions.

    Mathematical Formulation:
        min  sum_{e in H} c_e * x_e
        s.t. sum_{e in H: i in e} x_e = 1     for all talks i       (coverage)
             sum_{e in H_tau} x_e = p_tau     for all tuple types   (tuple count)
             x_e in {0, 1}                    for all e in H

    where:
        - H = union of H_tau (all feasible tuples)
        - H_tau = set of feasible n_tau-tuples
        - p_tau = number of tuples of size n_tau needed
        - c_e = missed attendance cost for tuple e
    """

    def __init__(self, env: Optional[gp.Env] = None):
        """
        Initialize the Phase 1 optimizer.

        Args:
            env: Gurobi environment for model creation (optional)
        """
        self.env = env
        self.model = None

        # Problem data
        self.problem_instance = None
        self.talks = []
        self.participants = []
        self.block_types = {}
        self.preferences = {}

        # Derived structures for optimization
        self.tuple_types = []  # List of (n_tau, p_tau) tuples
        self.tuples = []  # List of all feasible n-tuples
        self.tuple_costs = {}  # Dict: n-tuple -> cost
        self.tuple_sizes = {}  # Dict: n-tuple -> size (n_tau)

        # Decision variables
        self.x_vars = {}  # Dict: n-tuple -> binary variable

        # Solution
        self.selected_tuples = []
        self.tuples_by_size = {}

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and dispose model."""
        if self.model is not None:
            self.model.dispose()
        return False

    def set_problem_instance(self, problem_instance: ProblemInstance):
        """
        Load problem instance and derive tuple types.

        Args:
            problem_instance: ProblemInstance containing all problem data
        """
        self.problem_instance = problem_instance
        self.talks = problem_instance.talks
        self.participants = problem_instance.participants
        self.block_types = problem_instance.block_types
        self.preferences = problem_instance.preferences

        # Derive tuple types from block types
        self._derive_tuple_types()

        # Validate feasibility
        self._validate_feasibility()

    def _derive_tuple_types(self):
        """
        Derive tuple types T from block types B.

        Block types: {type_id: {"n": int, "k": int, "count": int}}
        Tuple types: List of (n_tau, p_tau) where:
            - n_tau = tuple size (number of parallel sessions)
            - p_tau = sum of (count * k) over all block types with this n
        """
        # Aggregate by tuple size (n)
        tuple_type_dict = {}

        for type_id, block_spec in self.block_types.items():
            n = block_spec["n"]  # Parallel sessions
            k = block_spec["k"]  # Talks per session
            count = block_spec["count"]  # Number of blocks

            # Tuples needed for this block type: count * k
            tuples_needed = count * k

            # Aggregate by size
            if n not in tuple_type_dict:
                tuple_type_dict[n] = 0
            tuple_type_dict[n] += tuples_needed

        # Convert to sorted list of (n_tau, p_tau)
        self.tuple_types = sorted(tuple_type_dict.items())

    def _validate_feasibility(self):
        """Validate that the problem is feasible (total slots = num talks)."""
        total_slots = sum(n_tau * p_tau for n_tau, p_tau in self.tuple_types)

        if total_slots != len(self.talks):
            raise ValueError(
                f"Infeasible configuration: "
                f"Total slots ({total_slots}) != number of talks ({len(self.talks)})\n"
                f"Block types: {self.block_types}\n"
                f"Tuple types: {self.tuple_types}"
            )

    def _calculate_dynamic_max_cost(
        self,
        talk_to_participants: Dict[str, Set[str]],
        max_tuple_size: int,
        filter_fn: Optional[Callable[[NTuple], bool]] = None,
        sample_size: int = 10000
    ) -> int:
        """
        Dynamically calculate max_cost threshold based on sampled cost distribution.

        This samples random tuples and computes their costs to find a good threshold
        that keeps approximately the bottom 50% of tuples by cost.

        Args:
            talk_to_participants: Reverse preference index
            max_tuple_size: Largest tuple size in the problem
            filter_fn: Structural filter (e.g., same presenter)
            sample_size: Number of tuples to sample for cost distribution

        Returns:
            Dynamic max_cost threshold (median cost from sample)
        """
        import random

        num_participants = len(self.participants)
        num_talks = len(self.talks)

        if num_participants == 0:
            return 2

        # Sample random tuples and compute their costs
        sampled_costs = []
        talks_list = list(self.talks)
        attempts = 0
        max_attempts = sample_size * 10

        print(
            f"  Sampling {sample_size} random tuples to estimate cost distribution...")

        while len(sampled_costs) < sample_size and attempts < max_attempts:
            attempts += 1
            # Sample a random tuple
            sample = tuple(sorted(random.sample(talks_list, max_tuple_size)))

            # Check structural filter
            if filter_fn is not None and not filter_fn(sample):
                continue

            # Compute cost
            cost = compute_tuple_cost_fast(sample, talk_to_participants)
            sampled_costs.append(cost)

        if not sampled_costs:
            print(
                f"  Warning: Could not sample any valid tuples, using default max_cost=10")
            return 10

        # Use percentile-based threshold (keep bottom 20% by cost for large problems)
        sampled_costs.sort()
        percentile_20 = sampled_costs[int(len(sampled_costs) * 0.20)]
        percentile_40 = sampled_costs[int(len(sampled_costs) * 0.40)]
        percentile_50 = sampled_costs[int(len(sampled_costs) * 0.50)]
        percentile_75 = sampled_costs[int(len(sampled_costs) * 0.75)]

        print(f"  Sampled {len(sampled_costs)} tuples: min={min(sampled_costs)}, "
              f"20th={percentile_20}, 40th={percentile_40}, median={percentile_50}, 75th={percentile_75}, max={max(sampled_costs)}")

        # Use 20th percentile for very large problems (>100 talks)
        # Use 40th percentile for medium problems
        num_talks = len(self.talks)
        if num_talks > 100:
            dynamic_max_cost = max(2, percentile_20)
            print(
                f"  Dynamic max_cost: {dynamic_max_cost} (keeps ~20% of tuples - aggressive for large problem)")
        else:
            dynamic_max_cost = max(2, percentile_40)
            print(
                f"  Dynamic max_cost: {dynamic_max_cost} (keeps ~40% of tuples)")

        return dynamic_max_cost

    def _generate_tuples(
        self,
        filter_fn: Optional[Callable[[NTuple], bool]] = None,
        use_parallel: bool = True,
        num_workers: Optional[int] = None,
        batch_size: int = 50000,
        use_heuristic_filter: bool = True,
        max_preferred: int = 10,
        max_cost: Optional[int] = None,  # None means auto-calculate
        popular_pairs: Optional[Set[Tuple[str, str]]] = None,
        min_tuples_per_talk: int = 50
    ):
        """
        Generate all feasible n-tuples and compute their costs.

        Uses optimized cost computation with reverse preference index.
        For large tuple counts, enables parallel processing.
        Uses heuristic filtering to dramatically reduce tuple count.

        Args:
            filter_fn: Optional function to filter out infeasible tuples
                       (e.g., same presenter constraint)
            use_parallel: Whether to use parallel processing for large problems
            num_workers: Number of parallel workers (default: CPU count - 1)
            batch_size: Batch size for parallel processing
            use_heuristic_filter: Whether to use heuristic cost-based filtering
            max_preferred: Kept for API compatibility (not actively used)
            max_cost: Max cost per tuple for heuristic filter (None = auto-calculate)
            popular_pairs: Set of (talk_i, talk_j) pairs that should not be scheduled together
            min_tuples_per_talk: Minimum tuples each talk should appear in
        """
        import time
        from math import comb
        start_time = time.time()

        # Print estimate of tuple counts
        print("Estimating tuple counts...")
        n_talks = len(self.talks)
        total_estimate = 0
        max_tuple_size = 1
        for n_tau, p_tau in self.tuple_types:
            estimate = comb(n_talks, n_tau)
            total_estimate += estimate
            max_tuple_size = max(max_tuple_size, n_tau)
            print(
                f"  Size {n_tau}: C({n_talks},{n_tau}) = {estimate:,} tuples (need to select {p_tau})")
        print(f"  Total: ~{total_estimate:,} tuples to evaluate")

        # Build reverse preference index for fast cost computation
        talk_to_participants = build_reverse_preference_index(self.preferences)
        preferred_talks = set(talk_to_participants.keys())
        all_talks_set = set(self.talks)

        # Auto-calculate max_cost based on sampled cost distribution
        if max_cost is None:
            max_cost = self._calculate_dynamic_max_cost(
                talk_to_participants, max_tuple_size, filter_fn
            )

        print(f"\nPreference analysis:")
        print(
            f"  {len(preferred_talks)} talks are preferred by at least one participant")
        print(
            f"  {len(all_talks_set) - len(preferred_talks)} talks have no preferences (always cost 0)")

        if use_heuristic_filter and total_estimate > 1_000_000:
            print(
                f"\nUsing heuristic filtering (max_preferred={max_preferred}, max_cost={max_cost})")
            print("  Tuples with ≤1 preferred talks: kept (cost = 0)")
            print("  Tuples with 2+ preferred talks: kept only if cost ≤ max_cost")
        print()

        print("Generating feasible n-tuples...")

        self.tuples = []
        self.tuple_costs = {}
        self.tuple_sizes = {}

        for n_tau, p_tau in self.tuple_types:
            gen_start = time.time()

            # Use streaming generation with filtering to avoid memory issues
            if use_heuristic_filter and total_estimate > 1_000_000:
                tuples_of_size = self._generate_tuples_with_heuristic_filter(
                    n_tau, filter_fn, talk_to_participants, preferred_talks,
                    max_preferred, max_cost, popular_pairs, min_tuples_per_talk
                )
            else:
                # Generate all n_tau-sized combinations
                tuples_of_size = list(combinations(self.talks, n_tau))
                # Apply structural filter if provided (e.g., same presenter)
                if filter_fn is not None:
                    tuples_of_size = [
                        t for t in tuples_of_size if filter_fn(t)]
                # Apply popular pairs filter if provided
                if popular_pairs is not None:
                    filtered_count = 0
                    filtered_tuples = []
                    for ntuple in tuples_of_size:
                        has_popular_pair = False
                        for i, talk_i in enumerate(ntuple):
                            for talk_j in ntuple[i+1:]:
                                pair = tuple(sorted([talk_i, talk_j]))
                                if pair in popular_pairs:
                                    has_popular_pair = True
                                    break
                            if has_popular_pair:
                                break
                        if not has_popular_pair:
                            filtered_tuples.append(ntuple)
                        else:
                            filtered_count += 1
                    tuples_of_size = filtered_tuples
                    if filtered_count > 0:
                        print(
                            f"    Filtered {filtered_count:,} tuples due to popular pairs")

            gen_time = time.time() - gen_start
            print(
                f"  Size {n_tau}: {len(tuples_of_size):,} tuples kept (gen: {gen_time:.1f}s)")

            # Compute costs and store
            cost_start = time.time()
            for ntuple in tuples_of_size:
                if ntuple not in self.tuple_costs:  # Avoid recomputing
                    cost = compute_tuple_cost_fast(
                        ntuple, talk_to_participants)
                    self.tuple_costs[ntuple] = cost
                else:
                    cost = self.tuple_costs[ntuple]
                self.tuples.append(ntuple)
                self.tuple_sizes[ntuple] = n_tau
            cost_time = time.time() - cost_start
            if cost_time > 0.1:
                print(f"    Cost computation: {cost_time:.1f}s")

        # Verify coverage
        covered = set()
        for ntuple in self.tuples:
            covered.update(ntuple)
        uncovered = all_talks_set - covered
        if uncovered:
            print(
                f"  WARNING: {len(uncovered)} talks not covered by any tuple!")

        total_time = time.time() - start_time
        print(
            f"\nGenerated {len(self.tuples):,} feasible n-tuples in {total_time:.1f}s")
        if total_estimate > 0:
            reduction = 100 * (1 - len(self.tuples) / total_estimate)
            print(
                f"  Reduction: {reduction:.1f}% (from {total_estimate:,} to {len(self.tuples):,})")

    def _generate_tuples_with_heuristic_filter(
        self,
        n_tau: int,
        filter_fn: Optional[Callable[[NTuple], bool]],
        talk_to_participants: Dict[str, Set[str]],
        preferred_talks: Set[str],
        max_preferred: int,
        max_cost: int,
        popular_pairs: Optional[Set[Tuple[str, str]]] = None,
        min_tuples_per_talk: int = 50
    ) -> List[NTuple]:
        """
        Generate tuples with adaptive heuristic filter to reduce problem size
        while guaranteeing feasibility.

        The filter works in order:
        1. Structural filter (same presenter)
        2. Popular pairs filter (fast - just set lookup)
        3. Cost filter (only computed if passed other filters)

        This ordering minimizes expensive cost computations.
        """
        kept_tuples = []
        talk_tuple_count = {talk: 0 for talk in self.talks}
        backup_tuples = []

        checked = 0
        passed_structural = 0
        passed_popular = 0
        filtered_by_popular_pairs = 0

        for ntuple in combinations(self.talks, n_tau):
            checked += 1

            # 1. Structural filter (e.g., same presenter) - fast
            if filter_fn is not None and not filter_fn(ntuple):
                continue
            passed_structural += 1

            # 2. Popular pairs filter - fast (set lookups)
            if popular_pairs is not None:
                has_popular_pair = False
                for i, talk_i in enumerate(ntuple):
                    for talk_j in ntuple[i+1:]:
                        pair = tuple(sorted([talk_i, talk_j]))
                        if pair in popular_pairs:
                            has_popular_pair = True
                            break
                    if has_popular_pair:
                        break
                if has_popular_pair:
                    filtered_by_popular_pairs += 1
                    continue
            passed_popular += 1

            # 3. Fast check: tuples with 0-1 preferred talks have cost 0
            n_preferred = sum(1 for t in ntuple if t in preferred_talks)

            if n_preferred <= 1:
                kept_tuples.append(ntuple)
                self.tuple_costs[ntuple] = 0
                for talk in ntuple:
                    talk_tuple_count[talk] += 1
                continue

            # 4. Cost filter - more expensive, only done if needed
            cost = compute_tuple_cost_fast(ntuple, talk_to_participants)
            self.tuple_costs[ntuple] = cost

            if cost <= max_cost:
                kept_tuples.append(ntuple)
                for talk in ntuple:
                    talk_tuple_count[talk] += 1
            else:
                backup_tuples.append((ntuple, cost))

            # Progress update
            if checked % 10_000_000 == 0:
                print(f"    Checked {checked:,}, kept {len(kept_tuples):,}...")

        if popular_pairs and filtered_by_popular_pairs > 0:
            print(
                f"    Filtered {filtered_by_popular_pairs:,} tuples due to popular pairs")

        # Ensure minimum coverage per talk
        low_coverage = [t for t, c in talk_tuple_count.items()
                        if c < min_tuples_per_talk]
        if low_coverage:
            print(
                f"    {len(low_coverage)} talks have < {min_tuples_per_talk} tuples, adding backups...")
            backup_tuples.sort(key=lambda x: x[1])

            for ntuple, cost in backup_tuples:
                if any(t in ntuple and talk_tuple_count[t] < min_tuples_per_talk for t in low_coverage):
                    kept_tuples.append(ntuple)
                    for talk in ntuple:
                        talk_tuple_count[talk] += 1
                    low_coverage = [
                        t for t in low_coverage if talk_tuple_count[t] < min_tuples_per_talk]
                    if not low_coverage:
                        break

        return kept_tuples

    def _compute_costs_parallel(
        self,
        tuples_of_size: List[NTuple],
        n_tau: int,
        talk_to_participants: Dict[str, Set[str]],
        num_workers: Optional[int] = None,
        batch_size: int = 50000
    ):
        """
        Compute tuple costs in parallel using multiple processes.

        Args:
            tuples_of_size: List of tuples to process
            n_tau: Tuple size (for storing in tuple_sizes)
            talk_to_participants: Reverse preference index
            num_workers: Number of worker processes
            batch_size: Number of tuples per batch
        """
        import time
        global _talk_to_participants_global
        _talk_to_participants_global = talk_to_participants

        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() - 1)

        # Split into batches
        batches = [
            tuples_of_size[i:i + batch_size]
            for i in range(0, len(tuples_of_size), batch_size)
        ]

        print(
            f"    Computing costs in parallel ({num_workers} workers, {len(batches)} batches)...")
        cost_start = time.time()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_compute_cost_batch, batches))

        # Flatten results and store
        for batch_result in results:
            for ntuple, cost in batch_result:
                self.tuples.append(ntuple)
                self.tuple_costs[ntuple] = cost
                self.tuple_sizes[ntuple] = n_tau

        cost_time = time.time() - cost_start
        print(f"    Parallel cost computation: {cost_time:.1f}s")

    def build_model(
        self,
        filter_fn: Optional[Callable[[NTuple], bool]] = None,
        time_limit: float = 300.0,
        mip_gap: float = 0.0,
        verbose: bool = True,
        use_parallel: bool = True,
        use_heuristic_filter: bool = True,
        max_preferred: int = 10,  # Relaxed - not a hard filter anymore
        # None = auto-calculate based on problem size
        max_cost: Optional[int] = None,
        min_tuples_per_talk: int = 50,  # Minimum tuples each talk should appear in
        use_popular_pairs_filter: bool = True,
        popular_pairs_min_fraction: float = 0.1
    ):
        """
        Build the Gurobi optimization model.

        Args:
            filter_fn: Optional function to filter infeasible tuples
            time_limit: Solver time limit in seconds
            mip_gap: MIP optimality gap tolerance
            verbose: Whether to show solver output
            use_parallel: Whether to use parallel processing for tuple generation
            use_heuristic_filter: Use heuristic to filter unlikely tuples
            max_preferred: Kept for API compatibility (no longer used)
            max_cost: Max cost per tuple (for heuristic). None = auto-calculate.
            min_tuples_per_talk: Minimum tuples each talk should appear in for feasibility
            use_popular_pairs_filter: Whether to filter tuples with popular talk pairs
            popular_pairs_min_fraction: Minimum fraction of profiles for a pair to be popular
        """
        print("\nBuilding Phase 1 optimization model...")

        # Compute popular pairs if enabled
        popular_pairs = None
        if use_popular_pairs_filter:
            max_parallel = max(n for n, p in self.tuple_types)
            popular_pairs = compute_popular_pairs(
                talks=self.talks,
                preferences=self.preferences,
                max_parallel=max_parallel,
                min_profile_fraction=popular_pairs_min_fraction,
                verbose=verbose
            )

        # Generate tuples
        self._generate_tuples(
            filter_fn,
            use_parallel=use_parallel,
            use_heuristic_filter=use_heuristic_filter,
            max_preferred=max_preferred,
            max_cost=max_cost,
            popular_pairs=popular_pairs,
            min_tuples_per_talk=min_tuples_per_talk
        )

        # Create model
        if self.env is not None:
            self.model = gp.Model("Phase1_ConferenceScheduling", env=self.env)
        else:
            self.model = gp.Model("Phase1_ConferenceScheduling")

        # Set parameters
        self.model.Params.TimeLimit = time_limit
        self.model.Params.MIPGap = mip_gap
        self.model.Params.LogFile = "gurobi_phase1.log"
        self.model.Params.MemLimit = 140  # Limit to 140 GB to avoid OOM
        if not verbose:
            self.model.Params.OutputFlag = 0

        # Build model components
        self._create_variables()
        self._add_constraints()
        self._set_objective()

        self.model.update()
        print("Model built successfully")

    def _create_variables(self):
        """Create binary decision variables for each n-tuple."""
        self.x_vars = {}
        for ntuple in self.tuples:
            var_name = f"x_{'_'.join(ntuple)}"
            self.x_vars[ntuple] = self.model.addVar(
                vtype=GRB.BINARY,
                name=var_name
            )
        print(f"Created {len(self.x_vars)} binary variables")

    def _add_constraints(self):
        """Add partitioning constraints to the model."""
        # Constraint 1: Coverage - each talk in exactly one selected tuple
        for talk_id in self.talks:
            tuples_with_talk = [nt for nt in self.tuples if talk_id in nt]

            self.model.addConstr(
                gp.quicksum(self.x_vars[nt] for nt in tuples_with_talk) == 1,
                name=f"coverage_{talk_id}"
            )

        # Constraint 2: Tuple count - select exactly p_tau tuples of each size
        for n_tau, p_tau in self.tuple_types:
            tuples_of_size = [
                nt for nt in self.tuples if self.tuple_sizes[nt] == n_tau]

            self.model.addConstr(
                gp.quicksum(self.x_vars[nt] for nt in tuples_of_size) == p_tau,
                name=f"tuple_count_n{n_tau}"
            )

        print(
            f"Added {len(self.talks)} coverage + {len(self.tuple_types)} tuple count constraints")

    def _set_objective(self):
        """Set objective to minimize total missed attendance."""
        objective = gp.quicksum(
            self.tuple_costs[nt] * self.x_vars[nt]
            for nt in self.tuples
        )
        self.model.setObjective(objective, GRB.MINIMIZE)

    def solve(self) -> int:
        """
        Solve the optimization model.

        Returns:
            Gurobi status code (GRB.OPTIMAL, GRB.INFEASIBLE, etc.)
        """
        print("\n" + "=" * 70)
        print("SOLVING PHASE 1 OPTIMIZATION")
        print("=" * 70)

        self.model.optimize()

        # Extract solution if optimal or time limit with solution
        if self.model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and self.model.SolCount > 0:
            self._extract_solution()

        return self.model.status

    def _extract_solution(self):
        """Extract the selected tuples from the solved model."""
        self.selected_tuples = [
            ntuple for ntuple in self.tuples
            if self.x_vars[ntuple].X > 0.5
        ]

        # Organize by size
        self.tuples_by_size = {}
        for ntuple in self.selected_tuples:
            size = self.tuple_sizes[ntuple]
            if size not in self.tuples_by_size:
                self.tuples_by_size[size] = []
            self.tuples_by_size[size].append(ntuple)

    def get_result(self) -> Optional[Phase1Result]:
        """
        Get Phase 1 result as a flat list of n-tuples.

        Returns:
            List of selected n-tuples, or None if no solution
        """
        if self.model is None or self.model.SolCount == 0:
            return None
        return self.selected_tuples

    def get_result_by_size(self) -> Optional[Phase1ResultByType]:
        """
        Get Phase 1 result organized by tuple size.

        Returns:
            Dict mapping tuple size to list of n-tuples, or None if no solution
        """
        if self.model is None or self.model.SolCount == 0:
            return None
        return self.tuples_by_size

    def get_objective_value(self) -> Optional[float]:
        """
        Get the optimal objective value (total missed attendance).

        Returns:
            Objective value, or None if no solution
        """
        if self.model is not None and self.model.SolCount > 0:
            return self.model.ObjVal
        return None

    def display_results(self, detailed: bool = True):
        """
        Display the optimization results.

        Args:
            detailed: If True, show conflict details per session
        """
        print("\n" + "=" * 70)
        print("PHASE 1 RESULTS")
        print("=" * 70)

        if self.model.status == GRB.OPTIMAL:
            print(f"Status: OPTIMAL")
        elif self.model.status == GRB.TIME_LIMIT and self.model.SolCount > 0:
            print(f"Status: TIME_LIMIT (solution found)")
        else:
            print(f"Status: {self.model.status} (no solution)")
            return

        print(f"Total Missed Attendance: {self.model.ObjVal:.0f}")

        total_prefs = sum(len(prefs) for prefs in self.preferences.values())
        if total_prefs > 0:
            satisfaction = 100 * (1 - self.model.ObjVal / total_prefs)
            print(f"Preference Satisfaction: {satisfaction:.1f}%")

        print(f"\nSelected {len(self.selected_tuples)} parallel sessions:")
        print("-" * 70)

        for n_tau in sorted(self.tuples_by_size.keys()):
            tuples = self.tuples_by_size[n_tau]
            print(f"\nSize {n_tau} ({len(tuples)} sessions):")

            for idx, ntuple in enumerate(tuples, 1):
                cost = self.tuple_costs[ntuple]
                print(f"  Session {idx}: {', '.join(ntuple)} - Missed: {cost}")

                if detailed and cost > 0:
                    # Show affected participants
                    affected = []
                    for p_id in self.participants:
                        prefs = self.preferences.get(p_id, set())
                        count_in_tuple = sum(1 for t in ntuple if t in prefs)
                        if count_in_tuple > 1:
                            affected.append(f"{p_id}({count_in_tuple})")
                    if affected:
                        print(f"    Conflicts: {', '.join(affected)}")

    def get_summary(self) -> Dict:
        """
        Export solution summary as a dictionary.

        Returns:
            Dictionary with solution metadata
        """
        if self.model is None or self.model.SolCount == 0:
            return {
                'status': self.model.status if self.model else None,
                'feasible': False
            }

        total_prefs = sum(len(prefs) for prefs in self.preferences.values())

        return {
            'status': self.model.status,
            'feasible': True,
            'objective_value': self.model.ObjVal,
            'missed_attendance': int(self.model.ObjVal),
            'total_preferences': total_prefs,
            'satisfaction_pct': 100 * (1 - self.model.ObjVal / total_prefs) if total_prefs > 0 else 100,
            'num_sessions': len(self.selected_tuples),
            'sessions_by_size': {size: len(tuples) for size, tuples in self.tuples_by_size.items()},
            'tuple_types': self.tuple_types
        }


def solve_phase1(
    instance: ProblemInstance,
    time_limit: float = 300.0,
    verbose: bool = True,
    method: Literal["explicit", "column_generation", "greedy"] = "explicit",
    cg_pricing_strategy: str = "local_search",
    cg_max_iterations: int = 100,
    max_cost: Optional[int] = None
) -> Phase1Result:
    """
    Solve Phase 1 for variable n (multiple block types).

    This is the main entry point for Phase 1 optimization.

    Args:
        instance: Problem instance with block types, talks, preferences
        time_limit: Solver time limit in seconds
        verbose: Whether to print solver output
        method: Solution method:
            - "explicit": Heuristic-filtered full enumeration (requires Gurobi)
            - "column_generation": Column generation approach (requires Gurobi)
            - "greedy": Greedy heuristic (no Gurobi required)
        cg_pricing_strategy: Column generation pricing strategy
            ('auto', 'enumeration', 'greedy', 'local_search', 'beam_search')
        cg_max_iterations: Maximum column generation iterations
        max_cost: For explicit method: max cost threshold for tuple filtering.
                  None = auto-calculate. Lower = more aggressive = smaller model.

    Returns:
        Phase1Result with selected n-tuples

    Raises:
        RuntimeError: If solver fails or no solution found
    """
    if method == "column_generation":
        return solve_phase1_column_generation(
            instance=instance,
            time_limit=time_limit,
            verbose=verbose,
            pricing_strategy=cg_pricing_strategy,
            max_iterations=cg_max_iterations
        )
    elif method == "greedy":
        return solve_phase1_greedy(
            instance=instance,
            time_limit=time_limit,
            verbose=verbose
        )

    # Default: explicit enumeration with heuristic filtering
    return solve_phase1_explicit(
        instance=instance,
        time_limit=time_limit,
        verbose=verbose,
        max_cost=max_cost
    )


def solve_phase1_explicit(
    instance: ProblemInstance,
    time_limit: float = 300.0,
    verbose: bool = True,
    max_cost: Optional[int] = None
) -> Phase1Result:
    """
    Solve Phase 1 using explicit enumeration with heuristic filtering.

    This is the original approach with optimizations for tuple filtering.

    Args:
        instance: Problem instance with block types, talks, preferences
        time_limit: Solver time limit in seconds
        verbose: Whether to print solver output
        max_cost: Maximum cost threshold for tuple filtering. None = auto-calculate.
                  Lower values = more aggressive filtering = smaller model.
                  Use values like 2-4 for very large problems (100+ talks).

    Returns:
        List of selected n-tuples (varying sizes)

    Raises:
        RuntimeError: If solver fails or no solution found
    """
    # Compute pairs with unavailability conflicts (can never be scheduled together)
    infeasible_pairs = compute_infeasible_talk_pairs(instance, verbose=verbose)

    # Filter: no two talks with same presenter, and no infeasible pairs
    def is_feasible(ntuple: NTuple) -> bool:
        # Check same presenter constraint
        if instance.talks_have_same_presenter(ntuple):
            return False

        # Check for infeasible pairs (presenter unavailability conflicts)
        if infeasible_pairs:
            for i, t1 in enumerate(ntuple):
                for t2 in ntuple[i+1:]:
                    pair = tuple(sorted([t1, t2]))
                    if pair in infeasible_pairs:
                        return False

        return True

    with Phase1Optimizer() as optimizer:
        optimizer.set_problem_instance(instance)
        optimizer.build_model(
            filter_fn=is_feasible,
            time_limit=time_limit,
            verbose=verbose,
            max_cost=max_cost
        )
        status = optimizer.solve()

        if verbose:
            optimizer.display_results(detailed=True)

        result = optimizer.get_result()
        if result is None:
            raise RuntimeError(f"Phase 1 solver failed with status {status}")

        return result


def solve_phase1_column_generation(
    instance: ProblemInstance,
    time_limit: float = 300.0,
    verbose: bool = True,
    pricing_strategy: str = "local_search",
    max_iterations: int = 100
) -> Phase1Result:
    """
    Solve Phase 1 using column generation.

    Efficiently solves large instances by generating only promising n-tuples
    instead of enumerating all possibilities.

    Args:
        instance: Problem instance with block types, talks, preferences
        time_limit: Solver time limit in seconds
        verbose: Whether to print solver output
        pricing_strategy: Strategy for pricing problem
            ('auto', 'enumeration', 'greedy', 'local_search', 'beam_search')
        max_iterations: Maximum number of CG iterations

    Returns:
        List of selected n-tuples (varying sizes)

    Raises:
        RuntimeError: If solver fails or no solution found
    """
    from .columngeneration_phase1.phase1_column_generation_enhanced import (
        Phase1ColumnGenerationEnhanced
    )

    # Derive tuple types from block types (same as in Phase1Optimizer)
    tuple_type_dict = {}
    for type_id, block_spec in instance.block_types.items():
        n = block_spec["n"]
        k = block_spec["k"]
        count = block_spec["count"]
        tuples_needed = count * k
        if n not in tuple_type_dict:
            tuple_type_dict[n] = 0
        tuple_type_dict[n] += tuples_needed

    tuple_types = sorted(tuple_type_dict.items())

    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 1 - COLUMN GENERATION METHOD")
        print("=" * 70)

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()

        solver = Phase1ColumnGenerationEnhanced(
            env=env,
            talks=instance.talks,
            participants=instance.participants,
            preferences=instance.preferences,
            tuple_types=tuple_types,
            pricing_strategy=pricing_strategy,
            verbose=verbose
        )

        result_dict = solver.solve(
            max_iterations=max_iterations,
            time_limit=time_limit
        )

        if result_dict['status'] not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            raise RuntimeError(
                f"Column generation solver failed with status {result_dict['status']}"
            )
        if result_dict.get('selected_tuples') is None:
            raise RuntimeError("Column generation: time limit reached with no feasible solution")

        result = solver.get_result()

        if verbose:
            print(f"\nColumn Generation Summary:")
            print(
                f"  Objective (missed attendance): {result_dict['objective']:.0f}")
            print(
                f"  Columns generated: {result_dict['stats']['final_columns']}")
            print(f"  Iterations: {result_dict['stats']['total_iterations']}")

        return result


def solve_phase1_greedy(
    instance: ProblemInstance,
    time_limit: float = 300.0,
    verbose: bool = True
) -> Phase1Result:
    """
    Solve Phase 1 using greedy heuristic (no Gurobi required).

    Strategy:
    1. Compute pairwise conflict costs between talks
    2. For each tuple: greedily select talks that minimize conflicts
    3. Start from less popular talks to ensure they get scheduled
    4. Avoid infeasible pairs (presenter unavailability conflicts)

    Args:
        instance: Problem instance
        time_limit: Time limit in seconds
        verbose: Print progress

    Returns:
        Phase1Result with selected tuples
    """
    import time as time_module
    from collections import defaultdict

    start_time = time_module.time()

    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 1 - GREEDY HEURISTIC (NO GUROBI)")
        print("=" * 70)

    # Compute pairs with unavailability conflicts (can never be scheduled together)
    infeasible_pairs = compute_infeasible_talk_pairs(instance, verbose=verbose)

    # Build presenter map for same-presenter check
    presenter_map = instance.talk_presenter

    # Build tuple types from block_types
    tuple_types = []
    for block_type, spec in instance.block_types.items():
        n, k, count = spec['n'], spec['k'], spec['count']
        total_tuples = k * count
        tuple_types.append((n, total_tuples))

    talks = list(instance.talks)
    preferences = instance.preferences

    # Calculate total slots needed
    total_slots = sum(n * count for n, count in tuple_types)

    if verbose:
        print(f"  Talks: {len(talks)}, Slots needed: {total_slots}")
        print(f"  Tuple types: {tuple_types}")

    # Build reverse index: talk_id -> set of participants who want it
    talk_to_participants: Dict[str, Set[str]] = defaultdict(set)
    for p_id, prefs in preferences.items():
        for talk_id in prefs:
            talk_to_participants[talk_id].add(p_id)

    # Compute pairwise conflict costs (use infinity for infeasible pairs)
    pair_costs: Dict[Tuple[str, str], float] = {}
    for i, t1 in enumerate(talks):
        p1 = talk_to_participants.get(t1, set())
        for t2 in talks[i+1:]:
            pair = tuple(sorted([t1, t2]))

            # Check for infeasible pairs
            if pair in infeasible_pairs:
                pair_costs[pair] = float('inf')
                continue

            # Check for same presenter
            if presenter_map.get(t1) == presenter_map.get(t2):
                pair_costs[pair] = float('inf')
                continue

            p2 = talk_to_participants.get(t2, set())
            cost = len(p1 & p2)  # participants who want both
            pair_costs[pair] = cost

    # Add placeholder talks if needed
    all_talks = list(talks)
    if len(all_talks) < total_slots:
        n_placeholders = total_slots - len(all_talks)
        if verbose:
            print(f"  Adding {n_placeholders} placeholder slots")
        for i in range(n_placeholders):
            all_talks.append(f"PLACEHOLDER_{i+1}")

    # Build tuples greedily
    tuples_by_n: Dict[int, List[Tuple[str, ...]]] = {}
    remaining_talks = set(all_talks)
    total_cost = 0

    # Process larger tuples first
    for n, count in sorted(tuple_types, key=lambda x: -x[0]):
        tuples_by_n[n] = []

        for _ in range(count):
            if len(remaining_talks) < n:
                break

            if time_module.time() - start_time > time_limit:
                if verbose:
                    print(f"  ⚠ Time limit reached")
                break

            # Start with least popular talk (to ensure it gets scheduled)
            candidates = sorted(
                remaining_talks,
                key=lambda t: len(talk_to_participants.get(t, set()))
            )

            best_tuple = None
            best_cost = float('inf')

            # Try building from different seeds
            for seed_idx in range(min(5, len(candidates))):
                current_tuple = [candidates[seed_idx]]
                pool = [t for t in remaining_talks if t != current_tuple[0]]

                while len(current_tuple) < n and pool:
                    # Find talk with minimum conflict with current tuple
                    # (infeasible pairs have cost=inf and will be skipped)
                    best_addition = None
                    best_addition_cost = float('inf')

                    for t in pool:
                        cost = sum(
                            pair_costs.get(tuple(sorted([t, ct])), 0)
                            for ct in current_tuple
                        )
                        if cost < best_addition_cost:
                            best_addition_cost = cost
                            best_addition = t

                    # Only add if not infeasible (cost < inf)
                    if best_addition and best_addition_cost < float('inf'):
                        current_tuple.append(best_addition)
                        pool.remove(best_addition)
                    else:
                        # All remaining candidates are infeasible with current tuple
                        break

                if len(current_tuple) == n:
                    cost = compute_tuple_cost(
                        tuple(current_tuple), preferences)
                    if cost < best_cost:
                        best_cost = cost
                        best_tuple = tuple(current_tuple)

            if best_tuple:
                tuples_by_n[n].append(best_tuple)
                total_cost += best_cost
                for t in best_tuple:
                    remaining_talks.discard(t)

    # Handle remaining talks
    if remaining_talks:
        if verbose:
            print(f"  Note: {len(remaining_talks)} remaining talks to assign")
        smallest_n = min(n for n, _ in tuple_types)

        while remaining_talks:
            talks_to_add = list(remaining_talks)[:smallest_n]
            if len(talks_to_add) < smallest_n:
                for i in range(smallest_n - len(talks_to_add)):
                    talks_to_add.append(f"DUMMY_PAD_{i}")
            tuples_by_n.setdefault(smallest_n, []).append(tuple(talks_to_add))
            for t in talks_to_add:
                remaining_talks.discard(t)

    elapsed = time_module.time() - start_time

    # Flatten to list
    selected_tuples: List[Tuple[str, ...]] = []
    for tuples in tuples_by_n.values():
        selected_tuples.extend(tuples)

    if verbose:
        print(f"\n  ✓ Created {len(selected_tuples)} tuples in {elapsed:.2f}s")
        print(f"  ✓ Total missed attendance cost: {total_cost}")

    return selected_tuples
