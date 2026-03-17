"""
Constrained Matching Pipeline: Bottom-Up Scheduling with Group/Keyword Constraints

A variant of the matching pipeline that enforces:
1. Talks can only be paired if they're in the SAME special group (required)
2. Talks can only be paired if they share at least one keyword (optional)

This ensures thematically coherent sessions by respecting pre-defined groups
and keyword-based topic similarity.
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import time
import pandas as pd

import gurobipy as gp
from gurobipy import GRB

from .instance import ProblemInstance, compute_infeasible_talk_pairs
from .matching_pipeline import (
    MatchingPipelineResult,
    build_co_preference_matrix,
    get_weight,
    solve_phase_c,
    solve_phase_d,
    _run_phase3,
    is_block_feasible
)
from .phase2 import Block


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TalkMetadata:
    """Metadata for constraining talk matching."""
    talk_id: str
    special_group: str  # e.g., "SpecialSession", "general"
    keywords: Set[str]  # e.g., {"Optimization", "Applications"}


@dataclass
class MatchingConstraints:
    """Constraints for the matching pipeline."""
    # Required: only match talks in same special group
    require_same_group: bool = True
    # Optional: only match talks with at least one keyword in common
    require_common_keyword: bool = False
    # Soft constraint mode: allow up to max_keyword_violations pairs without common keywords
    # If None, keyword constraint is hard. If 0+, allows violations up to this limit.
    max_keyword_violations: Optional[int] = None


def load_talk_metadata(
    talks_file: str = "",
    pre_pairings_file: str = "",
    algo_file: str = ""
) -> Dict[str, TalkMetadata]:
    """
    Load talk metadata (special groups and keywords) from data files.

    Args:
        talks_file: CSV with title, author, master_keywords columns
        pre_pairings_file: Excel with pre-defined special group pairings
        algo_file: CSV with talk_id and keywords (preferred source)

    Returns:
        Tuple of (metadata dict, title_to_keywords, special_groups, talkid_to_keywords)
    """
    from pathlib import Path

    metadata = {}

    # Primary source: talks_for_algorithm.csv (has talk_id -> keywords directly)
    talkid_to_keywords = {}
    if Path(algo_file).exists():
        algo_df = pd.read_csv(algo_file)
        for _, row in algo_df.iterrows():
            talk_id = f"T{row['talk_id']:03d}"
            kw_str = str(row.get('keywords', ''))
            if pd.notna(kw_str) and kw_str.lower() != 'nan':
                keywords = set(kw.strip()
                               for kw in kw_str.split(';') if kw.strip())
            else:
                keywords = set()
            talkid_to_keywords[talk_id] = keywords

    # Fallback: title-based keywords from talks_with_abstracts file
    title_to_keywords = {}
    if Path(talks_file).exists():
        keywords_df = pd.read_csv(talks_file)
        for _, row in keywords_df.iterrows():
            title = str(row['title']).strip()
            kw_str = str(row.get('master_keywords', ''))
            keywords = set(kw.strip()
                           for kw in kw_str.split(';') if kw.strip())
            title_to_keywords[title.lower()] = keywords

    # Load pre-pairings for special groups
    special_groups = {}  # talk_number -> group_name

    if Path(pre_pairings_file).exists():
        pairings_df = pd.read_excel(pre_pairings_file, engine='openpyxl')

        for _, row in pairings_df.iterrows():
            remark = str(row.get('Remark', ''))
            if not remark.strip():
                continue
            group_name = remark.split()[0]  # First word of Remark is the group name

            # Assign group to all talks in this row
            for col in ['Talk 1', 'Talk 2', 'Talk 3', 'Talk 4']:
                if col in row and pd.notna(row[col]):
                    talk_num = int(row[col])
                    special_groups[talk_num] = group_name

    return metadata, title_to_keywords, special_groups, talkid_to_keywords


def _find_keywords_for_title(title: str, title_to_keywords: Dict[str, Set[str]]) -> Set[str]:
    """
    Find keywords for a title, handling prefixed titles (e.g., 'Author-Title').

    Some keyword entries have author prefixes like 'Deleye-Locating a facility'
    while abstracts have just 'Locating a facility'. This function handles both.
    """
    title_lower = title.lower()

    # Try exact match first
    if title_lower in title_to_keywords:
        return title_to_keywords[title_lower]

    # Try suffix match (for prefixed titles like 'Author-Title')
    for kw_title, keywords in title_to_keywords.items():
        # Check if kw_title ends with our title (after a separator)
        if kw_title.endswith(title_lower):
            # Verify there's a separator (like '-' or ' - ')
            prefix = kw_title[:-len(title_lower)]
            if prefix.rstrip().endswith('-') or prefix.rstrip().endswith("'"):
                return keywords

    # Try substring match (more permissive - title contained in kw_title)
    for kw_title, keywords in title_to_keywords.items():
        if title_lower in kw_title:
            return keywords

    return set()


def build_metadata_from_instance(
    instance: ProblemInstance,
    title_to_keywords: Dict[str, Set[str]],
    special_groups: Dict[int, str],
    talk_titles: Dict[str, str] = None,
    talkid_to_keywords: Dict[str, Set[str]] = None
) -> Dict[str, TalkMetadata]:
    """
    Build TalkMetadata for each talk in the instance.

    Args:
        instance: Problem instance with talks
        title_to_keywords: Mapping from title to keywords (fallback)
        special_groups: Mapping from talk number to special group
        talk_titles: Optional mapping from talk_id to title
        talkid_to_keywords: Direct talk_id -> keywords mapping (preferred)

    Returns:
        Dictionary mapping talk_id to TalkMetadata
    """
    metadata = {}

    for talk_id in instance.talks:
        # Extract talk number from ID (e.g., "T001" -> 1)
        try:
            talk_num = int(talk_id[1:])
        except (ValueError, IndexError):
            talk_num = -1

        # Get special group (default to 'general')
        group = special_groups.get(talk_num, 'general')

        # Get keywords - prefer direct talk_id mapping, fallback to title-based
        keywords = set()
        if talkid_to_keywords and talk_id in talkid_to_keywords:
            keywords = talkid_to_keywords[talk_id]
        elif talk_titles and talk_id in talk_titles:
            title = talk_titles[talk_id]
            keywords = _find_keywords_for_title(title, title_to_keywords)

        metadata[talk_id] = TalkMetadata(
            talk_id=talk_id,
            special_group=group,
            keywords=keywords
        )

    return metadata


def can_match(
    talk_i: str,
    talk_j: str,
    metadata: Dict[str, TalkMetadata],
    constraints: MatchingConstraints
) -> bool:
    """
    Check if two talks can be matched based on constraints.

    Args:
        talk_i, talk_j: Talk IDs
        metadata: Talk metadata
        constraints: Matching constraints

    Returns:
        True if talks can be matched

    Note:
        Keyword constraint only applies to "general" group - special groups
        are already thematically coherent and don't need it.
    """
    meta_i = metadata.get(talk_i)
    meta_j = metadata.get(talk_j)

    if not meta_i or not meta_j:
        return True  # Allow if metadata missing

    # Check same group (required)
    if constraints.require_same_group:
        if meta_i.special_group != meta_j.special_group:
            return False

    # Check common keyword (optional) - ONLY for "general" group
    # Special groups are already thematically grouped
    if constraints.require_common_keyword:
        # Only apply keyword constraint if BOTH talks are in "general" group
        if meta_i.special_group == "general" and meta_j.special_group == "general":
            if not (meta_i.keywords & meta_j.keywords):
                return False

    return True


# =============================================================================
# PHASE A: CONSTRAINED PAIR MATCHING
# =============================================================================

def solve_phase_a_constrained(
    instance: ProblemInstance,
    n_3: int,
    n_4: int,
    metadata: Dict[str, TalkMetadata],
    constraints: MatchingConstraints,
    time_limit: float = 60.0,
    verbose: bool = True
) -> Tuple[List[Tuple[str, str]], List[str], int, float]:
    """
    Phase A with constraints: Maximum weight matching respecting group/keyword constraints.

    Only creates matching variables for pairs that satisfy the constraints.

    Args:
        instance: Problem instance
        n_3, n_4: Number of 3-blocks and 4-blocks needed
        metadata: Talk metadata with groups and keywords
        constraints: Matching constraints
        time_limit: Solver time limit
        verbose: Print progress

    Returns:
        Tuple of (pairs, singles, total_weight, solve_time)
    """
    start_time = time.time()

    talks = instance.talks
    preferences = instance.preferences
    M = n_3 + 2 * n_4

    if verbose:
        print(f"\n🎯 Goal: Create {M} pairs from {len(talks)} talks")
        print(f"   (Each pair will form part of a parallel session)")

    # Build co-preference weights
    weights = build_co_preference_matrix(talks, preferences)

    # Compute infeasible pairs due to presenter unavailability
    infeasible_pairs = compute_infeasible_talk_pairs(instance, verbose=verbose)

    # Count feasible pairs (respecting both keyword constraints and presenter availability)
    n_feasible = 0
    for i, talk_i in enumerate(talks):
        for talk_j in talks[i+1:]:
            if can_match(talk_i, talk_j, metadata, constraints):
                pair = tuple(sorted([talk_i, talk_j]))
                if pair not in infeasible_pairs:
                    n_feasible += 1

    if verbose:
        total_pairs = len(talks) * (len(talks) - 1) // 2
        print(f"\n📊 Pair feasibility after applying constraints:")
        print(f"   • Total possible pairs:  {total_pairs}")
        print(
            f"   • Feasible pairs:        {n_feasible} ({100*n_feasible/total_pairs:.1f}%)")
        print(f"   • Filtered out:          {total_pairs - n_feasible} pairs")
        if infeasible_pairs:
            print(
                f"   • Presenter conflicts:   {len(infeasible_pairs)} pairs excluded")

    # Create model
    model = gp.Model("PhaseA_Constrained")
    model.setParam('OutputFlag', 1 if verbose else 0)
    model.setParam('TimeLimit', time_limit)

    # Variables: y[i,j] = 1 if talks i,j are matched
    # ONLY create variables for feasible pairs (keyword/group constraints + presenter availability)
    y = {}
    for i, talk_i in enumerate(talks):
        for talk_j in talks[i+1:]:
            if can_match(talk_i, talk_j, metadata, constraints):
                pair = tuple(sorted([talk_i, talk_j]))
                if pair not in infeasible_pairs:
                    y[talk_i, talk_j] = model.addVar(
                        vtype=GRB.BINARY, name=f"y_{talk_i}_{talk_j}")

    if not y:
        raise ValueError("No feasible pairs - constraints too restrictive")

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
        relevant_vars = []
        for talk_i, talk_j in y.keys():
            if talk in (talk_i, talk_j):
                relevant_vars.append(y[talk_i, talk_j])

        if relevant_vars:
            model.addConstr(
                gp.quicksum(relevant_vars) <= 1,
                name=f"matching_{talk}"
            )

    # Constraint 2: Exactly M pairs
    model.addConstr(
        gp.quicksum(y[i, j] for i, j in y.keys()) == M,
        name="cardinality"
    )

    # Constraint 3: Special group talks MUST be matched (not left as singles)
    # This ensures special-group talks form complete blocks in Phase B
    if constraints.require_same_group:
        special_talks = [t for t in talks if metadata.get(
            t) and metadata[t].special_group != 'general']
        if special_talks and verbose:
            print(
                f"  Special group talks (must be paired): {len(special_talks)}")

        for talk in special_talks:
            relevant_vars = []
            for talk_i, talk_j in y.keys():
                if talk in (talk_i, talk_j):
                    relevant_vars.append(y[talk_i, talk_j])

            if relevant_vars:
                # This talk must be in exactly one pair
                model.addConstr(
                    gp.quicksum(relevant_vars) == 1,
                    name=f"special_must_match_{talk}"
                )
            else:
                # No feasible pairs for this special group talk - error
                raise ValueError(
                    f"Special group talk {talk} has no feasible pairs. "
                    f"Check that special group has enough members (need even count)."
                )

    # Solve
    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        raise ValueError(
            f"Phase A infeasible - need {M} pairs but constraints too restrictive. "
            f"Try relaxing require_common_keyword or check special group assignments."
        )

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


def needs_keyword_violation(
    talk_i: str,
    talk_j: str,
    metadata: Dict[str, TalkMetadata]
) -> bool:
    """
    Check if pairing these talks would violate the keyword constraint.
    Only applies to talks in "general" group (special groups are exempt).

    Returns:
        True if both talks are in "general" group and share no keywords
    """
    meta_i = metadata.get(talk_i)
    meta_j = metadata.get(talk_j)

    if not meta_i or not meta_j:
        return False

    # Only general group needs keyword matching
    if meta_i.special_group != "general" or meta_j.special_group != "general":
        return False

    # Violation if no common keywords
    return not (meta_i.keywords & meta_j.keywords)


def solve_phase_a_soft_constraints(
    instance: ProblemInstance,
    n_3: int,
    n_4: int,
    metadata: Dict[str, TalkMetadata],
    constraints: MatchingConstraints,
    time_limit: float = 60.0,
    verbose: bool = True
) -> Tuple[List[Tuple[str, str]], List[str], int, float, int]:
    """
    Phase A with SOFT keyword constraints: Minimizes keyword violations while maximizing quality.

    Model:
        max  sum_{i<j} w_ij * x_ij - penalty * sum_{i<j} v_ij
        s.t. sum_j x_ij <= 1                    (each talk matched once)
             sum_{i<j} x_ij = M                 (exactly M pairs)
             x_ij <= v_ij  for violating pairs  (track violations)
             sum v_ij <= max_violations         (violation budget)

    Uses iterative approach: starts with max_violations=0, increments until feasible.

    Args:
        instance: Problem instance
        n_3, n_4: Number of 3-blocks and 4-blocks needed
        metadata: Talk metadata with groups and keywords
        constraints: Matching constraints (uses max_keyword_violations as starting budget)
        time_limit: Solver time limit
        verbose: Print progress

    Returns:
        Tuple of (pairs, singles, total_weight, solve_time, n_violations)
    """
    start_time = time.time()

    talks = instance.talks
    preferences = instance.preferences
    M = n_3 + 2 * n_4

    # Build co-preference weights
    weights = build_co_preference_matrix(talks, preferences)

    # Compute infeasible pairs due to presenter unavailability
    infeasible_pairs = compute_infeasible_talk_pairs(instance, verbose=verbose)

    # Identify pairs that would violate keyword constraint (only for general group)
    violating_pairs = set()
    same_group_pairs = set()  # Pairs that satisfy same-group constraint

    for i, talk_i in enumerate(talks):
        for talk_j in talks[i+1:]:
            # Skip pairs with presenter unavailability conflicts
            pair = tuple(sorted([talk_i, talk_j]))
            if pair in infeasible_pairs:
                continue

            meta_i = metadata.get(talk_i)
            meta_j = metadata.get(talk_j)

            # Check same group constraint (still hard)
            if constraints.require_same_group:
                if meta_i and meta_j and meta_i.special_group != meta_j.special_group:
                    continue  # Skip cross-group pairs entirely

            same_group_pairs.add((talk_i, talk_j))

            # Check if this pair would violate keyword constraint
            if needs_keyword_violation(talk_i, talk_j, metadata):
                violating_pairs.add((talk_i, talk_j))

    if verbose:
        total_pairs = len(talks) * (len(talks) - 1) // 2
        print(f"\n🎯 Goal: Create {M} pairs from {len(talks)} talks")
        print(f"\n📊 Pair analysis:")
        print(f"   • Total possible pairs:     {total_pairs}")
        print(
            f"   • Same-group pairs (HARD):  {len(same_group_pairs)} ({100*len(same_group_pairs)/total_pairs:.1f}%)")
        print(
            f"   • Cross-group pairs:        {total_pairs - len(same_group_pairs)} (blocked)")
        print(f"   • Would violate keyword:    {len(violating_pairs)} pairs")
        print(
            f"   • Keyword-coherent pairs:   {len(same_group_pairs) - len(violating_pairs)}")

    if len(same_group_pairs) < M:
        raise ValueError(
            f"Not enough same-group pairs: need {M}, have {len(same_group_pairs)}. "
            f"Check special group assignments."
        )

    # Start with requested budget (or 0 if None)
    initial_budget = constraints.max_keyword_violations or 0
    max_budget = len(violating_pairs)  # Upper bound

    if verbose:
        print(f"\n🔄 Iterative Relaxation (Phase A):")
        print(f"   Starting with violation budget = {initial_budget}")

    for budget in range(initial_budget, max_budget + 1):
        if verbose and budget > initial_budget:
            print(
                f"   ⚠️  Budget {budget-1} infeasible → trying budget = {budget}...")

        # Give full time limit to each attempt (first one is most likely to succeed)
        # If it fails due to infeasibility, the next attempt will be quick
        result = _try_phase_a_with_budget(
            talks, same_group_pairs, violating_pairs, weights, M, budget,
            metadata, constraints,
            time_limit, verbose and budget == initial_budget
        )

        if result is not None:
            pairs, total_weight, n_violations = result

            # Get singles
            matched_talks = set()
            for t1, t2 in pairs:
                matched_talks.add(t1)
                matched_talks.add(t2)
            singles = [t for t in talks if t not in matched_talks]

            solve_time = time.time() - start_time

            if verbose:
                if n_violations == 0:
                    print(f"\n   ✅ Solution found with 0 keyword violations (perfect)")
                else:
                    print(
                        f"\n   ✅ Solution found with {n_violations} keyword violations")
                    print(f"      (These pairs have no shared keyword)")
                print(f"\n📊 Phase A Results:")
                print(f"   • Pairs created:     {len(pairs)}")
                print(f"   • Singles remaining: {len(singles)}")
                print(f"   • Co-preference:     {total_weight}")
                print(f"   • Time:              {solve_time:.2f}s")

            return pairs, singles, total_weight, solve_time, n_violations

    raise ValueError(
        f"Phase A infeasible even with all {max_budget} violations allowed")


def _try_phase_a_with_budget(
    talks: List[str],
    same_group_pairs: Set[Tuple[str, str]],
    violating_pairs: Set[Tuple[str, str]],
    weights: Dict[Tuple[str, str], int],
    M: int,
    max_violations: int,
    metadata: Dict[str, TalkMetadata],
    constraints: MatchingConstraints,
    time_limit: float,
    verbose: bool
) -> Optional[Tuple[List[Tuple[str, str]], int, int]]:
    """
    Try to solve Phase A with a specific violation budget.

    Returns:
        (pairs, total_weight, n_violations) if feasible, None if infeasible
    """
    model = gp.Model("PhaseA_SoftConstraints")
    model.setParam('OutputFlag', 1 if verbose else 0)
    model.setParam('TimeLimit', time_limit)

    # Decision variables
    x = {}  # x[i,j] = 1 if talks i,j are matched
    # v[i,j] = 1 if pair (i,j) is selected AND violates keyword constraint
    v = {}

    for (talk_i, talk_j) in same_group_pairs:
        x[talk_i, talk_j] = model.addVar(
            vtype=GRB.BINARY, name=f"x_{talk_i}_{talk_j}")

        if (talk_i, talk_j) in violating_pairs:
            v[talk_i, talk_j] = model.addVar(
                vtype=GRB.BINARY, name=f"v_{talk_i}_{talk_j}")

    # Objective: maximize weight (violations are constrained, not penalized)
    model.setObjective(
        gp.quicksum(
            get_weight(talk_i, talk_j, weights) * x[talk_i, talk_j]
            for talk_i, talk_j in x.keys()
        ),
        GRB.MAXIMIZE
    )

    # Constraint 1: Each talk in at most one pair
    for talk in talks:
        relevant_vars = []
        for (t1, t2) in x.keys():
            if talk == t1 or talk == t2:
                relevant_vars.append(x[t1, t2])

        if relevant_vars:
            model.addConstr(gp.quicksum(relevant_vars)
                            <= 1, name=f"match_{talk}")

    # Constraint 2: Exactly M pairs
    model.addConstr(gp.quicksum(x.values()) == M, name="cardinality")

    # Constraint 3: Special group talks MUST be matched (not left as singles)
    if constraints.require_same_group:
        special_talks = [t for t in talks if metadata.get(
            t) and metadata[t].special_group != 'general']
        for talk in special_talks:
            relevant_vars = []
            for (t1, t2) in x.keys():
                if talk == t1 or talk == t2:
                    relevant_vars.append(x[t1, t2])

            if relevant_vars:
                # This talk must be in exactly one pair
                model.addConstr(
                    gp.quicksum(relevant_vars) == 1,
                    name=f"special_must_match_{talk}"
                )

    # Constraint 4: Link x and v - if we select a violating pair, v must be 1
    for (talk_i, talk_j) in violating_pairs:
        if (talk_i, talk_j) in x:
            model.addConstr(
                x[talk_i, talk_j] <= v[talk_i, talk_j],
                name=f"link_{talk_i}_{talk_j}"
            )

    # Constraint 5: Violation budget
    if v:
        model.addConstr(
            gp.quicksum(v.values()) <= max_violations,
            name="violation_budget"
        )

    # Solve
    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        return None

    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        return None

    if model.SolCount == 0:
        return None

    # Extract solution
    pairs = []
    total_weight = 0
    n_violations = 0

    for (talk_i, talk_j), var in x.items():
        if var.X > 0.5:
            pairs.append((talk_i, talk_j))
            total_weight += get_weight(talk_i, talk_j, weights)
            if (talk_i, talk_j) in violating_pairs:
                n_violations += 1

    return pairs, total_weight, n_violations


# =============================================================================
# PHASE B: CONSTRAINED BLOCK FORMATION
# =============================================================================

def solve_phase_b_constrained(
    pairs: List[Tuple[str, str]],
    singles: List[str],
    n_3: int,
    n_4: int,
    preferences: Dict[str, Set[str]],
    metadata: Dict[str, TalkMetadata],
    constraints: MatchingConstraints,
    time_limit: float = 60.0,
    verbose: bool = True
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str, str]], int, float]:
    """
    Phase B with constraints: Form blocks respecting group/keyword constraints.

    Only allows combining pairs/singles that satisfy constraints.

    Args:
        pairs: Matched pairs from Phase A
        singles: Unmatched singles from Phase A
        n_3, n_4: Number of blocks needed
        preferences: Participant preferences
        metadata: Talk metadata
        constraints: Matching constraints
        time_limit: Solver time limit
        verbose: Print progress

    Returns:
        Tuple of (blocks_3, blocks_4, marginal_weight, solve_time)
    """
    start_time = time.time()

    if verbose:
        print(
            f"Phase B (Constrained): Forming {n_3} 3-blocks and {n_4} 4-blocks")

    # Build co-preference matrix
    all_talks = []
    for p in pairs:
        all_talks.extend(p)
    all_talks.extend(singles)
    weights = build_co_preference_matrix(all_talks, preferences)

    pair_idx = {p: i for i, p in enumerate(pairs)}
    single_idx = {s: i for i, s in enumerate(singles)}

    # Helper to check if all talks in a group can be matched
    def group_feasible(talk_list: List[str]) -> bool:
        """
        Check if all talks in the group satisfy constraints.

        Uses pairwise checking: every pair within the group must satisfy constraints.
        This is less strict than requiring a common keyword across ALL talks,
        which is often infeasible due to keyword distribution.

        For stricter coherence (all talks share one keyword), the evaluator's
        "incoherent_sessions" metric can be used post-hoc.
        """
        for i, t1 in enumerate(talk_list):
            for t2 in talk_list[i+1:]:
                if not can_match(t1, t2, metadata, constraints):
                    return False
        return True

    # Create model
    model = gp.Model("PhaseB_Constrained")
    model.setParam('OutputFlag', 1 if verbose else 0)
    model.setParam('TimeLimit', time_limit)

    # Variables for 4-blocks: z[p1, p2] - only if feasible
    z = {}
    n_feasible_4 = 0
    for i, p1 in enumerate(pairs):
        for p2 in pairs[i+1:]:
            if group_feasible(list(p1) + list(p2)):
                z[p1, p2] = model.addVar(
                    vtype=GRB.BINARY, name=f"z_{pair_idx[p1]}_{pair_idx[p2]}")
                n_feasible_4 += 1

    # Variables for 3-blocks: u[p, s] - only if feasible
    u = {}
    n_feasible_3 = 0
    for p in pairs:
        for s in singles:
            if group_feasible(list(p) + [s]):
                u[p, s] = model.addVar(
                    vtype=GRB.BINARY, name=f"u_{pair_idx[p]}_{single_idx[s]}")
                n_feasible_3 += 1

    if verbose:
        print(f"  Feasible 4-block combinations: {n_feasible_4}")
        print(f"  Feasible 3-block combinations: {n_feasible_3}")

    # Marginal weights
    def marginal_weight_4(p1, p2):
        i, j = p1
        k, l = p2
        return (get_weight(i, k, weights) + get_weight(i, l, weights) +
                get_weight(j, k, weights) + get_weight(j, l, weights))

    def marginal_weight_3(p, s):
        i, j = p
        return get_weight(i, s, weights) + get_weight(j, s, weights)

    # Objective
    model.setObjective(
        gp.quicksum(marginal_weight_4(p1, p2) * z[p1, p2] for p1, p2 in z.keys()) +
        gp.quicksum(marginal_weight_3(p, s) * u[p, s] for p, s in u.keys()),
        GRB.MAXIMIZE
    )

    # Constraint 1: Each pair used exactly once
    for p in pairs:
        z_terms = gp.quicksum(z[p1, p2]
                              for p1, p2 in z.keys() if p in (p1, p2))
        u_terms = gp.quicksum(u[p, s] for s in singles if (p, s) in u)
        model.addConstr(z_terms + u_terms == 1,
                        name=f"pair_once_{pair_idx[p]}")

    # Constraint 2: Each single used exactly once
    for s in singles:
        terms = gp.quicksum(u[p, s] for p in pairs if (p, s) in u)
        model.addConstr(terms == 1, name=f"single_once_{single_idx[s]}")

    # Constraint 3: Exactly n_4 4-blocks
    if z:
        model.addConstr(
            gp.quicksum(z[p1, p2] for p1, p2 in z.keys()) == n_4,
            name="count_4blocks"
        )
    elif n_4 > 0:
        raise ValueError(f"Need {n_4} 4-blocks but no feasible combinations")

    # Solve
    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Phase B infeasible - constraints too restrictive")

    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        raise ValueError(f"Phase B failed with status {model.Status}")

    # Extract solution
    blocks_4 = []
    for (p1, p2), var in z.items():
        if var.X > 0.5:
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


def needs_block_keyword_violation(talk_list: List[str], metadata: Dict[str, TalkMetadata]) -> int:
    """
    Count pairwise keyword violations within a block.

    Returns the number of pairs within the talk_list that violate keyword constraints.
    Only counts violations between "general" group talks (special groups excluded).
    """
    violations = 0
    for i, t1 in enumerate(talk_list):
        for t2 in talk_list[i+1:]:
            if needs_keyword_violation(t1, t2, metadata):
                violations += 1
    return violations


def count_block_violations(
    talk_list: List[str],
    metadata: Dict[str, TalkMetadata],
    constraints: MatchingConstraints
) -> Tuple[bool, bool]:
    """
    Check group feasibility and global keyword coherence within a block.

    Returns:
        Tuple of (group_feasible, is_coherent)
        - group_feasible: True if all talks have same special_group (hard constraint)
        - is_coherent: True if ALL talks share at least one common keyword (global)
    """
    group_feasible = True

    # Check group constraint (HARD)
    for i, t1 in enumerate(talk_list):
        for t2 in talk_list[i+1:]:
            meta_i = metadata.get(t1)
            meta_j = metadata.get(t2)

            if not meta_i or not meta_j:
                continue

            if constraints.require_same_group:
                if meta_i.special_group != meta_j.special_group:
                    group_feasible = False
                    break
        if not group_feasible:
            break

    # Check global keyword coherence (all talks share at least one keyword)
    is_coherent = True
    if constraints.require_common_keyword:
        # Special groups are considered coherent by definition
        all_general = all(
            metadata.get(t) and metadata.get(t).special_group == "general"
            for t in talk_list
        )

        if all_general:
            # Get keyword sets for all talks
            keyword_sets = []
            for t in talk_list:
                meta = metadata.get(t)
                if meta and meta.keywords:
                    keyword_sets.append(meta.keywords)
                else:
                    # Talk without keywords → cannot be coherent
                    keyword_sets.append(set())

            if keyword_sets:
                # Intersection of all keyword sets
                shared = keyword_sets[0].copy()
                for kw_set in keyword_sets[1:]:
                    shared = shared & kw_set
                is_coherent = len(shared) > 0
            else:
                is_coherent = True  # No keywords = coherent by default

    return group_feasible, is_coherent


def is_block_coherent(
    talk_list: List[str],
    metadata: Dict[str, TalkMetadata]
) -> Tuple[bool, Set[str]]:
    """
    Check if ALL talks in a block share at least one common keyword.

    This is stricter than pairwise keyword matching - it requires a keyword
    that appears in every talk's keyword set (intersection).

    Returns:
        Tuple of (is_coherent, shared_keywords)
    """
    keyword_sets = []
    for t in talk_list:
        meta = metadata.get(t)
        if meta and meta.keywords:
            keyword_sets.append(meta.keywords)
        else:
            keyword_sets.append(set())

    if not keyword_sets:
        return True, set()  # No keywords = coherent by default

    # Intersection of all keyword sets
    shared = keyword_sets[0].copy()
    for kw_set in keyword_sets[1:]:
        shared = shared & kw_set

    return len(shared) > 0, shared


def solve_phase_b_soft_constraints(
    pairs: List[Tuple[str, str]],
    singles: List[str],
    n_3: int,
    n_4: int,
    preferences: Dict[str, Set[str]],
    metadata: Dict[str, TalkMetadata],
    constraints: MatchingConstraints,
    talk_presenter: Optional[Dict[str, str]] = None,
    presenter_unavailability: Optional[Dict[str, Set[str]]] = None,
    all_timeslots: Optional[Set[str]] = None,
    time_limit: float = 60.0,
    verbose: bool = True
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str, str]], int, float, int]:
    """
    Phase B with SOFT keyword constraints: Form blocks with iterative violation budget.

    Group constraint (special sessions) is HARD - never allows mixing.
    Keyword constraint is SOFT - iteratively relaxed until feasible.
    Presenter unavailability is HARD - never allows infeasible blocks.

    Args:
        pairs: Matched pairs from Phase A
        singles: Unmatched singles from Phase A
        n_3, n_4: Number of blocks needed
        preferences: Participant preferences
        metadata: Talk metadata
        constraints: Matching constraints
        talk_presenter: Optional mapping from talk_id to presenter_id
        presenter_unavailability: Optional mapping from presenter_id to unavailable timeslots
        all_timeslots: Optional set of all timeslot IDs
        time_limit: Solver time limit
        verbose: Print progress

    Returns:
        Tuple of (blocks_3, blocks_4, marginal_weight, solve_time, n_incoherent)
    """
    import gurobipy as gp
    from gurobipy import GRB
    start_time = time.time()

    if verbose:
        print(f"\n🎯 Goal: Form {n_3} 3-blocks and {n_4} 4-blocks from pairs")
        print(f"   (Each block = talks running consecutively in ONE room)")

    # Build co-preference matrix
    all_talks = []
    for p in pairs:
        all_talks.extend(p)
    all_talks.extend(singles)
    weights = build_co_preference_matrix(all_talks, preferences)

    pair_idx = {p: i for i, p in enumerate(pairs)}
    single_idx = {s: i for i, s in enumerate(singles)}

    # Check if presenter unavailability checking is enabled
    check_presenter_feasibility = (
        talk_presenter is not None and
        presenter_unavailability is not None and
        all_timeslots is not None and
        len(all_timeslots) > 0
    )

    # Pre-compute all group-feasible combinations and their coherence status
    # Group constraint is HARD - only keep combinations with group_feasible=True
    # Presenter unavailability is HARD - only keep combinations that are schedulable
    # is_coherent: True if ALL talks share at least one keyword (global coherence)
    feasible_4 = {}  # (p1, p2) -> is_coherent (bool)
    n_presenter_infeasible_4 = 0
    for i, p1 in enumerate(pairs):
        for p2 in pairs[i+1:]:
            talk_list = list(p1) + list(p2)
            group_feasible, is_coherent = count_block_violations(
                talk_list, metadata, constraints)
            if group_feasible:
                # Also check presenter unavailability feasibility
                if check_presenter_feasibility:
                    if not is_block_feasible(tuple(talk_list), talk_presenter,
                                             presenter_unavailability, all_timeslots):
                        n_presenter_infeasible_4 += 1
                        continue  # Skip this block - presenter conflict
                feasible_4[p1, p2] = is_coherent

    feasible_3 = {}  # (p, s) -> is_coherent (bool)
    n_presenter_infeasible_3 = 0
    for p in pairs:
        for s in singles:
            talk_list = list(p) + [s]
            group_feasible, is_coherent = count_block_violations(
                talk_list, metadata, constraints)
            if group_feasible:
                # Also check presenter unavailability feasibility
                if check_presenter_feasibility:
                    if not is_block_feasible(tuple(talk_list), talk_presenter,
                                             presenter_unavailability, all_timeslots):
                        n_presenter_infeasible_3 += 1
                        continue  # Skip this block - presenter conflict
                feasible_3[p, s] = is_coherent

    if verbose:
        total_4 = len(pairs) * (len(pairs) - 1) // 2
        total_3 = len(pairs) * len(singles)
        n_coherent_4 = sum(1 for v in feasible_4.values() if v)
        n_coherent_3 = sum(1 for v in feasible_3.values() if v)
        n_incoherent_4 = sum(1 for v in feasible_4.values() if not v)
        n_incoherent_3 = sum(1 for v in feasible_3.values() if not v)

        print(f"\n📊 Block combination analysis (GLOBAL coherence):")
        print(f"   4-blocks (pair+pair):")
        print(f"      • Total possible:       {total_4}")
        print(
            f"      • Group-feasible:       {len(feasible_4) + n_presenter_infeasible_4} ({100*(len(feasible_4) + n_presenter_infeasible_4)/max(1, total_4):.1f}%)")
        if check_presenter_feasibility and n_presenter_infeasible_4 > 0:
            print(f"      • Presenter-infeasible: {n_presenter_infeasible_4}")
        print(f"      • Schedulable:          {len(feasible_4)}")
        print(
            f"      • Globally coherent:    {n_coherent_4} (all 4 talks share a keyword)")
        print(f"      • Incoherent:           {n_incoherent_4}")
        print(f"   3-blocks (pair+single):")
        print(f"      • Total possible:       {total_3}")
        print(
            f"      • Group-feasible:       {len(feasible_3) + n_presenter_infeasible_3} ({100*(len(feasible_3) + n_presenter_infeasible_3)/max(1, total_3):.1f}%)")
        if check_presenter_feasibility and n_presenter_infeasible_3 > 0:
            print(f"      • Presenter-infeasible: {n_presenter_infeasible_3}")
        print(f"      • Schedulable:          {len(feasible_3)}")
        print(
            f"      • Globally coherent:    {n_coherent_3} (all 3 talks share a keyword)")
        print(f"      • Incoherent:           {n_incoherent_3}")

    # Marginal weights
    def marginal_weight_4(p1, p2):
        i, j = p1
        k, l = p2
        return (get_weight(i, k, weights) + get_weight(i, l, weights) +
                get_weight(j, k, weights) + get_weight(j, l, weights))

    def marginal_weight_3(p, s):
        i, j = p
        return get_weight(i, s, weights) + get_weight(j, s, weights)

    # Separate coherent and incoherent combinations
    coherent_4 = {k for k, is_coh in feasible_4.items() if is_coh}
    incoherent_4 = {k for k, is_coh in feasible_4.items() if not is_coh}
    coherent_3 = {k for k, is_coh in feasible_3.items() if is_coh}
    incoherent_3 = {k for k, is_coh in feasible_3.items() if not is_coh}

    total_blocks_needed = n_3 + n_4
    max_incoherent = len(incoherent_4) + len(incoherent_3)  # Upper bound

    if verbose:
        print(f"\n🔄 Iterative Relaxation (Phase B) - GLOBAL coherence:")
        print(f"   Starting with 0 allowed incoherent blocks")
        print(
            f"   (All {total_blocks_needed} room-sessions must share a keyword)")

    result = None
    allowed_incoherent = 0

    for allowed_incoherent in range(max_incoherent + 1):
        if verbose and allowed_incoherent > 0:
            print(
                f"   ⚠️  {allowed_incoherent-1} incoherent blocks infeasible → trying {allowed_incoherent}")

        # Create model with all group-feasible combinations
        model = gp.Model("PhaseB_GlobalCoherence")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', max(
            time_limit / (max_incoherent + 1), 5.0))

        # Variables for all group-feasible combinations
        z = {}  # 4-blocks
        for p1, p2 in feasible_4.keys():
            z[p1, p2] = model.addVar(
                vtype=GRB.BINARY, name=f"z_{pair_idx[p1]}_{pair_idx[p2]}")

        u = {}  # 3-blocks
        for p, s in feasible_3.keys():
            u[p, s] = model.addVar(
                vtype=GRB.BINARY, name=f"u_{pair_idx[p]}_{single_idx[s]}")

        # Objective: maximize marginal weight
        model.setObjective(
            gp.quicksum(marginal_weight_4(p1, p2) * z[p1, p2] for p1, p2 in z.keys()) +
            gp.quicksum(marginal_weight_3(p, s) *
                        u[p, s] for p, s in u.keys()),
            GRB.MAXIMIZE
        )

        # Constraint 1: Each pair used exactly once
        for p in pairs:
            z_terms = gp.quicksum(z[p1, p2]
                                  for p1, p2 in z.keys() if p in (p1, p2))
            u_terms = gp.quicksum(u[p, s] for s in singles if (p, s) in u)
            model.addConstr(z_terms + u_terms == 1,
                            name=f"pair_once_{pair_idx[p]}")

        # Constraint 2: Each single used exactly once
        for s in singles:
            terms = gp.quicksum(u[p, s] for p in pairs if (p, s) in u)
            model.addConstr(terms == 1, name=f"single_once_{single_idx[s]}")

        # Constraint 3: Exactly n_4 4-blocks
        if n_4 > 0:
            if not z:
                continue  # Need 4-blocks but none available
            model.addConstr(
                gp.quicksum(z[p1, p2] for p1, p2 in z.keys()) == n_4,
                name="count_4blocks"
            )

        # Constraint 4: Limit number of incoherent blocks
        incoherent_terms = []
        for p1, p2 in incoherent_4:
            if (p1, p2) in z:
                incoherent_terms.append(z[p1, p2])
        for p, s in incoherent_3:
            if (p, s) in u:
                incoherent_terms.append(u[p, s])

        if incoherent_terms:
            model.addConstr(
                gp.quicksum(incoherent_terms) <= allowed_incoherent,
                name="max_incoherent"
            )

        # Solve
        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            continue  # Try allowing more incoherent blocks

        if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            continue

        if model.SolCount == 0:
            continue

        # Extract solution
        blocks_4 = []
        n_incoherent_selected = 0

        for (p1, p2), var in z.items():
            if var.X > 0.5:
                blocks_4.append(p1 + p2)
                if not feasible_4[p1, p2]:  # is_coherent = False
                    n_incoherent_selected += 1

        blocks_3 = []
        for (p, s), var in u.items():
            if var.X > 0.5:
                blocks_3.append(p + (s,))
                if not feasible_3[p, s]:  # is_coherent = False
                    n_incoherent_selected += 1

        result = (blocks_3, blocks_4, int(model.ObjVal), n_incoherent_selected)
        break

    solve_time = time.time() - start_time

    if result is None:
        raise ValueError(
            "Phase B infeasible even allowing all incoherent blocks")

    blocks_3, blocks_4, marginal_weight, n_incoherent_selected = result

    if verbose:
        if n_incoherent_selected == 0:
            print(f"\n   ✅ Solution found with 0 incoherent blocks (perfect!)")
        else:
            print(
                f"\n   ✅ Solution found allowing {allowed_incoherent} incoherent blocks")
            print(
                f"      Actually used: {n_incoherent_selected} incoherent blocks")

        print(f"\n📊 Phase B Results:")
        print(f"   • 3-blocks formed: {len(blocks_3)}")
        print(f"   • 4-blocks formed: {len(blocks_4)}")
        print(f"   • Marginal weight: {marginal_weight}")

        # Analyze block coherence (all talks share a keyword)
        n_coherent_3 = 0
        n_coherent_4 = 0
        incoherent_details = []

        for block in blocks_3:
            coherent, shared = is_block_coherent(list(block), metadata)
            if coherent:
                n_coherent_3 += 1
            else:
                kws = {}
                for t in block:
                    meta = metadata.get(t)
                    kws[t] = list(meta.keywords) if meta else []
                incoherent_details.append(("3-block", block, kws))

        for block in blocks_4:
            coherent, shared = is_block_coherent(list(block), metadata)
            if coherent:
                n_coherent_4 += 1
            else:
                kws = {}
                for t in block:
                    meta = metadata.get(t)
                    kws[t] = list(meta.keywords) if meta else []
                incoherent_details.append(("4-block", block, kws))

        total_blocks = len(blocks_3) + len(blocks_4)
        total_coherent = n_coherent_3 + n_coherent_4
        print(f"\n📊 Room-Session Coherence (GLOBAL = ALL talks share one keyword):")
        print(f"   • 3-talk sessions coherent: {n_coherent_3}/{len(blocks_3)}")
        print(f"   • 4-talk sessions coherent: {n_coherent_4}/{len(blocks_4)}")
        print(
            f"   • Total coherent:           {total_coherent}/{total_blocks} ({100*total_coherent/max(1, total_blocks):.0f}%)")

        if incoherent_details:
            print(
                f"\n⚠️  Incoherent room-sessions ({len(incoherent_details)} total):")
            print(f"   (No single keyword shared by ALL talks in the session)")
            show_count = min(5, len(incoherent_details))
            for block_type, block, kws in incoherent_details[:show_count]:
                print(f"   {block_type}: {', '.join(block)}")
                for t, kw_list in kws.items():
                    kw_str = ', '.join(kw_list[:3]) if kw_list else '(none)'
                    if len(kw_list) > 3:
                        kw_str += f" +{len(kw_list)-3} more"
                    print(f"      {t}: [{kw_str}]")
            if len(incoherent_details) > show_count:
                print(
                    f"   ... and {len(incoherent_details) - show_count} more")

        print(f"\n   Time: {solve_time:.2f}s")

    return blocks_3, blocks_4, marginal_weight, solve_time, n_incoherent_selected


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_constrained_matching_pipeline(
    instance: ProblemInstance,
    metadata: Dict[str, TalkMetadata],
    constraints: MatchingConstraints = None,
    time_limit: float = 300.0,
    verbose: bool = True,
    run_phase3: bool = True,
    phase3_method: str = "milp",
    max_feasibility_retries: int = 10
):
    """
    Run the constrained matching pipeline.

    Like run_matching_pipeline but respects group/keyword constraints.
    Includes feasibility checks for presenter unavailabilities with retry loop.

    Args:
        instance: Problem instance
        metadata: Talk metadata with groups and keywords
        constraints: Matching constraints (default: require same group only)
        time_limit: Total time limit
        verbose: Print progress
        run_phase3: Whether to run Phase 3
        phase3_method: "milp" or "hungarian"
        max_feasibility_retries: Max retries for Phase C when violations found

    Returns:
        Tuple of (MatchingPipelineResult, Phase3Result or None)
    """
    if constraints is None:
        constraints = MatchingConstraints(
            require_same_group=True,
            require_common_keyword=False
        )

    if verbose:
        print("=" * 70)
        print("CONSTRAINED MATCHING PIPELINE (MATCHING_KW)")
        print("=" * 70)
        print()
        print("📋 Constraint Mode:")
        print(
            f"   • Same special group (HARD): {constraints.require_same_group}")
        print(
            f"   • Common keyword (SOFT):     {constraints.require_common_keyword}")
        if constraints.max_keyword_violations is not None:
            print(
                f"   • Initial violation budget:  {constraints.max_keyword_violations}")
        print()
        print("ℹ️  How iterative relaxation works:")
        print("   Phase A/B start with 0 allowed keyword violations.")
        print("   If infeasible, the budget increases by 1 until a solution is found.")
        print("   This ensures minimum violations while guaranteeing feasibility.")
        print()

        # Show group distribution
        groups = {}
        for m in metadata.values():
            groups[m.special_group] = groups.get(m.special_group, 0) + 1
        print(f"📊 Talk distribution by group:")
        for g, count in sorted(groups.items(), key=lambda x: -x[1]):
            print(f"   • {g}: {count} talks")

        # Count talks with keywords
        n_with_kw = sum(1 for m in metadata.values() if m.keywords)
        n_without_kw = len(metadata) - n_with_kw
        print(f"\n📊 Keyword coverage:")
        print(f"   • With keywords:    {n_with_kw} talks")
        print(f"   • Without keywords: {n_without_kw} talks")

    # Extract block configuration
    n_3 = 0
    n_4 = 0
    tuple_requirements_3 = {}
    tuple_requirements_4 = {}

    for type_id, bt in instance.block_types.items():
        n = bt["n"]
        k = bt["k"]
        count = bt["count"]
        total_room_sessions = count * n

        if k == 3:
            n_3 += total_room_sessions
            tuple_requirements_3[n] = tuple_requirements_3.get(n, 0) + count
        elif k == 4:
            n_4 += total_room_sessions
            tuple_requirements_4[n] = tuple_requirements_4.get(n, 0) + count
        else:
            raise ValueError(f"Only k=3 or k=4 supported, got k={k}")

    if verbose:
        print(f"\n📐 Block Configuration:")
        print(f"   • 3-talk room-sessions: {n_3}")
        print(f"   • 4-talk room-sessions: {n_4}")
        print(f"   • Total room-sessions:  {n_3 + n_4}")

    # Phase A: Constrained pair matching
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE A: Pair Matching with Keyword Constraints")
        print("=" * 70)

    # Use soft constraints if max_keyword_violations is set
    n_keyword_violations = 0
    if constraints.max_keyword_violations is not None:
        pairs, singles, phase_a_weight, phase_a_time, n_keyword_violations = solve_phase_a_soft_constraints(
            instance, n_3, n_4, metadata, constraints,
            time_limit=time_limit * 0.15, verbose=verbose
        )
    else:
        pairs, singles, phase_a_weight, phase_a_time = solve_phase_a_constrained(
            instance, n_3, n_4, metadata, constraints,
            time_limit=time_limit * 0.15, verbose=verbose
        )

    # Phase B: Constrained block formation
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE B: Block Formation with Keyword Constraints")
        print("=" * 70)

    n_phase_b_violations = 0
    # Always use soft constraints approach for Phase B - allows iterative relaxation
    all_timeslots = instance.get_all_timeslots()
    blocks_3, blocks_4, phase_b_weight, phase_b_time, n_phase_b_violations = solve_phase_b_soft_constraints(
        pairs, singles, n_3, n_4,
        instance.preferences, metadata, constraints,
        talk_presenter=instance.talk_presenter,
        presenter_unavailability=instance.presenter_unavailability,
        all_timeslots=all_timeslots,
        time_limit=time_limit * 0.15, verbose=verbose
    )

    # Phase C/D/3 with feasibility retry loop
    forbidden_solutions_3: List[List[Tuple]] = []
    forbidden_solutions_4: List[List[Tuple]] = []

    phase_c_time_total = 0.0
    phase_d_time_total = 0.0
    feasibility_achieved = False
    phase3_result = None

    for retry in range(max_feasibility_retries):
        if verbose:
            print("\n" + "=" * 70)
            if retry == 0:
                print("PHASE C: Tuple Selection (no constraints)")
            else:
                print(f"PHASE C: Tuple Selection (retry {retry})")
            print("=" * 70)

        tuples_3, tuples_4, phase_c_cost, phase_c_time = solve_phase_c(
            blocks_3, blocks_4,
            tuple_requirements_3, tuple_requirements_4,
            instance.preferences,
            time_limit=time_limit * 0.5 / (retry + 1),
            verbose=verbose,
            forbidden_solutions_3=forbidden_solutions_3 if forbidden_solutions_3 else None,
            forbidden_solutions_4=forbidden_solutions_4 if forbidden_solutions_4 else None
        )
        phase_c_time_total += phase_c_time

        # Phase D: Talk ordering
        if verbose:
            print("\n" + "=" * 70)
            print("PHASE D: Talk Ordering")
            print("=" * 70)

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
                print("\n" + "=" * 70)
                print("PHASE 3: Room Assignment")
                print("=" * 70)

            phase3_result = _run_phase3(
                ordered_blocks, instance,
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
            feasibility_achieved = True
            break

    if not feasibility_achieved and verbose:
        print(
            f"  ⚠ Warning: Could not find feasible solution after {max_feasibility_retries} retries")

    # Summary
    if verbose:
        total_time = phase_a_time + phase_b_time + \
            phase_c_time_total + phase_d_time_total
        print("\n" + "=" * 70)
        print("MATCHING_KW PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\n📊 Final Summary:")
        print(f"   • Total time:            {total_time:.2f}s")
        print(f"   • Missed attendance:     {phase_c_cost}")
        print(
            f"   • Incoherent sessions:   {n_phase_b_violations} (out of {n_3 + n_4} room-sessions)")

        if len(forbidden_solutions_3) > 0 or len(forbidden_solutions_4) > 0:
            print(
                f"   • Feasibility retries:   {len(forbidden_solutions_3) + len(forbidden_solutions_4)}")

        if n_phase_b_violations == 0:
            print(f"\n✅ Perfect global coherence achieved!")
            print(f"   All room-sessions have talks sharing at least one keyword.")
        else:
            print(
                f"\n⚠️  {n_phase_b_violations} room-sessions without a shared keyword")
            print(f"   (Relaxation was needed to find a feasible solution)")

    result = MatchingPipelineResult(
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

    return result, phase3_result

    return result, phase3_result
