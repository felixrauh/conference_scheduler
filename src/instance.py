"""
Problem instance construction module.

Builds optimization-ready data structures from ConferenceData.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, FrozenSet
from itertools import combinations

from .data_loader import ConferenceData


@dataclass
class ProblemInstance:
    """
    Optimization-ready problem instance.

    Contains all sets and parameters needed for the three-phase optimization.
    """

    # Reference to source data
    conference_data: ConferenceData

    # Sets
    talks: List[str]  # List of talk_ids
    participants: List[str]  # List of participant_ids (derived from preferences)
    # {type_id: {"n": int, "k": int, "count": int}}
    block_types: Dict[str, Dict]

    # Preference matrix: q[p][i] = 1 if participant p wants to attend talk i
    # Stored as {participant_id: set of talk_ids}
    preferences: Dict[str, Set[str]] = field(default_factory=dict)

    # Talk -> presenter mapping
    talk_presenter: Dict[str, str] = field(default_factory=dict)

    # Presenter unavailability: {presenter_id: set of timeslot_ids}
    presenter_unavailability: Dict[str, Set[str]] = field(default_factory=dict)

    # Timeslots grouped by block type: {type_id: list of timeslot dicts}
    timeslots_by_type: Dict[str, List[Dict]] = field(default_factory=dict)

    def compute_tuple_cost(self, talk_tuple: Tuple[str, ...]) -> int:
        """
        Compute missed attendance coefficient c_e for an n-tuple of talks.

        c_e = sum over participants of max(0, (preferred talks in tuple) - 1)
        """
        cost = 0
        for p_id, prefs in self.preferences.items():
            preferred_in_tuple = sum(1 for t in talk_tuple if t in prefs)
            if preferred_in_tuple > 1:
                cost += preferred_in_tuple - 1
        return cost

    def get_presenter_for_talk(self, talk_id: str) -> str:
        """Get presenter_id for a talk."""
        return self.talk_presenter.get(talk_id)

    def talks_have_same_presenter(self, talk_ids: Tuple[str, ...]) -> bool:
        """Check if any two talks in the tuple have the same presenter."""
        presenters = [self.talk_presenter.get(t, f"__no_presenter_{t}__") for t in talk_ids]
        return len(presenters) != len(set(presenters))

    def get_all_timeslots(self) -> Set[str]:
        """Get the set of all timeslot IDs from the conference data."""
        return {ts["id"] for ts in self.conference_data.timeslots}

    def talks_have_unavailability_conflict(self, talk_ids: Tuple[str, ...]) -> bool:
        """
        Check if talks have conflicting presenter unavailabilities.

        A conflict exists when the union of all presenters' unavailable timeslots
        equals all available timeslots (i.e., no timeslot works for all presenters).

        Args:
            talk_ids: Tuple of talk IDs to check

        Returns:
            True if there's a conflict (can't be scheduled together), False otherwise
        """
        all_timeslots = self.get_all_timeslots()
        if not all_timeslots:
            return False  # No timeslots defined, can't determine conflict

        # Collect unavailable timeslots for all presenters
        combined_unavailable: Set[str] = set()
        for talk_id in talk_ids:
            presenter = self.talk_presenter.get(talk_id)
            if presenter and presenter in self.presenter_unavailability:
                combined_unavailable |= self.presenter_unavailability[presenter]

        # If combined unavailability covers all timeslots, there's a conflict
        return combined_unavailable >= all_timeslots


def compute_infeasible_talk_pairs(
    instance: ProblemInstance,
    verbose: bool = False,
    min_unavailable_threshold: int = 5
) -> Set[Tuple[str, str]]:
    """
    Compute all pairs of talks that should not be scheduled together.

    A pair (t1, t2) is considered infeasible if:
    - Their presenters' combined unavailabilities cover at least
      `min_unavailable_threshold` timeslots (default: 5 of 7)

    This is stricter than requiring ALL timeslots to be blocked, because
    pairs with very few available slots create scheduling bottlenecks in
    later phases.

    Args:
        instance: Problem instance with presenter unavailabilities
        verbose: Whether to print statistics
        min_unavailable_threshold: Minimum number of blocked timeslots to
            consider a pair infeasible (default 5 means pairs with only
            2 or fewer available slots are filtered out)

    Returns:
        Set of (talk_i, talk_j) pairs that cannot be in the same tuple
        (pairs are stored with talk_i < talk_j lexicographically)
    """
    all_timeslots = instance.get_all_timeslots()
    if not all_timeslots:
        return set()

    total_timeslots = len(all_timeslots)
    infeasible_pairs: Set[Tuple[str, str]] = set()
    talks = instance.talks

    # Only check talks whose presenters have unavailabilities
    talks_with_constraints = [
        t for t in talks
        if instance.talk_presenter.get(t) in instance.presenter_unavailability
    ]

    # Check all pairs of constrained talks
    for i, t1 in enumerate(talks_with_constraints):
        p1 = instance.talk_presenter[t1]
        unavail1 = instance.presenter_unavailability.get(p1, set())

        for t2 in talks_with_constraints[i+1:]:
            p2 = instance.talk_presenter[t2]
            unavail2 = instance.presenter_unavailability.get(p2, set())

            # Combined unavailability
            combined = unavail1 | unavail2
            # Mark as infeasible if combined unavailability >= threshold
            if len(combined) >= min_unavailable_threshold:
                # Store as sorted pair for consistent lookup
                pair = tuple(sorted([t1, t2]))
                infeasible_pairs.add(pair)

    if verbose and infeasible_pairs:
        print(f"\nInfeasible talk pairs (presenter unavailability conflicts):")
        print(
            f"  Threshold: {min_unavailable_threshold}/{total_timeslots} timeslots blocked")
        print(f"  Found {len(infeasible_pairs)} infeasible pairs")
        for t1, t2 in sorted(infeasible_pairs):
            p1, p2 = instance.talk_presenter[t1], instance.talk_presenter[t2]
            u1 = instance.presenter_unavailability.get(p1, set())
            u2 = instance.presenter_unavailability.get(p2, set())
            combined = u1 | u2
            available = total_timeslots - len(combined)
            print(f"    {t1} ({p1}) + {t2} ({p2}) → {available} slots available")

    return infeasible_pairs


def compute_forbidden_tuple_sizes(
    instance: ProblemInstance,
    verbose: bool = False
) -> Dict[str, Set[int]]:
    """
    Compute which tuple sizes each talk is forbidden from.

    A talk is forbidden from a tuple size if:
    - All block types of that size (n-value) map to timeslots the presenter can't attend

    Args:
        instance: Problem instance with block_types, timeslots_by_type, presenter_unavailability
        verbose: Whether to print restrictions

    Returns:
        Dict mapping talk_id -> set of forbidden tuple sizes (n values)
    """
    # Build mapping: tuple_size (n) -> set of possible timeslots
    size_to_timeslots: Dict[int, Set[str]] = {}
    for type_id, type_info in instance.block_types.items():
        n = type_info['n']  # tuple size
        if n not in size_to_timeslots:
            size_to_timeslots[n] = set()
        # Add all timeslots of this block type
        for ts in instance.timeslots_by_type.get(type_id, []):
            size_to_timeslots[n].add(ts['id'])

    # Also check conference_data.timeslots if timeslots_by_type is empty
    if not any(size_to_timeslots.values()) and instance.conference_data:
        for ts in instance.conference_data.timeslots:
            type_id = ts.get('type_id')
            if type_id and type_id in instance.block_types:
                n = instance.block_types[type_id]['n']
                if n not in size_to_timeslots:
                    size_to_timeslots[n] = set()
                size_to_timeslots[n].add(ts['id'])

    all_timeslots = instance.get_all_timeslots()

    forbidden: Dict[str, Set[int]] = {}

    for talk_id in instance.talks:
        presenter = instance.talk_presenter.get(talk_id)
        if not presenter or presenter not in instance.presenter_unavailability:
            continue  # No constraints

        unavail = instance.presenter_unavailability[presenter]
        avail = all_timeslots - unavail

        talk_forbidden = set()
        for n, timeslots_for_n in size_to_timeslots.items():
            # If none of the timeslots for this size are available to presenter
            if not (timeslots_for_n & avail):
                talk_forbidden.add(n)

        if talk_forbidden:
            forbidden[talk_id] = talk_forbidden

    if verbose and forbidden:
        print(f"\nTuple size restrictions (presenter availability):")
        for talk_id, sizes in sorted(forbidden.items()):
            presenter = instance.talk_presenter.get(talk_id, 'unknown')
            print(f"  {talk_id} ({presenter}): cannot be in size-{sizes} tuples")

    return forbidden


def compute_forbidden_block_types(
    instance: ProblemInstance,
    verbose: bool = False
) -> Dict[str, Set[str]]:
    """
    Compute which block TYPES each talk is forbidden from.

    This is more granular than `compute_forbidden_tuple_sizes` because it 
    distinguishes between block types with the same n value but different
    available timeslots.

    For example, if size-5 tuples can go to 5R4T (TA, TB, FB) or 5R3T (TC),
    and a presenter can only attend TA/FB, they are forbidden from 5R3T blocks
    but NOT from all size-5 tuples.

    Args:
        instance: Problem instance with block_types, timeslots_by_type, presenter_unavailability
        verbose: Whether to print restrictions

    Returns:
        Dict mapping talk_id -> set of forbidden block_type_ids
    """
    all_timeslots = instance.get_all_timeslots()

    # Build mapping: block_type -> set of timeslot IDs
    type_to_timeslots: Dict[str, Set[str]] = {}
    for type_id, type_info in instance.block_types.items():
        type_to_timeslots[type_id] = set()
        for ts in instance.timeslots_by_type.get(type_id, []):
            type_to_timeslots[type_id].add(ts['id'])

    # Also check conference_data.timeslots if timeslots_by_type is empty
    if not any(type_to_timeslots.values()) and instance.conference_data:
        for ts in instance.conference_data.timeslots:
            type_id = ts.get('type_id')
            if type_id and type_id in instance.block_types:
                if type_id not in type_to_timeslots:
                    type_to_timeslots[type_id] = set()
                type_to_timeslots[type_id].add(ts['id'])

    forbidden: Dict[str, Set[str]] = {}

    for talk_id in instance.talks:
        presenter = instance.talk_presenter.get(talk_id)
        if not presenter or presenter not in instance.presenter_unavailability:
            continue  # No constraints

        unavail = instance.presenter_unavailability[presenter]
        avail = all_timeslots - unavail

        talk_forbidden = set()
        for type_id, timeslots_for_type in type_to_timeslots.items():
            # If none of the timeslots for this block type are available to presenter
            if not (timeslots_for_type & avail):
                talk_forbidden.add(type_id)

        if talk_forbidden:
            forbidden[talk_id] = talk_forbidden

    if verbose and forbidden:
        print(f"\nBlock type restrictions (presenter availability):")
        for talk_id, types in sorted(forbidden.items()):
            presenter = instance.talk_presenter.get(talk_id, 'unknown')
            print(f"  {talk_id} ({presenter}): cannot be in block types {types}")

    return forbidden


def build_instance(data: ConferenceData) -> ProblemInstance:
    """
    Build a ProblemInstance from validated ConferenceData.
    """
    instance = ProblemInstance(
        conference_data=data,
        talks=list(data.talks["talk_id"]),
        participants=list(data.preference_matrix.keys()),
        block_types=data.block_types,
        preferences=data.preference_matrix,
        presenter_unavailability=data.presenter_unavailability,
    )

    # Build talk -> presenter mapping
    for _, row in data.talks.iterrows():
        instance.talk_presenter[row["talk_id"]] = row["presenter_id"]

    # Group timeslots by block type
    for ts in data.timeslots:
        type_id = ts["type_id"]
        if type_id not in instance.timeslots_by_type:
            instance.timeslots_by_type[type_id] = []
        instance.timeslots_by_type[type_id].append(ts)

    return instance
