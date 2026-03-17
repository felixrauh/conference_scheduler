"""
Swap Optimization: Post-Phase 3 Local Search

This module implements a local search phase that resolves presenter availability
violations by swapping talks within the schedule. It runs after Phase 3
(block scheduling) and before Phase 4 (room assignment).

Algorithm:
1. Detect presenter infeasibilities (talks scheduled when presenter unavailable)
2. For each infeasibility, find candidate swap partners (talks that can trade positions)
3. Score each swap by:
   - Feasibility: swap must not create new presenter violations
   - Missed attendance delta: prefer swaps that reduce conflicts
   - Keyword coherence (optional): prefer swaps that maintain session coherence
4. Apply the best feasible swap
5. Repeat until no violations remain or no feasible swaps exist

Complexity:
- O(V * T^2 * P) where V=violations, T=talks, P=participants
- In practice much faster due to early filtering

Usage:
    from src.swap_optimization import optimize_presenter_violations
    
    phase3_result = optimize_presenter_violations(
        phase3_result,
        talk_presenter,
        presenter_unavailability,
        preferences,
        talk_keywords=None,  # Optional: for coherence scoring
        verbose=True
    )
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import copy

from .phase2 import Block
from .phase3 import Phase3Result, RoomAssignment


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ViolationType(Enum):
    """Types of violations that can be resolved by swapping."""
    PRESENTER_UNAVAILABLE = "presenter_unavailable"
    DUMMY_IN_SHORT_BLOCK = "dummy_in_short_block"
    MULTIPLE_DUMMIES_IN_SESSION = "multiple_dummies_in_session"


@dataclass
class Violation:
    """A violation that should be resolved by swapping."""
    talk_id: str
    presenter_id: str       # Empty for dummy violations
    timeslot_id: str
    block_idx: int          # Index in Phase3Result.assignments
    tuple_idx: int          # Index within block.tuples
    room_idx: int           # Index within tuple
    violation_type: ViolationType = ViolationType.PRESENTER_UNAVAILABLE


@dataclass
class SwapCandidate:
    """A potential swap to resolve a violation."""
    # Source (the violating talk)
    src_talk_id: str
    src_block_idx: int
    src_tuple_idx: int
    src_room_idx: int

    # Target (the swap partner)
    tgt_talk_id: str
    tgt_block_idx: int
    tgt_tuple_idx: int
    tgt_room_idx: int

    # Scores
    feasible: bool              # Does not create new violations
    missed_attendance_delta: int  # Negative = improvement
    keyword_delta: float        # Higher = better coherence (0-1)
    combined_score: float       # Weighted combination


@dataclass
class SwapResult:
    """Results from the swap optimization phase."""
    phase3_result: Phase3Result  # Modified result
    violations_before: int       # Count before optimization
    violations_resolved: int
    violations_remaining: int
    swaps_applied: List[SwapCandidate]
    iterations: int


# =============================================================================
# VIOLATION DETECTION
# =============================================================================

def detect_violations(
    phase3_result: Phase3Result,
    talk_presenter: Dict[str, str],
    presenter_unavailability: Dict[str, Set[str]]
) -> List[Violation]:
    """
    Detect all presenter availability violations in the schedule.

    Args:
        phase3_result: Current Phase 3 result
        talk_presenter: talk_id -> presenter_id
        presenter_unavailability: presenter_id -> set of unavailable timeslot_ids

    Returns:
        List of Violation objects
    """
    violations = []

    for block_idx, assignment in enumerate(phase3_result.assignments):
        timeslot_id = assignment.timeslot.get('id', str(assignment.timeslot))
        block = assignment.block

        for tuple_idx, ntuple in enumerate(block.tuples):
            for room_idx, talk_id in enumerate(ntuple):
                presenter_id = talk_presenter.get(talk_id)
                if not presenter_id:
                    continue

                unavailable = presenter_unavailability.get(presenter_id, set())
                if timeslot_id in unavailable:
                    violations.append(Violation(
                        talk_id=talk_id,
                        presenter_id=presenter_id,
                        timeslot_id=timeslot_id,
                        block_idx=block_idx,
                        tuple_idx=tuple_idx,
                        room_idx=room_idx,
                        violation_type=ViolationType.PRESENTER_UNAVAILABLE
                    ))

    return violations


def detect_dummy_violations(
    phase3_result: Phase3Result,
    short_block_threshold: int = 3
) -> List[Violation]:
    """
    Detect dummy talk violations in the schedule.

    A dummy talk is considered in violation if:
    1. It's in a block with <= short_block_threshold slots (k <= 3)
    2. There are 2+ dummy talks in the same room-session

    Args:
        phase3_result: Current Phase 3 result
        short_block_threshold: Maximum k (slots) considered "short" (default: 3)

    Returns:
        List of Violation objects for dummy issues
    """
    violations = []

    for block_idx, assignment in enumerate(phase3_result.assignments):
        timeslot_id = assignment.timeslot.get('id', str(assignment.timeslot))
        block = assignment.block
        n_slots = len(block.tuples)  # k = number of timeslots in block
        n_rooms = len(block.tuples[0]) if block.tuples else 0

        # Track dummies per room-session (room_idx -> list of (tuple_idx, talk_id))
        dummies_by_room: Dict[int, List[Tuple[int, str]]] = defaultdict(list)

        for tuple_idx, ntuple in enumerate(block.tuples):
            for room_idx, talk_id in enumerate(ntuple):
                if talk_id.startswith('DUMMY'):
                    dummies_by_room[room_idx].append((tuple_idx, talk_id))

        # Check for violations
        for room_idx, dummy_list in dummies_by_room.items():
            if not dummy_list:
                continue

            # Violation type 1: Dummy in short block (k <= threshold)
            if n_slots <= short_block_threshold:
                for tuple_idx, talk_id in dummy_list:
                    violations.append(Violation(
                        talk_id=talk_id,
                        presenter_id="",  # No presenter for dummy
                        timeslot_id=timeslot_id,
                        block_idx=block_idx,
                        tuple_idx=tuple_idx,
                        room_idx=room_idx,
                        violation_type=ViolationType.DUMMY_IN_SHORT_BLOCK
                    ))

            # Violation type 2: Multiple dummies in same room-session
            elif len(dummy_list) >= 2:
                # Mark all but the first dummy as violations
                # (we want to spread them out)
                for tuple_idx, talk_id in dummy_list[1:]:
                    violations.append(Violation(
                        talk_id=talk_id,
                        presenter_id="",
                        timeslot_id=timeslot_id,
                        block_idx=block_idx,
                        tuple_idx=tuple_idx,
                        room_idx=room_idx,
                        violation_type=ViolationType.MULTIPLE_DUMMIES_IN_SESSION
                    ))

    return violations


def detect_all_violations(
    phase3_result: Phase3Result,
    talk_presenter: Dict[str, str],
    presenter_unavailability: Dict[str, Set[str]],
    check_dummy_violations: bool = True,
    short_block_threshold: int = 3
) -> List[Violation]:
    """
    Detect all violations (presenter + dummy) in the schedule.

    Args:
        phase3_result: Current Phase 3 result
        talk_presenter: talk_id -> presenter_id
        presenter_unavailability: presenter_id -> set of unavailable timeslot_ids
        check_dummy_violations: Whether to check for dummy talk violations
        short_block_threshold: Max slots for "short block" dummy violation

    Returns:
        Combined list of all violations
    """
    violations = detect_violations(
        phase3_result, talk_presenter, presenter_unavailability
    )

    if check_dummy_violations:
        dummy_violations = detect_dummy_violations(
            phase3_result, short_block_threshold
        )
        violations.extend(dummy_violations)

    return violations


# =============================================================================
# MISSED ATTENDANCE COMPUTATION
# =============================================================================

def compute_missed_attendance_for_timeslot(
    talks_at_timeslot: List[str],
    preferences: Dict[str, Set[str]]
) -> int:
    """
    Compute missed attendance for a single timeslot.

    If k+1 preferred talks are at the same time, k are missed.

    Args:
        talks_at_timeslot: All talks scheduled at this timeslot (in parallel)
        preferences: participant_id -> set of preferred talk_ids

    Returns:
        Total missed attendance count
    """
    talks_set = set(talks_at_timeslot)
    total_missed = 0

    for participant_id, prefs in preferences.items():
        preferred_here = len(prefs & talks_set)
        if preferred_here > 1:
            total_missed += (preferred_here - 1)

    return total_missed


def compute_total_missed_attendance(
    phase3_result: Phase3Result,
    preferences: Dict[str, Set[str]]
) -> int:
    """
    Compute total missed attendance across all timeslots.

    Args:
        phase3_result: Current schedule
        preferences: Participant preferences

    Returns:
        Total missed attendance
    """
    total = 0

    # Group talks by (timeslot, slot_within_block)
    # Truly parallel talks are at the same timeslot AND same slot position
    for assignment in phase3_result.assignments:
        timeslot_id = assignment.timeslot.get('id', str(assignment.timeslot))
        block = assignment.block

        # Each tuple represents talks at the same slot position (truly parallel)
        for tuple_idx, ntuple in enumerate(block.tuples):
            total += compute_missed_attendance_for_timeslot(
                list(ntuple), preferences
            )

    return total


def compute_swap_missed_attendance_delta(
    phase3_result: Phase3Result,
    swap: SwapCandidate,
    preferences: Dict[str, Set[str]]
) -> int:
    """
    Compute the change in missed attendance if we apply this swap.

    We only need to recompute for the affected timeslots (at most 4 tuples).

    Args:
        phase3_result: Current schedule
        swap: The swap candidate
        preferences: Participant preferences

    Returns:
        Delta (negative = improvement, positive = worse)
    """
    # Get current tuples
    src_assignment = phase3_result.assignments[swap.src_block_idx]
    tgt_assignment = phase3_result.assignments[swap.tgt_block_idx]

    src_tuple = list(src_assignment.block.tuples[swap.src_tuple_idx])
    tgt_tuple = list(tgt_assignment.block.tuples[swap.tgt_tuple_idx])

    # Compute missed attendance before swap
    before = (
        compute_missed_attendance_for_timeslot(src_tuple, preferences) +
        compute_missed_attendance_for_timeslot(tgt_tuple, preferences)
    )

    # Apply swap
    src_tuple[swap.src_room_idx] = swap.tgt_talk_id
    tgt_tuple[swap.tgt_room_idx] = swap.src_talk_id

    # Compute missed attendance after swap
    after = (
        compute_missed_attendance_for_timeslot(src_tuple, preferences) +
        compute_missed_attendance_for_timeslot(tgt_tuple, preferences)
    )

    return after - before


# =============================================================================
# KEYWORD COHERENCE
# =============================================================================

def compute_session_keyword_score(
    talks: List[str],
    talk_keywords: Dict[str, Set[str]]
) -> float:
    """
    Compute keyword coherence score for a session (room-session).

    Score = proportion of talks that share at least one keyword with all others.

    Args:
        talks: List of talk_ids in the session
        talk_keywords: talk_id -> set of keywords

    Returns:
        Score between 0 (no coherence) and 1 (all share common keyword)
    """
    if len(talks) < 2:
        return 1.0

    # Get keyword sets
    keyword_sets = [talk_keywords.get(t, set()) for t in talks]

    # Check if any keyword appears in all talks
    if all(keyword_sets):
        shared = keyword_sets[0]
        for kw_set in keyword_sets[1:]:
            shared = shared & kw_set
        if shared:
            return 1.0

    # Compute pairwise overlap score
    total_pairs = 0
    overlap_count = 0
    for i in range(len(keyword_sets)):
        for j in range(i + 1, len(keyword_sets)):
            total_pairs += 1
            if keyword_sets[i] & keyword_sets[j]:
                overlap_count += 1

    return overlap_count / total_pairs if total_pairs > 0 else 0.0


def compute_swap_keyword_delta(
    phase3_result: Phase3Result,
    swap: SwapCandidate,
    talk_keywords: Dict[str, Set[str]]
) -> float:
    """
    Compute the change in keyword coherence if we apply this swap.

    We compute the coherence for the affected room-sessions before and after.

    Args:
        phase3_result: Current schedule
        swap: The swap candidate
        talk_keywords: talk_id -> set of keywords

    Returns:
        Delta (positive = improvement in coherence)
    """
    if not talk_keywords:
        return 0.0

    # Get affected room-sessions (talks in the same room position across tuples)
    src_assignment = phase3_result.assignments[swap.src_block_idx]
    tgt_assignment = phase3_result.assignments[swap.tgt_block_idx]

    # Source room-session: all talks in src_room_idx across all tuples
    src_room_talks = [
        t[swap.src_room_idx] for t in src_assignment.block.tuples
    ]

    # Target room-session
    tgt_room_talks = [
        t[swap.tgt_room_idx] for t in tgt_assignment.block.tuples
    ]

    # Compute coherence before
    before = (
        compute_session_keyword_score(src_room_talks, talk_keywords) +
        compute_session_keyword_score(tgt_room_talks, talk_keywords)
    )

    # Apply swap
    src_room_talks_after = src_room_talks.copy()
    tgt_room_talks_after = tgt_room_talks.copy()

    src_room_talks_after[swap.src_tuple_idx] = swap.tgt_talk_id
    tgt_room_talks_after[swap.tgt_tuple_idx] = swap.src_talk_id

    # Compute coherence after
    after = (
        compute_session_keyword_score(src_room_talks_after, talk_keywords) +
        compute_session_keyword_score(tgt_room_talks_after, talk_keywords)
    )

    return after - before  # Positive = improvement


# =============================================================================
# SWAP CANDIDATE GENERATION
# =============================================================================

def check_swap_creates_violation(
    src_talk_id: str,
    tgt_talk_id: str,
    src_timeslot: str,
    tgt_timeslot: str,
    talk_presenter: Dict[str, str],
    presenter_unavailability: Dict[str, Set[str]]
) -> bool:
    """
    Check if swapping these talks would create a new presenter violation.

    After swap:
    - src_talk moves to tgt_timeslot
    - tgt_talk moves to src_timeslot

    Returns True if swap creates a violation.
    """
    # Check src_talk at tgt_timeslot
    src_presenter = talk_presenter.get(src_talk_id)
    if src_presenter:
        unavailable = presenter_unavailability.get(src_presenter, set())
        if tgt_timeslot in unavailable:
            return True

    # Check tgt_talk at src_timeslot
    tgt_presenter = talk_presenter.get(tgt_talk_id)
    if tgt_presenter:
        unavailable = presenter_unavailability.get(tgt_presenter, set())
        if src_timeslot in unavailable:
            return True

    return False


def generate_swap_candidates(
    violation: Violation,
    phase3_result: Phase3Result,
    talk_presenter: Dict[str, str],
    presenter_unavailability: Dict[str, Set[str]],
    preferences: Dict[str, Set[str]],
    talk_keywords: Optional[Dict[str, Set[str]]] = None,
    keyword_weight: float = 0.1
) -> List[SwapCandidate]:
    """
    Generate all feasible swap candidates for a given violation.

    A swap is feasible if:
    1. It moves the violating talk to a timeslot where the presenter is available
    2. It doesn't create a new violation for the swap partner

    Args:
        violation: The violation to resolve
        phase3_result: Current schedule
        talk_presenter: talk_id -> presenter_id
        presenter_unavailability: presenter_id -> set of unavailable timeslot_ids
        preferences: Participant preferences
        talk_keywords: Optional keywords for coherence scoring
        keyword_weight: Weight for keyword coherence in combined score

    Returns:
        List of SwapCandidate objects, sorted by combined_score (best first)
    """
    candidates = []

    src_talk_id = violation.talk_id
    src_presenter = violation.presenter_id
    src_unavailable = presenter_unavailability.get(src_presenter, set())

    # Iterate over all other positions in the schedule
    for tgt_block_idx, assignment in enumerate(phase3_result.assignments):
        tgt_timeslot_id = assignment.timeslot.get(
            'id', str(assignment.timeslot))

        # Skip if src presenter is also unavailable at target timeslot
        if tgt_timeslot_id in src_unavailable:
            continue

        block = assignment.block

        for tgt_tuple_idx, ntuple in enumerate(block.tuples):
            for tgt_room_idx, tgt_talk_id in enumerate(ntuple):
                # Skip self
                if (tgt_block_idx == violation.block_idx and
                    tgt_tuple_idx == violation.tuple_idx and
                        tgt_room_idx == violation.room_idx):
                    continue

                # Skip dummy talks
                if tgt_talk_id.startswith('DUMMY'):
                    continue

                # Get source timeslot
                src_timeslot_id = phase3_result.assignments[
                    violation.block_idx
                ].timeslot.get('id')

                # Check if swap creates new violation
                creates_violation = check_swap_creates_violation(
                    src_talk_id, tgt_talk_id,
                    src_timeslot_id, tgt_timeslot_id,
                    talk_presenter, presenter_unavailability
                )

                # Compute scores
                candidate = SwapCandidate(
                    src_talk_id=src_talk_id,
                    src_block_idx=violation.block_idx,
                    src_tuple_idx=violation.tuple_idx,
                    src_room_idx=violation.room_idx,
                    tgt_talk_id=tgt_talk_id,
                    tgt_block_idx=tgt_block_idx,
                    tgt_tuple_idx=tgt_tuple_idx,
                    tgt_room_idx=tgt_room_idx,
                    feasible=not creates_violation,
                    missed_attendance_delta=0,
                    keyword_delta=0.0,
                    combined_score=0.0
                )

                if candidate.feasible:
                    # Compute missed attendance delta
                    candidate.missed_attendance_delta = compute_swap_missed_attendance_delta(
                        phase3_result, candidate, preferences
                    )

                    # Compute keyword delta (if available)
                    if talk_keywords:
                        candidate.keyword_delta = compute_swap_keyword_delta(
                            phase3_result, candidate, talk_keywords
                        )

                    # Combined score: minimize missed attendance, maximize coherence
                    # Lower is better (we negate keyword delta since higher is better)
                    candidate.combined_score = (
                        candidate.missed_attendance_delta -
                        keyword_weight * candidate.keyword_delta
                    )

                    candidates.append(candidate)

    # Sort by combined score (lower is better)
    candidates.sort(key=lambda c: c.combined_score)

    return candidates


def check_swap_creates_dummy_violation(
    src_talk_id: str,
    tgt_talk_id: str,
    src_block_idx: int,
    tgt_block_idx: int,
    src_room_idx: int,
    tgt_room_idx: int,
    phase3_result: Phase3Result,
    short_block_threshold: int = 3
) -> bool:
    """
    Check if swapping would create a new dummy violation.

    A dummy violation is created if:
    1. Moving a non-dummy to a short block, or
    2. Moving a dummy to create 2+ dummies in a room-session

    Returns True if swap creates a dummy violation.
    """
    src_is_dummy = src_talk_id.startswith('DUMMY')
    tgt_is_dummy = tgt_talk_id.startswith('DUMMY')

    src_assignment = phase3_result.assignments[src_block_idx]
    tgt_assignment = phase3_result.assignments[tgt_block_idx]

    src_n_slots = len(src_assignment.block.tuples)
    tgt_n_slots = len(tgt_assignment.block.tuples)

    # If swapping a dummy to a different position:
    if src_is_dummy:
        # Check if target block is also short (would still be a violation)
        if tgt_n_slots <= short_block_threshold:
            return True

        # Check if target room-session already has a dummy
        tgt_room_talks = [t[tgt_room_idx] for t in tgt_assignment.block.tuples]
        existing_dummies = sum(
            1 for t in tgt_room_talks if t.startswith('DUMMY'))
        if existing_dummies > 0:
            return True

    # If swapping a non-dummy into the source position:
    if not src_is_dummy and tgt_is_dummy:
        # The non-dummy would go to source position - check if that creates issues
        # (This case shouldn't happen since we're resolving dummy violations)
        pass

    return False


def generate_dummy_swap_candidates(
    violation: Violation,
    phase3_result: Phase3Result,
    talk_presenter: Dict[str, str],
    presenter_unavailability: Dict[str, Set[str]],
    preferences: Dict[str, Set[str]],
    talk_keywords: Optional[Dict[str, Set[str]]] = None,
    keyword_weight: float = 0.1,
    short_block_threshold: int = 3
) -> List[SwapCandidate]:
    """
    Generate swap candidates for a dummy talk violation.

    For dummy violations, we want to swap the dummy with a non-dummy talk
    from a longer block (k > short_block_threshold), preferring swaps that
    don't harm missed attendance much.

    Args:
        violation: The dummy violation to resolve
        phase3_result: Current schedule
        talk_presenter: talk_id -> presenter_id
        presenter_unavailability: presenter_id -> set of unavailable timeslot_ids
        preferences: Participant preferences
        talk_keywords: Optional keywords for coherence scoring
        keyword_weight: Weight for keyword coherence in combined score
        short_block_threshold: Max slots for "short block" (default: 3)

    Returns:
        List of SwapCandidate objects, sorted by combined_score (best first)
    """
    candidates = []

    src_talk_id = violation.talk_id  # This is a DUMMY talk
    src_timeslot_id = phase3_result.assignments[violation.block_idx].timeslot.get(
        'id')

    # Iterate over all positions in longer blocks
    for tgt_block_idx, assignment in enumerate(phase3_result.assignments):
        tgt_timeslot_id = assignment.timeslot.get(
            'id', str(assignment.timeslot))
        block = assignment.block
        n_slots = len(block.tuples)

        # Only consider blocks with more slots than threshold
        if n_slots <= short_block_threshold:
            continue

        for tgt_tuple_idx, ntuple in enumerate(block.tuples):
            for tgt_room_idx, tgt_talk_id in enumerate(ntuple):
                # Skip self
                if (tgt_block_idx == violation.block_idx and
                    tgt_tuple_idx == violation.tuple_idx and
                        tgt_room_idx == violation.room_idx):
                    continue

                # Skip other dummy talks (we want to swap with non-dummies)
                if tgt_talk_id.startswith('DUMMY'):
                    continue

                # Check if moving the target talk to the short block creates presenter violation
                tgt_presenter = talk_presenter.get(tgt_talk_id)
                if tgt_presenter:
                    unavailable = presenter_unavailability.get(
                        tgt_presenter, set())
                    if src_timeslot_id in unavailable:
                        continue  # Can't move this talk to the dummy's position

                # Check if swap creates new dummy violation
                creates_dummy_violation = check_swap_creates_dummy_violation(
                    src_talk_id, tgt_talk_id,
                    violation.block_idx, tgt_block_idx,
                    violation.room_idx, tgt_room_idx,
                    phase3_result, short_block_threshold
                )

                if creates_dummy_violation:
                    continue

                # Check if target room-session would have multiple dummies after swap
                tgt_room_talks = [t[tgt_room_idx] for t in block.tuples]
                existing_dummies_in_target = sum(
                    1 for t in tgt_room_talks if t.startswith('DUMMY'))
                if existing_dummies_in_target > 0:
                    continue  # Would create multiple dummies in same room-session

                # Create candidate
                candidate = SwapCandidate(
                    src_talk_id=src_talk_id,
                    src_block_idx=violation.block_idx,
                    src_tuple_idx=violation.tuple_idx,
                    src_room_idx=violation.room_idx,
                    tgt_talk_id=tgt_talk_id,
                    tgt_block_idx=tgt_block_idx,
                    tgt_tuple_idx=tgt_tuple_idx,
                    tgt_room_idx=tgt_room_idx,
                    feasible=True,
                    missed_attendance_delta=0,
                    keyword_delta=0.0,
                    combined_score=0.0
                )

                # Compute missed attendance delta
                candidate.missed_attendance_delta = compute_swap_missed_attendance_delta(
                    phase3_result, candidate, preferences
                )

                # Compute keyword delta (if available)
                if talk_keywords:
                    candidate.keyword_delta = compute_swap_keyword_delta(
                        phase3_result, candidate, talk_keywords
                    )

                # Combined score: minimize missed attendance, maximize coherence
                candidate.combined_score = (
                    candidate.missed_attendance_delta -
                    keyword_weight * candidate.keyword_delta
                )

                candidates.append(candidate)

    # Sort by combined score (lower is better)
    candidates.sort(key=lambda c: c.combined_score)

    return candidates


# =============================================================================
# SWAP APPLICATION
# =============================================================================

def apply_swap(
    phase3_result: Phase3Result,
    swap: SwapCandidate
) -> Phase3Result:
    """
    Apply a swap to the Phase 3 result.

    This creates a new Phase3Result with the swap applied (doesn't modify original).

    Args:
        phase3_result: Current schedule
        swap: The swap to apply

    Returns:
        New Phase3Result with swap applied
    """
    # Deep copy the assignments
    new_assignments = []

    for idx, assignment in enumerate(phase3_result.assignments):
        # Deep copy the block with new tuples
        new_tuples = [list(t) for t in assignment.block.tuples]

        # Apply swap modifications to this assignment
        if idx == swap.src_block_idx:
            new_tuples[swap.src_tuple_idx][swap.src_room_idx] = swap.tgt_talk_id
        if idx == swap.tgt_block_idx:
            new_tuples[swap.tgt_tuple_idx][swap.tgt_room_idx] = swap.src_talk_id

        # Create new block
        new_block = Block(
            block_id=assignment.block.block_id,
            block_type=assignment.block.block_type,
            tuples=[tuple(t) for t in new_tuples],
            hopping_cost=assignment.block.hopping_cost  # May need recalc
        )

        # Create new assignment
        new_assignment = RoomAssignment(
            block=new_block,
            timeslot=assignment.timeslot,
            room_mapping=assignment.room_mapping.copy(),
            violations=assignment.violations
        )
        new_assignments.append(new_assignment)

    return Phase3Result(
        assignments=new_assignments,
        total_violations=phase3_result.total_violations,  # Will be recalculated
        total_capacity_gap=phase3_result.total_capacity_gap
    )


# =============================================================================
# MAIN OPTIMIZATION FUNCTION
# =============================================================================

def optimize_presenter_violations(
    phase3_result: Phase3Result,
    talk_presenter: Dict[str, str],
    presenter_unavailability: Dict[str, Set[str]],
    preferences: Dict[str, Set[str]],
    talk_keywords: Optional[Dict[str, Set[str]]] = None,
    keyword_weight: float = 0.1,
    max_iterations: int = 100,
    check_dummy_violations: bool = True,
    short_block_threshold: int = 3,
    verbose: bool = True
) -> SwapResult:
    """
    Optimize the schedule by resolving presenter and dummy violations.

    Uses a greedy local search approach:
    1. Find all violations (presenter unavailability + dummy placement)
    2. For each violation, find the best feasible swap
    3. Apply the best swap overall
    4. Repeat until no violations remain or no swaps possible

    Violation types handled:
    - PRESENTER_UNAVAILABLE: Presenter scheduled when unavailable
    - DUMMY_IN_SHORT_BLOCK: Dummy talk in a block with k <= threshold slots
    - MULTIPLE_DUMMIES_IN_SESSION: 2+ dummy talks in the same room-session

    Args:
        phase3_result: Input Phase 3 result
        talk_presenter: talk_id -> presenter_id
        presenter_unavailability: presenter_id -> set of unavailable timeslot_ids
        preferences: participant_id -> set of preferred talk_ids
        talk_keywords: Optional talk_id -> set of keywords for coherence
        keyword_weight: Weight for keyword coherence (0 = ignore, 1 = equal to missed attendance)
        max_iterations: Maximum swap iterations
        check_dummy_violations: Whether to check and resolve dummy violations
        short_block_threshold: Max slots for "short block" dummy violation (default: 3)
        verbose: Print progress

    Returns:
        SwapResult with optimized schedule and statistics
    """
    if verbose:
        print("\n" + "=" * 70)
        print("SWAP OPTIMIZATION: RESOLVING SCHEDULE VIOLATIONS")
        print("=" * 70)

    current_result = phase3_result
    swaps_applied = []

    # Track initial violations for summary
    initial_presenter_violations = detect_violations(
        phase3_result, talk_presenter, presenter_unavailability
    )
    initial_dummy_violations = (
        detect_dummy_violations(phase3_result, short_block_threshold)
        if check_dummy_violations else []
    )

    for iteration in range(max_iterations):
        # Detect current violations
        presenter_violations = detect_violations(
            current_result, talk_presenter, presenter_unavailability
        )
        dummy_violations = (
            detect_dummy_violations(current_result, short_block_threshold)
            if check_dummy_violations else []
        )

        all_violations = presenter_violations + dummy_violations

        if not all_violations:
            if verbose:
                print(f"\n  ✓ All violations resolved after {iteration} swaps")
            break

        if verbose and iteration == 0:
            if presenter_violations:
                print(f"\n  Presenter violations: {len(presenter_violations)}")
                for v in presenter_violations[:3]:
                    print(
                        f"    - {v.talk_id} ({v.presenter_id}) @ {v.timeslot_id}")
                if len(presenter_violations) > 3:
                    print(f"    ... and {len(presenter_violations) - 3} more")

            if dummy_violations:
                print(f"\n  Dummy violations: {len(dummy_violations)}")
                for v in dummy_violations[:3]:
                    vtype = "short block" if v.violation_type == ViolationType.DUMMY_IN_SHORT_BLOCK else "multi-dummy"
                    print(f"    - {v.talk_id} @ {v.timeslot_id} ({vtype})")
                if len(dummy_violations) > 3:
                    print(f"    ... and {len(dummy_violations) - 3} more")

        # Find best swap across all violations
        best_swap = None
        best_score = float('inf')

        # Handle presenter violations
        for violation in presenter_violations:
            candidates = generate_swap_candidates(
                violation, current_result,
                talk_presenter, presenter_unavailability,
                preferences, talk_keywords, keyword_weight
            )

            if candidates and candidates[0].combined_score < best_score:
                best_swap = candidates[0]
                best_score = candidates[0].combined_score

        # Handle dummy violations
        for violation in dummy_violations:
            candidates = generate_dummy_swap_candidates(
                violation, current_result,
                talk_presenter, presenter_unavailability,
                preferences, talk_keywords, keyword_weight,
                short_block_threshold
            )

            if candidates and candidates[0].combined_score < best_score:
                best_swap = candidates[0]
                best_score = candidates[0].combined_score

        if best_swap is None:
            if verbose:
                print(
                    f"\n  ⚠ No feasible swaps found for remaining {len(all_violations)} violations")
                print(
                    f"    ({len(presenter_violations)} presenter, {len(dummy_violations)} dummy)")
            break

        # Apply the best swap
        current_result = apply_swap(current_result, best_swap)
        swaps_applied.append(best_swap)

        if verbose:
            swap_type = "DUMMY" if best_swap.src_talk_id.startswith(
                'DUMMY') else "PRES"
            print(f"  Swap {iteration + 1} [{swap_type}]: {best_swap.src_talk_id} <-> {best_swap.tgt_talk_id} "
                  f"(Δmissed={best_swap.missed_attendance_delta:+d}, "
                  f"Δkeyword={best_swap.keyword_delta:+.2f})")

    # Final violation counts
    final_presenter_violations = detect_violations(
        current_result, talk_presenter, presenter_unavailability
    )
    final_dummy_violations = (
        detect_dummy_violations(current_result, short_block_threshold)
        if check_dummy_violations else []
    )
    final_all_violations = final_presenter_violations + final_dummy_violations

    # Update total_violations in result
    current_result = Phase3Result(
        assignments=current_result.assignments,
        # Only count presenter violations here
        total_violations=len(final_presenter_violations),
        total_capacity_gap=current_result.total_capacity_gap
    )

    initial_all = len(initial_presenter_violations) + \
        len(initial_dummy_violations)
    final_all = len(final_all_violations)

    if verbose:
        print(f"\n  Summary:")
        print(
            f"    Initial presenter violations: {len(initial_presenter_violations)}")
        print(
            f"    Final presenter violations:   {len(final_presenter_violations)}")
        if check_dummy_violations:
            print(
                f"    Initial dummy violations:     {len(initial_dummy_violations)}")
            print(
                f"    Final dummy violations:       {len(final_dummy_violations)}")
        print(f"    Swaps applied:                {len(swaps_applied)}")

    return SwapResult(
        phase3_result=current_result,
        violations_before=initial_all,
        violations_resolved=initial_all - final_all,
        violations_remaining=final_all,
        swaps_applied=swaps_applied,
        iterations=len(swaps_applied)
    )


# =============================================================================
# CONVENIENCE FUNCTION FOR INTEGRATION
# =============================================================================

def post_process_schedule(
    phase3_result: Phase3Result,
    data,
    instance,
    talk_keywords: Optional[Dict[str, Set[str]]] = None,
    keyword_weight: float = 0.1,
    verbose: bool = True
) -> Phase3Result:
    """
    Convenience wrapper to run swap optimization using data and instance objects.

    This is the main entry point for integration with the pipeline.

    Args:
        phase3_result: Input from Phase 3
        data: Data object with talks, preferences, etc.
        instance: Instance object with preferences and presenter info
        talk_keywords: Optional keyword mapping
        keyword_weight: Weight for keyword coherence
        verbose: Print progress

    Returns:
        Optimized Phase3Result
    """
    # Build talk_presenter mapping
    talk_presenter = {}
    if hasattr(data, 'talks') and data.talks is not None:
        for _, row in data.talks.iterrows():
            talk_id = str(row.get('talk_id', ''))
            presenter_id = str(row.get('presenter_id', ''))
            if talk_id and presenter_id:
                talk_presenter[talk_id] = presenter_id

    # Build presenter_unavailability mapping
    presenter_unavailability = {}
    if hasattr(instance, 'presenter_unavailability'):
        presenter_unavailability = instance.presenter_unavailability
    elif hasattr(data, 'presenter_unavailability'):
        presenter_unavailability = data.presenter_unavailability

    # Get preferences
    preferences = instance.preferences if hasattr(
        instance, 'preferences') else {}

    # Run optimization
    result = optimize_presenter_violations(
        phase3_result=phase3_result,
        talk_presenter=talk_presenter,
        presenter_unavailability=presenter_unavailability,
        preferences=preferences,
        talk_keywords=talk_keywords,
        keyword_weight=keyword_weight,
        verbose=verbose
    )

    return result.phase3_result
