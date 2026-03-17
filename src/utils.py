"""
Utility functions for conference scheduling.
"""

from typing import Dict, List, Set, Tuple
import time
from functools import wraps


def timer(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


def compute_attendance_stats(
    schedule_talks: List[Tuple[str, ...]],  # List of n-tuples (parallel talks)
    preferences: Dict[str, Set[str]]
) -> Dict:
    """
    Compute attendance statistics.

    Returns dict with:
    - total_preferences: Total number of preferences across all participants
    - total_attendable: Number of preferences that can be attended (no conflicts)
    - total_missed: Number of missed preferences (due to conflicts)
    - attendance_rate: Fraction of preferences that can be attended
    """
    total_prefs = sum(len(p) for p in preferences.values())

    missed = 0
    for ntuple in schedule_talks:
        for prefs in preferences.values():
            preferred_in_tuple = sum(1 for t in ntuple if t in prefs)
            if preferred_in_tuple > 1:
                missed += preferred_in_tuple - 1

    attendable = total_prefs - missed

    return {
        "total_preferences": total_prefs,
        "total_attendable": attendable,
        "total_missed": missed,
        "attendance_rate": attendable / total_prefs if total_prefs > 0 else 1.0,
    }


def compute_hopping_stats(
    blocks: List,  # List of Block objects
    preferences: Dict[str, Set[str]]
) -> Dict:
    """
    Compute session hopping statistics.

    Returns dict with:
    - total_hops: Total hops across all participants
    - avg_hops_per_participant: Average hops per participant
    - max_hops: Maximum hops by any single participant
    """
    from .phase2 import compute_hopping_number

    hops_per_participant = {}

    for p_id, prefs in preferences.items():
        total = 0
        for block in blocks:
            total += compute_hopping_number(block, prefs)
        hops_per_participant[p_id] = total

    all_hops = list(hops_per_participant.values())

    return {
        "total_hops": sum(all_hops),
        "avg_hops_per_participant": sum(all_hops) / len(all_hops) if all_hops else 0,
        "max_hops": max(all_hops) if all_hops else 0,
        "hops_by_participant": hops_per_participant,
    }


def format_time_delta(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
