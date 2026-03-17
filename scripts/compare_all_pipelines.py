#!/usr/bin/env python
"""
Compare All Scheduling Pipelines

Runs all three pipelines (traditional, heuristic, matching) on the same data,
generates schedules, and computes quality metrics for comparison.

Usage:
    python scripts/compare_all_pipelines.py
    python scripts/compare_all_pipelines.py --output-dir output/comparison
    python scripts/compare_all_pipelines.py --verbose
    python scripts/compare_all_pipelines.py --skip-traditional  # Skip if no Gurobi
    python scripts/compare_all_pipelines.py --skip-matching     # Skip matching pipelines

Output files (per pipeline):
    - schedule_{pipeline}.csv         - CSV format
    - schedule_{pipeline}.md          - Markdown format (human-readable)
    - schedule_{pipeline}.json        - JSON format (structured)
    - metrics_{pipeline}.json         - Quality metrics

Summary files:
    - comparison_summary.md           - Side-by-side comparison
    - comparison_metrics.json         - All metrics in one file

Dummy Talk Handling:
    If total_slots > total_talks (e.g., 123 slots but 120 talks), the script 
    automatically creates dummy talks (DUMMY_001, DUMMY_002, etc.) to fill
    empty slots. This allows flexible slot configurations without requiring
    an exact talk count match.
    
    To add more slots, modify the sessions.csv in your data directory:
    - Increase 'number of rooms' for a session, or
    - Increase 'number of talks per room' for a session, or  
    - Add a new session row
"""

import pandas as pd
import sys
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# SPECIAL SESSION CONFIGURATION
# =============================================================================
# Special-session talks are pre-specified sessions that must be scheduled in
# specific blocks. They are EXCLUDED from Phase 1-3 optimization and added in
# Phase 4 via fixed_block_sessions.

SPECIAL_SESSION_CONFIG = {
    # Example: pre-assign specific talks to specific blocks.
    # Uncomment and customize for your conference:
    #
    # "SpecialSession_1": {
    #     "talks": ["T039", "T046", "T051", "T057"],
    #     "block": "FA",
    # },
    # "SpecialSession_2": {
    #     "talks": ["T058", "T072", "T122", "T124"],
    #     "block": "FB",
    # },
}

# All special-session talk IDs for quick lookup
SPECIAL_SESSION_TALK_IDS: Set[str] = set()
for cfg in SPECIAL_SESSION_CONFIG.values():
    SPECIAL_SESSION_TALK_IDS.update(cfg["talks"])


def adjust_block_types_for_special_sessions(block_types: Dict[str, Dict], timeslots: List[Dict],
                                            special_blocks: List[str] = None, verbose: bool = False) -> Dict[str, Dict]:
    """
    Adjust block types by removing one room from each block that hosts a special session.

    Each special-session block has one room reserved for the pre-specified session,
    so the optimizer sees one fewer room for that timeslot.

    Args:
        block_types: Original block types from sessions.csv
        timeslots: Timeslot definitions
        special_blocks: Block IDs where special sessions are placed (default: ['FA', 'FB'])
        verbose: Print debug info

    Returns:
        New block_types dict with adjusted counts for reduced blocks
    """
    if special_blocks is None:
        special_blocks = ['FA', 'FB']

    # Build timeslot type lookup
    timeslot_types = {ts['id']: ts['type_id'] for ts in timeslots}

    # Count how many timeslots use each type
    type_counts = {}
    for ts in timeslots:
        t = ts['type_id']
        type_counts[t] = type_counts.get(t, 0) + 1

    # For each special-session block, reduce rooms by 1
    # This may change the block type (e.g., 5R4T -> 4R4T)
    reduced_types = {}
    for block_id in special_blocks:
        if block_id not in timeslot_types:
            continue
        old_type = timeslot_types[block_id]
        # Parse "nRkT" format
        n, k = int(old_type[0]), int(old_type[2])
        new_n = n - 1
        new_type = f"{new_n}R{k}T"
        reduced_types[block_id] = (old_type, new_type)

        if verbose:
            print(
                f"  Special session adjustment: {block_id} from {old_type} to {new_type}")

    # Rebuild block_types with adjusted counts
    new_block_types = {}

    for type_id, info in block_types.items():
        # How many timeslots originally use this type?
        original_count = type_counts.get(type_id, 0)

        # How many are being reduced from this type?
        reduced_from = sum(
            1 for (old, new) in reduced_types.values() if old == type_id)

        remaining = original_count - reduced_from
        if remaining > 0:
            new_block_types[type_id] = {
                "n": info["n"], "k": info["k"], "count": remaining}

    # Add the new reduced types
    for block_id, (old_type, new_type) in reduced_types.items():
        n, k = int(new_type[0]), int(new_type[2])
        if new_type in new_block_types:
            new_block_types[new_type]["count"] += 1
        else:
            new_block_types[new_type] = {"n": n, "k": k, "count": 1}

    return new_block_types


# =============================================================================
# ROOM CONFIGURATION
# =============================================================================
# Physical rooms with capacities (ORBEL 2026 - KU Leuven campus)
# Used for Phase 4 audience-based room assignment

ROOM_CONFIG = [
    {"id": "STUK", "name": "STUK 02.C004", "capacity": 250},
    {"id": "02.28", "name": "HOGC 02.28", "capacity": 126},
    {"id": "01.85", "name": "HOGM 01.85", "capacity": 72},
    {"id": "00.85", "name": "HOGM 00.85", "capacity": 72},
    {"id": "02.10", "name": "HOGS 02.10", "capacity": 40},
]

# Build helper dicts
ROOM_CAPACITIES = {r["id"]: r["capacity"] for r in ROOM_CONFIG}
ROOM_NAMES = {r["id"]: r["name"] for r in ROOM_CONFIG}


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PipelineResult:
    """Results from running a single pipeline."""
    name: str
    success: bool
    error_message: Optional[str] = None

    # Timing
    total_time: float = 0.0
    phase1_time: float = 0.0
    phase2_time: float = 0.0
    phase3_time: float = 0.0

    # Quality from pipeline
    missed_attendance_pipeline: int = 0
    room_switches_pipeline: int = 0
    presenter_violations: int = 0

    # Swap optimization tracking
    violations_before_swap: int = 0
    violations_after_swap: int = 0
    swaps_performed: int = 0

    # Quality from evaluator (post-hoc)
    missed_attendance_evaluated: int = 0
    session_hops_evaluated: int = 0
    incoherent_sessions: int = 0

    # Output files
    csv_file: Optional[str] = None
    md_file: Optional[str] = None
    json_file: Optional[str] = None
    metrics_file: Optional[str] = None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare all scheduling pipelines on 2026 data"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="output/comparison",
        help="Output directory for comparison results"
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default="examples/orbel2026",
        help="Input data directory"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--skip-traditional",
        action="store_true",
        help="Skip traditional pipeline (useful if Gurobi unavailable)"
    )
    parser.add_argument(
        "--skip-matching",
        action="store_true",
        help="Skip matching pipeline"
    )
    parser.add_argument(
        "--preferences",
        type=str,
        default="examples/orbel2026/preferences.csv",
        help="Path to preferences CSV for evaluation"
    )
    parser.add_argument(
        "--talks-with-keywords",
        type=str,
        default="examples/orbel2026/talks.csv",
        help="Path to talks CSV with keywords for evaluation"
    )
    return parser.parse_args()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data_and_instance(data_dir: str, verbose: bool = False):
    """Load data and build instance."""
    from src.data_loader import load_from_csv_dir
    from src.instance import build_instance

    if verbose:
        print("[Loading Data]")

    data = load_from_csv_dir(data_dir=data_dir, verbose=verbose)

    # Validate
    errors = data.validate() if hasattr(data, 'validate') else []
    if errors and verbose:
        print("  Validation warnings:")
        for err in errors[:5]:
            print(f"    - {err}")

    instance = build_instance(data)

    if verbose:
        print(f"  ✓ Loaded {len(instance.talks)} talks")
        print(
            f"  ✓ {sum(len(p) for p in instance.preferences.values())} preferences from {len(instance.participants)} participants")

    return data, instance


def add_dummy_talks_if_needed(
    data,
    instance,
    verbose: bool = False
) -> Tuple[Any, Any, List[str]]:
    """
    Add dummy talks if slots > talks (e.g., 123 slots but only 120 talks).

    Returns:
        Tuple of (modified_data, modified_instance, list_of_dummy_talk_ids)
    """
    from src.instance import build_instance
    import pandas as pd

    total_slots = sum(
        bt['n'] * bt['k'] * bt['count']
        for bt in data.block_types.values()
    )
    n_talks = len(data.talks)
    diff = total_slots - n_talks

    dummy_ids = []

    if diff <= 0:
        # No dummies needed (exact fit or too many talks)
        return data, instance, dummy_ids

    if verbose:
        print(
            f"  Adding {diff} dummy talk(s): {total_slots} slots > {n_talks} talks")

    # Create dummy talks
    for i in range(diff):
        dummy_id = f"DUMMY_{i+1:03d}"
        dummy_ids.append(dummy_id)

        # Add to data.talks DataFrame
        dummy_row = {
            'talk_id': dummy_id,
            'title': f'[Dummy Talk {i+1}]',
            'presenter_id': f'P_DUMMY_{i+1:03d}',
            'presenter_name': 'TBD',
            'author_names': 'TBD',
            'keywords': '',
            'abstract': '',
        }
        data.talks = pd.concat(
            [data.talks, pd.DataFrame([dummy_row])],
            ignore_index=True
        )

    if verbose:
        print(f"  Total talks (including dummies): {len(data.talks)}")

    # Rebuild instance with dummy talks
    instance = build_instance(data)

    return data, instance, dummy_ids


def load_data_excluding_special_sessions(data_dir: str, verbose: bool = False):
    """
    Load data with special-session talks excluded.

    Reads block structure from sessions.csv and dynamically adjusts for each
    special session:
    - Removes special-session talks from the talk list
    - Reduces the host block by one room (the special session occupies that room)
    - Recalculates block types based on the adjusted timeslots
    """
    from src.data_loader import load_from_csv_dir
    from src.instance import build_instance

    if verbose:
        print("[Loading Data (excluding special sessions)]")

    data = load_from_csv_dir(data_dir=data_dir, verbose=verbose)

    # Count original talks
    original_count = len(data.talks)

    # Exclude special-session talks from data.talks DataFrame
    data.talks = data.talks[~data.talks['talk_id'].isin(
        SPECIAL_SESSION_TALK_IDS)].copy()
    excluded_count = original_count - len(data.talks)

    if verbose:
        print(f"  ✓ Original talks: {original_count}")
        print(f"  ✓ Excluded special sessions: {excluded_count}")
        print(f"  ✓ Remaining talks: {len(data.talks)}")

    # Adjust block types: reduce rooms by 1 for blocks hosting special sessions
    special_blocks = [cfg['block'] for cfg in SPECIAL_SESSION_CONFIG.values()]
    original_block_types = data.block_types.copy()

    data.block_types = adjust_block_types_for_special_sessions(
        original_block_types, data.timeslots, special_blocks, verbose=verbose
    )

    # Also adjust timeslot type_ids and room lists to match reduced blocks
    for ts in data.timeslots:
        if ts['id'] in special_blocks:
            old_type = ts['type_id']
            n, k = int(old_type[0]), int(old_type[2])
            new_n = n - 1
            ts['type_id'] = f"{new_n}R{k}T"
            ts['rooms'] = ts['rooms'][:new_n]  # Reduce room count

    if verbose:
        print(f"  ✓ Block types: {data.block_types}")
        total_slots = sum(bt['n'] * bt['k'] * bt['count']
                          for bt in data.block_types.values())
        print(f"  ✓ Total slots: {total_slots}")
        adjusted = [ts['id']
                    for ts in data.timeslots if ts['id'] in special_blocks]
        print(f"  ✓ Adjusted timeslots: {adjusted}")

    # Build instance
    instance = build_instance(data)

    if verbose:
        print(f"  ✓ Instance has {len(instance.talks)} talks")

    return data, instance


# =============================================================================
# PIPELINE RUNNERS
# =============================================================================

def run_traditional_pipeline(instance, data, verbose: bool = False) -> Dict[str, Any]:
    """Run traditional (Phase 1→2→3) pipeline with special sessions excluded."""
    from src.pipelines.traditional import run_traditional_pipeline, PipelineConfig

    # NO fixed_sequences — special sessions are fully excluded and added in Phase 4
    print(f"\n📋 Configuration:")
    print(f"   • Method: Column Generation → Greedy Partition → MILP Assignment")
    print(f"   • Special sessions: excluded from Phase 1-3, added in Phase 4")
    print(f"   • Talks in Phase 1-3: {len(instance.talks)}")

    config = PipelineConfig(
        phase1_method="column_generation",
        phase1_time_limit=300.0,
        phase1_pricing_strategy="auto",
        phase2_partition_strategy="greedy",
        phase2_ordering_strategy="enumerate",
        phase2_local_search_iterations=2000,
        phase3_method="milp",
        phase3_time_limit=60.0,
        verbose=verbose,
        fixed_sequences=None  # special sessions handled via Phase 4
    )

    return run_traditional_pipeline(config, instance, data)


def run_heuristic_pipeline(instance, data, verbose: bool = False) -> Dict[str, Any]:
    """Run heuristic (greedy) pipeline with special sessions excluded."""
    from src.pipelines.heuristic import run_heuristic_pipeline, PipelineConfig

    # NO fixed_sequences — special sessions are fully excluded and added in Phase 4
    print(f"\n📋 Configuration:")
    print(f"   • Method: Greedy Tuples → Matching Partition → Hungarian Assignment")
    print(f"   • Special sessions: excluded from Phase 1-3, added in Phase 4")
    print(f"   • Talks in Phase 1-3: {len(instance.talks)}")

    config = PipelineConfig(
        phase1_time_limit=60.0,
        phase2_partition_strategy="matching",
        phase2_ordering_strategy="enumerate",
        phase2_local_search_iterations=100,
        verbose=verbose,
        fixed_sequences=None  # special sessions handled via Phase 4
    )

    return run_heuristic_pipeline(config, instance, data)


def run_matching_pipeline_wrapper(instance, data, verbose: bool = False) -> Dict[str, Any]:
    """Run matching (bottom-up) pipeline without keyword constraints."""
    from src.matching_pipeline import run_matching_pipeline
    from src.instance import build_instance

    n_talks = len(instance.talks)
    print(f"\n📋 Configuration:")
    print(f"   • Method: Pair Matching → Block Formation → Tuple Selection → Ordering")
    print(f"   • Special sessions: excluded from optimization, added in Phase 4")
    print(f"   • Talks in optimization: {n_talks}")

    pipeline_result, phase3_result = run_matching_pipeline(
        instance,
        time_limit=300.0,
        verbose=verbose,
        run_phase3=True,
        phase3_method="milp"
    )

    return {
        'phase1_time': pipeline_result.phase_a_time + pipeline_result.phase_b_time,
        'phase2_time': pipeline_result.phase_c_time + pipeline_result.phase_d_time,
        'phase3_time': 0,
        'phase1_objective': pipeline_result.phase_c_cost,
        'phase2_hopping': -pipeline_result.phase_d_benefit,
        'phase3_result': phase3_result,
        'pipeline_result': pipeline_result,
        'blocks': pipeline_result.ordered_blocks
    }


def run_matching_kw_pipeline_wrapper(instance, data, verbose: bool = False) -> Dict[str, Any]:
    """Run matching (bottom-up) pipeline WITH keyword constraints."""
    from src.matching_pipeline_constrained import (
        run_constrained_matching_pipeline,
        load_talk_metadata,
        build_metadata_from_instance,
        MatchingConstraints
    )
    from src.instance import build_instance

    n_talks = len(instance.talks)
    print(f"\n📋 Configuration:")
    print(f"   • Method: Constrained Pair Matching → Block Formation → Tuple Selection")
    print(f"   • Keyword Constraints: Pairs require common keywords")
    print(f"   • Special sessions: excluded from optimization, added in Phase 4")
    print(f"   • Talks in optimization: {n_talks}")

    # Build talk_titles mapping BEFORE any modifications
    talk_titles = {}
    talks_df = data.talks if hasattr(data.talks, 'iterrows') else None
    if talks_df is not None:
        for _, row in talks_df.iterrows():
            talk_id = str(row.get('talk_id', ''))
            title = str(row.get('title', ''))
            if talk_id and title:
                talk_titles[talk_id] = title

    # Load metadata with keywords - returns tuple: (metadata, title_to_keywords, special_groups, talkid_to_keywords)
    _, title_to_keywords, special_groups, talkid_to_keywords = load_talk_metadata()

    if verbose:
        print(f"  Built talk_titles for {len(talk_titles)} talks")
        if talkid_to_keywords:
            print(
                f"  Loaded {len(talkid_to_keywords)} talk_id->keyword mappings")

    # Build metadata for each talk in instance
    metadata = build_metadata_from_instance(
        instance, title_to_keywords, special_groups, talk_titles, talkid_to_keywords
    )

    # Check how many talks have keywords
    n_with_kw = sum(1 for m in metadata.values() if m.keywords)
    if verbose:
        print(f"  Talks with keywords: {n_with_kw}/{len(metadata)}")

    # Create constraints with keyword requirement
    constraints = MatchingConstraints(
        require_same_group=True,
        require_common_keyword=True,
        max_keyword_violations=None  # Hard constraint
    )

    pipeline_result, phase3_result = run_constrained_matching_pipeline(
        instance,
        metadata=metadata,
        constraints=constraints,
        time_limit=300.0,
        verbose=verbose,
        run_phase3=True,
        phase3_method="milp"
    )

    return {
        'phase1_time': pipeline_result.phase_a_time + pipeline_result.phase_b_time,
        'phase2_time': pipeline_result.phase_c_time + pipeline_result.phase_d_time,
        'phase3_time': 0,
        'phase1_objective': pipeline_result.phase_c_cost,
        'phase2_hopping': -pipeline_result.phase_d_benefit,
        'phase3_result': phase3_result,
        'pipeline_result': pipeline_result,
        'blocks': pipeline_result.ordered_blocks
    }


# =============================================================================
# SCHEDULE EXPORT
# =============================================================================

def add_special_sessions_to_schedule(rows: List[Dict], data, talk_info: Dict) -> None:
    """
    Add special-session talks to the schedule rows.

    Each special session occupies the last room of its target block (the room
    that was excluded from the optimizer's room count).
    """
    # Check what room naming convention is used in the rest of the schedule
    existing_rooms = set(r.get('Room', '') for r in rows)
    uses_letter_names = any('Room A' in str(r) or r ==
                            'Room A' for r in existing_rooms)

    for name, cfg in SPECIAL_SESSION_CONFIG.items():
        block = cfg["block"]
        talks = cfg["talks"]

        # Determine room name
        if block == "FA":
            room_name = "Room D" if uses_letter_names else "R3"
        else:  # FB
            room_name = "Room E" if uses_letter_names else "R4"

        for slot_idx, talk_id in enumerate(talks):
            info = talk_info.get(talk_id, {})
            # Extract paper_id from talk_id (T001 -> 1)
            paper_id = int(talk_id[1:]) if talk_id.startswith('T') else 0

            rows.append({
                'Session_ID': block,
                'Block_ID': f"FIXED_{block}",  # Consistent with Phase 4 naming
                'Room': room_name,             # For evaluator compatibility
                'Room_ID': room_name,
                'Room_Name': room_name,
                'Slot': slot_idx + 1,
                'Talk_ID': talk_id,
                'Paper_ID': paper_id,
                'Title': info.get('title', ''),
                'Primary_Contact_Author': info.get('presenter_name', ''),
                'Author_Names': info.get('author_names', ''),
                'Session_Total_Likes': 0,
                'Session_Unique_Attendees': 0,
                'Is_Fixed': True
            })


# =============================================================================
# SWAP OPTIMIZATION (POST PHASE 3, PRE PHASE 4)
# =============================================================================

def run_swap_optimization(
    results: Dict,
    data,
    data_full,
    talk_keywords: Optional[Dict[str, Set[str]]] = None,
    keyword_weight: float = 0.1,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run swap optimization to resolve presenter availability violations.

    This phase runs after Phase 3 (block scheduling) and before Phase 4 (room assignment).
    It swaps talks to eliminate violations while optimizing for missed attendance.

    Args:
        results: Pipeline results dict with 'phase3_result'
        data: Data with special sessions excluded
        data_full: Full data including special sessions
        talk_keywords: Optional talk_id -> set of keywords for coherence scoring
        keyword_weight: Weight for keyword coherence vs missed attendance (0-1)
        verbose: Print progress

    Returns:
        Modified results dict with optimized phase3_result
    """
    from src.swap_optimization import optimize_presenter_violations

    phase3_result = results.get('phase3_result')
    if not phase3_result:
        return results

    # Build talk_presenter mapping
    talk_presenter = {}
    if hasattr(data_full, 'talks') and data_full.talks is not None:
        for _, row in data_full.talks.iterrows():
            talk_id = str(row.get('talk_id', ''))
            presenter_id = str(row.get('presenter_id', ''))
            if talk_id and presenter_id:
                talk_presenter[talk_id] = presenter_id

    # Build presenter_unavailability mapping
    presenter_unavailability = {}

    # Try to load from constraints file
    constraints_file = Path(
        data_full.data_dir) / "constraints for the schedule.xlsx" if hasattr(data_full, 'data_dir') else None
    if constraints_file is None:
        # No constraints file available
        constraints_file = None

    if constraints_file and constraints_file.exists():
        try:
            constraints_df = pd.read_excel(constraints_file)
            # Build name -> presenter_id mapping
            name_to_pid = {}
            for _, row in data_full.talks.iterrows():
                pid = str(row.get('presenter_id', ''))
                name = str(row.get('presenter_name', ''))
                if pid and name and name.lower() != 'nan':
                    name_to_pid[name] = pid
                    # Also try last name
                    last_name = name.split()[-1] if name else ""
                    if last_name:
                        name_to_pid[last_name] = pid

            for idx, row in constraints_df.iterrows():
                participant = row.get('Unnamed: 1')
                sessions_to_avoid = row.get('Unnamed: 5')

                if pd.notna(participant) and pd.notna(sessions_to_avoid):
                    participant = str(participant).strip()
                    sessions_str = str(sessions_to_avoid).strip()
                    if sessions_str.lower() not in ['nan', 'sessions to avoid', 'all']:
                        sessions = set(s.strip()
                                       for s in sessions_str.split(',') if s.strip())
                        if sessions and participant not in ['participant', 'NaN']:
                            # Find presenter_id from name
                            pid = name_to_pid.get(participant)
                            if not pid:
                                # Try last name match
                                last_name = participant.split(
                                )[-1] if participant else ""
                                pid = name_to_pid.get(last_name)
                            if pid:
                                presenter_unavailability[pid] = sessions
        except Exception as e:
            if verbose:
                print(
                    f"  Warning: Could not load constraints for swap optimization: {e}")

    # Build preferences from data_full
    preferences: Dict[str, Set[str]] = {}
    if hasattr(data_full, 'preferences') and data_full.preferences is not None:
        for _, row in data_full.preferences.iterrows():
            pid = str(row.get('participant_id', ''))
            tid = str(row.get('talk_id', ''))
            if pid and tid:
                if pid not in preferences:
                    preferences[pid] = set()
                preferences[pid].add(tid)

    # Run swap optimization
    try:
        swap_result = optimize_presenter_violations(
            phase3_result=phase3_result,
            talk_presenter=talk_presenter,
            presenter_unavailability=presenter_unavailability,
            preferences=preferences,
            talk_keywords=talk_keywords,
            keyword_weight=keyword_weight,
            check_dummy_violations=True,  # Also handle dummy placement
            short_block_threshold=3,      # Dummies in blocks with k<=3 slots are violations
            verbose=verbose
        )

        # Update results with optimized phase3_result
        results = results.copy()
        results['phase3_result'] = swap_result.phase3_result
        results['swap_violations_before'] = swap_result.violations_before
        results['swap_violations_resolved'] = swap_result.violations_resolved
        results['swap_violations_remaining'] = swap_result.violations_remaining
        results['swap_count'] = swap_result.iterations

    except Exception as e:
        if verbose:
            print(f"  Warning: Swap optimization failed: {e}")
            import traceback
            traceback.print_exc()

    return results


def load_talk_keywords_for_swap(data_dir: str, data_full, verbose: bool = False) -> Dict[str, Set[str]]:
    """
    Load talk keywords for swap optimization coherence scoring.

    Args:
        data_dir: Data directory path
        data_full: Full data object
        verbose: Print progress

    Returns:
        talk_id -> set of keywords
    """
    talk_keywords: Dict[str, Set[str]] = {}

    # Try to load from talks_for_algorithm.csv first (has talk_id -> keywords directly)
    algo_file = Path(data_dir) / "talks_for_algorithm.csv"
    if algo_file.exists():
        try:
            algo_df = pd.read_csv(algo_file)
            for _, row in algo_df.iterrows():
                talk_id = str(row.get('talk_id', ''))
                kw_str = str(row.get('master_keywords', ''))
                if talk_id and kw_str and kw_str.lower() != 'nan':
                    keywords = set(kw.strip()
                                   for kw in kw_str.split(';') if kw.strip())
                    talk_keywords[talk_id] = keywords
            if verbose:
                print(
                    f"  Loaded keywords for {len(talk_keywords)} talks from talks_for_algorithm.csv")
            return talk_keywords
        except Exception as e:
            if verbose:
                print(
                    f"  Warning: Could not load from talks_for_algorithm.csv: {e}")

    # Fallback: load from talks_with_abstracts file and match by title
    talks_kw_file = Path(data_dir) / \
        "talks_with_abstracts_w_master_keywords.csv"
    if talks_kw_file.exists():
        try:
            talks_kw_df = pd.read_csv(talks_kw_file)
            title_to_keywords = {}
            for _, row in talks_kw_df.iterrows():
                title = str(row.get('title', '')).strip().lower()
                kw_str = str(row.get('master_keywords', ''))
                if title and kw_str and kw_str.lower() != 'nan':
                    keywords = set(kw.strip()
                                   for kw in kw_str.split(';') if kw.strip())
                    title_to_keywords[title] = keywords

            # Match to talk_ids via data_full.talks
            for _, row in data_full.talks.iterrows():
                talk_id = str(row.get('talk_id', ''))
                title = str(row.get('title', '')).strip().lower()
                if talk_id and title in title_to_keywords:
                    talk_keywords[talk_id] = title_to_keywords[title]

            if verbose:
                print(
                    f"  Loaded keywords for {len(talk_keywords)} talks via title matching")
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not load keywords: {e}")

    return talk_keywords


# =============================================================================
# PHASE 4 INTEGRATION
# =============================================================================

def run_phase4(results: Dict, data_full, verbose: bool = False) -> Optional[Any]:
    """
    Run Phase 4 to get proper room assignment based on audience size.

    Args:
        results: Pipeline results dict with 'phase3_result'
        data_full: Full data including special-session talks
        verbose: Print progress

    Returns:
        Phase4Result with finalized schedule, or None if failed
    """
    from src.phase4 import (
        Phase4Input, Phase4Result, FixedBlockSession, solve_phase4
    )

    phase3_result = results.get('phase3_result')
    if not phase3_result:
        return None

    # Build preferences dict from data_full
    preferences: Dict[str, Set[str]] = {}
    if hasattr(data_full, 'preferences') and data_full.preferences is not None:
        for _, row in data_full.preferences.iterrows():
            pid = str(row.get('participant_id', ''))
            tid = str(row.get('talk_id', ''))
            if pid and tid:
                if pid not in preferences:
                    preferences[pid] = set()
                preferences[pid].add(tid)

    # Build talk metadata from data_full
    talk_metadata: Dict[str, Dict] = {}
    for _, row in data_full.talks.iterrows():
        talk_id = str(row.get('talk_id', ''))
        # Use actual presenter_name and author_names if available
        presenter_name = row.get('presenter_name', row.get('presenter_id', ''))
        # Fall back to presenter name
        author_names = row.get('author_names', presenter_name)
        talk_metadata[talk_id] = {
            'title': row.get('title', ''),
            'presenter_id': row.get('presenter_id', ''),
            'primary_contact_author': presenter_name,
            'author_names': author_names,
        }

    # Build fixed block sessions from special session config
    fixed_block_sessions = [
        FixedBlockSession(
            name=name,
            block=cfg["block"],
            talks=cfg["talks"]
        )
        for name, cfg in SPECIAL_SESSION_CONFIG.items()
    ]

    # Create Phase 4 input
    phase4_input = Phase4Input(
        phase3_result=phase3_result,
        room_capacities=ROOM_CAPACITIES,
        preferences=preferences,
        talk_metadata=talk_metadata,
        fixed_block_sessions=fixed_block_sessions
    )

    # Run Phase 4
    try:
        phase4_result = solve_phase4(
            phase4_input,
            room_names=ROOM_NAMES,
            verbose=verbose
        )
        return phase4_result
    except Exception as e:
        if verbose:
            print(f"  Phase 4 error: {e}")
        return None


def validate_all_talks_scheduled(
    schedule_df: pd.DataFrame,
    expected_talks: set,
    verbose: bool = False
) -> Tuple[bool, set, set]:
    """
    Validate that all expected talks are scheduled exactly once.

    This is a HARD CONSTRAINT - if any talks are missing or duplicated,
    the schedule is invalid.

    Args:
        schedule_df: The generated schedule DataFrame
        expected_talks: Set of talk_ids that must be scheduled
        verbose: Print detailed info

    Returns:
        Tuple of (is_valid, missing_talks, duplicate_talks)

    Raises:
        ValueError: If any talks are missing (hard constraint violation)
    """
    scheduled_talks = set(schedule_df['Talk_ID'].unique())

    missing = expected_talks - scheduled_talks
    extra = scheduled_talks - expected_talks

    # Check for duplicates (same talk scheduled multiple times)
    talk_counts = schedule_df['Talk_ID'].value_counts()
    duplicates = set(talk_counts[talk_counts > 1].index)

    is_valid = len(missing) == 0 and len(duplicates) == 0

    if verbose or not is_valid:
        if missing:
            print(
                f"\n  ❌ HARD CONSTRAINT VIOLATION: {len(missing)} talks not scheduled!")
            for t in sorted(missing)[:10]:
                print(f"      Missing: {t}")
            if len(missing) > 10:
                print(f"      ... and {len(missing) - 10} more")

        if duplicates:
            print(
                f"\n  ❌ HARD CONSTRAINT VIOLATION: {len(duplicates)} talks scheduled multiple times!")
            for t in sorted(duplicates):
                count = talk_counts[t]
                print(f"      {t}: scheduled {count} times")

        if extra:
            print(
                f"\n  ⚠️  WARNING: {len(extra)} unexpected talks in schedule")
            for t in sorted(extra)[:5]:
                print(f"      Extra: {t}")

    if missing:
        raise ValueError(
            f"HARD CONSTRAINT VIOLATION: {len(missing)} talks not scheduled: "
            f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
        )

    if duplicates:
        raise ValueError(
            f"HARD CONSTRAINT VIOLATION: {len(duplicates)} talks scheduled multiple times: "
            f"{sorted(duplicates)}"
        )

    return is_valid, missing, duplicates


def export_schedule_csv(
    results: Dict,
    data,
    data_full,
    output_path: Path,
    data_dir: str = None,
    run_swap_opt: bool = True,
    keyword_weight: float = 0.1,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Export schedule to CSV using Phase 4 for proper room assignment.

    Optionally runs swap optimization before Phase 4 to resolve presenter violations.

    Args:
        results: Pipeline results dict
        data: Data with special sessions excluded (used during optimization)
        data_full: Full data including special-session talks (for talk metadata)
        output_path: Output file path
        data_dir: Path to data directory (for loading keywords)
        run_swap_opt: Whether to run swap optimization
        keyword_weight: Weight for keyword coherence in swap scoring (0-1)
        verbose: Print progress

    Returns:
        Tuple of (DataFrame with schedule, swap_info dict)
    """
    phase3_result = results.get('phase3_result')
    swap_info = {'enabled': run_swap_opt, 'violations_before': 0, 'violations_resolved': 0,
                 'violations_remaining': 0, 'swaps': 0}

    if not phase3_result:
        return None, swap_info

    # Run swap optimization before Phase 4 if enabled
    if run_swap_opt:
        if verbose:
            print(f"\n  Running swap optimization (pre-Phase 4)...")

        # Load keywords for coherence scoring
        talk_keywords = None
        if data_dir:
            talk_keywords = load_talk_keywords_for_swap(
                data_dir, data_full, verbose=verbose)

        # Run swap optimization
        results = run_swap_optimization(
            results, data, data_full,
            talk_keywords=talk_keywords,
            keyword_weight=keyword_weight,
            verbose=verbose
        )

        swap_info['violations_before'] = results.get(
            'swap_violations_before', 0)
        swap_info['violations_resolved'] = results.get(
            'swap_violations_resolved', 0)
        swap_info['violations_remaining'] = results.get(
            'swap_violations_remaining', 0)
        swap_info['swaps'] = results.get('swap_count', 0)

    # Run Phase 4 for audience-based room assignment
    phase4_result = run_phase4(results, data_full, verbose=verbose)

    if phase4_result:
        # Use Phase 4 results (proper room assignment)
        rows = []
        for session in phase4_result.sessions:
            # Determine a block identifier for the evaluator
            # For fixed sessions, use the fixed session name (e.g., "SpecialSession_1")
            # For regular sessions, use the session_id + room as a pseudo-block
            if session.is_fixed:
                block_id = f"FIXED_{session.block_id}"
            else:
                block_id = f"{session.block_id}_{session.room_id}"

            rows.append({
                'Session_ID': session.block_id,  # Timeslot ID (TA, FB, etc.)
                'Block_ID': block_id,            # Pseudo-block for evaluator
                'Room': session.room_id,         # Room ID for evaluator compatibility
                'Room_ID': session.room_id,
                'Room_Name': session.room_name,
                'Slot': session.slot,
                'Talk_ID': session.talk_id,
                'Paper_ID': session.paper_id,
                'Title': session.title,
                'Primary_Contact_Author': session.primary_contact_author,
                'Author_Names': session.author_names,
                'Session_Total_Likes': session.total_likes,
                'Session_Unique_Attendees': session.unique_attendees,
                'Is_Fixed': session.is_fixed
            })

        df = pd.DataFrame(rows)
        df = df.sort_values(['Session_ID', 'Room_ID', 'Slot'])

        # HARD CONSTRAINT: Validate all talks are scheduled
        expected_talks = set(data_full.talks['talk_id'])
        validate_all_talks_scheduled(df, expected_talks, verbose=verbose)

        df.to_csv(output_path, index=False)

        return df, swap_info
    else:
        # Fallback to old method (no proper room assignment)
        if verbose:
            print("  Warning: Phase 4 failed, using basic room assignment")

        # Build lookup tables from FULL data (including special sessions)
        talk_info = {
            row['talk_id']: {
                'title': row.get('title', ''),
                'presenter_id': row.get('presenter_id', ''),
                'presenter_name': row.get('presenter_name', row.get('presenter_id', '')),
                'author_names': row.get('author_names', row.get('presenter_name', '')),
            }
            for _, row in data_full.talks.iterrows()
        }

        rows = []
        for assignment in phase3_result.assignments:
            block = assignment.block
            timeslot = assignment.timeslot
            room_mapping = assignment.room_mapping

            ts_id = timeslot.get('id', timeslot) if isinstance(
                timeslot, dict) else str(timeslot)

            for slot_idx, ntuple in enumerate(block.tuples):
                for room_pos, talk_id in enumerate(ntuple):
                    room_id = room_mapping.get(room_pos, f"Room_{room_pos}")
                    info = talk_info.get(talk_id, {})
                    # Extract paper_id from talk_id (T001 -> 1)
                    paper_id = int(
                        talk_id[1:]) if talk_id.startswith('T') else 0

                    rows.append({
                        'Session_ID': ts_id,
                        # Pseudo-block for evaluator
                        'Block_ID': f"{ts_id}_{room_id}",
                        'Room': room_id,  # For evaluator compatibility
                        'Room_ID': room_id,
                        'Room_Name': room_id,
                        'Slot': slot_idx + 1,
                        'Talk_ID': talk_id,
                        'Paper_ID': paper_id,
                        'Title': info.get('title', ''),
                        'Primary_Contact_Author': info.get('presenter_name', ''),
                        'Author_Names': info.get('author_names', ''),
                        'Session_Total_Likes': 0,
                        'Session_Unique_Attendees': 0,
                        'Is_Fixed': False
                    })

        # Add special sessions to their designated blocks
        add_special_sessions_to_schedule(rows, data_full, talk_info)

        df = pd.DataFrame(rows)
        df = df.sort_values(['Session_ID', 'Room_ID', 'Slot'])

        # HARD CONSTRAINT: Validate all talks are scheduled
        expected_talks = set(data_full.talks['talk_id'])
        validate_all_talks_scheduled(df, expected_talks, verbose=verbose)

        df.to_csv(output_path, index=False)

        return df, swap_info


def export_schedule_markdown(
    schedule_df: pd.DataFrame,
    pipeline_name: str,
    output_path: Path,
    presenter_violations: List[Dict] = None
):
    """Export schedule to Markdown format with session metrics.

    Args:
        schedule_df: Schedule DataFrame
        pipeline_name: Name of the pipeline
        output_path: Output file path
        presenter_violations: List of violation dicts with keys:
            talk_id, presenter_id, timeslot, unavailable_timeslots
    """
    # Custom session ordering: Thursday (TA, TB, TC, TD) then Friday (FA, FB, FC)
    session_order = ['TA', 'TB', 'TC', 'TD', 'FA', 'FB', 'FC']

    def session_sort_key(session):
        try:
            return session_order.index(session)
        except ValueError:
            return len(session_order)  # Unknown sessions go last

    # Determine which column names are used
    room_col = 'Room_ID' if 'Room_ID' in schedule_df.columns else 'Room'
    room_name_col = 'Room_Name' if 'Room_Name' in schedule_df.columns else room_col
    author_col = 'Primary_Contact_Author' if 'Primary_Contact_Author' in schedule_df.columns else 'Presenter_Name'

    # Check if metrics are available
    has_metrics = 'Session_Unique_Attendees' in schedule_df.columns

    lines = [
        f"# Conference Schedule - {pipeline_name.upper()} Pipeline",
        f"",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        f"",
    ]

    # Add presenter violations section if any
    if presenter_violations:
        lines.extend([
            "## ⚠️ Presenter Constraint Violations",
            "",
        ])
        for v in presenter_violations:
            presenter = v.get('presenter_name') or v.get(
                'presenter_id', 'Unknown')
            session = v['timeslot']
            unavailable = ', '.join(v.get('unavailable_timeslots', []))
            talk_title = v.get('title', v.get('talk_id', 'Unknown talk'))
            lines.append(
                f"- **{presenter}** is scheduled in session **{session}** "
                f"despite constraint to avoid: {unavailable}"
            )
            lines.append(f"  - Talk: {talk_title}")
            lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Group by Session_ID with custom ordering
    for session_id in sorted(schedule_df['Session_ID'].unique(), key=session_sort_key):
        session_df = schedule_df[schedule_df['Session_ID'] == session_id]

        lines.append(f"## Session: {session_id}")
        lines.append("")

        # Group by Room
        unique_rooms = session_df[room_col].unique()
        # Sort by capacity (largest first) based on ROOM_CAPACITIES
        sorted_rooms = sorted(
            unique_rooms, key=lambda r: ROOM_CAPACITIES.get(r, 0), reverse=True)

        for room_id in sorted_rooms:
            room_df = session_df[session_df[room_col]
                                 == room_id].sort_values('Slot')
            room_name = room_df[room_name_col].iloc[0] if room_name_col in room_df.columns else room_id

            # Get session metrics (same for all talks in room)
            if has_metrics:
                attendees = room_df['Session_Unique_Attendees'].iloc[0] if len(
                    room_df) > 0 else 0
                likes = room_df['Session_Total_Likes'].iloc[0] if len(
                    room_df) > 0 else 0
                is_fixed = room_df['Is_Fixed'].iloc[0] if 'Is_Fixed' in room_df.columns else False
                fixed_tag = " 🏆" if is_fixed else ""
                lines.append(
                    f"### {room_name} ({attendees} attendees, {likes} total likes){fixed_tag}")
            else:
                lines.append(f"### {room_name}")

            lines.append("")
            lines.append("| Slot | Paper ID | Talk | Presenter |")
            lines.append("|------|----------|------|-----------|")

            for _, row in room_df.iterrows():
                title = str(row['Title'])
                presenter = row.get(author_col, row.get('Author_Names', ''))
                # Get paper_id from column or extract from talk_id
                if 'Paper_ID' in row and pd.notna(row['Paper_ID']):
                    paper_id = int(row['Paper_ID'])
                elif row['Talk_ID'].startswith('T'):
                    paper_id = int(row['Talk_ID'][1:])
                else:
                    paper_id = 0
                lines.append(
                    f"| {row['Slot']} | {paper_id} | {title} | {presenter} |")

            lines.append("")

        lines.append("---")
        lines.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def export_schedule_json(schedule_df: pd.DataFrame, pipeline_name: str, output_path: Path):
    """Export schedule to JSON format with session metrics."""
    # Determine which column names are used
    room_col = 'Room_ID' if 'Room_ID' in schedule_df.columns else 'Room'
    room_name_col = 'Room_Name' if 'Room_Name' in schedule_df.columns else room_col
    author_col = 'Primary_Contact_Author' if 'Primary_Contact_Author' in schedule_df.columns else 'Presenter_ID'

    schedule_dict = {
        "pipeline": pipeline_name,
        "generated": datetime.now().isoformat(),
        "sessions": []
    }

    for session_id in sorted(schedule_df['Session_ID'].unique()):
        session_df = schedule_df[schedule_df['Session_ID'] == session_id]

        session_data = {
            "session_id": session_id,
            "rooms": {}
        }

        for room_id in sorted(session_df[room_col].unique()):
            room_df = session_df[session_df[room_col]
                                 == room_id].sort_values('Slot')
            room_name = room_df[room_name_col].iloc[0] if room_name_col in room_df.columns else room_id

            # Get session-level metrics
            unique_attendees = int(room_df['Session_Unique_Attendees'].iloc[0]
                                   ) if 'Session_Unique_Attendees' in room_df.columns else 0
            total_likes = int(room_df['Session_Total_Likes'].iloc[0]
                              ) if 'Session_Total_Likes' in room_df.columns else 0
            is_fixed = bool(room_df['Is_Fixed'].iloc[0]
                            ) if 'Is_Fixed' in room_df.columns else False

            session_data["rooms"][room_id] = {
                "room_name": room_name,
                "capacity": ROOM_CAPACITIES.get(room_id, 0),
                "unique_attendees": unique_attendees,
                "total_likes": total_likes,
                "is_fixed": is_fixed,
                "talks": [
                    {
                        "slot": int(row['Slot']),
                        "talk_id": row['Talk_ID'],
                        "paper_id": int(row['Paper_ID']) if 'Paper_ID' in row and pd.notna(row['Paper_ID']) else (int(row['Talk_ID'][1:]) if row['Talk_ID'].startswith('T') else 0),
                        "title": row['Title'],
                        "presenter": row.get(author_col, row.get('Author_Names', ''))
                    }
                    for _, row in room_df.iterrows()
                ]
            }

        schedule_dict["sessions"].append(session_data)

    with open(output_path, 'w') as f:
        json.dump(schedule_dict, f, indent=2)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_schedule(
    schedule_df: pd.DataFrame,
    preferences_csv: str,
    talks_csv: str,
    verbose: bool = False
) -> Dict[str, Any]:
    """Evaluate schedule quality using the ScheduleEvaluator."""
    from src.schedule_evaluator import (
        ScheduleEvaluator,
        load_preferences_from_csv,
    )

    # Load preferences (pass schedule_df for title->talk_id mapping in matrix format)
    try:
        preferences = load_preferences_from_csv(
            preferences_csv, schedule_df=schedule_df)
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not load preferences: {e}")
        preferences = {}

    # Load keywords - try talks_for_algorithm.csv first (has talk_id), fallback to talks_with_keywords
    talk_keywords = {}
    # Use the same path as the pipeline: data/output/talks_for_algorithm.csv
    algo_file = Path("data/output/talks_for_algorithm.csv")

    if algo_file.exists():
        try:
            algo_df = pd.read_csv(algo_file)
            for _, row in algo_df.iterrows():
                # talk_id in this file is numeric, schedule uses T### format
                talk_id = f"T{row['talk_id']:03d}"
                kw_str = str(row.get('keywords', ''))
                if pd.notna(kw_str) and kw_str.lower() != 'nan':
                    keywords = set(kw.strip()
                                   for kw in kw_str.split(';') if kw.strip())
                else:
                    keywords = set()
                talk_keywords[talk_id] = keywords
            if verbose:
                print(
                    f"    Loaded keywords for {len(talk_keywords)} talks from talks_for_algorithm.csv")
        except Exception as e:
            if verbose:
                print(
                    f"  Warning: Could not load keywords from talks_for_algorithm.csv: {e}")

    # Fallback to talks_with_keywords file (title-based matching)
    if not talk_keywords:
        try:
            talks_kw_df = pd.read_csv(talks_csv)
            title_to_keywords = {}
            for _, row in talks_kw_df.iterrows():
                title = str(row['title']).strip().lower()
                kw_str = str(row.get('master_keywords', ''))
                if pd.isna(kw_str) or kw_str.lower() == 'nan':
                    keywords = set()
                else:
                    keywords = set(kw.strip()
                                   for kw in kw_str.split(';') if kw.strip())
                title_to_keywords[title] = keywords

            # Map talk_id -> keywords by title
            if 'Title' in schedule_df.columns:
                for _, row in schedule_df.iterrows():
                    talk_id = row['Talk_ID']
                    title = str(row['Title']).strip().lower()
                    talk_keywords[talk_id] = title_to_keywords.get(
                        title, set())
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not load keywords: {e}")

    # Build talk_presenter from schedule (derive presenter_id from talk_id: T029 -> P029)
    talk_presenter = {}
    presenter_names = {}  # presenter_id -> name
    for _, row in schedule_df.iterrows():
        talk_id = row['Talk_ID']
        # Derive presenter_id from talk_id (T029 -> P029)
        presenter_id = talk_id.replace(
            'T', 'P') if talk_id.startswith('T') else talk_id
        talk_presenter[talk_id] = presenter_id
        # Get presenter name for violation messages
        name = row.get('Primary_Contact_Author')
        if pd.notna(name):
            presenter_names[presenter_id] = name

    # Load presenter unavailability from constraints file
    presenter_unavailability = {}
    constraints_file = Path(preferences_csv).parent / \
        "constraints for the schedule.xlsx"
    if constraints_file.exists():
        try:
            constraints_df = pd.read_excel(constraints_file)
            # Find the right columns (participant, sessions to avoid)
            # The file has headers on row 2 (index 1)
            for idx, row in constraints_df.iterrows():
                participant = row.get('Unnamed: 1')
                sessions_to_avoid = row.get('Unnamed: 5')

                if pd.notna(participant) and pd.notna(sessions_to_avoid):
                    participant = str(participant).strip()
                    sessions_str = str(sessions_to_avoid).strip()
                    if sessions_str.lower() not in ['nan', 'sessions to avoid', 'all']:
                        # Parse comma-separated sessions
                        sessions = [s.strip()
                                    for s in sessions_str.split(',') if s.strip()]
                        if sessions and participant not in ['participant', 'NaN']:
                            presenter_unavailability[participant] = set(
                                sessions)
            if verbose and presenter_unavailability:
                print(
                    f"    Loaded {len(presenter_unavailability)} presenter constraints")
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not load constraints: {e}")

    # Build presenter_id -> unavailable sessions mapping
    # Match by name since constraints use names, schedule uses IDs
    presenter_id_unavailability = {}
    for p_id, p_name in presenter_names.items():
        if p_name:
            # Try exact match first
            if p_name in presenter_unavailability:
                presenter_id_unavailability[p_id] = presenter_unavailability[p_name]
            else:
                # Try partial match (last name)
                for constraint_name, sessions in presenter_unavailability.items():
                    # Check if last name matches
                    p_last = p_name.split()[-1].lower() if p_name else ""
                    c_last = constraint_name.split(
                    )[-1].lower() if constraint_name else ""
                    if p_last and c_last and p_last == c_last:
                        presenter_id_unavailability[p_id] = sessions
                        break

    # Evaluate
    evaluator = ScheduleEvaluator(
        schedule_df=schedule_df,
        preferences=preferences,
        talk_keywords=talk_keywords,
        presenter_unavailability=presenter_id_unavailability,
        talk_presenter=talk_presenter
    )

    metrics = evaluator.evaluate(verbose=verbose)

    # Enrich violation details with names and titles
    violation_details = []
    for v in metrics.presenter_violation_details:
        talk_id = v['talk_id']
        # Find talk row in schedule
        talk_row = schedule_df[schedule_df['Talk_ID'] == talk_id]
        if not talk_row.empty:
            v['title'] = talk_row.iloc[0]['Title']
            # Try different column names for presenter name
            presenter_name = (
                talk_row.iloc[0].get('Presenter_Name') or
                talk_row.iloc[0].get('Primary_Contact_Author') or
                v.get('presenter_id')
            )
            v['presenter_name'] = presenter_name
        violation_details.append(v)

    return {
        "missed_attendance": metrics.total_missed_attendance,
        "session_hops": metrics.total_session_hops,
        "incoherent_sessions": metrics.incoherent_sessions,
        "presenter_violations": metrics.presenter_violations,
        "presenter_violation_details": violation_details,
        "participants_evaluated": metrics.total_participants_with_preferences,
        "total_talks": metrics.total_talks,
        "total_timeslots": metrics.total_timeslots,
        "total_room_sessions": metrics.total_room_sessions
    }


# =============================================================================
# COMPARISON SUMMARY
# =============================================================================

def generate_comparison_summary(results: List[PipelineResult], output_path: Path):
    """Generate Markdown comparison summary."""
    lines = [
        "# Pipeline Comparison Summary",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "## Overview",
        "",
        "| Pipeline | Status | Time (s) | Missed Attendance | Session Hops | Incoherent Sessions | Presenter Violations |",
        "|----------|--------|----------|-------------------|--------------|---------------------|----------------------|",
    ]

    for r in results:
        status = "✅" if r.success else "❌"
        time_str = f"{r.total_time:.1f}" if r.success else "-"
        missed = str(r.missed_attendance_evaluated) if r.success else "-"
        hops = str(r.session_hops_evaluated) if r.success else "-"
        incoherent = str(r.incoherent_sessions) if r.success else "-"
        presenter_viol = str(r.presenter_violations) if r.success else "-"

        lines.append(
            f"| {r.name} | {status} | {time_str} | {missed} | {hops} | {incoherent} | {presenter_viol} |")

    # Add swap optimization section
    lines.extend([
        "",
        "## Swap Optimization (Before → After)",
        "",
        "| Pipeline | Violations Before | Swaps | Violations After | Resolved |",
        "|----------|-------------------|-------|------------------|----------|",
    ])

    for r in results:
        if r.success:
            before = r.violations_before_swap
            after = r.violations_after_swap
            swaps = r.swaps_performed
            resolved = before - after
            lines.append(
                f"| {r.name} | {before} | {swaps} | {after} | {resolved} |")
        else:
            lines.append(f"| {r.name} | - | - | - | - |")

    lines.extend([
        "",
        "## Detailed Timing",
        "",
        "| Pipeline | Phase 1 | Phase 2 | Phase 3 | Total |",
        "|----------|---------|---------|---------|-------|",
    ])

    for r in results:
        if r.success:
            lines.append(
                f"| {r.name} | {r.phase1_time:.1f}s | {r.phase2_time:.1f}s | "
                f"{r.phase3_time:.1f}s | {r.total_time:.1f}s |"
            )
        else:
            lines.append(f"| {r.name} | - | - | - | - |")

    lines.extend([
        "",
        "## Quality Metrics",
        "",
        "### Evaluated Metrics (Post-hoc evaluation)",
        "",
        "| Metric | " + " | ".join(r.name for r in results) + " |",
        "|--------|" + "|".join("-" * 12 for _ in results) + "|",
    ])

    metrics = [
        ("Missed Attendance", "missed_attendance_evaluated"),
        ("Session Hops", "session_hops_evaluated"),
        ("Incoherent Sessions", "incoherent_sessions"),
    ]

    for metric_name, attr in metrics:
        values = []
        for r in results:
            if r.success:
                values.append(str(getattr(r, attr, "-")))
            else:
                values.append("-")
        lines.append(f"| {metric_name} | " + " | ".join(values) + " |")

    lines.extend([
        "",
        "### Pipeline-Reported Metrics",
        "",
        "| Metric | " + " | ".join(r.name for r in results) + " |",
        "|--------|" + "|".join("-" * 12 for _ in results) + "|",
    ])

    for metric_name, attr in [
        ("Missed Attendance (reported)", "missed_attendance_pipeline"),
        ("Room Switches (reported)", "room_switches_pipeline"),
        ("Presenter Violations", "presenter_violations"),
    ]:
        values = []
        for r in results:
            if r.success:
                val = getattr(r, attr, 0)
                values.append(str(val) if val else "0")
            else:
                values.append("-")
        lines.append(f"| {metric_name} | " + " | ".join(values) + " |")

    lines.extend([
        "",
        "## Output Files",
        "",
    ])

    for r in results:
        if r.success:
            lines.extend([
                f"### {r.name}",
                f"- CSV: `{r.csv_file}`",
                f"- Markdown: `{r.md_file}`",
                f"- JSON: `{r.json_file}`",
                f"- Metrics: `{r.metrics_file}`",
                "",
            ])
        else:
            lines.extend([
                f"### {r.name}",
                f"- **Failed**: {r.error_message}",
                "",
            ])

    lines.extend([
        "---",
        "",
        "## Notes",
        "",
        "- **Missed Attendance**: Number of preferred talks missed due to parallel scheduling",
        "- **Session Hops**: Room switches required within blocks to attend preferred talks",
        "- **Incoherent Sessions**: Room-timeslot pairs where talks don't share keywords",
        "- **Presenter Violations**: Presenters scheduled in their unavailable timeslots",
        "- Metrics evaluated using survey preferences from preferences.csv",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def save_comparison_json(results: List[PipelineResult], output_path: Path):
    """Save all comparison metrics to JSON."""
    data = {
        "generated": datetime.now().isoformat(),
        "pipelines": []
    }

    for r in results:
        pipeline_data = {
            "name": r.name,
            "success": r.success,
            "error_message": r.error_message,
            "timing": {
                "total": r.total_time,
                "phase1": r.phase1_time,
                "phase2": r.phase2_time,
                "phase3": r.phase3_time,
            },
            "metrics_pipeline": {
                "missed_attendance": r.missed_attendance_pipeline,
                "room_switches": r.room_switches_pipeline,
                "presenter_violations": r.presenter_violations,
            },
            "metrics_evaluated": {
                "missed_attendance": r.missed_attendance_evaluated,
                "session_hops": r.session_hops_evaluated,
                "incoherent_sessions": r.incoherent_sessions,
            },
            "files": {
                "csv": r.csv_file,
                "markdown": r.md_file,
                "json": r.json_file,
                "metrics": r.metrics_file,
            }
        }
        data["pipelines"].append(pipeline_data)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PIPELINE COMPARISON - 2026 DATA")
    print("=" * 70)
    print(f"\nData directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"\nSpecial session handling:")
    print(f"  • {len(SPECIAL_SESSION_TALK_IDS)} special-session talks excluded from Phase 1-3 optimization")
    for cfg in SPECIAL_SESSION_CONFIG.values():
        print(f"  • {cfg['name']} ({len(cfg['talks'])} talks) added to {cfg['block']} in Phase 4")
    print()

    # Load FULL data once (for metadata in export)
    data_full, _ = load_data_and_instance(args.data_dir, args.verbose)

    # Define pipelines to run (traditional last since it's slowest)
    pipelines = []

    pipelines.append(("heuristic", run_heuristic_pipeline))

    if not args.skip_matching:
        pipelines.append(("matching", run_matching_pipeline_wrapper))
        pipelines.append(("matching_kw", run_matching_kw_pipeline_wrapper))

    if not args.skip_traditional:
        pipelines.append(("traditional", run_traditional_pipeline))

    results: List[PipelineResult] = []

    for pipeline_name, pipeline_func in pipelines:
        print(f"\n{'=' * 80}")
        print(f"  🚀 RUNNING: {pipeline_name.upper()} PIPELINE")
        print(f"{'=' * 80}")

        result = PipelineResult(name=pipeline_name, success=False)

        try:
            # Load data with special sessions excluded for optimization
            data, instance = load_data_excluding_special_sessions(
                args.data_dir, verbose=args.verbose)

            # Add dummy talks if slots > talks (handles 123 slots scenario)
            data, instance, dummy_ids = add_dummy_talks_if_needed(
                data, instance, verbose=args.verbose)
            if dummy_ids:
                print(
                    f"  Dummy talks added: {len(dummy_ids)} ({', '.join(dummy_ids[:3])}{'...' if len(dummy_ids) > 3 else ''})")

            start_time = time.time()
            pipeline_results = pipeline_func(instance, data, args.verbose)
            result.total_time = time.time() - start_time

            result.phase1_time = pipeline_results.get('phase1_time', 0)
            result.phase2_time = pipeline_results.get('phase2_time', 0)
            result.phase3_time = pipeline_results.get('phase3_time', 0)

            result.missed_attendance_pipeline = int(
                pipeline_results.get('phase1_objective', 0))
            result.room_switches_pipeline = int(
                pipeline_results.get('phase2_hopping', 0))

            phase3 = pipeline_results.get('phase3_result')
            if phase3 and hasattr(phase3, 'total_violations'):
                result.presenter_violations = phase3.total_violations

            # Export CSV (includes swap optimization and special sessions added in Phase 4)
            csv_path = output_dir / f"schedule_{pipeline_name}.csv"
            print(
                f"\n  Running swap optimization + Phase 4 (room assignment by audience)...")
            schedule_df, swap_info = export_schedule_csv(
                pipeline_results, data, data_full, csv_path,
                data_dir=args.data_dir,
                run_swap_opt=True,
                keyword_weight=0.1,
                verbose=args.verbose
            )
            result.csv_file = str(csv_path)

            # Log swap optimization results and store for comparison
            if swap_info:
                result.violations_before_swap = swap_info.get(
                    'violations_before', 0)
                result.violations_after_swap = swap_info.get(
                    'violations_remaining', 0)
                result.swaps_performed = swap_info.get('swaps', 0)
                if swap_info.get('swaps', 0) > 0:
                    print(f"\n  Swap optimization: {swap_info['violations_resolved']} violations resolved, "
                          f"{swap_info['violations_remaining']} remaining ({swap_info['swaps']} swaps)")

            if schedule_df is not None and not schedule_df.empty:
                # Evaluate first to get violation details
                print(f"\n  Evaluating schedule quality...")
                eval_metrics = evaluate_schedule(
                    schedule_df,
                    args.preferences,
                    args.talks_with_keywords,
                    args.verbose
                )

                result.missed_attendance_evaluated = eval_metrics["missed_attendance"]
                result.session_hops_evaluated = eval_metrics["session_hops"]
                result.incoherent_sessions = eval_metrics["incoherent_sessions"]
                # Always use evaluated presenter violations (after swap optimization)
                result.presenter_violations = eval_metrics["presenter_violations"]

                # Export Markdown with violation details
                md_path = output_dir / f"schedule_{pipeline_name}.md"
                export_schedule_markdown(
                    schedule_df,
                    pipeline_name,
                    md_path,
                    presenter_violations=eval_metrics.get(
                        "presenter_violation_details", [])
                )
                result.md_file = str(md_path)

                # Export JSON
                json_path = output_dir / f"schedule_{pipeline_name}.json"
                export_schedule_json(schedule_df, pipeline_name, json_path)
                result.json_file = str(json_path)

                # Save individual metrics
                metrics_path = output_dir / f"metrics_{pipeline_name}.json"
                with open(metrics_path, 'w') as f:
                    json.dump({
                        "pipeline": pipeline_name,
                        "timing": {
                            "total": result.total_time,
                            "phase1": result.phase1_time,
                            "phase2": result.phase2_time,
                            "phase3": result.phase3_time,
                        },
                        "metrics_pipeline": {
                            "missed_attendance": result.missed_attendance_pipeline,
                            "room_switches": result.room_switches_pipeline,
                            "presenter_violations": result.presenter_violations,
                        },
                        "metrics_evaluated": eval_metrics
                    }, f, indent=2)
                result.metrics_file = str(metrics_path)

                result.success = True

                print(
                    f"\n  ✓ {pipeline_name.upper()} completed in {result.total_time:.1f}s")
                print(
                    f"    Missed Attendance: {result.missed_attendance_evaluated}")
                print(f"    Session Hops: {result.session_hops_evaluated}")
                print(f"    Incoherent Sessions: {result.incoherent_sessions}")
            else:
                result.error_message = "No schedule generated"
                print(f"\n  ✗ {pipeline_name.upper()} produced no schedule")

        except ImportError as e:
            if "gurobipy" in str(e):
                result.error_message = "Gurobi not available"
                print(
                    f"\n  ✗ {pipeline_name.upper()} failed: Gurobi not available")
            else:
                result.error_message = str(e)
                print(f"\n  ✗ {pipeline_name.upper()} failed: {e}")
        except Exception as e:
            result.error_message = str(e)
            print(f"\n  ✗ {pipeline_name.upper()} failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

        results.append(result)

    # Generate comparison summary
    print(f"\n{'=' * 70}")
    print("GENERATING COMPARISON SUMMARY")
    print("=" * 70)

    summary_md = output_dir / "comparison_summary.md"
    generate_comparison_summary(results, summary_md)
    print(f"  ✓ Markdown summary: {summary_md}")

    comparison_json = output_dir / "comparison_metrics.json"
    save_comparison_json(results, comparison_json)
    print(f"  ✓ JSON metrics: {comparison_json}")

    # Print final summary
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Pipeline':<15} {'Status':<8} {'Time':>8} {'Missed':>8} {'Hops':>8} {'Incoherent':>10} {'Violations':>10}")
    print("-" * 80)

    for r in results:
        status = "✓" if r.success else "✗"
        time_str = f"{r.total_time:.1f}s" if r.success else "-"
        missed = str(r.missed_attendance_evaluated) if r.success else "-"
        hops = str(r.session_hops_evaluated) if r.success else "-"
        incoherent = str(r.incoherent_sessions) if r.success else "-"
        violations = str(r.presenter_violations) if r.success else "-"

        print(
            f"{r.name:<15} {status:<8} {time_str:>8} {missed:>8} {hops:>8} {incoherent:>10} {violations:>10}")

    print()
    print(f"Output files saved to: {output_dir}/")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
