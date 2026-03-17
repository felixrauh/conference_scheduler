"""
Schedule Evaluator Module

Computes quality metrics for a conference schedule:
1. Total Missed Attendance: participants who can't attend preferred talks due to conflicts
2. Required Session Hops: room switches within blocks to attend preferred talks
3. Incoherent Sessions: (room, timeslot) pairs where talks don't share keywords
4. Presenter Violations: presenters assigned to infeasible timeslots

Usage:
    from src.schedule_evaluator import ScheduleEvaluator, evaluate_schedule
    
    # Quick evaluation from files
    metrics = evaluate_schedule(
        schedule_csv="output/schedule.csv",
        preferences_csv="path/to/preferences.csv",
        talks_csv="path/to/talks.csv",
        availability_csv=None  # Optional
    )
    print(metrics)
    
    # Detailed evaluation with instance
    evaluator = ScheduleEvaluator(schedule, instance, talk_keywords)
    report = evaluator.full_report()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from pathlib import Path
import pandas as pd


@dataclass
class EvaluationMetrics:
    """Container for all schedule evaluation metrics."""

    # 1. Missed attendance: when k+1 preferred talks are at the same time, k are missed
    total_missed_attendance: int = 0
    missed_attendance_by_participant: Dict[str, int] = field(
        default_factory=dict)

    # 2. Session hops: room switches within blocks to attend preferred talks
    total_session_hops: int = 0
    session_hops_by_participant: Dict[str, int] = field(default_factory=dict)

    # 3. Incoherent sessions: (room, timeslot) pairs where talks don't share any keyword
    incoherent_sessions: int = 0
    incoherent_session_details: List[Dict] = field(default_factory=list)

    # 4. Presenter violations: presenters in infeasible timeslots
    presenter_violations: int = 0
    presenter_violation_details: List[Dict] = field(default_factory=list)

    # Additional statistics
    total_participants_with_preferences: int = 0
    total_talks: int = 0
    total_timeslots: int = 0
    total_room_sessions: int = 0

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "SCHEDULE EVALUATION METRICS",
            "=" * 60,
            f"Total Missed Attendance:     {self.total_missed_attendance}",
            f"  (participants who miss preferred talks due to conflicts)",
            f"",
            f"Total Session Hops:          {self.total_session_hops}",
            f"  (room switches within blocks to attend preferences)",
            f"",
            f"Incoherent Sessions:         {self.incoherent_sessions}",
            f"  (room-timeslot pairs without shared keywords)",
            f"",
            f"Presenter Violations:        {self.presenter_violations}",
            f"  (presenters in their unavailable timeslots)",
            "=" * 60,
            "STATISTICS",
            "=" * 60,
            f"Participants with preferences: {self.total_participants_with_preferences}",
            f"Total talks:                   {self.total_talks}",
            f"Total timeslots:               {self.total_timeslots}",
            f"Total room-sessions:           {self.total_room_sessions}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            "total_missed_attendance": self.total_missed_attendance,
            "total_session_hops": self.total_session_hops,
            "incoherent_sessions": self.incoherent_sessions,
            "presenter_violations": self.presenter_violations,
            "statistics": {
                "participants_with_preferences": self.total_participants_with_preferences,
                "total_talks": self.total_talks,
                "total_timeslots": self.total_timeslots,
                "total_room_sessions": self.total_room_sessions,
            }
        }


class ScheduleEvaluator:
    """
    Evaluates a conference schedule against various quality metrics.

    The evaluator takes a schedule and computes:
    1. Missed attendance from parallel conflicts
    2. Session hops required to attend preferred talks
    3. Keyword coherence of room-sessions
    4. Presenter availability violations
    """

    def __init__(
        self,
        schedule_df: pd.DataFrame,
        preferences: Dict[str, Set[str]],
        talk_keywords: Dict[str, Set[str]] = None,
        presenter_unavailability: Dict[str, Set[str]] = None,
        talk_presenter: Dict[str, str] = None
    ):
        """
        Initialize the evaluator.

        Args:
            schedule_df: DataFrame with columns [Session_ID, Block_ID, Slot, Room, Talk_ID, ...]
            preferences: Mapping participant_id -> set of preferred talk_ids
            talk_keywords: Mapping talk_id -> set of keyword strings (optional)
            presenter_unavailability: Mapping presenter_id -> set of unavailable timeslot_ids (optional)
            talk_presenter: Mapping talk_id -> presenter_id (optional)
        """
        self.schedule_df = schedule_df
        self.preferences = preferences
        self.talk_keywords = talk_keywords or {}
        self.presenter_unavailability = presenter_unavailability or {}
        self.talk_presenter = talk_presenter or {}

        # Build lookup structures
        self._build_schedule_index()

    def _build_schedule_index(self):
        """Build efficient lookup structures from schedule DataFrame."""
        # talks_by_timeslot: timeslot_id -> list of (room, slot, talk_id)
        self.talks_by_timeslot = defaultdict(list)

        # talks_by_parallel_slot: (session_id, slot) -> list of talk_ids
        # These are the talks that actually run in parallel (same session, same slot, different rooms)
        self.talks_by_parallel_slot = defaultdict(list)

        # talks_by_room_session: (timeslot_id, room) -> list of talk_ids (ordered by slot)
        self.talks_by_room_session = defaultdict(list)

        # talks_by_block: block_id -> list of (timeslot_id, room, slot, talk_id)
        self.talks_by_block = defaultdict(list)

        # block_info: block_id -> {"timeslot": str, "rooms": set}
        self.block_info = {}

        for _, row in self.schedule_df.iterrows():
            timeslot = row['Session_ID']
            block_id = row['Block_ID']
            slot = row['Slot']
            room = row['Room']
            talk_id = row['Talk_ID']

            self.talks_by_timeslot[timeslot].append((room, slot, talk_id))
            self.talks_by_parallel_slot[(timeslot, slot)].append(talk_id)
            self.talks_by_room_session[(timeslot, room)].append(
                (slot, talk_id))
            self.talks_by_block[block_id].append(
                (timeslot, room, slot, talk_id))

            if block_id not in self.block_info:
                self.block_info[block_id] = {
                    "timeslot": timeslot, "rooms": set()}
            self.block_info[block_id]["rooms"].add(room)

        # Sort room sessions by slot
        for key in self.talks_by_room_session:
            self.talks_by_room_session[key].sort(key=lambda x: x[0])

        # Get unique timeslots (in order they appear)
        self.timeslots = list(dict.fromkeys(self.schedule_df['Session_ID']))
        # Get unique parallel slots (session_id, slot) pairs
        self.parallel_slots = list(self.talks_by_parallel_slot.keys())
        self.all_talks = set(self.schedule_df['Talk_ID'])

    def compute_missed_attendance(self, verbose: bool = False) -> Tuple[int, Dict[str, int]]:
        """
        Compute total missed attendance allowing session hops.

        For each participant, at each parallel slot (session + slot position):
        - If k+1 preferred talks are scheduled in parallel, k are missed

        Talks are truly parallel only if they have the same Session_ID AND same Slot.
        This allows free room switching within a session (session hopping).

        Args:
            verbose: If True, print detailed logging of computation

        Returns:
            Tuple of (total_missed, per_participant_dict)
        """
        total_missed = 0
        per_participant = {}

        # Log summary statistics if verbose
        if verbose:
            print(f"\n  [Missed Attendance Computation]")
            print(f"    Participants: {len(self.preferences)}")
            total_prefs = sum(len(p) for p in self.preferences.values())
            print(f"    Total preferences: {total_prefs}")
            print(f"    Sessions: {len(self.timeslots)}")
            print(
                f"    Parallel slots (session, slot): {len(self.parallel_slots)}")
            print(f"    Talks in schedule: {len(self.all_talks)}")

            # Check overlap between preferences and scheduled talks
            all_pref_talks = set()
            for prefs in self.preferences.values():
                all_pref_talks.update(prefs)
            overlap = all_pref_talks & self.all_talks
            print(
                f"    Preferred talks in schedule: {len(overlap)} / {len(all_pref_talks)}")

        # Track per-slot contributions for logging
        slot_contributions = {} if verbose else None

        for p_id, prefs in self.preferences.items():
            participant_missed = 0

            # Iterate over parallel slots (session_id, slot) - these are truly parallel
            for parallel_slot, talks_at_slot in self.talks_by_parallel_slot.items():
                # Count preferred talks at this parallel slot
                preferred_at_slot = sum(
                    1 for talk_id in talks_at_slot
                    if talk_id in prefs
                )

                if preferred_at_slot > 1:
                    # Can attend 1, miss the rest
                    missed_here = preferred_at_slot - 1
                    participant_missed += missed_here

                    # Track for logging
                    if verbose:
                        if parallel_slot not in slot_contributions:
                            slot_contributions[parallel_slot] = 0
                        slot_contributions[parallel_slot] += missed_here

            if participant_missed > 0:
                per_participant[p_id] = participant_missed
                total_missed += participant_missed

        if verbose:
            print(f"    Participants with conflicts: {len(per_participant)}")
            if per_participant:
                top5 = sorted(per_participant.items(), key=lambda x: -x[1])[:5]
                print(f"    Top 5 participants by missed talks:")
                for p, m in top5:
                    print(f"      {p}: {m} missed")
            if slot_contributions:
                top_slots = sorted(slot_contributions.items(),
                                   key=lambda x: -x[1])[:5]
                print(f"    Parallel slots with most conflicts:")
                for (sess, slot), m in top_slots:
                    print(
                        f"      ({sess}, slot {slot}): {m} missed (across all participants)")
            print(f"    TOTAL MISSED ATTENDANCE: {total_missed}")

        return total_missed, per_participant

    def compute_session_hops(self) -> Tuple[int, Dict[str, int]]:
        """
        Compute required session hops for all participants.

        A session hop occurs when a participant changes rooms within a block
        (between consecutive slots) to attend a preferred talk.

        Uses dynamic programming to compute minimum hops needed.

        Note: Uses talks_by_timeslot (grouped by Session_ID) not talks_by_block
        (which is per-room). Session hopping is about switching rooms WITHIN
        a timeslot/block across ALL rooms.

        Returns:
            Tuple of (total_hops, per_participant_dict)
        """
        total_hops = 0
        per_participant = {}

        for p_id, prefs in self.preferences.items():
            participant_hops = 0

            # Use timeslots (e.g., FA, FB) not per-room blocks (e.g., FA_00.85)
            for timeslot, timeslot_talks in self.talks_by_timeslot.items():
                hops = self._compute_participant_block_hops(
                    timeslot_talks, prefs)
                participant_hops += hops

            if participant_hops > 0:
                per_participant[p_id] = participant_hops
                total_hops += participant_hops

        return total_hops, per_participant

    def _compute_participant_block_hops(
        self,
        block_talks: List[Tuple[str, int, str]],
        prefs: Set[str]
    ) -> int:
        """
        Compute minimum hops for one participant in one block/timeslot.

        Args:
            block_talks: List of (room, slot, talk_id) in the block
            prefs: Set of preferred talk_ids for this participant

        Returns:
            Minimum number of room switches needed
        """
        if not block_talks:
            return 0

        # Group by slot position
        slots = defaultdict(dict)  # slot -> {room -> talk_id}
        for (room, slot, talk_id) in block_talks:
            slots[slot][room] = talk_id

        # Get ordered slot indices
        slot_indices = sorted(slots.keys())
        if len(slot_indices) <= 1:
            return 0

        # Get set of rooms
        rooms = list(next(iter(slots.values())).keys())
        room_to_idx = {r: i for i, r in enumerate(rooms)}
        n_rooms = len(rooms)

        # Get preferred rooms at each slot
        preferred_rooms_by_slot = []
        for slot_idx in slot_indices:
            preferred_rooms = set()
            for room, talk_id in slots[slot_idx].items():
                if talk_id in prefs:
                    preferred_rooms.add(room_to_idx[room])
            preferred_rooms_by_slot.append(preferred_rooms)

        # Count slots with preferences
        attended_count = sum(
            1 for rooms_set in preferred_rooms_by_slot if rooms_set)
        if attended_count <= 1:
            return 0

        # DP: dp[r] = min switches to be in room r at current slot
        INF = float('inf')
        dp = [INF] * n_rooms

        # Initialize with first slot that has preferences
        first_idx = None
        for i, rooms_set in enumerate(preferred_rooms_by_slot):
            if rooms_set:
                first_idx = i
                for r in rooms_set:
                    dp[r] = 0
                break

        if first_idx is None:
            return 0

        # Process remaining slots
        for i in range(first_idx + 1, len(slot_indices)):
            if preferred_rooms_by_slot[i]:
                # Must attend one of these rooms
                new_dp = [INF] * n_rooms
                for r in preferred_rooms_by_slot[i]:
                    for prev_r in range(n_rooms):
                        if dp[prev_r] < INF:
                            cost = dp[prev_r] + (0 if prev_r == r else 1)
                            new_dp[r] = min(new_dp[r], cost)
                dp = new_dp

        return min(dp[r] for r in range(n_rooms) if dp[r] < INF)

    def compute_incoherent_sessions(self) -> Tuple[int, List[Dict]]:
        """
        Count room-sessions where talks don't share at least one keyword.

        A room-session is all talks in one room at one timeslot.
        It's incoherent if no keyword appears in all talks of that session.

        Returns:
            Tuple of (count, list of detail dicts)
        """
        if not self.talk_keywords:
            return 0, []

        incoherent_count = 0
        details = []

        for (timeslot, room), talk_list in self.talks_by_room_session.items():
            if len(talk_list) < 2:
                # Single talk or empty - coherent by definition
                continue

            # Get keywords for each talk in this room-session
            talk_ids = [talk_id for (_, talk_id) in talk_list]
            keyword_sets = [
                self.talk_keywords.get(talk_id, set())
                for talk_id in talk_ids
            ]

            # Check if there's any keyword shared by ALL talks
            # (intersection of all keyword sets)
            if not keyword_sets or not all(keyword_sets):
                # Some talks have no keywords - mark as incoherent
                shared_keywords = set()
            else:
                shared_keywords = keyword_sets[0]
                for kw_set in keyword_sets[1:]:
                    shared_keywords = shared_keywords & kw_set

            if not shared_keywords:
                incoherent_count += 1
                details.append({
                    "timeslot": timeslot,
                    "room": room,
                    "talks": talk_ids,
                    "keywords_per_talk": {
                        tid: list(self.talk_keywords.get(tid, set()))
                        for tid in talk_ids
                    }
                })

        return incoherent_count, details

    def compute_presenter_violations(self) -> Tuple[int, List[Dict]]:
        """
        Count presenters assigned to their unavailable timeslots.

        Returns:
            Tuple of (count, list of violation detail dicts)
        """
        if not self.presenter_unavailability:
            return 0, []

        violations = 0
        details = []

        for _, row in self.schedule_df.iterrows():
            talk_id = row['Talk_ID']
            timeslot = row['Session_ID']

            presenter_id = self.talk_presenter.get(talk_id)
            if not presenter_id:
                # Try to get from schedule if available
                presenter_id = row.get('Presenter_ID')

            if presenter_id:
                unavailable = self.presenter_unavailability.get(
                    presenter_id, set())
                if timeslot in unavailable:
                    violations += 1
                    details.append({
                        "talk_id": talk_id,
                        "presenter_id": presenter_id,
                        "timeslot": timeslot,
                        "unavailable_timeslots": list(unavailable)
                    })

        return violations, details

    def evaluate(self, verbose: bool = False) -> EvaluationMetrics:
        """
        Run all evaluations and return comprehensive metrics.

        Args:
            verbose: If True, print detailed logging during computation

        Returns:
            EvaluationMetrics object with all computed values
        """
        metrics = EvaluationMetrics()

        # 1. Missed attendance
        metrics.total_missed_attendance, metrics.missed_attendance_by_participant = \
            self.compute_missed_attendance(verbose=verbose)

        # 2. Session hops
        metrics.total_session_hops, metrics.session_hops_by_participant = \
            self.compute_session_hops()

        # 3. Incoherent sessions
        metrics.incoherent_sessions, metrics.incoherent_session_details = \
            self.compute_incoherent_sessions()

        # 4. Presenter violations
        metrics.presenter_violations, metrics.presenter_violation_details = \
            self.compute_presenter_violations()

        # Statistics
        metrics.total_participants_with_preferences = len(self.preferences)
        metrics.total_talks = len(self.all_talks)
        metrics.total_timeslots = len(self.timeslots)
        metrics.total_room_sessions = len(self.talks_by_room_session)

        return metrics

    def full_report(self, verbose: bool = True) -> EvaluationMetrics:
        """
        Run all evaluations and print a detailed report.

        Alias for evaluate(verbose=True) — provided for API compatibility
        with the usage shown in the module docstring.
        """
        return self.evaluate(verbose=verbose)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_schedule_csv(schedule_path: str) -> pd.DataFrame:
    """Load schedule from CSV file.

    Normalizes column names so the evaluator works with both Phase 4 export
    format (Block_ID, Room_ID) and the legacy format (Session_ID, Room).
    """
    df = pd.read_csv(schedule_path)

    # Normalize column names: Phase 4 exports Block_ID/Room_ID,
    # but the evaluator expects Session_ID/Room.
    if 'Session_ID' not in df.columns and 'Block_ID' in df.columns:
        df['Session_ID'] = df['Block_ID']
    if 'Room' not in df.columns and 'Room_ID' in df.columns:
        df['Room'] = df['Room_ID']

    return df


def load_preferences_from_csv(
    preferences_csv: str,
    talk_id_prefix: str = "T",
    schedule_df: pd.DataFrame = None
) -> Dict[str, Set[str]]:
    """
    Load preferences from CSV format.

    Supports multiple formats:
    1. Long format: participant_id, talk_id columns (one row per preference)
    2. Wide format: participant_id, preferences column (comma-separated talk IDs)
    3. Matrix format: First column as participant ID, other columns as talk titles,
       cells contain "I would like to attend:" or similar to indicate preference

    Args:
        preferences_csv: Path to CSV file
        talk_id_prefix: Prefix for talk IDs (e.g., "T" for T001)
        schedule_df: Optional schedule DataFrame to build title->talk_id mapping

    Returns:
        Dict mapping participant_id to set of preferred talk_ids
    """
    df = pd.read_csv(preferences_csv)

    preferences = defaultdict(set)

    if 'participant_id' in df.columns and 'talk_id' in df.columns:
        # Long format: one row per preference
        for _, row in df.iterrows():
            p_id = str(row['participant_id'])
            t_id = str(row['talk_id'])
            # Normalize numeric IDs to T### format (e.g. "4" -> "T004")
            if t_id.isdigit():
                t_id = f"T{int(t_id):03d}"
            preferences[p_id].add(t_id)

    elif 'participant_id' in df.columns and 'preferences' in df.columns:
        # Wide format: comma-separated talk IDs in preferences column
        for _, row in df.iterrows():
            p_id = str(row['participant_id'])
            prefs_str = str(row['preferences'])
            if pd.isna(prefs_str) or prefs_str.lower() == 'nan':
                continue
            # Parse comma-separated talk IDs
            for t_id in prefs_str.split(','):
                t_id = t_id.strip()
                if t_id:
                    preferences[p_id].add(t_id)

    else:
        # Try matrix format: first column is participant ID, other columns are talks
        # Detect by checking if many columns exist and values contain "attend" or are empty
        first_col = df.columns[0]
        other_cols = df.columns[1:]

        # Check if this looks like matrix format (many columns with bracketed talk names)
        has_bracket_cols = any('[' in str(col) for col in other_cols)

        if len(other_cols) > 10 or has_bracket_cols:
            # Build title -> talk_id mapping from schedule if provided
            title_to_talk_id = {}
            if schedule_df is not None and 'Title' in schedule_df.columns and 'Talk_ID' in schedule_df.columns:
                for _, row in schedule_df.iterrows():
                    title = str(row['Title']).strip().lower()
                    talk_id = str(row['Talk_ID'])
                    title_to_talk_id[title] = talk_id

            # Extract talk titles from column names (format: "[N. Title - Author]")
            col_to_title = {}
            for col in other_cols:
                col_str = str(col).strip()
                # Try to extract title from "[N. Title - Author]" format
                if '[' in col_str and ']' in col_str:
                    # Remove brackets
                    inner = col_str.split('[', 1)[1].rsplit(']', 1)[0]
                    # Remove leading number and dot
                    if '. ' in inner:
                        inner = inner.split('. ', 1)[1]
                    # Remove author after " - "
                    if ' - ' in inner:
                        inner = inner.rsplit(' - ', 1)[0]
                    col_to_title[col] = inner.strip().lower()
                else:
                    col_to_title[col] = col_str.lower()

            # Process each row
            for idx, row in df.iterrows():
                # Generate participant ID from row index
                p_id = f"P{idx+1:03d}"

                for col in other_cols:
                    cell_value = str(row[col]).strip(
                    ).lower() if pd.notna(row[col]) else ""

                    # Check if this indicates a preference
                    if 'attend' in cell_value or cell_value == 'x' or cell_value == '1':
                        title = col_to_title.get(col, str(col).lower())

                        # Try to find matching talk_id from schedule
                        talk_id = None
                        if title in title_to_talk_id:
                            talk_id = title_to_talk_id[title]
                        else:
                            # Try partial match
                            for sched_title, tid in title_to_talk_id.items():
                                if title[:30] in sched_title or sched_title[:30] in title:
                                    talk_id = tid
                                    break

                        if talk_id:
                            preferences[p_id].add(talk_id)
                        # If no mapping, store by column index as fallback
                        elif not title_to_talk_id:
                            # No schedule provided, use column title as key
                            # Truncate for safety
                            preferences[p_id].add(title[:50])

            if preferences:
                return dict(preferences)

        raise ValueError(
            "Expected either columns 'participant_id' and 'talk_id' (long format), "
            "or 'participant_id' and 'preferences' (wide format with comma-separated IDs), "
            "or matrix format with first column as participant ID and other columns as talk preferences."
        )

    return dict(preferences)


def load_preferences_from_instance(preferences_matrix: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """
    Load preferences directly from a ProblemInstance preference_matrix.

    Args:
        preferences_matrix: Dict mapping participant_id to set of talk_ids

    Returns:
        Same format (for API consistency)
    """
    return preferences_matrix


def load_keywords_from_csv(
    talks_csv: str,
    title_col: str = "title",
    keywords_col: str = "master_keywords",
    talk_id_col: str = None
) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    """
    Load talk keywords from CSV.

    Args:
        talks_csv: Path to talks CSV file
        title_col: Column containing talk titles
        keywords_col: Column containing semicolon-separated keywords
        talk_id_col: Column containing talk IDs (if None, creates mapping from title)

    Returns:
        Tuple of:
        - talk_id -> set of keywords (or title -> keywords if no talk_id)
        - title -> talk_id mapping (for cross-referencing)
    """
    df = pd.read_csv(talks_csv)

    keywords_by_id = {}
    title_to_id = {}

    for idx, row in df.iterrows():
        title = str(row[title_col]).strip()

        # Get talk_id
        if talk_id_col and talk_id_col in df.columns:
            talk_id = str(row[talk_id_col])
        else:
            # Use index-based ID
            talk_id = f"T{idx + 1:03d}"

        title_to_id[title.lower()] = talk_id

        # Parse keywords
        kw_str = str(row.get(keywords_col, ''))
        if pd.isna(kw_str) or kw_str.lower() == 'nan':
            keywords = set()
        else:
            keywords = set(kw.strip()
                           for kw in kw_str.split(';') if kw.strip())

        keywords_by_id[talk_id] = keywords

    return keywords_by_id, title_to_id


def load_availability_from_csv(
    availability_csv: str
) -> Dict[str, Set[str]]:
    """
    Load presenter unavailability from CSV.

    Expected columns: presenter_id, unavailable_timeslot

    Returns:
        Dict mapping presenter_id to set of unavailable timeslot_ids
    """
    df = pd.read_csv(availability_csv)

    unavailability = defaultdict(set)
    for _, row in df.iterrows():
        pres_id = str(row['presenter_id'])
        ts_id = str(row['unavailable_timeslot'])
        unavailability[pres_id].add(ts_id)

    return dict(unavailability)


def evaluate_schedule(
    schedule_csv: str,
    preferences: Dict[str, Set[str]] = None,
    preferences_csv: str = None,
    talk_keywords: Dict[str, Set[str]] = None,
    talks_csv: str = None,
    presenter_unavailability: Dict[str, Set[str]] = None,
    availability_csv: str = None,
    talk_presenter: Dict[str, str] = None,
    verbose: bool = True
) -> EvaluationMetrics:
    """
    Convenience function to evaluate a schedule from files.

    Args:
        schedule_csv: Path to schedule CSV file
        preferences: Pre-loaded preferences dict (optional)
        preferences_csv: Path to preferences CSV (if preferences not provided)
        talk_keywords: Pre-loaded keywords dict (optional)
        talks_csv: Path to talks CSV with keywords (if talk_keywords not provided)
        presenter_unavailability: Pre-loaded unavailability dict (optional)
        availability_csv: Path to availability CSV (if not provided)
        talk_presenter: Pre-loaded talk->presenter mapping (optional)
        verbose: Print results

    Returns:
        EvaluationMetrics object
    """
    # Load schedule
    schedule_df = load_schedule_csv(schedule_csv)

    # Load or use provided preferences
    if preferences is None:
        if preferences_csv:
            preferences = load_preferences_from_csv(preferences_csv)
        else:
            preferences = {}

    # Load or use provided keywords
    if talk_keywords is None:
        if talks_csv:
            talk_keywords, _ = load_keywords_from_csv(talks_csv)
        else:
            talk_keywords = {}

    # Load or use provided availability
    if presenter_unavailability is None:
        if availability_csv and Path(availability_csv).exists():
            presenter_unavailability = load_availability_from_csv(
                availability_csv)
        else:
            presenter_unavailability = {}

    # Build talk_presenter from schedule if not provided
    if talk_presenter is None and 'Presenter_ID' in schedule_df.columns:
        talk_presenter = {}
        for _, row in schedule_df.iterrows():
            talk_presenter[row['Talk_ID']] = row['Presenter_ID']

    # Create evaluator and run
    evaluator = ScheduleEvaluator(
        schedule_df=schedule_df,
        preferences=preferences,
        talk_keywords=talk_keywords,
        presenter_unavailability=presenter_unavailability,
        talk_presenter=talk_presenter or {}
    )

    metrics = evaluator.evaluate()

    if verbose:
        print(metrics)

    return metrics


def evaluate_from_instance(
    schedule_df: pd.DataFrame,
    instance,  # ProblemInstance
    talk_keywords: Dict[str, Set[str]] = None
) -> EvaluationMetrics:
    """
    Evaluate schedule using a ProblemInstance object.

    Args:
        schedule_df: Schedule DataFrame
        instance: ProblemInstance object
        talk_keywords: Optional keyword mapping

    Returns:
        EvaluationMetrics object
    """
    evaluator = ScheduleEvaluator(
        schedule_df=schedule_df,
        preferences=instance.preferences,
        talk_keywords=talk_keywords or {},
        presenter_unavailability=instance.presenter_unavailability,
        talk_presenter=instance.talk_presenter
    )

    return evaluator.evaluate()


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Command-line interface for schedule evaluation."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Evaluate a conference schedule quality"
    )
    parser.add_argument(
        "schedule",
        help="Path to schedule CSV file"
    )
    parser.add_argument(
        "--preferences",
        help="Path to preferences CSV (participant_id, talk_id format)"
    )
    parser.add_argument(
        "--talks",
        help="Path to talks CSV with keywords"
    )
    parser.add_argument(
        "--availability",
        help="Path to availability CSV"
    )
    parser.add_argument(
        "--output-json",
        help="Save metrics to JSON file"
    )

    args = parser.parse_args()

    metrics = evaluate_schedule(
        schedule_csv=args.schedule,
        preferences_csv=args.preferences,
        talks_csv=args.talks,
        availability_csv=args.availability,
        verbose=True
    )

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        print(f"\nMetrics saved to {args.output_json}")


if __name__ == "__main__":
    main()
