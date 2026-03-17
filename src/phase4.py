"""
Phase 4: Finalization & Room Assignment

This phase completes the schedule by:
1. Computing session interest metrics (total_likes, unique_attendees)
2. Assigning physical rooms based on total likes → room capacity matching
3. Adding fixed block sessions (pre-specified sessions assigned to specific blocks)

Terminology:
- Block: A set of parallel talks across all rooms during one timeslot
         (e.g., 5 rooms × 4 slots = 20 talks). Named TA, TB, FC, etc.
- Session (or room-session): The set of sequential talks in ONE room within
         a block (e.g., 4 talks in Room A during block TA).

The room assignment algorithm is greedy and provably optimal:
- Sort sessions by total_likes (ascending)
- Sort rooms by capacity (ascending)
- Match lowest total_likes to smallest room
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
import pandas as pd

from .phase3 import Phase3Result, RoomAssignment


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SessionMetrics:
    """Metrics for a single room-session."""
    block_id: str           # e.g., "TA", "FB"
    room_position: int      # 0-indexed position in block
    room_id: str            # Physical room ID after assignment
    talks: List[str]        # Ordered list of talk IDs
    total_likes: int        # Sum of preferences for all talks
    unique_attendees: int   # People who like at least one talk


@dataclass
class FixedBlockSession:
    """A pre-specified session assigned to a specific block."""
    name: str               # e.g., "SpecialSession_1"
    block: str              # Target block ID (e.g., "FA")
    talks: List[str]        # Talk IDs in sequence order


@dataclass
class Phase4Input:
    """Input data for Phase 4."""
    phase3_result: Phase3Result
    room_capacities: Dict[str, int]     # room_id -> capacity
    preferences: Dict[str, Set[str]]    # participant_id -> set of talk_ids
    # talk_id -> {title, primary_author, authors, ...}
    talk_metadata: Dict[str, Dict]
    fixed_block_sessions: List[FixedBlockSession] = field(default_factory=list)
    # block_id -> set of unavailable room_ids (e.g., {"TD": {"HOGM 00.85"}, "FC": {"HOGM 00.85"}})
    room_unavailability: Dict[str, Set[str]] = field(default_factory=dict)


@dataclass
class FinalSession:
    """A finalized session with all metadata."""
    block_id: str
    room_id: str
    room_name: str
    slot: int               # 1-indexed slot within session
    talk_id: str
    paper_id: int           # Numeric paper ID (talk_id without T prefix)
    title: str
    primary_contact_author: str
    author_names: str
    total_likes: int        # Session-level metric
    unique_attendees: int   # Session-level metric
    is_fixed: bool          # True if from fixed_block_sessions


@dataclass
class ScheduleResult:
    """Complete schedule output with all talk assignments and metrics."""
    sessions: List[FinalSession]        # All talk assignments with metrics
    session_metrics: List[SessionMetrics]  # Per-session metrics
    fixed_sessions_added: List[str]     # Names of fixed sessions added

    def to_markdown(self) -> str:
        """Generate a human-readable markdown schedule."""
        from collections import defaultdict

        lines = ["# Conference Schedule\n"]

        # Group sessions by block
        by_block: Dict[str, List[FinalSession]] = defaultdict(list)
        for s in self.sessions:
            by_block[s.block_id].append(s)

        # Group metrics by block for summary info
        metrics_by_block: Dict[str, List[SessionMetrics]] = defaultdict(list)
        for m in self.session_metrics:
            metrics_by_block[m.block_id].append(m)

        for block_id in sorted(by_block.keys()):
            block_sessions = by_block[block_id]
            lines.append(f"\n## Block {block_id}\n")

            # Group by room within the block
            by_room: Dict[str, List[FinalSession]] = defaultdict(list)
            for s in block_sessions:
                by_room[s.room_id].append(s)

            # Table header
            rooms = sorted(by_room.keys())
            max_slots = max(len(talks) for talks in by_room.values())

            # Build a table: rows = slots, columns = rooms
            lines.append("| Slot | " + " | ".join(rooms) + " |")
            lines.append("|------|" + "------|" * len(rooms))

            for slot in range(1, max_slots + 1):
                row = [f" {slot} "]
                for room in rooms:
                    room_talks = sorted(by_room[room], key=lambda s: s.slot)
                    talk = next((t for t in room_talks if t.slot == slot), None)
                    if talk:
                        cell = f" {talk.title} ({talk.primary_contact_author}) "
                        if talk.is_fixed:
                            cell += "[fixed] "
                    else:
                        cell = " "
                    row.append(cell)
                lines.append("|" + "|".join(row) + "|")

            # Room metrics summary
            block_metrics = sorted(
                metrics_by_block.get(block_id, []), key=lambda m: m.room_id)
            if block_metrics:
                lines.append("")
                for m in block_metrics:
                    lines.append(
                        f"- **{m.room_id}**: {m.unique_attendees} unique attendees, "
                        f"{m.total_likes} total likes")

        # Summary
        lines.append(f"\n---\n")
        lines.append(f"**Total blocks**: {len(by_block)}  ")
        lines.append(
            f"**Total sessions**: {len(self.session_metrics)}  ")
        lines.append(f"**Total talks**: {len(self.sessions)}")
        if self.fixed_sessions_added:
            lines.append(
                f"  \n**Fixed sessions**: {', '.join(self.fixed_sessions_added)}")

        return "\n".join(lines)

    def save_markdown(self, path) -> None:
        """Save schedule as a markdown file."""
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_markdown())

    def generate_personal_itinerary(
        self,
        participant_id: str,
        preferences: Dict[str, Set[str]]
    ) -> str:
        """
        Generate a personal itinerary for a participant.

        Shows which preferred talks they can attend per block,
        flags conflicts (multiple preferred talks in parallel),
        and tracks room switches.

        Args:
            participant_id: The participant to generate for
            preferences: Mapping of participant_id -> set of preferred talk_ids

        Returns:
            Markdown-formatted itinerary string
        """
        from collections import defaultdict

        prefs = preferences.get(participant_id, set())
        if not prefs:
            return f"No preferences recorded for participant {participant_id}."

        lines = [f"# Personal Itinerary for Participant {participant_id}\n"]

        # Group sessions by block
        by_block: Dict[str, List[FinalSession]] = defaultdict(list)
        for s in self.sessions:
            by_block[s.block_id].append(s)

        prev_room = None
        total_conflicts = 0
        total_hops = 0
        blocks_attended = 0

        for block_id in sorted(by_block.keys()):
            block_sessions = by_block[block_id]

            # Find preferred talks in this block, grouped by slot
            by_slot: Dict[int, List[FinalSession]] = defaultdict(list)
            for s in block_sessions:
                if s.talk_id in prefs:
                    by_slot[s.slot].append(s)

            if not by_slot:
                continue

            blocks_attended += 1
            lines.append(f"\n## Block {block_id}\n")

            for slot in sorted(by_slot.keys()):
                slot_talks = by_slot[slot]
                chosen = slot_talks[0]

                if len(slot_talks) > 1:
                    # Conflict: multiple preferred talks in parallel
                    total_conflicts += len(slot_talks) - 1
                    lines.append(
                        f"- **Slot {slot}**: {chosen.title} "
                        f"({chosen.primary_contact_author}) "
                        f"in **{chosen.room_name}**")
                    missed = [t for t in slot_talks[1:]]
                    missed_str = ", ".join(
                        f"{t.title} ({t.room_name})" for t in missed)
                    lines.append(
                        f"  - *Conflict — also wanted*: {missed_str}")
                else:
                    lines.append(
                        f"- **Slot {slot}**: {chosen.title} "
                        f"({chosen.primary_contact_author}) "
                        f"in **{chosen.room_name}**")

                # Track room switches
                if prev_room and prev_room != chosen.room_id:
                    total_hops += 1
                    lines.append(
                        f"  - *Room switch from {prev_room}*")
                prev_room = chosen.room_id

        # Summary
        attended_talks = sum(
            1 for s in self.sessions if s.talk_id in prefs)
        lines.append(f"\n---\n")
        lines.append(f"**Preferred talks in schedule**: {attended_talks}  ")
        lines.append(f"**Blocks with preferred talks**: {blocks_attended}  ")
        lines.append(f"**Parallel conflicts**: {total_conflicts}  ")
        lines.append(f"**Room switches**: {total_hops}")

        return "\n".join(lines)

    @classmethod
    def from_csv(cls, path) -> "ScheduleResult":
        """
        Reconstruct a ScheduleResult from a previously exported CSV.

        This enables the round-trip: export CSV -> edit -> reload -> evaluate/itinerary.
        """
        from collections import defaultdict
        from pathlib import Path

        df = pd.read_csv(Path(path))

        # Normalize column names (Phase 4 uses Block_ID/Room_ID)
        block_col = 'Block_ID' if 'Block_ID' in df.columns else 'Session_ID'
        room_col = 'Room_ID' if 'Room_ID' in df.columns else 'Room'
        room_name_col = 'Room_Name' if 'Room_Name' in df.columns else room_col

        sessions = []
        for _, row in df.iterrows():
            talk_id = str(row['Talk_ID'])
            paper_id = row.get('Paper_ID', 0)
            if pd.isna(paper_id):
                paper_id = 0

            sessions.append(FinalSession(
                block_id=str(row[block_col]),
                room_id=str(row[room_col]),
                room_name=str(row.get(room_name_col, row[room_col])),
                slot=int(row.get('Slot', 0)),
                talk_id=talk_id,
                paper_id=int(paper_id),
                title=str(row.get('Title', '')),
                primary_contact_author=str(
                    row.get('Primary_Contact_Author', '')),
                author_names=str(row.get('Author_Names', '')),
                total_likes=int(row.get('Session_Total_Likes', 0)),
                unique_attendees=int(
                    row.get('Session_Unique_Attendees', 0)),
                is_fixed=bool(row.get('Is_Fixed', False)),
            ))

        # Reconstruct session metrics by grouping
        by_room_block: Dict[
            Tuple[str, str], List[FinalSession]
        ] = defaultdict(list)
        for s in sessions:
            by_room_block[(s.block_id, s.room_id)].append(s)

        session_metrics = []
        for (block_id, room_id), room_sessions in by_room_block.items():
            room_sessions.sort(key=lambda s: s.slot)
            session_metrics.append(SessionMetrics(
                block_id=block_id,
                room_position=0,
                room_id=room_id,
                talks=[s.talk_id for s in room_sessions],
                total_likes=room_sessions[0].total_likes,
                unique_attendees=room_sessions[0].unique_attendees,
            ))

        # Infer fixed sessions from Is_Fixed column
        fixed_names = []
        if any(s.is_fixed for s in sessions):
            fixed_names = ["(loaded from CSV)"]

        return cls(
            sessions=sessions,
            session_metrics=session_metrics,
            fixed_sessions_added=fixed_names,
        )


# Backwards-compatibility alias
Phase4Result = ScheduleResult


# =============================================================================
# SESSION METRICS COMPUTATION
# =============================================================================

def compute_session_metrics(
    talks: List[str],
    preferences: Dict[str, Set[str]]
) -> Tuple[int, int]:
    """
    Compute metrics for a session (room-session).

    Args:
        talks: List of talk IDs in the session
        preferences: participant_id -> set of preferred talk_ids

    Returns:
        Tuple of (total_likes, unique_attendees)
        - total_likes: Sum of preferences for all talks in session
        - unique_attendees: Count of participants who like ≥1 talk
    """
    talk_set = set(talks)
    total_likes = 0
    unique_attendees = 0

    for participant_id, prefs in preferences.items():
        liked_in_session = prefs & talk_set
        count = len(liked_in_session)
        total_likes += count
        if count > 0:
            unique_attendees += 1

    return total_likes, unique_attendees


# =============================================================================
# ROOM ASSIGNMENT
# =============================================================================

def assign_rooms_by_audience(
    # (position, talks, total_likes, unique_attendees)
    sessions_with_metrics: List[Tuple[int, List[str], int, int]],
    room_capacities: Dict[str, int],
    unavailable_rooms: Set[str] = None,
    verbose: bool = False
) -> Dict[int, str]:
    """
    Assign physical rooms to session positions based on audience size.

    Algorithm (greedy, provably optimal):
    1. Sort sessions by total_likes (ascending)
    2. Sort rooms by capacity (ascending)
    3. Match: lowest total_likes → smallest room

    Args:
        sessions_with_metrics: List of (position, talks, total_likes, unique_attendees)
        room_capacities: room_id -> capacity
        unavailable_rooms: Set of room_ids that cannot be used for this block
        verbose: Print assignment details

    Returns:
        Dict mapping position -> room_id
    """
    if unavailable_rooms is None:
        unavailable_rooms = set()

    n_sessions = len(sessions_with_metrics)

    # Sort sessions by total_likes (ascending)
    # x[2] = total_likes
    sorted_sessions = sorted(sessions_with_metrics, key=lambda x: x[2])

    # Filter out unavailable rooms, then sort by capacity (ascending)
    available_rooms = {
        r: c for r, c in room_capacities.items() if r not in unavailable_rooms}
    sorted_rooms = sorted(available_rooms.items(),
                          key=lambda x: x[1])[:n_sessions]

    # Greedy matching
    room_mapping = {}
    for i, (pos, talks, total_likes, unique_attendees) in enumerate(sorted_sessions):
        room_id, capacity = sorted_rooms[i]
        room_mapping[pos] = room_id

        if verbose:
            print(
                f"    Position {pos}: {total_likes} total likes → {room_id} (cap: {capacity})")

    return room_mapping


# =============================================================================
# FIXED BLOCK SESSIONS
# =============================================================================

def validate_fixed_block_sessions(
    fixed_sessions: List[FixedBlockSession],
    block_ids: Set[str],
    all_talk_ids: Set[str],
    scheduled_talk_ids: Set[str]
) -> List[str]:
    """
    Validate fixed block sessions.

    Returns list of error messages (empty if valid).
    """
    errors = []

    for fs in fixed_sessions:
        # Check block exists
        if fs.block not in block_ids:
            errors.append(
                f"Fixed session '{fs.name}': block '{fs.block}' not found in schedule")

        # Check talks exist
        for talk_id in fs.talks:
            if talk_id not in all_talk_ids:
                errors.append(
                    f"Fixed session '{fs.name}': talk '{talk_id}' not found in talk list")
            if talk_id in scheduled_talk_ids:
                errors.append(
                    f"Fixed session '{fs.name}': talk '{talk_id}' already scheduled (should be excluded from earlier phases)")

    return errors


# =============================================================================
# MAIN PHASE 4 SOLVER
# =============================================================================

def solve_phase4(
    phase4_input: Phase4Input,
    room_names: Optional[Dict[str, str]] = None,
    verbose: bool = True
) -> ScheduleResult:
    """
    Solve Phase 4: Finalization and room assignment.

    Steps:
    1. For each block, compute session metrics (total_likes, unique_attendees)
    2. Assign rooms based on total likes (higher total_likes → larger room)
    3. Add fixed block sessions with their metrics
    4. Build final session list with all metadata

    Args:
        phase4_input: Phase4Input with all required data
        room_names: Optional mapping room_id -> display name
        verbose: Print progress

    Returns:
        ScheduleResult with finalized schedule
    """
    if room_names is None:
        room_names = {}

    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 4: FINALIZATION & ROOM ASSIGNMENT")
        print("=" * 70)

    phase3_result = phase4_input.phase3_result
    preferences = phase4_input.preferences
    room_capacities = phase4_input.room_capacities
    talk_metadata = phase4_input.talk_metadata
    fixed_block_sessions = phase4_input.fixed_block_sessions

    all_sessions: List[FinalSession] = []
    all_session_metrics: List[SessionMetrics] = []

    # Collect all timeslot IDs (session keys like TA, FB) and scheduled talks
    # Note: fixed_block_sessions uses timeslot/session keys, not internal block IDs
    timeslot_ids = set()
    scheduled_talk_ids = set()
    for assignment in phase3_result.assignments:
        # Get timeslot ID (session key like "TA", "FB")
        ts = assignment.timeslot
        ts_id = ts.get('id', ts) if isinstance(ts, dict) else str(ts)
        timeslot_ids.add(ts_id)
        for ntuple in assignment.block.tuples:
            for talk_id in ntuple:
                scheduled_talk_ids.add(talk_id)

    # Validate fixed block sessions using timeslot IDs (not block IDs)
    all_talk_ids = set(talk_metadata.keys())
    if fixed_block_sessions:
        errors = []
        for fs in fixed_block_sessions:
            # Check block (which is actually a timeslot/session key) exists
            if fs.block not in timeslot_ids:
                errors.append(
                    f"Fixed session '{fs.name}': block/session '{fs.block}' not found in schedule. Available: {sorted(timeslot_ids)}")

            # Check talks exist
            for talk_id in fs.talks:
                if talk_id not in all_talk_ids:
                    errors.append(
                        f"Fixed session '{fs.name}': talk '{talk_id}' not found in talk list")
                # Note: We don't check if already scheduled since talks should be excluded from earlier phases

        if errors:
            for err in errors:
                print(f"  WARNING: {err}")

    # Build fixed session lookup by timeslot ID (session key): "FA" -> [FixedBlockSession]
    fixed_by_timeslot: Dict[str, List[FixedBlockSession]] = {}
    for fs in fixed_block_sessions:
        if fs.block not in fixed_by_timeslot:
            fixed_by_timeslot[fs.block] = []
        fixed_by_timeslot[fs.block].append(fs)

    if verbose:
        print(f"\n  Processing {len(phase3_result.assignments)} blocks...")
        if fixed_block_sessions:
            print(
                f"  Adding {len(fixed_block_sessions)} fixed block sessions to timeslots: {list(fixed_by_timeslot.keys())}...")

    # Process each block
    for assignment in phase3_result.assignments:
        block = assignment.block
        timeslot = assignment.timeslot
        # Use timeslot ID as the session key (TA, FB, etc.)
        ts_id = timeslot.get('id', timeslot) if isinstance(
            timeslot, dict) else str(timeslot)

        if verbose:
            print(f"\n  Block {block.block_id} @ {ts_id}:")

        # Step 1: Compute metrics for each session (room position)
        n_rooms = len(block.tuples[0]) if block.tuples else 0
        sessions_with_metrics = []

        # Collect all talks already in this block (to check for duplicates)
        talks_in_block = set()
        for ntuple in block.tuples:
            for talk_id in ntuple:
                talks_in_block.add(talk_id)

        # Debug: show what's in this block if there are fixed sessions for this timeslot
        fixed_for_timeslot_debug = fixed_by_timeslot.get(ts_id, [])
        if fixed_for_timeslot_debug and verbose:
            print(
                f"    talks_in_block: {sorted(list(talks_in_block))[:10]}... ({len(talks_in_block)} total)")
            for fs in fixed_for_timeslot_debug:
                print(f"    {fs.name} talks: {fs.talks}")
                overlap = [t for t in fs.talks if t in talks_in_block]
                print(f"    overlap: {overlap}")

        for pos in range(n_rooms):
            # Get talks in this session (same room position across all tuples)
            talks_in_session = [ntuple[pos]
                                for ntuple in block.tuples if pos < len(ntuple)]
            total_likes, unique_attendees = compute_session_metrics(
                talks_in_session, preferences)
            sessions_with_metrics.append(
                (pos, talks_in_session, total_likes, unique_attendees, False, None))

        # Add fixed block sessions for this timeslot (if any were configured)
        # BUT only if the talks are not already in the block (Phase 2 may have attached them)
        fixed_for_timeslot = fixed_by_timeslot.get(ts_id, [])
        for fs in fixed_for_timeslot:
            # Check if fixed session talks are already in this block
            talks_already_in_block = any(t in talks_in_block for t in fs.talks)
            if talks_already_in_block:
                if verbose:
                    print(
                        f"    Skipping fixed session '{fs.name}': talks already in block (attached in Phase 2)")
                continue

            # Assign new positions
            pos = n_rooms + \
                len([x for x in fixed_for_timeslot if x.name < fs.name])
            total_likes, unique_attendees = compute_session_metrics(
                fs.talks, preferences)
            sessions_with_metrics.append(
                (pos, fs.talks, total_likes, unique_attendees, True, fs.name))
            if verbose:
                print(
                    f"    Adding fixed session '{fs.name}': {len(fs.talks)} talks, {unique_attendees} attendees")

        # Step 2: Assign rooms based on audience
        unavailable = phase4_input.room_unavailability.get(ts_id, set())
        room_mapping = assign_rooms_by_audience(
            [(pos, talks, tl, ua)
             for pos, talks, tl, ua, is_fixed, name in sessions_with_metrics],
            room_capacities,
            unavailable_rooms=unavailable,
            verbose=verbose
        )

        # Step 3: Build final sessions and metrics
        for pos, talks, total_likes, unique_attendees, is_fixed, fixed_name in sessions_with_metrics:
            room_id = room_mapping.get(pos, f"Room_{pos}")
            room_name = room_names.get(room_id, room_id)

            # Record session metrics (use ts_id as the block identifier in output)
            metrics = SessionMetrics(
                # Use timeslot ID (TA, FB, etc.) as block identifier
                block_id=ts_id,
                room_position=pos,
                room_id=room_id,
                talks=talks,
                total_likes=total_likes,
                unique_attendees=unique_attendees
            )
            all_session_metrics.append(metrics)

            # Build individual talk entries
            for slot_idx, talk_id in enumerate(talks):
                meta = talk_metadata.get(talk_id, {})
                # Extract paper_id: use metadata if available, otherwise parse from talk_id
                paper_id = meta.get('paper_id')
                if paper_id is None and talk_id.startswith('T'):
                    try:
                        paper_id = int(talk_id[1:])
                    except ValueError:
                        paper_id = 0

                session = FinalSession(
                    # Use timeslot ID (TA, FB, etc.) as block identifier
                    block_id=ts_id,
                    room_id=room_id,
                    room_name=room_name,
                    slot=slot_idx + 1,
                    talk_id=talk_id,
                    paper_id=paper_id,
                    title=meta.get('title', ''),
                    primary_contact_author=meta.get(
                        'primary_contact_author', ''),
                    author_names=meta.get('author_names', ''),
                    total_likes=total_likes,
                    unique_attendees=unique_attendees,
                    is_fixed=is_fixed
                )
                all_sessions.append(session)

    # Summary
    fixed_names = [fs.name for fs in fixed_block_sessions]

    if verbose:
        print(f"\n  Total sessions: {len(all_session_metrics)}")
        print(f"  Total talk assignments: {len(all_sessions)}")
        if fixed_names:
            print(f"  Fixed sessions added: {', '.join(fixed_names)}")

        # Print metrics summary
        total_unique = sum(m.unique_attendees for m in all_session_metrics)
        total_likes = sum(m.total_likes for m in all_session_metrics)
        print(f"\n  Metrics summary:")
        print(
            f"    Total unique_attendees (sum over sessions): {total_unique}")
        print(f"    Total likes (sum over all talks): {total_likes}")

    return ScheduleResult(
        sessions=all_sessions,
        session_metrics=all_session_metrics,
        fixed_sessions_added=fixed_names
    )


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def phase4_result_to_dataframe(result: ScheduleResult) -> pd.DataFrame:
    """
    Convert ScheduleResult to a pandas DataFrame for CSV export.

    Columns:
    - Block_ID, Room_ID, Room_Name, Slot
    - Talk_ID, Paper_ID, Title, Primary_Contact_Author, Author_Names
    - Session_Total_Likes, Session_Unique_Attendees
    - Is_Fixed
    """
    rows = []
    for s in result.sessions:
        rows.append({
            'Block_ID': s.block_id,
            'Room_ID': s.room_id,
            'Room_Name': s.room_name,
            'Slot': s.slot,
            'Talk_ID': s.talk_id,
            'Paper_ID': s.paper_id,
            'Title': s.title,
            'Primary_Contact_Author': s.primary_contact_author,
            'Author_Names': s.author_names,
            'Session_Total_Likes': s.total_likes,
            'Session_Unique_Attendees': s.unique_attendees,
            'Is_Fixed': s.is_fixed
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(['Block_ID', 'Room_ID', 'Slot'])
    return df


