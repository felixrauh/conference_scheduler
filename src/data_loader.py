"""
Data loading and validation module.

Reads CSV/JSON input files and validates data integrity.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class ConferenceData:
    """Container for all conference input data."""

    # Conference metadata
    conference_name: str
    rooms: List[str]

    # Block types: {type_id: {"n": int, "k": int, "count": int}}
    block_types: Dict[str, Dict]

    # Timeslots: list of {"id", "start_time", "type_id", "rooms"}
    timeslots: List[Dict]

    # Talks: DataFrame with columns [talk_id, title, presenter_id, track]
    talks: pd.DataFrame

    # Preferences: DataFrame with columns [participant_id, talk_id]
    # Participant IDs are inferred from this — no separate participants file needed.
    preferences: pd.DataFrame

    # Availability: DataFrame with columns [presenter_id, unavailable_timeslot]
    availability: pd.DataFrame

    # Derived data (populated by validate())
    preference_matrix: Dict[str, Set[str]] = field(default_factory=dict)
    presenter_unavailability: Dict[str, Set[str]] = field(default_factory=dict)

    # Room capacities: {room_id: capacity}
    room_capacities: Dict[str, int] = field(default_factory=dict)

    def total_talk_slots(self) -> int:
        """Calculate total number of talk slots from block types."""
        return sum(
            bt["n"] * bt["k"] * bt["count"]
            for bt in self.block_types.values()
        )

    def validate(self) -> List[str]:
        """
        Validate data integrity. Returns list of error messages (empty if valid).
        Also populates derived data structures.
        """
        errors = []

        # Check total slots match number of talks
        total_slots = self.total_talk_slots()
        num_talks = len(self.talks)
        if total_slots != num_talks:
            errors.append(
                f"Slot count mismatch: {total_slots} slots available, "
                f"but {num_talks} talks provided. "
                f"Extra slots will be filled with placeholders automatically."
            )

        # Check timeslot counts match block type counts
        timeslot_counts = {}
        for ts in self.timeslots:
            type_id = ts["type_id"]
            timeslot_counts[type_id] = timeslot_counts.get(type_id, 0) + 1

        for type_id, bt in self.block_types.items():
            expected = bt["count"]
            actual = timeslot_counts.get(type_id, 0)
            if expected != actual:
                errors.append(
                    f"Block type '{type_id}': expected {expected} timeslots, "
                    f"found {actual}"
                )

        # Check timeslot rooms match block type n
        for ts in self.timeslots:
            type_id = ts["type_id"]
            expected_n = self.block_types[type_id]["n"]
            actual_n = len(ts["rooms"])
            if expected_n != actual_n:
                errors.append(
                    f"Timeslot '{ts['id']}': type '{type_id}' expects {expected_n} rooms, "
                    f"but {actual_n} rooms specified"
                )

        # Check all rooms in timeslots are defined
        all_rooms = set(self.rooms)
        for ts in self.timeslots:
            for room in ts["rooms"]:
                if room not in all_rooms:
                    errors.append(
                        f"Timeslot '{ts['id']}' references undefined room '{room}'"
                    )

        # Check preferences reference valid talks
        pref_talks = set(self.preferences["talk_id"].unique())
        talk_ids = set(self.talks["talk_id"].unique())

        invalid_pref_talks = pref_talks - talk_ids
        if invalid_pref_talks:
            errors.append(
                f"Preferences reference unknown talks: {invalid_pref_talks}"
            )

        # Check availability references valid timeslots
        if len(self.availability) > 0:
            avail_timeslots = set(
                self.availability["unavailable_timeslot"].unique())
            timeslot_ids = {ts["id"] for ts in self.timeslots}

            invalid_avail_timeslots = avail_timeslots - timeslot_ids
            if invalid_avail_timeslots:
                errors.append(
                    f"Availability references unknown timeslots: {invalid_avail_timeslots}"
                )

        # Always build derived data structures (even with validation errors)
        # so that the data can still be used with warnings
        self._build_preference_matrix()
        self._build_presenter_unavailability()

        return errors

    def _build_preference_matrix(self):
        """Build participant -> set of preferred talk_ids mapping."""
        self.preference_matrix = {}
        for _, row in self.preferences.iterrows():
            p_id = row["participant_id"]
            t_id = row["talk_id"]
            if p_id not in self.preference_matrix:
                self.preference_matrix[p_id] = set()
            self.preference_matrix[p_id].add(t_id)

    def _build_presenter_unavailability(self):
        """Build presenter -> set of unavailable timeslot_ids mapping."""
        self.presenter_unavailability = {}
        for _, row in self.availability.iterrows():
            pres_id = row["presenter_id"]
            ts_id = row["unavailable_timeslot"]
            if pres_id not in self.presenter_unavailability:
                self.presenter_unavailability[pres_id] = set()
            self.presenter_unavailability[pres_id].add(ts_id)


def load_from_csv_dir(
    data_dir: str | Path,
    block_types: Optional[Dict[str, Dict]] = None,
    verbose: bool = False,
) -> "ConferenceData":
    """
    Load conference data from the standardized CSV format.

    This is the canonical data loader. It expects a directory with:
        talks.csv        (required): talk_id, presenter_id, keywords
        preferences.csv  (required): participant_id, talk_id
        sessions.csv     (required): session_id, n_rooms, n_talks_per_room
        availability.csv (optional): presenter_id, unavailable_timeslot

    IDs in the CSV files are plain integers (1, 2, ...).
    They are converted to T001/P001 format internally.

    Args:
        data_dir: Path to directory containing the CSV files.
        block_types: Override block types; if None, loaded from sessions.csv.
        verbose: Print loading progress.

    Returns:
        A populated ConferenceData instance.
    """
    data_path = Path(data_dir)

    # --- talks ---
    talks_df = pd.read_csv(data_path / "talks.csv")
    talks_df = talks_df.rename(columns={"keywords": "track"})
    talks_df["talk_id"] = talks_df["talk_id"].apply(lambda x: f"T{int(x):03d}")
    talks_df["presenter_id"] = talks_df["presenter_id"].apply(
        lambda x: f"P{int(x):03d}")
    talks_df["title"] = talks_df["talk_id"]
    talks_df["track"] = talks_df["track"].fillna("General")
    if verbose:
        print(f"  Loaded {len(talks_df)} talks")

    # --- preferences ---
    preferences_df = pd.read_csv(data_path / "preferences.csv")
    preferences_df["participant_id"] = preferences_df["participant_id"].apply(
        lambda x: f"P{int(x):03d}"
    )
    preferences_df["talk_id"] = preferences_df["talk_id"].apply(
        lambda x: f"T{int(x):03d}"
    )

    n_participants = preferences_df["participant_id"].nunique()
    if verbose:
        print(
            f"  Loaded {len(preferences_df)} preferences from {n_participants} participants")

    # --- sessions / block types ---
    if block_types is None:
        sessions_file = data_path / "sessions.csv"
        if not sessions_file.exists():
            raise FileNotFoundError(
                f"No sessions.csv found in {data_dir}. "
                "This file defines the conference structure (rooms × talks per session)."
            )
        sessions_df = pd.read_csv(sessions_file)
        block_types = {}
        for _, row in sessions_df.iterrows():
            sid = str(row["session_id"]).strip()
            block_types[sid] = {
                "n": int(row["n_rooms"]),
                "k": int(row["n_talks_per_room"]),
                "count": 1,
                "label": sid,
            }
        if verbose:
            print(f"  Loaded {len(block_types)} sessions from sessions.csv")

    max_rooms = max(bt["n"] for bt in block_types.values())
    room_list = [f"Room_{i + 1}" for i in range(max_rooms)]

    # Merge block types by shape so Phase 3 can reorder sessions of the same size
    type_map: Dict[Tuple[int, int], str] = {}
    merged_block_types: Dict[str, Dict] = {}
    for block_name, bt in block_types.items():
        n, k = bt["n"], bt["k"]
        key = (n, k)
        if key not in type_map:
            type_id = f"{n}R{k}T"
            type_map[key] = type_id
            merged_block_types[type_id] = {"n": n, "k": k, "count": 0}
        merged_block_types[type_map[key]]["count"] += 1

    timeslots = []
    for block_name, bt in block_types.items():
        type_id = type_map[(bt["n"], bt["k"])]
        timeslots.append(
            {
                "id": block_name,
                "start_time": block_name,
                "type_id": type_id,
                "rooms": room_list[: bt["n"]],
            }
        )

    # --- availability (optional) ---
    availability_file = data_path / "availability.csv"
    if availability_file.exists():
        availability_df = pd.read_csv(availability_file)
        if availability_df["presenter_id"].dtype in ("int64", "float64"):
            availability_df["presenter_id"] = availability_df["presenter_id"].apply(
                lambda x: f"P{int(x):03d}"
            )
        if verbose:
            print(f"  Loaded {len(availability_df)} availability constraints")
    else:
        availability_df = pd.DataFrame(
            columns=["presenter_id", "unavailable_timeslot"])

    return ConferenceData(
        conference_name="Conference",
        rooms=room_list,
        block_types=merged_block_types,
        timeslots=timeslots,
        talks=talks_df,
        preferences=preferences_df,
        availability=availability_df,
        room_capacities={},
    )


def load_sessions_from_excel(excel_path: str | Path) -> Dict:
    """
    Load conference session format from an Excel file.

    Expected columns:
    - Session key: Unique identifier for the session/timeslot (e.g., "TA", "TB")
    - number of rooms: Number of parallel rooms for this session
    - number of talks per room: Number of talks in each room (k)
    - total talks: (optional) Computed as rooms * talks per room

    Returns a dict compatible with format.json structure:
    {
        "conference_name": str,
        "rooms": List[str],
        "block_types": List[Dict],
        "timeslots": List[Dict]
    }
    """
    excel_path = Path(excel_path)
    df = pd.read_excel(excel_path, engine='openpyxl')

    # Normalize column names (strip whitespace, lowercase)
    df.columns = df.columns.str.strip().str.lower()

    # Map expected column names
    col_mapping = {
        'session key': 'session_key',
        'number of rooms': 'n_rooms',
        'number of talks per room': 'n_talks',
        'total talks': 'total_talks'
    }
    df = df.rename(columns=col_mapping)

    # Drop rows where session_key is NaN (empty rows)
    df = df.dropna(subset=['session_key'])

    # Convert numeric columns
    df['n_rooms'] = df['n_rooms'].astype(int)
    df['n_talks'] = df['n_talks'].astype(int)

    # Determine max rooms needed
    max_rooms = df['n_rooms'].max()
    # Room A, Room B, ...
    rooms = [f"Room {chr(65 + i)}" for i in range(max_rooms)]

    # Build block_types by grouping unique (n_rooms, n_talks) combinations
    block_types_list = []
    block_type_map = {}  # (n, k) -> type_id

    for _, row in df.iterrows():
        n = int(row['n_rooms'])
        k = int(row['n_talks'])
        key = (n, k)
        if key not in block_type_map:
            type_id = f"{n}R{k}T"
            block_type_map[key] = type_id
            block_types_list.append({
                "type_id": type_id,
                "n": n,
                "k": k,
                "count": 0
            })

    # Count occurrences of each block type
    for _, row in df.iterrows():
        n = int(row['n_rooms'])
        k = int(row['n_talks'])
        type_id = block_type_map[(n, k)]
        for bt in block_types_list:
            if bt["type_id"] == type_id:
                bt["count"] += 1
                break

    # Build timeslots
    timeslots = []
    for _, row in df.iterrows():
        session_key = str(row['session_key']).strip()
        n = int(row['n_rooms'])
        k = int(row['n_talks'])
        type_id = block_type_map[(n, k)]

        timeslots.append({
            "id": session_key,
            "start_time": session_key,  # Use session key as placeholder time
            "type_id": type_id,
            "rooms": rooms[:n]  # Use first n rooms
        })

    return {
        "conference_name": excel_path.stem,
        "rooms": rooms,
        "block_types": block_types_list,
        "timeslots": timeslots
    }


def load_conference_data(
    input_dir: str | Path,
    sessions_excel: Optional[str | Path] = None
) -> ConferenceData:
    """
    Load all conference data from input directory.

    Expected files:
    - format.json: Conference structure (or use sessions_excel parameter)
    - talks.csv: Talk information
    - preferences.csv: Participant preferences
    - availability.csv: Presenter availability constraints

    Args:
        input_dir: Path to directory containing input CSV files
        sessions_excel: Optional path to Excel file with session format.
                       If provided, takes precedence over format.json.
    """
    input_dir = Path(input_dir)

    # Load format data from Excel or JSON
    if sessions_excel is not None:
        format_data = load_sessions_from_excel(sessions_excel)
    else:
        with open(input_dir / "format.json", "r") as f:
            format_data = json.load(f)

    # Parse block_types into dict keyed by type_id
    block_types = {
        bt["type_id"]: {"n": bt["n"], "k": bt["k"], "count": bt["count"]}
        for bt in format_data["block_types"]
    }

    # Load CSVs
    talks = pd.read_csv(input_dir / "talks.csv")
    preferences = pd.read_csv(input_dir / "preferences.csv")

    # Availability might be empty
    availability_path = input_dir / "availability.csv"
    if availability_path.exists():
        availability = pd.read_csv(availability_path)
    else:
        availability = pd.DataFrame(
            columns=["presenter_id", "unavailable_timeslot"])

    return ConferenceData(
        conference_name=format_data["conference_name"],
        rooms=format_data["rooms"],
        block_types=block_types,
        timeslots=format_data["timeslots"],
        talks=talks,
        preferences=preferences,
        availability=availability,
    )


def load_talk_metadata_from_abstracts(
    abstracts_path: str | Path,
    talk_id_format: str = "T{:03d}"
) -> Dict[str, Dict]:
    """
    Load talk metadata from Abstracts.xlsx file.

    Extracts:
    - Paper ID → talk_id (formatted as T001, T002, etc.)
    - Paper Title → title
    - Primary Contact Author Name → primary_contact_author
    - Author Names → author_names

    Args:
        abstracts_path: Path to Abstracts.xlsx
        talk_id_format: Format string for talk ID (default: "T{:03d}")

    Returns:
        Dict mapping talk_id -> {title, primary_contact_author, author_names}
    """
    abstracts_path = Path(abstracts_path)
    df = pd.read_excel(abstracts_path, engine='openpyxl')

    metadata = {}
    for _, row in df.iterrows():
        paper_id = row.get('Paper ID')
        if pd.isna(paper_id):
            continue

        talk_id = talk_id_format.format(int(paper_id))

        metadata[talk_id] = {
            'paper_id': int(paper_id),
            'title': str(row.get('Paper Title', '')).strip() if pd.notna(row.get('Paper Title')) else '',
            'primary_contact_author': str(row.get('Primary Contact Author Name', '')).strip() if pd.notna(row.get('Primary Contact Author Name')) else '',
            'author_names': str(row.get('Author Names', '')).strip() if pd.notna(row.get('Author Names')) else '',
            'track_name': str(row.get('Track Name', '')).strip() if pd.notna(row.get('Track Name')) else '',
        }

    return metadata
