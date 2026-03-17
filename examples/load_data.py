"""
Simple data loader for standardized conference scheduling data.

This is a convenience wrapper around src.data_loader.load_from_csv_dir().
It demonstrates how to load data and validate it.

Expected directory structure:
    talks.csv:        talk_id, presenter_id, keywords
    preferences.csv:  participant_id, talk_id (long format, one row per preference)
    sessions.csv:     session_id, n_rooms, n_talks_per_room (conference structure)
    availability.csv: presenter_id, unavailable_timeslot (optional)

Usage:
    python examples/load_data.py examples/orbel2026
    python examples/load_data.py examples/orbel2017
"""

from src.data_loader import ConferenceData, load_from_csv_dir
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_conference_data(data_dir: str, **kwargs):
    """
    Load conference data from standardized CSV files.

    This delegates to src.data_loader.load_from_csv_dir().
    See that function for full documentation.

    Returns:
        Tuple of (ConferenceData, stats_dict)
    """
    data = load_from_csv_dir(data_dir, verbose=True, **kwargs)

    n_participants = data.preferences["participant_id"].nunique()
    stats = {
        "num_talks": len(data.talks),
        "num_participants": n_participants,
        "num_preferences": len(data.preferences),
        "num_blocks": len(data.block_types),
        "avg_preferences_per_participant": (
            len(data.preferences) / n_participants
            if n_participants > 0
            else 0
        ),
    }

    return data, stats


def main():
    """Test loading each example dataset."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and validate conference data")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="examples/orbel2026",
        help="Path to data directory",
    )
    args = parser.parse_args()

    print(f"Loading data from: {args.data_dir}")
    print("-" * 40)

    data, stats = load_conference_data(args.data_dir)

    print("-" * 40)
    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("-" * 40)
    print("Validation:")
    data.validate()
    print("  All checks passed!")

    return 0


if __name__ == "__main__":
    exit(main())
