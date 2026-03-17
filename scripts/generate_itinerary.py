#!/usr/bin/env python
"""
Generate a personal itinerary for a conference participant.

Shows which preferred talks the participant can attend, flags parallel
conflicts, and tracks room switches across blocks.

Usage:
    # Generate itinerary for participant 42
    python scripts/generate_itinerary.py output/schedule.csv \
        --participant 42 \
        --data-dir examples/orbel2026

    # Save to file
    python scripts/generate_itinerary.py output/schedule.csv \
        --participant 42 \
        --data-dir examples/orbel2026 \
        --output output/itinerary_42.md
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phase4 import ScheduleResult
from src.schedule_evaluator import load_preferences_from_csv


def main():
    parser = argparse.ArgumentParser(
        description="Generate a personal conference itinerary"
    )
    parser.add_argument(
        "schedule",
        help="Path to schedule CSV file"
    )
    parser.add_argument(
        "--participant", "-p",
        required=True,
        help="Participant ID (as it appears in preferences.csv)"
    )
    parser.add_argument(
        "--data-dir", "-d",
        help="Path to data directory containing preferences.csv"
    )
    parser.add_argument(
        "--preferences",
        help="Path to preferences CSV (alternative to --data-dir)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output markdown file (default: print to stdout)"
    )

    args = parser.parse_args()

    # Load schedule
    result = ScheduleResult.from_csv(args.schedule)

    # Load preferences
    prefs_path = args.preferences
    if not prefs_path and args.data_dir:
        prefs_path = str(Path(args.data_dir) / "preferences.csv")

    if not prefs_path:
        print("Error: provide --data-dir or --preferences", file=sys.stderr)
        sys.exit(1)

    preferences = load_preferences_from_csv(prefs_path)

    # Generate itinerary
    itinerary = result.generate_personal_itinerary(
        str(args.participant), preferences)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(itinerary)
        print(f"Itinerary saved to {output_path}")
    else:
        print(itinerary)


if __name__ == "__main__":
    main()
