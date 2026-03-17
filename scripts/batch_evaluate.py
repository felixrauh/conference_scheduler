#!/usr/bin/env python
"""Batch evaluate schedule CSV files with the ScheduleEvaluator for consistent metrics.

Usage:
    # Evaluate all schedule CSVs in a directory
    python scripts/batch_evaluate.py output/comparison/

    # Evaluate specific files
    python scripts/batch_evaluate.py output/schedule_traditional.csv output/schedule_heuristic.csv \\
        --preferences examples/orbel2026/preferences.csv

    # With custom preferences file
    python scripts/batch_evaluate.py output/ --preferences examples/orbel2017/preferences.csv
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schedule_evaluator import ScheduleEvaluator, load_schedule_csv, load_preferences_from_csv


def normalize_talk_id(tid) -> str:
    """Convert talk ID to T### format used in schedules."""
    tid = str(tid).strip()
    if tid.startswith('T'):
        return tid
    try:
        return f"T{int(tid):03d}"
    except ValueError:
        return tid


def evaluate_schedule(sched_path: str, pref_path: str):
    """Evaluate a schedule CSV and return (missed_attendance, session_hops)."""
    schedule_df = load_schedule_csv(sched_path)
    raw_preferences = load_preferences_from_csv(pref_path)

    # Normalize preference talk IDs to T### format
    preferences = {}
    for p_id, talks in raw_preferences.items():
        preferences[p_id] = {normalize_talk_id(t) for t in talks}

    # Build talk_presenter from schedule
    talk_presenter = {}
    for _, row in schedule_df.iterrows():
        talk_id = row['Talk_ID']
        presenter_id = talk_id.replace('T', 'P') if talk_id.startswith('T') else talk_id
        talk_presenter[talk_id] = presenter_id

    evaluator = ScheduleEvaluator(
        schedule_df=schedule_df,
        preferences=preferences,
        talk_keywords={},
        presenter_unavailability={},
        talk_presenter=talk_presenter
    )
    metrics = evaluator.evaluate()
    return metrics.total_missed_attendance, metrics.total_session_hops


def find_schedule_csvs(path: Path):
    """Find schedule CSV files in a path (file or directory)."""
    if path.is_file() and path.suffix == '.csv':
        return [path]
    elif path.is_dir():
        return sorted(path.glob('*.csv'))
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluate schedule CSV files with consistent metrics"
    )
    parser.add_argument(
        "paths", nargs="+",
        help="Schedule CSV files or directories containing them"
    )
    parser.add_argument(
        "--preferences", "-p",
        default="examples/orbel2026/preferences.csv",
        help="Path to preferences CSV (default: examples/orbel2026/preferences.csv)"
    )
    args = parser.parse_args()

    # Collect all schedule files
    schedule_files = []
    for p in args.paths:
        path = Path(p)
        if not path.exists():
            print(f"Warning: {p} does not exist, skipping")
            continue
        schedule_files.extend(find_schedule_csvs(path))

    if not schedule_files:
        print("No schedule CSV files found.")
        sys.exit(1)

    if not Path(args.preferences).exists():
        print(f"Error: preferences file not found: {args.preferences}")
        sys.exit(1)

    # Evaluate each schedule
    print(f"{'File':<50} {'Missed':>8} {'Session Hops':>14}")
    print('-' * 75)

    for sched_path in schedule_files:
        label = str(sched_path)
        try:
            missed, hops = evaluate_schedule(str(sched_path), args.preferences)
            print(f"{label:<50} {missed:>8} {hops:>14}")
        except Exception as e:
            print(f"{label:<50} ERROR: {e}")


if __name__ == "__main__":
    main()
