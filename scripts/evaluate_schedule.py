#!/usr/bin/env python
"""
Evaluate a conference schedule.

This script loads a schedule and computes quality metrics:
1. Total missed attendance (parallel conflicts)
2. Session hops required to attend preferences  
3. Incoherent sessions (no shared keywords)
4. Presenter violations (unavailable timeslots)

Usage:
    # Evaluate against a data directory (preferences, talks, availability)
    python scripts/evaluate_schedule.py output/schedule.csv \\
        --data-dir examples/orbel2026

    # Or point to the individual files
    python scripts/evaluate_schedule.py output/schedule.csv \\
        --preferences examples/orbel2026/preferences.csv \\
        --talks examples/orbel2026/talks.csv

    # Full evaluation with verbose output and JSON export
    python scripts/evaluate_schedule.py output/schedule.csv \\
        --data-dir examples/orbel2026 \\
        --output-json output/evaluation_metrics.json -v

Presenter IDs are read from talks.csv. Without --talks/--data-dir they are
derived from Talk_IDs (T029 -> P029).
"""

import sys
import argparse
import json
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schedule_evaluator import (  # noqa: E402
    ScheduleEvaluator,
    load_availability_from_csv,
    load_keywords_from_csv,
    load_preferences_from_csv,
    load_schedule_csv,
    load_talk_presenters_from_csv,
)


def load_constraints_from_excel(
    excel_path: str,
    presenter_names: dict,
    verbose: bool = False
) -> dict:
    """
    Load presenter unavailability from Excel constraints file.

    Args:
        excel_path: Path to 'constraints for the schedule.xlsx'
        presenter_names: Mapping of presenter_id -> name for matching
        verbose: Print debug info

    Returns:
        Dict of presenter_id -> set of unavailable session IDs
    """
    if not Path(excel_path).exists():
        return {}

    try:
        constraints_df = pd.read_excel(excel_path)
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not load constraints: {e}")
        return {}

    # Build name -> unavailable sessions from Excel
    name_unavailability = {}
    for _, row in constraints_df.iterrows():
        participant = row.get('Unnamed: 1')
        sessions_to_avoid = row.get('Unnamed: 5')

        if pd.notna(participant) and pd.notna(sessions_to_avoid):
            participant = str(participant).strip()
            sessions_str = str(sessions_to_avoid).strip()
            if sessions_str.lower() not in ['nan', 'sessions to avoid', 'all']:
                sessions = [s.strip()
                            for s in sessions_str.split(',') if s.strip()]
                if sessions and participant not in ['participant', 'NaN']:
                    name_unavailability[participant] = set(sessions)

    if verbose and name_unavailability:
        print(
            f"  Loaded {len(name_unavailability)} presenter constraints from Excel")

    # Match by name to get presenter_id -> unavailable sessions
    presenter_id_unavailability = {}
    for p_id, p_name in presenter_names.items():
        if not p_name:
            continue
        # Try exact match first
        if p_name in name_unavailability:
            presenter_id_unavailability[p_id] = name_unavailability[p_name]
        else:
            # Try partial match (last name)
            p_last = p_name.split()[-1].lower() if p_name else ""
            for constraint_name, sessions in name_unavailability.items():
                c_last = constraint_name.split(
                )[-1].lower() if constraint_name else ""
                if p_last and c_last and p_last == c_last:
                    presenter_id_unavailability[p_id] = sessions
                    break

    return presenter_id_unavailability


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a conference schedule quality"
    )
    parser.add_argument(
        "schedule",
        help="Path to schedule CSV file"
    )
    parser.add_argument(
        "--data-dir", "-d",
        help="Input data directory; supplies preferences.csv, talks.csv and "
             "availability.csv unless given explicitly"
    )
    parser.add_argument(
        "--preferences",
        help="Path to preferences CSV (participant_id, talk_id format)"
    )
    parser.add_argument(
        "--talks",
        help="Path to talks CSV (talk_id, presenter_id, keywords)"
    )
    parser.add_argument(
        "--availability",
        help="Path to availability CSV (presenter_id, unavailable_timeslot)"
    )
    parser.add_argument(
        "--constraints",
        help="Path to Excel constraints file (constraints for the schedule.xlsx)"
    )
    parser.add_argument(
        "--output-json",
        help="Save metrics to JSON file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output including per-participant breakdown"
    )

    args = parser.parse_args()

    # Fill in file paths from --data-dir
    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not args.preferences and (data_dir / "preferences.csv").exists():
            args.preferences = str(data_dir / "preferences.csv")
        if not args.talks and (data_dir / "talks.csv").exists():
            args.talks = str(data_dir / "talks.csv")
        if not args.availability and (data_dir / "availability.csv").exists():
            args.availability = str(data_dir / "availability.csv")

    # Load schedule
    print(f"Loading schedule from {args.schedule}...")
    schedule_df = load_schedule_csv(args.schedule)
    print(f"  Found {len(schedule_df)} talk assignments")
    print(f"  Timeslots: {schedule_df['Session_ID'].nunique()}")
    print(f"  Rooms: {schedule_df['Room'].nunique()}")

    # Load preferences (pass schedule_df for title->talk_id mapping in matrix format)
    preferences = {}
    if args.preferences:
        print(f"\nLoading preferences from {args.preferences}...")
        try:
            preferences = load_preferences_from_csv(
                args.preferences, schedule_df=schedule_df)
            print(f"  Found preferences for {len(preferences)} participants")
            total_prefs = sum(len(p) for p in preferences.values())
            print(f"  Total preference entries: {total_prefs}")
        except Exception as e:
            print(f"  Warning: Could not load preferences: {e}")

    # Load keywords and presenters from talks.csv
    talk_keywords = {}
    talk_presenter = {}
    if args.talks:
        print(f"\nLoading talks from {args.talks}...")
        try:
            talk_keywords, _ = load_keywords_from_csv(args.talks)
            talk_presenter = load_talk_presenters_from_csv(args.talks)
            talks_with_kw = sum(1 for kws in talk_keywords.values() if kws)
            print(
                f"  Found keywords for {talks_with_kw}/{len(talk_keywords)} talks")
        except Exception as e:
            print(f"  Warning: Could not load talks: {e}")

    # Fill in presenters missing from talks.csv (derive from Talk_ID: T029 -> P029)
    presenter_names = {}  # presenter_id -> name for constraint matching
    for _, row in schedule_df.iterrows():
        talk_id = row['Talk_ID']
        if talk_id not in talk_presenter:
            talk_presenter[talk_id] = talk_id.replace(
                'T', 'P', 1) if talk_id.startswith('T') else talk_id
        presenter_id = talk_presenter[talk_id]
        # Get presenter name for constraint matching
        name = row.get('Primary_Contact_Author')
        if pd.notna(name):
            presenter_names[presenter_id] = name

    # Load availability/constraints
    presenter_unavailability = {}
    if args.availability:
        print(f"\nLoading availability from {args.availability}...")
        presenter_unavailability = load_availability_from_csv(
            args.availability)
        print(
            f"  Found constraints for {len(presenter_unavailability)} presenters")
    elif args.constraints:
        print(f"\nLoading constraints from {args.constraints}...")
        presenter_unavailability = load_constraints_from_excel(
            args.constraints, presenter_names, verbose=args.verbose
        )
        print(
            f"  Found constraints for {len(presenter_unavailability)} presenters")
    else:
        # Try auto-detecting constraints file next to preferences
        if args.preferences:
            auto_constraints = Path(
                args.preferences).parent / "constraints for the schedule.xlsx"
            if auto_constraints.exists():
                print(f"\nAuto-detected constraints: {auto_constraints}")
                presenter_unavailability = load_constraints_from_excel(
                    str(auto_constraints), presenter_names, verbose=args.verbose
                )
                print(
                    f"  Found constraints for {len(presenter_unavailability)} presenters")

    # Create evaluator
    print("\n" + "=" * 60)
    print("EVALUATING SCHEDULE")
    print("=" * 60)

    evaluator = ScheduleEvaluator(
        schedule_df=schedule_df,
        preferences=preferences,
        talk_keywords=talk_keywords,
        presenter_unavailability=presenter_unavailability,
        talk_presenter=talk_presenter
    )

    metrics = evaluator.evaluate()
    print(metrics)

    # Verbose output
    if args.verbose:
        print("\n" + "=" * 60)
        print("DETAILED BREAKDOWN")
        print("=" * 60)

        if metrics.missed_attendance_by_participant:
            print("\nMissed Attendance by Participant (top 10):")
            sorted_missed = sorted(
                metrics.missed_attendance_by_participant.items(),
                key=lambda x: -x[1]
            )[:10]
            for p_id, missed in sorted_missed:
                print(f"  {p_id}: {missed} talks missed")

        if metrics.session_hops_by_participant:
            print("\nSession Hops by Participant (top 10):")
            sorted_hops = sorted(
                metrics.session_hops_by_participant.items(),
                key=lambda x: -x[1]
            )[:10]
            for p_id, hops in sorted_hops:
                print(f"  {p_id}: {hops} room switches")

        if metrics.incoherent_session_details:
            print(
                f"\nIncoherent Sessions (showing first 5 of {len(metrics.incoherent_session_details)}):")
            for detail in metrics.incoherent_session_details[:5]:
                print(
                    f"  {detail['timeslot']} / {detail['room']}: talks {detail['talks']}")

        if metrics.presenter_violation_details:
            print(f"\nPresenter Violations:")
            for detail in metrics.presenter_violation_details:
                print(
                    f"  {detail['presenter_id']} at {detail['timeslot']} (unavailable: {detail['unavailable_timeslots']})")

    # Save to JSON
    if args.output_json:
        output_data = metrics.to_dict()
        output_data["schedule_file"] = args.schedule

        with open(args.output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nMetrics saved to {args.output_json}")

    return metrics


if __name__ == "__main__":
    main()
