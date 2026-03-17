#!/usr/bin/env python
"""
Evaluate a conference schedule.

This script loads a schedule and computes quality metrics:
1. Total missed attendance (parallel conflicts)
2. Session hops required to attend preferences  
3. Incoherent sessions (no shared keywords)
4. Presenter violations (unavailable timeslots)

Usage:
    # Evaluate with preferences
    python scripts/evaluate_schedule.py output/schedule.csv \\
        --preferences examples/orbel2026/preferences.csv \\
        --talks examples/orbel2026/talks.csv

    # Quick evaluation (no preferences = just count violations and coherence)
    python scripts/evaluate_schedule.py output/schedule.csv \\
        --talks examples/orbel2026/talks.csv

    # Full evaluation with verbose output
    python scripts/evaluate_schedule.py output/schedule.csv \\
        --preferences examples/orbel2026/preferences.csv \\
        --talks examples/orbel2026/talks.csv \\
        --output-json output/evaluation_metrics.json -v

Note: Presenter IDs are derived from Talk_IDs (T029 -> P029), so no Presenter_ID
column is required in the schedule CSV.
"""

from collections import defaultdict
import pandas as pd
from src.schedule_evaluator import (
    ScheduleEvaluator,
    EvaluationMetrics,
    load_schedule_csv,
    load_keywords_from_csv,
    load_preferences_from_csv,
)
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_availability_from_csv(csv_path: str) -> dict:
    """Load presenter unavailability from CSV."""
    if not Path(csv_path).exists():
        return {}

    df = pd.read_csv(csv_path)
    unavailability = defaultdict(set)

    for _, row in df.iterrows():
        pres_id = str(row['presenter_id'])
        ts_id = str(row['unavailable_timeslot'])
        unavailability[pres_id].add(ts_id)

    return dict(unavailability)


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


def extract_talk_keywords_from_schedule(
    schedule_df: pd.DataFrame,
    talks_csv: str = None,
    verbose: bool = False
) -> dict:
    """
    Build talk_id -> keywords mapping.

    First tries data/output/talks_for_algorithm.csv (has talk_id directly),
    then falls back to title-based matching from talks_csv.
    """
    talk_keywords = {}

    # Try talks_for_algorithm.csv first (same as pipeline)
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
                    f"  Loaded keywords for {len(talk_keywords)} talks from talks_for_algorithm.csv")
            return talk_keywords
        except Exception as e:
            if verbose:
                print(
                    f"  Warning: Could not load from talks_for_algorithm.csv: {e}")

    # Fallback to talks_csv
    if not talks_csv:
        return talk_keywords

    talks_df = pd.read_csv(talks_csv)

    # If talks_csv has a 'keywords' column and numeric talk_ids, map directly
    if 'keywords' in talks_df.columns and 'talk_id' in talks_df.columns:
        try:
            for _, row in talks_df.iterrows():
                talk_id = f"T{int(row['talk_id']):03d}"
                kw_str = str(row.get('keywords', ''))
                if pd.isna(kw_str) or kw_str.lower() == 'nan':
                    keywords = set()
                else:
                    keywords = set(kw.strip()
                                   for kw in kw_str.split(';') if kw.strip())
                talk_keywords[talk_id] = keywords
            if verbose:
                print(f"  Loaded keywords for {len(talk_keywords)} talks")
            return talk_keywords
        except (ValueError, TypeError):
            pass  # Fall through to title-based matching

    # Title-based matching (for files with 'title' and 'master_keywords' columns)
    if 'title' not in talks_df.columns:
        if verbose:
            print("  Warning: Could not load keywords: talks CSV has no 'title' or 'keywords' column")
        return talk_keywords

    title_to_keywords = {}
    for _, row in talks_df.iterrows():
        title = str(row['title']).strip().lower()
        kw_str = str(row.get('master_keywords', ''))
        if pd.isna(kw_str) or kw_str.lower() == 'nan':
            keywords = set()
        else:
            keywords = set(kw.strip()
                           for kw in kw_str.split(';') if kw.strip())
        title_to_keywords[title] = keywords

    # Map talk_id -> keywords using titles from schedule
    if 'Title' in schedule_df.columns:
        for _, row in schedule_df.iterrows():
            talk_id = row['Talk_ID']
            title = str(row['Title']).strip().lower()
            talk_keywords[talk_id] = title_to_keywords.get(title, set())
    else:
        # Assume talk_id like "T001" corresponds to row 1 in talks_csv
        for talk_id in schedule_df['Talk_ID'].unique():
            try:
                idx = int(talk_id[1:]) - 1  # T001 -> index 0
                if 0 <= idx < len(talks_df):
                    title = str(talks_df.iloc[idx]['title']).strip().lower()
                    talk_keywords[talk_id] = title_to_keywords.get(
                        title, set())
            except (ValueError, IndexError):
                talk_keywords[talk_id] = set()

    return talk_keywords


def main():
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

    # Load keywords (tries talks_for_algorithm.csv first, then falls back to --talks)
    talk_keywords = {}
    print(f"\nLoading keywords...")
    try:
        talk_keywords = extract_talk_keywords_from_schedule(
            schedule_df, args.talks, verbose=args.verbose
        )
        talks_with_kw = sum(1 for kws in talk_keywords.values() if kws)
        print(
            f"  Found keywords for {talks_with_kw}/{len(talk_keywords)} talks")
    except Exception as e:
        print(f"  Warning: Could not load keywords: {e}")

    # Build talk_presenter from schedule (derive from Talk_ID: T029 -> P029)
    talk_presenter = {}
    presenter_names = {}  # presenter_id -> name for constraint matching
    for _, row in schedule_df.iterrows():
        talk_id = row['Talk_ID']
        # Derive presenter_id from talk_id (T029 -> P029)
        presenter_id = talk_id.replace(
            'T', 'P') if talk_id.startswith('T') else talk_id
        talk_presenter[talk_id] = presenter_id
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
