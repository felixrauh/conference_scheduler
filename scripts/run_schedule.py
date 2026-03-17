#!/usr/bin/env python
"""
Unified Conference Scheduler

Single entry point for all scheduling pipelines.
Configuration can be provided via config file and/or CLI arguments.

Usage:
    # Quick start with defaults (traditional pipeline, example data)
    python scripts/run_schedule.py --data-dir examples/orbel2026

    # Specify pipeline
    python scripts/run_schedule.py --pipeline matching --data-dir examples/orbel2026
    python scripts/run_schedule.py --pipeline heuristic --data-dir examples/orbel2026

    # Use config file (overrides defaults, CLI overrides config)
    python scripts/run_schedule.py --config config/scheduling_config.jsonc

    # Full example
    python scripts/run_schedule.py \\
        --pipeline traditional \\
        --data-dir examples/orbel2026 \\
        --output output/schedule.csv \\
        --verbose

Pipelines:
    traditional  - 3-phase optimization (Phase 1 → 2 → 3), requires Gurobi
    heuristic    - Greedy alternative, no Gurobi needed
    matching     - Bottom-up matching (Phase A → B → C → D → 3), requires Gurobi

Input Data:
    Expects a directory with: talks.csv, preferences.csv, sessions.csv
    See examples/orbel2026/ for the expected format.

See config/scheduling_config.jsonc for all configuration options.
"""

import sys
import time
import json
import re
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SchedulerConfig:
    """Complete scheduler configuration."""
    # Pipeline selection
    pipeline: str = "traditional"

    # Data configuration
    data_dir: str = "examples/orbel2026"

    # Output configuration
    output_file: str = "output/schedule.csv"
    verbose: bool = False

    # Block types (None = auto-detect from data)
    block_types: Optional[Dict] = None

    # Phase 1 settings
    phase1_method: str = "column_generation"
    phase1_time_limit: float = 300.0
    phase1_pricing_strategy: str = "auto"
    # For explicit method: tuple filtering threshold
    phase1_explicit_max_cost: Optional[int] = None

    # Phase 2 settings
    phase2_partition_strategy: str = "greedy"
    phase2_ordering_strategy: str = "enumerate"
    phase2_local_search_iterations: int = 2000
    # Fixed sequences: pre-specified sequential talks for special sessions
    # List of dicts with: name, talks, target_block_type, result_block_type
    fixed_sequences: Optional[List[Dict]] = None

    # Phase 3 settings
    phase3_method: str = "milp"
    phase3_time_limit: float = 60.0

    # Phase 4 settings (finalization)
    phase4_verbose: bool = True
    # Fixed block sessions: pre-specified sessions for specific blocks
    # List of dicts with: name, block, talks
    fixed_block_sessions: Optional[List[Dict]] = None
    # Room configuration: list of dicts with id, name, capacity
    rooms: Optional[List[Dict]] = None

    # Swap optimization (between Phase 3 and Phase 4)
    swap_enabled: bool = True
    swap_keyword_weight: float = 0.1
    swap_max_iterations: int = 100
    swap_check_dummy_violations: bool = True
    swap_short_block_threshold: int = 3

    # Matching pipeline settings
    matching_time_limit: float = 300.0
    require_common_keyword: bool = False  # For constrained matching
    # Soft constraint mode (None = hard constraint)
    max_keyword_violations: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict) -> 'SchedulerConfig':
        """Create config from dictionary (e.g., parsed JSON)."""
        config = cls()

        if 'pipeline' in d:
            config.pipeline = d['pipeline']

        if 'input' in d:
            inp = d['input']
            config.data_dir = inp.get('data_dir', config.data_dir)

        if 'output' in d:
            out = d['output']
            if 'schedule_csv' in out:
                config.output_file = str(
                    Path(out.get('dir', 'output')) / out['schedule_csv'])

        if 'block_types' in d and d['block_types'] != "auto":
            config.block_types = d['block_types']

        if 'phase1' in d:
            p1 = d['phase1']
            config.phase1_method = p1.get('method', config.phase1_method)
            config.phase1_time_limit = p1.get(
                'time_limit', config.phase1_time_limit)
            if 'explicit' in p1:
                config.phase1_explicit_max_cost = p1['explicit'].get(
                    'max_cost', config.phase1_explicit_max_cost)
            if 'column_generation' in p1:
                config.phase1_pricing_strategy = p1['column_generation'].get(
                    'pricing_strategy', config.phase1_pricing_strategy)

        if 'phase2' in d:
            p2 = d['phase2']
            config.phase2_partition_strategy = p2.get(
                'partition_strategy', config.phase2_partition_strategy)
            config.phase2_ordering_strategy = p2.get(
                'ordering_strategy', config.phase2_ordering_strategy)
            config.phase2_local_search_iterations = p2.get(
                'local_search_iterations', config.phase2_local_search_iterations)
            # Parse fixed sequences if present
            if 'fixed_sequences' in p2 and p2['fixed_sequences']:
                config.fixed_sequences = p2['fixed_sequences']

        if 'phase3' in d:
            p3 = d['phase3']
            config.phase3_method = p3.get('method', config.phase3_method)
            config.phase3_time_limit = p3.get(
                'time_limit', config.phase3_time_limit)

        if 'phase4' in d:
            p4 = d['phase4']
            # Parse fixed block sessions if present
            if 'fixed_block_sessions' in p4 and p4['fixed_block_sessions']:
                config.fixed_block_sessions = p4['fixed_block_sessions']

        # Parse rooms configuration
        if 'rooms' in d and d['rooms']:
            config.rooms = d['rooms']

        if 'swap_optimization' in d:
            sw = d['swap_optimization']
            config.swap_enabled = sw.get('enabled', config.swap_enabled)
            config.swap_keyword_weight = sw.get(
                'keyword_weight', config.swap_keyword_weight)
            config.swap_max_iterations = sw.get(
                'max_iterations', config.swap_max_iterations)
            config.swap_check_dummy_violations = sw.get(
                'check_dummy_violations', config.swap_check_dummy_violations)
            config.swap_short_block_threshold = sw.get(
                'short_block_threshold', config.swap_short_block_threshold)

        if 'matching_pipeline' in d:
            mp = d['matching_pipeline']
            if 'time_limit' in mp:
                config.matching_time_limit = mp['time_limit']
            else:
                # Legacy: sum per-phase time limits
                total_time = sum(
                    mp.get(phase, {}).get('time_limit', 60.0)
                    for phase in ['phase_a', 'phase_b', 'phase_c', 'phase_d']
                )
                if total_time > 0:
                    config.matching_time_limit = total_time

        return config


def load_jsonc(path: str) -> Dict:
    """Load JSONC (JSON with comments) file."""
    with open(path, 'r') as f:
        content = f.read()
    content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    content = re.sub(r',\s*([}\]])', r'\1', content)
    return json.loads(content)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Conference Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_schedule.py --data-dir examples/orbel2026
  python scripts/run_schedule.py --pipeline matching --data-dir examples/orbel2026
  python scripts/run_schedule.py --pipeline heuristic --data-dir examples/orbel2017
  python scripts/run_schedule.py --config config/scheduling_config.jsonc
        """
    )

    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to configuration file (JSONC format)")
    parser.add_argument("--pipeline", "-p", type=str,
                        choices=["traditional", "heuristic",
                                 "matching", "matching_constrained"],
                        default=None,
                        help="Scheduling pipeline to use (default: traditional)")
    parser.add_argument("--data-dir", "-d", type=str, default=None,
                        help="Directory containing input CSV files (talks.csv, preferences.csv, sessions.csv)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output CSV file path")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--phase1-method", type=str, choices=["explicit", "column_generation"],
                        default=None, help="Phase 1 method (traditional pipeline only)")
    parser.add_argument("--phase2-partition", type=str, choices=["greedy", "matching", "random"],
                        default=None, help="Phase 2 partition strategy (traditional pipeline only)")
    parser.add_argument("--phase3-method", type=str, choices=["milp", "hungarian"],
                        default=None, help="Phase 3 method")
    parser.add_argument("--time-limit", "-t", type=float, default=None,
                        help="Overall time limit in seconds")
    parser.add_argument("--require-keyword", action="store_true",
                        help="Require common keyword for matching (matching_constrained only)")
    parser.add_argument("--max-violations", type=int, default=None,
                        help="Max keyword violations allowed (soft constraint mode, 0 = start with 0 and increment)")

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SchedulerConfig:
    """Build configuration from config file and CLI arguments."""
    config = SchedulerConfig()

    if args.config:
        try:
            config = SchedulerConfig.from_dict(load_jsonc(args.config))
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")

    # CLI overrides
    if args.pipeline:
        config.pipeline = args.pipeline
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output:
        config.output_file = args.output
    if args.verbose:
        config.verbose = True
    if args.phase1_method:
        config.phase1_method = args.phase1_method
    if args.phase2_partition:
        config.phase2_partition_strategy = args.phase2_partition
    if args.phase3_method:
        config.phase3_method = args.phase3_method
    if args.time_limit:
        config.phase1_time_limit = args.time_limit
        config.matching_time_limit = args.time_limit
    if hasattr(args, 'require_keyword') and args.require_keyword:
        config.require_common_keyword = True
    if hasattr(args, 'max_violations') and args.max_violations is not None:
        config.max_keyword_violations = args.max_violations
        # Soft constraint implies keyword constraint
        config.require_common_keyword = True

    return config


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(config: SchedulerConfig):
    """Load conference data from standardized CSV directory."""
    from src.data_loader import load_from_csv_dir
    return load_from_csv_dir(
        data_dir=config.data_dir,
        block_types=config.block_types,
        verbose=config.verbose,
    )


def adjust_block_types(data, n_talks: int, verbose: bool = False):
    """Verify block types match actual talk count, raise if they don't."""
    current_slots = sum(bt["n"] * bt["k"] * bt["count"]
                        for bt in data.block_types.values())

    if current_slots == n_talks:
        return

    raise ValueError(
        f"sessions.csv defines {current_slots} total talk slots "
        f"but talks.csv contains {n_talks} talks. "
        f"Please update sessions.csv so the totals match."
    )


# =============================================================================
# PIPELINE DISPATCH
# =============================================================================

def run_pipeline(config: SchedulerConfig, instance, data):
    """Dispatch to appropriate pipeline."""
    if config.pipeline == "traditional":
        from src.pipelines.traditional import run_traditional_pipeline
        from src.pipelines.traditional import PipelineConfig as TraditionalConfig
        from src.pipelines.traditional import FixedSequenceConfig

        # Convert fixed_sequences from dict to FixedSequenceConfig objects
        fixed_sequences = None
        if config.fixed_sequences:
            fixed_sequences = []
            for fs in config.fixed_sequences:
                fixed_sequences.append(FixedSequenceConfig(
                    name=fs.get('name', 'FixedSequence'),
                    talks=fs['talks'],
                    target_block_type=fs['target_block_type'],
                    result_block_type=fs['result_block_type']
                ))

        pipeline_config = TraditionalConfig(
            phase1_method=config.phase1_method,
            phase1_time_limit=config.phase1_time_limit,
            phase1_pricing_strategy=config.phase1_pricing_strategy,
            phase1_explicit_max_cost=config.phase1_explicit_max_cost,
            phase2_partition_strategy=config.phase2_partition_strategy,
            phase2_ordering_strategy=config.phase2_ordering_strategy,
            phase2_local_search_iterations=config.phase2_local_search_iterations,
            phase3_method=config.phase3_method,
            phase3_time_limit=config.phase3_time_limit,
            verbose=config.verbose,
            fixed_sequences=fixed_sequences
        )
        return run_traditional_pipeline(pipeline_config, instance, data)

    elif config.pipeline == "heuristic":
        from src.pipelines.heuristic import run_heuristic_pipeline
        from src.pipelines.heuristic import PipelineConfig as HeuristicConfig
        from src.pipelines.heuristic import FixedSequenceConfig as HeuristicFixedSequenceConfig
        from src.pipelines.traditional import FixedSequenceConfig as TraditionalFixedSequenceConfig

        # Convert fixed_sequences to HeuristicFixedSequenceConfig format
        # Handle both dict and TraditionalFixedSequenceConfig inputs
        heuristic_fixed_sequences = None
        if config.fixed_sequences:
            heuristic_fixed_sequences = []
            for fs in config.fixed_sequences:
                if isinstance(fs, dict):
                    heuristic_fixed_sequences.append(HeuristicFixedSequenceConfig(
                        name=fs['name'],
                        talks=fs['talks'],
                        target_block_type=fs['target_block_type'],
                        result_block_type=fs['result_block_type']
                    ))
                elif isinstance(fs, TraditionalFixedSequenceConfig):
                    heuristic_fixed_sequences.append(HeuristicFixedSequenceConfig(
                        name=fs.name,
                        talks=fs.talks,
                        target_block_type=fs.target_block_type,
                        result_block_type=fs.result_block_type
                    ))
                else:
                    # Assume it's already a HeuristicFixedSequenceConfig
                    heuristic_fixed_sequences.append(fs)

        pipeline_config = HeuristicConfig(
            phase1_time_limit=config.phase1_time_limit,
            phase2_partition_strategy="matching",
            phase2_ordering_strategy=config.phase2_ordering_strategy,
            phase2_local_search_iterations=100,
            verbose=config.verbose,
            fixed_sequences=heuristic_fixed_sequences
        )
        return run_heuristic_pipeline(pipeline_config, instance, data)

    elif config.pipeline == "matching":
        from src.matching_pipeline import run_matching_pipeline
        from src.instance import build_instance

        # Add placeholder talks so talk count matches slot count from sessions.csv
        # (don't use adjust_block_types — matching pipeline needs original session structure)
        total_slots = sum(bt["n"] * bt["k"] * bt["count"]
                          for bt in data.block_types.values())
        n_real_talks = len(instance.talks)
        if n_real_talks < total_slots:
            import pandas as pd
            n_placeholders = total_slots - n_real_talks
            if config.verbose:
                print(
                    f"  Adding {n_placeholders} placeholder talks ({n_real_talks} talks, {total_slots} slots)")
            placeholder_rows = []
            for i in range(n_placeholders):
                pid = f"PLACEHOLDER_{i+1}"
                placeholder_rows.append({
                    "talk_id": pid, "presenter_id": pid,
                    "title": pid, "track": "Placeholder"
                })
            data.talks = pd.concat(
                [data.talks, pd.DataFrame(placeholder_rows)], ignore_index=True
            )
        instance = build_instance(data)

        print("\n[Running Matching Pipeline: A → B → C → D → 3]")

        pipeline_result, phase3_result = run_matching_pipeline(
            instance,
            time_limit=config.matching_time_limit,
            verbose=config.verbose,
            run_phase3=True,
            phase3_method=config.phase3_method
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

    elif config.pipeline == "matching_constrained":
        from src.matching_pipeline_constrained import (
            run_constrained_matching_pipeline,
            MatchingConstraints, TalkMetadata
        )
        from src.instance import build_instance

        adjust_block_types(data, len(instance.talks), config.verbose)
        instance = build_instance(data)

        # Build talk metadata from data
        metadata = {}
        for _, row in data.talks.iterrows():
            talk_id = row['talk_id']
            kw_str = str(row.get('keywords', ''))
            keywords = set(kw.strip()
                           for kw in kw_str.split(';') if kw.strip())
            group = row.get('special_group', 'general')
            metadata[talk_id] = TalkMetadata(
                talk_id=talk_id, special_group=group, keywords=keywords)

        # Constraints: same group required, common keyword optional
        constraints = MatchingConstraints(
            require_same_group=True,
            require_common_keyword=config.require_common_keyword,
            max_keyword_violations=config.max_keyword_violations
        )

        print("\n[Running Constrained Matching Pipeline]")

        pipeline_result, phase3_result = run_constrained_matching_pipeline(
            instance,
            metadata=metadata,
            constraints=constraints,
            time_limit=config.matching_time_limit,
            verbose=config.verbose,
            run_phase3=True,
            phase3_method=config.phase3_method
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

    else:
        raise ValueError(f"Unknown pipeline: {config.pipeline}")


# =============================================================================
# SWAP OPTIMIZATION (POST-PHASE 3, PRE-PHASE 4)
# =============================================================================

def run_swap_optimization(results, config: SchedulerConfig, instance, data):
    """
    Run swap optimization to resolve presenter unavailability violations.

    Swaps talks between positions to eliminate violations while minimizing
    impact on missed attendance and keyword coherence.
    """
    from src.swap_optimization import optimize_presenter_violations

    phase3_result = results.get('phase3_result')
    if not phase3_result:
        return results

    # Build talk_presenter mapping
    talk_presenter = {}
    for _, row in data.talks.iterrows():
        talk_id = str(row.get('talk_id', ''))
        presenter_id = str(row.get('presenter_id', ''))
        if talk_id and presenter_id:
            talk_presenter[talk_id] = presenter_id

    # Get presenter unavailability from data
    presenter_unavailability = getattr(data, 'presenter_unavailability', {})
    has_presenter_constraints = bool(presenter_unavailability)
    has_dummy_check = config.swap_check_dummy_violations

    if not has_presenter_constraints and not has_dummy_check:
        if config.verbose:
            print("\n[Swap Optimization] No violations to resolve — skipping")
        return results

    # Load talk keywords for coherence scoring (optional)
    talk_keywords = None
    try:
        keywords_col = 'keywords'
        if keywords_col in data.talks.columns:
            talk_keywords = {}
            for _, row in data.talks.iterrows():
                tid = row['talk_id']
                kw_str = str(row.get(keywords_col, ''))
                keywords = set(kw.strip()
                               for kw in kw_str.split(';') if kw.strip())
                if keywords:
                    talk_keywords[tid] = keywords
    except Exception:
        pass

    print("\n[Swap Optimization]")
    swap_result = optimize_presenter_violations(
        phase3_result=phase3_result,
        talk_presenter=talk_presenter,
        presenter_unavailability=presenter_unavailability,
        preferences=instance.preferences,
        talk_keywords=talk_keywords,
        keyword_weight=config.swap_keyword_weight,
        max_iterations=config.swap_max_iterations,
        check_dummy_violations=config.swap_check_dummy_violations,
        short_block_threshold=config.swap_short_block_threshold,
        verbose=config.verbose
    )

    results['phase3_result'] = swap_result.phase3_result
    results['swap_violations_before'] = swap_result.violations_before
    results['swap_violations_resolved'] = swap_result.violations_resolved
    results['swap_violations_remaining'] = swap_result.violations_remaining

    return results


# =============================================================================
# PHASE 4: FINALIZATION
# =============================================================================

def run_phase4(results, config: SchedulerConfig, instance, data):
    """
    Run Phase 4: Finalization & Room Assignment.

    This phase:
    1. Computes session metrics (total_likes, unique_attendees)
    2. Assigns rooms based on audience size
    3. Adds fixed block sessions
    """
    import time
    from pathlib import Path
    from src.phase4 import (
        solve_phase4, Phase4Input, FixedBlockSession
    )
    from src.data_loader import load_talk_metadata_from_abstracts

    phase3_result = results.get('phase3_result')
    if not phase3_result:
        print("No Phase 3 result to finalize")
        return None

    start_time = time.time()

    # Load talk metadata from Abstracts.xlsx
    abstracts_path = Path(config.data_dir) / "Abstracts.xlsx"
    if abstracts_path.exists():
        talk_metadata = load_talk_metadata_from_abstracts(abstracts_path)
    else:
        # Fall back to basic metadata from data.talks
        talk_metadata = {}
        for _, row in data.talks.iterrows():
            talk_id = row['talk_id']
            talk_metadata[talk_id] = {
                'title': row.get('title', ''),
                'primary_contact_author': row.get('presenter_id', ''),
                'author_names': '',
            }

    # Parse fixed block sessions from config
    fixed_block_sessions = []
    if config.fixed_block_sessions:
        for fs in config.fixed_block_sessions:
            fixed_block_sessions.append(FixedBlockSession(
                name=fs.get('name', 'FixedSession'),
                block=fs['block'],
                talks=fs['talks']
            ))

    # Build room capacities and names from config
    room_capacities = {}
    room_names = {}

    if config.rooms:
        # Use rooms from config
        for room in config.rooms:
            room_id = room['id']
            room_capacities[room_id] = room.get('capacity', 100)
            room_names[room_id] = room.get('name', room_id)
    else:
        # Try to get from data
        room_capacities = getattr(data, 'room_capacities', {})
        if not room_capacities:
            # Generate generic rooms matching the data's room list
            for room_id in data.rooms:
                room_capacities[room_id] = 100
                room_names[room_id] = room_id

    # Build Phase 4 input
    phase4_input = Phase4Input(
        phase3_result=phase3_result,
        room_capacities=room_capacities,
        preferences=instance.preferences,
        talk_metadata=talk_metadata,
        fixed_block_sessions=fixed_block_sessions
    )

    # Run Phase 4
    phase4_result = solve_phase4(
        phase4_input,
        room_names=room_names,
        verbose=config.phase4_verbose
    )

    phase4_time = time.time() - start_time
    results['phase4_result'] = phase4_result
    results['phase4_time'] = phase4_time

    return phase4_result


# =============================================================================
# OUTPUT
# =============================================================================

def export_schedule(results, config: SchedulerConfig, instance, data):
    """Export schedule to CSV."""
    import pandas as pd
    from src.phase4 import phase4_result_to_dataframe

    output_path = Path(config.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer Phase 4 result if available
    phase4_result = results.get('phase4_result')
    if phase4_result:
        df = phase4_result_to_dataframe(phase4_result)
        df.to_csv(output_path, index=False)

        # Also save markdown
        md_path = output_path.with_suffix('.md')
        phase4_result.save_markdown(md_path)

        print(f"\n✓ Schedule exported to {output_path}")
        print(f"  Markdown: {md_path}")
        print(f"  {len(df)} talk assignments")
        print(f"  {len(phase4_result.session_metrics)} sessions")
        if phase4_result.fixed_sessions_added:
            print(
                f"  Fixed sessions: {', '.join(phase4_result.fixed_sessions_added)}")
        return

    # Fall back to old export if Phase 4 not run
    phase3_result = results.get('phase3_result')
    if not phase3_result:
        print("No Phase 3 result to export")
        return

    # Build talk info map
    talk_info = {
        row['talk_id']: {'title': row.get(
            'title', ''), 'presenter_id': row.get('presenter_id', '')}
        for _, row in data.talks.iterrows()
    }

    # Build presenter names from talks (presenter_id is the name in anonymized data)
    presenter_names = {}
    for _, row in data.talks.iterrows():
        presenter_names[row['presenter_id']] = row['presenter_id']

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
                presenter_id = info.get('presenter_id', '')
                # Extract paper_id from talk_id (T001 -> 1)
                paper_id = int(talk_id[1:]) if talk_id.startswith('T') else 0

                rows.append({
                    'Session_ID': ts_id,
                    'Block_ID': block.block_id,
                    'Slot': slot_idx + 1,
                    'Room': room_id,
                    'Talk_ID': talk_id,
                    'Paper_ID': paper_id,
                    'Title': info.get('title', ''),
                    'Presenter_ID': presenter_id,
                    'Presenter_Name': presenter_names.get(presenter_id, presenter_id)
                })

    df = pd.DataFrame(rows)
    df = df.sort_values(['Session_ID', 'Block_ID', 'Slot', 'Room'])
    df.to_csv(output_path, index=False)

    print(f"\n✓ Schedule exported to {output_path}")
    print(f"  {len(df)} talk assignments")


def run_evaluation(results, config: SchedulerConfig, instance, data):
    """Run the schedule evaluator on the exported schedule."""
    from src.phase4 import phase4_result_to_dataframe
    from src.schedule_evaluator import evaluate_from_instance

    phase4_result = results.get('phase4_result')
    if not phase4_result:
        return None

    schedule_df = phase4_result_to_dataframe(phase4_result)

    # Normalize column names (Phase 4 uses Block_ID/Room_ID, evaluator expects Session_ID/Room)
    if 'Session_ID' not in schedule_df.columns and 'Block_ID' in schedule_df.columns:
        schedule_df['Session_ID'] = schedule_df['Block_ID']
    if 'Room' not in schedule_df.columns and 'Room_ID' in schedule_df.columns:
        schedule_df['Room'] = schedule_df['Room_ID']

    # Build talk keywords for coherence metric
    talk_keywords = {}
    if 'keywords' in data.talks.columns:
        for _, row in data.talks.iterrows():
            tid = row['talk_id']
            kw_str = str(row.get('keywords', ''))
            keywords = set(kw.strip() for kw in kw_str.split(';') if kw.strip())
            if keywords:
                talk_keywords[tid] = keywords

    return evaluate_from_instance(schedule_df, instance, talk_keywords)


def print_summary(results, config: SchedulerConfig, total_time: float,
                  eval_metrics=None):
    """Print final summary."""
    print("\n" + "=" * 70)
    print("SCHEDULING COMPLETE")
    print("=" * 70)

    print(f"\nPipeline: {config.pipeline.upper()}")
    print(f"Data: {config.data_dir}")

    print(f"\nTiming:")
    print(f"  Phase 1: {results.get('phase1_time', 0):.1f}s")
    print(f"  Phase 2: {results.get('phase2_time', 0):.1f}s")
    print(f"  Phase 3: {results.get('phase3_time', 0):.1f}s")
    print(f"  Phase 4: {results.get('phase4_time', 0):.1f}s")
    print(f"  Total:   {total_time:.1f}s")

    if eval_metrics:
        print(f"\nQuality (via ScheduleEvaluator):")
        print(f"  Missed attendance:   {eval_metrics.total_missed_attendance}")
        print(f"  Session hops:        {eval_metrics.total_session_hops}")
        print(f"  Incoherent sessions: {eval_metrics.incoherent_sessions}")
        print(f"  Presenter violations: {eval_metrics.presenter_violations}")
    else:
        # Fallback to pipeline-internal metrics
        print(f"\nQuality:")
        if 'phase1_objective' in results:
            print(f"  Missed attendances (pipeline estimate): {results['phase1_objective']:.0f}")

        phase3 = results.get('phase3_result')
        if phase3 and hasattr(phase3, 'total_violations'):
            print(f"  Presenter violations: {phase3.total_violations}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    config = build_config(args)

    print("=" * 70)
    print(f"CONFERENCE SCHEDULER - {config.pipeline.upper()} PIPELINE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Pipeline:    {config.pipeline}")
    print(f"  Data dir:    {config.data_dir}")
    print(f"  Output:      {config.output_file}")
    if args.config:
        print(f"  Config file: {args.config}")

    start_total = time.time()

    # Load data
    print("\n[Loading Data]")
    try:
        data = load_data(config)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    errors = data.validate() if hasattr(data, 'validate') else []
    if errors:
        print("Validation warnings:")
        for err in errors:
            print(f"  - {err}")

    # Build instance
    from src.instance import build_instance
    instance = build_instance(data)

    print(f"✓ Loaded {len(instance.talks)} talks")
    print(f"✓ {sum(len(p) for p in instance.preferences.values())} preferences from {len(instance.participants)} participants")

    # Exclude fixed_block_sessions talks from optimization and adjust block types
    if config.fixed_block_sessions:
        fixed_talk_ids = set()
        for fs in config.fixed_block_sessions:
            fixed_talk_ids.update(fs['talks'])

        if fixed_talk_ids:
            # Remove fixed talks from instance
            original_count = len(instance.talks)
            instance.talks = [
                t for t in instance.talks if t not in fixed_talk_ids]
            excluded_count = original_count - len(instance.talks)
            print(
                f"✓ Excluded {excluded_count} fixed_block_session talks from optimization")

            # Adjust block types: reduce one room per fixed_block_session
            # Each fixed_block_session takes one room slot from the target block's timeslot
            # We need to find which block type corresponds to each timeslot and reduce n by 1
            from src.pipelines.traditional import FixedSequenceConfig

            # Build timeslot -> block_type mapping from data.timeslots
            timeslot_block_types = {}
            for ts in data.timeslots:
                ts_id = ts.get('id', ts) if isinstance(ts, dict) else str(ts)
                type_id = ts.get('type_id', None)
                timeslot_block_types[ts_id] = type_id

            # For fixed_block_sessions, we have two approaches:
            #
            # Approach A (traditional pipeline): Convert to fixed_sequences for Phase 2 attachment
            #   - Phase 2 attaches fixed sequences to blocks (expanding them)
            #   - Phase 3 assigns blocks to timeslots (needs timeslot types unchanged)
            #   - Problem: Phase 3 doesn't know WHICH block goes to WHICH timeslot
            #
            # Approach B (heuristic pipeline with fixed_block_sessions): Let Phase 4 handle it
            #   - Update timeslot types to reduced sizes (e.g., FA: 4R4T → 3R4T)
            #   - Phase 1-3 create/assign blocks with reduced sizes
            #   - Phase 4 adds fixed sessions to the correct timeslots
            #
            # We use Approach B for the heuristic pipeline when fixed_block_sessions are specified

            if config.pipeline == "heuristic":
                # Approach B: Update timeslot type_ids for Phase 3 matching
                # Phase 4 will add the fixed sessions to the correct timeslots
                for fs in config.fixed_block_sessions:
                    block_id = fs['block']
                    name = fs.get('name', block_id)
                    result_block_type = timeslot_block_types.get(block_id)
                    if result_block_type:
                        import re
                        match = re.match(r'(\d+)R(\d+)T', result_block_type)
                        if match:
                            n, k = int(match.group(1)), int(match.group(2))
                            target_block_type = f"{n-1}R{k}T"

                            # Update timeslot type_id to the reduced type
                            for ts in data.timeslots:
                                ts_id_check = ts.get('id', ts) if isinstance(
                                    ts, dict) else str(ts)
                                if ts_id_check == block_id and isinstance(ts, dict):
                                    ts['type_id'] = target_block_type
                                    # Also update rooms list if present
                                    if 'rooms' in ts and len(ts['rooms']) > n - 1:
                                        ts['rooms'] = ts['rooms'][:n-1]
                                    break

                            # Also update instance.block_types for the heuristic pipeline
                            # Decrease count of result_block_type, increase count of target_block_type
                            if result_block_type in instance.block_types:
                                instance.block_types[result_block_type]['count'] -= 1
                                if instance.block_types[result_block_type]['count'] <= 0:
                                    del instance.block_types[result_block_type]

                            if target_block_type not in instance.block_types:
                                instance.block_types[target_block_type] = {
                                    'n': n-1, 'k': k, 'count': 1}
                            else:
                                instance.block_types[target_block_type]['count'] += 1

                            print(
                                f"  → {name}: timeslot {block_id} reduced to {target_block_type} (Phase 4 will add fixed session)")

                # Don't create fixed_sequences for heuristic pipeline
                # Phase 4 will handle fixed_block_sessions directly
            else:
                # Approach A: Convert to fixed_sequences for traditional pipeline
                auto_fixed_sequences = []
                for fs in config.fixed_block_sessions:
                    block_id = fs['block']
                    talks = fs['talks']
                    name = fs.get('name', block_id)

                    result_block_type = timeslot_block_types.get(block_id)
                    if result_block_type:
                        import re
                        match = re.match(r'(\d+)R(\d+)T', result_block_type)
                        if match:
                            n, k = int(match.group(1)), int(match.group(2))
                            target_block_type = f"{n-1}R{k}T"

                            auto_fixed_sequences.append(FixedSequenceConfig(
                                name=name,
                                talks=talks,
                                target_block_type=target_block_type,
                                result_block_type=result_block_type
                            ))
                            print(
                                f"  → {name}: auto-configured as {target_block_type} → {result_block_type}")

                if auto_fixed_sequences:
                    if config.fixed_sequences is None:
                        config.fixed_sequences = []
                    config.fixed_sequences.extend(auto_fixed_sequences)

    # Run pipeline
    try:
        results = run_pipeline(config, instance, data)
    except ImportError as e:
        if "gurobipy" in str(e) and config.pipeline != "heuristic":
            print(f"\n❌ Gurobi not available. Use --pipeline heuristic instead.")
            sys.exit(1)
        raise
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run swap optimization (between Phase 3 and Phase 4)
    if config.swap_enabled and results.get('phase3_result'):
        try:
            results = run_swap_optimization(results, config, instance, data)
        except Exception as e:
            print(f"\n⚠️ Swap optimization failed: {e}")
            import traceback
            traceback.print_exc()

    # Run Phase 4: Finalization
    try:
        run_phase4(results, config, instance, data)
    except Exception as e:
        print(f"\n⚠️ Phase 4 failed: {e}")
        import traceback
        traceback.print_exc()
        # Continue without Phase 4 - export_schedule will fall back

    # Export and summarize
    total_time = time.time() - start_total

    export_schedule(results, config, instance, data)

    # Run evaluator for consistent quality metrics across all pipelines
    eval_metrics = None
    try:
        eval_metrics = run_evaluation(results, config, instance, data)
    except Exception as e:
        if config.verbose:
            print(f"\n⚠️ Evaluation failed: {e}")

    print_summary(results, config, total_time, eval_metrics)

    print("\n" + "=" * 70)
    print(f"✓ Done! Schedule saved to {config.output_file}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
