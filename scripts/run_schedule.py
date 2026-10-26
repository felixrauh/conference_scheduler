#!/usr/bin/env python
"""
Unified Conference Scheduler

Single entry point for all scheduling pipelines.
Configuration can be provided via config file and/or CLI arguments.

Usage:
    # Quick start without Gurobi (heuristic pipeline)
    python scripts/run_schedule.py --pipeline heuristic --data-dir examples/orbel2026

    # Default pipeline (traditional, requires Gurobi)
    python scripts/run_schedule.py --data-dir examples/orbel2026

    # Other pipelines
    python scripts/run_schedule.py --pipeline matching --data-dir examples/orbel2026

    # Use config file (overrides defaults, CLI overrides config)
    python scripts/run_schedule.py --config config/scheduling_config.jsonc

    # Full example
    python scripts/run_schedule.py \\
        --pipeline traditional \\
        --data-dir examples/orbel2026 \\
        --output output/schedule.csv \\
        --verbose

Pipelines:
    traditional          - Phase 1 → 2 → 3 optimization, requires Gurobi
    heuristic            - Greedy alternative, no Gurobi needed
    matching             - Bottom-up matching (Phase A → B → C → D → 3), requires Gurobi
    matching_constrained - Matching with group/keyword constraints, requires Gurobi

Input Data:
    Expects a directory with: talks.csv, preferences.csv, sessions.csv
    (and optionally availability.csv). See examples/orbel2026/ for the format.

See config/scheduling_config.jsonc for all configuration options.
"""

import sys
import time
import json
import random
import re
import argparse
from collections import Counter
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).parent.parent))


class SchedulerError(Exception):
    """A problem with the input data or configuration, reported without traceback."""


# =============================================================================
# CONFIGURATION
# =============================================================================

# Allowed values for the algorithm choices (validated before running)
VALID_CHOICES = {
    "pipeline": ["traditional", "heuristic", "matching", "matching_constrained"],
    "phase1_method": ["column_generation", "explicit"],
    "phase1_pricing_strategy": ["auto", "enumeration", "greedy",
                                "local_search", "beam_search"],
    "phase2_partition_strategy": ["greedy", "matching", "random"],
    "phase2_ordering_strategy": ["enumerate", "greedy"],
    "phase3_method": ["milp", "hungarian"],
}

KNOWN_CONFIG_SECTIONS = {
    "pipeline", "seed", "input", "output", "block_types", "phase1", "phase2",
    "phase3", "phase4", "rooms", "swap_optimization", "matching_pipeline",
}


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

    # Seed for the randomized steps (local search, sampling); same input and
    # seed give the same schedule
    seed: int = 0

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

        unknown = sorted(set(d) - KNOWN_CONFIG_SECTIONS)
        if unknown:
            print(f"Warning: ignoring unknown config section(s): {', '.join(unknown)}")

        if 'pipeline' in d:
            config.pipeline = d['pipeline']

        if 'seed' in d:
            config.seed = d['seed']

        if 'input' in d:
            inp = d['input']
            config.data_dir = inp.get('data_dir', config.data_dir)

        if 'output' in d:
            out = d['output']
            if 'schedule_csv' in out or 'dir' in out:
                config.output_file = str(
                    Path(out.get('dir', 'output')) / out.get('schedule_csv', 'schedule.csv'))

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
            config.require_common_keyword = mp.get(
                'require_common_keyword', config.require_common_keyword)
            config.max_keyword_violations = mp.get(
                'max_keyword_violations', config.max_keyword_violations)

        return config

    def validate(self) -> List[str]:
        """Return a list of configuration errors (empty if valid)."""
        errors = []
        for attr, choices in VALID_CHOICES.items():
            value = getattr(self, attr)
            if value not in choices:
                errors.append(
                    f"Invalid {attr} '{value}'. Choose one of: {', '.join(choices)}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            errors.append(f"Invalid seed '{self.seed}'. Use an integer")
        if self.rooms:
            for room in self.rooms:
                if 'id' not in room:
                    errors.append(f"Room entry {room} has no 'id'")
        return errors


def strip_jsonc(text: str) -> str:
    """
    Turn JSONC (JSON with comments and trailing commas) into plain JSON.

    Removes // and /* */ comments and trailing commas, leaving string
    contents (e.g. "https://..." or "data//dir") untouched.
    """
    def scan(src: str, handle_outside):
        out, i, n, in_string = [], 0, len(src), False
        while i < n:
            c = src[i]
            if in_string:
                out.append(c)
                if c == '\\' and i + 1 < n:
                    out.append(src[i + 1])
                    i += 2
                    continue
                if c == '"':
                    in_string = False
                i += 1
            elif c == '"':
                in_string = True
                out.append(c)
                i += 1
            else:
                i = handle_outside(src, i, out)
        return ''.join(out)

    def drop_comment(src, i, out):
        if src.startswith('//', i):
            j = src.find('\n', i)
            return len(src) if j == -1 else j
        if src.startswith('/*', i):
            j = src.find('*/', i + 2)
            return len(src) if j == -1 else j + 2
        out.append(src[i])
        return i + 1

    def drop_trailing_comma(src, i, out):
        if src[i] == ',':
            m = re.match(r',\s*[}\]]', src[i:])
            if m:
                return i + 1
        out.append(src[i])
        return i + 1

    return scan(scan(text, drop_comment), drop_trailing_comma)


def load_jsonc(path: str) -> Dict:
    """Load JSONC (JSON with comments) file."""
    with open(path, 'r') as f:
        return json.loads(strip_jsonc(f.read()))


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Conference Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_schedule.py --pipeline heuristic --data-dir examples/orbel2026
  python scripts/run_schedule.py --data-dir examples/orbel2026
  python scripts/run_schedule.py --pipeline matching --data-dir examples/orbel2017
  python scripts/run_schedule.py --config config/scheduling_config.jsonc
        """
    )

    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to configuration file (JSONC format)")
    parser.add_argument("--pipeline", "-p", type=str,
                        choices=VALID_CHOICES["pipeline"],
                        default=None,
                        help="Scheduling pipeline to use (default: traditional, "
                             "which requires Gurobi; use 'heuristic' without Gurobi)")
    parser.add_argument("--data-dir", "-d", type=str, default=None,
                        help="Directory containing input CSV files (talks.csv, preferences.csv, "
                             "sessions.csv; default: examples/orbel2026)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output CSV file path (a .md file is written next to it)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: 0). The same input and seed give "
                             "the same schedule; other seeds give alternative schedules")
    parser.add_argument("--phase1-method", type=str, choices=VALID_CHOICES["phase1_method"],
                        default=None, help="Phase 1 method (traditional pipeline only)")
    parser.add_argument("--phase2-partition", type=str,
                        choices=VALID_CHOICES["phase2_partition_strategy"],
                        default=None, help="Phase 2 partition strategy (traditional pipeline only)")
    parser.add_argument("--phase3-method", type=str, choices=VALID_CHOICES["phase3_method"],
                        default=None, help="Phase 3 method (heuristic always uses hungarian)")
    parser.add_argument("--time-limit", "-t", type=float, default=None,
                        help="Time limit in seconds for Phase 1 (traditional, heuristic) "
                             "or the whole matching pipeline")
    parser.add_argument("--require-keyword", action="store_true",
                        help="Only pair talks that share a keyword (matching_constrained only)")
    parser.add_argument("--max-violations", type=int, default=None,
                        help="Soft keyword constraint: minimize keyword violations, starting "
                             "with this violation budget and raising it until feasible "
                             "(0 = as few as possible; matching_constrained only)")

    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> SchedulerConfig:
    """Build configuration from config file and CLI arguments."""
    config = SchedulerConfig()

    if args.config:
        try:
            config = SchedulerConfig.from_dict(load_jsonc(args.config))
        except FileNotFoundError:
            raise SchedulerError(f"Config file not found: {args.config}")
        except json.JSONDecodeError as e:
            raise SchedulerError(f"Could not parse config file {args.config}: {e}")

    # CLI overrides
    if args.pipeline:
        config.pipeline = args.pipeline
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output:
        config.output_file = args.output
    if args.verbose:
        config.verbose = True
    if args.seed is not None:
        config.seed = args.seed
    if args.phase1_method:
        config.phase1_method = args.phase1_method
    if args.phase2_partition:
        config.phase2_partition_strategy = args.phase2_partition
    if args.phase3_method:
        config.phase3_method = args.phase3_method
    if args.time_limit is not None:
        config.phase1_time_limit = args.time_limit
        config.matching_time_limit = args.time_limit
    if args.require_keyword:
        config.require_common_keyword = True
    if args.max_violations is not None:
        config.max_keyword_violations = args.max_violations
    if config.max_keyword_violations is not None:
        # Soft constraint implies keyword constraint
        config.require_common_keyword = True

    errors = config.validate()
    if errors:
        raise SchedulerError(
            "Invalid configuration:\n" + "\n".join(f"  - {e}" for e in errors))

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


def apply_fixed_block_sessions(data, fixed_block_sessions: List[Dict]) -> Set[str]:
    """
    Reserve one room per fixed block session in its target timeslot.

    The timeslot's room count is reduced by one per fixed session and block
    types are recomputed, so the pipelines schedule the remaining talks around
    the fixed sessions. Phase 4 inserts the fixed sessions afterwards.

    Must be called before build_instance(). Returns the fixed talk IDs.
    """
    timeslots_by_id = {ts['id']: ts for ts in data.timeslots}
    all_talks = set(data.talks['talk_id'])
    shape = {ts['id']: (data.block_types[ts['type_id']]['n'],
                        data.block_types[ts['type_id']]['k'])
             for ts in data.timeslots}

    errors = []
    fixed_talks: Set[str] = set()
    rooms_taken: Counter = Counter()
    for fs in fixed_block_sessions:
        name = fs.get('name', fs.get('block', '?'))
        block = fs.get('block')
        talks = fs.get('talks', [])
        if block not in timeslots_by_id:
            errors.append(f"Fixed session '{name}': block '{block}' is not a "
                          f"session_id in sessions.csv ({', '.join(timeslots_by_id)})")
            continue
        if len(talks) > shape[block][1]:
            errors.append(f"Fixed session '{name}' has {len(talks)} talks but "
                          f"block {block} only has {shape[block][1]} talks per room")
        for t in talks:
            if t not in all_talks:
                errors.append(f"Fixed session '{name}': unknown talk '{t}' "
                              f"(talk IDs look like T001)")
            elif t in fixed_talks:
                errors.append(f"Fixed session '{name}': talk '{t}' appears in "
                              f"more than one fixed session")
        fixed_talks.update(talks)
        rooms_taken[block] += 1

    for block, n_fixed in rooms_taken.items():
        if n_fixed >= shape[block][0]:
            errors.append(f"Block {block} has {shape[block][0]} rooms, all taken "
                          f"by fixed sessions; at least one room must remain")
    if errors:
        raise SchedulerError(
            "Invalid fixed_block_sessions:\n" + "\n".join(f"  - {e}" for e in errors))

    # Shrink the target timeslots and recompute block types by shape
    for block, n_fixed in rooms_taken.items():
        n, k = shape[block]
        shape[block] = (n - n_fixed, k)
        ts = timeslots_by_id[block]
        ts['rooms'] = ts['rooms'][:n - n_fixed]
        print(f"  → {block}: {n_fixed} room(s) reserved for fixed sessions, "
              f"{n - n_fixed} rooms left to optimize")

    block_types: Dict[str, Dict] = {}
    for ts in data.timeslots:
        n, k = shape[ts['id']]
        type_id = f"{n}R{k}T"
        ts['type_id'] = type_id
        block_types.setdefault(type_id, {"n": n, "k": k, "count": 0})
        block_types[type_id]["count"] += 1
    data.block_types = block_types

    return fixed_talks


def add_placeholder_talks(instance, n_slots: int, verbose: bool = False) -> int:
    """Pad instance.talks with placeholder talks until it fills n_slots."""
    n_missing = n_slots - len(instance.talks)
    for i in range(n_missing):
        pid = f"PLACEHOLDER_{i+1}"
        instance.talks.append(pid)
        instance.talk_presenter[pid] = pid
    if n_missing > 0 and verbose:
        print(f"  Adding {n_missing} placeholder talks "
              f"({len(instance.talks) - n_missing} talks, {n_slots} slots)")
    return max(n_missing, 0)


# =============================================================================
# PIPELINE DISPATCH
# =============================================================================

def run_pipeline(config: SchedulerConfig, instance, data):
    """Dispatch to appropriate pipeline."""
    if config.pipeline == "traditional":
        from src.pipelines.traditional import run_traditional_pipeline
        from src.pipelines.traditional import PipelineConfig as TraditionalConfig
        from src.pipelines.traditional import FixedSequenceConfig

        fixed_sequences = None
        if config.fixed_sequences:
            fixed_sequences = [FixedSequenceConfig(
                name=fs.get('name', 'FixedSequence'),
                talks=fs['talks'],
                target_block_type=fs['target_block_type'],
                result_block_type=fs['result_block_type']
            ) for fs in config.fixed_sequences]

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
        from src.pipelines.heuristic import FixedSequenceConfig

        fixed_sequences = None
        if config.fixed_sequences:
            fixed_sequences = [FixedSequenceConfig(
                name=fs.get('name', 'FixedSequence'),
                talks=fs['talks'],
                target_block_type=fs['target_block_type'],
                result_block_type=fs['result_block_type']
            ) for fs in config.fixed_sequences]

        pipeline_config = HeuristicConfig(
            phase1_time_limit=config.phase1_time_limit,
            phase2_partition_strategy="matching",
            phase2_ordering_strategy=config.phase2_ordering_strategy,
            phase2_local_search_iterations=100,
            verbose=config.verbose,
            fixed_sequences=fixed_sequences
        )
        return run_heuristic_pipeline(pipeline_config, instance, data)

    elif config.pipeline in ("matching", "matching_constrained"):
        # The matching pipelines need exactly one talk per slot
        add_placeholder_talks(
            instance, data.total_talk_slots(), verbose=config.verbose)

        if config.pipeline == "matching":
            from src.matching_pipeline import run_matching_pipeline

            print("\n[Running Matching Pipeline: A → B → C → D → 3]")

            pipeline_result, phase3_result = run_matching_pipeline(
                instance,
                time_limit=config.matching_time_limit,
                verbose=config.verbose,
                run_phase3=True,
                phase3_method=config.phase3_method
            )
        else:
            from src.matching_pipeline_constrained import (
                run_constrained_matching_pipeline,
                MatchingConstraints, TalkMetadata
            )

            # Build talk metadata for real talks (placeholders have none,
            # which lets them pair with any talk)
            talk_keywords = data.talk_keywords()
            scheduled = set(instance.talks)
            metadata = {}
            for _, row in data.talks.iterrows():
                talk_id = row['talk_id']
                if talk_id not in scheduled:
                    continue
                group = row.get('special_group', 'general')
                if not isinstance(group, str) or not group.strip():
                    group = 'general'
                metadata[talk_id] = TalkMetadata(
                    talk_id=talk_id, special_group=group.strip(),
                    keywords=talk_keywords.get(talk_id, set()))

            if config.require_common_keyword and not talk_keywords:
                raise SchedulerError(
                    "Keyword constraint requested, but talks.csv has no keywords")

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

    # Talk keywords for coherence scoring (optional)
    talk_keywords = data.talk_keywords() or None

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
    from src.phase4 import (
        solve_phase4, Phase4Input, FixedBlockSession
    )
    from src.data_loader import load_talk_metadata_from_abstracts

    phase3_result = results.get('phase3_result')
    if not phase3_result:
        print("No Phase 3 result to finalize")
        return None

    start_time = time.time()

    # Basic metadata from talks.csv (optional title / presenter_name / authors)
    talk_metadata = {}
    for _, row in data.talks.iterrows():
        talk_id = row['talk_id']
        talk_metadata[talk_id] = {
            'title': _str_or_empty(row.get('title')) or talk_id,
            'primary_contact_author': (_str_or_empty(row.get('presenter_name'))
                                       or row.get('presenter_id', '')),
            'author_names': _str_or_empty(row.get('authors')),
        }

    # Richer metadata from an Abstracts.xlsx export, if present
    abstracts_path = Path(config.data_dir) / "Abstracts.xlsx"
    if abstracts_path.exists():
        talk_metadata.update(load_talk_metadata_from_abstracts(abstracts_path))

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

    # Run Phase 4 (output follows the session order of sessions.csv)
    phase4_result = solve_phase4(
        phase4_input,
        room_names=room_names,
        verbose=config.phase4_verbose,
        timeslot_order=[ts['id'] for ts in data.timeslots]
    )

    phase4_time = time.time() - start_time
    results['phase4_result'] = phase4_result
    results['phase4_time'] = phase4_time

    return phase4_result


def _str_or_empty(value) -> str:
    """Return value as a stripped string, or '' for None/NaN."""
    if value is None or (isinstance(value, float) and value != value):
        return ''
    return str(value).strip()


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

    rows = []
    for assignment in phase3_result.assignments:
        block = assignment.block
        timeslot = assignment.timeslot
        room_mapping = assignment.room_mapping

        ts_id = timeslot.get('id', timeslot) if isinstance(
            timeslot, dict) else str(timeslot)

        for slot_idx, ntuple in enumerate(block.tuples):
            for room_pos, talk_id in enumerate(ntuple):
                room_id = room_mapping.get(room_pos, f"Room_{room_pos + 1}")
                info = talk_info.get(talk_id, {})
                presenter_id = info.get('presenter_id', '')
                # Extract paper_id from talk_id (T001 -> 1)
                paper_id = int(talk_id[1:]) if re.fullmatch(r'T\d+', talk_id) else 0

                rows.append({
                    'Session_ID': ts_id,
                    'Block_ID': block.block_id,
                    'Slot': slot_idx + 1,
                    'Room': room_id,
                    'Talk_ID': talk_id,
                    'Paper_ID': paper_id,
                    'Title': info.get('title', ''),
                    'Presenter_ID': presenter_id,
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

    return evaluate_from_instance(schedule_df, instance, data.talk_keywords())


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

def run_scheduler(config: SchedulerConfig) -> Dict:
    """
    Run the full scheduling workflow for a configuration.

    Loads data, runs the selected pipeline, swap optimization and Phase 4,
    exports the schedule and evaluates it. Returns the results dict, with the
    evaluator metrics under 'evaluation' and the wall time under 'total_time'.

    Raises SchedulerError for problems with the input data or configuration.
    """
    from src.instance import build_instance

    print("=" * 70)
    print(f"CONFERENCE SCHEDULER - {config.pipeline.upper()} PIPELINE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Pipeline:    {config.pipeline}")
    print(f"  Data dir:    {config.data_dir}")
    print(f"  Output:      {config.output_file}")
    print(f"  Seed:        {config.seed}")

    # Fix the randomized steps so that the same input and seed give the same
    # schedule (all of them use Python's global random generator)
    random.seed(config.seed)

    start_total = time.time()

    # Load data
    print("\n[Loading Data]")
    try:
        data = load_data(config)
    except Exception as e:
        raise SchedulerError(f"Error loading data from {config.data_dir}: {e}")

    errors = data.validate()
    if errors:
        print("Validation warnings:")
        for err in errors:
            print(f"  - {err}")

    n_talks, n_slots = len(data.talks), data.total_talk_slots()
    if n_talks > n_slots:
        raise SchedulerError(
            f"{n_talks} talks but only {n_slots} slots in sessions.csv. "
            f"Add rooms, talks per room, or sessions so every talk fits.")

    # Reserve rooms for fixed block sessions before building the instance
    fixed_talk_ids: Set[str] = set()
    if config.fixed_block_sessions:
        print(f"\n[Fixed Block Sessions] {len(config.fixed_block_sessions)} configured")
        fixed_talk_ids = apply_fixed_block_sessions(
            data, config.fixed_block_sessions)

    instance = build_instance(data)
    if fixed_talk_ids:
        instance.talks = [t for t in instance.talks if t not in fixed_talk_ids]
        print(f"✓ Excluded {len(fixed_talk_ids)} fixed_block_session talks from optimization")

    print(f"✓ Loaded {len(instance.talks)} talks")
    print(f"✓ {sum(len(p) for p in instance.preferences.values())} preferences from {len(instance.participants)} participants")
    n_empty = data.total_talk_slots() - len(instance.talks)
    if n_empty > 0:
        print(f"✓ {data.total_talk_slots()} slots: {n_empty} will be filled with placeholder talks")

    # Run pipeline
    results = run_pipeline(config, instance, data)

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
        if config.verbose:
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
        print(f"\n⚠️ Evaluation failed: {e}")

    print_summary(results, config, total_time, eval_metrics)

    print("\n" + "=" * 70)
    print(f"✓ Done! Schedule saved to {config.output_file}")
    print("=" * 70)

    results['evaluation'] = eval_metrics
    results['total_time'] = total_time
    return results


def main(argv=None):
    args = parse_args(argv)
    config = SchedulerConfig(pipeline=args.pipeline or "traditional")

    try:
        config = build_config(args)
        if args.config:
            print(f"Config file: {args.config}")
        run_scheduler(config)
    except SchedulerError as e:
        print(f"\n❌ {e}")
        return 1
    except ImportError as e:
        if "gurobipy" in str(e):
            print(f"\n❌ Gurobi (gurobipy) is not installed, but the "
                  f"'{config.pipeline}' pipeline needs it.\n"
                  f"   Use --pipeline heuristic, or install Gurobi: pip install gurobipy")
            return 1
        raise
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        if "license" in str(e).lower():
            print("   This looks like a Gurobi license problem. The pip-installed "
                  "gurobipy only includes a size-limited license, which is too small "
                  "for most conferences. Get a full (e.g. free academic) license from "
                  "https://www.gurobi.com/, or use --pipeline heuristic.")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
