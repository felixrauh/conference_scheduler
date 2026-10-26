#!/usr/bin/env python
"""
Compare All Scheduling Pipelines

Runs several pipelines on the same data through the main entry point
(scripts/run_schedule.py) and compares their evaluator metrics side by side.
Every pipeline uses the same configuration, so the comparison reflects exactly
what `run_schedule.py` produces.

Usage:
    python scripts/compare_all_pipelines.py --data-dir examples/orbel2026
    python scripts/compare_all_pipelines.py --skip-traditional    # faster
    python scripts/compare_all_pipelines.py --pipelines heuristic matching
    python scripts/compare_all_pipelines.py --config config/scheduling_config.jsonc

Pipelines that need Gurobi are reported as skipped when it is not installed.

Output files (in --output-dir, default output/comparison):
    schedule_{pipeline}.csv / .md  - schedule per pipeline
    metrics_{pipeline}.json        - evaluator metrics per pipeline
    comparison_summary.md          - side-by-side comparison
    comparison_metrics.json        - all metrics in one file
"""

import sys
import copy
import json
import argparse
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_schedule import (  # noqa: E402
    SchedulerConfig,
    SchedulerError,
    VALID_CHOICES,
    load_jsonc,
    run_scheduler,
)

ALL_PIPELINES = ["heuristic", "matching", "matching_constrained", "traditional"]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run several scheduling pipelines on the same data and compare them"
    )
    parser.add_argument("--data-dir", "-d", default="examples/orbel2026",
                        help="Input data directory (default: examples/orbel2026)")
    parser.add_argument("--output-dir", "-o", default="output/comparison",
                        help="Output directory (default: output/comparison)")
    parser.add_argument("--config", "-c", default=None,
                        help="Config file (JSONC) shared by all pipelines")
    parser.add_argument("--pipelines", nargs="+", choices=ALL_PIPELINES,
                        default=ALL_PIPELINES, help="Pipelines to run (default: all)")
    parser.add_argument("--skip-traditional", action="store_true",
                        help="Skip the traditional pipeline (the slowest)")
    parser.add_argument("--skip-matching", action="store_true",
                        help="Skip both matching pipelines")
    parser.add_argument("--time-limit", "-t", type=float, default=None,
                        help="Time limit for Phase 1 / the matching pipelines (seconds)")
    parser.add_argument("--require-keyword", action="store_true",
                        help="Hard keyword constraint for matching_constrained")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for all pipelines (default: 0, or the config file's)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    return parser.parse_args()


def build_base_config(args) -> SchedulerConfig:
    """Configuration shared by all pipelines."""
    if args.config:
        try:
            config = SchedulerConfig.from_dict(load_jsonc(args.config))
        except (OSError, json.JSONDecodeError) as e:
            raise SchedulerError(f"Could not load config file {args.config}: {e}")
    else:
        config = SchedulerConfig()

    config.data_dir = args.data_dir
    config.verbose = args.verbose
    config.phase4_verbose = args.verbose
    if args.time_limit is not None:
        config.phase1_time_limit = args.time_limit
        config.matching_time_limit = args.time_limit
    if args.require_keyword:
        config.require_common_keyword = True
    if args.seed is not None:
        config.seed = args.seed

    errors = [e for e in config.validate() if not e.startswith("Invalid pipeline")]
    if errors:
        raise SchedulerError("Invalid configuration:\n" + "\n".join(f"  - {e}" for e in errors))
    return config


def run_one(name: str, base_config: SchedulerConfig, output_dir: Path) -> dict:
    """Run one pipeline and return a summary entry."""
    config = copy.deepcopy(base_config)
    config.pipeline = name
    config.output_file = str(output_dir / f"schedule_{name}.csv")

    entry = {"pipeline": name, "status": "failed", "time": None, "metrics": None,
             "schedule_file": config.output_file, "error": None}
    try:
        results = run_scheduler(config)
    except ImportError as e:
        if "gurobipy" not in str(e):
            raise
        entry["status"] = "skipped"
        entry["error"] = "Gurobi (gurobipy) not installed"
        print(f"\n  ⏭  {name}: skipped (Gurobi not installed)")
        return entry
    except SchedulerError as e:
        entry["error"] = str(e)
        print(f"\n  ✗ {name} failed: {e}")
        return entry
    except Exception as e:
        entry["error"] = str(e)
        print(f"\n  ✗ {name} failed: {e}")
        if base_config.verbose:
            traceback.print_exc()
        return entry

    entry["time"] = results.get("total_time")
    metrics = results.get("evaluation")
    if metrics is None:
        entry["error"] = "no schedule produced"
        return entry

    entry["status"] = "ok"
    entry["metrics"] = metrics.to_dict()
    with open(output_dir / f"metrics_{name}.json", "w") as f:
        json.dump({**entry["metrics"], "pipeline": name, "time": entry["time"]}, f, indent=2)
    return entry


def write_summary(entries, args, output_dir: Path):
    """Write comparison_summary.md and comparison_metrics.json."""
    def fmt(entry, key):
        if entry["status"] != "ok":
            return "-"
        return str(entry["metrics"][key])

    lines = [
        "# Pipeline Comparison",
        "",
        f"- Data: `{args.data_dir}`",
        f"- Config: `{args.config or 'defaults'}`",
        f"- Generated: {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "All metrics are computed by the ScheduleEvaluator (lower is better).",
        "",
        "| Pipeline | Status | Time | Missed attendance | Session hops "
        "| Incoherent sessions | Presenter violations |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        time_str = f"{e['time']:.1f}s" if e["time"] is not None else "-"
        status = e["status"] if not e["error"] else f"{e['status']} ({e['error'].splitlines()[0]})"
        lines.append(
            f"| {e['pipeline']} | {status} | {time_str} | {fmt(e, 'total_missed_attendance')} "
            f"| {fmt(e, 'total_session_hops')} | {fmt(e, 'incoherent_sessions')} "
            f"| {fmt(e, 'presenter_violations')} |")

    (output_dir / "comparison_summary.md").write_text("\n".join(lines) + "\n")
    with open(output_dir / "comparison_metrics.json", "w") as f:
        json.dump({"data_dir": args.data_dir, "config": args.config,
                   "generated": datetime.now().isoformat(timespec="seconds"),
                   "pipelines": entries}, f, indent=2)


def main():
    args = parse_args()

    pipelines = [p for p in args.pipelines
                 if not (args.skip_traditional and p == "traditional")
                 and not (args.skip_matching and p.startswith("matching"))]
    assert set(pipelines) <= set(VALID_CHOICES["pipeline"])

    try:
        base_config = build_base_config(args)
    except SchedulerError as e:
        print(f"❌ {e}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PIPELINE COMPARISON")
    print("=" * 70)
    print(f"Data:      {args.data_dir}")
    print(f"Pipelines: {', '.join(pipelines)}")
    print(f"Output:    {output_dir}")

    entries = []
    for name in pipelines:
        print(f"\n{'#' * 70}\n# {name.upper()}\n{'#' * 70}")
        entries.append(run_one(name, base_config, output_dir))

    write_summary(entries, args, output_dir)

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY (ScheduleEvaluator metrics, lower is better)")
    print("=" * 70)
    print(f"{'Pipeline':<22}{'Status':<9}{'Time':>8}{'Missed':>8}{'Hops':>7}"
          f"{'Incoh.':>8}{'Viol.':>7}")
    print("-" * 69)
    for e in entries:
        m = e["metrics"] or {}
        time_str = f"{e['time']:.1f}s" if e["time"] is not None else "-"
        print(f"{e['pipeline']:<22}{e['status']:<9}{time_str:>8}"
              f"{m.get('total_missed_attendance', '-'):>8}{m.get('total_session_hops', '-'):>7}"
              f"{m.get('incoherent_sessions', '-'):>8}{m.get('presenter_violations', '-'):>7}")
    print(f"\nDetails: {output_dir / 'comparison_summary.md'}")

    return 0 if all(e["status"] != "failed" for e in entries) else 1


if __name__ == "__main__":
    sys.exit(main())
