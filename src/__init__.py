"""
Conference Scheduling Package

Based on: Vangerven et al. (2018) "Conference scheduling — A personalized approach"

Terminology:
- Block: A set of parallel talks across all rooms during one timeslot
         (e.g., 5 rooms × 4 slots = 20 talks). Named TA, TB, FC, etc.
- Session (or room-session): The set of sequential talks in ONE room within
         a block (e.g., 4 talks in Room A during block TA).

Includes:
- Original top-down pipeline: Phase 1 → Phase 2 → Phase 3 → Phase 4
- Alternative matching pipeline: Phase A → Phase B → Phase C → Phase D → Phase 3 → Phase 4
- Schedule evaluation: Quality metrics computation

The names below are imported lazily, so that importing the package (e.g. for
the heuristic pipeline or the evaluator) does not require Gurobi.
"""

import importlib

__version__ = "0.2.0"

_LAZY_EXPORTS = {
    # Main pipeline functions
    "solve_phase1": ".phase1",
    "solve_phase2": ".phase2",
    "solve_phase3": ".phase3",
    "solve_phase4": ".phase4",
    "run_matching_pipeline": ".matching_pipeline",
    # Post-processing
    "optimize_presenter_violations": ".swap_optimization",
    "SwapResult": ".swap_optimization",
    "ViolationType": ".swap_optimization",
    "detect_violations": ".swap_optimization",
    "detect_dummy_violations": ".swap_optimization",
    # Schedule evaluation
    "ScheduleEvaluator": ".schedule_evaluator",
    "EvaluationMetrics": ".schedule_evaluator",
    "evaluate_schedule": ".schedule_evaluator",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module = importlib.import_module(_LAZY_EXPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
