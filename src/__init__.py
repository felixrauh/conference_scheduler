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
"""

__version__ = "0.2.0"

# Main pipeline functions
from .phase1 import solve_phase1
from .phase2 import solve_phase2
from .phase3 import solve_phase3
from .phase4 import solve_phase4
from .matching_pipeline import run_matching_pipeline

# Post-processing
from .swap_optimization import (
    optimize_presenter_violations,
    SwapResult,
    ViolationType,
    detect_violations,
    detect_dummy_violations
)

# Schedule evaluation
from .schedule_evaluator import ScheduleEvaluator, EvaluationMetrics, evaluate_schedule
