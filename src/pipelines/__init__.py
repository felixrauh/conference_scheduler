"""
Pipeline implementations for conference scheduling.

Available pipelines:
- traditional: 3-phase optimization (Phase 1 → 2 → 3)
- heuristic: Greedy alternative (no Gurobi required)
- matching: Bottom-up matching (Phase A → B → C → D → 3)
"""

from .traditional import run_traditional_pipeline
from .heuristic import run_heuristic_pipeline

__all__ = ['run_traditional_pipeline', 'run_heuristic_pipeline']
