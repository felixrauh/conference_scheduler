"""
Column Generation module for Phase 1 optimization.

Provides an alternative solution method that generates columns (n-tuples)
dynamically rather than enumerating all possibilities upfront.

Main classes:
- Phase1ColumnGeneration: Basic column generation solver
- Phase1ColumnGenerationEnhanced: Advanced solver with multiple pricing strategies
"""

from .phase1_column_generation import (
    Phase1ColumnGeneration,
    compute_tuple_cost,
)

from .phase1_column_generation_enhanced import (
    Phase1ColumnGenerationEnhanced,
    PricingProblemSolver,
)

__all__ = [
    'Phase1ColumnGeneration',
    'Phase1ColumnGenerationEnhanced',
    'PricingProblemSolver',
    'compute_tuple_cost',
]
