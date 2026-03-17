# Column Generation for Phase 1

This module provides a column generation solver for Phase 1 tuple selection, useful for large instances where explicit enumeration is intractable.

## Files

- **`phase1_column_generation.py`** — Core solver with enumeration-based pricing
- **`phase1_column_generation_enhanced.py`** — Advanced solver with multiple pricing strategies (auto, enumeration, local_search, beam_search)

## Documentation

See [docs/column_generation.md](../../docs/column_generation.md) for the full algorithm description, mathematical formulation, usage guide, and tuning tips.
