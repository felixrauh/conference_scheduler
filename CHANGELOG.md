# Changelog

## [Unreleased]

### Added
- Modified slides from conference talk about the scheduling process with some visualizations and takeaways
- `scripts/batch_evaluate.py` — batch evaluation script for comparing schedules with consistent evaluator metrics
- Column name normalization in `load_schedule_csv()` so the `ScheduleEvaluator` works with Phase 4 CSV exports (`Block_ID`/`Room_ID` → `Session_ID`/`Room`)
- `--phase2-partition` CLI argument for `run_schedule.py` — choose `greedy`, `matching`, or `random` partition strategy directly from the command line
- Placeholder talk support in explicit Phase 1 method (handles slot count > talk count)
- **Presenter Unavailability Feasibility** for all pipelines
  - Traditional/Heuristic: infeasible pair filtering in Phase 1, retry with no-good cuts in Phase 2
  - Matching: infeasible pair/block filtering in Phases A/B, no-good cut retry in Phase C
- **Column Generation solver** for Phase 1 (`src/columngeneration_phase1/`)
  - Multiple pricing strategies: auto, enumeration, local search, beam search
  - 99.99% reduction in variables for large instances (125 vs 182M for 118 talks)
- **Adaptive Heuristic Filtering** for explicit enumeration

### Changed
- **Benchmarks use ScheduleEvaluator metrics exclusively** — consistent measurement across all pipelines
- Clarified that column generation is a near-optimal heuristic (generates columns on demand, solves LP optimally but uses only generated columns for the IP)
- Updated Phase 1 recommendation: `column_generation` for 50+ talks

---

## [1.0.0] - Original Implementation

### Features
- Three-phase conference scheduling optimization
- Phase 1: Tuple selection via MILP (explicit enumeration)
- Phase 2: Block assembly and ordering
- Phase 3: Timeslot scheduling and room assignment
- CSV export format

### Reference
Based on: Vangerven, B., et al. (2018). Conference scheduling — A personalized approach. *Omega*, 81, 38-47.
