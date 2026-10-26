# Changelog

## [Unreleased]

### Fixed
- Column generation built its initial columns by listing all combinations of talks (e.g. ~225M 5-tuples for ORBEL 2026: 7.9 GB of memory and ~6 minutes, after which no CG iteration was left within the time limit). Candidates are now sampled directly; the traditional pipeline on ORBEL 2026 takes ~20s instead of several minutes
- The `block_types` override in the config file ignored `count` (one block per entry)
- Infeasible talk pairs (presenter availability) were never detected with fewer than 5 timeslots
- Results were not reproducible: the greedy Phase 1 broke ties by set iteration order, which changes between Python processes (e.g. 143–196 missed attendances on ORBEL 2026 for the same input), and the randomized steps were unseeded. The greedy now breaks ties by the order of `talks.csv`, and all randomized steps use a seeded generator, so the same input and seed give the same schedule
- The `heuristic` pipeline and the evaluator now work without Gurobi (importing `src` no longer imports `gurobipy`; the Phase 2 retry falls back to networkx)
- Talk keywords were dropped by the data loader (renamed to `track`), so the incoherent-sessions metric was always 0, swap optimization ignored keywords and `matching_constrained` had no keywords
- Swap optimization ignored the `PLACEHOLDER_` talks created by the pipelines (it only recognized `DUMMY`), so empty slots could end up in short blocks or clustered in one session
- `matching_constrained` crashed when `sessions.csv` has more slots than talks (e.g. `examples/orbel2026`)
- `phase4.fixed_block_sessions` crashed with the `traditional` and `matching` pipelines and could place a session twice; all pipelines now reserve the room in the target block before optimizing
- Fixed sequences (`phase2.fixed_sequences`) did not keep their talk order
- Column generation crashed (`UnboundLocalError`) when initialization used up the Phase 1 time limit
- `scripts/evaluate_schedule.py` and `examples/load_data.py` failed with `ModuleNotFoundError`; `python -m src.schedule_evaluator --talks` failed on the standard `talks.csv`
- The evaluator derived presenters from talk IDs and did not normalize integer presenter IDs in `availability.csv`
- `scripts/compare_all_pipelines.py` optimized without preferences (it never called `validate()`), so its results were meaningless
- A broken or missing config file was silently ignored, and unknown option values (e.g. `phase1.method: "greedy"`) silently fell back to other behavior
- JSONC comment stripping corrupted strings containing `//` or `/*`
- `phase3.time_limit` was ignored by the traditional pipeline
- Pipelines mutated the shared block types when handling fixed sequences
- Too few configured rooms caused an unexplained Phase 4 failure; more talks than slots failed deep inside the pipelines
- Personal itineraries counted room changes between blocks as room switches
- Test suite: `tests/test_phase2.py` ran a 2-minute MILP at import time and wrote `session_schedule.csv`; tests returned values instead of asserting

### Changed
- `scripts/compare_all_pipelines.py` is now a thin wrapper around `run_schedule.py` (same configuration and metrics; new `--config`, `--pipelines`, `--time-limit` options)
- Presenter infeasibility after Phase 2 is first repaired by swapping tuples between blocks before re-partitioning (much faster, e.g. synthetic traditional run 31s → 2s)
- Output schedules (CSV, Markdown, itineraries) follow the block order of `sessions.csv` instead of alphabetical order
- More slots than talks is no longer reported as a validation warning; more talks than slots is an error
- The heuristic pipeline's Hungarian Phase 3 uses the shared implementation in `src/phase3.py`
- Column generation (traditional pipeline) is seeded with the heuristic pipeline's greedy solution, which also starts the final MIP, so its Phase 1 result is never worse than the greedy one (new `initial_columns` argument of `Phase1ColumnGenerationEnhanced`). docs/column_generation.md explains why the greedy solution could beat column generation before
- Documentation reviewed against the code: the specifications now describe the implemented hopping definition (room switches), partition strategies, Phase 1 filters, Phase D sampling, the column generation `auto` rule and parameters, and the soft keyword constraint (`--max-violations` is a starting budget); stale data structures, examples and non-existent config options were removed
- **Benchmarks use ScheduleEvaluator metrics exclusively** — consistent measurement across all pipelines
- Clarified that column generation is a near-optimal heuristic (generates columns on demand, solves LP optimally but uses only generated columns for the IP)
- Updated Phase 1 recommendation: `column_generation` for 50+ talks

### Added
- `seed` config key and `--seed` option (`run_schedule.py`, `compare_all_pipelines.py`; default 0) for reproducible results and alternative schedules
- `run_scheduler(config)` for programmatic use of the full workflow
- `--data-dir` option for `scripts/evaluate_schedule.py`
- Config keys `matching_pipeline.require_common_keyword` and `matching_pipeline.max_keyword_violations`
- Optional `title`, `presenter_name`, `authors` and `special_group` columns in `talks.csv`
- Tests for the workflow (`tests/test_workflow.py`) and real Phase 2 tests
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
