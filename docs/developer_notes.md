# Developer Notes

Internal implementation details for contributors. For user-facing configuration, see [configuration_and_algorithm_reference.md](configuration_and_algorithm_reference.md).

---

## Workflow Overview

`scripts/run_schedule.py` is the single entry point. `main()` parses CLI arguments and the config file into a `SchedulerConfig`, then calls `run_scheduler(config)`, which can also be used programmatically (the comparison script does so):

1. `load_from_csv_dir()` reads the CSVs into `ConferenceData`; `validate()` builds the preference and availability lookups.
2. `apply_fixed_block_sessions()` reserves rooms for `phase4.fixed_block_sessions` by shrinking their timeslots and recomputing block types.
3. `build_instance()` creates the `ProblemInstance`; fixed-session talks are removed from `instance.talks`.
4. The pipeline runs (placeholders are added where slots exceed talks), followed by swap optimization, Phase 4, export and evaluation.

User errors (bad config, inconsistent data) raise `SchedulerError` and are printed without a traceback.

`src/__init__.py` exports its main functions lazily, so `import src...` does not require Gurobi. Keep `gurobipy` imports inside functions (or in modules only the Gurobi pipelines import) so the heuristic pipeline and the evaluator keep working without it; the tests run with and without Gurobi.

### Reproducibility

`run_scheduler()` calls `random.seed(config.seed)` before anything else, so every step that uses Python's global `random` generator is reproducible. When adding code:

- Use the global `random` module (or a `random.Random` seeded from it) for randomized steps — never an unseeded generator such as `random.Random()` or `numpy.random.default_rng()`.
- Do not let results depend on the iteration order of a `set` of strings: string hashing (and thus set order) changes between Python processes. Iterate in a defined order instead, e.g. `sorted(talk_set, key=position.get)` as in `_greedy_phase1()`.

`tests/test_workflow.py` checks both (same seed → same schedule; greedy result independent of `PYTHONHASHSEED`).

### Placeholders

When `sessions.csv` has more slots than there are talks, the pipelines add talks named `PLACEHOLDER_<n>` (older scripts used `DUMMY_<n>`). Use `src.utils.is_placeholder()` to recognize them; swap optimization and the evaluator rely on it.

---

## Presenter Availability — Implementation Details

Presenter availability is enforced through multiple mechanisms. See the user-facing [Presenter Availability Constraints](configuration_and_algorithm_reference.md#presenter-availability-constraints) section for the high-level overview.

### Infeasible Talk Pairs

**Location**: `src/instance.py` → `compute_infeasible_talk_pairs()`

Two talks form an infeasible pair if their presenters' combined unavailabilities cover too many timeslots, leaving insufficient scheduling options.

```python
def compute_infeasible_talk_pairs(
    instance: ProblemInstance,
    min_unavailable_threshold: int = 5,  # not exposed in the config file
    verbose: bool = False
) -> Set[Tuple[str, str]]:
```

The threshold is capped at the number of timeslots, so with fewer than 5 timeslots only pairs without any common available timeslot are excluded.

**Threshold semantics** (assuming 7 total timeslots):
- `min_unavailable_threshold=7`: Exclude only pairs with no common available timeslot (fewest exclusions)
- `min_unavailable_threshold=5`: Exclude pairs with ≤2 available timeslots (default)
- `min_unavailable_threshold=3`: Exclude pairs with ≤4 available timeslots (most exclusions)

**Example**: If Talk A's presenter is unavailable for {TA, TB, TC} and Talk B's presenter is unavailable for {TC, TD, FA}, their union is {TA, TB, TC, TD, FA} = 5 timeslots unavailable. With `min_unavailable_threshold=5`, this pair is excluded.

**Where it's used**:

| Pipeline | Phase | Usage |
|----------|-------|-------|
| Traditional | Phase 1 (explicit) | Excludes infeasible tuples from enumeration |
| Traditional | Phase 1 (column_generation) | Not used yet (neither are same-presenter checks) |
| Heuristic | Phase 1 (greedy) | Prevents infeasible pairs from being grouped |
| Matching | Phase A | Prevents infeasible pairs from being matched |

### Tuple Size Restrictions

**Location**: `src/instance.py` → `compute_forbidden_tuple_sizes()`

Some talks cannot be placed in certain tuple sizes because:
- Size-3 tuples → 3R4T blocks → only scheduled in certain timeslots (e.g., FA only)
- Size-5 tuples → 5R4T/5R3T blocks → only scheduled in other timeslots (e.g., TA, TB, TC)

If a presenter is unavailable for ALL timeslots of a block type, their talk cannot be in tuples of that size.

**Where it's used**:

| Pipeline | Phase | Usage |
|----------|-------|-------|
| Heuristic | Phase 1 (greedy) | Prevents assigning talks to forbidden tuple sizes |
| Traditional | Phase 1 | Not yet implemented |

### Block Feasibility Checking

**Location**: `src/phase2.py` → `check_block_feasibility()`, `check_all_blocks_feasibility()`

After Phase 2 groups tuples into blocks, we verify each block can be assigned to at least one timeslot where all presenters are available.

```python
def check_block_feasibility(
    block: Block,
    talk_presenter: Dict[str, str],
    presenter_unavailability: Dict[str, Set[str]],
    all_timeslots: Set[str],
    timeslots_by_type: Optional[Dict[str, List[str]]] = None
) -> Tuple[bool, Set[str]]:
```

**Type-aware checking**: When `timeslots_by_type` is provided, only timeslots matching the block's type are considered feasible. The block type is accessed via `block.block_type`.

### Feasibility Repair

**Location**: `src/phase2.py` → `repair_block_feasibility()`, used by `solve_phase2_with_feasibility_check()`

When a block has no feasible timeslot, whole tuples are swapped between blocks with the same number of rooms (so every block keeps its shape). Each step applies the swap that makes the most infeasible blocks feasible without breaking a feasible one, preferring the smallest hopping increase; swapped blocks are re-ordered. Only if the repair fails does the retry loop re-partition (greedy: perturbed positions; matching: excluding a previous matching, which needs Gurobi — without it, the edge costs are randomly perturbed instead).

### Known Limitations

1. **Greedy Phase 1 can fail**: If too many talks have size restrictions, the greedy algorithm may consume eligible talks for larger sizes first, leaving insufficient talks for smaller sizes. This causes "Not enough n-tuples" errors.

2. **Phase 2 partition doesn't check cross-tuple constraints**: The partition step groups tuples into blocks without verifying that combining certain tuples creates feasible blocks. The feasibility check and repair happen AFTER partitioning. The no-good cuts used when re-partitioning with the `matching` strategy are approximate (they exclude tuple-index pairs, not the actual matching).

3. **Block type constraints are tight**: Some block types only have 1 valid timeslot (e.g., 3R4T → FA only). Talks that can only be in these block types have very limited scheduling flexibility.

### Debugging Presenter Violations

If you see presenter violations in the output:

1. **Check constraint loading**: Ensure `data.validate()` is called to populate `presenter_unavailability`
2. **Increase retries**: Raise `max_feasibility_retries` in the pipeline's `PipelineConfig` (`src/pipelines/*.py`; not exposed in the config file)
3. **Lower threshold**: Pass `min_unavailable_threshold=3` to `compute_infeasible_talk_pairs()` (code change) to exclude more pairs
4. **Check block type mappings**: Verify `timeslots_by_type` correctly maps block types to timeslots

---

## Swap Optimization — Implementation Details

### Violation Types

| Type | Description | Resolution |
|------|-------------|------------|
| `PRESENTER_UNAVAILABLE` | Presenter scheduled in unavailable timeslot | Swap to timeslot where presenter is available |
| `DUMMY_IN_SHORT_BLOCK` | Placeholder talk in a block with ≤3 timeslots | Swap with a real talk from a longer block |
| `MULTIPLE_DUMMIES_IN_SESSION` | 2+ placeholder talks in same room-session | Swap extra placeholders to different room-sessions |

### Algorithm Detail

1. **Detect Violations**: Find all presenter and dummy violations
2. **Generate Candidates**: For each violation, find all feasible swap partners
3. **Score Swaps**: Evaluate each swap by:
   - Feasibility: Does not create new violations
   - Missed attendance delta: Preference conflicts avoided/created
   - Keyword coherence (optional): Impact on session topical coherence
4. **Apply Best Swap**: Greedily apply the best-scoring feasible swap
5. **Iterate**: Repeat until no violations remain or no feasible swaps exist

### Scoring Formula

```
combined_score = missed_attendance_delta - keyword_weight × keyword_delta
```

- `missed_attendance_delta`: Change in conflicts (negative = improvement)
- `keyword_delta`: Change in session keyword coherence (positive = improvement)
- Lower combined score = better swap

### Placeholder (Dummy) Talk Handling

Placeholder talks (`PLACEHOLDER_1`, `PLACEHOLDER_2`, …; `DUMMY_*` is also recognized) fill empty slots when total_slots > total_talks. The swap optimization:

1. **Avoids short blocks**: Moves dummies from blocks with k≤3 slots to blocks with k>3 slots
2. **Spreads dummies**: Ensures no room-session has multiple dummies (avoids multiple empty slots in sequence)
3. **Minimizes impact**: Prefers swaps that don't increase missed attendance

### Complexity

- Time: O(V × T² × P) where V=violations, T=talks, P=participants
- In practice much faster due to early filtering of infeasible candidates

### Example Output

```
SWAP OPTIMIZATION: RESOLVING SCHEDULE VIOLATIONS
==================================================
  Presenter violations: 2
    - T042 (P042) @ TA
    - T089 (P089) @ FB

  Dummy violations: 3
    - PLACEHOLDER_1 @ FA (short block)
    - PLACEHOLDER_2 @ FA (multi-dummy)
    - PLACEHOLDER_3 @ FB (short block)

  Swap 1 [DUMMY]: PLACEHOLDER_1 <-> T071 (Δmissed=0, Δkeyword=+0.00)
  Swap 2 [PRES]: T042 <-> T098 (Δmissed=-2, Δkeyword=+0.15)
  Swap 3: T089 <-> T023 (Δmissed=0, Δkeyword=+0.10)

  Summary:
    Initial violations: 5
    Final violations:   0
    Swaps applied:      3
```

---

## Phase 1 — Non-Configurable Parameters

These parameters exist in `src/phase1.py` but are not exposed via config file. They use sensible defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_heuristic_filter` | `True` | Enable adaptive cost filtering for large instances (auto-activates when >1M tuples estimated) |

---

## Evaluator — Complexity

| Metric | Complexity |
|--------|------------|
| Missed Attendance | O(P × T) where P = participants, T = timeslots |
| Session Hops | O(P × B × k × n²) where B = blocks, k = slots/block, n = rooms |
| Incoherent Sessions | O(S × k × K) where S = room-sessions, K = avg keywords/talk |
| Presenter Violations | O(T) where T = total talks |

---

## Matching Pipeline — Solution Methods

The phase methods are fixed; there are no alternative implementations to switch to:

| Phase | Method | Solver |
|-------|--------|--------|
| A: Pair matching | Maximum weight matching with cardinality constraint (MILP) | Gurobi |
| B: Block formation | Joint MILP over pair+pair and pair+single blocks | Gurobi |
| C: Tuple selection | Set partitioning (MILP), with no-good cuts on retry | Gurobi |
| D: Talk ordering | Enumeration (≤ 100,000 orderings) or 50,000 random samples | None |

Greedy or Blossom-based variants would be natural additions for running the matching pipeline without Gurobi. See [matching_pipeline_specification.md](matching_pipeline_specification.md) for the formulations.
