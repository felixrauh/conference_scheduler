# Configuration & Algorithm Reference

This document provides a comprehensive reference for all configurable settings and algorithm choices. Each option shows its **config file key** (for `config/scheduling_config.jsonc`) and, where available, its **CLI flag**.

Settings are applied via config file, CLI flags, or both. CLI flags override config file values.

```bash
# Config file only
python scripts/run_schedule.py --config config/scheduling_config.jsonc

# CLI overrides
python scripts/run_schedule.py --config config/scheduling_config.jsonc --pipeline matching

# CLI only (uses defaults for everything not specified)
python scripts/run_schedule.py --pipeline heuristic --data-dir examples/synthetic
```

## Terminology

- **Block**: A set of parallel talks across all rooms during one timeslot (e.g., 5 rooms × 4 slots = 20 talks); one per row of `sessions.csv`, e.g. TA, TB, …, FC (Thursday-A, Friday-B, etc.).
- **Session** (or **room-session**): The set of sequential talks in ONE room within a block (e.g., 4 talks in Room A during block TA).
- **Tuple**: An n-tuple of talks scheduled in parallel at the same timeslot (one per room).

## Overview: Pipeline Types

The scheduler offers two pipeline families that differ in what they decide first:

- **Parallel-talks-first** (`traditional`, `heuristic`): Starts by choosing which talks should run at the same time (minimizing preference conflicts), then groups those parallel sets into session blocks, orders them, and assigns rooms. The `traditional` variant uses Gurobi for optimal tuple selection; the `heuristic` variant uses a greedy approach and requires no solver.

- **Sessions-first** (`matching`, `matching_constrained`): Starts by pairing talks that audiences want to see together, then assembles those pairs into complete session blocks, then decides which blocks run in parallel, and finally orders talks within blocks to allow beneficial room-hopping. The `matching_constrained` variant additionally supports special-group and keyword constraints within sessions (see [matching_constrained](#matching_constrained)). Both require Gurobi.

**Config key:** `pipeline` | **CLI:** `--pipeline`

| Pipeline | Approach | Solver Required | Speed (118 talks) |
|----------|----------|-----------------|-----|
| **traditional** | Parallel-talks-first | Gurobi | ~20 sec |
| **heuristic** | Parallel-talks-first | None (scipy only) | ~2 sec |
| **matching** | Sessions-first | Gurobi | ~1 min |
| **matching_constrained** | Sessions-first | Gurobi | ~1 min |

```jsonc
// Config example
"pipeline": "traditional"
```

### Phase Overview

**Parallel-talks-first:** Phase 1 (tuple selection) → Phase 2 (block assembly & ordering) → Phase 3 (block scheduling) → Swap optimization → Phase 4 (finalization & room assignment)

**Sessions-first:** Phase A (pair matching) → Phase B (block formation) → Phase C (tuple selection) → Phase D (talk ordering) → Phase 3 (block scheduling) → Swap optimization → Phase 4 (finalization & room assignment)

Phase 3, swap optimization, and Phase 4 are shared across both families.

---

## General Settings

| Config key | CLI | Default | Description |
|------------|-----|---------|-------------|
| `pipeline` | `--pipeline` | `"traditional"` | Pipeline to run (see above) |
| `seed` | `--seed` | `0` | Seed for the randomized steps; see [Reproducibility](#reproducibility) |
| `input.data_dir` | `--data-dir` | `"examples/orbel2026"` | Directory with `talks.csv`, `preferences.csv`, `sessions.csv` (and optionally `availability.csv`) |
| `output.dir`, `output.schedule_csv` | `--output` | `output/schedule.csv` | Output CSV; a Markdown version is written next to it |
| `block_types` | — | `"auto"` | `"auto"` reads the structure from `sessions.csv` (recommended). Alternatively a dict like `{"5R4T": {"n": 5, "k": 4, "count": 3}}`; this creates timeslots named `5R4T_1`, `5R4T_2`, … (or `5R4T` if `count` is 1), which `availability.csv` and fixed sessions must then use |
| — | `--verbose` | off | Detailed solver output |

Unknown top-level sections in the config file produce a warning; invalid option values stop the run with an error.

### Reproducibility

The same input files, configuration and `seed` always produce the same schedule (with the same package versions; for the Gurobi pipelines see the note below). Several steps are randomized — Phase 2 local search, column generation's initial sampling, the matching pipeline's Phase D sampling for large blocks — and all of them use the seeded generator; the greedy constructions break ties by the order of `talks.csv`.

Different seeds give alternative schedules. The seed affects the randomized steps only: mainly Phase 2's local search (so mostly the session hops) and, through the resulting block layout, swap optimization. The greedy Phase 1 of the `heuristic` pipeline and the greedy seed of column generation do not depend on it. To look for a better schedule, run a few seeds and compare the evaluator metrics:

```bash
for s in 0 1 2 3 4; do
    python scripts/run_schedule.py --pipeline heuristic --seed $s --output output/schedule_seed$s.csv
done
```

Gurobi is deterministic for the same model, parameters and machine, but results can differ between machines (e.g. with a different number of threads). If a time limit is reached, the solve stops at a wall-clock-dependent point, which can also change results on a slower or busy machine.

---

# Parallel-Talks-First Pipeline (traditional / heuristic)

## Phase 1: Tuple Selection

**Goal**: Select n-tuples of talks to run in parallel, minimizing missed attendance.

> For mathematical formulations and data structures, see [phase1_specification.md](phase1_specification.md).

**Config key:** `phase1.method` | **CLI:** `--phase1-method`

| Method | Description | Solver | Scalability | Solution Quality |
|--------|-------------|--------|-------------|------------------|
| `explicit` | Generate (almost) all tuples, filter, solve MIP | Gurobi | 30-100 talks | (Almost) Optimal (with filtering) |
| `column_generation` | Generate tuples on-demand via pricing, seeded with the greedy solution | Gurobi | 50-150+ talks | At least as good as the greedy seed; no optimality proof (see [column_generation.md](column_generation.md)) |

The `heuristic` pipeline uses a greedy construction instead (see [Greedy Heuristic](#greedy-heuristic)); it is not selectable via `phase1.method`.

```jsonc
// Config example: switch to column generation
"phase1": {
    "method": "column_generation",
    "time_limit": 300.0
}
```

### Explicit Enumeration Options

Config section: `phase1.explicit`

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `phase1.time_limit` | float | 300.0 (bundled config file: 120.0) | MIP solver time limit (seconds); CLI: `--time-limit` |
| `phase1.explicit.max_cost` | int | null | Cost filter threshold for tuple filtering. `null` = auto-calculate. Lower values = more aggressive filtering for large instances |

The heuristic cost filter (`use_heuristic_filter` in `src/phase1.py`) activates automatically for large instances and is not exposed via config.

**When to use**: small instances (fewer than ~50 talks), when you want a provably optimal Phase 1. For larger instances the number of tuples explodes (e.g. ~182M for 118 talks); above ~1M tuples heuristic filters are applied, so optimality is no longer guaranteed (see [phase1_specification.md](phase1_specification.md#filtering-infeasible-tuples)).

### Column Generation Options

Config section: `phase1.column_generation`. See [column_generation.md](column_generation.md) for full documentation.

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `phase1.time_limit` | float | 300.0 (bundled config file: 120.0) | Total time limit (seconds); CLI: `--time-limit` |
| `phase1.column_generation.pricing_strategy` | string | `"auto"` | Pricing problem solver strategy |

```jsonc
// Config example
"phase1": {
    "method": "column_generation",
    "time_limit": 300.0,
    "column_generation": {
        "pricing_strategy": "auto"
    }
}
```

#### Pricing Strategies

| Strategy | Description |
|----------|-------------|
| `auto` | `enumeration` if a tuple size has at most 10,000 candidate tuples (≈ 22 talks for 4 rooms), otherwise `local_search` for up to 5 rooms and `beam_search` beyond (recommended) |
| `enumeration` | Exact pricing; only practical for very small instances |
| `greedy` | Greedy construction; fastest |
| `local_search` | Greedy start plus swap improvement |
| `beam_search` | Beam search with pruning |

Only exact pricing (with convergence) gives a proven lower bound; see [column_generation.md](column_generation.md#bounds-and-optimality).

### Greedy Heuristic

Used automatically by the `heuristic` pipeline. No user-configurable options beyond `phase1.time_limit`. The `heuristic` pipeline also fixes its Phase 2 partition strategy (`matching`) and Phase 3 method (`hungarian`).

---

## Phase 2: Block Assembly & Ordering

**Goal**: Partition tuples into blocks and order them to minimize room hopping.

> For mathematical formulations (hopping number, DP, partitioning approaches), see [phase2_specification.md](phase2_specification.md).

### Partition Strategy

**Config key:** `phase2.partition_strategy` | **CLI:** `--phase2-partition` (traditional pipeline only)

| Strategy | Description | Quality | Dependencies |
|----------|-------------|---------|--------------|
| `greedy` | Conflict-based greedy assignment | Good | None |
| `random` | Random assignment | Baseline | None |
| `matching` | Assignment problem + perfect matching | Often better (dataset-dependent) | scipy, networkx |

The `matching` strategy follows the paper's approach: it builds a complete graph over tuples, solves a minimum-cost perfect matching, and merges matched pairs into blocks of 2 or 4 tuples. Blocks of other sizes (e.g. 3 talks per room) use a greedy partition. See [phase2_specification.md](phase2_specification.md#partition-strategies).

### Ordering Strategy

**Config key:** `phase2.ordering_strategy` (config file only, no CLI flag)

| Strategy | Description | Quality |
|----------|-------------|---------|
| `enumerate` | Try all k! permutations | Optimal (use for k ≤ 5) |
| `greedy` | Incremental greedy ordering | Good (use for k > 5) |

### All Phase 2 Options

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `phase2.partition_strategy` | string | `"greedy"` (bundled config file: `"matching"`) | Partition method |
| `phase2.ordering_strategy` | string | `"enumerate"` | Ordering method |
| `phase2.local_search_iterations` | int | 2000 | Post-optimization swap iterations (traditional pipeline; the heuristic pipeline uses 100) |
| `phase2.fixed_sequences` | array | `[]` | Pre-specified sessions (see [Pre-specified Sessions](#pre-specified-sessions-fixed-blocks)) |

```jsonc
// Config example: fast settings for large instances
"phase2": {
    "partition_strategy": "greedy",
    "ordering_strategy": "greedy",
    "local_search_iterations": 500
}
```

---

## Phase 3: Block Scheduling (shared across all pipelines)

**Goal**: Assign blocks to timeslots to minimize presenter availability violations.

> For the bipartite assignment formulation and room capacity matching, see [phase3_room_assignment.md](phase3_room_assignment.md).

**Config key:** `phase3.method` | **CLI:** `--phase3-method`

| Method | Description | Solver | Quality |
|--------|-------------|--------|---------|
| `milp` | Bipartite matching via MIP | Gurobi | Optimal |
| `hungarian` | Hungarian algorithm | scipy | Optimal |

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `phase3.method` | string | `"milp"` | `"milp"` (Gurobi) or `"hungarian"` (scipy); the `heuristic` pipeline always uses `"hungarian"` |
| `phase3.time_limit` | float | 60.0 | MIP solver time limit (seconds) |

```jsonc
// Config example: use scipy instead of Gurobi
"phase3": {
    "method": "hungarian"
}
```

---

## Swap Optimization (shared across all pipelines)

**Goal**: Resolve remaining presenter availability violations through local search, while minimizing impact on missed attendance.

After Phase 3, some violations may remain (presenters scheduled when unavailable, placeholder talks in undesirable positions). Swap optimization iteratively swaps talks between positions to fix these.

**Config section:** `swap_optimization`

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `swap_optimization.enabled` | bool | `true` | Enable/disable swap optimization |
| `swap_optimization.keyword_weight` | float | 0.1 | Weight for keyword coherence vs missed attendance (0.0–1.0). 0.0 = only minimize missed attendance |
| `swap_optimization.max_iterations` | int | 100 | Maximum number of swaps to attempt |
| `swap_optimization.check_dummy_violations` | bool | `true` | Also resolve placeholder (empty slot) placement issues |
| `swap_optimization.short_block_threshold` | int | 3 | Blocks with k ≤ this are "short" — placeholders placed there count as violations |

```jsonc
// Config example: disable keyword coherence, more iterations
"swap_optimization": {
    "enabled": true,
    "keyword_weight": 0.0,
    "max_iterations": 200
}
```

Presenter violations require `availability.csv` in the data directory. Placeholder violations occur when `sessions.csv` has more slots than there are talks: the empty slots are filled with `PLACEHOLDER_<n>` talks, which should not end up in short blocks or two in the same room-session.

### How It Works

1. Detects presenter violations (unavailable timeslots) and placeholder violations (placeholders in short blocks or clustered in one session)
2. For each violation, finds feasible swap partners
3. Scores each swap: `combined_score = missed_attendance_delta - keyword_weight × keyword_delta`
4. Greedily applies the best-scoring feasible swap
5. Repeats until no violations remain or no feasible swaps exist

---

## Phase 4: Finalization & Room Assignment (shared across all pipelines)

**Goal**: Assign physical rooms and insert any pre-specified sessions.

**Config section:** `phase4`, `rooms`

Room assignment is automatic — sessions with the largest expected audience get the largest rooms:
1. Sort sessions by total likes, i.e. the summed preferences for their talks (ascending)
2. Sort physical rooms by capacity (ascending)
3. Match: smallest audience → smallest room

If `rooms` is given, it must list at least as many rooms as the largest block (including fixed sessions) needs. Without it, rooms are named `Room_1`, `Room_2`, … with equal capacity.

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `rooms` | array | `[]` | Physical room definitions with `id`, `name`, `capacity` |
| `phase4.fixed_block_sessions` | array | `[]` | Pre-specified sessions (see [Pre-specified Sessions](#pre-specified-sessions-fixed-blocks)) |

```jsonc
// Config example
"rooms": [
    {"id": "A", "name": "Room A", "capacity": 250},
    {"id": "B", "name": "Room B", "capacity": 120},
    {"id": "C", "name": "Room C", "capacity": 80}
]
```

---

## Pre-specified Sessions (Fixed Blocks)

**Goal**: Allow certain talk groupings to be pre-specified and excluded from the main optimizer. Useful for award sessions, sponsored sessions, or any session whose content is already fixed before scheduling begins.

There are two mechanisms, depending on how much is pre-determined:

### Option A — `phase2.fixed_sequences`: timeslot chosen by optimizer

Use this when the talks and their order within one room are fixed, but the optimizer may still assign the resulting block to any matching timeslot.

The sequence is excluded from Phase 1 entirely. In Phase 2 it is attached as an extra room-column to a block of `target_block_type`, expanding it to `result_block_type`. Phase 3 then assigns that (now larger) block to a timeslot just like any other.

> Only available for the `traditional` and `heuristic` pipelines.

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Descriptive label (e.g., `"SpecialSession_1"`) |
| `talks` | array | Talk IDs in presentation order — exactly `k` entries |
| `target_block_type` | string | Block type to attach to (e.g., `"4R4T"`) |
| `result_block_type` | string | Block type after attachment (e.g., `"5R4T"`) |

```jsonc
// Config example: a 4-talk special session that adds one room to an existing 4R4T block
"phase2": {
    "fixed_sequences": [
        {
            "name": "SpecialSession_1",
            "talks": ["T101", "T102", "T103", "T104"],
            "target_block_type": "4R4T",
            "result_block_type": "5R4T"
        }
    ]
}
```

Keep `sessions.csv` as the final conference structure (including the room the sequence will occupy). For Phase 1, the optimizer turns one `result_block_type` block into a `target_block_type` block and attaches the sequence to it in Phase 2. The talks keep the given order.

### Option B — `phase4.fixed_block_sessions`: both grouping and timeslot are fixed

Use this when you know not only the talks but also exactly which timeslot the session must go to. The talks are excluded from Phases 1–3 entirely; Phase 4 inserts them into the named block and assigns a room.

Keep `sessions.csv` unchanged: for each fixed session, one room of the target block is reserved automatically before optimization (two fixed sessions in the same block reserve two rooms). A fixed session may have at most as many talks as the block has talks per room. Works with all pipelines.

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Descriptive label (e.g., `"SpecialSession_1"`) |
| `block` | string | Timeslot ID from `sessions.csv` (e.g., `"FA"`) |
| `talks` | array | Talk IDs in presentation order |

```jsonc
// Config example: two special sessions pinned to specific timeslots
"phase4": {
    "fixed_block_sessions": [
        {
            "name": "SpecialSession_1",
            "block": "FA",
            "talks": ["T101", "T102", "T103", "T104"]
        },
        {
            "name": "SpecialSession_2",
            "block": "FB",
            "talks": ["T105", "T106", "T107", "T108"]
        }
    ]
}
```

### Choosing between the two

| | `fixed_sequences` | `fixed_block_sessions` |
|---|---|---|
| Talks fixed | yes | yes |
| Timeslot fixed | no (optimizer decides) | yes |
| Pipeline support | traditional, heuristic | all pipelines |
| Phase applied | Phase 2 (attached to a block) | Room reserved before Phase 1, talks inserted in Phase 4 |

**When to use**:
- Award sessions with pre-determined speakers
- Sponsored sessions with a fixed lineup
- Invited or plenary sessions that must be in a specific slot

---

# Matching Pipeline (matching / matching_constrained)

The matching pipeline takes a sessions-first approach: it builds schedules by first grouping talks into coherent sessions, then deciding which sessions run in parallel. It has 4 pipeline-specific phases (A–D), followed by the shared Phase 3, swap optimization, and Phase 4.

Only sessions with 3 or 4 talks per room are supported. If `sessions.csv` has more slots than there are talks, placeholder talks fill the remaining slots.

**Config section:** `matching_pipeline`

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `matching_pipeline.time_limit` | float | 300.0 | Total time limit (CLI: `--time-limit`), distributed automatically: 15% pairing (A), 15% block formation (B), 50% tuple selection (C), 20% Phase 3. Phase D (enumeration or sampling) is not time-limited |
| `matching_pipeline.require_common_keyword` | bool | `false` | `matching_constrained` only: talks paired into a session must share a keyword (hard constraint; CLI: `--require-keyword`) |
| `matching_pipeline.max_keyword_violations` | int or null | `null` | `matching_constrained` only: soft keyword constraint. The value is the starting budget of pairs without a shared keyword; it is raised by 1 until a solution exists, so `0` means "as few violations as possible" (CLI: `--max-violations`) |

```jsonc
// Config example
"pipeline": "matching",
"matching_pipeline": {
    "time_limit": 300.0
}
```

The phase methods are fixed (MILP for Phases A–C, enumeration or sampling for Phase D). See [matching_pipeline_specification.md](matching_pipeline_specification.md) for the formulations.

### matching_constrained

Adds two kinds of pairing constraints to Phases A and B:

- **Special groups** (always on): if `talks.csv` has a `special_group` column, talks are only placed in a session with talks of the same group. Empty values mean `general`.
- **Keywords** (opt-in): with `require_common_keyword` every two talks in a session must share a keyword (hard constraint; pairwise, so a session can still lack one keyword common to all its talks). With `max_keyword_violations` the constraint becomes soft: the number of pairs (Phase A) and blocks (Phase B) without a shared keyword is limited by a budget that starts at the given value and grows until a solution exists. The hard constraint can be infeasible when many talks have no keywords (e.g. `examples/orbel2017`) — use `max_keyword_violations: 0` then.
- Special-group talks are exempt from the keyword constraint; talks without metadata (placeholders) can pair with any talk.

## Phase A: Pair Matching

**Goal**: Match talks into pairs to maximize co-preference (people wanting both talks).

Uses maximum weight matching via MIP (Gurobi).

## Phase B: Block Formation

**Goal**: Combine pairs/singles into 3-blocks and 4-blocks.

Uses joint MILP optimization (Gurobi).

## Phase C: Tuple Selection

**Goal**: Select which blocks run in parallel (similar to Phase 1 in the traditional pipeline).

Uses set partitioning via MIP (Gurobi).

## Phase D: Talk Ordering

**Goal**: Order talks within blocks to maximize room-hopping benefit.

Enumerates all orderings of the talks in each room of a block when there are at most 100,000 combinations, and otherwise evaluates 50,000 random orderings (e.g. 4 rooms × 4 talks: 24⁴ = 331,776 combinations). This step takes most of the matching pipeline's runtime on the example data.

> **Metric note:** Phase D optimizes _hopping benefit_ — the number of extra talks participants can attend by switching rooms within a block, compared to staying in a single room. This is **not** the same as the room-switching _cost_ minimized by the traditional pipeline's Phase 2. A higher hopping benefit means the schedule rewards room switching more. Use the evaluator's session hops to compare pipelines.

## Presenter Unavailability in Matching Pipeline

The matching pipeline enforces presenter availability at multiple levels:

| Phase | Mechanism |
|-------|-----------|
| **Phase A** | Infeasible talk pairs excluded from matching |
| **Phase B** | Blocks with infeasible presenter combinations excluded |
| **Phase C** | If Phase 3 finds violations, previous solution is forbidden and Phase C retries (up to 10 times) |

---

## Presenter Availability Constraints

Presenter availability (from `availability.csv`) is enforced through multiple mechanisms across all pipelines:

1. **Infeasible talk pairs** (explicit and heuristic Phase 1, matching Phase A): Talks whose presenters' combined unavailabilities leave too few valid timeslots are never placed together. Column generation does not use this filter yet; there, availability is handled from Phase 2 on.
2. **Tuple size restrictions** (heuristic Phase 1): If a presenter is unavailable for all timeslots of a block type, their talk is forbidden from that tuple size
3. **Block feasibility checking** (Phase 2): After grouping, each block is verified to have at least one feasible timeslot; infeasible blocks are repaired by swapping tuples between blocks, then by re-partitioning
4. **Block-to-timeslot assignment** (Phase 3): Optimal assignment minimizes remaining violations
5. **Swap optimization** (post-Phase 3): Resolves any remaining violations via talk swapping

If you see presenter violations in the final output:
- Ensure `availability.csv` is present in your data directory
- Increase `swap_optimization.max_iterations`
- Check that `swap_optimization.enabled` is `true`

---

## Decision Flowchart

```
Start
  │
  ├─ Have Gurobi license? ─── No ──→ pipeline = "heuristic"
  │         │                        (fixed choices: greedy Phase 1,
  │        Yes                        "matching" partition, Hungarian)
  │         │
  │         ├─ Want sessions grouped by shared audience? ──→ pipeline = "matching"
  │         │                                                (or "matching_constrained")
  │         ↓
  │    pipeline = "traditional"
  │         │
  │         ↓
  │    Talks ≥ 50? ─── Yes ──→ phase1.method = "column_generation"
  │         │                         │
  │        No                         ↓
  │         │                  pricing_strategy = "auto"
  │         ↓
  │    phase1.method = "explicit"
  │         │
  └─────────┴─────────────────────────→ Continue to Phase 2/3
```

---

## Performance Guidelines

### By Instance Size

| Talks | Recommended Configuration |
|-------|--------------------------|
| < 50 | `explicit` + `greedy` partition + `enumerate` ordering |
| 50-120 | `column_generation` (auto) + `greedy` or `matching` partition + `enumerate` ordering |
| 120+ | `column_generation` (beam) + `matching` + `greedy` ordering |

### By Priority

| Priority | Phase 1 | Phase 2 | Phase 3 |
|----------|---------|---------|---------|
| **Quality** | `explicit` (small instances) or `column_generation` with a higher time limit | `matching` + `enumerate` | `milp` |
| **Speed** | `heuristic` pipeline, or `column_generation` (beam) | `greedy` + `greedy` | `hungarian` |
| **Balanced** | `column_generation` (local_search) | `matching` + `enumerate` | `milp` |

---

## Dependencies by Algorithm

| Algorithm | Required Packages |
|-----------|-------------------|
| Phase 1: explicit | gurobipy |
| Phase 1: column_generation | gurobipy |
| Phase 1: greedy (heuristic pipeline) | numpy |
| Phase 2: matching | scipy, networkx |
| Phase 2: greedy/random | numpy |
| Phase 3: milp | gurobipy |
| Phase 3: hungarian | scipy |

**Minimum for heuristic pipeline**: pandas, numpy, scipy, networkx
**Full pipeline**: + gurobipy

---

## Schedule Evaluation

After generating a schedule, use the evaluator to compute quality metrics.

> **Pipeline-reported vs evaluator metrics:** Each pipeline reports quality metrics during optimization using its own internal accounting. These numbers may differ from the evaluator output, so **for consistent cross-pipeline comparison, always use the evaluator**. `run_schedule.py` and `compare_all_pipelines.py` print evaluator metrics at the end of every run; for schedule files use `scripts/evaluate_schedule.py` or `scripts/batch_evaluate.py`.

### Metrics

| Metric | Description | Lower is Better |
|--------|-------------|-----------------|
| **Missed Attendance** | For each participant: count talks missed due to parallel conflicts. If k+1 preferred talks are at the same timeslot, k are missed. | Yes |
| **Session Hops** | Room switches within blocks required to attend preferred talks. Uses DP to compute optimal path. | Yes |
| **Incoherent Sessions** | Room-sessions whose talks don't all share at least one keyword. Placeholders are ignored; talks without keywords count as incoherent. | Yes |
| **Presenter Violations** | Presenters assigned to their unavailable timeslots. | Yes |

### Command Line

```bash
# Reads preferences.csv, talks.csv and availability.csv from the data directory
python scripts/evaluate_schedule.py output/schedule.csv --data-dir examples/orbel2026 -v

# Or point to individual files
python scripts/evaluate_schedule.py output/schedule.csv \
    --preferences examples/orbel2026/preferences.csv \
    --talks examples/orbel2026/talks.csv

# Batch evaluation of multiple schedules
python scripts/batch_evaluate.py output/ \
    --preferences examples/orbel2026/preferences.csv
```

---
