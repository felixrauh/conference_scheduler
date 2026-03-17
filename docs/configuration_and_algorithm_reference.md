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

- **Block**: A set of parallel talks across all rooms during one timeslot (e.g., 5 rooms × 4 slots = 20 talks). Named TA, TB, TC, TD, FA, FB, FC (Thursday-A, Friday-B, etc.).
- **Session** (or **room-session**): The set of sequential talks in ONE room within a block (e.g., 4 talks in Room A during block TA).
- **Tuple**: An n-tuple of talks scheduled in parallel at the same timeslot (one per room).

## Overview: Pipeline Types

The scheduler offers two pipeline families that differ in what they decide first:

- **Parallel-talks-first** (`traditional`, `heuristic`): Starts by choosing which talks should run at the same time (minimizing preference conflicts), then groups those parallel sets into session blocks, orders them, and assigns rooms. The `traditional` variant uses Gurobi for optimal tuple selection; the `heuristic` variant uses a greedy approach and requires no solver.

- **Sessions-first** (`matching`, `matching_constrained`): Starts by pairing talks that audiences want to see together, then assembles those pairs into complete session blocks, then decides which blocks run in parallel, and finally orders talks within blocks to allow beneficial room-hopping. The `matching_constrained` variant additionally enforces keyword coherence within sessions. Both require Gurobi.

**Config key:** `pipeline` | **CLI:** `--pipeline`

| Pipeline | Approach | Solver Required | Speed (118 talks) |
|----------|----------|-----------------|-----|
| **traditional** | Parallel-talks-first | Gurobi | ~4 min |
| **heuristic** | Parallel-talks-first | None (scipy only) | ~3 sec |
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

# Parallel-Talks-First Pipeline (traditional / heuristic)

## Phase 1: Tuple Selection

**Goal**: Select n-tuples of talks to run in parallel, minimizing missed attendance.

> For mathematical formulations and data structures, see [phase1_specification.md](phase1_specification.md).

**Config key:** `phase1.method` | **CLI:** `--phase1-method`

| Method | Description | Solver | Scalability | Solution Quality |
|--------|-------------|--------|-------------|------------------|
| `explicit` | Generate (almost) all tuples, filter, solve MIP | Gurobi | 30-100 talks | (Almost) Optimal (with filtering) |
| `column_generation` | Generate tuples on-demand via pricing | Gurobi | 50-150+ talks | Feasible (see [column_generation.md](column_generation.md)) |
| `greedy` | Greedy construction based on pair costs | None | Any size | Good heuristic |

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
| `phase1.time_limit` | float | 120.0 | MIP solver time limit (seconds) |
| `phase1.explicit.max_cost` | int | null | Cost filter threshold for tuple filtering. `null` = auto-calculate. Lower values = more aggressive filtering for large instances |

The heuristic cost filter (`use_heuristic_filter` in `src/phase1.py`) activates automatically for large instances and is not exposed via config.

**When to use**: fewer than 100 talks, need guaranteed optimal solution, have sufficient memory.

### Column Generation Options

Config section: `phase1.column_generation`. See [column_generation.md](column_generation.md) for full documentation.

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `phase1.time_limit` | float | 300.0 | Total time limit (seconds) |
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

| Strategy | Description | Best For |
|----------|-------------|----------|
| `auto` | Automatic selection based on problem size | General use (recommended) |
| `enumeration` | Exact enumeration | < 60 talks (slow but optimal) |
| `greedy` | Greedy construction | Quick results |
| `local_search` | Local search from greedy start | 60–120 talks |
| `beam_search` | Beam search with pruning | 120+ talks |

### Greedy Heuristic

Used automatically by the `heuristic` pipeline. No user-configurable options beyond `phase1.time_limit`.

---

## Phase 2: Block Assembly & Ordering

**Goal**: Partition tuples into blocks and order them to minimize room hopping.

> For mathematical formulations (hopping number, DP, partitioning approaches), see [phase2_specification.md](phase2_specification.md).

### Partition Strategy

**Config key:** `phase2.partition_strategy` | **CLI:** `--phase2-partition`

| Strategy | Description | Quality | Dependencies |
|----------|-------------|---------|--------------|
| `greedy` | Conflict-based greedy assignment | Good | None |
| `random` | Random assignment | Baseline | None |
| `matching` | Assignment problem + perfect matching | Best | scipy, networkx |

The `matching` strategy (recommended) follows the paper's approach: builds a complete graph over tuples, solves minimum-cost perfect matching, and merges matched pairs into k-blocks.

### Ordering Strategy

**Config key:** `phase2.ordering_strategy` (config file only, no CLI flag)

| Strategy | Description | Quality |
|----------|-------------|---------|
| `enumerate` | Try all k! permutations | Optimal (use for k ≤ 5) |
| `greedy` | Incremental greedy ordering | Good (use for k > 5) |

### All Phase 2 Options

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `phase2.partition_strategy` | string | `"matching"` | Partition method |
| `phase2.ordering_strategy` | string | `"enumerate"` | Ordering method |
| `phase2.local_search_iterations` | int | 2000 | Post-optimization swap iterations |
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
| `phase3.method` | string | `"milp"` | `"milp"` (Gurobi) or `"hungarian"` (scipy) |
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

After Phase 3, some violations may remain (presenters scheduled when unavailable, dummy talks in undesirable positions). Swap optimization iteratively swaps talks between positions to fix these.

**Config section:** `swap_optimization`

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `swap_optimization.enabled` | bool | `true` | Enable/disable swap optimization |
| `swap_optimization.keyword_weight` | float | 0.1 | Weight for keyword coherence vs missed attendance (0.0–1.0). 0.0 = only minimize missed attendance |
| `swap_optimization.max_iterations` | int | 100 | Maximum number of swaps to attempt |
| `swap_optimization.check_dummy_violations` | bool | `true` | Also resolve dummy talk placement issues |
| `swap_optimization.short_block_threshold` | int | 3 | Blocks with k ≤ this are "short" — dummies placed there count as violations |

```jsonc
// Config example: disable keyword coherence, more iterations
"swap_optimization": {
    "enabled": true,
    "keyword_weight": 0.0,
    "max_iterations": 200
}
```

Requires `availability.csv` in the data directory to have any effect.

### How It Works

1. Detects presenter violations (unavailable timeslots) and dummy violations (dummies in short blocks or clustered in one session)
2. For each violation, finds feasible swap partners
3. Scores each swap: `combined_score = missed_attendance_delta - keyword_weight × keyword_delta`
4. Greedily applies the best-scoring feasible swap
5. Repeats until no violations remain or no feasible swaps exist

---

## Phase 4: Finalization & Room Assignment (shared across all pipelines)

**Goal**: Assign physical rooms and insert any pre-specified sessions.

**Config section:** `phase4`, `rooms`

Room assignment is automatic — sessions with the largest expected audience get the largest rooms:
1. Sort sessions by unique attendees (ascending)
2. Sort physical rooms by capacity (ascending)
3. Match: smallest audience → smallest room

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

`sessions.csv` must define one 4R4T timeslot less than would otherwise be needed (since this sequence consumes one room of a 4R4T slot). The optimizer promotes that slot to 5R4T after attachment.

### Option B — `phase4.fixed_block_sessions`: both grouping and timeslot are fixed

Use this when you know not only the talks but also exactly which timeslot the session must go to. The talks are excluded from Phases 1–3 entirely; Phase 4 inserts them into the named block and assigns a room.

`sessions.csv` must reduce the room count for the target timeslot by one (the special session occupies that room).

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
| Phase applied | Phase 2 (pre-ordering) | Phase 4 (post-assignment) |

**When to use**:
- Award sessions with pre-determined speakers
- Sponsored sessions with a fixed lineup
- Invited or plenary sessions that must be in a specific slot

---

# Matching Pipeline (matching / matching_constrained)

The matching pipeline takes a sessions-first approach: it builds schedules by first grouping talks into coherent sessions, then deciding which sessions run in parallel. It has 4 pipeline-specific phases (A–D), followed by the shared Phase 3, swap optimization, and Phase 4.

**Config section:** `matching_pipeline`

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `matching_pipeline.time_limit` | float | 300.0 | Total time limit across all phases (distributed automatically: 15% pairing, 15% block formation, 40% tuple selection, 30% ordering) |

```jsonc
// Config example
"pipeline": "matching",
"matching_pipeline": {
    "time_limit": 300.0
}
```

The individual phase methods (A–D) are not currently configurable via the config file — the pipeline uses optimal methods (MILP) by default. See [matching_pipeline_specification.md](matching_pipeline_specification.md) for the MILP formulations.

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

Uses exhaustive enumeration of all possible orderings within each block. This is feasible because k (talks per room) is typically 3–4, keeping the search space small.

> **Metric note:** Phase D optimizes _hopping benefit_ — the number of extra talks participants can attend by switching rooms within a block, compared to staying in a single room. This is **not** the same as the room-switching _cost_ minimized by the traditional pipeline's Phase 2. A higher hopping benefit means the schedule rewards room switching more. In benchmark output, this appears as a negative "Room switches" value (e.g., −126), which should be read as "126 extra attendances enabled by room hopping."

## Presenter Unavailability in Matching Pipeline

The matching pipeline enforces presenter availability at multiple levels:

| Phase | Mechanism |
|-------|-----------|
| **Phase A** | Infeasible talk pairs excluded from matching |
| **Phase B** | Blocks with infeasible presenter combinations excluded |
| **Phase C** | If Phase 3 finds violations, previous solution is forbidden and Phase C retries (up to `max_feasibility_retries` times) |

---

## Presenter Availability Constraints

Presenter availability (from `availability.csv`) is enforced through multiple mechanisms across all pipelines:

1. **Infeasible talk pairs** (Phase 1 / Phase A): Talks whose presenters' combined unavailabilities leave too few valid timeslots are never placed together
2. **Tuple size restrictions** (Phase 1): If a presenter is unavailable for all timeslots of a block type, their talk is forbidden from that tuple size
3. **Block feasibility checking** (Phase 2): After grouping, each block is verified to have at least one feasible timeslot
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
  │         │                              │
  │        Yes                             ↓
  │         │                    phase1.method = "greedy"
  │         ↓                    phase2.partition_strategy = "matching"
  │    pipeline = "traditional"  phase3.method = "hungarian"
  │         │
  │         ↓
  │    Talks > 100? ─── Yes ──→ phase1.method = "column_generation"
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
| 50-80 | `explicit` + `matching` partition + `enumerate` ordering |
| 80-120 | `column_generation` (auto) + `matching` + `enumerate` |
| 120+ | `column_generation` (beam) + `matching` + `greedy` ordering |

### By Priority

| Priority | Phase 1 | Phase 2 | Phase 3 |
|----------|---------|---------|---------|
| **Quality** | `explicit` or `column_generation` (enumeration) | `matching` + `enumerate` | `milp` |
| **Speed** | `greedy` or `column_generation` (beam) | `greedy` + `greedy` | `hungarian` |
| **Balanced** | `column_generation` (local_search) | `matching` + `enumerate` | `milp` |

---

## Dependencies by Algorithm

| Algorithm | Required Packages |
|-----------|-------------------|
| Phase 1: explicit | gurobipy |
| Phase 1: column_generation | gurobipy |
| Phase 1: greedy | numpy |
| Phase 2: matching | scipy, networkx |
| Phase 2: greedy/random | numpy |
| Phase 3: milp | gurobipy |
| Phase 3: hungarian | scipy |

**Minimum for heuristic pipeline**: numpy, scipy, networkx
**Full pipeline**: + gurobipy

---

## Schedule Evaluation

After generating a schedule, use the evaluator to compute quality metrics.

> **Pipeline-reported vs evaluator metrics:** Each pipeline reports quality metrics during optimization using its own internal accounting. These numbers may differ from the evaluator output, so **for consistent cross-pipeline comparison, always use the evaluator** (via `scripts/evaluate_schedule.py` or `scripts/batch_evaluate.py`).

### Metrics

| Metric | Description | Lower is Better |
|--------|-------------|-----------------|
| **Missed Attendance** | For each participant: count talks missed due to parallel conflicts. If k+1 preferred talks are at the same timeslot, k are missed. | Yes |
| **Session Hops** | Room switches within blocks required to attend preferred talks. Uses DP to compute optimal path. | Yes |
| **Incoherent Sessions** | Room-timeslot pairs where talks don't share at least one keyword. | Yes |
| **Presenter Violations** | Presenters assigned to their unavailable timeslots. | Yes |

### Command Line

```bash
python scripts/evaluate_schedule.py output/schedule.csv \
    --preferences examples/orbel2026/preferences.csv \
    --talks examples/orbel2026/talks.csv \
    -v

# Batch evaluation of multiple schedules
python scripts/batch_evaluate.py output/ \
    --preferences examples/orbel2026/preferences.csv
```

---
