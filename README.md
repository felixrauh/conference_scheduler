# Conference Scheduling System

An optimization system for scheduling academic conference talks. Based on the approach described in [Vangerven et al. (2018)](https://doi.org/10.1016/j.omega.2017.09.007) and some additional ideas.

This is not intended to be a polished production system without any flaws, and in fact, it was a bit of an experiment about the capabilities of different AI support tools, i.e., it has mostly been implemented by the use of OpenAI's GPT-5 models as well as Anthropic's Claude models within the Github CoPilot or Claude Code interface.
Moreover, MILP solving was also facilitated by the LLM support tool of Gurobi.

The tool provides feasible schedules optimizing different metrics and it worked reasonably well in practice when applying it to real conference scheduling (ORBEL 2026). Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

Nonetheless, manual review and post-processing are expected and encouraged. Conference organizers should review the output and adjust as needed based on specific requirements, practical constraints, and domain knowledge.

**Stay updated.** Pull the latest version regularly for improvements and bug fixes. We may also upload reports documenting experiences and insights from real-world applications.

**Update:** Added modified slides from conference talk about the scheduling process with some visualizations and takeaways.

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/felixrauh/conference_scheduler.git
cd conference_scheduler
pip install -r requirements.txt

# Optional: Gurobi solver (needed for all pipelines except "heuristic")
pip install gurobipy
```

> **Gurobi license.** `pip install gurobipy` ships with a size-limited license that is too small for real conferences ("Model too large for size-limited license"). Free academic licenses are available at [gurobi.com/academia](https://www.gurobi.com/academia/). See [docs/solver_guide.md](docs/solver_guide.md).

## Quick Start

```bash
# Run with example data (no Gurobi needed)
python scripts/run_schedule.py --pipeline heuristic --data-dir examples/orbel2026

# Try the small synthetic example
python scripts/run_schedule.py --pipeline heuristic --data-dir examples/synthetic

# With Gurobi: traditional (the default) and matching pipelines
python scripts/run_schedule.py --pipeline traditional --data-dir examples/orbel2026
python scripts/run_schedule.py --pipeline matching --data-dir examples/orbel2026

# Use a config file (CLI args override config)
python scripts/run_schedule.py --config config/scheduling_config.jsonc

# Evaluate the result, then compare all pipelines side by side
python scripts/evaluate_schedule.py output/schedule.csv --data-dir examples/orbel2026
python scripts/compare_all_pipelines.py --data-dir examples/orbel2026
```

The schedule is written to `output/schedule.csv` (plus a readable `output/schedule.md`), and a quality summary is printed at the end. If `--data-dir` is omitted, it defaults to `examples/orbel2026`; if `--pipeline` is omitted, it defaults to `traditional`.

### Command Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--data-dir` | `-d` | Input data directory (must contain talks.csv, preferences.csv, sessions.csv) |
| `--pipeline` | `-p` | `traditional` (default), `heuristic`, `matching`, or `matching_constrained` |
| `--config` | `-c` | Path to JSONC config file |
| `--output` | `-o` | Output CSV path (default `output/schedule.csv`; a `.md` file is written next to it) |
| `--phase1-method` | | `column_generation` (default) or `explicit` — traditional pipeline only |
| `--phase2-partition` | | `greedy` (default), `matching`, or `random` — traditional pipeline only |
| `--phase3-method` | | `milp` (default) or `hungarian` — the heuristic pipeline always uses `hungarian` |
| `--time-limit` | `-t` | Time limit (seconds) for Phase 1, or for the whole matching pipeline |
| `--require-keyword` | | `matching_constrained`: every two talks in a room-session must share a keyword (hard constraint) |
| `--max-violations` | | `matching_constrained`: soft keyword constraint — minimize keyword violations, starting from a budget of N that grows until feasible (`0` = as few as possible) |
| `--seed` | | Random seed (default `0`). The same input and seed give the same schedule; other seeds give alternatives (see [Reproducibility](docs/configuration_and_algorithm_reference.md#reproducibility)) |
| `--verbose` | `-v` | Detailed output |

## Pipelines

| Pipeline | Approach | Gurobi? | Time (118 talks) |
|----------|----------|---------|-----|
| **traditional** | Parallel talks first (Phase 1→2→3→4) | Yes | ~20 sec |
| **heuristic** | Greedy alternative | No | ~2 sec |
| **matching** | Sessions first (A→B→C→D→3→4) | Yes | ~1 min |
| **matching_constrained** | Like matching, but talks may only share a session if they belong to the same `special_group` and — with `--require-keyword` or `--max-violations` — share a keyword (see [Input Data](#input-data) and [the reference](docs/configuration_and_algorithm_reference.md#matching_constrained)) | Yes | ~1 min |

The matching pipelines only support sessions with 3 or 4 talks per room.

### Terminology

- **Block**: A set of parallel talks across all rooms during one timeslot (e.g., 5 rooms × 4 slots = 20 talks). Named TA, TB, FC, etc. (one per row of `sessions.csv`).
- **Session** (or **room-session**): The set of sequential talks in ONE room within a block (e.g., 4 talks in Room A during block TA).

### Traditional Pipeline
- **Phase 1**: Choose which talks run in parallel, minimizing missed attendance (column generation seeded with a greedy solution, MILP)
- **Phase 2**: Minimize room switching (matching + local search)
- **Phase 3**: Block scheduling (assign blocks to timeslots)
- **Swap Optimization**: Resolve presenter violations via local search
- **Phase 4**: Room assignment + session metrics

### Heuristic Pipeline
- **Phase 1**: Greedy tuple selection
- **Phase 2**: Matching-based partition
- **Phase 3**: Hungarian algorithm
- **Swap Optimization**: Resolve presenter violations via local search
- **Phase 4**: Room assignment + session metrics

### Matching Pipeline
- **Phase A**: Pair talks with shared audience interest
- **Phase B**: Form session blocks from pairs
- **Phase C**: Select which sessions run in parallel (similar to Phase 1 in traditional pipeline)
- **Phase D**: Order talks so that participants can attend more preferred talks by switching rooms (counterpart of Phase 2 in the traditional pipeline)
- **Phase 3**: Assign blocks to timeslots
- **Swap Optimization**: Resolve presenter violations via local search
- **Phase 4**: Room assignment + session metrics

> **Hypothesis**: Matching pipeline (bottom-up grouping) may produce more coherent sessions since talks are grouped based on shared audience interest at an earlier stage.

### Algorithm Choices at a Glance

| Component | How to set | Options | Default | Recommended |
|-----------|-----------|---------|---------|-------------|
| **Pipeline** | `--pipeline` or `pipeline` | `traditional`, `heuristic`, `matching`, `matching_constrained` | `traditional` | `traditional` or `matching` if Gurobi licensed |
| **Phase 1 method** | `--phase1-method` or `phase1.method` | `column_generation`, `explicit` | `column_generation` | `column_generation` for 50+ talks |
| **Phase 1 pricing** | `phase1.column_generation.pricing_strategy` | `auto`, `enumeration`, `greedy`, `local_search`, `beam_search` | `auto` | `auto` |
| **Phase 2 partition** | `--phase2-partition` or `phase2.partition_strategy` | `greedy`, `random`, `matching` | `greedy`* | dataset-dependent |
| **Phase 2 ordering** | `phase2.ordering_strategy` | `enumerate`, `greedy` | `enumerate` | `enumerate` for k ≤ 5 |
| **Phase 3 method** | `--phase3-method` or `phase3.method` | `milp`, `hungarian` | `milp` | `milp` if Gurobi available |
| **Swap optimization** | `swap_optimization.enabled` | `true`, `false` | `true` | `true` with `keyword_weight: 0.1` |

\* The bundled `config/scheduling_config.jsonc` sets `matching`, so runs with `--config` use that.

Phases 1–2 apply to the `traditional` pipeline; the `heuristic` pipeline uses its own fixed choices (greedy Phase 1, `matching` partition, Hungarian Phase 3). The `matching` pipelines have their own internal phases (A–D) documented in the reference.

The "How to set" column shows CLI flags (e.g., `--pipeline`) and config file keys (e.g., `phase1.method`). For a full explanation of each option, see [docs/configuration_and_algorithm_reference.md](docs/configuration_and_algorithm_reference.md).

## Configuration

### Config File

All algorithm choices above (and additional settings like time limits, rooms, and fixed sessions) can be specified in a JSONC file (JSON with comments). `config/scheduling_config.jsonc` documents every option:

```bash
python scripts/run_schedule.py --config config/scheduling_config.jsonc
```

CLI arguments override config file settings. The config file is optional — sensible defaults are used for everything. Invalid values (e.g. a misspelled method name) stop the run with an error message.

**Examples:**
```jsonc
// Switch to column generation for Phase 1:
"phase1": {
    "method": "column_generation",
    "time_limit": 300.0
}

// Use greedy ordering instead of enumeration:
"phase2": {
    "ordering_strategy": "greedy"  // faster for k > 5 timeslots
}
```

### Input Data

The scheduler needs three input files in one directory (plus an optional fourth). IDs in the CSV files are plain integers; internally talks become `T001`, `T002`, … and presenters/participants `P001`, … — use these internal IDs when referring to talks in the config file (e.g. for fixed sessions).

**talks.csv** (required) — one row per talk:
```csv
talk_id,presenter_id,keywords
1,1,Optimization; Applications
2,2,Machine Learning
3,3,
```
- `keywords` (optional, `;`-separated) are used by the evaluator to measure session coherence, by swap optimization, and by the `matching_constrained` pipeline.
- A presenter may give several talks (same `presenter_id`); see [Known Limitations](#known-limitations--improvement-opportunities) for how parallel talks of one presenter are handled.
- Further optional columns: `title`, `presenter_name` and `authors` (shown in the output schedule), and `special_group` (for `matching_constrained`: talks with the same group label are only placed in sessions with each other; empty means `general`).

**preferences.csv** (required) — one row per (participant, talk) pair:
```csv
participant_id,talk_id
1,4
1,6
1,14
2,3
```
Participant IDs are inferred automatically — no separate participants file is needed.

**sessions.csv** (required) — conference structure, one row per session block, in chronological order:
```csv
session_id,n_rooms,n_talks_per_room
TA,5,4
TB,5,4
TC,5,3
TD,4,3
FA,5,4
FB,5,4
FC,4,4
```
Each row defines one session block: `n_rooms` parallel rooms, each with `n_talks_per_room` sequential talks. The total number of talk slots (sum of `n_rooms × n_talks_per_room`) must be at least the number of talks (see [Slot Adjustment](#slot-adjustment)).

**availability.csv** (optional) — presenter unavailabilities:
```csv
presenter_id,unavailable_timeslot
1,TA
1,TB
2,FC
```
Timeslot IDs must match the `session_id` values from `sessions.csv`. Presenter IDs can be given as integers (`1`) or in internal form (`P001`).

See `examples/orbel2026/` for a complete working example, and check your own data with `python examples/load_data.py path/to/your_data`.

### Custom data formats

The canonical loader is `src/data_loader.load_from_csv_dir()`, which reads these CSVs and produces a `ConferenceData` object. If your raw data looks different, you can either:

1. Convert your data into the CSV format above, or
2. Write your own loader that produces a `ConferenceData` — see `src/data_loader.py` for the dataclass definition

### Key Features

- Four pipelines: Traditional (optimization), Heuristic (fast, no solver), Matching and Matching-constrained (bottom-up)
- Flexible configuration via CLI args, config file, or both
- Variable room counts per block (mix 3, 4, 5+ room sessions)
- Presenter availability constraints and pre-specified (fixed) sessions

### Performance Benchmarks

Benchmarks run on Apple M2, 8 cores, 16GB RAM, with default settings (`python scripts/compare_all_pipelines.py --data-dir examples/<dataset>`). All quality metrics are computed by the `ScheduleEvaluator`; lower is better.

#### Pipeline Comparison (Default Settings)

| Instance | Pipeline | Time | Missed Attendance | Session Hops | Incoherent Sessions |
|----------|----------|------|-------------------|--------------|---------------------|
| 2017 ORBEL (80 talks) | Heuristic | ~2s | 201 | 264 | 18 |
| 2017 ORBEL (80 talks) | Traditional | ~20s | 201 | 252 | 16 |
| 2017 ORBEL (80 talks) | Matching | ~50s | 164 | 193 | 18 |
| 2017 ORBEL (80 talks) | Matching (constrained) | ~50s | 164 | 193 | 18 |
| 2026 ORBEL (118 talks) | Heuristic | ~2s | 166 | 347 | 4 |
| 2026 ORBEL (118 talks) | Traditional | ~20s | 180 | 271 | 6 |
| 2026 ORBEL (118 talks) | Matching | ~65s | 156 | 179 | 7 |
| 2026 ORBEL (118 talks) | Matching (constrained) | ~65s | 159 | 197 | 7 |

The ORBEL example data contain no presenter availability constraints, so presenter violations are 0 throughout. The quality numbers are reproducible with the default seed (`--seed 0`); runtimes depend on the machine. Other seeds give somewhat different schedules, e.g. `--seed 1` on the 2026 data: heuristic 165 missed attendance / 341 session hops, traditional 179 / 271. In the 2017 data, 46 of 80 talks have no keywords, so most sessions count as incoherent. `matching_constrained` ran without keyword constraint here; on the 2026 data, `--require-keyword` reduced incoherent sessions to 1 (151 missed attendance, 206 session hops).

#### Traditional Pipeline: Algorithm Choices

Traditional pipeline (column generation Phase 1) with different Phase 2 and Phase 3 settings, default seed:

| Instance | Phase 2 Partition | Phase 3 | Time | Missed | Session Hops |
|----------|-------------------|---------|------|--------|--------------|
| 2017 ORBEL | greedy | milp | ~23s | 201 | 252 |
| 2017 ORBEL | matching | milp | ~23s | 201 | 246 |
| 2017 ORBEL | greedy | hungarian | ~23s | 201 | 252 |
| 2026 ORBEL | greedy | milp | ~19s | 180 | 271 |
| 2026 ORBEL | matching | milp | ~19s | 169 | 277 |
| 2026 ORBEL | greedy | hungarian | ~19s | 180 | 271 |

**Key observations:**
- **What each phase controls:** missed attendance is fixed in Phase 1 (which talks run in parallel), session hops in Phase 2 (which tuples form a block, and their order). Phase 3 only decides which block goes to which timeslot, so `milp` and `hungarian` give the same schedule quality (identical rows above) and differ only in the solver used. The `matching` rows differ because the partition changes the blocks (session hops) and, through the block layout, which talks swap optimization moves (missed attendance). With other seeds such comparisons can come out differently, so compare settings over several seeds.
- **Column generation is a heuristic** — it generates columns (tuples) on demand rather than enumerating all possibilities, so Phase 1 output is not provably optimal. It is seeded with the greedy solution and in these runs never improved on it, so the Phase 1 result is that of the greedy seed. See [docs/column_generation.md](docs/column_generation.md#why-a-greedy-solution-can-beat-column-generation).
- **Explicit enumeration** is the provably optimal Phase 1 approach but impractical for larger instances (C(80,5) = 24M possible 5-tuples for 80 talks). Use `column_generation` for datasets with 50+ talks.
- **Phase 2 `matching` partition** can reduce session hops but the effect is dataset-dependent. It solves a min-cost matching problem to align rooms optimally.
- **Runtime** (~20s) is dominated by Phase 2 (ordering and local search); Phase 1 takes a few seconds.

## Output

The scheduler produces two output files and a terminal summary. Blocks appear in the order of `sessions.csv`.

### CSV Export (`output/schedule.csv`)

One row per talk:

```csv
Block_ID,Room_ID,Room_Name,Slot,Talk_ID,Paper_ID,Title,Primary_Contact_Author,Author_Names,Session_Total_Likes,Session_Unique_Attendees,Is_Fixed
TA,Room_1,Room_1,1,T012,12,T012,P012,,31,22,False
TA,Room_1,Room_1,2,T047,47,T047,P047,,31,22,False
...
```

Empty slots (when there are more slots than talks) appear as `PLACEHOLDER_<n>` talks. Room names and capacities come from the `rooms` section of the config file; without it, rooms are numbered `Room_1`, `Room_2`, ….

### Markdown Export (`output/schedule.md`)

A human-readable schedule with tables per block and room metrics, generated automatically alongside the CSV.

### Personal Itineraries

Generate a personal itinerary for a specific participant, showing their preferred talks, parallel conflicts, and room switches:

```bash
python scripts/generate_itinerary.py output/schedule.csv \
    --participant 42 \
    --data-dir examples/orbel2026
```

## Schedule Evaluation

`run_schedule.py` evaluates every schedule it produces. To evaluate a schedule file (for instance one you edited by hand):

```bash
# Uses preferences.csv, talks.csv and availability.csv from the data directory
python scripts/evaluate_schedule.py output/schedule.csv --data-dir examples/orbel2026 -v

# Evaluate several schedules at once (missed attendance and session hops)
python scripts/batch_evaluate.py output/comparison/ \
    --preferences examples/orbel2026/preferences.csv
```

### Quality Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Missed Attendance** | Preferred talks missed due to parallel conflicts | Minimize |
| **Session Hops** | Room switches within blocks to attend preferences | Minimize |
| **Incoherent Sessions** | Room-sessions whose talks don't share a common keyword (placeholders ignored) | Minimize |
| **Presenter Violations** | Presenters in unavailable timeslots | Zero |

📖 **See [docs/configuration_and_algorithm_reference.md](docs/configuration_and_algorithm_reference.md#schedule-evaluation) for detailed evaluation documentation.**

## Algorithm Details

📖 **See [docs/configuration_and_algorithm_reference.md](docs/configuration_and_algorithm_reference.md) for detailed algorithm documentation**, including mathematical formulations, all configurable parameters, and performance guidelines by instance size.

### Slot Adjustment

The total number of talk slots (from `sessions.csv`) must be at least the number of talks. If there are more slots than talks, the remaining slots are filled with placeholder talks (`PLACEHOLDER_<n>`, i.e. empty slots). Swap optimization then moves placeholders out of short blocks (≤ `short_block_threshold` talks per room) and avoids two empty slots in the same room-session. If there are more talks than slots, the run stops with an error.

### Pre-specified Sessions (Fixed Blocks)

If certain sessions have a pre-determined lineup (e.g., award or sponsored sessions), you can lock them before optimization runs. Two options:

- **`phase4.fixed_block_sessions`** — talks, order *and* timeslot are fixed. Keep `sessions.csv` unchanged: one room of the target block is reserved automatically. Available for all pipelines.
- **`phase2.fixed_sequences`** — talks and order are fixed; the optimizer still picks the timeslot. Available for the `traditional` and `heuristic` pipelines.

📖 **See [docs/configuration_and_algorithm_reference.md](docs/configuration_and_algorithm_reference.md#pre-specified-sessions-fixed-blocks) for configuration details and examples.**

## Project Structure

```
conference_scheduler/
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── config/                       # Configuration
│   └── scheduling_config.jsonc   # Main config file (optional, documents all options)
│
├── src/                          # Core source code
│   ├── phase1.py                 # Phase 1: tuple selection (explicit enumeration)
│   ├── phase2.py                 # Phase 2: partition + ordering
│   ├── phase3.py                 # Phase 3: block scheduling (MILP or Hungarian)
│   ├── phase4.py                 # Phase 4: room assignment, metrics, ScheduleResult
│   ├── matching_pipeline.py      # Bottom-up matching pipeline
│   ├── matching_pipeline_constrained.py  # Group/keyword-constrained variant
│   ├── swap_optimization.py      # Post-phase 3 violation resolution
│   ├── schedule_evaluator.py     # Quality metrics computation
│   ├── instance.py               # Problem instance + presenter-availability helpers
│   ├── data_loader.py            # CSV data loading
│   ├── utils.py                  # Shared utilities
│   ├── pipelines/                # Pipeline implementations
│   │   ├── traditional.py        # Top-down (Gurobi) pipeline
│   │   └── heuristic.py          # Greedy (no solver) pipeline
│   └── columngeneration_phase1/  # Column generation for Phase 1 (see docs/column_generation.md)
│
├── scripts/                      # Entry points
│   ├── run_schedule.py           # MAIN ENTRY POINT (all pipelines)
│   ├── evaluate_schedule.py      # Schedule quality evaluation
│   ├── generate_itinerary.py     # Personal itinerary generation
│   ├── batch_evaluate.py         # Batch evaluation utility
│   └── compare_all_pipelines.py  # Run and compare all pipelines
│
├── examples/                     # Example datasets
│   ├── orbel2017/                # Anonymized 2017 data (80 talks)
│   ├── orbel2026/                # Anonymized 2026 data (118 talks)
│   ├── synthetic/                # Small test dataset (16 talks)
│   └── load_data.py              # Load and validate a data directory
│
├── tests/                        # Unit and integration tests
│
└── docs/                         # Technical documentation
    ├── configuration_and_algorithm_reference.md  # Config options & algorithm overview
    ├── column_generation.md              # Column generation for Phase 1
    ├── phase1_specification.md           # Phase 1 mathematical formulations
    ├── phase2_specification.md           # Phase 2 mathematical formulations
    ├── phase3_room_assignment.md         # Phase 3 assignment formulations
    ├── matching_pipeline_specification.md # Matching pipeline MILP formulations
    ├── developer_notes.md                # Implementation details for contributors
    └── solver_guide.md                   # Gurobi setup & alternatives
```

## Running Tests

```bash
pip install pytest
python -m pytest
```

Tests that need Gurobi are skipped automatically when it is not installed.

## Known Limitations & Improvement Opportunities

- **Phase 1 (traditional pipeline):** Our column generation implementation is a straightforward adaptation and not heavily tailored to the problem structure. It is seeded with the heuristic pipeline's greedy solution, so its Phase 1 result is never worse than that, but on the example data it rarely improves on the seed: the LP relaxation is weak and the final MIP only combines generated columns (see [docs/column_generation.md](docs/column_generation.md#why-a-greedy-solution-can-beat-column-generation)). Stronger pricing, branch-and-price, better preprocessing (e.g., dominance-based filtering) or a tighter formulation could improve solution quality. Column generation also does not yet use presenter availability or same-presenter checks when building tuples (later phases handle availability).
- **Phase 2 (traditional pipeline):** We did not do a very thorough analysis of all parts of our implementation, so there are likely optimization opportunities and discrepancies in the way we implemented the different strategies from the paper. Presenter availability is handled by a feasibility check after partitioning with a local repair step (swapping tuples between blocks), not inside the partitioning itself.
- **Keyword coherence:** The `matching_constrained` pipeline can enforce keyword coherence at solve time, but the other pipelines only address it post-hoc via swap optimization. Integrating keyword-awareness earlier in the traditional pipeline could improve thematic grouping.
- **Matching pipelines** support only sessions with 3 or 4 talks per room.
- **Presenters with several talks:** the `heuristic` pipeline and `explicit` Phase 1 never put two talks of the same presenter in parallel; column generation and the matching pipelines do not check this yet, so review such presenters in the output.

## Solver Options

The `traditional` and `matching` pipelines require [Gurobi](https://www.gurobi.com/), a commercial solver (free for academics). The `heuristic` pipeline works without any solver.

See [docs/solver_guide.md](docs/solver_guide.md) for license information and alternatives.

## Requirements

- Python 3.10+
- pandas, numpy, scipy, networkx, openpyxl
- Gurobi (optional, for `traditional` and `matching` pipelines)

<!-- ## Citing This Work

If you use this software in your research, please cite: -->


## License

This project is MIT licensed — you're free to use, copy, modify, and distribute it for any purpose, including commercially. The only requirement is to include the license text. See [LICENSE](LICENSE) for details.
