# Conference Scheduling System

An optimization system for scheduling academic conference talks. Based on the approach described in [Vangerven et al. (2018)](https://doi.org/10.1016/j.ejor.2018.01.023) and some additional ideas.

This is not intended to be a polished production system without any flaws, and in fact, it was a bit of an experiment about the capabilities of different AI support tools, i.e., it has mostly been implemented by the use of OpenAI's GPT-5 models as well as Anthropic's Claude models within the Github CoPilot or Claude Code interface.
Moreover, MILP solving was also facilitated by the LLM support tool of Gurobi.

The tool provides feasible schedules optimizing different metrics and it worked reasonably well in practice when applying it to real conference scheduling (ORBEL 2026). Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

Nonetheless, manual review and post-processing are expected and encouraged. Conference organizers should review the output and adjust as needed based on specific requirements, practical constraints, and domain knowledge.

**Stay updated.** Pull the latest version regularly for improvements and bug fixes. We may also upload reports documenting experiences and insights from real-world applications.

**Update:** Added modified slides from conference talk about the scheduling process with some visualizations and takeaways.

## Installation

```bash
git clone https://github.com/felixrauh/conference_scheduler.git
cd conference_scheduler
pip install -r requirements.txt

# Optional: Gurobi solver (requires license)
pip install gurobipy
```

## Quick Start

```bash
# Run with example data (no Gurobi needed)
python scripts/run_schedule.py --pipeline heuristic --data-dir examples/orbel2026

# Try the small synthetic example
python scripts/run_schedule.py --pipeline heuristic --data-dir examples/synthetic

# With Gurobi: traditional and matching pipelines
python scripts/run_schedule.py --pipeline traditional --data-dir examples/orbel2026
python scripts/run_schedule.py --pipeline matching --data-dir examples/orbel2026

# Different dataset
python scripts/run_schedule.py --data-dir examples/orbel2017

# Use config file (CLI args override config)
python scripts/run_schedule.py --config config/scheduling_config.jsonc

# Full example
python scripts/run_schedule.py \
    --pipeline matching \
    --data-dir examples/orbel2026 \
    --output output/my_schedule.csv \
    --verbose
```

> **Default data directory**: If `--data-dir` is omitted, it defaults to `examples/orbel2026`.

### Command Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--data-dir` | `-d` | Path to input data directory (must contain talks.csv, preferences.csv, sessions.csv) |
| `--pipeline` | `-p` | `traditional`, `heuristic`, `matching`, or `matching_constrained` |
| `--config` | `-c` | Path to JSONC config file |
| `--output` | `-o` | Output CSV path |
| `--phase1-method` | | `explicit` or `column_generation` |
| `--phase2-partition` | | `greedy`, `matching`, or `random` |
| `--phase3-method` | | `milp` or `hungarian` |
| `--time-limit` | `-t` | Overall time limit (seconds) |
| `--verbose` | `-v` | Detailed output |

## Pipelines

| Pipeline | Approach | Gurobi? | Speed (118 talks) |
|----------|----------|---------|-----|
| **traditional** | Tuples (i.e. parallel talks) first (Phase 1→2→3→4) | Yes | ~4 min |
| **heuristic** | Greedy alternative | No | ~3 sec |
| **matching** | Sessions first (A→B→C→D→3→4) | Yes | ~1 min |
| **matching_constrained** | Like matching, but enforces keyword coherence within sessions (requires `keywords` column in talks.csv — see [Input Data](#input-data)) | Yes | ~1 min |

### Terminology

- **Block**: A set of parallel talks across all rooms during one timeslot (e.g., 5 rooms × 4 slots = 20 talks). Named TA, TB, FC, etc.
- **Session** (or **room-session**): The set of sequential talks in ONE room within a block (e.g., 4 talks in Room A during block TA).

### Traditional Pipeline
- **Phase 1**: Maximize co-attendance (MILP column generation)
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
- **Phase D**: Order talks to minimize room hopping (similar to Phase 2 in traditional pipeline)
- **Phase 3**: Assign blocks to timeslots
- **Swap Optimization**: Resolve presenter violations via local search
- **Phase 4**: Room assignment + session metrics

> **Hypothesis**: Matching pipeline (bottom-up grouping) may produce more coherent sessions since talks are grouped based on shared audience interest at an earlier stage.

### Algorithm Choices at a Glance

| Component | How to set | Options | Recommended |
|-----------|-----------|---------|-------------|
| **Pipeline** | `--pipeline` or `pipeline` in config | `traditional`, `heuristic`, `matching`, `matching_constrained` | `traditional` if Gurobi licensed |
| **Phase 1 method** | `--phase1-method` or `phase1.method` | `explicit`, `column_generation` | `column_generation` for 50+ talks |
| **Phase 1 pricing** | `phase1.column_generation.pricing_strategy` | `auto`, `enumeration`, `greedy`, `local_search`, `beam_search` | `auto` |
| **Phase 2 partition** | `--phase2-partition` or `phase2.partition_strategy` | `greedy`, `random`, `matching` | `matching` |
| **Phase 2 ordering** | `phase2.ordering_strategy` | `enumerate`, `greedy` | `enumerate` for k ≤ 5 |
| **Phase 3 method** | `--phase3-method` or `phase3.method` | `milp`, `hungarian` | `milp` if Gurobi available |
| **Swap optimization** | `swap_optimization.enabled` | `true`, `false` | `true` with `keyword_weight: 0.1` |

Phases 1–2 apply to the `traditional`/`heuristic` pipelines. The `matching` pipelines have their own internal phases (A–D) documented in the reference.

The "How to set" column shows CLI flags (e.g., `--pipeline`) and config file keys (e.g., `phase1.method`). These correspond to sections in the config file described below. For a full explanation of each option, see [docs/configuration_and_algorithm_reference.md](docs/configuration_and_algorithm_reference.md).

## Configuration

### Config File

All algorithm choices above (and additional settings like time limits) can be specified in `config/scheduling_config.jsonc`:

```bash
python scripts/run_schedule.py --config config/scheduling_config.jsonc
```

CLI arguments override config file settings. The config file is optional — sensible defaults are used for everything.

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

The scheduler needs three input files, placed in a directory:

**talks.csv** (required) — one row per talk:
```csv
talk_id,presenter_id,keywords
1,1,Optimization; Applications
2,2,Machine Learning
3,3,
```
The `keywords` column is optional but used by the `matching_constrained` pipeline to enforce thematic coherence within sessions, and by the evaluator to measure session coherence (see [Evaluation](#evaluation)).

**preferences.csv** (required) — one row per (participant, talk) pair:
```csv
participant_id,talk_id
1,4
1,6
1,14
2,3
```

**sessions.csv** (required) — conference structure, one row per session block:
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
Each row defines one session block: `n_rooms` parallel rooms, each with `n_talks_per_room` sequential talks. The total number of talk slots (sum of `n_rooms × n_talks_per_room` across all rows) should match the number of talks.

**availability.csv** (optional) — presenter unavailabilities:
```csv
presenter_id,unavailable_timeslot
1,TA
1,TB
2,FC
```
Timeslot IDs must match the `session_id` values from `sessions.csv`.

Participant IDs are inferred automatically from `preferences.csv` — no separate participants file is needed.

See `examples/orbel2026/` for a complete working example.

### Custom data formats

The canonical loader is `src/data_loader.load_from_csv_dir()`, which reads these CSVs and produces a `ConferenceData` object. If your raw data looks different, you can either:

1. Convert your data into the CSV format above, or
2. Write your own loader that produces a `ConferenceData` — see `src/data_loader.py` for the dataclass definition

### Key Features

- Three pipelines: Traditional (optimization), Heuristic (fast), Matching (bottom-up)
- Flexible configuration via CLI args, config file, or both
- Variable room counts per block (mix 3, 4, 5+ room sessions)

### Performance Benchmarks

Benchmarks run on Apple M2, 8 cores, 16GB RAM. All traditional variants use default settings unless noted.

#### Pipeline Comparison (Default Settings)

Traditional uses column generation + greedy partition + MILP scheduling. All quality metrics are computed by the `ScheduleEvaluator` for consistency across pipelines.

| Instance | Pipeline | Time | Missed Attendance | Session Hops |
|----------|----------|------|-------------------|--------------|
| 2017 ORBEL (80 talks) | Traditional | ~54s | 319 | 275 |
| 2017 ORBEL (80 talks) | Heuristic | ~3s | 288 | 278 |
| 2017 ORBEL (80 talks) | Matching | ~45s | 231 | 209 |
| 2026 ORBEL (118 talks) | Traditional | ~4min | 214 | 278 |
| 2026 ORBEL (118 talks) | Heuristic | ~3s | 172 | 283 |
| 2026 ORBEL (118 talks) | Matching | ~57s | 180 | 154 |

#### Traditional Pipeline: Algorithm Choice Impact (2026 ORBEL)

These results show how different algorithm combinations in the traditional pipeline affect quality and runtime on the 118-talk instance. Quality metrics are computed by the `ScheduleEvaluator`.

| Phase 1 | Phase 2 Partition | Phase 3 | Time | Missed | Session Hops |
|---------|-------------------|---------|------|--------|--------------|
| column_generation | greedy | milp | ~3.5min | 204 | 289 |
| column_generation | matching | milp | ~8min | 219 | 290 |
| column_generation | greedy | hungarian | ~19min | 228 | 275 |

#### Traditional Pipeline: Algorithm Choice Impact (2017 ORBEL)

| Phase 1 | Phase 2 Partition | Phase 3 | Time | Missed | Session Hops |
|---------|-------------------|---------|------|--------|--------------|
| column_generation | greedy | milp | ~52s | 326 | 270 |
| column_generation | matching | milp | ~56s | 311 | 267 |
| column_generation | greedy | hungarian | ~53s | 328 | **256** |

**Key observations:**
- **Column generation is a heuristic** — it generates columns (tuples) on demand rather than enumerating all possibilities, so Phase 1 output is not provably optimal. It also uses randomized pricing, so results vary between runs.
- **Explicit enumeration** is the provably optimal Phase 1 approach but impractical for 80+ talks (C(80,5) = 24M tuples). Use `column_generation` for datasets with 50+ talks.
- **Phase 2 `matching` partition** can improve session hops but the effect is dataset-dependent. It solves a min-cost matching problem to align rooms optimally.
- **Phase 3 `hungarian`** tends to produce slightly fewer session hops than MILP, at the cost of slightly more missed attendance. MILP also considers presenter availability.
- **Runtime** is dominated by Phase 1 (column generation) and Phase 2 (ordering), not Phase 3. CG runtime is variable due to randomized pricing — the ~19min outlier above hit a slow pricing sequence.

## Output

The scheduler produces two output files and a terminal summary:

### CSV Export (`output/schedule.csv`)

One row per talk, with columns for block, room, slot, and various metadata:

```csv
Block_ID,Room_ID,Room_Name,Slot,Talk_ID,Paper_ID,Title,...
TA,Room_1,Room_1,1,T001,...
TA,Room_1,Room_1,2,T002,...
...
```

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

After generating a schedule, evaluate its quality using the `ScheduleEvaluator`:

```bash
# Evaluate with all metrics
python scripts/evaluate_schedule.py output/schedule.csv \
    --preferences examples/orbel2026/preferences.csv \
    --talks examples/orbel2026/talks.csv \
    -v
```

### Quality Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Missed Attendance** | Preferred talks missed due to parallel conflicts | Minimize |
| **Session Hops** | Room switches within blocks to attend preferences | Minimize |
| **Incoherent Sessions** | Room-timeslots where talks don't share keywords | Minimize |
| **Presenter Violations** | Presenters in unavailable timeslots | Zero |

📖 **See [docs/configuration_and_algorithm_reference.md](docs/configuration_and_algorithm_reference.md#schedule-evaluation) for detailed evaluation documentation.**

## Algorithm Details

📖 **See [docs/configuration_and_algorithm_reference.md](docs/configuration_and_algorithm_reference.md) for detailed algorithm documentation**, including mathematical formulations, all configurable parameters, and performance guidelines by instance size.

### Slot Adjustment

The total number of talk slots (from `sessions.csv`) should ideally match the number of talks. If not, the scheduler adds placeholder ("dummy") slots to fill remaining spaces and prints a `Slot count mismatch` warning — this is expected when your sessions.csv has slightly more slots than talks. For best results, define your `sessions.csv` to match your talk count exactly.

### Pre-specified Sessions (Fixed Blocks)

If certain sessions have a pre-determined lineup (e.g., award or sponsored sessions), you can lock them before optimization runs. Two options:

- **`phase2.fixed_sequences`** — talks and order are fixed; the optimizer still picks the timeslot. Available for `traditional` and `heuristic` pipelines.
- **`phase4.fixed_block_sessions`** — talks *and* timeslot are both fixed. Available for all pipelines.

📖 **See [docs/configuration_and_algorithm_reference.md](docs/configuration_and_algorithm_reference.md#pre-specified-sessions-fixed-blocks) for configuration details and examples.**

## Project Structure

```
conference_scheduler/
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── config/                       # Configuration
│   └── scheduling_config.jsonc   # Main config file (optional)
│
├── src/                          # Core source code
│   ├── phase1.py                 # Phase 1: tuple selection (explicit or column generation)
│   ├── phase2.py                 # Phase 2: partition + ordering
│   ├── phase3.py                 # Phase 3: scheduling + room assignment
│   ├── phase4.py                 # Phase 4: room assignment, metrics, ScheduleResult
│   ├── matching_pipeline.py      # Bottom-up matching pipeline
│   ├── matching_pipeline_constrained.py  # Keyword-constrained variant
│   ├── swap_optimization.py      # Post-phase 3 violation resolution
│   ├── schedule_evaluator.py     # Quality metrics computation
│   ├── instance.py               # Data structures
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
│   └── load_data.py              # Data loader utility
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
    └── solver_guide.md                   # Gurobi setup & alternatives
```

## Known Limitations & Improvement Opportunities

- **Phase 1 (traditional pipeline):** The tuple selection phase is the computational bottleneck. Our column generation implementation is a straightforward adaptation and not heavily tailored to the problem structure. Better preprocessing (e.g., dominance-based filtering, tighter LP relaxations) and more problem-specific pricing strategies could significantly reduce solve times and improve solution quality for large instances.
For Phase 2, we did not do a very thorough analysis of all parts of our implementation, so there are likely optimization opprtunities and discrepancies in the way we implemented the different strategies from the paper.
- **Keyword coherence:** The `matching_constrained` pipeline enforces keyword coherence at solve time, but the other pipelines only address it post-hoc via swap optimization. Integrating keyword-awareness earlier in the traditional pipeline could improve thematic grouping.

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
