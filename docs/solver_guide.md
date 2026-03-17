# Solver Guide

## About the Optimization Solver

This project uses [Gurobi](https://www.gurobi.com/) to solve the mixed-integer linear programming (MILP) models at the core of the scheduling algorithm. **You should verify that your Gurobi license covers your intended use case** - for example, academic licenses are free but cannot be used for commercial conference scheduling.

Of course, other ways of solving these MILP models exist as well. The corresponding model implementations can be found in the files listed below, and you can re-implement them with a solver of your choice (best case with pull request here afterwards), or use the heuristic approach which requires no commercial solver.


## MILP Model Locations

The following files contain Gurobi MILP models that you may want to re-implement with an alternative solver:

| File | Model Name | Problem Type |
|------|------------|--------------|
| [src/phase1.py](../src/phase1.py) | `Phase1_ConferenceScheduling` | Set partitioning - selects n-tuples to minimize missed attendance |
| [src/phase2.py](../src/phase2.py) | `min_cost_perfect_matching` | Min-cost perfect matching - pairs tuples into blocks |
| [src/phase3.py](../src/phase3.py) | `block_scheduling` | Bipartite assignment - assigns blocks to timeslots |
| [src/matching_pipeline.py](../src/matching_pipeline.py) | `PhaseA_PairMatching`, `PhaseB_BlockFormation`, `PhaseC_*` | Matching pipeline models |
| [src/matching_pipeline_constrained.py](../src/matching_pipeline_constrained.py) | `PhaseA_Constrained`, `PhaseA_SoftConstraints`, `PhaseB_Constrained`, `PhaseB_GlobalCoherence` | Constrained variants with soft/hard constraints |
| [src/columngeneration_phase1/](../src/columngeneration_phase1/) | Column generation | Alternative Phase 1 for large instances (50+ talks). See [column_generation.md](column_generation.md) |

## Alternative: Heuristic Pipeline (No Solver Required)

If you don't have Gurobi or prefer not to use a commercial solver:

```bash
python scripts/run_schedule.py --pipeline heuristic
```

This uses scipy and greedy algorithms. It's faster but may produce suboptimal solutions.
