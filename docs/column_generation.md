# Column Generation for Phase 1

## Overview

Column generation is an alternative to explicit enumeration for solving Phase 1 (tuple selection). Instead of enumerating all possible n-tuples upfront, it generates only promising tuples on-demand by iteratively solving a restricted master problem and a pricing subproblem. This makes it the only feasible option for large instances where explicit enumeration is intractable.

However, column generation is not an ideal fit for this problem structure. The pricing subproblem is NP-hard with no exploitable structure (conflicts can occur between any pair of talks), so the implementation prices heuristically. And the LP relaxation of the master problem turns out to be weak on the example data (see [Why a greedy solution can beat column generation](#why-a-greedy-solution-can-beat-column-generation)), so solving it well does not translate into a good integer solution.

Column generation remains necessary for instances with 50+ talks simply because explicit enumeration runs out of memory. To guarantee a reasonable result, the traditional pipeline **seeds** column generation with the greedy solution of the heuristic pipeline: its tuples are added to the column pool and used as the starting solution of the final MIP, so the Phase 1 result is never worse than the greedy one.

In practice, the implementation uses heuristic pricing and stops at its iteration limit (50) within seconds, so we have no proven bounds and cannot assess how far the solution is from optimal. On the example data, column generation rarely improves on its greedy seed:

Phase 1 with default settings (Apple M2; times for the greedy seed, CG loop and final MIP):

| Instance | Talks | Explicit Enum | Columns | Time | Greedy seed | Phase 1 objective |
|----------|-------|---------------|---------|------|-------------|-------------------|
| orbel2017 | 80 | Feasible | ~90 | ~1.5s | 201 | 201 |
| orbel2026 | 118 (+5 placeholders) | Intractable (~225M tuples of size 5) | ~150 | ~4s | 189 | 189 |

(The greedy seed is deterministic. Before it was, its result depended on Python's set iteration order and varied between runs — on ORBEL 2026 from 143 to 196 — and column generation improved on its seed in only one of 13 such runs, by one missed attendance. The objective is the number of missed attendances; the evaluator's final number can differ because swap optimization moves talks afterwards.)

Letting CG run until no improving column is found (about 200 iterations, 5–50s) did not lower the objective noticeably in our tests, which is why the iteration limit stays at 50.

## Bounds and Optimality

Column generation provides an **upper bound** on the true optimal: any feasible integer solution found is at least as costly as the true optimum. However, obtaining a valid **lower bound** is generally not possible until convergence is achieved.

The restricted master LP (with only the generated columns) gives an LP objective $z_{LP}(\Omega_k)$, but this is **not** a valid lower bound before convergence. For minimization, fewer columns means fewer options, so $z_{LP}(\Omega_k) \geq z_{LP}(H)$ — the restricted LP can only be *worse* than the full LP. The dual solution from the restricted master violates constraints for columns not yet generated (that's exactly what a negative reduced cost means), so it is not dual-feasible for the full problem and cannot serve as a bound.

**Only after convergence** (no negative reduced cost columns found with exact pricing) does $z_{LP}(\Omega_k) = z_{LP}(H)$, giving a valid lower bound. At that point the integrality gap $z_{IP} - z_{LP}$ is meaningful.

The solver reports convergence status and the LP/IP values:

```
Convergence: YES (no improving columns found)
IP solution: 51
Restricted master LP: 42.30          ← valid lower bound only because CG converged
LP-IP gap: 20.57% (proven bound)     ← true integrality gap (exact pricing only)
  True optimal is between 42.3 and 51
```

(Shown with `--verbose`. With heuristic pricing or without convergence, the gap is labelled "NOT a proven bound".)

| Scenario | What you can claim |
|----------|-------------------|
| **CG converged with exact pricing** (`enumeration`) | LP is a proven lower bound. Gap is the true integrality gap. Optimal is between LP and IP. |
| **CG converged with heuristic pricing** (`local_search`, `beam_search`) | Pricing may have missed negative reduced cost columns, so "convergence" is not guaranteed. LP value is not a proven bound. |
| **CG stopped early** (time/iteration limit) | No bound on optimality. Can only say: "found a feasible solution with value X". |

Even a proven LP bound does not make the final schedule optimal: the final MIP only chooses among the generated columns (sometimes called *price-and-branch*), so the true integer optimum may need columns that were never generated. Proving integer optimality would require branch-and-price, i.e. generating columns inside the branch-and-bound tree.

In practice, for the instances in this project, the column generation implementation uses heuristic pricing and stops at its iteration limit. The reported solutions are feasible schedules, but we cannot make claims about how close they are to optimal.

## Why a greedy solution can beat column generation

Column generation "converging" means it found the best *fractional* combination of tuples among those it can price; the integer schedule is then assembled from the generated columns only. One run on ORBEL 2026 without the greedy seed, with CG run to convergence (234 iterations, `auto` = local-search pricing):

| | Missed attendances (Phase 1 objective) |
|---|---|
| Restricted master LP at convergence | 96.4 |
| Greedy solution (heuristic pipeline's Phase 1) | 196 |
| Final MIP over the 492 generated columns | 226 |
| Same MIP with the 26 greedy tuples added | 196 |

Two effects explain this:

1. **Integrality gap.** The fractional optimum (96) is far below any known schedule (196). The generated columns are good building blocks for the fractional solution, but few of them fit together into a complete integer partition; the greedy schedule simply was not among the integer solutions the final MIP could form. Once its tuples are in the pool, the MIP finds it.
2. **Heuristic pricing.** Under the final dual values, one of the greedy tuples still had a negative reduced cost (−0.61), so the "converged" LP was not truly optimal and 96 is not a proven lower bound. (By LP duality, the reduced costs of the greedy tuples sum to $G - z_{LP} = 196 - 96.4$; any tuple with negative reduced cost is one the pricing missed.) In this run the effect on the LP value was negligible.

The true optimum therefore lies somewhere between roughly 96 and the best schedule found. Seeding with the greedy solution makes column generation a safe refinement step; substantial improvements would need stronger pricing, branch-and-price, or a tighter formulation.

## Algorithm

```
1. INITIALIZATION
   ├─ Generate initial feasible columns using a greedy heuristic
   │  (for each tuple, the cheapest of up to 500 sampled candidates)
   └─ Add the seed columns (traditional pipeline: the tuples of the
      heuristic pipeline's greedy solution)

2. COLUMN GENERATION LOOP
   ├─ Solve restricted master LP (relaxation)
   ├─ Extract dual values (π*, λ*)
   ├─ Solve pricing problems (for each tuple size)
   │  └─ Find columns with negative reduced cost
   ├─ Add new columns to master
   └─ Repeat until no improving columns found, or the
      iteration / time limit is reached

3. FINAL MIP
   └─ Solve integer program over all generated columns,
      starting from the seed solution
```

## Mathematical Formulation

### Master Problem (LP Relaxation)

**Variables:** $x_e \in [0, 1]$ for each n-tuple $e$ in the current column pool $\Omega$

**Objective:** $\min \sum_{e \in \Omega} c_e \cdot x_e$

where $c_e = \sum_{p \in P} \max\{0, |e \cap Q_p| - 1\}$ (missed attendance cost)

**Constraints:**
1. **Coverage** (duals: $\pi_i$): $\sum_{e \in \Omega: i \in e} x_e = 1 \quad \forall i \in X$
2. **Tuple count** (duals: $\lambda_\tau$): $\sum_{e \in \Omega: |e|=n_\tau} x_e = p_\tau \quad \forall \tau$

### Pricing Problem

For each tuple size $n_\tau$, find the n-tuple with minimum reduced cost:

$$\min_{S \subseteq X: |S|=n_\tau} \left\{ c(S) - \sum_{i \in S} \pi_i^* \right\}$$

If the minimum reduced cost is negative ($\bar{c} < -\epsilon$), add the column to the master. Otherwise, the LP is optimal.

## Pricing Strategies

The pricing problem is NP-hard in general. Four strategies are available:

| Strategy | Method | Complexity | Exact? |
|----------|--------|-----------|--------|
| `enumeration` | Try all n-tuples | $O(\binom{|X|}{n} \cdot |P|)$ | Yes |
| `greedy` | Greedy construction | $O(n \cdot |X| \cdot |P|)$ | No |
| `local_search` | Greedy + swap improvement | $O(k \cdot n \cdot |X| \cdot |P|)$ | No |
| `beam_search` | Beam search with pruning | $O(n \cdot w \cdot |X| \cdot |P|)$ | No |

`pricing_strategy: "auto"` (default) chooses per tuple size $n$:

- `enumeration` if there are at most 10,000 candidate tuples, i.e. $\binom{|X|}{n} \le 10{,}000$ (about 22 talks for $n = 4$, 16 talks for $n = 5$);
- otherwise `local_search` for $n \le 5$;
- otherwise `beam_search` (6+ parallel rooms).

So on realistic instances `auto` uses heuristic pricing, and the LP value is not a proven bound (see above).

## Usage

Column generation is selected via the config or CLI:

```bash
# column_generation is the default Phase 1 method of the traditional pipeline
python scripts/run_schedule.py --pipeline traditional --phase1-method column_generation --time-limit 300
```

Or in config:
```jsonc
"phase1": {
    "method": "column_generation",
    "column_generation": {
        "pricing_strategy": "auto"
    }
}
```

### Parameters

| Parameter | Config key / CLI | Default | Description |
|-----------|------------------|---------|-------------|
| `pricing_strategy` | `phase1.column_generation.pricing_strategy` | `"auto"` | `auto`, `enumeration`, `greedy`, `local_search`, `beam_search` |
| `time_limit` | `phase1.time_limit` / `--time-limit` | 300.0 (bundled config file: 120.0) | Time limit for the CG loop in seconds; the final MIP runs afterwards |
| `max_iterations` | — | 50 | Maximum CG iterations (set in `src/pipelines/traditional.py`) |
| `optimality_gap` | — | 1e-4 | Columns are added only if their reduced cost is below `-optimality_gap` (set in `src/pipelines/traditional.py`) |

### Tuning Tips

- **Faster solutions:** lower `phase1.time_limit`, or use `pricing_strategy: "greedy"`
- **Better quality:** raise `phase1.time_limit`; use `enumeration` only for small instances
- **Very large instances:** `beam_search` and a higher `phase1.time_limit`

## Implementation

Two solver implementations are provided in `src/columngeneration_phase1/`:

- **`phase1_column_generation.py`** — Core solver with enumeration-based pricing
- **`phase1_column_generation_enhanced.py`** — Advanced solver with all four pricing strategies (recommended). Its `initial_columns` argument seeds the column pool and the final MIP with a known solution

The traditional pipeline calls the enhanced solver directly (`_run_phase1_column_generation` in `src/pipelines/traditional.py`). For programmatic use, `solve_phase1(instance, method="column_generation")` in `src/phase1.py` does the same.

## Limitations

The current column generation implementation is a straightforward adaptation and not heavily tailored to the conference scheduling problem structure. Better preprocessing (e.g., dominance-based filtering, tighter LP relaxations) and more problem-specific pricing strategies could improve performance. See the [Known Limitations](../README.md#known-limitations--improvement-opportunities) section in the README.
