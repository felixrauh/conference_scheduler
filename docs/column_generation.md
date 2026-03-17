# Column Generation for Phase 1

## Overview

Column generation is an alternative to explicit enumeration for solving Phase 1 (tuple selection). Instead of enumerating all possible n-tuples upfront, it generates only promising tuples on-demand by iteratively solving a restricted master problem and a pricing subproblem. This makes it the only feasible option for large instances where explicit enumeration is intractable.

However, column generation is not an ideal fit for this problem structure. The master problem (set partitioning) is Gurobi's sweet spot — tight LP relaxation, excellent presolve — while the pricing subproblem is NP-hard with no exploitable structure (conflicts can occur between any pair of talks). This means CG shifts complexity away from where the solver excels and into repeated heuristic pricing, often doing more total work without guaranteeing better solutions than explicit enumeration would give on smaller instances.

Despite this, column generation remains necessary for instances with 50+ talks simply because explicit enumeration runs out of memory. The table below shows that it produces workable solutions, but there is significant room for improvement through better pricing strategies or alternative decomposition approaches.

In practice, the implementation uses heuristic pricing and typically stops at a time limit, so we have no proven dual bounds and cannot assess solution quality. This means that despite Phase 1 directly optimizing for missed attendance, other pipelines (heuristic, matching) may actually achieve comparable or better missed-attendance scores through their own structural choices — the CG approach is not guaranteed to outperform them on this metric.


| Instance | Talks | Explicit Enum | Column Gen | Variables | Time |
|----------|-------|---------------|------------|-----------|------|
| orbel2017 | 80 | Feasible | Feasible | 47 (0.9%) | 17s |
| orbel2026 | 118 | Intractable (182M vars) | Feasible | 125 (0.0001%) | 105s |

## Bounds and Optimality

Column generation provides an **upper bound** on the true optimal: any feasible integer solution found is at least as costly as the true optimum. However, obtaining a valid **lower bound** is generally not possible until convergence is achieved.

The restricted master LP (with only the generated columns) gives an LP objective $z_{LP}(\Omega_k)$, but this is **not** a valid lower bound before convergence. For minimization, fewer columns means fewer options, so $z_{LP}(\Omega_k) \geq z_{LP}(H)$ — the restricted LP can only be *worse* than the full LP. The dual solution from the restricted master violates constraints for columns not yet generated (that's exactly what a negative reduced cost means), so it is not dual-feasible for the full problem and cannot serve as a bound.

**Only after convergence** (no negative reduced cost columns found with exact pricing) does $z_{LP}(\Omega_k) = z_{LP}(H)$, giving a valid lower bound. At that point the integrality gap $z_{IP} - z_{LP}$ is meaningful.

The solver reports convergence status and the LP/IP values:

```
Convergence: YES (no improving columns found)
LP value:    42.30    ← valid lower bound only because CG converged
IP solution: 51
LP-IP gap:   20.57%   ← true integrality gap
```

| Scenario | What you can claim |
|----------|-------------------|
| **CG converged with exact pricing** (`enumeration`) | LP is a proven lower bound. Gap is the true integrality gap. Optimal is between LP and IP. |
| **CG converged with heuristic pricing** (`local_search`, `beam_search`) | Pricing may have missed negative reduced cost columns, so "convergence" is not guaranteed. LP value is not a proven bound. |
| **CG stopped early** (time/iteration limit) | No bound on optimality. Can only say: "found a feasible solution with value X". |

In practice, for the instances in this project, the column generation implementation uses heuristic pricing and often stops at a time limit. The reported solutions are feasible schedules, but we cannot make claims about how close they are to optimal.

## Algorithm

```
1. INITIALIZATION
   └─ Generate initial feasible columns using greedy heuristic

2. COLUMN GENERATION LOOP
   ├─ Solve restricted master LP (relaxation)
   ├─ Extract dual values (π*, λ*)
   ├─ Solve pricing problems (for each tuple size)
   │  └─ Find columns with negative reduced cost
   ├─ Add new columns to master
   └─ Repeat until no improving columns found

3. FINAL MIP
   └─ Solve integer program with generated columns
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

| Strategy | Method | Complexity | Best for |
|----------|--------|-----------|----------|
| `enumeration` | Exact: try all n-tuples | $O(\binom{|X|}{n} \cdot |P|)$ | < 20 talks |
| `greedy` | Greedy construction | $O(n \cdot |X| \cdot |P|)$ | Quick warm-start |
| `local_search` | Greedy + swap improvement | $O(k \cdot n \cdot |X| \cdot |P|)$ | 20–50 talks (recommended) |
| `beam_search` | Beam search with pruning | $O(n \cdot w \cdot |X| \cdot |P|)$ | 50+ talks |

Use `pricing_strategy='auto'` (default) for automatic selection based on problem size.

## Usage

Column generation is selected via the config or CLI:

```bash
python scripts/run_schedule.py --pipeline traditional --phase1-method column_generation
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

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pricing_strategy` | `"auto"` | Pricing problem solver (auto, enumeration, local_search, beam_search) |
| `time_limit` | 120.0 | Total solver time limit (seconds) |
| `max_iterations` | 1000 | Maximum CG iterations |
| `gap_tolerance` | 0.01 | Optimality gap tolerance |

### Tuning Tips

- **Faster solutions:** Use `local_search`, increase `gap_tolerance` to 1e-3, reduce `max_iterations` to 30
- **Better quality:** Use `enumeration` (if feasible), decrease `gap_tolerance` to 1e-6
- **Very large instances:** Use `beam_search`, increase `time_limit`

## Implementation

Two solver implementations are provided in `src/columngeneration_phase1/`:

- **`phase1_column_generation.py`** — Core solver with enumeration-based pricing
- **`phase1_column_generation_enhanced.py`** — Advanced solver with all four pricing strategies (recommended)

Both integrate with the existing pipeline through `src/phase1.py`, which dispatches to column generation when `method = "column_generation"`.

## Limitations

The current column generation implementation is a straightforward adaptation and not heavily tailored to the conference scheduling problem structure. Better preprocessing (e.g., dominance-based filtering, tighter LP relaxations) and more problem-specific pricing strategies could improve performance. See the [Known Limitations](../README.md#known-limitations--improvement-opportunities) section in the README.
