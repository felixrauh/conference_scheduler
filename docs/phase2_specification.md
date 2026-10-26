# Phase 2: Minimize Room Hopping

## Problem Statement

Given the tuples of parallel talks selected in Phase 1, **assemble them into blocks** and **order them within each block** so that participants need to switch rooms as little as possible.

A **block** consists of $k$ consecutive tuples (the $k$ talk slots of a session block). Position $r$ of every tuple in a block is the same room, so a room-session is the sequence of talks at position $r$. A participant **hops** when they have to change rooms between consecutive talks they want to attend.

Phase 2 is used by the `traditional` and `heuristic` pipelines. The matching pipelines order talks in their own Phase D (see [matching_pipeline_specification.md](matching_pipeline_specification.md)).

---

## Hopping

### Definition (as implemented)

For a participant $p$ and an ordered block $B = (e_1, \ldots, e_k)$, let $R_j \subseteq \{1, \ldots, n\}$ be the rooms in which tuple $e_j$ contains a talk that $p$ wants to attend. The participant attends one preferred talk in every tuple where $R_j \neq \emptyset$ and stays put elsewhere. The **hopping number** $h(p, B)$ is the minimum number of room changes over all such choices:

$$h(p, B) = \min_{\rho_j \in R_j} \; \bigl|\{\, j : \rho_j \neq \rho_{j'} \,\}\bigr|$$

where $j'$ is the previous tuple with $R_{j'} \neq \emptyset$. It is computed with a small dynamic program over rooms (`compute_participant_hopping` in `src/phase2.py`). Participants with preferred talks in at most one tuple of the block never hop.

**Example:** Preferred rooms $(R_1, R_2, R_3, R_4) = (\{1\}, \emptyset, \{2, 3\}, \{3\})$ give $h = 1$: stay in room 1, skip slot 2, then go to room 3 for slots 3 and 4.

The total hopping of a block is $H(B) = \sum_{p \in P} h(p, B)$. The same definition is used by the evaluator's "session hops" metric.

> **Note:** This counts room switches. The paper's hopping number may be defined differently (see the README's Known Limitations on possible discrepancies with the paper).

### Ordering within a Block

$H(B)$ depends on the order of the tuples. Two strategies are available (`phase2.ordering_strategy`):

| Strategy | Method | Complexity |
|----------|--------|------------|
| `enumerate` (default) | Try all $k!$ orders, keep the best | $O(k! \cdot k \cdot n^2 \cdot \lvert P\rvert)$ — optimal; fine for $k \le 5$ |
| `greedy` | Build the order position by position, adding the tuple with the smallest hopping increase | Faster, for larger $k$ |

---

## Input and Output

```python
@dataclass
class Phase2Input:
    # Tuples from Phase 1, grouped by tuple size n: {n: [tuples of size n]}
    tuples_by_n: Dict[int, List[Tuple[str, ...]]]

    # Block specifications: [(n, k, count, block_type), ...]
    # e.g. [(5, 4, 4, "5R4T"), (5, 3, 1, "5R3T"), (4, 3, 1, "4R3T")]
    block_specs: List[Tuple[int, int, int, str]]

    # Participant preferences: {participant_id: set of talk_ids}
    preferences: Dict[str, Set[str]]

    # Optional pre-specified sequences (see "Fixed Sequences" below)
    fixed_sequences: Optional[List[FixedSequence]] = None


@dataclass
class Block:
    block_id: str                   # "B01", "B02", ...
    block_type: str                 # e.g. "5R4T"; Phase 3 matches it to timeslot types
    tuples: List[Tuple[str, ...]]   # Ordered: tuples[j][r] is the talk in slot j, room r
    hopping_cost: int


@dataclass
class Phase2Result:
    blocks: List[Block]
    total_hopping: int
```

Tuples of size $n$ can only go into blocks with $n$ rooms.

---

## Algorithm

`solve_phase2()` in `src/phase2.py` runs these steps:

```
1. Partition     tuples → unordered groups of k tuples, one per block (strategy below)
2. Local search  random swaps of tuples between blocks with the same n (keep improvements;
                 randomized, reproducible through the `seed` option)
3. Fixed seqs    attach pre-specified sequences as an extra room (optional)
4. Ordering      order the tuples of every block (enumerate | greedy)
```

With presenter availability constraints, the pipelines call `solve_phase2_with_feasibility_check()`, which wraps these steps in a feasibility check, repair and retry loop (see below).

### Partition Strategies

**Config key:** `phase2.partition_strategy` | **CLI:** `--phase2-partition` (traditional pipeline; the heuristic pipeline always uses `matching`)

| Strategy | Method |
|----------|--------|
| `greedy` (default) | Fills blocks one at a time. Each next tuple is the one with the fewest participants who also want to attend a tuple already in the block (conflict count). |
| `random` | Random assignment; a baseline for comparison. |
| `matching` | The paper's approach, for blocks with $k \in \{2, 4\}$ (see below). Other $k$ (e.g. 3) use a greedy partition with a presenter-availability compatibility check. |

**Matching-based partition** (`partition_tuples_matching_based`):

1. Build a complete graph whose nodes are the tuples.
2. The cost of an edge between two tuples is the optimal value of an assignment problem that aligns their talks room by room. Aligning talk $i$ (first tuple) with talk $j$ (second tuple) costs $d_{ij}$, the number of participants who want $i$ and another talk of the second tuple, or $j$ and another talk of the first tuple.
3. A minimum-cost perfect matching (networkx) pairs tuples into 2-blocks.
4. For $k = 4$, the 2-blocks are matched again into 4-blocks.

Edges that would create a block with no feasible timeslot for its presenters are excluded. If no perfect matching exists, the greedy partition is used instead.

When several block types share a tuple size (e.g. `5R4T` and `5R3T`), the tuples are split between them in list order before partitioning; block types are then assigned to the resulting blocks so that presenters avoid block types whose timeslots they cannot attend.

### Fixed Sequences

A `FixedSequence` (config `phase2.fixed_sequences`) is a list of $k$ talks that run one after another in one room. Phase 1 schedules one block of `result_block_type` as `target_block_type` (one room fewer); after partitioning, the sequence is attached to such a block as an extra room. The sequence keeps its talk order: with `enumerate` ordering, the block's own tuples are ordered around it to minimize hopping, and the block is not re-ordered afterwards.

### Presenter Availability: Check, Repair, Retry

After partitioning, every block must have at least one timeslot of its type at which all of its presenters are available (`check_block_feasibility`). If some block has none:

1. **Repair** (`repair_block_feasibility`): swap whole tuples between blocks with the same $n$. Each step applies the swap that makes the most infeasible blocks feasible without breaking a feasible one (ties: smallest hopping increase).
2. **Retry**: if the repair fails, re-partition — greedy with perturbed choices, or matching excluding the previous solution (Gurobi; without Gurobi the edge costs are randomly perturbed) — up to `max_feasibility_retries` (10) times.

Remaining violations are left to Phase 3 (which minimizes them) and swap optimization.

---

## Complexity

The number of ways to partition tuples into blocks is very large. For example, 20 tuples into 4 blocks of 3 and 2 blocks of 4:

- choose 12 tuples for the 3-blocks: $\binom{20}{12}$
- split those into 4 groups of 3: $\frac{12!}{(3!)^4 \cdot 4!}$
- split the remaining 8 into 2 groups of 4: $\frac{8!}{(4!)^2 \cdot 2!}$

Hence the heuristic partition strategies. Ordering within a block is cheap: 6 orders for $k = 3$, 24 for $k = 4$.

---

## Example

Requires Gurobi (for Phase 1):

```python
from collections import defaultdict

from src.data_loader import load_from_csv_dir
from src.instance import build_instance
from src.phase1 import solve_phase1
from src.phase2 import Phase2Input, solve_phase2

data = load_from_csv_dir("examples/synthetic")
data.validate()
instance = build_instance(data)

# Phase 1: tuples of parallel talks, grouped by size n
tuples_by_n = defaultdict(list)
for t in solve_phase1(instance, time_limit=60.0, verbose=False):
    tuples_by_n[len(t)].append(t)

# Phase 2: one block spec per block type: (n, k, count, block_type)
block_specs = [(bt["n"], bt["k"], bt["count"], type_id)
               for type_id, bt in instance.block_types.items()]
phase2_result = solve_phase2(
    Phase2Input(tuples_by_n=dict(tuples_by_n), block_specs=block_specs,
                preferences=instance.preferences),
    partition_strategy="matching",
    verbose=False,
)

print(f"Total hopping: {phase2_result.total_hopping}")
for block in phase2_result.blocks:
    print(f"Block {block.block_id} ({block.block_type}): hopping={block.hopping_cost}")
    for i, t in enumerate(block.tuples, 1):
        print(f"  Position {i}: {t}")
```

---

## References

- Vangerven et al. (2018), Section 5.2: "Phase 2: minimizing session hopping"
