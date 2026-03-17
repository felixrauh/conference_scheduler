# Phase 2: Minimize Session Hopping

## Problem Statement

Given the selected n-tuples from Phase 1, determine how to **assemble them into blocks** to minimize "session hopping."

A **block** consists of $k$ consecutive n-tuples (timeslots within a session). A participant "hops" when they attend non-consecutive talks within a session (e.g., talk 1 and talk 3, but not talk 2).

---

## Background from Paper

### Definition: Hopping Number

For a participant $p$ and a block $B = (e_1, e_2, \ldots, e_k)$ of $k$ consecutive tuples:

The **hopping number** $h(p, B)$ is computed as follows:
1. Let $a_j = 1$ if participant $p$ attends a talk in tuple $e_j$, else $a_j = 0$
2. Find the first and last attended tuples: $\text{first} = \min\{j : a_j = 1\}$, $\text{last} = \max\{j : a_j = 1\}$
3. Count the gaps: $h(p, B) = \sum_{j=\text{first}}^{\text{last}} (1 - a_j)$

**Example:** If $(a_1, a_2, a_3, a_4) = (1, 0, 1, 1)$, then first=1, last=4, and hopping = 1 (the gap at position 2).

### Total Hopping for a Block

$$H(B) = \sum_{p \in P} h(p, B)$$

### Dynamic Programming for Block Cost

For a given set of $k$ tuples $\{e_1, \ldots, e_k\}$ (unordered), the optimal ordering that minimizes hopping can be computed in $O(k! \cdot |P|)$ by trying all permutations.

However, the paper uses a DP formulation. For each participant, define:

$$f(j, s) = \text{minimum hopping to schedule first } j \text{ tuples ending in state } s$$

where state $s \in \{\text{not-started}, \text{in-session}, \text{finished}\}$.

---

## Input from Phase 1

Phase 1 returns a list of selected n-tuples:

```python
Phase1Result = List[Tuple[str, ...]]
# Example: [('T001', 'T002', 'T010', 'T012'), ('T004', 'T005', 'T007', 'T011'), ...]
```

All tuples have the same size $n$ (number of parallel rooms).

---

## Phase 2 Problem

### Given
- Selected tuples $E = \{e_1, e_2, \ldots, e_m\}$ from Phase 1
- Block types $\mathcal{B}$ with parameters $(n_\beta, k_\beta, r_\beta)$
- Participant preferences

### Find
- Partition of tuples into blocks
- Ordering of tuples within each block

### Minimize
Total hopping across all blocks and participants:
$$\min \sum_{B \in \text{Blocks}} H(B)$$

---

## Extended Formulation (Variable $k$)

### Challenge

In our setup, different block types have different $k$ values:
- Block type `4R3T`: $k = 3$ (3 talks per session → 3 tuples per block)
- Block type `4R4T`: $k = 4$ (4 talks per session → 4 tuples per block)

We have 20 tuples total and need to partition them into:
- 4 blocks of size 3 (from `4R3T`: 4 blocks × 3 = 12 tuples)
- 2 blocks of size 4 (from `4R4T`: 2 blocks × 4 = 8 tuples)
- Total: 12 + 8 = 20 tuples ✓

### Key Insight

Since all tuples have the same $n$ (from Phase 1), we need to decide:
1. **Which tuples go into which block type** (size-3 vs size-4 blocks)
2. **How to order tuples within each block** to minimize hopping

---

## Mathematical Formulation

### Sets

- $E$: Set of selected tuples from Phase 1 (all of size $n$)
- $\mathcal{K}$: Set of block sizes needed, with multiplicity
  - Derived from block types: $\{(k_1, c_1), (k_2, c_2), \ldots\}$ where $c_i$ = count of blocks of size $k_i$
- $P$: Set of participants

### Parameters

- $h(e_1, e_2, \ldots, e_k)$: Hopping cost for a block with tuples in given order
  - Computed by evaluating all permutations or using DP

### Decision

Partition $E$ into blocks and order tuples within each block.

### Approaches

#### Approach A: Two-Stage

1. **Stage 1: Partition tuples into block-size groups**
   - Assign each tuple to a "slot" in some block
   - This is a set partitioning / assignment problem

2. **Stage 2: Order tuples within each block**
   - For each block, find optimal permutation minimizing hopping
   - Can be done by enumeration for small $k$

#### Approach B: Direct IP Formulation

**Decision Variables:**
- $y_{e,b,j} \in \{0,1\}$: 1 if tuple $e$ is assigned to block $b$ at position $j$

**Constraints:**
1. Each tuple assigned to exactly one (block, position):
$$\sum_{b} \sum_{j=1}^{k_b} y_{e,b,j} = 1 \quad \forall e \in E$$

2. Each (block, position) has exactly one tuple:
$$\sum_{e \in E} y_{e,b,j} = 1 \quad \forall b, j$$

**Objective:**
Minimize total hopping (requires linearization of the hopping function).

#### Approach C: Heuristic (Greedy Assembly)

1. Compute pairwise "compatibility" between tuples based on hopping reduction
2. Greedily assemble blocks by picking compatible tuples
3. Order tuples within each block optimally

---

## Data Structures

### Input

```python
@dataclass
class Phase2Input:
    """Input to Phase 2."""
    
    # Selected tuples from Phase 1
    tuples: List[Tuple[str, ...]]
    
    # Block sizes needed: [(k, count), ...]
    # e.g., [(3, 4), (4, 2)] means 4 blocks of size 3 and 2 blocks of size 4
    block_sizes: List[Tuple[int, int]]
    
    # Participant preferences: {participant_id: set of talk_ids}
    preferences: Dict[str, Set[str]]
```

### Output

```python
@dataclass
class Block:
    """A block of k ordered tuples."""
    
    block_id: str
    block_type: str  # e.g., '4R3T'
    tuples: List[Tuple[str, ...]]  # Ordered list of tuples
    hopping_cost: int

@dataclass
class Phase2Result:
    """Output from Phase 2."""
    
    blocks: List[Block]
    total_hopping: int
```

---

## Hopping Computation

### Per-Participant Hopping for a Block

```python
def compute_participant_hopping(
    block_tuples: List[Tuple[str, ...]],
    participant_prefs: Set[str]
) -> int:
    """
    Compute hopping for one participant in a block.
    
    Args:
        block_tuples: Ordered list of tuples in the block
        participant_prefs: Set of talk_ids this participant wants to attend
    
    Returns:
        Number of "hops" (gaps between attended tuples)
    """
    # For each tuple position, check if participant attends any talk
    attendance = []
    for ntuple in block_tuples:
        attends = any(talk_id in participant_prefs for talk_id in ntuple)
        attendance.append(1 if attends else 0)
    
    # Find first and last attendance
    if sum(attendance) <= 1:
        return 0  # No hopping possible with 0 or 1 attended
    
    first = attendance.index(1)
    last = len(attendance) - 1 - attendance[::-1].index(1)
    
    # Count gaps
    hopping = sum(1 - attendance[j] for j in range(first, last + 1))
    return hopping
```

### Total Block Hopping

```python
def compute_block_hopping(
    block_tuples: List[Tuple[str, ...]],
    preferences: Dict[str, Set[str]]
) -> int:
    """Compute total hopping for a block across all participants."""
    total = 0
    for p_id, prefs in preferences.items():
        total += compute_participant_hopping(block_tuples, prefs)
    return total
```

---

## Design Decisions

### Q1: How to partition tuples into different block sizes?

**Decision:** Start with **heuristic** (greedy by compatibility), but design with a clean interface so it can be **replaced by an IP-based approach** later.

**Implementation:**
```python
def partition_tuples_into_blocks(
    tuples: List[Tuple[str, ...]],
    block_sizes: List[Tuple[int, int]],  # [(k, count), ...]
    preferences: Dict[str, Set[str]],
    strategy: str = "greedy"  # "greedy" | "ip" | "random"
) -> List[List[Tuple[str, ...]]]:
    """
    Partition tuples into groups for each block.
    Returns list of unordered tuple groups, one per block.
    """
    if strategy == "greedy":
        return _partition_greedy(tuples, block_sizes, preferences)
    elif strategy == "ip":
        return _partition_ip(tuples, block_sizes, preferences)
    else:
        return _partition_random(tuples, block_sizes)
```

### Q2: How to handle ordering within blocks?

**Decision:** Start with **enumeration** of all permutations (feasible for k ≤ 4), but encapsulate in a function that can be **replaced by DP or other methods** later.

**Implementation:**
```python
def optimize_block_ordering(
    tuples: List[Tuple[str, ...]],  # Unordered tuples for one block
    preferences: Dict[str, Set[str]],
    strategy: str = "enumerate"  # "enumerate" | "dp" | "greedy"
) -> Tuple[List[Tuple[str, ...]], int]:
    """
    Find optimal ordering of tuples within a block.
    Returns (ordered_tuples, hopping_cost).
    """
    if strategy == "enumerate":
        return _order_by_enumeration(tuples, preferences)
    elif strategy == "dp":
        return _order_by_dp(tuples, preferences)
    else:
        return _order_greedy(tuples, preferences)
```

### Q3: Exact vs Heuristic (Overall)

**Clarification:** This was asking the same as Q1 at a higher level. Since n=4 (≥3), the problem is NP-hard per the paper. The decision is:
- **Heuristic first** (greedy partition + enumeration for ordering)
- **IP-based partition as optional upgrade** (clean interface allows swapping)

### Q4: Interface with Phase 3

**Decision:** Yes, include `block_type` in the Block output so Phase 3 can match blocks to compatible timeslots.

---

## Architecture: Modular Design

```
Phase2Input
    │
    ▼
┌─────────────────────────────────┐
│  partition_tuples_into_blocks() │  ← Strategy: "greedy" | "ip"
│  Returns: unordered tuple groups│
└─────────────────────────────────┘
    │
    ▼ (for each group)
┌─────────────────────────────────┐
│  optimize_block_ordering()      │  ← Strategy: "enumerate" | "dp"
│  Returns: ordered tuples + cost │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Build Block objects            │
│  Include: block_type, tuples,   │
│           hopping_cost          │
└─────────────────────────────────┘
    │
    ▼
Phase2Result
```

---

## Complexity Analysis

### Tuple Enumeration
- Number of ways to partition 20 tuples into blocks: combinatorially large
- For 4 blocks of 3 and 2 blocks of 4:
  - Choose 12 tuples for size-3 blocks: $\binom{20}{12}$
  - Partition those 12 into 4 groups of 3: $\frac{12!}{(3!)^4 \cdot 4!}$
  - Partition remaining 8 into 2 groups of 4: $\frac{8!}{(4!)^2 \cdot 2!}$
  - This is very large → need heuristic or optimization

### Ordering Within Blocks
- For $k = 3$: 6 permutations per block
- For $k = 4$: 24 permutations per block
- With 6 blocks total: manageable

---

## Example

```python
from src.phase2 import assemble_blocks, Phase2Input

# Input from Phase 1
phase1_result = [
    ('T001', 'T027', 'T045', 'T069'),
    ('T002', 'T004', 'T043', 'T065'),
    # ... 18 more tuples
]

# Block sizes needed
block_sizes = [
    (3, 4),  # 4 blocks of size 3 (from 4R3T)
    (4, 2),  # 2 blocks of size 4 (from 4R4T)
]

# Run Phase 2
phase2_input = Phase2Input(
    tuples=phase1_result,
    block_sizes=block_sizes,
    preferences=instance.preferences
)

phase2_result = assemble_blocks(phase2_input)

print(f"Total hopping: {phase2_result.total_hopping}")
for block in phase2_result.blocks:
    print(f"Block {block.block_id} ({block.block_type}): hopping={block.hopping_cost}")
    for i, t in enumerate(block.tuples, 1):
        print(f"  Position {i}: {t}")
```

---

## References

- Vangerven et al. (2018), Section 5.2: "Phase 2: minimizing session hopping"
- Equations (4)-(7) in the paper (hopping number definition and DP)
