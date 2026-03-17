# Matching-Based Pipeline: Bottom-Up Conference Scheduling

## Overview

This document specifies an alternative pipeline for conference scheduling that builds sessions **bottom-up** by first matching talks into pairs, then assembling pairs into session blocks, and finally optimizing across parallel sessions.

### Philosophy

The original pipeline (Phase 1 → Phase 2 → Phase 3) works **top-down**:
1. First decide which talks run in parallel (tuple selection)
2. Then group tuples into blocks (block assembly)
3. Finally assign to rooms and order talks

This **matching-based pipeline** works **bottom-up**:
1. First pair talks that people want to see together (pair matching)
2. Assemble pairs into session blocks (block formation)
3. Select which blocks run in parallel (tuple selection)
4. Optimize talk ordering for room-hopping (talk ordering)

### Motivation

- Talks paired early are guaranteed not to conflict
- Natural way to preserve "must-see-together" relationships
- May produce different (potentially better) solutions than top-down approach
- More intuitive: "keep related talks together" before "separate conflicting talks"

---

## Phase A: Pair Matching

### Problem Statement

Match talks into pairs such that the total "co-preference weight" is maximized, subject to matching exactly the required number of pairs.

### Input

- $X$: Set of all talks
- $P$: Set of all participants  
- $q(p)_i \in \{0, 1\}$: 1 if participant $p$ wants to attend talk $i$
- Block configuration determining:
  - $n_3$: Number of sessions with 3 talks each
  - $n_4$: Number of sessions with 4 talks each

### Derived Parameters

**Number of pairs to match:**
$$M = n_3 + 2 \cdot n_4$$

Rationale:
- Each 3-talk session needs 1 pair + 1 single
- Each 4-talk session needs 2 pairs (pair + pair)

**Co-preference weight** between talks $i$ and $j$:
$$w_{ij} = \sum_{p \in P} q(p)_i \cdot q(p)_j$$

This counts how many participants want to attend **both** talks.

### Mathematical Formulation

**Decision Variables:**
- $y_{ij} \in \{0, 1\}$: 1 if talks $i$ and $j$ are matched as a pair

**Model:**
$$\max \sum_{i < j} w_{ij} \cdot y_{ij}$$

Subject to:

1. **Matching constraint** — each talk is in at most one pair:
$$\sum_{j: j \neq i} y_{ij} \leq 1 \quad \forall i \in X$$

2. **Cardinality constraint** — exactly $M$ pairs:
$$\sum_{i < j} y_{ij} = M$$

3. **Binary constraints:**
$$y_{ij} \in \{0, 1\} \quad \forall i < j$$

4. **Presenter unavailability constraint** — infeasible pairs are excluded:
$$y_{ij} = 0 \quad \forall (i, j) \in \mathcal{I}$$

where $\mathcal{I}$ is the set of talk pairs whose presenters' combined unavailabilities cover all timeslots.

### Output

- $\mathcal{P}$: Set of matched pairs, $|\mathcal{P}| = M$
- $\mathcal{S}$: Set of unmatched singles, $|\mathcal{S}| = |X| - 2M$

### Verification

$$|\mathcal{S}| = |X| - 2M = |X| - 2(n_3 + 2n_4) = |X| - 2n_3 - 4n_4$$

Since $|X| = 3n_3 + 4n_4$ (total talks = sum over all sessions):
$$|\mathcal{S}| = 3n_3 + 4n_4 - 2n_3 - 4n_4 = n_3$$

This is exactly the number of singles needed (one per 3-talk session). ✓

### Complexity

- Graph: $|X|$ nodes, $\binom{|X|}{2}$ edges
- This is a **maximum weight $b$-matching** with cardinality constraint
- Can be solved as MILP or using specialized matching algorithms
- For $|X| \approx 80$: ~3,200 edges, easily solvable

### Alternative Formulations

**Variant A: Minimum weight matching (co-conflict instead of co-preference)**
- Instead of maximizing $w_{ij}$, minimize co-preference for pairs that will be in the same room
- Useful if interpretation is "pairs in same session don't conflict"

**Variant B: Weighted by preference strength**
- Use weighted preferences: $w_{ij} = \sum_p q(p)_i \cdot q(p)_j \cdot \text{strength}(p, i) \cdot \text{strength}(p, j)$

---

## Phase B: Block Formation

### Problem Statement

Given matched pairs and unmatched singles, form session blocks:
- 3-blocks: pair + single
- 4-blocks: pair + pair

Maximize the total pairwise preference weight within each block.

### Input

- $\mathcal{P}$: Set of matched pairs from Phase A
- $\mathcal{S}$: Set of unmatched singles from Phase A  
- $n_3$: Number of 3-talk sessions (each needs one 3-block)
- $n_4$: Number of 4-talk sessions (each needs one 4-block)
- $w_{ij}$: Co-preference weight between talks $i$ and $j$

### Block Weight Definitions

For a **pair** $P = \{i, j\}$:
$$W(P) = w_{ij}$$

For a **3-block** $B_3 = \{P, s\}$ where $P = \{i, j\}$ is a pair and $s$ is a single:
$$W(B_3) = w_{ij} + w_{is} + w_{js}$$

The pair contribution $w_{ij}$ is already counted, so the **marginal weight** of adding single $s$ to pair $P$:
$$\Delta W(P, s) = w_{is} + w_{js}$$

For a **4-block** $B_4 = \{P_1, P_2\}$ where $P_1 = \{i, j\}$ and $P_2 = \{k, l\}$:
$$W(B_4) = w_{ij} + w_{kl} + w_{ik} + w_{il} + w_{jk} + w_{jl}$$

The **marginal weight** of combining two pairs:
$$\Delta W(P_1, P_2) = w_{ik} + w_{il} + w_{jk} + w_{jl}$$

### Mathematical Formulation (Joint MILP)

We optimize which pairs form 4-blocks (pair+pair) and which pairs form 3-blocks (pair+single) jointly:

**Decision Variables:**
- $z_{P_1, P_2} \in \{0, 1\}$: pairs $P_1, P_2$ form a 4-block
- $u_{P, s} \in \{0, 1\}$: pair $P$ and single $s$ form a 3-block

**Model:**
$$\max \sum_{P_1 < P_2} \Delta W(P_1, P_2) \cdot z_{P_1, P_2} + \sum_{P, s} \Delta W(P, s) \cdot u_{P, s}$$

Subject to:

1. **Each pair used exactly once** (either in a 4-block or 3-block):
$$\sum_{P_2 \neq P} z_{P, P_2} + \sum_{s \in \mathcal{S}} u_{P, s} = 1 \quad \forall P \in \mathcal{P}$$

2. **Each single used exactly once:**
$$\sum_{P \in \mathcal{P}} u_{P, s} = 1 \quad \forall s \in \mathcal{S}$$

3. **Correct number of 4-blocks:**
$$\sum_{P_1 < P_2} z_{P_1, P_2} = n_4$$

4. **Binary constraints:**
$$z_{P_1, P_2}, u_{P, s} \in \{0, 1\}$$

5. **Presenter unavailability constraint** — infeasible blocks are excluded:
$$z_{P_1, P_2} = 0 \quad \forall (P_1, P_2) \text{ where combined presenter unavailabilities cover all timeslots}$$
$$u_{P, s} = 0 \quad \forall (P, s) \text{ where combined presenter unavailabilities cover all timeslots}$$

### Output

- $\mathcal{B}_3$: Set of 3-blocks, $|\mathcal{B}_3| = n_3$
- $\mathcal{B}_4$: Set of 4-blocks, $|\mathcal{B}_4| = n_4$
- $\mathcal{B} = \mathcal{B}_3 \cup \mathcal{B}_4$: All blocks

### Complexity

- Part 1: Perfect matching on $2n_4$ nodes → $O(n_4^3)$ with Hungarian or MILP
- Part 2: Bipartite matching on $n_3 + n_3$ nodes → $O(n_3^3)$
- Joint: MILP with $O(M^2 + M \cdot |\mathcal{S}|)$ variables

---

## Phase C: Tuple Selection

### Problem Statement

Given the formed blocks, select which blocks run in parallel (forming "tuples") to minimize missed attendance.

This is analogous to Phase 1 of the original pipeline, but operating on **blocks** instead of **individual talks**.

### Input

- $\mathcal{B}$: Set of all blocks from Phase B
- $\mathcal{B}_3, \mathcal{B}_4$: Subsets of 3-blocks and 4-blocks
- $P$: Set of participants
- $q(p)_i$: Preference of participant $p$ for talk $i$
- Block configuration (variable n supported):
  - Tuple requirements for 3-blocks: $R_3 = \{(n_1, c_1), (n_2, c_2), \ldots\}$ where we need $c_i$ tuples of size $n_i$
  - Tuple requirements for 4-blocks: $R_4 = \{(n_1, c_1), (n_2, c_2), \ldots\}$
  - Example: If we have 2 blocks of type "4R3T" (4 rooms, 3 talks) and 1 block of type "3R3T" (3 rooms, 3 talks), then $R_3 = \{(4, 2), (3, 1)\}$

### Block Preference

Define whether participant $p$ wants to attend block $B$:
$$Q(p, B) = \max_{i \in B} q(p)_i$$

If $q(p)_i \in \{0, 1\}$, this is 1 if the participant wants any talk in the block.

For weighted preferences:
$$Q(p, B) = \max_{i \in B} q(p)_i \quad \text{(best talk in block)}$$

### Tuple Definition

A **tuple** $\tau = (B_1, B_2, \ldots, B_n)$ is an ordered collection of $n$ blocks, one per parallel room, where $n$ can vary per tuple based on the block types.

**Constraint:** All blocks in a tuple must have the same size (all 3-blocks or all 4-blocks) since they occupy the same timeslot.

### Tuple Cost (Missed Attendance — Talk-Level)

For a tuple $\tau = (B_1, \ldots, B_n)$ and participant $p$:

1. **Count interested talks per block:**
$$n_j(p) = \sum_{i \in B_j} q(p)_i$$

2. **Participant chooses their best block** (the one with most interesting talks):
$$B^*(p) = \arg\max_j n_j(p)$$

3. **Missed talks** = talks in other blocks that they wanted:
$$\text{missed}(p, \tau) = \sum_{j: B_j \neq B^*(p)} n_j(p) = \sum_{j=1}^{r} n_j(p) - \max_j n_j(p)$$

**Total tuple cost:**
$$c_\tau = \sum_{p \in P} \text{missed}(p, \tau) = \sum_{p \in P} \left( \sum_{B \in \tau} \sum_{i \in B} q(p)_i - \max_{B \in \tau} \sum_{i \in B} q(p)_i \right)$$

This counts the **number of specific talks missed** assuming each participant attends their best block entirely.

### Mathematical Formulation (Variable n)

**Sets:**
- $\mathcal{B}_3$: Set of 3-blocks from Phase B
- $\mathcal{B}_4$: Set of 4-blocks from Phase B
- For each required tuple size $n$:
  - $\mathcal{T}_3^{(n)}$: Feasible $n$-tuples from 3-blocks
  - $\mathcal{T}_4^{(n)}$: Feasible $n$-tuples from 4-blocks
- $\mathcal{T}_3 = \bigcup_n \mathcal{T}_3^{(n)}$, $\mathcal{T}_4 = \bigcup_n \mathcal{T}_4^{(n)}$

**Parameters:**
- $R_3$: Dictionary mapping tuple size $n$ to count needed for 3-blocks
- $R_4$: Dictionary mapping tuple size $n$ to count needed for 4-blocks
- Example: $R_3 = \{4: 2, 3: 1\}$ means we need 2 4-tuples and 1 3-tuple of 3-blocks

**Decision Variables:**
- $x_\tau \in \{0, 1\}$: 1 if tuple $\tau$ is selected

**Model:**
$$\min \sum_{\tau \in \mathcal{T}_3 \cup \mathcal{T}_4} c_\tau \cdot x_\tau$$

Subject to:

1. **Coverage constraint** — each block appears in exactly one tuple:
$$\sum_{\tau: B \in \tau} x_\tau = 1 \quad \forall B \in \mathcal{B}_3 \cup \mathcal{B}_4$$

2. **Tuple count constraints per size** — select exactly the required number of tuples of each size:
$$\sum_{\tau \in \mathcal{T}_3^{(n)}} x_\tau = R_3[n] \quad \forall n \in R_3$$
$$\sum_{\tau \in \mathcal{T}_4^{(n)}} x_\tau = R_4[n] \quad \forall n \in R_4$$

3. **Binary constraints:**
$$x_\tau \in \{0, 1\}$$

4. **No-good cuts** (for feasibility retry):
$$\sum_{\tau \in S^{(k)}} x_\tau \leq |S^{(k)}| - 1 \quad \forall k \in \text{previous violations}$$

where $S^{(k)}$ is the set of tuples selected in iteration $k$ that led to a Phase 3 presenter violation.

### Complexity

- Number of 3-block tuples: $\binom{|\mathcal{B}_3|}{r} \cdot r!$ (ordered) or $\binom{|\mathcal{B}_3|}{r}$ (unordered)
- For small $r$ (e.g., 4 rooms), this is manageable
- Similar to original Phase 1, can use column generation for large instances

### Output

- Selected tuples partitioning all blocks
- Each tuple represents a timeslot with $r$ parallel sessions

---

## Phase D: Talk Ordering Within Blocks

### Problem Statement

Given the tuple assignment, optimize the **order of talks within each block** to maximize room-hopping opportunities.

### Background: Room Hopping

When multiple blocks run in parallel (as a tuple), attendees might want talks from different blocks. If talks are properly ordered, they can "hop" between rooms:

**Example:** Tuple with 2 parallel 3-blocks:
- Room 1: $[A_1, A_2, A_3]$
- Room 2: $[B_1, B_2, B_3]$

A participant wanting $A_1$ and $B_3$ can:
1. Attend $A_1$ in Room 1 (timeslot 1)
2. Hop to Room 2 after $A_1$
3. Attend $B_3$ in Room 2 (timeslot 3)

This is only possible if $A_1$ is scheduled before $B_3$.

### Input

- Selected tuples from Phase C
- For each tuple $\tau = (B_1, \ldots, B_r)$: the blocks and their talks
- Participant preferences $q(p)_i$

### Ordering Cost

For a tuple with ordered blocks:
- $\sigma_j$: permutation of talks within block $B_j$
- $B_j[\sigma_j] = (t_{j,1}, t_{j,2}, \ldots, t_{j,k})$: ordered talks in room $j$

A participant can attend a **feasible schedule**: a sequence of (room, timeslot) pairs that is non-decreasing in timeslot.

**Participant's achievable attendance:**
$$A(p, \tau, \sigma) = \max_{\text{feasible schedule } S} \sum_{(j, k) \in S} q(p)_{t_{j,k}}$$

**Room-hopping benefit:**
$$\text{Benefit} = \sum_p A(p, \tau, \sigma) - \sum_p \max_j \sum_{i \in B_j} q(p)_i$$

The second term is what they'd get without hopping (just their best block).

### Mathematical Formulation

This can be formulated per-tuple. For a tuple $\tau = (B_1, \ldots, B_r)$:

**Decision Variables:**
- $\pi_{j,i,k} \in \{0, 1\}$: 1 if talk $i$ from block $B_j$ is in position $k$

**Constraints:**
$$\sum_k \pi_{j,i,k} = 1 \quad \forall j, i \in B_j \quad \text{(each talk has one position)}$$
$$\sum_{i \in B_j} \pi_{j,i,k} = 1 \quad \forall j, k \quad \text{(each position has one talk)}$$

The objective involves computing achievable attendance, which requires auxiliary variables for participant schedules.

### Simplified Heuristic Approach

Given small block sizes ($k \in \{3, 4\}$) and moderate room count ($r \approx 4$):

**Total orderings per tuple:** $(k!)^r$
- For 4 rooms with 3-blocks: $(3!)^4 = 1296$
- For 4 rooms with 4-blocks: $(4!)^4 = 331,776$

**Approach:** Enumerate all orderings, compute benefit, select best.

### Algorithm: Greedy Schedule for Participant

Given ordered blocks, compute participant $p$'s achievable attendance:

```
def achievable_attendance(p, ordered_blocks):
    # Dynamic programming over timeslots
    # State: current timeslot, set of attended talks
    # Transition: for each room, decide attend or not
    
    k = len(ordered_blocks[0])  # talks per block
    
    # dp[t] = max talks attended by end of timeslot t
    dp = [0] * (k + 1)
    
    for t in range(1, k + 1):
        # Available talks at timeslot t
        available = [block[t-1] for block in ordered_blocks]
        best_at_t = max(q(p, talk) for talk in available)
        dp[t] = dp[t-1] + best_at_t
    
    return dp[k]
```

### Output

- For each tuple: optimal ordering of talks within each block
- Equivalently: for each block, a permutation of its talks

---

## Summary: Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT                                     │
│  - Talks X                                                       │
│  - Participant preferences q(p)_i                                │
│  - Block configuration (n_3, n_4, r)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               PHASE A: Pair Matching                             │
│  - Build co-preference graph                                     │
│  - Find maximum weight matching with cardinality M = n_3 + 2n_4  │
│  - Output: M pairs, n_3 singles                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               PHASE B: Block Formation                           │
│  - Match pairs to pairs → n_4 4-blocks                           │
│  - Match pairs to singles → n_3 3-blocks                         │
│  - Maximize pairwise weight within blocks                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               PHASE C: Tuple Selection                           │
│  - Enumerate tuples of blocks (r blocks per tuple)               │
│  - Cost = missed attendance (choosing best block in tuple)       │
│  - Select tuples covering all blocks                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               PHASE D: Talk Ordering                             │
│  - For each tuple, order talks within blocks                     │
│  - Maximize within-session room-hopping opportunities            │
│  - (Attend talks from multiple parallel blocks by switching)     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               PHASE 3: Room Assignment (from original pipeline)  │
│  - Assign tuples to physical timeslots                           │
│  - Assign blocks to physical rooms                               │
│  - Handle presenter availability constraints                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUT                                    │
│  - Complete schedule: rooms × timeslots → talks                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Comparison with Original Pipeline

| Aspect | Original Pipeline | Matching Pipeline |
|--------|------------------|-------------------|
| Direction | Top-down | Bottom-up |
| Phase 1 | Tuple selection (which talks parallel) | Pair matching (which talks together) |
| Phase 2 | Block assembly (order tuples) | Block formation (pairs → blocks) |
| Phase 3 | Room assignment | Tuple selection + ordering |
| Conflict handling | Minimize first | Maximize togetherness first |
| Key optimization | Avoid conflicts in parallel | Group co-preferred talks |

### Theoretical Properties

**Original pipeline:**
- Directly minimizes total conflicts
- May separate talks that people want together

**Matching pipeline:**
- First ensures related talks are grouped
- Then minimizes cross-group conflicts
- May produce different Pareto-optimal solutions

---

## Implementation Notes

### Solver Requirements

- **Phase A:** Maximum weight matching with cardinality constraint → MILP or Blossom algorithm
- **Phase B:** Bipartite/perfect matching → Hungarian algorithm or MILP
- **Phase C:** Set partitioning → MILP (similar to original Phase 1)
- **Phase D:** Enumeration or MILP for small instances

### Heuristic Alternatives

1. **Phase A:** Greedy matching by weight
2. **Phase B:** Greedy assignment
3. **Phase C:** Greedy or local search
4. **Phase D:** Random restarts with local optimization

### Presenter Unavailability Feasibility

The pipeline enforces presenter availability constraints at multiple levels:

**Phase A (Pair Matching):**
- Compute infeasible talk pairs where combined presenter unavailabilities cover all timeslots
- Add constraints `y[i,j] = 0` for all infeasible pairs
- Example: If presenter A is unavailable for TA/TB and presenter B is unavailable for TC/TD/FA/FB/FC, their talks cannot be paired

**Phase B (Block Formation):**
- For each candidate 3-block or 4-block, check if combined unavailabilities cover all timeslots
- Exclude infeasible blocks from the optimization model
- Uses `is_block_feasible()` helper to test each combination

**Phase C (Tuple Selection) with Retry:**
- After Phase D ordering, run Phase 3 to check for presenter violations
- If violations found, add no-good cut to exclude the current solution:
  ```
  sum(x[τ] for τ in previous_solution) ≤ |previous_solution| - 1
  ```
- Retry with reduced time limit (up to `max_feasibility_retries` attempts)
- This ensures the optimizer finds alternative tuple assignments

**Sample Output:**
```
PHASE A: Pair Matching
  Infeasible talk pairs (presenter unavailability conflicts):
    Found 4 infeasible pairs
      T011 (P011) + T112 (P112)
      T066 (P066) + T112 (P112)
      ...
    Added 4 infeasible pair constraints

PHASE B: Block Formation
  Excluded 3 infeasible 4-blocks, 1 infeasible 3-blocks

PHASE C: Tuple Selection (retry 2)
  Added 2 no-good cuts
```

### Configuration Parameters

```json
{
  "matching_pipeline": {
    "phase_a": {
      "method": "milp",           // "milp", "greedy", "blossom"
      "weight_type": "co_preference"  // "co_preference", "co_conflict_inverse"
    },
    "phase_b": {
      "method": "joint_milp",     // "sequential", "joint_milp"
      "weight_type": "marginal"   // "marginal", "total"
    },
    "phase_c": {
      "method": "milp",           // "milp", "greedy", "column_generation"
      "cost_type": "block_level"  // "block_level", "talk_level"
    },
    "phase_d": {
      "method": "enumeration",    // "enumeration", "milp", "greedy"
      "objective": "room_hopping" // "room_hopping", "session_continuity"
    }
  }
}
```

---

## Design Decisions (Confirmed)

| Decision | Choice | Rationale |
|----------|--------|----------|
| **Phase A Objective** | Maximize co-preference | Pairs in same room means you CAN attend both |
| **Phase B Approach** | Joint MILP | Better global optimization, computationally feasible for ~120 talks |
| **4-Block Formation** | Pair + pair only | Set up in Phase A with exact cardinality |
| **Tuple Cost** | Talk-level counting | More precise: counts specific missed talks, not just blocks |
| **Phase D Objective** | Within-session room hopping | Reorder talks to enable attending multiple parallel blocks |
| **Phase 3** | Reuse from original pipeline | Room assignment + presenter availability |

---

## References

- Vangerven et al. (2018): Original conference scheduling formulation
- Original pipeline: [phase1_specification.md](phase1_specification.md), [phase2_specification.md](phase2_specification.md)
