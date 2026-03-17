# Developer Notes

Internal implementation details for contributors. For user-facing configuration, see [configuration_and_algorithm_reference.md](configuration_and_algorithm_reference.md).

---

## Presenter Availability — Implementation Details

Presenter availability is enforced through multiple mechanisms. See the user-facing [Presenter Availability Constraints](configuration_and_algorithm_reference.md#presenter-availability-constraints) section for the high-level overview.

### Infeasible Talk Pairs

**Location**: `src/instance.py` → `compute_infeasible_talk_pairs()`

Two talks form an infeasible pair if their presenters' combined unavailabilities cover too many timeslots, leaving insufficient scheduling options.

```python
def compute_infeasible_talk_pairs(
    instance: ProblemInstance,
    min_unavailable_threshold: int = 5,  # Configurable strictness
    verbose: bool = False
) -> Set[Tuple[str, str]]:
```

**Threshold semantics** (assuming 7 total timeslots):
- `min_unavailable_threshold=7`: Only exclude pairs where ALL 7 timeslots are blocked (strictest minimum)
- `min_unavailable_threshold=5`: Exclude pairs with ≤2 available timeslots (recommended default)
- `min_unavailable_threshold=3`: Exclude pairs with ≤4 available timeslots (more conservative)

**Example**: If Talk A's presenter is unavailable for {TA, TB, TC} and Talk B's presenter is unavailable for {TC, TD, FA}, their union is {TA, TB, TC, TD, FA} = 5 timeslots unavailable. With `min_unavailable_threshold=5`, this pair is excluded.

**Where it's used**:

| Pipeline | Phase | Usage |
|----------|-------|-------|
| Traditional | Phase 1 (explicit) | Excludes infeasible tuples from enumeration |
| Heuristic | Phase 1 (greedy) | Prevents infeasible pairs from being grouped |
| Matching | Phase A | Prevents infeasible pairs from being matched |

### Tuple Size Restrictions

**Location**: `src/instance.py` → `compute_forbidden_tuple_sizes()`

Some talks cannot be placed in certain tuple sizes because:
- Size-3 tuples → 3R4T blocks → only scheduled in certain timeslots (e.g., FA only)
- Size-5 tuples → 5R4T/5R3T blocks → only scheduled in other timeslots (e.g., TA, TB, TC)

If a presenter is unavailable for ALL timeslots of a block type, their talk cannot be in tuples of that size.

**Where it's used**:

| Pipeline | Phase | Usage |
|----------|-------|-------|
| Heuristic | Phase 1 (greedy) | Prevents assigning talks to forbidden tuple sizes |
| Traditional | Phase 1 | (Should be added for completeness) |

### Block Feasibility Checking

**Location**: `src/phase2.py` → `check_block_feasibility()`, `check_all_blocks_feasibility()`

After Phase 2 groups tuples into blocks, we verify each block can be assigned to at least one timeslot where all presenters are available.

```python
def check_block_feasibility(
    block: Block,
    talk_presenter: Dict[str, str],
    presenter_unavailability: Dict[str, Set[str]],
    all_timeslots: Set[str],
    timeslots_by_type: Optional[Dict[str, List[str]]] = None
) -> Tuple[bool, Set[str]]:
```

**Type-aware checking**: When `timeslots_by_type` is provided, only timeslots matching the block's type are considered feasible. The block type is accessed via `block.block_type`.

### Known Limitations

1. **Greedy Phase 1 can fail**: If too many talks have size restrictions, the greedy algorithm may consume eligible talks for larger sizes first, leaving insufficient talks for smaller sizes. This causes "Not enough n-tuples" errors.

2. **Phase 2 partition doesn't check cross-tuple constraints**: The partition step groups tuples into blocks without verifying that combining certain tuples creates feasible blocks. The feasibility check happens AFTER partitioning, which may require retries.

3. **Block type constraints are tight**: Some block types only have 1 valid timeslot (e.g., 3R4T → FA only). Talks that can only be in these block types have very limited scheduling flexibility.

### Debugging Presenter Violations

If you see presenter violations in the output:

1. **Check constraint loading**: Ensure `data.validate()` is called to populate `presenter_unavailability`
2. **Increase retries**: Set `max_feasibility_retries` higher in config
3. **Lower threshold**: Use `min_unavailable_threshold=3` for stricter pair filtering
4. **Check block type mappings**: Verify `timeslots_by_type` correctly maps block types to timeslots

---

## Swap Optimization — Implementation Details

### Violation Types

| Type | Description | Resolution |
|------|-------------|------------|
| `PRESENTER_UNAVAILABLE` | Presenter scheduled in unavailable timeslot | Swap to timeslot where presenter is available |
| `DUMMY_IN_SHORT_BLOCK` | Dummy talk in a block with ≤3 timeslots | Swap with non-dummy from longer block |
| `MULTIPLE_DUMMIES_IN_SESSION` | 2+ dummy talks in same room-session | Swap extra dummies to different room-sessions |

### Algorithm Detail

1. **Detect Violations**: Find all presenter and dummy violations
2. **Generate Candidates**: For each violation, find all feasible swap partners
3. **Score Swaps**: Evaluate each swap by:
   - Feasibility: Does not create new violations
   - Missed attendance delta: Preference conflicts avoided/created
   - Keyword coherence (optional): Impact on session topical coherence
4. **Apply Best Swap**: Greedily apply the best-scoring feasible swap
5. **Iterate**: Repeat until no violations remain or no feasible swaps exist

### Scoring Formula

```
combined_score = missed_attendance_delta - keyword_weight × keyword_delta
```

- `missed_attendance_delta`: Change in conflicts (negative = improvement)
- `keyword_delta`: Change in session keyword coherence (positive = improvement)
- Lower combined score = better swap

### Dummy Talk Handling

Dummy talks (DUMMY_001, DUMMY_002, etc.) are placeholders for empty slots when total_slots > total_talks. The swap optimization:

1. **Avoids short blocks**: Moves dummies from blocks with k≤3 slots to blocks with k>3 slots
2. **Spreads dummies**: Ensures no room-session has multiple dummies (avoids multiple empty slots in sequence)
3. **Minimizes impact**: Prefers swaps that don't increase missed attendance

### Complexity

- Time: O(V × T² × P) where V=violations, T=talks, P=participants
- In practice much faster due to early filtering of infeasible candidates

### Example Output

```
SWAP OPTIMIZATION: RESOLVING SCHEDULE VIOLATIONS
==================================================
  Presenter violations: 2
    - T042 (P042) @ TA
    - T089 (P089) @ FB

  Dummy violations: 3
    - DUMMY_001 @ FA (short block)
    - DUMMY_002 @ FA (multi-dummy)
    - DUMMY_003 @ FB (short block)

  Swap 1 [DUMMY]: DUMMY_001 <-> T071 (Δmissed=0, Δkeyword=+0.00)
  Swap 2 [PRES]: T042 <-> T098 (Δmissed=-2, Δkeyword=+0.15)
  Swap 3: T089 <-> T023 (Δmissed=0, Δkeyword=+0.10)

  Summary:
    Initial violations: 5
    Final violations:   0
    Swaps applied:      3
```

---

## Phase 1 — Non-Configurable Parameters

These parameters exist in `src/phase1.py` but are not exposed via config file. They use sensible defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_heuristic_filter` | `True` | Enable adaptive cost filtering for large instances (auto-activates when >1M tuples estimated) |

---

## Evaluator — Complexity

| Metric | Complexity |
|--------|------------|
| Missed Attendance | O(P × T) where P = participants, T = timeslots |
| Session Hops | O(P × B × k × n²) where B = blocks, k = slots/block, n = rooms |
| Incoherent Sessions | O(S × k × K) where S = room-sessions, K = avg keywords/talk |
| Presenter Violations | O(T) where T = total talks |

---

## Matching Pipeline — Available Methods (Not User-Configurable)

These methods exist in the code but are hardcoded to use optimal (MILP) variants:

### Phase A: Pair Matching

| Method | Description | Solver |
|--------|-------------|--------|
| `milp` | Maximum weight matching via MIP | Gurobi |
| `greedy` | Greedy by edge weight | None |
| `blossom` | Blossom algorithm | networkx |

### Phase B: Block Formation

| Method | Description | Solver |
|--------|-------------|--------|
| `joint_milp` | Joint optimization | Gurobi |
| `sequential` | Two-stage matching | scipy |

### Phase C: Tuple Selection

| Method | Description | Solver |
|--------|-------------|--------|
| `milp` | Set partitioning | Gurobi |
| `greedy` | Greedy construction | None |
