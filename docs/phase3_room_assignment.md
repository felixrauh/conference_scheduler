# Phase 3: Block-to-Timeslot Assignment & Room Allocation

## Overview

After Phase 2, we have **blocks** where each block groups k tuples. Within a block, position $r$ across all tuples forms a **room session** - all talks that will be in the same physical room.

Phase 3 does two things:
1. **Assign blocks to timeslots** (satisfying presenter availability)
2. **Assign room positions to physical rooms** (maximize capacity gap)

---

Room capacities are configured via the `rooms` section in the config file or inferred from input data. Phase 4 assigns rooms using a greedy algorithm (smallest audience to smallest room).

---

## Problem Statement

### Input
- **Blocks** from Phase 2: each block $B$ has $k$ tuples, each tuple has $n$ talks
- **Room sessions**: For block $B$, room position $r$ has talks $\{t_{1,r}, t_{2,r}, \ldots, t_{k,r}\}$
- **Presenter unavailability**: presenter $p$ unavailable at timeslot $s$
- **Room capacities**: $C_r$ for each physical room $r$
- **Participant preferences**: who wants to attend which talks

### Output
- Assignment of each block to a timeslot
- Assignment of each room position to a physical room

### Objectives
1. **Constraint**: Minimize/avoid presenter availability violations
2. **Objective**: Maximize capacity gap (room capacity - expected attendance)

---

## Room Session Interest

For a room session (all talks in room position $r$ of block $B$):

$$\text{Interest}(B, r) = |\{p : p \text{ likes at least one talk in room position } r \text{ of block } B\}|$$

This is the number of participants who would want to attend at least one talk in that room session.

```python
def compute_room_session_interest(
    block: Block,
    room_position: int,
    preferences: Dict[str, Set[str]]
) -> int:
    """
    Count participants interested in at least one talk 
    in this room position across all timeslots of the block.
    """
    # Get all talks in this room position
    room_talks = {block.tuples[slot][room_position] for slot in range(len(block.tuples))}
    
    # Count participants with at least one preference in these talks
    interested = sum(
        1 for prefs in preferences.values()
        if prefs & room_talks  # intersection not empty
    )
    return interested
```

---

## Room Assignment (Simple Greedy)

For each block independently:
1. Compute interest for each room position (0 to n-1)
2. Sort room positions by interest (descending)
3. Sort physical rooms by capacity (descending)
4. Match: highest interest → largest room

This is **optimal** for maximizing minimum capacity gap within each block.

### Example

Block B01 with 4 room positions:
- Position 0: 45 interested participants
- Position 1: 38 interested participants  
- Position 2: 28 interested participants
- Position 3: 22 interested participants

Physical rooms (sorted by capacity):
- Room A: 250
- Room B: 120
- Room C: 80
- Room D: 60

Assignment:
- Position 0 (45) → Room A (250): gap = 205
- Position 1 (38) → Room B (120): gap = 82
- Position 2 (28) → Room C (80): gap = 52
- Position 3 (22) → Room D (60): gap = 38

Minimum gap = 44 ✓

---

## Block-to-Timeslot Assignment

### Sets
- $B$: Set of blocks (from Phase 2)
- $S$: Set of timeslots (compatible with block types)

### Parameters
- $u_{b,s}$: Number of presenter availability violations if block $b$ is assigned to timeslot $s$

### Decision Variables
- $y_{b,s} \in \{0,1\}$: 1 if block $b$ assigned to timeslot $s$

### Formulation

$$\min \sum_{b \in B} \sum_{s \in S} u_{b,s} \cdot y_{b,s}$$

Subject to:
$$\sum_{s \in S} y_{b,s} = 1 \quad \forall b \in B \quad \text{(each block to one timeslot)}$$
$$\sum_{b \in B} y_{b,s} \leq 1 \quad \forall s \in S \quad \text{(each timeslot at most one block)}$$

---

## Complete Phase 3 Pipeline

```
Input: Blocks from Phase 2, presenter unavailability, room capacities

Step 1: Block-to-Timeslot Assignment
  - Compute violation costs u[b,s] for all block-timeslot pairs
  - Solve assignment IP (or use Hungarian algorithm)
  - Output: block → timeslot mapping

Step 2: Room Position to Physical Room Assignment  
  - For each block:
    a. Compute interest for each room position
    b. Greedy match: sort positions by interest, rooms by capacity
    c. Assign highest interest → largest room
  - Output: (block, position) → physical room mapping

Output: Complete schedule with talks assigned to (timeslot, room)
```

---

## Implementation Notes

### Violation Cost Computation

```python
def compute_violation_cost(block, timeslot, talk_presenter, presenter_unavailability):
    """Count presenters in block who are unavailable at timeslot."""
    violations = 0
    all_talks = [talk for ntuple in block.tuples for talk in ntuple]
    
    for talk_id in all_talks:
        presenter = talk_presenter.get(talk_id)
        if presenter and timeslot['id'] in presenter_unavailability.get(presenter, set()):
            violations += 1
    return violations
```

### Room Configuration

Room capacities and names are defined in the config file (`rooms` section) or via the `ConferenceData` object. See [configuration_and_algorithm_reference.md](configuration_and_algorithm_reference.md#phase-4-finalization--room-assignment) for configuration details.
