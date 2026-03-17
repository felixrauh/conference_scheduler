# Phase 1: Maximize Total Attendance

## Problem Statement

Given participant preferences, determine which talks should run in parallel to **minimize missed attendance**.

A participant "misses" attendance when two or more of their preferred talks are scheduled at the same time (in parallel). Since they can only attend one, the others are missed.

---

## Mathematical Formulation

### Original Paper Formulation (Fixed $n$)

The paper assumes a **fixed number of parallel sessions $n$** across the entire conference.

**Sets:**
- $X$: Set of all talks
- $P$: Set of all participants  
- $H \subseteq X^n$: Set of all feasible n-tuples of distinct talks

**Parameters:**
- $q(p)_i \in \{0, 1\}$: 1 if participant $p$ wants to attend talk $i$
- $c_e$: Missed attendance cost for n-tuple $e \in H$ 

$$c_e = \sum_{p \in P} \max\left\{0, \sum_{i \in e} q(p)_i - 1\right\}$$
(i.e each participant contributes +k for missed attendance if they have k+1 preferred talks in the tuple)

**Decision Variables:**
- $x_e \in \{0, 1\}$: 1 if n-tuple $e$ is selected

**Model:**
$$\min \sum_{e \in H} c_e \cdot x_e$$

Subject to:
$$\sum_{e \in H: i \in e} x_e = 1 \quad \forall i \in X$$
$$x_e \in \{0, 1\} \quad \forall e \in H$$

---

## Extended Formulation (Variable $n$)

Our extension supports **different numbers of parallel sessions** across different parts of the conference.

### Additional Sets

- $\mathcal{B}$: Set of block types, each with $(n_\tau, k_\tau, r_\tau)$
  - $n_\tau$: Number of parallel sessions (rooms)
  - $k_\tau$: Number of talks per session
  - $r_\tau$: Number of blocks of this type

- $\mathcal{T}$: Set of tuple types, each with $(n_\tau, p_\tau)$, derived from $\mathcal{B}$ by aggregating over blocks with the same $n_\tau$:
  - $n_\tau$: Tuple size (number of parallel sessions)
  - $p_\tau$: Number of tuples of this type needed:
  $$p_\tau = \sum_{\beta \in \mathcal{B}: n_\beta = n_\tau} r_\beta \cdot k_\beta$$

- $H_\tau \subseteq X^{n_\tau}$: Set of feasible $n_\tau$-tuples (tuples of size $n_\tau$)
- $H = \bigcup_{\tau \in \mathcal{T}} H_\tau$: Set of all feasible tuples across all sizes

### Extended Model

**Decision Variables:**
- $x_e \in \{0, 1\}$: 1 if tuple $e \in H$ is selected

**Model:**
$$\min \sum_{e \in H} c_e \cdot x_e$$

Subject to:

1. **Coverage constraint** — each talk appears in exactly one selected tuple:
$$\sum_{e \in H: i \in e} x_e = 1 \quad \forall i \in X$$

2. **Tuple count constraint** — for each tuple type, select exactly the required number:
$$\sum_{e \in H_\tau} x_e = p_\tau \quad \forall \tau \in \mathcal{T}$$

3. **Binary constraints:**
$$x_e \in \{0, 1\} \quad \forall e \in H$$

### Notes

- The coverage constraint ensures every talk is scheduled exactly once.
- The tuple count constraint ensures the right number of parallel sessions of each size.
- Together, these constraints partition talks into tuples of varying sizes according to the conference format.
- The model is **unified**: no need to pre-assign talks to block types. The optimization decides which talks go into which tuple size.

### Feasibility Condition

For the model to be feasible, the total number of slots must equal the number of talks:
$$|X| = \sum_{\tau \in \mathcal{T}} n_\tau \cdot p_\tau = \sum_{\beta \in \mathcal{B}} n_\beta \cdot k_\beta \cdot r_\beta$$

---

## Data Types

### Input Data Structures

```python
@dataclass
class ConferenceData:
    """Container for all conference input data."""
    
    conference_name: str
    rooms: List[str]
    
    # Block types: {type_id: {"n": int, "k": int, "count": int}}
    block_types: Dict[str, Dict]
    
    # Timeslots: list of {"id", "start_time", "type_id", "rooms"}
    timeslots: List[Dict]
    
    # Talks: DataFrame [talk_id, title, presenter_id, track]
    talks: pd.DataFrame
    
    # Participants: DataFrame [participant_id, name, email]
    participants: pd.DataFrame
    
    # Preferences: DataFrame [participant_id, talk_id]
    preferences: pd.DataFrame
    
    # Availability: DataFrame [presenter_id, unavailable_timeslot]
    availability: pd.DataFrame
    
    # Derived: {participant_id: set of preferred talk_ids}
    preference_matrix: Dict[str, Set[str]]
```

### Problem Instance

```python
@dataclass
class ProblemInstance:
    """Optimization-ready problem instance."""
    
    conference_data: ConferenceData
    
    talks: List[str]              # List of talk_ids
    participants: List[str]        # List of participant_ids
    block_types: Dict[str, Dict]   # Block type definitions
    
    # {participant_id: set of preferred talk_ids}
    preferences: Dict[str, Set[str]]
    
    # {talk_id: presenter_id}
    talk_presenter: Dict[str, str]
    
    # {presenter_id: set of unavailable timeslot_ids}
    presenter_unavailability: Dict[str, Set[str]]
    
    # {type_id: list of timeslot dicts}
    timeslots_by_type: Dict[str, List[Dict]]
```

### Phase 1 Output

```python
# For fixed-n approach: List of selected n-tuples
Phase1Result = List[Tuple[str, ...]]

# For variable-n approach: Mapping from block type to n-tuples
Phase1ResultByType = Dict[str, List[Tuple[str, ...]]]
```

### N-Tuple Representation

```python
# An n-tuple is a tuple of talk_ids that run in parallel
NTuple = Tuple[str, ...]  # e.g., ("T001", "T002", "T003", "T004")

# Cost coefficient for an n-tuple
def compute_tuple_cost(ntuple: NTuple, preferences: Dict[str, Set[str]]) -> int:
    """
    c_e = sum over participants of max(0, preferred_in_tuple - 1)
    """
    cost = 0
    for p_id, prefs in preferences.items():
        preferred_count = sum(1 for t in ntuple if t in prefs)
        if preferred_count > 1:
            cost += preferred_count - 1
    return cost
```

---

## Filtering Infeasible Tuples

To reduce the size of $H$, filter out n-tuples that violate hard constraints:

### Filter Functions

```python
def filter_same_presenter(ntuple: NTuple, talk_presenter: Dict[str, str]) -> bool:
    """Reject if any two talks have the same presenter."""
    presenters = [talk_presenter[t] for t in ntuple]
    return len(presenters) == len(set(presenters))

def filter_track_constraints(ntuple: NTuple, talks_df: pd.DataFrame, max_per_track: int) -> bool:
    """Reject if too many talks from same track in parallel."""
    tracks = [talks_df.loc[talks_df['talk_id'] == t, 'track'].iloc[0] for t in ntuple]
    from collections import Counter
    return all(count <= max_per_track for count in Counter(tracks).values())

def filter_must_not_overlap(ntuple: NTuple, forbidden_pairs: Set[FrozenSet[str]]) -> bool:
    """Reject if any forbidden pair of talks are both in tuple."""
    for t1, t2 in combinations(ntuple, 2):
        if frozenset({t1, t2}) in forbidden_pairs:
            return False
    return True
```

---

## Example

```python
from src.data_loader import load_from_csv_dir
from src.instance import build_instance
from src.phase1 import solve_phase1

# Load data
data = load_from_csv_dir("examples/orbel2026")
data.validate()
instance = build_instance(data)

# Solve the unified model
selected_tuples = solve_phase1(
    instance=instance,
    time_limit=300.0,
    verbose=True
)

# Result: tuples of varying sizes
for t in selected_tuples:
    print(f"Parallel ({len(t)} rooms): {t}")
```

---

## References

- Vangerven et al. (2018), Section 5.1: "Phase 1: maximizing total attendance"
- Equations (1)-(3) in the paper
