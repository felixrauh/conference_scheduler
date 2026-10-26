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

If `sessions.csv` provides more slots than there are talks, the pipelines add placeholder talks (`PLACEHOLDER_<n>`, no preferences, so cost 0) to $X$ until equality holds. More talks than slots is rejected before optimization.

---

## Data Types

### Input Data Structures

Defined in `src/data_loader.py` (abbreviated):

```python
@dataclass
class ConferenceData:
    """Container for all conference input data."""

    conference_name: str
    rooms: List[str]              # Room_1 ... Room_<max rooms>

    # Block types, merged by shape: {"5R4T": {"n": 5, "k": 4, "count": 4}, ...}
    block_types: Dict[str, Dict]

    # One timeslot per row of sessions.csv: {"id": "TA", "type_id": "5R4T", "rooms": [...]}
    timeslots: List[Dict]

    # Talks: DataFrame [talk_id, presenter_id, title, keywords, ...]
    talks: pd.DataFrame

    # Preferences: DataFrame [participant_id, talk_id] (participants are derived from it)
    preferences: pd.DataFrame

    # Availability: DataFrame [presenter_id, unavailable_timeslot]
    availability: pd.DataFrame

    # Derived by validate(): {participant_id: set of preferred talk_ids}
    preference_matrix: Dict[str, Set[str]]
    # Derived by validate(): {presenter_id: set of unavailable timeslot_ids}
    presenter_unavailability: Dict[str, Set[str]]
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
# List of selected tuples (of varying sizes)
Phase1Result = List[Tuple[str, ...]]

# The same tuples grouped by tuple size n (input format of Phase 2)
Phase1ResultByType = Dict[int, List[Tuple[str, ...]]]
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

To reduce the size of $H$, the explicit method (`solve_phase1_explicit` in `src/phase1.py`) only generates tuples that pass these filters:

1. **Same presenter** — no two talks by the same presenter in one tuple (`ProblemInstance.talks_have_same_presenter`).
2. **Presenter availability** — no pair of talks whose presenters' combined unavailabilities block too many timeslots (`compute_infeasible_talk_pairs` in `src/instance.py`; see [developer_notes.md](developer_notes.md#infeasible-talk-pairs)).
3. **Heuristic filters** — only when more than ~1M tuples would be generated:
   - tuples containing a *popular pair* (two talks many participants want to see together) are dropped;
   - tuples containing two or more preferred talks are kept only if their cost is at most `max_cost` (config `phase1.explicit.max_cost`; `null` = derived from a sample of the cost distribution);
   - talks left with fewer than 50 candidate tuples get the cheapest rejected tuples back.

   These filters can exclude the optimal solution, so the result is then no longer provably optimal.

The column generation method (default) does not enumerate $H$; see [column_generation.md](column_generation.md).

---

## Example

```python
from src.data_loader import load_from_csv_dir
from src.instance import build_instance
from src.phase1 import solve_phase1

# Load data
data = load_from_csv_dir("examples/synthetic")
data.validate()
instance = build_instance(data)

# Solve the unified model (explicit enumeration by default)
selected_tuples = solve_phase1(instance, time_limit=60.0, verbose=False)

# Result: tuples of varying sizes
for t in selected_tuples:
    print(f"Parallel ({len(t)} rooms): {t}")
```

For larger instances (50+ talks) use `method="column_generation"`; explicit enumeration of e.g. the 118-talk ORBEL 2026 data would need about 182M variables.

---

## References

- Vangerven et al. (2018), Section 5.1: "Phase 1: maximizing total attendance"
- Equations (1)-(3) in the paper
