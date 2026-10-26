# Examples

This directory contains example data for the conference scheduling system.

## Datasets

### `orbel2017/`
Anonymized data from ORBEL 2017 conference (80 talks, 104 participants).

### `orbel2026/`
Anonymized data from ORBEL 2026 conference (118 talks, 99 participants).
Includes keyword topics for clustering (also anonymized/randomized).

### `synthetic/`
Small synthetic dataset (16 talks, 8 participants) for quick testing.

## Data Format

All datasets use a simple, standardized format:

### `talks.csv`
```
talk_id,presenter_id,keywords
1,1,"optimization; scheduling"
2,2,"machine learning"
```
`keywords` is optional. Further optional columns: `title`, `presenter_name`, `authors`, `special_group` (see the main README).

### `preferences.csv` (long format)
```
participant_id,talk_id
1,5
1,12
2,3
```

### `sessions.csv` (required)
```
session_id,n_rooms,n_talks_per_room
TA,5,4
TB,5,4
TC,4,3
```
Each row defines one session block (timeslot): `n_rooms` parallel rooms, each with `n_talks_per_room` sequential talks. The total slots (`sum of n_rooms × n_talks_per_room`) must be at least the number of talks; leftover slots are filled with placeholders (empty slots).

### `availability.csv` (optional)
```
presenter_id,unavailable_timeslot
P001,TB
P002,TA
```
Specifies timeslots where a presenter is unavailable. Presenter IDs refer to the `presenter_id` column of `talks.csv`, either as integers (`1`) or in internal form (`P001`). All pipelines take these constraints into account.

Participant IDs are inferred automatically from `preferences.csv` — no separate participants file is needed.

## Loading Data

```bash
# Load and validate a data directory from the command line
python examples/load_data.py examples/orbel2026
```

```python
# Or use programmatically
import sys
sys.path.insert(0, ".")
from examples.load_data import load_conference_data
from src.instance import build_instance

data, stats = load_conference_data("examples/orbel2026")
instance = build_instance(data)
```

## Command Line

```bash
# Load and validate
python examples/load_data.py examples/orbel2026

# Run the scheduler (heuristic pipeline, no Gurobi needed)
python scripts/run_schedule.py --pipeline heuristic --data-dir examples/orbel2026
```

## Creating Your Own Data

1. Create a directory with `talks.csv`, `preferences.csv`, and `sessions.csv`
2. Use numeric IDs (1, 2, 3, ...) for talks, presenters and participants
3. Keywords are optional but enable the coherence metric and keyword constraints
4. Check it with `python examples/load_data.py path/to/my_conference`

Minimal example:
```
my_conference/
├── talks.csv          # talk_id, presenter_id, keywords
├── preferences.csv    # participant_id, talk_id
└── sessions.csv       # session_id, n_rooms, n_talks_per_room
```
