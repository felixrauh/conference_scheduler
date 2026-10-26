# ORBEL 2017 Anonymized Data

Anonymized conference scheduling data in standardized format.

## Files

- `talks.csv` - Talk metadata
  - `talk_id`: Numeric talk identifier
  - `presenter_id`: Numeric presenter identifier  
  - `keywords`: Track/topic (COMEX field); empty for 46 of the 80 talks

- `preferences.csv` - Attendee preferences (long format)
  - `participant_id`: Numeric participant identifier
  - `talk_id`: Talk the participant wants to attend

- `sessions.csv` - Conference structure (rooms × talks per session block):
  5 blocks of 4 rooms × 4 talks = 80 slots, so no slots stay empty

There is no `availability.csv` for this dataset.

Because most talks have no keyword, the incoherent-sessions metric is high for every pipeline, and the hard keyword constraint of `matching_constrained` (`--require-keyword`) is infeasible; use `--max-violations 0` instead.

## Statistics
- 80 talks
- 104 participants
- 1200 total preferences
- Average 11.5 preferences per participant
