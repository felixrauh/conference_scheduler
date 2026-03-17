# ORBEL 2026 Anonymized Data

Anonymized conference scheduling data in standardized format.

## Files

- `talks.csv` - Talk metadata
  - `talk_id`: Numeric talk identifier (original paper ID)
  - `presenter_id`: Numeric presenter identifier
  - `keywords`: Topic keywords for clustering

- `preferences.csv` - Attendee preferences (long format)
  - `participant_id`: Numeric participant identifier
  - `talk_id`: Talk the participant wants to attend

- `sessions.csv` - Conference structure (rooms × talks per session block)

## Statistics
- 118 talks
- 99 participants
- 1358 total preferences
- Average 13.7 preferences per participant
