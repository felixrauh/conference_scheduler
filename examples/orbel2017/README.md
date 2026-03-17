# ORBEL 2017 Anonymized Data

Anonymized conference scheduling data in standardized format.

## Files

- `talks.csv` - Talk metadata
  - `talk_id`: Numeric talk identifier
  - `presenter_id`: Numeric presenter identifier  
  - `keywords`: Track/topic (COMEX field)

- `preferences.csv` - Attendee preferences (long format)
  - `participant_id`: Numeric participant identifier
  - `talk_id`: Talk the participant wants to attend

- `sessions.csv` - Conference structure (rooms × talks per session block)

## Statistics
- 80 talks
- 104 participants
- 1200 total preferences
- Average 11.5 preferences per participant
