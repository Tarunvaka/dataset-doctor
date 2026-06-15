# Mode: leakage  (the core mode)

Goal: determine whether any feature carries information about the target that would
not be available at prediction time. This is the single most common cause of models
that score 0.99 in validation and fail in the real world.

## Steps
1. Read config/profile.yml for the task, target, known_safe columns, and domain notes.
2. Run `scripts/leakage_check.py` with the right --task. Pass --test if a held-out file exists.
3. Read the JSON. For each finding, decide if it is real leakage or genuine signal,
   using the profile notes and the column's meaning. A feature flagged for high single-
   feature power is leakage if it could only be known after the outcome; it is just a
   strong predictor if it is legitimately available beforehand.
4. Rank: train/test overlap and post-outcome features are high severity. ID-like columns
   are medium (memorisation risk). Suspicious names are leads to confirm, not verdicts.

## Reasoning you must add on top of the script
- The script cannot know your timeline. You must ask, for each suspect: "Would this value
  exist at the moment of prediction?" If no, it is leakage regardless of statistics.
- A perfect separator can be real (e.g. a deterministic business rule). Flag it as a
  question, not a conviction.
- Always end by naming what these heuristics do NOT catch: subtle temporal leakage,
  group leakage (same entity in train and test under different rows), and leakage through
  preprocessing fit on the full dataset.
