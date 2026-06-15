# Mode: diagnose

Goal: a complete data-quality read. Run BOTH scripts and synthesise.

## Steps
1. Read config/profile.yml if present.
2. Run `scripts/leakage_check.py` then `scripts/profile_data.py`.
3. If leakage_check returns any high-severity finding, switch to the leakage mode's
   reasoning and lead the whole report with it. Quality issues are secondary until
   leakage is resolved.
4. Otherwise, walk the quality findings in priority order: missingness that forces a
   drop/impute decision, severe class imbalance, then skew/outliers/cardinality.
5. For each issue, give the fix decision, not just the observation.

Follow the output contract in _shared.md.
