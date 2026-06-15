# dataset-doctor

**Catch the data leak before it ships.** dataset-doctor is an agent that diagnoses a
tabular dataset *before* you train on it — and its main job is finding the leakage that
makes a model score 0.99 in validation and then fall apart in production.

It runs on the open [Agent Skills](https://github.com/agentskills/agentskills) format,
so it works in Claude Code, Cursor, Codex CLI, Gemini CLI, and anything else that speaks
the standard. Four modes: `diagnose`, `leakage`, `fix`, `report`.

```
You:  diagnose examples/customers.csv, target=churn

dataset-doctor:
  VERDICT: LIKELY LEAKAGE — do not trust validation scores yet.

  HIGH SEVERITY
  • refund_issued  — a single feature predicts churn almost perfectly
                     (cv accuracy 1.00, baseline 0.73). This is only known
                     AFTER a customer churns. It is leakage. Drop it.

  MEDIUM
  • customer_id    — unique per row; a model can memorise through it. Drop.

  NOT CHECKED
  • temporal/group leakage, preprocessing-time leakage. See limits below.
```

## Why this exists

Every data-cleaning library will hand you a wall of statistics. None of them know that
`refund_issued` is filled in *after* the thing you're predicting. dataset-doctor reads
your dataset **and your context** (`config/profile.yml`) and reasons about whether a
column would actually exist at prediction time. The scripts produce evidence; the agent
does the judgment. That's the difference between a profiler and a second pair of eyes.

Data leakage is the most common reason student and junior ML projects silently fail, and
there is no good agentic tool aimed squarely at it. That's the gap this fills.

## Quickstart

```bash
git clone https://github.com/<you>/dataset-doctor
cd dataset-doctor
pip install -r scripts/requirements.txt
cp config/profile.example.yml config/profile.yml   # edit with your task + target
```

Then, in any skills-compatible agent, point it at the repo and say:

```
diagnose path/to/your.csv, target=<column>, task=classification
```

Or run the engine directly, no agent required:

```bash
python scripts/leakage_check.py --data your.csv --target churn --task classification
python scripts/profile_data.py  --data your.csv --target churn --task classification
```

## What it checks

**Leakage (`leakage_check.py`)**
- Single-feature predictive power (a column that alone nearly predicts the target)
- Near-perfect target correlation (regression)
- Train/test row overlap, when you pass a `--test` file
- Suspicious column names (post-outcome / future-information tokens)
- Identifier-like columns a model can memorise through
- Perfect class separators among categoricals

**Quality (`profile_data.py`)**
- Missingness with drop/impute guidance
- Constant and near-constant columns
- High-cardinality categoricals
- Skew and extreme outliers (sentinel detection)
- Class imbalance with the right metric to use
- Duplicate rows

## Limits (read this)

These are **tabular heuristics, not proof**. dataset-doctor does **not** catch:
temporal leakage in time-series, group leakage (the same entity split across train and
test), or leakage introduced when preprocessing is fit on the whole dataset instead of
the training fold. Absence of a flag is not a guarantee of safety. It is a sharp first
pass that catches the common, expensive mistakes — treat it as a reviewer, not an oracle.

## Layout

```
dataset-doctor/
├── AGENTS.md              # canonical agent instructions
├── CLAUDE.md              # Claude Code wrapper (imports AGENTS.md)
├── config/profile.example.yml
├── modes/                 # diagnose / leakage / fix / report
├── scripts/               # the deterministic engine + requirements
└── examples/customers.csv # demo dataset with a planted leak
```

## License

MIT. Recommendations are not guarantees; you are responsible for what you ship.
