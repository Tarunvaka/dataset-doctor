#!/usr/bin/env python3
"""
leakage_check.py - Detect data leakage in a tabular dataset before training.

This is the core differentiator of dataset-doctor. It runs a battery of checks
that catch the most common reason ML projects silently fail: information from the
target (or from the future) bleeding into the features, producing models that look
great in validation and collapse in production.

Output is JSON on stdout so the agent layer can read and reason about it.

Usage:
    python leakage_check.py --data data.csv --target churn --task classification
    python leakage_check.py --data data.parquet --target price --task regression \
        --test data_test.csv

Checks performed:
    1. Single-feature predictive power   - any one column that nearly predicts the
                                           target alone is a prime leakage suspect.
    2. Target correlation                - numeric features with near-perfect
                                           correlation to the target.
    3. Train/test row overlap            - identical rows appearing in both splits
                                           (contamination), if --test is provided.
    4. Suspicious column names           - columns whose names imply they are
                                           recorded at or after the outcome.
    5. Identifier-like columns           - near-unique columns that can let a model
                                           memorise rows.
    6. Perfect separators                - categorical values that map 1:1 to a class.
"""

import argparse
import json
import sys
import warnings

warnings.filterwarnings("ignore")

# Column-name tokens that frequently indicate post-outcome / future information.
SUSPICIOUS_TOKENS = [
    "outcome", "result", "label", "target", "score", "rank", "final",
    "post", "after", "future", "next", "resolved", "closed", "settled",
    "refund", "chargeback", "repaid", "default", "fraud_flag", "is_fraud",
    "churned", "cancelled", "returned", "approved", "rejected", "decision",
    "predicted", "prediction", "y_true", "y_pred", "actual",
]


def _load(path):
    import pandas as pd
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith((".tsv", ".tab")):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _single_feature_power(df, target, task):
    """Fit a trivial model on each feature alone; flag any that score implausibly high."""
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.model_selection import cross_val_score

    findings = []
    y = df[target]
    if task == "classification":
        y = LabelEncoder().fit_transform(y.astype(str))

    for col in df.columns:
        if col == target:
            continue
        x = df[[col]].copy()
        try:
            if x[col].dtype == object or str(x[col].dtype).startswith("category"):
                x[col] = LabelEncoder().fit_transform(x[col].astype(str))
            x = x.fillna(x.median(numeric_only=True))
            if x[col].nunique() < 2:
                continue
            if task == "classification":
                model = DecisionTreeClassifier(max_depth=3, random_state=0)
                scoring = "accuracy"
                # baseline = majority class rate
                baseline = pd.Series(y).value_counts(normalize=True).max()
            else:
                model = DecisionTreeRegressor(max_depth=3, random_state=0)
                scoring = "r2"
                baseline = 0.0
            score = float(np.mean(cross_val_score(model, x, y, cv=3, scoring=scoring)))
            # flag features that single-handedly explain the target far beyond baseline.
            # use "error reduction": fraction of the gap from baseline to perfect that
            # this one feature closes. robust to class imbalance (high baselines).
            if task == "classification":
                gap = 1.0 - baseline
                err_reduction = (score - baseline) / gap if gap > 1e-9 else 0.0
                if score > 0.99 or (score > 0.9 and err_reduction > 0.9):
                    findings.append({
                        "column": col, "metric": "accuracy", "score": round(score, 4),
                        "baseline": round(float(baseline), 4),
                        "error_reduction": round(float(err_reduction), 4),
                        "why": "A single feature predicts the target almost perfectly. "
                               "Real signal is rarely this strong; this usually means the "
                               "column encodes the answer.",
                    })
            elif task == "regression" and score > 0.97:
                findings.append({
                    "column": col, "metric": "r2", "score": round(score, 4),
                    "why": "A single feature explains nearly all variance in the target. "
                           "Check whether it is derived from or recorded after the target.",
                })
        except Exception:
            continue
    return findings


def _target_correlation(df, target, task):
    import numpy as np
    if task != "regression":
        return []
    num = df.select_dtypes("number")
    if target not in num.columns:
        return []
    corr = num.corr(numeric_only=True)[target].drop(target).abs()
    out = []
    for col, c in corr.items():
        if c > 0.95:
            out.append({"column": col, "abs_correlation": round(float(c), 4),
                        "why": "Almost perfectly correlated with the target. Likely a "
                               "transformed copy or a post-outcome measurement."})
    return out


def _train_test_overlap(df, test_df):
    if test_df is None:
        return None
    merged = df.merge(test_df, how="inner")
    n = len(merged)
    return {
        "identical_rows_in_both": int(n),
        "share_of_test": round(n / max(len(test_df), 1), 4),
        "why": "Rows appear in both train and test. Validation scores will be "
               "optimistic because the model has already seen these examples."
    } if n > 0 else None


def _suspicious_names(df, target):
    out = []
    for col in df.columns:
        if col == target:
            continue
        low = col.lower()
        hits = [t for t in SUSPICIOUS_TOKENS if t in low]
        if hits:
            out.append({"column": col, "matched_tokens": hits,
                        "why": "Column name suggests it may be recorded at or after the "
                               "outcome you are predicting. Confirm it would be available "
                               "at prediction time."})
    return out


def _identifier_like(df, target):
    import pandas as pd
    out = []
    n = len(df)
    for col in df.columns:
        if col == target:
            continue
        # continuous floats are naturally near-unique and are not identifiers
        if pd.api.types.is_float_dtype(df[col]):
            continue
        nun = df[col].nunique(dropna=True)
        if n > 20 and nun / n > 0.95:
            out.append({"column": col, "unique_ratio": round(nun / n, 4),
                        "why": "Nearly unique per row (looks like an ID). A flexible model "
                               "can memorise rows through it. Usually should be dropped."})
    return out


def _perfect_separators(df, target, task):
    if task != "classification":
        return []
    out = []
    cats = df.select_dtypes(include=["object", "category"]).columns
    for col in cats:
        if col == target or df[col].nunique() > 50:
            continue
        grp = df.groupby(col)[target].nunique()
        pure = (grp == 1).mean()
        if pure > 0.9 and df[col].nunique() > 1:
            out.append({"column": col, "pure_value_share": round(float(pure), 4),
                        "why": "Most values of this category map to exactly one class. "
                               "Often a leak, sometimes a genuinely strong signal - verify."})
    return out


def main():
    ap = argparse.ArgumentParser(description="Detect data leakage in a tabular dataset.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--task", choices=["classification", "regression"], required=True)
    ap.add_argument("--test", default=None, help="optional held-out file to check overlap")
    args = ap.parse_args()

    try:
        df = _load(args.data)
    except Exception as e:
        print(json.dumps({"error": f"could not load data: {e}"}))
        sys.exit(1)

    if args.target not in df.columns:
        print(json.dumps({"error": f"target '{args.target}' not in columns"}))
        sys.exit(1)

    test_df = _load(args.test) if args.test else None

    report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "target": args.target,
        "task": args.task,
        "checks": {
            "single_feature_predictive_power": _single_feature_power(df, args.target, args.task),
            "target_correlation": _target_correlation(df, args.target, args.task),
            "train_test_overlap": _train_test_overlap(df, test_df),
            "suspicious_column_names": _suspicious_names(df, args.target),
            "identifier_like_columns": _identifier_like(df, args.target),
            "perfect_separators": _perfect_separators(df, args.target, args.task),
        },
    }

    # crude severity roll-up so the agent can lead with the worst thing
    high = (report["checks"]["single_feature_predictive_power"]
            + report["checks"]["target_correlation"]
            + ([report["checks"]["train_test_overlap"]]
               if report["checks"]["train_test_overlap"] else []))
    report["summary"] = {
        "high_severity_findings": len(high),
        "verdict": "LIKELY LEAKAGE - do not trust validation scores until resolved"
        if high else "No high-severity leakage detected by these heuristics",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
