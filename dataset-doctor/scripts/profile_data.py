#!/usr/bin/env python3
"""
profile_data.py - Diagnose data-quality problems in a tabular dataset.

Produces a JSON diagnosis the agent layer reads and explains in plain language.
This covers the problems that wreck models even when there is no leakage:
missingness, skew, outliers, constant columns, high-cardinality categoricals,
class imbalance, and duplicate rows.

Usage:
    python profile_data.py --data data.csv --target churn --task classification
    python profile_data.py --data data.parquet            # target/task optional
"""

import argparse
import json
import sys
import warnings

warnings.filterwarnings("ignore")


def _load(path):
    import pandas as pd
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith((".tsv", ".tab")):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _missingness(df):
    out = []
    n = len(df)
    miss = df.isna().sum()
    for col, m in miss.items():
        if m == 0:
            continue
        frac = m / n
        note = "high - consider whether the column is usable" if frac > 0.4 else \
               "moderate - needs an imputation or drop decision" if frac > 0.05 else "low"
        out.append({"column": col, "missing": int(m), "fraction": round(float(frac), 4),
                    "note": note})
    return sorted(out, key=lambda x: -x["fraction"])


def _constant_or_near(df):
    out = []
    n = len(df)
    for col in df.columns:
        nun = df[col].nunique(dropna=False)
        if nun <= 1:
            out.append({"column": col, "issue": "constant",
                        "why": "Single value - carries no information, drop it."})
        else:
            top_share = df[col].value_counts(normalize=True, dropna=False).iloc[0]
            if top_share > 0.99 and n > 50:
                out.append({"column": col, "issue": "near-constant",
                            "dominant_value_share": round(float(top_share), 4),
                            "why": "One value dominates; usually adds noise more than signal."})
    return out


def _high_cardinality(df, target):
    out = []
    n = len(df)
    cats = df.select_dtypes(include=["object", "category"]).columns
    for col in cats:
        if col == target:
            continue
        nun = df[col].nunique(dropna=True)
        if nun > 50 and nun / n < 0.95:
            out.append({"column": col, "unique_values": int(nun),
                        "why": "Many distinct categories. One-hot encoding will explode "
                               "dimensionality; consider target/frequency encoding or grouping."})
    return out


def _skew_outliers(df, target):
    import numpy as np
    out = []
    num = df.select_dtypes("number")
    for col in num.columns:
        if col == target:
            continue
        s = df[col].dropna()
        if len(s) < 20 or s.nunique() < 5:
            continue
        skew = float(s.skew())
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            outliers = ((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).mean()
        else:
            outliers = 0.0
        entry = {"column": col}
        flagged = False
        if abs(skew) > 2:
            entry["skew"] = round(skew, 3)
            entry["skew_note"] = "Heavily skewed; a log or power transform may help linear models."
            flagged = True
        if outliers > 0.01:
            entry["extreme_outlier_fraction"] = round(float(outliers), 4)
            entry["outlier_note"] = "Notable share of extreme values; check for sentinels (e.g. -999) or errors."
            flagged = True
        if flagged:
            out.append(entry)
    return out


def _class_balance(df, target, task):
    if task != "classification" or target not in df.columns:
        return None
    vc = df[target].value_counts(normalize=True)
    minority = float(vc.min())
    return {
        "class_fractions": {str(k): round(float(v), 4) for k, v in vc.items()},
        "minority_fraction": round(minority, 4),
        "note": "Severe imbalance - accuracy will mislead; use precision/recall/AUC and "
                "consider resampling or class weights." if minority < 0.1 else
                "Moderate imbalance - prefer AUC/F1 over raw accuracy." if minority < 0.3 else
                "Reasonably balanced.",
    }


def _duplicates(df):
    d = int(df.duplicated().sum())
    return {"duplicate_rows": d,
            "why": "Exact duplicate rows can bias training and inflate validation if "
                   "they straddle a split."} if d else None


def main():
    ap = argparse.ArgumentParser(description="Diagnose tabular data-quality issues.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", default=None)
    ap.add_argument("--task", choices=["classification", "regression"], default=None)
    args = ap.parse_args()

    try:
        df = _load(args.data)
    except Exception as e:
        print(json.dumps({"error": f"could not load data: {e}"}))
        sys.exit(1)

    report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "checks": {
            "missingness": _missingness(df),
            "constant_columns": _constant_or_near(df),
            "high_cardinality_categoricals": _high_cardinality(df, args.target),
            "skew_and_outliers": _skew_outliers(df, args.target),
            "class_balance": _class_balance(df, args.target, args.task),
            "duplicate_rows": _duplicates(df),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
