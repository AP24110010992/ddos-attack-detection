"""
Compare form-compatible (compact) feature sets with the full 62-feature models.
Validation INSIDE the training day only; the test day (03-11) is NOT used.

Also checks two facts the form design depends on:
  1) Are the rate columns exactly total / duration? (this reveals the unit
     of Flow Duration)
  2) How often are the SYN / ACK flag counts non-zero?

Usage (from the project root):
    python src\\compare_compact.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from compact_features import (
    ALL_COMPACT_COLUMNS, COMPACT_FEATURES, compact_from_raw,
)
from data_processing import TARGET, TRAIN_DAY, build_preprocessor, load_sample
from evaluate import compute_metrics, make_time_blocks, summarize_folds

ROOT = Path(__file__).resolve().parent.parent
FULL_PATH = ROOT / "artifacts" / "model_validation.json"
BASE_PATH = ROOT / "artifacts" / "baseline_validation.json"
OUT_PATH = ROOT / "artifacts" / "compact_validation.json"
N_FOLDS = 5
SEED = 42
UNITS = {"microseconds": 1e6, "milliseconds": 1e3, "seconds": 1.0}


def make_models():
    """name -> (factory, needs_scaling). Names match model_validation.json."""
    return {
        "Decision Tree (depth 4)": (
            lambda: DecisionTreeClassifier(
                max_depth=4, class_weight="balanced", random_state=SEED), False),
        "Decision Tree (depth 10)": (
            lambda: DecisionTreeClassifier(
                max_depth=10, class_weight="balanced", random_state=SEED), False),
        "Random Forest (200 trees)": (
            lambda: RandomForestClassifier(
                n_estimators=200, class_weight="balanced", n_jobs=-1,
                random_state=SEED), False),
        "Logistic Regression (scaled)": (
            lambda: LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=SEED), True),
    }


def check_rate_formulas(compact):
    """Share of rows where the dataset's rate equals total / (duration in a unit)."""
    positive = compact["Flow Duration"] > 0
    out = {}
    for rate_col, total_col in (("Flow Packets/s", "Total Packets"),
                                ("Flow Bytes/s", "Total Bytes")):
        reported = compact.loc[positive, rate_col]
        for unit, per_second in UNITS.items():
            seconds = compact.loc[positive, "Flow Duration"] / per_second
            recomputed = compact.loc[positive, total_col] / seconds
            usable = np.isfinite(reported) & np.isfinite(recomputed)
            match = np.isclose(reported[usable], recomputed[usable],
                               rtol=1e-3, atol=0.0)
            out[(rate_col, unit)] = (int(match.sum()), int(usable.sum()))
    return out


def evaluate_schema(X, y, groups, models):
    results = {name: [] for name in models}
    for tr, va in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        for name, (factory, scale) in models.items():
            pipe = Pipeline([
                ("pre", build_preprocessor(scale=scale)),
                ("model", factory()),
            ])
            pipe.fit(X.iloc[tr], y[tr])
            results[name].append(compute_metrics(
                y[va], pipe.predict(X.iloc[va]),
                pipe.predict_proba(X.iloc[va])[:, 1]))
    return {name: summarize_folds(rows) for name, rows in results.items()}


def header():
    return (f"  {'Model':<44}{'Acc':>7}{'Prec':>7}{'Recall':>8}{'FPR':>7}"
            f"{'F1':>7}{'ValBA':>7}{'AUC':>7}{'BAstd':>7}")


def line(label, s):
    return (f"  {label:<44}{s['accuracy']:>7.3f}{s['precision']:>7.3f}"
            f"{s['recall']:>8.3f}{s['fpr']:>7.3f}{s['f1']:>7.3f}"
            f"{s['balanced_accuracy']:>7.3f}{s['roc_auc']:>7.3f}"
            f"{s['balanced_accuracy_std']:>7.3f}")


def main():
    df = load_sample()
    train = df[df["day"] == TRAIN_DAY].reset_index(drop=True)
    y = train[TARGET].to_numpy()
    groups = make_time_blocks(train, N_FOLDS).to_numpy()
    compact = compact_from_raw(train)
    print(f"Training day {TRAIN_DAY}: {len(train):,} rows. "
          f"{N_FOLDS}-fold time-blocked validation.")

    # 1) rate formulas and unit of Flow Duration
    rate_check = check_rate_formulas(compact)
    print("\n1) Do the rate columns equal total / duration? "
          "(rows with duration > 0)")
    for (col, unit), (m, n) in rate_check.items():
        share = 100 * m / n if n else 0.0
        print(f"   {col:<15} if duration is in {unit:<13}: "
              f"{m:>6,} of {n:>6,} rows match ({share:.1f}%)")

    # 2) flag counts
    cls = train[TARGET].map({0: "BENIGN", 1: "ATTACK"}).rename("class")
    print("\n2) Flag counts by class (rows per value)")
    for col in ("SYN Flag Count", "ACK Flag Count"):
        print(f"\n   {col}")
        print(pd.crosstab(train[col], cls).to_string())

    # 3) model comparison
    models = make_models()
    schemas = {
        "compact-7 (no SYN)": COMPACT_FEATURES,
        "compact-8 (with SYN)": ALL_COMPACT_COLUMNS,
    }
    summaries = {}
    for schema_name, cols in schemas.items():
        print(f"\nrunning {schema_name} ...", flush=True)
        summaries[schema_name] = evaluate_schema(compact[cols], y, groups, models)

    full = {}
    if FULL_PATH.exists():
        with open(FULL_PATH, encoding="utf-8") as f:
            full = json.load(f)["models"]

    print("\n3) Validation inside the training day (mean over folds)")
    print(header())
    if BASE_PATH.exists():
        with open(BASE_PATH, encoding="utf-8") as f:
            base = json.load(f)
        print(line("Baseline: majority class", base["majority_class"]))
        print(line("Baseline: best one-feature stump",
                   base["stumps_ranked"][0]))
    for name in models:
        print(f"\n  {name}")
        if name in full:
            print(line("   full schema, 62 features", full[name]))
        for schema_name in schemas:
            print(line(f"   {schema_name}", summaries[schema_name][name]))

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "validation": f"{N_FOLDS}-fold time-blocked, inside training day {TRAIN_DAY}",
            "rate_formula_check": {
                f"{c}|{u}": {"matching": m, "checked": n}
                for (c, u), (m, n) in rate_check.items()
            },
            "schemas": {
                name: {"features": cols, "models": summaries[name]}
                for name, cols in schemas.items()
            },
        }, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()