"""
Fill-strategy comparison for undefined rate values (NaN / infinity).

Flow Packets/s and Flow Bytes/s are undefined when Flow Duration is 0. This
script compares three ways of filling them, on the full schema and on the
compact 7-feature schema, using time-blocked validation INSIDE the training
day. The test day (03-11) is NOT used.

  median (current)       : training-fold median (what data_processing.py does)
  zero                   : fill with 0
  median + missing flag  : median fill plus a 0/1 column marking filled rows

Besides overall scores, it reports errors on the zero-duration rows
specifically, because those are the only rows the fill strategy can affect.

Usage (from the project root):
    python src\\compare_fill.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from compact_features import COMPACT_FEATURES, compact_from_raw
from data_processing import (
    TARGET, TRAIN_DAY, _replace_non_finite, load_sample, prepare_features,
)
from evaluate import compute_metrics, make_time_blocks, summarize_folds

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "artifacts" / "fill_strategy_validation.json"
N_FOLDS = 5
SEED = 42

STRATEGIES = {
    "median (current)": lambda: SimpleImputer(
        strategy="median", keep_empty_features=True),
    "zero": lambda: SimpleImputer(
        strategy="constant", fill_value=0.0, keep_empty_features=True),
    "median + missing flag": lambda: SimpleImputer(
        strategy="median", keep_empty_features=True, add_indicator=True),
}

MODELS = {
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


def build_pre(make_imputer, scale):
    """Same steps as data_processing.build_preprocessor, with a swappable imputer."""
    steps = [
        ("non_finite_to_nan", FunctionTransformer(
            _replace_non_finite, feature_names_out="one-to-one")),
        ("impute", make_imputer()),
        ("drop_constant", VarianceThreshold(threshold=0.0)),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def main():
    df = load_sample()
    train = df[df["day"] == TRAIN_DAY].reset_index(drop=True)
    y = train[TARGET].to_numpy()
    groups = make_time_blocks(train, N_FOLDS).to_numpy()
    schemas = {
        "full schema": prepare_features(train),
        "compact schema": compact_from_raw(train)[COMPACT_FEATURES],
    }

    zero_dur = (train["Flow Duration"] == 0).to_numpy()
    n_b = int(((y == 0) & zero_dur).sum())
    n_a = int(((y == 1) & zero_dur).sum())
    print(f"Training day {TRAIN_DAY}: {len(train):,} rows. "
          f"Zero-duration rows (undefined rates): {int(zero_dur.sum()):,} "
          f"({n_b} benign, {n_a} attack).")

    folds = list(GroupKFold(n_splits=N_FOLDS).split(
        schemas["full schema"], y, groups))
    results = {}
    for schema_name, X in schemas.items():
        for model_name, (factory, scale) in MODELS.items():
            print(f"running {schema_name} / {model_name} ...", flush=True)
            for strat_name, make_imp in STRATEGIES.items():
                rows = []
                oof = np.zeros(len(y), dtype=int)   # out-of-fold predictions
                for tr, va in folds:
                    pipe = Pipeline([
                        ("pre", build_pre(make_imp, scale)),
                        ("model", factory()),
                    ])
                    pipe.fit(X.iloc[tr], y[tr])
                    pred = pipe.predict(X.iloc[va])
                    rows.append(compute_metrics(
                        y[va], pred, pipe.predict_proba(X.iloc[va])[:, 1]))
                    oof[va] = pred
                results.setdefault(schema_name, {}).setdefault(model_name, {})[
                    strat_name] = {
                    "metrics": summarize_folds(rows),
                    "zero_duration": {
                        "benign": n_b, "attack": n_a,
                        "false_alarms": int(((oof == 1) & (y == 0) & zero_dur).sum()),
                        "missed_attacks": int(((oof == 0) & (y == 1) & zero_dur).sum()),
                    },
                }

    names = list(STRATEGIES)
    print("\n1) Validation inside the training day (mean over folds): "
          "balanced accuracy / false-positive rate")
    print(f"  {'':<44}" + "".join(f"{n:>23}" for n in names))
    for schema_name in schemas:
        for model_name in MODELS:
            cells = ""
            for s in names:
                m = results[schema_name][model_name][s]["metrics"]
                cells += f"{m['balanced_accuracy']:>12.3f} / {m['fpr']:>8.3f}"
            print(f"  {schema_name + ' | ' + model_name:<44}{cells}")

    print(f"\n2) Errors on the {int(zero_dur.sum()):,} zero-duration rows only "
          f"(pooled out-of-fold predictions; {n_b} benign, {n_a} attack)")
    print(f"  {'':<44}" + "".join(f"{n:>26}" for n in names))
    for schema_name in schemas:
        for model_name in MODELS:
            cells = ""
            for s in names:
                z = results[schema_name][model_name][s]["zero_duration"]
                cells += f"{z['false_alarms']}/{n_b} FP, {z['missed_attacks']}/{n_a} FN".rjust(26)
            print(f"  {schema_name + ' | ' + model_name:<44}{cells}")
    print("  (FP = benign flow flagged as attack, FN = attack flow missed)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "validation": f"{N_FOLDS}-fold time-blocked, inside training day {TRAIN_DAY}",
            "zero_duration_rows": {"benign": n_b, "attack": n_a},
            "results": results,
        }, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()