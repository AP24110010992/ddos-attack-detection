"""
Baselines, evaluated with time-blocked validation INSIDE the training day.
The test day (03-11) is NOT used here.

  1. Majority class : always predicts the most common training class.
  2. One-feature stump : a decision tree of depth 1 (one threshold on one
     feature). Every feature is tried; they are ranked by validation
     balanced accuracy (average of recall and 1 - false-positive rate).

Preprocessing is re-fitted inside every fold using only that fold's training
rows, so nothing from the validation block leaks in.

Usage (from the project root):
    python src\\baselines.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

from data_processing import (
    TARGET, TRAIN_DAY, build_preprocessor, get_output_feature_names,
    load_sample, prepare_features,
)
from evaluate import compute_metrics, make_time_blocks, summarize_folds

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "artifacts" / "baseline_validation.json"
N_FOLDS = 5
SEED = 42
LABELS = {0: "BENIGN", 1: "ATTACK"}


def format_row(label, s):
    return (f"  {label:<44}{s['accuracy']:>9.3f}{s['recall']:>8.3f}"
            f"{s['fpr']:>8.3f}{s['balanced_accuracy']:>8.3f}{s['roc_auc']:>8.3f}")


def main():
    df = load_sample()
    train = df[df["day"] == TRAIN_DAY].reset_index(drop=True)
    X = prepare_features(train)
    y = train[TARGET].to_numpy()
    groups = make_time_blocks(train, N_FOLDS).to_numpy()
    print(f"Training day {TRAIN_DAY}: {len(train):,} rows, "
          f"{int((y == 0).sum()):,} benign, {int((y == 1).sum()):,} attack.")
    print(f"Time-blocked validation with {N_FOLDS} folds:")

    majority_rows, stump_rows, fold_info = [], {}, []
    splitter = GroupKFold(n_splits=N_FOLDS)
    for fold, (tr, va) in enumerate(splitter.split(X, y, groups), start=1):
        n_benign = int((y[va] == 0).sum())
        print(f"  fold {fold}: train {len(tr):,} rows, validation {len(va):,} rows "
              f"({n_benign:,} benign)")
        fold_info.append({"fold": fold, "train_rows": int(len(tr)),
                          "validation_rows": int(len(va)),
                          "validation_benign": n_benign})

        # Baseline 1: majority class
        dummy = DummyClassifier(strategy="most_frequent").fit(X.iloc[tr], y[tr])
        majority_rows.append(compute_metrics(
            y[va], dummy.predict(X.iloc[va]),
            dummy.predict_proba(X.iloc[va])[:, 1]))

        # Baseline 2: one-feature stumps (preprocessing fitted on this fold's train rows)
        pre = build_preprocessor().fit(X.iloc[tr])
        Z_tr, Z_va = pre.transform(X.iloc[tr]), pre.transform(X.iloc[va])
        for j, name in enumerate(get_output_feature_names(pre)):
            stump = DecisionTreeClassifier(
                max_depth=1, class_weight="balanced", random_state=SEED
            ).fit(Z_tr[:, [j]], y[tr])
            stump_rows.setdefault(name, []).append(compute_metrics(
                y[va], stump.predict(Z_va[:, [j]]),
                stump.predict_proba(Z_va[:, [j]])[:, 1]))

    majority = summarize_folds(majority_rows)
    stumps = {name: summarize_folds(rows) for name, rows in stump_rows.items()}
    ranked = sorted(stumps.items(), key=lambda kv: -kv[1]["balanced_accuracy"])

    print("\nValidation results inside the training day (mean of the folds)")
    print(f"  {'Model':<44}{'Accuracy':>9}{'Recall':>8}{'FPR':>8}{'BalAcc':>8}{'AUC':>8}")
    print(format_row("Majority class", majority))
    for name, s in ranked[:8]:
        print(format_row(f"Stump on {name}", s))

    # Refit the best stump on the whole training day to show its rule
    best_name = ranked[0][0]
    pre = build_preprocessor().fit(X)
    names = get_output_feature_names(pre)
    Z = pre.transform(X)
    j = names.index(best_name)
    stump = DecisionTreeClassifier(
        max_depth=1, class_weight="balanced", random_state=SEED
    ).fit(Z[:, [j]], y)

    rule = None
    if stump.tree_.node_count == 3:
        threshold = float(stump.tree_.threshold[0])
        left = int(stump.classes_[np.argmax(stump.tree_.value[1][0])])
        right = int(stump.classes_[np.argmax(stump.tree_.value[2][0])])
        rule = (f"if {best_name} <= {threshold:,.4g} then {LABELS[left]} "
                f"else {LABELS[right]}")
        print(f"\nBest single-feature rule (fitted on the whole training day):\n  {rule}")
    print(f"Best stump balanced accuracy: {ranked[0][1]['balanced_accuracy']:.3f} "
          f"(std across folds {ranked[0][1]['balanced_accuracy_std']:.3f})")

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "validation": f"{N_FOLDS}-fold time-blocked, inside training day {TRAIN_DAY}",
            "folds": fold_info,
            "majority_class": majority,
            "best_stump_feature": best_name,
            "best_stump_rule": rule,
            "stumps_ranked": [{"feature": n, **s} for n, s in ranked],
        }, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()