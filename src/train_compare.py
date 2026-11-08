"""
Compare candidate models with time-blocked validation INSIDE the training day.
The test day (03-11) is NOT used here.

For every model and fold:
  * preprocessing is fitted on that fold's training rows only
  * the model is scored on the held-out time block (validation)
  * the model is also scored on its own training rows, so that a large gap
    (overfitting) becomes visible

Usage (from the project root):
    python src\\train_compare.py
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from data_processing import (
    TARGET, TRAIN_DAY, build_preprocessor, load_sample, prepare_features,
)
from evaluate import compute_metrics, make_time_blocks, summarize_folds

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "artifacts" / "model_validation.json"
BASELINE_PATH = ROOT / "artifacts" / "baseline_validation.json"
N_FOLDS = 5
SEED = 42


def make_models():
    """name -> (factory, needs_scaling). A factory builds a fresh model per fold."""
    models = {}
    for depth in (2, 4, 6, 10, None):
        name = f"Decision Tree (depth {depth if depth else 'unlimited'})"
        models[name] = (
            lambda d=depth: DecisionTreeClassifier(
                max_depth=d, class_weight="balanced", random_state=SEED),
            False,
        )
    models["Random Forest (200 trees)"] = (
        lambda: RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1,
            random_state=SEED),
        False,
    )
    models["Logistic Regression (scaled)"] = (
        lambda: LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=SEED),
        True,
    )
    return models


def header():
    return (f"  {'Model':<34}{'TrainBA':>8}{'Acc':>7}{'Prec':>7}{'Recall':>8}"
            f"{'FPR':>7}{'F1':>7}{'ValBA':>7}{'AUC':>7}{'BAstd':>7}")


def row(label, s, train_ba=None):
    tr = "-" if train_ba is None else f"{train_ba:.3f}"
    return (f"  {label:<34}{tr:>8}{s['accuracy']:>7.3f}{s['precision']:>7.3f}"
            f"{s['recall']:>8.3f}{s['fpr']:>7.3f}{s['f1']:>7.3f}"
            f"{s['balanced_accuracy']:>7.3f}{s['roc_auc']:>7.3f}"
            f"{s['balanced_accuracy_std']:>7.3f}")


def main():
    df = load_sample()
    train = df[df["day"] == TRAIN_DAY].reset_index(drop=True)
    X = prepare_features(train)
    y = train[TARGET].to_numpy()
    groups = make_time_blocks(train, N_FOLDS).to_numpy()
    models = make_models()
    print(f"Training day {TRAIN_DAY}: {len(train):,} rows. "
          f"{N_FOLDS}-fold time-blocked validation, {len(models)} models.")

    results = {name: [] for name in models}
    seconds = {name: 0.0 for name in models}
    for fold, (tr, va) in enumerate(
            GroupKFold(n_splits=N_FOLDS).split(X, y, groups), start=1):
        print(f"  fold {fold}/{N_FOLDS} ...", flush=True)
        for name, (factory, scale) in models.items():
            start = time.time()
            pipe = Pipeline([
                ("pre", build_preprocessor(scale=scale)),
                ("model", factory()),
            ])
            pipe.fit(X.iloc[tr], y[tr])
            score = compute_metrics(
                y[va], pipe.predict(X.iloc[va]),
                pipe.predict_proba(X.iloc[va])[:, 1])
            on_train = compute_metrics(y[tr], pipe.predict(X.iloc[tr]))
            score["train_balanced_accuracy"] = on_train["balanced_accuracy"]
            results[name].append(score)
            seconds[name] += time.time() - start

    summary = {}
    for name, rows in results.items():
        s = summarize_folds(rows)
        s["train_balanced_accuracy"] = float(
            np.mean([r["train_balanced_accuracy"] for r in rows]))
        s["total_seconds"] = round(seconds[name], 1)
        summary[name] = s

    print("\nValidation inside the training day (mean over folds). "
          "TrainBA = balanced accuracy on the model's own training rows.")
    print(header())
    if BASELINE_PATH.exists():
        with open(BASELINE_PATH, encoding="utf-8") as f:
            base = json.load(f)
        best = base["stumps_ranked"][0]
        print(row("Baseline: majority class", base["majority_class"]))
        print(row(f"Baseline: stump on {best['feature']}", best))
    for name, s in summary.items():
        print(row(name, s, s["train_balanced_accuracy"]))

    print("\nGap between training and validation balanced accuracy "
          "(large gap = overfitting):")
    for name, s in summary.items():
        gap = s["train_balanced_accuracy"] - s["balanced_accuracy"]
        print(f"  {name:<34}{gap:>+7.3f}   ({s['total_seconds']}s for {N_FOLDS} folds)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "validation": f"{N_FOLDS}-fold time-blocked, inside training day {TRAIN_DAY}",
            "models": summary,
        }, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()