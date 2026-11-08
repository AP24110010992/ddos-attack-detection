"""
Cross-day evaluation: train on the 01-12 day, test on the 03-11 day.

This opens the test day. The candidate list, schemas, preprocessing and the
0.5 decision threshold were fixed BEFORE running. Results are saved to
artifacts/final_evaluation.json.

Parts:
  1) Main results on the full (62) and compact (7) schemas
  2) Per traffic type on the test day (NetBIOS never appears in training)
  3) Sensitivity: add Source Port, Destination Port, Inbound to the full schema
  4) Preliminary random 70/30 split of the whole sample, for comparison

Usage (from the project root):
    python src\\final_evaluation.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from compact_features import COMPACT_FEATURES, compact_from_raw
from data_processing import (
    TARGET, TEST_DAY, TRAIN_DAY, build_preprocessor, load_sample,
    prepare_features,
)
from evaluate import compute_metrics

ROOT = Path(__file__).resolve().parent.parent
BASE_PATH = ROOT / "artifacts" / "baseline_validation.json"
OUT_PATH = ROOT / "artifacts" / "final_evaluation.json"
SEED = 42
EXTRA_COLUMNS = ["Source Port", "Destination Port", "Inbound"]
ATTACK_TYPES = ["LDAP", "Syn", "NetBIOS"]


class OneFeatureStump:
    """Preprocess, keep one feature, fit a depth-1 tree (a single threshold)."""

    def __init__(self, feature):
        self.feature = feature

    def fit(self, X, y):
        self.pre_ = build_preprocessor().fit(X)
        names = list(self.pre_.get_feature_names_out(list(X.columns)))
        self.j_ = names.index(self.feature)
        self.tree_ = DecisionTreeClassifier(
            max_depth=1, class_weight="balanced", random_state=SEED
        ).fit(self._z(X), y)
        return self

    def _z(self, X):
        return self.pre_.transform(X)[:, [self.j_]]

    def predict(self, X):
        return self.tree_.predict(self._z(X))

    def predict_proba(self, X):
        return self.tree_.predict_proba(self._z(X))


def pipe(model, scale=False):
    return Pipeline([("pre", build_preprocessor(scale=scale)), ("model", model)])


def candidates(stump_feature, columns, skip_baselines=False):
    """name -> factory returning a fresh, unfitted estimator."""
    def tree(depth):
        return DecisionTreeClassifier(
            max_depth=depth, class_weight="balanced", random_state=SEED)

    models = {}
    if not skip_baselines:
        models["Baseline: majority class"] = lambda: pipe(
            DummyClassifier(strategy="most_frequent"))
        if stump_feature in columns:
            models[f"Baseline: stump on {stump_feature}"] = (
                lambda: OneFeatureStump(stump_feature))
    models["Decision Tree (depth 4)"] = lambda: pipe(tree(4))
    models["Decision Tree (depth 10)"] = lambda: pipe(tree(10))
    models["Random Forest (200 trees)"] = lambda: pipe(RandomForestClassifier(
        n_estimators=200, class_weight="balanced", n_jobs=-1,
        random_state=SEED))
    models["Logistic Regression (scaled)"] = lambda: pipe(
        LogisticRegression(class_weight="balanced", max_iter=2000,
                           random_state=SEED), scale=True)
    return models


def run(models, X_tr, y_tr, X_te, y_te, test_types=None):
    out = {}
    for name, make in models.items():
        model = make().fit(X_tr, y_tr)
        pred = np.asarray(model.predict(X_te))
        entry = compute_metrics(y_te, pred, model.predict_proba(X_te)[:, 1])
        if test_types is not None:
            for t in ATTACK_TYPES:
                sel = (test_types == t).to_numpy()
                entry[f"recall_{t}"] = (
                    float(pred[sel].mean()) if sel.any() else float("nan"))
        out[name] = entry
    return out


def table_header():
    return (f"  {'Model':<40}{'Acc':>7}{'Prec':>7}{'Recall':>8}{'FPR':>7}"
            f"{'F1':>7}{'BalAcc':>7}{'AUC':>7}{'TN':>7}{'FP':>6}{'FN':>6}{'TP':>7}")


def table_row(label, m):
    return (f"  {label:<40}{m['accuracy']:>7.3f}{m['precision']:>7.3f}"
            f"{m['recall']:>8.3f}{m['fpr']:>7.3f}{m['f1']:>7.3f}"
            f"{m['balanced_accuracy']:>7.3f}{m['roc_auc']:>7.3f}"
            f"{m['tn']:>7}{m['fp']:>6}{m['fn']:>6}{m['tp']:>7}")


def main():
    df = load_sample().reset_index(drop=True)
    train = df[df["day"] == TRAIN_DAY].reset_index(drop=True)
    test = df[df["day"] == TEST_DAY].reset_index(drop=True)
    y_tr, y_te = train[TARGET].to_numpy(), test[TARGET].to_numpy()

    stump_feature = "Flow IAT Std"
    if BASE_PATH.exists():
        with open(BASE_PATH, encoding="utf-8") as f:
            stump_feature = json.load(f)["best_stump_feature"]

    print("=" * 70)
    print(f"TEST DAY OPENED. Train: {TRAIN_DAY} ({len(train):,} rows).  "
          f"Test: {TEST_DAY} ({len(test):,} rows).")
    print(f"Test-day composition: {test['attack_type'].value_counts().to_dict()}")
    print("=" * 70)

    schemas = {
        "full (62 features)": lambda d: prepare_features(d),
        "compact (7 features)": lambda d: compact_from_raw(d)[COMPACT_FEATURES],
        "SENSITIVITY: full + ports + Inbound": lambda d: pd.concat(
            [prepare_features(d), d[EXTRA_COLUMNS]], axis=1),
    }

    results = {}
    for sname, build in schemas.items():
        X_tr, X_te = build(train), build(test)
        sens = sname.startswith("SENSITIVITY")
        models = candidates(stump_feature, X_tr.columns, skip_baselines=sens)
        print(f"\n--- {sname}: {X_tr.shape[1]} input features ---", flush=True)
        results[sname] = run(models, X_tr, y_tr, X_te, y_te, test["attack_type"])
        print(table_header())
        for name, m in results[sname].items():
            print(table_row(name, m))
        print(f"\n  Share detected per traffic type on the test day "
              f"(benign: share wrongly flagged = FPR)")
        print(f"  {'Model':<40}" + "".join(f"{t:>10}" for t in ATTACK_TYPES))
        for name, m in results[sname].items():
            print(f"  {name:<40}"
                  + "".join(f"{m['recall_' + t]:>10.3f}" for t in ATTACK_TYPES))

    # Preliminary random split (whole sample), full schema, for comparison only
    X_all = prepare_features(df)
    y_all = df[TARGET].to_numpy()
    tr_idx, te_idx = train_test_split(
        np.arange(len(df)), test_size=0.3, stratify=y_all, random_state=SEED)
    rnd = run(candidates(stump_feature, X_all.columns, skip_baselines=True),
              X_all.iloc[tr_idx], y_all[tr_idx], X_all.iloc[te_idx], y_all[te_idx])
    results["PRELIMINARY random 70/30 split (full)"] = rnd

    print("\n--- Random split vs cross-day (full schema): balanced accuracy / FPR ---")
    print(f"  {'Model':<40}{'random split':>22}{'cross-day':>22}")
    for name, m in rnd.items():
        c = results["full (62 features)"][name]
        print(f"  {name:<40}"
              f"{m['balanced_accuracy']:>12.3f} /{m['fpr']:>7.3f}"
              f"{c['balanced_accuracy']:>12.3f} /{c['fpr']:>7.3f}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "train_day": TRAIN_DAY, "test_day": TEST_DAY,
            "threshold": 0.5, "stump_feature": stump_feature,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()