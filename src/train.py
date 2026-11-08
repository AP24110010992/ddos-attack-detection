"""
Train the two dashboard models on the TRAINING day (01-12) and save them.

  full    : Decision Tree (depth 4), full schema    -> CSV upload page
  compact : Logistic Regression (scaled), 7 features -> manual-entry form

Why these two: selected with a rule fixed BEFORE the test day was opened
(false-positive rate <= 0.02, then highest recall on all test-day attacks;
for the compact schema, the same rule picks Logistic Regression). The choice
was made after seeing test-day results, so the reported test scores are
slightly optimistic. This is recorded in the metadata.

The script refuses to save unless the retrained models reproduce the
confusion counts in artifacts/final_evaluation.json.

Usage (from the project root):
    python src\\train.py
"""

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from compact_features import COMPACT_FEATURES, compact_from_raw
from data_processing import (
    EXCLUDED_COLUMNS, FEATURE_COLUMNS, TARGET, TEST_DAY, TRAIN_DAY,
    build_preprocessor, load_sample, prepare_features,
)
from evaluate import compute_metrics

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
META_PATH = ROOT / "artifacts" / "model_metadata.json"
FINAL_PATH = ROOT / "artifacts" / "final_evaluation.json"
SEED = 42
ATTACK_TYPES = ["LDAP", "Syn", "NetBIOS"]

SPECS = {
    "full": {
        "file": "ddos_full_tree_d4.joblib",
        "model_name": "Decision Tree (depth 4)",
        "schema_name": "full (62 features)",
        "columns": FEATURE_COLUMNS,
        "build": lambda d: prepare_features(d),
        "make": lambda: Pipeline([
            ("pre", build_preprocessor()),
            ("model", DecisionTreeClassifier(
                max_depth=4, class_weight="balanced", random_state=SEED)),
        ]),
    },
    "compact": {
        "file": "ddos_compact_logreg.joblib",
        "model_name": "Logistic Regression (scaled)",
        "schema_name": "compact (7 features)",
        "columns": COMPACT_FEATURES,
        "build": lambda d: compact_from_raw(d)[COMPACT_FEATURES],
        "make": lambda: Pipeline([
            ("pre", build_preprocessor(scale=True)),
            ("model", LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=SEED)),
        ]),
    },
}


def row_hashes(X):
    return pd.util.hash_pandas_object(X.astype("float64"), index=False).to_numpy()


def explain(pipe, columns):
    """Feature importances (tree) or coefficients (logistic regression)."""
    names = list(pipe.named_steps["pre"].get_feature_names_out(list(columns)))
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        kind, values = "importance", model.feature_importances_
    else:
        kind, values = "coefficient (on standardised inputs)", model.coef_[0]
    order = np.argsort(-np.abs(values))
    top = [{"feature": names[i], "value": float(values[i])}
           for i in order[:15] if values[i] != 0]
    return kind, top


def training_ranges(X):
    finite = X.replace([np.inf, -np.inf], np.nan)
    return {
        c: {"min": float(finite[c].min()), "median": float(finite[c].median()),
            "max": float(finite[c].max())}
        for c in X.columns
    }


def main():
    df = load_sample().reset_index(drop=True)
    train = df[df["day"] == TRAIN_DAY].reset_index(drop=True)
    test = df[df["day"] == TEST_DAY].reset_index(drop=True)
    y_tr, y_te = train[TARGET].to_numpy(), test[TARGET].to_numpy()
    types = test["attack_type"].to_numpy()

    saved = None
    if FINAL_PATH.exists():
        with open(FINAL_PATH, encoding="utf-8") as f:
            saved = json.load(f)["results"]
    else:
        print("WARNING: final_evaluation.json not found, reproduction check skipped.")

    MODEL_DIR.mkdir(exist_ok=True)
    entries = {}
    for key, spec in SPECS.items():
        X_tr, X_te = spec["build"](train), spec["build"](test)
        pipe = spec["make"]().fit(X_tr, y_tr)
        pred = np.asarray(pipe.predict(X_te))
        score = pipe.predict_proba(X_te)[:, 1]
        m = compute_metrics(y_te, pred, score)

        print(f"\n[{key}] {spec['model_name']} on the {spec['schema_name']} schema")
        print(f"  trained on {TRAIN_DAY}: {len(train):,} rows, "
              f"{X_tr.shape[1]} input columns")
        print(f"  test-day confusion counts: tn={m['tn']} fp={m['fp']} "
              f"fn={m['fn']} tp={m['tp']}")
        if saved is not None:
            ref = saved[spec["schema_name"]][spec["model_name"]]
            if not all(m[k] == ref[k] for k in ("tn", "fp", "fn", "tp")):
                raise SystemExit("Counts differ from final_evaluation.json. "
                                 "Nothing was saved. Do not continue.")
            print("  reproduces final_evaluation.json            [OK]")

        # Test rows that are NOT exact copies of a training row
        dup = np.isin(row_hashes(X_te), row_hashes(X_tr))
        nd = ~dup
        m_nd = compute_metrics(y_te[nd], pred[nd], score[nd])
        n_nd_attack = int(((y_te == 1) & nd).sum())
        print(f"  test rows identical to a training row: {int(dup.sum()):,}; "
              f"non-duplicate rows: {int(nd.sum()):,} "
              f"({int(((y_te == 0) & nd).sum()):,} benign, {n_nd_attack:,} attack)")
        print(f"  on non-duplicate rows: recall {m_nd['recall']:.3f}, "
              f"FPR {m_nd['fpr']:.3f}, balanced accuracy "
              f"{m_nd['balanced_accuracy']:.3f}")

        path = MODEL_DIR / spec["file"]
        joblib.dump(pipe, path)
        print(f"  saved {path.relative_to(ROOT)} "
              f"({path.stat().st_size / 1024:,.1f} KB)")

        kind, top = explain(pipe, spec["columns"])
        entries[key] = {
            "file": f"models/{spec['file']}",
            "model_name": spec["model_name"],
            "schema_name": spec["schema_name"],
            "feature_columns": list(spec["columns"]),
            "n_input_features": int(X_tr.shape[1]),
            "n_features_after_preprocessing": int(
                len(pipe.named_steps["pre"].get_feature_names_out(
                    list(spec["columns"])))),
            "trained_on": {
                "day": TRAIN_DAY, "rows": int(len(train)),
                "benign": int((y_tr == 0).sum()), "attack": int((y_tr == 1).sum()),
            },
            "test_day": TEST_DAY,
            "test_metrics": m,
            "test_recall_by_type": {
                t: float(pred[types == t].mean()) for t in ATTACK_TYPES},
            "test_non_duplicate": {
                **m_nd, "rows": int(nd.sum()), "attack_rows": n_nd_attack},
            "test_rows_identical_to_training": int(dup.sum()),
            "explanation_kind": kind,
            "explanation_top": top,
            "training_ranges": training_ranges(X_tr) if key == "compact" else None,
            "protocol_values_in_training": (
                sorted(int(p) for p in X_tr["Protocol"].unique())),
        }

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "library_versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "seed": SEED, "decision_threshold": 0.5,
        "train_day": TRAIN_DAY, "test_day": TEST_DAY,
        "selection": {
            "rule": "FPR <= 0.02, then highest recall on all test-day attacks",
            "note": ("The rule was fixed before the test day was opened, but the "
                     "models were selected after seeing test-day results, so the "
                     "test scores are slightly optimistic."),
            "caveat": ("NetBIOS never appears in training. Detection of it differs "
                       "strongly between models and should not be generalised."),
        },
        "compact_schema_notes": {
            "Flow Duration unit": "microseconds (verified: rates = total / duration)",
            "Total Packets": "Total Fwd Packets + Total Backward Packets",
            "Total Bytes": "Total Length of Fwd Packets + Total Length of Bwd Packets",
            "SYN Flag Count": "left out: nonzero in only 9 of 10,000 training rows",
        },
        "excluded_columns": EXCLUDED_COLUMNS,
        "models": entries,
    }
    META_PATH.parent.mkdir(exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved: {META_PATH.relative_to(ROOT)}")
    print("TRAINING AND SAVING COMPLETE")


if __name__ == "__main__":
    main()