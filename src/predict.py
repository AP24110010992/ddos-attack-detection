"""
Prediction helpers shared by the dashboard and the tests.

Loads the saved models and turns input data into predictions using the SAME
feature preparation as training (prepare_features / prepare_numeric), so the
schema cannot drift between training and the dashboard.

  load_model(key)            -> saved scikit-learn Pipeline (cached)
  predict_frame(df, key)     -> DataFrame: prediction, label, attack probability

Keys: "full"    = 74 raw CIC columns (uploaded CSV page)
      "compact" = the 7 compact features (manual-entry form)

NOTE for app.py: the saved pipelines reference functions in
data_processing.py, so the src folder must be on sys.path when loading.

Self-check in a fresh process (from the project root):
    python src\\predict.py
"""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from compact_features import COMPACT_FEATURES, compact_from_raw, prepare_numeric
from data_processing import FEATURE_COLUMNS, TARGET, TEST_DAY, load_sample, prepare_features
from evaluate import compute_metrics

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
META_PATH = ROOT / "artifacts" / "model_metadata.json"
MODEL_FILES = {"full": "ddos_full_tree_d4.joblib",
               "compact": "ddos_compact_logreg.joblib"}
LABELS = {0: "BENIGN", 1: "DDoS ATTACK"}


class ModelNotTrainedError(FileNotFoundError):
    """Raised when a saved model or its metadata is missing."""


def _load_path(path):
    path = Path(path)
    if not path.exists():
        raise ModelNotTrainedError(
            f"{path.name} not found. Train the models first: python src\\train.py")
    return joblib.load(path)


@lru_cache(maxsize=None)
def load_model(key):
    if key not in MODEL_FILES:
        raise KeyError(f"Unknown model key {key!r}; use one of {list(MODEL_FILES)}")
    return _load_path(MODEL_DIR / MODEL_FILES[key])


def load_metadata():
    if not META_PATH.exists():
        raise ModelNotTrainedError(
            "model_metadata.json not found. Train the models first: "
            "python src\\train.py")
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)


def prepare_inputs(df, key):
    """Validate `df` and return the model inputs in the training column order."""
    if key == "full":
        return prepare_features(df)
    return prepare_numeric(df, COMPACT_FEATURES)


def predict_frame(df, key="full"):
    model = load_model(key)
    X = prepare_inputs(df, key)
    pred = np.asarray(model.predict(X)).astype(int)
    proba = model.predict_proba(X)[:, 1]
    return pd.DataFrame({
        "prediction": pred,
        "prediction_label": [LABELS[p] for p in pred],
        "attack_probability": proba,
    }, index=df.index)


def main():
    meta = load_metadata()
    print("Fresh-process reload check. Models trained "
          f"{meta['created_utc']} with scikit-learn "
          f"{meta['library_versions']['scikit-learn']}.")

    df = load_sample().reset_index(drop=True)
    test = df[df["day"] == TEST_DAY].reset_index(drop=True)
    y_te = test[TARGET].to_numpy()
    inputs = {
        "full": test,
        "compact": compact_from_raw(test),
    }

    for key in ("full", "compact"):
        entry = meta["models"][key]
        out = predict_frame(inputs[key], key)

        m = compute_metrics(y_te, out["prediction"].to_numpy(),
                            out["attack_probability"].to_numpy())
        ref = entry["test_metrics"]
        same = all(m[k] == ref[k] for k in ("tn", "fp", "fn", "tp"))
        print(f"\n[{key}] {entry['model_name']}")
        print(f"  confusion counts after reload: tn={m['tn']} fp={m['fp']} "
              f"fn={m['fn']} tp={m['tp']}  ->  "
              f"{'matches metadata [OK]' if same else 'MISMATCH'}")
        assert same, "reloaded model disagrees with saved evaluation"

        p = out["attack_probability"]
        assert p.between(0, 1).all(), "probability outside [0, 1]"
        print(f"  probabilities within [0, 1]                [OK]")

        # Column order of the input must not matter
        cols = list(reversed(list(inputs[key].columns)))
        shuffled = predict_frame(inputs[key][cols].head(200), key)
        assert (shuffled["prediction"].to_numpy()
                == out["prediction"].head(200).to_numpy()).all()
        print(f"  shuffled column order gives same predictions [OK]")

        # Missing column gives a clear error
        try:
            predict_frame(inputs[key].drop(columns=["Flow Duration"]), key)
            raise AssertionError("missing column was not detected")
        except ValueError as err:
            print(f"  missing-column error                        [OK]  ({err})")

    # Missing model file gives setup instructions
    try:
        _load_path(MODEL_DIR / "does_not_exist.joblib")
        raise AssertionError("missing model file was not detected")
    except ModelNotTrainedError as err:
        print(f"\nMissing-model message                        [OK]  ({err})")

    # Two REAL held-out test-day records (not synthetic)
    print("\nExample real held-out records (test day, never used in training):")
    for label, name in ((0, "BENIGN"), (1, "ATTACK")):
        row = test[test[TARGET] == label].head(1)
        o = predict_frame(row, "full").iloc[0]
        print(f"  actual {name:<7} -> predicted {o['prediction_label']:<12} "
              f"(attack probability {o['attack_probability']:.3f})")

    print("\nALL RELOAD CHECKS PASSED")


if __name__ == "__main__":
    main()