"""
Data processing: feature schema, input validation and preprocessing pipeline.

Training, evaluation and the dashboard all import from this module so the
feature schema (names AND order) can never differ between them.

Decisions are documented in EXCLUDED_COLUMNS and in the notes below:
  * Identifiers and capture artefacts are excluded (see leakage_check.py).
  * Infinite values become NaN; NaN is filled with the TRAINING-set median.
  * Constant columns are removed, detected on the TRAINING set only.

Run the self-check (from the project root):
    python src\\data_processing.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = ROOT / "data" / "processed" / "sample_20k.parquet"

TARGET = "label_binary"      # 0 = BENIGN, 1 = attack
TRAIN_DAY = "01-12"
TEST_DAY = "03-11"

# Columns that are NOT model inputs, with the reason for each exclusion.
EXCLUDED_COLUMNS = {
    "source_row_id": "row position in the raw file; leaks capture order",
    "Flow ID": "identifier built from IPs, ports and protocol",
    "Source IP": "identifies a host, not traffic behaviour",
    "Destination IP": "identifies a host, not traffic behaviour",
    "Timestamp": "capture time; not a behavioural feature",
    "SimillarHTTP": "text field; not a traffic measurement",
    "Source Port": "single-column shortcut (cross-day balanced accuracy 0.68)",
    "Destination Port": "single-column shortcut (cross-day balanced accuracy 0.89)",
    "Inbound": "single-column shortcut (cross-day balanced accuracy 0.90)",
    "Subflow Fwd Packets": "exact copy of Total Fwd Packets in our sample",
    "Subflow Bwd Packets": "exact copy of Total Backward Packets in our sample",
    "Fwd Header Length.1": "exact copy of Fwd Header Length in our sample",
    "RST Flag Count": "identical to Fwd PSH Flags in our sample",
}

# The model's input schema: names AND order. Never reorder silently.
FEATURE_COLUMNS = [
    "Protocol", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s",
    "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
    "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags",
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length",
    "Max Packet Length", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
    "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Bytes",
    "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std",
    "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]


def load_sample():
    """Load the 20,000-row working sample created by build_sample.py."""
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"{SAMPLE_PATH} not found. Run python src\\build_sample.py first."
        )
    return pd.read_parquet(SAMPLE_PATH)


def prepare_features(df):
    """
    Validate a DataFrame and return the model-input columns in schema order.

    Works for the sample AND for uploaded raw CIC CSV files: column names are
    stripped of spaces, missing columns raise a clear error, and any value
    that is not a number raises an error (it is never silently replaced).
    """
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        shown = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        raise ValueError(
            f"{len(missing)} required column(s) missing: {shown}"
        )

    raw = df[FEATURE_COLUMNS]
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    not_numeric = numeric.isna() & raw.notna()   # had a value, but not a number
    if not_numeric.to_numpy().any():
        bad_cols = [c for c in FEATURE_COLUMNS if not_numeric[c].any()]
        raise ValueError(f"Non-numeric values found in column(s): {bad_cols[:8]}")
    return numeric


def _replace_non_finite(X):
    """Turn +/-infinity (and NaN) into NaN so the imputer can fill them."""
    X = np.asarray(X, dtype="float64")
    return np.where(np.isfinite(X), X, np.nan)


def build_preprocessor(scale=False):
    """
    Preprocessing pipeline (fit it on TRAINING data only):
      1. infinity -> NaN
      2. NaN -> median of the training data
      3. remove columns that are constant in the training data
      4. (optional) standardise, needed for Logistic Regression
    """
    steps = [
        ("non_finite_to_nan", FunctionTransformer(
            _replace_non_finite, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("drop_constant", VarianceThreshold(threshold=0.0)),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def get_output_feature_names(fitted_preprocessor):
    """Names of the columns that remain after preprocessing, in order."""
    return list(fitted_preprocessor.get_feature_names_out(FEATURE_COLUMNS))


def main():
    df = load_sample()
    train = df[df["day"] == TRAIN_DAY]
    test = df[df["day"] == TEST_DAY]

    # --- schema sanity ------------------------------------------------------
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS), "duplicate names in schema"
    overlap = set(FEATURE_COLUMNS) & set(EXCLUDED_COLUMNS)
    assert not overlap, f"excluded columns found in the schema: {overlap}"
    print(f"Feature columns defined in schema : {len(FEATURE_COLUMNS)}")
    print(f"Excluded columns documented       : {len(EXCLUDED_COLUMNS)}")

    # --- fit on the training day, transform both ---------------------------
    X_train = prepare_features(train)
    X_test = prepare_features(test)
    bad_train = int((~np.isfinite(X_train.to_numpy())).sum())
    bad_test = int((~np.isfinite(X_test.to_numpy())).sum())
    print(f"\nNon-finite cells before preprocessing: train day {bad_train:,}, "
          f"test day {bad_test:,} (total {bad_train + bad_test:,})")

    pre = build_preprocessor().fit(X_train)
    Z_train = pre.transform(X_train)
    Z_test = pre.transform(X_test)
    names = get_output_feature_names(pre)
    dropped = [c for c in FEATURE_COLUMNS if c not in names]

    print(f"Features after preprocessing        : {len(names)}")
    print(f"Dropped as constant on training day : {len(dropped)}")
    print(f"  {dropped}")
    print(f"Output shapes: train {Z_train.shape}, test {Z_test.shape}")
    assert Z_train.shape[1] == Z_test.shape[1] == len(names)
    assert np.isfinite(Z_train).all() and np.isfinite(Z_test).all()
    print("Non-finite cells after preprocessing: 0  [OK]")

    median_pps = pre.named_steps["impute"].statistics_[FEATURE_COLUMNS.index("Flow Packets/s")]
    print(f"Fill value used for Flow Packets/s (training-day median): {median_pps:,.2f}")

    # --- infinity / NaN in new data are handled ----------------------------
    probe = X_test.head(5).copy()
    probe["Flow Packets/s"] = np.inf
    probe.iloc[0, probe.columns.get_loc("Flow Bytes/s")] = np.nan
    assert np.isfinite(pre.transform(probe)).all()
    print("Infinity/NaN probe handled              [OK]")

    # --- column order in the input does not matter -------------------------
    shuffled = test.head(5)[list(reversed(FEATURE_COLUMNS))]
    assert prepare_features(shuffled).equals(X_test.head(5))
    print("Shuffled column order re-aligned        [OK]")

    # --- bad input is rejected with a clear message ------------------------
    try:
        prepare_features(test.drop(columns=["Flow Duration"]))
        raise AssertionError("missing column was not detected")
    except ValueError as err:
        print(f"Missing-column check                    [OK]  ({err})")

    bad = test.head(3).copy()
    bad["Protocol"] = ["abc", "6", "17"]
    try:
        prepare_features(bad)
        raise AssertionError("non-numeric value was not detected")
    except ValueError as err:
        print(f"Non-numeric check                       [OK]  ({err})")

    print("\nALL PREPROCESSING CHECKS PASSED")


if __name__ == "__main__":
    main()