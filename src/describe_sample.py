"""
Describe the working sample (data/processed/sample_20k.parquet).

READ-ONLY with respect to the data: it prints statistics and saves two small
summary files for later use (artifacts/sample_statistics.csv and
artifacts/sample_profile.json).

Usage (from the project root):
    python src\\describe_sample.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = ROOT / "data" / "processed" / "sample_20k.parquet"
STATS_PATH = ROOT / "artifacts" / "sample_statistics.csv"
PROFILE_PATH = ROOT / "artifacts" / "sample_profile.json"

# Columns that describe the row itself, not the traffic. Never model inputs.
NOT_FEATURES = ["day", "source_file", "source_row_id", "label_original",
                "attack_type", "label_binary"]
ID_COLS = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "SimillarHTTP"]
KEY_FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Flow Bytes/s", "Flow Packets/s", "SYN Flag Count", "ACK Flag Count",
    "Protocol",
]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")


def main():
    if not SAMPLE_PATH.exists():
        raise SystemExit("Sample not found. Run python src\\build_sample.py first.")

    df = pd.read_parquet(SAMPLE_PATH)
    print(f"Loaded {SAMPLE_PATH.name}: {df.shape[0]:,} rows x {df.shape[1]} columns")
    class_name = df["label_binary"].map({0: "BENIGN", 1: "ATTACK"}).rename("class")

    # 1. Column types
    print("\n1) Column types")
    print(df.dtypes.astype(str).value_counts().to_string())
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    print(f"Non-numeric columns ({len(non_numeric)}): {non_numeric}")
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in NOT_FEATURES
    ]
    print(f"Numeric traffic columns (candidate features): {len(numeric_cols)}")

    # 2. Class distribution
    print("\n2) Class distribution")
    print(pd.crosstab(df["attack_type"], df["day"], margins=True).to_string())
    for day, pct in (df.groupby("day")["label_binary"].mean() * 100).items():
        print(f"  Attack share on {day}: {pct:.1f}%")

    # 3. Missing and infinite values
    print("\n3) Missing (NaN) and infinite values in numeric columns")
    numeric = df[numeric_cols]
    problems = pd.DataFrame({
        "NaN": numeric.isna().sum(),
        "infinite": np.isinf(numeric).sum(),
    })
    problems = problems[(problems["NaN"] > 0) | (problems["infinite"] > 0)]
    if problems.empty:
        print("  None found.")
    else:
        print(problems.to_string())
        print(f"\n  Rows affected, split by class "
              f"(class sizes: {class_name.value_counts().to_dict()}):")
        affected = {}
        for c in problems.index:
            bad = df[c].isna() | np.isinf(df[c])
            affected[c] = bad.groupby(class_name).sum()
        print(pd.DataFrame(affected).T.to_string())

    # 4. Constant columns (ignoring infinite values)
    finite = numeric.replace([np.inf, -np.inf], np.nan)
    n_unique = finite.nunique(dropna=True)
    constant = sorted(n_unique[n_unique <= 1].index)
    print(f"\n4) Constant columns in the sample ({len(constant)}): {constant}")

    # 5. Duplicates and label conflicts, using numeric traffic values only
    print("\n5) Duplicates (numeric traffic values only)")
    dup = int(df.duplicated(subset=numeric_cols, keep=False).sum())
    row_hash = pd.util.hash_pandas_object(df[numeric_cols], index=False)
    labels_per_hash = df.groupby(row_hash)["label_binary"].nunique()
    conflicts = int((labels_per_hash > 1).sum())
    print(f"  Rows sharing identical values with another row: {dup:,}")
    print(f"  Groups of identical rows with CONFLICTING labels: {conflicts:,}")

    # 6. Basic statistics of key features
    key = [c for c in KEY_FEATURES if c in finite.columns]
    absent = [c for c in KEY_FEATURES if c not in finite.columns]
    if absent:
        print(f"\nWARNING: expected columns not found: {absent}")
    print("\n6) Basic statistics of key features (infinite values ignored)")
    print(finite[key].describe().T[["count", "mean", "std", "min", "50%", "max"]]
          .to_string())

    # 7. Medians by class, and protocol by class
    print("\n7) Median of key features by class")
    print(finite[key].groupby(class_name).median().T.to_string())
    print("\n   Protocol by class (6 = TCP, 17 = UDP)")
    print(pd.crosstab(df["Protocol"], class_name).to_string())

    # 8. Identifier columns
    print("\n8) Identifier columns (NOT used as features): unique values")
    for c in ID_COLS:
        print(f"   {c:<16}{df[c].nunique():>8,}")

    # Save summaries for later use (dashboard, report)
    STATS_PATH.parent.mkdir(exist_ok=True)
    finite.describe().T.to_csv(STATS_PATH)
    profile = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "numeric_traffic_columns": len(numeric_cols),
        "class_counts": {k: int(v) for k, v in class_name.value_counts().items()},
        "rows_by_day_and_attack_type": {
            f"{d}|{t}": int(n)
            for (d, t), n in df.groupby(["day", "attack_type"]).size().items()
        },
        "constant_columns": constant,
        "columns_with_nan_or_inf": {
            c: {"nan": int(r["NaN"]), "infinite": int(r["infinite"])}
            for c, r in problems.iterrows()
        },
        "identical_feature_rows": dup,
        "identical_feature_label_conflicts": conflicts,
    }
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    print(f"\nSaved: {STATS_PATH}")
    print(f"Saved: {PROFILE_PATH}")


if __name__ == "__main__":
    main()