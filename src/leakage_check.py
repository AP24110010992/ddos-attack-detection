"""
Leakage and shortcut check (DIAGNOSTIC ONLY).

Question: which columns can separate BENIGN from ATTACK on their own,
and would that be a legitimate reason to use them?

IMPORTANT: columns are excluded from the model on PRINCIPLE (identifiers and
capture artefacts are not legitimate traffic behaviour). The numbers below
justify that decision; they are not used to tune or choose features.

  A) Suspicious columns (IPs, ports, flow id, row index ...), evaluated two ways:
       - cross-day : learn on 01-12, test on 03-11
       - random    : random 70/30 split of the whole sample
  B) Single-feature power of every ordinary traffic feature (cross-day AUC)
  C) Traffic columns that are exact copies of each other
  D) Do NaN/infinite rate values come from zero-duration flows?

Usage (from the project root):
    python src\\leakage_check.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = ROOT / "data" / "processed" / "sample_20k.parquet"
OUT_PATH = ROOT / "artifacts" / "leakage_check.json"

SEED = 42
TARGET = "label_binary"
TRAIN_DAY, TEST_DAY = "01-12", "03-11"

NOT_FEATURES = ["day", "source_file", "source_row_id", "label_original",
                "attack_type", "label_binary"]
CATEGORICAL_SUSPECTS = ["Source IP", "Destination IP", "Flow ID", "SimillarHTTP"]
NUMERIC_SUSPECTS = ["Source Port", "Destination Port", "Inbound", "source_row_id"]

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")


def rule_categorical(train, test, col):
    """Predict each test row by the majority class of its value in training."""
    mapping = (train.groupby(col)[TARGET].mean() >= 0.5).astype(int)
    default = int(train[TARGET].mean() >= 0.5)  # used for values never seen
    pred = test[col].map(mapping).fillna(default).astype(int)
    unseen = float((~test[col].isin(mapping.index)).mean())
    return balanced_accuracy_score(test[TARGET], pred), unseen


def rule_numeric(train, test, col):
    """Small decision tree that sees ONLY this one column."""
    tree = DecisionTreeClassifier(max_depth=3, random_state=SEED)
    tree.fit(train[[col]], train[TARGET])
    return balanced_accuracy_score(test[TARGET], tree.predict(test[[col]]))


def main():
    if not SAMPLE_PATH.exists():
        raise SystemExit("Sample not found. Run python src\\build_sample.py first.")

    df = pd.read_parquet(SAMPLE_PATH)
    train = df[df["day"] == TRAIN_DAY]
    test = df[df["day"] == TEST_DAY]
    rnd_train, rnd_test = train_test_split(
        df, test_size=0.3, stratify=df[TARGET], random_state=SEED
    )
    print(f"Loaded {len(df):,} rows. Train day {TRAIN_DAY}: {len(train):,} rows, "
          f"test day {TEST_DAY}: {len(test):,} rows.")

    # ---- A) suspicious columns --------------------------------------------
    print("\nA) Suspicious columns. Score = balanced accuracy "
          "(0.500 = no better than guessing, 1.000 = perfect)")
    rows = []
    for col in CATEGORICAL_SUSPECTS:
        ba_day, unseen = rule_categorical(train, test, col)
        ba_rnd, _ = rule_categorical(rnd_train, rnd_test, col)
        rows.append({"column": col, "kind": "categorical", "cross_day": ba_day,
                     "random_split": ba_rnd, "unseen_in_test_day": unseen})
    for col in NUMERIC_SUSPECTS:
        rows.append({"column": col, "kind": "numeric",
                     "cross_day": rule_numeric(train, test, col),
                     "random_split": rule_numeric(rnd_train, rnd_test, col),
                     "unseen_in_test_day": np.nan})
    table = pd.DataFrame(rows).set_index("column")
    print(table.to_string())

    # ---- B) single-feature power of ordinary traffic features -------------
    traffic_cols = [c for c in df.select_dtypes("number").columns
                    if c not in NOT_FEATURES and c not in NUMERIC_SUSPECTS]
    clean = df[traffic_cols].replace([np.inf, -np.inf], np.nan)
    x_train = clean[df["day"] == TRAIN_DAY]
    x_test = clean[df["day"] == TEST_DAY]
    medians = x_train.median()          # from the TRAINING day only
    x_train = x_train.fillna(medians)
    x_test = x_test.fillna(medians)

    results, skipped = [], []
    for c in traffic_cols:
        if x_train[c].nunique() <= 1:
            skipped.append(c)
            continue
        a_train = roc_auc_score(train[TARGET], x_train[c])
        sign = 1 if a_train >= 0.5 else -1   # direction learned on training day
        a_test = roc_auc_score(test[TARGET], sign * x_test[c])
        results.append((c, max(a_train, 1 - a_train), a_test))
    results.sort(key=lambda r: -r[2])
    varying = [r[0] for r in results]

    print(f"\nB) Single-feature ROC-AUC, learned on {TRAIN_DAY}, measured on {TEST_DAY}")
    print(f"   Features evaluated: {len(results)} "
          f"(skipped {len(skipped)} constant columns on the training day)")
    print(f"   Features with test-day AUC >= 0.95: "
          f"{sum(r[2] >= 0.95 for r in results)}")
    print(f"   {'feature':<32}{'AUC train day':>15}{'AUC test day':>15}")
    for name, a_tr, a_te in results[:10]:
        print(f"   {name:<32}{a_tr:>15.3f}{a_te:>15.3f}")

    # ---- C) exact duplicate columns ----------------------------------------
    dup_pairs = [(a, b) for i, a in enumerate(varying)
                 for b in varying[i + 1:] if df[a].equals(df[b])]
    print("\nC) Traffic columns that are exact copies of each other:")
    print("   " + (", ".join(f"{a} == {b}" for a, b in dup_pairs)
                    if dup_pairs else "none"))

    # ---- D) where do NaN / infinite rate values come from? ----------------
    bad = np.zeros(len(df), dtype=bool)
    for c in ["Flow Bytes/s", "Flow Packets/s"]:
        bad |= (df[c].isna() | np.isinf(df[c])).to_numpy()
    zero_duration = (df["Flow Duration"] == 0).to_numpy()
    print("\nD) NaN/infinite rate values vs zero-duration flows")
    print(f"   Rows with NaN/inf in a rate feature     : {int(bad.sum()):,}")
    print(f"   ... of which Flow Duration == 0         : {int((bad & zero_duration).sum()):,}")
    print(f"   Rows with Flow Duration == 0 (all rows) : {int(zero_duration.sum()):,}")

    # ---- save ---------------------------------------------------------------
    report = {
        "suspicious_columns": json.loads(table.to_json(orient="index")),
        "single_feature_auc": [
            {"feature": n, "auc_train_day": float(a), "auc_test_day": float(b)}
            for n, a, b in results
        ],
        "constant_on_train_day": skipped,
        "duplicate_column_pairs": dup_pairs,
        "rate_nan_inf_rows": int(bad.sum()),
        "rate_nan_inf_with_zero_duration": int((bad & zero_duration).sum()),
    }
    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()