"""
Evaluation helpers shared by training, evaluation and the dashboard.

  compute_metrics  : accuracy, precision, recall, F1, false-positive rate,
                     balanced accuracy and ROC-AUC
  make_time_blocks : cut each capture file into consecutive time blocks,
                     used for validation INSIDE the training day
  summarize_folds  : mean and standard deviation of metrics across folds

Run the self-check (from the project root):
    python src\\evaluate.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from data_processing import SAMPLE_PATH, TARGET, TRAIN_DAY

METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "fpr",
               "balanced_accuracy", "roc_auc"]


def compute_metrics(y_true, y_pred, y_score=None):
    """
    Metrics for the positive class (1 = attack).
      recall = tp / (tp + fn)    share of real attacks that were detected
      fpr    = fp / (fp + tn)    share of benign flows wrongly flagged
    y_score is the model's attack probability (needed only for ROC-AUC).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    if y_score is not None and len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, y_score))
    else:
        auc = float("nan")

    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "balanced_accuracy": float((recall + (1 - fpr)) / 2),
        "roc_auc": auc,
    }


def make_time_blocks(df, n_blocks=5):
    """
    Block number (0 .. n_blocks-1) for every row. Within each source file the
    rows are ordered by Timestamp and cut into equal consecutive blocks, so
    block k holds the k-th time slice of EVERY file. Holding out one block
    therefore keeps every traffic type in both training and validation, while
    the validation flows come from a different moment than the training flows.
    """
    ts = pd.to_datetime(df["Timestamp"])
    blocks = pd.Series(0, index=df.index, dtype="int64")
    for _, idx in df.groupby("source_file").groups.items():
        ranks = ts.loc[idx].rank(method="first")
        blocks.loc[idx] = (((ranks - 1) * n_blocks) // len(idx)).astype("int64")
    return blocks


def summarize_folds(rows):
    """Mean and standard deviation of every metric over a list of fold results."""
    out = {}
    for key in METRIC_KEYS:
        values = np.array([r[key] for r in rows], dtype=float)
        out[key] = float(np.nanmean(values))
        out[key + "_std"] = float(np.nanstd(values))
    return out


def main():
    # 1) Metrics against a hand calculation
    y_true = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    y_pred = [0, 0, 1, 0, 1, 1, 1, 0, 1, 1]
    y_score = [0.1, 0.2, 0.9, 0.3, 0.8, 0.7, 0.9, 0.4, 0.6, 0.95]
    m = compute_metrics(y_true, y_pred, y_score)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 1, 1, 5)
    assert abs(m["accuracy"] - 0.8) < 1e-9
    assert abs(m["precision"] - 5 / 6) < 1e-9
    assert abs(m["recall"] - 5 / 6) < 1e-9
    assert abs(m["f1"] - 5 / 6) < 1e-9
    assert abs(m["fpr"] - 0.25) < 1e-9
    assert abs(m["roc_auc"] - 0.8125) < 1e-9
    print("Metrics match the hand calculation (tn=3, fp=1, fn=1, tp=5)  [OK]")

    # 2) Time blocks on the real training day
    if not SAMPLE_PATH.exists():
        raise SystemExit("Sample not found. Run python src\\build_sample.py first.")
    df = pd.read_parquet(SAMPLE_PATH)
    train = df[df["day"] == TRAIN_DAY].reset_index(drop=True)
    blocks = make_time_blocks(train, 5)
    table = pd.crosstab(blocks.rename("time_block"), train["attack_type"])
    table["total"] = table.sum(axis=1)
    print("\nRows per time block on the training day:")
    print(table.to_string())
    assert blocks.nunique() == 5
    if (table.drop(columns="total") == 0).any().any():
        print("WARNING: some block lacks a traffic type. Tell your mentor.")
    print("\nALL EVALUATION CHECKS PASSED")


if __name__ == "__main__":
    main()