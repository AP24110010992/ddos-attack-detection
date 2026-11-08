"""
Exploratory data analysis (EDA) on the working sample.

Every chart is calculated from the real sample. Feature-level charts use the
TRAINING day only (01-12) so the test day does not influence exploration.
Charts are saved to artifacts/figures/ and a numeric summary to
artifacts/eda_summary.json.

Usage (from the project root):
    python src\\eda.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_processing import (
    FEATURE_COLUMNS, TARGET, TRAIN_DAY, build_preprocessor,
    get_output_feature_names, load_sample, prepare_features,
)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "artifacts" / "figures"
SUMMARY_PATH = ROOT / "artifacts" / "eda_summary.json"

COLORS = {"BENIGN": "#2a9d8f", "ATTACK": "#e76f51"}
DIST_FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Flow Packets/s", "Packet Length Mean",
]
MISSING_COLUMNS = ["Flow Bytes/s", "Flow Packets/s"]


def class_names(df):
    return df[TARGET].map({0: "BENIGN", 1: "ATTACK"})


def signed_log(x):
    """log10(1 + |x|) with the original sign; tames heavy-tailed features."""
    return np.sign(x) * np.log10(1 + np.abs(x))


def proto_name(p):
    return {6: "TCP", 17: "UDP"}.get(int(p), f"Other ({int(p)})")


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


# ---------------------------------------------------------------- charts ----
def fig_class_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    by_class = pd.crosstab(df["day"], class_names(df))[["BENIGN", "ATTACK"]]
    by_class.plot(kind="bar", ax=axes[0], rot=0,
                  color=[COLORS[c] for c in by_class.columns])
    axes[0].set_title("Benign vs attack rows by capture day")
    axes[0].set_xlabel("Capture day")
    axes[0].set_ylabel("Rows")

    by_type = pd.crosstab(df["attack_type"], df["day"])
    by_type.plot(kind="bar", ax=axes[1], rot=0, color=["#264653", "#e9c46a"])
    axes[1].set_title("Rows by traffic type and capture day")
    axes[1].set_xlabel("Traffic type")
    axes[1].set_ylabel("Rows")
    for ax in axes:
        for container in ax.containers:
            ax.bar_label(container, fontsize=8)
    save(fig, "fig01_class_distribution.png")
    return by_class, by_type


def fig_missing(df, X):
    bad = pd.DataFrame(~np.isfinite(X[MISSING_COLUMNS].to_numpy()),
                       columns=MISSING_COLUMNS, index=df.index)
    pct = bad.groupby(class_names(df)).mean() * 100
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pct.T[["BENIGN", "ATTACK"]].plot(
        kind="bar", ax=ax, rot=0, color=[COLORS["BENIGN"], COLORS["ATTACK"]])
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f%%", fontsize=8)
    ax.set_title("Rows with NaN or infinite value, by class")
    ax.set_ylabel("% of rows in the class")
    save(fig, "fig02_missing_values.png")
    n_cols = int((~np.isfinite(X.to_numpy())).any(axis=0).sum())
    return pct, n_cols


def fig_feature_distributions(train, X_train):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    y = train[TARGET].to_numpy()
    for ax, feat in zip(axes.ravel(), DIST_FEATURES):
        values = X_train[feat].to_numpy(dtype=float)
        finite = np.isfinite(values)
        transformed = signed_log(values[finite])
        bins = np.linspace(transformed.min(), transformed.max(), 41)
        for label, name in ((0, "BENIGN"), (1, "ATTACK")):
            sel = finite & (y == label)
            ax.hist(signed_log(values[sel]), bins=bins, alpha=0.6,
                    density=True, color=COLORS[name], label=name)
        ax.set_title(feat)
        ax.set_xlabel("signed log10(1 + |value|)")
        ax.legend(fontsize=8)
    fig.suptitle(f"Feature distributions by class (training day {TRAIN_DAY}, "
                 "infinite values excluded)")
    save(fig, "fig03_feature_distributions.png")


def fig_correlation(train, X_train):
    pre = build_preprocessor().fit(X_train)
    Z = pre.transform(X_train)
    names = get_output_feature_names(pre)
    Zdf = pd.DataFrame(Z, columns=names)
    corr = Zdf.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(15, 13))
    sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, square=True,
                xticklabels=True, yticklabels=True,
                cbar_kws={"shrink": 0.6}, ax=ax)
    ax.tick_params(labelsize=6)
    ax.set_title(f"Spearman correlation of the {len(names)} model features "
                 f"(training day {TRAIN_DAY})")
    save(fig, "fig04_correlation_heatmap.png")

    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = corr.abs().where(mask).stack()
    high = pairs[pairs > 0.95].sort_values(ascending=False)
    y = pd.Series(train[TARGET].to_numpy())
    with_target = Zdf.corrwith(y, method="spearman").sort_values(
        key=np.abs, ascending=False)
    return high, with_target, len(names)


def fig_protocol(df):
    proto = df["Protocol"].map(proto_name)
    by_class = pd.crosstab(proto, class_names(df))[["BENIGN", "ATTACK"]]
    by_type = pd.crosstab(proto, df["attack_type"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    by_class.plot(kind="bar", stacked=True, ax=axes[0], rot=0,
                  color=[COLORS["BENIGN"], COLORS["ATTACK"]])
    axes[0].set_title("Protocol by class")
    by_type.plot(kind="bar", stacked=True, ax=axes[1], rot=0)
    axes[1].set_title("Protocol by traffic type")
    for ax in axes:
        ax.set_xlabel("Protocol")
        ax.set_ylabel("Rows")
    save(fig, "fig05_protocol.png")
    return by_class, by_type


# ------------------------------------------------------------------ main ----
def main():
    df = load_sample()
    train = df[df["day"] == TRAIN_DAY]
    X_all = prepare_features(df)
    X_train = prepare_features(train)
    print(f"Loaded {len(df):,} rows; exploring features on the training day "
          f"({len(train):,} rows).\nSaving charts:")

    by_class, by_type = fig_class_distribution(df)
    pct, n_bad_cols = fig_missing(df, X_all)
    fig_feature_distributions(train, X_train)
    high, with_target, n_features = fig_correlation(train, X_train)
    proto_class, proto_type = fig_protocol(df)

    print("\n1) Rows by class and day\n" + by_class.to_string())
    print("\n   Rows by traffic type and day\n" + by_type.to_string())

    print(f"\n2) Columns (of {len(FEATURE_COLUMNS)} schema columns) containing "
          f"NaN or infinity: {n_bad_cols}")
    print("   % of rows affected, by class:\n" + pct.round(2).to_string())

    print(f"\n4) Feature pairs with |Spearman correlation| > 0.95 among the "
          f"{n_features} model features: {len(high)}")
    for (a, b), r in high.head(10).items():
        print(f"   {r:.3f}   {a}  <->  {b}")
    print("\n   Strongest correlations with the label (training day, "
          "exploration only):")
    for name, r in with_target.head(10).items():
        print(f"   {r:+.3f}   {name}")

    print("\n5) Protocol by class\n" + proto_class.to_string())
    print("\n   Protocol by traffic type\n" + proto_type.to_string())

    summary = {
        "rows_by_class_and_day": json.loads(by_class.to_json()),
        "rows_by_type_and_day": json.loads(by_type.to_json()),
        "schema_columns_with_nan_or_inf": n_bad_cols,
        "pct_rows_nan_or_inf_by_class": json.loads(pct.to_json()),
        "high_correlation_pairs": int(len(high)),
        "top_label_correlations": {k: float(v) for k, v in with_target.head(10).items()},
        "protocol_by_class": json.loads(proto_class.to_json()),
        "protocol_by_type": json.loads(proto_type.to_json()),
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {SUMMARY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()