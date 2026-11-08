"""
Detail report for the cross-day evaluation (train 01-12, test 03-11).

  1) Exact duplicates between the two days (leakage check)
  2) Retrains the same fixed-seed models and verifies that the confusion
     counts reproduce artifacts/final_evaluation.json
  3) Seen vs unseen attack types: NetBIOS never appears in training, so the
     test day is also scored WITHOUT NetBIOS
  4) Figures: confusion matrices, ROC curves, seen/unseen comparison

No tuning happens here. Models, schemas and the 0.5 threshold are unchanged.

Usage (from the project root):
    python src\\evaluation_report.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

from compact_features import COMPACT_FEATURES, compact_from_raw
from data_processing import (
    TARGET, TEST_DAY, TRAIN_DAY, load_sample, prepare_features,
)
from evaluate import compute_metrics
from final_evaluation import candidates

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "artifacts" / "figures"
FINAL_PATH = ROOT / "artifacts" / "final_evaluation.json"
OUT_PATH = ROOT / "artifacts" / "final_evaluation_detail.json"
UNSEEN_TYPE = "NetBIOS"

SCHEMAS = {
    "full (62 features)": lambda d: prepare_features(d),
    "compact (7 features)": lambda d: compact_from_raw(d)[COMPACT_FEATURES],
}
DISPLAY = {
    "full (62 features)": "Full schema (74 inputs, 62 after constants dropped)",
    "compact (7 features)": "Compact schema (7 features)",
}
SHORT = {
    "Decision Tree (depth 4)": "Tree d4",
    "Decision Tree (depth 10)": "Tree d10",
    "Random Forest (200 trees)": "Forest",
    "Logistic Regression (scaled)": "LogReg",
}


def save(fig, name, rect=None):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.tight_layout(rect=rect) if rect else fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def cross_day_duplicates(X_tr, y_tr, X_te, y_te):
    """Test rows whose feature values exactly equal some training row."""
    h_tr = pd.util.hash_pandas_object(X_tr.astype("float64"), index=False).to_numpy()
    h_te = pd.util.hash_pandas_object(X_te.astype("float64"), index=False).to_numpy()
    train_labels = (pd.DataFrame({"h": h_tr, "y": y_tr})
                    .groupby("h")["y"].agg(["min", "max"]))
    test_rows = pd.DataFrame({"h": h_te, "y": y_te})
    matched = test_rows.join(train_labels, on="h", how="inner")
    agree = (matched["min"] == matched["max"]) & (matched["min"] == matched["y"])
    return {
        "test_rows_matching_train": int(len(matched)),
        "same_label": int(agree.sum()),
        "conflicting_label": int((~agree).sum()),
        "matched_benign": int((matched["y"] == 0).sum()),
        "matched_attack": int((matched["y"] == 1).sum()),
    }


def fit_predict(models, X_tr, y_tr, X_te):
    out = {}
    for name, make in models.items():
        model = make().fit(X_tr, y_tr)
        out[name] = (np.asarray(model.predict(X_te)),
                     model.predict_proba(X_te)[:, 1])
    return out


# ---------------------------------------------------------------- figures --
def fig_confusion(preds_by_schema, y_te):
    names = list(next(iter(preds_by_schema.values())))
    rows, cols = len(preds_by_schema), len(names)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.9 * rows))
    axes = np.atleast_2d(axes)
    for r, (sname, preds) in enumerate(preds_by_schema.items()):
        for c, name in enumerate(names):
            cm = confusion_matrix(y_te, preds[name][0], labels=[0, 1])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        ax=axes[r, c], xticklabels=["BENIGN", "ATTACK"],
                        yticklabels=["BENIGN", "ATTACK"])
            axes[r, c].set_title(f"{SHORT[name]}\n{DISPLAY[sname]}", fontsize=9)
            axes[r, c].set_xlabel("Predicted")
            axes[r, c].set_ylabel("Actual")
    fig.suptitle(f"Confusion matrices on test day {TEST_DAY} "
                 f"(models trained on {TRAIN_DAY})")
    save(fig, "fig06_confusion_matrices.png", rect=(0, 0, 1, 0.96))


def fig_roc(preds_by_schema, y_te, results):
    fig, axes = plt.subplots(1, len(preds_by_schema), figsize=(6 * len(preds_by_schema), 5))
    axes = np.atleast_1d(axes)
    for ax, (sname, preds) in zip(axes, preds_by_schema.items()):
        for name, (_, score) in preds.items():
            fpr, tpr, _ = roc_curve(y_te, score)
            auc = results[sname][name]["all"]["roc_auc"]
            ax.plot(fpr, tpr, label=f"{SHORT[name]} (AUC {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_title(DISPLAY[sname], fontsize=10)
        ax.set_xlabel("False-positive rate")
        ax.set_ylabel("True-positive rate (recall)")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"ROC curves on test day {TEST_DAY}")
    save(fig, "fig07_roc_curves.png", rect=(0, 0, 1, 0.95))


def fig_comparison(results):
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    colors = ["#2a9d8f", "#e9c46a", "#e76f51"]
    for ax, (sname, res) in zip(axes, results.items()):
        names = list(res)
        x, w = np.arange(len(names)), 0.26
        series = [
            ("Recall on LDAP + Syn (seen in training)",
             [res[n]["seen"]["recall"] for n in names]),
            (f"Recall on {UNSEEN_TYPE} (never seen in training)",
             [res[n]["recall_unseen"] for n in names]),
            ("False-positive rate (benign flagged as attack)",
             [res[n]["all"]["fpr"] for n in names]),
        ]
        for i, (label, vals) in enumerate(series):
            bars = ax.bar(x + (i - 1) * w, vals, w, label=label, color=colors[i])
            ax.bar_label(bars, fmt="%.2f", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[n] for n in names])
        ax.set_ylim(0, 1.15)
        ax.set_title(DISPLAY[sname], fontsize=10)
    axes[0].set_ylabel("Share of flows")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(f"Seen vs unseen attack types on test day {TEST_DAY}")
    save(fig, "fig08_seen_vs_unseen.png", rect=(0, 0.07, 1, 0.95))


# ------------------------------------------------------------------- main --
def main():
    df = load_sample().reset_index(drop=True)
    train = df[df["day"] == TRAIN_DAY].reset_index(drop=True)
    test = df[df["day"] == TEST_DAY].reset_index(drop=True)
    y_tr, y_te = train[TARGET].to_numpy(), test[TARGET].to_numpy()
    types = test["attack_type"].to_numpy()
    seen, unseen = types != UNSEEN_TYPE, types == UNSEEN_TYPE

    saved = None
    if FINAL_PATH.exists():
        with open(FINAL_PATH, encoding="utf-8") as f:
            saved = json.load(f)["results"]

    print(f"Train {TRAIN_DAY} ({len(train):,} rows), test {TEST_DAY} "
          f"({len(test):,} rows; {int(unseen.sum()):,} {UNSEEN_TYPE} rows).")

    detail = {"duplicates": {}, "reproduces_saved_results": {}}
    results, preds_by_schema = {}, {}

    print("\n1) Exact duplicates between the training day and the test day "
          "(identical model-input values)")
    for sname, build in SCHEMAS.items():
        X_tr, X_te = build(train), build(test)
        dup = cross_day_duplicates(X_tr, y_tr, X_te, y_te)
        detail["duplicates"][sname] = dup
        print(f"   {DISPLAY[sname]}")
        print(f"     test rows identical to a training row : "
              f"{dup['test_rows_matching_train']:,}  "
              f"({dup['matched_benign']:,} benign, {dup['matched_attack']:,} attack)")
        print(f"     ... with the same label               : {dup['same_label']:,}")
        print(f"     ... with a CONFLICTING label          : {dup['conflicting_label']:,}")

        models = candidates("Flow IAT Std", X_tr.columns, skip_baselines=True)
        preds = fit_predict(models, X_tr, y_tr, X_te)
        preds_by_schema[sname] = preds
        results[sname] = {}
        for name, (pred, score) in preds.items():
            m_all = compute_metrics(y_te, pred, score)
            m_seen = compute_metrics(y_te[seen], pred[seen], score[seen])
            results[sname][name] = {
                "all": m_all,
                "seen": m_seen,
                "recall_unseen": float(pred[unseen].mean()),
                "recall_LDAP": float(pred[types == "LDAP"].mean()),
                "recall_Syn": float(pred[types == "Syn"].mean()),
            }

    print("\n2) Do the retrained models reproduce final_evaluation.json?")
    if saved is None:
        print("   final_evaluation.json not found, check skipped.")
    else:
        for sname in SCHEMAS:
            for name, r in results[sname].items():
                s = saved[sname][name]
                ok = all(r["all"][k] == s[k] for k in ("tn", "fp", "fn", "tp"))
                detail["reproduces_saved_results"][f"{sname}|{name}"] = ok
                print(f"   {'[OK]' if ok else '[MISMATCH]'}  {sname} | {name}")

    print(f"\n3) Test day scored WITHOUT {UNSEEN_TYPE}: the 2,000 benign flows plus "
          f"LDAP and Syn attacks only. 'BalAcc all' includes {UNSEEN_TYPE}.")
    for sname in SCHEMAS:
        print(f"\n   {DISPLAY[sname]}")
        print(f"   {'Model':<30}{'Recall LDAP+Syn':>16}{'FPR':>8}"
              f"{'BalAcc seen':>13}{f'Recall {UNSEEN_TYPE}':>16}{'BalAcc all':>12}")
        for name, r in results[sname].items():
            print(f"   {name:<30}{r['seen']['recall']:>16.3f}{r['all']['fpr']:>8.3f}"
                  f"{r['seen']['balanced_accuracy']:>13.3f}"
                  f"{r['recall_unseen']:>16.3f}"
                  f"{r['all']['balanced_accuracy']:>12.3f}")

    print("\n4) Figures")
    fig_confusion(preds_by_schema, y_te)
    fig_roc(preds_by_schema, y_te, results)
    fig_comparison(results)

    detail["results"] = results
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2)
    print(f"\nSaved: {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()