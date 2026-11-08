"""
Data helpers for the dashboard's Model Performance page.

  metrics_from_counts / metric_rows : recompute every metric from the four
                                      confusion-matrix counts and compare it
                                      with the value saved by train.py
  evidence_table   : the same models under three evaluation methods
  seen_unseen_table: detection on attack types seen / never seen in training
  sensitivity_table: what happens if the excluded columns were used
  importance_frame : feature importances (tree) or coefficients (logistic regression)
  tree_rules       : the decision rules of a tree model, as readable text

This module does NOT import streamlit, so it can be tested on its own.
Run the self-check (from the project root):
    python src\\performance_data.py
"""

import numpy as np
import pandas as pd
from sklearn.tree import export_text

from dashboard_data import load_json

NAN = float("nan")
SENSITIVITY_KEY = "SENSITIVITY: full + ports + Inbound"
RANDOM_SPLIT_KEY = "PRELIMINARY random 70/30 split (full)"


def metrics_from_counts(tn, fp, fn, tp):
    """Every count-based metric, calculated from the confusion matrix."""
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
        "balanced_accuracy": (recall + (1 - fpr)) / 2,
    }


def metric_rows(m):
    """
    One row per metric: meaning, the calculation written with the real counts,
    the recomputed value, and whether it matches the value saved by train.py.
    Columns starting with an underscore are numeric helpers for tests.
    """
    tn, fp, fn, tp = m["tn"], m["fp"], m["fn"], m["tp"]
    total = tn + fp + fn + tp
    calc = metrics_from_counts(tn, fp, fn, tp)
    spec = [
        ("Accuracy", "accuracy",
         "Share of all test flows classified correctly. It can look good even "
         "when benign flows are mishandled, because attack flows outnumber "
         "benign flows in our sample.",
         f"({tp:,} + {tn:,}) ÷ {total:,}"),
        ("Precision (attack)", "precision",
         "Of the flows the model flagged as attack, the share that really "
         "were attacks. It depends on how common attacks are in the sample, "
         "so it would be lower on real traffic.",
         f"{tp:,} ÷ ({tp:,} + {fp:,})"),
        ("Recall (attack)", "recall",
         "Of the real attacks, the share the model caught. A missed attack "
         "(false negative) is the costly error for a detector.",
         f"{tp:,} ÷ ({tp:,} + {fn:,})"),
        ("F1-score (attack)", "f1",
         "One score combining precision and recall (their harmonic mean).",
         "2 × precision × recall ÷ (precision + recall)"),
        ("False-positive rate", "fpr",
         "Of the benign flows, the share wrongly flagged as attack (false "
         "alarms). Lower is better. Unlike precision, it does not depend on "
         "how many attacks are in the sample.",
         f"{fp:,} ÷ ({fp:,} + {tn:,})"),
        ("Balanced accuracy", "balanced_accuracy",
         "Average of recall and (1 − false-positive rate). It gives benign and "
         "attack flows equal weight, so it is not inflated by the attack "
         "majority.",
         "(recall + (1 − false-positive rate)) ÷ 2"),
    ]
    rows = []
    for name, key, meaning, how in spec:
        value, saved = calc[key], m[key]
        rows.append({
            "Metric": name, "What it means in our problem": meaning,
            "Calculation with the test-day counts": how,
            "Value": f"{value * 100:.2f}%",
            "Check against the saved value": (
                "matches" if abs(value - saved) < 1e-9 else "DIFFERS"),
            "_value": value, "_saved": saved,
        })
    auc = m["roc_auc"]
    rows.append({
        "Metric": "ROC-AUC",
        "What it means in our problem": (
            "Chance that a randomly chosen attack flow gets a higher attack "
            "score than a randomly chosen benign flow (1.0 = perfect ranking, "
            "0.5 = guessing). A shallow tree has few distinct scores, so its "
            "AUC is close to its balanced accuracy."),
        "Calculation with the test-day counts": (
            "Computed by scikit-learn from the model's scores"),
        "Value": f"{auc:.3f}",
        "Check against the saved value": (
            "taken from the saved results (needs the scores, not just counts)"),
        "_value": auc, "_saved": auc,
    })
    return pd.DataFrame(rows)


def evidence_table(key, meta, final, validation=None, baseline=None,
                   compact_val=None):
    """
    The same models under three evaluation methods (balanced accuracy):
      val_ba  : validation inside the training day (time-blocked folds)
      rand_ba : random 70/30 split of the whole sample (PRELIMINARY, full only)
      test_ba : cross-day test, trained on 01-12 and tested on 03-11
    """
    entry = meta["models"][key]
    test = final["results"][entry["schema_name"]]
    random_split = final["results"].get(RANDOM_SPLIT_KEY, {}) if key == "full" else {}

    in_day = {}
    if key == "full" and validation:
        in_day.update(validation["models"])
    if key == "compact" and compact_val:
        in_day.update(compact_val["schemas"]["compact-7 (no SYN)"]["models"])
    if baseline:
        in_day["Baseline: majority class"] = baseline["majority_class"]
        if key == "full" and baseline.get("best_stump_feature"):
            in_day[f"Baseline: stump on {baseline['best_stump_feature']}"] = (
                baseline["stumps_ranked"][0])

    rows = []
    for name, t in test.items():
        rows.append({
            "Model": name,
            "val_ba": in_day.get(name, {}).get("balanced_accuracy", NAN),
            "rand_ba": random_split.get(name, {}).get("balanced_accuracy", NAN),
            "test_ba": t["balanced_accuracy"],
            "test_recall": t["recall"],
            "test_fpr": t["fpr"],
        })
    return pd.DataFrame(rows)


def seen_unseen_table(key, meta, detail):
    """Detection on attack types seen in training (LDAP, Syn) vs never seen."""
    entry = meta["models"][key]
    results = detail["results"][entry["schema_name"]]
    rows = [{
        "Model": name,
        "recall_seen": r["seen"]["recall"],
        "recall_unseen": r["recall_unseen"],
        "fpr": r["all"]["fpr"],
        "ba_seen": r["seen"]["balanced_accuracy"],
        "ba_all": r["all"]["balanced_accuracy"],
    } for name, r in results.items()]
    return pd.DataFrame(rows)


def sensitivity_table(final):
    """Test-day scores when the excluded columns (ports, Inbound) are added."""
    block = final["results"].get(SENSITIVITY_KEY, {})
    return pd.DataFrame([{
        "Model": name, "recall": r["recall"], "fpr": r["fpr"],
        "balanced_accuracy": r["balanced_accuracy"],
    } for name, r in block.items()])


def importance_frame(entry):
    """(DataFrame of feature and value sorted by size, kind of explanation)."""
    rows = entry.get("explanation_top") or []
    df = pd.DataFrame(rows, columns=["feature", "value"])
    order = df["value"].abs().sort_values(ascending=False).index
    return df.loc[order].reset_index(drop=True), entry.get("explanation_kind", "")


def tree_rules(pipe, columns):
    """The tree's decision rules as text, with class names instead of 0 and 1."""
    names = list(pipe.named_steps["pre"].get_feature_names_out(list(columns)))
    text = export_text(pipe.named_steps["model"], feature_names=names,
                       max_depth=8)
    return (text.replace("class: 0", "class: BENIGN")
                .replace("class: 1", "class: DDoS ATTACK"))


# ----------------------------------------------------------- self-check ----
def main():
    from predict import load_model
    from pathlib import Path

    files = {name: load_json(name) for name in (
        "model_metadata.json", "final_evaluation.json",
        "final_evaluation_detail.json", "model_validation.json",
        "baseline_validation.json", "compact_validation.json")}
    absent = [n for n, v in files.items() if v is None]
    if absent:
        raise SystemExit(f"Missing artifact files: {absent}. "
                         "Run the earlier scripts first.")
    meta, final, detail = (files["model_metadata.json"],
                           files["final_evaluation.json"],
                           files["final_evaluation_detail.json"])

    for key in ("full", "compact"):
        entry = meta["models"][key]
        m = entry["test_metrics"]
        ref = final["results"][entry["schema_name"]][entry["model_name"]]
        assert all(m[k] == ref[k] for k in ("tn", "fp", "fn", "tp")), key
        print(f"\n[{key}] {entry['model_name']}  counts tn={m['tn']} fp={m['fp']} "
              f"fn={m['fn']} tp={m['tp']} equal final_evaluation.json   [OK]")

        table = metric_rows(m)
        assert len(table) == 7
        counted = table[table["Metric"] != "ROC-AUC"]
        assert ((counted["_value"] - counted["_saved"]).abs() < 1e-9).all()
        assert (counted["Check against the saved value"] == "matches").all()
        print("  metrics recomputed from the counts match the saved values   [OK]")
        for _, r in table.iterrows():
            print(f"    {r['Metric']:<22}{r['Value']:>9}   {r['Calculation with the test-day counts']}")

        ev = evidence_table(key, meta, final, files["model_validation.json"],
                            files["baseline_validation.json"],
                            files["compact_validation.json"])
        saved_row = ev[ev["Model"] == entry["model_name"]].iloc[0]
        assert abs(saved_row["test_ba"] - m["balanced_accuracy"]) < 1e-12
        if key == "full":
            assert ev["rand_ba"].notna().sum() == 4
        else:
            assert ev["rand_ba"].isna().all()
        print("  evaluation methods compared (balanced accuracy)   [OK]")
        show = ev.copy()
        for c in ("val_ba", "rand_ba", "test_ba"):
            show[c] = show[c].map(lambda v: "n/a" if v != v else f"{v:.3f}")
        print(show[["Model", "val_ba", "rand_ba", "test_ba"]].to_string(index=False))

        su = seen_unseen_table(key, meta, detail)
        assert len(su) == 4
        print("  seen vs unseen attack types (recall: seen / NetBIOS):")
        for _, r in su.iterrows():
            print(f"    {r['Model']:<30}{r['recall_seen']:.3f} / {r['recall_unseen']:.3f}")

        imp, kind = importance_frame(entry)
        assert len(imp) > 0
        if kind == "importance":
            assert imp["value"].sum() <= 1 + 1e-9
            assert (imp["value"] >= 0).all()
        print(f"  top 5 by {kind}:")
        for _, r in imp.head(5).iterrows():
            print(f"    {r['value']:+.4f}   {r['feature']}")

    sens = sensitivity_table(final)
    assert len(sens) == 4
    print("\nSensitivity (excluded columns added), balanced accuracy:")
    for _, r in sens.iterrows():
        print(f"  {r['Model']:<30}{r['balanced_accuracy']:.3f}")

    pipe = load_model("full")
    rules = tree_rules(pipe, meta["models"]["full"]["feature_columns"])
    assert "class: BENIGN" in rules and "class: DDoS ATTACK" in rules
    depth = pipe.named_steps["model"].get_depth()
    assert depth <= 4, depth
    print(f"\nDecision rules of the saved tree (depth {depth}):")
    print(rules)

    fig_dir = Path(__file__).resolve().parent.parent / "artifacts" / "figures"
    for name in ("fig06_confusion_matrices.png", "fig07_roc_curves.png",
                 "fig08_seen_vs_unseen.png"):
        assert (fig_dir / name).exists(), name
    print("Figures fig06, fig07 and fig08 exist   [OK]")
    print("\nALL PERFORMANCE-PAGE CHECKS PASSED")


if __name__ == "__main__":
    main()