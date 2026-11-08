"""
DDoS Attack Detection Dashboard (Streamlit).

Run from the project root:
    streamlit run app.py

This is a model-evaluation and prediction dashboard. It does NOT monitor live
network traffic, and a prediction is a model output, not proof of an attack.
"""

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))   # lets the saved models find our modules

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from dashboard_data import (
    load_json, pct, raw_totals, sample_distribution, types_seen_in_training,
)

import numpy as np

from compact_features import COMPACT_FEATURES
from manual_input import (
    MAX_BYTES, MAX_DURATION_MS, MAX_PACKETS, derive_row, range_warnings,
    validate_inputs,
)
from predict import ModelNotTrainedError, predict_frame

from csv_prediction import (
    MAX_FILE_MB, MAX_ROWS, build_download_frame, check_schema,
    evaluate_against_labels, find_label_column, ground_truth_from_labels,
    protocol_outside_training, read_upload,
)
from data_processing import prepare_features
from dataset_explorer import (
    SAMPLE_PATH, cleaned_copy, label_distribution, list_demo_files,
    load_demo_file, load_sample_frame, missing_summary, numeric_statistics,
    overview, raw_files_table, signed_log,
)

from performance_data import (
    evidence_table, importance_frame, metric_rows, seen_unseen_table,
    sensitivity_table, tree_rules,
)
from predict import load_model

st.set_page_config(page_title="DDoS Detection Dashboard", page_icon="🛡️",
                   layout="wide")

PAGES = ["Dashboard Overview", "Dataset Explorer", "Manual Detection",
         "CSV Prediction", "Model Performance"]
MODEL_CHOICES = {
    "Full schema, CSV upload model": "full",
    "Compact schema, manual-form model": "compact",
}


# ------------------------------------------------------------ helpers ------
def show_setup_instructions():
    st.error("No trained model found yet, so there is nothing to display.")
    st.markdown(
        "Run these commands in the project folder (with the virtual "
        "environment active), then refresh this page. The raw CSV files must "
        "already be in `data\\raw`."
    )
    st.code(
        "python src\\inspect_raw.py\n"
        "python src\\build_sample.py\n"
        "python src\\describe_sample.py\n"
        "python src\\baselines.py\n"
        "python src\\final_evaluation.py\n"
        "python src\\train.py",
        language="powershell",
    )


def draw_confusion(m):
    cm = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["BENIGN", "ATTACK"],
                yticklabels=["BENIGN", "ATTACK"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def section_header(title, kind):
    st.subheader(title)
    st.caption(kind)


# ------------------------------------------------------------- Page A ------
def page_overview():
    st.title("🛡️ DDoS Attack Detection Dashboard")
    st.caption("Machine-learning classification of network-flow records "
               "(CIC-DDoS2019). Every number below is read from files created "
               "by our own scripts.")
    st.info("This is an offline evaluation dashboard. It does not monitor live "
            "traffic, and a prediction is a model output, not proof of an "
            "active attack.")

    meta = load_json("model_metadata.json")
    if meta is None:
        show_setup_instructions()
        return
    raw = load_json("raw_inspection.json")
    profile = load_json("sample_profile.json")
    final = load_json("final_evaluation.json")
    train_day, test_day = meta["train_day"], meta["test_day"]

    # ---- 1. dataset statistics ------------------------------------------
    section_header("1. Dataset statistics",
                   "What is in the data we processed. These are not model "
                   "results.")
    totals = raw_totals(raw) if raw else None
    if totals:
        c = st.columns(4)
        c[0].metric("Raw records scanned", f"{totals['rows']:,}",
                    help=f"Rows read from {totals['files']} official "
                         "CIC-DDoS2019 CSV files (src/inspect_raw.py).")
        c[1].metric("Benign in raw files", f"{totals['benign']:,}")
        c[2].metric("Attack in raw files", f"{totals['attack']:,}")
        c[3].metric("Benign share of raw data",
                    pct(totals["benign"] / totals["rows"], 2))
    else:
        st.warning("raw_inspection.json not found. Run python "
                   "src\\inspect_raw.py to see raw-file statistics.")

    if profile:
        counts = profile["class_counts"]
        c = st.columns(4)
        c[0].metric("Working sample size", f"{profile['rows']:,}",
                    help="Random sample drawn by src/build_sample.py.")
        c[1].metric("Benign in sample", f"{counts.get('BENIGN', 0):,}")
        c[2].metric("Attack in sample", f"{counts.get('ATTACK', 0):,}")
        full = meta["models"]["full"]
        c[3].metric("Model input features",
                    f"{full['n_input_features']} / "
                    f"{meta['models']['compact']['n_input_features']}",
                    help="Full schema / compact schema. The full schema has "
                         f"{full['n_features_after_preprocessing']} features "
                         "left after constant columns are removed.")
        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Rows in the working sample by traffic type and "
                        "capture day**")
            st.bar_chart(sample_distribution(profile))
        with right:
            st.markdown("**Same data as a table**")
            st.dataframe(sample_distribution(profile))
            st.caption(f"Training day: {train_day}. Test day: {test_day}. "
                       "The sample deliberately contains far more benign "
                       "traffic than the raw data.")
    else:
        st.warning("sample_profile.json not found. Run python "
                   "src\\describe_sample.py.")

    # ---- 2. test-set evaluation ----------------------------------------
    st.divider()
    section_header(f"2. Test-set evaluation (trained on {train_day}, "
                   f"tested on {test_day})",
                   "Results on capture-day data the model never saw during "
                   "training.")
    label = st.radio("Model", list(MODEL_CHOICES), horizontal=True)
    key = MODEL_CHOICES[label]
    entry = meta["models"][key]
    m = entry["test_metrics"]

    st.markdown(f"**{entry['model_name']}**, {entry['schema_name']}, trained "
                f"on {entry['trained_on']['rows']:,} rows from {train_day}.")
    c = st.columns(6)
    c[0].metric("Accuracy", pct(m["accuracy"]))
    c[1].metric("Precision (attack)", pct(m["precision"]))
    c[2].metric("Recall (attack)", pct(m["recall"]),
                help="Share of real attacks that were detected.")
    c[3].metric("F1-score (attack)", pct(m["f1"]))
    c[4].metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    c[5].metric("False-positive rate", pct(m["fpr"], 2),
                help="Share of benign flows wrongly flagged as attack.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix** (test day)")
        fig = draw_confusion(m)
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"Missed attacks (false negatives): {m['fn']:,}. "
                   f"False alarms (false positives): {m['fp']:,}.")
    with right:
        st.markdown("**Detection rate by traffic type** (test day)")
        seen = (types_seen_in_training(profile, train_day)
                if profile else set())
        rows = []
        for kind, rate in entry["test_recall_by_type"].items():
            if profile:
                flag = "yes" if kind in seen else "NO, never seen"
            else:
                flag = "n/a"
            rows.append({"Traffic type": kind, "In training data?": flag,
                         "Detected": pct(rate)})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        nd = entry["test_non_duplicate"]
        st.markdown(f"**On test rows that are not exact copies of a training "
                    f"row** ({nd['rows']:,} rows)")
        st.write(f"Recall {pct(nd['recall'])}, false-positive rate "
                 f"{pct(nd['fpr'], 2)}, balanced accuracy "
                 f"{pct(nd['balanced_accuracy'])}.")

    with st.expander("How to read these results (limitations)"):
        n_test = m["tn"] + m["fp"] + m["fn"] + m["tp"]
        ident = entry["test_rows_identical_to_training"]
        benign_share = (pct(totals["benign"] / totals["rows"], 2)
                        if totals else "far lower")
        st.markdown(
            f"- **Precision and accuracy depend on the sample mix.** Our "
            f"working sample is 20% benign, while the raw data is "
            f"{benign_share} benign. Real-world precision would be lower at "
            f"the same false-positive rate. Recall and false-positive rate "
            f"transfer better.\n"
            f"- **These models were chosen after seeing the test results.** "
            f"The selection rule was fixed beforehand, but the scores are "
            f"slightly optimistic.\n"
            f"- **One attack type is unseen.** NetBIOS never appears in "
            f"training; detection of it varies a lot between models and "
            f"should not be generalised.\n"
            f"- **Some test flows repeat training flows.** {ident:,} of "
            f"{n_test:,} test rows have identical model-input values to a "
            f"training row, which makes those rows easier.\n"
            f"- **Offline only.** Nothing here is live monitoring, and a "
            f"prediction is not proof of an attack.")

    # ---- 3. model comparison -------------------------------------------
    st.divider()
    section_header("3. Model comparison on the test day",
                   "All candidate models, trained on the same training day "
                   "and scored on the same test day.")
    if final is None:
        st.warning("final_evaluation.json not found. Run python "
                   "src\\final_evaluation.py.")
    else:
        schemas = [s for s in final["results"]
                   if s.startswith(("full", "compact"))]
        choice = st.selectbox("Feature schema", schemas)
        res = final["results"][choice]
        table = pd.DataFrame({
            name: {"Recall": r["recall"],
                   "False-positive rate": r["fpr"],
                   "Balanced accuracy": r["balanced_accuracy"],
                   "ROC-AUC": r["roc_auc"]}
            for name, r in res.items()
        }).T
        chart = table[["Recall", "False-positive rate", "Balanced accuracy"]]
        fig, ax = plt.subplots(figsize=(9, 0.9 * len(chart) + 1.5))
        chart.iloc[::-1].plot(kind="barh", ax=ax, width=0.8,
                              color=["#e76f51", "#264653", "#2a9d8f"])
        ax.set_xlim(0, 1.4)
        ax.set_xlabel("Value from 0 to 1 (higher is better, except "
                      "false-positive rate)")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        shown = pd.DataFrame(index=table.index)
        for col in ["Recall", "False-positive rate", "Balanced accuracy"]:
            shown[col] = (table[col] * 100).map("{:.2f}%".format)
        shown["ROC-AUC"] = table["ROC-AUC"].map("{:.3f}".format)
        shown.insert(0, "Saved in dashboard", [
            "yes" if (name == entry_["model_name"]
                      and choice == entry_["schema_name"]) else ""
            for name in shown.index
            for entry_ in [meta["models"][
                "full" if choice.startswith("full") else "compact"]]
        ])
        st.dataframe(shown)
        st.caption("Baselines are the majority-class rule and a single-feature "
                   "threshold. A real model should clearly beat them.")

    # ---- 4. predictions on new records ---------------------------------
    st.divider()
    section_header("4. Predictions on new records",
                   "Separate from the results above.")
    st.info("No predictions are made on this page. Use **Manual Detection** "
            "to test one flow, or **CSV Prediction** to test many.")


# ------------------------------------------------------------- Page C ------
def fmt_number(value):
    """Readable number for tables; undefined (NaN or infinite) values are labelled."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "undefined"
    if number.is_integer():
        return f"{int(number):,}"
    if number != 0 and abs(number) < 0.01:
        return f"{number:.3g}"
    return f"{number:,.2f}"


def fmt_probability(p):
    """Percentage text that never implies certainty."""
    if p < 0.001:
        return "<0.1%"
    if p > 0.999:
        return ">99.9%"
    return f"{p * 100:.1f}%"


def page_manual_detection():
    st.title("Manual Network-Flow Detection")
    st.info("Enter summary values for ONE network flow and press DETECT. The "
            "result is a model prediction, not proof of an active attack. The "
            "pre-filled values are demonstration inputs, not live "
            "measurements.")

    meta = load_json("model_metadata.json")
    if meta is None:
        show_setup_instructions()
        return
    entry = meta["models"]["compact"]
    if list(entry["feature_columns"]) != list(COMPACT_FEATURES):
        st.error("Feature-schema mismatch: the saved model was trained on "
                 "different columns than this dashboard expects, so "
                 "prediction is disabled. Retrain with python src\\train.py "
                 "or update src\\compact_features.py.")
        st.write("Saved model expects:", entry["feature_columns"])
        st.write("Dashboard expects:", list(COMPACT_FEATURES))
        return

    trained_protocols = {int(p) for p in entry["protocol_values_in_training"]}
    protocol_choices = {"TCP": 6, "UDP": 17,
                        "Other (not supported by this model)": None}

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Flow values")
        st.caption("Pre-filled with DEMONSTRATION values (850 packets in 1.5 "
                   "seconds). They are not measurements of real traffic.")
        proto_label = st.selectbox("Protocol", list(protocol_choices),
                                   key="m_protocol")
        duration_ms = st.number_input(
            "Flow duration (milliseconds)", min_value=0.0,
            max_value=float(MAX_DURATION_MS), value=1500.0, step=100.0,
            format="%.3f", key="m_duration_ms",
            help="The model works in microseconds; the form converts "
                 "automatically.")
        packets = st.number_input(
            "Total packets (forward + backward)", min_value=1,
            max_value=MAX_PACKETS, value=850, step=1, key="m_packets")
        total_bytes = st.number_input(
            "Total bytes (forward + backward)", min_value=0,
            max_value=MAX_BYTES, value=124000, step=1, key="m_bytes")
        ack_label = st.selectbox(
            "ACK flag count", ["1 (ACK flag seen)", "0 (ACK flag not seen)"],
            key="m_ack",
            help="In our data sample this count is only ever 0 or 1.")
    ack = 1 if ack_label.startswith("1") else 0
    protocol_number = protocol_choices[proto_label]

    with right:
        st.subheader("Calculated by the form")
        preview = derive_row(duration_ms, packets, total_bytes, ack,
                             protocol_number or 0)
        st.metric("Flow duration sent to the model",
                  f"{preview.at[0, 'Flow Duration']:,.0f} µs")
        st.metric("Packets per second",
                  fmt_number(preview.at[0, "Flow Packets/s"]))
        st.metric("Bytes per second",
                  fmt_number(preview.at[0, "Flow Bytes/s"]))
        st.caption("The rates are calculated from the totals and the "
                   "duration, the way the dataset defines them. They are not "
                   "typed in, so they can never contradict the totals.")
        with st.expander("What the model does not use"):
            st.markdown(
                f"- **SYN flag count:** {meta['compact_schema_notes']['SYN Flag Count']}.\n"
                "- **IP addresses, ports, timestamps and flow identifiers** "
                "are excluded on purpose. They identify this lab network, "
                "not traffic behaviour.\n"
                "- **Other protocols:** only the protocols seen in training "
                "are accepted.")

    detect = st.button("DETECT", type="primary")
    if not detect:
        st.caption("Press DETECT to run the saved model on these values.")
        return

    # ---- validate ---------------------------------------------------------
    if protocol_number is None or protocol_number not in trained_protocols:
        st.warning("This protocol is not supported. The training data "
                   "contains only TCP, UDP and one small group of protocol-0 "
                   "records, so a prediction for any other protocol would be "
                   "a guess. Choose TCP or UDP.")
        return
    errors = validate_inputs(duration_ms, packets, total_bytes, ack,
                             protocol_number, trained_protocols)
    if errors:
        st.error("The entered values are not valid, so nothing was predicted:")
        for message in errors:
            st.write(f"- {message}")
        return

    # ---- predict ----------------------------------------------------------
    row = derive_row(duration_ms, packets, total_bytes, ack, protocol_number)
    try:
        out = predict_frame(row, "compact")
    except ModelNotTrainedError:
        show_setup_instructions()
        return
    except ValueError as err:
        st.error(f"The input does not match the model's feature schema: {err}")
        return
    except Exception as err:   # keep the dashboard alive whatever happens
        st.error(f"Prediction failed ({type(err).__name__}): {err}")
        return

    result = out.iloc[0]
    probability = float(result["attack_probability"])
    is_attack = int(result["prediction"]) == 1
    label = result["prediction_label"]

    st.divider()
    st.subheader("Result")
    if is_attack:
        st.error("## 🚨 DDoS ATTACK\nThe model classified this flow as attack "
                 "traffic.")
    else:
        st.success("## ✅ BENIGN\nThe model classified this flow as "
                   "legitimate traffic.")

    c1, c2 = st.columns([1, 3])
    c1.metric("Estimated attack probability", fmt_probability(probability))
    with c2:
        st.progress(min(max(probability, 0.0), 1.0))
        trained = entry["trained_on"]
        attack_share = trained["attack"] / trained["rows"]
        st.caption(
            f"This is the model's score, not a calibrated real-world "
            f"probability. The training sample was {attack_share * 100:.0f}% "
            f"attack and the model used class weights, so scores do not "
            f"reflect how common attacks are on a real network.")

    proto_name = proto_label.split()[0]
    st.markdown(
        f"**Summary.** For a {proto_name} flow lasting "
        f"{fmt_number(duration_ms)} ms with {packets:,} packets "
        f"({total_bytes:,} bytes) and an ACK count of {ack}, the saved "
        f"{entry['model_name']} gives an attack probability of "
        f"{fmt_probability(probability)} and classifies it as **{label}** "
        f"(decision threshold {meta['decision_threshold'] * 100:.0f}%).")

    # ---- warnings ---------------------------------------------------------
    if duration_ms == 0:
        message = ("The flow duration is 0, so packets per second and bytes "
                   "per second are undefined. The saved pipeline fills them "
                   "with the training-day median, as it did for such rows "
                   "in training.")
        zero = load_json("fill_strategy_validation.json")
        if zero:
            z = zero["zero_duration_rows"]
            message += (f" In the training day, {z['attack']:,} of the "
                        f"{z['attack'] + z['benign']:,} zero-duration flows "
                        f"were attacks, so the model may lean towards ATTACK "
                        f"for this input. That reflects how this dataset was "
                        f"captured, not a general rule about networks.")
        st.warning(message)

    outside = range_warnings(row, entry.get("training_ranges") or {})
    if outside:
        lines = [f"- {col}: {fmt_number(value)} (training range "
                 f"{fmt_number(lo)} to {fmt_number(hi)})"
                 for col, value, lo, hi in outside]
        st.warning("Some values are outside the range seen in training "
                   "(flow duration is in microseconds here), so this "
                   "prediction is an extrapolation and less reliable:\n\n"
                   + "\n".join(lines))

    # ---- what was entered, and what the model received ---------------------
    left, right = st.columns(2)
    with left:
        st.markdown("**Values you entered**")
        st.dataframe(pd.DataFrame({
            "Field": ["Protocol", "Flow duration (ms)", "Total packets",
                      "Total bytes", "ACK flag count"],
            "Value": [proto_name, fmt_number(duration_ms), f"{packets:,}",
                      f"{total_bytes:,}", str(ack)],
        }), hide_index=True)
    with right:
        st.markdown("**Exact row sent to the model**")
        st.dataframe(pd.DataFrame({
            "Model feature": [c + (" (µs)" if c == "Flow Duration" else "")
                              for c in COMPACT_FEATURES],
            "Value": [fmt_number(row.at[0, c]) for c in COMPACT_FEATURES],
        }), hide_index=True)

    with st.expander("How this prediction was made, and how far to trust it"):
        m = entry["test_metrics"]
        netbios = entry["test_recall_by_type"].get("NetBIOS")
        st.markdown(
            f"1. Your values were checked, the duration was converted from "
            f"milliseconds to microseconds, and the two rates were "
            f"calculated.\n"
            f"2. The row was passed through the same saved pipeline used in "
            f"training: infinite or undefined values are filled with the "
            f"training median, constant columns are dropped, and inputs are "
            f"standardised.\n"
            f"3. {entry['model_name']} produced the attack probability, and "
            f"the {meta['decision_threshold'] * 100:.0f}% threshold turned it "
            f"into a label.\n\n"
            f"**Reliability.** On the unseen test day ({meta['test_day']}), "
            f"this model detected {pct(m['recall'])} of attacks and wrongly "
            f"flagged {pct(m['fpr'], 2)} of benign flows. It detected "
            f"{pct(netbios) if netbios is not None else 'n/a'} of NetBIOS "
            f"attacks, a type it never saw in training. One flow's summary "
            f"is far less information than the traffic around it, and a real "
            f"DDoS attack involves very large numbers of flows. Treat this "
            f"as a model output, not as proof of an attack.")


# ------------------------------------------------------------- Page D ------
@st.cache_data(show_spinner="Reading and scoring the file ...")
def score_upload(file_bytes, model_stamp):
    """
    Read, check and score an uploaded CSV. Cached, so clicking a widget does
    not score the file again. model_stamp is part of the cache key, so
    retraining invalidates old results.
    """
    df, truncated = read_upload(io.BytesIO(file_bytes), MAX_ROWS)
    report = check_schema(df)
    results = None if report["missing"] else predict_frame(df, "full")
    csv_bytes = None
    if results is not None:
        csv_bytes = build_download_frame(df, results).to_csv(
            index=False).encode("utf-8")
    return df, truncated, report, results, csv_bytes


def page_csv_prediction():
    st.title("CSV-Based Prediction")
    st.info("Upload a CSV of network-flow records and the saved model will "
            "predict every row. Predictions are model outputs, not proof of "
            "attacks. No labels are needed to predict.")

    meta = load_json("model_metadata.json")
    if meta is None:
        show_setup_instructions()
        return
    entry = meta["models"]["full"]
    allowed = {int(p) for p in entry["protocol_values_in_training"]}
    st.caption(
        f"Model: {entry['model_name']}, trained on "
        f"{entry['trained_on']['rows']:,} rows from {meta['train_day']}. "
        "Uploaded files are processed in memory. This page never saves your "
        "file and never changes anything in data\\raw.")

    with st.expander("Which files can I upload?"):
        st.markdown(
            f"- A CSV that contains the {len(entry['feature_columns'])} "
            f"columns listed below. Names are matched after removing spaces, "
            f"column order does not matter, and extra columns (IP addresses, "
            f"timestamps, labels) are ignored.\n"
            f"- Values must be numbers. **Flow Duration is in microseconds**, "
            f"as in the raw CIC-DDoS2019 files. Empty or infinite values in "
            f"the rate columns are handled as in training.\n"
            f"- No label column is needed.\n"
            f"- Limits: {MAX_FILE_MB} MB, and only the first {MAX_ROWS:,} rows "
            f"are used, to keep the app responsive on a laptop.\n"
            f"- The first rows of a raw CIC file are usually one attack type "
            f"from one short time window, so they say little about the model. "
            f"To try this page, use the demo files created by "
            f"`python src\\make_demo_csv.py` (folder `data\\processed\\demo`).")
        st.code(", ".join(entry["feature_columns"]), language="text")

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is None:
        st.caption("No file uploaded yet.")
        return
    if uploaded.size > MAX_FILE_MB * 1024 * 1024:
        st.error(f"This file is {uploaded.size / 1024 ** 2:,.1f} MB and the "
                 f"limit is {MAX_FILE_MB} MB. Upload a smaller sample.")
        return

    try:
        df, truncated, report, results, csv_bytes = score_upload(
            uploaded.getvalue(), meta["created_utc"])
    except ModelNotTrainedError:
        show_setup_instructions()
        return
    except ValueError as err:
        st.error(f"This file cannot be used: {err}")
        return
    except Exception as err:   # keep the dashboard alive whatever happens
        st.error(f"Unexpected problem while reading or scoring the file "
                 f"({type(err).__name__}): {err}")
        return

    # ---- 1. file check ------------------------------------------------------
    st.subheader("1. File check")
    found = report["n_required"] - len(report["missing"])
    c = st.columns(3)
    c[0].metric("Rows read", f"{len(df):,}")
    c[1].metric("Columns in file", f"{df.shape[1]:,}")
    c[2].metric("Required columns found", f"{found} / {report['n_required']}")
    if truncated:
        st.warning(f"The file has more than {MAX_ROWS:,} rows. Only the first "
                   f"{MAX_ROWS:,} were used and the rest were ignored.")
    if report["missing"]:
        st.error(f"{len(report['missing'])} required column(s) are missing, "
                 "so nothing was predicted. Missing columns:")
        st.code(", ".join(report["missing"]), language="text")
        return
    st.success(f"All {report['n_required']} required columns are present. "
               f"{len(report['extra'])} other column(s) were ignored.")
    if report["extra"]:
        with st.expander("Columns that were ignored"):
            st.write(", ".join(report["extra"]))
    n_outside = protocol_outside_training(df, allowed)
    if n_outside:
        st.warning(f"{n_outside:,} row(s) have a Protocol value that never "
                   "occurred in the training data. They were still predicted, "
                   "but treat those predictions with extra caution.")

    # ---- 2. summary of the predictions --------------------------------------
    st.divider()
    st.subheader("2. Prediction summary")
    st.caption("These are predictions on your file, not evaluation results.")
    n = len(results)
    n_attack = int((results["prediction"] == 1).sum())
    c = st.columns(4)
    c[0].metric("Rows predicted", f"{n:,}")
    c[1].metric("Predicted BENIGN", f"{n - n_attack:,}")
    c[2].metric("Predicted DDoS ATTACK", f"{n_attack:,}")
    c[3].metric("Share flagged as attack", pct(n_attack / n))
    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Estimated attack probability of every row**")
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.hist(results["attack_probability"], bins=20, range=(0, 1),
                color="#e76f51")
        ax.set_xlabel("Estimated attack probability")
        ax.set_ylabel("Rows")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**What this means**")
        st.write(
            f"A row is classified as DDoS ATTACK when its probability is at "
            f"least {meta['decision_threshold'] * 100:.0f}%. The probabilities "
            f"are model scores, not calibrated real-world chances: the "
            f"training sample was 80% attack. Without known labels, accuracy "
            f"cannot be calculated, so none is shown here.")

    # ---- 3. predicted records --------------------------------------------------
    st.divider()
    st.subheader("3. Predicted records")
    choice = st.radio("Show", ["All rows", "Predicted ATTACK only",
                               "Predicted BENIGN only"], horizontal=True)
    by_probability = st.checkbox("Sort by attack probability (highest first)")
    numeric = prepare_features(df)
    view = pd.DataFrame({
        "row_number": np.arange(1, n + 1),
        "prediction_label": results["prediction_label"].to_numpy(),
        "attack_probability": results["attack_probability"].round(4).to_numpy(),
    })
    label_col = find_label_column(df)
    if label_col:
        view[f"{label_col} (from your file)"] = df[label_col].astype(str).to_numpy()
    for col in ("Protocol", "Flow Duration", "Total Fwd Packets",
                "Total Backward Packets", "Total Length of Fwd Packets",
                "Total Length of Bwd Packets"):
        view[col] = numeric[col].to_numpy()
    if choice.startswith("Predicted ATTACK"):
        view = view[view["prediction_label"] == "DDoS ATTACK"]
    elif choice.startswith("Predicted BENIGN"):
        view = view[view["prediction_label"] == "BENIGN"]
    if by_probability:
        view = view.sort_values("attack_probability", ascending=False)
    st.dataframe(view.head(1000), hide_index=True)
    st.caption(f"Showing {min(len(view), 1000):,} of {len(view):,} matching "
               f"rows (the table is limited to 1,000 rows for speed). The "
               f"download below contains every row.")
    st.download_button(
        "Download all predictions as CSV", data=csv_bytes,
        file_name="ddos_predictions.csv", mime="text/csv")
    st.caption("The download contains your original columns plus row_number, "
               "prediction (0 or 1), prediction_label and attack_probability.")

    # ---- 4. optional evaluation ---------------------------------------------
    st.divider()
    st.subheader("4. Optional: evaluate against labels in your file")
    if label_col is None:
        st.caption("No label column (Label, label_original or label_binary) "
                   "was found, so no accuracy can be calculated. Accuracy is "
                   "never computed from predictions alone.")
        return
    y, table = ground_truth_from_labels(df[label_col])
    st.write(f"Column **{label_col}** was found. This is how its values would "
             "be read (BENIGN is 0, any other label is an attack, empty "
             "values are ignored):")
    st.dataframe(table, hide_index=True)
    if not st.checkbox("These labels are correct: evaluate the model on this "
                       "file"):
        return
    try:
        m, n_known = evaluate_against_labels(y, results)
    except ValueError as err:
        st.error(str(err))
        return

    n_benign_known = m["tn"] + m["fp"]
    n_attack_known = m["fn"] + m["tp"]
    st.markdown(f"**Evaluation on {n_known:,} labelled rows** "
                f"({n_benign_known:,} benign, {n_attack_known:,} attack)")
    c = st.columns(6)
    c[0].metric("Accuracy", pct(m["accuracy"]))
    c[1].metric("Precision (attack)",
                pct(m["precision"]) if (m["tp"] + m["fp"]) else "n/a")
    c[2].metric("Recall (attack)", pct(m["recall"]) if n_attack_known else "n/a")
    c[3].metric("F1-score (attack)", pct(m["f1"]) if n_attack_known else "n/a")
    c[4].metric("ROC-AUC",
                f"{m['roc_auc']:.3f}" if m["roc_auc"] == m["roc_auc"] else "n/a")
    c[5].metric("False-positive rate",
                pct(m["fpr"], 2) if n_benign_known else "n/a")
    left, right = st.columns(2)
    with left:
        fig = draw_confusion(m)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.caption(
            "These numbers describe the model on THIS file only. If the file "
            f"contains flows from the training day ({meta['train_day']}), the "
            "model has already seen similar rows and the numbers are "
            "optimistic. Metrics shown as n/a cannot be calculated because "
            "the file lacks benign or attack rows, or has too few of one "
            "class for the metric to exist.")



# ------------------------------------------------------------- Page B ------
SAMPLE_CHOICE = "Working sample (20,000 rows)"


@st.cache_data(show_spinner="Loading the working sample ...")
def cached_sample(file_stamp):
    return load_sample_frame()


@st.cache_data(show_spinner="Reading the demo file ...")
def cached_demo(name, file_stamp):
    return load_demo_file(name)


@st.cache_data(show_spinner="Reading the uploaded file ...")
def cached_upload(file_bytes):
    return read_upload(io.BytesIO(file_bytes), MAX_ROWS)


def render_explorer(df, name, truncated):
    """Everything below the dataset chooser."""
    ov = overview(df)

    # ---- 1. overview ------------------------------------------------------
    st.subheader("1. Overview")
    st.caption(f"Dataset: {name}")
    if truncated:
        st.warning(f"The file has more than {MAX_ROWS:,} rows. Only the first "
                   f"{MAX_ROWS:,} are shown and analysed.")
    c = st.columns(5)
    c[0].metric("Rows", f"{ov['rows']:,}")
    c[1].metric("Columns", f"{ov['columns']:,}")
    c[2].metric("Numeric / text columns",
                f"{ov['numeric_columns']} / {ov['text_columns']}")
    c[3].metric("Memory", f"{ov['memory_mb']:.1f} MB")
    c[4].metric("Rows identical to another row", f"{ov['identical_rows']:,}",
                help=f"Compared on {ov['columns_compared']} numeric columns "
                     "(row positions and label copies are ignored).")

    # ---- 2. preview ----------------------------------------------------------
    st.divider()
    st.subheader("2. Preview")
    n_show = st.slider("Rows to show", 5, 100, 20, key="ex_rows")
    random_rows = st.checkbox("Show random rows instead of the first rows",
                              key="ex_random")
    shown = (df.sample(n=min(n_show, len(df)), random_state=0)
             if random_rows else df.head(n_show))
    st.dataframe(shown)

    # ---- 3. label distribution ------------------------------------------------
    st.divider()
    st.subheader("3. Label distribution")
    label_col = find_label_column(df)
    if label_col is None:
        st.caption("No label column (Label, label_original or label_binary) "
                   "was found, so there is no label distribution. This is "
                   "normal for files meant only for prediction.")
    else:
        table = label_distribution(df, label_col)
        left, right = st.columns([2, 3])
        with left:
            st.markdown(f"**Values in column `{label_col}`**")
            st.dataframe(table.assign(Share=table["Share"].map("{:.2%}".format)),
                         hide_index=True)
        with right:
            st.bar_chart(table.set_index("Label value")["Rows"])
        y, _ = ground_truth_from_labels(df[label_col])
        n_benign, n_attack = int((y == 0).sum()), int((y == 1).sum())
        c = st.columns(3)
        c[0].metric("BENIGN (0)", f"{n_benign:,}")
        c[1].metric("Attack (1)", f"{n_attack:,}")
        c[2].metric("Rows without a usable label", f"{int(np.isnan(y).sum()):,}")
        st.caption("BENIGN counts as 0 and every other label as an attack, "
                   "the same rule as on the CSV Prediction page.")
        if {"attack_type", "day"} <= set(df.columns):
            with st.expander("Traffic type by capture day"):
                st.dataframe(pd.crosstab(df["attack_type"], df["day"],
                                         margins=True))

    # ---- 4. missing and infinite values -----------------------------------------
    st.divider()
    st.subheader("4. Missing and infinite values")
    miss, totals = missing_summary(df)
    c = st.columns(3)
    c[0].metric("Columns with a problem", f"{totals['columns_with_issues']:,}")
    c[1].metric("Rows with a problem", f"{totals['rows_with_issues']:,}")
    c[2].metric("Share of rows", pct(totals["share_rows_with_issues"], 2))
    if miss.empty:
        st.success("No missing, blank or infinite values were found.")
    else:
        st.dataframe(miss.assign(
            **{"Share of rows": miss["Share of rows"].map("{:.2%}".format)}),
            hide_index=True)
        if {"Flow Bytes/s", "Flow Packets/s"} & set(miss["Column"]):
            st.caption("In CIC-DDoS2019, empty or infinite values appear in "
                       "the two rate columns when Flow Duration is 0, because "
                       "the rate would divide by zero. The model's pipeline "
                       "replaces them with the median of the training data "
                       "instead of dropping the rows.")

    # ---- 5. statistics -----------------------------------------------------------
    st.divider()
    st.subheader("5. Feature statistics")
    stats = numeric_statistics(df)
    st.caption("Numeric columns only. Infinite values are ignored. Text "
               "columns such as IP addresses are not summarised.")
    display_stats = stats.round(3)
    roles = {"label_binary": "target label, not a feature",
             "source_row_id": "row position, not a feature",
             "row_number": "row position, not a feature"}
    display_stats.insert(1, "Role", display_stats["Column"].map(roles).fillna(""))
    st.dataframe(display_stats, hide_index=True)

    if not stats.empty:
        names = stats["Column"].tolist()
        feature = st.selectbox(
            "Show the distribution of one feature", names,
            index=names.index("Flow Duration") if "Flow Duration" in names else 0)
        values = df[feature].to_numpy(dtype="float64", na_value=np.nan)
        finite = np.isfinite(values)
        if not finite.any():
            st.caption("This column has no finite values to plot.")
        else:
            transformed = signed_log(values)
            lo, hi = transformed[finite].min(), transformed[finite].max()
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            bins = np.linspace(lo, hi, 41)
            fig, ax = plt.subplots(figsize=(7.5, 3.6))
            if label_col is not None:
                y, _ = ground_truth_from_labels(df[label_col])
                for value, label, color in ((0, "BENIGN", "#2a9d8f"),
                                            (1, "ATTACK", "#e76f51")):
                    selected = finite & (y == value)
                    if selected.any():
                        ax.hist(transformed[selected], bins=bins, alpha=0.6,
                                density=True, color=color, label=label)
                ax.legend(fontsize=8)
            else:
                ax.hist(transformed[finite], bins=bins, color="#264653")
            ax.set_xlabel(f"{feature}: signed log10(1 + |value|)")
            ax.set_ylabel("Density")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("A log scale is used because traffic features span "
                       "many orders of magnitude. Infinite values are left out.")

    # ---- 6. downloads --------------------------------------------------------------
    st.divider()
    st.subheader("6. Download results")
    st.caption("These are separate files you save yourself. Nothing here "
               "modifies the original dataset.")
    c = st.columns(3)
    c[0].download_button("Feature statistics (CSV)",
                         stats.to_csv(index=False).encode("utf-8"),
                         "feature_statistics.csv", "text/csv")
    c[1].download_button("Missing-value summary (CSV)",
                         miss.to_csv(index=False).encode("utf-8"),
                         "missing_value_summary.csv", "text/csv")
    if label_col is not None:
        c[2].download_button(
            "Label distribution (CSV)",
            label_distribution(df, label_col).to_csv(index=False).encode("utf-8"),
            "label_distribution.csv", "text/csv")
    if st.checkbox("Prepare a cleaned copy of this dataset (infinite values "
                   "replaced by blanks)"):
        st.download_button("Download cleaned copy (CSV)",
                           cleaned_copy(df).to_csv(index=False).encode("utf-8"),
                           "cleaned_dataset.csv", "text/csv")


def page_dataset_explorer():
    st.title("Dataset Explorer")
    st.info("Look inside a dataset: its size, labels, missing values and "
            "feature statistics. Nothing on this page changes the original "
            "data. Uploaded files are held in memory and never saved.")

    tab_data, tab_raw = st.tabs(["Explore a dataset",
                                 "Raw CIC-DDoS2019 files (pre-computed)"])

    with tab_raw:
        raw = load_json("raw_inspection.json")
        if raw is None:
            st.warning("raw_inspection.json not found. Run python "
                       "src\\inspect_raw.py to scan the raw files.")
        else:
            totals = raw_totals(raw)
            st.caption("These numbers come from an earlier scan of the four "
                       "raw CSV files. The files themselves are up to 1.8 GB "
                       "each and are never loaded into this app.")
            c = st.columns(4)
            c[0].metric("Raw files scanned", f"{totals['files']}")
            c[1].metric("Rows", f"{totals['rows']:,}")
            c[2].metric("BENIGN rows", f"{totals['benign']:,}")
            c[3].metric("Benign share", pct(totals["benign"] / totals["rows"], 2))
            st.dataframe(raw_files_table(raw), hide_index=True)
            st.caption("Benign traffic is a tiny fraction of the raw data, "
                       "which is why our working sample was rebalanced to "
                       "20% benign. Duplicate counts are hash-based "
                       "estimates within each file.")

    with tab_data:
        source = st.radio("Dataset", [SAMPLE_CHOICE, "Demo file",
                                      "Upload a CSV file"], horizontal=True)
        df, truncated, name = None, False, ""
        try:
            if source == SAMPLE_CHOICE:
                if not SAMPLE_PATH.exists():
                    st.error("The working sample was not found. Create it "
                             "with: python src\\build_sample.py")
                    return
                df = cached_sample(SAMPLE_PATH.stat().st_mtime_ns)
                name = ("working sample drawn from the raw CIC-DDoS2019 files "
                        "(data\\processed\\sample_20k.parquet)")
            elif source == "Demo file":
                demos = list_demo_files()
                if not demos:
                    st.error("No demo files found. Create them with: "
                             "python src\\make_demo_csv.py")
                    return
                choice = st.selectbox("Demo file", demos)
                stamp = (ROOT / "data" / "processed" / "demo" / choice
                         ).stat().st_mtime_ns
                df, truncated = cached_demo(choice, stamp)
                name = f"demo file {choice}"
            else:
                uploaded = st.file_uploader("Upload a CSV file", type=["csv"],
                                            key="explorer_upload")
                if uploaded is None:
                    st.caption("No file uploaded yet.")
                    return
                if uploaded.size > MAX_FILE_MB * 1024 * 1024:
                    st.error(f"This file is {uploaded.size / 1024 ** 2:,.1f} "
                             f"MB and the limit is {MAX_FILE_MB} MB.")
                    return
                df, truncated = cached_upload(uploaded.getvalue())
                name = f"uploaded file {uploaded.name} (kept in memory only)"
        except ValueError as err:
            st.error(f"This file cannot be used: {err}")
            return
        except Exception as err:   # keep the dashboard alive whatever happens
            st.error(f"Unexpected problem while reading the file "
                     f"({type(err).__name__}): {err}")
            return
        render_explorer(df, name, truncated)


# ------------------------------------------------------------- Page E ------
FIG_DIR = ROOT / "artifacts" / "figures"


def pct_or_na(value, digits=1):
    return "n/a" if value is None or value != value else pct(value, digits)


def page_model_performance():
    st.title("Model Performance")
    st.info("Every number on this page is read from files written by our "
            "training and evaluation scripts, or recomputed from the saved "
            "confusion-matrix counts. Nothing is typed in by hand.")

    meta = load_json("model_metadata.json")
    final = load_json("final_evaluation.json")
    if meta is None or final is None:
        show_setup_instructions()
        return
    detail = load_json("final_evaluation_detail.json")
    validation = load_json("model_validation.json")
    baseline = load_json("baseline_validation.json")
    compact_val = load_json("compact_validation.json")

    label = st.radio("Model", list(MODEL_CHOICES), horizontal=True,
                     key="perf_model")
    key = MODEL_CHOICES[label]
    entry = meta["models"][key]
    m = entry["test_metrics"]
    total = m["tn"] + m["fp"] + m["fn"] + m["tp"]
    st.markdown(
        f"**{entry['model_name']}**, {entry['schema_name']} schema. Trained "
        f"on {entry['trained_on']['rows']:,} rows from {meta['train_day']} "
        f"and evaluated once on {total:,} rows from {meta['test_day']}, a "
        f"different capture day.")
    with st.expander("Where do these numbers come from?"):
        st.markdown(
            f"1. `python src\\train.py` trains the model on the "
            f"{meta['train_day']} day only.\n"
            f"2. It predicts every {meta['test_day']} flow and saves the "
            f"four confusion-matrix counts to `artifacts\\model_metadata.json`.\n"
            f"3. Before saving, it checks that the counts equal the ones in "
            f"`artifacts\\final_evaluation.json`, written earlier by "
            f"`src\\final_evaluation.py`.\n"
            f"4. This page recomputes every percentage from those counts, so "
            f"a mismatch would show as DIFFERS in the table below.")

    # ---- 1. metrics -----------------------------------------------------------
    st.divider()
    st.subheader("1. Metrics on the unseen test day")
    table = metric_rows(m)
    st.table(table[[c for c in table.columns if not c.startswith("_")]]
             .set_index("Metric"))
    if (table["Check against the saved value"] == "DIFFERS").any():
        st.error("A recomputed metric differs from the saved value. Retrain "
                 "with python src\\train.py.")

    # ---- 2. confusion matrix -------------------------------------------------
    st.divider()
    st.subheader("2. Confusion matrix")
    left, right = st.columns([2, 3])
    with left:
        fig = draw_confusion(m)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown(
            f"In DDoS detection:\n\n"
            f"- **True negatives ({m['tn']:,})**: legitimate flows correctly "
            f"passed as BENIGN.\n"
            f"- **False positives ({m['fp']:,})**: legitimate flows wrongly "
            f"flagged as attack. These are false alarms.\n"
            f"- **False negatives ({m['fn']:,})**: real attack flows the "
            f"model called BENIGN. These are the dangerous errors.\n"
            f"- **True positives ({m['tp']:,})**: attack flows correctly "
            f"flagged.")
        st.caption("Rows are the actual class, columns are the model's "
                   "prediction.")

    # ---- 3. evaluation methods -------------------------------------------------
    st.divider()
    st.subheader("3. How much does the evaluation method matter?")
    st.caption("Balanced accuracy of the same models under three evaluation "
               "methods. Only the last column is a test on a different "
               "capture day.")
    ev = evidence_table(key, meta, final, validation, baseline, compact_val)
    shown = pd.DataFrame({
        "Model": [n + ("  (saved in dashboard)" if n == entry["model_name"]
                       else "") for n in ev["Model"]],
        "Validation inside the training day": ev["val_ba"].map(pct_or_na),
        "Random 70/30 split (preliminary)": ev["rand_ba"].map(pct_or_na),
        "Cross-day test (strongest)": ev["test_ba"].map(pct_or_na),
        "Cross-day recall": ev["test_recall"].map(pct_or_na),
        "Cross-day false-positive rate": ev["test_fpr"].map(
            lambda v: pct_or_na(v, 2)),
    })
    st.dataframe(shown, hide_index=True)
    st.markdown(
        "- **Random split (preliminary):** train and test rows are drawn from "
        "the same captures, so near-identical neighbouring flows appear on "
        "both sides. It is shown only to demonstrate how it can flatter a "
        "model. Treat it as preliminary.\n"
        "- **Validation inside the training day:** the held-out rows are a "
        "different time block of the same capture.\n"
        "- **Cross-day test:** train on one day, test on another. This is the "
        "strongest evidence of generalisation we have, though it is still one "
        "dataset recorded in one lab.")
    if key == "compact":
        st.caption("No random-split run was made for the compact schema, so "
                   "that column shows n/a.")

    # ---- 4. seen vs unseen ---------------------------------------------------------
    st.divider()
    st.subheader("4. Attack types seen and never seen in training")
    if detail is None:
        st.warning("final_evaluation_detail.json not found. Run python "
                   "src\\evaluation_report.py.")
    else:
        su = seen_unseen_table(key, meta, detail)
        st.dataframe(pd.DataFrame({
            "Model": [n + ("  (saved in dashboard)" if n == entry["model_name"]
                           else "") for n in su["Model"]],
            "Detected: LDAP and Syn (seen)": su["recall_seen"].map(pct_or_na),
            "Detected: NetBIOS (never seen)": su["recall_unseen"].map(pct_or_na),
            "False-positive rate": su["fpr"].map(lambda v: pct_or_na(v, 2)),
            "Balanced acc., seen types only": su["ba_seen"].map(pct_or_na),
            "Balanced acc., all attacks": su["ba_all"].map(pct_or_na),
        }), hide_index=True)
        st.caption("NetBIOS is in the test day but not in the training day. "
                   "Whether a model catches it differs strongly from model to "
                   "model, so one attack type from one day is not evidence "
                   "that any model detects unknown attacks in general.")

    # ---- 5. feature importance ---------------------------------------------------------
    st.divider()
    st.subheader("5. What the model relies on")
    imp, kind = importance_frame(entry)
    if imp.empty:
        st.caption("No feature explanation was saved for this model.")
    else:
        plot = imp.iloc[::-1]
        colors = ["#e76f51" if v >= 0 else "#2a9d8f" for v in plot["value"]]
        fig, ax = plt.subplots(figsize=(8, 0.42 * len(plot) + 1.5))
        ax.barh(plot["feature"], plot["value"], color=colors)
        if kind == "importance":
            ax.set_xlabel("Feature importance (the values add up to 1)")
        else:
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Coefficient on standardised inputs (red pushes "
                          "towards ATTACK, teal towards BENIGN)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        if kind == "importance":
            st.caption("Importance is each feature's share of the tree's "
                       "total error reduction. Many CIC features are near-"
                       "copies of each other (see the correlation heatmap in "
                       "our analysis), so they split the credit, and a low "
                       "value does not mean a feature is useless.")
        else:
            st.caption("Coefficients are measured on standardised inputs, so "
                       "their sizes are comparable. Protocol enters as a "
                       "plain number (6 or 17), which is a crude encoding "
                       "for a linear model. Related features also share "
                       "credit here.")
            st.caption("Total Packets and Total Bytes have large coefficients "
                       "of opposite sign. They measure related things, so the "
                       "model probably uses them together. Do not read either "
                       "one alone.")
        st.caption("These describe what the model uses, not what causes "
                   "attacks.")
    if kind == "importance":
        with st.expander("The tree's decision rules"):
            try:
                st.code(tree_rules(load_model(key), entry["feature_columns"]),
                        language="text")
                st.caption("Read top to bottom: each line is a yes/no test on "
                           "one feature (units as in the dataset, for example "
                           "microseconds for durations). Missing or infinite "
                           "values were filled with the training median "
                           "before these tests.")
            except Exception as err:
                st.caption(f"The saved model could not be loaded "
                           f"({type(err).__name__}).")

    # ---- 6. extra evidence -------------------------------------------------------------
    st.divider()
    st.subheader("6. Supporting evidence")
    with st.expander("Sensitivity: what if the excluded columns were used?"):
        sens = sensitivity_table(final)
        if sens.empty:
            st.caption("No sensitivity results found.")
        else:
            st.dataframe(pd.DataFrame({
                "Model": sens["Model"],
                "Recall": sens["recall"].map(pct_or_na),
                "False-positive rate": sens["fpr"].map(
                    lambda v: pct_or_na(v, 2)),
                "Balanced accuracy": sens["balanced_accuracy"].map(pct_or_na),
            }), hide_index=True)
        st.caption("Source and destination ports and the Inbound flag were "
                   "added to the full schema for this check. These columns "
                   "describe this lab network, not traffic behaviour, so they "
                   "are excluded from the saved models. This table shows "
                   "what the exclusion costs, and why a near-perfect score "
                   "built on them would be a shortcut rather than detection.")
    with st.expander("Figures from the evaluation scripts"):
        for name, caption in (
                ("fig06_confusion_matrices.png",
                 "Confusion matrices of all candidate models on the test day"),
                ("fig07_roc_curves.png", "ROC curves on the test day"),
                ("fig08_seen_vs_unseen.png",
                 "Seen vs unseen attack types and false-positive rates")):
            path = FIG_DIR / name
            if path.exists():
                st.image(str(path), caption=caption)
            else:
                st.caption(f"{name} not found. Run python "
                           "src\\evaluation_report.py.")
    with st.expander("Limitations"):
        netbios = entry["test_recall_by_type"].get("NetBIOS")
        st.markdown(
            f"- **Offline evaluation only.** Nothing here is live monitoring, "
            f"and a prediction is a model output, not proof of an attack.\n"
            f"- **One dataset, one lab.** Both days come from CIC-DDoS2019, "
            f"recorded in one test network. Performance on other networks is "
            f"unknown.\n"
            f"- **Chosen after seeing test results.** The selection rule was "
            f"fixed beforehand, but the models were picked after the "
            f"{meta['test_day']} results were known, so these scores are "
            f"slightly optimistic.\n"
            f"- **Sample mix.** Our sample is 20% benign, while the raw data "
            f"is under 1% benign. Precision and accuracy would be lower on "
            f"real traffic. Recall and false-positive rate transfer better.\n"
            f"- **Unseen attack type.** NetBIOS was detected "
            f"{pct_or_na(netbios)} of the time by this model. That is one "
            f"attack type from one day.\n"
            f"- **Repeated flows.** {entry['test_rows_identical_to_training']:,} "
            f"test rows have model-input values identical to a training row. "
            f"The Overview page shows the scores on the remaining rows.")
        

# ---------------------------------------------------- placeholder pages ----
def page_placeholder(title):
    st.title(title)
    st.info("This page has not been built yet. It is a later task in Phase 8.")


# ---------------------------------------------------------------- shell ----
def main():
    st.sidebar.title("🛡️ DDoS Detection")
    page = st.sidebar.radio("Navigation", PAGES)
    meta = load_json("model_metadata.json")
    if meta:
        st.sidebar.success("Model artifacts loaded")
        st.sidebar.caption(f"Trained {meta['created_utc']} (UTC)")
    else:
        st.sidebar.error("No trained model found")
    st.sidebar.divider()
    st.sidebar.caption(
        "Dataset: CIC-DDoS2019 (Sharafaldin, Lashkari, Hakak, Ghorbani, 2019), "
        "Canadian Institute for Cybersecurity, UNB. Files obtained through a "
        "Kaggle mirror.")

    if page == "Dashboard Overview":
        page_overview()
    elif page == "Manual Detection":
        page_manual_detection()
    elif page == "CSV Prediction":
        page_csv_prediction()
    elif page == "Dataset Explorer":
        page_dataset_explorer()
    elif page == "Model Performance":
        page_model_performance()
    else:
        page_placeholder(page)


main()