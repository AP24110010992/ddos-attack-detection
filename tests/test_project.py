"""
Automated tests for the DDoS attack detection project.

Run from the project root (virtual environment active):
    python -m pytest tests -v

The tests use the real sample and the saved models, and check that the
pipeline, the saved artifacts and the dashboard helpers agree with each
other. A test is SKIPPED (not failed) when an artifact it needs has not been
created yet. Test-day rows are used only as held-out data: the saved models
were trained on the training day only.
"""

import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compact_features import COMPACT_FEATURES, compact_from_raw
from csv_prediction import (
    MAX_ROWS, build_download_frame, check_schema, evaluate_against_labels,
    find_label_column, ground_truth_from_labels, protocol_outside_training,
    read_upload,
)
from dashboard_data import load_json, raw_totals, sample_distribution
from data_processing import (
    EXCLUDED_COLUMNS, FEATURE_COLUMNS, TARGET, TEST_DAY, TRAIN_DAY,
    build_preprocessor, load_sample, prepare_features,
)
from dataset_explorer import missing_summary, overview, snapshot_data_dir
from evaluate import compute_metrics, make_time_blocks
from manual_input import (
    MAX_DURATION_MS, MS_TO_US, derive_frame, derive_row, range_warnings,
    validate_inputs,
)
from performance_data import metric_rows, metrics_from_counts
from predict import (
    ModelNotTrainedError, _load_path, load_metadata, load_model, predict_frame,
)

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_ID_COLUMNS = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]


# =========================================================== fixtures ======
@pytest.fixture(scope="session")
def sample():
    if not (ROOT / "data" / "processed" / "sample_20k.parquet").exists():
        pytest.skip("Working sample missing. Run: python src\\build_sample.py")
    return load_sample().reset_index(drop=True)


@pytest.fixture(scope="session")
def train_rows(sample):
    return sample[sample["day"] == TRAIN_DAY].reset_index(drop=True)


@pytest.fixture(scope="session")
def held_out(sample):
    """Test-day rows: never used to train the saved models."""
    return sample[sample["day"] == TEST_DAY].reset_index(drop=True)


@pytest.fixture(scope="session")
def meta():
    try:
        return load_metadata()
    except ModelNotTrainedError:
        pytest.skip("Saved models missing. Run: python src\\train.py")


@pytest.fixture(scope="session")
def final_eval():
    data = load_json("final_evaluation.json")
    if data is None:
        pytest.skip("Run: python src\\final_evaluation.py")
    return data["results"]


@pytest.fixture(scope="session")
def allowed_protocols(meta):
    return {int(p) for p in
            meta["models"]["compact"]["protocol_values_in_training"]}


def make_upload_frame(held_out, per_type=100):
    """Real test-day rows, `per_type` of each traffic type, with a Label column."""
    picked = held_out.groupby("attack_type").sample(n=per_type, random_state=42)
    frame = picked[UPLOAD_ID_COLUMNS + FEATURE_COLUMNS].copy()
    frame["Label"] = picked["label_original"].to_numpy()
    return frame.reset_index(drop=True)


def as_csv_file(frame):
    return io.BytesIO(frame.to_csv(index=False).encode("utf-8"))


def model_inputs(held_out, key):
    return held_out if key == "full" else compact_from_raw(held_out)


# ============================================ 1. schema and preprocessing ==
def test_schema_has_74_unique_features_and_no_excluded_columns():
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS)) == 74
    assert not set(FEATURE_COLUMNS) & set(EXCLUDED_COLUMNS)
    assert not any("label" in c.lower() for c in FEATURE_COLUMNS)


def test_compact_schema_is_documented_and_has_no_syn_flag():
    assert COMPACT_FEATURES == [
        "Protocol", "Flow Duration", "Total Packets", "Total Bytes",
        "Flow Packets/s", "Flow Bytes/s", "ACK Flag Count"]


def test_prepare_features_cleans_names_and_restores_order(held_out):
    rows = held_out.head(20)
    scrambled = rows[list(reversed(FEATURE_COLUMNS))].copy()
    scrambled.columns = [" " + c for c in scrambled.columns]
    pd.testing.assert_frame_equal(prepare_features(scrambled),
                                  prepare_features(rows))


def test_prepare_features_reports_missing_columns_and_text(held_out):
    with pytest.raises(ValueError, match="missing"):
        prepare_features(held_out.drop(columns=["Flow Duration"]))
    bad = held_out.head(3).copy()
    bad["Protocol"] = ["abc", "6", "17"]
    with pytest.raises(ValueError, match="Non-numeric"):
        prepare_features(bad)


def test_imputer_uses_the_training_day_median(train_rows):
    X = prepare_features(train_rows)
    pre = build_preprocessor().fit(X)
    column = "Flow Packets/s"
    expected = X[column].replace([np.inf, -np.inf], np.nan).median()
    filled = pre.named_steps["impute"].statistics_[FEATURE_COLUMNS.index(column)]
    assert filled == pytest.approx(expected)


def test_preprocessing_removes_non_finite_values_and_constant_columns(
        train_rows, held_out):
    X_train, X_test = prepare_features(train_rows), prepare_features(held_out)
    # Regression check: counts measured earlier in this project
    assert int((~np.isfinite(X_train.to_numpy(dtype="float64"))).sum()) == 1268
    assert int((~np.isfinite(X_test.to_numpy(dtype="float64"))).sum()) == 780
    pre = build_preprocessor().fit(X_train)
    Z_train, Z_test = pre.transform(X_train), pre.transform(X_test)
    assert Z_train.shape == (10000, 62) and Z_test.shape == (10000, 62)
    assert np.isfinite(Z_train).all() and np.isfinite(Z_test).all()


# ============================================================ 2. metrics ===
def test_compute_metrics_against_a_hand_calculation():
    y_true = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    y_pred = [0, 0, 1, 0, 1, 1, 1, 0, 1, 1]
    y_score = [0.1, 0.2, 0.9, 0.3, 0.8, 0.7, 0.9, 0.4, 0.6, 0.95]
    m = compute_metrics(y_true, y_pred, y_score)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 1, 1, 5)
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["precision"] == pytest.approx(5 / 6)
    assert m["recall"] == pytest.approx(5 / 6)
    assert m["fpr"] == pytest.approx(0.25)
    assert m["balanced_accuracy"] == pytest.approx((5 / 6 + 0.75) / 2)
    assert m["roc_auc"] == pytest.approx(0.8125)


def test_metrics_stay_defined_when_a_class_is_missing():
    m = compute_metrics([1, 1, 1], [1, 1, 0], [0.9, 0.8, 0.4])
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (0, 0, 1, 2)
    assert m["fpr"] == 0.0
    assert m["recall"] == pytest.approx(2 / 3)
    assert np.isnan(m["roc_auc"])


def test_count_based_metrics_agree_with_compute_metrics():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    pred = np.where(rng.random(500) < 0.9, y, 1 - y)
    m = compute_metrics(y, pred)
    c = metrics_from_counts(m["tn"], m["fp"], m["fn"], m["tp"])
    for key, value in c.items():
        assert value == pytest.approx(m[key]), key


def test_time_blocks_split_every_capture_file_evenly(train_rows):
    blocks = make_time_blocks(train_rows, 5)
    assert sorted(blocks.unique()) == [0, 1, 2, 3, 4]
    table = pd.crosstab(blocks, train_rows["attack_type"])
    assert (table.to_numpy() > 0).all()          # every traffic type in every block
    assert blocks.value_counts().between(1999, 2001).all()


# ===================================== 3. saved models and predictions =====
@pytest.mark.parametrize("key", ["full", "compact"])
def test_saved_model_reproduces_its_saved_test_results(key, meta, held_out):
    out = predict_frame(model_inputs(held_out, key), key)
    m = compute_metrics(held_out[TARGET].to_numpy(),
                        out["prediction"].to_numpy(),
                        out["attack_probability"].to_numpy())
    saved = meta["models"][key]["test_metrics"]
    keys = ("tn", "fp", "fn", "tp")
    assert [m[k] for k in keys] == [saved[k] for k in keys]


@pytest.mark.parametrize("key", ["full", "compact"])
def test_predictions_are_consistent_and_probabilities_valid(key, meta, held_out):
    out = predict_frame(model_inputs(held_out, key), key)
    assert list(out.columns) == ["prediction", "prediction_label",
                                 "attack_probability"]
    assert len(out) == len(held_out)
    p = out["attack_probability"].to_numpy()
    pred = out["prediction"].to_numpy()
    assert ((p >= 0) & (p <= 1)).all()
    decided = p != 0.5
    assert ((p[decided] > 0.5).astype(int) == pred[decided]).all()
    expected = np.where(pred == 1, "DDoS ATTACK", "BENIGN")
    assert (out["prediction_label"].to_numpy() == expected).all()


@pytest.mark.parametrize("key", ["full", "compact"])
def test_column_order_of_the_input_does_not_matter(key, meta, held_out):
    inputs = model_inputs(held_out, key).head(300)
    scrambled = inputs[list(reversed(list(inputs.columns)))]
    a = predict_frame(inputs, key)["prediction"].to_numpy()
    b = predict_frame(scrambled, key)["prediction"].to_numpy()
    assert (a == b).all()


@pytest.mark.parametrize("key", ["full", "compact"])
def test_missing_column_is_reported_clearly(key, meta, held_out):
    inputs = model_inputs(held_out, key).drop(columns=["Flow Duration"])
    with pytest.raises(ValueError, match="Flow Duration"):
        predict_frame(inputs, key)


def test_missing_model_file_tells_the_user_how_to_train():
    with pytest.raises(ModelNotTrainedError, match="python src"):
        _load_path(ROOT / "models" / "does_not_exist.joblib")


def test_two_real_held_out_flows_are_classified_correctly(meta, held_out):
    # Two examples to show the workflow end to end, not a performance claim.
    benign = held_out[held_out[TARGET] == 0].head(1)
    attack = held_out[held_out[TARGET] == 1].head(1)
    assert predict_frame(benign, "full")["prediction"].iloc[0] == 0
    assert predict_frame(attack, "full")["prediction"].iloc[0] == 1


def test_saved_models_were_trained_on_the_training_day_only(meta):
    for key, columns in (("full", FEATURE_COLUMNS), ("compact", COMPACT_FEATURES)):
        entry = meta["models"][key]
        assert entry["trained_on"]["day"] == TRAIN_DAY
        assert entry["trained_on"]["rows"] == 10000
        imputer = load_model(key).named_steps["pre"].named_steps["impute"]
        assert len(imputer.statistics_) == len(columns)


def test_models_load_and_agree_in_a_fresh_python_process(meta, sample):
    """The 'restart the app' test: a new interpreter loads the saved files."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "predict.py")],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=600)
    assert result.returncode == 0, (result.stdout[-1500:] + result.stderr[-1500:])
    assert "ALL RELOAD CHECKS PASSED" in result.stdout


# ===================================================== 4. manual input =====
VALID_FORMS = [
    pytest.param((1500.0, 850, 124000, 1, 6), id="demonstration values"),
    pytest.param((0.0, 1, 0, 0, 17), id="all minimums, UDP"),
    pytest.param((MAX_DURATION_MS, 1, 65_535, 1, 6),
                 id="duration at the limit, 65535 bytes in one packet"),
    pytest.param((0.001, 2, 120, 0, 6), id="duration of one microsecond"),
]
INVALID_FORMS = [
    pytest.param((-1.0, 850, 124000, 1, 6), "negative", id="negative duration"),
    pytest.param((1500.0, 0, 0, 1, 6), "at least 1", id="zero packets"),
    pytest.param((1500.0, 2.5, 100, 1, 6), "whole number",
                 id="fractional packets"),
    pytest.param((1500.0, 850, -5, 1, 6), "at least 0", id="negative bytes"),
    pytest.param((1500.0, 1, 70000, 1, 6), "per packet",
                 id="more than 65535 bytes in one packet"),
    pytest.param((1500.0, 850, 124000, 2, 6), "0 or 1", id="ACK flag of 2"),
    pytest.param((1500.0, 850, 124000, 1, 1),
                 "does not appear in the training data",
                 id="protocol 1 (not in training)"),
    pytest.param((float("nan"), 850, 124000, 1, 6), "finite",
                 id="NaN duration"),
    pytest.param((1500.0, "abc", 124000, 1, 6), "not a number",
                 id="text instead of a number"),
    pytest.param((MAX_DURATION_MS * 2, 850, 124000, 1, 6),
                 "above the form limit", id="duration above the limit"),
]


def test_form_values_become_the_documented_model_row():
    row = derive_row(1500.0, 850, 124000, 1, 6)
    assert list(row.columns) == COMPACT_FEATURES
    assert row.at[0, "Flow Duration"] == pytest.approx(1_500_000.0)
    assert row.at[0, "Flow Packets/s"] == pytest.approx(850 / 1.5)
    assert row.at[0, "Flow Bytes/s"] == pytest.approx(124000 / 1.5)
    assert row.at[0, "Protocol"] == 6 and row.at[0, "ACK Flag Count"] == 1


@pytest.mark.parametrize("protocol", [6, 17])
def test_zero_duration_leaves_rates_undefined_but_still_predicts(protocol, meta):
    row = derive_row(0.0, 2, 120, 0, protocol)
    assert row[["Flow Packets/s", "Flow Bytes/s"]].isna().all().all()
    out = predict_frame(row, "compact")
    assert out["prediction"].iloc[0] in (0, 1)
    assert 0 <= out["attack_probability"].iloc[0] <= 1


@pytest.mark.parametrize("protocol", [6, 17])
def test_tcp_and_udp_forms_both_produce_a_prediction(protocol, meta):
    out = predict_frame(derive_row(1500.0, 850, 124000, 1, protocol), "compact")
    assert out["prediction_label"].iloc[0] in ("BENIGN", "DDoS ATTACK")


@pytest.mark.parametrize("values", VALID_FORMS)
def test_valid_form_values_are_accepted(values, allowed_protocols):
    assert validate_inputs(*values, allowed_protocols) == []


@pytest.mark.parametrize("values, fragment", INVALID_FORMS)
def test_invalid_form_values_are_rejected_with_a_reason(
        values, fragment, allowed_protocols):
    errors = validate_inputs(*values, allowed_protocols)
    assert errors, "invalid input was accepted"
    assert any(fragment in message for message in errors), errors


def test_every_real_flow_passes_the_form_validation(held_out, allowed_protocols):
    c = compact_from_raw(held_out)
    rejected = sum(
        bool(validate_inputs(dur / MS_TO_US, pk, by, ack, proto,
                             allowed_protocols))
        for proto, dur, pk, by, ack in zip(
            c["Protocol"], c["Flow Duration"], c["Total Packets"],
            c["Total Bytes"], c["ACK Flag Count"]))
    assert rejected == 0


def test_rebuilding_real_flows_from_form_style_values_reproduces_features(
        meta, held_out):
    ref = compact_from_raw(held_out)[COMPACT_FEATURES]
    ms = ref["Flow Duration"].to_numpy(dtype="float64") / MS_TO_US
    rebuilt = derive_frame(ms, ref["Total Packets"].to_numpy(),
                           ref["Total Bytes"].to_numpy(),
                           ref["ACK Flag Count"].to_numpy(),
                           ref["Protocol"].to_numpy())
    positive = (ref["Flow Duration"] > 0).to_numpy()
    for col in ("Flow Packets/s", "Flow Bytes/s"):
        a = ref[col].to_numpy(dtype="float64")
        b = rebuilt[col].to_numpy(dtype="float64")
        np.testing.assert_allclose(b[positive], a[positive], rtol=1e-6)
        assert not np.isfinite(a[~positive]).any()   # undefined in the dataset
        assert np.isnan(b[~positive]).all()          # undefined in the form
    flips = int((predict_frame(ref, "compact")["prediction"].to_numpy()
                 != predict_frame(rebuilt, "compact")["prediction"].to_numpy()).sum())
    assert flips <= 10                                # at most 0.1% of rows


def test_range_warnings_fire_only_outside_the_training_range(meta):
    ranges = meta["models"]["compact"]["training_ranges"]
    assert range_warnings(derive_row(1500.0, 850, 124000, 1, 6), ranges) == []
    extreme = range_warnings(derive_row(80_000_000.0, 850, 124000, 1, 6), ranges)
    assert "Flow Duration" in [w[0] for w in extreme]


# ================================================ 5. CSV prediction page ===
def test_labelled_upload_predicts_like_direct_prediction(meta, held_out):
    frame = make_upload_frame(held_out)
    df, truncated = read_upload(as_csv_file(frame))
    assert len(df) == 400 and not truncated
    report = check_schema(df)
    assert report["missing"] == []
    assert set(report["extra"]) == set(UPLOAD_ID_COLUMNS + ["Label"])
    via_csv, direct = predict_frame(df, "full"), predict_frame(frame, "full")
    assert (via_csv["prediction"].to_numpy() == direct["prediction"].to_numpy()).all()
    np.testing.assert_allclose(via_csv["attack_probability"],
                               direct["attack_probability"], atol=1e-9)


def test_evaluation_uses_the_labels_in_the_file(meta, held_out):
    df, _ = read_upload(as_csv_file(make_upload_frame(held_out)))
    assert find_label_column(df) == "Label"
    y, _ = ground_truth_from_labels(df["Label"])
    assert int((y == 0).sum()) == 100 and int((y == 1).sum()) == 300
    metrics, n_known = evaluate_against_labels(y, predict_frame(df, "full"))
    assert n_known == 400
    assert metrics["tn"] + metrics["fp"] == 100
    assert metrics["fn"] + metrics["tp"] == 300


def test_file_without_labels_and_with_spaced_column_names(meta, held_out):
    frame = make_upload_frame(held_out).drop(columns=["Label"])
    spaced = frame.copy()
    spaced.columns = [" " + c for c in spaced.columns]
    df, _ = read_upload(as_csv_file(spaced))
    assert find_label_column(df) is None
    assert check_schema(df)["missing"] == []
    a = predict_frame(df, "full")["prediction"].to_numpy()
    b = predict_frame(frame, "full")["prediction"].to_numpy()
    assert (a == b).all()


def test_missing_columns_are_listed_in_full(held_out):
    broken = make_upload_frame(held_out).head(50).drop(
        columns=["Protocol", "Flow Duration"])
    df, _ = read_upload(as_csv_file(broken))
    assert set(check_schema(df)["missing"]) == {"Protocol", "Flow Duration"}


def test_text_in_a_numeric_column_is_rejected(meta, held_out):
    frame = make_upload_frame(held_out).head(50).copy()
    frame["Total Fwd Packets"] = frame["Total Fwd Packets"].astype(str)
    frame.loc[2, "Total Fwd Packets"] = "abc"
    df, _ = read_upload(as_csv_file(frame))
    with pytest.raises(ValueError, match="Non-numeric"):
        predict_frame(df, "full")


def test_protocols_never_seen_in_training_are_counted_but_still_predicted(
        meta, held_out, allowed_protocols):
    frame = make_upload_frame(held_out).head(50).copy()
    frame.loc[frame.index[:5], "Protocol"] = 1
    df, _ = read_upload(as_csv_file(frame))
    assert protocol_outside_training(df, allowed_protocols) == 5
    assert len(predict_frame(df, "full")) == 50


@pytest.mark.parametrize("content, fragment", [
    pytest.param(b"", "empty", id="empty file"),
    pytest.param(b"a,b,c\n", "no data rows", id="header only"),
])
def test_unusable_files_are_rejected_with_a_message(content, fragment):
    with pytest.raises(ValueError, match=fragment):
        read_upload(io.BytesIO(content))


def test_binary_junk_never_passes_as_a_valid_file():
    junk = bytes([0xFF, 0xFE, 0x00, 0x81]) * 200
    try:
        df, _ = read_upload(io.BytesIO(junk))
    except ValueError:
        return                                   # rejected while reading
    assert check_schema(df)["missing"]           # or rejected by the schema check


def test_files_above_the_row_cap_are_truncated_and_flagged():
    big = pd.DataFrame({"a": np.arange(MAX_ROWS + 5)})
    df, truncated = read_upload(as_csv_file(big))
    assert truncated and len(df) == MAX_ROWS


def test_label_interpretation_and_no_evaluation_without_usable_labels():
    y, table = ground_truth_from_labels(
        pd.Series(["BENIGN", " benign", "Syn", "DrDoS_UDP", None, ""]))
    assert list(y[:4]) == [0, 0, 1, 1] and np.isnan(y[4:]).all()
    y01, _ = ground_truth_from_labels(pd.Series([0, 1, 1, 0]))
    assert list(y01) == [0.0, 1.0, 1.0, 0.0]
    results = pd.DataFrame({"prediction": [0, 1],
                            "attack_probability": [0.2, 0.8]})
    with pytest.raises(ValueError, match="no usable values"):
        evaluate_against_labels(np.array([np.nan, np.nan]), results)


def test_download_frame_keeps_the_users_columns(meta, held_out):
    frame = make_upload_frame(held_out).head(3).copy()
    frame["prediction"] = "keep"
    out = build_download_frame(frame, predict_frame(frame, "full"))
    assert out.columns[0] == "row_number" and len(out) == 3
    assert (out["prediction_in_upload"] == "keep").all()
    assert set(out["prediction"].unique()) <= {0, 1}


# ============================== 6. dashboard numbers equal the saved files ==
@pytest.mark.parametrize("key", ["full", "compact"])
def test_saved_metadata_matches_the_evaluation_script_output(
        key, meta, final_eval):
    entry = meta["models"][key]
    reference = final_eval[entry["schema_name"]][entry["model_name"]]
    for k in ("tn", "fp", "fn", "tp"):
        assert entry["test_metrics"][k] == reference[k]


@pytest.mark.parametrize("key", ["full", "compact"])
def test_every_metric_on_the_performance_page_matches_the_saved_value(key, meta):
    table = metric_rows(meta["models"][key]["test_metrics"])
    counted = table[table["Metric"] != "ROC-AUC"]
    assert len(counted) == 6
    assert (counted["Check against the saved value"] == "matches").all()


def test_overview_numbers_agree_with_the_data_files(sample):
    profile, raw = load_json("sample_profile.json"), load_json("raw_inspection.json")
    if profile is None or raw is None:
        pytest.skip("Run src\\describe_sample.py and src\\inspect_raw.py first")
    assert profile["rows"] == len(sample)
    assert profile["class_counts"]["BENIGN"] == int((sample[TARGET] == 0).sum())
    assert profile["class_counts"]["ATTACK"] == int((sample[TARGET] == 1).sum())
    assert int(sample_distribution(profile).to_numpy().sum()) == len(sample)
    totals = raw_totals(raw)
    # Regression check: values measured by src\inspect_raw.py
    assert (totals["rows"], totals["benign"]) == (10_197_998, 42_918)


def test_dashboard_schema_equals_the_trained_schema(meta):
    assert meta["models"]["full"]["feature_columns"] == FEATURE_COLUMNS
    assert meta["models"]["compact"]["feature_columns"] == COMPACT_FEATURES


def test_profiling_a_dataset_never_changes_the_data_folder(sample):
    before = snapshot_data_dir()
    overview(sample)
    missing_summary(sample)
    assert snapshot_data_dir() == before


def test_gitignore_keeps_raw_data_and_the_environment_out_of_git():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for line in (".venv/", "data/raw/", "data/processed/"):
        assert line in text, f"{line} missing from .gitignore"