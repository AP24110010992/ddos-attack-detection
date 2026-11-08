"""
CSV-upload helpers for the dashboard's CSV Prediction page.

  read_upload               : read an uploaded CSV safely (row cap, clear errors)
  check_schema              : which required columns are present or missing
  find_label_column         : locate an optional ground-truth column
  ground_truth_from_labels  : map labels to 0/1 and show exactly how
  evaluate_against_labels   : metrics, ONLY when ground-truth labels exist
  protocol_outside_training : rows whose protocol was never seen in training
  build_download_frame      : original columns plus the prediction columns

This module does NOT import streamlit, so it can be tested on its own.
First create the demo files, then run the self-check (from the project root):
    python src\\make_demo_csv.py
    python src\\csv_prediction.py
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd

from data_processing import FEATURE_COLUMNS
from evaluate import compute_metrics

ROOT = Path(__file__).resolve().parent.parent
MAX_ROWS = 20_000          # rows scored per upload (keeps the app responsive)
MAX_FILE_MB = 50           # largest upload accepted
LABEL_NAMES = ("label", "label_original", "label_binary")
PREDICTION_COLUMNS = ("prediction", "prediction_label", "attack_probability")


def read_upload(file_obj, max_rows=MAX_ROWS):
    """
    Read an uploaded CSV. Returns (DataFrame with cleaned column names,
    truncated), where truncated is True when the file had more than max_rows
    rows. Raises ValueError with a readable message when the file is unusable.
    """
    try:
        df = pd.read_csv(file_obj, nrows=max_rows + 1, low_memory=False)
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty.") from None
    except UnicodeDecodeError:
        raise ValueError("The file could not be read as text. Upload a plain "
                         "CSV file saved as UTF-8.") from None
    except pd.errors.ParserError as err:
        raise ValueError(f"The file could not be parsed as CSV ({err}).") from None

    if len(df) == 0:
        raise ValueError("The file has a header row but no data rows.")

    df.columns = df.columns.astype(str).str.strip()
    repeated = df.columns[df.columns.duplicated()].unique().tolist()
    if repeated:
        raise ValueError("Some column names repeat after removing spaces: "
                         f"{repeated[:5]}")
    truncated = len(df) > max_rows
    return df.iloc[:max_rows].reset_index(drop=True), truncated


def check_schema(df):
    """Required columns that are missing, and extra columns that will be ignored."""
    present = set(df.columns)
    required = set(FEATURE_COLUMNS)
    return {
        "missing": [c for c in FEATURE_COLUMNS if c not in present],
        "extra": [c for c in df.columns if c not in required],
        "n_required": len(FEATURE_COLUMNS),
    }


def find_label_column(df):
    """Name of the ground-truth column if the file has one, otherwise None."""
    for wanted in LABEL_NAMES:
        for col in df.columns:
            if col.lower() == wanted:
                return col
    return None


def _treated_as(value, binary_numeric):
    if value == "(empty)":
        return "ignored (no label)"
    if binary_numeric:
        return "BENIGN (0)" if float(value) == 0 else "ATTACK (1)"
    return "BENIGN (0)" if value.lower() == "benign" else "ATTACK (1)"


def ground_truth_from_labels(series):
    """
    Turn a label column into 0/1 and say exactly how.
      * a column that contains only 0 and 1 is used as it is
      * otherwise "BENIGN" (any capitalisation) is 0 and every other
        non-empty value is an attack (1)
      * empty values are ignored
    Returns (y, table): y is a float array with 0.0, 1.0 or NaN (ignored);
    table lists each label value, how many rows have it, and how it is treated.
    """
    text = series.astype("string").str.strip().fillna("").astype(object)
    empty = (text == "").to_numpy(dtype=bool)
    known = ~empty
    as_number = pd.to_numeric(text.where(~empty), errors="coerce").to_numpy(dtype="float64")
    binary_numeric = bool(known.any()) and bool(
        np.isin(as_number[known], [0.0, 1.0]).all())

    y = np.full(len(text), np.nan)
    if binary_numeric:
        y[known] = as_number[known]
    else:
        is_benign = text.str.lower().eq("benign").to_numpy(dtype=bool)
        y[known] = np.where(is_benign[known], 0.0, 1.0)

    shown = text.where(~empty, "(empty)")
    table = shown.value_counts().rename_axis("Label value").reset_index(name="Rows")
    table["Treated as"] = [_treated_as(v, binary_numeric)
                           for v in table["Label value"]]
    return y, table


def evaluate_against_labels(y, results):
    """Metrics against ground truth. Rows without a usable label are excluded."""
    known = ~np.isnan(y)
    if not known.any():
        raise ValueError("The label column has no usable values.")
    metrics = compute_metrics(
        y[known].astype(int),
        results["prediction"].to_numpy()[known],
        results["attack_probability"].to_numpy()[known])
    return metrics, int(known.sum())


def protocol_outside_training(df, allowed):
    """Number of rows whose Protocol value never occurred in the training data."""
    values = pd.to_numeric(df["Protocol"], errors="coerce")
    return int((values.notna() & ~values.isin(list(allowed))).sum())


def build_download_frame(df, results):
    """The uploaded data with row_number and the prediction columns added."""
    out = df.copy()
    for col in ("row_number",) + PREDICTION_COLUMNS:
        if col in out.columns:                       # never overwrite the user's data
            out = out.rename(columns={col: col + "_in_upload"})
    out.insert(0, "row_number", np.arange(1, len(out) + 1))
    for col in PREDICTION_COLUMNS:
        out[col] = results[col].to_numpy()
    return out


# ------------------------------------------------------------ self-check ----
def main():
    from data_processing import TEST_DAY, load_sample
    from predict import load_metadata, predict_frame

    demo_dir = ROOT / "data" / "processed" / "demo"
    names = ["demo_test_day_with_labels.csv", "demo_test_day_no_labels.csv",
             "demo_missing_columns.csv", "demo_text_value.csv",
             "demo_unseen_protocol.csv", "demo_empty.csv",
             "demo_header_only.csv"]
    absent = [n for n in names if not (demo_dir / n).exists()]
    if absent:
        raise SystemExit(f"Demo files missing: {absent}. "
                         "Run python src\\make_demo_csv.py first.")

    meta = load_metadata()
    allowed = {int(p) for p in meta["models"]["full"]["protocol_values_in_training"]}

    def load(name):
        with open(demo_dir / name, "rb") as f:
            return read_upload(f)

    # 1) Labelled demo file
    df, truncated = load(names[0])
    assert not truncated and len(df) == 400
    report = check_schema(df)
    assert report["missing"] == []
    assert set(report["extra"]) == {"Flow ID", "Source IP", "Destination IP",
                                    "Timestamp", "Label"}
    print(f"Labelled demo file: {len(df)} rows, {df.shape[1]} columns, "
          f"{report['n_required']} required columns found, extras ignored: "
          f"{report['extra']}   [OK]")
    results = predict_frame(df, "full")

    # Round trip: predictions from the CSV must equal predictions made directly
    # on the same rows of the sample (matched on Flow ID + Timestamp)
    sample = load_sample()
    test = sample[sample["day"] == TEST_DAY]
    key = (test["Flow ID"].astype(str) + "|" + test["Timestamp"].astype(str)).to_numpy()
    direct = pd.Series(predict_frame(test, "full")["attack_probability"].to_numpy(),
                       index=key)
    direct = direct[~direct.index.duplicated(keep=False)]
    csv_key = df["Flow ID"].astype(str) + "|" + df["Timestamp"].astype(str)
    matched = csv_key.isin(direct.index).to_numpy()
    diff = float(np.abs(results["attack_probability"].to_numpy()[matched]
                        - direct.loc[csv_key[matched]].to_numpy()).max())
    assert matched.sum() >= 390 and diff < 1e-9
    print(f"CSV round trip: {int(matched.sum())} of 400 rows matched to the "
          f"sample, largest probability difference {diff:.1e}   [OK]")

    label_col = find_label_column(df)
    assert label_col == "Label"
    y, table = ground_truth_from_labels(df[label_col])
    print("\nHow the label column is read:")
    print(table.to_string(index=False))
    metrics, n_known = evaluate_against_labels(y, results)
    assert n_known == 400
    assert metrics["tn"] + metrics["fp"] == 100
    assert metrics["fn"] + metrics["tp"] == 300
    print(f"\nEvaluation on {n_known} labelled rows: tn={metrics['tn']} "
          f"fp={metrics['fp']} fn={metrics['fn']} tp={metrics['tp']}  "
          f"(recall {metrics['recall']:.3f}, false-positive rate {metrics['fpr']:.3f})")
    print("Predictions by traffic type:")
    print(pd.crosstab(df[label_col], results["prediction_label"]).to_string())

    # 2) Unlabelled file whose column names start with spaces
    header = pd.read_csv(demo_dir / names[1], nrows=0).columns
    assert header[0] != header[0].strip(), "demo file should have spaced names"
    df2, _ = load(names[1])
    assert find_label_column(df2) is None and check_schema(df2)["missing"] == []
    r2 = predict_frame(df2, "full")
    assert (r2["prediction"].to_numpy() == results["prediction"].to_numpy()).all()
    print("\nNo-label file with spaced column names: names cleaned, no label "
          "found, same predictions as the labelled file   [OK]")

    # 3) Missing columns are listed in full
    df3, _ = load(names[2])
    missing = check_schema(df3)["missing"]
    assert set(missing) == {"Protocol", "Flow Duration"}
    print(f"Missing-columns file: reported {missing}   [OK]")

    # 4) Text in a numeric column is rejected with a clear message
    df4, _ = load(names[3])
    assert check_schema(df4)["missing"] == []
    try:
        predict_frame(df4, "full")
        raise AssertionError("text value was not detected")
    except ValueError as err:
        assert "Non-numeric" in str(err)
        print(f"Text-value file: {err}   [OK]")

    # 5) Protocol never seen in training
    df5, _ = load(names[4])
    n_out = protocol_outside_training(df5, allowed)
    assert n_out == 5
    predict_frame(df5, "full")
    print(f"Unseen-protocol file: {n_out} rows flagged, predictions still "
          f"made   [OK]")

    # 6) Empty and header-only files
    for name, phrase in ((names[5], "empty"), (names[6], "no data rows")):
        try:
            load(name)
            raise AssertionError(f"{name} was accepted")
        except ValueError as err:
            assert phrase in str(err)
            print(f"{name}: {err}   [OK]")

    # 7) Binary junk
    junk = io.BytesIO(bytes([0xFF, 0xFE, 0x00, 0x81]) * 200)
    try:
        junk_df, _ = read_upload(junk)
        assert check_schema(junk_df)["missing"], "binary junk passed the schema check"
        outcome = "read as text, then rejected by the schema check"
    except ValueError as err:
        outcome = f"rejected while reading ({err})"
    print(f"Binary junk: {outcome}   [OK]")

    # 8) Row cap
    big = pd.concat([df] * 51, ignore_index=True).head(MAX_ROWS + 5)
    out, trunc = read_upload(io.BytesIO(big.to_csv(index=False).encode("utf-8")))
    assert trunc and len(out) == MAX_ROWS
    print(f"File with {len(big):,} rows: only the first {len(out):,} used and "
          f"flagged as truncated   [OK]")

    # 9) Label interpretation
    y_t, t_t = ground_truth_from_labels(
        pd.Series(["BENIGN", " benign", "Syn", "DrDoS_UDP", None, ""]))
    assert np.array_equal(y_t[:4], [0, 0, 1, 1]) and np.isnan(y_t[4:]).all()
    y_n, _ = ground_truth_from_labels(pd.Series([0, 1, 1, 0]))
    assert np.array_equal(y_n, [0.0, 1.0, 1.0, 0.0])
    y_f, _ = ground_truth_from_labels(pd.Series([0.0, 1.0, np.nan]))
    assert y_f[0] == 0 and y_f[1] == 1 and np.isnan(y_f[2])
    print("Label interpretation (text, 0/1 numbers, blanks)   [OK]")
    print(t_t.to_string(index=False))

    # 10) Download frame keeps the user's data and never overwrites it
    clash = df.head(3).copy()
    clash["prediction"] = ["keep", "keep", "keep"]
    dl = build_download_frame(clash, predict_frame(clash, "full"))
    assert list(dl.columns[:1]) == ["row_number"]
    assert (dl["prediction_in_upload"] == "keep").all()
    assert set(dl["prediction"].unique()) <= {0, 1}
    print("Download frame: row_number added, a clashing user column is "
          "renamed instead of overwritten   [OK]")

    print("\nALL CSV-PAGE CHECKS PASSED")


if __name__ == "__main__":
    main()