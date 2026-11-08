"""
Helpers for the dashboard's Dataset Explorer page.

  load_sample_frame / load_demo_file : read-only loaders (demo names come from
                                       a fixed list, never from free text)
  overview             : rows, columns, types, memory, identical rows
  missing_summary      : NaN, blank and infinite values per column
  numeric_statistics   : basic statistics with infinite values ignored
  label_distribution   : counts per label value
  cleaned_copy         : a copy with infinity replaced by blanks (original untouched)
  raw_files_table      : the pre-computed scan of the raw CIC files
  snapshot_data_dir    : (size, modified time) of every file in data/, used to
                         prove that nothing here changes the data

This module does NOT import streamlit, so it can be tested on its own.
Run the self-check (from the project root):
    python src\\dataset_explorer.py
"""

import numpy as np
import pandas as pd

from csv_prediction import (
    MAX_ROWS, find_label_column, ground_truth_from_labels, read_upload,
)
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SAMPLE_PATH = DATA_DIR / "processed" / "sample_20k.parquet"
DEMO_DIR = DATA_DIR / "processed" / "demo"

# Columns that are a row position or a copy of the label, not traffic values.
# Left out when counting rows with identical values.
IGNORE_FOR_DUPLICATES = {"source_row_id", "unnamed: 0", "row_number", "label_binary"}


def signed_log(x):
    """log10(1 + |x|) with the original sign; tames heavy-tailed features."""
    return np.sign(x) * np.log10(1 + np.abs(x))


# ------------------------------------------------------------- loading -----
def list_demo_files():
    if not DEMO_DIR.exists():
        return []
    return sorted(p.name for p in DEMO_DIR.glob("*.csv"))


def load_sample_frame():
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            "The working sample was not found. Create it with: "
            "python src\\build_sample.py")
    return pd.read_parquet(SAMPLE_PATH)


def load_demo_file(name):
    """Read one demo file. Only names from the fixed list are accepted."""
    if name not in list_demo_files():
        raise ValueError(f"{name!r} is not one of the demo files.")
    with open(DEMO_DIR / name, "rb") as f:
        return read_upload(f, MAX_ROWS)


# ----------------------------------------------------------- profiling -----
def overview(df):
    numeric = df.select_dtypes("number").columns
    dup_cols = [c for c in numeric if c.lower() not in IGNORE_FOR_DUPLICATES]
    identical = (int(df.duplicated(subset=dup_cols, keep=False).sum())
                 if dup_cols else 0)
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "numeric_columns": int(len(numeric)),
        "text_columns": int(df.shape[1] - len(numeric)),
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1024 ** 2),
        "identical_rows": identical,
        "columns_compared": int(len(dup_cols)),
    }


def missing_summary(df):
    """
    Returns (table, totals). The table lists only columns with a problem:
    NaN (or blank text) values and infinite values.
    """
    n = len(df)
    any_bad = np.zeros(n, dtype=bool)
    rows = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            values = s.to_numpy(dtype="float64", na_value=np.nan)
            missing_mask = np.isnan(values)
            inf_mask = np.isinf(values)
            kind = "numeric"
        else:
            stripped = s.astype("string").str.strip()
            missing_mask = (s.isna() | (stripped == "").fillna(False)).to_numpy(dtype=bool)
            inf_mask = np.zeros(n, dtype=bool)
            kind = "text"
        bad = missing_mask | inf_mask
        if bad.any():
            any_bad |= bad
            rows.append({
                "Column": col, "Type": kind,
                "Missing": int(missing_mask.sum()),
                "Infinite": int(inf_mask.sum()),
                "Rows affected": int(bad.sum()),
                "Share of rows": float(bad.sum() / n),
            })
    table = pd.DataFrame(rows, columns=["Column", "Type", "Missing", "Infinite",
                                        "Rows affected", "Share of rows"])
    table = table.sort_values("Rows affected", ascending=False, ignore_index=True)
    totals = {
        "columns_with_issues": int(len(table)),
        "rows_with_issues": int(any_bad.sum()),
        "share_rows_with_issues": float(any_bad.sum() / n) if n else 0.0,
    }
    return table, totals


def numeric_statistics(df):
    """count, mean, std, min, quartiles and max of every numeric column."""
    numeric = df.select_dtypes("number")
    if numeric.shape[1] == 0:
        return pd.DataFrame(columns=["Column", "count", "mean", "std", "min",
                                     "25%", "50%", "75%", "max"])
    finite = numeric.replace([np.inf, -np.inf], np.nan)
    stats = finite.describe().T
    stats.index.name = "Column"
    return stats.reset_index()


def label_distribution(df, label_col):
    text = df[label_col].astype("string").str.strip().fillna("(empty)")
    text = text.where(text != "", "(empty)")
    table = text.value_counts().rename_axis("Label value").reset_index(name="Rows")
    table["Share"] = table["Rows"] / len(df)
    return table


def cleaned_copy(df):
    """A copy with +/- infinity replaced by blanks. The original is not changed."""
    out = df.copy()
    numeric = list(out.select_dtypes("number").columns)
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)
    return out


def raw_files_table(raw_inspection):
    """One row per raw CSV, from the pre-computed scan (artifacts/raw_inspection.json)."""
    rows = []
    for e in raw_inspection:
        labels = e["label_counts"]
        benign = int(labels.get("BENIGN", 0))
        parts = e["file"].replace("\\", "/").split("/")
        rows.append({
            "File": "/".join(parts[-2:]),
            "Rows": int(e["rows"]),
            "Columns": int(e["n_columns"]),
            "BENIGN rows": benign,
            "Attack rows": int(e["rows"]) - benign,
            "Labels in the file": "; ".join(
                f"{k}: {v:,}" for k, v in sorted(labels.items(), key=lambda kv: -kv[1])),
            "Cells with NaN": int(sum(e["missing_by_column"].values())),
            "Cells with infinity": int(sum(e["infinite_by_column"].values())),
            "Duplicate rows (approx.)": int(e["duplicate_rows_approx"]),
            "Capture start": e["timestamp_min"],
            "Capture end": e["timestamp_max"],
        })
    return pd.DataFrame(rows)


def snapshot_data_dir():
    """(size, modified time) for every file under data/. Only reads file metadata."""
    if not DATA_DIR.exists():
        return {}
    return {
        str(p.relative_to(ROOT)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in DATA_DIR.rglob("*") if p.is_file()
    }


# ----------------------------------------------------------- self-check ----
def main():
    import io
    import json

    before = snapshot_data_dir()
    print(f"Recorded {len(before)} files in the data folder (size and modified time).")

    # 1) Working sample, compared with numbers saved earlier by other scripts
    with open(ROOT / "artifacts" / "sample_profile.json", encoding="utf-8") as f:
        profile = json.load(f)
    df = load_sample_frame()
    ov = overview(df)
    print(f"\nWorking sample: {ov['rows']:,} rows x {ov['columns']} columns "
          f"({ov['numeric_columns']} numeric, {ov['text_columns']} text), "
          f"{ov['memory_mb']:.1f} MB in memory")
    assert ov["rows"] == profile["rows"] and ov["columns"] == profile["columns"]
    assert ov["identical_rows"] == profile["identical_feature_rows"], (
        ov["identical_rows"], profile["identical_feature_rows"])
    print(f"Rows identical to another row ({ov['columns_compared']} numeric columns): "
          f"{ov['identical_rows']:,}  (describe_sample.py found "
          f"{profile['identical_feature_rows']:,})   [OK]")

    miss, totals = missing_summary(df)
    expected = profile["columns_with_nan_or_inf"]
    assert set(miss["Column"]) == set(expected), set(miss["Column"])
    for _, r in miss.iterrows():
        assert r["Missing"] == expected[r["Column"]]["nan"]
        assert r["Infinite"] == expected[r["Column"]]["infinite"]
    print(f"Missing/infinite values: {totals['columns_with_issues']} columns, "
          f"{totals['rows_with_issues']:,} rows "
          f"({totals['share_rows_with_issues'] * 100:.2f}%)   [OK]")
    print(miss.to_string(index=False))
    leak = ROOT / "artifacts" / "leakage_check.json"
    if leak.exists():
        with open(leak, encoding="utf-8") as f:
            assert totals["rows_with_issues"] == json.load(f)["rate_nan_inf_rows"]
        print("Rows with a problem equal leakage_check.py's count   [OK]")

    stats = numeric_statistics(df)
    assert len(stats) == ov["numeric_columns"]
    counts = stats.set_index("Column")["count"]
    for col, c in expected.items():
        assert counts[col] == len(df) - c["nan"] - c["infinite"]
    print(f"Statistics for {len(stats)} numeric columns; counts exclude "
          f"infinite/NaN values   [OK]")

    label_col = find_label_column(df)
    dist = label_distribution(df, label_col)
    assert dist["Rows"].sum() == len(df)
    benign = int(dist.loc[dist["Label value"] == "BENIGN", "Rows"].sum())
    assert benign == profile["class_counts"]["BENIGN"]
    y, _ = ground_truth_from_labels(df[label_col])
    assert int((y == 1).sum()) == profile["class_counts"]["ATTACK"]
    print(f"\nLabel column {label_col!r}:")
    print(dist.assign(Share=dist["Share"].map("{:.2%}".format)).to_string(index=False))
    print("Binary split matches sample_profile.json   [OK]")

    # 2) Demo files, including the deliberately broken ones
    print("\nDemo files:")
    for name in list_demo_files():
        try:
            d, _ = load_demo_file(name)
            o = overview(d)
            lc = find_label_column(d)
            print(f"  {name:<32}{o['rows']:>6} rows {o['columns']:>3} cols   "
                  f"label column: {lc}")
            if name == "demo_test_day_with_labels.csv":
                assert (o["rows"], o["columns"]) == (400, 79)
                assert set(label_distribution(d, lc)["Rows"]) == {100}
            if name == "demo_text_value.csv":
                assert "Total Fwd Packets" not in numeric_statistics(d)["Column"].tolist()
                m, _ = missing_summary(d)
        except ValueError as err:
            print(f"  {name:<32}rejected: {err}")
            assert name in ("demo_empty.csv", "demo_header_only.csv"), name
    print("Valid demo files profile correctly, broken ones are rejected   [OK]")

    # 3) An upload is read from memory
    buf = io.BytesIO(df.head(500).to_csv(index=False).encode("utf-8"))
    up, truncated = read_upload(buf, MAX_ROWS)
    overview(up)
    missing_summary(up)
    numeric_statistics(up)
    assert len(up) == 500 and not truncated
    print("In-memory upload profiled   [OK]")

    # 4) Cleaned copy
    clean = cleaned_copy(df)
    assert clean.shape == df.shape
    assert not np.isinf(clean.select_dtypes("number").to_numpy(dtype="float64")).any()
    cells = sum(c["nan"] + c["infinite"] for c in expected.values())
    nan_after = int(np.isnan(clean.select_dtypes("number").to_numpy(dtype="float64")).sum())
    assert nan_after == cells, (nan_after, cells)
    assert np.isinf(df.select_dtypes("number").to_numpy(dtype="float64")).any()
    print(f"Cleaned copy: infinity became blanks ({nan_after:,} blank cells), "
          f"the original still has its infinite values   [OK]")

    # 5) Raw-file scan table
    raw_path = ROOT / "artifacts" / "raw_inspection.json"
    if raw_path.exists():
        with open(raw_path, encoding="utf-8") as f:
            table = raw_files_table(json.load(f))
        print(f"\nRaw files: {len(table)} files, {table['Rows'].sum():,} rows, "
              f"{table['BENIGN rows'].sum():,} benign, "
              f"{table['Duplicate rows (approx.)'].sum():,} duplicate rows")
        print(table[["File", "Rows", "BENIGN rows", "Attack rows",
                     "Duplicate rows (approx.)"]].to_string(index=False))

    # 6) A demo name that is not on the list is refused
    for bad in ("..\\..\\raw\\01-12\\Syn.csv", "../sample_20k.parquet", "nope.csv"):
        try:
            load_demo_file(bad)
            raise AssertionError(f"{bad!r} was accepted")
        except ValueError:
            pass
    print("\nNames outside the demo list are refused   [OK]")

    # 7) Nothing in data/ changed
    after = snapshot_data_dir()
    assert before == after, "the data folder changed while profiling!"
    print(f"Data folder unchanged ({len(after)} files: same sizes and modified "
          f"times)   [OK]")
    print("\nALL EXPLORER CHECKS PASSED")


if __name__ == "__main__":
    main()