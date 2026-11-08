"""
Inspect the raw CIC-DDoS2019 CSV files WITHOUT modifying them.

Reads each file in chunks (low memory) and reports real column names,
label values, missing/infinite values, constant columns, duplicates and
timestamp range. A full report is saved to artifacts/raw_inspection.json.

Usage (from the project root):
    python src\\inspect_raw.py
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
REPORT_PATH = ROOT / "artifacts" / "raw_inspection.json"
CHUNK_ROWS = 200_000

# Text values that mean "infinite" or "missing" when a numeric column was
# read as text.
SPECIAL_TOKENS = {"infinity", "-infinity", "inf", "-inf", "nan", ""}


def find_column(columns, wanted):
    """Find a column whose name matches `wanted` ignoring spaces and case."""
    for col in columns:
        if col.strip().lower() == wanted:
            return col
    return None


def inspect_file(path):
    """Scan one CSV in chunks and return a dictionary of findings."""
    header = pd.read_csv(path, nrows=0)
    columns = list(header.columns)
    label_col = find_column(columns, "label")
    time_col = find_column(columns, "timestamp")

    # Columns named "Unnamed: ..." are just a saved row index. They are
    # excluded when looking for duplicate rows, otherwise no two rows could
    # ever match.
    hash_cols = [c for c in columns if not c.startswith("Unnamed")]

    total_rows = 0
    label_counts = Counter()
    missing = pd.Series(0, index=columns, dtype="int64")
    inf_counts = {}
    special_text = {}
    col_min, col_max = {}, {}
    hash_parts = []
    ts_min, ts_max = None, None
    dtypes_first = None
    non_numeric = {}

    start = time.time()
    reader = pd.read_csv(path, chunksize=CHUNK_ROWS, low_memory=False)
    for i, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)

        if dtypes_first is None:
            dtypes_first = {c: str(t) for c, t in chunk.dtypes.items()}
            for c in columns:
                if not pd.api.types.is_numeric_dtype(chunk[c]):
                    non_numeric[c] = {
                        "dtype": str(chunk[c].dtype),
                        "examples": [str(v) for v in chunk[c].dropna().unique()[:3]],
                    }

        if label_col is not None:
            label_counts.update(chunk[label_col].value_counts(dropna=False).to_dict())

        missing += chunk.isna().sum()

        numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(chunk[c])]
        if numeric_cols:
            num = chunk[numeric_cols]
            inf_here = np.isinf(num).sum()
            for c, n in inf_here.items():
                if n:
                    inf_counts[c] = inf_counts.get(c, 0) + int(n)
            finite = num.replace([np.inf, -np.inf], np.nan)
            for c, v in finite.min().items():
                if pd.notna(v):
                    col_min[c] = min(col_min.get(c, v), v)
            for c, v in finite.max().items():
                if pd.notna(v):
                    col_max[c] = max(col_max.get(c, v), v)

        for c in non_numeric:
            if c == label_col or c == time_col:
                continue
            tokens = chunk[c].astype(str).str.strip().str.lower()
            n = int(tokens.isin(SPECIAL_TOKENS).sum())
            if n:
                special_text[c] = special_text.get(c, 0) + n

        if time_col is not None:
            t = chunk[time_col].astype(str)
            lo, hi = t.min(), t.max()
            ts_min = lo if ts_min is None else min(ts_min, lo)
            ts_max = hi if ts_max is None else max(ts_max, hi)

        hash_parts.append(
            pd.util.hash_pandas_object(chunk[hash_cols], index=False).to_numpy()
        )

        print(f"    chunk {i}: {total_rows:,} rows read "
              f"({time.time() - start:.0f}s)", flush=True)

    all_hashes = np.concatenate(hash_parts)
    duplicates = int(len(all_hashes) - len(np.unique(all_hashes)))

    constant_cols = [
        c for c in col_min if c in col_max and col_min[c] == col_max[c]
    ]

    return {
        "file": str(path.relative_to(ROOT)),
        "rows": int(total_rows),
        "columns": columns,
        "n_columns": len(columns),
        "label_column": label_col,
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "missing_by_column": {c: int(n) for c, n in missing.items() if n},
        "infinite_by_column": inf_counts,
        "special_text_by_column": special_text,
        "non_numeric_columns": non_numeric,
        "constant_numeric_columns": constant_cols,
        "duplicate_rows_approx": duplicates,
        "timestamp_min": ts_min,
        "timestamp_max": ts_max,
    }


def print_summary(r):
    print(f"\n  Rows: {r['rows']:,}   Columns: {r['n_columns']}")
    print(f"  Label column: {r['label_column']!r}")
    print("  Label counts:")
    for label, n in sorted(r["label_counts"].items(), key=lambda kv: -kv[1]):
        print(f"    {label!r:<28}{n:>12,}   ({100 * n / r['rows']:.4f}%)")
    print(f"  Timestamp range: {r['timestamp_min']}  ->  {r['timestamp_max']}")
    print(f"  Columns with missing values: {len(r['missing_by_column'])}")
    for c, n in list(r["missing_by_column"].items())[:10]:
        print(f"    {c!r}: {n:,}")
    print(f"  Numeric columns containing +/-infinity: {len(r['infinite_by_column'])}")
    for c, n in list(r["infinite_by_column"].items())[:10]:
        print(f"    {c!r}: {n:,}")
    if r["special_text_by_column"]:
        print("  Text columns with Infinity/NaN/blank tokens:")
        for c, n in r["special_text_by_column"].items():
            print(f"    {c!r}: {n:,}")
    print(f"  Non-numeric columns ({len(r['non_numeric_columns'])}):")
    for c, info in r["non_numeric_columns"].items():
        print(f"    {c!r} [{info['dtype']}] e.g. {info['examples']}")
    print(f"  Constant numeric columns: {len(r['constant_numeric_columns'])}")
    print(f"  Duplicate rows (hash-based, excluding index column): "
          f"{r['duplicate_rows_approx']:,}")


def main():
    files = sorted(RAW_DIR.rglob("*.csv"))
    if not files:
        print(f"No CSV files found under {RAW_DIR}")
        return

    reports = []
    for path in files:
        print(f"\n=== {path.relative_to(ROOT)} "
              f"({path.stat().st_size / 1024 ** 2:,.0f} MB) ===")
        report = inspect_file(path)
        print_summary(report)
        reports.append(report)

    # Cross-file comparison of column names (after stripping spaces)
    print("\n=== Column names across files ===")
    base = [c.strip() for c in reports[0]["columns"]]
    all_same = True
    for r in reports[1:]:
        cols = [c.strip() for c in r["columns"]]
        if cols != base:
            all_same = False
            print(f"  DIFFERENT: {r['file']}")
            print(f"    only in first file : {sorted(set(base) - set(cols))}")
            print(f"    only in this file  : {sorted(set(cols) - set(base))}")
    if all_same:
        print("  All files have identical column names and order "
              "(after stripping spaces).")

    spaced = [c for c in reports[0]["columns"] if c != c.strip()]
    print(f"  Columns with leading/trailing spaces in first file: {len(spaced)}")

    print("\n=== Column names (first file) ===")
    for i, c in enumerate(reports[0]["columns"]):
        print(f"  {i:>3}  {c!r}")

    REPORT_PATH.parent.mkdir(exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, default=str)
    print(f"\nFull report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()