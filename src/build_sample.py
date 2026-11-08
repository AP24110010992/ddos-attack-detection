"""
Build the working sample (about 20,000 rows) from the four raw CSV files.

Design (documented so it can be defended in the viva):
  * The two capture days stay separate: 01-12 (training day) and
    03-11 (testing day).
  * BENIGN rows: all benign rows of a day are collected, exact duplicates
    are removed, then a random subset is drawn.
  * Attack rows: a random fraction of each attack type is drawn while the
    file is read in chunks (low memory), exact duplicates are removed, and
    the result is trimmed to the quota.
  * The raw files are never modified. Output goes to data/processed/.

Requires artifacts/raw_inspection.json (created by src/inspect_raw.py).

Usage (from the project root):
    python src\\build_sample.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
REPORT_PATH = ROOT / "artifacts" / "raw_inspection.json"
SUMMARY_PATH = ROOT / "artifacts" / "sample_summary.json"
SAMPLE_PATH = OUT_DIR / "sample_20k.parquet"

SEED = 42
CHUNK_ROWS = 200_000
OVERSAMPLE = 1.3  # draw 30% extra so that removing duplicates still leaves enough

BENIGN = "BENIGN"
FILES = [
    ("01-12", "DrDoS_LDAP.csv"),
    ("01-12", "Syn.csv"),
    ("03-11", "LDAP.csv"),
    ("03-11", "Syn.csv"),
]
# Quota of attack rows per (day, ORIGINAL label as written in the file)
ATTACK_QUOTAS = {
    ("01-12", "DrDoS_LDAP"): 4000,
    ("01-12", "Syn"): 4000,
    ("03-11", "LDAP"): 3000,
    ("03-11", "NetBIOS"): 1000,
    ("03-11", "Syn"): 4000,
}
BENIGN_PER_DAY = 2000

# Original label -> harmonised attack name (the two days spell LDAP differently)
ATTACK_NAME = {
    "DrDoS_LDAP": "LDAP",
    "LDAP": "LDAP",
    "NetBIOS": "NetBIOS",
    "Syn": "Syn",
}

STRING_COLS = ["Flow ID", "Source IP", "Destination IP",
               "Timestamp", "SimillarHTTP", "Label"]
META_COLS = ["day", "source_file", "source_row_id"]
FRONT_COLS = ["day", "source_file", "source_row_id",
              "label_original", "attack_type", "label_binary"]


def read_chunks(path):
    """Yield cleaned chunks: stripped column names, string columns as text."""
    for chunk in pd.read_csv(path, chunksize=CHUNK_ROWS, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        chunk = chunk.rename(columns={"Unnamed: 0": "source_row_id"})
        # Force text so a chunk containing only "0" is not read as a number
        for c in STRING_COLS:
            chunk[c] = chunk[c].astype(str).str.strip()
        yield chunk


def collect_rows(rng, report):
    """First pass: gather every benign row and a random draw of attack rows."""
    benign = {"01-12": [], "03-11": []}
    attack = {}
    known_labels = {BENIGN} | set(ATTACK_NAME)

    for day, fname in FILES:
        path = RAW_DIR / day / fname
        key = str(path.relative_to(ROOT))
        if key not in report:
            raise SystemExit(f"{key} not found in {REPORT_PATH.name}. "
                             "Re-run src\\inspect_raw.py.")
        counts = report[key]["label_counts"]
        print(f"\nReading {key}")
        start = time.time()

        for chunk in read_chunks(path):
            chunk["day"] = day
            chunk["source_file"] = fname
            labels = chunk["Label"]

            unknown = set(labels.unique()) - known_labels
            if unknown:
                raise ValueError(f"Unknown label(s) {unknown} in {fname}")

            is_benign = labels == BENIGN
            if is_benign.any():
                benign[day].append(chunk[is_benign])

            for label in labels[~is_benign].unique():
                quota = ATTACK_QUOTAS.get((day, label))
                if quota is None:
                    continue
                p = min(1.0, OVERSAMPLE * quota / counts[label])
                part = chunk[labels == label]
                keep = rng.random(len(part)) < p
                if keep.any():
                    attack.setdefault((day, label), []).append(part[keep])

        print(f"  done in {time.time() - start:.0f}s")
    return benign, attack


def finalize(benign, attack):
    """Remove exact duplicates and trim each group to its quota."""
    parts = []

    print("\nBenign rows")
    for day, frames in benign.items():
        df = pd.concat(frames, ignore_index=True)
        before = len(df)
        df = df.drop_duplicates(subset=[c for c in df.columns if c not in META_COLS])
        take = min(BENIGN_PER_DAY, len(df))
        print(f"  {day}: {before:,} found, {before - len(df):,} duplicates removed, "
              f"{len(df):,} unique, taking {take:,}")
        if take < BENIGN_PER_DAY:
            print(f"  WARNING: fewer than {BENIGN_PER_DAY:,} benign rows available for {day}")
        parts.append(df.sample(n=take, random_state=SEED))

    print("\nAttack rows")
    for (day, label), frames in sorted(attack.items()):
        df = pd.concat(frames, ignore_index=True)
        before = len(df)
        df = df.drop_duplicates(subset=[c for c in df.columns if c not in META_COLS])
        quota = ATTACK_QUOTAS[(day, label)]
        take = min(quota, len(df))
        print(f"  {day} {label}: {before:,} drawn, {before - len(df):,} duplicates removed, "
              f"taking {take:,} (quota {quota:,})")
        if take < quota:
            print(f"  WARNING: quota not met for {day} {label}")
        parts.append(df.sample(n=take, random_state=SEED))

    sample = pd.concat(parts, ignore_index=True)
    sample = sample.rename(columns={"Label": "label_original"})
    sample["attack_type"] = sample["label_original"].map(
        lambda v: BENIGN if v == BENIGN else ATTACK_NAME[v]
    )
    sample["label_binary"] = (sample["label_original"] != BENIGN).astype(int)
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return sample[FRONT_COLS + [c for c in sample.columns if c not in FRONT_COLS]]


def report_and_save(sample):
    print(f"\nSample shape: {sample.shape[0]:,} rows x {sample.shape[1]} columns")
    print("\nRows by attack type and day:")
    print(pd.crosstab(sample["attack_type"], sample["day"], margins=True))
    print("\nBinary label (0 = BENIGN, 1 = attack) by day:")
    print(pd.crosstab(sample["label_binary"], sample["day"], margins=True))

    # Duplicate checks on numeric feature values only (no IDs or timestamps)
    feature_cols = [c for c in sample.select_dtypes("number").columns
                    if c not in ("source_row_id", "label_binary")]
    hashes = pd.util.hash_pandas_object(sample[feature_cols], index=False)
    train_hashes = set(hashes[sample["day"] == "01-12"])
    test_hashes = hashes[sample["day"] == "03-11"]
    print(f"\nRows sharing identical numeric feature values with another row "
          f"in the sample: {int(hashes.duplicated(keep=False).sum()):,}")
    print(f"Test-day rows whose numeric features also appear in the "
          f"training-day sample: {int(test_hashes.isin(train_hashes).sum()):,}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(SAMPLE_PATH, index=False)
    summary = {
        "seed": SEED,
        "rows": int(len(sample)),
        "columns": int(sample.shape[1]),
        "rows_by_day_and_attack_type": {
            f"{d}|{t}": int(n)
            for (d, t), n in sample.groupby(["day", "attack_type"]).size().items()
        },
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved sample  : {SAMPLE_PATH}")
    print(f"Saved summary : {SUMMARY_PATH}")


def main():
    if not REPORT_PATH.exists():
        raise SystemExit("artifacts\\raw_inspection.json not found. "
                         "Run python src\\inspect_raw.py first.")
    with open(REPORT_PATH, encoding="utf-8") as f:
        report = {entry["file"]: entry for entry in json.load(f)}

    rng = np.random.default_rng(SEED)
    benign, attack = collect_rows(rng, report)
    sample = finalize(benign, attack)
    report_and_save(sample)


if __name__ == "__main__":
    main()