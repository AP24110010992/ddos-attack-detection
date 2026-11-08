"""
Create demonstration CSV files for testing the CSV Prediction page.

The two "valid" files contain REAL held-out records: 100 randomly drawn rows
of each traffic type (BENIGN, LDAP, Syn, NetBIOS) from the TEST day (03-11)
of our working sample. The other files are deliberately damaged copies, used
to test the page's error messages. Output goes to data/processed/demo/.

Usage (from the project root):
    python src\\make_demo_csv.py
"""

from pathlib import Path

import pandas as pd

from data_processing import FEATURE_COLUMNS, TEST_DAY, load_sample

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "processed" / "demo"
SEED = 42
PER_TYPE = 100
TYPES = ["BENIGN", "LDAP", "Syn", "NetBIOS"]
EXTRA_COLUMNS = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]


def main():
    df = load_sample()
    test = df[df["day"] == TEST_DAY]
    parts = [test[test["attack_type"] == t].sample(n=PER_TYPE, random_state=SEED)
             for t in TYPES]
    demo = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

    labeled = demo[EXTRA_COLUMNS + FEATURE_COLUMNS].copy()
    labeled["Label"] = demo["label_original"].to_numpy()

    unlabeled = labeled.drop(columns=["Label"]).copy()
    unlabeled.columns = [" " + c for c in unlabeled.columns]   # like the raw files

    missing = labeled.head(50).drop(columns=["Protocol", "Flow Duration"])

    text = labeled.head(50).copy()
    text["Total Fwd Packets"] = text["Total Fwd Packets"].astype(str)
    text.loc[text.index[2], "Total Fwd Packets"] = "abc"

    unseen = labeled.head(50).copy()
    unseen.loc[unseen.index[:5], "Protocol"] = 1

    files = {
        "demo_test_day_with_labels.csv": (labeled, "valid, has a Label column"),
        "demo_test_day_no_labels.csv": (
            unlabeled, "valid, no labels, names start with a space like the raw files"),
        "demo_missing_columns.csv": (missing, "BROKEN: Protocol and Flow Duration removed"),
        "demo_text_value.csv": (text, "BROKEN: the text 'abc' in Total Fwd Packets"),
        "demo_unseen_protocol.csv": (unseen, "valid, but 5 rows use protocol 1 (never in training)"),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing demo files to {OUT_DIR.relative_to(ROOT)}\n")
    print(f"  {'File':<32}{'Rows':>6}{'Cols':>6}{'KB':>8}   Purpose")
    for name, (frame, note) in files.items():
        path = OUT_DIR / name
        frame.to_csv(path, index=False)
        print(f"  {name:<32}{len(frame):>6}{frame.shape[1]:>6}"
              f"{path.stat().st_size / 1024:>8.0f}   {note}")

    (OUT_DIR / "demo_empty.csv").write_bytes(b"")
    (OUT_DIR / "demo_header_only.csv").write_text("a,b,c\n", encoding="utf-8")
    print(f"  {'demo_empty.csv':<32}{0:>6}{0:>6}{0:>8}   BROKEN: a completely empty file")
    print(f"  {'demo_header_only.csv':<32}{0:>6}{3:>6}{0:>8}   BROKEN: header row only, no data")

    print("\nTraffic types in demo_test_day_with_labels.csv:")
    print(labeled["Label"].value_counts().to_string())
    print("\nThese rows come from the test day, so they were NOT used in training.")


if __name__ == "__main__":
    main()