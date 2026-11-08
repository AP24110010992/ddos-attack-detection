"""
Compact feature schema for the manual-entry form (Page C of the dashboard).

Every compact feature maps to documented CIC-DDoS2019 column(s):

  Compact feature   Source column(s) in the dataset
  ---------------   -----------------------------------------------------
  Protocol          Protocol (6 = TCP, 17 = UDP, 0 = IP protocol 0)
  Flow Duration     Flow Duration (unit is checked in compare_compact.py)
  Total Packets     Total Fwd Packets + Total Backward Packets
  Total Bytes       Total Length of Fwd Packets + Total Length of Bwd Packets
  Flow Packets/s    Flow Packets/s
  Flow Bytes/s      Flow Bytes/s
  ACK Flag Count    ACK Flag Count
  SYN Flag Count    SYN Flag Count (optional; tested in compare_compact.py)

Training and the dashboard both use these functions, so the compact schema
cannot drift between them.

Run the self-check (from the project root):
    python src\\compact_features.py
"""

import pandas as pd

from data_processing import load_sample

COMPACT_FEATURES = [
    "Protocol", "Flow Duration", "Total Packets", "Total Bytes",
    "Flow Packets/s", "Flow Bytes/s", "ACK Flag Count",
]
OPTIONAL_FEATURES = ["SYN Flag Count"]
ALL_COMPACT_COLUMNS = COMPACT_FEATURES + OPTIONAL_FEATURES

RAW_COLUMNS_NEEDED = [
    "Protocol", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Flow Packets/s", "Flow Bytes/s", "ACK Flag Count", "SYN Flag Count",
]


def prepare_numeric(df, columns):
    """Select `columns` (in this order); every value must be a number or empty."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} required column(s) missing: {', '.join(missing)}"
        )
    raw = df[columns]
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    bad = numeric.isna() & raw.notna()      # had a value, but not a number
    if bad.to_numpy().any():
        names = [c for c in columns if bad[c].any()]
        raise ValueError(f"Non-numeric values found in column(s): {names}")
    return numeric


def compact_from_raw(df):
    """Build the compact columns from raw CIC-DDoS2019 columns."""
    num = prepare_numeric(df, RAW_COLUMNS_NEEDED)
    out = pd.DataFrame({
        "Protocol": num["Protocol"],
        "Flow Duration": num["Flow Duration"],
        "Total Packets": num["Total Fwd Packets"] + num["Total Backward Packets"],
        "Total Bytes": (num["Total Length of Fwd Packets"]
                        + num["Total Length of Bwd Packets"]),
        "Flow Packets/s": num["Flow Packets/s"],
        "Flow Bytes/s": num["Flow Bytes/s"],
        "ACK Flag Count": num["ACK Flag Count"],
        "SYN Flag Count": num["SYN Flag Count"],
    })
    return out[ALL_COMPACT_COLUMNS]


def main():
    df = load_sample()
    compact = compact_from_raw(df)
    print(f"Compact frame: {compact.shape[0]:,} rows x {compact.shape[1]} columns")
    print(compact.head(3).to_string())

    assert list(compact.columns) == ALL_COMPACT_COLUMNS
    expected_packets = df["Total Fwd Packets"] + df["Total Backward Packets"]
    expected_bytes = (df["Total Length of Fwd Packets"]
                      + df["Total Length of Bwd Packets"])
    assert (compact["Total Packets"] == expected_packets).all()
    assert (compact["Total Bytes"] == expected_bytes).all()
    print("\nDerived columns equal the raw sums          [OK]")

    # A form-style row. DEMONSTRATION values only, not measured traffic.
    # 1.5 seconds = 1,500,000 microseconds.
    demo = pd.DataFrame([{
        "Protocol": 6, "Flow Duration": 1_500_000, "Total Packets": 850,
        "Total Bytes": 124_000, "Flow Packets/s": 566.67,
        "Flow Bytes/s": 82_666.7, "ACK Flag Count": 1,
    }])
    assert list(prepare_numeric(demo, COMPACT_FEATURES).columns) == COMPACT_FEATURES
    print("Form-style demo row accepted                [OK]")

    try:
        prepare_numeric(compact.drop(columns=["Flow Duration"]), COMPACT_FEATURES)
        raise AssertionError("missing column was not detected")
    except ValueError as err:
        print(f"Missing-column check                        [OK]  ({err})")

    bad = compact.head(3).copy()
    bad["Protocol"] = ["abc", "6", "17"]
    try:
        prepare_numeric(bad, COMPACT_FEATURES)
        raise AssertionError("non-numeric value was not detected")
    except ValueError as err:
        print(f"Non-numeric check                           [OK]  ({err})")

    print("\nALL COMPACT-FEATURE CHECKS PASSED")


if __name__ == "__main__":
    main()