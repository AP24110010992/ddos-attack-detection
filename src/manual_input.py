"""
Manual-entry helpers for the dashboard's Manual Detection page.

  derive_frame / derive_row : turn form values into the 7-feature row the
                              compact model was trained on
  validate_inputs           : reject impossible values with clear messages
  range_warnings            : flag values outside the training range

This module does NOT import streamlit, so it can be tested on its own.
Run the self-check (from the project root):
    python src\\manual_input.py
"""

import numpy as np
import pandas as pd

from compact_features import COMPACT_FEATURES, compact_from_raw

MS_TO_US = 1000.0                      # the dataset stores Flow Duration in microseconds
MAX_DURATION_MS = 86_400_000.0         # form limit: 24 hours
MAX_PACKETS = 100_000_000              # form limit
MAX_BYTES = 1_000_000_000_000          # form limit
MAX_BYTES_PER_PACKET = 65_535          # an IP packet cannot be larger


def derive_frame(duration_ms, total_packets, total_bytes, ack, protocol):
    """
    Build compact-schema rows from form-style values (arrays or scalars).

    The two rates are CALCULATED, exactly as the dataset defines them:
        Flow Packets/s = Total Packets / (duration in seconds)
        Flow Bytes/s   = Total Bytes   / (duration in seconds)
    When the duration is 0 they are undefined (NaN); the saved pipeline fills
    them with the training-day median, as it did for such rows in training.
    """
    ms = np.atleast_1d(np.asarray(duration_ms, dtype="float64"))
    packets = np.atleast_1d(np.asarray(total_packets, dtype="float64"))
    nbytes = np.atleast_1d(np.asarray(total_bytes, dtype="float64"))
    duration_us = ms * MS_TO_US
    seconds = duration_us / 1e6
    with np.errstate(divide="ignore", invalid="ignore"):
        packets_per_s = np.where(seconds > 0, packets / seconds, np.nan)
        bytes_per_s = np.where(seconds > 0, nbytes / seconds, np.nan)
    return pd.DataFrame({
        "Protocol": np.atleast_1d(protocol),
        "Flow Duration": duration_us,
        "Total Packets": packets,
        "Total Bytes": nbytes,
        "Flow Packets/s": packets_per_s,
        "Flow Bytes/s": bytes_per_s,
        "ACK Flag Count": np.atleast_1d(ack),
    }, columns=COMPACT_FEATURES)


def derive_row(duration_ms, total_packets, total_bytes, ack, protocol):
    """One-row version of derive_frame (what the dashboard form uses)."""
    return derive_frame([duration_ms], [total_packets], [total_bytes],
                        [ack], [protocol])


def validate_inputs(duration_ms, total_packets, total_bytes, ack, protocol,
                    allowed_protocols):
    """Return a list of error messages; an empty list means the input is valid."""
    errors = []

    def as_number(label, value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            errors.append(f"{label} is missing or is not a number.")
            return None
        if not np.isfinite(number):
            errors.append(f"{label} must be a finite number.")
            return None
        return number

    duration = as_number("Flow duration", duration_ms)
    packets = as_number("Total packets", total_packets)
    nbytes = as_number("Total bytes", total_bytes)
    ack_value = as_number("ACK flag count", ack)
    proto = as_number("Protocol", protocol)

    if duration is not None:
        if duration < 0:
            errors.append("Flow duration cannot be negative.")
        elif duration > MAX_DURATION_MS:
            errors.append(f"Flow duration is above the form limit of "
                          f"{MAX_DURATION_MS:,.0f} ms (24 hours).")

    packets_ok = False
    if packets is not None:
        if not packets.is_integer() or packets < 1:
            errors.append("Total packets must be a whole number, at least 1.")
        elif packets > MAX_PACKETS:
            errors.append(f"Total packets is above the form limit of "
                          f"{MAX_PACKETS:,}.")
        else:
            packets_ok = True

    bytes_ok = False
    if nbytes is not None:
        if not nbytes.is_integer() or nbytes < 0:
            errors.append("Total bytes must be a whole number, at least 0.")
        elif nbytes > MAX_BYTES:
            errors.append(f"Total bytes is above the form limit of "
                          f"{MAX_BYTES:,}.")
        else:
            bytes_ok = True

    if packets_ok and bytes_ok and nbytes > packets * MAX_BYTES_PER_PACKET:
        errors.append(
            f"Total bytes cannot exceed {MAX_BYTES_PER_PACKET:,} bytes per "
            f"packet ({packets * MAX_BYTES_PER_PACKET:,.0f} bytes for "
            f"{packets:,.0f} packets).")

    if ack_value is not None and ack_value not in (0.0, 1.0):
        errors.append("ACK flag count must be 0 or 1.")

    if proto is not None and (
            not proto.is_integer() or int(proto) not in allowed_protocols):
        errors.append(f"Protocol {protocol} does not appear in the training "
                      "data, so the model cannot make a defensible "
                      "prediction for it.")
    return errors


def range_warnings(row, ranges):
    """
    Features of a one-row frame outside the training range.
    Returns a list of (feature, value, training_min, training_max).
    """
    found = []
    for col in COMPACT_FEATURES:
        value = float(row.at[row.index[0], col])
        bounds = ranges.get(col)
        if bounds is None or not np.isfinite(value):
            continue
        if value < bounds["min"] or value > bounds["max"]:
            found.append((col, value, bounds["min"], bounds["max"]))
    return found


# ------------------------------------------------------------ self-check ----
def main():
    from data_processing import TARGET, TEST_DAY, load_sample
    from predict import load_metadata, predict_frame

    meta = load_metadata()
    entry = meta["models"]["compact"]
    allowed = {int(p) for p in entry["protocol_values_in_training"]}
    ranges = entry["training_ranges"]

    # 1) Demonstration row (NOT measured traffic): 850 packets in 1.5 seconds
    demo = derive_row(1500.0, 850, 124000, 1, 6)
    assert list(demo.columns) == COMPACT_FEATURES
    assert np.isclose(demo.at[0, "Flow Duration"], 1_500_000.0)
    assert np.isclose(demo.at[0, "Flow Packets/s"], 850 / 1.5)
    assert np.isclose(demo.at[0, "Flow Bytes/s"], 124000 / 1.5)
    print(f"Demo row: duration {demo.at[0, 'Flow Duration']:,.0f} us, "
          f"{demo.at[0, 'Flow Packets/s']:,.2f} packets/s, "
          f"{demo.at[0, 'Flow Bytes/s']:,.2f} bytes/s   [OK]")

    # 2) Zero duration gives undefined rates, and the pipeline still predicts
    zero = derive_row(0.0, 2, 120, 0, 17)
    assert zero[["Flow Packets/s", "Flow Bytes/s"]].isna().all().all()
    p0 = predict_frame(zero, "compact").iloc[0]
    print(f"Zero duration: rates undefined, prediction still works "
          f"({p0['prediction_label']})   [OK]")

    # 3) Validation accepts a good input and rejects impossible ones
    assert validate_inputs(1500.0, 850, 124000, 1, 6, allowed) == []
    bad_cases = {
        "negative duration": (-1.0, 850, 124000, 1, 6),
        "zero packets": (1500.0, 0, 0, 1, 6),
        "fractional packets": (1500.0, 2.5, 100, 1, 6),
        "negative bytes": (1500.0, 850, -5, 1, 6),
        "bytes above 65,535 per packet": (1500.0, 1, 70000, 1, 6),
        "ACK flag of 2": (1500.0, 850, 124000, 2, 6),
        "protocol 1 (not in training)": (1500.0, 850, 124000, 1, 1),
        "NaN duration": (float("nan"), 850, 124000, 1, 6),
        "text instead of a number": (1500.0, "abc", 124000, 1, 6),
        "duration above the form limit": (MAX_DURATION_MS * 2, 850, 124000, 1, 6),
    }
    print("Validation:")
    for name, args in bad_cases.items():
        errs = validate_inputs(*args, allowed)
        assert errs, f"not rejected: {name}"
        print(f"  rejected: {name:<32}-> {errs[0]}")

    # 4) Real held-out rows: rebuilding them from form-style values must
    #    reproduce the training features
    df = load_sample().reset_index(drop=True)
    test = df[df["day"] == TEST_DAY].reset_index(drop=True)
    ref = compact_from_raw(test)[COMPACT_FEATURES]
    rebuilt = derive_frame(
        ref["Flow Duration"].to_numpy(dtype="float64") / MS_TO_US,
        ref["Total Packets"].to_numpy(), ref["Total Bytes"].to_numpy(),
        ref["ACK Flag Count"].to_numpy(), ref["Protocol"].to_numpy())
    for col in ("Protocol", "Total Packets", "Total Bytes", "ACK Flag Count"):
        assert np.array_equal(ref[col].to_numpy(dtype="float64"),
                              rebuilt[col].to_numpy(dtype="float64")), col
    assert np.allclose(ref["Flow Duration"], rebuilt["Flow Duration"], rtol=1e-9)
    positive = (ref["Flow Duration"] > 0).to_numpy()
    print(f"\nRebuilding {len(ref):,} real test-day rows "
          f"({int(positive.sum()):,} with duration > 0):")
    for col in ("Flow Packets/s", "Flow Bytes/s"):
        a = ref[col].to_numpy(dtype="float64")
        b = rebuilt[col].to_numpy(dtype="float64")
        rel = np.abs(a[positive] - b[positive]) / np.maximum(np.abs(b[positive]), 1e-12)
        assert rel.max() < 1e-3, f"{col}: calculated rate differs from the dataset"
        assert not np.isfinite(a[~positive]).any()
        assert np.isnan(b[~positive]).all()
        print(f"  {col:<15} max relative difference from the dataset's own "
              f"value: {rel.max():.2e}   [OK]")

    # 5) Predictions from rebuilt rows agree with predictions from the originals
    p_ref = predict_frame(ref, "compact")
    p_new = predict_frame(rebuilt, "compact")
    flips = int((p_ref["prediction"].to_numpy() != p_new["prediction"].to_numpy()).sum())
    max_dp = float(np.abs(p_ref["attack_probability"].to_numpy()
                          - p_new["attack_probability"].to_numpy()).max())
    print(f"  predictions that changed after rebuilding: {flips} of {len(ref):,} "
          f"(largest probability change {max_dp:.2e})")
    assert flips <= 0.01 * len(ref), "more than 1% of predictions changed"

    # 6) The validator must accept every real row (otherwise it is too strict)
    rejected = 0
    for r in ref.itertuples(index=False):
        if validate_inputs(r[1] / MS_TO_US, r[2], r[3], r[6], r[0], allowed):
            rejected += 1
    print(f"  real rows rejected by validate_inputs: {rejected} of {len(ref):,}")
    assert rejected == 0

    # 7) Range warnings fire for an extreme input
    extreme = derive_row(80_000_000.0, 850, 124000, 1, 6)
    flagged = [w[0] for w in range_warnings(extreme, ranges)]
    assert "Flow Duration" in flagged
    print(f"Range warning for an 80,000,000 ms flow: {flagged}   [OK]")

    # 8) Two REAL held-out test-day records, entered the way the form would
    print("\nTwo real held-out test-day records through the compact model:")
    y = test[TARGET].to_numpy()
    for label, name in ((0, "BENIGN"), (1, "ATTACK")):
        i = int(np.flatnonzero(y == label)[0])
        out = predict_frame(rebuilt.iloc[[i]], "compact").iloc[0]
        print(f"  actual {name:<7} -> predicted {out['prediction_label']:<12} "
              f"(attack probability {out['attack_probability']:.3f})")

    print("\nALL MANUAL-INPUT CHECKS PASSED")


if __name__ == "__main__":
    main()