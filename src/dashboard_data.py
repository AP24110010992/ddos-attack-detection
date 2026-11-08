"""
Data helpers for the dashboard. Everything shown on the dashboard is read
from the artifact files created by the training pipeline, so no performance
number is ever typed in by hand.

This module does NOT import streamlit, so it can be tested on its own.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

TYPE_ORDER = ["BENIGN", "LDAP", "Syn", "NetBIOS"]


def load_json(name):
    """Return the parsed JSON file from artifacts/, or None if it is missing."""
    path = ARTIFACTS / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def raw_totals(raw_inspection):
    """Totals over the raw CSV files scanned by src/inspect_raw.py."""
    rows = sum(e["rows"] for e in raw_inspection)
    benign = sum(e["label_counts"].get("BENIGN", 0) for e in raw_inspection)
    return {
        "files": len(raw_inspection),
        "rows": rows,
        "benign": benign,
        "attack": rows - benign,
    }


def sample_distribution(profile):
    """Rows per traffic type (index) and capture day (columns) in the sample."""
    records = []
    for key, n in profile["rows_by_day_and_attack_type"].items():
        day, kind = key.split("|")
        records.append({"day": day, "type": kind, "rows": n})
    table = pd.DataFrame(records).pivot(index="type", columns="day", values="rows")
    table = table.fillna(0).astype(int)
    order = [t for t in TYPE_ORDER if t in table.index]
    order += [t for t in table.index if t not in order]
    return table.loc[order]


def types_seen_in_training(profile, train_day):
    """Traffic types that occur in the training-day sample."""
    return {k.split("|")[1] for k in profile["rows_by_day_and_attack_type"]
            if k.startswith(train_day + "|")}


def pct(value, digits=1):
    """Format a fraction as a percentage string; NaN becomes 'n/a'."""
    if value is None or value != value:
        return "n/a"
    return f"{value * 100:.{digits}f}%"