"""
Setup smoke test.

Purpose: confirm that every library installs and works together.
NOTE: The data here is synthetic (random numbers). It is used ONLY to test
the tools and is NOT part of the project results.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to a file instead of opening a window
import matplotlib.pyplot as plt

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import streamlit
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Project root = the folder that contains "tests"
ROOT = Path(__file__).resolve().parent.parent


def main():
    # 1. Environment and versions
    print("Python executable:", sys.executable)
    print("Python version   :", sys.version.split()[0])
    for name, module in [
        ("pandas", pd), ("numpy", np), ("scikit-learn", sklearn),
        ("matplotlib", matplotlib), ("seaborn", sns),
        ("joblib", joblib), ("streamlit", streamlit),
    ]:
        print(f"{name:<13}: {module.__version__}")

    # 2. Synthetic data: two features, two classes
    rng = np.random.default_rng(42)
    n = 400
    benign = pd.DataFrame({
        "feature_a": rng.normal(0, 1, n),
        "feature_b": rng.normal(0, 1, n),
        "label": 0,
    })
    attack = pd.DataFrame({
        "feature_a": rng.normal(3, 1, n),
        "feature_b": rng.normal(3, 1, n),
        "label": 1,
    })
    df = pd.concat([benign, attack], ignore_index=True)
    print("\nSynthetic data shape:", df.shape)

    # 3. Check that we can detect infinity and NaN values
    df.loc[0, "feature_a"] = np.inf
    df.loc[1, "feature_b"] = np.nan
    bad_cells = int((~np.isfinite(df[["feature_a", "feature_b"]])).sum().sum())
    print("Non-finite cells planted and detected:", bad_cells)
    assert bad_cells == 2, "Infinity/NaN detection failed"
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # 4. Train a tiny model
    X = df[["feature_a", "feature_b"]]
    y = df["label"]
    model = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
    acc = accuracy_score(y, model.predict(X))
    print(f"Tiny model trained (synthetic data, training accuracy: {acc:.3f})")

    # 5. Save and reload with joblib
    model_path = ROOT / "models" / "_setup_test.joblib"
    joblib.dump(model, model_path)
    reloaded = joblib.load(model_path)
    assert (reloaded.predict(X) == model.predict(X)).all()
    print("Joblib save/load: OK")

    # 6. Draw one seaborn chart and save it
    plot_path = ROOT / "artifacts" / "_setup_test_plot.png"
    sns.scatterplot(data=df, x="feature_a", y="feature_b", hue="label")
    plt.title("Setup test (synthetic data)")
    plt.savefig(plot_path, dpi=100)
    plt.close()
    assert plot_path.exists()
    print("Seaborn/matplotlib chart saved: OK")

    # 7. Clean up the temporary files
    model_path.unlink()
    plot_path.unlink()
    print("Temporary test files removed.")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()