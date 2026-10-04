"""
Checks the files in ./artifacts against the values printed in the Colab
notebook outputs. Exits with code 1 if anything does not match.

    python verify_artifacts.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).parent
A = ROOT / "artifacts"
DATA_FILE = ROOT / "data" / "Most Streamed Artists on Spotify (17_07_2026) V1.1.csv"

# ----- Values copied from the notebook's printed outputs -----
EXPECTED_THRESHOLD = 18020.05                      # cell 17
EXPECTED_COUNTS = {0: 375, 1: 125}                 # cell 17
EXPECTED_MEANS = {0: 10855.594667, 1: 34703.656800}  # cell 48
EXPECTED_METRICS = {                               # cell 52
    "Logistic Regression": [0.75, 0.500000, 0.000000, 0.00, 0.000000],
    "Random Forest":       [0.62, 0.493333, 0.240000, 0.24, 0.240000],
    "XGBoost":             [0.74, 0.546667, 0.444444, 0.16, 0.235294],
}
METRIC_COLS = ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1 Score"]
# Derived from the classification reports in cell 32 (support 75 / 25)
EXPECTED_CM = {
    "y_pred_logistic": [[75, 0], [25, 0]],
    "y_pred_rf":       [[56, 19], [19, 6]],
    "y_pred_xgb":      [[70, 5], [21, 4]],
}
EXPECTED_TOP15 = [                                 # cell 35
    ("categorical__Country of Origin_Mexico", 0.056198),
    ("categorical__Country of Origin_United States", 0.050147),
    ("categorical__Primary Genre_Regional Mexican", 0.049389),
    ("categorical__Country of Origin_Colombia", 0.048257),
    ("categorical__Artist Type_Group", 0.046974),
    ("categorical__Primary Language_Portuguese", 0.044512),
    ("numerical__Debut Year", 0.043146),
    ("categorical__Country of Origin_Brazil", 0.041672),
    ("categorical__Primary Language_English", 0.041029),
    ("categorical__Primary Genre_EDM", 0.040373),
    ("categorical__Primary Genre_Rock", 0.039923),
    ("categorical__Primary Genre_R&B", 0.039476),
    ("categorical__Primary Genre_Pop", 0.038714),
    ("categorical__Sex_Mixed", 0.036938),
    ("categorical__Primary Language_Hindi", 0.034123),
]
EXPECTED_CHI = {                                   # cells 53 / 71
    "Sex": (0.3885, 2, 0.8235),
    "Country of Origin": (45.2673, 42, 0.3372),
    "Primary Language": (12.1101, 10, 0.2778),
    "Primary Genre": (27.9634, 22, 0.1769),
    "Artist Type": (7.1246, 1, 0.0076),
}

failures = []


def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


meta = json.loads((A / "metadata.json").read_text())
df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()
df["Streaming Performance"] = (
    df["Total Streams (in millions)"] >= meta["stream_threshold"]
).astype(int)

print("Target")
check(close(meta["stream_threshold"], EXPECTED_THRESHOLD), f"threshold = {meta['stream_threshold']}")
vc = df["Streaming Performance"].value_counts().to_dict()
check(vc == EXPECTED_COUNTS, f"class counts = {vc}")
means = df.groupby("Streaming Performance")["Total Streams (in millions)"].mean()
for k, v in EXPECTED_MEANS.items():
    check(close(means[k], v, 1e-5), f"mean streams class {k} = {means[k]:.6f}")

print("Model results (final_model_results.csv)")
res = pd.read_csv(A / "final_model_results.csv").set_index("Model")
for m, vals in EXPECTED_METRICS.items():
    got = res.loc[m, METRIC_COLS].tolist()
    check(all(close(g, e, 1e-6) for g, e in zip(got, vals)),
          f"{m}: {[round(g, 6) for g in got]}")

print("Test predictions / confusion matrices")
tp = pd.read_csv(A / "test_predictions.csv")
check(len(tp) == 100, f"test set size = {len(tp)}")
check(tp["y_test"].tolist() == df.loc[tp["row_index"], "Streaming Performance"].tolist(),
      "y_test matches dataset rows")
for col, cm_exp in EXPECTED_CM.items():
    cm = confusion_matrix(tp["y_test"], tp[col], labels=[0, 1]).tolist()
    check(cm == cm_exp, f"{col}: {cm}")

print("Saved models reproduce saved test predictions")
X_test = df.loc[tp["row_index"], meta["features"]]
for fname, col in [("logistic_model.joblib", "y_pred_logistic"),
                   ("random_forest_model.joblib", "y_pred_rf"),
                   ("xgboost_model.joblib", "y_pred_xgb")]:
    model = joblib.load(A / fname)
    check((model.predict(X_test) == tp[col].values).all(), f"{fname} -> {col}")

print("XGBoost feature importance (top 15)")
fi = pd.read_csv(A / "feature_importance.csv")
for i, (name, val) in enumerate(EXPECTED_TOP15):
    check(fi.iloc[i]["Feature"] == name and close(fi.iloc[i]["Importance"], val, 5e-7),
          f"#{i + 1} {fi.iloc[i]['Feature']} = {fi.iloc[i]['Importance']:.6f}")

print("Chi-square tests")
for col, (c, d, p) in EXPECTED_CHI.items():
    stat, pv, dof, _ = chi2_contingency(pd.crosstab(df[col], df["Streaming Performance"]))
    check(round(stat, 4) == c and dof == d and round(pv, 4) == p,
          f"{col}: chi2={round(stat, 4)}, df={dof}, p={round(pv, 4)}")

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("All checks passed: artifacts match the Colab notebook outputs.")
