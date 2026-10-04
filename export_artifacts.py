"""
Offline export of the Colab project's fitted models and results.

The dashboard (app.py) NEVER trains anything. It only loads the files that
this script writes into ./artifacts.

Preferred route: run the export cell from `colab_export_cell.py` at the end of
the Google Colab notebook, which saves the exact in-memory objects from Colab.

This script is the fallback when the Colab objects are not available. Every
modelling line below is copied verbatim from the notebook
(Spotify_Artist_Success_ML_Project.ipynb, cells 5-35). The settings are fully
deterministic (fixed split, random_state=42 everywhere), and verify_artifacts.py
checks the output against the numbers printed in the notebook.

    python export_artifacts.py
    python verify_artifacts.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "data" / "Most Streamed Artists on Spotify (17_07_2026) V1.1.csv"
OUT = ROOT / "artifacts"

# ---------- Notebook cell 5 / 7: load + clean column names ----------
df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

# ---------- Notebook cell 17 / 18: target ----------
stream_threshold = df["Total Streams (in millions)"].quantile(0.75)
df["Streaming Performance"] = (
    df["Total Streams (in millions)"] >= stream_threshold
).astype(int)
df["Streaming Performance Label"] = df["Streaming Performance"].map({
    0: "Lower Streaming Performance",
    1: "Higher Streaming Performance"
})

# ---------- Notebook cell 22 / 23: features ----------
features = [
    "Sex",
    "Country of Origin",
    "Primary Language",
    "Primary Genre",
    "Artist Type",
    "Debut Year"
]
X = df[features]
y = df["Streaming Performance"]

categorical_features = [
    "Sex",
    "Country of Origin",
    "Primary Language",
    "Primary Genre",
    "Artist Type"
]
numerical_features = [
    "Debut Year"
]

# ---------- Notebook cell 24: split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ---------- Notebook cell 25: preprocessing ----------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

# ---------- Notebook cells 27-29: models ----------
logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]
)
logistic_model.fit(X_train, y_train)

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        ))
    ]
)
random_forest_model.fit(X_train, y_train)

xgboost_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        ))
    ]
)
xgboost_model.fit(X_train, y_train)

# ---------- Notebook cell 30: test predictions ----------
y_pred_logistic = logistic_model.predict(X_test)
y_pred_rf = random_forest_model.predict(X_test)
y_pred_xgb = xgboost_model.predict(X_test)

# ---------- Notebook cell 52: final_model_results ----------
predictions = {
    "Logistic Regression": y_pred_logistic,
    "Random Forest": y_pred_rf,
    "XGBoost": y_pred_xgb
}
balanced_results = []
for model_name, predictions_value in predictions.items():
    balanced_results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions_value),
        "Balanced Accuracy": balanced_accuracy_score(y_test, predictions_value),
        "Precision": precision_score(y_test, predictions_value, zero_division=0),
        "Recall": recall_score(y_test, predictions_value, zero_division=0),
        "F1 Score": f1_score(y_test, predictions_value, zero_division=0)
    })
final_model_results = pd.DataFrame(balanced_results)

# ---------- Notebook cell 35: XGBoost feature importance ----------
xgb_model = xgboost_model.named_steps["model"]
xgb_preprocessor = xgboost_model.named_steps["preprocessor"]
feature_names = xgb_preprocessor.get_feature_names_out()
importance = xgb_model.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})
feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)


# ---------- Save (identical to colab_export_cell.py) ----------
def save_artifacts(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(logistic_model, out_dir / "logistic_model.joblib")
    joblib.dump(random_forest_model, out_dir / "random_forest_model.joblib")
    joblib.dump(xgboost_model, out_dir / "xgboost_model.joblib")
    pd.DataFrame({
        "row_index": y_test.index,
        "y_test": y_test.values,
        "y_pred_logistic": y_pred_logistic,
        "y_pred_rf": y_pred_rf,
        "y_pred_xgb": y_pred_xgb,
    }).to_csv(out_dir / "test_predictions.csv", index=False)
    final_model_results.to_csv(out_dir / "final_model_results.csv", index=False)
    feature_importance_df.to_csv(out_dir / "feature_importance.csv", index=False)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({
            "stream_threshold": float(stream_threshold),
            "features": features,
            "categorical_features": categorical_features,
            "numerical_features": numerical_features,
            "sklearn_version": sklearn.__version__,
            "xgboost_version": xgboost.__version__,
            "pandas_version": pd.__version__,
            "source": "export_artifacts.py (notebook code re-run locally)",
        }, f, indent=2)


if __name__ == "__main__":
    save_artifacts(OUT)
    print("Artifacts written to", OUT.resolve())
    print(final_model_results.to_string(index=False))
