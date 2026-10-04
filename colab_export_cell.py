# ===== EXPORT CELL: paste at the END of the Colab notebook and run it =====
# Run all notebook cells first (Runtime > Run all). This cell only SAVES the
# objects that already exist in the Colab session; it does not train anything.
# It downloads artifacts.zip -> unzip it into the dashboard's artifacts/ folder.
import json, os, shutil
import joblib, sklearn, xgboost
import pandas as pd

required = ["df", "stream_threshold", "features", "categorical_features",
            "numerical_features", "y_test", "y_pred_logistic", "y_pred_rf",
            "y_pred_xgb", "logistic_model", "random_forest_model", "xgboost_model",
            "final_model_results", "feature_importance_df"]
missing = [name for name in required if name not in globals()]
if missing:
    raise NameError("Run all notebook cells above first. Missing: " + ", ".join(missing))

out_dir = "/content/artifacts"
os.makedirs(out_dir, exist_ok=True)

joblib.dump(logistic_model, f"{out_dir}/logistic_model.joblib")
joblib.dump(random_forest_model, f"{out_dir}/random_forest_model.joblib")
joblib.dump(xgboost_model, f"{out_dir}/xgboost_model.joblib")

pd.DataFrame({
    "row_index": y_test.index,
    "y_test": y_test.values,
    "y_pred_logistic": y_pred_logistic,
    "y_pred_rf": y_pred_rf,
    "y_pred_xgb": y_pred_xgb,
}).to_csv(f"{out_dir}/test_predictions.csv", index=False)
final_model_results.to_csv(f"{out_dir}/final_model_results.csv", index=False)
feature_importance_df.to_csv(f"{out_dir}/feature_importance.csv", index=False)

with open(f"{out_dir}/metadata.json", "w") as f:
    json.dump({
        "stream_threshold": float(stream_threshold),
        "features": features,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "pandas_version": pd.__version__,
        "source": "colab",
    }, f, indent=2)

print("scikit-learn", sklearn.__version__, "| xgboost", xgboost.__version__,
      "| pandas", pd.__version__)
print("If these differ from requirements.txt, update requirements.txt to match.")

shutil.make_archive("/content/artifacts", "zip", out_dir)
from google.colab import files
files.download("/content/artifacts.zip")
