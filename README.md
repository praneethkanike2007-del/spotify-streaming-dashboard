
# Spotify Artist Streaming Performance Analysis — Interactive Machine Learning Dashboard

A Streamlit + Plotly web dashboard for the Google Colab project
**Spotify_Artist_Success_ML_Project.ipynb**. It is a port of the notebook's
ipywidgets dashboard (section 21) with the same seven sections, calculations,
and interpretation text.

> Interactive visualization of the machine-learning analysis performed in Google
> Colab. **No model retraining is performed by the dashboard.**

| Item | Value (from the notebook) |
|---|---|
| Dataset | Kaggle *Most Streamed Artists on Spotify* — `V1.1.csv`, 500 artists |
| Target | `Streaming Performance` = 1 if `Total Streams (in millions)` ≥ 75th percentile (18,020.05 M) |
| Features | Sex, Country of Origin, Primary Language, Primary Genre, Artist Type, Debut Year |
| Preprocessing | `OneHotEncoder(handle_unknown="ignore")` on 5 categoricals, Debut Year passthrough |
| Split | `train_test_split(test_size=0.20, random_state=42, stratify=y)` → 400 / 100 |
| Models | Logistic Regression, Random Forest (300 trees, balanced), XGBoost (300, depth 4, lr 0.05) |

## Project structure

```
app.py                     Streamlit dashboard (loads files only, never trains)
requirements.txt           Pinned to the versions the models were fitted with
.streamlit/config.toml     Spotify dark theme
data/
  Most Streamed Artists on Spotify (17_07_2026) V1.1.csv
artifacts/                 Exported project results (loaded by app.py)
  logistic_model.joblib      fitted Pipeline
  random_forest_model.joblib fitted Pipeline
  xgboost_model.joblib       fitted Pipeline
  test_predictions.csv       y_test + y_pred_logistic / y_pred_rf / y_pred_xgb
  final_model_results.csv    notebook cell 52 table
  feature_importance.csv     notebook cell 35 table (all encoded features)
  metadata.json              threshold, feature lists, library versions, source
colab_export_cell.py       Paste into Colab to export artifacts/ from the real session
export_artifacts.py        Offline fallback: notebook code re-run locally
verify_artifacts.py        Checks artifacts/ against the numbers printed in the notebook
```

## Dashboard sections

1. **Overview** — totals, class counts, threshold, group means, distribution and Top-N chart.
2. **Artist Explorer** — filters (genre, country, language, sex, artist type, debut-year range),
   match counts, top-25 chart and artist table (up to 100 rows, as in Colab).
3. **Group Analysis** — % Higher Streaming or counts by characteristic, minimum group size,
   descriptive-not-causal note.
4. **Model Performance** — saved metrics for all three models, model selector, confusion matrix and
   classification report computed from the saved test predictions, class-imbalance discussion.
5. **Feature Importance** — saved XGBoost importances with Top-N control, labelled predictive/not causal.
6. **Statistical Tests** — chi-square table (variable, χ², degrees of freedom, p-value,
   significance at α = 0.05, interpretation) plus the Artist Type analysis.
7. **Try a Prediction** — the three saved pipelines' `predict` / `predict_proba` on the six inputs,
   number of models predicting Higher, probability chart, and the notebook's limitations.

## Step 1 — export the real artifacts from Colab (important)

The original notebook never saved its fitted models, so they must be exported once:

1. Open the notebook in Colab and run **Runtime → Run all**.
2. Add a new cell at the end, paste the contents of `colab_export_cell.py`, run it.
3. It downloads `artifacts.zip`. Unzip it and replace everything in the `artifacts/` folder.
4. Check the printed versions. If they are not `scikit-learn 1.6.1` and `xgboost 2.1.4`,
   change those two pins in `requirements.txt` to the printed versions.
5. Optional check: `python verify_artifacts.py` — every line should say `OK`.

Until this is done, `artifacts/` holds files from `export_artifacts.py`, which re-runs the
notebook's exact code locally with scikit-learn 1.6.1 / XGBoost 2.1.4. Those files reproduce the
notebook's threshold, class counts, every metric, every test prediction and confusion matrix,
and the chi-square results. **The XGBoost trees are not bit-identical** (XGBoost's random
subsampling differs between Windows and Colab's Linux), so the XGBoost feature importances and
the prediction probabilities differ slightly. The dashboard shows an orange **Artifact
provenance** banner whenever `metadata.json` does not say `"source": "colab"`.

## Step 2 — run locally

Use Python 3.10–3.13.

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

(macOS/Linux: `source .venv/bin/activate`)

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

Then open http://localhost:8501.

## Step 3 — deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository, including `data/`, `artifacts/` and `.streamlit/`
   (about 7 MB in total).
2. Go to https://share.streamlit.io → **Create app** → choose the repository and branch,
   and set the main file path to `app.py`.
3. Under **Advanced settings**, choose Python **3.12** (or 3.11 / 3.13; all are supported by the
   pinned packages).
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt`; no Colab access is needed.

## Data and model rules

* The app loads only `data/` and `artifacts/`. It never calls `fit`, so no model is retrained.
  The chi-square tests and group percentages are recalculated from the dataset with the
  same pandas/scipy code as the Colab dashboard. These are descriptive statistics, not model training.
* If any required file is missing, the app stops and lists the exact missing files instead of
  generating substitutes.
* If the installed scikit-learn / XGBoost versions differ from those recorded in
  `metadata.json`, the app shows a version-mismatch warning.

## Interpretation (from the project)

* The classes are imbalanced (75% / 25%), so accuracy alone is misleading. Logistic Regression
  reaches 75% accuracy by predicting Lower Streaming for everyone.
* XGBoost has the highest balanced accuracy (≈0.547) and precision (≈0.444), but recall of only 0.16.
* Feature importance is model-based and predictive, not causal.
* Only Artist Type shows a statistically significant association (χ² = 7.1246, p = 0.0076):
  Solo 28.15% vs Group 15.75% Higher Streaming. This is an association, not causation.
* The models classify streaming performance among already highly streamed artists in this
  dataset. They do not predict whether an arbitrary new artist will become successful.

## References

1. Kang, I., Mandulak, M., & Szymanski, B. K. (2022). Analyzing and predicting success of
   professional musicians. *Scientific Reports*.
2. Kaggle dataset: *Most Streamed Artists on Spotify*.
3. scikit-learn and XGBoost documentation.

# spotify-streaming-dashboard
 6e468072370a2224fd9aed5ad3604ed360280e2c
