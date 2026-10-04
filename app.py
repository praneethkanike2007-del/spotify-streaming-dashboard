"""
Spotify Artist Streaming Performance Analysis - Interactive Machine Learning Dashboard

Streamlit + Plotly port of the ipywidgets dashboard in
Spotify_Artist_Success_ML_Project.ipynb (section 21, dashboard cells 1-5).

The app only LOADS the exported dataset, fitted models and results from
./data and ./artifacts. It never fits, refits or tunes any model.

    streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import sklearn
import streamlit as st
import xgboost
from scipy.stats import chi2_contingency
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# Paths and constants (names follow the Colab dashboard)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_FILE = ROOT / "data" / "Most Streamed Artists on Spotify (17_07_2026) V1.1.csv"
ART = ROOT / "artifacts"
REQUIRED_FILES = [
    DATA_FILE,
    ART / "metadata.json",
    ART / "logistic_model.joblib",
    ART / "random_forest_model.joblib",
    ART / "xgboost_model.joblib",
    ART / "test_predictions.csv",
    ART / "final_model_results.csv",
    ART / "feature_importance.csv",
]

TARGET = "Streaming Performance"
STREAMS = "Total Streams (in millions)"
LOWER, HIGHER = "Lower Streaming", "Higher Streaming"
MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost"]
MODEL_FILES = {
    "Logistic Regression": "logistic_model.joblib",
    "Random Forest": "random_forest_model.joblib",
    "XGBoost": "xgboost_model.joblib",
}
PRED_COLS = {
    "Logistic Regression": "y_pred_logistic",
    "Random Forest": "y_pred_rf",
    "XGBoost": "y_pred_xgb",
}
METRICS = ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1 Score"]
GROUP_COLUMNS = ["Primary Genre", "Country of Origin", "Primary Language", "Artist Type", "Sex"]

# Spotify-inspired dark theme (same values as the Colab dashboard)
BG, PANEL, PANEL_2, TEXT, MUTED = "#121212", "#181818", "#202020", "#FFFFFF", "#B3B3B3"
GREEN, GREY, GRID = "#1DB954", "#7F7F7F", "#2A2A2A"
GROUP_COLORS = {LOWER: GREY, HIGHER: GREEN}
MODEL_COLORS = {"Logistic Regression": "#7F7F7F", "Random Forest": "#B3B3B3", "XGBoost": GREEN}

st.set_page_config(
    page_title="Spotify Artist Streaming Performance Analysis",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, button, input, select, textarea {{
    font-family:'Inter', 'Segoe UI', Helvetica, Arial, sans-serif; }}
.stApp {{ background:
    radial-gradient(1200px 500px at 85% -10%, rgba(29,185,84,0.10), transparent 60%), {BG}; }}
header[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ padding-top:4.5rem; padding-bottom:3rem; max-width:1280px; }}
h1, h2, h3, h4 {{ color:{TEXT}; letter-spacing:-0.015em; }}

/* ---------- Hero banner ---------- */
.hero {{ position:relative; overflow:hidden; border-radius:18px; padding:30px 34px 26px 34px;
    margin-bottom:20px; border:1px solid rgba(29,185,84,0.28);
    background:linear-gradient(120deg, #0F3D22 0%, #10261A 38%, {PANEL} 75%); }}
.hero-eq {{ position:absolute; right:34px; top:50%; transform:translateY(-50%);
    display:flex; align-items:flex-end; gap:7px; height:110px; opacity:0.9; }}
.hero-eq i {{ display:block; width:12px; border-radius:6px 6px 2px 2px;
    background:linear-gradient(180deg, {GREEN}, rgba(29,185,84,0.25)); }}
.hero-kicker {{ display:inline-flex; align-items:center; gap:8px; color:{GREEN};
    font-size:0.78rem; font-weight:700; letter-spacing:0.14em; text-transform:uppercase; }}
.hero-kicker .dot {{ width:8px; height:8px; border-radius:50%; background:{GREEN};
    box-shadow:0 0 0 4px rgba(29,185,84,0.18); }}
.hero-title {{ font-size:2.55rem; font-weight:800; color:{TEXT}; margin:10px 0 0 0;
    line-height:1.1; letter-spacing:-0.025em; max-width:820px; }}
.hero-title span {{ color:{GREEN}; }}
.hero-sub {{ color:#D6D6D6; font-size:1.18rem; font-weight:500; margin:8px 0 16px 0; }}
.hero-chips {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px; max-width:820px; }}
.hero-chip {{ background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.10);
    color:{TEXT}; border-radius:999px; padding:6px 13px; font-size:0.82rem; font-weight:500; }}
.hero-chip b {{ color:{GREEN}; font-weight:700; }}
.hero-note {{ color:{MUTED}; font-size:0.84rem; border-top:1px solid rgba(255,255,255,0.08);
    padding-top:12px; max-width:820px; }}
.hero-note b {{ color:{GREEN}; font-weight:600; }}
@media (max-width: 900px) {{
    .hero {{ padding:22px 20px; }} .hero-eq {{ display:none; }}
    .hero-title {{ font-size:1.8rem; }} .hero-sub {{ font-size:1rem; }}
}}

/* ---------- Section headings ---------- */
div[data-testid="stHeadingWithActionElements"] h3 {{ display:flex; align-items:center; gap:10px; }}
div[data-testid="stHeadingWithActionElements"] h3::before {{ content:""; width:5px; height:1.05em;
    border-radius:3px; background:{GREEN}; display:inline-block; }}

.card {{ background:linear-gradient(160deg, {PANEL_2} 0%, {PANEL} 100%); border:1px solid {GRID};
    border-left:4px solid {GREEN}; border-radius:12px; padding:14px 16px; height:100%;
    box-shadow:0 1px 0 rgba(255,255,255,0.03) inset, 0 6px 18px rgba(0,0,0,0.25);
    transition:border-color .15s ease; }}
.card:hover {{ border-color:#3A3A3A; border-left-color:{GREEN}; }}
.card .lbl {{ color:{MUTED}; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em; }}
.card .val {{ color:{TEXT}; font-size:1.55rem; font-weight:700; margin-top:2px;
    font-variant-numeric:tabular-nums; }}
.card .sub {{ color:{GREEN}; font-size:0.78rem; margin-top:2px; }}
.note {{ background:{PANEL}; border-left:4px solid {GREEN}; border-radius:8px;
    padding:12px 16px; margin:10px 0 14px 0; color:{TEXT}; font-size:0.93rem; line-height:1.55; }}
.note.grey {{ border-left-color:{GREY}; color:#DADADA; }}
.note.warn {{ border-left-color:#F59B23; }}
.section-cap {{ color:{MUTED}; font-size:0.92rem; margin-top:-6px; margin-bottom:12px; }}
.stTabs [data-baseweb="tab-list"], .stTabs [role="tablist"] {{ gap:6px; height:auto; flex-wrap:wrap; background:{PANEL};
    border:1px solid {GRID}; border-radius:14px; padding:6px; margin-bottom:8px; }}
.stTabs [data-baseweb="tab"], .stTabs [role="tab"] {{ background:transparent; border-radius:10px;
    height:auto; padding:9px 16px !important; color:{MUTED}; font-weight:600; }}
.stTabs [role="tab"] p {{ font-weight:600; font-size:0.92rem; }}
.stTabs [data-baseweb="tab"]:hover, .stTabs [role="tab"]:hover {{ color:{TEXT}; background:{PANEL_2}; }}
.stTabs [aria-selected="true"] {{ color:#0B0B0B !important; background:{GREEN} !important; }}
.stTabs [aria-selected="true"] p {{ color:#0B0B0B !important; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display:none; }}
div[data-testid="stForm"] {{ background:{PANEL}; border:1px solid {GRID}; border-radius:10px; }}
.footer {{ color:#8A8A8A; font-size:0.8rem; border-top:1px solid {GRID};
    padding-top:12px; margin-top:28px; }}
</style>
""",
    unsafe_allow_html=True,
)


def card(label, value, sub=""):
    return (f"<div class='card'><div class='lbl'>{label}</div>"
            f"<div class='val'>{value}</div><div class='sub'>{sub or '&nbsp;'}</div></div>")


def note(text, kind=""):
    st.markdown(f"<div class='note {kind}'>{text}</div>", unsafe_allow_html=True)


def cards_row(items):
    for col, (label, value, sub) in zip(st.columns(len(items)), items):
        col.markdown(card(label, value, sub), unsafe_allow_html=True)


def style(fig, title, height=450):
    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color=TEXT), x=0, xanchor="left"),
        template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=13),
        height=height, margin=dict(l=20, r=30, t=70, b=40),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, title=None),
        hoverlabel=dict(bgcolor=PANEL_2, font_color=TEXT, bordercolor=GRID),
        bargap=0.25,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def table(df, height=None, highlight=None, formats=None):
    styler = df.style
    if formats:
        styler = styler.format(formats)
    if highlight is not None:
        styler = styler.apply(
            lambda r: [f"background-color:#1F3D2B" if highlight(r) else "" for _ in r], axis=1)
    kwargs = {"height": height} if height else {}
    st.dataframe(styler, width="stretch", hide_index=True, **kwargs)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
EQ_BARS = [38, 72, 54, 96, 64, 110, 82, 48, 90, 60, 34]
st.markdown(
    "<div class='hero'>"
    "<div class='hero-eq'>" + "".join(f"<i style='height:{h}px'></i>" for h in EQ_BARS) + "</div>"
    "<div class='hero-kicker'><span class='dot'></span>Research Project · Spotify Streaming Data</div>"
    "<div class='hero-title' role='heading' aria-level='1'>Spotify Artist <span>Streaming Performance</span> Analysis</div>"
    "<div class='hero-sub'>Interactive Machine Learning Dashboard</div>"
    "<div class='hero-chips'>"
    "<span class='hero-chip'><b>500</b> artists</span>"
    "<span class='hero-chip'><b>3</b> models · Logistic Regression, Random Forest, XGBoost</span>"
    "<span class='hero-chip'><b>6</b> artist features</span>"
    "<span class='hero-chip'>Target: top <b>25%</b> by Total Streams</span>"
    "</div>"
    "<div class='hero-note'>Interactive visualization of the machine-learning analysis performed in "
    "Google Colab. <b>No model retraining is performed by the dashboard.</b></div>"
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Required-file check: stop with an exact list instead of inventing anything
# ---------------------------------------------------------------------------
missing = [p.relative_to(ROOT).as_posix() for p in REQUIRED_FILES if not p.exists()]
if missing:
    st.error(
        "The dashboard cannot start because these exported project files are missing:\n\n"
        + "\n".join(f"- `{m}`" for m in missing)
        + "\n\nExport them from the Colab notebook with `colab_export_cell.py` "
        "(see README.md). The dashboard does not train models to fill the gap."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Cached loaders (read-only)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_metadata():
    return json.loads((ART / "metadata.json").read_text())


@st.cache_data(show_spinner=False)
def load_data(stream_threshold):
    # Notebook cells 5, 7, 17, 18 and dashboard cell 2
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()
    df[TARGET] = (df[STREAMS] >= stream_threshold).astype(int)
    df["Group"] = df[TARGET].map({0: LOWER, 1: HIGHER})
    return df


@st.cache_data(show_spinner=False)
def load_results():
    return (
        pd.read_csv(ART / "final_model_results.csv"),
        pd.read_csv(ART / "test_predictions.csv"),
        pd.read_csv(ART / "feature_importance.csv"),
    )


@st.cache_resource(show_spinner=False)
def load_models():
    return {name: joblib.load(ART / f) for name, f in MODEL_FILES.items()}


@st.cache_data(show_spinner=False)
def chi_square_table(df, categorical_features):
    # Same test and output as dashboard cell 2 (dash_chi_df)
    rows = []
    for col in categorical_features:
        ct = pd.crosstab(df[col], df[TARGET])
        stat, p, dof, _ = chi2_contingency(ct)
        sig = p < 0.05
        rows.append({
            "Variable": col,
            "Chi-square": round(stat, 4),
            "Degrees of freedom": int(dof),
            "p-value": round(p, 4),
            "Significant at alpha = 0.05": "Yes" if sig else "No",
            "Interpretation": (
                "Statistically significant association with streaming group"
                if sig else "No sufficient evidence of an association"),
        })
    return pd.DataFrame(rows)


def group_stats(df, col):
    # dash_group_stats from dashboard cell 2
    t = df.groupby(col)[TARGET].agg(n="size", higher="sum").reset_index()
    t["lower"] = t["n"] - t["higher"]
    t["pct_higher"] = 100 * t["higher"] / t["n"]
    return t


def clean_feature(name, categorical_features):
    # dash_clean_feature from dashboard cell 2
    name = str(name).split("__", 1)[-1]
    for c in categorical_features:
        if name.startswith(c + "_"):
            return f"{c}: {name[len(c) + 1:]}"
    return name


try:
    meta = load_metadata()
    features = meta["features"]
    categorical_features = meta["categorical_features"]
    stream_threshold = float(meta["stream_threshold"])
    df = load_data(stream_threshold)
    final_model_results, test_preds, feature_importance_df = load_results()
    models = load_models()
except Exception as exc:  # corrupt / incompatible export
    st.error(f"Could not load the exported project files: {exc}")
    st.stop()

# Consistency guards: the exported threshold must equal the notebook's rule
if abs(df[STREAMS].quantile(0.75) - stream_threshold) > 1e-9:
    st.error("metadata.json stream_threshold does not match the 75th percentile of the dataset.")
    st.stop()

if meta.get("source", "").lower() != "colab":
    note(
        "<b>Artifact provenance:</b> the model files in <code>artifacts/</code> were not exported "
        "from the Colab session (" + meta.get("source", "unknown") + "). Metrics, predictions and "
        "confusion matrices match the notebook, but the XGBoost feature importances and predicted "
        "probabilities can differ slightly from Colab. Replace the folder with the export from "
        "<code>colab_export_cell.py</code> before presenting results.", "warn")

version_issues = [
    f"{lib} {meta[key]} (export) vs {cur} (installed)"
    for lib, key, cur in [("scikit-learn", "sklearn_version", sklearn.__version__),
                          ("xgboost", "xgboost_version", xgboost.__version__)]
    if key in meta and meta[key] != cur
]
if version_issues:
    note("<b>Library version mismatch:</b> " + "; ".join(version_issues)
         + ". Pin the export versions in requirements.txt so the saved models load identically.",
         "warn")

# Derived, read-only values (dashboard cell 2)
n_total = len(df)
n_lower = int((df[TARGET] == 0).sum())
n_higher = int((df[TARGET] == 1).sum())
mean_lower = df.loc[df[TARGET] == 0, STREAMS].mean()
mean_higher = df.loc[df[TARGET] == 1, STREAMS].mean()
overall_pct = 100 * n_higher / n_total
y_test = test_preds["y_test"].values
options = {c: sorted(df[c].dropna().unique().tolist()) for c in categorical_features}
year_min, year_max = int(df["Debut Year"].min()), int(df["Debut Year"].max())
fi = feature_importance_df.copy()
fi["Display"] = fi["Feature"].apply(lambda f: clean_feature(f, categorical_features))
fi = fi.sort_values("Importance", ascending=False).reset_index(drop=True)
chi_df = chi_square_table(df, tuple(categorical_features))

tabs = st.tabs(["Overview", "Artist Explorer", "Group Analysis", "Model Performance",
                "Feature Importance", "Statistical Tests", "Try a Prediction"])

# ===========================================================================
# 1. OVERVIEW
# ===========================================================================
with tabs[0]:
    st.subheader("Overview")
    st.markdown(
        f"<p class='section-cap'>Target: <b>Streaming Performance</b>. Artists with "
        f"Total Streams at or above the 75th percentile ({stream_threshold:,.2f} M) are labelled "
        f"Higher Streaming; all others Lower Streaming.</p>", unsafe_allow_html=True)
    cards_row([
        ("Total Artists", f"{n_total}", "in the dataset"),
        ("Lower Streaming", f"{n_lower}", f"{100 * n_lower / n_total:.0f}% of artists"),
        ("Higher Streaming", f"{n_higher}", f"{100 * n_higher / n_total:.0f}% of artists"),
        ("% Higher Streaming", f"{overall_pct:.1f}%", "share of all artists"),
    ])
    st.write("")
    cards_row([
        ("Threshold (75th pct)", f"{stream_threshold:,.2f} M", "total streams"),
        ("Mean Streams: Lower", f"{mean_lower:,.0f} M", f"exact: {mean_lower:,.2f} M"),
        ("Mean Streams: Higher", f"{mean_higher:,.0f} M", f"exact: {mean_higher:,.2f} M"),
    ])
    st.write("")

    c1, c2 = st.columns([1, 1.6])
    with c1:
        counts = df["Group"].value_counts().reindex([LOWER, HIGHER])
        fig = go.Figure(go.Bar(
            x=counts.index, y=counts.values, marker_color=[GROUP_COLORS[g] for g in counts.index],
            text=[f"{v} ({100 * v / n_total:.0f}%)" for v in counts.values],
            textposition="outside", textfont=dict(color=TEXT),
            hovertemplate="%{x}: %{y} artists<extra></extra>", width=0.55,
        ))
        fig.update_yaxes(title="Number of Artists", range=[0, counts.max() * 1.18])
        show(style(fig, "Streaming Performance groups"))
    with c2:
        fig = go.Figure()
        for g in [LOWER, HIGHER]:
            fig.add_trace(go.Histogram(
                x=df.loc[df["Group"] == g, STREAMS], name=g, marker_color=GROUP_COLORS[g],
                nbinsx=40, marker_line=dict(color=PANEL, width=1),
                hovertemplate="%{x} M: %{y} artists<extra>" + g + "</extra>",
            ))
        fig.update_layout(barmode="stack", legend_traceorder="normal")
        fig.add_vline(x=stream_threshold, line_dash="dash", line_color=TEXT,
                      annotation_text=f"Threshold: {stream_threshold:,.2f} M",
                      annotation_font_color=TEXT)
        fig.update_xaxes(title="Total Streams (millions)")
        fig.update_yaxes(title="Number of Artists")
        show(style(fig, "Distribution of Total Spotify Streams"))

    top_n = st.slider("Top N artists", min_value=5, max_value=30, value=15, step=5, key="top_n")
    d = df.nlargest(top_n, STREAMS).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d[STREAMS], y=d["Artist Name"], orientation="h",
        marker_color=[GROUP_COLORS[g] for g in d["Group"]],
        hovertemplate="%{y}: %{x:,.1f} M<extra></extra>",
    ))
    fig.update_xaxes(title="Total Streams (millions)")
    show(style(fig, f"Top {top_n} Artists by Total Streams", height=max(400, 24 * top_n + 120)))

# ===========================================================================
# 2. ARTIST EXPLORER
# ===========================================================================
with tabs[1]:
    st.subheader("Artist Explorer")
    f1, f2, f3 = st.columns(3)
    sel = {
        "Primary Genre": f1.selectbox("Primary Genre", ["All"] + options["Primary Genre"], key="ex_genre"),
        "Country of Origin": f2.selectbox("Country of Origin", ["All"] + options["Country of Origin"], key="ex_country"),
        "Primary Language": f3.selectbox("Primary Language", ["All"] + options["Primary Language"], key="ex_lang"),
    }
    f4, f5, f6 = st.columns(3)
    sel["Sex"] = f4.selectbox("Sex", ["All"] + options["Sex"], key="ex_sex")
    sel["Artist Type"] = f5.selectbox("Artist Type", ["All"] + options["Artist Type"], key="ex_type")
    lo, hi = f6.slider("Debut Year range", year_min, year_max, (year_min, year_max), key="ex_year")

    d = df
    for col, val in sel.items():
        if val != "All":
            d = d[d[col] == val]
    d = d[(d["Debut Year"] >= lo) & (d["Debut Year"] <= hi)]
    n = len(d)
    n_hi = int(d[TARGET].sum())
    pct = f"{100 * n_hi / n:.1f}%" if n else "n/a"

    cards_row([
        ("Matching artists", f"{n}", f"of {n_total}"),
        ("Higher Streaming", f"{n_hi}", "in the filtered set"),
        ("% Higher Streaming", pct, f"overall: {overall_pct:.0f}%"),
    ])
    st.write("")

    if n:
        top = d.nlargest(25, STREAMS).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=top[STREAMS], y=top["Artist Name"], orientation="h",
            marker_color=[GROUP_COLORS[g] for g in top["Group"]],
            hovertemplate="%{y}: %{x:,.1f} M<extra></extra>",
        ))
        fig.update_xaxes(title="Total Streams (millions)")
        show(style(fig, "Top filtered artists (up to 25)", height=max(300, 24 * len(top) + 120)))
        st.caption("Green bars = Higher Streaming, grey bars = Lower Streaming. "
                   "Table shows up to 100 artists, sorted by total streams.")
        cols = ["Artist Name", "Primary Genre", "Country of Origin", "Primary Language",
                "Sex", "Artist Type", "Debut Year", STREAMS, "Group"]
        out = d.sort_values(STREAMS, ascending=False)[cols].head(100).rename(
            columns={"Group": "Streaming group"})
        table(out, height=420, formats={STREAMS: "{:,.1f}", "Debut Year": "{:d}"},
              highlight=lambda r: r["Streaming group"] == HIGHER)
    else:
        note("No artists match the selected filters.", "grey")

# ===========================================================================
# 3. GROUP ANALYSIS
# ===========================================================================
with tabs[2]:
    st.subheader("Group Analysis")
    g1, g2, g3 = st.columns([1.2, 1, 1.2])
    g_col = g1.selectbox("Characteristic", GROUP_COLUMNS, index=0, key="g_col")
    g_mode = g2.radio("Display", ["Percentages", "Counts"], horizontal=True, key="g_mode")
    g_min = g3.slider("Minimum group size", 1, 50, 5, key="g_min")

    t = group_stats(df, g_col)
    t = t[t["n"] >= g_min]
    if len(t) == 0:
        note("No category meets the minimum group size.", "grey")
    else:
        fig = go.Figure()
        if g_mode == "Counts":
            tp = t.sort_values("n").tail(20)
            fig.add_trace(go.Bar(y=tp[g_col], x=tp["lower"], name=LOWER, orientation="h",
                                 marker_color=GREY,
                                 hovertemplate="%{y}: %{x} Lower<extra></extra>"))
            fig.add_trace(go.Bar(y=tp[g_col], x=tp["higher"], name=HIGHER, orientation="h",
                                 marker_color=GREEN,
                                 hovertemplate="%{y}: %{x} Higher<extra></extra>"))
            fig.update_layout(barmode="stack", legend_traceorder="normal")
            fig.update_xaxes(title="Number of Artists")
            title = f"Artist counts by {g_col} (groups with n ≥ {g_min})"
        else:
            tp = t.sort_values("pct_higher").tail(20)
            fig.add_trace(go.Bar(
                y=tp[g_col], x=tp["pct_higher"], orientation="h", marker_color=GREEN,
                text=[f"{p:.1f}%  (n={k})" for p, k in zip(tp["pct_higher"], tp["n"])],
                textposition="outside", textfont=dict(color=TEXT), name="% Higher Streaming",
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            ))
            fig.add_vline(x=overall_pct, line_dash="dash", line_color=TEXT,
                          annotation_text=f"Overall: {overall_pct:.0f}%", annotation_position="bottom right",
                          annotation_font_color=TEXT)
            fig.update_xaxes(title="Higher Streaming artists (%)", range=[0, 115])
            title = f"% Higher Streaming by {g_col} (groups with n ≥ {g_min})"
        show(style(fig, title, height=max(320, 28 * len(tp) + 140)))
        if len(t) > 20:
            st.caption(f"Chart shows the top 20 of {len(t)} qualifying categories; "
                       "the table below lists all of them.")

        out = t.sort_values("pct_higher" if g_mode == "Percentages" else "n", ascending=False)
        out = out.rename(columns={"n": "Artists", "higher": "Higher Streaming",
                                  "lower": "Lower Streaming", "pct_higher": "% Higher Streaming"})
        table(out[[g_col, "Artists", "Lower Streaming", "Higher Streaming", "% Higher Streaming"]],
              height=min(420, 38 * len(out) + 40), formats={"% Higher Streaming": "{:.2f}%"})

    note("Percentages from very small groups are unstable (one artist can equal 100%). "
         "Use the minimum group size control before interpreting. <b>These are descriptive "
         "patterns, not causal effects.</b>", "grey")

# ===========================================================================
# 4. MODEL PERFORMANCE
# ===========================================================================
with tabs[3]:
    st.subheader("Model Performance")
    st.markdown("<p class='section-cap'>Existing results on the held-out test set "
                "(100 artists: 75 Lower, 25 Higher). Loaded from the exported project files; "
                "nothing is retrained.</p>", unsafe_allow_html=True)

    fig = go.Figure()
    for _, r in final_model_results.iterrows():
        fig.add_trace(go.Bar(
            x=METRICS, y=[r[m] for m in METRICS], name=r["Model"],
            marker_color=MODEL_COLORS.get(r["Model"], GREY),
            text=[f"{r[m]:.2f}" for m in METRICS], textposition="outside",
            textfont=dict(color=TEXT),
            hovertemplate="%{x}: %{y:.3f}<extra>" + r["Model"] + "</extra>",
        ))
    fig.update_layout(barmode="group", bargroupgap=0.08)
    fig.update_yaxes(range=[0, 1.05], title="Score")
    show(style(fig, "Model comparison on the test set (100 artists)"))

    table(final_model_results[["Model"] + METRICS], formats={m: "{:.3f}" for m in METRICS},
          highlight=lambda r: r["Model"] == "XGBoost")

    note("<b>Reading accuracy carefully:</b> Logistic Regression's 75% accuracy is largely "
         "a result of the 75/25 class imbalance. It predicted Lower Streaming for every test "
         "artist, so its precision, recall and F1 for the Higher Streaming class are 0 and "
         "its balanced accuracy is 0.50. Balanced accuracy, recall and F1 are more "
         "informative here than accuracy alone.")

    m_sel = st.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index("XGBoost"), key="m_sel")
    preds = test_preds[PRED_COLS[m_sel]].values
    c1, c2 = st.columns([1, 1.15])
    with c1:
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        fig = go.Figure(go.Heatmap(
            z=cm, x=["Predicted Lower", "Predicted Higher"], y=["Actual Lower", "Actual Higher"],
            colorscale=[[0, PANEL_2], [1, GREEN]], text=cm, texttemplate="%{text}",
            textfont=dict(size=24, color=TEXT), showscale=False, xgap=3, ygap=3,
            hovertemplate="%{y} / %{x}: %{z}<extra></extra>",
        ))
        fig.update_yaxes(autorange="reversed")
        show(style(fig, f"Confusion Matrix: {m_sel}", height=420))
    with c2:
        st.markdown(f"##### Classification report: {m_sel}")
        rep = classification_report(y_test, preds, labels=[0, 1], target_names=[LOWER, HIGHER],
                                    zero_division=0, output_dict=True)
        rows = [{"Class": k, "Precision": rep[k]["precision"], "Recall": rep[k]["recall"],
                 "F1": rep[k]["f1-score"], "Support": int(rep[k]["support"])}
                for k in [LOWER, HIGHER, "macro avg", "weighted avg"]]
        rows.insert(2, {"Class": "accuracy", "Precision": None, "Recall": None,
                        "F1": rep["accuracy"], "Support": int(rep["macro avg"]["support"])})
        table(pd.DataFrame(rows),
              formats={"Precision": "{:.2f}", "Recall": "{:.2f}", "F1": "{:.2f}"})
        tn, fp, fn, tp_ = cm.ravel()
        st.caption(f"{tp_} of {tp_ + fn} Higher Streaming test artists were identified "
                   f"(recall {tp_ / (tp_ + fn):.2f}); {fp} Lower Streaming artists were "
                   f"predicted as Higher.")

    with st.expander("Interpretation from the project (Results & Discussion)"):
        st.markdown(
            "- **Logistic Regression** achieved an accuracy of 75%, but it failed to identify the "
            "Higher Streaming class, resulting in zero precision, recall, and F1-score for that class.\n"
            "- **Random Forest** achieved 62% accuracy, with a precision, recall, and F1-score of "
            "approximately 0.24 for the Higher Streaming class.\n"
            "- **XGBoost** achieved 74% accuracy and the highest balanced accuracy among the tested "
            "models at approximately 0.547. It also achieved the highest precision for the Higher "
            "Streaming class at approximately 0.444. However, its recall was only 0.16, indicating "
            "that many Higher Streaming artists were not identified by the model.\n\n"
            "The Higher Streaming group represented only 25% of the dataset, creating a class "
            "imbalance. Therefore, accuracy alone was not sufficient for evaluating the models, and "
            "balanced accuracy, precision, recall, and F1-score were also considered. Overall, the "
            "models demonstrated limited ability to reliably distinguish Higher Streaming artists "
            "from Lower Streaming artists using the available artist characteristics.")

# ===========================================================================
# 5. FEATURE IMPORTANCE
# ===========================================================================
with tabs[4]:
    st.subheader("Feature Importance (XGBoost)")
    fi_n = st.slider("Top N features", 5, min(30, len(fi)), 15, key="fi_n")
    d = fi.head(fi_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d["Importance"], y=d["Display"], orientation="h", marker_color=GREEN,
        customdata=d["Feature"],
        hovertemplate="%{y}: %{x:.4f}<br><span style='color:#B3B3B3'>%{customdata}</span>"
                      "<extra></extra>",
    ))
    fig.update_xaxes(title="XGBoost importance")
    show(style(fig, f"Top {fi_n} XGBoost features", height=max(380, 26 * fi_n + 120)))

    note("Feature importance is <b>model-based and predictive, not causal</b>. It shows which "
         "encoded features the XGBoost model used most when splitting, not that a characteristic "
         "causes higher streaming. The model itself has low recall for the Higher Streaming "
         "class, so these importances should be read with caution.", "grey")

    with st.expander(f"All {len(fi)} encoded features"):
        out = fi[["Feature", "Display", "Importance"]].rename(
            columns={"Feature": "Encoded feature name", "Display": "Readable name"})
        out.insert(0, "Rank", range(1, len(out) + 1))
        table(out, height=420, formats={"Importance": "{:.6f}"})

# ===========================================================================
# 6. STATISTICAL TESTS
# ===========================================================================
with tabs[5]:
    st.subheader("Chi-square tests of association with Streaming Performance")
    st.markdown("<p class='section-cap'>scipy.stats.chi2_contingency on the contingency table of "
                "each characteristic vs. Streaming Performance (all 500 artists), α = 0.05.</p>",
                unsafe_allow_html=True)
    table(chi_df, formats={"Chi-square": "{:.4f}", "p-value": "{:.4f}"},
          highlight=lambda r: r["Significant at alpha = 0.05"] == "Yes")

    at = group_stats(df, "Artist Type").set_index("Artist Type")
    at_chi = chi_df[chi_df["Variable"] == "Artist Type"].iloc[0]
    at_text = "; ".join(
        f"<b>{k}</b>: {int(r['higher'])} of {int(r['n'])} artists are Higher Streaming "
        f"({r['pct_higher']:.2f}%)" for k, r in at.iterrows())
    note(f"<b>Artist Type</b> is the only tested variable with a statistically significant "
         f"association (chi-square = {at_chi['Chi-square']:.4f}, p = {at_chi['p-value']:.4f}). "
         f"{at_text}. This is an association in this dataset, not evidence of causation.")

    c1, c2 = st.columns([1, 1])
    with c1:
        t = group_stats(df, "Artist Type").sort_values("pct_higher", ascending=False)
        fig = go.Figure(go.Bar(
            x=t["Artist Type"], y=t["pct_higher"], marker_color=GREEN, width=0.5,
            text=[f"{p:.2f}%  (n={k})" for p, k in zip(t["pct_higher"], t["n"])],
            textposition="outside", textfont=dict(color=TEXT),
            hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
        ))
        fig.add_hline(y=overall_pct, line_dash="dash", line_color=MUTED,
                      annotation_text=f"Overall: {overall_pct:.0f}%", annotation_position="bottom right",
                      annotation_font_color=MUTED)
        fig.update_yaxes(title="Higher Streaming artists (%)", range=[0, 40])
        show(style(fig, "Solo vs Group: share of Higher Streaming artists", height=400))
    with c2:
        st.markdown("##### Artist Type contingency table")
        ct = pd.crosstab(df["Artist Type"], df["Group"])[[LOWER, HIGHER]].reset_index()
        ct["Total"] = ct[LOWER] + ct[HIGHER]
        table(ct)
        st.markdown(
            "<div class='note grey'>The tests for Sex, Country of Origin, Primary Language, and "
            "Primary Genre did not provide sufficient evidence of a statistically significant "
            "association at the 0.05 significance level. Several of these variables have many "
            "small categories, so these results should be read together with the Group Analysis "
            "minimum-size caution.</div>", unsafe_allow_html=True)

# ===========================================================================
# 7. TRY A PREDICTION
# ===========================================================================
with tabs[6]:
    st.subheader("Try a Prediction")
    st.markdown("<p class='section-cap'>Enter artist characteristics. The three saved, already "
                "fitted pipelines are applied to the input; nothing is retrained.</p>",
                unsafe_allow_html=True)

    def default_index(col, value):
        return options[col].index(value) if value in options[col] else 0

    with st.form("predict_form"):
        p1, p2, p3 = st.columns(3)
        row = {
            "Sex": p1.selectbox("Sex", options["Sex"], index=0),
            "Country of Origin": p2.selectbox(
                "Country of Origin", options["Country of Origin"],
                index=default_index("Country of Origin", "United States")),
            "Primary Language": p3.selectbox(
                "Primary Language", options["Primary Language"],
                index=default_index("Primary Language", "English")),
        }
        p4, p5, p6 = st.columns(3)
        row["Primary Genre"] = p4.selectbox("Primary Genre", options["Primary Genre"],
                                            index=default_index("Primary Genre", "Pop"))
        row["Artist Type"] = p5.selectbox("Artist Type", options["Artist Type"],
                                          index=default_index("Artist Type", "Solo"))
        row["Debut Year"] = p6.slider("Debut Year", year_min, year_max,
                                      min(max(2010, year_min), year_max))
        submitted = st.form_submit_button("▶  Predict", type="primary")

    if submitted:
        # Same logic as dash_run_prediction (dashboard cell 4)
        pred_input = pd.DataFrame([row])[features]   # same six columns, same order as X
        rows, p_high = [], {}
        for name in MODEL_NAMES:
            model = models[name]                     # existing fitted pipelines only
            pred = int(model.predict(pred_input)[0])
            proba = model.predict_proba(pred_input)[0]
            classes = list(model.classes_)
            pl, ph = float(proba[classes.index(0)]), float(proba[classes.index(1)])
            p_high[name] = ph
            rows.append({"Model": name, "Predicted class": HIGHER if pred == 1 else LOWER,
                         "P(Lower Streaming)": pl, "P(Higher Streaming)": ph})
        st.session_state["prediction"] = (row, pd.DataFrame(rows), p_high)

    if "prediction" in st.session_state:
        used, res, p_high = st.session_state["prediction"]
        votes = int((res["Predicted class"] == HIGHER).sum())
        st.markdown("##### Results for: " + ", ".join(f"{k} = <b>{v}</b>" for k, v in used.items()),
                    unsafe_allow_html=True)
        cards_row([
            ("Models predicting Higher", f"{votes} of 3", "majority" if votes >= 2 else "minority"),
            ("Lowest P(Higher)", f"{min(p_high.values()):.1%}", min(p_high, key=p_high.get)),
            ("Highest P(Higher)", f"{max(p_high.values()):.1%}", max(p_high, key=p_high.get)),
        ])
        st.write("")
        c1, c2 = st.columns([1.1, 1])
        with c1:
            table(res, formats={"P(Lower Streaming)": "{:.1%}", "P(Higher Streaming)": "{:.1%}"},
                  highlight=lambda r: r["Predicted class"] == HIGHER)
            note(f"<b>Comparison:</b> {votes} of 3 models predict Higher Streaming. "
                 f"Model-estimated probability of Higher Streaming ranges from "
                 f"{min(p_high.values()):.1%} to {max(p_high.values()):.1%}. These are model "
                 f"estimates, not certainties; on the test set the models identified few "
                 f"Higher Streaming artists (low recall).")
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(p_high.keys()), y=[1 - v for v in p_high.values()], name=LOWER,
                marker_color=GREY, text=[f"{1 - v:.1%}" for v in p_high.values()],
                textposition="inside", hovertemplate="%{x}: %{y:.1%}<extra>P(Lower)</extra>",
            ))
            fig.add_trace(go.Bar(
                x=list(p_high.keys()), y=list(p_high.values()), name=HIGHER,
                marker_color=GREEN, text=[f"{v:.1%}" for v in p_high.values()],
                textposition="inside", hovertemplate="%{x}: %{y:.1%}<extra>P(Higher)</extra>",
            ))
            fig.update_layout(barmode="group")
            fig.add_hline(y=0.5, line_dash="dash", line_color=TEXT,
                          annotation_text="0.5 decision line", annotation_font_color=TEXT)
            fig.update_yaxes(range=[0, 1.1], title="Predicted probability", tickformat=".0%")
            show(style(fig, "Predicted class probabilities by model", height=400))

        note("This prediction classifies streaming performance among artists represented in "
             "this dataset. It does not predict whether an arbitrary new artist will become "
             "successful, and the results should not be interpreted as causal.", "grey")
    else:
        st.info("Choose the characteristics above and click **Predict**.")

# ---------------------------------------------------------------------------
# Footer: limitations from the notebook (section 18) and references (section 20)
# ---------------------------------------------------------------------------
with st.expander("Limitations of the analysis"):
    st.markdown(
        "1. The dataset contains 500 artists, which limits the size of the available sample for "
        "machine learning.\n"
        "2. The dataset consists of already highly streamed artists. Therefore, the results should "
        "not be interpreted as predicting whether an unknown artist will become successful in the "
        "general music industry.\n"
        "3. The streaming-performance classes are imbalanced, with 75% of artists in the Lower "
        "Streaming group and 25% in the Higher Streaming group.\n"
        "4. The available dataset contains a limited number of artist-level characteristics "
        "compared with the broader data used in the reference research paper.\n"
        "5. The machine learning models showed low recall for the Higher Streaming class, "
        "indicating that the models missed a substantial number of Higher Streaming artists.\n"
        "6. Feature importance identifies variables that are useful to the model, but it does not "
        "establish causation.\n"
        "7. The statistical tests identify associations within this dataset and should not be "
        "interpreted as proof that a particular artist characteristic causes higher streaming "
        "performance.\n"
        "8. The results are based on the available dataset and may not generalize to all Spotify "
        "artists or the wider global music industry.")

st.markdown(
    "<div class='footer'>Data: Kaggle, <i>Most Streamed Artists on Spotify</i> (V1.1, 500 artists). "
    "Reference study: Kang, I., Mandulak, M., &amp; Szymanski, B. K. (2022). Analyzing and "
    "predicting success of professional musicians. <i>Scientific Reports</i>. "
    f"Models: scikit-learn {sklearn.__version__}, XGBoost {xgboost.__version__}.</div>",
    unsafe_allow_html=True,
)
