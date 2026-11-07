"""
Breast cancer FNA classifier — Streamlit front end for ML Assignment 2.

The app only LOADS the five pre-trained pipelines from model/ and scores the
uploaded test CSV. No model is (re)trained here.
"""
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
SAMPLE_CSV = os.path.join(BASE_DIR, "test_data.csv")

# H&E stain palette: hematoxylin violet, eosin pink, slide-glass white
VIOLET = "#3E2A6B"
EOSIN = "#C9487A"
LILAC = "#E7E1F2"
INK = "#232033"

st.set_page_config(page_title="FNA Diagnosis Classifier", page_icon="🔬", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Franklin:wght@400;600;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText {{ font-family: 'Libre Franklin', sans-serif; }}
h1 {{ color: {VIOLET}; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.1rem; }}
h2, h3 {{ color: {INK}; font-weight: 600; }}
.lede {{ color: #5A5470; max-width: 68ch; font-size: 1.02rem; line-height: 1.55; }}
.metric-strip {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr));
    border-top: 3px solid {VIOLET}; border-bottom: 1px solid {LILAC}; margin: 0.6rem 0 1.4rem; }}
.metric-strip div {{ padding: 0.9rem 0.8rem; border-right: 1px solid {LILAC}; }}
.metric-strip div:last-child {{ border-right: none; }}
.metric-strip .v {{ font-size: 1.9rem; font-weight: 800; color: {VIOLET}; line-height: 1.1; }}
.metric-strip .k {{ font-size: 0.85rem; color: #6B6580; }}
@media (max-width: 800px) {{ .metric-strip {{ grid-template-columns: repeat(3, 1fr); }} }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_artifacts():
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        meta = json.load(f)
    models = {name: joblib.load(os.path.join(MODEL_DIR, fname))
              for name, fname in meta["model_files"].items()}
    return meta, models


def prepare_data(df, features, target):
    """Validate the uploaded frame. Returns (X, y, error_message)."""
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    missing = [c for c in features if c not in df.columns]
    if missing:
        return None, None, (f"The file is missing {len(missing)} required feature column(s), "
                            f"for example: {', '.join(missing[:5])}. "
                            "Use test_data.csv from the repository as a template.")
    if target not in df.columns:
        return None, None, (f"The file has no '{target}' column. Add the true labels "
                            "(1 = malignant, 0 = benign) so the app can compute metrics.")
    y = df[target]
    if not pd.api.types.is_numeric_dtype(y):  # accept the original UCI M/B coding too
        y = y.astype(str).str.strip().str.upper().map({"M": 1, "B": 0})
    if y.isna().any() or not set(pd.unique(y)).issubset({0, 1}):
        return None, None, f"'{target}' must contain only 0/1 (or B/M) values."
    X = df[features].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        return None, None, f"Non-numeric or empty values found in: {', '.join(bad[:5])}."
    return X, y.astype(int), None


def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob) if y.nunique() == 2 else np.nan
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }, y_pred, y_prob


def fmt(v):
    return "n/a" if pd.isna(v) else f"{v:.4f}"


# ---------------------------------------------------------------- app
try:
    meta, models = load_artifacts()
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load the saved models from model/: {e}")
    st.stop()

FEATURES, TARGET = meta["features"], meta["target"]
CLASS_NAMES = ["Benign", "Malignant"]

with st.sidebar:
    st.header("Test data")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"],
                                help=f"Needs the {len(FEATURES)} feature columns and '{TARGET}'.")
    use_sample = st.checkbox("Use the bundled test_data.csv instead", value=uploaded is None)
    with open(SAMPLE_CSV, "rb") as f:
        st.download_button("Download test_data.csv", f, file_name="test_data.csv", mime="text/csv")

    st.header("Model")
    model_name = st.selectbox("Choose a classifier", list(models.keys()), index=4)
    st.caption("Models were trained offline in the notebook and are only loaded here.")

st.title("Breast mass FNA diagnosis classifier")
st.markdown(
    '<p class="lede">Scores fine needle aspirate measurements from the UCI Breast Cancer Wisconsin '
    '(Diagnostic) dataset as benign or malignant, using one of five pre-trained models. '
    'Upload labelled test data to see how each model performs on it.</p>',
    unsafe_allow_html=True)

if uploaded is not None and not use_sample:
    raw_df = pd.read_csv(uploaded)
    source = uploaded.name
elif use_sample:
    raw_df = pd.read_csv(SAMPLE_CSV)
    source = "test_data.csv (bundled)"
else:
    st.info("Upload a test CSV in the sidebar, or tick the box to use the bundled test_data.csv.")
    st.stop()

X, y, err = prepare_data(raw_df, FEATURES, TARGET)
if err:
    st.error(err)
    st.stop()

st.caption(f"Data: {source} — {len(X)} rows, {int(y.sum())} malignant, {int((y == 0).sum())} benign")

tab_model, tab_compare, tab_data = st.tabs(["Selected model", "Compare all models", "Data preview"])

with tab_model:
    model = models[model_name]
    metrics, y_pred, y_prob = compute_metrics(model, X, y)
    st.subheader(model_name)
    strip = "".join(f'<div><div class="v">{fmt(v)}</div><div class="k">{k}</div></div>'
                    for k, v in metrics.items())
    st.markdown(f'<div class="metric-strip">{strip}</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1.25])
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.4, 3.6))
        sns.heatmap(cm, annot=True, fmt="d", cbar=False, ax=ax,
                    cmap=sns.light_palette(VIOLET, as_cmap=True),
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    annot_kws={"size": 15, "weight": "bold"})
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)
        fn, fp = cm[1, 0], cm[0, 1]
        st.caption(f"{fn} malignant case(s) missed, {fp} benign case(s) flagged as malignant.")
    with right:
        st.markdown("**Classification report**")
        rep = classification_report(y, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
                                    output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(4))

    out = raw_df.copy()
    out["predicted"] = y_pred
    out["malignant_probability"] = np.round(y_prob, 4)
    st.download_button("Download predictions", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_name.replace(' ', '_').lower()}.csv",
                       mime="text/csv")

with tab_compare:
    rows = {name: compute_metrics(m, X, y)[0] for name, m in models.items()}
    comp = pd.DataFrame(rows).T
    comp.index.name = "ML Model Name"
    st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#F3C6D7"))
    best = comp["MCC"].idxmax()
    st.markdown(f"Highest MCC on this data: **{best}** ({comp.loc[best, 'MCC']:.4f}).")

    fig, ax = plt.subplots(figsize=(10, 3.8))
    comp.plot.bar(ax=ax, rot=0, width=0.82,
                  color=[VIOLET, "#6F5AA8", "#A08BD1", EOSIN, "#E58AAE", "#F3C6D7"])
    ax.set_ylim(max(0, np.nanmin(comp.values) - 0.05), 1.01)
    ax.legend(ncol=6, loc="lower right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("Score")
    st.pyplot(fig)
    plt.close(fig)

with tab_data:
    st.dataframe(raw_df.head(50))
    st.caption("First 50 rows of the evaluated file.")
