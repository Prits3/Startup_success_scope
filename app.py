from pathlib import Path

import pandas as pd
import streamlit as st

from model.predict import load_metrics
from model.train import save_artifacts, train_model
from pages import comparison, dashboard, model_insights, startup_analysis
from ui import inject_theme


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "startup_data.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "startup_success_model.joblib"

st.set_page_config(page_title="VentureScope AI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
inject_theme()

if not DATA_PATH.exists():
    st.error("Dataset not found. Run `python data/generate_data.py` first.")
    st.stop()

if not MODEL_PATH.exists():
    with st.spinner("Model artifact missing. Training model..."):
        pipeline, metrics = train_model(DATA_PATH)
        save_artifacts(pipeline, metrics, PROJECT_ROOT / "model")
    st.success("Model trained successfully.")

df = pd.read_csv(DATA_PATH)
metrics = load_metrics()

bar_l, bar_r = st.columns([6, 1])
with bar_l:
    st.markdown(
        "<div class='topbar'><div><div class='brand-title'>VentureScope AI</div>"
        "<div class='brand-sub'>AI Venture Intelligence Copilot</div></div></div>",
        unsafe_allow_html=True,
    )
with bar_r:
    st.button("⚙️", width="stretch", key="settings_btn")

with st.sidebar:
    st.markdown("### Navigation")
    section = st.radio(
        "Pages",
        ["Overview", "Analyze", "Compare", "Model"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Predict. Compare. Decide.")
    if metrics:
        st.caption(f"Accuracy: {metrics.get('accuracy', 0) * 100:.1f}%")
        st.caption(f"ROC-AUC: {metrics.get('roc_auc', 0):.3f}")

try:
    if section == "Overview":
        dashboard.render(df)
    elif section == "Analyze":
        startup_analysis.render(df)
    elif section == "Compare":
        comparison.render(df)
    else:
        model_insights.render(df)
except Exception as exc:
    st.error("A runtime error occurred in this page.")
    st.code(str(exc))
