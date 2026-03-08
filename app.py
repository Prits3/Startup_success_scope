import os
import pickle
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "model.pkl"
METRICS_PATH = PROJECT_ROOT / "model" / "metrics.pkl"

st.set_page_config(page_title="VentureScope AI", page_icon="🚀", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0A1628; color: #E0E0E0; }
section[data-testid="stSidebar"] { background-color: #0D1F38; }
h1, h2, h3 { color: #00B4D8; }
.stButton>button {
    background-color: #00B4D8;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}
.card {
    background:#0D1F38;
    padding:20px;
    border-radius:12px;
    border:1px solid #1E3A5F;
}
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("# VentureScope AI")
    st.markdown("*AI-Powered Startup Intelligence*")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Dashboard", "Startup Analysis", "Market Insights", "Compare Startups"],
    )
    st.markdown("---")
    st.markdown("**Model:** Random Forest")

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "rb") as f:
            metrics = pickle.load(f)
        st.markdown(f"**Rows:** {metrics.get('rows', 0):,}")
        st.markdown(f"**Accuracy:** {metrics.get('accuracy', 0):.1%}")
        st.markdown(f"**ROC-AUC:** {metrics.get('roc_auc', 0):.3f}")

if not MODEL_PATH.exists():
    with st.spinner("Training model on startup data..."):
        from model.train import train_model

        train_model()
    st.success("Model is ready.")

if page == "Dashboard":
    st.markdown("# VentureScope AI")
    st.markdown("### AI-Powered Venture Capital Intelligence Platform")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.markdown("<div class='card'><h3>Startup Analysis</h3><p>Enter startup details and get an AI-powered success probability score.</p></div>", unsafe_allow_html=True)
    col2.markdown("<div class='card'><h3>Market Insights</h3><p>Explore industry trends, funding patterns, and geographic clusters.</p></div>", unsafe_allow_html=True)
    col3.markdown("<div class='card'><h3>Compare Startups</h3><p>Compare multiple startups side-by-side across key investment signals.</p></div>", unsafe_allow_html=True)

elif page == "Startup Analysis":
    from pages.startup_analysis import show

    show()
elif page == "Market Insights":
    from pages.market_insights import show

    show()
else:
    from pages.comparison import show

    show()
