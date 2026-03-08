from pathlib import Path
import pickle

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model.predict import predict_startup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATEGORIES_PATH = PROJECT_ROOT / "model" / "categories.pkl"
DATA_FALLBACK = PROJECT_ROOT / "data" / "startups_raw.csv"


def _load_options():
    if CATEGORIES_PATH.exists():
        with open(CATEGORIES_PATH, "rb") as f:
            cats = pickle.load(f)
        return cats["categories"], cats["countries"]

    if DATA_FALLBACK.exists():
        import pandas as pd

        df = pd.read_csv(DATA_FALLBACK, usecols=["category_list", "country_code"])
        categories = (
            df["category_list"].fillna("Unknown").astype(str).apply(lambda x: x.split("|")[0]).unique().tolist()
        )
        countries = df["country_code"].fillna("Unknown").astype(str).unique().tolist()
        return sorted(categories), sorted(countries)

    return ["Unknown"], ["Unknown"]


def show():
    st.markdown("## Startup Analysis")

    categories, countries = _load_options()
    categories = categories[:80]

    with st.form("analysis_form"):
        col1, col2 = st.columns(2)
        with col1:
            funding = st.number_input("Total Funding ($)", 0, 500_000_000, 1_000_000, 50_000)
            rounds = st.slider("Funding Rounds", 1, 10, 2)
            category = st.selectbox("Industry", categories)
        with col2:
            country = st.selectbox("Country", countries)
            days_to_funding = st.number_input("Days from Founded to First Funding", 0, 3000, 365)
            funding_duration = st.number_input("Funding Duration (days)", 0, 5000, 500)

        submitted = st.form_submit_button("Analyze", use_container_width=True)

    if not submitted:
        return

    result = predict_startup(funding, rounds, category, country, days_to_funding, funding_duration)
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("Success Probability", f"{result['probability']}%")
    c2.metric("Risk Level", result["risk_level"])
    c3.metric("Signal", result["signal"])

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=result["probability"],
            title={"text": "Success Probability", "font": {"color": "white"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": result["color"]},
                "steps": [
                    {"range": [0, 20], "color": "#1a1a2e"},
                    {"range": [20, 40], "color": "#16213e"},
                    {"range": [40, 100], "color": "#0f3460"},
                ],
            },
        )
    )
    gauge.update_layout(paper_bgcolor="#0A1628", font_color="white", height=290)
    st.plotly_chart(gauge, use_container_width=True)

    st.markdown("### Key Success Factors")
    imp = dict(sorted(result["feature_importance"].items(), key=lambda x: x[1]))
    fig2 = px.bar(
        x=list(imp.values()),
        y=list(imp.keys()),
        orientation="h",
        color=list(imp.values()),
        color_continuous_scale=["#0A1628", "#00B4D8"],
    )
    fig2.update_layout(
        paper_bgcolor="#0A1628",
        plot_bgcolor="#0A1628",
        font_color="white",
        showlegend=False,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

    if "compared_startups" not in st.session_state:
        st.session_state.compared_startups = []

    if st.button("Add to Comparison"):
        st.session_state.compared_startups.append(
            {
                "Industry": category,
                "Country": country,
                "Funding": funding,
                "Rounds": rounds,
                "Probability (%)": result["probability"],
                "Risk": result["risk_level"],
            }
        )
        st.success("Added to comparison list.")
