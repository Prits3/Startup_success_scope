import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model.predict import get_feature_importance, predict_startup


def _confidence_score(payload: dict, df: pd.DataFrame) -> float:
    cols = ["funding_amount", "num_funding_rounds", "num_milestones", "team_size", "years_active"]
    inside = 0
    for col in cols:
        low, high = df[col].quantile(0.05), df[col].quantile(0.95)
        if low <= payload[col] <= high:
            inside += 1
    return round(58 + 42 * (inside / len(cols)), 1)


def _analyst_note(payload: dict, prob: float, risk: str) -> str:
    return (
        f"This startup is positioned in {payload['industry_sector']} with {payload['num_funding_rounds']} funding rounds and "
        f"{payload['num_milestones']} milestones. Current model output indicates {prob:.1f}% success probability "
        f"with {risk.lower()} profile. Main watchpoint is execution consistency as the company scales."
    )


def render(df: pd.DataFrame) -> None:
    st.markdown("<p class='page-title'>Analyze</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='page-sub'>Evaluate one startup and get AI score, confidence, recommendation, and reasoning.</p>",
        unsafe_allow_html=True,
    )

    industries = sorted(df["industry_sector"].unique().tolist())
    locations = sorted(df["location"].unique().tolist())

    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### Input")
        with st.container(border=True):
            industry = st.selectbox("Industry", industries)
            location = st.selectbox("Location", locations)
            funding = st.slider("Funding Amount (M$)", 0.5, 250.0, 12.0, 0.5)
            rounds = st.slider("Funding Rounds", 1, 10, 3)
            milestones = st.slider("Milestones", 0, 20, 7)
            team = st.slider("Team Size", 2, 500, 35)
            years = st.slider("Years Active", 0.2, 15.0, 3.0)
            run = st.button("Analyze Startup", type="primary", width='stretch', key="analyze_submit")

    with right:
        st.markdown("#### Result")
        if not run:
            st.info("Run analysis to generate result.")
            return

        payload = {
            "funding_amount": float(funding),
            "num_funding_rounds": int(rounds),
            "industry_sector": industry,
            "location": location,
            "num_milestones": int(milestones),
            "team_size": int(team),
            "years_active": float(years),
        }

        with st.spinner("AI is evaluating startup profile..."):
            try:
                pred = predict_startup(payload)
            except Exception as exc:
                st.error(
                    "Prediction failed. Ensure model artifacts exist by running "
                    "`python model/train.py` from the project root."
                )
                st.code(str(exc))
                return

        prob = pred["success_probability"]
        risk = pred["risk_level"]
        confidence = _confidence_score(payload, df)
        recommendation = {
            "High Potential": "Prioritize for deeper diligence",
            "Moderate Potential": "Proceed with conditional diligence",
            "High Risk": "Deprioritize until stronger traction",
        }[risk]

        r1, r2 = st.columns(2)
        r1.metric("Success Probability", f"{prob:.1f}%")
        r2.metric("Confidence", f"{confidence:.1f}%")
        st.metric("Recommendation", recommendation)

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2bc6f4"},
                    "steps": [
                        {"range": [0, 45], "color": "#44292d"},
                        {"range": [45, 70], "color": "#4d4026"},
                        {"range": [70, 100], "color": "#1f4d3a"},
                    ],
                },
            )
        )
        gauge.update_layout(paper_bgcolor="#0A101A", font_color="#e8f0fb", margin=dict(l=8, r=8, t=8, b=8), height=250)
        st.plotly_chart(gauge, width='stretch')

    st.markdown("#### AI Reasoning")
    st.markdown("<div class='ai-brief'><b>AI Analyst Note</b><br/>" + _analyst_note(payload, prob, risk) + "</div>", unsafe_allow_html=True)

    try:
        fi = get_feature_importance(top_n=8).sort_values("importance")
    except Exception as exc:
        st.warning("Feature importance is temporarily unavailable.")
        st.code(str(exc))
        return
    fig = px.bar(
        fi,
        x="importance",
        y="feature",
        orientation="h",
        title="Top Drivers",
        color_discrete_sequence=["#2bc6f4"],
    )
    fig.update_layout(
        paper_bgcolor="#0A101A",
        plot_bgcolor="#0A101A",
        font_color="#e8f0fb",
        xaxis_title="Relative Impact",
        yaxis_title="",
    )
    st.plotly_chart(fig, width='stretch')
